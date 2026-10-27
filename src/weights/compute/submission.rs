use super::Compute;
use crate::gpu::{CommandMemory, Commands};
use anyhow::{Result, anyhow};
use block2::RcBlock;
use futures_channel::oneshot;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::{
    cell::RefCell,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::NonNull,
    sync::Mutex,
};

impl Compute {
    /// Submit work eagerly and signal when its mutable buffers can be reused.
    /// Dropping the future abandons only the wait: command memory, pipelines, buffers,
    /// and mapped pages remain owned by the callback until GPU completion.
    /// Callers must give up host access to submitted writable storage until then.
    pub fn submit(
        &self,
        encode: impl FnOnce(&Commands),
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let completed = objc2::rc::autoreleasepool(|_| -> Result<_> {
            let memory = self.idle.lock().expect("command pool poisoned").take();
            let mut memory = match memory {
                None => {
                    let device = self.queue.device();
                    let weights = self.weights.clone();
                    CommandMemory::new(&device, weights)?
                }
                Some(memory) => memory,
            };
            memory.begin();
            let encoder = memory.command.computeCommandEncoder();
            let encoder =
                encoder.ok_or_else(|| anyhow!("Metal compute encoder creation failed"))?;
            // Preserving queue order across submissions as well as dependencies within each pass.
            encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
            let commands = Commands {
                encoder,
                memory: RefCell::new(memory),
            };
            commands.stamp_start();
            encode(&commands);
            commands.encoder.endEncoding();
            let memory = commands.memory.into_inner();
            memory.end();
            let command = memory.command.clone();
            let (sender, completed) = oneshot::channel();
            let idle = std::sync::Arc::downgrade(&self.idle);
            let queue = self.queue.clone();
            // Taking captures exactly once, independently of the waiter's lifetime.
            let completion = Mutex::new(Some((sender, memory, queue)));
            let handler = RcBlock::new(
                move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                    let completion = completion.lock();
                    let Ok(mut completion) = completion else {
                        return;
                    };
                    let Some((sender, memory, queue)) = completion.take() else {
                        return;
                    };
                    drop(completion);
                    // Containing both internal and consumer-waker panics at the foreign callback boundary.
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        let error = unsafe { feedback.as_ref() }.error();
                        drop(queue);
                        if let Some(error) = error {
                            return Err(anyhow!("Metal command failed: {error}"));
                        }
                        let mut memory = memory;
                        memory.resolve_profile();
                        if let Some(idle) = idle.upgrade() {
                            let mut idle = idle.lock().expect("command pool poisoned");
                            if idle.is_none() {
                                *idle = Some(memory);
                            }
                        }
                        Ok(())
                    }));
                    let result = match result {
                        Err(_) => Err(anyhow!("Metal completion handler panicked")),
                        Ok(result) => result,
                    };
                    let _ = catch_unwind(AssertUnwindSafe(|| sender.send(result)));
                },
            );
            let options = MTL4CommitOptions::new();
            let mut pointer = NonNull::from(&*command);
            unsafe {
                options.addFeedbackHandler(RcBlock::as_ptr(&handler));
                self.queue
                    .commit_count_options(NonNull::from(&mut pointer), 1, &options);
            }
            Ok(completed)
        });
        async move {
            completed?
                .await
                .map_err(|_| anyhow!("Metal completion handler was dropped"))?
        }
    }
}
