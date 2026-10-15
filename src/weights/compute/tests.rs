use super::{Compute, Gpu};
use anyhow::Result;
use memmap2::MmapMut;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTL4CommandQueue, MTLDevice, MTLSharedEvent};
use std::sync::Arc;

type CommandQueueRef = ProtocolObject<dyn MTL4CommandQueue>;
type SharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

/// Queued GPU wait released on drop, including when an assertion unwinds.
pub(crate) struct Gate(SharedEvent);

impl Gate {
    pub(crate) fn new(queue: &CommandQueueRef) -> Self {
        let event = queue.device().newSharedEvent().expect("shared event");
        queue.waitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        Self(event)
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.0.setSignaledValue(1);
    }
}

#[test]
fn dropped_submission_keeps_mapped_pages_until_gpu_completion() -> Result<()> {
    for shared in [false, true] {
        let mapping = MmapMut::map_anon(16384)?;
        let mapping = mapping.make_read_only()?;
        let mapping = Arc::new(mapping);
        let weak = Arc::downgrade(&mapping);
        objc2::rc::autoreleasepool(|_| -> Result<()> {
            let gpu = Gpu::new()?;
            let (storage, residency) = if shared {
                let regions = crate::gpu::MappedRegions::new(&gpu, &mapping, [(0, mapping.len())])?;
                let storage = regions.get(0, mapping.len())?;
                (storage, Some(regions.residency))
            } else {
                let storage = gpu.load(&mapping, 0, mapping.len())?;
                (storage, None)
            };
            let compute = Compute::new(&gpu, residency);
            let gate = Gate::new(&compute.queue);
            let submission = compute.submit(|encoder| {
                let (buffer, _) = storage.binding();
                encoder.resident(buffer);
            });
            drop(storage);
            drop(mapping);
            drop(submission);
            assert!(
                weak.strong_count() > 0,
                "queued Metal resources must retain their mapped pages"
            );
            let finished = compute.submit(|_| {});
            drop(compute);
            drop(gpu);
            assert!(
                weak.strong_count() > 0,
                "the callback must outlive model and session owners"
            );
            drop(gate);
            // Draining the test queue before checking that Metal released the final mapping owner.
            futures_lite::future::block_on(finished)?;
            Ok(())
        })?;
        assert_eq!(
            weak.strong_count(),
            0,
            "completed resources must release pages"
        );
    }
    Ok(())
}

/// Wait for earlier submissions and their completion callbacks on this queue.
pub(crate) fn drain(queue: &CommandQueueRef) -> Result<()> {
    let compute = Compute {
        queue: queue.retain(),
        idle: Default::default(),
        weights: None,
    };
    futures_lite::future::block_on(compute.submit(|_| {}))
}
