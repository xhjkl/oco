use super::{Bindings, Compute, Gpu};
use crate::{
    gpu::{BufferView, arena::Plan},
    tensor::{Tensor, WeightFormat},
    weights::{operands::Projection, pipelines::Builder, state::CommonScratch},
};
use anyhow::Result;
use half::{bf16, f16};
use memmap2::MmapMut;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTL4CommandQueue, MTLDevice, MTLSharedEvent};
use std::sync::Arc;

type CommandQueueRef = ProtocolObject<dyn MTL4CommandQueue>;
type SharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

/// Borrow the queue to gate operations through inference entry points outside `weights`.
pub(crate) fn command_queue(compute: &Compute) -> &CommandQueueRef {
    &compute.queue
}

#[test]
fn selected_projection_rows_preserve_values_and_output_guards() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let (input, output) = (2880usize, 512usize);
    let kernels = builder.kernels(input, 1e-5, false)?;
    let mut plan = Plan::with_capacity(20);
    let scratch = CommonScratch::plan(&mut plan, 512, input, 64, 1, 2, 1)?;
    let arena = plan.allocate(&gpu)?;
    let scratch = scratch.bind(&arena);
    for format in [WeightFormat::F32, WeightFormat::Q4_0, WeightFormat::Q8_0] {
        let mut coefficients = Vec::new();
        match format {
            WeightFormat::F32 => {
                coefficients = bytemuck::cast_slice(&vec![1.0f32; input * output]).to_vec()
            }
            _ => {
                for _ in 0..input * output / 32 {
                    coefficients.extend(f16::from_f32(1.0).to_bits().to_le_bytes());
                    match format {
                        WeightFormat::Q4_0 => coefficients.extend([0x99u8; 16]),
                        _ => coefficients.extend([1u8; 32]),
                    }
                }
            }
        }
        let storage = gpu.upload(&coefficients);
        let tensor = Tensor::bound(storage, format, &[input, output])?;
        let matrix = builder.dense(tensor)?;
        let projection = Projection { matrix, bias: None };
        // Covering vector and tiled selections, including Q8's 128-thread vector kernel.
        for rows in [2, 129] {
            let row = rows - 1;
            let mut values = vec![bf16::from_f32(1.0).to_bits(); 512 * input];
            values[row * input..(row + 1) * input].fill(bf16::from_f32(2.0).to_bits());
            scratch.norm.write(0, &values);
            // Keeping excess storage guarded; a transposed vector grid stays in bounds and is observable.
            let buffer = gpu.new_buffer((512 * output + 64) * 2);
            let storage = BufferView::new(&buffer, 0, (512 * output + 64) * 2);
            storage.write(0, &vec![0x7fc1u16; 512 * output + 64]);
            let y = storage.slice(64, output * 2);
            let submission = compute.submit(|encoder| {
                let compute = Bindings {
                    kernels: &kernels,
                    scratch: &scratch,
                    encoder,
                };
                projection.apply_row(&compute, &y, rows, row);
            });
            futures_lite::future::block_on(submission)?;
            let mut actual = vec![0u16; 512 * output + 64];
            storage.read(&mut actual);
            let expected = bf16::from_f32((2 * input) as f32).to_bits();
            assert!(
                actual[32..32 + output].iter().all(|&v| v == expected),
                "{format:?}, rows={rows}"
            );
            assert!(
                actual[..32]
                    .iter()
                    .chain(&actual[32 + output..])
                    .all(|&v| v == 0x7fc1),
                "{format:?}, rows={rows}"
            );
        }
    }
    Ok(())
}

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
