use super::{BufferView, Gpu};
use crate::gpu::Buffer;
use anyhow::{Context, Result, ensure};
use objc2_metal::MTLDevice;

/// Storage request that lets session planning refer to a buffer before its physical placement is known.
#[derive(Clone, Copy)]
pub struct Slot(usize);

struct Region {
    index: usize,
    bytes: usize,
    buffer: usize,
    offset: usize,
}

/// Session storage requirements, collected before allocation so whole bindings can be packed together.
#[derive(Default)]
pub struct Plan {
    regions: Vec<Region>,
}

impl Plan {
    pub fn with_capacity(bindings: usize) -> Self {
        Self {
            regions: Vec::with_capacity(bindings),
        }
    }

    /// Reserve one indivisible shader binding, checking element indices before converting to bytes.
    pub fn slot(&mut self, elements: usize, element_bytes: usize) -> Result<Slot> {
        ensure!(
            elements > 0 && elements <= u32::MAX as usize,
            "arena binding exceeds Metal element index range"
        );
        ensure!(element_bytes > 0, "zero-sized arena element");
        let bytes = elements
            .checked_mul(element_bytes)
            .context("arena byte length overflow")?;
        let index = self.regions.len();
        self.regions.push(Region {
            index,
            bytes,
            buffer: 0,
            offset: 0,
        });
        Ok(Slot(index))
    }

    pub fn allocate(mut self, gpu: &Gpu) -> Result<Arena> {
        let limit = gpu.device.maxBufferLength();
        let budget = usize::try_from(gpu.device.recommendedMaxWorkingSetSize())?;
        self.regions
            .sort_unstable_by_key(|r| (std::cmp::Reverse(r.bytes), r.index));
        let mut lengths: Vec<usize> = Vec::new();
        // Packing largest bindings first, then restoring slot order in the same allocation.
        for region in &mut self.regions {
            ensure!(
                region.bytes <= limit,
                "one arena binding exceeds Metal's buffer limit"
            );
            let mut placement = None;
            for (buffer, used) in lengths.iter().enumerate() {
                let start = used
                    .checked_next_multiple_of(256)
                    .context("arena alignment overflow")?;
                if start <= limit && region.bytes <= limit - start {
                    placement = Some((buffer, start));
                    break;
                }
            }
            let (buffer, start) = placement.unwrap_or_else(|| {
                lengths.push(0);
                (lengths.len() - 1, 0)
            });
            lengths[buffer] = start
                .checked_add(region.bytes)
                .context("arena size overflow")?;
            region.buffer = buffer;
            region.offset = start;
        }
        for bytes in &mut lengths {
            *bytes = (*bytes).max(4);
            ensure!(
                *bytes <= limit,
                "minimum Metal allocation exceeds buffer limit"
            );
        }
        let total = lengths
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .context("aggregate arena size overflow")?;
        ensure!(
            total <= budget,
            "session arenas exceed the per-session working-set budget"
        );
        self.regions.sort_unstable_by_key(|r| r.index);
        let mut buffers = Vec::with_capacity(lengths.len());
        for bytes in lengths {
            let buffer = gpu.new_buffer(bytes);
            buffers.push(buffer);
        }
        Ok(Arena {
            buffers,
            regions: self.regions,
        })
    }
}

/// Allocated backing buffers from which session operations obtain their named storage views.
pub struct Arena {
    buffers: Vec<Buffer>,
    regions: Vec<Region>,
}

impl Arena {
    pub fn get(&self, slot: Slot) -> BufferView {
        let r = &self.regions[slot.0];
        BufferView::new(&self.buffers[r.buffer], r.offset, r.bytes)
    }
}
