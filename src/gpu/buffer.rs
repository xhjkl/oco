use super::Gpu;
use crate::gpu::{Buffer, BufferRef};
use anyhow::{Result, ensure};
use block2::RcBlock;
use bytemuck::Pod;
use memmap2::Mmap;
use objc2::{Message, rc::Retained};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ptr::NonNull;
use std::sync::Arc;

/// Bounded byte range within a session allocation, used by both the host and Metal kernels.
///
/// Multiple views can share one Metal buffer without allocating or copying their bytes.
/// Cloning aliases the same bytes. Callers must finish all GPU access before reading,
/// writing, or reusing them on the host; range checks do not provide synchronization.
/// Session progress enforces that boundary for inference callers.
#[derive(Clone)]
pub struct BufferView {
    buffer: Buffer,
    // Stable shared-memory address, resolved once to keep host access free of Objective-C calls.
    pointer: usize,
    offset: usize,
    length: usize,
}

impl BufferView {
    pub fn new(buffer: &BufferRef, offset: usize, length: usize) -> Self {
        assert!(offset <= buffer.length() && length <= buffer.length() - offset);
        Self {
            buffer: buffer.retain(),
            pointer: buffer.contents().as_ptr() as usize + offset,
            offset,
            length,
        }
    }

    pub fn binding(&self) -> (&BufferRef, usize) {
        (&self.buffer, self.offset)
    }

    pub fn at(&self, offset: usize) -> (&BufferRef, usize) {
        assert!(offset <= self.length);
        (&self.buffer, self.offset + offset)
    }

    pub fn read<T: Pod>(&self, values: &mut [T]) {
        let bytes = bytemuck::cast_slice_mut(values);
        assert!(bytes.len() <= self.length, "read exceeds storage range");
        unsafe {
            std::ptr::copy_nonoverlapping(self.pointer(), bytes.as_mut_ptr(), bytes.len());
        }
    }

    pub fn write<T: Pod>(&self, offset: usize, values: &[T]) {
        let bytes = bytemuck::cast_slice(values);
        assert!(
            offset <= self.length && bytes.len() <= self.length - offset,
            "write exceeds storage range"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.pointer().add(offset), bytes.len());
        }
    }

    pub fn zero(&self) {
        unsafe {
            std::ptr::write_bytes(self.pointer(), 0, self.length);
        }
    }

    fn pointer(&self) -> *mut u8 {
        self.pointer as *mut u8
    }
}

/// Shared view of learned weights backed by the mapped GGUF file.
/// The Metal allocation retains the mapping, including while submitted work outlives
/// the model handle. Neither host code nor shaders may modify these bytes.
#[derive(Clone)]
pub struct ReadOnly {
    buffer: Buffer,
    offset: usize,
    length: usize,
}

impl ReadOnly {
    pub fn binding(&self) -> (&BufferRef, usize) {
        (&self.buffer, self.offset)
    }

    /// Narrow this view to a byte range within it.
    pub fn slice(&self, offset: usize, length: usize) -> ReadOnly {
        assert!(offset <= self.length && length <= self.length - offset);
        ReadOnly {
            buffer: self.buffer.clone(),
            offset: self.offset + offset,
            length,
        }
    }
}

/// Import-time lookup that lets selected tensors share page-aligned Metal registrations.
/// Runtime operands retain their resolved views, so this index can be discarded after loading.
pub struct MappedRegions {
    regions: Vec<(usize, ReadOnly)>,
    pub residency: Arc<super::ResidentWeights>,
}

impl MappedRegions {
    pub fn new(
        gpu: &Gpu,
        map: &Arc<Mmap>,
        spans: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self> {
        let limit = gpu.device.maxBufferLength();
        let page = page_size()?;
        let limit = limit / page * page;
        let mut spans: Vec<_> = spans.into_iter().collect();
        spans.sort_unstable();
        let mut regions = Vec::new();
        let mut index = 0;
        while index < spans.len() {
            let begin = spans[index].0 / page * page;
            let mut end = begin;
            while let Some(&(offset, bytes)) = spans.get(index) {
                ensure!(
                    offset <= map.len() && bytes <= map.len() - offset,
                    "tensor outside mapped storage"
                );
                let next = (offset + bytes)
                    .checked_next_multiple_of(page)
                    .ok_or_else(|| anyhow::anyhow!("mapped range overflow"))?;
                // Splitting at unused pages to limit registration size; this is a resource policy.
                if offset / page * page > end || next - begin > limit {
                    break;
                }
                end = next;
                index += 1;
            }
            ensure!(end > begin, "tensor exceeds Metal's buffer limit");
            let storage = gpu.load(map, begin, (end - begin).min(map.len() - begin))?;
            regions.push((begin, storage));
        }
        let resident = super::ResidentWeights::new(
            &gpu.device,
            regions.iter().map(|(_, region)| &*region.buffer),
        )?;
        let residency = Arc::new(resident);
        Ok(Self { regions, residency })
    }

    pub fn get(&self, offset: usize, bytes: usize) -> Result<ReadOnly> {
        let region = self.regions.iter().find(|(begin, s)| {
            offset >= *begin && offset - begin <= s.length && bytes <= s.length - (offset - begin)
        });
        let (begin, s) =
            region.ok_or_else(|| anyhow::anyhow!("tensor has no containing Metal region"))?;
        Ok(ReadOnly {
            buffer: s.buffer.clone(),
            offset: s.offset + offset - begin,
            length: bytes,
        })
    }
}

fn page_size() -> Result<usize> {
    unsafe extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }
    let page = unsafe { getpagesize() };
    ensure!(
        page > 0 && (page as usize).is_power_of_two(),
        "invalid host page size"
    );
    Ok(page as usize)
}

impl Gpu {
    pub fn new_buffer(&self, bytes: usize) -> Buffer {
        let buffer = self
            .device
            .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .expect("Metal buffer allocation failed");
        // Sharing the immutable handle; access to its bytes requires the session's completion fence.
        unsafe { Retained::cast_unchecked(buffer) }
    }

    /// Load a byte range without copying and retain its pages for the Metal buffer's lifetime.
    /// The returned storage is read-only; shaders must not write to it.
    pub fn load(&self, mapping: &Arc<Mmap>, offset: usize, bytes: usize) -> Result<ReadOnly> {
        ensure!(
            offset <= mapping.len() && bytes <= mapping.len() - offset,
            "buffer outside mapped storage"
        );
        let page = page_size()?;
        ensure!(
            (mapping.as_ptr() as usize).is_multiple_of(page),
            "mmap base is not page aligned"
        );
        let begin = offset / page * page;
        let offset = offset - begin;
        let length = (offset + bytes)
            .checked_next_multiple_of(page)
            .ok_or_else(|| anyhow::anyhow!("mapped range overflow"))?;
        let mapped = mapping
            .len()
            .checked_next_multiple_of(page)
            .ok_or_else(|| anyhow::anyhow!("mapped length overflow"))?;
        // Including only the OS-mapped final partial page; logical tensor ranges remain inside the file.
        ensure!(
            begin <= mapped && length <= mapped - begin,
            "buffer exceeds mapped pages"
        );
        ensure!(
            length <= self.device.maxBufferLength(),
            "buffer exceeds Metal's buffer limit"
        );
        let pointer = unsafe { mapping.as_ptr().add(begin) };
        let mapping = mapping.clone();
        let deallocator = RcBlock::new(move |_, _| {
            // Retaining mapped pages in the block until Metal releases its buffer view.
            let _mapping = &mapping;
        });
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(pointer.cast_mut().cast()).expect("non-null mapped pointer"),
                    length,
                    MTLResourceOptions::StorageModeShared,
                    Some(&deallocator),
                )
        }
        .ok_or_else(|| anyhow::anyhow!("Metal mapping registration failed"))?;
        // Sharing immutable mapped bytes; the deallocator owns their mapping.
        let buffer = unsafe { Retained::cast_unchecked(buffer) };
        Ok(ReadOnly {
            buffer,
            offset,
            length: bytes,
        })
    }
}
