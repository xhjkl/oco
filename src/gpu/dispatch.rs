use super::{Buffer, BufferRef, Pipeline, profile};
use anyhow::{Result, anyhow};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::*;
use std::{cell::RefCell, collections::HashMap};

const PARAMETER_PAGE: usize = 64 * 1024;
/// Timestamps per submission when profiling; later dispatches go unstamped.
const PROFILE_STAMPS: usize = 4096;

/// Reusable command storage, leased exclusively until its GPU completion callback.
/// Parameter pages keep every dispatch's constants at a distinct, stable GPU address.
/// Cached resource owners cover direct and indirect addresses; Metal 4 does not retain them.
pub struct CommandMemory {
    pub allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    pub command: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    buffers: HashMap<usize, Buffer>,
    pipelines: HashMap<usize, Pipeline>,
    parameters: Vec<Buffer>,
    cursor: usize,
    weights: Option<std::sync::Arc<super::ResidentWeights>>,
    /// Optional GPU timestamp heap and the label of each stamped dispatch, in order.
    heap: Option<Retained<ProtocolObject<dyn MTL4CounterHeap>>>,
    labels: Vec<String>,
}

// Moving exclusive ownership between encoding and completion threads; none of the
// non-thread-safe command objects are accessed concurrently or exposed to callers.
unsafe impl Send for CommandMemory {}

impl CommandMemory {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        weights: Option<std::sync::Arc<super::ResidentWeights>>,
    ) -> Result<Self> {
        let allocator = device.newCommandAllocator();
        let allocator =
            allocator.ok_or_else(|| anyhow!("Metal command allocator creation failed"))?;
        let command = device.newCommandBuffer();
        let command = command.ok_or_else(|| anyhow!("Metal command buffer creation failed"))?;
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(31);
        let table = device.newArgumentTableWithDescriptor_error(&descriptor)?;
        let residency =
            device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())?;
        let heap = if profile::enabled() {
            let descriptor = MTL4CounterHeapDescriptor::new();
            descriptor.setType(MTL4CounterHeapType::Timestamp);
            unsafe { descriptor.setCount(PROFILE_STAMPS) };
            Some(device.newCounterHeapWithDescriptor_error(&descriptor)?)
        } else {
            None
        };
        Ok(Self {
            allocator,
            command,
            table,
            residency,
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            parameters: Vec::new(),
            cursor: 0,
            weights,
            heap,
            labels: Vec::new(),
        })
    }

    pub fn begin(&mut self) {
        self.allocator.reset();
        self.cursor = 0;
        self.labels.clear();
        self.command
            .beginCommandBufferWithAllocator(&self.allocator);
    }

    /// Read back this submission's timestamps after GPU completion and fold them into the profile.
    pub fn resolve_profile(&mut self) {
        let Some(heap) = &self.heap else { return };
        if self.labels.is_empty() {
            return;
        }
        let count = self.labels.len() + 1;
        let data = unsafe { heap.resolveCounterRange(NSRange::new(0, count)) };
        if let Some(data) = data {
            let bytes = data.to_vec();
            let stamps: Vec<u64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| u64::from_le_bytes(*chunk))
                .collect();
            profile::record(&self.labels, &stamps);
        }
        unsafe { heap.invalidateCounterRange(NSRange::new(0, count)) };
        self.labels.clear();
    }

    pub fn end(&self) {
        self.residency.commit();
        self.command.useResidencySet(&self.residency);
        if let Some(weights) = &self.weights {
            self.command
                .useResidencySet(ProtocolObject::from_ref(&*weights.set));
        }
        self.command.endCommandBuffer();
    }

    fn retain(&mut self, buffer: &BufferRef) {
        let key = std::ptr::from_ref(buffer) as usize;
        if self
            .weights
            .as_ref()
            .is_some_and(|weights| weights.allocations.contains_key(&key))
        {
            return;
        }
        self.buffers.entry(key).or_insert_with(|| {
            self.residency
                .addAllocation(ProtocolObject::from_ref(buffer));
            buffer.retain()
        });
    }

    fn parameters(&mut self, bytes: &[u8]) -> u64 {
        let length = bytes.len().max(1).next_multiple_of(16);
        assert!(
            length <= PARAMETER_PAGE,
            "dispatch parameter block exceeds page size"
        );
        if self.cursor % PARAMETER_PAGE + length > PARAMETER_PAGE {
            self.cursor = self.cursor.next_multiple_of(PARAMETER_PAGE);
        }
        let page = self.cursor / PARAMETER_PAGE;
        let offset = self.cursor % PARAMETER_PAGE;
        if page == self.parameters.len() {
            let device = self.command.device();
            let buffer = device
                .newBufferWithLength_options(PARAMETER_PAGE, MTLResourceOptions::StorageModeShared);
            let buffer = buffer.expect("Metal parameter allocation failed");
            // The submission exclusively writes these bytes before commit.
            let buffer: Buffer = unsafe { Retained::cast_unchecked(buffer) };
            self.retain(&buffer);
            self.parameters.push(buffer);
        }
        let buffer = &self.parameters[page];
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
        self.cursor += length;
        buffer.gpuAddress() + offset as u64
    }
}

/// Encoding-time access to leased Metal command memory and its argument table.
/// This context carries resource lifetimes and copied constants, never inference progress.
pub struct Commands {
    pub encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    pub memory: RefCell<CommandMemory>,
}

impl Commands {
    /// Retain and make resident an allocation reached through a GPU address inside another buffer.
    pub fn resident(&self, buffer: &BufferRef) {
        self.memory.borrow_mut().retain(buffer);
    }

    /// Stamp the submission start when profiling.
    pub fn stamp_start(&self) {
        let memory = self.memory.borrow();
        if let Some(heap) = &memory.heap {
            unsafe {
                self.encoder.writeTimestampWithGranularity_intoHeap_atIndex(
                    MTL4TimestampGranularity::Precise,
                    heap,
                    0,
                );
            }
        }
    }

    /// Stamp the completion of the dispatch just encoded, attributing it to its pipeline.
    fn stamp(&self, pipeline: &Pipeline) {
        let mut memory = self.memory.borrow_mut();
        let Some(heap) = &memory.heap else { return };
        let index = memory.labels.len() + 1;
        if index >= PROFILE_STAMPS {
            return;
        }
        unsafe {
            self.encoder.writeTimestampWithGranularity_intoHeap_atIndex(
                MTL4TimestampGranularity::Precise,
                heap,
                index,
            );
        }
        let label = pipeline
            .label()
            .map(|label| label.to_string())
            .unwrap_or_else(|| "unnamed".to_owned());
        memory.labels.push(label);
    }
}

pub fn dispatch(
    commands: &Commands,
    pipeline: &Pipeline,
    buffers: &[(&BufferRef, usize)],
    params: &[u8],
    groups: [usize; 3],
    threads: usize,
) {
    bind(commands, pipeline, buffers, params, true);
    commands.encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups[0],
            height: groups[1],
            depth: groups[2],
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    commands.stamp(pipeline);
}

/// Dispatch without waiting for earlier work; the caller guarantees independence from every
/// dispatch since the previous barrier, which the next ordered dispatch still waits for.
pub fn dispatch_concurrent(
    commands: &Commands,
    pipeline: &Pipeline,
    buffers: &[(&BufferRef, usize)],
    params: &[u8],
    groups: [usize; 3],
    threads: usize,
) {
    bind(commands, pipeline, buffers, params, false);
    commands.encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups[0],
            height: groups[1],
            depth: groups[2],
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    commands.stamp(pipeline);
}

pub fn dispatch_indirect(
    commands: &Commands,
    pipeline: &Pipeline,
    buffers: &[(&BufferRef, usize)],
    params: &[u8],
    (buffer, offset): (&BufferRef, usize),
    threads: usize,
) {
    commands.resident(buffer);
    bind(commands, pipeline, buffers, params, true);
    // Retaining the indirect grid until completion; the dispatch barrier covers its producer.
    unsafe {
        commands
            .encoder
            .dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(
                buffer.gpuAddress() + offset as u64,
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
    }
    commands.stamp(pipeline);
}

/// Bind operands and copied parameters, ordering access unless the caller guarantees independence.
fn bind(
    commands: &Commands,
    pipeline: &Pipeline,
    buffers: &[(&BufferRef, usize)],
    params: &[u8],
    ordered: bool,
) {
    assert!(buffers.len() < 31);
    let mut memory = commands.memory.borrow_mut();
    let key = Retained::as_ptr(pipeline) as usize;
    memory
        .pipelines
        .entry(key)
        .or_insert_with(|| pipeline.clone());
    // Covering producer/consumer hazards and scratch aliases, including GPU-generated indirect grids.
    if ordered {
        commands
            .encoder
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
    }
    commands.encoder.setComputePipelineState(pipeline);
    for (index, &(buffer, offset)) in buffers.iter().enumerate() {
        memory.retain(buffer);
        unsafe {
            memory
                .table
                .setAddress_atIndex(buffer.gpuAddress() + offset as u64, index);
        }
    }
    let address = memory.parameters(params);
    unsafe {
        memory.table.setAddress_atIndex(address, buffers.len());
    }
    commands.encoder.setArgumentTable(Some(&memory.table));
}
