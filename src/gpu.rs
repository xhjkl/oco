//! Metal execution and raw storage, independent of tensor formats and model architecture.
pub(crate) mod arena;
mod buffer;
mod dispatch;
pub(crate) mod profile;
#[cfg(test)]
pub(crate) mod tests;

use anyhow::{Result, anyhow, ensure};
pub use buffer::{BufferView, MappedRegions, ReadOnly};
pub use dispatch::{CommandMemory, Commands, dispatch, dispatch_concurrent, dispatch_indirect};
use dispatch2::DispatchData;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{collections::HashMap, ptr::NonNull};

// Loading CoreGraphics registers the default Metal device in command-line processes.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

// Buffer handles have immutable allocation metadata; CPU/GPU byte access is fenced by Session.
pub type BufferRef = ProtocolObject<dyn MTLBuffer + Send + Sync>;
pub type Buffer = Retained<BufferRef>;
pub type LibraryRef = ProtocolObject<dyn MTLLibrary>;
pub type Library = Retained<LibraryRef>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Device used to register mapped weights, prepare immutable kernels, and allocate session buffers.
pub struct Gpu {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice();
        let device = device.ok_or_else(|| anyhow!("a Metal GPU is required"))?;
        let supported = device.supportsFamily(MTLGPUFamily::Metal4);
        ensure!(supported, "a Metal 4 GPU is required");
        Ok(Self { device })
    }

    /// Load precompiled Metal IR supplied by the caller.
    pub fn new_library(&self, bytes: &[u8]) -> Result<Library> {
        let data = DispatchData::from_bytes(bytes);
        let library = self.device.newLibraryWithData_error(&data)?;
        Ok(library)
    }

    /// Specialize a library function and create its device-specific pipeline.
    pub fn new_pipeline(
        compiler: &ProtocolObject<dyn MTL4Compiler>,
        library: &LibraryRef,
        symbol: &str,
        constants: &[(u64, u32)],
    ) -> Result<Pipeline> {
        objc2::rc::autoreleasepool(|_| {
            let values = MTLFunctionConstantValues::new();
            for &(index, value) in constants {
                // Copying scalar constants with the types declared by the shader.
                unsafe {
                    values.setConstantValue_type_atIndex(
                        NonNull::from(&value).cast(),
                        MTLDataType::UInt,
                        index as usize,
                    );
                }
            }
            let name = NSString::from_str(symbol);
            let function = MTL4LibraryFunctionDescriptor::new();
            function.setLibrary(Some(library));
            function.setName(Some(&name));
            let specialized = MTL4SpecializedFunctionDescriptor::new();
            specialized.setFunctionDescriptor(Some(&function));
            specialized.setConstantValues(Some(&values));
            let descriptor = MTL4ComputePipelineDescriptor::new();
            descriptor.setComputeFunctionDescriptor(Some(&specialized));
            // Naming pipelines by symbol and specialization for the optional kernel profile.
            let suffix: Vec<String> = constants
                .iter()
                .map(|(_, value)| value.to_string())
                .collect();
            let label = NSString::from_str(&format!("{symbol}:{}", suffix.join("/")));
            descriptor.setLabel(Some(&label));
            let pipeline = compiler
                .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(
                    &descriptor,
                    None,
                )?;
            Ok(pipeline)
        })
    }
}

/// Immutable weight allocations shared by all commands without repeated residency registration.
pub(crate) struct ResidentWeights {
    pub set: Retained<ProtocolObject<dyn MTLResidencySet + Send + Sync>>,
    pub allocations: HashMap<usize, Buffer>,
}

impl ResidentWeights {
    fn new<'a>(
        device: &ProtocolObject<dyn MTLDevice>,
        buffers: impl Iterator<Item = &'a BufferRef>,
    ) -> Result<Self> {
        let set = device.newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())?;
        let mut allocations = HashMap::new();
        for buffer in buffers {
            set.addAllocation(ProtocolObject::from_ref(buffer));
            let buffer = buffer.retain();
            allocations.insert(Retained::as_ptr(&buffer) as usize, buffer);
        }
        set.commit();
        set.requestResidency();
        // Publishing only after construction; commands read the set until its final owner drops.
        let set = unsafe { Retained::cast_unchecked(set) };
        Ok(Self { set, allocations })
    }
}

impl Drop for ResidentWeights {
    fn drop(&mut self) {
        self.set.endResidency();
    }
}
