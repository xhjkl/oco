use crate::gpu::ReadOnly;
use anyhow::Result;

/// Encodings used to read learned coefficients from the checkpoint inside Metal kernels.
/// Quantized formats pack groups of coefficients with reconstruction data; their
/// block layout determines storage size and which kernel can interpret the bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum WeightFormat {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Bf16 = 30,
    Mxfp4 = 39,
}

impl WeightFormat {
    /// Return elements and bytes per GGUF block.
    pub fn block_layout(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Mxfp4 => (32, 17),
        }
    }
}

/// Imported weight array whose dimensions and encoding guide preparation of an execution operand.
/// The operand keeps the shared storage view; the file's tensor name is no longer needed.
pub struct Tensor {
    pub(crate) storage: ReadOnly,
    pub shape: [usize; 3],
    pub kind: WeightFormat,
}

impl Tensor {
    pub fn bound(storage: ReadOnly, kind: WeightFormat, shape: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            !shape.is_empty() && shape.len() <= 3,
            "unsupported runtime tensor geometry"
        );
        let mut geometry = [1; 3];
        geometry[..shape.len()].copy_from_slice(shape);
        Ok(Self {
            storage,
            kind,
            shape: geometry,
        })
    }
}
