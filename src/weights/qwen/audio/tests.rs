use super::*;
use crate::{gpu::MappedRegions, tensor::Tensor};
use memmap2::MmapMut;

/// Build a small convolution/projector graph for audio submission lifecycle tests.
pub(crate) fn encoder(gpu: &Gpu, output: usize) -> Result<AudioEncoder> {
    let mut builder = crate::weights::pipelines::Builder::new(gpu)?;
    let projection =
        |builder: &mut crate::weights::pipelines::Builder, input, output| -> Result<Projection> {
            let storage = gpu.upload(&vec![0u8; input * output * 2]);
            let tensor = Tensor::bound(storage, WeightFormat::F16, &[input, output])?;
            let matrix = builder.dense(tensor)?;
            Ok(Projection { matrix, bias: None })
        };
    let mut convolution = |frames, outputs| -> Result<Conv> {
        let geometry = Im2col::new(1, frames, 1, (1, 1), 2, (0, 0));
        let weights = projection(&mut builder, geometry.k as usize, outputs)?;
        Ok(Conv { weights, geometry })
    };
    let first = convolution(100, 1)?;
    let second = convolution(50, 1)?;
    let third = convolution(25, 64)?;
    let convolutions = [first, second, third];
    let conv_out = projection(&mut builder, 64, 64)?;
    let up = projection(&mut builder, 64, 64)?;
    let down = projection(&mut builder, 64, output)?;
    let projector = GeluFfn {
        up,
        down,
        form: GeluForm::Erf,
    };
    let attention = builder.encoder_attention(64, WINDOW_TOKENS)?;
    let kernels = builder.kernels(64, 1e-5, false)?;
    let pipelines = builder.audio()?;
    let map = MmapMut::map_anon(16384)?;
    let map = map.make_read_only()?;
    let map = Arc::new(map);
    let regions = MappedRegions::new(gpu, &map, [])?;
    let positions = gpu.upload(&vec![0u8; CHUNK_TOKENS * 64 * 4]);
    let weight = gpu.upload(bytemuck::cast_slice(&[1.0f32; 64]));
    let bias = gpu.upload(&[0u8; 64 * 4]);
    Ok(AudioEncoder {
        kernels,
        pipelines,
        attention: AttentionPlan::new(Rope::unrotated(1, 1, 64), attention),
        layer_norm: LayerNorm {
            n: 64,
            epsilon: 1e-5,
        },
        convolutions,
        conv_out,
        positions,
        layers: Vec::new(),
        post_norm: LayerNormWeights { weight, bias },
        projector,
        mels: 1,
        width: 64,
        ffn_width: 64,
        output,
        _residency: regions.residency,
    })
}
