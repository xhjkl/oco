use super::{GUARD, SENTINEL};
use crate::{
    gpu::{
        Gpu, dispatch,
        tests::{buffer, read},
    },
    tensor::WeightFormat,
    weights::{compute::Compute, pipelines::Builder},
};
use anyhow::Result;
use half::bf16;

#[test]
fn gpu_argmax_rejects_nonfinite_logits_and_preserves_finite_ordering() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let pipeline = builder.get("argmax", WeightFormat::F32, 0)?;
    let mut cases = Vec::new();
    for bits in [0x7f80, 0xff80, 0x7fc1, 0xffc1, 0x7f81] {
        let mut input = vec![bf16::from_f32(2.0).to_bits(); 2051];
        input[2049] = bits;
        cases.push((input, u32::MAX));
    }
    cases.extend([
        (vec![0xff7f], 0),
        (vec![0xff7f, 0x7f7f], 1),
        (vec![0x8000, 0, 0x8001], 0),
        (vec![0x8001, 0, 1, 2], 3),
    ]);
    for tied in [[31, 32], [1023, 1024], [1024, 2048]] {
        let mut input = vec![bf16::from_f32(-2.0).to_bits(); 2051];
        for &index in &tied {
            input[index] = bf16::from_f32(3.0).to_bits();
        }
        cases.push((input, tied[0] as u32));
    }
    for (input, expected) in cases {
        let x = buffer(&gpu, &input);
        let mut output = vec![SENTINEL; 1 + 2 * GUARD];
        let y = buffer(&gpu, &output);
        let n = input.len() as u32;
        let submission = compute.submit(|encoder| {
            dispatch(
                encoder,
                &pipeline,
                &[(&x, 0), (&y, GUARD * 4)],
                bytemuck::bytes_of(&n),
                [1, 1, 1],
                1024,
            );
        });
        futures_lite::future::block_on(submission)?;
        read(&y, &mut output);
        assert!(output[..GUARD].iter().all(|&word| word == SENTINEL));
        assert!(output[GUARD + 1..].iter().all(|&word| word == SENTINEL));
        assert_eq!(
            output[GUARD],
            expected,
            "argmax n={}, first nonfinite={:?}",
            input.len(),
            input.iter().position(|&bits| bits & 0x7f80 == 0x7f80)
        );
    }
    Ok(())
}
