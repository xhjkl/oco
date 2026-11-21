use super::*;

#[test]
fn small_f32_projections_preserve_tiny_coefficients_across_model_sizes() -> Result<()> {
    use crate::gpu::{
        dispatch,
        tests::{buffer, read},
    };
    use crate::tensor::Tensor;
    use crate::weights::{compute::Compute, operands::DensePlan};
    use half::{bf16, f16};
    use memmap2::MmapMut;
    use std::sync::Arc;

    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let coefficient = 2.0f32.powi(-30);
    assert_eq!(f16::from_f32(coefficient).to_f32(), 0.0);
    // K = 96 is not a whole number of tensor steps, so tensor GPUs keep its vector rows too.
    for (k, n, m) in [(1024usize, 16usize, 2usize), (6144, 128, 9), (96, 48, 3), (64, 129, 2)] {
        let mut coefficients = vec![0.0f32; k * n];
        for column in 0..n {
            coefficients[column * k + (column * 31) % k] = coefficient;
            coefficients[column * k + (column * 31 + k / 2) % k] = -0.5 * coefficient;
        }
        let bytes = bytemuck::cast_slice(&coefficients);
        let mut mapping = MmapMut::map_anon(bytes.len().next_multiple_of(16384))?;
        mapping[..bytes.len()].copy_from_slice(bytes);
        let mapping = mapping.make_read_only()?;
        let mapping = Arc::new(mapping);
        let storage = gpu.load(&mapping, 0, bytes.len())?;
        let tensor = Tensor::bound(storage, WeightFormat::F32, &[k, n])?;
        let matrix = builder.dense(tensor)?;
        if n > 128 {
            assert!(matches!(
                matrix.plan,
                DensePlan::Tiled { .. } | DensePlan::Tensor { .. }
            ));
            continue;
        }
        // Exact FP32 vector rows, or on tensor GPUs FP32-staged tensor tiles.
        let (pipeline, groups, threads) = match &matrix.plan {
            DensePlan::Vector(vector) => {
                assert_eq!(vector.column_groups, n);
                let expected = &builder.cache[&("matvec_split", WeightFormat::F32, 0)];
                assert_eq!(
                    Retained::as_ptr(&vector.pipeline),
                    Retained::as_ptr(expected)
                );
                (&vector.pipeline, [vector.column_groups, m, 1], 256)
            }
            DensePlan::Tensor { tiles, .. } => {
                let expected = &builder.cache[&("matmul_tensor_f32_32", WeightFormat::F32, 0)];
                assert_eq!(
                    Retained::as_ptr(&tiles[0].pipeline),
                    Retained::as_ptr(expected)
                );
                (&tiles[0].pipeline, [m.div_ceil(32), tiles[0].column_groups, 1], 128)
            }
            _ => panic!("F32 {k}x{n} must not round tiny coefficients through FP16"),
        };
        let input: Vec<_> = (0..m * k)
            .map(|i| bf16::from_f32(((i * 17 + i / k * 5) % 61) as f32 / 32.0 - 0.875).to_bits())
            .collect();
        let expected: Vec<_> = input
            .chunks_exact(k)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    let first = bf16::from_bits(row[(column * 31) % k]).to_f64();
                    let second = bf16::from_bits(row[(column * 31 + k / 2) % k]).to_f64();
                    bf16::from_f64(f64::from(coefficient) * (first - 0.5 * second)).to_bits()
                })
            })
            .collect();
        assert!(expected.iter().any(|&value| value & 0x7fff != 0));
        let input = buffer(&gpu, &input);
        let bias = buffer(&gpu, &[0.0f32]);
        let ids = buffer(&gpu, &[0u32]);
        const GUARD: usize = 16;
        const MARKER: u16 = 0x7fc1;
        let output = buffer(&gpu, &vec![MARKER; GUARD + m * n + GUARD]);
        let params = [k as u32, n as u32, m as u32, 0, 4, 1, 0];
        let submission = compute.submit(|encoder| {
            dispatch(
                encoder,
                pipeline,
                &[
                    matrix.values.binding(),
                    (&input, 0),
                    (&output, GUARD * 2),
                    (&bias, 0),
                    (&ids, 0),
                ],
                bytemuck::cast_slice(&params),
                groups,
                threads,
            );
        });
        futures_lite::future::block_on(submission)?;
        let mut actual = vec![0u16; GUARD + m * n + GUARD];
        read(&output, &mut actual);
        assert!(actual[..GUARD].iter().all(|&value| value == MARKER));
        assert!(actual[GUARD + m * n..].iter().all(|&value| value == MARKER));
        assert_eq!(
            &actual[GUARD..GUARD + m * n],
            expected,
            "F32 {k}x{n}, M={m}"
        );
    }
    Ok(())
}
