// Dense products on the Metal 4 tensor operations of GPUs with neural accelerators (Apple10):
// NT token rows × 64 output columns per threadgroup, K steps of 64. Each thread decodes one
// 32-coefficient block of one column per step, staged [column][k] in threadgroup memory as S:
// half with relaxed precision for encoded weights, exact float for FP32 weights (tiny recurrent
// coefficients). BF16 activations feed the operation straight from device memory, so the half
// operand range needs no row gains. FP32 sums take the bias and round to BF16 in place. A grid of
// more than one layer splits K across them for products with too few tiles to fill the GPU: each
// split stores raw FP32 sums through `y` into layer-major partials for matmul_split_sum.
template <int NT, typename S>
[[kernel, max_total_threads_per_threadgroup(128)]] void
matmul_tensor(const device uchar *w [[buffer(0)]], device bfloat *x [[buffer(1)]],
              device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
              constant Matrix &p [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]],
              uint3 grid [[threadgroups_per_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NR = 64, NK = 64;
    const bool Q4 = weight_type == WeightFormat::Q4_0 || weight_type == WeightFormat::Q4_1;
    threadgroup S staged[NR * NK];
    const uint n0 = group.y * NR, m0 = group.x * NT;
    // Tail columns repeat the last valid one and are never stored.
    const ushort wn = tid / 2, offset = tid % 2 * 32;
    const device uchar *column = w + ulong(min(n0 + wn, p.n - 1)) * packed_row_bytes(p.k);
    const uint block_bytes = weight_type == WeightFormat::Q4_0 ? 18 : 20;
    threadgroup vec<S, 4> *slot = reinterpret_cast<threadgroup vec<S, 4> *>(staged + wn * NK + offset);
    device bfloat *rows = x + ulong(m0) * p.k;
    const int live = int(min(uint(NT), p.m - m0));
    auto tw = tensor<threadgroup S, dextents<int32_t, 2>, tensor_inline>(staged, dextents<int32_t, 2>(NK, NR));
    auto tx = tensor(rows, dextents<int32_t, 2>(NK, live), array<int, 2>({1, int(p.k)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NT, NR, NK, false, true, is_same<S, half>::value,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    // Destination cooperative tensors start at zero; explicit clearing measured up to 19% slower
    // in the expert tiles.
    auto sums = mm.template get_destination_cooperative_tensor<decltype(tx), decltype(tw), float>();
    // The builder routes only K divisible by 64 here. Q4 blocks are fetched one step ahead so
    // their device loads overlap the current step's products.
    const uint span = p.k / grid.z, begin = group.z * span, end = begin + span;
    Q4Raw raw;
    if (Q4)
        raw = q4_fetch(column + (begin + offset) / 32 * block_bytes, 0);
    for (uint k0 = begin; k0 < end; k0 += NK) {
        uint k = k0 + offset;
        vec<S, 4> values[8];
        if (Q4) {
#pragma unroll
            for (short h = 0; h < 2; ++h) {
                half4x4 decoded = q4_decode(raw, h);
#pragma unroll
                for (short j = 0; j < 4; ++j)
                    values[h * 4 + j] = vec<S, 4>(decoded[j]);
            }
            if (k0 + NK < end)
                raw = q4_fetch(column + (k + NK) / 32 * block_bytes, 0);
        } else if (weight_type == WeightFormat::F32) {
            // GGUF guarantees only four-byte alignment for F32 tensors.
            const device packed_float4 *f = reinterpret_cast<const device packed_float4 *>(column) + k / 4;
#pragma unroll
            for (short j = 0; j < 8; ++j)
                values[j] = vec<S, 4>(float4(f[j]));
        } else {
            float decoded[16];
#pragma unroll
            for (short h = 0; h < 2; ++h) {
                decode_weights16(column, k + h * 16, decoded);
#pragma unroll
                for (short j = 0; j < 4; ++j)
                    values[h * 4 + j] = vec<S, 4>(decoded[4 * j], decoded[4 * j + 1], decoded[4 * j + 2], decoded[4 * j + 3]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (short j = 0; j < 8; ++j)
            slot[j] = values[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto step = tensor(rows + k0, dextents<int32_t, 2>(NK, live), array<int, 2>({1, int(p.k)}));
        mm.run(step, tw, sums);
    }
#pragma unroll
    for (uint16_t i = 0; i < sums.get_capacity(); ++i) {
        if (!sums.is_valid_element(i))
            continue;
        auto at = sums.get_multidimensional_index(i);
        uint n = n0 + at[0], m = m0 + at[1];
        if (n >= p.n || m >= p.m)
            continue;
        if (grid.z > 1)
            reinterpret_cast<device float *>(y)[(group.z * p.m + m) * p.n + n] = sums[i];
        else
            y[m * p.n + n] = bfloat(sums[i] + (p.has_bias ? float(bfloat(bias[n])) : 0));
    }
}

#define MATMUL_TENSOR(NAME, NT, S)                                                                \
    template [[host_name(NAME)]] kernel void matmul_tensor<NT, S>(                                 \
        const device uchar *, device bfloat *, device bfloat *, const device float *,              \
        constant Matrix &, uint3, uint3, ushort);
MATMUL_TENSOR("matmul_tensor_32", 32, half)
MATMUL_TENSOR("matmul_tensor_64", 64, half)
MATMUL_TENSOR("matmul_tensor_128", 128, half)
MATMUL_TENSOR("matmul_tensor_f32_32", 32, float)
MATMUL_TENSOR("matmul_tensor_f32_64", 64, float)
MATMUL_TENSOR("matmul_tensor_f32_128", 128, float)



