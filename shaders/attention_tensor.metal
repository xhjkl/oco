// Prompt attention on Metal 4 tensor operations (Apple10), after FlashAttention-2 and
// llama.cpp's tensor kernel: 32 queries of one head per threadgroup, 128 keys per step, eight
// SIMD groups. Queries, keys and values feed the operations straight from device memory (BF16
// queries, half cache), and the score scale applies to the FP32 scores, which leaves the
// threadgroup memory to 128-key score and probability tiles. The online softmax runs in FP32
// with half probabilities and lazy rescaling (the running maximum moves only when it grows by
// more than 8, so probabilities stay within half), and outputs accumulate in an FP32 cooperative
// tensor. Causality, windows and sinks follow the vector kernel; the key range starts exactly at
// the first row's window so expired cache rows never enter a product.
template <uint D>
[[kernel, max_total_threads_per_threadgroup(256)]] void
attention_tensor(const device bfloat *q [[buffer(0)]], device half *k [[buffer(1)]],
                    device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
                    device bfloat *out [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
                    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int Q = 32, C = 128, NSG = 8, NQ = Q / NSG, NC = C / 32;
    threadgroup float ss[Q * C];
    threadgroup half sp[Q * C];
    threadgroup float sr[Q];
    threadgroup atomic_uint rescaled;
    const uint h = group.x, t0 = group.y * Q, kv = h / (p.heads / p.kv_heads);
    const uint rows = min(uint(Q), p.m - t0);
    const uint end = key_end(p, t0 + rows - 1);
    const uint first_end = key_end(p, t0);
    const uint begin = p.window && first_end > p.window ? first_end - p.window : 0;
    float peak[NQ], mass[NQ];
#pragma unroll
    for (int jj = 0; jj < NQ; ++jj) {
        peak[jj] = -FLT_MAX / 2;
        mass[jj] = 0;
    }
    const int stride = int(p.kv_heads * D);
    device half *keys = k + kv * D, *values = v + kv * D;
    auto tq = tensor(const_cast<device bfloat *>(q) + (t0 * p.heads + h) * D, dextents<int32_t, 2>(D, int(rows)),
                     array<int, 2>({1, int(p.heads * D)}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(C, Q));
    auto tp = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(C, Q));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, D, false, true, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, D, static_cast<int>(dynamic_extent), false, false, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;
    auto tv0 = tensor(values, dextents<int32_t, 2>(D, C), array<int, 2>({1, stride}));
    // Destination cooperative tensors start at zero; explicit clearing measured up to 19% slower
    // in the expert tiles.
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    if (tid == 0)
        atomic_store_explicit(&rescaled, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint step = 0;
    for (uint ic = begin; ic < end; ic += C, ++step) {
        const int live = int(min(uint(C), end - ic));
        auto tk = tensor(keys + ic * stride, dextents<int32_t, 2>(D, live), array<int, 2>({1, stride}));
        mm_qk.run(tq, tk, ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (int jj = 0; jj < NQ; ++jj) {
            const uint j = jj * NSG + sg, t = t0 + j;
            const uint row_end = key_end(p, t);
            const uint row_begin = p.window && row_end > p.window ? row_end - p.window : 0;
            float s[NC];
            float m = peak[jj];
#pragma unroll
            for (int ii = 0; ii < NC; ++ii) {
                uint column = ii * 32 + lane, key = ic + column;
                bool visible = j < rows && int(column) < live && key < row_end && key >= row_begin;
                s[ii] = visible ? ss[j * C + column] * p.scale : -INFINITY;
                m = max(m, s[ii]);
            }
            m = simd_max(m);
            float scale = 1.0f;
            if (m > peak[jj] + 8.0f) {
                scale = fast::exp(peak[jj] - m);
                peak[jj] = m;
                if (lane == 0)
                    atomic_store_explicit(&rescaled, step + 1, memory_order_relaxed);
            }
            float sum = 0;
#pragma unroll
            for (int ii = 0; ii < NC; ++ii) {
                half probability = half(fast::exp(s[ii] - peak[jj]));
                sp[j * C + ii * 32 + lane] = probability;
                sum += float(probability);
            }
            mass[jj] = mass[jj] * scale + simd_sum(sum);
            if (lane == 0)
                sr[j] = scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&rescaled, memory_order_relaxed) == step + 1) {
#pragma unroll
            for (uint16_t i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i))
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
        auto tpl = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            sp, dextents<int32_t, 2>(live, Q), array<int, 2>({1, C}));
        auto tv = tensor(values + ic * stride, dextents<int32_t, 2>(D, live), array<int, 2>({1, stride}));
        mm_pv.run(tpl, tv, co);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (int jj = 0; jj < NQ; ++jj) {
        const uint j = jj * NSG + sg;
        float scale = 1.0f;
        if (p.sinks) {
            float sink = float(bfloat(sinks[h])), m = max(peak[jj], sink);
            scale = fast::exp(peak[jj] - m);
            mass[jj] = mass[jj] * scale + fast::exp(sink - m);
        }
        if (lane == 0)
            sr[j] = mass[jj] == 0.0f ? 0.0f : scale / mass[jj];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint16_t i = 0; i < co.get_capacity(); ++i) {
        if (!co.is_valid_element(i))
            continue;
        auto at = co.get_multidimensional_index(i);
        uint d = at[0], j = at[1];
        if (j < rows)
            out[((t0 + j) * p.heads + h) * D + d] = bfloat(co[i] * sr[j]);
    }
}

#define ATTENTION_TENSOR(D)                                                                        \
    template [[host_name("attention_tensor_" #D)]] kernel void attention_tensor<D>(                \
        const device bfloat *, device half *, device half *, const device float *, device bfloat *, \
        constant AttentionParams &, uint3, ushort, ushort, ushort);
ATTENTION_TENSOR(64)
ATTENTION_TENSOR(128)
ATTENTION_TENSOR(256)
