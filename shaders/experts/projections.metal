#include "../matrix/mxfp4-dot.metal"

// Clipped SwiGLU of one (gate, up) pair, matching the standalone gate kernel's arithmetic.
inline float swiglu_clipped(float g, float u) {
    if ((as_type<uint>(g) & 0x7f800000u) == 0x7f800000u ||
        (as_type<uint>(u) & 0x7f800000u) == 0x7f800000u)
        return as_type<float>(0x7fc00000u);
    g = min(g, 7.0f);
    u = clamp(u, -6.0f, 8.0f);
    float e = fast::exp(-abs(g) * 1.702f);
    float sigmoid = (g < 0 ? e : 1.0f) / (1.0f + e);
    return sigmoid * g * u;
}

// Each code byte maps to its two signed E2M1 coefficients (low nibble: weight j, high
// nibble: weight j + 16) through one threadgroup table lookup. The block's power-of-two
// scale applies once after the block's products.
// Decode finite E8M0 powers exactly; the reserved255 header represents NaN, not infinity.
inline float expert_mxfp4_scale(uint exponent) {
    if (exponent == 255)
        return as_type<float>(0x7fc00000u);
    return exponent == 0 ? as_type<float>(0x00400000u) : as_type<float>(exponent << 23);
}

__attribute__((always_inline)) inline void mxfp4_fill_table(threadgroup float2 *lut, uint tid) {
    constexpr float magnitudes[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    uint lo = tid & 15, hi = tid >> 4;
    float a = magnitudes[lo & 7], b = magnitudes[hi & 7];
    lut[tid] = float2(lo & 8 ? -a : a, hi & 8 ? -b : b);
}

__attribute__((always_inline)) inline float
mxfp4_half_block(const device uchar *block, uint half_index, thread const float *lo,
                 thread const float *hi, const threadgroup float2 *lut) {
    const device packed_uchar4 *codes =
        reinterpret_cast<const device packed_uchar4 *>(block + 1 + half_index * 8);
    uchar4 c0 = uchar4(codes[0]), c1 = uchar4(codes[1]);
    float2 v0 = lut[c0.x], v1 = lut[c0.y], v2 = lut[c0.z], v3 = lut[c0.w];
    float2 v4 = lut[c1.x], v5 = lut[c1.y], v6 = lut[c1.z], v7 = lut[c1.w];
    float acc = 0;
    acc = fma(lo[0], v0.x, acc);
    acc = fma(lo[1], v1.x, acc);
    acc = fma(lo[2], v2.x, acc);
    acc = fma(lo[3], v3.x, acc);
    acc = fma(lo[4], v4.x, acc);
    acc = fma(lo[5], v5.x, acc);
    acc = fma(lo[6], v6.x, acc);
    acc = fma(lo[7], v7.x, acc);
    acc = fma(hi[0], v0.y, acc);
    acc = fma(hi[1], v1.y, acc);
    acc = fma(hi[2], v2.y, acc);
    acc = fma(hi[3], v3.y, acc);
    acc = fma(hi[4], v4.y, acc);
    acc = fma(hi[5], v5.y, acc);
    acc = fma(hi[6], v6.y, acc);
    acc = fma(hi[7], v7.y, acc);
    uint exponent = block[0];
    float scale = expert_mxfp4_scale(exponent);
    return acc * scale;
}

// Four rows over the blocks one lane pair covers; the activation loader fills sixteen values.
template <typename Load>
__attribute__((always_inline)) inline void
mxfp4_rows(const device uchar *const rows[4], uint blocks, uint ix, uint half_index,
           thread float *sums, const threadgroup float2 *lut, Load load) {
    for (uint ib = ix; ib < blocks; ib += 16) {
        float lo[8], hi[8];
        load(ib, lo, hi);
#pragma unroll
        for (uint r = 0; r < 4; r++)
            sums[r] += mxfp4_half_block(rows[r] + ib * 17, half_index, lo, hi, lut);
    }
}

// Share normalized input loads across two up rows and their two gate rows per SIMD group.
// The clipped SwiGLU of each column is applied here for the following contraction.
[[kernel, max_total_threads_per_threadgroup(256)]] void
expert_expand(const device uchar *uw [[buffer(0)]], const device uchar *gw [[buffer(1)]],
              const device bfloat *x [[buffer(2)]], device float *uy [[buffer(3)]],
              device float *gy [[buffer(4)]], const device float *ub [[buffer(5)]],
              const device float *gb [[buffer(6)]], device uint *ids [[buffer(7)]],
              const device float *logits [[buffer(8)]], device float *probs [[buffer(9)]],
              constant Matrix &p [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]],
              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 lut[256];
    mxfp4_fill_table(lut, tid);
    // Every SIMD group selects the experts from the shared logits; one lane records them.
    uint selected[4];
    float weights[4];
    select_experts(logits, lane, selected, weights);
    if (group.x == 0 && group.y == 0 && tid == 0) {
        for (uint r = 0; r < 4; r++) {
            ids[r] = selected[r];
            probs[r] = weights[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = group.x * 16 + sg * 2, route = group.y;
    uint expert = route == 0 ? selected[0] : route == 1 ? selected[1] : route == 2 ? selected[2] : selected[3];
    uint blocks = p.k / 32, stride = blocks * 17;
    const device uchar *rows[4];
#pragma unroll
    for (uint r = 0; r < 2; r++) {
        uint clamped = min(row + r, p.n - 1);
        rows[r] = uw + (expert * p.n + clamped) * stride;
        rows[2 + r] = gw + (expert * p.n + clamped) * stride;
    }
    uint ix = lane / 2, half_index = lane % 2;
    const device bfloat *xv = x + (route / p.routes) * p.k + half_index * 8;
    float sums[4] = {0, 0, 0, 0};
    mxfp4_rows(rows, blocks, ix, half_index, sums, lut, [&](uint ib, thread float *lo, thread float *hi) {
        const device bfloat *yb = xv + ib * 32;
        uint4 packed_lo = *reinterpret_cast<const device uint4 *>(yb);
        uint4 packed_hi = *reinterpret_cast<const device uint4 *>(yb + 16);
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            lo[2 * j] = as_type<float>(packed_lo[j] << 16);
            lo[2 * j + 1] = as_type<float>(packed_lo[j] & 0xffff0000u);
            hi[2 * j] = as_type<float>(packed_hi[j] << 16);
            hi[2 * j + 1] = as_type<float>(packed_hi[j] & 0xffff0000u);
        }
    });
#pragma unroll
    for (uint r = 0; r < 4; r++)
        sums[r] = simd_sum(sums[r]);
    // Both halves of a column land in one lane, so the clipped SwiGLU is applied here; the
    // activated value is written where the following contraction reads its input.
    if (lane < 2 && row + lane < p.n) {
        uint column = row + lane;
        float up = sums[lane] + float(bfloat(ub[expert * p.n + column] + 1.0f));
        float gate = sums[2 + lane] + float(bfloat(gb[expert * p.n + column]));
        uy[route * p.n + column] = up;
        gy[route * p.n + column] = swiglu_clipped(gate, up);
    }
}

// Single-token contraction over activated route-major FP32 inputs; four rows per SIMD group.
[[kernel, max_total_threads_per_threadgroup(256)]] void
matvec_mxfp4(const device uchar *w [[buffer(0)]], const device float *x [[buffer(1)]],
                  device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
                  const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
                  uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 lut[256];
    mxfp4_fill_table(lut, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = group.x * 32 + sg * 4, route = group.y, expert = p.experts > 1 ? ids[route] : 0;
    uint blocks = p.k / 32, stride = blocks * 17;
    const device uchar *rows[4];
#pragma unroll
    for (uint r = 0; r < 4; r++)
        rows[r] = w + (expert * p.n + min(row + r, p.n - 1)) * stride;
    uint ix = lane / 2, half_index = lane % 2;
    const device float *xv = x + route * p.k + half_index * 8;
    float sums[4];
#pragma unroll
    for (uint r = 0; r < 4; r++)
        sums[r] = 0;
    mxfp4_rows(rows, blocks, ix, half_index, sums, lut, [&](uint ib, thread float *lo, thread float *hi) {
        const device float *yb = xv + ib * 32;
        float4 l0 = *reinterpret_cast<const device float4 *>(yb);
        float4 l1 = *reinterpret_cast<const device float4 *>(yb + 4);
        float4 h0 = *reinterpret_cast<const device float4 *>(yb + 16);
        float4 h1 = *reinterpret_cast<const device float4 *>(yb + 20);
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            lo[j] = l0[j];
            lo[4 + j] = l1[j];
            hi[j] = h0[j];
            hi[4 + j] = h1[j];
        }
    });
#pragma unroll
    for (uint r = 0; r < 4; r++) {
        float v = simd_sum(sums[r]);
        if (lane == 0 && row + r < p.n)
            y[route * p.n + row + r] =
                bfloat(v + (p.has_bias ? float(bfloat(bias[expert * p.n + row + r])) : 0));
    }
}

constant float e2m1[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6};

// Routed expert tiles on Metal 4 tensor operations: 32 routes × 64 output rows per threadgroup
// with K steps of 32. MXFP4 blocks decode to half through the signed E2M1 table and stage as
// [row][k]; gathered activations stage as half [route][k]. Two 16×64 cooperative products
// accumulate in FP32 and share one 8 KiB block with the output staging; the upper half is
// skipped when the expert owns at most sixteen routes in the tile. Up tiles write FP32 up and
// gate projections with their biases; down tiles contract activated FP32 inputs to BF16.
template <bool Up>
inline void expert_mxfp4_tensor(const device uchar *w, const device uchar *x, device uchar *y,
                                const device float *bias, const device uint *plan, Matrix p,
                                threadgroup float *shared, uint3 group, uint tid, uint lane,
                                uint sg, uint n0) {
    constexpr int NR0 = 64, NR1 = 32, NK = 32, NR1H = NR1 / 2;
    threadgroup half *sa = reinterpret_cast<threadgroup half *>(shared);
    threadgroup half *sb = sa + NR0 * NK;
    uint expert = plan[6 + group.y * 2], tile_row = plan[6 + group.y * 2 + 1];
    uint base = 6 + 2 * ((p.m + 15) / 16 + 32) + expert * (p.m + 1), count = plan[base];
    if (tile_row >= count)
        return;
    uint nr1 = min(uint(NR1), count - tile_row);
    bool has_hi = nr1 > NR1H;
    uint blocks = p.k / 32;
    uint wn = tid / 2, wb = tid % 2;
    const device uchar *row_blocks = w + (expert * p.n + min(n0 + wn, p.n - 1)) * blocks * 17;
    uint xr = tid / 4, xk = (tid % 4) * 8;
    uint route = plan[base + 1 + tile_row + min(xr, nr1 - 1)];
    uint source = Up ? route / p.routes : route;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb + NR1H * NK, dextents<int32_t, 2>(NK, NR1H));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto c0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto c1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
    for (uint k0 = 0; k0 < p.k; k0 += NK) {
        const device uchar *block = row_blocks + (k0 / 32) * 17;
        float scale = expert_mxfp4_scale(block[0]);
        half decoded[16];
        uint shift = wb * 4;
#pragma unroll
        for (uint j = 0; j < 16; ++j)
            decoded[j] = half(e2m1[(block[1 + j] >> shift) & 15] * scale);
        half4 staged[2];
        if (Up) {
            const device bfloat4 *values = reinterpret_cast<const device bfloat4 *>(
                reinterpret_cast<const device bfloat *>(x) + source * p.k + k0 + xk);
            staged[0] = half4(float4(values[0]));
            staged[1] = half4(float4(values[1]));
        } else {
            const device float4 *values = reinterpret_cast<const device float4 *>(
                reinterpret_cast<const device float *>(x) + source * p.k + k0 + xk);
            staged[0] = half4(values[0]);
            staged[1] = half4(values[1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half4 *wd = reinterpret_cast<threadgroup half4 *>(sa + wn * NK + wb * 16);
#pragma unroll
        for (uint j = 0; j < 4; ++j)
            wd[j] = half4(decoded[4 * j], decoded[4 * j + 1], decoded[4 * j + 2], decoded[4 * j + 3]);
        threadgroup half4 *xd = reinterpret_cast<threadgroup half4 *>(sb + xr * NK + xk);
        xd[0] = staged[0];
        xd[1] = staged[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tA.slice(0, 0);
        auto b0 = tB0.slice(0, 0);
        mm.run(b0, a, c0);
        if (has_hi) {
            auto b1 = tB1.slice(0, 0);
            mm.run(b1, a, c1);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared, dextents<int32_t, 2>(NR0, NR1H));
    c0.store(tC0);
    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared + NR1H * NR0, dextents<int32_t, 2>(NR0, NR1H));
        c1.store(tC1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = sg; j < nr1; j += 4) {
        uint out = plan[base + 1 + tile_row + j];
        for (uint i = lane; i < NR0; i += 32) {
            uint col = n0 + i;
            if (col >= p.n)
                continue;
            float value = shared[j * NR0 + i];
            if (Up) {
                value += float(bfloat(bias[expert * p.n + col] + (p.has_bias == 2 ? 1.0f : 0.0f)));
                reinterpret_cast<device float *>(y)[out * p.n + col] = value;
            } else {
                value += p.has_bias ? float(bfloat(bias[expert * p.n + col])) : 0.0f;
                reinterpret_cast<device bfloat *>(y)[out * p.n + col] = bfloat(value);
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void
expert_up_tensor(const device uchar *uw [[buffer(0)]], const device uchar *gw [[buffer(1)]],
                 const device bfloat *x [[buffer(2)]], device float *uy [[buffer(3)]],
                 device float *gy [[buffer(4)]], const device float *ub [[buffer(5)]],
                 const device float *gb [[buffer(6)]], const device uint *ids [[buffer(7)]],
                 constant Matrix &p [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32 * 64];
    uint column_tiles = (p.n + 63) / 64;
    bool gate = group.x >= column_tiles;
    uint n0 = (group.x % column_tiles) * 64;
    Matrix q = p;
    // The up projection carries the model's +1 bias offset; the gate does not.
    q.has_bias = gate ? 1 : 2;
    expert_mxfp4_tensor<true>(gate ? gw : uw, reinterpret_cast<const device uchar *>(x),
                              reinterpret_cast<device uchar *>(gate ? gy : uy), gate ? gb : ub, ids, q,
                              shared, group, tid, lane, sg, n0);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void
expert_down_tensor(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                   device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
                   const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
                   uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32 * 64];
    expert_mxfp4_tensor<false>(w, reinterpret_cast<const device uchar *>(x),
                               reinterpret_cast<device uchar *>(y), bias, ids, p, shared, group, tid,
                               lane, sg, group.x * 64);
}
