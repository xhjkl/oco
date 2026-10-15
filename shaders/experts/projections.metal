#include "../matrix/mxfp4-dot.metal"

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
// The following gate dispatch applies clipped SwiGLU to these FP32 projection outputs.
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
    if (lane < 2 && row + lane < p.n) {
        uint column = row + lane;
        float up = sums[lane] + float(bfloat(ub[expert * p.n + column] + 1.0f));
        float gate = sums[2 + lane] + float(bfloat(gb[expert * p.n + column]));
        uy[route * p.n + column] = up;
        gy[route * p.n + column] = gate;
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

// Routed expert tiles: BM routes × 64 rows per threadgroup, four SIMD groups each owning
// (BM/2) × 32. MXFP4 blocks decode to half through the signed E2M1 table and stage as [row][k],
// loading transposed; gathered activations stage as [route][k]. Products accumulate in FP32.
// This native inference path uses FP16 operands; it does not promise the full BF16/FP32 range.
constant float e2m1[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6};

template <uint BM, bool Up>
inline void expert_tile(const device uchar *w, const device uchar *x, device uchar *y,
                        const device float *bias, const device uint *ids, Matrix p,
                        threadgroup half *wt, threadgroup half *xs, uint expert, uint tile_row,
                        uint count, uint plan_base, uint n0, uint tid, uint lane, uint sg) {
    constexpr uint BK = 32, STRIDE = BK + 8, R = BM / 16;
    uint quad = lane / 4, lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    // Weights: thread stages sixteen coefficients of one row.
    uint wn = tid / 2, wk = (tid % 2) * 16;
    const device uchar *blocks = w + (expert * p.n + min(n0 + wn, p.n - 1)) * (p.k / 32) * 17;
    // Activations: thread stages eight values of one route row per work item.
    constexpr uint ITEMS = BM * 4;
    uint route[(ITEMS + 127) / 128];
#pragma unroll
    for (uint i = 0; i < (ITEMS + 127) / 128; ++i) {
        uint item = tid + 128 * i, row = tile_row + item / 4;
        route[i] = row < count ? (p.experts > 1 ? ids[plan_base + 1 + row] : row) : 0;
    }
    simdgroup_half8x8 a[4], b[R];
    simdgroup_float8x8 accum[R][4];
#pragma unroll
    for (uint r = 0; r < R; ++r)
#pragma unroll
        for (uint c = 0; c < 4; ++c)
            accum[r][c] = simdgroup_float8x8(0);
    for (uint base = 0; base < p.k; base += BK) {
        const device uchar *block = blocks + (base / 32) * 17;
        uint exponent = block[0];
        float scale = expert_mxfp4_scale(exponent);
        half decoded[16];
        uint shift = wk / 16 * 4;
#pragma unroll
        for (uint j = 0; j < 16; ++j)
            decoded[j] = half(e2m1[(block[1 + j] >> shift) & 15] * scale);
        half4 staged[(ITEMS + 127) / 128][2];
#pragma unroll
        for (uint i = 0; i < (ITEMS + 127) / 128; ++i) {
            uint item = tid + 128 * i, xk = (item % 4) * 8;
            if (item < ITEMS) {
                // Padding inactive routes before any device read; sparse experts need not fill a tile.
                if (tile_row + item / 4 >= count) {
                    staged[i][0] = half4(0);
                    staged[i][1] = half4(0);
                } else if (Up) {
                    uint input = p.experts > 1 && !p.route_major ? route[i] / p.routes : route[i];
                    const device bfloat4 *values = reinterpret_cast<const device bfloat4 *>(
                        reinterpret_cast<const device bfloat *>(x) + input * p.k + base + xk);
                    staged[i][0] = half4(float4(values[0]));
                    staged[i][1] = half4(float4(values[1]));
                } else {
                    const device float4 *values = reinterpret_cast<const device float4 *>(
                        reinterpret_cast<const device float *>(x) + route[i] * p.k + base + xk);
                    staged[i][0] = half4(values[0]);
                    staged[i][1] = half4(values[1]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half4 *wd = reinterpret_cast<threadgroup half4 *>(wt + wn * STRIDE + wk);
#pragma unroll
        for (uint j = 0; j < 4; ++j)
            wd[j] = half4(decoded[4 * j], decoded[4 * j + 1], decoded[4 * j + 2], decoded[4 * j + 3]);
#pragma unroll
        for (uint i = 0; i < (ITEMS + 127) / 128; ++i) {
            uint item = tid + 128 * i;
            if (item < ITEMS) {
                threadgroup half4 *xd = reinterpret_cast<threadgroup half4 *>(
                    xs + (item / 4) * STRIDE + (item % 4) * 8);
                xd[0] = staged[i][0];
                xd[1] = staged[i][1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *wa = wt + (sg % 2) * 32 * STRIDE;
        threadgroup const half *xb = xs + (sg / 2) * (BM / 2) * STRIDE;
#pragma unroll
        for (uint k = 0; k < BK; k += 8) {
#pragma unroll
            for (uint c = 0; c < 4; ++c)
                simdgroup_load(a[c], wa + c * 8 * STRIDE + k, STRIDE, ulong2(0, 0), true);
#pragma unroll
            for (uint r = 0; r < R; ++r)
                simdgroup_load(b[r], xb + r * 8 * STRIDE + k, STRIDE);
#pragma unroll
            for (uint r = 0; r < R; ++r)
#pragma unroll
                for (uint c = 0; c < 4; ++c)
                    simdgroup_multiply_accumulate(accum[r][c], b[r], a[c], accum[r][c]);
        }
    }
    uint col0 = n0 + (sg % 2) * 32 + lc;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
        uint row = tile_row + (sg / 2) * (BM / 2) + r * 8 + lr;
        if (row >= count)
            continue;
        uint out = p.experts > 1 ? ids[plan_base + 1 + row] : row;
#pragma unroll
        for (uint c = 0; c < 4; ++c) {
#pragma unroll
            for (uint j = 0; j < 2; ++j) {
                uint col = col0 + c * 8 + j;
                if (col >= p.n)
                    continue;
                float value = accum[r][c].thread_elements()[j];
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
}

template <uint BM>
[[kernel, max_total_threads_per_threadgroup(128)]] void
expert_up_tiled(const device uchar *uw [[buffer(0)]], const device uchar *gw [[buffer(1)]],
                const device bfloat *x [[buffer(2)]], device float *uy [[buffer(3)]],
                device float *gy [[buffer(4)]], const device float *ub [[buffer(5)]],
                const device float *gb [[buffer(6)]], const device uint *ids [[buffer(7)]],
                constant Matrix &p [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half wt[64 * 40];
    threadgroup half xs[BM * 40];
    uint column_tiles = (p.n + 63) / 64;
    bool gate = group.x >= column_tiles;
    uint n0 = (group.x % column_tiles) * 64;
    uint expert = p.experts > 1 ? ids[6 + group.y * 2] : 0;
    uint tile_row = p.experts > 1 ? ids[6 + group.y * 2 + 1] : group.y * BM;
    uint plan_base = 6 + 2 * ((p.m + 15) / 16 + 32) + expert * (p.m + 1),
         count = p.experts > 1 ? ids[plan_base] : p.m;
    if (tile_row >= count)
        return;
    // The up projection carries the model's +1 bias offset; the gate does not.
    if (gate) {
        Matrix g = p;
        g.has_bias = 1;
        expert_tile<BM, true>(gw, reinterpret_cast<const device uchar *>(x),
                              reinterpret_cast<device uchar *>(gy), gb, ids, g, wt, xs, expert,
                              tile_row, count, plan_base, n0, tid, lane, sg);
    } else {
        Matrix u = p;
        u.has_bias = 2;
        expert_tile<BM, true>(uw, reinterpret_cast<const device uchar *>(x),
                              reinterpret_cast<device uchar *>(uy), ub, ids, u, wt, xs, expert,
                              tile_row, count, plan_base, n0, tid, lane, sg);
    }
}

template <uint BM>
[[kernel, max_total_threads_per_threadgroup(128)]] void
expert_down_tiled(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                  device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
                  const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
                  uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half wt[64 * 40];
    threadgroup half xs[BM * 40];
    uint n0 = group.x * 64;
    uint expert = p.experts > 1 ? ids[6 + group.y * 2] : 0;
    uint tile_row = p.experts > 1 ? ids[6 + group.y * 2 + 1] : group.y * BM;
    uint plan_base = 6 + 2 * ((p.m + 15) / 16 + 32) + expert * (p.m + 1),
         count = p.experts > 1 ? ids[plan_base] : p.m;
    if (tile_row >= count)
        return;
    expert_tile<BM, false>(w, reinterpret_cast<const device uchar *>(x),
                           reinterpret_cast<device uchar *>(y), bias, ids, p, wt, xs, expert,
                           tile_row, count, plan_base, n0, tid, lane, sg);
}

template [[host_name("expert_up")]] kernel void expert_up_tiled<16>(
    const device uchar *, const device uchar *, const device bfloat *, device float *, device float *,
    const device float *, const device float *, const device uint *, constant Matrix &, uint3, uint, uint, uint);
template [[host_name("expert_up_large")]] kernel void expert_up_tiled<32>(
    const device uchar *, const device uchar *, const device bfloat *, device float *, device float *,
    const device float *, const device float *, const device uint *, constant Matrix &, uint3, uint, uint, uint);
template [[host_name("expert_down")]] kernel void expert_down_tiled<16>(
    const device uchar *, const device bfloat *, device bfloat *, const device float *,
    const device uint *, constant Matrix &, uint3, uint, uint, uint);
template [[host_name("expert_down_large")]] kernel void expert_down_tiled<32>(
    const device uchar *, const device bfloat *, device bfloat *, const device float *,
    const device uint *, constant Matrix &, uint3, uint, uint, uint);
