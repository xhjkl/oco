// Round decoded coefficients and activations to half; products accumulate in FP32.
// Large Q4 products apply the cached row gain before half conversion and its inverse after MMA.
// Coefficient rounding remains intentional and requires model-quality validation.
// A packed Q4 block header and the eight code words one lane decodes, fetched ahead of use
// so the next step's device loads overlap the current step's matrix products.
struct Q4Raw {
    ushort scale, offset;
    ushort qs[8];
};

__attribute__((always_inline)) inline Q4Raw q4_fetch(const device uchar *block, short high) {
    const device ushort *words = reinterpret_cast<const device ushort *>(block);
    Q4Raw raw;
    raw.scale = words[0];
    raw.offset = weight_type == WeightFormat::Q4_1 ? words[1] : 0;
    const device ushort *qs = words + (weight_type == WeightFormat::Q4_0 ? 1 : 2);
#pragma unroll
    for (short i = 0; i < 8; ++i)
        raw.qs[i] = qs[i];
    return raw;
}

__attribute__((always_inline)) inline half4x4 q4_decode(Q4Raw raw, short high) {
    float scale = float(as_type<half>(raw.scale));
    float d1 = high ? scale / 16.0f : scale, d2 = d1 / 256.0f;
    float offset = weight_type == WeightFormat::Q4_0 ? -8.0f * scale : float(as_type<half>(raw.offset));
    ushort mask0 = high ? 0x00f0 : 0x000f, mask1 = mask0 << 8;
    half4x4 values;
#pragma unroll
    for (short i = 0; i < 8; ++i) {
        values[i / 2][2 * (i % 2)] = half(d1 * (raw.qs[i] & mask0) + offset);
        values[i / 2][2 * (i % 2) + 1] = half(d2 * (raw.qs[i] & mask1) + offset);
    }
    return values;
}

inline half4x4 q4_block(const device uchar *block, short high) {
    return q4_decode(q4_fetch(block, high), high);
}

// Choose finite normal reciprocal power-of-two factors from a whole-row peak.
// Zero rows use one. FP32-subnormal peaks saturate the positive exponent clamp.
// Scaling cannot preserve arbitrary within-row dynamic range; tiny values can still round in half.
inline float2 q4_input_scale_for_peak(float peak) {
    int exponent = int((as_type<uint>(peak) >> 23) & 255u);
    int shift = peak == 0.0f ? 0 : clamp(141 - exponent, -120, 120);
    return float2(as_type<float>(uint(127 + shift) << 23),
                  as_type<float>(uint(127 - shift) << 23));
}

// Wide tiles: 64 tokens × 64 rows per threadgroup, four SIMD groups each owning 32 × 32.
// Weights decode to half and stage as [row][k], loading transposed; activations stage as
// [token][k]. Each staged weight serves 64 products, and outputs leave registers directly
// through the SIMD-group matrix element layout. Q4 reads packed codes as ushort pairs; other
// formats share the generic sixteen-coefficient decoder.
template <bool Q4, bool Scaled>
inline void matmul_tile_wide(const device uchar *w, const device bfloat *x, device bfloat *y,
                             const device float *bias, constant Matrix &p,
                             threadgroup half *wt, threadgroup half *xs,
                             const device float2 *scales, uint3 group, ushort tid, ushort lane, ushort sg) {
    constexpr ushort BK = 32, STRIDE = BK + 8, BM = 64;
    uint n0 = group.y * 64, m0 = group.x * BM;
    // Clamping tail rows to valid storage; their outputs are discarded.
    ushort wn = tid / 2, wk = (tid % 2) * 16;
    ushort xm = tid / 2, xk = (tid % 2) * 16;
    uint block_bytes = weight_type == WeightFormat::Q4_0 ? 18 : 20;
    const device uchar *weights = w + min(n0 + wn, p.n - 1) * packed_row_bytes(p.k);
    const device uchar *block = weights;
    const device bfloat *row = x + min(m0 + xm, p.m - 1) * p.k + xk;
    float gain = Scaled ? scales[min(m0 + xm, p.m - 1)].x : 1.0f;
    simdgroup_half8x8 a[4], b[4];
    simdgroup_float8x8 accum[4][4];
#pragma unroll
    for (short r = 0; r < 4; ++r)
#pragma unroll
        for (short c = 0; c < 4; ++c)
            accum[r][c] = simdgroup_float8x8(0);
    for (uint base = 0; base < p.k; base += BK) {
        half4x4 decoded;
        if (Q4) {
            decoded = q4_block(block, wk / 16);
            block += block_bytes;
        } else {
            float values[16];
            decode_weights16(weights, base + wk, values);
#pragma unroll
            for (short i = 0; i < 16; ++i)
                decoded[i / 4][i % 4] = half(values[i]);
        }
        const device bfloat4 *values = reinterpret_cast<const device bfloat4 *>(row);
        half4 staged[4];
#pragma unroll
        for (short j = 0; j < 4; ++j)
            staged[j] = half4(float4(values[j]) * gain);
        row += BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half4 *wd = reinterpret_cast<threadgroup half4 *>(wt + wn * STRIDE + wk);
#pragma unroll
        for (short j = 0; j < 4; ++j)
            wd[j] = decoded[j];
        threadgroup half4 *xd = reinterpret_cast<threadgroup half4 *>(xs + xm * STRIDE + xk);
#pragma unroll
        for (short j = 0; j < 4; ++j)
            xd[j] = staged[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *wa = wt + (sg % 2) * 32 * STRIDE;
        threadgroup const half *xb = xs + (sg / 2) * 32 * STRIDE;
#pragma unroll
        for (short k = 0; k < BK; k += 8) {
#pragma unroll
            for (short c = 0; c < 4; ++c)
                simdgroup_load(a[c], wa + c * 8 * STRIDE + k, STRIDE, ulong2(0, 0), true);
#pragma unroll
            for (short r = 0; r < 4; ++r)
                simdgroup_load(b[r], xb + r * 8 * STRIDE + k, STRIDE);
#pragma unroll
            for (short r = 0; r < 4; ++r)
#pragma unroll
                for (short c = 0; c < 4; ++c)
                    simdgroup_multiply_accumulate(accum[r][c], b[r], a[c], accum[r][c]);
        }
    }
    // Element (lr, lc) and (lr, lc + 1) of each 8x8 matrix belong to this lane.
    ushort quad = lane / 4;
    ushort lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    uint row0 = m0 + (sg / 2) * 32 + lr, col0 = n0 + (sg % 2) * 32 + lc;
#pragma unroll
    for (short r = 0; r < 4; ++r) {
        uint m = row0 + r * 8;
        if (m >= p.m)
            continue;
        float inverse_gain = Scaled ? scales[m].y : 1.0f;
#pragma unroll
        for (short c = 0; c < 4; ++c) {
            uint n = col0 + c * 8;
#pragma unroll
            for (short j = 0; j < 2; ++j) {
                if (n + j < p.n)
                    y[m * p.n + n + j] = bfloat(accum[r][c].thread_elements()[j] * inverse_gain +
                                                (p.has_bias ? float(bfloat(bias[n + j])) : 0));
            }
        }
    }
}

// Narrow tiles: 8·R tokens × 64 rows per threadgroup for prompts shorter than a wide tile
// and for the tail rows of longer prompts. Each SIMD group owns 16 rows across all token tiles,
// so the per-threadgroup weight decoding is amortized over exactly the tokens present.
template <bool Q4, bool Scaled, short R>
inline void matmul_tile_rows(const device uchar *w, const device bfloat *x, device bfloat *y,
                             const device float *bias, constant Matrix &p,
                             threadgroup half *wt, threadgroup half *xs,
                             const device float2 *scales, uint3 group, ushort tid, ushort lane, ushort sg) {
    constexpr ushort BK = 32, STRIDE = BK + 8, BM = 8 * R;
    uint n0 = group.y * 64, m0 = group.x * BM;
    ushort wn = tid / 2, wk = (tid % 2) * 16;
    uint block_bytes = weight_type == WeightFormat::Q4_0 ? 18 : 20;
    const device uchar *weights = w + min(n0 + wn, p.n - 1) * packed_row_bytes(p.k);
    const device uchar *block = weights;
    // Each of the first 4·BM threads stages eight activations of one token row.
    bool stager = tid < 4 * BM;
    ushort xm = tid / 4, xk = (tid % 4) * 8;
    const device bfloat *row = x + min(m0 + xm, p.m - 1) * p.k + xk;
    float gain = Scaled && stager ? scales[min(m0 + xm, p.m - 1)].x : 1.0f;
    simdgroup_half8x8 a[2], b[R];
    simdgroup_float8x8 accum[R][2];
#pragma unroll
    for (short r = 0; r < R; ++r)
#pragma unroll
        for (short c = 0; c < 2; ++c)
            accum[r][c] = simdgroup_float8x8(0);
    for (uint base = 0; base < p.k; base += BK) {
        half4x4 decoded;
        if (Q4) {
            decoded = q4_block(block, wk / 16);
            block += block_bytes;
        } else {
            float values[16];
            decode_weights16(weights, base + wk, values);
#pragma unroll
            for (short i = 0; i < 16; ++i)
                decoded[i / 4][i % 4] = half(values[i]);
        }
        half4 staged[2];
        if (stager) {
            const device bfloat4 *values = reinterpret_cast<const device bfloat4 *>(row);
            staged[0] = half4(float4(values[0]) * gain);
            staged[1] = half4(float4(values[1]) * gain);
            row += BK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half4 *wd = reinterpret_cast<threadgroup half4 *>(wt + wn * STRIDE + wk);
#pragma unroll
        for (short j = 0; j < 4; ++j)
            wd[j] = decoded[j];
        if (stager) {
            threadgroup half4 *xd = reinterpret_cast<threadgroup half4 *>(xs + xm * STRIDE + xk);
            xd[0] = staged[0];
            xd[1] = staged[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *wa = wt + sg * 16 * STRIDE;
#pragma unroll
        for (short k = 0; k < BK; k += 8) {
#pragma unroll
            for (short c = 0; c < 2; ++c)
                simdgroup_load(a[c], wa + c * 8 * STRIDE + k, STRIDE, ulong2(0, 0), true);
#pragma unroll
            for (short r = 0; r < R; ++r)
                simdgroup_load(b[r], xs + r * 8 * STRIDE + k, STRIDE);
#pragma unroll
            for (short r = 0; r < R; ++r)
#pragma unroll
                for (short c = 0; c < 2; ++c)
                    simdgroup_multiply_accumulate(accum[r][c], b[r], a[c], accum[r][c]);
        }
    }
    ushort quad = lane / 4;
    ushort lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    uint row0 = m0 + lr, col0 = n0 + sg * 16 + lc;
#pragma unroll
    for (short r = 0; r < R; ++r) {
        uint m = row0 + r * 8;
        if (m >= p.m)
            continue;
        float inverse_gain = Scaled ? scales[m].y : 1.0f;
#pragma unroll
        for (short c = 0; c < 2; ++c) {
            uint n = col0 + c * 8;
#pragma unroll
            for (short j = 0; j < 2; ++j) {
                if (n + j < p.n)
                    y[m * p.n + n + j] = bfloat(accum[r][c].thread_elements()[j] * inverse_gain +
                                                (p.has_bias ? float(bfloat(bias[n + j])) : 0));
            }
        }
    }
}
