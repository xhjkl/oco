#include "mxfp4-dot.metal"
#include "q8-dot.metal"

// Read four packed codes without requiring alignment beyond the GGUF byte layout.
__attribute__((always_inline)) inline uint4 packed_codes4(const device uchar *codes) {
    return uint4(*reinterpret_cast<const device packed_uchar4 *>(codes));
}

// Share K-quant metadata across sixteen coefficients and retain FP32 decoded weights.
__attribute__((always_inline)) inline float4 matvec_kquant4(const device uchar *w,
                                                         const device bfloat *x, uint k,
                                                         uint rows, uint lane) {
    uint size = weight_type == WeightFormat::Q4K ? 144 : weight_type == WeightFormat::Q5K ? 176 : 210;
    uint stride = k / 256 * size;
    float4 sums = 0;
    for (uint block = lane / 16; block < k / 256; block += 2) {
        uint section = lane % 16;
        float4 a[4];
#pragma unroll
        for (uint chunk = 0; chunk < 4; chunk++)
            a[chunk] = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(
                x + block * 256 + section * 16 + chunk * 4));
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            if (r >= rows)
                continue;
            const device uchar *b = w + r * stride + block * size;
            float4 partial = 0;
            if (weight_type == WeightFormat::Q4K || weight_type == WeightFormat::Q5K) {
                uint s = section / 2, offset = section % 2 * 16;
                const device uchar *scales = b + 4;
                uint sc = s < 4 ? scales[s] & 63 : (scales[s + 4] & 15) | ((scales[s - 4] >> 6) << 4);
                uint mn = s < 4 ? scales[s + 4] & 63 : (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4);
                float scale = fp16(b) * sc, minimum = fp16(b + 2) * mn;
                const device uchar *codes = b + (weight_type == WeightFormat::Q4K ? 16 : 48) +
                                           s / 2 * 32 + offset;
#pragma unroll
                for (uint chunk = 0; chunk < 4; chunk++) {
                    uint4 q = (packed_codes4(codes + chunk * 4) >> (s % 2 * 4)) & 15;
                    if (weight_type == WeightFormat::Q5K)
                        q |= ((packed_codes4(b + 16 + offset + chunk * 4) >> s) & 1) << 4;
                    // Converting small codes exactly through the float mantissa.
                    float4 values = as_type<float4>(uint4(0x4b000000u) | q) - 8388608.0f;
                    partial = fma(a[chunk], scale * values - minimum, partial);
                }
            } else {
                uint h = section / 8, q = section % 8 / 2, offset = section % 2 * 16;
                const device uchar *low = b + h * 64 + q % 2 * 32 + offset;
                const device uchar *high = b + 128 + h * 32 + offset;
                float scale = fp16(b + 208) * reinterpret_cast<const device char *>(b + 192)[section];
#pragma unroll
                for (uint chunk = 0; chunk < 4; chunk++) {
                    uint4 lo = (packed_codes4(low + chunk * 4) >> (q / 2 * 4)) & 15;
                    uint4 hi = (packed_codes4(high + chunk * 4) >> (q * 2)) & 3;
                    // Converting six-bit codes exactly through the float mantissa, then centering at 32.
                    float4 values = as_type<float4>(uint4(0x4b000000u) | lo | (hi << 4)) - 8388640.0f;
                    partial = fma(a[chunk], scale * values, partial);
                }
            }
            sums[r] += (partial.x + partial.y) + (partial.z + partial.w);
        }
    }
    return sums;
}

// Four output rows per SIMD group.
kernel void matvec(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                   device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
                   const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
                   uint3 group [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group.x * 32 + sg * 4, route = group.y;
    uint expert = p.experts > 1 ? ids[route] : 0;
    uint input = p.experts > 1 && !p.route_major ? route / p.routes : route;
    float sums[4] = {0, 0, 0, 0};
    if (weight_type == WeightFormat::Mxfp4) {
        // Sharing signed E2M1 values across block lookups; coefficient scaling remains before each FMA.
        threadgroup float lut[16];
        constexpr float magnitudes[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
        if (sg == 0 && lane < 16)
            lut[lane] = as_type<float>(as_type<uint>(magnitudes[lane & 7]) | ((lane & 8) << 28));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint stride = p.k / 32 * 17;
        const device uchar *first = w + (expert * p.n + row) * stride;
        uint rows = min(4u, p.n - min(row, p.n));
        float4 result;
        if (p.route_major)
            result = mxfp4_dot4(first, reinterpret_cast<const device float *>(x) + input * p.k,
                               p.k, rows, lane, lut);
        else
            result = mxfp4_dot4(first, x + input * p.k, p.k, rows, lane, lut);
#pragma unroll
        for (uint r = 0; r < 4; r++)
            sums[r] = result[r];
    } else if (weight_type == WeightFormat::Q8_0) {
        const device uchar *first = w + (expert * p.n + row) * (p.k / 32) * 34;
        float4 result = q8_dot4(first, x + input * p.k, p.k,
                               min(4u, p.n - min(row, p.n)), lane);
#pragma unroll
        for (uint r = 0; r < 4; r++)
            sums[r] = result[r];
    } else if (weight_type == WeightFormat::Q4K || weight_type == WeightFormat::Q5K ||
               weight_type == WeightFormat::Q6K) {
        uint size = weight_type == WeightFormat::Q4K ? 144 : weight_type == WeightFormat::Q5K ? 176 : 210;
        const device uchar *first = w + (expert * p.n + row) * (p.k / 256) * size;
        float4 result = matvec_kquant4(first, x + input * p.k, p.k,
                                     min(4u, p.n - min(row, p.n)), lane);
#pragma unroll
        for (uint r = 0; r < 4; r++)
            sums[r] = result[r];
    } else {
        for (uint k = lane * 4; k < p.k; k += 128) {
            float4 a =
                float4(*reinterpret_cast<const device vec<bfloat, 4> *>(x + input * p.k + k));
#pragma unroll
            for (uint r = 0; r < 4; ++r) {
                if (row + r >= p.n)
                    continue;
                uint index = (uint(expert) * p.n + row + r) * p.k + k;
                float4 b =
                    float4(weight(w, index, weight_type), weight(w, index + 1, weight_type),
                           weight(w, index + 2, weight_type), weight(w, index + 3, weight_type));
                sums[r] += dot(a, b);
            }
        }
    }
#pragma unroll
    for (uint r = 0; r < 4; ++r) {
        float v = simd_sum(sums[r]);
        if (lane == 0 && row + r < p.n) {
            v += p.has_bias
                     ? float(bfloat(bias[expert * p.n + row + r] + (p.has_bias == 2 ? 1.0f : 0.0f)))
                     : 0;
            if (p.experts > 1 && !p.route_major)
                reinterpret_cast<device float *>(y)[route * p.n + row + r] = v;
            else
                y[route * p.n + row + r] = bfloat(v);
        }
    }
}

// Q4_0/Q4_1 dot products in masked-nibble form:
// the activation is pre-scaled by 1, 1/256, 1/16 and 1/4096 so each nibble is used in place,
// and the Q4_0 offset of 8 is folded into one FMA with the activation sum.
// Two lanes share a 32-weight block; each SIMD group accumulates four output rows.
[[kernel, max_total_threads_per_threadgroup(256)]] void
matvec_q4(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
          device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
          const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
          uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint block_bytes = weight_type == WeightFormat::Q4_0 ? 18 : 20;
    const uint code_offset = weight_type == WeightFormat::Q4_0 ? 2 : 4;
    uint row = group.x * 32 + sg * 4, route = group.y;
    uint expert = p.experts > 1 ? ids[route] : 0,
         input = p.experts > 1 && !p.route_major ? route / p.routes : route;
    uint blocks = p.k / 32;
    const device uchar *rows[4];
#pragma unroll
    for (uint r = 0; r < 4; r++)
        // Clamping tail rows to valid storage; their results are discarded below.
        rows[r] = w + (expert * p.n + min(row + r, p.n - 1)) * blocks * block_bytes;
    uint ix = lane / 2, il = lane % 2 * 8;
    const device bfloat *yb = x + input * p.k + ix * 32 + il;
    float sums[4] = {0, 0, 0, 0};
    for (uint ib = ix; ib < blocks; ib += 16) {
        uint4 lo = *reinterpret_cast<const device uint4 *>(yb);
        uint4 hi = *reinterpret_cast<const device uint4 *>(yb + 16);
        float yl[16];
        float sumy = 0;
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            float a0 = as_type<float>(lo[j] << 16), a1 = as_type<float>(lo[j] & 0xffff0000u);
            float b0 = as_type<float>(hi[j] << 16), b1 = as_type<float>(hi[j] & 0xffff0000u);
            sumy += (a0 + a1) + (b0 + b1);
            yl[2 * j] = a0;
            yl[2 * j + 1] = a1 * (1.0f / 256.0f);
            yl[2 * j + 8] = b0 * (1.0f / 16.0f);
            yl[2 * j + 9] = b1 * (1.0f / 4096.0f);
        }
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            const device uchar *b = rows[r] + ib * block_bytes;
            float d = fp16(b);
            const device ushort *qs = reinterpret_cast<const device ushort *>(b + code_offset + il);
            float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                ushort q = qs[j];
                acc0 = fma(yl[2 * j], float(q & 0x000F), acc0);
                acc1 = fma(yl[2 * j + 1], float(q & 0x0F00), acc1);
                acc2 = fma(yl[2 * j + 8], float(q & 0x00F0), acc2);
                acc3 = fma(yl[2 * j + 9], float(q & 0xF000), acc3);
            }
            float acc = (acc0 + acc1) + (acc2 + acc3);
            if (weight_type == WeightFormat::Q4_0)
                sums[r] = fma(d, fma(sumy, -8.0f, acc), sums[r]);
            else
                sums[r] = fma(d, acc, fma(sumy, fp16(b + 2), sums[r]));
        }
        yb += 16 * 32;
    }
#pragma unroll
    for (uint r = 0; r < 4; r++) {
        float v = simd_sum(sums[r]);
        if (lane == 0 && row + r < p.n)
            y[route * p.n + row + r] =
                bfloat(v + (p.has_bias ? float(bfloat(bias[expert * p.n + row + r] +
                                                      (p.has_bias == 2 ? 1.0f : 0.0f)))
                                       : 0));
    }
}

// Q6_K rows: two superblocks per SIMD group, sixteen lanes per superblock,
// two output rows per SIMD group.
[[kernel, max_total_threads_per_threadgroup(256)]] void
matvec_q6k(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
           device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
           const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
           uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR = 2;
    uint row = (group.x * 8 + sg) * NR, route = group.y;
    uint expert = p.experts > 1 ? ids[route] : 0;
    uint input = p.experts > 1 && !p.route_major ? route / p.routes : route;
    uint blocks = p.k / 256;
    const device uchar *base = w + (expert * p.n + min(row, p.n - 1)) * blocks * 210;
    uint stride = row + 1 < p.n ? blocks * 210 : 0;
    const device bfloat *yy = x + input * p.k;
    uint t = lane / 2, ix = lane % 2, ip = t / 8, il = t % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
    uint y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
    float sums[NR] = {0, 0};
    for (uint i = ix; i < blocks; i += 2) {
        const device bfloat *yv = yy + i * 256 + y_offset;
        float yl[16];
#pragma unroll
        for (uint l = 0; l < 4; ++l) {
            yl[4 * l + 0] = float(yv[l + 0]);
            yl[4 * l + 1] = float(yv[l + 32]);
            yl[4 * l + 2] = float(yv[l + 64]);
            yl[4 * l + 3] = float(yv[l + 96]);
        }
#pragma unroll
        for (uint r = 0; r < NR; ++r) {
            const device uchar *b = base + r * stride + i * 210;
            const device uchar *q1 = b + q_offset_l, *q2 = q1 + 32, *qh = b + 128 + q_offset_h;
            const device char *sc = reinterpret_cast<const device char *>(b + 192) + is;
            float4 s = 0;
#pragma unroll
            for (uint l = 0; l < 4; ++l) {
                uint h = qh[l];
                s[0] = fma(yl[4 * l + 0], float(int((q1[l] & 0xF) | ((h & 0x03) << 4)) - 32), s[0]);
                s[1] = fma(yl[4 * l + 1], float(int((q2[l] & 0xF) | ((h & 0x0C) << 2)) - 32), s[1]);
                s[2] = fma(yl[4 * l + 2], float(int((q1[l] >> 4) | ((h & 0x30) << 0)) - 32), s[2]);
                s[3] = fma(yl[4 * l + 3], float(int((q2[l] >> 4) | ((h & 0xC0) >> 2)) - 32), s[3]);
            }
            sums[r] += fp16(b + 208) * (s[0] * sc[0] + s[1] * sc[2] + s[2] * sc[4] + s[3] * sc[6]);
        }
    }
#pragma unroll
    for (uint r = 0; r < NR; ++r) {
        float v = simd_sum(sums[r]);
        if (lane == 0 && row + r < p.n) {
            v += p.has_bias ? float(bfloat(bias[expert * p.n + row + r] +
                                          (p.has_bias == 2 ? 1.0f : 0.0f))) : 0;
            if (p.experts > 1 && !p.route_major)
                reinterpret_cast<device float *>(y)[route * p.n + row + r] = v;
            else
                y[route * p.n + row + r] = bfloat(v);
        }
    }
}

// Q5_K rows: four superblocks per SIMD group, eight lanes per superblock,
// two output rows per SIMD group.
[[kernel, max_total_threads_per_threadgroup(256)]] void
matvec_q5k(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
           device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
           const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
           uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR = 2;
    uint row = (group.x * 8 + sg) * NR, route = group.y;
    uint expert = p.experts > 1 ? ids[route] : 0;
    uint input = p.experts > 1 && !p.route_major ? route / p.routes : route;
    uint blocks = p.k / 256;
    const device uchar *base = w + (expert * p.n + min(row, p.n - 1)) * blocks * 176;
    uint stride = row + 1 < p.n ? blocks * 176 : 0;
    const device bfloat *yy = x + input * p.k;
    uint t = lane / 4, ix = lane % 4, iq = t / 4, ir = t % 4;
    uint l0 = 8 * ir, q_offset = 32 * iq + l0, y_offset = 64 * iq + l0;
    uint hm1 = 1u << (2 * iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
    float sums[NR] = {0, 0};
    for (uint i = ix; i < blocks; i += 4) {
        const device bfloat *y1 = yy + i * 256 + y_offset, *y2 = y1 + 128;
        float yl[16], yh[16];
        float4 sumy = 0;
#pragma unroll
        for (uint l = 0; l < 8; ++l) {
            yl[l + 0] = float(y1[l + 0]); sumy[0] += yl[l + 0];
            yl[l + 8] = float(y1[l + 32]); sumy[1] += yl[l + 8];
            yh[l + 0] = float(y2[l + 0]); sumy[2] += yh[l + 0];
            yh[l + 8] = float(y2[l + 32]); sumy[3] += yh[l + 8];
        }
#pragma unroll
        for (uint r = 0; r < NR; ++r) {
            const device uchar *b = base + r * stride + i * 176;
            const device uchar *q1 = b + 48 + q_offset, *q2 = q1 + 64, *qh = b + 16 + l0;
            const device ushort *a = reinterpret_cast<const device ushort *>(b + 4) + iq;
            ushort sc16[4];
            sc16[0] = a[0] & 0x3f3f;
            sc16[1] = a[2] & 0x3f3f;
            sc16[2] = ((a[4] >> 0) & 0x0f0f) | ((a[0] & 0xc0c0) >> 2);
            sc16[3] = ((a[4] >> 4) & 0x0f0f) | ((a[2] & 0xc0c0) >> 2);
            float4 acc1 = 0, acc2 = 0;
#pragma unroll
            for (uint l = 0; l < 8; ++l) {
                uint h = qh[l];
                acc1[0] = fma(yl[l + 0], float(q1[l] & 0x0F), acc1[0]);
                acc1[1] = fma(yl[l + 8], float(q1[l] & 0xF0), acc1[1]);
                acc1[2] = fma(yh[l + 0], float(q2[l] & 0x0F), acc1[2]);
                acc1[3] = fma(yh[l + 8], float(q2[l] & 0xF0), acc1[3]);
                acc2[0] += h & hm1 ? yl[l + 0] : 0.0f;
                acc2[1] += h & hm2 ? yl[l + 8] : 0.0f;
                acc2[2] += h & hm3 ? yh[l + 0] : 0.0f;
                acc2[3] += h & hm4 ? yh[l + 8] : 0.0f;
            }
            float s0 = float(sc16[0] & 0xff), s1 = float(sc16[0] >> 8);
            float s2 = float(sc16[1] & 0xff), s3 = float(sc16[1] >> 8);
            float s4 = float(sc16[2] & 0xff), s5 = float(sc16[2] >> 8);
            float s6 = float(sc16[3] & 0xff), s7 = float(sc16[3] >> 8);
            sums[r] += fp16(b) * (s0 * (acc1[0] + 16.0f * acc2[0]) +
                                  s1 * (acc1[1] * (1.0f / 16.0f) + 16.0f * acc2[1]) +
                                  s4 * (acc1[2] + 16.0f * acc2[2]) +
                                  s5 * (acc1[3] * (1.0f / 16.0f) + 16.0f * acc2[3])) -
                       fp16(b + 2) * (sumy[0] * s2 + sumy[1] * s3 + sumy[2] * s6 + sumy[3] * s7);
        }
    }
#pragma unroll
    for (uint r = 0; r < NR; ++r) {
        float v = simd_sum(sums[r]);
        if (lane == 0 && row + r < p.n) {
            v += p.has_bias ? float(bfloat(bias[expert * p.n + row + r] +
                                          (p.has_bias == 2 ? 1.0f : 0.0f))) : 0;
            if (p.experts > 1 && !p.route_major)
                reinterpret_cast<device float *>(y)[route * p.n + row + r] = v;
            else
                y[route * p.n + row + r] = bfloat(v);
        }
    }
}

// Split-K geometry for small dense projections.
kernel void matvec_split(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                         device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
                         const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    uint row = group.x;
    float sum = 0;
    for (uint k = sg * 128 + lane * 4; k < p.k; k += 1024) {
        float4 a = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(x + group.y * p.k + k));
        uint index = uint(row) * p.k + k;
        float4 b = float4(weight(w, index, weight_type), weight(w, index + 1, weight_type),
                          weight(w, index + 2, weight_type), weight(w, index + 3, weight_type));
        sum += dot(a, b);
    }
    sum = simd_sum(sum);
    if (lane == 0)
        partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(tid < 8 ? partial[tid] : 0);
    if (tid == 0)
        y[group.y * p.n + row] = bfloat(sum + (p.has_bias ? float(bfloat(bias[row])) : 0));
}
