// Decode sixteen Q8 coefficients per lane, sharing each block's half scale.
__attribute__((always_inline)) inline float4
q8_dot4(const device uchar *w, const device bfloat *x, uint k, uint rows, uint lane) {
    uint stride = k / 32 * 34, first = lane % 2 * 16;
    float4 sums = 0;
    for (uint block = lane / 2; block < k / 32; block += 16) {
        float4 a[4];
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            a[j] = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(
                x + block * 32 + first + j * 4));
            if (RELU_SQR_INPUT) {
                a[j] = max(a[j], 0.0f);
                a[j] *= a[j];
            }
        }
#pragma unroll
        for (uint row = 0; row < 4; row++) {
            if (row >= rows)
                continue;
            const device uchar *b = w + row * stride + block * 34;
            float scale = fp16(b);
            float4 partial = 0;
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                float4 q = float4(*reinterpret_cast<const device packed_char4 *>(
                    b + 2 + first + j * 4));
                partial = fma(a[j], scale * q, partial);
            }
            sums[row] += (partial.x + partial.y) + (partial.z + partial.w);
        }
    }
    return sums;
}

// Q8_0 single-row product with K split across the four SIMD groups of a 128-thread group:
// two output rows per threadgroup, four lanes per 32-coefficient block, eight blocks per
// SIMD group per step. The short per-lane loops keep many loads in flight for narrow matrices.
[[kernel, max_total_threads_per_threadgroup(128)]] void
matvec_q8_split(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
                const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
                uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4][2];
    uint row = group.x * 2, route = group.y;
    uint blocks = p.k / 32;
    const device bfloat *xr = x + route * p.k;
    uint ix = lane / 4, il = lane % 4 * 8;
    const device uchar *rows[2];
#pragma unroll
    for (uint r = 0; r < 2; r++)
        rows[r] = w + min(row + r, p.n - 1) * blocks * 34;
    float sums[2] = {0, 0};
    for (uint ib = sg * 8 + ix; ib < blocks; ib += 32) {
        const device bfloat *xb = xr + ib * 32 + il;
        float4 a0 = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(xb));
        float4 a1 = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(xb + 4));
        if (RELU_SQR_INPUT) {
            a0 = max(a0, 0.0f);
            a0 *= a0;
            a1 = max(a1, 0.0f);
            a1 *= a1;
        }
#pragma unroll
        for (uint r = 0; r < 2; r++) {
            const device uchar *b = rows[r] + ib * 34;
            float4 q0 = float4(*reinterpret_cast<const device packed_char4 *>(b + 2 + il));
            float4 q1 = float4(*reinterpret_cast<const device packed_char4 *>(b + 6 + il));
            sums[r] += fp16(b) * (dot(a0, q0) + dot(a1, q1));
        }
    }
#pragma unroll
    for (uint r = 0; r < 2; r++) {
        float v = simd_sum(sums[r]);
        if (lane == 0)
            partial[sg][r] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2 && row + tid < p.n) {
        float v = partial[0][tid] + partial[1][tid] + partial[2][tid] + partial[3][tid];
        if (p.has_bias)
            v += float(bfloat(bias[row + tid]));
        store_result(y, route * p.n + row + tid, v);
    }
}
