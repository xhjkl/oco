// Decode sixteen Q8 coefficients per lane, sharing each block's half scale.
__attribute__((always_inline)) inline float4
q8_dot4(const device uchar *w, const device bfloat *x, uint k, uint rows, uint lane) {
    uint stride = k / 32 * 34, first = lane % 2 * 16;
    float4 sums = 0;
    for (uint block = lane / 2; block < k / 32; block += 16) {
        float4 a[4];
#pragma unroll
        for (uint j = 0; j < 4; j++)
            a[j] = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(
                x + block * 32 + first + j * 4));
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
