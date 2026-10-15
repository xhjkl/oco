#pragma once

// Decode signed E2M1 values before applying their shared power-of-two scale.
__attribute__((always_inline)) inline float4 mxfp4_values(uchar4 codes, const threadgroup float *lut) {
    return float4(lut[codes.x], lut[codes.y], lut[codes.z], lut[codes.w]);
}

// Sum four packed rows; two lanes share each block and reuse its header for 16 products.
template <typename Input>
__attribute__((always_inline)) inline float4
mxfp4_dot4(const device uchar *w, const device Input *x, uint k, uint rows, uint lane, const threadgroup float *lut) {
    uint stride = k / 32 * 17, first = lane % 2 * 8;
    float4 sums = 0;
    for (uint block = lane / 2; block < k / 32; block += 16) {
        const device Input *a = x + block * 32 + first;
        float4 a0 = float4(*reinterpret_cast<const device vec<Input, 4> *>(a)),
               a1 = float4(*reinterpret_cast<const device vec<Input, 4> *>(a + 4)),
               a2 = float4(*reinterpret_cast<const device vec<Input, 4> *>(a + 16)),
               a3 = float4(*reinterpret_cast<const device vec<Input, 4> *>(a + 20));
#pragma unroll
        for (uint row = 0; row < 4; row++) {
            if (row >= rows)
                continue;
            const device uchar *b = w + row * stride + block * 17;
            const device uchar *q = b + 1 + first;
            // Reading bytes individually; GGUF's 17-byte blocks need not be vector-aligned.
            uchar4 q0(q[0], q[1], q[2], q[3]), q1(q[4], q[5], q[6], q[7]);
            uint exponent = b[0];
            float scale = exponent == 0 ? as_type<float>(0x00400000u)
                                        : as_type<float>(exponent << 23);
            // Preserving coefficient scaling; explicit FMAs prevent fast math from factoring it out.
            float4 values = mxfp4_values(q0 & uchar4(15), lut) * scale;
            float4 products = fma(a0, values, float4(0));
            values = mxfp4_values(q1 & uchar4(15), lut) * scale;
            products = fma(a1, values, products);
            values = mxfp4_values(q0 >> 4, lut) * scale;
            products = fma(a2, values, products);
            values = mxfp4_values(q1 >> 4, lut) * scale;
            products = fma(a3, values, products);
            float partial = (products.x + products.y) + (products.z + products.w);
            sums[row] += partial;
        }
    }
    return sums;
}
