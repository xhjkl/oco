#include <metal_simdgroup_matrix>
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// GGUF format codes.
enum WeightFormat : uint {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Bf16 = 30,
    Mxfp4 = 39,
};

constant uint weight_type [[function_constant(0)]];
// Vector kernels may square the positive part of their input on load (relu² feed-forward).
constant uint input_relu_sqr [[function_constant(2)]];
#define RELU_SQR_INPUT (is_function_constant_defined(input_relu_sqr) && input_relu_sqr != 0)
// Vector kernels may keep their results in FP32 (final logits, recurrent time steps).
constant uint output_f32 [[function_constant(3)]];
#define OUTPUT_F32 (is_function_constant_defined(output_f32) && output_f32 != 0)
__attribute__((always_inline)) inline void store_result(device bfloat *y, uint i, float v) {
    if (OUTPUT_F32)
        reinterpret_cast<device float *>(y)[i] = v;
    else
        y[i] = bfloat(v);
}

__attribute__((always_inline)) inline float fp16(const device uchar *p) {
    return float(*reinterpret_cast<const device half *>(p));
}

// Decode one weight from a 17-byte GGUF MXFP4 block.
__attribute__((always_inline)) inline float mxfp4(const device uchar *block, uint j) {
    uint q = (block[1 + j % 16] >> (j / 16 * 4)) & 15;
    uint exponent = block[0];
    constexpr float mag[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    float scale = exponent == 0 ? as_type<float>(0x00400000u) : as_type<float>(exponent << 23);
    return (q & 8 ? -mag[q & 7] : mag[q & 7]) * scale;
}

__attribute__((always_inline)) inline float weight(const device uchar *w, uint i, uint type) {
    if (type == WeightFormat::F32)
        return reinterpret_cast<const device float *>(w)[i];
    if (type == WeightFormat::F16)
        return float(reinterpret_cast<const device half *>(w)[i]);
    if (type == WeightFormat::Bf16)
        return float(reinterpret_cast<const device bfloat *>(w)[i]);
    if (type == WeightFormat::Q4_0 || type == WeightFormat::Q4_1) {
        const device uchar *b = w + (i / 32) * (type == WeightFormat::Q4_0 ? 18 : 20);
        uint j = i % 32;
        uint q = (b[(type == WeightFormat::Q4_0 ? 2 : 4) + j % 16] >> (j / 16 * 4)) & 15;
        return type == WeightFormat::Q4_0 ? fp16(b) * (int(q) - 8) : fp16(b) * q + fp16(b + 2);
    }
    if (type == WeightFormat::Q8_0) {
        const device uchar *b = w + i / 32 * 34;
        return fp16(b) * float(reinterpret_cast<const device char *>(b + 2)[i % 32]);
    }
    if (type == WeightFormat::Mxfp4)
        return mxfp4(w + i / 32 * 17, i % 32);
    if (type == WeightFormat::Q4K || type == WeightFormat::Q5K) {
        const device uchar *b = w + i / 256 * (type == WeightFormat::Q4K ? 144 : 176);
        uint j = i % 256, s = j / 32;
        const device uchar *scales = b + 4;
        uint sc = s < 4 ? scales[s] & 63 : (scales[s + 4] & 15) | ((scales[s - 4] >> 6) << 4);
        uint mn = s < 4 ? scales[s + 4] & 63 : (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4);
        uint q = (b[(type == WeightFormat::Q4K ? 16 : 48) + j / 64 * 32 + j % 32] >> (s % 2 * 4)) & 15;
        if (type == WeightFormat::Q5K)
            q |= ((b[16 + j % 32] >> s) & 1) << 4;
        return fp16(b) * sc * q - fp16(b + 2) * mn;
    }
    // Q6_K: sixteen signed scales and 256 six-bit values per superblock.
    const device uchar *b = w + i / 256 * 210;
    uint j = i % 256, h = j / 128, q = j % 128 / 32, l = j % 32;
    uint lo = (b[h * 64 + (q % 2) * 32 + l] >> (q / 2 * 4)) & 15;
    uint hi = (b[128 + h * 32 + l] >> (q * 2)) & 3;
    int sc = reinterpret_cast<const device char *>(b + 192)[j / 16];
    return fp16(b + 208) * sc * (int(lo | (hi << 4)) - 32);
}

struct Matrix {
    uint k, n, m, route_major, routes, experts, has_bias;
};

kernel void embed(const device uchar *w [[buffer(0)]], const device uint *tokens [[buffer(1)]],
                  device bfloat *out [[buffer(2)]], constant Matrix &p [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= p.m * p.k)
        return;
    uint index = uint(tokens[i / p.k]) * p.k + i % p.k;
    // Copying native BF16 embeddings without decoding through float.
    if (weight_type == WeightFormat::Bf16)
        out[i] = reinterpret_cast<const device bfloat *>(w)[index];
    else
        out[i] = bfloat(weight(w, index, weight_type));
}
