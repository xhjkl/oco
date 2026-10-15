// Return one packed row's byte span; GGUF validation requires block-aligned rows.
inline uint packed_row_bytes(uint k) {
    switch (weight_type) {
        case WeightFormat::Q4_0: return k / 32 * 18;
        case WeightFormat::Q4_1: return k / 32 * 20;
        case WeightFormat::Q8_0: return k / 32 * 34;
        case WeightFormat::Q4K: return k / 256 * 144;
        case WeightFormat::Q5K: return k / 256 * 176;
        case WeightFormat::Q6K: return k / 256 * 210;
        case WeightFormat::F32: return k * 4;
        default: return k * 2;
    }
}

// Decode an aligned sixteen-coefficient slice, sharing its packed block header.
inline void decode_weights16(const device uchar *row, uint k, thread float *out) {
    if (weight_type == WeightFormat::Q8_0) {
        const device uchar *b = row + k / 32 * 34;
        float scale = fp16(b);
        const device char *q = reinterpret_cast<const device char *>(b + 2) + k % 32;
#pragma unroll
        for (uint j = 0; j < 16; ++j)
            out[j] = float(scale * float(q[j]));
    } else if (weight_type == WeightFormat::Q4_0 || weight_type == WeightFormat::Q4_1) {
        const device uchar *b = row + k / 32 * (weight_type == WeightFormat::Q4_0 ? 18 : 20);
        float scale = fp16(b), offset = weight_type == WeightFormat::Q4_1 ? fp16(b + 2) : 0;
        const device uchar *q = b + (weight_type == WeightFormat::Q4_0 ? 2 : 4);
        uint shift = k % 32 / 16 * 4;
#pragma unroll
        for (uint j = 0; j < 16; ++j) {
            uint value = (q[j] >> shift) & 15;
            out[j] = float(weight_type == WeightFormat::Q4_0 ? scale * (int(value) - 8)
                                                              : scale * value + offset);
        }
    } else if (weight_type == WeightFormat::Q4K || weight_type == WeightFormat::Q5K) {
        const device uchar *b = row + k / 256 * (weight_type == WeightFormat::Q4K ? 144 : 176);
        uint local = k % 256, section = local / 32;
        const device uchar *scales = b + 4;
        uint sc = section < 4 ? scales[section] & 63
                             : (scales[section + 4] & 15) | ((scales[section - 4] >> 6) << 4);
        uint mn = section < 4 ? scales[section + 4] & 63
                             : (scales[section + 4] >> 4) | ((scales[section] >> 6) << 4);
        float scale = fp16(b) * sc, minimum = fp16(b + 2) * mn;
        const device uchar *q = b + (weight_type == WeightFormat::Q4K ? 16 : 48)
                                   + local / 64 * 32 + local % 32;
#pragma unroll
        for (uint j = 0; j < 16; ++j) {
            uint value = (q[j] >> (section % 2 * 4)) & 15;
            if (weight_type == WeightFormat::Q5K)
                value |= ((b[16 + local % 32 + j] >> section) & 1) << 4;
            out[j] = float(scale * value - minimum);
        }
    } else if (weight_type == WeightFormat::Q6K) {
        const device uchar *b = row + k / 256 * 210;
        uint local = k % 256, h = local / 128, q = local % 128 / 32, l = local % 32;
        float scale = fp16(b + 208) * reinterpret_cast<const device char *>(b + 192)[local / 16];
#pragma unroll
        for (uint j = 0; j < 16; ++j) {
            uint lo = (b[h * 64 + (q % 2) * 32 + l + j] >> (q / 2 * 4)) & 15;
            uint hi = (b[128 + h * 32 + l + j] >> (q * 2)) & 3;
            out[j] = float(scale * (int(lo | (hi << 4)) - 32));
        }
    } else {
#pragma unroll
        for (uint j = 0; j < 16; ++j)
            out[j] = float(weight(row, k + j, weight_type));
    }
}

