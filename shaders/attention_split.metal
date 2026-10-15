// Decode grouped query heads together, splitting keys across independent workgroups.
// This entry accepts D in {64,96,128,256}, integral GQA, and 1..32 splits.
struct SplitAttentionParams {
    uint heads, kv_heads, dim, pos, window, sinks;
    float scale;
    uint m, splits;
};

[[kernel, max_total_threads_per_threadgroup(128)]] void
attention_gqa_split(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
                    const device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
                    device float *partials [[buffer(4)]], constant SplitAttentionParams &p [[buffer(5)]],
                    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sq[8 * 256];
    threadgroup float ss[8 * 32];
    threadgroup half sp[8 * 32];
    threadgroup float rowmax[8], rowsum[8], rowfactor[8];
    uint ratio = p.heads / p.kv_heads, head_tiles = (ratio + 7) / 8;
    uint kv = group.x / head_tiles, first_head = kv * ratio + (group.x % head_tiles) * 8;
    uint head_end = (kv + 1) * ratio, t = group.y, part = group.z, D = p.dim;
    uint kstride = p.kv_heads * D;
    uint end = p.pos + t + 1;
    uint begin = p.window && end > p.window ? end - p.window : 0;
    uint active = end - begin;
    uint span = active / p.splits, extra = active % p.splits;
    uint split_begin = begin + part * span + min(part, extra);
    uint split_end = split_begin + span + uint(part < extra);
    for (uint i = tid; i < 8 * D; i += 128) {
        uint h = first_head + i / D, d = i % D;
        float value = h < head_end ? float(q[(t * p.heads + h) * D + d]) * p.scale : 0.0f;
        sq[i] = half(value);
    }
    if (tid < 8) {
        uint h = first_head + tid;
        bool sink = p.sinks && part == 0 && h < head_end;
        // Keeping empty split metadata finite; zero mass removes it in the merge.
        rowmax[tid] = sink ? float(bfloat(sinks[h])) : -0x1.fffffep127f;
        rowsum[tid] = sink ? 1.0f : 0.0f;
    }
    simdgroup_float8x8 o[8];
#pragma unroll
    for (uint i = 0; i < 8; i++)
        o[i] = simdgroup_float8x8(0);
    uint tiles = D / 32, dbase = sg * (D / 4);
    uint quad = lane / 4, lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb = split_begin; kb < split_end; kb += 32) {
        simdgroup_float8x8 s(0);
        uint key0 = kb + 8 * sg;
        if (key0 < split_end) {
            const device half *kp = k + (key0 * p.kv_heads + kv) * D;
            for (uint d8 = 0; d8 < D; d8 += 8) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, sq + d8, D);
                if (key0 + 8 <= split_end) {
                    simdgroup_load(mk, kp + d8, kstride, ulong2(0, 0), true);
                } else {
                    for (uint j = 0; j < 2; j++) {
                        uint key = key0 + lc + j;
                        mk.thread_elements()[j] = key < split_end
                            ? k[(key * p.kv_heads + kv) * D + d8 + lr] : half(0);
                    }
                }
                simdgroup_multiply_accumulate(s, mq, mk, s);
            }
        }
        simdgroup_store(s, ss + 8 * sg, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < 8) {
            uint r = lane, h = first_head + r;
            float previous = rowmax[r], peak = previous, scores[32];
            for (uint c = 0; c < 32; c++) {
                bool valid = h < head_end && kb + c < split_end;
                scores[c] = valid ? ss[r * 32 + c] : -INFINITY;
                peak = max(peak, scores[c]);
            }
            bool empty = peak == -INFINITY;
            float factor = empty ? 1.0f : fast::exp(previous - peak), mass = 0;
            for (uint c = 0; c < 32; c++) {
                float e = empty ? 0.0f : fast::exp(scores[c] - peak);
                sp[r * 32 + c] = half(e);
                mass += e;
            }
            rowsum[r] = rowsum[r] * factor + mass;
            rowmax[r] = peak;
            rowfactor[r] = factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float factor = rowfactor[lr];
        for (uint i = 0; i < tiles; i++) {
            o[i].thread_elements()[0] *= factor;
            o[i].thread_elements()[1] *= factor;
        }
        for (uint kk = 0; kk < 32 && kb + kk < split_end; kk += 8) {
            simdgroup_half8x8 mp;
            simdgroup_load(mp, sp + kk, 32);
            const device half *vp = v + ((kb + kk) * p.kv_heads + kv) * D + dbase;
            for (uint i = 0; i < tiles; i++) {
                simdgroup_half8x8 mv;
                if (kb + kk + 8 <= split_end) {
                    simdgroup_load(mv, vp + i * 8, kstride);
                } else {
                    for (uint j = 0; j < 2; j++) {
                        uint key = kb + kk + lr;
                        mv.thread_elements()[j] = key < split_end
                            ? v[(key * p.kv_heads + kv) * D + dbase + i * 8 + lc + j] : half(0);
                    }
                }
                simdgroup_multiply_accumulate(o[i], mp, mv, o[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Keeping partials in FP32; only the final merged output rounds to BF16.
    uint h = first_head + lr;
    if (h >= head_end)
        return;
    uint base = ((t * p.heads + h) * p.splits + part) * (D + 2);
    if (sg == 0 && lc == 0) {
        partials[base] = rowmax[lr];
        partials[base + 1] = rowsum[lr];
    }
    for (uint i = 0; i < tiles; i++) {
#pragma unroll
        for (uint j = 0; j < 2; j++)
            partials[base + 2 + dbase + i * 8 + lc + j] = o[i].thread_elements()[j];
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void
attention_gqa_merge(const device float *partials [[buffer(0)]], device bfloat *out [[buffer(1)]],
                    constant SplitAttentionParams &p [[buffer(2)]],
                    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float factors[32], denominator;
    uint row = group.y * p.heads + group.x, stride = p.dim + 2;
    const device float *src = partials + row * p.splits * stride;
    if (sg == 0) {
        float peak = lane < p.splits ? src[lane * stride] : -INFINITY;
        float global = simd_max(peak);
        float factor = lane < p.splits ? fast::exp(peak - global) : 0.0f;
        float mass = lane < p.splits ? src[lane * stride + 1] : 0.0f;
        if (lane < p.splits)
            factors[lane] = factor;
        float total = simd_sum(mass * factor);
        if (lane == 0)
            denominator = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < p.dim; d += 128) {
        float value = 0;
        for (uint part = 0; part < p.splits; part++)
            value += src[part * stride + 2 + d] * factors[part];
        out[row * p.dim + d] = bfloat(value / denominator);
    }
}
