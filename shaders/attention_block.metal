// Attention for long query batches: 32 queries per threadgroup, each SIMD group owning eight
// rows across every key it may see, with 32-key tiles of keys and values staged once per step.
// The online softmax runs in registers: a fragment row lives in four lanes, so row statistics
// reduce with two shuffles. Masking covers causal, windowed, and bidirectional (`keys`) use.
template <uint D>
[[kernel, max_total_threads_per_threadgroup(128)]] void
attention_block(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
                const device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
                device bfloat *out [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
                uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint KEYS = 32, ROWS = 32, STRIDE = D + 8, VECTORS = D / 8;
    threadgroup half sq[ROWS * STRIDE];
    threadgroup half sk[KEYS * STRIDE];
    threadgroup half sv[KEYS * STRIDE];
    uint h = group.x, t0 = group.y * ROWS, kv = h / (p.heads / p.kv_heads);
    for (uint i = tid; i < ROWS * D; i += 128) {
        uint r = i / D, d = i % D, t = t0 + r;
        sq[r * STRIDE + d] = half(t < p.m ? float(q[(t * p.heads + h) * D + d]) * p.scale : 0.0f);
    }
    uint quad = lane / 4, lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    uint row = t0 + sg * 8 + lr;
    uint last = min(t0 + ROWS - 1, p.m - 1);
    uint end = key_end(p, last);
    uint begin = p.window && p.pos + t0 + 1 > p.window ? p.pos + t0 + 1 - p.window : 0;
    // Rows past the batch see no keys and keep zero mass.
    uint row_end = row < p.m ? key_end(p, row) : 0;
    uint row_begin = p.window && p.pos + row + 1 > p.window ? p.pos + row + 1 - p.window : 0;
    float rowmax = p.sinks ? float(bfloat(sinks[h])) : -INFINITY, rowsum = p.sinks ? 1.0f : 0.0f;
    simdgroup_float8x8 o[VECTORS];
#pragma unroll
    for (uint i = 0; i < VECTORS; i++)
        o[i] = simdgroup_float8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb = begin; kb < end; kb += KEYS) {
        // Stage this step's keys and values, sixteen bytes per item, zero beyond the stored range.
        for (uint i = tid; i < KEYS * VECTORS; i += 128) {
            uint r = i / VECTORS, c = i % VECTORS * 8, key = kb + r;
            uint4 kw = 0, vw = 0;
            if (key < end) {
                uint at = (key * p.kv_heads + kv) * D + c;
                kw = *reinterpret_cast<const device uint4 *>(k + at);
                vw = *reinterpret_cast<const device uint4 *>(v + at);
            }
            *reinterpret_cast<threadgroup uint4 *>(sk + r * STRIDE + c) = kw;
            *reinterpret_cast<threadgroup uint4 *>(sv + r * STRIDE + c) = vw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[4];
#pragma unroll
        for (uint j = 0; j < 4; j++)
            s[j] = simdgroup_float8x8(0);
        for (uint d8 = 0; d8 < D; d8 += 8) {
            simdgroup_half8x8 mq;
            simdgroup_load(mq, sq + sg * 8 * STRIDE + d8, STRIDE);
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, sk + j * 8 * STRIDE + d8, STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], mq, mk, s[j]);
            }
        }
        // This lane holds columns lc and lc + 1 of each fragment for row lr.
        float scores[8];
        float peak = -INFINITY;
#pragma unroll
        for (uint j = 0; j < 4; j++) {
#pragma unroll
            for (uint e = 0; e < 2; e++) {
                uint key = kb + j * 8 + lc + e;
                bool valid = key < row_end && key >= row_begin;
                scores[j * 2 + e] = valid ? s[j].thread_elements()[e] : -INFINITY;
                peak = max(peak, scores[j * 2 + e]);
            }
        }
        peak = max(peak, simd_shuffle_xor(peak, 1));
        peak = max(peak, simd_shuffle_xor(peak, 8));
        float newmax = max(rowmax, peak);
        bool empty = newmax == -INFINITY;
        float factor = empty ? 1.0f : fast::exp(rowmax - newmax);
        float mass = 0;
        simdgroup_half8x8 mp[4];
#pragma unroll
        for (uint j = 0; j < 4; j++) {
#pragma unroll
            for (uint e = 0; e < 2; e++) {
                float weight = empty ? 0.0f : fast::exp(scores[j * 2 + e] - newmax);
                mp[j].thread_elements()[e] = half(weight);
                mass += weight;
            }
        }
        mass += simd_shuffle_xor(mass, 1);
        mass += simd_shuffle_xor(mass, 8);
        rowsum = rowsum * factor + mass;
        rowmax = newmax;
#pragma unroll
        for (uint i = 0; i < VECTORS; i++) {
            o[i].thread_elements()[0] *= factor;
            o[i].thread_elements()[1] *= factor;
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, sv + j * 8 * STRIDE + i * 8, STRIDE);
                simdgroup_multiply_accumulate(o[i], mp[j], mv, o[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row >= p.m)
        return;
    float inv = rowsum > 0.0f ? 1.0f / rowsum : 0.0f;
#pragma unroll
    for (uint i = 0; i < VECTORS; i++) {
#pragma unroll
        for (uint e = 0; e < 2; e++)
            out[(row * p.heads + h) * D + i * 8 + lc + e] = bfloat(o[i].thread_elements()[e] * inv);
    }
}

template [[host_name("attention_block_64")]] kernel void attention_block<64>(
    const device bfloat *, const device half *, const device half *, const device float *,
    device bfloat *, constant AttentionParams &, uint3, uint, uint, uint);
template [[host_name("attention_block_128")]] kernel void attention_block<128>(
    const device bfloat *, const device half *, const device half *, const device float *,
    device bfloat *, constant AttentionParams &, uint3, uint, uint, uint);
