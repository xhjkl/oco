struct Rope {
    uint heads, kv_heads, dim, rope_dim, m, pos, q_stride;
    float base, factor, low, high, scale;
};
kernel void rope(const device bfloat *q [[buffer(0)]], const device bfloat *k [[buffer(1)]],
                 const device bfloat *v [[buffer(2)]], device bfloat *qo [[buffer(3)]],
                 device half *kc [[buffer(4)]], device half *vc [[buffer(5)]],
                 constant Rope &p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    uint total = (p.heads + 2 * p.kv_heads) * p.dim;
    if (i >= p.m * total)
        return;
    uint t = i / total, h = i % total / p.dim, d = i % p.dim;
    bool isq = h < p.heads, isk = h >= p.heads && h < p.heads + p.kv_heads;
    uint head = isq ? h : (isk ? h - p.heads : h - p.heads - p.kv_heads);
    const device bfloat *src =
        isq ? q + t * p.heads * p.dim * p.q_stride + head * p.dim * p.q_stride
            : (isk ? k : v) + (t * p.kv_heads + head) * p.dim;
    float a = float(src[d]);
    if ((isq || isk) && d < p.rope_dim) {
        uint halfdim = p.rope_dim / 2, pair = d % halfdim,
             other = d < halfdim ? d + halfdim : d - halfdim;
        float freq = pow(p.base, -float(2 * pair) / float(p.rope_dim));
        float ramp = clamp((float(pair) - p.low) / (p.high - p.low), 0.0f, 1.0f);
        freq *= 1.0f - ramp + ramp / p.factor;
        float theta = float(p.pos + t) * freq;
        a = (a * cos(theta) + (d < halfdim ? -1.0f : 1.0f) * float(src[other]) * sin(theta)) *
            p.scale;
    }
    if (isq)
        qo[(t * p.heads + head) * p.dim + d] = bfloat(a);
    else if (isk)
        kc[((p.pos + t) * p.kv_heads + head) * p.dim + d] = half(a);
    else
        vc[((p.pos + t) * p.kv_heads + head) * p.dim + d] = half(a);
}

struct AttentionParams {
    uint heads, kv_heads, dim, pos, window, sinks;
    float scale;
    uint m;
};
// Keys and values are retained as half so prefill can load them straight into SIMD-group matrices.
// Online softmax and cross-SIMD reduction, with contiguous KV storage.
[[kernel, max_total_threads_per_threadgroup(1024)]] void attention(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
                      const device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
                      device bfloat *out [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float maxs[32], sums[32], outputs[1024];
    uint h = group.x, t = group.y, kv = h / (p.heads / p.kv_heads), elems = p.dim / 32;
    float qq[8], oo[8];
    for (uint j = 0; j < elems; j++) {
        qq[j] = float(q[(t * p.heads + h) * p.dim + lane * elems + j]) * p.scale;
        oo[j] = 0;
    }
    float mx = p.sinks && sg == 0 ? float(bfloat(sinks[h])) : -INFINITY,
          sm = p.sinks && sg == 0 ? 1 : 0;
    uint end = p.pos + t + 1, begin = p.window && end > p.window ? end - p.window : 0;
    for (uint s = begin + sg; s < end; s += 32) {
        float score = 0;
        for (uint j = 0; j < elems; j++)
            score += qq[j] * float(k[(s * p.kv_heads + kv) * p.dim + lane * elems + j]);
        score = simd_sum(score);
        float nm = max(mx, score), f = fast::exp(mx - nm), e = fast::exp(score - nm);
        sm = sm * f + e;
        mx = nm;
        for (uint j = 0; j < elems; j++)
            oo[j] = oo[j] * f + e * float(v[(s * p.kv_heads + kv) * p.dim + lane * elems + j]);
    }
    if (lane == 0) {
        maxs[sg] = mx;
        sums[sg] = sm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global = simd_max(maxs[lane]);
    float factor = fast::exp(maxs[lane] - global);
    float den = simd_sum(sums[lane] * factor);
    for (uint j = 0; j < elems; j++) {
        outputs[lane * 32 + sg] = oo[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float val = simd_sum(outputs[sg * 32 + lane] * factor) / den;
        if (lane == 0)
            out[(t * p.heads + h) * p.dim + sg * elems + j] = bfloat(val);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Reducing complete output rows once; short key ranges need fewer cooperating SIMD groups.
template <uint Groups>
kernel void attention_compact(const device bfloat *q [[buffer(0)]],
                              const device half *k [[buffer(1)]],
                              const device half *v [[buffer(2)]],
                              const device float *sinks [[buffer(3)]],
                              device bfloat *out [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float maxs[Groups], sums[Groups], outputs[Groups * 256];
    uint h = group.x, t = group.y, kv = h / (p.heads / p.kv_heads), elems = p.dim / 32;
    float qq[8], oo[8];
    for (uint j = 0; j < elems; j++) {
        qq[j] = float(q[(t * p.heads + h) * p.dim + lane * elems + j]) * p.scale;
        oo[j] = 0;
    }
    // Keeping empty groups finite; their zero mass removes them from the reduction.
    float mx = p.sinks && sg == 0 ? float(bfloat(sinks[h])) : -0x1.fffffep127f;
    float sm = p.sinks && sg == 0 ? 1 : 0;
    uint end = p.pos + t + 1, begin = p.window && end > p.window ? end - p.window : 0;
    for (uint s = begin + sg; s < end; s += Groups) {
        float score = 0;
        for (uint j = 0; j < elems; j++)
            score += qq[j] * float(k[(s * p.kv_heads + kv) * p.dim + lane * elems + j]);
        score = simd_sum(score);
        float nm = max(mx, score), f = fast::exp(mx - nm), e = fast::exp(score - nm);
        sm = sm * f + e;
        mx = nm;
        for (uint j = 0; j < elems; j++)
            oo[j] = oo[j] * f + e * float(v[(s * p.kv_heads + kv) * p.dim + lane * elems + j]);
    }
    if (lane == 0) {
        maxs[sg] = mx;
        sums[sg] = sm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_max = lane < Groups ? maxs[lane] : -0x1.fffffep127f;
    float mass = lane < Groups ? sums[lane] : 0;
    float global = simd_max(local_max);
    float den = simd_sum(mass * fast::exp(local_max - global));
    float factor = fast::exp(mx - global);
    for (uint j = 0; j < elems; j++)
        outputs[(lane * elems + j) * Groups + sg] = oo[j] * factor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < p.dim; d += Groups * 32) {
        float val = 0;
#pragma unroll
        for (uint g = 0; g < Groups; g++)
            val += outputs[d * Groups + g];
        out[(t * p.heads + h) * p.dim + d] = bfloat(val / den);
    }
}

template [[host_name("attention_256")]] kernel void attention_compact<8>(
    const device bfloat *, const device half *, const device half *, const device float *,
    device bfloat *, constant AttentionParams &, uint3, uint, uint, uint);

// Query-tiled attention for prompts: eight queries of one head per threadgroup and 32 keys per
// step. Scores and weighted values use SIMD-group matrices with FP32 accumulation; the online
// softmax runs in FP32 with half probabilities. Each SIMD group scores eight keys, then
// accumulates a quarter of the head dimension. Sinks seed the running maximum and mass, and
// windows and causality mask per query row. Partial key tiles are zero-filled before multiplication.
[[kernel, max_total_threads_per_threadgroup(128)]] void
attention_tiled(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
                const device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
                device bfloat *out [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
                uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sq[8 * 256];
    threadgroup float ss[8 * 32];
    threadgroup half sp[8 * 32];
    threadgroup float rowmax[8], rowsum[8], rowfactor[8];
    uint h = group.x, t0 = group.y * 8, kv = h / (p.heads / p.kv_heads), D = p.dim;
    uint kstride = p.kv_heads * D;
    for (uint i = tid; i < 8 * D; i += 128) {
        uint r = i / D, d = i % D, t = t0 + r;
        float value = t < p.m ? float(q[(t * p.heads + h) * D + d]) * p.scale : 0.0f;
        sq[i] = half(value);
    }
    if (tid < 8) {
        rowmax[tid] = p.sinks ? float(bfloat(sinks[h])) : -INFINITY;
        rowsum[tid] = p.sinks ? 1.0f : 0.0f;
    }
    simdgroup_float8x8 o[8];
#pragma unroll
    for (uint i = 0; i < 8; i++)
        o[i] = simdgroup_float8x8(0);
    uint tiles = D / 32, dbase = sg * (D / 4);
    uint quad = lane / 4, lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    uint end = p.pos + min(t0 + 7, p.m - 1) + 1;
    uint begin = p.window && p.pos + t0 + 1 > p.window ? p.pos + t0 + 1 - p.window : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb = begin; kb < end; kb += 32) {
        simdgroup_float8x8 s(0);
        uint key0 = kb + 8 * sg;
        if (key0 < end) {
            const device half *kp = k + (key0 * p.kv_heads + kv) * D;
            for (uint d8 = 0; d8 < D; d8 += 8) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, sq + d8, D);
                if (key0 + 8 <= end) {
                    simdgroup_load(mk, kp + d8, kstride, ulong2(0, 0), true);
                } else {
                    // Zeroing the last fragment; masked probabilities must never multiply stale NaNs.
                    for (uint j = 0; j < 2; j++) {
                        uint key = key0 + lc + j;
                        mk.thread_elements()[j] = key < end
                            ? k[(key * p.kv_heads + kv) * D + d8 + lr] : half(0);
                    }
                }
                simdgroup_multiply_accumulate(s, mq, mk, s);
            }
        }
        simdgroup_store(s, ss + 8 * sg, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < 8) {
            uint r = lane, t = t0 + r, qpos = p.pos + t;
            uint rbegin = p.window && qpos + 1 > p.window ? qpos + 1 - p.window : 0;
            float previous = rowmax[r], peak = previous;
            float scores[32];
            for (uint c = 0; c < 32; c++) {
                uint key = kb + c;
                bool valid = t < p.m && key <= qpos && key >= rbegin;
                scores[c] = valid ? ss[r * 32 + c] : -INFINITY;
                peak = max(peak, scores[c]);
            }
            // Fully masked rows keep zero mass without producing NaN from exp(-inf - -inf).
            bool empty = peak == -INFINITY;
            float factor = empty ? 1.0f : fast::exp(previous - peak);
            float mass = 0;
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
        for (uint kk = 0; kk < 32 && kb + kk < end; kk += 8) {
            simdgroup_half8x8 mp;
            simdgroup_load(mp, sp + kk, 32);
            const device half *vp = v + ((kb + kk) * p.kv_heads + kv) * D + dbase;
            for (uint i = 0; i < tiles; i++) {
                simdgroup_half8x8 mv;
                if (kb + kk + 8 <= end) {
                    simdgroup_load(mv, vp + i * 8, kstride);
                } else {
                    for (uint j = 0; j < 2; j++) {
                        uint key = kb + kk + lr;
                        mv.thread_elements()[j] = key < end
                            ? v[(key * p.kv_heads + kv) * D + dbase + i * 8 + lc + j] : half(0);
                    }
                }
                simdgroup_multiply_accumulate(o[i], mp, mv, o[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint t = t0 + lr;
    if (t >= p.m)
        return;
    float inv = 1.0f / rowsum[lr];
    for (uint i = 0; i < tiles; i++) {
#pragma unroll
        for (uint j = 0; j < 2; j++)
            out[(t * p.heads + h) * D + dbase + i * 8 + lc + j] =
                bfloat(o[i].thread_elements()[j] * inv);
    }
}

struct QkNorm {
    uint dim, stride;
    float epsilon;
};
kernel void qk_norm(device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                    constant QkNorm &p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    uint offset = row * p.dim * p.stride;
    float val = tid < p.dim ? float(x[offset + tid]) : 0, s = simd_sum(val * val);
    if (lane == 0)
        sums[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(tid < 8 ? sums[tid] : 0);
    threadgroup float inv;
    if (tid == 0)
        inv = rsqrt(s / float(p.dim) + p.epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.dim)
        x[offset + tid] = bfloat(float(bfloat(val * inv)) * float(bfloat(w[tid])));
}
struct AttentionGateParams {
    uint n, dim;
};
kernel void attention_gate(device bfloat *out [[buffer(0)]], const device bfloat *q [[buffer(1)]],
                      constant AttentionGateParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n)
        out[i] = bfloat(float(out[i]) /
                        (1 + fast::exp(-float(q[i / p.dim * 2 * p.dim + p.dim + i % p.dim]))));
}
