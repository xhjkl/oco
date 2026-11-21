// Single-query (and short-batch) attention with several keys in flight per SIMD group: a few
// lanes share one key, so each lane streams a contiguous slice of it with wide loads and the
// four or two keys a SIMD group scores per step hide each other's memory latency. The 32 or 16
// running softmax states merge once at the end through threadgroup memory.
template <uint D>
[[kernel, max_total_threads_per_threadgroup(256)]] void
attention_vector(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
                 const device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
                 device bfloat *out [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
                 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint LANES = D == 256 ? 16 : 8, SLOTS = 32 / LANES, PER_LANE = D / LANES;
    constexpr uint VECTORS = PER_LANE / 8, STATES = 8 * SLOTS;
    threadgroup float maxs[STATES], sums[STATES];
    threadgroup float outputs[STATES * D];
    uint h = group.x, t = group.y, kv = h / (p.heads / p.kv_heads);
    uint key_lane = lane % LANES, slot = lane / LANES, state = sg * SLOTS + slot;
    float qq[PER_LANE];
    const device bfloat *qp = q + (t * p.heads + h) * D + key_lane * PER_LANE;
#pragma unroll
    for (uint j = 0; j < PER_LANE; j++)
        qq[j] = float(qp[j]) * p.scale;
    float oo[PER_LANE];
#pragma unroll
    for (uint j = 0; j < PER_LANE; j++)
        oo[j] = 0;
    bool seed = p.sinks && state == 0;
    float mx = seed ? float(bfloat(sinks[h])) : -INFINITY, sm = seed ? 1.0f : 0.0f;
    uint end = key_end(p, t), begin = p.window && end > p.window ? end - p.window : 0;
    for (uint s = begin + state; s < end; s += STATES) {
        const device uint4 *kp = reinterpret_cast<const device uint4 *>(k + (s * p.kv_heads + kv) * D + key_lane * PER_LANE);
        const device uint4 *vp = reinterpret_cast<const device uint4 *>(v + (s * p.kv_heads + kv) * D + key_lane * PER_LANE);
        uint4 kw[VECTORS], vw[VECTORS];
#pragma unroll
        for (uint i = 0; i < VECTORS; i++) {
            kw[i] = kp[i];
            vw[i] = vp[i];
        }
        float score = 0;
#pragma unroll
        for (uint i = 0; i < VECTORS; i++) {
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                score += qq[i * 8 + 2 * j] * float(as_type<half>(ushort(kw[i][j] & 0xffffu)));
                score += qq[i * 8 + 2 * j + 1] * float(as_type<half>(ushort(kw[i][j] >> 16)));
            }
        }
        // Reduce across the lanes sharing this key; they form an aligned group of LANES lanes.
#pragma unroll
        for (uint shift = 1; shift < LANES; shift <<= 1)
            score += simd_shuffle_xor(score, shift);
        float nm = max(mx, score), f = fast::exp(mx - nm), e = fast::exp(score - nm);
        sm = sm * f + e;
        mx = nm;
#pragma unroll
        for (uint i = 0; i < VECTORS; i++) {
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                oo[i * 8 + 2 * j] = oo[i * 8 + 2 * j] * f + e * float(as_type<half>(ushort(vw[i][j] & 0xffffu)));
                oo[i * 8 + 2 * j + 1] = oo[i * 8 + 2 * j + 1] * f + e * float(as_type<half>(ushort(vw[i][j] >> 16)));
            }
        }
    }
    if (key_lane == 0) {
        maxs[state] = mx;
        sums[state] = sm;
    }
#pragma unroll
    for (uint j = 0; j < PER_LANE; j++)
        outputs[state * D + key_lane * PER_LANE + j] = oo[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Merge the running states: empty ones carry -inf peaks and zero mass.
    float peak = -INFINITY;
    for (uint i = 0; i < STATES; i++)
        peak = max(peak, maxs[i]);
    float den = 0;
    for (uint i = 0; i < STATES; i++)
        den += sums[i] * (maxs[i] == -INFINITY ? 0.0f : fast::exp(maxs[i] - peak));
    for (uint d = tid; d < D; d += 256) {
        float val = 0;
        for (uint i = 0; i < STATES; i++)
            val += outputs[i * D + d] * (maxs[i] == -INFINITY ? 0.0f : fast::exp(maxs[i] - peak));
        out[(t * p.heads + h) * D + d] = bfloat(val / den);
    }
}

template [[host_name("attention_vector_64")]] kernel void attention_vector<64>(
    const device bfloat *, const device half *, const device half *, const device float *,
    device bfloat *, constant AttentionParams &, uint3, uint, uint, uint);
template [[host_name("attention_vector_128")]] kernel void attention_vector<128>(
    const device bfloat *, const device half *, const device half *, const device float *,
    device bfloat *, constant AttentionParams &, uint3, uint, uint, uint);
template [[host_name("attention_vector_256")]] kernel void attention_vector<256>(
    const device bfloat *, const device half *, const device half *, const device float *,
    device bfloat *, constant AttentionParams &, uint3, uint, uint, uint);

// Split-key decode attention (flash-decoding) for one query row: one threadgroup per KV head and
// key chunk serves all G query heads of that KV head, so each key and value is read once. Each
// SIMD group takes every eighth key of the chunk with lanes splitting the head width; the eight
// running softmax states per head merge in threadgroup memory. Partials store, per head and
// chunk, the peak, the mass and the unnormalized output. Empty states keep a finite floor peak
// (fast math does not honor infinities), which exponentiates to zero weight.
template <uint D, uint G>
[[kernel, max_total_threads_per_threadgroup(256)]] void
attention_split(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
                const device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
                device float *partials [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
                uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]],
                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint PER = D / 32;
    threadgroup float stats[8 * G * 2];
    threadgroup float merged[8 * D];
    const uint kv = group.x, chunk = group.y, chunks = grid.y;
    const uint end = key_end(p, 0), begin = p.window && end > p.window ? end - p.window : 0;
    const uint span = (end - begin + chunks - 1) / chunks;
    const uint first = begin + chunk * span, last = min(first + span, end);
    float qq[G][PER], oo[G][PER], mx[G], sm[G];
#pragma unroll
    for (uint g = 0; g < G; ++g) {
        const device bfloat *qp = q + (kv * G + g) * D + lane * PER;
#pragma unroll
        for (uint j = 0; j < PER; ++j) {
            qq[g][j] = float(qp[j]) * p.scale;
            oo[g][j] = 0;
        }
        mx[g] = -FLT_MAX / 2;
        sm[g] = 0;
    }
    for (uint s = first + sg; s < last; s += 8) {
        const device half *kp = k + (s * p.kv_heads + kv) * D + lane * PER;
        const device half *vp = v + (s * p.kv_heads + kv) * D + lane * PER;
        float kk[PER], vv[PER];
#pragma unroll
        for (uint j = 0; j < PER; ++j) {
            kk[j] = float(kp[j]);
            vv[j] = float(vp[j]);
        }
#pragma unroll
        for (uint g = 0; g < G; ++g) {
            float score = 0;
#pragma unroll
            for (uint j = 0; j < PER; ++j)
                score += qq[g][j] * kk[j];
            score = simd_sum(score);
            float nm = max(mx[g], score), f = fast::exp(mx[g] - nm), e = fast::exp(score - nm);
            sm[g] = sm[g] * f + e;
            mx[g] = nm;
#pragma unroll
            for (uint j = 0; j < PER; ++j)
                oo[g][j] = oo[g][j] * f + e * vv[j];
        }
    }
    if (lane == 0) {
#pragma unroll
        for (uint g = 0; g < G; ++g) {
            stats[(sg * G + g) * 2] = mx[g];
            stats[(sg * G + g) * 2 + 1] = sm[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint g = 0; g < G; ++g) {
        float peak = -FLT_MAX / 2;
        for (uint i = 0; i < 8; ++i)
            peak = max(peak, stats[(i * G + g) * 2]);
        float own = fast::exp(mx[g] - peak);
#pragma unroll
        for (uint j = 0; j < PER; ++j)
            merged[sg * D + lane * PER + j] = oo[g][j] * own;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device float *out = partials + ((kv * G + g) * chunks + chunk) * (D + 2);
        for (uint d = tid; d < D; d += 256) {
            float value = 0;
            for (uint i = 0; i < 8; ++i)
                value += merged[i * D + d];
            out[2 + d] = value;
        }
        if (tid == 0) {
            float mass = 0;
            for (uint i = 0; i < 8; ++i)
                mass += stats[(i * G + g) * 2 + 1] * fast::exp(stats[(i * G + g) * 2] - peak);
            out[0] = peak;
            out[1] = mass;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Combine the `p.m` chunk partials of attention_split for each head; sinks join the denominator.
template <uint D>
[[kernel, max_total_threads_per_threadgroup(256)]] void
attention_merge(const device float *partials [[buffer(0)]], const device float *sinks [[buffer(1)]],
                device bfloat *out [[buffer(2)]], constant AttentionParams &p [[buffer(3)]],
                uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint chunks = p.m;
    const device float *base = partials + h * chunks * (D + 2);
    float peak = p.sinks ? float(bfloat(sinks[h])) : -FLT_MAX / 2;
    for (uint c = 0; c < chunks; ++c)
        peak = max(peak, base[c * (D + 2)]);
    float den = p.sinks ? fast::exp(float(bfloat(sinks[h])) - peak) : 0.0f;
    for (uint c = 0; c < chunks; ++c)
        den += base[c * (D + 2) + 1] * fast::exp(base[c * (D + 2)] - peak);
    for (uint d = tid; d < D; d += 256) {
        float value = 0;
        for (uint c = 0; c < chunks; ++c)
            value += base[c * (D + 2) + 2 + d] * fast::exp(base[c * (D + 2)] - peak);
        out[h * D + d] = bfloat(value / den);
    }
}

#define ATTENTION_SPLIT(D, G)                                                                      \
    template [[host_name("attention_split_" #D "_" #G)]] kernel void attention_split<D, G>(        \
        const device bfloat *, const device half *, const device half *, const device float *,     \
        device float *, constant AttentionParams &, uint3, uint3, uint, uint, uint);
ATTENTION_SPLIT(256, 6)
ATTENTION_SPLIT(128, 16)
ATTENTION_SPLIT(64, 8)
#define ATTENTION_MERGE(D)                                                                         \
    template [[host_name("attention_merge_" #D)]] kernel void attention_merge<D>(                  \
        const device float *, const device float *, device bfloat *, constant AttentionParams &,   \
        uint, uint);
ATTENTION_MERGE(256)
ATTENTION_MERGE(128)
ATTENTION_MERGE(64)
