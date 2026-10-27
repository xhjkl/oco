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
