// Inkling 2 (inkling2_loop): residual streams mixed over the depth, causal short convolutions,
// clipped attention scored with a relative-position bias and per-query sinks, and routed
// experts with the SITU activation. Activations are BF16; reductions run in FP32.

struct InklingMix {
    uint n;       // stream width, at most 9 channels per thread
    uint states;  // block states before the current stream, 0 to 4
    uint queries; // learned queries, 1 to 4
    uint norms;   // queries whose result is RMS-normalized with a learned gain (0 or `queries`)
    float epsilon;
    float norm_epsilon;
};

// Attention over the depth, for one row per threadgroup: RMS-normalize each block state and the
// current stream, score them against each learned query (scaled by 1/sqrt(n)), and write the
// softmax-weighted sum of the normalized states, optionally RMS-normalized again with the next
// operation's learned gain. Order does not matter: the mix is symmetric.
[[kernel, max_total_threads_per_threadgroup(1024)]] void
inkling_mix(const device bfloat *s0 [[buffer(0)]], const device bfloat *s1 [[buffer(1)]],
            const device bfloat *s2 [[buffer(2)]], const device bfloat *s3 [[buffer(3)]],
            const device bfloat *cur [[buffer(4)]], const device float *query [[buffer(5)]],
            device bfloat *out0 [[buffer(6)]], device bfloat *out1 [[buffer(7)]],
            device bfloat *out2 [[buffer(8)]], device bfloat *out3 [[buffer(9)]],
            const device float *gain0 [[buffer(10)]], const device float *gain1 [[buffer(11)]],
            const device float *gain2 [[buffer(12)]], constant InklingMix &p [[buffer(13)]],
            uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
            uint threads [[threads_per_threadgroup]]) {
    constexpr uint S = 5, Q = 4, SUMS = S * (Q + 1), CH = 9;
    threadgroup float partial[32][SUMS];
    threadgroup float weights[Q][S];
    threadgroup float scales[Q];
    const uint states = p.states + 1, groups = threads / 32;
    const device bfloat *rows[S] = {s0 + t * p.n, s1 + t * p.n, s2 + t * p.n, s3 + t * p.n, cur + t * p.n};
    rows[p.states] = cur + t * p.n;
    device bfloat *outs[Q] = {out0 + t * p.n, out1 + t * p.n, out2 + t * p.n, out3 + t * p.n};
    const device float *gains[3] = {gain0, gain1, gain2};
    float sums[SUMS];
#pragma unroll
    for (uint i = 0; i < SUMS; i++)
        sums[i] = 0;
    for (uint c = tid; c < p.n; c += threads) {
        float qv[Q];
#pragma unroll
        for (uint q = 0; q < Q; q++)
            qv[q] = q < p.queries ? query[q * p.n + c] : 0.0f;
#pragma unroll
        for (uint s = 0; s < S; s++) {
            if (s >= states)
                continue;
            float x = float(rows[s][c]);
            sums[s * (Q + 1)] = fma(x, x, sums[s * (Q + 1)]);
#pragma unroll
            for (uint q = 0; q < Q; q++)
                sums[s * (Q + 1) + 1 + q] = fma(x, qv[q], sums[s * (Q + 1) + 1 + q]);
        }
    }
#pragma unroll
    for (uint i = 0; i < SUMS; i++) {
        float v = simd_sum(sums[i]);
        if (lane == 0)
            partial[sg][i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.queries) {
        uint q = tid;
        float inv[S], score[S], peak = -INFINITY;
        for (uint s = 0; s < states; s++) {
            float sq = 0, dot = 0;
            for (uint g = 0; g < groups; g++) {
                sq += partial[g][s * (Q + 1)];
                dot += partial[g][s * (Q + 1) + 1 + q];
            }
            inv[s] = rsqrt(sq / float(p.n) + p.epsilon);
            score[s] = dot * inv[s] * rsqrt(float(p.n));
            peak = max(peak, score[s]);
        }
        float total = 0;
        for (uint s = 0; s < states; s++) {
            score[s] = exp(score[s] - peak);
            total += score[s];
        }
        for (uint s = 0; s < states; s++)
            weights[q][s] = score[s] / total * inv[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mixed[Q][CH], squares[Q] = {0, 0, 0, 0};
    for (uint i = 0, c = tid; i < CH && c < p.n; i++, c += threads) {
        float x[S];
#pragma unroll
        for (uint s = 0; s < S; s++)
            x[s] = s < states ? float(rows[s][c]) : 0.0f;
#pragma unroll
        for (uint q = 0; q < Q; q++) {
            float v = 0;
#pragma unroll
            for (uint s = 0; s < S; s++)
                if (s < states)
                    v = fma(weights[q][s], x[s], v);
            mixed[q][i] = v;
            squares[q] = fma(v, v, squares[q]);
        }
    }
    if (p.norms) {
        // Reusing the first partial rows: every thread has read the weights above.
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint q = 0; q < Q; q++) {
            float v = simd_sum(squares[q]);
            if (lane == 0)
                partial[sg][q] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < p.norms) {
            float total = 0;
            for (uint g = 0; g < groups; g++)
                total += partial[g][tid];
            scales[tid] = rsqrt(total / float(p.n) + p.norm_epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0, c = tid; i < CH && c < p.n; i++, c += threads) {
        for (uint q = 0; q < p.queries; q++) {
            float v = mixed[q][i];
            if (p.norms)
                v = v * scales[q] * gains[q][c];
            outs[q][c] = bfloat(v);
        }
    }
}

struct InklingRows {
    uint n;
    uint m;
    float epsilon;
};

// Loop-controller input: [rmsnorm(cur); rmsnorm(message)] per row, without learned gains.
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling_control_input(const device bfloat *cur [[buffer(0)]], const device bfloat *message [[buffer(1)]],
                      device bfloat *out [[buffer(2)]], constant InklingRows &p [[buffer(3)]],
                      uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8][2];
    const device bfloat *a = cur + t * p.n, *b = message + t * p.n;
    float sa = 0, sb = 0;
    for (uint c = tid; c < p.n; c += 256) {
        float x = float(a[c]), y = float(b[c]);
        sa = fma(x, x, sa);
        sb = fma(y, y, sb);
    }
    sa = simd_sum(sa);
    sb = simd_sum(sb);
    if (lane == 0) {
        partial[sg][0] = sa;
        partial[sg][1] = sb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ta = 0, tb = 0;
    for (uint g = 0; g < 8; g++) {
        ta += partial[g][0];
        tb += partial[g][1];
    }
    float ia = rsqrt(ta / float(p.n) + p.epsilon), ib = rsqrt(tb / float(p.n) + p.epsilon);
    device bfloat *o = out + t * 2 * p.n;
    for (uint c = tid; c < p.n; c += 256) {
        o[c] = bfloat(float(a[c]) * ia);
        o[p.n + c] = bfloat(float(b[c]) * ib);
    }
}

struct InklingControl {
    uint n;
    uint norm; // RMS-normalize the result with `w` (the next operation's input norm)
    float epsilon;
};

// Loop-controller output, one row per threadgroup: cur - tanh(forget) * cur + tanh(gain) * message,
// optionally RMS-normalized with a learned gain. `out` may alias `cur`.
[[kernel, max_total_threads_per_threadgroup(1024)]] void
inkling_control(const device bfloat *cur [[buffer(0)]], const device bfloat *message [[buffer(1)]],
                const device bfloat *gain [[buffer(2)]], const device bfloat *forget [[buffer(3)]],
                device bfloat *out [[buffer(4)]], const device float *w [[buffer(5)]],
                constant InklingControl &p [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    constexpr uint CH = 9;
    threadgroup float partial[32];
    const uint base = t * p.n;
    float y[CH], sq = 0;
    for (uint i = 0, c = tid; i < CH && c < p.n; i++, c += threads) {
        float x = float(cur[base + c]);
        y[i] = x - precise::tanh(float(forget[base + c])) * x + precise::tanh(float(gain[base + c])) * float(message[base + c]);
        sq = fma(y[i], y[i], sq);
    }
    float scale = 1.0f;
    if (p.norm) {
        sq = simd_sum(sq);
        if (lane == 0)
            partial[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0;
        for (uint g = 0; g < threads / 32; g++)
            total += partial[g];
        scale = rsqrt(total / float(p.n) + p.epsilon);
    }
    for (uint i = 0, c = tid; i < CH && c < p.n; i++, c += threads)
        out[base + c] = bfloat(p.norm ? y[i] * scale * w[c] : y[i]);
}

struct InklingConv {
    uint channels;
    uint m;
    uint stride;     // input elements between rows
    uint head_width; // channels sharing one set of dynamic taps
    uint taps;       // dynamic tap elements per row (four per head), zero for static only
    float scale;
    uint accumulate; // add into `out` instead of overwriting it
    uint routes;     // input is the probability-weighted sum of this many routed rows per row
};

// Causal short convolution (four taps) with a residual, for one channel per thread over the rows
// in order: y = (x + sum_k w[k] x[t - 3 + k] + dynamic taps) * scale. The three most recent
// inputs of each channel carry across calls in FP32. Dynamic taps (full-attention keys and values)
// weigh lags 0..3 by per-row, per-head coefficients.
kernel void inkling_conv(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                         const device bfloat *taps [[buffer(2)]], device float *state [[buffer(3)]],
                         device bfloat *out [[buffer(4)]], const device float *probs [[buffer(5)]],
                         constant InklingConv &p [[buffer(6)]], uint c [[thread_position_in_grid]]) {
    if (c >= p.channels)
        return;
    float a = state[c * 3], b = state[c * 3 + 1], d = state[c * 3 + 2];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    uint head = c / p.head_width;
    for (uint t = 0; t < p.m; t++) {
        float v = 0;
        if (p.routes) {
            for (uint r = 0; r < p.routes; r++)
                v = fma(float(x[(t * p.routes + r) * p.channels + c]), probs[t * p.routes + r], v);
        } else {
            v = float(x[t * p.stride + c]);
        }
        float s = fma(w0, a, fma(w1, b, fma(w2, d, w3 * v)));
        if (p.taps) {
            const device bfloat *tp = taps + t * p.taps + head * 4;
            s = fma(float(tp[0]), v, fma(float(tp[1]), d, fma(float(tp[2]), b, fma(float(tp[3]), a, s))));
        }
        float y = (v + s) * p.scale;
        uint o = t * p.channels + c;
        out[o] = bfloat(p.accumulate ? float(out[o]) + y : y);
        a = b;
        b = d;
        d = v;
    }
    state[c * 3] = a;
    state[c * 3 + 1] = b;
    state[c * 3 + 2] = d;
}

struct InklingQkv {
    uint heads, kv_heads, dim, rel;
    uint m, pos;
    float clip, rel_clip, epsilon, alpha;
    uint floor, scaled; // log-scaled queries and relative vectors (full-attention layers)
    uint conv;          // single row: convolve raw keys and values here (with their state)
    uint taps;          // dynamic taps per KV head (full-attention layers), else zero
    float conv_scale;
    uint ring;          // rows of a sliding-window ring cache (position p at row p % ring), or 0
};

// Per head vector, one SIMD group each: clip queries (after their per-channel scale) and keys to
// an L2 norm of at most `clip`, RMS-normalize values without gains, clip the relative vectors
// (after their scale) to `rel_clip`, and log-scale queries and relative vectors by position on
// full-attention layers. Keys and values go to the cache as half at their sequence positions (or
// their ring rows); queries stay BF16 in place and relative vectors become FP32.
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling_qkv(device bfloat *q [[buffer(0)]], const device bfloat *k [[buffer(1)]],
            const device bfloat *v [[buffer(2)]], const device bfloat *r [[buffer(3)]],
            const device float *q_scale [[buffer(4)]], const device float *r_scale [[buffer(5)]],
            device half *k_cache [[buffer(6)]], device half *v_cache [[buffer(7)]],
            device float *r_out [[buffer(8)]], device float *state [[buffer(9)]],
            const device float *k_conv [[buffer(10)]], const device float *v_conv [[buffer(11)]],
            const device bfloat *k_taps [[buffer(12)]], const device bfloat *v_taps [[buffer(13)]],
            constant InklingQkv &p [[buffer(14)]],
            uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint per_row = 2 * p.heads + 2 * p.kv_heads, item = group * 8 + sg;
    if (item >= p.m * per_row)
        return;
    const uint t = item / per_row, j = item % per_row, pos = p.pos + t, row = p.ring ? pos % p.ring : pos;
    const float tau = p.scaled ? 1.0f + p.alpha * log(max(1.0f, float(pos + 1) / float(p.floor))) : 1.0f;
    constexpr uint E = 4; // 128-wide heads over 32 lanes
    if (j < p.heads + 2 * p.kv_heads) {
        float x[E], sq = 0;
        uint h;
        if (j < p.heads) {
            h = j;
            const device bfloat *src = q + (t * p.heads + h) * p.dim + lane * E;
#pragma unroll
            for (uint e = 0; e < E; e++)
                x[e] = float(src[e]) * q_scale[h * p.dim + lane * E + e];
        } else {
            h = (j - p.heads) % p.kv_heads;
            const bool is_v = j >= p.heads + p.kv_heads;
            const device bfloat *src = (is_v ? v : k) + (t * p.kv_heads + h) * p.dim + lane * E;
#pragma unroll
            for (uint e = 0; e < E; e++)
                x[e] = float(src[e]);
            if (p.conv) {
                // One row: the causal four-tap convolution of inkling_conv, fused, state included.
                const uint first = h * p.dim + lane * E;
                device float *st = state + ((is_v ? p.kv_heads * p.dim : 0) + first) * 3;
                const device float *w = (is_v ? v_conv : k_conv) + first * 4;
                float4 tp = 0;
                if (p.taps)
                    tp = float4(*reinterpret_cast<const device bfloat4 *>((is_v ? v_taps : k_taps) + h * 4));
#pragma unroll
                for (uint e = 0; e < E; e++) {
                    float a = st[e * 3], b = st[e * 3 + 1], d = st[e * 3 + 2], raw = x[e];
                    float s = fma(w[e * 4], a, fma(w[e * 4 + 1], b, fma(w[e * 4 + 2], d, w[e * 4 + 3] * raw)));
                    if (p.taps)
                        s = fma(tp.x, raw, fma(tp.y, d, fma(tp.z, b, fma(tp.w, a, s))));
                    x[e] = (raw + s) * p.conv_scale;
                    st[e * 3] = b;
                    st[e * 3 + 1] = d;
                    st[e * 3 + 2] = raw;
                }
            }
        }
#pragma unroll
        for (uint e = 0; e < E; e++)
            sq = fma(x[e], x[e], sq);
        sq = simd_sum(sq);
        if (j < p.heads + p.kv_heads) {
            float f = clamp(p.clip / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
            if (j < p.heads) {
                device bfloat *dst = q + (t * p.heads + h) * p.dim + lane * E;
#pragma unroll
                for (uint e = 0; e < E; e++)
                    dst[e] = bfloat(x[e] * f * tau);
            } else {
                device half *dst = k_cache + (ulong(row) * p.kv_heads + h) * p.dim + lane * E;
#pragma unroll
                for (uint e = 0; e < E; e++)
                    dst[e] = half(x[e] * f);
            }
        } else {
            float f = rsqrt(sq / float(p.dim) + p.epsilon);
            device half *dst = v_cache + (ulong(row) * p.kv_heads + h) * p.dim + lane * E;
#pragma unroll
            for (uint e = 0; e < E; e++)
                dst[e] = half(x[e] * f);
        }
        return;
    }
    // Relative vectors: p.rel (16) lanes per head.
    uint h = j - p.heads - 2 * p.kv_heads;
    float x = 0;
    if (lane < p.rel)
        x = float(r[(t * p.heads + h) * p.rel + lane]) * r_scale[h * p.rel + lane];
    float sq = simd_sum(x * x);
    float f = clamp(p.rel_clip / (sqrt(sq) + 1e-6f), 0.0f, 1.0f);
    if (lane < p.rel)
        r_out[(t * p.heads + h) * p.rel + lane] = x * f * tau;
}

struct InklingAttention {
    uint heads, kv_heads, pos, window, m, extent;
    float scale;
    uint ring; // rows of a sliding-window ring cache (position s at row s % ring), or 0
};

// Single-query and short-batch attention over 128-wide heads, after attention_vector: eight lanes
// share one key, four keys per SIMD group step, 32 running softmax states per threadgroup.
// Each score adds the query's relative-position bias r . table[d] for distances d below the
// extent (each of the eight lanes contributes two of the sixteen terms before the reduction).
// The per-query, per-head sink logit seeds the softmax mass, and the output is gated by
// sigmoid(gate) before the output projection.
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling_attention(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
                  const device half *v [[buffer(2)]], const device float *r [[buffer(3)]],
                  const device float *table [[buffer(4)]], const device bfloat *sinks [[buffer(5)]],
                  const device bfloat *gate [[buffer(6)]], device bfloat *out [[buffer(7)]],
                  constant InklingAttention &p [[buffer(8)]],
                  uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128, LANES = 8, SLOTS = 32 / LANES, PER_LANE = D / LANES;
    constexpr uint VECTORS = PER_LANE / 8, STATES = 8 * SLOTS, REL = 16, RPL = REL / LANES;
    threadgroup float maxs[STATES], sums[STATES];
    threadgroup float outputs[STATES * D];
    uint h = group.x, t = group.y, kv = h / (p.heads / p.kv_heads);
    uint key_lane = lane % LANES, slot = lane / LANES, state = sg * SLOTS + slot;
    float qq[PER_LANE];
    const device bfloat *qp = q + (t * p.heads + h) * D + key_lane * PER_LANE;
#pragma unroll
    for (uint j = 0; j < PER_LANE; j++)
        qq[j] = float(qp[j]) * p.scale;
    float rr[RPL];
#pragma unroll
    for (uint j = 0; j < RPL; j++)
        rr[j] = r[(t * p.heads + h) * REL + key_lane * RPL + j];
    float oo[PER_LANE];
#pragma unroll
    for (uint j = 0; j < PER_LANE; j++)
        oo[j] = 0;
    bool seed = state == 0;
    float mx = seed ? float(sinks[t * p.heads + h]) : -INFINITY, sm = seed ? 1.0f : 0.0f;
    uint qpos = p.pos + t, end = qpos + 1, begin = p.window && end > p.window ? end - p.window : 0;
    // Cache rows advance with positions and wrap at the ring size (never, for linear caches).
    const uint wrap = p.ring ? p.ring : 0xffffffffu;
    uint row = p.ring ? (begin + state) % p.ring : begin + state;
    for (uint s = begin + state; s < end; s += STATES, row = row + STATES >= wrap ? row + STATES - wrap : row + STATES) {
        const device uint4 *kp = reinterpret_cast<const device uint4 *>(k + (row * p.kv_heads + kv) * D + key_lane * PER_LANE);
        const device uint4 *vp = reinterpret_cast<const device uint4 *>(v + (row * p.kv_heads + kv) * D + key_lane * PER_LANE);
        uint4 kw[VECTORS], vw[VECTORS];
#pragma unroll
        for (uint i = 0; i < VECTORS; i++) {
            kw[i] = kp[i];
            vw[i] = vp[i];
        }
        float score = 0;
        uint distance = qpos - s;
        if (distance < p.extent) {
            const device float *tb = table + distance * REL + key_lane * RPL;
#pragma unroll
            for (uint j = 0; j < RPL; j++)
                score = fma(rr[j], tb[j], score);
        }
#pragma unroll
        for (uint i = 0; i < VECTORS; i++) {
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                score += qq[i * 8 + 2 * j] * float(as_type<half>(ushort(kw[i][j] & 0xffffu)));
                score += qq[i * 8 + 2 * j + 1] * float(as_type<half>(ushort(kw[i][j] >> 16)));
            }
        }
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
        uint o = (t * p.heads + h) * D + d;
        float g = float(gate[o]);
        out[o] = bfloat(val / den / (1.0f + exp(-g)));
    }
}

struct InklingSitu {
    uint width; // expert intermediate width (half of the merged gate/up rows)
    uint n;     // routes times width
    float beta, linear_beta;
};

// SITU over merged [gate | up] expert rows: beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb).
kernel void inkling_situ(const device bfloat *x [[buffer(0)]], device bfloat *out [[buffer(1)]],
                         constant InklingSitu &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n)
        return;
    uint route = i / p.width, j = i % p.width;
    float g = float(x[route * 2 * p.width + j]), u = float(x[route * 2 * p.width + p.width + j]);
    float act = p.beta * precise::tanh(g / p.beta) / (1.0f + exp(-g));
    out[i] = bfloat(act * p.linear_beta * precise::tanh(u / p.linear_beta));
}

struct InklingCombine {
    uint hidden, batch, routes;
};

// Weighted sum of each row's routed expert outputs for multi-row chunks, written (not added) for
// the post-convolution; single rows fold this sum into inkling_conv instead.
kernel void inkling_combine(const device bfloat *routes [[buffer(0)]], const device float *weights [[buffer(1)]],
                            device bfloat *out [[buffer(2)]], constant InklingCombine &p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.hidden * p.batch)
        return;
    uint t = i / p.hidden, d = i % p.hidden;
    float s = 0;
    for (uint r = 0; r < p.routes; r++)
        s = fma(float(routes[(t * p.routes + r) * p.hidden + d]), weights[t * p.routes + r], s);
    out[i] = bfloat(s);
}

// Router logits, eight experts per threadgroup (one per SIMD group): a single decode row spreads
// over sixteen threadgroups instead of one. Same FP32 accumulation order as router_sigmoid.
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling_router_logits(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                      device float *logits [[buffer(2)]], constant RouterParams &p [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    uint e = group.x * 8 + sg, t = group.y;
    if (e >= p.experts)
        return;
    float4 accum = 0;
    for (uint k = lane * 4; k < p.n; k += 128) {
        float4 a = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(x + t * p.n + k));
        float4 b = *reinterpret_cast<const device float4 *>(w + e * p.n + k);
        accum = fma(a, b, accum);
    }
    float sum = simd_sum((accum.x + accum.y) + (accum.z + accum.w));
    if (lane == 0)
        logits[t * p.experts + e] = sum;
}

// Sigmoid top-k over precomputed logits, one SIMD group per row, by router_sigmoid's rule: select
// by sigmoid plus bias, weigh by the chosen sigmoids normalized to unit sum and scaled.
[[kernel, max_total_threads_per_threadgroup(32)]] void
inkling_route(const device float *logits [[buffer(0)]], const device float *bias [[buffer(1)]],
              device uint *ids [[buffer(2)]], device float *probs [[buffer(3)]],
              constant RouterParams &p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
              uint lane [[thread_index_in_simdgroup]]) {
    const device float *row = logits + t * p.experts;
    float selection[4];
    bool invalid = false;
#pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint e = lane + 32 * j;
        bool valid = e < p.experts;
        float logit = valid ? row[e] : 0.0f;
        invalid = invalid || (as_type<uint>(logit) & 0x7f800000u) == 0x7f800000u;
        selection[j] = valid ? 1.0f / (1.0f + exp(-logit)) + bias[e] : -INFINITY;
    }
    invalid = simd_any(invalid);
    float weights[8];
    uint chosen[8];
    float total = 0;
    for (uint r = 0; r < p.active; r++) {
        float local = selection[0];
        uint which = 0;
#pragma unroll
        for (uint j = 1; j < 4; j++) {
            if (selection[j] > local) {
                local = selection[j];
                which = j;
            }
        }
        float peak = simd_max(local);
        uint id = simd_min(local == peak ? lane + 32 * which : 0xffffffffu);
        // Non-finite scores can match nothing; a placeholder keeps the reads in bounds, and the
        // invalid flag already poisons the weights.
        if (id >= p.experts) {
            id = r;
            invalid = true;
        }
        chosen[r] = id;
        if (lane == id % 32)
            selection[id / 32] = -INFINITY;
        float weight = 1.0f / (1.0f + exp(-row[id]));
        weights[r] = weight;
        total += weight;
    }
    // llama.cpp's expert_weights_norm_eps for Inkling: sum + 1e-20, not a clamp.
    float divisor = p.normalize ? total + 1e-20f : 1.0f;
    if (lane == 0) {
        for (uint r = 0; r < p.active; r++) {
            ids[t * p.active + r] = chosen[r];
            probs[t * p.active + r] = invalid ? as_type<float>(0x7fc00000u) : weights[r] / divisor * p.scale;
        }
    }
}

struct InklingLowRank {
    uint n;    // outputs per matrix
    uint rank; // 64
    uint m;    // rows
};

// Gain and forget of a loop controller in one dispatch: one thread per (row, output, matrix)
// dots the rank-64 vector with one FP32 weight row of 256 bytes.
[[kernel, max_total_threads_per_threadgroup(256)]] void
inkling_lowrank(const device bfloat *low [[buffer(0)]], const device float *gain_w [[buffer(1)]],
                const device float *forget_w [[buffer(2)]], device bfloat *gain [[buffer(3)]],
                device bfloat *forget [[buffer(4)]], constant InklingLowRank &p [[buffer(5)]],
                uint2 position [[thread_position_in_grid]]) {
    uint i = position.x, t = position.y;
    if (i >= 2 * p.n || t >= p.m)
        return;
    bool is_gain = i < p.n;
    uint j = is_gain ? i : i - p.n;
    const device float4 *w = reinterpret_cast<const device float4 *>((is_gain ? gain_w : forget_w) + j * p.rank);
    const device bfloat4 *g = reinterpret_cast<const device bfloat4 *>(low + t * p.rank);
    float4 acc = 0;
    for (uint k = 0; k < p.rank / 4; k++)
        acc = fma(w[k], float4(g[k]), acc);
    float v = (acc.x + acc.y) + (acc.z + acc.w);
    (is_gain ? gain : forget)[t * p.n + j] = bfloat(v);
}
