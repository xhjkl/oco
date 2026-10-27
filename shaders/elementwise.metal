struct Norm {
    uint n;
    float epsilon;
    uint full_layer;
};
kernel void norm(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                 device bfloat *y [[buffer(2)]], constant Norm &p [[buffer(3)]],
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32], inv;
    float s = 0;
    for (uint i = tid; tid < 256 && i < p.n; i += 256) {
        float v = float(x[row * p.n + i]);
        s += v * v;
    }
    s = simd_sum(s);
    if (lane == 0)
        shared[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(tid < 8 ? shared[tid] : 0);
    if (tid == 0)
        inv = rsqrt(s / float(p.n) + p.epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; tid < 256 && i < p.n; i += 256) {
        float normalized = float(x[row * p.n + i]) * inv;
        y[row * p.n + i] = p.full_layer ? bfloat(normalized * w[i])
                                        : bfloat(float(bfloat(normalized)) * float(bfloat(w[i])));
    }
}

// Residual add followed by the next normalization in one pass: the summed row is stored back
// to the residual stream and its normalized form written for the following transformation.
kernel void add_norm(device bfloat *x [[buffer(0)]], const device bfloat *y [[buffer(1)]],
                     const device float *w [[buffer(2)]], device bfloat *out [[buffer(3)]],
                     constant Norm &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32], inv;
    float s = 0;
    for (uint i = tid; tid < 256 && i < p.n; i += 256) {
        bfloat summed = bfloat(float(x[row * p.n + i]) + float(y[row * p.n + i]));
        x[row * p.n + i] = summed;
        float v = float(summed);
        s += v * v;
    }
    s = simd_sum(s);
    if (lane == 0)
        shared[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(tid < 8 ? shared[tid] : 0);
    if (tid == 0)
        inv = rsqrt(s / float(p.n) + p.epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; tid < 256 && i < p.n; i += 256) {
        float normalized = float(x[row * p.n + i]) * inv;
        out[row * p.n + i] = p.full_layer ? bfloat(normalized * w[i])
                                          : bfloat(float(bfloat(normalized)) * float(bfloat(w[i])));
    }
}

kernel void add(device bfloat *x [[buffer(0)]], const device bfloat *y [[buffer(1)]],
                constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n)
        x[i] = bfloat(float(x[i]) + float(y[i]));
}
struct Gate {
    uint n;
    uint clipped;
};
kernel void gate(const device bfloat *g [[buffer(0)]], const device bfloat *u [[buffer(1)]],
                 device bfloat *y [[buffer(2)]], constant Gate &p [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.n)
        return;
    float a = p.clipped ? reinterpret_cast<const device float *>(g)[i] : float(g[i]);
    float b = p.clipped ? reinterpret_cast<const device float *>(u)[i] : float(u[i]);
    // Rejecting invalid operands before min/clamp can turn NaN into a finite clipping bound.
    if ((as_type<uint>(a) & 0x7f800000u) == 0x7f800000u ||
        (as_type<uint>(b) & 0x7f800000u) == 0x7f800000u) {
        float invalid = as_type<float>(0x7fc00000u);
        if (p.clipped)
            reinterpret_cast<device float *>(y)[i] = invalid;
        else
            y[i] = bfloat(invalid);
        return;
    }
    if (p.clipped) {
        a = min(a, 7.0f);
        b = clamp(b, -6.0f, 8.0f);
    }
    float e = fast::exp(-abs(a) * (p.clipped ? 1.702f : 1.0f));
    // Dividing the negative tail directly; subtracting sigmoid(abs(a)) from one loses it.
    float sigmoid = (a < 0 ? e : 1.0f) / (1.0f + e);
    float activated = sigmoid * a;
    if (p.clipped)
        reinterpret_cast<device float *>(y)[i] = activated * b;
    else
        y[i] = bfloat(float(bfloat(activated)) * b);
}

// Mean-centered normalization with learned gain and offset (audio encoders and Whisper).
struct LayerNorm {
    uint n;
    float epsilon;
};
// Reduce one row's mean and inverse deviation across the 256-thread group.
inline float2 layer_norm_stats(const device bfloat *x, uint n, float epsilon, uint tid,
                               uint lane, uint sg, threadgroup float *shared) {
    float s = 0;
    for (uint i = tid; i < n; i += 256)
        s += float(x[i]);
    s = simd_sum(s);
    if (lane == 0)
        shared[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every SIMD group reduces the eight partial sums itself, so no second broadcast is needed.
    float mean = simd_sum(lane < 8 ? shared[lane] : 0) / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0;
    for (uint i = tid; i < n; i += 256) {
        float d = float(x[i]) - mean;
        v += d * d;
    }
    v = simd_sum(v);
    if (lane == 0)
        shared[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(simd_sum(lane < 8 ? shared[lane] : 0) / float(n) + epsilon);
    return float2(mean, inv);
}
kernel void layer_norm(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                       const device float *b [[buffer(2)]], device bfloat *y [[buffer(3)]],
                       constant LayerNorm &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[8];
    x += row * p.n;
    float2 stats = layer_norm_stats(x, p.n, p.epsilon, tid, lane, sg, shared);
    for (uint i = tid; i < p.n; i += 256)
        y[row * p.n + i] = bfloat((float(x[i]) - stats.x) * stats.y * w[i] + b[i]);
}

// Residual add stored back to the stream, then the next layer normalization in one pass.
kernel void add_layer_norm(device bfloat *x [[buffer(0)]], const device bfloat *y [[buffer(1)]],
                           const device float *w [[buffer(2)]], const device float *b [[buffer(3)]],
                           device bfloat *out [[buffer(4)]], constant LayerNorm &p [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[8];
    x += row * p.n;
    for (uint i = tid; i < p.n; i += 256)
        x[i] = bfloat(float(x[i]) + float(y[row * p.n + i]));
    threadgroup_barrier(mem_flags::mem_device);
    float2 stats = layer_norm_stats(x, p.n, p.epsilon, tid, lane, sg, shared);
    for (uint i = tid; i < p.n; i += 256)
        out[row * p.n + i] = bfloat((float(x[i]) - stats.x) * stats.y * w[i] + b[i]);
}

struct Gelu {
    uint n;
    // 1 evaluates the exact erf form; 0 the tanh approximation ggml uses for Whisper.
    uint erf_form;
};
// Abramowitz and Stegun 7.1.26, within 1.5e-7 of erf; Metal has no erf intrinsic.
inline float erf_approx(float x) {
    float a = abs(x), t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t +
                  0.254829592f) * t;
    return copysign(1.0f - poly * exp(-a * a), x);
}
kernel void gelu(device bfloat *x [[buffer(0)]], constant Gelu &p [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.n)
        return;
    float v = float(x[i]);
    // The tanh form as a logistic: 0.5 (1 + tanh(z)) = 1 / (1 + exp(-2z)), which stays finite
    // where a fast-math tanh of a large argument would divide infinity by infinity.
    float g = p.erf_form ? 0.5f * v * (1.0f + erf_approx(v * 0.70710678118654752f))
                         : v / (1.0f + exp(-1.5957691216057308f * (v + 0.044715f * v * v * v)));
    x[i] = bfloat(g);
}

// Add FP32 table rows to BF16 feature rows: positional embeddings indexed by sequence position,
// optionally repeating with a period (per-chunk positions in audio encoders).
struct AddRows {
    uint n, rows, offset, period;
};
kernel void add_rows(device bfloat *x [[buffer(0)]], const device float *table [[buffer(1)]],
                     constant AddRows &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.rows)
        return;
    uint row = (p.offset + i / p.n) % p.period;
    x[i] = bfloat(float(x[i]) + table[row * p.n + i % p.n]);
}

// Unfold a batch of channel-minor (B, H, W, C) images into convolution rows so the strided
// kernel becomes one dense product: row (b, oh, ow) holds taps (th, tw, c) zero-padded to k.
struct Im2col {
    uint c, h, w, oh, ow, kh, kw, stride, pad_h, pad_w, k, rows;
};
kernel void im2col(const device bfloat *x [[buffer(0)]], device bfloat *y [[buffer(1)]],
                   constant Im2col &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.k)
        return;
    uint row = i / p.k, j = i % p.k;
    uint taps = p.kh * p.kw;
    float value = 0;
    if (j < taps * p.c) {
        uint tap = j / p.c, c = j % p.c;
        uint b = row / (p.oh * p.ow), o = row % (p.oh * p.ow), oh = o / p.ow, ow = o % p.ow;
        int ih = int(oh * p.stride + tap / p.kw) - int(p.pad_h);
        int iw = int(ow * p.stride + tap % p.kw) - int(p.pad_w);
        if (ih >= 0 && iw >= 0 && uint(ih) < p.h && uint(iw) < p.w)
            value = float(x[((b * p.h + uint(ih)) * p.w + uint(iw)) * p.c + c]);
    }
    y[i] = bfloat(value);
}
