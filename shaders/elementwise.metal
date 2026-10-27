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
