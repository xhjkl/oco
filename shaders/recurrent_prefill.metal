// Keep state rows in registers across the prompt, with FP32 recurrence and BF16 output.
kernel void delta_prepare(const device bfloat *qkv [[buffer(0)]],
                          const device bfloat *alpha [[buffer(1)]],
                          const device bfloat *beta [[buffer(2)]],
                          const device float *a [[buffer(3)]],
                          const device float *dt [[buffer(4)]],
                          device float *qk [[buffer(5)]], device float *metadata [[buffer(6)]],
                          constant Gdn &p [[buffer(7)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint hk = group.x, t = group.y, keydim = p.kh * 128;
    float q[4], k[4], qs = 0, ks = 0;
    for (uint j = 0; j < 4; ++j) {
        uint d = lane + 32 * j;
        q[j] = float(qkv[t * p.conv_dim + hk * 128 + d]);
        k[j] = float(qkv[t * p.conv_dim + keydim + hk * 128 + d]);
        qs += q[j] * q[j];
        ks += k[j] * k[j];
    }
    float qi = rsqrt(simd_sum(qs) + 1e-6f) * rsqrt(128.0f),
          ki = rsqrt(simd_sum(ks) + 1e-6f), kq = 0;
    for (uint j = 0; j < 4; ++j) {
        q[j] *= qi;
        k[j] *= ki;
        kq += q[j] * k[j];
        uint index = t * 2 * keydim + hk * 128 + lane + 32 * j;
        qk[index] = q[j];
        qk[index + keydim] = k[j];
    }
    kq = simd_sum(kq);
    if (lane != 0)
        return;
    uint base = t * (2 * p.vh + p.kh);
    metadata[base + 2 * p.vh + hk] = kq;
    for (uint hv = hk; hv < p.vh; hv += p.kh) {
        float br = float(beta[t * p.vh + hv]);
        float be = 1 / (1 + fast::exp(-br));
        float ar = float(alpha[t * p.vh + hv]) + dt[hv];
        float e = fast::exp(-abs(ar)), tail;
        if (e < 0.0625f) {
            float u = e / (2.0f + e), u2 = u * u;
            tail = 2.0f * u * fma(u2, fma(u2, 1.0f / 5.0f, 1.0f / 3.0f), 1.0f);
        } else {
            tail = log(1.0f + e);
        }
        float sp = max(ar, 0.0f) + tail;
        metadata[base + hv] = fast::exp(a[hv] * sp);
        metadata[base + p.vh + hv] = be;
    }
}

// Four SIMD groups own four value rows, retaining four state values per lane.
kernel void delta_persistent(const device float *qk [[buffer(0)]],
                             const device float *metadata [[buffer(1)]],
                             const device bfloat *qkv [[buffer(2)]],
                             device float *state [[buffer(3)]],
                             device float *mixed [[buffer(4)]],
                             constant Gdn &p [[buffer(5)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    uint hv = group.y, dv = group.x * 4 + sg;
    if (dv >= p.vd)
        return;
    uint hk = hv % p.kh, keydim = p.kh * 128, vdim = p.vh * p.vd;
    uint row = (hv * p.vd + dv) * 128;
    float s[4];
    for (uint j = 0; j < 4; ++j)
        s[j] = state[row + lane + 32 * j];
    for (uint t = 0; t < p.m; ++t) {
        float q[4], k[4], sq = 0, sk = 0;
        for (uint j = 0; j < 4; ++j) {
            uint index = t * 2 * keydim + hk * 128 + lane + 32 * j;
            q[j] = qk[index];
            k[j] = qk[index + keydim];
            sq += s[j] * q[j];
            sk += s[j] * k[j];
        }
        sq = simd_sum(sq);
        sk = simd_sum(sk);
        uint base = t * (2 * p.vh + p.kh);
        float decay = metadata[base + hv], be = metadata[base + p.vh + hv];
        float kq = metadata[base + 2 * p.vh + hk];
        float v = float(qkv[t * p.conv_dim + 2 * keydim + hv * p.vd + dv]);
        float change = be * (v - decay * sk);
        for (uint j = 0; j < 4; ++j)
            s[j] = decay * s[j] + k[j] * change;
        if (lane == 0)
            mixed[t * vdim + hv * p.vd + dv] = decay * sq + change * kq;
    }
    for (uint j = 0; j < 4; ++j)
        state[row + lane + 32 * j] = s[j];
}

kernel void delta_output(const device float *mixed [[buffer(0)]],
                         const device bfloat *z [[buffer(1)]],
                         const device float *norm [[buffer(2)]],
                         device bfloat *out [[buffer(3)]], constant Gdn &p [[buffer(4)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4], inv;
    uint hv = group.x, t = group.y, base = (t * p.vh + hv) * p.vd;
    float value = tid < p.vd ? mixed[base + tid] : 0;
    float sq = simd_sum(value * value);
    if (lane == 0)
        scratch[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq = simd_sum(tid < 4 ? scratch[tid] : 0);
    if (tid == 0)
        inv = rsqrt(sq / float(p.vd) + p.epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.vd) {
        float zz = float(z[base + tid]);
        out[base + tid] = bfloat(value * inv * norm[tid] * zz / (1 + fast::exp(-zz)));
    }
}
