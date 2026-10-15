struct Gdn {
    uint kh, vh, vd, m, conv_dim;
    float epsilon;
};
inline void conv_rows(const device bfloat *x, const device float *w,
                      device float *state, device bfloat *out, Gdn p, uint c) {
    if (c >= p.conv_dim)
        return;
    float a = state[c * 3], b = state[c * 3 + 1], d = state[c * 3 + 2];
    for (uint t = 0; t < p.m; t++) {
        float v = float(x[t * p.conv_dim + c]);
        float s = w[c * 4] * a + w[c * 4 + 1] * b + w[c * 4 + 2] * d + w[c * 4 + 3] * v;
        out[t * p.conv_dim + c] = bfloat(s / (1 + fast::exp(-s)));
        a = b;
        b = d;
        d = v;
    }
    state[c * 3] = a;
    state[c * 3 + 1] = b;
    state[c * 3 + 2] = d;
}
kernel void conv(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                 device float *state [[buffer(2)]], device bfloat *out [[buffer(3)]],
                 constant Gdn &p [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    conv_rows(x, w, state, out, p, c);
}

// Each head owns its recurrent state across the batch.
inline void delta_rows(const device bfloat *qkv, const device bfloat *z,
                       const device bfloat *alpha, const device bfloat *beta,
                       const device float *a, const device float *dt,
                       const device float *norm, device float *state, device bfloat *out,
                       Gdn p, uint hv, uint tid, uint lane, uint sg, uint groups,
                       threadgroup float *shared, threadgroup float *scratch,
                       threadgroup float &inv) {
    // Following GGUF's tiled value-head order.
    uint hk = hv % p.kh, keydim = p.kh * 128, vdim = p.vh * p.vd;
    for (uint t = 0; t < p.m; t++) {
        float q[4], k[4], qs = 0, ks = 0;
        for (uint j = 0; j < 4; j++) {
            uint d = lane + 32 * j;
            q[j] = float(qkv[t * p.conv_dim + hk * 128 + d]);
            k[j] = float(qkv[t * p.conv_dim + keydim + hk * 128 + d]);
            qs += q[j] * q[j];
            ks += k[j] * k[j];
        }
        float qi = rsqrt(simd_sum(qs) + 1e-6f) * rsqrt(128.0f), ki = rsqrt(simd_sum(ks) + 1e-6f),
              kq = 0;
        for (uint j = 0; j < 4; j++) {
            q[j] *= qi;
            k[j] *= ki;
            kq += q[j] * k[j];
        }
        kq = simd_sum(kq);
        float br = float(beta[t * p.vh + hv]);
        float be = 1 / (1 + fast::exp(-br));
        float ar = float(alpha[t * p.vh + hv]) + dt[hv];
        float e = fast::exp(-abs(ar)), tail;
        if (e < 0.0625f) {
            // Expanding log1p through atanh; adding a tiny e to one would erase the decay.
            float u = e / (2.0f + e), u2 = u * u;
            tail = 2.0f * u * fma(u2, fma(u2, 1.0f / 5.0f, 1.0f / 3.0f), 1.0f);
        } else {
            tail = log(1.0f + e);
        }
        float sp = max(ar, 0.0f) + tail;
        float decay = fast::exp(a[hv] * sp);
        for (uint dv = sg; dv < p.vd; dv += groups) {
            uint row = (hv * p.vd + dv) * 128;
            float s[4], sq = 0, sk = 0;
            for (uint j = 0; j < 4; j++) {
                s[j] = state[row + lane + 32 * j];
                sq += s[j] * q[j];
                sk += s[j] * k[j];
            }
            sq = simd_sum(sq);
            sk = simd_sum(sk);
            float v = float(qkv[t * p.conv_dim + 2 * keydim + hv * p.vd + dv]);
            float change = be * (v - decay * sk);
            for (uint j = 0; j < 4; j++)
                state[row + lane + 32 * j] = decay * s[j] + k[j] * change;
            if (lane == 0)
                shared[dv] = decay * sq + change * kq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sq = tid < p.vd ? shared[tid] * shared[tid] : 0;
        sq = simd_sum(sq);
        if (lane == 0)
            scratch[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sq = simd_sum(tid < groups ? scratch[tid] : 0);
        if (tid == 0)
            inv = rsqrt(sq / float(p.vd) + p.epsilon);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < p.vd) {
            float zz = float(z[t * vdim + hv * p.vd + tid]);
            out[t * vdim + hv * p.vd + tid] =
                bfloat(shared[tid] * inv * norm[tid] * zz / (1 + fast::exp(-zz)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
[[kernel, max_total_threads_per_threadgroup(512)]] void delta(const device bfloat *qkv [[buffer(0)]], const device bfloat *z [[buffer(1)]],
                  const device bfloat *alpha [[buffer(2)]], const device bfloat *beta [[buffer(3)]],
                  const device float *a [[buffer(4)]], const device float *dt [[buffer(5)]],
                  const device float *norm [[buffer(6)]], device float *state [[buffer(7)]],
                  device bfloat *out [[buffer(8)]], constant Gdn &p [[buffer(9)]],
                  uint hv [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[128], scratch[16], inv;
    delta_rows(qkv, z, alpha, beta, a, dt, norm, state, out, p,
               hv, tid, lane, sg, 16, shared, scratch, inv);
}

#include "recurrent_prefill.metal"
