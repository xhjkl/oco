// Nemotron-H blocks: Mamba-2 recurrence over grouped state, and sigmoid-routed relu² experts.
struct Ssm {
    uint channels, m, stride, offset, heads, head_dim, groups, state, dt_offset;
    float epsilon;
};

// Depthwise causal convolution over the x/B/C columns of the input projection, with bias and
// SiLU. Each channel carries its three most recent inputs across calls.
kernel void ssm_conv(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                     const device float *bias [[buffer(2)]], device float *state [[buffer(3)]],
                     device float *out [[buffer(4)]], constant Ssm &p [[buffer(5)]],
                     uint c [[thread_position_in_grid]]) {
    if (c >= p.channels)
        return;
    float a = state[c * 3], b = state[c * 3 + 1], d = state[c * 3 + 2];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], bc = bias[c];
    for (uint t = 0; t < p.m; t++) {
        float v = float(x[t * p.stride + p.offset + c]);
        float s = fma(w0, a, fma(w1, b, fma(w2, d, fma(w3, v, bc))));
        out[t * p.channels + c] = s / (1 + fast::exp(-s));
        a = b;
        b = d;
        d = v;
    }
    state[c * 3] = a;
    state[c * 3 + 1] = b;
    state[c * 3 + 2] = d;
}

// Selective scan: one SIMD group owns one (head, channel) state row of 128 values, four per
// lane, and walks the chunk in token order. The FP32 time steps pass through softplus after
// their bias; A is stored negated in the checkpoint so the decay is exp(dt * A).
[[kernel, max_total_threads_per_threadgroup(128)]] void
mamba_scan(const device float *xbc [[buffer(0)]], const device float *dt_in [[buffer(1)]],
           const device float *a [[buffer(2)]], const device float *d [[buffer(3)]],
           const device float *dt_bias [[buffer(4)]], device float *state [[buffer(5)]],
           device float *y [[buffer(6)]], constant Ssm &p [[buffer(7)]],
           uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
           uint sg [[simdgroup_index_in_threadgroup]]) {
    uint h = group.y, ch = group.x * 4 + sg;
    if (ch >= p.head_dim)
        return;
    uint g = h / (p.heads / p.groups), inner = p.heads * p.head_dim;
    uint row = (h * p.head_dim + ch) * p.state;
    float s[4];
#pragma unroll
    for (uint j = 0; j < 4; j++)
        s[j] = state[row + lane + 32 * j];
    float ah = a[h], dh = d[h], bias = dt_bias[h];
    const device float *bcol = xbc + inner + g * p.state;
    const device float *ccol = bcol + p.groups * p.state;
    for (uint t = 0; t < p.m; t++) {
        float dt = dt_in[t * p.heads + h] + bias;
        dt = dt <= 20.0f ? log(1.0f + exp(dt)) : dt;
        float decay = exp(dt * ah);
        float xv = xbc[t * p.channels + h * p.head_dim + ch];
        float xdt = xv * dt, acc = 0;
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint n = t * p.channels + lane + 32 * j;
            s[j] = fma(s[j], decay, bcol[n] * xdt);
            acc = fma(s[j], ccol[n], acc);
        }
        acc = simd_sum(acc);
        if (lane == 0)
            y[t * inner + h * p.head_dim + ch] = fma(dh, xv, acc);
    }
#pragma unroll
    for (uint j = 0; j < 4; j++)
        state[row + lane + 32 * j] = s[j];
}

// Gate the scan output with SiLU(z), then RMS-normalize each group of channels in FP32.
[[kernel, max_total_threads_per_threadgroup(256)]] void
mamba_gate_norm(const device float *y [[buffer(0)]], const device bfloat *zxbcdt [[buffer(1)]],
                const device float *w [[buffer(2)]], device bfloat *out [[buffer(3)]],
                constant Ssm &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[8], inv;
    uint inner = p.heads * p.head_dim, size = inner / p.groups;
    uint t = group.y, base = group.x * size;
    float v[2] = {0, 0}, sq = 0;
#pragma unroll
    for (uint k = 0; k < 2; k++) {
        uint i = tid + 256 * k;
        if (i < size) {
            float z = float(zxbcdt[t * p.stride + base + i]);
            v[k] = y[t * inner + base + i] * z / (1 + fast::exp(-z));
            sq = fma(v[k], v[k], sq);
        }
    }
    sq = simd_sum(sq);
    if (lane == 0)
        shared[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq = simd_sum(tid < 8 ? shared[tid] : 0);
    if (tid == 0)
        inv = rsqrt(sq / float(size) + p.epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint k = 0; k < 2; k++) {
        uint i = tid + 256 * k;
        if (i < size)
            out[t * inner + base + i] = bfloat(v[k] * inv * w[base + i]);
    }
}

struct RouterParams {
    uint n, experts, active, normalize;
    float scale;
};

// Sigmoid routing over up to 128 experts: the selection adds a per-expert bias, the
// combination weights are the unbiased sigmoids of the chosen experts, optionally normalized
// to unit sum and scaled. Eight SIMD groups share the logits; the first selects.
[[kernel, max_total_threads_per_threadgroup(256)]] void
router_sigmoid(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
               const device float *bias [[buffer(2)]], device uint *ids [[buffer(3)]],
               device float *probs [[buffer(4)]], constant RouterParams &p [[buffer(5)]],
               uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float logits[128];
    for (uint row = sg; row < p.experts; row += 8) {
        float4 accum = 0;
        for (uint k = lane * 4; k < p.n; k += 128) {
            float4 a = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(x + t * p.n + k));
            float4 b = *reinterpret_cast<const device float4 *>(w + row * p.n + k);
            accum = fma(a, b, accum);
        }
        float sum = simd_sum((accum.x + accum.y) + (accum.z + accum.w));
        if (lane == 0)
            logits[row] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0)
        return;
    float score[4], selection[4];
    bool invalid = false;
#pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint e = lane + 32 * j;
        bool valid = e < p.experts;
        float logit = valid ? logits[e] : 0.0f;
        invalid = invalid || (as_type<uint>(logit) & 0x7f800000u) == 0x7f800000u;
        score[j] = 1.0f / (1.0f + exp(-logit));
        selection[j] = valid ? score[j] + bias[e] : -INFINITY;
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
        chosen[r] = id;
        if (lane == id % 32)
            selection[id / 32] = -INFINITY;
        float weight = 1.0f / (1.0f + exp(-logits[id]));
        weights[r] = weight;
        total += weight;
    }
    float divisor = p.normalize ? max(total, 6.103515625e-5f) : 1.0f;
    if (lane == 0) {
        for (uint r = 0; r < p.active; r++) {
            ids[t * p.active + r] = chosen[r];
            // Carrying invalid arithmetic to the final-logit check rather than clamping it away.
            probs[t * p.active + r] = invalid ? as_type<float>(0x7fc00000u)
                                              : weights[r] / divisor * p.scale;
        }
    }
}

kernel void relu_sqr(device bfloat *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float v = max(float(x[i]), 0.0f);
        x[i] = bfloat(v * v);
    }
}

struct MoeAdd {
    uint hidden, batch, routes, shared;
};
// Add each token's weighted expert outputs, and optionally its shared expert, to its residual.
kernel void moe_add(const device bfloat *routes [[buffer(0)]], const device float *weights [[buffer(1)]],
                    device bfloat *out [[buffer(2)]], const device bfloat *shared [[buffer(3)]],
                    constant MoeAdd &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.hidden * p.batch)
        return;
    uint t = i / p.hidden, d = i % p.hidden;
    float s = p.shared ? float(shared[i]) : 0.0f;
    for (uint r = 0; r < p.routes; r++)
        s = fma(float(routes[(t * p.routes + r) * p.hidden + d]), weights[t * p.routes + r], s);
    out[i] = bfloat(float(out[i]) + s);
}

// Routed tile plan for gathered expert products: per-expert route lists and one (expert, first
// route) pair per tile, plus the two indirect grids that consume them. One thread per expert.
struct RoutePlanN {
    uint routes, up_tiles, down_tiles, experts, tile_rows;
};
inline uint plan_tiles(uint routes, uint tile_rows, uint experts) {
    return (routes + tile_rows - 1) / tile_rows + experts;
}
[[kernel, max_total_threads_per_threadgroup(128)]] void
route_plan_n(const device uint *ids [[buffer(0)]], device uint *plan [[buffer(1)]],
             constant RoutePlanN &p [[buffer(2)]], uint expert [[thread_index_in_threadgroup]]) {
    uint routes = p.routes, lists = 6 + 2 * plan_tiles(routes, p.tile_rows, p.experts);
    uint base = lists + expert * (routes + 1), count = 0;
    if (expert < p.experts) {
        for (uint i = 0; i < routes; i++)
            if (ids[i] == expert)
                plan[base + 1 + count++] = i;
        plan[base] = count;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (expert == 0) {
        uint tiles = 0;
        for (uint e = 0; e < p.experts; e++)
            for (uint begin = 0; begin < plan[lists + e * (routes + 1)]; begin += p.tile_rows) {
                plan[6 + tiles * 2] = e;
                plan[6 + tiles * 2 + 1] = begin;
                tiles++;
            }
        plan[0] = p.up_tiles;
        plan[1] = tiles;
        plan[2] = 1;
        plan[3] = p.down_tiles;
        plan[4] = tiles;
        plan[5] = 1;
    }
}

// Gathered Q4_0 expert tiles for sparse routing: BM routes × 64 output rows per threadgroup,
// K steps of 32. Weights decode to half and are scattered into [k-block][row-block][k][row]
// so SIMD-group loads read contiguous, untransposed 8×8 blocks; activations stage likewise as
// [k-block][route-block][route][k]. Four SIMD groups split routes RS ways and columns 4/RS ways.
template <uint BM, uint RS, bool TokenInput>
[[kernel, max_total_threads_per_threadgroup(128)]] void
expert_q4_blocked(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                  device bfloat *y [[buffer(2)]], const device uint *plan [[buffer(3)]],
                  constant Matrix &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BN = 64, BK = 32, KB = BK / 8, CS = 4 / RS, COLS = BN / CS, CT = COLS / 8;
    constexpr uint ROWS = BM / RS, R = ROWS / 8, RB = BM / 8;
    threadgroup half sa[BN * BK];
    threadgroup half sb[BM * BK];
    uint expert = plan[6 + group.y * 2], tile_row = plan[6 + group.y * 2 + 1];
    uint lists = 6 + 2 * plan_tiles(p.m, BM, p.experts);
    uint base = lists + expert * (p.m + 1), count = plan[base];
    if (tile_row >= count)
        return;
    uint n0 = group.x * BN, blocks = p.k / 32;
    uint quad = lane / 4, lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    // Weights: two threads per row, one nibble half (sixteen k values) each.
    uint wn = tid / 2, wb = tid % 2;
    const device uchar *row_blocks = w + (expert * p.n + min(n0 + wn, p.n - 1)) * blocks * 18;
    // Activations: each work item stages eight k values of one route.
    constexpr uint ITEMS = BM * (BK / 8), PER_THREAD = (ITEMS + 127) / 128;
    uint source[PER_THREAD];
#pragma unroll
    for (uint i = 0; i < PER_THREAD; ++i) {
        uint item = tid + 128 * i, row = tile_row + item / KB;
        uint route = row < count ? plan[base + 1 + row] : 0;
        source[i] = TokenInput ? route / p.routes : route;
    }
    simdgroup_half8x8 a[CT], b[R];
    simdgroup_float8x8 accum[R][CT];
#pragma unroll
    for (uint r = 0; r < R; ++r)
#pragma unroll
        for (uint c = 0; c < CT; ++c)
            accum[r][c] = simdgroup_float8x8(0);
    for (uint k0 = 0; k0 < p.k; k0 += BK) {
        half4x4 decoded = q4_block(row_blocks + (k0 / 32) * 18, wb);
        half4 staged[PER_THREAD][2];
#pragma unroll
        for (uint i = 0; i < PER_THREAD; ++i) {
            uint item = tid + 128 * i, xk = (item % KB) * 8;
            staged[i][0] = half4(0);
            staged[i][1] = half4(0);
            if (item < ITEMS && tile_row + item / KB < count) {
                const device bfloat4 *values =
                    reinterpret_cast<const device bfloat4 *>(x + source[i] * p.k + k0 + xk);
                float4 v0 = float4(values[0]), v1 = float4(values[1]);
                if (RELU_SQR_INPUT) {
                    v0 = max(v0, 0.0f);
                    v0 *= v0;
                    v1 = max(v1, 0.0f);
                    v1 *= v1;
                }
                staged[i][0] = half4(v0);
                staged[i][1] = half4(v1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A(row, k) lives at block (k/8, row/8), element (k%8, row%8): the right operand's layout.
#pragma unroll
        for (uint j = 0; j < 16; ++j) {
            uint k = wb * 16 + j;
            sa[64 * ((k / 8) * (BN / 8) + wn / 8) + 8 * (k % 8) + wn % 8] = decoded[j / 4][j % 4];
        }
#pragma unroll
        for (uint i = 0; i < PER_THREAD; ++i) {
            uint item = tid + 128 * i;
            if (item < ITEMS) {
                uint r = item / KB, kb = item % KB;
                threadgroup half4 *xd =
                    reinterpret_cast<threadgroup half4 *>(sb + 64 * (kb * RB + r / 8) + 8 * (r % 8));
                xd[0] = staged[i][0];
                xd[1] = staged[i][1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint col_block = (sg % CS) * (COLS / 8), row_block = (sg / CS) * (ROWS / 8);
#pragma unroll
        for (uint kb = 0; kb < KB; ++kb) {
#pragma unroll
            for (uint c = 0; c < CT; ++c)
                simdgroup_load(a[c], sa + 64 * (kb * (BN / 8) + col_block + c), 8);
#pragma unroll
            for (uint r = 0; r < R; ++r)
                simdgroup_load(b[r], sb + 64 * (kb * RB + row_block + r), 8);
#pragma unroll
            for (uint r = 0; r < R; ++r)
#pragma unroll
                for (uint c = 0; c < CT; ++c)
                    simdgroup_multiply_accumulate(accum[r][c], b[r], a[c], accum[r][c]);
        }
    }
    uint col0 = n0 + (sg % CS) * COLS + lc;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
        uint row = tile_row + (sg / CS) * ROWS + r * 8 + lr;
        if (row >= count)
            continue;
        uint out = plan[base + 1 + row];
#pragma unroll
        for (uint c = 0; c < CT; ++c) {
#pragma unroll
            for (uint j = 0; j < 2; ++j) {
                uint col = col0 + c * 8 + j;
                if (col < p.n)
                    y[out * p.n + col] = bfloat(accum[r][c].thread_elements()[j]);
            }
        }
    }
}

#define EXPERT_Q4_BLOCKED(NAME, BM, RS, TOKEN)                                                    \
    template [[host_name(NAME)]] kernel void expert_q4_blocked<BM, RS, TOKEN>(                    \
        const device uchar *, const device bfloat *, device bfloat *, const device uint *,        \
        constant Matrix &, uint3, uint, uint, uint);
EXPERT_Q4_BLOCKED("expert_q4_up_8", 8, 1, true)
EXPERT_Q4_BLOCKED("expert_q4_down_8", 8, 1, false)

// Gathered Q4_0 expert tiles for sparse routing: BM routes × 64 output rows per threadgroup,
// K steps of 32. Weights decode to half and are scattered into [k-block][row-block][k][row]
// so SIMD-group loads read contiguous, untransposed 8×8 blocks; activations stage likewise as
// [k-block][route-block][route][k]. Four SIMD groups split routes RS ways and columns 4/RS ways.
template <uint BM, uint RS, bool TokenInput>
[[kernel, max_total_threads_per_threadgroup(128)]] void
expert_q4_staged(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                  device bfloat *y [[buffer(2)]], const device uint *plan [[buffer(3)]],
                  constant Matrix &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BN = 64, BK = 32, KB = BK / 8, CS = 4 / RS, COLS = BN / CS, CT = COLS / 8;
    constexpr uint ROWS = BM / RS, R = ROWS / 8, RB = BM / 8;
    // Operands share one block with the FP32 output staging that follows the K loop.
    threadgroup float shared[(BM * BN > (BN * BK + BM * BK) / 2) ? BM * BN : (BN * BK + BM * BK) / 2];
    threadgroup half *sa = reinterpret_cast<threadgroup half *>(shared);
    threadgroup half *sb = sa + BN * BK;
    uint expert = plan[6 + group.y * 2], tile_row = plan[6 + group.y * 2 + 1];
    uint lists = 6 + 2 * plan_tiles(p.m, BM, p.experts);
    uint base = lists + expert * (p.m + 1), count = plan[base];
    if (tile_row >= count)
        return;
    uint n0 = group.x * BN, blocks = p.k / 32;
    uint quad = lane / 4, lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    // Weights: two threads per row, one nibble half (sixteen k values) each.
    uint wn = tid / 2, wb = tid % 2;
    const device uchar *row_blocks = w + (expert * p.n + min(n0 + wn, p.n - 1)) * blocks * 18;
    // Activations: each work item stages eight k values of one route.
    constexpr uint ITEMS = BM * (BK / 8), PER_THREAD = (ITEMS + 127) / 128;
    uint source[PER_THREAD];
#pragma unroll
    for (uint i = 0; i < PER_THREAD; ++i) {
        uint item = tid + 128 * i, row = tile_row + item / KB;
        uint route = row < count ? plan[base + 1 + row] : 0;
        source[i] = TokenInput ? route / p.routes : route;
    }
    simdgroup_half8x8 a[CT], b[R];
    simdgroup_float8x8 accum[R][CT];
#pragma unroll
    for (uint r = 0; r < R; ++r)
#pragma unroll
        for (uint c = 0; c < CT; ++c)
            accum[r][c] = simdgroup_float8x8(0);
    for (uint k0 = 0; k0 < p.k; k0 += BK) {
        half4x4 decoded = q4_block(row_blocks + (k0 / 32) * 18, wb);
        half4 staged[PER_THREAD][2];
#pragma unroll
        for (uint i = 0; i < PER_THREAD; ++i) {
            uint item = tid + 128 * i, xk = (item % KB) * 8;
            staged[i][0] = half4(0);
            staged[i][1] = half4(0);
            if (item < ITEMS && tile_row + item / KB < count) {
                const device bfloat4 *values =
                    reinterpret_cast<const device bfloat4 *>(x + source[i] * p.k + k0 + xk);
                float4 v0 = float4(values[0]), v1 = float4(values[1]);
                if (RELU_SQR_INPUT) {
                    v0 = max(v0, 0.0f);
                    v0 *= v0;
                    v1 = max(v1, 0.0f);
                    v1 *= v1;
                }
                staged[i][0] = half4(v0);
                staged[i][1] = half4(v1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A(row, k) lives at block (k/8, row/8), element (k%8, row%8): the right operand's layout.
#pragma unroll
        for (uint j = 0; j < 16; ++j) {
            uint k = wb * 16 + j;
            sa[64 * ((k / 8) * (BN / 8) + wn / 8) + 8 * (k % 8) + wn % 8] = decoded[j / 4][j % 4];
        }
#pragma unroll
        for (uint i = 0; i < PER_THREAD; ++i) {
            uint item = tid + 128 * i;
            if (item < ITEMS) {
                uint r = item / KB, kb = item % KB;
                threadgroup half4 *xd =
                    reinterpret_cast<threadgroup half4 *>(sb + 64 * (kb * RB + r / 8) + 8 * (r % 8));
                xd[0] = staged[i][0];
                xd[1] = staged[i][1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint col_block = (sg % CS) * (COLS / 8), row_block = (sg / CS) * (ROWS / 8);
#pragma unroll
        for (uint kb = 0; kb < KB; ++kb) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (uint c = 0; c < CT; ++c)
                simdgroup_load(a[c], sa + 64 * (kb * (BN / 8) + col_block + c), 8);
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (uint r = 0; r < R; ++r)
                simdgroup_load(b[r], sb + 64 * (kb * RB + row_block + r), 8);
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (uint r = 0; r < R; ++r)
#pragma unroll
                for (uint c = 0; c < CT; ++c)
                    simdgroup_multiply_accumulate(accum[r][c], b[r], a[c], accum[r][c]);
        }
    }
    // Stage the FP32 tile, then write whole rows so each SIMD group stores 128 contiguous bytes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint col_base = (sg % CS) * COLS, row_base = (sg / CS) * ROWS;
#pragma unroll
    for (uint r = 0; r < R; ++r)
#pragma unroll
        for (uint c = 0; c < CT; ++c)
            simdgroup_store(accum[r][c], shared + (row_base + r * 8) * BN + col_base + c * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nr = min(uint(BM), count - tile_row);
    for (uint j = sg; j < nr; j += 4) {
        uint out = plan[base + 1 + tile_row + j];
        uint col = n0 + lane * 2;
        if (col + 1 < p.n) {
            float2 v = *reinterpret_cast<threadgroup float2 *>(shared + j * BN + lane * 2);
            *reinterpret_cast<device bfloat2 *>(y + out * p.n + col) = bfloat2(v);
        } else if (col < p.n) {
            y[out * p.n + col] = bfloat(shared[j * BN + lane * 2]);
        }
    }
}

#define EXPERT_Q4_STAGED(NAME, BM, RS, TOKEN)                                                    \
    template [[host_name(NAME)]] kernel void expert_q4_staged<BM, RS, TOKEN>(                    \
        const device uchar *, const device bfloat *, device bfloat *, const device uint *,        \
        constant Matrix &, uint3, uint, uint, uint);
EXPERT_Q4_STAGED("expert_q4_up_8_s", 8, 1, true)
EXPERT_Q4_STAGED("expert_q4_up_16_s", 16, 2, true)
EXPERT_Q4_STAGED("expert_q4_up_32_s", 32, 2, true)
EXPERT_Q4_STAGED("expert_q4_down_8_s", 8, 1, false)
EXPERT_Q4_STAGED("expert_q4_down_16_s", 16, 2, false)
EXPERT_Q4_STAGED("expert_q4_down_32_s", 32, 2, false)

// Gathered Q4_0 expert tiles on Metal 4 tensor operations: 32 routes × 64 output rows per
// threadgroup with K steps of 32. Weights decode to half and stage [row][k]; gathered BF16
// activations stage [route][k], optionally squared through relu on load. Two 16×64 cooperative
// products per step accumulate in FP32; the upper half is skipped when the expert owns at most
// sixteen routes in the tile. Operands and the FP32 output staging share one 8 KiB block.
template <bool TokenInput>
[[kernel, max_total_threads_per_threadgroup(128)]] void
expert_q4_tensor(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
                  device bfloat *y [[buffer(2)]], const device uint *plan [[buffer(3)]],
                  constant Matrix &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NR1 = 32, NK = 32, NR1H = NR1 / 2;
    threadgroup float shared[NR1 * NR0];
    threadgroup half *sa = reinterpret_cast<threadgroup half *>(shared);
    threadgroup half *sb = sa + NR0 * NK;
    uint expert = plan[6 + group.y * 2], tile_row = plan[6 + group.y * 2 + 1];
    uint lists = 6 + 2 * plan_tiles(p.m, NR1, p.experts);
    uint base = lists + expert * (p.m + 1), count = plan[base];
    if (tile_row >= count)
        return;
    uint nr1 = min(uint(NR1), count - tile_row);
    bool has_hi = nr1 > NR1H;
    uint n0 = group.x * NR0, blocks = p.k / 32;
    uint wn = tid / 2, wb = tid % 2;
    const device uchar *row_blocks = w + (expert * p.n + min(n0 + wn, p.n - 1)) * blocks * 18;
    uint xr = tid / 4, xk = (tid % 4) * 8;
    uint route = plan[base + 1 + tile_row + min(xr, nr1 - 1)];
    uint source = TokenInput ? route / p.routes : route;
    const device bfloat4 *xrow = reinterpret_cast<const device bfloat4 *>(x + source * p.k + xk);
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb + NR1H * NK, dextents<int32_t, 2>(NK, NR1H));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto c0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto c1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
    for (uint k0 = 0; k0 < p.k; k0 += NK) {
        half4x4 decoded = q4_block(row_blocks + (k0 / 32) * 18, wb);
        float4 v0 = float4(xrow[k0 / 4]), v1 = float4(xrow[k0 / 4 + 1]);
        if (RELU_SQR_INPUT) {
            v0 = max(v0, 0.0f);
            v0 *= v0;
            v1 = max(v1, 0.0f);
            v1 *= v1;
        }
        // Rows past the expert's routes recompute a valid row; their outputs are never stored.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half4 *wd = reinterpret_cast<threadgroup half4 *>(sa + wn * NK + wb * 16);
#pragma unroll
        for (uint j = 0; j < 4; ++j)
            wd[j] = decoded[j];
        threadgroup half4 *xd = reinterpret_cast<threadgroup half4 *>(sb + xr * NK + xk);
        xd[0] = half4(v0);
        xd[1] = half4(v1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);
        mm.run(sB0, sA, c0);
        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);
            mm.run(sB1, sA, c1);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared, dextents<int32_t, 2>(NR0, NR1H));
    c0.store(tC0);
    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(shared + NR1H * NR0, dextents<int32_t, 2>(NR0, NR1H));
        c1.store(tC1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = sg; j < nr1; j += 4) {
        uint out = plan[base + 1 + tile_row + j];
        for (uint i = lane; i < NR0; i += 32) {
            uint col = n0 + i;
            if (col < p.n)
                y[out * p.n + col] = bfloat(shared[j * NR0 + i]);
        }
    }
}

template [[host_name("expert_q4_tensor_up")]] kernel void expert_q4_tensor<true>(
    const device uchar *, const device bfloat *, device bfloat *, const device uint *,
    constant Matrix &, uint3, uint, uint, uint);
template [[host_name("expert_q4_tensor_down")]] kernel void expert_q4_tensor<false>(
    const device uchar *, const device bfloat *, device bfloat *, const device uint *,
    constant Matrix &, uint3, uint, uint, uint);
