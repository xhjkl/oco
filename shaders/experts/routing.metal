struct RoutePlan {
    uint routes, up_tiles, down_tiles;
};
template <uint TileRows>
kernel void route_plan_tiled(const device uint *ids [[buffer(0)]], device uint *plan [[buffer(1)]],
                       constant RoutePlan &p [[buffer(2)]],
                       uint expert [[thread_index_in_threadgroup]]) {
    uint routes = p.routes, count = 0, grouped = 6 + 2 * ((routes + 15) / 16 + 32),
         base = grouped + expert * (routes + 1);
    for (uint i = 0; i < routes; i++)
        if (ids[i] == expert)
            plan[base + 1 + count++] = i;
    plan[base] = count;
    threadgroup_barrier(mem_flags::mem_device);
    if (expert == 0) {
        uint tiles = 0;
        for (uint e = 0; e < 32; e++)
            for (uint begin = 0; begin < plan[grouped + e * (routes + 1)]; begin += TileRows) {
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

template [[host_name("route_plan")]] kernel void route_plan_tiled<16>(
    const device uint *, device uint *, constant RoutePlan &, uint);
template [[host_name("route_plan_large")]] kernel void route_plan_tiled<32>(
    const device uint *, device uint *, constant RoutePlan &, uint);

// Keep top-four indices valid even when invalid arithmetic reaches the router.
// Bit tests survive fast-math compilation; NaN probabilities carry failure to final-logit validation.
inline void select_expert_values(float value, uint lane, thread uint *selected, thread float *probs) {
    bool finite = (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
    bool invalid = simd_any(!finite), available = true;
    float v = finite ? value : -INFINITY, chosen[4];
    for (uint r = 0; r < 4; r++) {
        float mx = simd_max(available ? v : -INFINITY);
        uint id = simd_min(available && v == mx ? lane : 0xffffffffu);
        selected[r] = id;
        chosen[r] = float(bfloat(mx));
        available = available && lane != id;
    }
    for (uint r = 0; r < 4; r++)
        invalid = invalid || (as_type<uint>(chosen[r]) & 0x7f800000u) == 0x7f800000u;
    if (invalid) {
        for (uint r = 0; r < 4; r++)
            probs[r] = as_type<float>(0x7fc00000u);
        return;
    }
    float sum = 0;
    for (uint r = 0; r < 4; r++)
        sum += exp(chosen[r] - chosen[0]);
    for (uint r = 0; r < 4; r++)
        probs[r] = float(bfloat(exp(chosen[r] - chosen[0]) / sum));
}

kernel void router(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
                   const device float *bias [[buffer(2)]], device uint *ids [[buffer(3)]],
                   device float *probs [[buffer(4)]], constant uint &n [[buffer(5)]],
                   uint t [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float logits[32];
    for (uint row = sg; row < 32; row += 8) {
        float4 accum = 0;
        for (uint k = lane * 4; k < n; k += 128) {
            float4 a = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(x + t * n + k));
            float4 b = float4(bfloat(w[row * n + k]), bfloat(w[row * n + k + 1]),
                              bfloat(w[row * n + k + 2]), bfloat(w[row * n + k + 3]));
            accum = fma(a, b, accum);
        }
        float sum = simd_sum((accum.x + accum.y) + (accum.z + accum.w));
        if (lane == 0)
            logits[row] = sum + float(bfloat(bias[row]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        uint selected[4];
        float weights[4];
        select_expert_values(logits[tid], tid, selected, weights);
        if (tid == 0) {
            for (uint r = 0; r < 4; r++) {
                ids[t * 4 + r] = selected[r];
                probs[t * 4 + r] = weights[r];
            }
        }
    }
}

// Assign one SIMD group to each logit while retaining the batched router's FMA tree.
[[kernel, max_total_threads_per_threadgroup(32)]] void
router_logits(const device bfloat *x [[buffer(0)]], const device float *w [[buffer(1)]],
              const device float *bias [[buffer(2)]], device float *logits [[buffer(3)]],
              constant uint &n [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
              uint lane [[thread_index_in_simdgroup]]) {
    float4 accum = 0;
    for (uint k = lane * 4; k < n; k += 128) {
        float4 a = float4(*reinterpret_cast<const device vec<bfloat, 4> *>(x + k));
        float4 b = float4(bfloat(w[row * n + k]), bfloat(w[row * n + k + 1]),
                          bfloat(w[row * n + k + 2]), bfloat(w[row * n + k + 3]));
        accum = fma(a, b, accum);
    }
    float sum = simd_sum((accum.x + accum.y) + (accum.z + accum.w));
    if (lane == 0)
        logits[row] = sum + float(bfloat(bias[row]));
}

// Top-four selection over 32 logits by one SIMD group, matching the batched router's rounding.
__attribute__((always_inline)) inline void
select_experts(const device float *logits, uint lane, thread uint *selected, thread float *probs) {
    select_expert_values(logits[lane], lane, selected, probs);
}

struct Moe {
    uint hidden, batch;
};
kernel void moe_sum(const device bfloat *routes [[buffer(0)]],
                    const device float *weights [[buffer(1)]], device bfloat *out [[buffer(2)]],
                    constant Moe &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.hidden * p.batch)
        return;
    uint t = i / p.hidden, d = i % p.hidden;
    float s = 0;
    for (uint r = 0; r < 4; r++)
        s += float(routes[(t * 4 + r) * p.hidden + d]) * weights[t * 4 + r];
    // Rounding the expert sum before the residual, matching the two-pass BF16 boundary.
    out[i] = bfloat(float(out[i]) + float(bfloat(s)));
}
