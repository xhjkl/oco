// Greedy selection over FP32 logits: finite values order as integers after folding the sign, and
// non-finite ones are classified from their raw bits, since fast math assumes finite floats.
inline void select_token(const device float *logits, device uint *output,
                             uint n, uint threads, uint tid, uint lane, uint sg,
                             threadgroup uint *vals, threadgroup uint *ids,
                             threadgroup uint *invalids) {
    uint val = 0;
    uint id = 0xffffffffu;
    uint invalid = 0;
    for (uint i = tid; i < n; i += threads) {
        uint bits = as_type<uint>(logits[i]);
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            invalid = 1;
            continue;
        }
        bits = (bits & 0x7fffffffu) == 0 ? 0u : bits;
        uint v = bits ^ ((bits & 0x80000000u) != 0 ? 0xffffffffu : 0x80000000u);
        if (v > val || (v == val && i < id)) {
            val = v;
            id = i;
        }
    }
    uint mx = simd_max(val);
    uint ix = simd_min(val == mx ? id : 0xffffffffu);
    invalid = simd_max(invalid);
    if (lane == 0) {
        vals[sg] = mx;
        ids[sg] = ix;
        invalids[sg] = invalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        mx = simd_max(tid < threads / 32 ? vals[tid] : 0);
        ix = simd_min(tid < threads / 32 && vals[tid] == mx ? ids[tid] : 0xffffffffu);
        invalid = simd_max(tid < threads / 32 ? invalids[tid] : 0);
        if (tid == 0) {
            output[0] = invalid != 0 ? 0xffffffffu : ix;
        }
    }
}
[[kernel, max_total_threads_per_threadgroup(1024)]] void argmax_f32(const device float *logits [[buffer(0)]], device uint *output [[buffer(1)]],
                   constant uint &n [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint vals[32], ids[32], invalids[32];
    select_token(logits, output, n, 1024, tid, lane, sg, vals, ids, invalids);
}
