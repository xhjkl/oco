constant uint row_tiles [[function_constant(1)]];

#include "tile.metal"
#include "q4_tile.metal"

// Dispatching by the row-tile specialization: 0 selects wide tiles, 1..4 narrow tiles.
// Threadgroup arrays live in the kernel body; the unused variant's arrays are discarded.
#define MATMUL_DISPATCH(Q4, SCALED, SCALES)                                                                      \
    if (row_tiles == 0) {                                                                                        \
        threadgroup half wt[64 * 40];                                                                            \
        threadgroup half xs[64 * 40];                                                                            \
        matmul_tile_wide<Q4, SCALED>(w, x, y, bias, p, wt, xs, SCALES, group, tid, lane, sg);                    \
        return;                                                                                                  \
    }                                                                                                            \
    threadgroup half wt[64 * 40];                                                                                \
    threadgroup half xs[32 * 40];                                                                                \
    switch (row_tiles) {                                                                                         \
        case 1: matmul_tile_rows<Q4, SCALED, 1>(w, x, y, bias, p, wt, xs, SCALES, group, tid, lane, sg); break;  \
        case 2: matmul_tile_rows<Q4, SCALED, 2>(w, x, y, bias, p, wt, xs, SCALES, group, tid, lane, sg); break;  \
        case 3: matmul_tile_rows<Q4, SCALED, 3>(w, x, y, bias, p, wt, xs, SCALES, group, tid, lane, sg); break;  \
        default: matmul_tile_rows<Q4, SCALED, 4>(w, x, y, bias, p, wt, xs, SCALES, group, tid, lane, sg); break; \
    }

[[kernel, max_total_threads_per_threadgroup(128)]] void
matmul_q4(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
          device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
          const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
          uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
          ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    MATMUL_DISPATCH(true, false, reinterpret_cast<const device float2 *>(ids))
}

[[kernel, max_total_threads_per_threadgroup(128)]] void
matmul(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
       device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
       const device uint *ids [[buffer(4)]], constant Matrix &p [[buffer(5)]],
       uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
       ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    MATMUL_DISPATCH(false, false, reinterpret_cast<const device float2 *>(ids))
}

// Compute each activation row's gain once for all following output-column tiles.
// Exactly 256 threads (eight 32-lane SIMD groups) cooperate on one row; the caller orders
// this write before matmul_q4_half and keeps gain/input row bases aligned.
[[kernel, max_total_threads_per_threadgroup(256)]] void
q4_input_scales(const device bfloat *x [[buffer(0)]], device float2 *scales [[buffer(1)]],
                constant Matrix &p [[buffer(2)]], uint3 group [[threadgroup_position_in_grid]],
                ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group.x;
    if (row >= p.m)
        return;
    float peak = 0.0f;
    for (uint column = tid; column < p.k; column += 256)
        peak = max(peak, abs(float(x[row * p.k + column])));
    peak = simd_max(peak);
    threadgroup float peaks[8];
    if (lane == 0)
        peaks[sg] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0)
        return;
    peak = simd_max(lane < 8 ? peaks[lane] : 0.0f);
    if (lane == 0)
        scales[row] = q4_input_scale_for_peak(peak);
}

// Consume the group-shared gain rows with full-half operands and FP32 accumulators.
[[kernel, max_total_threads_per_threadgroup(128)]] void
matmul_q4_half(const device uchar *w [[buffer(0)]], const device bfloat *x [[buffer(1)]],
               device bfloat *y [[buffer(2)]], const device float *bias [[buffer(3)]],
               const device float2 *scales [[buffer(4)]], constant Matrix &p [[buffer(5)]],
               uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
               ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    MATMUL_DISPATCH(true, true, scales)
}
