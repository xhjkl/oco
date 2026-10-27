// Share each K/V fragment across two query tiles for measured full 512-row prefill geometries.
[[kernel, max_total_threads_per_threadgroup(128)]] void
attention_wide16(const device bfloat *q [[buffer(0)]], const device half *k [[buffer(1)]],
               const device half *v [[buffer(2)]], const device float *sinks [[buffer(3)]],
               device bfloat *out [[buffer(4)]], constant AttentionParams &p [[buffer(5)]],
               uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint QueryTiles = 2, Rows = 16;
    threadgroup half sq[Rows * 256];
    threadgroup float ss[Rows * 32];
    threadgroup half sp[Rows * 32];
    threadgroup float rowmax[Rows], rowsum[Rows], rowfactor[Rows];
    uint h = group.x, t0 = group.y * Rows, kv = h / (p.heads / p.kv_heads), D = p.dim;
    uint kstride = p.kv_heads * D;
    for (uint i = tid; i < Rows * D; i += 128) {
        uint r = i / D, d = i % D, t = t0 + r;
        float value = t < p.m ? float(q[(t * p.heads + h) * D + d]) * p.scale : 0.0f;
        sq[i] = half(value);
    }
    if (tid < Rows) {
        rowmax[tid] = p.sinks ? float(bfloat(sinks[h])) : -INFINITY;
        rowsum[tid] = p.sinks ? 1.0f : 0.0f;
    }
    simdgroup_float8x8 o[QueryTiles][8];
#pragma unroll
    for (uint qt = 0; qt < QueryTiles; qt++) {
#pragma unroll
        for (uint i = 0; i < 8; i++)
            o[qt][i] = simdgroup_float8x8(0);
    }
    uint tiles = D / 32, dbase = sg * (D / 4);
    uint quad = lane / 4, lr = (quad & 4) + (lane / 2) % 4, lc = (quad & 2) * 2 + (lane % 2) * 2;
    uint end = key_end(p, min(t0 + Rows - 1, p.m - 1));
    uint begin = p.window && p.pos + t0 + 1 > p.window ? p.pos + t0 + 1 - p.window : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb = begin; kb < end; kb += 32) {
        simdgroup_float8x8 scores[QueryTiles];
#pragma unroll
        for (uint qt = 0; qt < QueryTiles; qt++)
            scores[qt] = simdgroup_float8x8(0);
        uint key0 = kb + 8 * sg;
        if (key0 < end) {
            const device half *kp = k + (key0 * p.kv_heads + kv) * D;
            for (uint d8 = 0; d8 < D; d8 += 8) {
                simdgroup_half8x8 mk;
                if (key0 + 8 <= end) {
                    simdgroup_load(mk, kp + d8, kstride, ulong2(0, 0), true);
                } else {
                    // Zeroing partial fragments before arithmetic; masked NaNs must not propagate.
                    for (uint j = 0; j < 2; j++) {
                        uint key = key0 + lc + j;
                        mk.thread_elements()[j] = key < end
                            ? k[(key * p.kv_heads + kv) * D + d8 + lr] : half(0);
                    }
                }
#pragma unroll
                for (uint qt = 0; qt < QueryTiles; qt++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, sq + qt * 8 * D + d8, D);
                    simdgroup_multiply_accumulate(scores[qt], mq, mk, scores[qt]);
                }
            }
        }
#pragma unroll
        for (uint qt = 0; qt < QueryTiles; qt++)
            simdgroup_store(scores[qt], ss + qt * 8 * 32 + 8 * sg, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < Rows) {
            uint r = lane, t = t0 + r, qpos = p.pos + t;
            uint rbegin = p.window && qpos + 1 > p.window ? qpos + 1 - p.window : 0;
            float previous = rowmax[r], peak = previous;
            float values[32];
            for (uint c = 0; c < 32; c++) {
                uint key = kb + c;
                bool valid = t < p.m && key < key_end(p, t) && key >= rbegin;
                values[c] = valid ? ss[r * 32 + c] : -INFINITY;
                peak = max(peak, values[c]);
            }
            bool empty = peak == -INFINITY;
            float factor = empty ? 1.0f : fast::exp(previous - peak);
            float mass = 0;
            for (uint c = 0; c < 32; c++) {
                float e = empty ? 0.0f : fast::exp(values[c] - peak);
                sp[r * 32 + c] = half(e);
                mass += e;
            }
            rowsum[r] = rowsum[r] * factor + mass;
            rowmax[r] = peak;
            rowfactor[r] = factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint qt = 0; qt < QueryTiles; qt++) {
            float factor = rowfactor[qt * 8 + lr];
            for (uint i = 0; i < tiles; i++) {
                o[qt][i].thread_elements()[0] *= factor;
                o[qt][i].thread_elements()[1] *= factor;
            }
        }
        for (uint kk = 0; kk < 32 && kb + kk < end; kk += 8) {
            simdgroup_half8x8 mp[QueryTiles];
#pragma unroll
            for (uint qt = 0; qt < QueryTiles; qt++)
                simdgroup_load(mp[qt], sp + qt * 8 * 32 + kk, 32);
            const device half *vp = v + ((kb + kk) * p.kv_heads + kv) * D + dbase;
            for (uint i = 0; i < tiles; i++) {
                simdgroup_half8x8 mv;
                if (kb + kk + 8 <= end) {
                    simdgroup_load(mv, vp + i * 8, kstride);
                } else {
                    for (uint j = 0; j < 2; j++) {
                        uint key = kb + kk + lr;
                        mv.thread_elements()[j] = key < end
                            ? v[(key * p.kv_heads + kv) * D + dbase + i * 8 + lc + j] : half(0);
                    }
                }
#pragma unroll
                for (uint qt = 0; qt < QueryTiles; qt++)
                    simdgroup_multiply_accumulate(o[qt][i], mp[qt], mv, o[qt][i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint qt = 0; qt < QueryTiles; qt++) {
        uint t = t0 + qt * 8 + lr;
        if (t >= p.m)
            continue;
        float inv = 1.0f / rowsum[qt * 8 + lr];
        for (uint i = 0; i < tiles; i++) {
#pragma unroll
            for (uint j = 0; j < 2; j++)
                out[(t * p.heads + h) * D + dbase + i * 8 + lc + j] =
                    bfloat(o[qt][i].thread_elements()[j] * inv);
        }
    }
}
