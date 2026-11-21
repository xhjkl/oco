//! Validate Inkling 2 loop weights and bind them to the concrete model.
use super::{Control, Inkling, InklingLayer, LoopLayer};
use crate::error::{Context, Error, Result, ensure};
use crate::{
    gguf::{self, Gguf},
    gpu::Gpu,
    tensor::{Tensor, WeightFormat},
    weights::{
        Architecture,
        common::{CommonWeights, Kernels},
        compute::{InklingPlan, Norm},
        load::{Geometry, Import, Parameters, common_selection, import},
        operands::{Projection, Router},
    },
};
use std::sync::Arc;

const PREFIX: &str = "inkling2_loop";

struct Selected<'a> {
    full: bool,
    /// Three Q/K/V input norms on full layers, one shared norm on SWA layers.
    norms: Vec<gguf::Tensor<'a>>,
    attn_query: Option<gguf::Tensor<'a>>,
    dense: [gguf::Tensor<'a>; 7],
    taps: Option<[gguf::Tensor<'a>; 2]>,
    floats: [gguf::Tensor<'a>; 9],
    router: [gguf::Tensor<'a>; 2],
    experts: [gguf::Tensor<'a>; 2],
}

struct SelectedLoop<'a> {
    messages: gguf::Tensor<'a>,
    controls: [[gguf::Tensor<'a>; 3]; 4],
}

/// Read a per-layer metadata value that must be uniform across layers (an array or a scalar).
fn uniform(file: &Gguf, key: &str, count: usize) -> Result<usize> {
    let key = format!("{PREFIX}.{key}");
    let values = match file.unsigned_integers(&key) {
        Ok(values) => values,
        Err(_) => vec![file.number(&key)? as u64; count],
    };
    ensure!(
        values.len() == count && values.iter().all(|&v| v == values[0]) && values[0] > 0,
        Error::Checkpoint("{key} must be one positive value for every layer")
    );
    Ok(values[0] as usize)
}

pub fn load(file: &Gguf, p: &Parameters<'_>) -> Result<Architecture> {
    let hidden = p.integer("embedding_length")?;
    let count = p.integer("block_count")?;
    let heads = uniform(file, "attention.head_count", count)?;
    let kv_heads = uniform(file, "attention.head_count_kv", count)?;
    let dim = p.integer("attention.key_length")?;
    let context = p.integer("context_length")?;
    let epsilon = p.number("attention.layer_norm_rms_epsilon")? as f32;
    let experts = p.integer("expert_count")?;
    let active = p.integer("expert_used_count")?;
    let expert_width = p.integer("expert_feed_forward_length")?;
    let unpadded = p.integer("vocab_size_unpadded")?;
    let window = p.integer("attention.sliding_window")?;
    let taps = p.integer("shortconv.l_cache")?;
    let block = p.integer("attn_res.block_size")?;
    let rel = p.integer("attention.relative_length")?;
    let extent = p.integer("attention.relative_extent")?;
    let clip = p.number("attention.qk_clip")? as f32;
    let rel_clip = p.number("attention.relative_clip")? as f32;
    let floor = p.integer("attention.log_scale_floor")?;
    let alpha = p.number("attention.log_scale_alpha")? as f32;
    let route_scale = p.optional("expert_weights_scale", 1.0)? as f32;
    let situ_beta = p.number("activation.situ_beta")? as f32;
    let situ_linear_beta = p.number("activation.situ_linear_beta")? as f32;
    let core = p.integer("loop.start")?..p.integer("loop.end")?;
    let passes = p.integer("loop.count")?;
    let rank = p.integer("loop.gate_rank")?;
    let varying = format!("{PREFIX}.loop.varying");
    ensure!(
        !file.meta.contains_key(&varying) || !file.boolean(&varying)?,
        Error::Checkpoint("variable-pass Inkling loops are unsupported")
    );
    let pattern_key = format!("{PREFIX}.attention.sliding_window_pattern");
    let pattern = file.booleans(&pattern_key)?;
    ensure!(
        pattern.len() == count,
        Error::Checkpoint("{pattern_key} must cover every layer")
    );
    ensure!(
        dim == 128
            && heads.is_multiple_of(kv_heads)
            && hidden.is_multiple_of(256)
            && heads * dim <= 8192
            && rel == 16
            && taps == 4
            && rank == 64
            && experts <= 128
            && active <= 8
            && active <= experts
            && expert_width.is_multiple_of(64)
            && block > 0
            && core.start > 0
            && core.start < core.end
            && core.end <= count
            && core.start.is_multiple_of(block)
            && core.end.is_multiple_of(block)
            && core.len() == 2 * block
            && count - core.end == block
            && core.start == block
            && passes >= 2
            && unpadded > 0
            && [
                epsilon,
                clip,
                rel_clip,
                situ_beta,
                situ_linear_beta,
                route_scale
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
            && alpha.is_finite()
            && alpha >= 0.0,
        Error::Checkpoint("unsupported Inkling geometry")
    );
    let g = Geometry {
        hidden,
        heads,
        kv: kv_heads,
        dim,
        ffn_width: expert_width,
        context,
        epsilon,
        base: 10000.0,
        count,
    };
    let directory = file.directory()?;
    let (mut s, common) = common_selection(&directory, &g)?;
    let vocabulary = common.embedding.shape[1];
    ensure!(
        unpadded <= vocabulary,
        Error::Checkpoint("unpadded vocabulary exceeds the embedding table")
    );
    let embedding_norm = s.float("token_embd_norm.weight", &[hidden])?;
    let output_query = s.float("output_res_query.weight", &[hidden])?;
    let mut selected = Vec::with_capacity(count);
    let qw = heads * dim;
    let kw = kv_heads * dim;
    for (index, &swa) in pattern.iter().enumerate() {
        let b = |name: &str| format!("blk.{index}.{name}");
        let full = !swa;
        let norms = if full {
            vec![
                s.float(&b("attn_q_inp_norm.weight"), &[hidden])?,
                s.float(&b("attn_k_inp_norm.weight"), &[hidden])?,
                s.float(&b("attn_v_inp_norm.weight"), &[hidden])?,
            ]
        } else {
            vec![s.float(&b("attn_norm.weight"), &[hidden])?]
        };
        let attn_query = if index == 0 {
            None
        } else {
            let shape: &[usize] = if full { &[hidden, 3] } else { &[hidden] };
            Some(s.float(&b("attn_res_query.weight"), shape)?)
        };
        let dense = [
            s.matrix(&b("attn_q"), hidden, qw)?,
            s.matrix(&b("attn_k"), hidden, kw)?,
            s.matrix(&b("attn_v"), hidden, kw)?,
            s.matrix(&b("attn_gate"), hidden, qw)?,
            s.matrix(&b("attn_output"), qw, hidden)?,
            s.matrix(&b("attn_rel"), hidden, heads * rel)?,
            s.matrix(&b("attn_sink_proj"), hidden, heads)?,
        ];
        let tap_matrices = if full {
            Some([
                s.matrix(&b("attn_k_conv_proj"), hidden, taps * kv_heads)?,
                s.matrix(&b("attn_v_conv_proj"), hidden, taps * kv_heads)?,
            ])
        } else {
            None
        };
        let floats = [
            s.float(&b("attn_q_clip.weight"), &[dim, heads])?,
            s.float(&b("attn_rel_clip.weight"), &[rel, heads])?,
            s.float(
                &b("attn_rel_bias.weight"),
                &[rel, if full { extent } else { window }],
            )?,
            s.float(&b("attn_k_conv.weight"), &[taps, kw])?,
            s.float(&b("attn_v_conv.weight"), &[taps, kw])?,
            s.float(&b("attn_post_conv.weight"), &[taps, hidden])?,
            s.float(&b("ffn_post_conv.weight"), &[taps, hidden])?,
            s.float(&b("ffn_res_query.weight"), &[hidden])?,
            s.float(&b("ffn_norm.weight"), &[hidden])?,
        ];
        let router = s.matrix(&b("ffn_gate_inp"), hidden, experts)?;
        ensure!(
            router.kind == WeightFormat::F32 && router.offset.is_multiple_of(16),
            Error::Checkpoint("router weights must be 16-byte aligned float32")
        );
        let router_bias = s.float(&b("exp_probs_b.bias"), &[experts])?;
        let gate_up = s.tensor(&b("ffn_gate_up_exps.weight"))?;
        let down = s.tensor(&b("ffn_down_exps.weight"))?;
        ensure!(
            gate_up.shape == [hidden, 2 * expert_width, experts]
                && down.shape == [expert_width, hidden, experts]
                && gate_up.kind == down.kind
                && matches!(gate_up.kind, WeightFormat::Iq4Nl | WeightFormat::Q8_0),
            Error::Checkpoint("unsupported expert banks in layer {index}")
        );
        selected.push(Selected {
            full,
            norms,
            attn_query,
            dense,
            taps: tap_matrices,
            floats,
            router: [router, router_bias],
            experts: [gate_up, down],
        });
    }
    let mut loops = Vec::with_capacity(core.len());
    for index in core.clone() {
        let b = |name: &str| format!("blk.{index}.loop_{name}");
        let messages = s.float(&b("msg_query.weight"), &[hidden, 4])?;
        let mut control = |path: &str| -> Result<[gguf::Tensor<'_>; 3]> {
            Ok([
                s.matrix(&b(&format!("{path}_ctrl_down")), 2 * hidden, rank)?,
                s.matrix(&b(&format!("{path}_ctrl_gain")), rank, hidden)?,
                s.matrix(&b(&format!("{path}_ctrl_forget")), rank, hidden)?,
            ])
        };
        let controls = [control("q")?, control("k")?, control("v")?, control("ffn")?];
        loops.push(SelectedLoop { messages, controls });
    }
    // Projections run without row gains, so Q4 encodings (which need them) are refused.
    for l in &selected {
        for t in l.dense.iter().chain(l.taps.iter().flatten()) {
            ensure!(
                matches!(
                    t.kind,
                    WeightFormat::Q8_0 | WeightFormat::F32 | WeightFormat::F16 | WeightFormat::Bf16
                ),
                Error::Checkpoint("unsupported Inkling projection encoding {:?}", t.kind)
            );
        }
    }
    // final_logit_softcapping (30 tanh(x / 30)) is monotone, so greedy argmax over the raw
    // logits selects the same token; it matters only once logits or sampling are exposed.
    let gpu = Gpu::new()?;
    let mut i = import(file, s.spans, &gpu)?;
    // The embedding and output tables are padded past the vocabulary; only real rows are used.
    let embedding = i.tensor(common.embedding)?;
    let embedding = Tensor::bound(embedding.storage, embedding.kind, &[hidden, unpadded])?;
    let embedding = i.builder.embedding(embedding)?;
    let output = common
        .output
        .context(Error::Checkpoint, "Inkling requires an output projection")?;
    ensure!(
        output.bias.is_none(),
        Error::Checkpoint("Inkling's output projection has no bias")
    );
    let output = i.tensor(output.matrix)?;
    let output = Tensor::bound(output.storage, output.kind, &[hidden, unpadded])?;
    let output = i.builder.output(output, None)?;
    let output_norm = i.binding(common.norm)?;
    let pipelines = i.builder.elementwise()?;
    let embedding_norm = i.binding(embedding_norm)?;
    let output_query = i.binding(output_query)?;
    let dense = |i: &mut Import, t: gguf::Tensor<'_>| -> Result<Projection> {
        let t = i.tensor(t)?;
        Ok(Projection {
            matrix: i.builder.dense(t)?,
            bias: None,
        })
    };
    let mut layers = Vec::with_capacity(count);
    for l in selected {
        let mut bound = Vec::with_capacity(3);
        for norm in l.norms {
            bound.push(i.binding(norm)?);
        }
        // One SWA norm serves all three paths.
        while bound.len() < 3 {
            bound.push(bound[0].clone());
        }
        let norms: [crate::gpu::ReadOnly; 3] = bound
            .try_into()
            .map_err(|_| Error::Checkpoint("three attention input norms".into()))?;
        let attn_query = l.attn_query.map(|t| i.binding(t)).transpose()?;
        let [q, k, v, gate, out, rel, sink] = l.dense;
        let (q, k, v) = (dense(&mut i, q)?, dense(&mut i, k)?, dense(&mut i, v)?);
        let (gate, out) = (dense(&mut i, gate)?, dense(&mut i, out)?);
        let (rel, sink) = (dense(&mut i, rel)?, dense(&mut i, sink)?);
        let (k_taps, v_taps) = match l.taps {
            Some([kt, vt]) => (Some(dense(&mut i, kt)?), Some(dense(&mut i, vt)?)),
            None => (None, None),
        };
        let [
            q_clip,
            rel_clip,
            rel_table,
            k_conv,
            v_conv,
            out_conv,
            ffn_conv,
            ffn_query,
            ffn_norm,
        ] = l.floats;
        let [router, router_bias] = l.router;
        let [gate_up, down] = l.experts;
        let expert_bank = usize::from(gate_up.kind == WeightFormat::Q8_0);
        layers.push(InklingLayer {
            full: l.full,
            norms,
            attn_query,
            q,
            k,
            v,
            gate,
            out,
            rel,
            sink,
            k_taps,
            v_taps,
            q_clip: i.binding(q_clip)?,
            rel_clip: i.binding(rel_clip)?,
            rel_table: i.binding(rel_table)?,
            k_conv: i.binding(k_conv)?,
            v_conv: i.binding(v_conv)?,
            out_conv: i.binding(out_conv)?,
            ffn_conv: i.binding(ffn_conv)?,
            ffn_query: i.binding(ffn_query)?,
            ffn_norm: i.binding(ffn_norm)?,
            router: Router {
                values: i.binding(router)?,
                bias: i.binding(router_bias)?,
            },
            gate_up: i.binding(gate_up)?,
            down: i.binding(down)?,
            expert_bank,
        });
    }
    let mut loop_layers = Vec::with_capacity(loops.len());
    for l in loops {
        let messages = i.binding(l.messages)?;
        let mut controls = Vec::with_capacity(4);
        for [d, g, f] in l.controls {
            let matrix = |i: &mut Import, t| -> Result<_> {
                let t = i.tensor(t)?;
                i.builder.dense(t)
            };
            controls.push(Control {
                down: matrix(&mut i, d)?,
                gain: matrix(&mut i, g)?,
                forget: matrix(&mut i, f)?,
            });
        }
        let controls: [Control; 4] = controls
            .try_into()
            .map_err(|_| Error::Checkpoint("four loop controllers per core layer".into()))?;
        loop_layers.push(LoopLayer { messages, controls });
    }
    let plan = InklingPlan {
        pipelines: i.builder.inkling()?,
        experts: [
            i.builder
                .inkling_experts(WeightFormat::Iq4Nl, 2 * expert_width)?,
            i.builder
                .inkling_experts(WeightFormat::Q8_0, 2 * expert_width)?,
        ],
        hidden,
        heads,
        kv_heads,
        dim,
        rel,
        extent,
        window,
        epsilon,
        clip,
        rel_clip,
        alpha,
        floor,
        expert_count: experts,
        active,
        expert_width,
        route_scale,
        situ_beta,
        situ_linear_beta,
        taps,
    };
    let norm = Norm {
        n: hidden as u32,
        epsilon,
        full_layer: 1,
    };
    let residency = i.finish()?;
    let base = CommonWeights {
        gpu,
        residency: Some(residency),
        context,
        embedding,
        output,
        output_norm,
        kernels: Kernels { pipelines, norm },
    };
    Ok(Architecture::Inkling(Arc::new(Inkling {
        base,
        embedding_norm,
        output_query,
        layers,
        loops: loop_layers,
        core,
        passes,
        block,
        plan,
    })))
}
