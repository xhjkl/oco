use super::*;
use crate::error::{Error, Result, bail, ensure};
use crate::gpu::BufferView;
use compute::{Bindings, Compute};
use state::*;
use std::{borrow::Cow, ops::Range};

mod saved;

/// Mutable sequence state and scratch storage sharing immutable model weights.
///
/// Attention caches and recurrent memories retain consumed context across inference
/// runs. Invalid input leaves state unchanged. Failed or dropped in-flight GPU work
/// makes the session unusable; submitted resources remain alive until Metal finishes.
pub struct Session {
    pub(super) compute: Compute,
    inner: Execution,
    progress: Progress,
    pub(super) capacity: usize,
    prefill_rows: usize,
    /// Audio feature rows filled so far; IDs `vocabulary..vocabulary + features` select them.
    pub(super) features: usize,
    /// Consumed IDs, including any a rewind left for the next prefill to replay.
    history: Vec<u32>,
    /// Retained recurrent memories, oldest first: the earliest and the most recent.
    checkpoints: Vec<Checkpoint>,
}

/// Host access permission for mutable GPU state, restored only after a validated completion.
enum Progress {
    /// GPU state holds the first `resident` IDs of the history; later ones await replay.
    Ready {
        resident: usize,
    },
    Unusable,
}

/// Recurrent memory copied at a history position, so rewinding there needs no replay.
/// Attention-only models keep just the position; their caches rewind in place.
struct Checkpoint {
    position: usize,
    state: Vec<u8>,
}

/// Immutable weights and mutable memory paired by layout so every layer receives the state it expects.
#[allow(clippy::large_enum_variant)]
enum Execution {
    GptOss {
        weights: Arc<gpt_oss::GptOss>,
        memory: gpt_oss::Memory,
    },
    QwenAttention {
        weights: Arc<qwen::attention::QwenAttention>,
        memory: qwen::attention::Memory,
    },
    QwenDelta {
        weights: Arc<qwen::delta::QwenDelta>,
        memory: qwen::delta::Memory,
    },
    QwenHybrid {
        weights: Arc<qwen::hybrid::QwenHybrid>,
        memory: qwen::hybrid::Memory,
    },
    Nemotron {
        weights: Arc<nemotron::Nemotron>,
        memory: nemotron::Memory,
    },
    Inkling {
        weights: Arc<inkling::Inkling>,
        memory: inkling::Memory,
    },
}

impl Session {
    pub(super) fn new(weights: &Architecture, capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= weights.base().context,
            Error::Request("invalid context capacity")
        );
        let prefill_rows = capacity.min(MAX_PREFILL_ROWS);
        let inner = match weights {
            Architecture::GptOss(weights) => {
                let memory = gpt_oss::Memory::new(weights, capacity, prefill_rows)?;
                Execution::GptOss {
                    weights: weights.clone(),
                    memory,
                }
            }
            Architecture::QwenAttention(weights) => {
                let memory = qwen::attention::Memory::new(weights, capacity, prefill_rows)?;
                Execution::QwenAttention {
                    weights: weights.clone(),
                    memory,
                }
            }
            Architecture::QwenDelta(weights) => {
                let memory = qwen::delta::Memory::new(weights, prefill_rows)?;
                Execution::QwenDelta {
                    weights: weights.clone(),
                    memory,
                }
            }
            Architecture::QwenHybrid(weights) => {
                let memory = qwen::hybrid::Memory::new(weights, capacity, prefill_rows)?;
                Execution::QwenHybrid {
                    weights: weights.clone(),
                    memory,
                }
            }
            Architecture::Nemotron(weights) => {
                let memory = nemotron::Memory::new(weights, capacity, prefill_rows)?;
                Execution::Nemotron {
                    weights: weights.clone(),
                    memory,
                }
            }
            Architecture::Inkling(weights) => {
                let memory = inkling::Memory::new(weights, capacity, prefill_rows)?;
                Execution::Inkling {
                    weights: weights.clone(),
                    memory,
                }
            }
        };
        let residency = weights.base().residency.clone();
        let compute = Compute::new(&weights.base().gpu, residency);
        Ok(Self {
            compute,
            inner,
            progress: Progress::Ready { resident: 0 },
            capacity,
            prefill_rows,
            features: 0,
            history: Vec::new(),
            checkpoints: Vec::new(),
        })
    }

    /// Encode 16 kHz mono audio into feature rows the next prompt can reference.
    ///
    /// Returns one ID per audio token, each at or beyond the vocabulary; splice them into
    /// the prompt between the checkpoint's audio markers. Rows accumulate until [`Self::reset`]
    /// and count against the session capacity. Requires a model loaded with [`crate::load_asr`].
    pub async fn embed_audio(&mut self, samples: &[f32]) -> Result<Vec<u32>> {
        self.position()?;
        let Execution::QwenAttention { weights, memory } = &self.inner else {
            bail!(Error::Request("this model has no audio encoder"));
        };
        let (Some(encoder), Some((scratch, features))) = (&weights.audio, &memory.audio) else {
            bail!(Error::Request("this model has no audio encoder"));
        };
        let windows = encoder.windows(samples)?;
        let total: usize = windows.iter().map(qwen::audio::Window::tokens).sum();
        let mut used = self.features;
        ensure!(
            total <= self.capacity - used,
            Error::Request("audio exceeds the session capacity")
        );
        let vocabulary = self.vocabulary_size();
        let width = encoder.output * 2;
        let mut ids = Vec::with_capacity(total);
        let progress = std::mem::replace(&mut self.progress, Progress::Unusable);
        for window in &windows {
            scratch.mel.write(0, &window.mel);
            let rows = window.tokens();
            let destination = features.slice(used * width, rows * width);
            let submission = self.compute.submit(|commands| {
                let compute = Bindings {
                    kernels: &encoder.kernels,
                    scratch: &scratch.common,
                    encoder: commands,
                };
                encoder.encode(&compute, scratch, window.chunks, &destination);
            });
            submission.await?;
            ids.extend((used..used + rows).map(|row| (vocabulary + row) as u32));
            used += rows;
        }
        self.features = used;
        self.progress = progress;
        Ok(ids)
    }

    fn common(&self) -> (&CommonWeights, &CommonScratch) {
        with_architecture!(&self.inner, Execution { weights, memory } => (&weights.base, &memory.common))
    }

    /// Return the number of token IDs accepted by this session's model.
    pub(super) fn vocabulary_size(&self) -> usize {
        self.common().0.embedding.vocabulary
    }

    /// Count consumed tokens, excluding the last emitted prediction.
    /// Return an error if incomplete GPU work left the state unusable.
    pub fn position(&self) -> Result<usize> {
        self.resident()?;
        Ok(self.history.len())
    }

    /// Consumed input IDs, oldest first. The last emitted prediction is not among them.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Count the consumed tokens the GPU state already holds.
    fn resident(&self) -> Result<usize> {
        match self.progress {
            Progress::Unusable => Err(Error::Request(
                "session is unusable; create a new session".into(),
            )),
            Progress::Ready { resident } => Ok(resident),
        }
    }

    /// Consume input already validated by `infer`, after any IDs a rewind left to replay,
    /// and predict its successor.
    pub(super) async fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        let mut position = self.resident()?;
        let input = match &self.history[position..] {
            [] => Cow::Borrowed(tokens),
            replay => Cow::Owned([replay, tokens].concat()),
        };
        // Invalidating before submission; cancellation cannot expose abandoned mutable buffers.
        self.progress = Progress::Unusable;
        let mut chunks = input.chunks(self.prefill_rows).peekable();
        while let Some(chunk) = chunks.next() {
            let scratch = self.common().1;
            scratch.input.write(0, chunk);
            let output_row = match chunks.peek() {
                Some(_) => None,
                None => Some(chunk.len() - 1),
            };
            self.forward(&scratch.input, position..position + chunk.len(), output_row)
                .await?;
            position += chunk.len();
        }
        self.history.extend_from_slice(tokens);
        self.accept_prediction(position)
    }

    /// Feed back the GPU-resident prediction `token` and return its successor.
    pub(super) async fn decode(&mut self, token: u32) -> Result<u32> {
        let position = self.resident()?;
        ensure!(
            position > 0 && position == self.history.len(),
            Error::Request("decode requires a prefilled session")
        );
        ensure!(
            position < self.capacity,
            Error::Request("context capacity exceeded")
        );
        self.progress = Progress::Unusable;
        self.forward(&self.common().1.next, position..position + 1, Some(0))
            .await?;
        self.history.push(token);
        self.accept_prediction(position + 1)
    }

    /// Accept a valid GPU prediction as the session's next usable state.
    /// The caller must await GPU completion first; invalid output leaves the session unusable.
    fn accept_prediction(&mut self, position: usize) -> Result<u32> {
        let mut selected = [0u32];
        self.common().1.next.read(&mut selected);
        let [selected] = selected;
        ensure!(
            (selected as usize) < self.vocabulary_size(),
            Error::Gpu("non-finite logits produced no valid token")
        );
        self.progress = Progress::Ready { resident: position };
        Ok(selected)
    }

    /// Forget the sequence and its checkpoints and reuse the storage for an unrelated prompt.
    ///
    /// Capacity and scratch size stay fixed, and predictions become unavailable
    /// until the next prefill. Reset is allowed only for healthy idle sessions;
    /// it cannot recover one whose GPU work failed or was cancelled.
    pub fn reset(&mut self) -> Result<()> {
        self.resident()?;
        self.checkpoints.clear();
        self.rewind(0)?;
        self.features = 0;
        Ok(())
    }

    /// Keep only the first `position` consumed tokens; the next `infer` continues after them.
    ///
    /// Capacity-long attention caches rewind in place. Recurrent memory and ring caches return
    /// to the newest checkpoint at or before `position` (the empty sequence always counts as
    /// one), and the next `infer` re-consumes the retained IDs between that checkpoint and
    /// `position` before its own input. Checkpoints past `position` are forgotten.
    pub fn rewind(&mut self, position: usize) -> Result<()> {
        let resident = self.resident()?;
        ensure!(
            position <= self.history.len(),
            Error::Request("cannot rewind past the consumed context")
        );
        self.history.truncate(position);
        self.checkpoints.retain(|c| c.position <= position);
        if position >= resident {
            return Ok(());
        }
        let resident = match self.recurrent() {
            [] => position,
            recurrent => match self.checkpoints.last() {
                Some(checkpoint) => {
                    self.restore(&checkpoint.state, checkpoint.position);
                    checkpoint.position
                }
                // Restoring the empty sequence, whose memory is all zeros.
                None => {
                    recurrent.iter().for_each(Recurrence::reset);
                    0
                }
            },
        };
        self.progress = Progress::Ready { resident };
        Ok(())
    }

    /// Remember recurrent memory at the current position so later rewinds to it, or past it,
    /// replay nothing before it. A session retains its earliest and its most recent checkpoint.
    /// Each is a host copy of all recurrent memory (about 150 MiB for Qwen3.8-27B, the same
    /// bytes every saved context carries), plus the filled rows of any ring caches (up to
    /// 240 MiB per checkpoint for Inkling 2 Mini sessions longer than 1536 positions, which
    /// [`Weights::capacity_within`] does not count); attention-only models store nothing. The
    /// empty sequence needs no checkpoint, so taking one there does nothing.
    ///
    /// To checkpoint a prompt without an extra forward pass, pull its first prediction, drop
    /// the `Inference`, call this, and continue with `infer(vec![prediction])`: that prefill
    /// does exactly the work of the decode step it replaces. With IDs still awaiting replay
    /// after a rewind, the memory they start from is already retained and this does nothing.
    pub fn checkpoint(&mut self) -> Result<()> {
        let position = self.resident()?;
        if position == 0
            || position < self.history.len()
            || self
                .checkpoints
                .last()
                .is_some_and(|c| c.position == position)
        {
            return Ok(());
        }
        // Reusing the replaced checkpoint's allocation.
        let state = match self.checkpoints.len() {
            2 => self.checkpoints.pop().map(|c| c.state).unwrap_or_default(),
            _ => Vec::new(),
        };
        let state = self.snapshot(position, state);
        self.checkpoints.push(Checkpoint { position, state });
        Ok(())
    }

    /// Copy recurrent memory and the rows ring caches hold at `position` into `state`.
    fn snapshot(&self, position: usize, mut state: Vec<u8>) -> Vec<u8> {
        let recurrent = self.recurrent_views().map(BufferView::length).sum();
        state.resize(recurrent, 0);
        let mut offset = 0;
        for view in self.recurrent_views() {
            let end = offset + view.length();
            view.read(&mut state[offset..end]);
            offset = end;
        }
        self.append_rings(position, state)
    }

    /// Append the rows ring caches hold at `position` to a copy of recurrent memory.
    fn append_rings(&self, position: usize, mut state: Vec<u8>) -> Vec<u8> {
        let mut offset = state.len();
        state.resize(offset + self.ring_bytes(position), 0);
        for (kv, ring) in self.histories() {
            if ring > 0 {
                let rows = position.min(ring) * self.kv_row_bytes();
                for view in [&kv.k, &kv.v] {
                    view.read(&mut state[offset..offset + rows]);
                    offset += rows;
                }
            }
        }
        state
    }

    /// Bytes of the rows ring caches hold at `position`: a prefix until the ring first fills,
    /// then all of it.
    fn ring_bytes(&self, position: usize) -> usize {
        self.histories()
            .filter(|&(_, ring)| ring > 0)
            .map(|(_, ring)| 2 * position.min(ring) * self.kv_row_bytes())
            .sum()
    }

    /// Every attention history in layer (visit) order with its ring size in rows, or zero for
    /// a capacity-long cache.
    fn histories(&self) -> impl Iterator<Item = (&Kv, usize)> {
        let caches: Box<dyn Iterator<Item = (&Kv, usize)>> = match &self.inner {
            Execution::Inkling { memory, .. } => {
                Box::new(memory.caches.iter().map(|(kv, ring)| (kv, *ring)))
            }
            _ => Box::new(self.kv().iter().map(|kv| (kv, 0))),
        };
        caches
    }

    /// Per-layer attention history, in layer order.
    fn kv(&self) -> &[Kv] {
        match &self.inner {
            Execution::QwenDelta { .. } => &[],
            Execution::GptOss { memory, .. } => &memory.states,
            Execution::QwenAttention { memory, .. } => &memory.states,
            Execution::QwenHybrid { memory, .. } => &memory.kv_states,
            Execution::Nemotron { memory, .. } => &memory.kv_states,
            Execution::Inkling { memory, .. } => &memory.kv_states,
        }
    }

    /// Recurrent memory buffers in checkpoint and saved-context order.
    fn recurrent_views(&self) -> impl Iterator<Item = &BufferView> {
        self.recurrent().iter().flat_map(|r| [&r.conv, &r.ssm])
    }

    /// Overwrite recurrent memory and ring caches with a [`Self::snapshot`] taken at `position`.
    fn restore(&self, state: &[u8], position: usize) {
        let mut offset = 0;
        for view in self.recurrent_views() {
            let end = offset + view.length();
            view.write(0, &state[offset..end]);
            offset = end;
        }
        for (kv, ring) in self.histories() {
            if ring > 0 {
                let rows = position.min(ring) * self.kv_row_bytes();
                for view in [&kv.k, &kv.v] {
                    view.write(0, &state[offset..offset + rows]);
                    offset += rows;
                }
            }
        }
    }

    /// Per-layer recurrent memory, in layer order; empty for attention-only models.
    fn recurrent(&self) -> &[Recurrence] {
        match &self.inner {
            Execution::GptOss { .. } | Execution::QwenAttention { .. } => &[],
            Execution::QwenDelta { memory, .. } => &memory.states,
            Execution::QwenHybrid { memory, .. } => &memory.recurrent,
            Execution::Nemotron { memory, .. } => &memory.recurrent,
            Execution::Inkling { memory, .. } => &memory.recurrent,
        }
    }

    /// Consume the supplied sequence positions and optionally score one chunk-relative output row.
    /// The caller must forbid host access to mutable buffers before entering this operation.
    async fn forward(
        &self,
        input: &BufferView,
        positions: Range<usize>,
        output_row: Option<usize>,
    ) -> Result<()> {
        let (base, scratch) = self.common();
        let features = match &self.inner {
            Execution::QwenAttention { memory, .. } => memory
                .audio
                .as_ref()
                .map(|(_, features)| features.binding()),
            _ => None,
        };
        let submission = self.compute.submit(|encoder| {
            let compute = Bindings {
                kernels: &base.kernels,
                scratch,
                encoder,
            };
            let features = features.unwrap_or(scratch.next.binding());
            compute.embed(&base.embedding, input, positions.len(), features);
            with_architecture!(&self.inner, Execution { weights, memory } =>
                weights.encode_layers(&compute, memory, positions, output_row)
            );
            // Scoring only the last row; earlier rows establish context for the next prediction.
            if let Some(row) = output_row {
                compute.norm(
                    &base.output_norm,
                    scratch.hidden.at(row * base.embedding.width * 2),
                    &scratch.norm,
                    1,
                );
                compute.output(&base.output);
                compute.argmax(base.output.vocabulary);
            }
        });
        submission.await
    }
}
