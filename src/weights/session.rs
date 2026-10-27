use super::*;
use crate::gpu::BufferView;
use anyhow::{Result, anyhow, bail, ensure};
use compute::{Bindings, Compute};
use state::*;
use std::ops::Range;
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
}

/// Host access permission for mutable GPU state, restored only after a validated completion.
enum Progress {
    Empty,
    Ready { position: usize },
    Unusable,
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
}

impl Session {
    pub(super) fn new(weights: &Architecture, capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= weights.base().context,
            "invalid context capacity"
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
        };
        let residency = weights.base().residency.clone();
        let compute = Compute::new(&weights.base().gpu, residency);
        Ok(Self {
            compute,
            inner,
            progress: Progress::Empty,
            capacity,
            prefill_rows,
            features: 0,
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
            bail!("this model has no audio encoder");
        };
        let (Some(encoder), Some((scratch, features))) = (&weights.audio, &memory.audio) else {
            bail!("this model has no audio encoder");
        };
        let windows = encoder.windows(samples)?;
        let total: usize = windows.iter().map(qwen::audio::Window::tokens).sum();
        let mut used = self.features;
        ensure!(
            total <= self.capacity - used,
            "audio exceeds the session capacity"
        );
        let vocabulary = self.vocabulary_size();
        let width = encoder.output * 2;
        let mut ids = Vec::with_capacity(total);
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
            // Partial encodings are harmless: unreferenced rows are never read.
            submission.await?;
            ids.extend((used..used + rows).map(|row| (vocabulary + row) as u32));
            used += rows;
        }
        self.features = used;
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
        match self.progress {
            Progress::Unusable => Err(anyhow!("session is unusable; create a new session")),
            Progress::Empty => Ok(0),
            Progress::Ready { position, .. } => Ok(position),
        }
    }

    /// Consume input already validated by `infer` and predict its successor.
    pub(super) async fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        let mut position = self.position()?;
        // Invalidating before submission; cancellation cannot expose abandoned mutable buffers.
        self.progress = Progress::Unusable;
        let mut chunks = tokens.chunks(self.prefill_rows).peekable();
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
        self.accept_prediction(position)
    }

    /// Feed back the GPU-resident prediction and return its successor.
    pub(super) async fn decode(&mut self) -> Result<u32> {
        let position = self.position()?;
        ensure!(position > 0, "decode requires a prefilled session");
        ensure!(position < self.capacity, "context capacity exceeded");
        self.progress = Progress::Unusable;
        self.forward(&self.common().1.next, position..position + 1, Some(0))
            .await?;
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
            "non-finite logits produced no valid token"
        );
        self.progress = Progress::Ready { position };
        Ok(selected)
    }

    /// Forget the sequence and reuse its allocated storage for an unrelated prompt.
    ///
    /// Capacity and scratch size stay fixed, and predictions become unavailable
    /// until the next prefill. Reset is allowed only for healthy idle sessions;
    /// it cannot recover one whose GPU work failed or was cancelled.
    pub fn reset(&mut self) -> Result<()> {
        let position = self.position()?;
        if position == 0 {
            return Ok(());
        }
        // Forgetting the KV prefix by position; recurrent memory must be zeroed because
        // every update reads it regardless of the new sequence's length.
        let recurrent: &[Recurrence] = match &self.inner {
            Execution::GptOss { .. } | Execution::QwenAttention { .. } => &[],
            Execution::QwenDelta { memory, .. } => &memory.states,
            Execution::QwenHybrid { memory, .. } => &memory.recurrent,
            Execution::Nemotron { memory, .. } => &memory.recurrent,
        };
        for state in recurrent {
            state.reset();
        }
        self.progress = Progress::Empty;
        self.features = 0;
        Ok(())
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
