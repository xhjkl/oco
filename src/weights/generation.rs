use super::Session;
use anyhow::{Result, bail, ensure};
use futures_channel::mpsc::{Receiver, channel};
use std::{
    borrow::Cow,
    future::{Future, poll_fn},
};

/// Pull greedy token predictions from one borrowed session.
///
/// The first pull consumes the supplied input. Each later pull consumes the previous
/// prediction and returns its successor. Stop IDs, output limits, and text decoding
/// belong to the caller. The engine ends the stream when the session is full.
///
/// Pausing between completed pulls is safe. Dropping a pending pull makes the session
/// unusable because submitted GPU work may still be changing its state. Dropping an
/// unpolled pull or an idle inference does not change the session.
#[must_use = "inference runs only when next is polled"]
pub struct Inference<'a> {
    session: &'a mut Session,
    phase: Phase<'a>,
}

/// Next operation, including an abort marker installed before any GPU wait.
enum Phase<'a> {
    Prefill(Cow<'a, [u32]>),
    Decode,
    Complete,
    Aborted,
}

impl Session {
    /// Append input and pull predictions without repeating the retained context.
    ///
    /// Supply only the new suffix. The last emitted prediction has not been consumed;
    /// include it in this input if continuing a previous run. Input must be nonempty,
    /// contain valid token IDs, and fit the remaining session capacity.
    /// Validation leaves the session unchanged; GPU work starts on the first pull.
    pub fn infer<'a>(&'a mut self, tokens: impl Into<Cow<'a, [u32]>>) -> Result<Inference<'a>> {
        let tokens = tokens.into();
        let position = self.position()?;
        ensure!(!tokens.is_empty(), "prefill needs at least one token");
        ensure!(
            tokens.len() <= self.capacity - position,
            "context capacity exceeded"
        );
        let vocabulary = self.vocabulary_size();
        ensure!(
            tokens.iter().all(|&t| (t as usize) < vocabulary),
            "token outside vocabulary"
        );
        Ok(Inference {
            session: self,
            phase: Phase::Prefill(tokens),
        })
    }
}

impl<'a> Inference<'a> {
    /// Advance inference by one prediction, returning `None` after context exhaustion.
    ///
    /// Consuming the final available input position still yields its prediction.
    /// That prediction cannot be fed back into this full session. Errors abort the
    /// run; they never appear as a normal end of stream.
    pub async fn next(&mut self) -> Result<Option<u32>> {
        // Aborting before awaiting; a dropped wait cannot resume a partially updated state.
        let phase = std::mem::replace(&mut self.phase, Phase::Aborted);
        let token = match phase {
            Phase::Aborted => bail!("inference was aborted; create a new session"),
            Phase::Complete => {
                self.phase = Phase::Complete;
                return Ok(None);
            }
            Phase::Prefill(input) => self.session.prefill(&input).await?,
            Phase::Decode => self.session.decode().await?,
        };
        let position = self.session.position()?;
        self.phase = if position == self.session.capacity {
            Phase::Complete
        } else {
            Phase::Decode
        };
        Ok(Some(token))
    }

    /// Separate production from consumption with a one-token handoff.
    ///
    /// Drive the returned future on your runtime while reading the receiver. No task
    /// or thread is spawned here. Backpressure bounds queued plus in-progress tokens
    /// to one. Dropping the receiver stops production after the current pull finishes;
    /// dropping the producer during a pending GPU operation makes its session unusable.
    /// Errors are sent through the receiver before it closes. After the producer
    /// finishes, the session position may include the last received token: a pull
    /// already in progress consumes it even if its prediction is never delivered.
    pub fn into_producer(
        mut self,
    ) -> (impl Future<Output = ()> + Send + 'a, Receiver<Result<u32>>) {
        // Reserving only futures-channel's single sender slot; the sender is never cloned.
        let (mut sender, receiver) = channel(0);
        let produce = async move {
            loop {
                let ready = poll_fn(|cx| sender.poll_ready(cx)).await;
                if ready.is_err() {
                    break;
                }
                let next = self.next().await;
                let Some(item) = next.transpose() else {
                    break;
                };
                let failed = item.is_err();
                let sent = sender.start_send(item);
                if sent.is_err() || failed || matches!(self.phase, Phase::Complete) {
                    break;
                }
            }
        };
        (produce, receiver)
    }
}

#[cfg(test)]
mod tests;
