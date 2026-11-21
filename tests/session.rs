//! Session reuse: greedy continuations after checkpoints, rewinds, and saved contexts against
//! a fresh session fed the same IDs. Set QWEN_PATH, GPT_OSS_PATH, and NEMOTRON_PATH to GGUF
//! files; tests without a path are skipped.
mod common;
use anyhow::{Context, Result};
use common::{Family, WEATHER, family, prompt};
use oco::{Session, Weights};
use std::{sync::Mutex, time::Instant};

// These models do not fit in GPU memory together, even when the test harness runs in parallel.
static GPU: Mutex<()> = Mutex::new(());

/// Greedy tokens generated after each input.
const GENERATED: usize = 24;
/// Prompt IDs that follow the checkpointed preamble.
const TAIL: usize = 24;

#[test]
fn qwen_reuses_sessions() -> Result<()> {
    reuse("QWEN_PATH")
}

#[test]
fn gpt_oss_reuses_sessions() -> Result<()> {
    reuse("GPT_OSS_PATH")
}

#[test]
fn nemotron_reuses_sessions() -> Result<()> {
    reuse("NEMOTRON_PATH")
}

/// Qwen3.8-27B's full 262k context needs about 16 GiB of KV cache: within the device's working
/// set alone, but not beside the 16 GB of weights it must share that set with.
#[test]
fn qwen_refuses_sessions_beyond_the_working_set() -> Result<()> {
    let Some(path) = std::env::var_os("QWEN_PATH") else {
        eprintln!("QWEN_PATH unset; skipping");
        return Ok(());
    };
    let _gpu = GPU.lock().unwrap_or_else(|error| error.into_inner());
    let weights = oco::load(path)?;
    assert!(weights.new_session(weights.context_limit()).is_err());
    assert!(weights.new_session(32_768).is_ok());
    Ok(())
}

fn reuse(variable: &str) -> Result<()> {
    let Some(path) = std::env::var_os(variable) else {
        eprintln!("{variable} unset; skipping");
        return Ok(());
    };
    let _gpu = GPU.lock().unwrap_or_else(|error| error.into_inner());
    let weights = oco::load(path)?;
    let ids = prompt(
        weights.tokenizer(),
        "You are a helpful assistant. For any question about the weather, you must call the get_weather function instead of answering from memory.",
        "What's the weather like in Paris right now? Also tell me whether I need an umbrella.",
        Some(&WEATHER),
    )?;
    // Recurrent models replay rewound IDs in exactly the original prefill; attention-only
    // models recompute rewound rows in place, as the suffix-append path does.
    let recurrent = family(weights.tokenizer()) != Family::GptOss;
    let split = ids.len() - TAIL;
    let (preamble, tail) = ids.split_at(split);
    let capacity = ids.len() + 2 * GENERATED;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let fresh = generate(&mut weights.new_session(capacity)?, &ids).await?;

        // Checkpointing the consumed preamble costs no forward pass of its own.
        let mut session = weights.new_session(capacity)?;
        let first = {
            let mut inference = session.infer(preamble)?;
            inference.next().await?.context("no prediction")?
        };
        assert_eq!(session.history(), preamble);
        let started = Instant::now();
        session.checkpoint()?;
        let checkpoint = started.elapsed();
        let file = std::env::temp_dir().join(format!("oco-session-{}.ctx", std::process::id()));
        let started = Instant::now();
        session.save(&file)?;
        let saved = started.elapsed();
        let bytes = std::fs::metadata(&file)?.len();
        let appended = generate(&mut session, tail).await?;
        let agreement = common_prefix(&appended, &fresh);
        eprintln!(
            "{variable}: {} + {} IDs; append path agrees with one prefill for {agreement}/{GENERATED} tokens; checkpoint {checkpoint:.2?}, save {saved:.2?} ({:.1} MiB)",
            preamble.len(),
            tail.len(),
            bytes as f64 / (1 << 20) as f64,
        );
        assert_eq!(appended[0], fresh[0], "append path changed the first prediction");

        // Continuing through the first prediction matches plain decoding.
        let mut continued = weights.new_session(capacity)?;
        let mut inference = continued.infer(&ids)?;
        let head = inference.next().await?.context("no prediction")?;
        drop(inference);
        continued.checkpoint()?;
        let rest = generate(&mut continued, &[head]).await?;
        assert_eq!([&[head], &rest[..GENERATED - 1]].concat(), fresh);

        // A restored context continues exactly like the session that saved it.
        let started = Instant::now();
        let mut opened = weights.open_session(&file, capacity)?;
        let open = started.elapsed();
        assert_eq!(opened.history(), preamble);
        assert_eq!(generate(&mut opened, tail).await?, appended);

        // Rewinding inside the tail replays it from the preamble checkpoint (or rewinds the
        // caches in place) and continues as the uninterrupted session did.
        let started = Instant::now();
        session.rewind(split + 5)?;
        assert_eq!(session.history(), &ids[..split + 5]);
        let rewound = generate(&mut session, &tail[5..]).await?;
        let replay = started.elapsed();
        let check = |got: &[u32], want: &[u32], what: &str| {
            if recurrent {
                assert_eq!(got, want, "{what}");
            } else {
                assert_agrees(got, want, agreement, what);
            }
        };
        check(&rewound, &appended, "rewind past the checkpoint");

        // Rewinding before every checkpoint replays from the empty sequence.
        session.rewind(split - 3)?;
        let replayed = generate(&mut session, &ids[split - 3..]).await?;
        check(&replayed, &fresh, "rewind before the checkpoint");

        // Rewinding exactly to a checkpoint, twice while a replay is pending, and past a
        // checkpoint call that must do nothing because a replay is pending.
        let mut paths = weights.open_session(&file, capacity)?;
        check(&generate(&mut paths, tail).await?, &appended, "opened session");
        paths.rewind(split)?;
        check(&generate(&mut paths, tail).await?, &appended, "rewind to the checkpoint");
        paths.rewind(split + 10)?;
        paths.rewind(split + 7)?;
        assert_eq!(paths.history(), &ids[..split + 7]);
        paths.checkpoint()?;
        check(&generate(&mut paths, &tail[7..]).await?, &appended, "repeated rewinds");
        paths.rewind(split + 9)?;
        check(&generate(&mut paths, &tail[9..]).await?, &appended, "rewind past a no-op checkpoint");

        // Saving with a replay pending keeps the pending IDs for the opened session.
        paths.rewind(split + 5)?;
        let pending = std::env::temp_dir().join(format!("oco-pending-{}.ctx", std::process::id()));
        paths.save(&pending)?;
        let mut reopened = weights.open_session(&pending, capacity)?;
        std::fs::remove_file(&pending)?;
        assert_eq!(reopened.history(), &ids[..split + 5]);
        check(&generate(&mut reopened, &tail[5..]).await?, &appended, "saved replay");

        // An opened session rewinds before its checkpoint by replaying from the start.
        let mut early = weights.open_session(&file, capacity)?;
        early.rewind(split - 3)?;
        check(&generate(&mut early, &ids[split - 3..]).await?, &fresh, "opened, rewound early");

        // Of four checkpoints only the earliest and the most recent remain; rewinding between
        // them replays from the earliest.
        let mut many = weights.open_session(&file, capacity)?;
        for range in [0..8, 8..16, 16..TAIL] {
            let mut inference = many.infer(&tail[range])?;
            inference.next().await?.context("no prediction")?;
            drop(inference);
            many.checkpoint()?;
        }
        many.rewind(split + 12)?;
        check(&generate(&mut many, &tail[12..]).await?, &appended, "rewind among checkpoints");
        session.reset()?;
        assert!(session.history().is_empty());
        assert_eq!(generate(&mut session, &ids).await?, fresh);
        eprintln!(
            "{variable}: open {open:.2?}, rewind + {} ID prefill + {GENERATED} tokens {replay:.2?}; first preamble prediction {first}",
            tail.len() - 5
        );

        // Unreadable files are refused so the caller can prefill instead; telling a file
        // apart from one saved for other weights or builds is the caller's naming.
        let contents = std::fs::read(&file)?;
        std::fs::write(&file, &contents[..contents.len() - 1])?;
        assert!(weights.open_session(&file, capacity).is_err());
        std::fs::write(&file, b"not a context")?;
        assert!(weights.open_session(&file, capacity).is_err());
        std::fs::remove_file(&file)?;
        assert!(weights.open_session(&file, capacity).is_err());
        anyhow::Ok(())
    })?;
    check_capacity(&weights)
}

/// A saved context larger than the opening session must be refused.
fn check_capacity(weights: &Weights) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let ids = weights
            .tokenizer()
            .encode("Capacity check for saved contexts.")?;
        let mut session = weights.new_session(64)?;
        generate(&mut session, &ids).await?;
        let file = std::env::temp_dir().join(format!("oco-capacity-{}.ctx", std::process::id()));
        session.save(&file)?;
        let consumed = session.history().len();
        assert!(weights.open_session(&file, consumed - 1).is_err());
        assert!(weights.open_session(&file, consumed).is_ok());
        std::fs::remove_file(&file)?;
        anyhow::Ok(())
    })
}

/// Consume `input` and return the next `GENERATED` greedy predictions.
async fn generate(session: &mut Session, input: &[u32]) -> Result<Vec<u32>> {
    let mut inference = session.infer(input)?;
    let mut tokens = Vec::with_capacity(GENERATED);
    while tokens.len() < GENERATED {
        tokens.push(inference.next().await?.context("stream ended early")?);
    }
    Ok(tokens)
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

/// Require a rewound continuation to agree with its reference at least as far as the
/// suffix-append path agrees with one prefill (`floor` tokens).
fn assert_agrees(got: &[u32], want: &[u32], floor: usize, what: &str) {
    let agreement = common_prefix(got, want);
    eprintln!("{what}: {agreement}/{GENERATED} tokens agree");
    assert!(
        agreement >= floor,
        "{what} agrees for fewer than {floor} tokens"
    );
}
