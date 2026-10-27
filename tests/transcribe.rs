//! Speech recognition acceptance on the JFK sample: Whisper (whisper.cpp ggml checkpoint) and
//! Qwen3-ASR (text GGUF plus audio encoder GGUF) must both produce the famous sentence.
//! Set WHISPER_PATH, and QWEN3_ASR_PATH with QWEN3_ASR_MMPROJ; tests without paths are skipped.
use anyhow::{Context, Result};
use std::sync::Mutex;
use std::time::Instant;

static GPU: Mutex<()> = Mutex::new(());

const EXPECTED: [&str; 8] = [
    "fellow americans",
    "ask not what your country can do for you",
    "ask what you can do for your country",
    "and so",
    "my",
    "can",
    "do",
    "for",
];

fn samples() -> Result<Vec<f32>> {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/jfk.wav"))?;
    oco::audio::wav_samples(&bytes)
}

fn check(text: &str) {
    let normalized: String = text
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    for phrase in EXPECTED {
        assert!(
            normalized.contains(phrase),
            "missing {phrase:?} in {text:?}"
        );
    }
}

#[test]
fn whisper_transcribes_jfk() -> Result<()> {
    let Some(path) = std::env::var_os("WHISPER_PATH") else {
        eprintln!("WHISPER_PATH unset; skipping");
        return Ok(());
    };
    let _gpu = GPU.lock().unwrap_or_else(|error| error.into_inner());
    let samples = samples()?;
    let whisper = oco::Whisper::load(path)?;
    let mut transcriber = whisper.new_transcriber()?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    // Detected language first, then forced English; both must transcribe the sentence.
    let started = Instant::now();
    let detected = runtime.block_on(transcriber.transcribe(&samples, None))?;
    eprintln!(
        "whisper ({}, {} tokens, {:?}): {}",
        detected.language,
        detected.tokens,
        started.elapsed(),
        detected.text
    );
    assert_eq!(detected.language, "en");
    check(&detected.text);
    let forced = runtime.block_on(transcriber.transcribe(&samples, Some("en")))?;
    assert_eq!(forced.text, detected.text);
    assert!(whisper.languages().contains(&"de"));
    assert!(
        runtime
            .block_on(transcriber.transcribe(&samples, Some("xx")))
            .is_err()
    );
    Ok(())
}

#[test]
fn qwen3_asr_transcribes_jfk() -> Result<()> {
    let (Some(text), Some(audio)) = (
        std::env::var_os("QWEN3_ASR_PATH"),
        std::env::var_os("QWEN3_ASR_MMPROJ"),
    ) else {
        eprintln!("QWEN3_ASR_PATH or QWEN3_ASR_MMPROJ unset; skipping");
        return Ok(());
    };
    let _gpu = GPU.lock().unwrap_or_else(|error| error.into_inner());
    let samples = samples()?;
    let weights = oco::load_asr(text, audio)?;
    let tokenizer = weights.tokenizer();
    let special = |name: &str| {
        tokenizer
            .special_token(name)
            .with_context(|| format!("missing {name}"))
    };
    let mut session = weights.new_session(1024)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let started = Instant::now();
    let audio_ids = runtime.block_on(session.embed_audio(&samples))?;
    let encoded = started.elapsed();
    // One frame per 160-sample hop plus one, in windows of 800 frames, 13 tokens per 100-frame
    // chunk: eleven seconds give 1101 frames, so 8 + 4 chunks.
    let frames = samples.len() / 160 + 1;
    let chunks: usize = (0..frames)
        .step_by(800)
        .map(|start| (frames - start).min(800).div_ceil(100))
        .sum();
    assert_eq!(audio_ids.len(), chunks * 13);
    assert!(
        audio_ids
            .iter()
            .all(|&id| id as usize >= weights.vocabulary_size())
    );
    // Feature rows past the encoded ones stay out of reach.
    assert!(session.infer(vec![audio_ids[0] + 1000]).is_err());
    let mut prompt = vec![special("<|im_start|>")?];
    prompt.extend(tokenizer.encode("user\n")?);
    prompt.push(special("<|audio_start|>")?);
    prompt.extend(&audio_ids);
    prompt.push(special("<|audio_end|>")?);
    prompt.push(special("<|im_end|>")?);
    prompt.extend(tokenizer.encode("\n")?);
    prompt.push(special("<|im_start|>")?);
    prompt.extend(tokenizer.encode("assistant\n")?);
    let stop = special("<|im_end|>")?;
    let mut generated = Vec::new();
    runtime.block_on(async {
        let mut inference = session.infer(prompt)?;
        while let Some(token) = inference.next().await? {
            if token == stop || generated.len() >= 96 {
                break;
            }
            generated.push(token);
        }
        anyhow::Ok(())
    })?;
    let text = tokenizer.decode(&generated)?;
    eprintln!(
        "qwen3-asr (encode {encoded:?}, {} tokens): {text}",
        generated.len()
    );
    let (header, transcript) = text
        .split_once("<asr_text>")
        .context("missing <asr_text> marker")?;
    assert!(header.contains("English"), "unexpected header {header:?}");
    check(transcript);
    // Resetting forgets the audio rows along with the conversation.
    session.reset()?;
    assert!(session.infer(vec![audio_ids[0]]).is_err());
    Ok(())
}
