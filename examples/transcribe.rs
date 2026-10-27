//! Transcribe a 16 kHz mono WAV file with Whisper or Qwen3-ASR.
//! Usage: transcribe ggml-whisper.bin AUDIO.wav [LANGUAGE]
//!        transcribe QWEN3-ASR.gguf MMPROJ.gguf AUDIO.wav
use anyhow::{Context, Result, ensure};
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "usage: transcribe ggml-whisper.bin AUDIO.wav [LANGUAGE] | transcribe MODEL.gguf MMPROJ.gguf AUDIO.wav";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    ensure!((2..=3).contains(&args.len()), "{USAGE}");
    let is_gguf = |path: &OsString| Path::new(path).extension().is_some_and(|e| e == "gguf");
    if is_gguf(&args[0]) {
        ensure!(args.len() == 3, "{USAGE}");
        qwen(&args[0], &args[1], &args[2]).await
    } else {
        let language = args
            .get(2)
            .map(|l| l.to_str().context("language must be UTF-8"))
            .transpose()?;
        whisper(&args[0], &args[1], language).await
    }
}

async fn whisper(model: &OsString, wav: &OsString, language: Option<&str>) -> Result<()> {
    let samples = oco::audio::wav_samples(&std::fs::read(wav)?)?;
    let loading = Instant::now();
    let whisper = oco::Whisper::load(model)?;
    let mut transcriber = whisper.new_transcriber()?;
    let loaded = loading.elapsed();
    let started = Instant::now();
    let transcript = transcriber.transcribe(&samples, language).await?;
    let elapsed = started.elapsed();
    println!("{}", transcript.text);
    let t = transcript.timings;
    eprintln!(
        "{:.1} s of audio ({}) transcribed in {elapsed:.2?}: spectrogram {:.2?}, encode {:.2?}, decode {:.2?} for {} tokens; load {loaded:.2?}",
        samples.len() as f64 / oco::audio::SAMPLE_RATE as f64,
        transcript.language,
        t.spectrogram,
        t.encoding,
        t.decoding,
        transcript.tokens
    );
    if let Some(report) = oco::profile_report() {
        eprintln!("{report}");
    }
    Ok(())
}

async fn qwen(text: &OsString, audio: &OsString, wav: &OsString) -> Result<()> {
    let samples = oco::audio::wav_samples(&std::fs::read(wav)?)?;
    let weights = oco::load_asr(text, audio)?;
    let tokenizer = weights.tokenizer();
    let special = |name: &str| {
        tokenizer
            .special_token(name)
            .with_context(|| format!("missing {name}"))
    };
    let mut session = weights.new_session(2048.min(weights.context_limit()))?;
    let started = Instant::now();
    let audio_ids = session.embed_audio(&samples).await?;
    let encoded = started.elapsed();
    let mut tokens = vec![special("<|im_start|>")?];
    tokens.extend(tokenizer.encode("user\n")?);
    tokens.push(special("<|audio_start|>")?);
    tokens.extend(&audio_ids);
    tokens.push(special("<|audio_end|>")?);
    tokens.push(special("<|im_end|>")?);
    tokens.extend(tokenizer.encode("\n")?);
    tokens.push(special("<|im_start|>")?);
    tokens.extend(tokenizer.encode("assistant\n")?);
    let stop = special("<|im_end|>")?;
    let mut inference = session.infer(tokens)?;
    let mut decoder = tokenizer.decoder();
    let mut output = io::stdout().lock();
    let mut count = 0;
    let decoding = Instant::now();
    while let Some(token) = inference.next().await? {
        if token == stop {
            break;
        }
        count += 1;
        write!(output, "{}", decoder.push(token)?)?;
        output.flush()?;
    }
    writeln!(output, "{}", decoder.finish())?;
    eprintln!(
        "{:.1} s of audio -> {} audio tokens in {encoded:.2?}; {count} text tokens in {:.2?}",
        samples.len() as f64 / oco::audio::SAMPLE_RATE as f64,
        audio_ids.len(),
        decoding.elapsed()
    );
    Ok(())
}
