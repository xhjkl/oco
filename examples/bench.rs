//! Prefill and decode throughput for one checkpoint, comparable to llama-bench's pp/tg rows.
//! Usage: bench MODEL.gguf [PROMPT_TOKENS=512] [GENERATED_TOKENS=128] [REPEATS=3]
use anyhow::{Context, Result, ensure};
use std::time::Instant;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: bench MODEL.gguf [PROMPT] [GENERATED] [REPEATS]")?;
    let prompt_tokens: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(512);
    let generated: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(128);
    let repeats: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);
    let started = Instant::now();
    let weights = oco::load(&path)?;
    eprintln!("loaded {path} in {:.2?}", started.elapsed());
    let tokenizer = weights.tokenizer();
    let text = "The quick brown fox jumps over the lazy dog while the engineer inspects log files, counts tokens, and measures memory bandwidth on Apple Silicon. ";
    let mut prompt = Vec::new();
    while prompt.len() < prompt_tokens {
        prompt.extend(tokenizer.encode(text)?);
    }
    prompt.truncate(prompt_tokens);
    ensure!(prompt.len() == prompt_tokens, "prompt too short");
    let mut prefill_rates = Vec::new();
    let mut decode_rates = Vec::new();
    for repeat in 0..repeats {
        let mut session = weights.new_session(prompt_tokens + generated)?;
        let mut inference = session.infer(prompt.clone())?;
        let begin = Instant::now();
        let first = inference.next().await?.context("no prediction")?;
        let prefill = begin.elapsed();
        if let Some(report) = oco::profile_report() {
            eprintln!("--- prefill kernels (run {repeat})\n{report}");
        }
        let begin = Instant::now();
        let mut last = first;
        let mut distinct = false;
        for _ in 1..generated {
            let token = inference.next().await?.context("stream ended early")?;
            distinct |= token != last;
            last = token;
        }
        let decode = begin.elapsed();
        if let Some(report) = oco::profile_report() {
            eprintln!("--- decode kernels (run {repeat}, {} steps)\n{report}", generated - 1);
        }
        let pp = prompt_tokens as f64 / prefill.as_secs_f64();
        let tg = (generated - 1) as f64 / decode.as_secs_f64();
        eprintln!(
            "run {repeat}: pp{prompt_tokens} {pp:8.2} t/s ({prefill:.3?})  tg{generated} {tg:7.2} t/s ({decode:.3?}){}",
            if distinct { "" } else { "  [degenerate output]" }
        );
        prefill_rates.push(pp);
        decode_rates.push(tg);
    }
    let best = |rates: &[f64]| rates.iter().cloned().fold(0.0, f64::max);
    let mean = |rates: &[f64]| rates.iter().sum::<f64>() / rates.len() as f64;
    println!(
        "{}\tpp{prompt_tokens}\tmean {:.2}\tbest {:.2}\ttg{generated}\tmean {:.2}\tbest {:.2}",
        std::path::Path::new(&path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&path),
        mean(&prefill_rates),
        best(&prefill_rates),
        mean(&decode_rates),
        best(&decode_rates)
    );
    Ok(())
}
