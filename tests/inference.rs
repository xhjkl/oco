use anyhow::{Context, Result};
use futures_lite::StreamExt;
use std::{sync::Mutex, time::Instant};

// These models do not fit in GPU memory together, even when the test harness runs in parallel.
static GPU: Mutex<()> = Mutex::new(());

#[test]
fn qwen_infers() -> Result<()> {
    infer("QWEN_PATH", QWEN_INPUT)
}

#[test]
fn gpt_oss_infers() -> Result<()> {
    infer("GPT_OSS_PATH", GPT_INPUT)
}

fn infer(variable: &str, input: &[u32]) -> Result<()> {
    let path =
        std::env::var_os(variable).with_context(|| format!("set {variable} to a GGUF file"))?;
    let _gpu = GPU.lock().unwrap_or_else(|error| error.into_inner());
    let started = Instant::now();
    let weights = oco::load(path)?;
    let vocabulary = weights.vocabulary_size();
    let tokenizer = weights.tokenizer();
    for text in [
        "",
        "ქართული café 你好 🤍✨",
        "  log\t1970-01-01\r\n",
        "<|im_end|><|return|><think>",
    ] {
        let ids = tokenizer.encode(text)?;
        assert!(
            ids.iter()
                .all(|&id| tokenizer.token_kind(id) == Some(oco::TokenKind::Normal))
        );
        assert_eq!(tokenizer.decode(&ids)?, text);
        let mut decoder = tokenizer.decoder();
        let mut output = String::new();
        for id in ids {
            output.push_str(&decoder.push(id)?);
        }
        output.push_str(&decoder.finish());
        assert_eq!(output, text);
    }
    let eos = tokenizer.eos_token().context("missing EOS")?;
    assert!((eos as usize) < vocabulary);
    assert_eq!(
        tokenizer.special_token(&tokenizer.decode(&[eos])?),
        Some(eos)
    );
    let capacity = input.len() + 127;
    let mut session = weights.new_session(capacity)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let mut output = Vec::with_capacity(128);
        let mut inference = session.infer(input)?;
        for _ in 0..64 {
            let token = inference.next().await?.context("stream ended early")?;
            output.push(token);
        }
        drop(inference);
        assert_eq!(session.position()?, input.len() + 63);

        // Continuing through the producer consumes the last prediction exactly once.
        let (produce, mut tokens) = session.infer(vec![output[63]])?.into_producer();
        let consume = async {
            while let Some(token) = tokens.next().await {
                output.push(token?);
            }
            anyhow::Ok(())
        };
        let ((), result) = tokio::join!(produce, consume);
        result?;
        assert_eq!(output.len(), 128);
        assert_eq!(session.position()?, capacity);
        assert!(output.iter().all(|&token| (token as usize) < vocabulary));
        assert!(
            output.iter().any(|&token| token != output[0]),
            "generation collapsed to one token"
        );
        let mut decoder = tokenizer.decoder();
        let mut text = String::new();
        for &token in &output {
            text.push_str(&decoder.push(token)?);
        }
        text.push_str(&decoder.finish());
        assert_eq!(text, tokenizer.decode(&output)?);

        session.reset()?;
        assert_eq!(session.position()?, 0);
        let mut inference = session.infer(input)?;
        let first = inference.next().await?;
        assert_eq!(
            first,
            Some(output[0]),
            "reset retained stale sequence state"
        );
        anyhow::Ok(())
    })?;
    drop(session);
    drop(weights);
    let elapsed = started.elapsed();
    eprintln!("{variable}: {elapsed:.2?} door-to-door (load, 128 tokens, continuation, reset)");
    Ok(())
}

const QWEN_INPUT: &[u32] = &[
    248045, 846, 198, 814, 20139, 6653, 57154, 440, 264, 6237, 3010, 13, 248046, 198, 248045,
    74455, 198, 248068, 271, 248069, 271,
];
const GPT_INPUT: &[u32] = &[
    200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359, 22203, 656, 7788,
    17527, 558, 87447, 100594, 25, 220, 1323, 19, 12, 3218, 198, 6576, 3521, 25, 220, 1323, 21, 12,
    3114, 12, 2922, 279, 30377, 289, 25, 4465, 279, 2, 13888, 18403, 25, 8450, 11, 1721, 13, 21030,
    2804, 413, 7360, 395, 1753, 3176, 13, 200007, 200006, 1428, 200008, 176289, 9926, 67854, 13,
    200007, 200006, 173781,
];
