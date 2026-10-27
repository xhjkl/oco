//! Conversation-level acceptance: each checkpoint answers a factual question and issues a
//! well-formed tool call under its own protocol. Set QWEN_PATH, GPT_OSS_PATH, and
//! NEMOTRON_PATH to GGUF files; tests without a path are skipped.
mod common;
use anyhow::Result;
use common::{Family, WEATHER, family, generate, prompt};
use std::sync::Mutex;

// These models do not fit in GPU memory together, even when the test harness runs in parallel.
static GPU: Mutex<()> = Mutex::new(());

#[test]
fn qwen_answers_and_calls_tools() -> Result<()> {
    converse("QWEN_PATH")
}

#[test]
fn gpt_oss_answers_and_calls_tools() -> Result<()> {
    converse("GPT_OSS_PATH")
}

#[test]
fn nemotron_answers_and_calls_tools() -> Result<()> {
    converse("NEMOTRON_PATH")
}

fn converse(variable: &str) -> Result<()> {
    let Some(path) = std::env::var_os(variable) else {
        eprintln!("{variable} unset; skipping");
        return Ok(());
    };
    let _gpu = GPU.lock().unwrap_or_else(|error| error.into_inner());
    let weights = oco::load(path)?;
    let tokenizer = weights.tokenizer();
    let family = family(tokenizer);

    let question = prompt(
        tokenizer,
        "You are a concise assistant.",
        "What is the capital of France? Reply with just the city name.",
        None,
    )?;
    let answer = generate(&weights, &question, 160)?;
    eprintln!(
        "{variable} answer ({:?}, prefill {:.2?}, {} tokens in {:.2?}): {}",
        family,
        answer.prefill,
        answer.tokens.len(),
        answer.decode,
        answer.text.trim()
    );
    assert!(
        final_text(family, &answer.text)
            .to_lowercase()
            .contains("paris"),
        "expected Paris in {:?}",
        answer.text
    );

    // Greedy decoding of these checkpoints answers weather questions from memory unless the
    // system prompt directs them to the tool; all three follow this instruction.
    let request = prompt(
        tokenizer,
        "You are a helpful assistant. For any question about the weather, you must call the get_weather function instead of answering from memory.",
        "What's the weather like in Paris right now?",
        Some(&WEATHER),
    )?;
    let call = generate(&weights, &request, 200)?;
    eprintln!(
        "{variable} tool call ({} tokens in {:.2?}): {}",
        call.tokens.len(),
        call.decode,
        call.text.trim()
    );
    // Only Nemotron reasons before its call; Harmony's commentary channel is the call itself.
    let text = match family {
        Family::Nemotron => final_text(family, &call.text),
        _ => call.text.clone(),
    };
    let text = text.as_str();
    match family {
        Family::Qwen | Family::Nemotron => {
            assert!(text.contains("<tool_call>"), "no <tool_call> in {text:?}");
            assert!(
                text.contains("<function=get_weather>"),
                "wrong function in {text:?}"
            );
            assert!(
                text.contains("<parameter=city>"),
                "no city parameter in {text:?}"
            );
            assert!(
                text.contains("</function>"),
                "unterminated call in {text:?}"
            );
            let last = call.tokens.last().copied();
            assert_eq!(
                last,
                tokenizer.special_token("<|im_end|>"),
                "call must end the turn"
            );
        }
        Family::GptOss => {
            assert!(
                text.contains("commentary to=functions.get_weather"),
                "no commentary call in {text:?}"
            );
            assert!(text.contains("\"city\""), "no city argument in {text:?}");
            let last = call.tokens.last().copied();
            assert_eq!(
                last,
                tokenizer.special_token("<|call|>"),
                "call must hand over to the tool"
            );
        }
    }
    assert!(text.contains("Paris"), "no Paris in {text:?}");
    Ok(())
}

/// Strip reasoning and channel preambles so the assertion reads the user-visible answer.
fn final_text(family: Family, text: &str) -> String {
    match family {
        Family::GptOss => text.rsplit("<|message|>").next().unwrap_or(text).to_owned(),
        Family::Nemotron => text.rsplit("</think>").next().unwrap_or(text).to_owned(),
        Family::Qwen => text.to_owned(),
    }
}
