use anyhow::{Context, Result, ensure};
use oco::Tokenizer;
use std::path::PathBuf;

pub fn arguments() -> Result<(PathBuf, String)> {
    let mut args = std::env::args_os();
    args.next();
    let path = args
        .next()
        .context("usage: <example> MODEL.gguf [QUESTION]")?;
    let question = args.next();
    let extra = args.next();
    ensure!(extra.is_none(), "quote the question as one argument");
    let question = match &question {
        None => {
            "Explain three practical ways to inspect a large log file. Keep it under 100 words."
        }
        Some(question) => question.to_str().context("question must be UTF-8")?,
    };
    Ok((path.into(), question.to_owned()))
}

/// Frame one user question and open an assistant answer in the checkpoint's protocol.
/// This example owns conversation formatting; Oco only encodes text and looks up markers.
pub fn prompt(tokenizer: &Tokenizer, question: &str) -> Result<Vec<u32>> {
    let special = |name| {
        tokenizer
            .special_token(name)
            .with_context(|| format!("missing {name}"))
    };
    let mut tokens = Vec::new();
    if let Some(start) = tokenizer.special_token("<|im_start|>") {
        tokens.push(start);
        tokens.extend(tokenizer.encode("user\n")?);
        tokens.extend(tokenizer.encode(question)?);
        tokens.push(special("<|im_end|>")?);
        tokens.extend(tokenizer.encode("\n")?);
        tokens.push(start);
        tokens.extend(tokenizer.encode("assistant\n")?);
        // Opening a completed thinking section requests a direct answer. Qwen pads the
        // section with blank lines; Nemotron (recognizable by Tekken's markers) keeps it empty.
        let padded = tokenizer.special_token("[INST]").is_none();
        tokens.push(special("<think>")?);
        if padded {
            tokens.extend(tokenizer.encode("\n\n")?);
        }
        tokens.push(special("</think>")?);
        if padded {
            tokens.extend(tokenizer.encode("\n\n")?);
        }
        return Ok(tokens);
    }

    tokens.push(special("<|start|>")?);
    tokens.extend(tokenizer.encode("user")?);
    tokens.push(special("<|message|>")?);
    tokens.extend(tokenizer.encode(question)?);
    tokens.push(special("<|end|>")?);
    tokens.push(special("<|start|>")?);
    tokens.extend(tokenizer.encode("assistant")?);
    tokens.push(special("<|channel|>")?);
    tokens.extend(tokenizer.encode("final")?);
    tokens.push(special("<|message|>")?);
    Ok(tokens)
}
