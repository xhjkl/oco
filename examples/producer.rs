use anyhow::{Context, Result};
use futures_lite::StreamExt;
use std::io::{self, Write};
mod common;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let (path, question) = common::arguments()?;
    let weights = oco::load(path)?;
    let tokenizer = weights.tokenizer();
    let tokens = common::prompt(tokenizer, &question)?;
    let stops = [tokenizer.eos_token(), tokenizer.special_token("<|call|>")];
    let mut decoder = tokenizer.decoder();
    let capacity = tokens
        .len()
        .checked_add(255)
        .context("context capacity overflow")?;
    let capacity = capacity.min(weights.context_limit());
    let mut session = weights.new_session(capacity)?;
    let (produce, mut tokens) = session.infer(tokens)?.into_producer();
    let consume = async move {
        let mut output = io::stdout().lock();
        for _ in 0..256 {
            let token = tokens.next().await;
            let Some(token) = token else {
                break;
            };
            let token = token?;
            if stops.contains(&Some(token)) {
                break;
            }
            let text = decoder.push(token)?;
            write!(output, "{text}")?;
            output.flush()?;
        }
        writeln!(output, "{}", decoder.finish())?;
        anyhow::Ok(())
    };
    let ((), result) = tokio::join!(produce, consume);
    result
}
