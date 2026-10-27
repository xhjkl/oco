//! Teacher-forced agreement: feed a token sequence and count how often Oco's greedy prediction
//! matches the token that actually follows. Usage: agree MODEL.gguf PROMPT_LEN ID ID ID ...
use anyhow::{Context, Result, ensure};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: agree MODEL.gguf PROMPT_LEN IDS...")?;
    let prompt_len: usize = args.next().context("PROMPT_LEN")?.parse()?;
    let ids: Vec<u32> = args.map(|s| s.parse()).collect::<Result<_, _>>()?;
    ensure!(ids.len() > prompt_len + 1, "need tokens after the prompt");
    let weights = oco::load(&path)?;
    let tokenizer = weights.tokenizer();
    let mut session = weights.new_session(ids.len() + 1)?;
    let mut predicted = session
        .infer(ids[..prompt_len].to_vec())?
        .next()
        .await?
        .context("no prediction")?;
    let (mut agree, mut total) = (0usize, 0usize);
    let mut misses = Vec::new();
    for i in prompt_len..ids.len() {
        total += 1;
        if predicted == ids[i] {
            agree += 1;
        } else if misses.len() < 12 {
            misses.push(format!(
                "@{i} got {:?} want {:?}",
                tokenizer.decode(&[predicted])?,
                tokenizer.decode(&[ids[i]])?
            ));
        }
        if i + 1 < ids.len() {
            predicted = session
                .infer(vec![ids[i]])?
                .next()
                .await?
                .context("no prediction")?;
        }
    }
    println!(
        "{}: {agree}/{total} = {:.1}% agreement",
        std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
        100.0 * agree as f64 / total as f64
    );
    for miss in misses {
        println!("  {miss}");
    }
    Ok(())
}
