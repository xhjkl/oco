//! Teacher-forced agreement: feed a token sequence and count how often Oco's greedy prediction
//! matches the token that actually follows. Usage: agree MODEL.gguf PROMPT_LEN ID ID ID ...
use std::error::Error;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: agree MODEL.gguf PROMPT_LEN IDS...")?;
    let prompt_len: usize = args.next().ok_or("PROMPT_LEN")?.parse()?;
    let ids: Vec<u32> = args.map(|s| s.parse()).collect::<Result<_, _>>()?;
    if ids.len() <= prompt_len + 1 {
        return Err("need tokens after the prompt".into());
    }
    let weights = oco::load(&path)?;
    let tokenizer = weights.tokenizer();
    let mut session = weights.new_session(ids.len() + 1)?;
    let mut predicted = session
        .infer(ids[..prompt_len].to_vec())?
        .next()
        .await?
        .ok_or("no prediction")?;
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
                .ok_or("no prediction")?;
        }
    }
    println!(
        "{}: {agree}/{total} = {:.1}% agreement",
        std::path::Path::new(&path)
            .file_name()
            .unwrap()
            .to_string_lossy(),
        100.0 * agree as f64 / total as f64
    );
    for miss in misses {
        println!("  {miss}");
    }
    Ok(())
}
