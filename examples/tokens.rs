//! Print the token IDs Oco assigns to a text, for cross-checking against other tokenizers.
//! Usage: tokens MODEL.gguf TEXT
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: tokens MODEL.gguf TEXT")?;
    let text = args.next().ok_or("usage: tokens MODEL.gguf TEXT")?;
    let weights = oco::load(path)?;
    let ids = weights.tokenizer().encode(&text)?;
    let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    println!("{}", ids.join(" "));
    Ok(())
}
