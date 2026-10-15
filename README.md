# ⚫️ oco

A small Rust+Metal inference engine for GPT-OSS and dense Qwen3.5/3.8 GGUF
models on Apple Silicon.

```rust,no_run
# async fn example() -> anyhow::Result<()> {
let weights = oco::load("model.gguf")?;
let tokenizer = weights.tokenizer();
let prompt = tokenizer.encode("Three useful shell commands for inspecting log files are:\n1.")?;
let mut decoder = tokenizer.decoder();
let mut session = weights.new_session(2048)?;
let mut inference = session.infer(prompt)?;
for _ in 0..128 {
    let Some(token) = inference.next().await? else { break };
    if Some(token) == tokenizer.eos_token() { break; }
    print!("{}", decoder.push(token)?);
}
println!("{}", decoder.finish());
# Ok(())
# }
```

Use `into_producer()` for a token stream with backpressure. Drive the producer
and consumer together on your runtime:

```rust,no_run
# async fn example() -> anyhow::Result<()> {
use futures_lite::StreamExt;

let weights = oco::load("model.gguf")?;
let tokenizer = weights.tokenizer();
let prompt = tokenizer.encode("Three useful shell commands for inspecting log files are:\n1.")?;
let mut decoder = tokenizer.decoder();
let mut session = weights.new_session(2048)?;
let (produce, mut tokens) = session.infer(prompt)?.into_producer();
let consume = async move {
    for _ in 0..128 {
        let Some(token) = tokens.next().await else { break };
        let token = token?;
        if Some(token) == tokenizer.eos_token() { break; }
        print!("{}", decoder.push(token)?);
    }
    println!("{}", decoder.finish());
    anyhow::Ok(())
};
let ((), result) = tokio::join!(produce, consume);
result?;
# Ok(())
# }
```

Oco generates tokens greedily. The tokenizer comes from the GGUF: `encode` handles
ordinary text, `decode` handles complete token sequences, and `decoder` streams text.
`special_token("<|im_end|>")`, `token_kind`, and BOS/EOS metadata let callers build
their own conversation formatting and stop policy. Sessions retain context between
input suffixes and can be cleared with `reset()`.

The snippets above perform text completion. The runnable examples add Qwen/GPT-OSS
question framing in [examples/common](examples/common/mod.rs):
`cargo run --release --example infer -- MODEL.gguf "How can I inspect a large log file?"`
or `--example producer`. Set `QWEN_PATH` and `GPT_OSS_PATH` to your GGUF files and run
`cargo test -- --nocapture` for both model tests and elapsed times;
`cargo test --lib` runs the numerical and safety checks without model files.
