# ⚫️ oco

Oco runs Qwen, gpt-oss, and Whisper on Apple Silicon, in Rust and Metal, and very little else.
Serving so few architectures is the point: every kernel is written for the model it runs.

From the same GGUF, it generates gpt-oss-20b tokens 22% faster than llama.cpp and Qwen3.8-27B
tokens 5% faster; it transcribes long recordings 11% faster than whisper.cpp at the same
accuracy. An agent's next turn costs only its new tokens: a session rewinds to where the
conversation diverged, and a saved prompt reopens in milliseconds instead of being read again.

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

`infer` appends to what a session already holds. `rewind` returns to an earlier position,
`checkpoint` keeps that cheap for recurrent layers, and `save` with `open_session` carries a
prefilled prompt across processes. The tokenizer and special tokens come from the checkpoint.

Runnable programs live in [examples](examples): `cargo run --release --example NAME -- ARGS`.
