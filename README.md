# ⚫️ oco

`oco` is an inference engine for Apple Silicon, written in Rust and Metal. It generates text
with Qwen, gpt-oss, and Nemotron, transcribes speech with Whisper and Qwen3-ASR, and runs
nothing else. Supporting so few kinds of model is by design: it lets every kernel be written
for a layer and a weight format that one of these models actually uses.

From the same GGUF file, `oco` generates tokens 22% faster than llama.cpp with gpt-oss-20b
and 5% faster with Qwen3.8-27B. It transcribes long recordings 11% faster than whisper.cpp
at the same accuracy.

```rust,no_run
# async fn example() -> oco::Result<()> {
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

A session can be rewound, or saved to disk and restored later, so a long prompt need not be
processed twice.

Start exploring with the [examples](examples):

```sh
cargo run --release --example infer -- model.gguf "Why is the sky blue?"
```
