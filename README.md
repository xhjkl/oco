# ⚫️ oco

Oco is an inference engine written in Rust and Metal for Apple Silicon. It generates
text greedily and transcribes speech with these model architectures:

| Models | Checkpoints |
| --- | --- |
| GPT-OSS | GGUF `gpt-oss`, with MXFP4 experts |
| Qwen3 dense | GGUF `qwen3` and the `qwen3vl` text stack |
| Qwen3.5/3.8 dense | GGUF `qwen35`, with gated attention and gated delta layers |
| Nemotron-H MoE (Nemotron 3.5 Lightning) | GGUF `nemotron_h_moe`, with Mamba-2, attention, and routed experts |
| Whisper, tiny through large-v3 and large-v3-turbo | whisper.cpp `ggml-*.bin` |
| Qwen3-ASR | Text GGUF paired with its `mmproj` audio encoder GGUF |

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

Sessions retain context between input suffixes; `reset()` starts a fresh sequence.
The tokenizer and special tokens come from the checkpoint.

## Examples

Run an example with `cargo run --release --example NAME -- ARGS`:

| Example | Arguments | Purpose |
| --- | --- | --- |
| [infer](examples/infer.rs) | `MODEL.gguf [QUESTION]` | Stream an answer to a question |
| [producer](examples/producer.rs) | `MODEL.gguf [QUESTION]` | Consume a token stream with backpressure |
| [bench](examples/bench.rs) | `MODEL.gguf [PROMPT_TOKENS] [GENERATED] [REPEATS]` | Measure prefill and decode throughput |
| [agree](examples/agree.rs) | `MODEL.gguf PROMPT_LEN IDS...` | Measure agreement with a reference token sequence |
| [tokens](examples/tokens.rs) | `MODEL.gguf TEXT` | Print token IDs |
| [transcribe](examples/transcribe.rs) | `ggml-MODEL.bin speech.wav [LANGUAGE]` | Transcribe with Whisper |
| [transcribe](examples/transcribe.rs) | `Qwen3-ASR.gguf mmproj.gguf speech.wav` | Transcribe with Qwen3-ASR |

The text examples provide conversation framing in [examples/common](examples/common/mod.rs).
Speech examples take 16 kHz WAV audio.
