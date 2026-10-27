# ⚫️ oco

A small Rust+Metal inference engine for GPT-OSS, dense Qwen3/3.5/3.8, and Nemotron-H MoE
(Nemotron 3.5 Lightning) GGUF models on Apple Silicon, plus speech recognition with Whisper
and Qwen3-ASR.

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

The snippets above perform text completion. The runnable examples add Qwen/Nemotron/GPT-OSS
question framing in [examples/common](examples/common/mod.rs):
`cargo run --release --example infer -- MODEL.gguf "How can I inspect a large log file?"`
or `--example producer`. Set `QWEN_PATH`, `GPT_OSS_PATH`, and `NEMOTRON_PATH` to your GGUF
files and run `cargo test --release -- --nocapture` for the model tests: `tests/inference.rs`
checks tokenizer round trips and generation mechanics, and `tests/conversation.rs` checks that
each model answers a factual question and issues a well-formed tool call under its own
protocol (ChatML with `<tool_call>` blocks for Qwen and Nemotron, Harmony commentary channels
for GPT-OSS). `cargo test --lib` runs the numerical and safety checks without model files.

Supported checkpoints: GPT-OSS (`gpt-oss`, MXFP4 experts), Qwen3 dense (`qwen3` and the
`qwen3vl` text stack), Qwen3.5/3.8 dense (`qwen35`,
gated attention and gated delta layers, Q4/Q5/Q6/Q8 encodings), and Nemotron-H MoE
(`nemotron_h_moe`: Mamba-2 blocks, position-free attention, sigmoid-routed relu² experts with a
shared expert, Q4_0 experts with Q8_0 dense weights, Tekken tokenizer). Nemotron prefill runs
the routed experts as gathered tiles on Metal 4 tensor operations; decode uses gathered
single-row products with relu² folded into the down projections. Activations are BF16 with
FP32 accumulation; the final vocabulary scores and Nemotron's recurrent time steps stay in FP32.

`cargo run --release --example bench -- MODEL.gguf [PROMPT_TOKENS] [GENERATED] [REPEATS]`
reports prefill and decode throughput, comparable to llama-bench's pp/tg rows. Set
`OCO_PROFILE=1` to print per-kernel GPU time (precise timestamps, so absolute numbers run
slower than a free pass; the attribution is what matters). `--example agree` measures
teacher-forced agreement with another engine's greedy continuation, and `--example tokens`
prints token IDs for cross-checking tokenizers.

## Speech recognition

```rust,no_run
# async fn example() -> anyhow::Result<()> {
let samples = oco::audio::wav_samples(&std::fs::read("speech.wav")?)?; // 16 kHz mono
let whisper = oco::Whisper::load("ggml-large-v3-turbo.bin")?;
let mut transcriber = whisper.new_transcriber()?;
let transcript = transcriber.transcribe(&samples, None).await?; // or Some("en")
println!("[{}] {}", transcript.language, transcript.text);
# Ok(())
# }
```

`oco::Whisper` reads whisper.cpp's `ggml-*.bin` checkpoints (tiny through large-v3 and
large-v3-turbo). Transcription is greedy without timestamps, one 30-second window at a time,
with the language detected from the first window unless given. The log-mel front end runs on
the host and matches whisper.cpp's; the encoder, cross-attention keys, and decoder run on Metal.

Qwen3-ASR is a Qwen3 language model with a companion audio encoder, both as GGUF files from
ggml-org (`Qwen3-ASR-*.gguf` and `mmproj-Qwen3-ASR-*.gguf`). Load them together and splice the
audio into an ordinary ChatML prompt; the model answers with `language English<asr_text>...`:

```rust,no_run
# async fn example() -> anyhow::Result<()> {
let samples = oco::audio::wav_samples(&std::fs::read("speech.wav")?)?;
let weights = oco::load_asr("Qwen3-ASR-1.7B-Q8_0.gguf", "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")?;
let tokenizer = weights.tokenizer();
let mut session = weights.new_session(2048)?;
let audio = session.embed_audio(&samples).await?; // IDs past the vocabulary select audio rows
let special = |name| tokenizer.special_token(name).unwrap();
let mut prompt = vec![special("<|im_start|>")];
prompt.extend(tokenizer.encode("user\n")?);
prompt.push(special("<|audio_start|>"));
prompt.extend(audio);
prompt.push(special("<|audio_end|>"));
prompt.push(special("<|im_end|>"));
prompt.extend(tokenizer.encode("\n")?);
prompt.push(special("<|im_start|>"));
prompt.extend(tokenizer.encode("assistant\n")?);
let mut inference = session.infer(prompt)?;
# Ok(())
# }
```

`cargo run --release --example transcribe -- ggml-large-v3-turbo.bin speech.wav [LANGUAGE]` or
`-- Qwen3-ASR.gguf mmproj.gguf speech.wav` transcribes a file and prints timings. Set
`WHISPER_PATH`, `QWEN3_ASR_PATH`, and `QWEN3_ASR_MMPROJ` to run `tests/transcribe.rs`, which
transcribes the JFK sample in `tests/data/jfk.wav` (from whisper.cpp) with both models.

`oco::inspect(path)` reads checkpoint identity, parameter count, and file size without
allocating GPU resources. It validates GGUF storage and supported encodings; full
model topology and tokenizer validation still happen during `oco::load`.

Building requires the Xcode Metal 4.1 compiler. If `xcrun` cannot find an installed
component, set `OCO_METAL_TOOLCHAIN` to its `Metal.xctoolchain` directory.
