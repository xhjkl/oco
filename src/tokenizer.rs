use crate::gguf::Gguf;
use anyhow::{Context, Result, bail, ensure};
use tokenizers::{
    SplitDelimiterBehavior,
    models::bpe::{BPE, Vocab},
    normalizers::unicode::NFC,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

/// Token classification declared by the checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

/// Checkpoint vocabulary and text codec; conversation framing stays with the caller.
pub struct Tokenizer {
    encoder: tokenizers::Tokenizer,
    kinds: Vec<TokenKind>,
    bos: Option<u32>,
    eos: Option<u32>,
}

/// Incremental UTF-8 decoding, retaining only an unfinished character between pushes.
pub struct Decoder<'a> {
    tokenizer: &'a Tokenizer,
    pending: Vec<u8>,
}

impl Tokenizer {
    pub(crate) fn load(file: &Gguf, vocabulary: usize) -> Result<Self> {
        ensure!(
            file.text("tokenizer.ggml.model")? == "gpt2",
            "unsupported tokenizer model"
        );
        let pre = file.text("tokenizer.ggml.pre")?;
        let pattern = match pre {
            "gpt-4o" => {
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            "qwen35" => {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            "qwen2" => {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            _ => bail!("unsupported tokenizer preprocessor {pre}"),
        };
        let tokens = file.strings("tokenizer.ggml.tokens")?;
        let kinds = file.integers("tokenizer.ggml.token_type")?;
        ensure!(
            tokens.len() == vocabulary && kinds.len() == vocabulary,
            "tokenizer vocabulary does not match model weights"
        );
        let kinds = kinds
            .into_iter()
            .map(|kind| {
                Ok(match kind {
                    1 => TokenKind::Normal,
                    2 => TokenKind::Unknown,
                    3 => TokenKind::Control,
                    4 => TokenKind::UserDefined,
                    5 => TokenKind::Unused,
                    6 => TokenKind::Byte,
                    _ => bail!("invalid tokenizer token kind {kind}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut vocab = Vocab::with_capacity(vocabulary);
        for (id, token) in tokens.into_iter().enumerate() {
            ensure!(!token.is_empty(), "empty vocabulary token");
            let previous = vocab.insert(token.to_owned(), id as u32);
            ensure!(previous.is_none(), "duplicate vocabulary token {token}");
        }
        // Requiring the complete byte alphabet; BPE otherwise silently drops unknown characters.
        for character in ByteLevel::alphabet() {
            let id = vocab
                .get(&character.to_string())
                .context("incomplete byte vocabulary")?;
            ensure!(
                kinds[*id as usize] == TokenKind::Normal,
                "invalid byte vocabulary kind"
            );
        }
        let merges = file.strings("tokenizer.ggml.merges")?;
        let merges = merges
            .into_iter()
            .map(|merge| {
                let (left, right) = merge.split_once(' ').context("invalid BPE merge")?;
                ensure!(
                    !left.is_empty() && !right.is_empty() && !right.contains(' '),
                    "invalid BPE merge"
                );
                Ok((left.to_owned(), right.to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(anyhow::Error::from_boxed)?;
        let split = Split::new(
            SplitPattern::Regex(pattern.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(anyhow::Error::from_boxed)?;
        let mut encoder = tokenizers::Tokenizer::new(model);
        encoder.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, false, false).into(),
        ])));
        if pre == "qwen35" {
            encoder
                .with_normalizer(Some(NFC))
                .map_err(anyhow::Error::from_boxed)?;
        }
        // Leaving added-token recognition disabled; callers insert control IDs explicitly.
        let boundary = |key| -> Result<Option<u32>> {
            let id = file.optional_unsigned(key)?;
            let Some(id) = id else { return Ok(None) };
            ensure!(id < vocabulary as u64, "out-of-range {key}");
            Ok(Some(id as u32))
        };
        let bos = boundary("tokenizer.ggml.bos_token_id")?;
        let eos = boundary("tokenizer.ggml.eos_token_id")?;
        Ok(Self {
            encoder,
            kinds,
            bos,
            eos,
        })
    }

    /// Encode ordinary text without inserting BOS/EOS or recognizing control-token spellings.
    /// Qwen35 applies its tokenizer's NFC normalization.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self
            .encoder
            .encode_fast(text, false)
            .map_err(anyhow::Error::from_boxed)?;
        Ok(encoded.get_ids().to_vec())
    }

    /// Decode IDs, preserving control-token spellings and replacing malformed or incomplete UTF-8.
    /// Use [`Self::decoder`] when tokens arrive incrementally.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &token in tokens {
            self.append_bytes(token, &mut bytes)?;
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn decoder(&self) -> Decoder<'_> {
        Decoder {
            tokenizer: self,
            pending: Vec::new(),
        }
    }

    /// Look up a control or user-defined marker by its exact checkpoint spelling.
    pub fn special_token(&self, spelling: &str) -> Option<u32> {
        let id = self.encoder.token_to_id(spelling)?;
        match self.token_kind(id)? {
            TokenKind::Control | TokenKind::UserDefined => Some(id),
            _ => None,
        }
    }

    pub fn token_kind(&self, id: u32) -> Option<TokenKind> {
        self.kinds.get(id as usize).copied()
    }

    /// Return the checkpoint's beginning-of-sequence marker, without inserting it automatically.
    pub fn bos_token(&self) -> Option<u32> {
        self.bos
    }

    /// Return the declared end-of-sequence marker; protocol-specific stop policy belongs to the caller.
    pub fn eos_token(&self) -> Option<u32> {
        self.eos
    }

    fn append_bytes(&self, id: u32, output: &mut Vec<u8>) -> Result<()> {
        let kind = self.token_kind(id).context("token ID outside vocabulary")?;
        let token = self
            .encoder
            .id_to_token(id)
            .context("missing vocabulary token")?;
        match kind {
            TokenKind::Normal => {
                for character in token.chars() {
                    // Reversing the byte-level alphabet, whose extra codepoints represent gaps in Latin-1.
                    let byte = match character as u32 {
                        c @ (33..=126 | 161..=172 | 174..=255) => c as u8,
                        c @ 256..=288 => (c - 256) as u8,
                        c @ 289..=322 => (c - 289 + 127) as u8,
                        323 => 173,
                        _ => bail!("invalid byte-level vocabulary token {id}"),
                    };
                    output.push(byte);
                }
            }
            TokenKind::Byte => {
                let hex = token
                    .strip_prefix("<0x")
                    .and_then(|s| s.strip_suffix('>'))
                    .context("invalid byte token")?;
                let byte = u8::from_str_radix(hex, 16)?;
                output.push(byte);
            }
            _ => output.extend_from_slice(token.as_bytes()),
        }
        Ok(())
    }
}

impl Decoder<'_> {
    /// Append a token and return newly completed text, retaining any unfinished UTF-8 suffix.
    /// Control tokens remain visible; invalid byte sequences become replacement characters.
    pub fn push(&mut self, token: u32) -> Result<String> {
        self.tokenizer.append_bytes(token, &mut self.pending)?;
        let mut text = String::new();
        let mut consumed = 0;
        while consumed < self.pending.len() {
            let bytes = &self.pending[consumed..];
            match std::str::from_utf8(bytes) {
                Err(error) => {
                    let valid = error.valid_up_to();
                    text.push_str(std::str::from_utf8(&bytes[..valid])?);
                    consumed += valid;
                    let Some(invalid) = error.error_len() else {
                        break;
                    };
                    text.push('\u{fffd}');
                    consumed += invalid;
                }
                Ok(rest) => {
                    text.push_str(rest);
                    consumed = self.pending.len();
                }
            }
        }
        self.pending.drain(..consumed);
        Ok(text)
    }

    /// Finish decoding, replacing a truncated final character if one remains.
    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

#[cfg(test)]
mod tests;
