//! Whisper's token table and the special IDs whisper.cpp derives from the vocabulary size.
use anyhow::{Result, ensure};

/// Language codes in Whisper's token order; the token for index `i` is `sot + 1 + i`.
pub const LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Byte spellings of the text tokens plus the control IDs the decoder is steered with.
pub struct Vocabulary {
    tokens: Vec<Vec<u8>>,
    pub size: usize,
    pub end_of_text: u32,
    pub start_of_transcript: u32,
    pub translate: u32,
    pub transcribe: u32,
    pub speaker_turn: u32,
    pub previous: u32,
    pub no_speech: u32,
    pub no_timestamps: u32,
    /// First timestamp token; every ID from here on is a timestamp.
    pub timestamps: u32,
    pub languages: usize,
    pub space: Option<u32>,
}

impl Vocabulary {
    /// Lay out the specials the way whisper.cpp does for the checkpoint's vocabulary size.
    pub fn new(tokens: Vec<Vec<u8>>, size: usize) -> Result<Self> {
        ensure!(
            tokens.len() <= size && size >= 51_864,
            "unexpected Whisper vocabulary size {size}"
        );
        let multilingual = size >= 51_865;
        let shift = u32::from(multilingual);
        let languages = if multilingual { size - 51_766 } else { 0 };
        ensure!(
            languages <= LANGUAGES.len(),
            "vocabulary declares {languages} languages"
        );
        // Multilingual models insert the language tokens after the start token.
        let dt = if multilingual {
            (languages as u32).saturating_sub(98)
        } else {
            0
        };
        let timestamps = 50_363 + dt;
        ensure!(
            (timestamps as usize) < size,
            "vocabulary has no room for timestamp tokens"
        );
        let space = tokens.iter().position(|t| t == b" ").map(|id| id as u32);
        Ok(Self {
            tokens,
            size,
            end_of_text: 50_256 + shift,
            start_of_transcript: 50_257 + shift,
            translate: 50_357 + dt,
            transcribe: 50_358 + dt,
            speaker_turn: 50_359 + dt,
            previous: 50_360 + dt,
            no_speech: 50_361 + dt,
            no_timestamps: 50_362 + dt,
            timestamps,
            languages,
            space,
        })
    }

    /// Token for an ISO language code, when the model knows it.
    pub fn language(&self, code: &str) -> Option<u32> {
        let index = LANGUAGES[..self.languages]
            .iter()
            .position(|&c| c == code)?;
        Some(self.start_of_transcript + 1 + index as u32)
    }

    /// Language code of a language token.
    pub fn language_code(&self, token: u32) -> Option<&'static str> {
        let index = token.checked_sub(self.start_of_transcript + 1)? as usize;
        LANGUAGES[..self.languages].get(index).copied()
    }

    /// Text tokens are the IDs below the end-of-text marker.
    pub fn is_text(&self, token: u32) -> bool {
        token < self.end_of_text
    }

    /// Append a text token's bytes; control tokens contribute nothing.
    pub fn append(&self, token: u32, out: &mut Vec<u8>) {
        if let Some(bytes) = self
            .tokens
            .get(token as usize)
            .filter(|_| self.is_text(token))
        {
            out.extend_from_slice(bytes);
        }
    }
}
