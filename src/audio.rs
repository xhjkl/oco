//! Log-mel spectrogram front end shared by Whisper and Qwen3-ASR, computed on the host from
//! 16 kHz mono samples. The arithmetic follows whisper.cpp and llama.cpp's audio preprocessor
//! so transcripts match those references: periodic Hann window, 400-point FFT, power spectrum,
//! Slaney-normalized triangular filters, log10, then clamping to eight decades below the peak
//! and rescaling into roughly [-1, 1].
use anyhow::{Context, Result, ensure};

pub const SAMPLE_RATE: usize = 16_000;
pub const FFT_SIZE: usize = 400;
pub const HOP: usize = 160;
/// Frequency bins of the one-sided power spectrum.
pub const BINS: usize = FFT_SIZE / 2 + 1;
/// Frames in one 30-second Whisper window.
pub const WHISPER_FRAMES: usize = 30 * SAMPLE_RATE / HOP;

/// Triangular mel filters over the one-sided spectrum, `mels` rows of [`BINS`] weights.
pub struct MelFilters {
    pub mels: usize,
    pub weights: Vec<f32>,
}

impl MelFilters {
    /// Filters a checkpoint carries (Whisper embeds its librosa-generated bank).
    pub fn new(mels: usize, weights: Vec<f32>) -> Result<Self> {
        ensure!(
            mels > 0 && weights.len() == mels * BINS,
            "mel filter bank does not match {BINS} spectrum bins"
        );
        Ok(Self { mels, weights })
    }

    /// librosa-style bank on the Slaney scale with area normalization, 0 Hz to half the sample rate.
    pub fn slaney(mels: usize) -> Self {
        let hz_to_mel = |f: f64| {
            if f < 1000.0 {
                f * 3.0 / 200.0
            } else {
                15.0 + (f / 1000.0).ln() / (6.4f64.ln() / 27.0)
            }
        };
        let mel_to_hz = |m: f64| {
            if m < 15.0 {
                m * 200.0 / 3.0
            } else {
                1000.0 * ((m - 15.0) * (6.4f64.ln() / 27.0)).exp()
            }
        };
        let high = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
        let edges: Vec<f64> = (0..mels + 2)
            .map(|i| mel_to_hz(high * i as f64 / (mels + 1) as f64))
            .collect();
        let bin_hz = SAMPLE_RATE as f64 / FFT_SIZE as f64;
        let mut weights = vec![0.0; mels * BINS];
        for m in 0..mels {
            let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = 2.0 / (right - left).max(1e-30);
            for k in 0..BINS {
                let f = k as f64 * bin_hz;
                let w = if f >= left && f <= center {
                    (f - left) / (center - left).max(1e-30)
                } else if f > center && f <= right {
                    (right - f) / (right - center).max(1e-30)
                } else {
                    0.0
                };
                weights[m * BINS + k] = (w * norm) as f32;
            }
        }
        Self { mels, weights }
    }
}

/// Normalized log-mel frames, mel-major: `data[mel * frames + frame]`.
pub struct Spectrogram {
    pub mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl Spectrogram {
    /// One frame's mel values in frame-major order for `frames` frames starting at `start`;
    /// frames past the end are zero, which is how both references pad a short window.
    pub fn window(&self, start: usize, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0; frames * self.mels];
        for (i, frame) in out.chunks_exact_mut(self.mels).enumerate() {
            let source = start + i;
            if source < self.frames {
                for (m, value) in frame.iter_mut().enumerate() {
                    *value = self.data[m * self.frames + source];
                }
            }
        }
        out
    }
}

/// Whisper's spectrogram: 200 reflected samples in front, the audio, then 30 seconds of
/// silence, so every 30-second window read from it is complete.
pub fn whisper_log_mel(samples: &[f32], filters: &MelFilters) -> Result<Spectrogram> {
    const FLOOR: f64 = 1e-10;
    ensure!(!samples.is_empty(), "no audio samples");
    let half = FFT_SIZE / 2;
    let mut padded = Vec::with_capacity(samples.len() + 30 * SAMPLE_RATE + 2 * half);
    let reflect = half.min(samples.len() - 1);
    padded.extend(std::iter::repeat_n(0.0, half - reflect));
    padded.extend(samples[1..1 + reflect].iter().rev());
    padded.extend_from_slice(samples);
    padded.resize(samples.len() + 30 * SAMPLE_RATE + 2 * half, 0.0);
    let frames = (padded.len() - FFT_SIZE) / HOP;
    let computed = ((samples.len() + half) / HOP + 1).min(frames);
    Ok(spectrogram(&padded, frames, computed, filters, FLOOR))
}

/// Centered spectrogram with 200 reflected samples at both ends, as Qwen3-ASR's feature
/// extractor produces (one frame per hop, plus one).
pub fn centered_log_mel(samples: &[f32], filters: &MelFilters) -> Result<Spectrogram> {
    // llama.cpp's extractor floors mel energies at 2^-24 before the logarithm.
    const FLOOR: f64 = 5.960_464_477_539_063e-8;
    ensure!(
        samples.len() > FFT_SIZE / 2,
        "audio shorter than one analysis window"
    );
    let half = FFT_SIZE / 2;
    let mut padded = Vec::with_capacity(samples.len() + 2 * half);
    padded.extend(samples[1..=half].iter().rev());
    padded.extend_from_slice(samples);
    let tail = samples.len() - 1;
    padded.extend(samples[tail - half..tail].iter().rev());
    let frames = (padded.len() - FFT_SIZE) / HOP + 1;
    let kept = (samples.len() / HOP + 1).min(frames);
    let full = spectrogram(&padded, frames, frames, filters, FLOOR);
    let mut data = Vec::with_capacity(kept * full.mels);
    for m in 0..full.mels {
        data.extend_from_slice(&full.data[m * frames..m * frames + kept]);
    }
    Ok(Spectrogram {
        mels: full.mels,
        frames: kept,
        data,
    })
}

/// Power-spectrum frames projected onto the filters, in log10, then normalized like Whisper.
/// Frames at or past `computed` hold the floor instead of being analyzed.
fn spectrogram(
    padded: &[f32],
    frames: usize,
    computed: usize,
    filters: &MelFilters,
    floor: f64,
) -> Spectrogram {
    let mels = filters.mels;
    let fft = Fft::new();
    let mut data = vec![floor.log10() as f32; mels * frames];
    // Frames are independent; spread them over the available cores in contiguous spans.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let span = computed.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for first in (0..computed).step_by(span) {
            let last = (first + span).min(computed);
            let (fft, filters) = (&fft, filters);
            handles.push(scope.spawn(move || {
                let mut frame = Frame::new();
                let mut out = vec![0.0f32; (last - first) * mels];
                for (i, values) in out.chunks_exact_mut(mels).enumerate() {
                    frame.analyze(fft, padded, (first + i) * HOP, filters, floor, values);
                }
                out
            }));
        }
        for (block, handle) in handles.into_iter().enumerate() {
            let out = handle.join().expect("spectrogram worker panicked");
            let first = block * span;
            for (i, values) in out.chunks_exact(mels).enumerate() {
                for (m, &value) in values.iter().enumerate() {
                    data[m * frames + first + i] = value;
                }
            }
        }
    });
    let peak = data.iter().copied().fold(f32::MIN, f32::max) - 8.0;
    for value in &mut data {
        *value = (value.max(peak) + 4.0) / 4.0;
    }
    Spectrogram { mels, frames, data }
}

/// Twiddles for whisper.cpp's transform: even/odd splitting down to a direct 25-point DFT,
/// all angles taken from one table of 400 steps around the circle.
struct Fft {
    hann: Vec<f32>,
    sin: Vec<f32>,
    cos: Vec<f32>,
}

impl Fft {
    fn new() -> Self {
        let angle = |i: usize| 2.0 * std::f64::consts::PI * i as f64 / FFT_SIZE as f64;
        Self {
            hann: (0..FFT_SIZE)
                .map(|i| (0.5 * (1.0 - angle(i).cos())) as f32)
                .collect(),
            sin: (0..FFT_SIZE).map(|i| angle(i).sin() as f32).collect(),
            cos: (0..FFT_SIZE).map(|i| angle(i).cos() as f32).collect(),
        }
    }

    /// Real input of length `n` (400 or one of its even halves down to 25) to interleaved
    /// complex output in `out[..2n]`. `input` holds the values followed by `n` slots of
    /// scratch; `out` needs `8n` slots, the tail serving the recursion.
    fn transform(&self, input: &mut [f32], n: usize, out: &mut [f32]) {
        if n == 1 {
            out[0] = input[0];
            out[1] = 0.0;
            return;
        }
        let step = FFT_SIZE / n;
        if n % 2 == 1 {
            for k in 0..n {
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for (i, &x) in input[..n].iter().enumerate() {
                    let index = (k * i * step) % FFT_SIZE;
                    re += x * self.cos[index];
                    im -= x * self.sin[index];
                }
                out[2 * k] = re;
                out[2 * k + 1] = im;
            }
            return;
        }
        let half = n / 2;
        let (values, scratch) = input.split_at_mut(n);
        let (head, tail) = out.split_at_mut(2 * n);
        for i in 0..half {
            scratch[i] = values[2 * i];
        }
        self.transform(scratch, half, tail);
        for i in 0..half {
            scratch[i] = values[2 * i + 1];
        }
        let (even, rest) = tail.split_at_mut(n);
        self.transform(scratch, half, rest);
        for k in 0..half {
            let (re, im) = (self.cos[k * step], -self.sin[k * step]);
            let (ore, oim) = (rest[2 * k], rest[2 * k + 1]);
            let (tre, tim) = (re * ore - im * oim, re * oim + im * ore);
            head[2 * k] = even[2 * k] + tre;
            head[2 * k + 1] = even[2 * k + 1] + tim;
            head[2 * (k + half)] = even[2 * k] - tre;
            head[2 * (k + half) + 1] = even[2 * k + 1] - tim;
        }
    }
}

/// Per-thread scratch for analyzing one frame at a time.
struct Frame {
    windowed: Vec<f32>,
    spectrum: Vec<f32>,
}

impl Frame {
    fn new() -> Self {
        Self {
            windowed: vec![0.0; FFT_SIZE * 3],
            spectrum: vec![0.0; FFT_SIZE * 8],
        }
    }

    fn analyze(
        &mut self,
        fft: &Fft,
        padded: &[f32],
        offset: usize,
        filters: &MelFilters,
        floor: f64,
        out: &mut [f32],
    ) {
        let available = padded.len().saturating_sub(offset).min(FFT_SIZE);
        for (i, w) in self.windowed[..FFT_SIZE].iter_mut().enumerate() {
            *w = if i < available {
                fft.hann[i] * padded[offset + i]
            } else {
                0.0
            };
        }
        fft.transform(&mut self.windowed, FFT_SIZE, &mut self.spectrum);
        let power = &mut self.windowed[..BINS];
        for (k, p) in power.iter_mut().enumerate() {
            let (re, im) = (self.spectrum[2 * k], self.spectrum[2 * k + 1]);
            *p = re * re + im * im;
        }
        for (m, value) in out.iter_mut().enumerate() {
            let bank = &filters.weights[m * BINS..(m + 1) * BINS];
            let sum: f64 = power
                .iter()
                .zip(bank)
                .map(|(&p, &w)| f64::from(p) * f64::from(w))
                .sum();
            *value = sum.max(floor).log10() as f32;
        }
    }
}

/// Decode a RIFF WAVE file into 16 kHz mono samples in [-1, 1].
/// Accepts 16-bit PCM and 32-bit float, mixing channels down by averaging.
pub fn wav_samples(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a RIFF WAVE file"
    );
    let u16_at = |at: usize| -> Result<u16> {
        let b = bytes.get(at..at + 2).context("truncated WAV header")?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    };
    let u32_at = |at: usize| -> Result<u32> {
        let b = bytes.get(at..at + 4).context("truncated WAV header")?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let mut format = None;
    let mut cursor = 12;
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32_at(cursor + 4)? as usize;
        let body = cursor + 8;
        ensure!(body + size <= bytes.len(), "truncated WAV chunk");
        match id {
            b"fmt " => {
                let tag = u16_at(body)?;
                let channels = u16_at(body + 2)? as usize;
                let rate = u32_at(body + 4)? as usize;
                let bits = u16_at(body + 14)? as usize;
                ensure!(channels > 0, "WAV without channels");
                ensure!(
                    rate == SAMPLE_RATE,
                    "WAV sample rate {rate} Hz; resample to {SAMPLE_RATE} Hz"
                );
                format = Some((tag, channels, bits));
            }
            b"data" => {
                let (tag, channels, bits) =
                    format.context("WAV data chunk before its format chunk")?;
                let data = &bytes[body..body + size];
                let samples: Vec<f32> = match (tag, bits) {
                    (1, 16) => data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| f32::from(i16::from_le_bytes(b)) / 32768.0)
                        .collect(),
                    (3, 32) => data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| f32::from_le_bytes(b))
                        .collect(),
                    _ => anyhow::bail!("unsupported WAV encoding (format {tag}, {bits} bits)"),
                };
                return Ok(samples
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                    .collect());
            }
            _ => {}
        }
        cursor = body + size + (size & 1);
    }
    anyhow::bail!("WAV without a data chunk")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_direct_transform_for_whisper_length() {
        let input: Vec<f32> = (0..FFT_SIZE)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let fft = Fft::new();
        let mut scratch = vec![0.0f32; FFT_SIZE * 3];
        scratch[..FFT_SIZE].copy_from_slice(&input);
        let mut spectrum = vec![0.0f32; FFT_SIZE * 8];
        fft.transform(&mut scratch, FFT_SIZE, &mut spectrum);
        let fast: Vec<(f32, f32)> = (0..FFT_SIZE)
            .map(|k| (spectrum[2 * k], spectrum[2 * k + 1]))
            .collect();
        for k in [0, 1, 17, 199, 200, 201, 399] {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, &x) in input.iter().enumerate() {
                let theta = 2.0 * std::f64::consts::PI * (k * i) as f64 / FFT_SIZE as f64;
                re += f64::from(x) * theta.cos();
                im -= f64::from(x) * theta.sin();
            }
            assert!((f64::from(fast[k].0) - re).abs() < 1e-3, "bin {k} real");
            assert!(
                (f64::from(fast[k].1) - im).abs() < 1e-3,
                "bin {k} imaginary"
            );
        }
    }

    #[test]
    fn slaney_filters_partition_the_spectrum() {
        let filters = MelFilters::slaney(128);
        // Every filter has mass, and each bin above the lowest edge is covered.
        for m in 0..128 {
            let mass: f32 = filters.weights[m * BINS..(m + 1) * BINS].iter().sum();
            assert!(mass > 0.0, "empty filter {m}");
        }
        let coverage: Vec<f32> = (0..BINS)
            .map(|k| (0..128).map(|m| filters.weights[m * BINS + k]).sum())
            .collect();
        assert!(coverage[1..BINS - 1].iter().all(|&c| c > 0.0));
    }

    #[test]
    fn whisper_window_pads_to_thirty_seconds() {
        let samples = vec![0.1f32; SAMPLE_RATE];
        let filters = MelFilters::slaney(80);
        let mel = whisper_log_mel(&samples, &filters).unwrap();
        assert!(mel.frames >= WHISPER_FRAMES);
        let window = mel.window(0, WHISPER_FRAMES);
        assert_eq!(window.len(), WHISPER_FRAMES * 80);
        // Clamping to eight decades below the peak bounds the normalized range to two units.
        let (low, high) = window
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(low.is_finite() && high - low <= 2.0 + 1e-5);
        let centered = centered_log_mel(&samples, &filters).unwrap();
        assert_eq!(centered.frames, SAMPLE_RATE / HOP + 1);
    }
}
