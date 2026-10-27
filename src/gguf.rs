use crate::tensor::WeightFormat;
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Cursor, Read},
    ops::Range,
    path::Path,
    sync::Arc,
};

const MAX_ENTRIES: usize = 100_000;
const MAX_VALUES: usize = 10_000_000;

#[derive(Debug)]
pub struct Tensor<'a> {
    pub shape: &'a [usize],
    pub kind: WeightFormat,
    pub offset: usize,
    pub bytes: usize,
}

/// Structural directory entry; execution encoding is checked only for selected operands.
#[derive(Debug)]
pub struct Entry {
    shape: [usize; 4],
    rank: u8,
    code: u32,
    pub offset: usize,
    extent: usize,
}

impl Entry {
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank as usize]
    }

    pub fn select(&self) -> Result<Tensor<'_>> {
        let shape = self.shape();
        let kind = weight_format(self.code)?;
        // Checking the shader's scalar casts independently of the file's declared alignment.
        let alignment = match kind {
            WeightFormat::F32 => 4,
            WeightFormat::Mxfp4 => 1,
            WeightFormat::F16
            | WeightFormat::Bf16
            | WeightFormat::Q4_0
            | WeightFormat::Q4_1
            | WeightFormat::Q8_0
            | WeightFormat::Q4K
            | WeightFormat::Q5K
            | WeightFormat::Q6K => 2,
        };
        ensure!(
            self.offset.is_multiple_of(alignment),
            "unaligned tensor data"
        );
        let (block, size) = kind.block_layout();
        ensure!(
            shape[0].is_multiple_of(block),
            "invalid tensor block alignment"
        );
        let elements = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .context("tensor size overflow")?;
        ensure!(
            elements <= u32::MAX as usize,
            "tensor exceeds the supported Metal index range"
        );
        let bytes = (elements / block)
            .checked_mul(size)
            .context("tensor size overflow")?;
        ensure!(
            bytes <= self.extent,
            "truncated or overlapping selected tensor"
        );
        Ok(Tensor {
            shape,
            kind,
            offset: self.offset,
            bytes,
        })
    }
}

/// Selected metadata's type and validated byte range in the mapped file.
pub struct Field {
    kind: u32,
    bytes: Range<usize>,
}

/// Temporary metadata index over immutable mapped storage; only weight views retain the mapping.
pub struct Gguf {
    pub map: Arc<Mmap>,
    pub meta: HashMap<String, Field>,
    tensor_count: usize,
    tensor_offset: u64,
}

impl Gguf {
    /// Index selected metadata and structurally validate all metadata without materializing values.
    pub fn open(path: &Path, keep: &[&str]) -> Result<Self> {
        let file = File::open(path)?;
        // Retaining immutable pages through the last submitted Metal view.
        let map = unsafe { Mmap::map(&file)? };
        let map = Arc::new(map);
        let mut r = Reader::new(&map);
        let magic = r.u32()?;
        ensure!(magic == 0x46554747, "not a GGUF file");
        let version = r.u32()?;
        ensure!(version == 3, "only GGUF version 3 is supported");
        let nt = r.usize()?;
        let nm = r.usize()?;
        ensure!(nm <= MAX_ENTRIES, "excessive GGUF metadata count");
        ensure!(
            nm <= r.remaining() / 12,
            "truncated GGUF metadata directory"
        );
        let mut meta = HashMap::new();
        let mut keys = HashSet::with_capacity(nm);
        for _ in 0..nm {
            let key = r.string()?;
            ensure!(keys.insert(key), "duplicate GGUF metadata key");
            let kind = r.u32()?;
            let start = r.0.position() as usize;
            r.skip_value(kind, 0)?;
            if keep.contains(&key) {
                meta.insert(
                    key.to_owned(),
                    Field {
                        kind,
                        bytes: start..r.0.position() as usize,
                    },
                );
            }
        }
        let tensor_offset = r.0.position();
        Ok(Self {
            map,
            meta,
            tensor_count: nt,
            tensor_offset,
        })
    }

    /// Check directory structure without imposing execution support on unused tensors.
    /// Borrow names from the mapped file; this index is discarded after operand import.
    pub fn directory(&self) -> Result<HashMap<&str, Entry>> {
        ensure!(
            self.tensor_count <= MAX_ENTRIES,
            "excessive GGUF tensor count"
        );
        let mut r = Reader::new(&self.map);
        r.0.set_position(self.tensor_offset);
        ensure!(
            self.tensor_count <= r.remaining() / 32,
            "truncated GGUF tensor directory"
        );
        let mut tensors = HashMap::with_capacity(self.tensor_count);
        for _ in 0..self.tensor_count {
            let name = r.string()?;
            let dims = r.u32()?;
            ensure!((1..=4).contains(&dims), "invalid dimensions for {name}");
            let mut shape = [0; 4];
            for dim in &mut shape[..dims as usize] {
                let d = r.usize()?;
                ensure!(d > 0, "zero tensor dimension");
                *dim = d;
            }
            let code = r.u32()?;
            let offset = r.usize()?;
            let previous = tensors.insert(
                name,
                Entry {
                    shape,
                    rank: dims as u8,
                    code,
                    offset,
                    extent: 0,
                },
            );
            ensure!(previous.is_none(), "duplicate tensor name");
        }
        let alignment = self.optional_unsigned("general.alignment")?.unwrap_or(32);
        let alignment = usize::try_from(alignment).context("GGUF alignment overflow")?;
        ensure!(
            alignment.is_power_of_two() && alignment <= 4096,
            "invalid GGUF alignment"
        );
        let start = (r.0.position() as usize)
            .checked_next_multiple_of(alignment)
            .context("GGUF data offset overflow")?;
        let mut sorted: Vec<_> = tensors.values_mut().collect();
        sorted.sort_unstable_by_key(|t| t.offset);
        for i in 0..sorted.len() {
            let offset = sorted[i].offset;
            ensure!(offset.is_multiple_of(alignment), "unaligned tensor");
            let absolute = start
                .checked_add(offset)
                .context("tensor offset overflow")?;
            ensure!(absolute < self.map.len(), "tensor offset outside file");
            let end = if let Some(next) = sorted.get(i + 1) {
                ensure!(offset < next.offset, "overlapping tensor offsets");
                start
                    .checked_add(next.offset)
                    .context("tensor offset overflow")?
            } else {
                self.map.len()
            };
            ensure!(end <= self.map.len(), "tensor offset outside file");
            sorted[i].offset = absolute;
            sorted[i].extent = end - absolute;
        }
        Ok(tensors)
    }

    fn value(&self, key: &str) -> Result<Value<'_>> {
        let field = self
            .meta
            .get(key)
            .with_context(|| format!("missing metadata {key}"))?;
        ensure!(field.kind != 9, "expected scalar metadata {key}");
        let mut r = Reader::new(&self.map[field.bytes.clone()]);
        r.value(field.kind)
    }

    pub fn text(&self, key: &str) -> Result<&str> {
        match self.value(key)? {
            Value::String(value) => Ok(value),
            _ => bail!("expected string {key}"),
        }
    }

    pub fn number(&self, key: &str) -> Result<f64> {
        let value = match self.value(key)? {
            Value::Unsigned(value) => value as f64,
            Value::Signed(value) => value as f64,
            Value::Float(value) => value,
            _ => bail!("expected number {key}"),
        };
        ensure!(value.is_finite(), "non-finite number {key}");
        Ok(value)
    }

    pub fn boolean(&self, key: &str) -> Result<bool> {
        match self.value(key)? {
            Value::Bool(value) => Ok(value),
            Value::Unsigned(value) if value <= 1 => Ok(value == 1),
            _ => bail!("expected boolean {key}"),
        }
    }

    pub fn optional_unsigned(&self, key: &str) -> Result<Option<u64>> {
        if !self.meta.contains_key(key) {
            return Ok(None);
        }
        match self.value(key)? {
            Value::Unsigned(value) => Ok(Some(value)),
            Value::Signed(value) if value >= 0 => Ok(Some(value as u64)),
            _ => bail!("expected unsigned integer {key}"),
        }
    }

    /// Read a selected, already bounded array without exposing the GGUF representation.
    fn array(&self, key: &str, kind: u32) -> Result<(Reader<'_>, usize)> {
        let field = self
            .meta
            .get(key)
            .with_context(|| format!("missing metadata {key}"))?;
        ensure!(field.kind == 9, "expected array metadata {key}");
        let mut reader = Reader::new(&self.map[field.bytes.clone()]);
        let (element, count) = reader.array_header()?;
        ensure!(element == kind, "unexpected array element type for {key}");
        Ok((reader, count))
    }

    pub fn strings(&self, key: &str) -> Result<Vec<&str>> {
        let (mut reader, count) = self.array(key, 8)?;
        (0..count).map(|_| reader.string()).collect()
    }

    /// Read an integer array of any GGUF element width as non-negative values.
    pub fn unsigned_integers(&self, key: &str) -> Result<Vec<u64>> {
        let field = self
            .meta
            .get(key)
            .with_context(|| format!("missing metadata {key}"))?;
        ensure!(field.kind == 9, "expected array metadata {key}");
        let mut reader = Reader::new(&self.map[field.bytes.clone()]);
        let (element, count) = reader.array_header()?;
        (0..count)
            .map(|_| match reader.value(element)? {
                Value::Unsigned(value) => Ok(value),
                Value::Signed(value) if value >= 0 => Ok(value as u64),
                _ => bail!("expected unsigned integer array {key}"),
            })
            .collect()
    }

    pub fn integers(&self, key: &str) -> Result<Vec<i32>> {
        let (mut reader, count) = self.array(key, 5)?;
        (0..count)
            .map(|_| {
                let bytes = reader.bytes()?;
                Ok(i32::from_le_bytes(bytes))
            })
            .collect()
    }
}

/// Scalar import values, borrowing string bytes from mapped storage.
enum Value<'a> {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(&'a str),
}

/// Bounded little-endian reads, with one aggregate element budget for the whole metadata scan.
struct Reader<'a>(Cursor<&'a [u8]>, usize);
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self(Cursor::new(bytes), MAX_VALUES)
    }

    fn remaining(&self) -> usize {
        self.0
            .get_ref()
            .len()
            .saturating_sub(self.0.position() as usize)
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.0.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.bytes()?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn usize(&mut self) -> Result<usize> {
        let bytes = self.bytes()?;
        usize::try_from(u64::from_le_bytes(bytes)).context("GGUF length overflow")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let start = self.0.position() as usize;
        let end = start.checked_add(n).context("GGUF value size overflow")?;
        let source = *self.0.get_ref();
        let bytes = source.get(start..end).context("truncated GGUF value")?;
        self.0.set_position(end as u64);
        Ok(bytes)
    }

    fn string(&mut self) -> Result<&'a str> {
        let n = self.usize()?;
        let bytes = self.take(n)?;
        Ok(std::str::from_utf8(bytes)?)
    }

    fn array_header(&mut self) -> Result<(u32, usize)> {
        let kind = self.u32()?;
        let n = self.usize()?;
        let minimum = match kind {
            8 => 8,
            9 => 12,
            _ => scalar_bytes(kind)?,
        };
        ensure!(
            n <= MAX_VALUES && n <= self.remaining() / minimum,
            "excessive or truncated metadata array"
        );
        Ok((kind, n))
    }

    fn skip_value(&mut self, kind: u32, depth: usize) -> Result<()> {
        ensure!(depth < 4, "nested metadata is too deep");
        self.1 = self
            .1
            .checked_sub(1)
            .context("metadata exceeds aggregate value budget")?;
        match kind {
            8 => {
                let n = self.usize()?;
                self.take(n)?;
            }
            9 => {
                let (kind, n) = self.array_header()?;
                self.skip_array(kind, n, depth)?;
            }
            _ => {
                let n = scalar_bytes(kind)?;
                self.take(n)?;
            }
        }
        Ok(())
    }

    fn skip_array(&mut self, kind: u32, n: usize, depth: usize) -> Result<()> {
        ensure!(n <= self.1, "metadata exceeds aggregate value budget");
        if kind != 8 && kind != 9 {
            ensure!(depth + 1 < 4, "nested metadata is too deep");
            self.1 -= n;
            let bytes = scalar_bytes(kind)?
                .checked_mul(n)
                .context("GGUF array size overflow")?;
            self.take(bytes)?;
        } else {
            for _ in 0..n {
                self.skip_value(kind, depth + 1)?;
            }
        }
        Ok(())
    }

    fn value(&mut self, kind: u32) -> Result<Value<'a>> {
        Ok(match kind {
            0 => {
                let v = self.bytes::<1>()?;
                Value::Unsigned(v[0] as u64)
            }
            1 => {
                let v = self.bytes::<1>()?;
                Value::Signed(v[0] as i8 as i64)
            }
            2 => {
                let v = self.bytes()?;
                Value::Unsigned(u16::from_le_bytes(v) as u64)
            }
            3 => {
                let v = self.bytes()?;
                Value::Signed(i16::from_le_bytes(v) as i64)
            }
            4 => {
                let v = self.bytes()?;
                Value::Unsigned(u32::from_le_bytes(v) as u64)
            }
            5 => {
                let v = self.bytes()?;
                Value::Signed(i32::from_le_bytes(v) as i64)
            }
            6 => {
                let v = self.bytes()?;
                Value::Float(f32::from_le_bytes(v) as f64)
            }
            7 => {
                let v = self.bytes::<1>()?;
                ensure!(v[0] <= 1, "invalid GGUF boolean");
                Value::Bool(v[0] == 1)
            }
            8 => Value::String(self.string()?),
            10 => {
                let v = self.bytes()?;
                Value::Unsigned(u64::from_le_bytes(v))
            }
            11 => {
                let v = self.bytes()?;
                Value::Signed(i64::from_le_bytes(v))
            }
            12 => {
                let v = self.bytes()?;
                Value::Float(f64::from_le_bytes(v))
            }
            _ => bail!("unknown metadata type {kind}"),
        })
    }
}

fn scalar_bytes(kind: u32) -> Result<usize> {
    match kind {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        10..=12 => Ok(8),
        _ => bail!("unknown metadata type {kind}"),
    }
}

/// Translate file-format codes into the execution encodings fixed by the packed shader ABI.
fn weight_format(code: u32) -> Result<WeightFormat> {
    use WeightFormat::*;
    Ok(match code {
        0 => F32,
        1 => F16,
        2 => Q4_0,
        3 => Q4_1,
        8 => Q8_0,
        12 => Q4K,
        13 => Q5K,
        14 => Q6K,
        30 => Bf16,
        39 => Mxfp4,
        _ => bail!("unsupported GGUF tensor type {code}"),
    })
}

#[cfg(test)]
pub(crate) mod tests;
