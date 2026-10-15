use super::{Gguf, Reader};
use anyhow::Result;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);

impl Fixture {
    fn new(bytes: &[u8]) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir();
        let pid = std::process::id();
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!("oco-gguf-{pid}-{id}"));
        std::fs::write(&path, bytes)?;
        Ok(Self(path))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}

fn header(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut bytes = Vec::from(b"GGUF".as_slice());
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(tensors.to_le_bytes());
    bytes.extend(metadata.to_le_bytes());
    bytes
}

#[test]
fn metadata_scan_checks_truncation_and_skips_unused_arrays() -> Result<()> {
    let mut bytes = header(0, 2);
    string(&mut bytes, "unused");
    bytes.extend(9u32.to_le_bytes()); // Array of two strings.
    bytes.extend(8u32.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    string(&mut bytes, "one");
    string(&mut bytes, "two");
    string(&mut bytes, "count");
    bytes.extend(10u32.to_le_bytes());
    bytes.extend(23u64.to_le_bytes());
    let file = Fixture::new(&bytes)?;
    let gguf = Gguf::open(&file.0, &["count"])?;
    assert_eq!(gguf.meta.len(), 1);
    assert_eq!(gguf.optional_unsigned("count")?, Some(23));
    drop(gguf);
    let gguf = Gguf::open(&file.0, &["unused", "count"])?;
    assert_eq!(gguf.strings("unused")?, ["one", "two"]);
    assert!(gguf.integers("unused").is_err());
    assert!(gguf.strings("count").is_err());
    drop(gguf);
    for cut in 0..bytes.len() {
        std::fs::write(&file.0, &bytes[..cut])?;
        let opened = Gguf::open(&file.0, &["count"]);
        assert!(opened.is_err(), "truncation at byte {cut}");
    }
    Ok(())
}

#[test]
fn metadata_work_and_nesting_are_bounded() {
    let mut bytes = Vec::new();
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(u64::MAX.to_le_bytes());
    let mut reader = Reader::new(&bytes);
    let skipped = reader.skip_value(9, 0);
    assert!(skipped.is_err(), "unbounded array count");

    bytes.clear();
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(5u64.to_le_bytes());
    bytes.extend([0; 5]);
    let mut reader = Reader::new(&bytes);
    reader.1 = 5; // The array and its elements share one work budget.
    let skipped = reader.skip_value(9, 0);
    assert!(skipped.is_err(), "aggregate value budget");

    bytes.clear();
    for _ in 0..4 {
        bytes.extend(9u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
    }
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    bytes.push(0);
    let mut reader = Reader::new(&bytes);
    let skipped = reader.skip_value(9, 0);
    let error = skipped.expect_err("nested arrays must be bounded");
    assert!(error.to_string().contains("too deep"));
}

#[test]
fn selected_tensors_require_aligned_complete_payloads() -> Result<()> {
    let mut bytes = header(1, 0);
    string(&mut bytes, "weight");
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(4u64.to_le_bytes());
    bytes.extend(0u32.to_le_bytes()); // Four F32 elements at data offset zero.
    bytes.extend(0u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend([0; 16]);
    let file = Fixture::new(&bytes)?;
    let gguf = Gguf::open(&file.0, &[])?;
    let mut directory = gguf.directory()?;
    let mut entry = directory.remove("weight").unwrap();
    assert_eq!(entry.select()?.bytes, 16);
    entry.offset += 1;
    assert!(entry.select().is_err(), "unaligned F32 storage");
    drop(directory);
    drop(gguf);

    bytes.pop();
    std::fs::write(&file.0, bytes)?;
    let gguf = Gguf::open(&file.0, &[])?;
    let directory = gguf.directory()?;
    assert!(
        directory["weight"].select().is_err(),
        "payload crosses the file boundary"
    );
    Ok(())
}
