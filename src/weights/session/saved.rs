//! Consumed context written to a file and read straight back into session buffers.
//!
//! Layout, little-endian: consumed and resident counts, the consumed IDs, then each attention
//! layer's resident key and value rows and each recurrent layer's convolution and state memory,
//! all in layer order. Files carry no identity: the caller names them by the weights, build,
//! and IDs that produced them, and a file opened for anything else restores garbage.
use super::{Architecture, Checkpoint, Progress, Session};
use crate::error::{Context, Error, Result, ensure};
use crate::gpu::BufferView;
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, IntoInnerError, Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

/// Distinguishes concurrent saves within this process, which share its ID.
static SAVES: AtomicU64 = AtomicU64::new(0);

impl Session {
    /// Write the consumed context (IDs, attention caches, recurrent memory) to `path`,
    /// atomically: a reader sees either the previous file or the complete new one.
    /// Contexts holding spliced audio are refused; their feature rows are not saved.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let resident = self.resident()?;
        let vocabulary = self.vocabulary_size();
        ensure!(
            self.history.iter().all(|&t| (t as usize) < vocabulary),
            Error::Request("contexts with spliced audio cannot be saved")
        );
        let path = path.as_ref();
        let name = path
            .file_name()
            .context(Error::Request, "context path has no file name")?;
        let mut temporary = name.to_owned();
        let save = SAVES.fetch_add(1, Ordering::Relaxed);
        temporary.push(format!(".{}.{save}.tmp", std::process::id()));
        let temporary = path.with_file_name(temporary);
        let written = self.write(&temporary, resident);
        let renamed = written.and_then(|()| Ok(std::fs::rename(&temporary, path)?));
        if renamed.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        renamed
    }

    /// Write the complete file to a fresh path that no other save can be using.
    fn write(&self, path: &Path, resident: usize) -> Result<()> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut file = BufWriter::new(file);
        file.write_all(&(self.history.len() as u64).to_le_bytes())?;
        file.write_all(&(resident as u64).to_le_bytes())?;
        file.write_all(bytemuck::cast_slice(&self.history))?;
        let rows = resident * self.kv_row_bytes();
        for kv in self.kv() {
            kv.k.store(rows, &mut file)?;
            kv.v.store(rows, &mut file)?;
        }
        for view in self.recurrent_views() {
            view.store(view.length(), &mut file)?;
        }
        // Syncing before the rename so a crash cannot leave a complete-looking torn file.
        file.into_inner()
            .map_err(IntoInnerError::into_error)?
            .sync_all()?;
        Ok(())
    }

    /// Allocate a session holding a context written by [`Session::save`], checkpointed there.
    pub(in crate::weights) fn open(
        weights: &Architecture,
        path: &Path,
        capacity: usize,
    ) -> Result<Self> {
        let mut session = Self::new(weights, capacity)?;
        let mut file = File::open(path)?;
        let mut count = || -> Result<usize> {
            let mut bytes = [0; 8];
            file.read_exact(&mut bytes)?;
            usize::try_from(u64::from_le_bytes(bytes))
                .context(Error::Request, "saved context exceeds the session capacity")
        };
        let (consumed, resident) = (count()?, count()?);
        // Trusting the file everywhere except where it could overrun the session's buffers.
        ensure!(
            resident <= consumed && consumed <= capacity,
            Error::Request("saved context exceeds the session capacity")
        );
        let rows = resident * session.kv_row_bytes();
        let state: usize = session.recurrent_views().map(BufferView::length).sum();
        let mut history = vec![0u32; consumed];
        file.read_exact(bytemuck::cast_slice_mut(&mut history))?;
        for kv in session.kv() {
            kv.k.load(rows, &mut file)?;
            kv.v.load(rows, &mut file)?;
        }
        // Reading recurrent memory once into its checkpoint, then copying it into place.
        let mut state = vec![0; state];
        file.read_exact(&mut state)?;
        session.restore(&state);
        session.history = history;
        session.checkpoints = vec![Checkpoint {
            position: resident,
            state,
        }];
        session.progress = Progress::Ready { resident };
        Ok(session)
    }

    /// Bytes of one position's keys (or values) in one attention layer's cache.
    fn kv_row_bytes(&self) -> usize {
        self.kv()
            .first()
            .map_or(0, |kv| kv.k.length() / self.capacity)
    }
}
