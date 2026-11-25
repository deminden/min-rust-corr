//! Output stream: plain file, gzip file or stdout.

use anyhow::{Context, Result};
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::{BufWriter, Stdout, Write};
use std::path::Path;

/// A writable output that must be finished explicitly.
pub enum Out {
    /// Plain file.
    Plain(BufWriter<File>),
    /// Gzip-compressed file.
    Gz(Box<GzEncoder<BufWriter<File>>>),
    /// Standard output.
    Stdout(BufWriter<Stdout>),
}

impl Out {
    /// Create `path` ("-" = stdout), gzip-compressed when `gz`.
    pub fn create(path: &Path, gz: bool) -> Result<Self> {
        if path.as_os_str() == "-" {
            return Ok(Out::Stdout(BufWriter::new(std::io::stdout())));
        }
        let f = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
        let w = BufWriter::with_capacity(1 << 20, f);
        Ok(if gz { Out::Gz(Box::new(GzEncoder::new(w, Compression::fast()))) } else { Out::Plain(w) })
    }

    /// Flush everything (and write the gzip trailer).
    pub fn finish(self) -> Result<()> {
        match self {
            Out::Plain(mut w) => w.flush()?,
            Out::Gz(g) => g.finish()?.flush()?,
            Out::Stdout(mut w) => w.flush()?,
        }
        Ok(())
    }
}

impl Write for Out {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Out::Plain(w) => w.write(buf),
            Out::Gz(w) => w.write(buf),
            Out::Stdout(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Out::Plain(w) => w.flush(),
            Out::Gz(w) => w.flush(),
            Out::Stdout(w) => w.flush(),
        }
    }
}
