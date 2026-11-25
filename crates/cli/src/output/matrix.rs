//! Matrix writers: TSV (optionally gzipped) and NumPy .npy.

use super::{Out, format_value};
use anyhow::Result;
use rayon::prelude::*;
use std::io::Write;
use std::sync::Arc;

/// Writes consecutive matrix rows.
pub trait MatrixSink {
    /// Write rows starting at global row `first`, `ncols` values each.
    fn rows(&mut self, first: usize, data: &[f64], ncols: usize) -> Result<()>;
    /// Flush and close.
    fn finish(self: Box<Self>) -> Result<()>;
}

/// TSV with a header of column IDs and a leading row-ID column. Rows are
/// formatted (and, for `.gz`, compressed into independent gzip members)
/// in parallel; members are written in order, which is a valid gzip
/// stream readable by any multi-member decoder (gzip, zcat, R, pandas).
pub struct TsvWriter {
    out: Out,
    row_ids: Arc<Vec<String>>,
    digits: Option<usize>,
    gz: bool,
}

const ROWS_PER_CHUNK: usize = 8;

impl TsvWriter {
    /// Create and write the header. `out` must be uncompressed; `gz`
    /// selects member-wise compression.
    pub fn new(
        mut out: Out,
        row_ids: Arc<Vec<String>>,
        col_ids: &[String],
        digits: Option<usize>,
        gz: bool,
    ) -> Result<Self> {
        let header = format!("\t{}\n", col_ids.join("\t"));
        out.write_all(&encode(header.into_bytes(), gz)?)?;
        Ok(Self { out, row_ids, digits, gz })
    }

    fn chunk(&self, first: usize, data: &[f64], ncols: usize) -> Result<Vec<u8>> {
        let mut text = String::with_capacity(data.len() * 20);
        let mut ry = ryu::Buffer::new();
        for (k, row) in data.chunks(ncols.max(1)).enumerate() {
            text.push_str(&self.row_ids[first + k]);
            for &v in row {
                text.push('\t');
                format_value(v, self.digits, &mut ry, &mut text);
            }
            text.push('\n');
        }
        encode(text.into_bytes(), self.gz)
    }
}

fn encode(bytes: Vec<u8>, gz: bool) -> Result<Vec<u8>> {
    if !gz {
        return Ok(bytes);
    }
    let mut e =
        flate2::write::GzEncoder::new(Vec::with_capacity(bytes.len() / 3), flate2::Compression::fast());
    e.write_all(&bytes)?;
    Ok(e.finish()?)
}

impl MatrixSink for TsvWriter {
    fn rows(&mut self, first: usize, data: &[f64], ncols: usize) -> Result<()> {
        let step = ROWS_PER_CHUNK * ncols.max(1);
        // Waves of a few chunks per thread bound the buffered text.
        let wave = step * 4 * rayon::current_num_threads();
        for (w, part) in data.chunks(wave).enumerate() {
            let base = first + w * (wave / ncols.max(1));
            let parts: Vec<Result<Vec<u8>>> = part
                .par_chunks(step)
                .enumerate()
                .map(|(k, c)| self.chunk(base + k * ROWS_PER_CHUNK, c, ncols))
                .collect();
            for p in parts {
                self.out.write_all(&p?)?;
            }
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        self.out.finish()
    }
}

/// NumPy v1.0 .npy file of little-endian float64 in C order.
pub struct NpyWriter {
    out: Out,
}

impl NpyWriter {
    /// Create and write the header for a `rows × cols` matrix.
    pub fn new(mut out: Out, rows: usize, cols: usize) -> Result<Self> {
        let dict = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        // magic(6) + version(2) + len(2) + dict + padding + '\n' ≡ 0 (mod 64)
        let unpadded = 10 + dict.len() + 1;
        let pad = (64 - unpadded % 64) % 64;
        let header = format!("{dict}{}\n", " ".repeat(pad));
        out.write_all(b"\x93NUMPY\x01\x00")?;
        out.write_all(&(header.len() as u16).to_le_bytes())?;
        out.write_all(header.as_bytes())?;
        Ok(Self { out })
    }
}

impl MatrixSink for NpyWriter {
    fn rows(&mut self, _: usize, data: &[f64], _: usize) -> Result<()> {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.out.write_all(&bytes)?;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        self.out.finish()
    }
}
