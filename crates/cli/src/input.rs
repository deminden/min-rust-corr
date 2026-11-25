//! Streaming matrix reader: plain / .gz / single-file .tar.gz, tab or
//! comma separated, strict numeric parsing, input row order preserved.

use crate::args::{Delimiter, InputArgs};
use anyhow::{Context, Result, bail};
use flate2::read::MultiGzDecoder;
use ndarray::Array2;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A loaded matrix with its labels.
pub struct Matrix {
    /// Row identifiers (input order unless sorted).
    pub row_ids: Vec<String>,
    /// Column identifiers.
    pub col_ids: Vec<String>,
    /// Values, rows × columns.
    pub data: Array2<f64>,
}

fn open(path: &Path) -> Result<Box<dyn Read>> {
    let name = path.to_string_lossy();
    let file = File::open(path).with_context(|| format!("cannot open {name}"))?;
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let mut archive = tar::Archive::new(MultiGzDecoder::new(file));
        for entry in archive.entries()? {
            let mut entry = entry?;
            if entry.header().entry_type().is_file() {
                let mut buf = Vec::new();
                entry.read_to_end(&mut buf)?;
                return Ok(Box::new(std::io::Cursor::new(buf)));
            }
        }
        bail!("no regular file found in archive {name}");
    }
    if name.ends_with(".gz") {
        return Ok(Box::new(MultiGzDecoder::new(file)));
    }
    Ok(Box::new(file))
}

fn sniff(header: &str, d: Delimiter) -> char {
    match d {
        Delimiter::Tab => '\t',
        Delimiter::Comma => ',',
        Delimiter::Auto if header.contains('\t') => '\t',
        Delimiter::Auto if header.contains(',') => ',',
        Delimiter::Auto => '\t',
    }
}

fn parse_cell(s: &str, na: &[String], line: usize, col: usize) -> Result<f64> {
    let t = s.trim();
    if t.is_empty() || na.iter().any(|n| n == t) {
        return Ok(f64::NAN);
    }
    t.parse::<f64>().map_err(|_| anyhow::anyhow!("line {line}, column {col}: cannot parse '{t}' as a number"))
}

/// Read a matrix file.
pub fn read_matrix(path: &Path, opts: &InputArgs) -> Result<Matrix> {
    let mut lines = BufReader::with_capacity(1 << 20, open(path)?).lines();
    let header = lines.next().context("input is empty")??;
    let header = header.trim_end_matches('\r');
    let delim = sniff(header, opts.delimiter);
    let mut col_ids: Vec<String> = header.split(delim).map(|s| s.trim().to_string()).collect();
    let (mut row_ids, mut values) = (Vec::new(), Vec::new());
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut ncol: Option<usize> = None;
    for (k, line) in lines.enumerate() {
        let lineno = k + 2;
        let line = line?;
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split(delim);
        let id = fields.next().unwrap_or("").trim().to_string();
        if id.is_empty() {
            bail!("line {lineno}: empty row ID");
        }
        if let Some(prev) = seen.insert(id.clone(), lineno) {
            bail!("line {lineno}: duplicate row ID '{id}' (first seen on line {prev})");
        }
        let start = values.len();
        for (c, f) in fields.enumerate() {
            values.push(parse_cell(f, &opts.na_strings, lineno, c + 2)?);
        }
        let got = values.len() - start;
        match ncol {
            None => ncol = Some(got),
            Some(n) if n != got => bail!("line {lineno}: {got} values, expected {n} (ragged row)"),
            _ => {}
        }
        row_ids.push(id);
    }
    let ncol = ncol.context("input has a header but no data rows")?;
    // Header either names the ID column too, or (R write.table style) only the samples.
    if col_ids.len() == ncol + 1 {
        col_ids.remove(0);
    } else if col_ids.len() != ncol {
        bail!("header has {} fields but rows have {} values", col_ids.len(), ncol);
    }
    if ncol < 2 {
        bail!("at least two sample columns are required");
    }
    let mut m = Matrix { data: Array2::from_shape_vec((row_ids.len(), ncol), values)?, row_ids, col_ids };
    if opts.sort_rows {
        sort_rows(&mut m);
    }
    Ok(m)
}

fn sort_rows(m: &mut Matrix) {
    let mut order: Vec<usize> = (0..m.row_ids.len()).collect();
    order.sort_by(|&a, &b| m.row_ids[a].cmp(&m.row_ids[b]));
    m.data = m.data.select(ndarray::Axis(0), &order);
    m.row_ids = order.iter().map(|&i| m.row_ids[i].clone()).collect();
}
