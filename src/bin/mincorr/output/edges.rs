//! Long-format edge list: `row  col  r  [p]`, optionally thresholded and
//! limited to the strongest K edges per row. Symmetric runs write the
//! upper triangle only; single-matrix runs never write the diagonal.

use super::{Out, Sink, format_value};
use anyhow::Result;
use mincorr::RowBlock;
use std::io::Write;
use std::sync::Arc;

/// Edge filters.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeFilter {
    /// Keep |r| ≥ this.
    pub min_abs_r: Option<f64>,
    /// Keep p ≤ this.
    pub max_p: Option<f64>,
    /// Keep the K strongest edges per row.
    pub top_k: Option<usize>,
}

/// Which cells of a block become edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cells {
    /// Single matrix, symmetric statistic: j > i.
    Upper,
    /// Single matrix, asymmetric statistic: j ≠ i.
    OffDiagonal,
    /// Cross matrix: every cell.
    All,
}

/// Edge-list sink.
pub struct EdgesSink {
    out: Out,
    row_ids: Arc<Vec<String>>,
    col_ids: Arc<Vec<String>>,
    cells: Cells,
    filter: EdgeFilter,
    digits: Option<usize>,
    cand: Vec<(usize, f64, f64)>,
    line: String,
}

impl EdgesSink {
    /// Create and write the header.
    pub fn new(
        mut out: Out,
        row_ids: Arc<Vec<String>>,
        col_ids: Arc<Vec<String>>,
        cells: Cells,
        with_p: bool,
        filter: EdgeFilter,
        digits: Option<usize>,
    ) -> Result<Self> {
        writeln!(out, "row\tcol\tr{}", if with_p { "\tp" } else { "" })?;
        Ok(Self { out, row_ids, col_ids, cells, filter, digits, cand: Vec::new(), line: String::new() })
    }

    fn keep(&self, r: f64, p: f64) -> bool {
        !r.is_nan()
            && self.filter.min_abs_r.is_none_or(|t| r.abs() >= t)
            && self.filter.max_p.is_none_or(|t| p <= t)
    }

    fn write_row(&mut self, i: usize, with_p: bool) -> Result<()> {
        if let Some(k) = self.filter.top_k {
            self.cand.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0)));
            self.cand.truncate(k);
            self.cand.sort_by_key(|c| c.0);
        }
        let mut ry = ryu::Buffer::new();
        for &(j, r, p) in &self.cand {
            self.line.clear();
            self.line.push_str(&self.row_ids[i]);
            self.line.push('\t');
            self.line.push_str(&self.col_ids[j]);
            self.line.push('\t');
            format_value(r, self.digits, &mut ry, &mut self.line);
            if with_p {
                self.line.push('\t');
                format_value(p, None, &mut ry, &mut self.line);
            }
            self.line.push('\n');
            self.out.write_all(self.line.as_bytes())?;
        }
        Ok(())
    }
}

impl Sink for EdgesSink {
    fn block(&mut self, b: &RowBlock<'_>) -> Result<()> {
        for i in b.rows.clone() {
            let (cr, pr) = (b.corr_row(i), b.pvalue_row(i));
            let j0 = if self.cells == Cells::Upper { i + 1 } else { 0 };
            self.cand.clear();
            for j in j0..b.n_cols {
                if j == i && self.cells == Cells::OffDiagonal {
                    continue;
                }
                let p = pr.map_or(f64::NAN, |p| p[j]);
                if self.keep(cr[j], p) {
                    self.cand.push((j, cr[j], p));
                }
            }
            self.write_row(i, pr.is_some())?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        self.out.finish()
    }
}
