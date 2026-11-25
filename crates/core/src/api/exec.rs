//! Layout executors: the one place where full / cross / packed-upper /
//! streamed outputs are assembled from engine-filled row blocks.

use crate::engine::{Engine, fill_parallel};
use crate::error::Result;
use crate::layout::blocks::{block_rows, ranges};
use crate::layout::fill::{mirror_upper, packed_len, row_offset};
use crate::layout::{CorrOutput, PackedOutput, RowBlock};
use ndarray::Array2;

/// Consumes a built engine.
pub trait EngineUser {
    /// Result type.
    type Out;
    /// Run with engine `e` producing `n_rows` output rows.
    fn run<E: Engine>(self, e: &E, n_rows: usize, with_p: bool) -> Result<Self::Out>;
}

fn alloc(n: usize, with_p: bool) -> (Vec<f64>, Option<Vec<f64>>) {
    (vec![0.0; n], with_p.then(|| vec![0.0; n]))
}

fn to_matrix(v: Vec<f64>, r: usize, c: usize) -> Result<Array2<f64>> {
    Array2::from_shape_vec((r, c), v).map_err(|e| crate::CorrError::Unsupported(e.to_string()))
}

/// In-memory symmetric matrix: upper triangle computed, then mirrored.
pub struct Full;

impl EngineUser for Full {
    type Out = CorrOutput;

    fn run<E: Engine>(self, e: &E, n: usize, with_p: bool) -> Result<CorrOutput> {
        let (mut c, mut p) = alloc(n * n, with_p);
        fill_parallel(e, 0..n, &mut c, p.as_deref_mut(), true);
        let mut corr = to_matrix(c, n, n)?;
        mirror_upper(&mut corr);
        let pvalues = match p {
            Some(p) => {
                let mut m = to_matrix(p, n, n)?;
                mirror_upper(&mut m);
                Some(m)
            }
            None => None,
        };
        Ok(CorrOutput { corr, pvalues })
    }
}

/// In-memory cross matrix.
pub struct Cross;

impl EngineUser for Cross {
    type Out = CorrOutput;

    fn run<E: Engine>(self, e: &E, n: usize, with_p: bool) -> Result<CorrOutput> {
        let nb = e.n_cols();
        let (mut c, mut p) = alloc(n * nb, with_p);
        fill_parallel(e, 0..n, &mut c, p.as_deref_mut(), false);
        let pvalues = p.map(|p| to_matrix(p, n, nb)).transpose()?;
        Ok(CorrOutput { corr: to_matrix(c, n, nb)?, pvalues })
    }
}

/// Packed upper triangle, assembled block by block.
pub struct Upper {
    pub block_rows: Option<usize>,
}

impl EngineUser for Upper {
    type Out = PackedOutput;

    fn run<E: Engine>(self, e: &E, n: usize, with_p: bool) -> Result<PackedOutput> {
        let (mut corr, mut pv) = alloc(packed_len(n), with_p);
        let br = block_rows(n, n, with_p, self.block_rows);
        let (mut bc, mut bp) = alloc(br.min(n) * n, with_p);
        for r in ranges(n, br) {
            let len = r.len() * n;
            fill_parallel(e, r.clone(), &mut bc[..len], bp.as_deref_mut().map(|p| &mut p[..len]), true);
            for (k, i) in r.enumerate() {
                let dst = row_offset(n, i);
                let src = k * n + i..(k + 1) * n;
                corr[dst..dst + n - i].copy_from_slice(&bc[src.clone()]);
                if let (Some(pv), Some(bp)) = (pv.as_mut(), bp.as_ref()) {
                    pv[dst..dst + n - i].copy_from_slice(&bp[src]);
                }
            }
        }
        Ok(PackedOutput { n, corr, pvalues: pv })
    }
}

/// Streaming: complete rows handed to a sink block by block.
pub struct Blocks<F> {
    pub sink: F,
    pub block_rows: Option<usize>,
}

impl<F: FnMut(RowBlock<'_>) -> Result<()>> EngineUser for Blocks<F> {
    type Out = ();

    fn run<E: Engine>(mut self, e: &E, n: usize, with_p: bool) -> Result<()> {
        let nb = e.n_cols();
        let br = block_rows(n, nb, with_p, self.block_rows);
        let (mut bc, mut bp) = alloc(br.min(n) * nb, with_p);
        for r in ranges(n, br) {
            let len = r.len() * nb;
            fill_parallel(e, r.clone(), &mut bc[..len], bp.as_deref_mut().map(|p| &mut p[..len]), false);
            (self.sink)(RowBlock {
                rows: r,
                n_cols: nb,
                corr: &bc[..len],
                pvalues: bp.as_deref().map(|p| &p[..len]),
            })?;
        }
        Ok(())
    }
}
