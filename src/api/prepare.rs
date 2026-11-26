//! Matrix-level preprocessing that must see every row at once.

use crate::options::Method;
use crate::prep::logratio::log_clr;
use ndarray::{Array2, ArrayView2};

/// For ρp with `clr`, the centred log-ratio transform of all parts jointly
/// (the reference is the mean log over every row); `None` otherwise.
pub fn proportional(m: &Method, parts: &[ArrayView2<'_, f64>]) -> Option<Vec<Array2<f64>>> {
    match m {
        Method::Proportionality(p) if p.clr => Some(log_clr(parts, p.pseudocount)),
        _ => None,
    }
}
