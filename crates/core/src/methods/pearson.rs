//! Pearson correlation.
//!
//! r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² Σ(y−ȳ)²). As a Gram method each row is
//! centred and scaled to unit norm; r is then a dot product. P-values use
//! t = r√((n−2)/(1−r²)) with n−2 degrees of freedom.

use crate::engine::RowTransform;
use crate::prep::RowStatus;
use crate::prep::standardize::center_unit_norm;

/// Pearson row transform: centre, then scale to unit Euclidean norm.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pearson;

impl RowTransform for Pearson {
    type Scratch = ();

    fn transform(&self, x: &[f64], out: &mut [f64], _: &mut ()) -> RowStatus {
        center_unit_norm(x, out)
    }
}
