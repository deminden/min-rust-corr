//! Proportionality coefficient ρp (Lovell et al. 2015, PLoS Comput. Biol.
//! 11(3); Erb & Notredame 2016, Theory Biosci. 135):
//!
//! ρp(x, y) = 1 − var(x − y) / (var x + var y) = 2 cov(x, y) / (var x + var y)
//!
//! on log-ratio transformed rows (clr by default, see
//! [`crate::prep::logratio::log_clr`]). As a Gram method each row is only
//! centred; ρp follows from the dot product and the two squared norms.

use crate::engine::RowTransform;
use crate::prep::RowStatus;
use crate::prep::standardize::{all_equal, mean};

/// ρp row transform: centre (no scaling).
#[derive(Debug, Clone, Copy, Default)]
pub struct Rho;

impl RowTransform for Rho {
    type Scratch = ();

    fn transform(&self, x: &[f64], out: &mut [f64], _: &mut ()) -> RowStatus {
        if x.len() < 2 || all_equal(x) {
            out.fill(0.0);
            return RowStatus::Degenerate;
        }
        let m = mean(x);
        out.iter_mut().zip(x).for_each(|(o, &v)| *o = v - m);
        RowStatus::Ok
    }

    fn combine(&self, dot: f64, sq_a: f64, sq_b: f64) -> f64 {
        let r = 2.0 * dot / (sq_a + sq_b);
        r.clamp(-1.0, 1.0)
    }

    fn pvalue(&self, _r: f64, _n: usize) -> f64 {
        f64::NAN
    }
}
