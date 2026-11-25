//! Hellinger correlation (placeholder until the clean-room module lands).

use crate::engine::{PairKernel, PairStat};
use crate::error::{CorrError, Result};
use crate::options::{HellcorParams, PvalueSpec};

/// Hellinger correlation kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hellcor {
    #[allow(dead_code)]
    params: HellcorParams,
}

impl Hellcor {
    /// Create with validated parameters.
    pub fn new(params: HellcorParams) -> Self {
        Self { params }
    }
}

impl PairKernel for Hellcor {
    type Shared = ();
    type Prepared = ();
    type Scratch = ();
    fn shared(&self, _: usize, _: &PvalueSpec) -> Result<()> {
        Err(CorrError::Unsupported("hellcor not yet implemented".into()))
    }
    fn scratch(&self, _: usize) {}
    fn prepare(&self, _: &[f64], _: &(), _: &mut ()) -> Option<()> {
        None
    }
    fn eval(&self, _: &(), _: &(), _: &(), _: &mut ()) -> PairStat {
        PairStat::nan(0)
    }
    fn eval_gathered(&self, x: &[f64], _: &[f64], _: &(), _: &mut ()) -> PairStat {
        PairStat::nan(x.len())
    }
    fn pvalue(&self, _: &PairStat, _: &()) -> f64 {
        f64::NAN
    }
}
