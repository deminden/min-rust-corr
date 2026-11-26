//! Hellinger correlation (Geenens & Lafaye de Micheaux 2022, JASA 117).
//! Clean-room implementation written from the paper; see docs/hellcor.md.
//!
//! Rows are reduced to half-rank indices once; every marginal quantity
//! (Beta quantile, density, Legendre basis) comes from per-n tables, and
//! each pair costs O(n² + n·K·L).

pub mod basis;
pub mod coeffs;
pub mod estimate;
pub mod eta;
pub mod marginal;
pub mod neighbors;
pub mod null;
pub mod select;

use crate::engine::{PairKernel, PairStat};
use crate::error::Result;
use crate::options::{HellcorParams, PvalueSpec};
use crate::prep::standardize::all_equal;
use crate::stats::empirical::upper_tail_p;
use estimate::{Work, estimate};
use marginal::{Tables, half_ranks};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Hellinger correlation kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hellcor {
    params: HellcorParams,
}

impl Hellcor {
    /// Create with validated parameters.
    pub fn new(params: HellcorParams) -> Self {
        Self { params }
    }

    fn max_order(&self) -> usize {
        self.params.kmax.max(self.params.lmax)
    }
}

type Lazy<T> = Mutex<HashMap<usize, Arc<OnceLock<T>>>>;

/// Shared state: tables for the full sample size, lazily built tables and
/// null distributions for other effective sample sizes.
pub struct HellcorShared {
    main: Arc<Tables>,
    tables: Lazy<Arc<Tables>>,
    nulls: Lazy<Vec<f64>>,
    mc: Option<(usize, u64)>,
}

fn lazy<T: Clone>(map: &Lazy<T>, n: usize, make: impl FnOnce() -> T) -> T {
    let cell = match map.lock() {
        Ok(mut m) => m.entry(n).or_default().clone(),
        Err(p) => p.into_inner().entry(n).or_default().clone(),
    };
    cell.get_or_init(make).clone()
}

impl HellcorShared {
    fn tables_for(&self, k: &Hellcor, n: usize) -> Arc<Tables> {
        if n == self.main.n {
            return self.main.clone();
        }
        lazy(&self.tables, n, || Arc::new(Tables::new(n, k.params.alpha, k.max_order())))
    }

    fn null_for(&self, k: &Hellcor, n: usize) -> Option<Arc<Vec<f64>>> {
        let (samples, seed) = self.mc?;
        let tab = self.tables_for(k, n);
        Some(Arc::new(lazy(&self.nulls, n, || null::null_distribution(&tab, &k.params, samples, seed))))
    }
}

/// Per-thread scratch.
#[derive(Default)]
pub struct HellcorScratch {
    work: Work,
    idx: Vec<u32>,
    hx: Vec<u32>,
    hy: Vec<u32>,
}

impl PairKernel for Hellcor {
    type Shared = HellcorShared;
    type Prepared = Vec<u32>;
    type Scratch = HellcorScratch;

    fn shared(&self, n_obs: usize, _n_rows: usize, pv: &PvalueSpec) -> Result<HellcorShared> {
        let mc = match *pv {
            PvalueSpec::MonteCarlo { samples, seed } => Some((samples, seed)),
            _ => None,
        };
        let main = Arc::new(Tables::new(n_obs, self.params.alpha, self.max_order()));
        let sh = HellcorShared { main, tables: Mutex::default(), nulls: Mutex::default(), mc };
        if mc.is_some() && n_obs > 3 {
            let _ = sh.null_for(self, n_obs);
        }
        Ok(sh)
    }

    fn scratch(&self, _: usize) -> HellcorScratch {
        HellcorScratch::default()
    }

    fn prepare(&self, x: &[f64], _: &HellcorShared, s: &mut HellcorScratch) -> Option<Vec<u32>> {
        if x.len() < 2 || all_equal(x) {
            return None;
        }
        let mut h = Vec::with_capacity(x.len());
        half_ranks(x, &mut s.idx, &mut h);
        Some(h)
    }

    fn eval(&self, a: &Vec<u32>, b: &Vec<u32>, sh: &HellcorShared, s: &mut HellcorScratch) -> PairStat {
        let value = estimate(a, b, &sh.main, &self.params, &mut s.work);
        PairStat { value, n_eff: a.len(), aux: 0.0 }
    }

    fn eval_gathered(&self, x: &[f64], y: &[f64], sh: &HellcorShared, s: &mut HellcorScratch) -> PairStat {
        let n = x.len();
        if n < 2 || all_equal(x) || all_equal(y) {
            return PairStat::nan(n);
        }
        half_ranks(x, &mut s.idx, &mut s.hx);
        half_ranks(y, &mut s.idx, &mut s.hy);
        let tab = sh.tables_for(self, n);
        PairStat { value: estimate(&s.hx, &s.hy, &tab, &self.params, &mut s.work), n_eff: n, aux: 0.0 }
    }

    fn pvalue(&self, st: &PairStat, sh: &HellcorShared) -> f64 {
        if st.n_eff <= 3 {
            return f64::NAN;
        }
        sh.null_for(self, st.n_eff).map_or(f64::NAN, |null| upper_tail_p(st.value, &null))
    }
}
