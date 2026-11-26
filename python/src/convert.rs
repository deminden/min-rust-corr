//! Python options → core `Correlator`.

use mincorr::{Correlator, Method, MethodSettings, NanPolicy, PvalueSpec};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// All options, extracted from the Python dict built by `mincorr._options`.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub struct Options {
    method: String,
    pvalues: bool,
    nan_policy: String,
    alpha: f64,
    kmax: usize,
    lmax: usize,
    max_p_outliers: f64,
    mc_samples: usize,
    seed: u64,
}

/// Map a core error to `ValueError`.
pub fn core_err(e: mincorr::CorrError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Build a correlator from the Python-side options.
pub fn correlator(p: &Options) -> PyResult<Correlator> {
    let settings =
        MethodSettings { alpha: p.alpha, kmax: p.kmax, lmax: p.lmax, max_p_outliers: p.max_p_outliers };
    let method = Method::with_settings(&p.method, &settings).map_err(core_err)?;
    let spec = if p.pvalues { PvalueSpec::auto(&method, p.mc_samples, p.seed) } else { PvalueSpec::None };
    let nan: NanPolicy = p.nan_policy.parse().map_err(core_err)?;
    Ok(Correlator::new(method).pvalues(spec).nan_policy(nan))
}
