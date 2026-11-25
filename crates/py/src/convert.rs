//! Python argument → core option conversion.

use mincorr::{BicorParams, Correlator, HellcorParams, Method, NanPolicy, PvalueSpec};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

/// Method parameters passed from Python.
pub struct Params {
    pub alpha: f64,
    pub kmax: usize,
    pub lmax: usize,
    pub max_p_outliers: f64,
    pub mc_samples: usize,
    pub seed: u64,
}

fn err(e: impl std::fmt::Display) -> pyo3::PyErr {
    PyValueError::new_err(e.to_string())
}

/// Build a correlator from string options.
pub fn correlator(method: &str, pvalues: bool, nan_policy: &str, p: &Params) -> PyResult<Correlator> {
    let method = match Method::from_name(method).map_err(err)? {
        Method::Bicor(_) => {
            Method::Bicor(BicorParams { max_p_outliers: p.max_p_outliers, ..Default::default() })
        }
        Method::Hellcor(_) => Method::Hellcor(HellcorParams {
            alpha: p.alpha,
            kmax: p.kmax,
            lmax: p.lmax,
            ..Default::default()
        }),
        m => m,
    };
    let spec = match (pvalues, &method) {
        (false, _) => PvalueSpec::None,
        (true, Method::Hellcor(_)) => PvalueSpec::MonteCarlo { samples: p.mc_samples, seed: p.seed },
        (true, _) => PvalueSpec::Analytic,
    };
    let nan = match nan_policy {
        "pairwise" | "pairwise_complete" => NanPolicy::PairwiseComplete,
        "propagate" => NanPolicy::Propagate,
        "error" | "raise" => NanPolicy::Error,
        other => return Err(err(format!("unknown nan_policy '{other}'"))),
    };
    Ok(Correlator::new(method).pvalues(spec).nan_policy(nan))
}

/// Map a core error to `ValueError`.
pub fn core_err(e: mincorr::CorrError) -> pyo3::PyErr {
    err(e)
}
