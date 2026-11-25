//! Generic entry points; the GIL is released during computation.

use crate::convert::{Params, core_err, correlator};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Out<'py> = (Bound<'py, PyAny>, Option<Bound<'py, PyAny>>);

/// compute(data, other, method, layout, pvalues, nan_policy, alpha, kmax, lmax,
/// max_p_outliers, mc_samples, seed) -> (corr, pvalues | None)
#[pyfunction]
#[pyo3(signature = (data, other, method, layout, pvalues, nan_policy, alpha, kmax, lmax, max_p_outliers, mc_samples, seed))]
#[allow(clippy::too_many_arguments)]
pub fn compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    other: Option<PyReadonlyArray2<'py, f64>>,
    method: &str,
    layout: &str,
    pvalues: bool,
    nan_policy: &str,
    alpha: f64,
    kmax: usize,
    lmax: usize,
    max_p_outliers: f64,
    mc_samples: usize,
    seed: u64,
) -> PyResult<Out<'py>> {
    let params = Params { alpha, kmax, lmax, max_p_outliers, mc_samples, seed };
    let c = correlator(method, pvalues, nan_policy, &params)?;
    let a = data.as_array();
    let b = other.as_ref().map(|o| o.as_array());
    match (layout, b) {
        ("upper", None) => {
            let o = py.detach(|| c.upper(a)).map_err(core_err)?;
            let p = o.pvalues.map(|p| p.into_pyarray(py).into_any());
            Ok((o.corr.into_pyarray(py).into_any(), p))
        }
        ("full", None) | ("cross", None) => {
            let o = py.detach(|| c.full(a)).map_err(core_err)?;
            Ok((o.corr.into_pyarray(py).into_any(), o.pvalues.map(|p| p.into_pyarray(py).into_any())))
        }
        ("full", Some(b)) | ("cross", Some(b)) => {
            let o = py.detach(|| c.cross(a, b)).map_err(core_err)?;
            Ok((o.corr.into_pyarray(py).into_any(), o.pvalues.map(|p| p.into_pyarray(py).into_any())))
        }
        (l, _) => Err(PyValueError::new_err(format!(
            "unsupported layout '{l}' (full, cross, upper; upper needs one matrix)"
        ))),
    }
}

/// pair(x, y, method, pvalues, nan_policy, ...) -> (r, p | None, n_eff)
#[pyfunction]
#[pyo3(signature = (x, y, method, pvalues, nan_policy, alpha, kmax, lmax, max_p_outliers, mc_samples, seed))]
#[allow(clippy::too_many_arguments)]
pub fn pair(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    method: &str,
    pvalues: bool,
    nan_policy: &str,
    alpha: f64,
    kmax: usize,
    lmax: usize,
    max_p_outliers: f64,
    mc_samples: usize,
    seed: u64,
) -> PyResult<(f64, Option<f64>, usize)> {
    let params = Params { alpha, kmax, lmax, max_p_outliers, mc_samples, seed };
    let c = correlator(method, pvalues, nan_policy, &params)?;
    let (xa, ya) = (x.as_array(), y.as_array());
    let r = py.detach(|| match (xa.as_slice(), ya.as_slice()) {
        (Some(xs), Some(ys)) => c.pair(xs, ys),
        _ => c.pair(&xa.to_vec(), &ya.to_vec()),
    });
    let r = r.map_err(core_err)?;
    Ok((r.corr, r.pvalue, r.n_eff))
}
