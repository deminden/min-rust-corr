//! Generic entry points; the GIL is released during computation.

use crate::convert::{Options, core_err, correlator};
use mincorr::Correlator;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Out<'py> = (Bound<'py, PyAny>, Option<Bound<'py, PyAny>>);

fn array<'py, A: IntoPyArray>(py: Python<'py>, a: A) -> Bound<'py, PyAny> {
    a.into_pyarray(py).into_any()
}

fn with_covariates(c: Correlator, cov: Option<PyReadonlyArray2<'_, f64>>) -> Correlator {
    match cov {
        Some(z) => c.covariates(z.as_array().to_owned()),
        None => c,
    }
}

/// compute(data, other, layout, options, covariates) -> (corr, pvalues | None)
#[pyfunction]
#[pyo3(signature = (data, other, layout, options, covariates=None))]
pub fn compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    other: Option<PyReadonlyArray2<'py, f64>>,
    layout: &str,
    options: Options,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Out<'py>> {
    let c = with_covariates(correlator(&options)?, covariates);
    let a = data.as_array();
    match (layout, other.as_ref().map(|o| o.as_array())) {
        ("upper", None) => {
            let o = py.detach(|| c.upper(a)).map_err(core_err)?;
            Ok((array(py, o.corr), o.pvalues.map(|p| array(py, p))))
        }
        ("full" | "cross", b) => {
            let o = py.detach(|| match b {
                Some(b) => c.cross(a, b),
                None => c.full(a),
            });
            let o = o.map_err(core_err)?;
            Ok((array(py, o.corr), o.pvalues.map(|p| array(py, p))))
        }
        (l, _) => Err(PyValueError::new_err(format!(
            "unsupported layout '{l}' (full, cross, upper; upper takes one matrix)"
        ))),
    }
}

/// pair(x, y, options, covariates) -> (r, p | None, n_eff)
#[pyfunction]
#[pyo3(signature = (x, y, options, covariates=None))]
pub fn pair(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    options: Options,
    covariates: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<(f64, Option<f64>, usize)> {
    let c = with_covariates(correlator(&options)?, covariates);
    let (xa, ya) = (x.as_array(), y.as_array());
    let r = py.detach(|| match (xa.as_slice(), ya.as_slice()) {
        (Some(xs), Some(ys)) => c.pair(xs, ys),
        _ => c.pair(&xa.to_vec(), &ya.to_vec()),
    });
    let r = r.map_err(core_err)?;
    Ok((r.corr, r.pvalue, r.n_eff))
}
