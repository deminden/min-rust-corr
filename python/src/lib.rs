//! Python extension `mincorr._core`: two generic entry points; the public
//! Python API (including all v0.2 function names) lives in
//! `python/mincorr/__init__.py`.

mod convert;
mod dispatch;

use pyo3::prelude::*;

/// The `_core` extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dispatch::compute, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch::pair, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
