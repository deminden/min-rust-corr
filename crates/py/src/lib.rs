use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn pair_matrix_from_slices(x: &[f64], y: &[f64]) -> Array2<f64> {
    let n = x.len();
    let mut flat = Vec::with_capacity(n * 2);
    flat.extend_from_slice(x);
    flat.extend_from_slice(y);
    Array2::from_shape_vec((2, n), flat).expect("pair matrix shape mismatch")
}

fn owned_pair(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let x_owned = x.as_array().to_owned();
    let y_owned = y.as_array().to_owned();

    if x_owned.len() != y_owned.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }

    let x_slice = x_owned.as_slice().expect("owned array is contiguous");
    let y_slice = y_owned.as_slice().expect("owned array is contiguous");

    Ok((x_slice.to_vec(), y_slice.to_vec()))
}

#[pyfunction]
#[pyo3(signature = (x, y, alpha = 6.0))]
fn hellcor_pair(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    alpha: f64,
) -> PyResult<f64> {
    let (x_vals, y_vals) = owned_pair(x, y)?;
    Ok(py.detach(|| mincorr_core::hellcor_pair(&x_vals, &y_vals, alpha)))
}

#[pyfunction]
fn pearson_pair(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let (x_vals, y_vals) = owned_pair(x, y)?;
    let data = pair_matrix_from_slices(&x_vals, &y_vals);
    let corr = py.detach(|| mincorr_core::pearson::correlation_matrix(&data));
    Ok(corr[[0, 1]])
}

#[pyfunction]
fn spearman_pair(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let (x_vals, y_vals) = owned_pair(x, y)?;
    let data = pair_matrix_from_slices(&x_vals, &y_vals);
    let corr = py.detach(|| mincorr_core::spearman::correlation_matrix(&data));
    Ok(corr[[0, 1]])
}

#[pyfunction]
fn kendall_pair(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let (x_vals, y_vals) = owned_pair(x, y)?;
    let data = pair_matrix_from_slices(&x_vals, &y_vals);
    let corr = py.detach(|| mincorr_core::kendall::correlation_matrix(&data));
    Ok(corr[[0, 1]])
}

#[pyfunction]
fn bicor_pair(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let (x_vals, y_vals) = owned_pair(x, y)?;
    let data = pair_matrix_from_slices(&x_vals, &y_vals);
    let corr = py.detach(|| mincorr_core::bicor::correlation_matrix(&data));
    Ok(corr[[0, 1]])
}

#[pyfunction]
fn pearson_matrix(py: Python<'_>, data: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray2<f64>>> {
    let data_owned = data.as_array().to_owned();
    let corr = py.detach(|| mincorr_core::pearson::correlation_matrix(&data_owned));
    Ok(corr.into_pyarray(py).into())
}

#[pyfunction]
fn spearman_matrix(py: Python<'_>, data: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray2<f64>>> {
    let data_owned = data.as_array().to_owned();
    let corr = py.detach(|| mincorr_core::spearman::correlation_matrix(&data_owned));
    Ok(corr.into_pyarray(py).into())
}

#[pyfunction]
fn kendall_matrix(py: Python<'_>, data: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray2<f64>>> {
    let data_owned = data.as_array().to_owned();
    let corr = py.detach(|| mincorr_core::kendall::correlation_matrix(&data_owned));
    Ok(corr.into_pyarray(py).into())
}

#[pyfunction]
fn bicor_matrix(py: Python<'_>, data: PyReadonlyArray2<'_, f64>) -> PyResult<Py<PyArray2<f64>>> {
    let data_owned = data.as_array().to_owned();
    let corr = py.detach(|| mincorr_core::bicor::correlation_matrix(&data_owned));
    Ok(corr.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (data, alpha = 6.0))]
fn hellcor_matrix(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    alpha: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let data_owned = data.as_array().to_owned();
    let corr = py.detach(|| mincorr_core::hellcor::correlation_matrix_with_alpha(&data_owned, alpha));
    Ok(corr.into_pyarray(py).into())
}

#[pymodule]
fn mincorr(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hellcor_pair, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_pair, m)?)?;
    m.add_function(wrap_pyfunction!(spearman_pair, m)?)?;
    m.add_function(wrap_pyfunction!(kendall_pair, m)?)?;
    m.add_function(wrap_pyfunction!(bicor_pair, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(spearman_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(kendall_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(bicor_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(hellcor_matrix, m)?)?;
    m.add("__all__", vec![
        "hellcor_pair",
        "pearson_pair",
        "spearman_pair",
        "kendall_pair",
        "bicor_pair",
        "pearson_matrix",
        "spearman_matrix",
        "kendall_matrix",
        "bicor_matrix",
        "hellcor_matrix",
    ])?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
