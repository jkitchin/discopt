use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

/// Create a batch of float64 arrays with shape [batch_size, n_vars].
#[pyfunction]
fn create_batch<'py>(
    py: Python<'py>,
    batch_size: usize,
    n_vars: usize,
) -> Bound<'py, PyArray2<f64>> {
    let mut arr = Array2::<f64>::zeros((batch_size, n_vars));
    for i in 0..batch_size {
        for j in 0..n_vars {
            arr[[i, j]] = (i * n_vars + j) as f64 * 0.001;
        }
    }
    arr.into_pyarray(py)
}

/// Create a batch of float32 arrays with shape [batch_size, n_vars].
#[pyfunction]
fn create_batch_f32<'py>(
    py: Python<'py>,
    batch_size: usize,
    n_vars: usize,
) -> Bound<'py, PyArray2<f32>> {
    let mut arr = Array2::<f32>::zeros((batch_size, n_vars));
    for i in 0..batch_size {
        for j in 0..n_vars {
            arr[[i, j]] = (i * n_vars + j) as f32 * 0.001;
        }
    }
    arr.into_pyarray(py)
}

/// Receive a float64 numpy array and return the sum of all elements.
#[pyfunction]
fn sum_array(arr: PyReadonlyArray2<f64>) -> f64 {
    arr.as_array().sum()
}

/// Receive a float32 numpy array and return the sum of all elements.
#[pyfunction]
fn sum_array_f32(arr: PyReadonlyArray2<f32>) -> f32 {
    arr.as_array().sum()
}

/// Return the data pointer of a float64 numpy array as an integer.
#[pyfunction]
fn data_pointer(arr: PyReadonlyArray2<f64>) -> usize {
    arr.as_array().as_ptr() as usize
}

/// Return the data pointer of a float32 numpy array as an integer.
#[pyfunction]
fn data_pointer_f32(arr: PyReadonlyArray2<f32>) -> usize {
    arr.as_array().as_ptr() as usize
}

/// Create a float64 batch and also return its data pointer.
#[pyfunction]
fn create_batch_with_ptr<'py>(
    py: Python<'py>,
    batch_size: usize,
    n_vars: usize,
) -> (Bound<'py, PyArray2<f64>>, usize) {
    let arr = Array2::<f64>::zeros((batch_size, n_vars));
    let pyarr = arr.into_pyarray(py);
    let ptr = pyarr.as_raw_array().as_ptr() as usize;
    (pyarr, ptr)
}

/// Create a float32 batch and also return its data pointer.
#[pyfunction]
fn create_batch_f32_with_ptr<'py>(
    py: Python<'py>,
    batch_size: usize,
    n_vars: usize,
) -> (Bound<'py, PyArray2<f32>>, usize) {
    let arr = Array2::<f32>::zeros((batch_size, n_vars));
    let pyarr = arr.into_pyarray(py);
    let ptr = pyarr.as_raw_array().as_ptr() as usize;
    (pyarr, ptr)
}

#[pymodule]
fn discopt_spike(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_batch, m)?)?;
    m.add_function(wrap_pyfunction!(create_batch_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array_f32, m)?)?;
    m.add_function(wrap_pyfunction!(data_pointer, m)?)?;
    m.add_function(wrap_pyfunction!(data_pointer_f32, m)?)?;
    m.add_function(wrap_pyfunction!(create_batch_with_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(create_batch_f32_with_ptr, m)?)?;
    Ok(())
}
