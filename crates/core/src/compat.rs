//! v0.2 function names, kept as thin deprecated wrappers over
//! [`crate::Correlator`]. Errors (shape mismatch, invalid parameters) are
//! reported as NaN-filled outputs instead of panics; use the
//! [`crate::Correlator`] API to get proper `Result`s.
#![allow(deprecated)]

use crate::{Correlator, Method, PvalueSpec};
use ndarray::{Array2, ArrayBase, Data, Ix2};

fn nan2(r: usize, c: usize) -> Array2<f64> {
    Array2::from_elem((r, c), f64::NAN)
}

pub(crate) fn full<S: Data<Elem = f64>>(
    m: Method,
    p: PvalueSpec,
    d: &ArrayBase<S, Ix2>,
) -> (Array2<f64>, Array2<f64>) {
    let n = d.nrows();
    match Correlator::new(m).pvalues(p).full(d.view()) {
        Ok(o) => (o.corr, o.pvalues.unwrap_or_else(|| nan2(0, 0))),
        Err(_) => (nan2(n, n), nan2(n, n)),
    }
}

pub(crate) fn cross<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    m: Method,
    p: PvalueSpec,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> (Array2<f64>, Array2<f64>) {
    let (r, c) = (a.nrows(), b.nrows());
    match Correlator::new(m).pvalues(p).cross(a.view(), b.view()) {
        Ok(o) => (o.corr, o.pvalues.unwrap_or_else(|| nan2(0, 0))),
        Err(_) => (nan2(r, c), nan2(r, c)),
    }
}

pub(crate) fn upper<S: Data<Elem = f64>>(m: Method, d: &ArrayBase<S, Ix2>) -> Vec<f64> {
    let n = d.nrows();
    Correlator::new(m).upper(d.view()).map(|o| o.corr).unwrap_or_else(|_| vec![f64::NAN; n * (n + 1) / 2])
}

macro_rules! legacy_module {
    ($name:ident, $method:expr, $pv:expr $(, $extra:item)*) => {
        #[doc = concat!("v0.2 `", stringify!($name), "` functions (deprecated).")]
        pub mod $name {
            use super::*;
            /// Full correlation matrix.
            #[deprecated(since = "0.3.0", note = "use Correlator::full")]
            pub fn correlation_matrix<S: Data<Elem = f64>>(d: &ArrayBase<S, Ix2>) -> Array2<f64> {
                full($method, PvalueSpec::None, d).0
            }
            /// Alias of `correlation_matrix`.
            #[deprecated(since = "0.3.0", note = "use Correlator::full")]
            pub fn matrix<S: Data<Elem = f64>>(d: &ArrayBase<S, Ix2>) -> Array2<f64> {
                full($method, PvalueSpec::None, d).0
            }
            /// Cross matrix.
            #[deprecated(since = "0.3.0", note = "use Correlator::cross")]
            pub fn correlation_cross_matrix<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
                a: &ArrayBase<S1, Ix2>,
                b: &ArrayBase<S2, Ix2>,
            ) -> Array2<f64> {
                cross($method, PvalueSpec::None, a, b).0
            }
            /// Packed upper triangle.
            #[deprecated(since = "0.3.0", note = "use Correlator::upper")]
            pub fn correlation_upper_triangle<S: Data<Elem = f64>>(d: &ArrayBase<S, Ix2>) -> Vec<f64> {
                upper($method, d)
            }
            /// Full matrix with p-values.
            #[deprecated(since = "0.3.0", note = "use Correlator::full with pvalues")]
            pub fn correlation_matrix_with_pvalues<S: Data<Elem = f64>>(d: &ArrayBase<S, Ix2>) -> (Array2<f64>, Array2<f64>) {
                full($method, $pv, d)
            }
            /// Cross matrix with p-values.
            #[deprecated(since = "0.3.0", note = "use Correlator::cross with pvalues")]
            pub fn correlation_cross_matrix_with_pvalues<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
                a: &ArrayBase<S1, Ix2>,
                b: &ArrayBase<S2, Ix2>,
            ) -> (Array2<f64>, Array2<f64>) {
                cross($method, $pv, a, b)
            }
            $($extra)*
        }
    };
}

legacy_module!(pearson, Method::Pearson, PvalueSpec::Analytic);
legacy_module!(spearman, Method::Spearman, PvalueSpec::Analytic);
legacy_module!(kendall, Method::Kendall, PvalueSpec::Analytic);
legacy_module!(bicor, Method::Bicor(Default::default()), PvalueSpec::Analytic);
legacy_module!(
    hellcor,
    Method::Hellcor(Default::default()),
    PvalueSpec::MonteCarlo { samples: 10_000, seed: 42 },
    pub use super::hellcor_alpha::with_alpha::*;
);

mod hellcor_alpha;
pub use hellcor_alpha::hellcor_pair;
