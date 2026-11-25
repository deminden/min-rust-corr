//! Monte-Carlo null distribution of η̂ under independence.
//!
//! With continuous margins the statistic depends only on ranks, so the
//! null depends only on (n, α, kmax, lmax): x-ranks are fixed to 1..n and
//! y-ranks are a uniformly random permutation. Replicate b uses its own
//! ChaCha8 stream (`set_stream(b)`) of the user seed, so the result does
//! not depend on the number of threads.

use super::estimate::{Work, estimate};
use super::marginal::Tables;
use crate::options::HellcorParams;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};
use rayon::prelude::*;

/// Uniform integer in [0, n) by Lemire's multiply-and-reject method.
fn below(rng: &mut ChaCha8Rng, n: u64) -> u64 {
    let mut m = u128::from(rng.next_u64()) * u128::from(n);
    if (m as u64) < n {
        let threshold = n.wrapping_neg() % n;
        while (m as u64) < threshold {
            m = u128::from(rng.next_u64()) * u128::from(n);
        }
    }
    (m >> 64) as u64
}

/// Sorted null sample of size `samples` for sample size `tab.n`.
pub fn null_distribution(tab: &Tables, p: &HellcorParams, samples: usize, seed: u64) -> Vec<f64> {
    let n = tab.n;
    let hx: Vec<u32> = (0..n as u32).map(|r| 2 * r).collect();
    let mut null: Vec<f64> = (0..samples)
        .into_par_iter()
        .map_init(
            || (Work::default(), hx.clone()),
            |(w, hy), b| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                rng.set_stream(b as u64);
                hy.copy_from_slice(&hx);
                for i in (1..n).rev() {
                    let j = below(&mut rng, i as u64 + 1) as usize;
                    hy.swap(i, j);
                }
                estimate(&hx, hy, tab, p, w)
            },
        )
        .collect();
    null.sort_unstable_by(f64::total_cmp);
    null
}
