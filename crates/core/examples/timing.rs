//! Quick wall-clock timing: `cargo run --release --example timing -- <method> <rows> <cols> [pvalues]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use mincorr::{Correlator, Method, PvalueSpec};
use ndarray::Array2;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let method = Method::from_name(a.get(1).map_or("pearson", |s| s)).unwrap();
    let rows: usize = a.get(2).map_or(1000, |s| s.parse().unwrap());
    let cols: usize = a.get(3).map_or(77, |s| s.parse().unwrap());
    let with_p = a.get(4).is_some();
    let mut s = 12345u64;
    let data = Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / (1u64 << 53) as f64
    });
    let pv = match (with_p, &method) {
        (false, _) => PvalueSpec::None,
        (true, Method::Hellcor(_)) => PvalueSpec::MonteCarlo { samples: 10_000, seed: 42 },
        (true, _) => PvalueSpec::Analytic,
    };
    let t = Instant::now();
    let out = Correlator::new(method.clone()).pvalues(pv).full(data.view()).unwrap();
    let el = t.elapsed().as_secs_f64();
    println!(
        "{} {rows}x{cols} p={with_p}: {el:.3} s (threads={}) checksum={:.6}",
        method.name(),
        rayon::current_num_threads(),
        out.corr.sum()
    );
}
