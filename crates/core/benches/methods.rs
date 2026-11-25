//! Criterion benchmarks: every method on (rows × obs) ∈ {500×77, 3000×77,
//! 2000×600} (Hellcor on 500×77 only; its 3000×77 cost is reported from
//! examples/timing.rs in docs/benchmarks.md).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mincorr::{Correlator, Method};
use ndarray::Array2;
use std::hint::black_box;
use std::time::Duration;

fn data(rows: usize, cols: usize) -> Array2<f64> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    })
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("full");
    g.sample_size(10).measurement_time(Duration::from_secs(10));
    for (rows, cols) in [(500, 77), (3000, 77), (2000, 600)] {
        let x = data(rows, cols);
        for name in ["pearson", "spearman", "bicor", "kendall", "hellcor"] {
            if name == "hellcor" && rows > 500 {
                continue;
            }
            let corr = Correlator::new(Method::from_name(name).unwrap());
            g.bench_with_input(BenchmarkId::new(name, format!("{rows}x{cols}")), &x, |b, x| {
                b.iter(|| black_box(corr.full(x.view()).unwrap()))
            });
        }
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
