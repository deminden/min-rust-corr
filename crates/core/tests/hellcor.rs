//! Hellinger correlation: invariances, sanity and Monte-Carlo p-values.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
use mincorr::{Correlator, HellcorParams, Method, PvalueSpec};

fn hell() -> Correlator {
    Correlator::new(Method::Hellcor(HellcorParams::default()))
}

/// Deterministic pseudo-random normals (Box–Muller on an LCG).
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut u = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n).map(|_| (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()).collect()
}

#[test]
fn invariances() {
    let x = normals(77, 1);
    let e = normals(77, 2);
    let y: Vec<f64> = x.iter().zip(&e).map(|(a, b)| a + 0.7 * b).collect();
    let base = hell().pair(&x, &y).unwrap().corr;
    // Strictly increasing transforms of either variable.
    let xt: Vec<f64> = x.iter().map(|v| v.exp()).collect();
    let yt: Vec<f64> = y.iter().map(|v| v * v * v + 2.0).collect();
    assert_eq!(hell().pair(&xt, &yt).unwrap().corr, base);
    // Swapping x and y (kmax = lmax).
    assert_eq!(hell().pair(&y, &x).unwrap().corr, base);
    // Permuting the samples.
    let perm: Vec<usize> = (0..77).map(|i| (i * 31) % 77).collect();
    let xp: Vec<f64> = perm.iter().map(|&i| x[i]).collect();
    let yp: Vec<f64> = perm.iter().map(|&i| y[i]).collect();
    // (summation order changes, so equality holds up to rounding)
    let d = (hell().pair(&xp, &yp).unwrap().corr - base).abs();
    assert!(d < 1e-12, "permutation changed eta by {d:e}");
}

#[test]
fn sanity() {
    let x = normals(200, 3);
    let noise = normals(200, 4);
    let indep = hell().pair(&x, &noise).unwrap().corr;
    let parab: Vec<f64> = x.iter().zip(&noise).map(|(a, b)| a * a + 0.05 * b).collect();
    let dep = hell().pair(&x, &parab).unwrap().corr;
    println!("independent η = {indep:.3}, parabola η = {dep:.3}");
    assert!((0.0..=1.0).contains(&indep) && (0.0..=1.0).contains(&dep));
    assert!(dep > 0.8 && indep < 0.4, "indep {indep} dep {dep}");
    // n <= 3 returns 0 with NaN p-value.
    let r = hell()
        .pvalues(PvalueSpec::MonteCarlo { samples: 50, seed: 1 })
        .pair(&[1.0, 2.0, 3.0], &[3.0, 1.0, 2.0]);
    let r = r.unwrap();
    assert_eq!(r.corr, 0.0);
    assert!(r.pvalue.unwrap().is_nan());
}

#[test]
fn pvalues_power_and_resolution() {
    let x = normals(60, 5);
    let e = normals(60, 6);
    let y: Vec<f64> = x.iter().zip(&e).map(|(a, b)| a + 0.3 * b).collect();
    let c = hell().pvalues(PvalueSpec::MonteCarlo { samples: 199, seed: 42 });
    let p_dep = c.pair(&x, &y).unwrap().pvalue.unwrap();
    assert!((p_dep - 1.0 / 200.0).abs() < 1e-15, "p {p_dep}");
    let p_ind = c.pair(&x, &e).unwrap().pvalue.unwrap();
    assert!(p_ind > 0.01, "p {p_ind}");
}

#[test]
#[ignore = "statistical, slow: run in the nightly job"]
fn pvalues_uniform_under_independence() {
    let c = hell().pvalues(PvalueSpec::MonteCarlo { samples: 499, seed: 9 });
    let mut ps: Vec<f64> = (0..300)
        .map(|k| c.pair(&normals(40, 1000 + k), &normals(40, 5000 + k)).unwrap().pvalue.unwrap())
        .collect();
    ps.sort_by(f64::total_cmp);
    let n = ps.len() as f64;
    let d =
        ps.iter().enumerate().map(|(i, p)| ((i as f64 + 1.0) / n - p).abs().max((p - i as f64 / n).abs()));
    let ks = d.fold(0.0, f64::max);
    assert!(ks < 1.63 / n.sqrt(), "KS statistic {ks}");
}
