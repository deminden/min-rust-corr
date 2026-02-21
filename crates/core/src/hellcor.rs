// Compute pairwise Hellinger correlations (HellCor) for matrix rows.
// References in comments are to the Hellinger correlation paper https://arxiv.org/abs/1810.10276

use ndarray::{Array2, ArrayBase, ArrayView1, Data, Ix2};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::hellinger::{dbeta, qbeta};

const DEFAULT_KMAX: usize = 20;
const DEFAULT_LMAX: usize = 20;
const DEFAULT_ALPHA: f64 = 6.0;
const SQRT2: f64 = 1.4142135623730951;

struct HellcorRow {
    u: Vec<f64>,
    t: Vec<f64>,
    w: Vec<f64>,
    leg: Vec<f64>,
    n: usize,
    valid: bool,
}

fn ranks_avg_ties(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(f64, usize)> = values
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && indexed[j].0 == indexed[j + 1].0 {
            j += 1;
        }
        let avg_rank = (i + j + 2) as f64 / 2.0;
        for k in i..=j {
            ranks[indexed[k].1] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

fn legendre_poly(x: f64, k: usize) -> f64 {
    // M_n = sqrt((2n+1)/2) * P_n, with P_n the Legendre polynomials on [-1,1].
    // Orthonormal on [0,1]: sqrt(2) * P_n(2x - 1), using (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}.
    match k {
        0 => 0.707106781186547,
        1 => 1.22474487139159 * x,
        2 => -0.790569415042095 + 2.37170824512628 * x.powi(2),
        3 => -2.80624304008046 * x + 4.67707173346743 * x.powi(3),
        4 => 0.795495128834866 - 7.95495128834866 * x.powi(2) + 9.28077650307344 * x.powi(4),
        5 => 4.39726477483446 * x - 20.5205689492275 * x.powi(3) + 18.4685120543048 * x.powi(5),
        6 => {
            -0.796721798998873 + 16.7311577789763 * x.powi(2) - 50.193473336929 * x.powi(4)
                + 36.8085471137479 * x.powi(6)
        }
        7 => {
            -5.99071547271275 * x + 53.9164392544148 * x.powi(3) - 118.616166359713 * x.powi(5)
                + 73.4290553655363 * x.powi(7)
        }
        8 => {
            0.797200454373381 - 28.6992163574417 * x.powi(2) + 157.845689965929 * x.powi(4)
                - 273.599195940944 * x.powi(6)
                + 146.570997825506 * x.powi(8)
        }
        9 => {
            7.58511879271573 * x - 111.248408959831 * x.powi(3) + 433.86879494334 * x.powi(5)
                - 619.812564204771 * x.powi(7)
                + 292.689266430031 * x.powi(9)
        }
        10 => {
            -0.797434890624405 + 43.8589189843422 * x.powi(2) - 380.110631197633 * x.powi(4)
                + 1140.3318935929 * x.powi(6)
                - 1384.68872793423 * x.powi(8)
                + 584.646351794454 * x.powi(10)
        }
        11 => {
            -9.17998960606603 * x + 198.899774798097 * x.powi(3) - 1193.39864878858 * x.powi(5)
                + 2898.25386134371 * x.powi(7)
                - 3059.26796475169 * x.powi(9)
                + 1168.0841319961 * x.powi(11)
        }
        12 => {
            0.797566730732873 - 62.2102049971641 * x.powi(2) + 777.627562464552 * x.powi(4)
                - 3525.2449498393 * x.powi(6)
                + 7176.39150503 * x.powi(8)
                - 6697.96540469467 * x.powi(10)
                + 2334.13945921178 * x.powi(12)
        }
        13 => {
            10.7751235804364 * x - 323.253707413091 * x.powi(3) + 2747.65651301127 * x.powi(5)
                - 9943.89976137412 * x.powi(7)
                + 17401.8245824047 * x.powi(9)
                - 14554.2532871021 * x.powi(11)
                + 4664.82477150709 * x.powi(13)
        }
        14 => {
            -0.797648110941312 + 83.7530516488378 * x.powi(2) - 1423.80187803024 * x.powi(4)
                + 9017.41189419154 * x.powi(6)
                - 27052.2356825746 * x.powi(8)
                + 41480.0947132811 * x.powi(10)
                - 31424.3141767281 * x.powi(12)
                + 9323.69761287537 * x.powi(14)
        }
        15 => {
            -12.37042008527 * x + 490.693330049044 * x.powi(3) - 5593.9039625591 * x.powi(5)
                + 27969.5198127955 * x.powi(7)
                - 71477.6617438108 * x.powi(9)
                + 97469.5387415601 * x.powi(11)
                - 67478.9114364647 * x.powi(13)
                + 18637.0326824522 * x.powi(15)
        }
        16 => {
            0.79770183004505 - 108.487448886127 * x.powi(2) + 2404.80511697581 * x.powi(4)
                - 20200.3629825968 * x.powi(6)
                + 82965.7765356655 * x.powi(8)
                - 184368.392301479 * x.powi(10)
                + 226270.299642724 * x.powi(12)
                - 144216.234937121 * x.powi(14)
                + 37255.8606920895 * x.powi(16)
        }
        17 => {
            13.9658239139855 * x - 707.601744975264 * x.powi(3) + 10401.7456511364 * x.powi(5)
                - 68354.3285646105 * x.powi(7)
                + 237341.41862712 * x.powi(9)
                - 466052.240213253 * x.powi(11)
                + 519827.498699398 * x.powi(13)
                - 306945.761136787 * x.powi(15)
                + 74479.4861581911 * x.powi(17)
        }
        18 => {
            -0.797739132849908 + 136.413391717334 * x.powi(2) - 3819.57496808536 * x.powi(4)
                + 40996.7713241162 * x.powi(6)
                - 219625.560664908 * x.powi(8)
                + 658876.681994724 * x.powi(10)
                - 1158025.68350588 * x.powi(12)
                + 1183476.79742909 * x.powi(14)
                - 650912.238585997 * x.powi(16)
                + 148901.492486993 * x.powi(18)
        }
        19 => {
            -15.5613022863318 * x + 980.362044038905 * x.powi(3) - 18038.6616103158 * x.powi(5)
                + 150322.180085965 * x.powi(7)
                - 676449.810386844 * x.powi(9)
                + 1783367.68192895 * x.powi(11)
                - 2835097.34050244 * x.powi(13)
                + 2673091.77818801 * x.powi(15)
                - 1375856.06230265 * x.powi(17)
                + 297699.849738001 * x.powi(19)
        }
        20 => {
            0.797766083041056 - 167.530877438622 * x.powi(2) + 5779.81527163245 * x.powi(4)
                - 77064.203621766 * x.powi(6)
                + 520183.37444692 * x.powi(8)
                - 2011375.71452809 * x.powi(10)
                + 4723685.39017961 * x.powi(12)
                - 6851939.2472935 * x.powi(14)
                + 5995446.84138182 * x.powi(16)
                - 2899758.60302127 * x.powi(18)
                + 595213.607988577 * x.powi(20)
        }
        _ => {
            let kf = k as f64;
            ((2.0 * kf + 1.0).sqrt()
                * ((2.0 * kf - 1.0).sqrt() * x * legendre_poly(x, k - 1)
                    - (kf - 1.0) * legendre_poly(x, k - 2) / (2.0 * kf - 3.0).sqrt()))
                / kf
        }
    }
}

fn prepare_row(row: ArrayView1<f64>, kmax: usize, alpha: f64) -> HellcorRow {
    let n = row.len();
    if row.iter().any(|v| !v.is_finite()) {
        return HellcorRow {
            u: vec![0.0; n],
            t: vec![0.0; n],
            w: vec![0.0; n],
            leg: vec![0.0; (kmax + 1) * n],
            n,
            valid: false,
        };
    }

    // Copula pseudo-observations: rank with average ties, then divide by (n + 1).
    let ranks = ranks_avg_ties(row.as_slice().expect("row is contiguous"));
    let mut u = vec![0.0; n];
    for (i, r) in ranks.iter().enumerate() {
        u[i] = r / (n as f64 + 1.0);
    }

    let mut t = vec![0.0; n];
    let mut w = vec![0.0; n];
    // Marginal transformation (Appendix B): qbeta + weights from dbeta.
    for i in 0..n {
        let ti = qbeta(u[i], alpha, alpha, true, false);
        let wi = dbeta(ti, alpha, alpha, false);
        if !ti.is_finite() || !wi.is_finite() {
            return HellcorRow {
                u,
                t,
                w,
                leg: vec![0.0; (kmax + 1) * n],
                n,
                valid: false,
            };
        }
        t[i] = ti;
        w[i] = wi;
    }

    let mut leg = vec![0.0; (kmax + 1) * n];
    // Compute b_k(û_ij) = sqrt(2) * LegendrePoly(2u - 1).
    for k in 0..=kmax {
        for i in 0..n {
            let x = 2.0 * u[i] - 1.0;
            leg[k * n + i] = SQRT2 * legendre_poly(x, k);
        }
    }

    HellcorRow {
        u,
        t,
        w,
        leg,
        n,
        valid: true,
    }
}

fn leg_value(row: &HellcorRow, k: usize, i: usize) -> f64 {
    row.leg[k * row.n + i]
}

fn etafunc(h2: f64) -> f64 {
    let b = 1.0 - h2;
    let b2 = b * b;
    let b4 = b2 * b2;
    2.0 * (-2.0 + b4 + (4.0 - 3.0 * b4).sqrt()).sqrt() / b2
}

fn write_vec(path: &str, values: &[f64]) -> io::Result<()> {
    let mut file = File::create(path)?;
    for v in values {
        writeln!(file, "{:.17e}", v)?;
    }
    Ok(())
}

fn write_matrix(path: &str, values: &[f64], rows: usize, cols: usize) -> io::Result<()> {
    let mut file = File::create(path)?;
    for r in 0..rows {
        for c in 0..cols {
            if c > 0 {
                write!(file, "\t")?;
            }
            write!(file, "{:.17e}", values[r * cols + c])?;
        }
        writeln!(file)?;
    }
    Ok(())
}

fn dump_debug(
    i: usize,
    j: usize,
    x: &HellcorRow,
    y: &HellcorRow,
    r1: &[f64],
    r2: &[f64],
    betahat: &[f64],
    kmax: usize,
    lmax: usize,
    alpha: f64,
) {
    let _ = write_vec(&format!("hellcor_debug_u1_{}_{}.tsv", i, j), &x.u);
    let _ = write_vec(&format!("hellcor_debug_u2_{}_{}.tsv", i, j), &y.u);
    let _ = write_vec(&format!("hellcor_debug_t1_{}_{}.tsv", i, j), &x.t);
    let _ = write_vec(&format!("hellcor_debug_t2_{}_{}.tsv", i, j), &y.t);
    let _ = write_vec(&format!("hellcor_debug_R1_{}_{}.tsv", i, j), r1);
    let _ = write_vec(&format!("hellcor_debug_R2_{}_{}.tsv", i, j), r2);
    let _ = write_matrix(
        &format!("hellcor_debug_betahat_{}_{}.tsv", i, j),
        betahat,
        kmax + 1,
        lmax + 1,
    );
    if let Ok(mut file) = File::create(&format!("hellcor_debug_meta_{}_{}.txt", i, j)) {
        let _ = writeln!(file, "i={}", i);
        let _ = writeln!(file, "j={}", j);
        let _ = writeln!(file, "n={}", x.n);
        let _ = writeln!(file, "kmax={}", kmax);
        let _ = writeln!(file, "lmax={}", lmax);
        let _ = writeln!(file, "alpha={:.17e}", alpha);
    }
}

fn hellcor_pair_impl(
    i: usize,
    j: usize,
    x: &HellcorRow,
    y: &HellcorRow,
    kmax: usize,
    lmax: usize,
    alpha: f64,
    debug_pair: Option<(usize, usize)>,
    debug_written: &AtomicBool,
) -> f64 {
    if !x.valid || !y.valid || x.n != y.n {
        return f64::NAN;
    }

    let n = x.n;
    if n <= 3 {
        return 0.0;
    }

    // sqrt{xi1}(t1) * sqrt{xi2}(t2) per Equ. (B.1).
    let mut weight = vec![0.0; n];
    for i in 0..n {
        weight[i] = (x.w[i] * y.w[i]).sqrt();
    }

    let mut r1 = vec![f64::INFINITY; n];
    let mut r2 = vec![f64::INFINITY; n];
    let mut ii1 = vec![usize::MAX; n];

    // dist[i][j] = ||T_i - T_j|| (exact O(n^2), matching C++ reference).
    for i in 0..n - 1 {
        for j in i + 1..n {
            let dx = x.t[i] - x.t[j];
            let dy = y.t[i] - y.t[j];
            let d = (dx * dx + dy * dy).sqrt();

            if d < r1[i] {
                r2[i] = r1[i];
                r1[i] = d;
                ii1[i] = j;
            } else if d < r2[i] {
                r2[i] = d;
            }

            if d < r1[j] {
                r2[j] = r1[j];
                r1[j] = d;
                ii1[j] = i;
            } else if d < r2[j] {
                r2[j] = d;
            }
        }
    }

    // Apply weights to nearest-neighbor radii.
    for i in 0..n {
        r1[i] *= weight[i];
        r2[i] *= weight[i];
    }

    let cte1 = 2.0 * ((n - 1) as f64).sqrt() / n as f64;
    let cte2 = 2.0 * ((n - 2) as f64).sqrt() / (n as f64 - 1.0);

    // Compute beta-hat (Equ. 5.2).
    let mut betahat = vec![0.0; (kmax + 1) * (lmax + 1)];
    for k in 0..=kmax {
        for l in 0..=lmax {
            let mut tmp = 0.0;
            for i in 0..n {
                tmp += r1[i] * leg_value(x, k, i) * leg_value(y, l, i);
            }
            betahat[k * (lmax + 1) + l] = cte1 * tmp;
        }
    }

    if let Some((di, dj)) = debug_pair {
        if di == i && dj == j && !debug_written.swap(true, Ordering::SeqCst) {
            dump_debug(i, j, x, y, &r1, &r2, &betahat, kmax, lmax, alpha);
        }
    }

    // Aterm[K][L] = sum_{k<=K,l<=L} beta-hat^2 (denominator of Equ. 5.3, no sqrt).
    let mut aterm = vec![0.0; (kmax + 1) * (lmax + 1)];
    for l in 0..=lmax {
        let mut tmp = 0.0;
        for k in 0..=kmax {
            let v = betahat[k * (lmax + 1) + l];
            tmp += v * v;
            aterm[k * (lmax + 1) + l] = tmp;
        }
    }
    for k in 0..=kmax {
        let mut tmp = 0.0;
        for l in 0..=lmax {
            tmp += aterm[k * (lmax + 1) + l];
            aterm[k * (lmax + 1) + l] = tmp;
        }
    }

    // Leave-one-out beta-hat in Appendix C.
    let mut betahatminusi = vec![0.0; n * (kmax + 1) * (lmax + 1)];
    for i in 0..n {
        let mut rminusi = r1.clone();
        for iprime in 0..n {
            if ii1[iprime] == i {
                rminusi[iprime] = r2[iprime];
            }
        }
        rminusi[i] = 0.0;
        for k in 0..=kmax {
            for l in 0..=lmax {
                let mut tmp = 0.0;
                for iprime in 0..n {
                    if iprime != i {
                        tmp += rminusi[iprime] * leg_value(x, k, iprime) * leg_value(y, l, iprime);
                    }
                }
                let idx = i * (kmax + 1) * (lmax + 1) + k * (lmax + 1) + l;
                betahatminusi[idx] = cte2 * tmp;
            }
        }
    }

    // Cross-validation selection for (Khat, Lhat).
    let mut max_val = f64::NEG_INFINITY;
    let mut khat = 0usize;
    let mut lhat = 0usize;

    for k in 0..=kmax {
        for l in 0..=lmax {
            let mut term = 0.0;
            for i in 0..n {
                let mut num = 0.0;
                let mut denom = 0.0;
                for kk in 0..=k {
                    for ll in 0..=l {
                        let idx = i * (kmax + 1) * (lmax + 1) + kk * (lmax + 1) + ll;
                        let b = betahatminusi[idx];
                        num += b * leg_value(x, kk, i) * leg_value(y, ll, i);
                        denom += b * b;
                    }
                }
                term += r1[i] * num / denom.sqrt();
            }
            term *= cte1;
            if term > max_val {
                max_val = term;
                khat = k;
                lhat = l;
            }
        }
    }

    if khat < 1 {
        khat = 1;
    }
    if lhat < 1 {
        lhat = 1;
    }

    // H^2 = 1 - beta_00 / sqrt(Aterm[Khat][Lhat]); then eta = etafunc(H^2).
    let denom = aterm[khat * (lmax + 1) + lhat].sqrt();
    let h2 = 1.0 - betahat[0] / denom;
    etafunc(h2)
}

pub fn hellcor_pair(x: &[f64], y: &[f64], alpha: f64) -> f64 {
    if x.len() != y.len() {
        return f64::NAN;
    }

    let kmax = DEFAULT_KMAX;
    let lmax = DEFAULT_LMAX;
    let max_k = kmax.max(lmax);

    let x_row = prepare_row(ArrayView1::from(x), max_k, alpha);
    let y_row = prepare_row(ArrayView1::from(y), max_k, alpha);

    let debug_written = AtomicBool::new(false);
    hellcor_pair_impl(
        0,
        1,
        &x_row,
        &y_row,
        kmax,
        lmax,
        alpha,
        None,
        &debug_written,
    )
}

fn parse_debug_pair() -> Option<(usize, usize)> {
    std::env::var("HELLCOR_DEBUG_PAIR").ok().and_then(|value| {
        let parts: Vec<&str> = value.split(',').collect();
        if parts.len() != 2 {
            return None;
        }
        let i = parts[0].trim().parse::<usize>().ok()?;
        let j = parts[1].trim().parse::<usize>().ok()?;
        Some((i, j))
    })
}

fn correlation_matrix_impl<S>(data: &ArrayBase<S, Ix2>, alpha: f64) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (n_rows, n_cols) = data.dim();
    if n_rows == 0 || n_cols == 0 {
        return Array2::<f64>::zeros((n_rows, n_rows));
    }

    let kmax = DEFAULT_KMAX;
    let lmax = DEFAULT_LMAX;
    let max_k = kmax.max(lmax);

    let rows: Vec<HellcorRow> = (0..n_rows)
        .into_par_iter()
        .map(|i| prepare_row(data.row(i), max_k, alpha))
        .collect();

    let debug_pair = parse_debug_pair();
    let debug_written = AtomicBool::new(false);

    let row_results: Vec<Vec<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n_rows];
            row[i] = 1.0;
            for j in i + 1..n_rows {
                row[j] = hellcor_pair_impl(
                    i,
                    j,
                    &rows[i],
                    &rows[j],
                    kmax,
                    lmax,
                    alpha,
                    debug_pair,
                    &debug_written,
                );
            }
            row
        })
        .collect();

    let mut corr = Array2::<f64>::from_elem((n_rows, n_rows), f64::NAN);
    for i in 0..n_rows {
        for j in i..n_rows {
            let val = row_results[i][j];
            corr[[i, j]] = val;
            corr[[j, i]] = val;
        }
    }

    corr
}

pub fn correlation_matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    correlation_matrix_impl(data, DEFAULT_ALPHA)
}

pub fn correlation_matrix_with_alpha<S>(data: &ArrayBase<S, Ix2>, alpha: f64) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    correlation_matrix_impl(data, alpha)
}

pub fn matrix<S>(data: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    correlation_matrix(data)
}

pub fn matrix_with_alpha<S>(data: &ArrayBase<S, Ix2>, alpha: f64) -> Array2<f64>
where
    S: Data<Elem = f64> + Sync,
{
    correlation_matrix_with_alpha(data, alpha)
}

fn correlation_cross_matrix_impl<S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
    alpha: f64,
) -> Array2<f64>
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = f64> + Sync,
{
    let (lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();
    assert_eq!(
        lhs_cols, rhs_cols,
        "Hellcor cross-correlation requires equal sample count in both matrices"
    );

    if lhs_rows == 0 || rhs_rows == 0 || lhs_cols == 0 {
        return Array2::<f64>::zeros((lhs_rows, rhs_rows));
    }

    let kmax = DEFAULT_KMAX;
    let lmax = DEFAULT_LMAX;
    let max_k = kmax.max(lmax);

    let lhs_prepared: Vec<HellcorRow> = (0..lhs_rows)
        .into_par_iter()
        .map(|i| prepare_row(lhs.row(i), max_k, alpha))
        .collect();
    let rhs_prepared: Vec<HellcorRow> = (0..rhs_rows)
        .into_par_iter()
        .map(|i| prepare_row(rhs.row(i), max_k, alpha))
        .collect();

    let debug_pair = parse_debug_pair();
    let debug_written = AtomicBool::new(false);

    let row_results: Vec<Vec<f64>> = (0..lhs_rows)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; rhs_rows];
            for j in 0..rhs_rows {
                row[j] = hellcor_pair_impl(
                    i,
                    j,
                    &lhs_prepared[i],
                    &rhs_prepared[j],
                    kmax,
                    lmax,
                    alpha,
                    debug_pair,
                    &debug_written,
                );
            }
            row
        })
        .collect();

    let mut corr = Array2::<f64>::from_elem((lhs_rows, rhs_rows), f64::NAN);
    for i in 0..lhs_rows {
        for j in 0..rhs_rows {
            corr[[i, j]] = row_results[i][j];
        }
    }
    corr
}

pub fn correlation_cross_matrix<S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Array2<f64>
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = f64> + Sync,
{
    correlation_cross_matrix_impl(lhs, rhs, DEFAULT_ALPHA)
}

pub fn correlation_cross_matrix_with_alpha<S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
    alpha: f64,
) -> Array2<f64>
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = f64> + Sync,
{
    correlation_cross_matrix_impl(lhs, rhs, alpha)
}

fn correlation_upper_triangle_impl<S>(data: &ArrayBase<S, Ix2>, alpha: f64) -> Vec<f64>
where
    S: Data<Elem = f64> + Sync,
{
    let (n_rows, n_cols) = data.dim();
    if n_rows == 0 || n_cols == 0 {
        return Vec::new();
    }

    let kmax = DEFAULT_KMAX;
    let lmax = DEFAULT_LMAX;
    let max_k = kmax.max(lmax);

    let rows: Vec<HellcorRow> = (0..n_rows)
        .into_par_iter()
        .map(|i| prepare_row(data.row(i), max_k, alpha))
        .collect();

    let debug_pair = parse_debug_pair();
    let debug_written = AtomicBool::new(false);

    let row_results: Vec<Vec<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f64::NAN; n_rows - i];
            row[0] = 1.0;
            for j in i + 1..n_rows {
                row[j - i] = hellcor_pair_impl(
                    i,
                    j,
                    &rows[i],
                    &rows[j],
                    kmax,
                    lmax,
                    alpha,
                    debug_pair,
                    &debug_written,
                );
            }
            row
        })
        .collect();

    let mut packed = Vec::with_capacity(crate::upper::upper_triangular_len(n_rows));
    for row in row_results {
        packed.extend_from_slice(&row);
    }
    packed
}

pub fn correlation_upper_triangle<S>(data: &ArrayBase<S, Ix2>) -> Vec<f64>
where
    S: Data<Elem = f64> + Sync,
{
    correlation_upper_triangle_impl(data, DEFAULT_ALPHA)
}

pub fn correlation_upper_triangle_with_alpha<S>(data: &ArrayBase<S, Ix2>, alpha: f64) -> Vec<f64>
where
    S: Data<Elem = f64> + Sync,
{
    correlation_upper_triangle_impl(data, alpha)
}
