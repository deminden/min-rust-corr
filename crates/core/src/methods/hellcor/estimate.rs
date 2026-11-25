//! The Hellinger-correlation estimator for one pair of half-rank vectors
//! (Geenens & Lafaye de Micheaux 2022, §5 and Appendices B–C).

use super::coeffs::{PairData, inverse_neighbors, mise_cross, s_matrix};
use super::eta::eta_from_affinity;
use super::marginal::Tables;
use super::neighbors::Neighbors;
use super::select::{NormalizedWork, best, normalized_table, prefix2d};
use crate::options::{HellcorCriterion, HellcorParams};

/// Reusable buffers for one thread.
pub struct Work {
    tx: Vec<f64>,
    ty: Vec<f64>,
    nb: Neighbors,
    r: Vec<f64>,
    rp: Vec<f64>,
    bx: Vec<f64>,
    by: Vec<f64>,
    s: Vec<f64>,
    m: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    start: Vec<u32>,
    list: Vec<u32>,
    norm: NormalizedWork,
}

impl Default for Work {
    fn default() -> Self {
        Self {
            tx: vec![],
            ty: vec![],
            nb: Neighbors::default(),
            r: vec![],
            rp: vec![],
            bx: vec![],
            by: vec![],
            s: vec![],
            m: vec![],
            u: vec![],
            v: vec![],
            start: vec![],
            list: vec![],
            norm: NormalizedWork { beta: vec![], num: vec![], den: vec![] },
        }
    }
}

/// Estimate η̂ from half-rank indices `hx`, `hy` (same length n = tab.n).
/// Returns 0.0 for n ≤ 3, as the reference implementation does.
pub fn estimate(hx: &[u32], hy: &[u32], tab: &Tables, p: &HellcorParams, w: &mut Work) -> f64 {
    let n = hx.len();
    if n <= 3 {
        return 0.0;
    }
    // η is symmetric when kmax = lmax; fix the argument order so that
    // η(x, y) and η(y, x) are bit-identical despite summation order.
    let (hx, hy) = if p.kmax == p.lmax && hy < hx { (hy, hx) } else { (hx, hy) };
    let (k1, l1) = (p.kmax + 1, p.lmax + 1);
    // Transformed points, nearest neighbours and weighted radii.
    w.tx.clear();
    w.tx.extend(hx.iter().map(|&h| tab.t[h as usize]));
    w.ty.clear();
    w.ty.extend(hy.iter().map(|&h| tab.t[h as usize]));
    w.nb.compute(&w.tx, &w.ty);
    w.r.clear();
    w.rp.clear();
    for i in 0..n {
        let wt = tab.w[hx[i] as usize] * tab.w[hy[i] as usize];
        w.r.push(w.nb.d1[i].sqrt() * wt);
        w.rp.push(w.nb.d2[i].sqrt() * wt);
    }
    w.bx.clear();
    w.by.clear();
    for i in 0..n {
        w.bx.extend_from_slice(&tab.b(hx[i])[..k1]);
        w.by.extend_from_slice(&tab.b(hy[i])[..l1]);
    }
    let nf = n as f64;
    let c1 = 2.0 * (nf - 1.0).sqrt() / nf;
    let c2 = 2.0 * (nf - 2.0).sqrt() / (nf - 1.0);
    let d = PairData { r: &w.r, rp: &w.rp, nn: &w.nb.nn, bx: &w.bx, by: &w.by, k1, l1 };
    s_matrix(&d, &mut w.s);
    let (kh, lh) = match p.fixed_orders {
        Some((k, l)) => (k.min(p.kmax), l.min(p.lmax)),
        None => select(
            &d,
            &w.s,
            c1,
            c2,
            p.criterion,
            &mut w.m,
            &mut w.u,
            &mut w.v,
            &mut w.start,
            &mut w.list,
            &mut w.norm,
        ),
    };
    // Normalised affinity: β00 / ‖β_{≤K̂,≤L̂}‖ (c1 cancels).
    let mut ss = 0.0;
    for k in 0..=kh {
        for l in 0..=lh {
            let b = w.s[k * l1 + l];
            ss += b * b;
        }
    }
    eta_from_affinity(w.s[0] / ss.sqrt())
}

#[allow(clippy::too_many_arguments)]
fn select(
    d: &PairData<'_>,
    s: &[f64],
    c1: f64,
    c2: f64,
    crit: HellcorCriterion,
    m: &mut Vec<f64>,
    u: &mut Vec<f64>,
    v: &mut Vec<f64>,
    start: &mut Vec<u32>,
    list: &mut Vec<u32>,
    norm: &mut NormalizedWork,
) -> (usize, usize) {
    let (k1, l1) = (d.k1, d.l1);
    match crit {
        HellcorCriterion::Mise => {
            // A²(K,L) − 2B(K,L) with A² = c1²·ΣS², B = c1·ΣM.
            mise_cross(d, s, c2, m, u, v);
            prefix2d(m, k1, l1);
            u.clear();
            u.extend(s.iter().map(|x| c1 * c1 * x * x));
            prefix2d(u, k1, l1);
            for (a, b) in u.iter_mut().zip(m.iter()) {
                *a -= 2.0 * c1 * b;
            }
            best(u, k1, l1, |a, b| a < b)
        }
        HellcorCriterion::NormalizedLoo => {
            inverse_neighbors(d.nn, start, list);
            normalized_table(d, s, c2, start, list, norm, m);
            best(m, k1, l1, |a, b| a > b)
        }
    }
}
