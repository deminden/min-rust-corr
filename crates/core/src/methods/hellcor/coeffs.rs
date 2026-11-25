//! Coefficient matrices of the Legendre expansion of √c and the
//! leave-one-out quantities needed by cross-validation, all in
//! O(n·K·L) per pair.
//!
//! With φ_i = b(U1_i) b(U2_i)ᵀ and S = Σ_j R_j φ_j (so β̂ = c1·S), the
//! leave-one-out coefficients are
//! β⁽⁻ⁱ⁾ = c2·[S − R_i φ_i − Σ_{j: nn_j = i} (R_j − R'_j) φ_j],
//! because removing point i only changes the radius of the points whose
//! nearest neighbour was i (they fall back to their second nearest).

/// Dense K1 × L1 matrix stored row-major.
pub type Mat = Vec<f64>;

/// Inputs for one pair, gathered into contiguous arrays.
pub struct PairData<'a> {
    /// Radii R_i (nearest-neighbour distance × weights).
    pub r: &'a [f64],
    /// Radii R'_i (second nearest).
    pub rp: &'a [f64],
    /// Nearest-neighbour index of each point.
    pub nn: &'a [u32],
    /// Basis of the first variable, n × k1.
    pub bx: &'a [f64],
    /// Basis of the second variable, n × l1.
    pub by: &'a [f64],
    /// Number of first-variable basis functions.
    pub k1: usize,
    /// Number of second-variable basis functions.
    pub l1: usize,
}

impl PairData<'_> {
    fn n(&self) -> usize {
        self.r.len()
    }
    fn bx(&self, i: usize) -> &[f64] {
        &self.bx[i * self.k1..(i + 1) * self.k1]
    }
    fn by(&self, i: usize) -> &[f64] {
        &self.by[i * self.l1..(i + 1) * self.l1]
    }
}

/// out += c · u vᵀ.
#[inline]
fn add_outer(out: &mut [f64], c: f64, u: &[f64], v: &[f64]) {
    let l1 = v.len();
    for (k, &uk) in u.iter().enumerate() {
        let a = c * uk;
        for (o, &vl) in out[k * l1..(k + 1) * l1].iter_mut().zip(v) {
            *o += a * vl;
        }
    }
}

/// S = Σ_i R_i φ_i.
pub fn s_matrix(d: &PairData<'_>, s: &mut Mat) {
    s.clear();
    s.resize(d.k1 * d.l1, 0.0);
    for i in 0..d.n() {
        add_outer(s, d.r[i], d.bx(i), d.by(i));
    }
}

/// M = Σ_i R_i β⁽⁻ⁱ⁾ ∘ φ_i (the cross term of the MISE criterion), using
/// M/c2 = S∘S − Σ_i R_i² (b_i∘b_i)(b_i∘b_i)ᵀ − Σ_j D_j R_{nn_j} (b_{nn_j}∘b_j)(…)ᵀ.
pub fn mise_cross(d: &PairData<'_>, s: &[f64], c2: f64, m: &mut Mat, u: &mut Vec<f64>, v: &mut Vec<f64>) {
    m.clear();
    m.extend(s.iter().map(|x| x * x));
    u.resize(d.k1, 0.0);
    v.resize(d.l1, 0.0);
    for i in 0..d.n() {
        sq(d.bx(i), d.bx(i), u);
        sq(d.by(i), d.by(i), v);
        add_outer(m, -d.r[i] * d.r[i], u, v);
        let p = d.nn[i] as usize;
        let dj = d.r[i] - d.rp[i];
        if dj != 0.0 {
            sq(d.bx(p), d.bx(i), u);
            sq(d.by(p), d.by(i), v);
            add_outer(m, -dj * d.r[p], u, v);
        }
    }
    m.iter_mut().for_each(|x| *x *= c2);
}

#[inline]
fn sq(a: &[f64], b: &[f64], out: &mut [f64]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x * y;
    }
}

/// Inverse neighbour lists in CSR form: points j with nn_j = i are
/// `list[start[i]..start[i+1]]`.
pub fn inverse_neighbors(nn: &[u32], start: &mut Vec<u32>, list: &mut Vec<u32>) {
    let n = nn.len();
    start.clear();
    start.resize(n + 1, 0);
    for &p in nn {
        start[p as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    list.clear();
    list.resize(n, 0);
    let mut fill = start.clone();
    for (j, &p) in nn.iter().enumerate() {
        list[fill[p as usize] as usize] = j as u32;
        fill[p as usize] += 1;
    }
}

/// Leave-one-out coefficients β⁽⁻ⁱ⁾ (scaled by c2) for one point.
pub fn loo_beta(d: &PairData<'_>, s: &[f64], c2: f64, i: usize, inv: &[u32], beta: &mut Mat) {
    beta.clear();
    beta.extend_from_slice(s);
    add_outer(beta, -d.r[i], d.bx(i), d.by(i));
    for &j in inv {
        let j = j as usize;
        add_outer(beta, -(d.r[j] - d.rp[j]), d.bx(j), d.by(j));
    }
    beta.iter_mut().for_each(|x| *x *= c2);
}
