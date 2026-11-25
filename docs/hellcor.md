# Hellinger correlation: algorithm notes

Clean-room implementation written from Geenens, G. & Lafaye de Micheaux, P.
(2022), "The Hellinger Correlation", JASA 117(538), 639–653 (preprint
arXiv:1810.10276, §5 and Appendices B–C). No code of the HellCor package was
consulted.

## Estimator (per pair, n complete observations)

1. Pseudo-observations U = rank/(n+1) (average ranks for ties).
2. Marginal transform T = Q(U) with Q the Beta(α, α) quantile (α = 6 by default);
   weights √ξ(T) with ξ the Beta(α, α) density (paper, Appendix B).
3. Nearest and second-nearest neighbour distances d₁, d₂ in the T-plane
   (brute force, squared distances, lowest index wins ties).
   R_i = d₁·√ξ(T1_i)·√ξ(T2_i), R'_i = d₂·√ξ(T1_i)·√ξ(T2_i).
4. Orthonormal shifted Legendre basis b_k(u) = √(2k+1)·P_k(2u−1) via the
   three-term recurrence.
5. β̂_kl = c₁ Σ_i R_i b_k(U1_i) b_l(U2_i), c₁ = 2√(n−1)/n (paper eq. 5.2).
6. Orders (K̂, L̂) over 0 ≤ K ≤ kmax, 0 ≤ L ≤ lmax, first optimum in K-major
   scan order, then forced to ≥ 1 (see "Order selection").
7. B̂ = β̂₀₀ / (Σ_{k≤K̂, l≤L̂} β̂²_kl)^{1/2} (eq. 5.3) and
   η̂ = (2/B̂²)·{B̂⁴ + (4 − 3B̂⁴)^{1/2} − 2}^{1/2} (eq. 5.4). The normalisation
   guarantees B̂ ∈ (0, 1]; a negative radicand from rounding gives η̂ = 0.
8. n ≤ 3 → 0.0.

## Order selection

Leave-one-out coefficients use c₂ = 2√(n−2)/(n−1) and, for point j whose
nearest neighbour was the removed point i, the second-nearest radius R'_j.

- `HellcorCriterion::Mise` (default, paper Appendix C): minimise
  A²(K,L) − 2B(K,L), A² = Σ β̂², B = c₁ Σ_i R_i √ĉ^{(−i)}(U_i).
- `HellcorCriterion::NormalizedLoo`: maximise
  Σ_i R_i ⟨β^{(−i)}, φ_i⟩ / ‖β^{(−i)}‖ (the variant described in the rewrite
  specification). **Discrepancy flagged:** the specification and the paper
  describe different criteria; the paper was followed by default. Whichever
  matches HellCor's black-box output on the fixtures should become the default
  (owner decision).

## Complexity: O(n² + n·K·L) per pair

With φ_i = b(U1_i) b(U2_i)ᵀ and S = Σ_j R_j φ_j:

β^{(−i)} = c₂·[S − R_i φ_i − Σ_{j: nn_j = i} (R_j − R'_j) φ_j].

For the MISE criterion the cross term needs only
M = Σ_i R_i β^{(−i)} ∘ φ_i
  = c₂·[S∘S − Σ_i R_i² (b_i∘b_i)(b_i∘b_i)ᵀ − Σ_j (R_j − R'_j) R_{nn_j} (b_{nn_j}∘b_j)(b_{nn_j}∘b_j)ᵀ],
three rank-n accumulations of (K+1)×(L+1) matrices; every candidate (K, L)
then comes from 2-D prefix sums of M and S∘S. The normalised variant
materialises β^{(−i)} per point, still O(n·K·L) in total because every j
appears in exactly one inverse-neighbour list.

All marginal quantities depend only on the half-rank r ∈ {1, 1.5, …, n} and are
tabulated once per (n, α) (Beta quantiles by symmetry Q(1−u) = 1 − Q(u) for half
the grid). A prepared row is a vector of half-rank indices.

## P-values

Monte-Carlo permutation null per distinct effective sample size: x-ranks fixed
to 1..n, y-ranks a uniform random permutation (Fisher–Yates with Lemire's
unbiased bounded integers), replicate b on ChaCha8 stream b of the user seed.
p = (1 + #{null ≥ obs}) / (B + 1). Not bit-reproducible against v0.2.

## Numerical differences vs R

Summation orders differ from the reference implementation, so differences at the
1e-12–1e-10 level are expected and, rarely, a different (K̂, L̂) when two
candidates are within rounding. Fixture parity vs `HellCor::HellCor` is pending
(the HellCor package was not available in the build environment; run
`Rscript tools/make_fixtures.R` with HellCor installed and re-run the tests).
