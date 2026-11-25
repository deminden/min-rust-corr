//! Hellinger affinity B → Hellinger correlation η (paper eq. 4.3):
//! η = (2/B²)·√(B⁴ + √(4 − 3B⁴) − 2).

/// η from a normalised affinity B ∈ (0, 1]. B above 1 (rounding) is
/// treated as 1; a negative radicand from rounding gives η = 0. NaN when
/// B is not positive and finite.
pub fn eta_from_affinity(b: f64) -> f64 {
    if !(b > 0.0 && b.is_finite()) {
        return f64::NAN;
    }
    let b = b.min(1.0);
    let b2 = b * b;
    let b4 = b2 * b2;
    let rad = b4 + (4.0 - 3.0 * b4).sqrt() - 2.0;
    if rad <= 0.0 { 0.0 } else { 2.0 / b2 * rad.sqrt() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints() {
        assert_eq!(eta_from_affinity(1.0), 0.0);
        // η is increasing as B decreases; check against the closed form of
        // h(ρ): for ρ = 0.5, B = 2(1−ρ²)^{1/4}/(4−ρ²)^{1/2}.
        let rho: f64 = 0.5;
        let b = 2.0 * (1.0 - rho * rho).powf(0.25) / (4.0 - rho * rho).sqrt();
        assert!((eta_from_affinity(b) - rho).abs() < 1e-12);
        assert!(eta_from_affinity(0.0).is_nan());
    }
}
