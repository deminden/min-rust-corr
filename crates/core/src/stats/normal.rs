//! Two-sided standard-normal p-values.

use super::special::erfc;
use std::f64::consts::SQRT_2;

/// P(|Z| ≥ |z|) = erfc(|z| / √2). NaN for non-finite `z`.
pub fn two_sided_p(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    erfc(z.abs() / SQRT_2).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_pnorm() {
        // 2 * pnorm(-z) from R 4.3.3.
        let cases = [
            (0.1, 0.920_344_325_445_942_02),
            (1.0, 0.317_310_507_862_914_09),
            (2.0, 0.045_500_263_896_358_424),
            (3.0, 0.002_699_796_063_260_189_2),
            (5.0, 5.733_031_437_583_878_2e-7),
            (8.0, 1.244_192_114_854_357e-15),
            (12.0, 3.552_964_224_155_358_1e-33),
            (20.0, 5.507_248_237_212_467_5e-89),
            (37.0, 1.145_114_244_504_915_4e-299),
        ];
        for (z, want) in cases {
            let got = two_sided_p(z);
            assert!(((got - want) / want).abs() < 5e-13, "z={z} got={got} want={want}");
        }
    }
}
