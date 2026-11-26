//! Bit-counting kernels for the Kendall bitset path, with runtime dispatch
//! to the hardware POPCNT instruction on x86-64.
//!
//! This is the crate's only `unsafe` code. Portable x86-64 builds cannot
//! assume POPCNT, so `count_ones` falls back to a bit-twiddling sequence;
//! the same loop compiled with POPCNT enabled is 2–3× faster for Kendall
//! (measured at n = 77…2000; see docs/benchmarks.md). Other targets
//! (aarch64 always has a vector `cnt`) and builds that already enable
//! POPCNT use the portable loop directly. Both paths compute the same exact
//! integer, so results never depend on the CPU.

/// Σ popcount(a[k] ^ b[k]).
#[inline(always)]
fn xor_portable(a: &[u64], b: &[u64]) -> u64 {
    a.iter().zip(b).map(|(x, y)| u64::from((x ^ y).count_ones())).sum()
}

/// Σ popcount(m[k] & (a[k] ^ b[k])).
#[inline(always)]
fn masked_xor_portable(a: &[u64], b: &[u64], m: &[u64]) -> u64 {
    a.iter().zip(b).zip(m).map(|((x, y), z)| u64::from((z & (x ^ y)).count_ones())).sum()
}

/// Σ popcount(a[k] & b[k]).
#[inline(always)]
fn and_portable(a: &[u64], b: &[u64]) -> u64 {
    a.iter().zip(b).map(|(x, y)| u64::from((x & y).count_ones())).sum()
}

/// Σ popcount(ma[k] & mb[k] & (a[k] ^ b[k])).
#[inline(always)]
fn masked2_xor_portable(a: &[u64], b: &[u64], ma: &[u64], mb: &[u64]) -> u64 {
    let it = a.iter().zip(b).zip(ma.iter().zip(mb));
    it.map(|((x, y), (p, q))| u64::from((p & q & (x ^ y)).count_ones())).sum()
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "popcnt")))]
#[allow(unsafe_code)]
mod hw {
    // SAFETY (for every function below): each is compiled with the
    // `popcnt` target feature and is only called by `dispatch!` after
    // `is_x86_feature_detected!("popcnt")` returned true on this CPU, so
    // the instruction is available. The bodies are the safe portable loops.
    #[target_feature(enable = "popcnt")]
    pub unsafe fn xor(a: &[u64], b: &[u64]) -> u64 {
        super::xor_portable(a, b)
    }
    #[target_feature(enable = "popcnt")]
    pub unsafe fn masked_xor(a: &[u64], b: &[u64], m: &[u64]) -> u64 {
        super::masked_xor_portable(a, b, m)
    }
    #[target_feature(enable = "popcnt")]
    pub unsafe fn and(a: &[u64], b: &[u64]) -> u64 {
        super::and_portable(a, b)
    }
    #[target_feature(enable = "popcnt")]
    pub unsafe fn masked2_xor(a: &[u64], b: &[u64], ma: &[u64], mb: &[u64]) -> u64 {
        super::masked2_xor_portable(a, b, ma, mb)
    }
}

/// Call the POPCNT version when the CPU has it, else the portable loop.
macro_rules! dispatch {
    ($hw:ident, $portable:ident ( $($arg:expr),* )) => {{
        #[cfg(all(target_arch = "x86_64", not(target_feature = "popcnt")))]
        {
            if std::is_x86_feature_detected!("popcnt") {
                // SAFETY: the feature was detected just above (see `hw`).
                #[allow(unsafe_code)]
                return unsafe { hw::$hw($($arg),*) };
            }
        }
        $portable($($arg),*)
    }};
}

/// Number of positions where `a` and `b` differ.
pub fn xor_count(a: &[u64], b: &[u64]) -> u64 {
    dispatch!(xor, xor_portable(a, b))
}

/// Number of positions in mask `m` where `a` and `b` differ.
pub fn masked_xor_count(a: &[u64], b: &[u64], m: &[u64]) -> u64 {
    dispatch!(masked_xor, masked_xor_portable(a, b, m))
}

/// Number of positions set in both `a` and `b`.
pub fn and_count(a: &[u64], b: &[u64]) -> u64 {
    dispatch!(and, and_portable(a, b))
}

/// Number of positions in both masks where `a` and `b` differ.
pub fn masked2_xor_count(a: &[u64], b: &[u64], ma: &[u64], mb: &[u64]) -> u64 {
    dispatch!(masked2_xor, masked2_xor_portable(a, b, ma, mb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_matches_portable() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let v: Vec<Vec<u64>> = (0..4).map(|_| (0..37).map(|_| next()).collect()).collect();
        assert_eq!(xor_count(&v[0], &v[1]), xor_portable(&v[0], &v[1]));
        assert_eq!(masked_xor_count(&v[0], &v[1], &v[2]), masked_xor_portable(&v[0], &v[1], &v[2]));
        assert_eq!(and_count(&v[0], &v[1]), and_portable(&v[0], &v[1]));
        let want = masked2_xor_portable(&v[0], &v[1], &v[2], &v[3]);
        assert_eq!(masked2_xor_count(&v[0], &v[1], &v[2], &v[3]), want);
    }
}
