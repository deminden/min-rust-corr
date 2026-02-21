use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

type Case = (f64, f64, f64, bool, bool);

fn run_r(cases: &[Case]) -> Result<Vec<f64>, String> {
    let mut input_path = std::env::temp_dir();
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    input_path.push(format!("qbeta_cases_{}_{}.tsv", pid, ts));

    let mut file = File::create(&input_path).map_err(|e| e.to_string())?;
    writeln!(file, "p\ta\tb\tlower_tail\tlog_p").map_err(|e| e.to_string())?;
    for (p, a, b, lower_tail, log_p) in cases {
        writeln!(
            file,
            "{:.17e}\t{:.17e}\t{:.17e}\t{}\t{}",
            p,
            a,
            b,
            if *lower_tail { "TRUE" } else { "FALSE" },
            if *log_p { "TRUE" } else { "FALSE" }
        )
        .map_err(|e| e.to_string())?;
    }

    let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/qbeta_ref.R");
    let output = Command::new("Rscript")
        .arg(script_path)
        .arg(&input_path)
        .output()
        .map_err(|e| format!("failed to run Rscript: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Rscript failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut results = Vec::with_capacity(cases.len());
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: f64 = trimmed
            .parse()
            .map_err(|e| format!("parse R output: {e}"))?;
        results.push(val);
    }

    if results.len() != cases.len() {
        return Err(format!(
            "R output length mismatch: got {}, expected {}",
            results.len(),
            cases.len()
        ));
    }

    let _ = std::fs::remove_file(&input_path);
    Ok(results)
}

fn ulp_diff(a: f64, b: f64) -> u64 {
    let mut ia = a.to_bits() as i64;
    let mut ib = b.to_bits() as i64;
    if ia < 0 {
        ia = i64::MIN - ia;
    }
    if ib < 0 {
        ib = i64::MIN - ib;
    }
    (ia - ib).abs() as u64
}

fn approx_equal(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive();
    }
    let diff = (a - b).abs();
    let scale = 1.0_f64.max(b.abs());
    if diff <= 2e-15 * scale {
        return true;
    }
    if b.abs() < 1e-5 && diff <= 1e-13 {
        return true;
    }
    ulp_diff(a, b) <= 16
}

fn deterministic_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let ps: [f64; 8] = [
        0.0,
        1.0,
        0.5,
        1e-300,
        1e-50,
        1e-10,
        1.0 - 1e-12,
        1.0 - 1e-30,
    ];
    let ab: [f64; 6] = [0.5, 1.0, 2.0, 10.0, 1e-8, 1e8];

    for &a in &ab {
        for &b in &ab {
            for &p in &ps {
                cases.push((p, a, b, true, false));
                cases.push((p, a, b, false, false));
                if p > 0.0 {
                    cases.push((p.ln(), a, b, true, true));
                    cases.push((p.ln(), a, b, false, true));
                }
            }
        }
    }

    cases.push((f64::NEG_INFINITY, 0.5, 0.5, true, true));
    cases.push((f64::NEG_INFINITY, 0.5, 0.5, false, true));
    cases
}

fn rand_cases(n: usize) -> Vec<Case> {
    let mut out = Vec::with_capacity(n * 2);
    let mut state: u64 = 0x9e3779b97f4a7c15;

    for i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (state >> 11) as f64 / ((1u64 << 53) as f64);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (state >> 11) as f64 / ((1u64 << 53) as f64);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u3 = (state >> 11) as f64 / ((1u64 << 53) as f64);

        let mut p = if i % 10 == 0 {
            let logp = -690.0 + 690.0 * u1;
            logp.exp()
        } else {
            u1
        };
        if p == 0.0 {
            p = f64::MIN_POSITIVE;
        }
        let a = (1e-6_f64.ln() + (1e6_f64.ln() - 1e-6_f64.ln()) * u2).exp();
        let b = (1e-6_f64.ln() + (1e6_f64.ln() - 1e-6_f64.ln()) * u3).exp();

        out.push((p, a, b, true, false));
        out.push((p, a, b, false, false));
        out.push((p.ln(), a, b, true, true));
        out.push((p.ln(), a, b, false, true));
    }

    out
}

#[test]
fn qbeta_against_r() {
    if Command::new("Rscript").arg("--version").output().is_err() {
        eprintln!("Rscript not available; skipping qbeta_against_r");
        return;
    }

    let mut cases = deterministic_cases();
    cases.extend(rand_cases(1000));

    let r_vals = match run_r(&cases) {
        Ok(vals) => vals,
        Err(err) => panic!("Rscript required for tests: {err}"),
    };

    for (idx, ((p, a, b, lower_tail, log_p), r_val)) in cases.iter().zip(r_vals.iter()).enumerate()
    {
        let extreme_p = if *log_p {
            *p <= 1e-300_f64.ln()
        } else {
            *p <= 1e-300
        };
        let huge_ab = a.max(*b) >= 1e6 || a.min(*b) <= 1e-6;
        if extreme_p && huge_ab {
            continue;
        }
        let rust_val = mincorr::hellinger::qbeta(*p, *a, *b, *lower_tail, *log_p);
        if !approx_equal(rust_val, *r_val) {
            let diff = (rust_val - r_val).abs();
            let rel = diff / 1.0_f64.max(r_val.abs()).max(rust_val.abs());
            let ulp = ulp_diff(rust_val, *r_val);
            let rust_p_at_r = mincorr::hellinger::pbeta(*r_val, *a, *b, *lower_tail, false);
            let rust_p_at_rust = mincorr::hellinger::pbeta(rust_val, *a, *b, *lower_tail, false);
            panic!(
                "mismatch at {idx}: p={p:.17e} a={a:.17e} b={b:.17e} lower_tail={lower_tail} log_p={log_p} rust={rust_val:.17e} r={r_val:.17e} abs={diff:.3e} rel={rel:.3e} ulp={ulp} rust_p_at_r={rust_p_at_r:.17e} rust_p_at_rust={rust_p_at_rust:.17e}"
            );
        }
    }
}
