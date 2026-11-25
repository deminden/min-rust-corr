//! End-to-end tests of the `mincorr` binary.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use assert_cmd::Command;
use flate2::read::GzDecoder;
use mincorr::{Correlator, Method, PvalueSpec};
use predicates::prelude::*;
use std::io::Read;
use std::path::{Path, PathBuf};

const INPUT: &str = "zeta\t0.1\t0.4\t0.35\t0.8\t0.9\n\
alpha\t1\t2\t3\t4\t5\n\
mid\t5\t3\tNA\t2\t1\n\
beta\t2\t1\t4\t3\t6\n";

fn setup() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("expr.tsv");
    std::fs::write(&p, format!("gene\ts1\ts2\ts3\ts4\ts5\n{INPUT}")).unwrap();
    (dir, p)
}

fn cmd(dir: &Path) -> Command {
    let mut c = Command::cargo_bin("mincorr").unwrap();
    c.current_dir(dir);
    c
}

fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut s = String::new();
    if path.extension().is_some_and(|e| e == "gz") {
        flate2::read::MultiGzDecoder::new(std::fs::File::open(path).unwrap()).read_to_string(&mut s).unwrap();
    } else {
        s = std::fs::read_to_string(path).unwrap();
    }
    let mut ids = Vec::new();
    let rows = s
        .lines()
        .skip(1)
        .map(|l| {
            let mut f = l.split('\t');
            ids.push(f.next().unwrap().to_string());
            f.map(|v| v.parse::<f64>().unwrap()).collect()
        })
        .collect();
    (ids, rows)
}

#[test]
fn default_output_preserves_order_and_matches_library() {
    let (dir, input) = setup();
    cmd(dir.path()).arg(&input).arg("spearman").arg("--with-pvalues").assert().success().stdout("");
    let (ids, rows) = read_tsv(&dir.path().join("expr_spearman_correlations.tsv.gz"));
    assert_eq!(ids, ["zeta", "alpha", "mid", "beta"]);
    let data = ndarray::Array2::from_shape_vec(
        (4, 5),
        INPUT
            .lines()
            .flat_map(|l| l.split('\t').skip(1).map(|v| v.parse::<f64>().unwrap_or(f64::NAN)))
            .collect(),
    )
    .unwrap();
    let want = Correlator::new(Method::Spearman).pvalues(PvalueSpec::Analytic).full(data.view()).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let (g, w) = (rows[i][j], want.corr[[i, j]]);
            assert!(g == w || (g.is_nan() && w.is_nan()));
        }
    }
    assert!(dir.path().join("expr_spearman_pvalues.tsv.gz").exists());
}

#[test]
fn sort_rows_and_npy_agree_with_tsv() {
    let (dir, input) = setup();
    cmd(dir.path()).arg(&input).args(["pearson", "--sort-rows", "--format", "tsv"]).assert().success();
    let (ids, rows) = read_tsv(&dir.path().join("expr_pearson_correlations.tsv"));
    assert_eq!(ids, ["alpha", "beta", "mid", "zeta"]);
    cmd(dir.path())
        .arg(&input)
        .args(["pearson", "--sort-rows", "--format", "npy", "-o", "m.npy"])
        .assert()
        .success();
    let bytes = std::fs::read(dir.path().join("m.npy")).unwrap();
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let vals: Vec<f64> =
        bytes[10 + hlen..].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let flat: Vec<f64> = rows.concat();
    assert_eq!(vals.len(), flat.len());
    assert!(vals.iter().zip(&flat).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
}

#[test]
fn legacy_tar_naming_and_v02_flags() {
    let (dir, input) = setup();
    cmd(dir.path())
        .arg(&input)
        .args(["kendall", "2", "--time", "--with-pvalues", "--pvalue-mode", "analytic", "--legacy-tar"])
        .args(["--subset-size", "3", "--subset-seed", "7"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Total time"));
    let tar = dir.path().join("expr_subset3_seed7_kendall_correlations.tar.gz");
    let mut a = tar::Archive::new(GzDecoder::new(std::fs::File::open(tar).unwrap()));
    let names: Vec<String> =
        a.entries().unwrap().map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(
        names,
        ["expr_subset3_seed7_kendall_correlations.tsv", "expr_subset3_seed7_kendall_pvalues.tsv"]
    );
}

#[test]
fn cross_modes_snap_overlaps_and_edges() {
    let (dir, input) = setup();
    cmd(dir.path())
        .arg(&input)
        .args(["pearson", "--subset-a-rows", "alpha,beta", "--subset-b-rows", "beta,zeta", "--format", "tsv"])
        .assert()
        .success();
    let (ids, rows) = read_tsv(&dir.path().join("expr_Asubsetlist2_Bsubsetlist2_pearson_correlations.tsv"));
    assert_eq!(ids, ["alpha", "beta"]);
    // B keeps input order: [zeta, beta]; beta×beta is snapped to exactly 1.
    assert_eq!(rows[1][1], 1.0);
    cmd(dir.path())
        .arg(&input)
        .args([
            "bicor",
            "--subset-rows",
            "alpha",
            "--subset-vs-all",
            "--format",
            "edges",
            "--top-k",
            "1",
            "-o",
            "e.gz",
        ])
        .assert()
        .success();
    let mut s = String::new();
    GzDecoder::new(std::fs::File::open(dir.path().join("e.gz")).unwrap()).read_to_string(&mut s).unwrap();
    assert_eq!(s.lines().count(), 2);
}

#[test]
fn errors_are_reported_not_panics() {
    let (dir, input) = setup();
    let bad = |content: &str, msg: &str| {
        let p = dir.path().join("bad.tsv");
        std::fs::write(&p, content).unwrap();
        cmd(dir.path()).arg(&p).arg("pearson").assert().failure().stderr(predicate::str::contains(msg));
    };
    bad("id\ta\tb\nx\t1\toops\n", "line 2, column 3");
    bad("id\ta\tb\nx\t1\t2\nx\t3\t4\n", "duplicate row ID 'x'");
    bad("id\ta\tb\nx\t1\t2\ny\t3\n", "ragged");
    cmd(dir.path())
        .arg(&input)
        .arg("nope")
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown method"));
    cmd(dir.path())
        .arg(&input)
        .args(["hellcor", "--with-pvalues", "--pvalue-mode", "analytic"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("analytic p-values are not available"));
    cmd(dir.path()).arg(&input).args(["pearson", "--nan-policy", "error"]).assert().failure();
    cmd(dir.path())
        .arg(&input)
        .args(["pearson", "--subset-size", "2", "--subset-rows", "alpha"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("only one subset mode"));
}

#[test]
fn csv_and_gzip_input_and_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("m.csv.gz");
    let mut enc =
        flate2::write::GzEncoder::new(std::fs::File::create(&p).unwrap(), flate2::Compression::fast());
    std::io::Write::write_all(&mut enc, b"s1,s2,s3\na,1,2,3\nb,3,1,2\n").unwrap();
    enc.finish().unwrap();
    cmd(dir.path())
        .arg(&p)
        .args(["pearson", "-o", "-", "--format", "tsv", "-q", "--digits", "3"])
        .assert()
        .success()
        .stdout(predicate::str::contains("a\t1.000\t-0.500"));
}
