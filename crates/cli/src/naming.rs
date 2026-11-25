//! Output file naming, including the v0.2 legacy scheme
//! `<base>[_subset…]_<method>_correlations…`.

use std::path::{Path, PathBuf};

/// v0.2 stem: `file_stem()` with a trailing ".tar" removed.
pub fn legacy_stem(input: &Path) -> String {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("data");
    stem.strip_suffix(".tar").unwrap_or(stem).to_string()
}

/// Stem with compression, archive and table extensions removed.
pub fn clean_stem(input: &Path) -> String {
    let mut s = input.file_name().and_then(|s| s.to_str()).unwrap_or("data").to_string();
    for ext in [".gz", ".tgz", ".tar", ".tsv", ".csv", ".txt"] {
        if let Some(t) = s.strip_suffix(ext) {
            s = t.to_string();
        }
    }
    if s.is_empty() { "data".into() } else { s }
}

/// Base name with subset tags: `<stem>_<tag>` or `<stem>_A<tagA>_B<tagB>`.
pub fn base_name(stem: &str, tag_a: Option<&str>, tag_b: Option<&str>, cross: bool) -> String {
    if cross {
        format!("{stem}_A{}_B{}", tag_a.unwrap_or("all"), tag_b.unwrap_or("all"))
    } else if let Some(t) = tag_a {
        format!("{stem}_{t}")
    } else {
        stem.to_string()
    }
}

/// Insert `_pvalues` before the (possibly double) extension of `p`, or
/// replace a `correlations` token when present.
pub fn pvalue_path(p: &Path) -> PathBuf {
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("out");
    let new = if name.contains("_correlations") {
        name.replacen("_correlations", "_pvalues", 1)
    } else {
        let (stem, ext) = split_ext(name);
        format!("{stem}_pvalues{ext}")
    };
    p.with_file_name(new)
}

fn split_ext(name: &str) -> (&str, &str) {
    for ext in [".tsv.gz", ".tsv", ".npy", ".gz", ".txt", ".csv"] {
        if let Some(s) = name.strip_suffix(ext) {
            return (s, &name[s.len()..]);
        }
    }
    (name, "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(legacy_stem(Path::new("dir/x.tar.gz")), "x");
        assert_eq!(legacy_stem(Path::new("x.tsv.gz")), "x.tsv");
        assert_eq!(clean_stem(Path::new("dir/x.tsv.gz")), "x");
        assert_eq!(base_name("x", Some("subset5_seed1"), None, false), "x_subset5_seed1");
        assert_eq!(base_name("x", Some("t"), None, true), "x_At_Ball");
        assert_eq!(
            pvalue_path(Path::new("o/a_pearson_correlations.tsv.gz")),
            Path::new("o/a_pearson_pvalues.tsv.gz")
        );
        assert_eq!(pvalue_path(Path::new("res.npy")), Path::new("res_pvalues.npy"));
    }
}
