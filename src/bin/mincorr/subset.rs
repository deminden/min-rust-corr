//! Row selection (ported from the author-original v0.2 CLI): random
//! subset, list file, or explicit IDs. Selected rows keep input order.

use anyhow::{Result, bail};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// One selector group (A or B), independent of its flag prefix.
#[derive(Debug, Default, Clone)]
pub struct Selector {
    pub size: Option<usize>,
    pub seed: Option<u64>,
    pub file: Option<PathBuf>,
    pub rows: Option<Vec<String>>,
}

impl Selector {
    /// True when any selection mode is set.
    pub fn has_any(&self) -> bool {
        self.size.is_some() || self.file.is_some() || self.rows.is_some()
    }
}

/// Indices of the selected rows (sorted, i.e. input order) plus the tag
/// used in output names; `None` tag when no selection applies.
pub fn apply(ids: &[String], s: &Selector) -> Result<(Vec<usize>, Option<String>)> {
    let modes = [s.size.is_some(), s.file.is_some(), s.rows.is_some()].iter().filter(|b| **b).count();
    if modes == 0 {
        return Ok(((0..ids.len()).collect(), None));
    }
    if modes > 1 {
        bail!("use only one subset mode: --subset-size, --subset-file, or --subset-rows");
    }
    if s.seed.is_some() && s.size.is_none() {
        bail!("--subset-seed can only be used with --subset-size");
    }
    if let Some(size) = s.size {
        if size == 0 {
            bail!("--subset-size must be greater than 0");
        }
        if size > ids.len() {
            bail!("--subset-size {size} exceeds available rows {}", ids.len());
        }
        let seed = s.seed.unwrap_or(42);
        let mut picked = random_subset(ids.len(), size, seed);
        picked.sort_unstable();
        return Ok((picked, Some(format!("subset{size}_seed{seed}"))));
    }
    if let Some(path) = &s.file {
        let content = std::fs::read_to_string(path)?;
        let req: Vec<String> =
            content.lines().map(str::trim).filter(|x| !x.is_empty()).map(String::from).collect();
        if req.is_empty() {
            bail!("no row IDs found in subset file: {}", path.display());
        }
        let sel = select_named(ids, &req)?;
        let tag = format!("subsetfile{}", sel.len());
        return Ok((sel, Some(tag)));
    }
    let rows = s.rows.clone().unwrap_or_default();
    let sel = select_named(ids, &rows)?;
    let tag = format!("subsetlist{}", sel.len());
    Ok((sel, Some(tag)))
}

/// Partial Fisher–Yates: `k` distinct indices out of `n`.
fn random_subset(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let span = (n - i) as u64;
        let j = i + ((u128::from(rng.next_u64()) * u128::from(span)) >> 64) as usize;
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

fn select_named(ids: &[String], requested: &[String]) -> Result<Vec<usize>> {
    let pos: HashMap<&str, usize> = ids.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let mut seen = HashSet::new();
    let (mut sel, mut missing) = (Vec::new(), Vec::new());
    for r in requested.iter().map(|r| r.trim()).filter(|r| !r.is_empty()) {
        if !seen.insert(r) {
            continue;
        }
        match pos.get(r) {
            Some(&i) => sel.push(i),
            None => missing.push(r.to_string()),
        }
    }
    if !missing.is_empty() {
        let sample = missing.iter().take(10).cloned().collect::<Vec<_>>().join(", ");
        bail!("{} requested row IDs not found (first up to 10): {sample}", missing.len());
    }
    if sel.is_empty() {
        bail!("subset selection produced zero rows");
    }
    sel.sort_unstable();
    Ok(sel)
}
