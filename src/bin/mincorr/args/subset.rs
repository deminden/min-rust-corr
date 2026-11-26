//! Row-selection flag groups A and B (A also answers to the v0.2
//! `--subset-*` names).

use crate::subset::Selector;
use clap::Args;
use std::path::PathBuf;

macro_rules! subset_group {
    ($name:ident, $doc:literal, $size:literal, $seed:literal, $file:literal, $rows:literal $(, $alias:literal)*) => {
        #[doc = $doc]
        #[derive(Debug, Args, Default, Clone)]
        pub struct $name {
            /// Random subset of N rows.
            #[arg(id = $size, long = $size $(, alias = concat!($alias, "size"))*)]
            pub size: Option<usize>,
            /// Seed of the random subset (default 42).
            #[arg(id = $seed, long = $seed $(, alias = concat!($alias, "seed"))*)]
            pub seed: Option<u64>,
            /// File with one row ID per line.
            #[arg(id = $file, long = $file $(, alias = concat!($alias, "file"))*)]
            pub file: Option<PathBuf>,
            /// Comma-separated row IDs.
            #[arg(id = $rows, long = $rows, value_delimiter = ',' $(, alias = concat!($alias, "rows"))*)]
            pub rows: Option<Vec<String>>,
        }

        impl $name {
            /// The prefix-independent selector.
            pub fn selector(&self) -> Selector {
                Selector { size: self.size, seed: self.seed, file: self.file.clone(), rows: self.rows.clone() }
            }
        }
    };
}

subset_group!(
    SubsetA,
    "Row selection A (also the v0.2 --subset-* flags).",
    "subset-a-size",
    "subset-a-seed",
    "subset-a-file",
    "subset-a-rows",
    "subset-"
);
subset_group!(
    SubsetB,
    "Row selection B (cross mode A × B).",
    "subset-b-size",
    "subset-b-seed",
    "subset-b-file",
    "subset-b-rows"
);
