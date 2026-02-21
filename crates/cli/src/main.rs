// Compute pairwise correlations (Pearson, Spearman, Kendall, biweight midcorrelation, hellcor) in matrices.
// BLAS accelerates Pearson/Spearman

use csv::{ReaderBuilder, WriterBuilder};
use flate2::{
    Compression,
    read::{GzDecoder, MultiGzDecoder},
    write::GzEncoder,
};
use ndarray::{Array1, Array2};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{
    collections::{HashMap, HashSet},
    env,
    error::Error,
    fs::File,
    io::{Cursor, Read},
    time::Instant,
};
use strum_macros::{Display, EnumString};
use tar::{Archive, Builder, Header};

use mincorr::bicor::{
    correlation_cross_matrix as bicor_cross_matrix, correlation_matrix as bicor_correlation_matrix,
};
use mincorr::hellcor::{
    correlation_cross_matrix as hellcor_cross_matrix,
    correlation_matrix as hellcor_correlation_matrix,
};
use mincorr::kendall::{
    correlation_cross_matrix as kendall_cross_matrix,
    correlation_matrix as kendall_correlation_matrix,
};
use mincorr::pearson::{
    correlation_cross_matrix as pearson_cross_matrix,
    correlation_matrix as pearson_correlation_matrix,
};
use mincorr::spearman::{
    correlation_cross_matrix as spearman_cross_matrix,
    correlation_matrix as spearman_correlation_matrix,
};

#[derive(EnumString, Display)]
#[strum(ascii_case_insensitive)]
enum CorrelationType {
    #[strum(serialize = "Pearson")]
    Pearson,
    #[strum(serialize = "Spearman")]
    Spearman,
    #[strum(serialize = "Kendall")]
    Kendall,
    #[strum(serialize = "Bicor", serialize = "Biweight", to_string = "Bicor")]
    Bicor,
    #[strum(serialize = "Hellcor", serialize = "Hellinger", to_string = "Hellcor")]
    Hellcor,
}

#[derive(Default)]
struct SubsetConfig {
    size: Option<usize>,
    seed: Option<u64>,
    file: Option<String>,
    rows: Option<Vec<String>>,
}

impl SubsetConfig {
    fn has_any(&self) -> bool {
        self.size.is_some() || self.file.is_some() || self.rows.is_some()
    }
}

fn build_data_matrix(row_ids: &[String], row_data: &HashMap<String, Array1<f64>>) -> Array2<f64> {
    let n_rows = row_ids.len();
    let n_cols = row_data.values().next().map(|v| v.len()).unwrap_or(0);

    let mut matrix = Array2::<f64>::zeros((n_rows, n_cols));
    for (i, row_id) in row_ids.iter().enumerate() {
        if let Some(values) = row_data.get(row_id) {
            matrix.row_mut(i).assign(values);
        }
    }
    matrix
}

// rank_data is defined in src/rank.rs for shared use.

fn read_matrix_data<R: Read>(reader: R) -> Result<HashMap<String, Array1<f64>>, Box<dyn Error>> {
    let mut row_data = HashMap::new();
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(reader);

    // File should contain at least two columns (row ID + ≥2 samples)
    let header_len = rdr.headers()?.len();
    if header_len < 3 {
        return Err("Input file must contain at least two columns for correlation analysis".into());
    }

    for (idx, record) in rdr.records().enumerate() {
        let record = record?;

        // Validate row ID field
        let raw_row_id = record.get(0).unwrap_or("").trim();
        if raw_row_id.is_empty() {
            return Err(format!(
                "Empty row ID encountered on line {}",
                idx + 2 /* header offset */
            )
            .into());
        }

        if row_data.contains_key(raw_row_id) {
            return Err(format!(
                "Duplicate row ID '{}' encountered on line {}",
                raw_row_id,
                idx + 2
            )
            .into());
        }

        let expression_values: Array1<f64> = record
            .iter()
            .skip(1)
            .map(|s| s.parse().unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
            .into();

        row_data.insert(raw_row_id.to_string(), expression_values);
    }
    Ok(row_data)
}

fn parse_args() -> Result<
    (
        String,
        CorrelationType,
        Option<usize>,
        bool,
        bool,
        SubsetConfig,
        SubsetConfig,
    ),
    Box<dyn Error>,
> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        return Err("Usage: program <input_file> <correlation_type> [num_threads] [--time] [--subset-size N --subset-seed S] [--subset-file PATH] [--subset-rows ID1,ID2,...] [--subset-a-* ... --subset-b-* ...] [--subset-vs-all]\nCorrelation types: pearson, spearman, kendall, bicor, hellcor\nnum_threads: number of threads to use (default: all available)\n--time: enable detailed timing output".into());
    }

    let correlation_type: CorrelationType = args[2].parse()?;

    let mut num_threads = None;
    let mut time_tracking = false;
    let mut subset_vs_all = false;
    let mut subset_a = SubsetConfig::default();
    let mut subset_b = SubsetConfig::default();

    // Parse remaining arguments.
    let mut i = 3;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--time" {
            time_tracking = true;
            i += 1;
        } else if arg == "--subset-vs-all" {
            subset_vs_all = true;
            i += 1;
        } else if arg == "--subset-a-size" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-a-size")?;
            subset_a.size = Some(value.parse().map_err(|_| "Invalid --subset-a-size value")?);
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-a-size=") {
            subset_a.size = Some(value.parse().map_err(|_| "Invalid --subset-a-size value")?);
            i += 1;
        } else if arg == "--subset-a-seed" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-a-seed")?;
            subset_a.seed = Some(value.parse().map_err(|_| "Invalid --subset-a-seed value")?);
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-a-seed=") {
            subset_a.seed = Some(value.parse().map_err(|_| "Invalid --subset-a-seed value")?);
            i += 1;
        } else if arg == "--subset-a-file" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-a-file")?;
            subset_a.file = Some(value.clone());
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-a-file=") {
            subset_a.file = Some(value.to_string());
            i += 1;
        } else if arg == "--subset-a-rows" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-a-rows")?;
            subset_a.rows = Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(ToString::to_string)
                    .collect(),
            );
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-a-rows=") {
            subset_a.rows = Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(ToString::to_string)
                    .collect(),
            );
            i += 1;
        } else if arg == "--subset-b-size" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-b-size")?;
            subset_b.size = Some(value.parse().map_err(|_| "Invalid --subset-b-size value")?);
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-b-size=") {
            subset_b.size = Some(value.parse().map_err(|_| "Invalid --subset-b-size value")?);
            i += 1;
        } else if arg == "--subset-b-seed" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-b-seed")?;
            subset_b.seed = Some(value.parse().map_err(|_| "Invalid --subset-b-seed value")?);
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-b-seed=") {
            subset_b.seed = Some(value.parse().map_err(|_| "Invalid --subset-b-seed value")?);
            i += 1;
        } else if arg == "--subset-b-file" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-b-file")?;
            subset_b.file = Some(value.clone());
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-b-file=") {
            subset_b.file = Some(value.to_string());
            i += 1;
        } else if arg == "--subset-b-rows" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-b-rows")?;
            subset_b.rows = Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(ToString::to_string)
                    .collect(),
            );
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-b-rows=") {
            subset_b.rows = Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(ToString::to_string)
                    .collect(),
            );
            i += 1;
        } else if arg == "--subset-size" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-size")?;
            subset_a.size = Some(value.parse().map_err(|_| "Invalid --subset-size value")?);
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-size=") {
            subset_a.size = Some(value.parse().map_err(|_| "Invalid --subset-size value")?);
            i += 1;
        } else if arg == "--subset-seed" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-seed")?;
            subset_a.seed = Some(value.parse().map_err(|_| "Invalid --subset-seed value")?);
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-seed=") {
            subset_a.seed = Some(value.parse().map_err(|_| "Invalid --subset-seed value")?);
            i += 1;
        } else if arg == "--subset-file" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-file")?;
            subset_a.file = Some(value.clone());
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-file=") {
            subset_a.file = Some(value.to_string());
            i += 1;
        } else if arg == "--subset-rows" {
            let value = args.get(i + 1).ok_or("Missing value for --subset-rows")?;
            subset_a.rows = Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(ToString::to_string)
                    .collect(),
            );
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--subset-rows=") {
            subset_a.rows = Some(
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(ToString::to_string)
                    .collect(),
            );
            i += 1;
        } else if let Ok(threads) = arg.parse::<usize>() {
            num_threads = Some(threads);
            i += 1;
        } else {
            return Err(format!("Unknown argument: {}", arg).into());
        }
    }

    Ok((
        args[1].clone(),
        correlation_type,
        num_threads,
        time_tracking,
        subset_vs_all,
        subset_a,
        subset_b,
    ))
}

fn load_subset_rows_from_file(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let content = std::fs::read_to_string(path)?;
    let rows: Vec<String> = content
        .lines()
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(ToString::to_string)
        .collect();
    if rows.is_empty() {
        return Err(format!("No row IDs found in subset file: {}", path).into());
    }
    Ok(rows)
}

fn select_named_rows(
    all_rows: &[String],
    requested: &[String],
) -> Result<Vec<String>, Box<dyn Error>> {
    let available: HashSet<&str> = all_rows.iter().map(String::as_str).collect();
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    let mut missing = Vec::new();
    for row in requested {
        let name = row.trim();
        if name.is_empty() {
            continue;
        }
        if !seen.insert(name.to_string()) {
            continue;
        }
        if available.contains(name) {
            selected.push(name.to_string());
        } else {
            missing.push(name.to_string());
        }
    }
    if !missing.is_empty() {
        let sample = missing
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{} requested row IDs not found (first up to 10): {}",
            missing.len(),
            sample
        )
        .into());
    }
    if selected.is_empty() {
        return Err("Subset selection produced zero rows".into());
    }
    Ok(selected)
}

fn apply_subset(
    row_ids: &[String],
    subset: &SubsetConfig,
) -> Result<(Vec<String>, Option<String>), Box<dyn Error>> {
    let mut mode_count = 0;
    if subset.size.is_some() {
        mode_count += 1;
    }
    if subset.file.is_some() {
        mode_count += 1;
    }
    if subset.rows.is_some() {
        mode_count += 1;
    }

    if mode_count == 0 {
        return Ok((row_ids.to_vec(), None));
    }
    if mode_count > 1 {
        return Err(
            "Use only one subset mode: --subset-size, --subset-file, or --subset-rows".into(),
        );
    }

    if subset.seed.is_some() && subset.size.is_none() {
        return Err("--subset-seed can only be used with --subset-size".into());
    }

    if let Some(size) = subset.size {
        if size == 0 {
            return Err("--subset-size must be greater than 0".into());
        }
        if size > row_ids.len() {
            return Err(format!(
                "--subset-size {} exceeds available rows {}",
                size,
                row_ids.len()
            )
            .into());
        }
        let seed = subset.seed.unwrap_or(42);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut picked = row_ids.to_vec();
        picked.shuffle(&mut rng);
        picked.truncate(size);
        picked.sort();
        let tag = format!("subset{}_seed{}", size, seed);
        return Ok((picked, Some(tag)));
    }

    if let Some(path) = &subset.file {
        let requested = load_subset_rows_from_file(path)?;
        let selected = select_named_rows(row_ids, &requested)?;
        let tag = format!("subsetfile{}", selected.len());
        return Ok((selected, Some(tag)));
    }

    if let Some(rows) = &subset.rows {
        let selected = select_named_rows(row_ids, rows)?;
        let tag = format!("subsetlist{}", selected.len());
        return Ok((selected, Some(tag)));
    }

    Ok((row_ids.to_vec(), None))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (
        file_path,
        correlation_type,
        num_threads,
        time_tracking,
        subset_vs_all,
        subset_cfg_a,
        subset_cfg_b,
    ) = parse_args()?;

    // Configure thread pool
    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| format!("Failed to set thread pool: {}", e))?;
        println!("Using {} threads.", threads);
    } else {
        println!("Using all available CPU cores.");
    }

    // Data loading
    let load_start = if time_tracking {
        Some(Instant::now())
    } else {
        None
    };
    let row_data: HashMap<String, Array1<f64>> = if file_path.ends_with(".tar.gz") {
        let file = File::open(&file_path)?;
        let gz = GzDecoder::new(file);
        let mut archive = Archive::new(gz);
        let mut data = None;

        for entry in archive.entries()? {
            let mut entry = entry?;
            if entry.header().entry_type().is_file() {
                data = Some(read_matrix_data(&mut entry)?);
                break;
            }
        }
        data.ok_or("No readable file found in tar archive")?
    } else if file_path.ends_with(".gz") {
        let file = File::open(&file_path)?;
        read_matrix_data(MultiGzDecoder::new(file))?
    } else {
        read_matrix_data(File::open(&file_path)?)?
    };
    let load_duration = load_start.map(|start| start.elapsed());

    if let Some(duration) = load_duration {
        println!(
            "Expression data loaded in {:.3} seconds.",
            duration.as_secs_f64()
        );
    }

    let mut all_row_ids: Vec<String> = row_data.keys().cloned().collect();
    all_row_ids.sort();

    if subset_vs_all && subset_cfg_b.has_any() {
        return Err("--subset-vs-all cannot be combined with --subset-b-* selectors".into());
    }
    if subset_vs_all && !subset_cfg_a.has_any() {
        return Err("--subset-vs-all requires subset A selection (--subset-size/--subset-file/--subset-rows or --subset-a-*)".into());
    }

    let (row_ids_a, subset_tag_a) = apply_subset(&all_row_ids, &subset_cfg_a)?;
    if let Some(tag) = &subset_tag_a {
        println!(
            "Applying subset A mode: {} ({} rows selected).",
            tag,
            row_ids_a.len()
        );
    }

    let use_subset_b = subset_cfg_b.has_any() || subset_vs_all;
    let (row_ids_b, subset_tag_b) = if subset_vs_all {
        println!(
            "Applying subset B mode: all rows ({} rows selected).",
            all_row_ids.len()
        );
        (all_row_ids.clone(), Some("all".to_string()))
    } else if subset_cfg_b.has_any() {
        let (rows, tag) = apply_subset(&all_row_ids, &subset_cfg_b)?;
        if let Some(tagv) = &tag {
            println!(
                "Applying subset B mode: {} ({} rows selected).",
                tagv,
                rows.len()
            );
        }
        (rows, tag)
    } else {
        (Vec::new(), None)
    };

    if use_subset_b {
        println!(
            "Matrix dimensions: {} rows x {} columns (A x B), source samples: {}",
            row_ids_a.len(),
            row_ids_b.len(),
            row_data.values().next().map(|v| v.len()).unwrap_or(0)
        );
    } else {
        println!(
            "Matrix dimensions: {} rows x {} columns",
            row_ids_a.len(),
            row_data.values().next().map(|v| v.len()).unwrap_or(0)
        );
    }

    let data_matrix_a = build_data_matrix(&row_ids_a, &row_data);
    let data_matrix_b = if use_subset_b {
        Some(build_data_matrix(&row_ids_b, &row_data))
    } else {
        None
    };

    // Correlation calculation
    let calc_start = if time_tracking {
        Some(Instant::now())
    } else {
        None
    };
    let mut correlation_matrix = match correlation_type {
        CorrelationType::Pearson => {
            println!("Computing Pearson correlations...");
            if let Some(ref b) = data_matrix_b {
                pearson_cross_matrix(&data_matrix_a, b)
            } else {
                pearson_correlation_matrix(&data_matrix_a)
            }
        }
        CorrelationType::Spearman => {
            println!("Computing Spearman correlations...");
            if let Some(ref b) = data_matrix_b {
                spearman_cross_matrix(&data_matrix_a, b)
            } else {
                spearman_correlation_matrix(&data_matrix_a)
            }
        }
        CorrelationType::Kendall => {
            println!("Computing Kendall correlations...");
            if let Some(ref b) = data_matrix_b {
                kendall_cross_matrix(&data_matrix_a, b)
            } else {
                kendall_correlation_matrix(&data_matrix_a)
            }
        }
        CorrelationType::Bicor => {
            println!("Computing biweight midcorrelations (bicor)...");
            if let Some(ref b) = data_matrix_b {
                bicor_cross_matrix(&data_matrix_a, b)
            } else {
                bicor_correlation_matrix(&data_matrix_a)
            }
        }
        CorrelationType::Hellcor => {
            println!("Computing Hellinger correlations (hellcor)...");
            if let Some(ref b) = data_matrix_b {
                hellcor_cross_matrix(&data_matrix_a, b)
            } else {
                hellcor_correlation_matrix(&data_matrix_a)
            }
        }
    };

    if use_subset_b {
        // Keep exact diagonal semantics for overlapping A/B row IDs.
        let b_index: HashMap<&str, usize> = row_ids_b
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        for (i, id) in row_ids_a.iter().enumerate() {
            if let Some(&j) = b_index.get(id.as_str()) {
                correlation_matrix[[i, j]] = 1.0;
            }
        }
    }

    let output_row_ids = row_ids_a.clone();
    let output_col_ids = if use_subset_b {
        row_ids_b.clone()
    } else {
        row_ids_a.clone()
    };

    let calc_duration = calc_start.map(|start| start.elapsed());

    if let Some(duration) = calc_duration {
        println!(
            "{} correlations calculated in {:.3} seconds.",
            correlation_type,
            duration.as_secs_f64()
        );
    }

    // Output writing
    let output_start = if time_tracking {
        Some(Instant::now())
    } else {
        None
    };
    let mut csv_buf = Vec::<u8>::new();
    {
        let mut wtr = WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(&mut csv_buf);

        wtr.write_record(std::iter::once("").chain(output_col_ids.iter().map(String::as_str)))?;

        for (i, row_id) in output_row_ids.iter().enumerate() {
            let row_vals: Vec<String> = correlation_matrix
                .row(i)
                .iter()
                .map(|&r| r.to_string())
                .collect();
            wtr.write_record(
                std::iter::once(row_id.as_str()).chain(row_vals.iter().map(String::as_str)),
            )?;
        }
        wtr.flush()?;
    }

    let correlation_suffix = correlation_type.to_string().to_lowercase();

    // Extract base filename from input path
    let input_basename = std::path::Path::new(&file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("data");

    // Remove .tar if present (for .tar.gz files)
    let input_basename = if input_basename.ends_with(".tar") {
        &input_basename[..input_basename.len() - 4]
    } else {
        input_basename
    };

    let output_base = if use_subset_b {
        let tag_a = subset_tag_a.clone().unwrap_or_else(|| "all".to_string());
        let tag_b = subset_tag_b.clone().unwrap_or_else(|| "all".to_string());
        format!("{}_A{}_B{}", input_basename, tag_a, tag_b)
    } else if let Some(tag) = &subset_tag_a {
        format!("{}_{}", input_basename, tag)
    } else {
        input_basename.to_string()
    };

    let tar_gz_path = format!("{}_{}_correlations.tar.gz", output_base, correlation_suffix);
    let tar_gz_file = File::create(&tar_gz_path)?;
    let enc = GzEncoder::new(tar_gz_file, Compression::default());
    let mut tar_builder = Builder::new(enc);

    let mut header = Header::new_gnu();
    header.set_size(csv_buf.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();

    let csv_name = format!("{}_{}_correlations.tsv", output_base, correlation_suffix);
    tar_builder.append_data(&mut header, csv_name, &mut Cursor::new(csv_buf))?;
    tar_builder.finish()?;
    let output_duration = output_start.map(|start| start.elapsed());

    if let Some(duration) = output_duration {
        println!("Output written in {:.3} seconds.", duration.as_secs_f64());
    }

    if time_tracking {
        if let (Some(load_dur), Some(calc_dur), Some(output_dur)) =
            (load_duration, calc_duration, output_duration)
        {
            let total_duration = load_dur + calc_dur + output_dur;

            println!(
                "Data loading:           {:8.3} seconds",
                load_dur.as_secs_f64()
            );
            println!(
                "Correlation calculation: {:8.3} seconds",
                calc_dur.as_secs_f64()
            );
            println!(
                "Output writing:         {:8.3} seconds",
                output_dur.as_secs_f64()
            );
            println!(
                "Total time:             {:8.3} seconds",
                total_duration.as_secs_f64()
            );
        }
    }

    Ok(())
}
