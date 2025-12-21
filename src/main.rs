// Compute pairwise correlations (Pearson, Spearman, Kendall, biweight midcorrelation, hellcor) in matrices.
// BLAS accelerates Pearson/Spearman

use std::{collections::HashMap, error::Error, fs::File, io::{Cursor, Read}, env, time::Instant};
use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array1, Array2};
use flate2::{read::{GzDecoder, MultiGzDecoder}, write::GzEncoder, Compression};
use tar::{Archive, Builder, Header};
use strum_macros::{EnumString, Display};

mod pearson;
mod spearman;
mod kendall;
mod bicor;
mod hellcor;
mod hellinger;
mod rank;

use pearson::correlation_matrix as pearson_correlation_matrix;
use spearman::correlation_matrix as spearman_correlation_matrix;
use kendall::correlation_matrix as kendall_correlation_matrix;
use bicor::correlation_matrix as bicor_correlation_matrix;
use hellcor::correlation_matrix as hellcor_correlation_matrix;

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
            return Err(format!("Empty row ID encountered on line {}", idx + 2 /* header offset */).into());
        }

        if row_data.contains_key(raw_row_id) {
            return Err(format!("Duplicate row ID '{}' encountered on line {}", raw_row_id, idx + 2).into());
        }

        let expression_values: Array1<f64> = record.iter().skip(1)
            .map(|s| s.parse().unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
            .into();

        row_data.insert(raw_row_id.to_string(), expression_values);
    }
    Ok(row_data)
}

fn parse_args() -> Result<(String, CorrelationType, Option<usize>, bool), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        return Err("Usage: program <input_file> <correlation_type> [num_threads] [--time]\nCorrelation types: pearson, spearman, kendall, bicor, hellcor\nnum_threads: number of threads to use (default: all available)\n--time: enable detailed timing output".into());
    }

    let correlation_type: CorrelationType = args[2].parse()?;

    let mut num_threads = None;
    let mut time_tracking = false;

    // Parse remaining arguments
    for arg in args.iter().skip(3) {
        if arg == "--time" {
            time_tracking = true;
        } else if let Ok(threads) = arg.parse::<usize>() {
            num_threads = Some(threads);
        } else {
            return Err(format!("Unknown argument: {}", arg).into());
        }
    }

    Ok((args[1].clone(), correlation_type, num_threads, time_tracking))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (file_path, correlation_type, num_threads, time_tracking) = parse_args()?;
    
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
    let load_start = if time_tracking { Some(Instant::now()) } else { None };
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
        println!("Expression data loaded in {:.3} seconds.", duration.as_secs_f64());
    }

    let mut row_ids: Vec<String> = row_data.keys().cloned().collect();
    row_ids.sort();
    
    println!("Matrix dimensions: {} rows x {} columns", row_ids.len(), 
             row_data.values().next().map(|v| v.len()).unwrap_or(0));
    
    let data_matrix = build_data_matrix(&row_ids, &row_data);
    
    // Correlation calculation
    let calc_start = if time_tracking { Some(Instant::now()) } else { None };
    let correlation_matrix = match correlation_type {
        CorrelationType::Pearson => {
            println!("Computing Pearson correlations...");
            pearson_correlation_matrix(&data_matrix)
        },
        CorrelationType::Spearman => {
            println!("Computing Spearman correlations...");
            spearman_correlation_matrix(&data_matrix)
        },
        CorrelationType::Kendall => {
            println!("Computing Kendall correlations...");
            kendall_correlation_matrix(&data_matrix)
        },
        CorrelationType::Bicor => {
            println!("Computing biweight midcorrelations (bicor)...");
            bicor_correlation_matrix(&data_matrix)
        },
        CorrelationType::Hellcor => {
            println!("Computing Hellinger correlations (hellcor)...");
            hellcor_correlation_matrix(&data_matrix)
        },
    };
    let calc_duration = calc_start.map(|start| start.elapsed());
    
    if let Some(duration) = calc_duration {
        println!("{} correlations calculated in {:.3} seconds.",
                 correlation_type, duration.as_secs_f64());
    }

    // Output writing
    let output_start = if time_tracking { Some(Instant::now()) } else { None };
    let mut csv_buf = Vec::<u8>::new();
    {
        let mut wtr = WriterBuilder::new().delimiter(b'\t').from_writer(&mut csv_buf);

        wtr.write_record(std::iter::once("").chain(row_ids.iter().map(String::as_str)))?;

        for (i, row_id) in row_ids.iter().enumerate() {
            let row_vals: Vec<String> = correlation_matrix.row(i).iter().map(|&r| r.to_string()).collect();
            wtr.write_record(std::iter::once(row_id.as_str()).chain(row_vals.iter().map( String::as_str )))?;
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

    let tar_gz_path = format!("{}_{}_correlations.tar.gz", input_basename, correlation_suffix);
    let tar_gz_file = File::create(&tar_gz_path)?;
    let enc = GzEncoder::new(tar_gz_file, Compression::default());
    let mut tar_builder = Builder::new(enc);

    let mut header = Header::new_gnu();
    header.set_size(csv_buf.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();

    let csv_name = format!("{}_{}_correlations.tsv", input_basename, correlation_suffix);
    tar_builder.append_data(&mut header, csv_name, &mut Cursor::new(csv_buf))?;
    tar_builder.finish()?;
    let output_duration = output_start.map(|start| start.elapsed());
    
    if let Some(duration) = output_duration {
        println!("Output written in {:.3} seconds.", duration.as_secs_f64());
    }
    
    if time_tracking {
        if let (Some(load_dur), Some(calc_dur), Some(output_dur)) = (load_duration, calc_duration, output_duration) {
            let total_duration = load_dur + calc_dur + output_dur;
            
            println!("Data loading:           {:8.3} seconds", load_dur.as_secs_f64());
            println!("Correlation calculation: {:8.3} seconds", calc_dur.as_secs_f64());
            println!("Output writing:         {:8.3} seconds", output_dur.as_secs_f64());
            println!("Total time:             {:8.3} seconds", total_duration.as_secs_f64());
        }
    }

    Ok(())
}
