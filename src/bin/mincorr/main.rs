// `mincorr` command-line tool.

mod args;
mod input;
mod naming;
mod output;
mod run;
mod sinks;
mod subset;
mod timing;

use clap::Parser;

fn main() -> std::process::ExitCode {
    let cli = args::Cli::parse();
    let threads = cli.threads.or(cli.threads_positional);
    if let Some(t) = threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(t).build_global() {
            eprintln!("error: cannot configure {t} threads: {e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    match run::run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
