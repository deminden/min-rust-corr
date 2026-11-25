//! Progress and timing messages (stderr only; stdout carries data).

use std::time::{Duration, Instant};

/// Collects phase durations and prints messages unless quiet.
pub struct Reporter {
    quiet: bool,
    timing: bool,
    phases: Vec<(&'static str, Duration)>,
    start: Instant,
}

impl Reporter {
    /// New reporter.
    pub fn new(quiet: bool, timing: bool) -> Self {
        Self { quiet, timing, phases: Vec::new(), start: Instant::now() }
    }

    /// Print a progress message.
    pub fn say(&self, msg: impl AsRef<str>) {
        if !self.quiet {
            eprintln!("{}", msg.as_ref());
        }
    }

    /// Close the current phase under `name`.
    pub fn phase(&mut self, name: &'static str) {
        let now = Instant::now();
        self.phases.push((name, now - self.start));
        self.start = now;
    }

    /// Print the timing table when `--time` was given.
    pub fn summary(&self) {
        if !self.timing {
            return;
        }
        let mut total = Duration::ZERO;
        for (name, d) in &self.phases {
            eprintln!("{name:<24}{:9.3} seconds", d.as_secs_f64());
            total += *d;
        }
        eprintln!("{:<24}{:9.3} seconds", "Total time:", total.as_secs_f64());
    }
}
