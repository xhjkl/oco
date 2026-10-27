//! Opt-in per-kernel GPU timing. Set `OCO_PROFILE=1` to stamp every dispatch with a GPU
//! timestamp; completed submissions accumulate kernel durations by pipeline label, and
//! [`report`] renders and clears the totals. Precise stamps serialize the GPU, so the
//! absolute numbers overstate a free-running pass; the attribution is what matters.
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[derive(Default)]
struct Totals {
    nanoseconds: HashMap<String, (u64, f64)>,
    submissions: u64,
}

static TOTALS: OnceLock<Option<Mutex<Totals>>> = OnceLock::new();

fn totals() -> Option<&'static Mutex<Totals>> {
    TOTALS
        .get_or_init(|| {
            std::env::var_os("OCO_PROFILE")
                .filter(|value| !value.is_empty() && value != "0")
                .map(|_| Mutex::new(Totals::default()))
        })
        .as_ref()
}

pub fn enabled() -> bool {
    totals().is_some()
}

/// Fold one completed submission's consecutive timestamps into the running totals.
pub fn record(labels: &[String], stamps: &[u64]) {
    let Some(totals) = totals() else { return };
    let mut totals = totals.lock().unwrap_or_else(|error| error.into_inner());
    totals.submissions += 1;
    for (index, label) in labels.iter().enumerate() {
        let (Some(&start), Some(&end)) = (stamps.get(index), stamps.get(index + 1)) else {
            break;
        };
        let entry = totals.nanoseconds.entry(label.clone()).or_insert((0, 0.0));
        entry.0 += 1;
        // Metal timestamps follow the Mach timebase: 24 MHz on Apple silicon.
        entry.1 += end.saturating_sub(start) as f64 * (125.0 / 3.0);
    }
}

/// Render the accumulated kernel totals, longest first, and start a new accumulation.
pub fn report() -> Option<String> {
    let totals = totals()?;
    let mut totals = totals.lock().unwrap_or_else(|error| error.into_inner());
    let mut rows: Vec<_> = totals.nanoseconds.drain().collect();
    let submissions = std::mem::take(&mut totals.submissions).max(1);
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    let total: f64 = rows.iter().map(|row| row.1.1).sum();
    let mut out = format!(
        "{:<36} {:>8} {:>10} {:>9} {:>7}\n",
        "kernel", "calls", "total ms", "avg us", "share"
    );
    for (label, (calls, nanoseconds)) in &rows {
        out.push_str(&format!(
            "{:<36} {:>8} {:>10.3} {:>9.1} {:>6.1}%\n",
            label,
            calls,
            nanoseconds / 1e6,
            nanoseconds / *calls as f64 / 1e3,
            100.0 * nanoseconds / total.max(1.0)
        ));
    }
    out.push_str(&format!(
        "{:<36} {:>8} {:>10.3}  ({} submissions, {:.3} ms each)\n",
        "total",
        rows.iter().map(|row| row.1.0).sum::<u64>(),
        total / 1e6,
        submissions,
        total / 1e6 / submissions as f64
    ));
    Some(out)
}
