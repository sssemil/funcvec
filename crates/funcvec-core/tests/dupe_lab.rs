use std::{path::PathBuf, process::Command};

use funcvec_core::{ProviderKind, ReportConfig, run_eval};

#[test]
fn dupe_lab_eval_finds_labeled_duplicate_candidates() {
    if !rust_analyzer_available() {
        eprintln!("skipping dupe_lab eval because rust-analyzer is not installed");
        return;
    }

    let fixture = fixture_path();
    let config = ReportConfig {
        provider: ProviderKind::Lexical,
        threshold: 0.55,
        top_k: 50,
        min_lines: 2,
        min_tokens: 8,
        ..ReportConfig::default()
    };

    let eval = run_eval(&fixture, &config).expect("dupe lab eval should run");
    assert!(eval.report.functions_count >= 6);
    assert!(
        eval.report
            .candidates
            .iter()
            .any(|candidate| candidate.expected_match),
        "expected at least one labeled duplicate pair in candidates"
    );
    assert!(
        eval.hybrid.true_positives >= 1,
        "hybrid strategy should flag at least one labeled pair"
    );
}

fn rust_analyzer_available() -> bool {
    Command::new("rustup")
        .args(["which", "rust-analyzer"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("fixtures/dupe_lab")
}
