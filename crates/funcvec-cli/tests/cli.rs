use std::process::Command;

#[test]
fn no_args_outside_rust_project_exits_with_clear_error() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_funcvec"))
        .current_dir(temp_dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("is not inside a Rust project"),
        "stderr was: {stderr}"
    );
}

#[test]
fn cargo_funcvec_binary_exposes_same_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_cargo-funcvec"))
        .arg("funcvec")
        .arg("--help")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage: funcvec"));
}
