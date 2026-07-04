use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

fn main() -> ExitCode {
    let Some(funcvec) = funcvec_binary() else {
        eprintln!("failed to locate the `funcvec` binary next to `cargo-funcvec`");
        return ExitCode::FAILURE;
    };

    let mut args: Vec<_> = env::args_os().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "funcvec") {
        args.remove(0);
    }

    match Command::new(funcvec).args(args).status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
        Err(error) => {
            eprintln!("failed to run `funcvec`: {error}");
            ExitCode::FAILURE
        }
    }
}

fn funcvec_binary() -> Option<PathBuf> {
    let current = env::current_exe().ok()?;
    let dir = current.parent()?;
    let name = if cfg!(windows) {
        "funcvec.exe"
    } else {
        "funcvec"
    };
    Some(dir.join(Path::new(name)))
}
