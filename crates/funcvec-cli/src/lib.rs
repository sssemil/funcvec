use std::{env, ffi::OsString, path::PathBuf};

use anyhow::{Result, bail};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum, error::ErrorKind};
use funcvec_core::{
    ModelEvalStatus, OutputFormat, ProviderKind, ReportConfig, format_eval, format_eval_matrix,
    format_report, run_eval, run_eval_matrix, run_report,
};

#[derive(Debug, Parser)]
#[command(name = "funcvec", bin_name = "funcvec")]
#[command(
    about = "Find likely duplicate Rust functions with language-server extraction and vectors.",
    args_conflicts_with_subcommands = true
)]
#[command(after_help = "Examples:
  funcvec
  funcvec --provider none --top-k 10
  funcvec fixtures/dupe_lab --provider lexical --threshold 0.72
  funcvec report fixtures/dupe_lab --provider nomic
  funcvec eval fixtures/dupe_lab --provider nomic --models nomic-v1,nomic-v1.5
  cargo funcvec --provider none

Native Nomic downloads model files once into a local model cache and runs without a daemon.
Ollama setup, optional:
  ollama serve
  ollama pull nomic-embed-text")]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Report(ReportArgs),
    Eval(EvalArgs),
}

#[derive(Debug, Parser)]
struct ReportArgs {
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Debug, Parser)]
struct EvalArgs {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(long, value_delimiter = ',')]
    models: Vec<String>,
}

#[derive(Debug, Parser)]
struct CommonArgs {
    #[arg(default_value = ".")]
    path: PathBuf,

    #[arg(long, value_enum, default_value_t = CliFormat::Table)]
    format: CliFormat,

    #[arg(long, value_enum, default_value_t = CliProvider::Nomic)]
    provider: CliProvider,

    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    threshold: Option<f32>,

    #[arg(long, default_value_t = 25)]
    top_k: usize,

    #[arg(long, default_value_t = 3)]
    min_lines: usize,

    #[arg(long, default_value_t = 12)]
    min_tokens: usize,

    #[arg(long)]
    cache_dir: Option<PathBuf>,

    #[arg(long)]
    model_cache_dir: Option<PathBuf>,

    #[arg(long)]
    native_threads: Option<usize>,

    #[arg(long)]
    allow_source_upload: bool,

    #[arg(long)]
    ollama_host: Option<String>,

    #[arg(long)]
    allow_nonlocal_ollama_host: bool,

    #[arg(long, default_value_t = 120)]
    ollama_timeout_secs: u64,

    #[arg(long)]
    ollama_keep_alive: Option<String>,

    #[arg(long)]
    ollama_dimensions: Option<usize>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliFormat {
    Table,
    Json,
    Markdown,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliProvider {
    Openai,
    Ollama,
    Nomic,
    Lexical,
    None,
}

pub fn main() -> Result<()> {
    let args = parse_args(normalized_args()).unwrap_or_else(|error| error.exit());
    run(args)
}

fn run(args: Args) -> Result<()> {
    match args.command {
        None => run_report_command(args.common),
        Some(Command::Report(args)) => run_report_command(args.common),
        Some(Command::Eval(args)) => {
            let format = output_format(args.common.format);
            let config = config(&args.common)?;
            if config.provider == ProviderKind::Ollama || config.provider == ProviderKind::Nomic {
                let models = eval_models(&args, config.provider);
                let eval = run_eval_matrix(&args.common.path, &config, &models)?;
                if models.len() == 1
                    && let Some(result) = eval.models.first()
                    && matches!(result.status, ModelEvalStatus::Failure)
                {
                    bail!(
                        "{}",
                        result.error.as_deref().unwrap_or("provider model failed")
                    );
                }
                print!("{}", format_eval_matrix(&eval, format)?);
            } else {
                ensure_models_allowed(&args, config.provider)?;
                let eval = run_eval(&args.common.path, &config)?;
                print!("{}", format_eval(&eval, format)?);
            }
            Ok(())
        }
    }
}

fn ensure_models_allowed(args: &EvalArgs, provider: ProviderKind) -> Result<()> {
    if !args.models.is_empty()
        && provider != ProviderKind::Ollama
        && provider != ProviderKind::Nomic
    {
        bail!("--models is only valid with --provider ollama or --provider nomic");
    }
    Ok(())
}

fn normalized_args() -> Vec<OsString> {
    let mut args: Vec<OsString> = env::args_os().collect();
    let invoked_as_cargo_plugin = args
        .first()
        .and_then(|arg| std::path::Path::new(arg).file_stem())
        .is_some_and(|stem| stem == "cargo-funcvec");
    if invoked_as_cargo_plugin && args.get(1).is_some_and(|arg| arg == "funcvec") {
        args.remove(1);
    }
    args
}

fn run_report_command(args: CommonArgs) -> Result<()> {
    let format = output_format(args.format);
    let config = config(&args)?;
    if config.provider == ProviderKind::Ollama && config.model.is_none() {
        bail!("--model is required for `report --provider ollama`");
    }
    let report = run_report(&args.path, &config)?;
    print!("{}", format_report(&report, format)?);
    Ok(())
}

fn config(args: &CommonArgs) -> Result<ReportConfig> {
    let provider = provider_kind(args.provider);
    let threshold = args
        .threshold
        .unwrap_or_else(|| default_threshold(provider));
    if !(0.0..=1.0).contains(&threshold) {
        bail!("--threshold must be between 0.0 and 1.0");
    }
    if args.ollama_timeout_secs == 0 {
        bail!("--ollama-timeout-secs must be greater than 0");
    }
    if args.native_threads == Some(0) {
        bail!("--native-threads must be greater than 0");
    }
    let model = match (&args.model, provider) {
        (Some(model), _) => Some(model.clone()),
        (None, ProviderKind::Nomic) => Some(funcvec_core::default_nomic_model().to_owned()),
        (None, _) => None,
    };
    Ok(ReportConfig {
        provider,
        model,
        ollama_host: args
            .ollama_host
            .clone()
            .unwrap_or_else(funcvec_core::default_ollama_host),
        allow_nonlocal_ollama_host: args.allow_nonlocal_ollama_host,
        ollama_timeout_secs: args.ollama_timeout_secs,
        ollama_keep_alive: args.ollama_keep_alive.clone(),
        ollama_dimensions: args.ollama_dimensions,
        ollama_truncate: true,
        threshold,
        top_k: args.top_k,
        min_lines: args.min_lines,
        min_tokens: args.min_tokens,
        cache_dir: args.cache_dir.clone(),
        model_cache_dir: args.model_cache_dir.clone(),
        native_threads: args.native_threads,
        allow_source_upload: args.allow_source_upload,
    })
}

fn parse_args<I, T>(args: I) -> clap::error::Result<Args>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let parsed = Args::try_parse_from(args)?;
    reject_reserved_path_after_top_level_options(parsed)
}

#[cfg(test)]
fn verify_cli() {
    Args::command().debug_assert();
}

fn reject_reserved_path_after_top_level_options(args: Args) -> clap::error::Result<Args> {
    if args.command.is_none() {
        let path = args.common.path.to_string_lossy();
        if matches!(path.as_ref(), "report" | "eval") {
            return Err(Args::command().error(
                ErrorKind::ArgumentConflict,
                format!(
                    "`{path}` is a subcommand; put options after it, for example `funcvec {path} --provider none`"
                ),
            ));
        }
    }
    Ok(args)
}

fn output_format(format: CliFormat) -> OutputFormat {
    match format {
        CliFormat::Table => OutputFormat::Table,
        CliFormat::Json => OutputFormat::Json,
        CliFormat::Markdown => OutputFormat::Markdown,
    }
}

fn provider_kind(provider: CliProvider) -> ProviderKind {
    match provider {
        CliProvider::Openai => ProviderKind::OpenAi,
        CliProvider::Ollama => ProviderKind::Ollama,
        CliProvider::Nomic => ProviderKind::Nomic,
        CliProvider::Lexical => ProviderKind::Lexical,
        CliProvider::None => ProviderKind::None,
    }
}

fn default_threshold(provider: ProviderKind) -> f32 {
    match provider {
        ProviderKind::Nomic => 0.95,
        ProviderKind::Lexical | ProviderKind::None => 0.72,
        ProviderKind::OpenAi | ProviderKind::Ollama => 0.85,
    }
}

fn eval_models(args: &EvalArgs, provider: ProviderKind) -> Vec<String> {
    if !args.models.is_empty() {
        return args.models.clone();
    }
    if let Some(model) = &args.common.model {
        return vec![model.clone()];
    }
    let models = match provider {
        ProviderKind::Ollama => default_ollama_models(),
        ProviderKind::Nomic => default_nomic_models(),
        _ => Vec::new(),
    };
    models.into_iter().map(ToOwned::to_owned).collect()
}

fn default_ollama_models() -> Vec<&'static str> {
    vec![
        "embeddinggemma",
        "all-minilm",
        "nomic-embed-text",
        "mxbai-embed-large",
        "bge-m3",
        "qwen3-embedding:0.6b",
    ]
}

fn default_nomic_models() -> Vec<&'static str> {
    vec![funcvec_core::default_nomic_model()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;
    use funcvec_core::ProviderKind;

    #[test]
    fn clap_definition_is_valid() {
        verify_cli();
    }

    #[test]
    fn no_args_defaults_to_report_current_dir_with_nomic() {
        let args = parse_args(["funcvec"]).unwrap();

        assert!(args.command.is_none());
        assert_eq!(args.common.path, PathBuf::from("."));
        let config = config(&args.common).unwrap();
        assert_eq!(config.provider, ProviderKind::Nomic);
        assert_eq!(
            config.model.as_deref(),
            Some(funcvec_core::default_nomic_model())
        );
        assert_eq!(config.threshold, 0.95);
    }

    #[test]
    fn top_level_path_is_report_path() {
        let args = parse_args(["funcvec", "."]).unwrap();

        assert!(args.command.is_none());
        assert_eq!(args.common.path, PathBuf::from("."));
    }

    #[test]
    fn top_level_provider_none_uses_lexical_threshold() {
        let args = parse_args(["funcvec", "--provider", "none"]).unwrap();
        let config = config(&args.common).unwrap();

        assert_eq!(config.provider, ProviderKind::None);
        assert_eq!(config.threshold, 0.72);
    }

    #[test]
    fn report_subcommand_defaults_to_current_dir() {
        let args = parse_args(["funcvec", "report"]).unwrap();
        let Some(Command::Report(report)) = args.command else {
            panic!("expected report subcommand");
        };

        assert_eq!(report.common.path, PathBuf::from("."));
    }

    #[test]
    fn report_subcommand_accepts_path_and_local_options() {
        let args = parse_args(["funcvec", "report", ".", "--provider", "none"]).unwrap();
        let Some(Command::Report(report)) = args.command else {
            panic!("expected report subcommand");
        };
        let config = config(&report.common).unwrap();

        assert_eq!(report.common.path, PathBuf::from("."));
        assert_eq!(config.provider, ProviderKind::None);
    }

    #[test]
    fn eval_subcommand_uses_nomic_matrix_defaults() {
        let args = parse_args(["funcvec", "eval"]).unwrap();
        let Some(Command::Eval(eval)) = args.command else {
            panic!("expected eval subcommand");
        };
        let config = config(&eval.common).unwrap();

        assert_eq!(config.provider, ProviderKind::Nomic);
        assert_eq!(
            eval_models(&eval, config.provider),
            vec![funcvec_core::default_nomic_model().to_owned()]
        );
    }

    #[test]
    fn mixed_top_level_options_with_subcommands_are_rejected() {
        let error = parse_args(["funcvec", "--provider", "none", "report"]).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
    }

    #[test]
    fn models_are_rejected_for_non_matrix_providers() {
        let args = EvalArgs {
            common: CommonArgs {
                path: PathBuf::from("."),
                format: CliFormat::Table,
                provider: CliProvider::Lexical,
                model: None,
                threshold: None,
                top_k: 25,
                min_lines: 3,
                min_tokens: 12,
                cache_dir: None,
                model_cache_dir: None,
                native_threads: None,
                allow_source_upload: false,
                ollama_host: None,
                allow_nonlocal_ollama_host: false,
                ollama_timeout_secs: 120,
                ollama_keep_alive: None,
                ollama_dimensions: None,
            },
            models: vec!["nomic-v1.5".to_owned()],
        };
        let config = config(&args.common).unwrap();

        let error = ensure_models_allowed(&args, config.provider).unwrap_err();
        assert_eq!(
            error.to_string(),
            "--models is only valid with --provider ollama or --provider nomic"
        );
    }
}
