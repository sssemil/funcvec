mod cache;
mod discover;
mod embeddings;
mod extract;
mod lsp_client;
mod models;
mod normalize;
mod report;
mod score;

pub use embeddings::{ProviderKind, default_nomic_model};
pub use models::{
    Candidate, EmbeddingStats, EvalMatrixReport, EvalReport, FunctionRecord, FunctionSummary,
    ModelEvalResult, ModelEvalStatus, OutputFormat, Report, ReportConfig, ScoreBreakdown,
    default_ollama_host,
};
pub use report::{
    format_eval, format_eval_matrix, format_report, run_eval, run_eval_matrix, run_report,
};
