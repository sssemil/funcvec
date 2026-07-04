use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Table,
    Json,
    Markdown,
}

#[derive(Debug, Clone)]
pub struct ReportConfig {
    pub provider: crate::embeddings::ProviderKind,
    pub model: Option<String>,
    pub ollama_host: String,
    pub allow_nonlocal_ollama_host: bool,
    pub ollama_timeout_secs: u64,
    pub ollama_keep_alive: Option<String>,
    pub ollama_dimensions: Option<usize>,
    pub ollama_truncate: bool,
    pub model_cache_dir: Option<PathBuf>,
    pub native_threads: Option<usize>,
    pub threshold: f32,
    pub top_k: usize,
    pub min_lines: usize,
    pub min_tokens: usize,
    pub cache_dir: Option<PathBuf>,
    pub allow_source_upload: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            provider: crate::embeddings::ProviderKind::Lexical,
            model: None,
            ollama_host: default_ollama_host(),
            allow_nonlocal_ollama_host: false,
            ollama_timeout_secs: 120,
            ollama_keep_alive: None,
            ollama_dimensions: None,
            ollama_truncate: true,
            model_cache_dir: None,
            native_threads: None,
            threshold: 0.72,
            top_k: 25,
            min_lines: 3,
            min_tokens: 12,
            cache_dir: None,
            allow_source_upload: false,
        }
    }
}

pub fn default_ollama_host() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".to_owned())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRecord {
    pub id: String,
    pub name: String,
    pub file: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub source: String,
    pub normalized: String,
    pub token_count: usize,
    pub line_count: usize,
    pub content_hash: String,
    pub expected_group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSummary {
    pub id: String,
    pub name: String,
    pub file: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub token_count: usize,
    pub line_count: usize,
    pub expected_group: Option<String>,
}

impl From<&FunctionRecord> for FunctionSummary {
    fn from(function: &FunctionRecord) -> Self {
        Self {
            id: function.id.clone(),
            name: function.name.clone(),
            file: function.file.clone(),
            start_line: function.start_line,
            end_line: function.end_line,
            token_count: function.token_count,
            line_count: function.line_count,
            expected_group: function.expected_group.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub clone: f32,
    pub semantic: Option<f32>,
    pub hybrid: f32,
    pub clone_flag: bool,
    pub semantic_flag: bool,
    pub hybrid_flag: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub left: FunctionRecord,
    pub right: FunctionRecord,
    pub scores: ScoreBreakdown,
    pub reasons: Vec<String>,
    pub expected_match: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub project_root: PathBuf,
    pub provider: String,
    pub model: Option<String>,
    pub functions_count: usize,
    pub embedding_stats: EmbeddingStats,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub dimensions: Option<usize>,
    pub elapsed_ms: u64,
    pub model_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyEval {
    pub flagged: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub known_pairs: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub report: Report,
    pub functions: Vec<FunctionSummary>,
    pub clone: StrategyEval,
    pub semantic: StrategyEval,
    pub hybrid: StrategyEval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMatrixReport {
    pub project_root: PathBuf,
    pub provider: String,
    pub functions_count: usize,
    pub known_pairs: usize,
    pub threshold: f32,
    pub top_k: usize,
    pub functions: Vec<FunctionSummary>,
    pub models: Vec<ModelEvalResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvalResult {
    pub model: String,
    pub status: ModelEvalStatus,
    pub error_kind: Option<String>,
    pub error: Option<String>,
    pub report: Option<Report>,
    pub clone: Option<StrategyEval>,
    pub semantic: Option<StrategyEval>,
    pub hybrid: Option<StrategyEval>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelEvalStatus {
    Success,
    Failure,
}
