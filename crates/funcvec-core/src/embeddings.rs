use std::{
    net::IpAddr,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    cache,
    models::{EmbeddingStats, FunctionRecord, ReportConfig},
    normalize::content_hash,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAi,
    Ollama,
    Nomic,
    Lexical,
    None,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Ollama => "ollama",
            Self::Nomic => "nomic",
            Self::Lexical => "lexical",
            Self::None => "none",
        }
    }
}

pub fn embeddings_for(
    functions: &[FunctionRecord],
    config: &ReportConfig,
    cache_root: &Path,
) -> Result<(Vec<Option<Vec<f32>>>, EmbeddingStats)> {
    let started = Instant::now();
    match config.provider {
        ProviderKind::None => Ok((vec![None; functions.len()], elapsed_stats(started))),
        ProviderKind::Lexical => {
            let vectors: Vec<_> = functions
                .iter()
                .map(|function| Some(lexical_embedding(&function.normalized)))
                .collect();
            let mut stats = elapsed_stats(started);
            stats.cache_misses = functions.len();
            stats.dimensions = vectors
                .iter()
                .find_map(|vector| vector.as_ref().map(Vec::len));
            Ok((vectors, stats))
        }
        ProviderKind::OpenAi => {
            if !config.allow_source_upload {
                bail!(
                    "openai provider would send source-derived text; rerun with --allow-source-upload to opt in"
                );
            }
            let model = config.model.as_deref().unwrap_or("text-embedding-3-small");
            let provider =
                OpenAiProvider::new(model, Duration::from_secs(config.ollama_timeout_secs))?;
            let mut out = Vec::with_capacity(functions.len());
            let mut stats = elapsed_stats(started);
            for function in functions {
                let text = embedding_text(function);
                let key = content_hash(&format!("openai:{model}:text={}", content_hash(&text)));
                if let Some(vector) = cache::load_embedding(cache_root, &key)? {
                    stats.cache_hits += 1;
                    stats.dimensions.get_or_insert(vector.len());
                    out.push(Some(vector));
                    continue;
                }
                let vector = provider.embed(&text)?;
                stats.cache_misses += 1;
                stats.dimensions.get_or_insert(vector.len());
                cache::save_embedding(cache_root, &key, &vector)?;
                out.push(Some(vector));
            }
            stats.elapsed_ms = elapsed_ms(started);
            Ok((out, stats))
        }
        ProviderKind::Ollama => {
            let model = config
                .model
                .as_deref()
                .context("--model is required when using --provider ollama")?;
            let provider = OllamaProvider::new(config, model)?;
            provider.embed_functions(functions, cache_root, started)
        }
        ProviderKind::Nomic => {
            let provider = NativeNomicProvider::new(config)?;
            provider.embed_functions(functions, cache_root, started)
        }
    }
}

fn embedding_text(function: &FunctionRecord) -> String {
    format!(
        "name: {}\nlines: {}-{}\ncode:\n{}",
        function.name, function.start_line, function.end_line, function.normalized
    )
}

fn lexical_embedding(text: &str) -> Vec<f32> {
    const DIMS: usize = 96;
    let mut vector = vec![0.0; DIMS];
    for token in text.split_whitespace() {
        let hash = content_hash(token);
        let bucket = usize::from_str_radix(&hash[..8], 16).unwrap_or(0) % DIMS;
        vector[bucket] += 1.0;
    }
    normalize(&mut vector);
    vector
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn elapsed_stats(started: Instant) -> EmbeddingStats {
    EmbeddingStats {
        elapsed_ms: elapsed_ms(started),
        ..EmbeddingStats::default()
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[derive(Debug, Clone)]
struct NomicModel {
    alias: &'static str,
    model: EmbeddingModel,
}

struct NativeNomicProvider {
    model: NomicModel,
    model_cache_dir: PathBuf,
    native_threads: Option<usize>,
}

impl NativeNomicProvider {
    fn new(config: &ReportConfig) -> Result<Self> {
        let model = parse_nomic_model(config.model.as_deref())?;
        let model_cache_dir = model_cache_dir(config)?;
        Ok(Self {
            model,
            model_cache_dir,
            native_threads: config.native_threads,
        })
    }

    fn embed_functions(
        &self,
        functions: &[FunctionRecord],
        cache_root: &Path,
        started: Instant,
    ) -> Result<(Vec<Option<Vec<f32>>>, EmbeddingStats)> {
        let mut stats = elapsed_stats(started);
        let mut out = vec![None; functions.len()];
        let mut pending_indices = Vec::new();
        let mut pending_inputs = Vec::new();
        let mut pending_keys = Vec::new();

        for (idx, function) in functions.iter().enumerate() {
            let text = nomic_embedding_text(function);
            let key = self.cache_key(function, &text);
            if let Some(vector) = cache::load_embedding(cache_root, &key)? {
                stats.cache_hits += 1;
                stats.dimensions.get_or_insert(vector.len());
                out[idx] = Some(vector);
            } else {
                pending_indices.push(idx);
                pending_inputs.push(text);
                pending_keys.push(key);
            }
        }

        if !pending_inputs.is_empty() {
            let mut options = TextInitOptions::new(self.model.model.clone())
                .with_cache_dir(self.model_cache_dir.clone())
                .with_show_download_progress(false);
            if let Some(threads) = self.native_threads {
                options = options.with_intra_threads(threads);
            }

            let mut model = TextEmbedding::try_new(options).with_context(|| {
                format!(
                    "failed to initialize native Nomic model `{}`",
                    self.model.alias
                )
            })?;
            let vectors = model.embed(&pending_inputs, None).with_context(|| {
                format!(
                    "failed to embed {} functions with native Nomic model `{}`",
                    pending_inputs.len(),
                    self.model.alias
                )
            })?;
            if vectors.len() != pending_inputs.len() {
                bail!(
                    "native Nomic returned {} embeddings for {} inputs",
                    vectors.len(),
                    pending_inputs.len()
                );
            }

            for (idx, (key, vector)) in pending_indices
                .into_iter()
                .zip(pending_keys.into_iter().zip(vectors))
            {
                let dimension = vector.len();
                if let Some(expected) = stats.dimensions {
                    if expected != dimension {
                        bail!(
                            "native Nomic returned inconsistent embedding dimensions: expected {expected}, got {dimension}"
                        );
                    }
                } else {
                    stats.dimensions = Some(dimension);
                }
                cache::save_embedding(cache_root, &key, &vector)?;
                stats.cache_misses += 1;
                out[idx] = Some(vector);
            }
        }

        stats.elapsed_ms = elapsed_ms(started);
        Ok((out, stats))
    }

    fn cache_key(&self, function: &FunctionRecord, embedding_text: &str) -> String {
        content_hash(&nomic_cache_key_seed(
            self.model.alias,
            function,
            embedding_text,
        ))
    }
}

fn nomic_cache_key_seed(
    model_alias: &str,
    function: &FunctionRecord,
    embedding_text: &str,
) -> String {
    format!(
        "nomic-fastembed-v1:model={model_alias}:prefix=clustering:function={}:text={}",
        function.content_hash,
        content_hash(embedding_text)
    )
}

fn nomic_embedding_text(function: &FunctionRecord) -> String {
    format!("clustering: {}", embedding_text(function))
}

fn parse_nomic_model(model: Option<&str>) -> Result<NomicModel> {
    match model.unwrap_or(default_nomic_model()) {
        "nomic-v1" | "nomic-embed-text-v1" => Ok(NomicModel {
            alias: "nomic-v1",
            model: EmbeddingModel::NomicEmbedTextV1,
        }),
        "nomic-v1.5" | "nomic-embed-text-v1.5" | "nomic-embed-text" => Ok(NomicModel {
            alias: "nomic-v1.5",
            model: EmbeddingModel::NomicEmbedTextV15,
        }),
        value => bail!(
            "unsupported native Nomic model `{value}`; supported models: nomic-v1, nomic-v1.5"
        ),
    }
}

pub fn default_nomic_model() -> &'static str {
    "nomic-v1.5"
}

fn model_cache_dir(config: &ReportConfig) -> Result<PathBuf> {
    if let Some(path) = &config.model_cache_dir {
        return Ok(path.clone());
    }
    if let Some(path) = std::env::var_os("FUNCVEC_MODEL_CACHE_DIR")
        .or_else(|| std::env::var_os("RFV_MODEL_CACHE_DIR"))
    {
        return Ok(PathBuf::from(path));
    }
    let cache_dir = dirs::cache_dir()
        .context("could not determine OS cache directory; pass --model-cache-dir")?;
    Ok(cache_dir.join("funcvec").join("models"))
}

struct OllamaProvider {
    client: reqwest::blocking::Client,
    host: String,
    model: String,
    keep_alive: Option<String>,
    dimensions: Option<usize>,
    truncate: bool,
}

impl OllamaProvider {
    fn new(config: &ReportConfig, model: &str) -> Result<Self> {
        let host = normalize_ollama_host(&config.ollama_host)?;
        if !config.allow_nonlocal_ollama_host && !is_loopback_url(&host)? {
            bail!(
                "refusing to send source-derived text to non-loopback Ollama host `{host}`; rerun with --allow-nonlocal-ollama-host to opt in"
            );
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.ollama_timeout_secs))
            .build()?;
        Ok(Self {
            client,
            host,
            model: model.to_owned(),
            keep_alive: config.ollama_keep_alive.clone(),
            dimensions: config.ollama_dimensions,
            truncate: config.ollama_truncate,
        })
    }

    fn embed_functions(
        &self,
        functions: &[FunctionRecord],
        cache_root: &Path,
        started: Instant,
    ) -> Result<(Vec<Option<Vec<f32>>>, EmbeddingStats)> {
        let model_digest = self.model_digest().unwrap_or(None);
        let mut stats = elapsed_stats(started);
        stats.model_digest = model_digest.clone();

        let mut out = vec![None; functions.len()];
        let mut pending_indices = Vec::new();
        let mut pending_inputs = Vec::new();
        let mut pending_keys = Vec::new();

        for (idx, function) in functions.iter().enumerate() {
            let text = embedding_text(function);
            let key = self.cache_key(function, &text, model_digest.as_deref());
            if let Some(vector) = cache::load_embedding(cache_root, &key)? {
                stats.cache_hits += 1;
                stats.dimensions.get_or_insert(vector.len());
                out[idx] = Some(vector);
            } else {
                pending_indices.push(idx);
                pending_inputs.push(text);
                pending_keys.push(key);
            }
        }

        if !pending_inputs.is_empty() {
            let vectors = self.embed_batch(&pending_inputs)?;
            if vectors.len() != pending_inputs.len() {
                bail!(
                    "ollama returned {} embeddings for {} inputs",
                    vectors.len(),
                    pending_inputs.len()
                );
            }
            for (idx, (key, vector)) in pending_indices
                .into_iter()
                .zip(pending_keys.into_iter().zip(vectors))
            {
                let dimension = vector.len();
                if let Some(expected) = stats.dimensions {
                    if expected != dimension {
                        bail!(
                            "ollama returned inconsistent embedding dimensions: expected {expected}, got {dimension}"
                        );
                    }
                } else {
                    stats.dimensions = Some(dimension);
                }
                cache::save_embedding(cache_root, &key, &vector)?;
                stats.cache_misses += 1;
                out[idx] = Some(vector);
            }
        }

        stats.elapsed_ms = elapsed_ms(started);
        Ok((out, stats))
    }

    fn cache_key(
        &self,
        function: &FunctionRecord,
        embedding_text: &str,
        model_digest: Option<&str>,
    ) -> String {
        content_hash(&format!(
            "ollama:api_embed_v1:host={}:model={}:digest={}:truncate={}:dimensions={:?}:function={}:text={}",
            self.host,
            self.model,
            model_digest.unwrap_or("unknown"),
            self.truncate,
            self.dimensions,
            function.content_hash,
            content_hash(embedding_text)
        ))
    }

    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = OllamaEmbedRequest {
            model: &self.model,
            input: inputs,
            truncate: self.truncate,
            keep_alive: self.keep_alive.as_deref(),
            dimensions: self.dimensions,
        };
        let url = format!("{}/api/embed", self.host.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .map_err(|err| ollama_transport_error(err, &self.host))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            if status.as_u16() == 404 {
                bail!(
                    "ollama model `{}` is not available at {}; run `ollama pull {}` ({})",
                    self.model,
                    self.host,
                    self.model,
                    body.trim()
                );
            }
            bail!(
                "ollama embed request failed with HTTP {status} at {}: {}",
                self.host,
                body.trim()
            );
        }
        let response: OllamaEmbedResponse = response.json()?;
        Ok(response.embeddings)
    }

    fn model_digest(&self) -> Result<Option<String>> {
        let url = format!("{}/api/tags", self.host.trim_end_matches('/'));
        let response = self
            .client
            .get(url)
            .send()
            .map_err(|err| ollama_transport_error(err, &self.host))?;
        if !response.status().is_success() {
            return Ok(None);
        }
        let tags: OllamaTagsResponse = response.json()?;
        Ok(tags
            .models
            .into_iter()
            .find(|model| model.name == self.model || model.model == self.model)
            .and_then(|model| model.digest))
    }
}

fn normalize_ollama_host(host: &str) -> Result<String> {
    let trimmed = host.trim();
    if trimmed.is_empty() {
        bail!("--ollama-host cannot be empty");
    }
    let host = if trimmed.contains("://") {
        trimmed.to_owned()
    } else {
        format!("http://{trimmed}")
    };
    let parsed = url::Url::parse(&host).with_context(|| format!("invalid Ollama host `{host}`"))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        bail!("Ollama host must use http or https: {host}");
    }
    Ok(host.trim_end_matches('/').to_owned())
}

fn is_loopback_url(host: &str) -> Result<bool> {
    let parsed = url::Url::parse(host)?;
    let Some(host) = parsed.host_str() else {
        return Ok(false);
    };
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(true);
    }
    Ok(host.parse::<IpAddr>().is_ok_and(|addr| addr.is_loopback()))
}

fn ollama_transport_error(err: reqwest::Error, host: &str) -> anyhow::Error {
    if err.is_connect() {
        anyhow!("could not connect to Ollama at {host}; start it with `ollama serve`")
    } else if err.is_timeout() {
        anyhow!("timed out waiting for Ollama at {host}")
    } else {
        anyhow!("ollama request to {host} failed: {err}")
    }
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    truncate: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTagModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    name: String,
    model: String,
    digest: Option<String>,
}

struct OpenAiProvider {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
}

impl OpenAiProvider {
    fn new(model: &str, timeout: Duration) -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY is required when using --provider openai")?;
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(timeout)
                .build()?,
            api_key,
            model: model.to_owned(),
        })
    }

    fn embed(&self, input: &str) -> Result<Vec<f32>> {
        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "input": input,
            }))
            .send()?
            .error_for_status()?
            .json::<OpenAiEmbeddingResponse>()?;
        response
            .data
            .into_iter()
            .next()
            .map(|item| item.embedding)
            .context("OpenAI embedding response did not contain an embedding")
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingItem {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_embeddings_are_deterministic() {
        assert_eq!(lexical_embedding("a b c"), lexical_embedding("a b c"));
    }

    #[test]
    fn rejects_non_loopback_ollama_hosts_by_default() {
        let mut config = ReportConfig {
            provider: ProviderKind::Ollama,
            model: Some("nomic-embed-text".to_owned()),
            ollama_host: "http://example.com:11434".to_owned(),
            ..ReportConfig::default()
        };
        assert!(OllamaProvider::new(&config, "nomic-embed-text").is_err());

        config.allow_nonlocal_ollama_host = true;
        assert!(OllamaProvider::new(&config, "nomic-embed-text").is_ok());
    }

    #[test]
    fn accepts_loopback_ollama_hosts() {
        let config = ReportConfig {
            provider: ProviderKind::Ollama,
            model: Some("nomic-embed-text".to_owned()),
            ollama_host: "127.0.0.1:11434".to_owned(),
            ..ReportConfig::default()
        };
        assert!(OllamaProvider::new(&config, "nomic-embed-text").is_ok());
    }

    #[test]
    fn parses_native_nomic_model_aliases() {
        assert_eq!(parse_nomic_model(None).unwrap().alias, "nomic-v1.5");
        assert_eq!(
            parse_nomic_model(Some("nomic-embed-text-v1"))
                .unwrap()
                .alias,
            "nomic-v1"
        );
        assert_eq!(
            parse_nomic_model(Some("nomic-embed-text")).unwrap().alias,
            "nomic-v1.5"
        );
        assert!(parse_nomic_model(Some("nomic-v1.5-q")).is_err());
    }

    #[test]
    fn native_nomic_embedding_text_uses_clustering_prefix() {
        let function = sample_function();
        let text = nomic_embedding_text(&function);
        assert!(text.starts_with("clustering: name: sample"));
    }

    #[test]
    fn native_nomic_cache_seed_versions_embedding_behavior() {
        let function = sample_function();
        let text = nomic_embedding_text(&function);
        let seed = nomic_cache_key_seed("nomic-v1.5", &function, &text);
        assert!(seed.contains("nomic-fastembed-v1"));
        assert!(seed.contains("model=nomic-v1.5"));
        assert!(seed.contains("prefix=clustering"));
        assert!(seed.contains("function=abc123"));
    }

    #[test]
    fn native_nomic_smoke_test_is_explicitly_opted_in() {
        if std::env::var_os("FUNCVEC_RUN_NATIVE_MODEL_TESTS")
            .or_else(|| std::env::var_os("RFV_RUN_NATIVE_MODEL_TESTS"))
            .is_none()
        {
            return;
        }

        let config = ReportConfig {
            provider: ProviderKind::Nomic,
            model: Some(default_nomic_model().to_owned()),
            native_threads: Some(1),
            ..ReportConfig::default()
        };
        let provider = NativeNomicProvider::new(&config).unwrap();
        let function = sample_function();
        let cache_root = tempfile::tempdir().unwrap();
        let (embeddings, stats) = provider
            .embed_functions(&[function], cache_root.path(), Instant::now())
            .unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(stats.dimensions, Some(768));
    }

    fn sample_function() -> FunctionRecord {
        FunctionRecord {
            id: "id".to_owned(),
            name: "sample".to_owned(),
            file: "src/lib.rs".into(),
            start_line: 1,
            end_line: 3,
            source: "fn sample() -> i32 { 1 }".to_owned(),
            normalized: "fn ID ( ) -> ID { NUM }".to_owned(),
            token_count: 8,
            line_count: 3,
            content_hash: "abc123".to_owned(),
            expected_group: None,
        }
    }
}
