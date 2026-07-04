use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct CachedEmbedding {
    vector: Vec<f32>,
}

pub fn cache_root(project_root: &Path, explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(|| project_root.join(".funcvec"))
}

pub fn load_embedding(cache_root: &Path, key: &str) -> Result<Option<Vec<f32>>> {
    let path = embedding_path(cache_root, key);
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read embedding cache {}", path.display()))?;
    let cached: CachedEmbedding = serde_json::from_str(&text)?;
    Ok(Some(cached.vector))
}

pub fn save_embedding(cache_root: &Path, key: &str, vector: &[f32]) -> Result<()> {
    let path = embedding_path(cache_root, key);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let cached = CachedEmbedding {
        vector: vector.to_vec(),
    };
    fs::write(&path, serde_json::to_vec_pretty(&cached)?)?;
    Ok(())
}

fn embedding_path(cache_root: &Path, key: &str) -> PathBuf {
    cache_root.join("embeddings").join(format!("{key}.json"))
}
