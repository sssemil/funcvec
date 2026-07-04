use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use cargo_metadata::{Metadata, MetadataCommand};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct ProjectFiles {
    pub root: PathBuf,
    pub files: Vec<PathBuf>,
}

pub fn discover_project(path: &Path) -> Result<ProjectFiles> {
    let path = path.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize project path {}",
            path.to_string_lossy()
        )
    })?;
    let manifest = manifest_for_path(&path)
        .with_context(|| format!("{} is not inside a Rust project", path.display()))?;
    let root = manifest
        .parent()
        .context("manifest has no parent directory")?
        .to_path_buf();
    let metadata = MetadataCommand::new()
        .manifest_path(&manifest)
        .no_deps()
        .exec()
        .ok();
    let scan_root = scan_root(&path, &root);
    let allowed_roots = workspace_package_roots(metadata.as_ref(), &scan_root);
    let mut files = Vec::new();

    if scan_root.is_file() {
        if scan_root.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(scan_root);
        }
        return Ok(ProjectFiles { root, files });
    }

    for entry in WalkDir::new(&scan_root) {
        let entry = entry.with_context(|| format!("failed to walk {}", scan_root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let file = entry.path();
        if file.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        if should_skip(file) {
            continue;
        }
        if let Some(allowed_roots) = &allowed_roots
            && !allowed_roots.iter().any(|root| file.starts_with(root))
        {
            continue;
        }
        files.push(file.to_path_buf());
    }

    files.sort();
    Ok(ProjectFiles { root, files })
}

fn manifest_for_path(path: &Path) -> Option<PathBuf> {
    if path.is_file() && path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml") {
        Some(path.to_path_buf())
    } else if path.is_file() {
        find_ancestor_manifest(path)
    } else {
        let direct = path.join("Cargo.toml");
        if direct.exists() {
            Some(direct)
        } else {
            find_ancestor_manifest(path)
        }
    }
}

fn scan_root(path: &Path, manifest_root: &Path) -> PathBuf {
    if path.is_file() && path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml") {
        manifest_root.to_path_buf()
    } else {
        path.to_path_buf()
    }
}

fn workspace_package_roots(metadata: Option<&Metadata>, scan_root: &Path) -> Option<Vec<PathBuf>> {
    let metadata = metadata?;
    let workspace_root: PathBuf = metadata.workspace_root.clone().into();
    if workspace_root != scan_root {
        return None;
    }

    let roots: Vec<PathBuf> = metadata
        .packages
        .iter()
        .filter_map(|package| {
            let manifest_path: PathBuf = package.manifest_path.clone().into();
            manifest_path.parent().map(Path::to_path_buf)
        })
        .collect();
    if roots.is_empty() { None } else { Some(roots) }
}

fn find_ancestor_manifest(path: &Path) -> Option<PathBuf> {
    let mut dir = if path.is_file() { path.parent()? } else { path };
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists() {
            return Some(manifest);
        }
        dir = dir.parent()?;
    }
}

fn should_skip(path: &Path) -> bool {
    path.components().any(|component| {
        let text = component.as_os_str().to_string_lossy();
        matches!(
            text.as_ref(),
            "target" | ".git" | ".funcvec" | ".rfv" | "node_modules" | "vendor"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::discover_project;
    use std::path::PathBuf;

    #[test]
    fn member_path_does_not_widen_to_workspace() {
        let member = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let project = discover_project(&member).unwrap();

        assert_eq!(project.root, member.canonicalize().unwrap());
        assert!(project.files.iter().all(|file| file.starts_with(&member)));
    }

    #[test]
    fn workspace_root_uses_cargo_members_not_excluded_fixtures() {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let project = discover_project(&workspace).unwrap();

        assert!(
            project
                .files
                .iter()
                .all(|file| !file.to_string_lossy().contains("fixtures/dupe_lab"))
        );
    }
}
