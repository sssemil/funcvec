use std::{fs, path::Path};

use anyhow::{Context, Result};
use lsp_types::SymbolKind;

use crate::{
    discover,
    lsp_client::{RustAnalyzer, position_to_offset},
    models::{FunctionRecord, ReportConfig},
    normalize::{content_hash, normalize_source, tokens},
};

pub fn extract_functions(
    path: &Path,
    config: &ReportConfig,
) -> Result<(std::path::PathBuf, Vec<FunctionRecord>)> {
    let project = discover::discover_project(path)?;
    let mut analyzer = RustAnalyzer::start(&project.root)?;
    let mut functions = Vec::new();

    for file in &project.files {
        let text = fs::read_to_string(file)
            .with_context(|| format!("failed to read Rust source file {}", file.display()))?;
        let symbols = analyzer
            .document_symbols(file, &text)
            .with_context(|| format!("failed to read document symbols for {}", file.display()))?;

        for symbol in symbols {
            if !is_function_kind(symbol.kind) {
                continue;
            }

            let start = position_to_offset(&text, symbol.range.start);
            let end = position_to_offset(&text, symbol.range.end);
            if start >= end || end > text.len() {
                continue;
            }

            let source = text[start..end].to_owned();
            let normalized = normalize_source(&source);
            let token_count = tokens(&normalized).len();
            let start_line = symbol.selection_range.start.line as usize + 1;
            let end_line = symbol.range.end.line as usize + 1;
            let line_count = source.lines().count().max(1);

            if line_count < config.min_lines || token_count < config.min_tokens {
                continue;
            }
            if is_trivial(&normalized) {
                continue;
            }

            let normalized_hash = content_hash(&normalized);
            let relative = file.strip_prefix(&project.root).unwrap_or(file);
            let id_seed = format!(
                "{}:{}:{}:{}",
                relative.display(),
                symbol.name,
                start_line,
                normalized_hash
            );
            let id = content_hash(&id_seed)[..16].to_owned();
            let expected_group = expected_group_in_source(&source)
                .or_else(|| expected_group_before(&text, start_line));

            functions.push(FunctionRecord {
                id,
                name: symbol.name,
                file: relative.to_path_buf(),
                start_line,
                end_line,
                source,
                normalized,
                token_count,
                line_count,
                content_hash: normalized_hash,
                expected_group,
            });
        }
    }

    functions.sort_by(|a, b| {
        a.file
            .cmp(&b.file)
            .then(a.start_line.cmp(&b.start_line))
            .then(a.name.cmp(&b.name))
    });
    Ok((project.root, functions))
}

fn is_function_kind(kind: SymbolKind) -> bool {
    kind == SymbolKind::FUNCTION || kind == SymbolKind::METHOD || kind == SymbolKind::CONSTRUCTOR
}

fn is_trivial(normalized: &str) -> bool {
    let compact = normalized.replace(' ', "");
    compact.contains("{self.ID}")
        || compact.contains("{&self.ID}")
        || compact.contains("{*self.ID}")
}

fn expected_group_before(text: &str, start_line_one_based: usize) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let start = start_line_one_based.saturating_sub(1);
    let lower = start.saturating_sub(6);

    for line in lines[lower..start].iter().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.starts_with("///") && !trimmed.starts_with("//!") && !trimmed.starts_with("//")
        {
            break;
        }
        if let Some(group) = expected_group_marker(trimmed) {
            return Some(group.trim().to_owned());
        }
    }

    None
}

fn expected_group_in_source(source: &str) -> Option<String> {
    source.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .split_once("funcvec:group=")
            .or_else(|| trimmed.split_once("rfv:group="))
            .map(|(_, group)| group.trim().to_owned())
    })
}

fn expected_group_marker(text: &str) -> Option<&str> {
    text.split_once("funcvec:group=")
        .or_else(|| text.split_once("rfv:group="))
        .map(|(_, group)| group)
}

#[cfg(test)]
mod tests {
    use super::is_trivial;

    #[test]
    fn does_not_treat_closure_returning_identifier_as_trivial_getter() {
        assert!(!is_trivial(
            "fn ID ( ) { ID . ID ( | ID | if ID { ID } else { CHAR } ) }"
        ));
    }

    #[test]
    fn treats_simple_self_field_getter_as_trivial() {
        assert!(is_trivial("pub fn ID ( & self ) -> ID { self . ID }"));
    }
}
