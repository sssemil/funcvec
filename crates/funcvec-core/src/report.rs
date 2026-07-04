use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use anyhow::Result;

use crate::{
    cache, embeddings, extract,
    models::{
        Candidate, EvalMatrixReport, EvalReport, FunctionRecord, FunctionSummary, ModelEvalResult,
        ModelEvalStatus, OutputFormat, Report, ReportConfig, StrategyEval,
    },
    score,
};

pub fn run_report(path: &Path, config: &ReportConfig) -> Result<Report> {
    let (project_root, functions) = extract::extract_functions(path, config)?;
    build_report(project_root, functions, config, config.top_k)
}

pub fn run_eval(path: &Path, config: &ReportConfig) -> Result<EvalReport> {
    let display_top_k = config.top_k;
    let (project_root, functions) = extract::extract_functions(path, config)?;
    let known_pairs = known_pair_count(&functions);
    let function_summaries = function_summaries(&functions);
    let (report, all_candidates) =
        build_eval_report(project_root, functions, config, display_top_k)?;

    Ok(EvalReport {
        functions: function_summaries,
        clone: strategy_eval(&all_candidates, known_pairs, |candidate| {
            candidate.scores.clone_flag
        }),
        semantic: strategy_eval(&all_candidates, known_pairs, |candidate| {
            candidate.scores.semantic_flag
        }),
        hybrid: strategy_eval(&all_candidates, known_pairs, |candidate| {
            candidate.scores.hybrid_flag
        }),
        report,
    })
}

pub fn run_eval_matrix(
    path: &Path,
    config: &ReportConfig,
    models: &[String],
) -> Result<EvalMatrixReport> {
    let display_top_k = config.top_k;
    let (project_root, functions) = extract::extract_functions(path, config)?;
    let known_pairs = known_pair_count(&functions);
    let function_summaries = function_summaries(&functions);
    let mut results = Vec::with_capacity(models.len());

    for model in models {
        let mut model_config = config.clone();
        model_config.model = Some(model.clone());
        match build_eval_report(
            project_root.clone(),
            functions.clone(),
            &model_config,
            display_top_k,
        ) {
            Ok((report, all_candidates)) => {
                results.push(ModelEvalResult {
                    model: model.clone(),
                    status: ModelEvalStatus::Success,
                    error_kind: None,
                    error: None,
                    clone: Some(strategy_eval(&all_candidates, known_pairs, |candidate| {
                        candidate.scores.clone_flag
                    })),
                    semantic: Some(strategy_eval(&all_candidates, known_pairs, |candidate| {
                        candidate.scores.semantic_flag
                    })),
                    hybrid: Some(strategy_eval(&all_candidates, known_pairs, |candidate| {
                        candidate.scores.hybrid_flag
                    })),
                    report: Some(report),
                });
            }
            Err(error) => {
                results.push(ModelEvalResult {
                    model: model.clone(),
                    status: ModelEvalStatus::Failure,
                    error_kind: Some(classify_eval_error(&error).to_owned()),
                    error: Some(error.to_string()),
                    report: None,
                    clone: None,
                    semantic: None,
                    hybrid: None,
                });
            }
        }
    }

    Ok(EvalMatrixReport {
        project_root,
        provider: config.provider.as_str().to_owned(),
        functions_count: functions.len(),
        known_pairs,
        threshold: config.threshold,
        top_k: display_top_k,
        functions: function_summaries,
        models: results,
    })
}

fn build_report(
    project_root: std::path::PathBuf,
    functions: Vec<crate::models::FunctionRecord>,
    config: &ReportConfig,
    top_k: usize,
) -> Result<Report> {
    let cache_root = cache::cache_root(&project_root, config.cache_dir.as_deref());
    let (embeddings, embedding_stats) =
        embeddings::embeddings_for(&functions, config, &cache_root)?;
    let candidates = score::score_candidates(&functions, &embeddings, config.threshold, top_k);

    Ok(Report {
        project_root,
        provider: config.provider.as_str().to_owned(),
        model: config.model.clone(),
        functions_count: functions.len(),
        embedding_stats,
        candidates,
    })
}

fn build_eval_report(
    project_root: std::path::PathBuf,
    functions: Vec<crate::models::FunctionRecord>,
    config: &ReportConfig,
    display_top_k: usize,
) -> Result<(Report, Vec<Candidate>)> {
    let cache_root = cache::cache_root(&project_root, config.cache_dir.as_deref());
    let (embeddings, embedding_stats) =
        embeddings::embeddings_for(&functions, config, &cache_root)?;
    let all_candidates =
        score::score_candidates(&functions, &embeddings, config.threshold, usize::MAX);
    let mut display_candidates = all_candidates.clone();
    display_candidates.truncate(display_top_k);

    let report = Report {
        project_root,
        provider: config.provider.as_str().to_owned(),
        model: config.model.clone(),
        functions_count: functions.len(),
        embedding_stats,
        candidates: display_candidates,
    };

    Ok((report, all_candidates))
}

pub fn format_report(report: &Report, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(report)?),
        OutputFormat::Markdown => Ok(format_markdown(report)),
        OutputFormat::Table => Ok(format_table(report)),
    }
}

pub fn format_eval(eval: &EvalReport, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(eval)?),
        OutputFormat::Markdown => {
            let mut out = String::new();
            out.push_str("# Funcvec Eval\n\n");
            out.push_str(&format!(
                "- functions: {}\n- candidates: {}\n\n",
                eval.report.functions_count,
                eval.report.candidates.len()
            ));
            out.push_str(
                "| strategy | flagged | true positives | false positives | known pairs | precision | recall | f1 |\n",
            );
            out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
            write_eval_row(&mut out, "clone", &eval.clone);
            write_eval_row(&mut out, "semantic", &eval.semantic);
            write_eval_row(&mut out, "hybrid", &eval.hybrid);
            out.push_str("\n## Functions\n\n");
            append_functions_markdown(&mut out, &eval.functions);
            out.push('\n');
            out.push_str(&format_markdown(&eval.report));
            Ok(out)
        }
        OutputFormat::Table => {
            let mut out = String::new();
            out.push_str("strategy   flagged  true+  false+  known   prec  recall  f1\n");
            out.push_str("-------------------------------------------------------------\n");
            table_eval_row(&mut out, "clone", &eval.clone);
            table_eval_row(&mut out, "semantic", &eval.semantic);
            table_eval_row(&mut out, "hybrid", &eval.hybrid);
            out.push('\n');
            append_functions_table(&mut out, &eval.functions);
            out.push('\n');
            out.push_str(&format_table(&eval.report));
            Ok(out)
        }
    }
}

pub fn format_eval_matrix(eval: &EvalMatrixReport, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Json => Ok(serde_json::to_string_pretty(eval)?),
        OutputFormat::Markdown => Ok(format_eval_matrix_markdown(eval)),
        OutputFormat::Table => Ok(format_eval_matrix_table(eval)),
    }
}

fn classify_eval_error(error: &anyhow::Error) -> &'static str {
    let text = error.to_string().to_ascii_lowercase();
    if text.contains("could not connect") || text.contains("timed out") {
        "daemon_unavailable"
    } else if text.contains("ollama model") && text.contains("not available") {
        "model_missing"
    } else if text.contains("non-loopback") {
        "nonlocal_host_rejected"
    } else if text.contains("embedding") {
        "provider_response"
    } else {
        "provider_error"
    }
}

fn known_pair_count(functions: &[crate::models::FunctionRecord]) -> usize {
    let mut groups: HashMap<&str, HashSet<&str>> = HashMap::new();
    for function in functions {
        if let Some(group) = function.expected_group.as_deref() {
            groups.entry(group).or_default().insert(&function.id);
        }
    }
    groups
        .values()
        .map(|ids| {
            let count = ids.len();
            count.saturating_sub(1) * count / 2
        })
        .sum()
}

fn function_summaries(functions: &[FunctionRecord]) -> Vec<FunctionSummary> {
    functions.iter().map(FunctionSummary::from).collect()
}

fn strategy_eval(
    candidates: &[Candidate],
    known_pairs: usize,
    flagged: impl Fn(&Candidate) -> bool,
) -> StrategyEval {
    let mut result = StrategyEval {
        flagged: 0,
        true_positives: 0,
        false_positives: 0,
        known_pairs,
        precision: 0.0,
        recall: 0.0,
        f1: 0.0,
    };
    for candidate in candidates {
        if !flagged(candidate) {
            continue;
        }
        result.flagged += 1;
        if candidate.expected_match {
            result.true_positives += 1;
        } else {
            result.false_positives += 1;
        }
    }
    result.precision = if result.flagged == 0 {
        0.0
    } else {
        result.true_positives as f32 / result.flagged as f32
    };
    result.recall = if result.known_pairs == 0 {
        0.0
    } else {
        result.true_positives as f32 / result.known_pairs as f32
    };
    result.f1 = if result.precision + result.recall == 0.0 {
        0.0
    } else {
        2.0 * result.precision * result.recall / (result.precision + result.recall)
    };
    result
}

fn format_table(report: &Report) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "project: {}\nprovider: {}\nfunctions: {}\ncandidates: {}\ncache: {} hit / {} miss\nembedding dims: {}\n\n",
        report.project_root.display(),
        report.provider,
        report.functions_count,
        report.candidates.len(),
        report.embedding_stats.cache_hits,
        report.embedding_stats.cache_misses,
        report
            .embedding_stats
            .dimensions
            .map(|value| value.to_string())
            .unwrap_or_else(|| "--".to_owned())
    ));
    out.push_str("hybrid  clone  sem    left -> right\n");
    out.push_str("------------------------------------\n");
    for candidate in &report.candidates {
        let sem = candidate
            .scores
            .semantic
            .map(|score| format!("{score:.2}"))
            .unwrap_or_else(|| "--".to_owned());
        out.push_str(&format!(
            "{:.2}    {:.2}   {:<5}  {}:{} {} -> {}:{} {}\n",
            candidate.scores.hybrid,
            candidate.scores.clone,
            sem,
            candidate.left.file.display(),
            candidate.left.start_line,
            candidate.left.name,
            candidate.right.file.display(),
            candidate.right.start_line,
            candidate.right.name
        ));
        if !candidate.reasons.is_empty() {
            out.push_str(&format!(
                "        reasons: {}\n",
                candidate.reasons.join(", ")
            ));
        }
    }
    out
}

fn format_markdown(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("# Funcvec Report\n\n");
    out.push_str(&format!(
        "- project: `{}`\n- provider: `{}`\n- functions: `{}`\n- candidates: `{}`\n- cache: `{}` hit / `{}` miss\n- embedding dims: `{}`\n\n",
        report.project_root.display(),
        report.provider,
        report.functions_count,
        report.candidates.len(),
        report.embedding_stats.cache_hits,
        report.embedding_stats.cache_misses,
        report
            .embedding_stats
            .dimensions
            .map(|value| value.to_string())
            .unwrap_or_else(|| "--".to_owned())
    ));
    out.push_str("| hybrid | clone | semantic | left | right | reasons |\n");
    out.push_str("| ---: | ---: | ---: | --- | --- | --- |\n");
    for candidate in &report.candidates {
        let sem = candidate
            .scores
            .semantic
            .map(|score| format!("{score:.2}"))
            .unwrap_or_else(|| "--".to_owned());
        out.push_str(&format!(
            "| {:.2} | {:.2} | {} | `{}`:{} `{}` | `{}`:{} `{}` | {} |\n",
            candidate.scores.hybrid,
            candidate.scores.clone,
            sem,
            candidate.left.file.display(),
            candidate.left.start_line,
            candidate.left.name,
            candidate.right.file.display(),
            candidate.right.start_line,
            candidate.right.name,
            candidate.reasons.join(", ")
        ));
    }
    out
}

fn write_eval_row(out: &mut String, name: &str, eval: &StrategyEval) {
    out.push_str(&format!(
        "| {name} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2} |\n",
        eval.flagged,
        eval.true_positives,
        eval.false_positives,
        eval.known_pairs,
        eval.precision,
        eval.recall,
        eval.f1
    ));
}

fn table_eval_row(out: &mut String, name: &str, eval: &StrategyEval) {
    out.push_str(&format!(
        "{name:<10} {:>7} {:>6} {:>7} {:>6}  {:>5.2}  {:>6.2}  {:>4.2}\n",
        eval.flagged,
        eval.true_positives,
        eval.false_positives,
        eval.known_pairs,
        eval.precision,
        eval.recall,
        eval.f1
    ));
}

fn format_eval_matrix_table(eval: &EvalMatrixReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "project: {}\nprovider: {}\nfunctions: {}\nknown pairs: {}\nthreshold: {:.2}\n\n",
        eval.project_root.display(),
        eval.provider,
        eval.functions_count,
        eval.known_pairs,
        eval.threshold
    ));
    append_functions_table(&mut out, &eval.functions);
    out.push('\n');
    out.push_str(
        "model                         strategy   status   dims   hit  miss flagged  true+ false+  prec recall  f1  error\n",
    );
    out.push_str(
        "--------------------------------------------------------------------------------------------------------\n",
    );
    for result in &eval.models {
        match (
            &result.status,
            &result.report,
            &result.clone,
            &result.semantic,
            &result.hybrid,
        ) {
            (ModelEvalStatus::Success, Some(report), Some(clone), Some(semantic), Some(hybrid)) => {
                table_eval_matrix_row(&mut out, &result.model, "clone", report, clone);
                table_eval_matrix_row(&mut out, &result.model, "semantic", report, semantic);
                table_eval_matrix_row(&mut out, &result.model, "hybrid", report, hybrid);
            }
            _ => {
                out.push_str(&format!(
                    "{:<29} {:<10} failure  {:>4} {:>5} {:>5} {:>7} {:>6} {:>6}  {:>4} {:>6} {:>4}  {}\n",
                    result.model,
                    "-",
                    "--",
                    0,
                    0,
                    0,
                    0,
                    0,
                    "--",
                    "--",
                    "--",
                    result.error.as_deref().unwrap_or("unknown provider error")
                ));
            }
        }
    }
    out
}

fn table_eval_matrix_row(
    out: &mut String,
    model: &str,
    strategy: &str,
    report: &Report,
    eval: &StrategyEval,
) {
    out.push_str(&format!(
        "{model:<29} {strategy:<10} success  {dims:>4} {hit:>5} {miss:>5} {flagged:>7} {true_pos:>6} {false_pos:>6}  {precision:>4.2} {recall:>6.2} {f1:>4.2}  \n",
        dims = report
            .embedding_stats
            .dimensions
            .map(|value| value.to_string())
            .unwrap_or_else(|| "--".to_owned()),
        hit = report.embedding_stats.cache_hits,
        miss = report.embedding_stats.cache_misses,
        flagged = eval.flagged,
        true_pos = eval.true_positives,
        false_pos = eval.false_positives,
        precision = eval.precision,
        recall = eval.recall,
        f1 = eval.f1
    ));
}

fn format_eval_matrix_markdown(eval: &EvalMatrixReport) -> String {
    let mut out = String::new();
    out.push_str("# Funcvec Eval Matrix\n\n");
    out.push_str(&format!(
        "- project: `{}`\n- provider: `{}`\n- functions: `{}`\n- known pairs: `{}`\n- threshold: `{:.2}`\n\n",
        eval.project_root.display(),
        eval.provider,
        eval.functions_count,
        eval.known_pairs,
        eval.threshold
    ));
    out.push_str("## Functions\n\n");
    append_functions_markdown(&mut out, &eval.functions);
    out.push('\n');
    out.push_str("| model | strategy | status | dims | cache hit | cache miss | flagged | true+ | false+ | precision | recall | f1 | error |\n");
    out.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n");
    for result in &eval.models {
        match (
            &result.status,
            &result.report,
            &result.clone,
            &result.semantic,
            &result.hybrid,
        ) {
            (ModelEvalStatus::Success, Some(report), Some(clone), Some(semantic), Some(hybrid)) => {
                markdown_eval_matrix_row(&mut out, &result.model, "clone", report, clone);
                markdown_eval_matrix_row(&mut out, &result.model, "semantic", report, semantic);
                markdown_eval_matrix_row(&mut out, &result.model, "hybrid", report, hybrid);
            }
            _ => {
                out.push_str(&format!(
                    "| `{}` | - | failure | -- | 0 | 0 | 0 | 0 | 0 | -- | -- | -- | {} |\n",
                    result.model,
                    result.error.as_deref().unwrap_or("unknown provider error")
                ));
            }
        }
    }
    out
}

fn markdown_eval_matrix_row(
    out: &mut String,
    model: &str,
    strategy: &str,
    report: &Report,
    eval: &StrategyEval,
) {
    out.push_str(&format!(
        "| `{model}` | {strategy} | success | {dims} | {hit} | {miss} | {flagged} | {true_pos} | {false_pos} | {precision:.2} | {recall:.2} | {f1:.2} |  |\n",
        dims = report
            .embedding_stats
            .dimensions
            .map(|value| value.to_string())
            .unwrap_or_else(|| "--".to_owned()),
        hit = report.embedding_stats.cache_hits,
        miss = report.embedding_stats.cache_misses,
        flagged = eval.flagged,
        true_pos = eval.true_positives,
        false_pos = eval.false_positives,
        precision = eval.precision,
        recall = eval.recall,
        f1 = eval.f1
    ));
}

fn append_functions_table(out: &mut String, functions: &[FunctionSummary]) {
    out.push_str("functions\n");
    out.push_str("---------\n");
    for function in functions {
        out.push_str(&format!(
            "{}:{}-{}  {:<34} lines={:<3} tokens={:<3} group={}\n",
            function.file.display(),
            function.start_line,
            function.end_line,
            function.name,
            function.line_count,
            function.token_count,
            function.expected_group.as_deref().unwrap_or("-")
        ));
    }
}

fn append_functions_markdown(out: &mut String, functions: &[FunctionSummary]) {
    out.push_str("| function | location | lines | tokens | expected group |\n");
    out.push_str("| --- | --- | ---: | ---: | --- |\n");
    for function in functions {
        out.push_str(&format!(
            "| `{}` | `{}`:{}-{} | {} | {} | {} |\n",
            function.name,
            function.file.display(),
            function.start_line,
            function.end_line,
            function.line_count,
            function.token_count,
            function.expected_group.as_deref().unwrap_or("-")
        ));
    }
}
