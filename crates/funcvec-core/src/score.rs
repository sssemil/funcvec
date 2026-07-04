use std::collections::HashSet;

use crate::models::{Candidate, FunctionRecord, ScoreBreakdown};

pub fn score_candidates(
    functions: &[FunctionRecord],
    embeddings: &[Option<Vec<f32>>],
    threshold: f32,
    top_k: usize,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();

    for left_idx in 0..functions.len() {
        for right_idx in (left_idx + 1)..functions.len() {
            let left = &functions[left_idx];
            let right = &functions[right_idx];
            let clone = token_jaccard(&left.normalized, &right.normalized);
            let semantic = embeddings[left_idx]
                .as_ref()
                .zip(embeddings[right_idx].as_ref())
                .map(|(left, right)| cosine(left, right));
            let hybrid = match semantic {
                Some(semantic) => (clone * 0.45) + (semantic * 0.55),
                None => clone,
            };

            let clone_flag = clone >= threshold;
            let semantic_flag = semantic.is_some_and(|score| score >= threshold);
            let hybrid_flag = hybrid >= threshold;
            if !clone_flag && !semantic_flag && !hybrid_flag {
                continue;
            }

            let mut reasons = Vec::new();
            if clone_flag {
                reasons.push("normalized tokens overlap".to_owned());
            }
            if semantic_flag {
                reasons.push("embedding vectors are close".to_owned());
            }
            if hybrid_flag {
                reasons.push("hybrid score crosses threshold".to_owned());
            }
            if left.name == right.name {
                reasons.push("same symbol name".to_owned());
            }

            let expected_match =
                left.expected_group.is_some() && left.expected_group == right.expected_group;
            let id = format!("{}:{}", left.id, right.id);
            candidates.push(Candidate {
                id,
                left: left.clone(),
                right: right.clone(),
                scores: ScoreBreakdown {
                    clone,
                    semantic,
                    hybrid,
                    clone_flag,
                    semantic_flag,
                    hybrid_flag,
                },
                reasons,
                expected_match,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.scores
            .hybrid
            .total_cmp(&a.scores.hybrid)
            .then(b.scores.clone.total_cmp(&a.scores.clone))
            .then(a.left.file.cmp(&b.left.file))
            .then(a.left.start_line.cmp(&b.left.start_line))
    });
    candidates.truncate(top_k);
    candidates
}

fn token_jaccard(left: &str, right: &str) -> f32 {
    let left: HashSet<&str> = left.split_whitespace().collect();
    let right: HashSet<&str> = right.split_whitespace().collect();
    if left.is_empty() && right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(&right).count() as f32;
    let union = left.union(&right).count() as f32;
    intersection / union
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_tokens_score_one() {
        assert_eq!(token_jaccard("fn ID ( )", "fn ID ( )"), 1.0);
    }
}
