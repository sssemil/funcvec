use sha2::{Digest, Sha256};

pub fn content_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    hex::encode(digest)
}

pub fn normalize_source(source: &str) -> String {
    let mut out = String::new();
    let mut ident = String::new();
    let mut in_string = false;
    let mut chars = source.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_string {
            if ch == '\\' {
                let _ = chars.next();
            } else if ch == '"' {
                in_string = false;
                out.push_str(" STR ");
            }
            continue;
        }

        if ch == '/' && matches!(chars.peek(), Some('/')) {
            flush_ident(&mut ident, &mut out);
            for next in chars.by_ref() {
                if next == '\n' {
                    break;
                }
            }
        } else if ch == '"' {
            flush_ident(&mut ident, &mut out);
            in_string = true;
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            ident.push(ch);
        } else if ch.is_ascii_digit() {
            flush_ident(&mut ident, &mut out);
            while matches!(chars.peek(), Some(next) if next.is_ascii_digit() || *next == '_') {
                let _ = chars.next();
            }
            out.push_str(" NUM ");
        } else {
            flush_ident(&mut ident, &mut out);
            if !ch.is_whitespace() {
                out.push(' ');
                out.push(ch);
                out.push(' ');
            }
        }
    }

    flush_ident(&mut ident, &mut out);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn tokens(normalized: &str) -> Vec<String> {
    normalized
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect()
}

fn flush_ident(ident: &mut String, out: &mut String) {
    if ident.is_empty() {
        return;
    }

    let keyword = matches!(
        ident.as_str(),
        "fn" | "pub"
            | "impl"
            | "trait"
            | "for"
            | "if"
            | "else"
            | "match"
            | "while"
            | "loop"
            | "return"
            | "let"
            | "mut"
            | "self"
            | "Self"
            | "struct"
            | "enum"
            | "use"
            | "mod"
            | "async"
            | "await"
            | "move"
            | "where"
            | "const"
            | "static"
            | "ref"
            | "in"
            | "break"
            | "continue"
            | "Some"
            | "None"
            | "Ok"
            | "Err"
    );

    out.push(' ');
    if keyword {
        out.push_str(ident);
    } else {
        out.push_str("ID");
    }
    out.push(' ');
    ident.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_identifiers_and_literals() {
        let a = normalize_source("fn add_price(total: i32) -> i32 { total + 42 }");
        let b = normalize_source("fn sum_cost(value: i32) -> i32 { value + 7 }");
        assert_eq!(a, b);
    }

    #[test]
    fn keeps_code_after_url_like_string_literals() {
        let normalized =
            normalize_source(r#"fn endpoint() -> &'static str { "https://example.com"; "done" }"#);
        assert!(normalized.contains("STR ; STR"));
    }
}
