use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio},
};

use anyhow::{Context, Result, anyhow, bail};
use lsp_types::{
    DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse, InitializeParams,
    InitializedParams, PartialResultParams, Position, Range, SymbolKind,
    TextDocumentClientCapabilities, TextDocumentIdentifier, TextDocumentItem,
    TextDocumentSyncClientCapabilities, Uri, WorkDoneProgressParams, WorkspaceFolder,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct SymbolRange {
    pub name: String,
    pub kind: SymbolKind,
    pub range: Range,
    pub selection_range: Range,
}

pub struct RustAnalyzer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr: Option<ChildStderr>,
    next_id: i64,
}

impl RustAnalyzer {
    pub fn start(root: &Path) -> Result<Self> {
        let executable = rust_analyzer_path();
        let mut child = Command::new(&executable)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context(
                "failed to start rust-analyzer; install it with `rustup component add rust-analyzer rust-src` or set FUNCVEC_RUST_ANALYZER",
            )?;

        let stdin = child.stdin.take().context("rust-analyzer stdin missing")?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .context("rust-analyzer stdout missing")?,
        );
        let stderr = child.stderr.take();
        let mut this = Self {
            child,
            stdin,
            stdout,
            stderr,
            next_id: 1,
        };
        this.initialize(root)?;
        Ok(this)
    }

    pub fn document_symbols(&mut self, file: &Path, text: &str) -> Result<Vec<SymbolRange>> {
        let uri = file_uri(file)?;
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": TextDocumentItem {
                    uri: uri.clone(),
                    language_id: "rust".to_owned(),
                    version: 1,
                    text: text.to_owned(),
                }
            }),
        )?;

        let params = DocumentSymbolParams {
            text_document: TextDocumentIdentifier { uri },
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: PartialResultParams::default(),
        };

        let response: Option<DocumentSymbolResponse> =
            self.request("textDocument/documentSymbol", serde_json::to_value(params)?)?;

        let mut out = Vec::new();
        if let Some(response) = response {
            match response {
                DocumentSymbolResponse::Nested(symbols) => {
                    for symbol in symbols {
                        collect_symbol(None, symbol, &mut out);
                    }
                }
                DocumentSymbolResponse::Flat(symbols) => {
                    for symbol in symbols {
                        out.push(SymbolRange {
                            name: symbol.name,
                            kind: symbol.kind,
                            range: symbol.location.range,
                            selection_range: symbol.location.range,
                        });
                    }
                }
            }
        }
        Ok(out)
    }

    fn initialize(&mut self, root: &Path) -> Result<()> {
        let uri = file_uri(root)?;
        let name = root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("workspace")
            .to_owned();

        #[allow(deprecated)]
        let params = InitializeParams {
            process_id: None,
            root_path: None,
            root_uri: Some(uri.clone()),
            initialization_options: None,
            capabilities: lsp_types::ClientCapabilities {
                text_document: Some(TextDocumentClientCapabilities {
                    synchronization: Some(TextDocumentSyncClientCapabilities {
                        dynamic_registration: Some(false),
                        will_save: Some(false),
                        will_save_wait_until: Some(false),
                        did_save: Some(false),
                    }),
                    document_symbol: Some(lsp_types::DocumentSymbolClientCapabilities {
                        dynamic_registration: Some(false),
                        symbol_kind: None,
                        hierarchical_document_symbol_support: Some(true),
                        tag_support: None,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            trace: None,
            workspace_folders: Some(vec![WorkspaceFolder { uri, name }]),
            client_info: None,
            locale: None,
            work_done_progress_params: WorkDoneProgressParams::default(),
        };

        let _: Value = self.request("initialize", serde_json::to_value(params)?)?;
        self.notify("initialized", serde_json::to_value(InitializedParams {})?)?;
        Ok(())
    }

    fn request<T: DeserializeOwned>(&mut self, method: &str, params: Value) -> Result<T> {
        let id = self.next_id;
        self.next_id += 1;
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))?;

        loop {
            let message = self.read_message().with_context(|| {
                format!("rust-analyzer did not return a valid response for {method}")
            })?;
            if message.get("id").and_then(Value::as_i64) != Some(id) {
                self.handle_unrelated_message(&message)?;
                continue;
            }
            if let Some(error) = message.get("error") {
                bail!("rust-analyzer request `{method}` failed: {error}");
            }
            let result = message.get("result").cloned().unwrap_or(Value::Null);
            return serde_json::from_value(result)
                .with_context(|| format!("failed to decode rust-analyzer response for {method}"));
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
    }

    fn handle_unrelated_message(&mut self, message: &Value) -> Result<()> {
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return Ok(());
        };
        let Some(id) = message.get("id").cloned() else {
            return Ok(());
        };

        self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": Value::Null,
        }))
        .with_context(|| format!("failed to respond to rust-analyzer request `{method}`"))
    }

    fn write_message(&mut self, value: &Value) -> Result<()> {
        let body = serde_json::to_vec(value)?;
        write!(self.stdin, "Content-Length: {}\r\n\r\n", body.len())?;
        self.stdin.write_all(&body)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_message(&mut self) -> Result<Value> {
        let mut headers = HashMap::new();
        loop {
            let mut line = String::new();
            let count = self.stdout.read_line(&mut line)?;
            if count == 0 {
                let mut stderr = String::new();
                if let Some(mut child_stderr) = self.stderr.take() {
                    let _ = child_stderr.read_to_string(&mut stderr);
                }
                if !stderr.trim().is_empty() {
                    bail!("rust-analyzer exited before responding: {}", stderr.trim());
                }
                bail!(
                    "rust-analyzer exited before responding; install it with `rustup component add rust-analyzer rust-src`"
                );
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }
            if let Some((key, value)) = trimmed.split_once(':') {
                headers.insert(key.to_ascii_lowercase(), value.trim().to_owned());
            }
        }

        let len = headers
            .get("content-length")
            .ok_or_else(|| anyhow!("rust-analyzer response missing Content-Length"))?
            .parse::<usize>()?;
        let mut body = vec![0; len];
        self.stdout.read_exact(&mut body)?;
        Ok(serde_json::from_slice(&body)?)
    }
}

impl Drop for RustAnalyzer {
    fn drop(&mut self) {
        let _ = self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "shutdown",
        }));
        let _ = self.write_message(&json!({
            "jsonrpc": "2.0",
            "method": "exit",
        }));
        let _ = self.child.kill();
    }
}

fn collect_symbol(parent: Option<&str>, symbol: DocumentSymbol, out: &mut Vec<SymbolRange>) {
    let name = match parent {
        Some(parent) => format!("{parent}::{}", symbol.name),
        None => symbol.name.clone(),
    };

    out.push(SymbolRange {
        name: name.clone(),
        kind: symbol.kind,
        range: symbol.range,
        selection_range: symbol.selection_range,
    });

    if let Some(children) = symbol.children {
        for child in children {
            collect_symbol(Some(&name), child, out);
        }
    }
}

fn file_uri(path: &Path) -> Result<Uri> {
    let url = url::Url::from_file_path(path)
        .map_err(|_| anyhow!("failed to convert path to file URI: {}", path.display()))?;
    Ok(url.as_str().parse()?)
}

fn rust_analyzer_path() -> PathBuf {
    if let Some(path) =
        std::env::var_os("FUNCVEC_RUST_ANALYZER").or_else(|| std::env::var_os("RFV_RUST_ANALYZER"))
    {
        return PathBuf::from(path);
    }

    if let Ok(output) = Command::new("rustup")
        .args(["which", "rust-analyzer"])
        .output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }

    PathBuf::from("rust-analyzer")
}

pub fn position_to_offset(text: &str, position: Position) -> usize {
    let mut line = 0_u32;
    let mut line_start = 0;
    for (idx, ch) in text.char_indices() {
        if line == position.line {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = idx + ch.len_utf8();
        }
    }

    let mut units = 0_u32;
    for (relative, ch) in text[line_start..].char_indices() {
        if ch == '\n' || units >= position.character {
            return line_start + relative;
        }
        units += ch.len_utf16() as u32;
    }
    text.len()
}
