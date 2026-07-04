# funcvec

`funcvec` finds likely duplicate Rust functions. It uses rust-analyzer/LSP
symbols to discover functions, builds vectors for each function, and reports
candidate pairs that may be worth merging or refactoring.

The default mode runs local Nomic embeddings through Rust libraries. No Ollama
daemon or external embedding API is required for the default workflow.

## Install

```sh
cargo install funcvec
```

For local development from this repository:

```sh
cargo install --path crates/funcvec-cli
```

The install provides both `funcvec` and `cargo-funcvec`, so either form works:

```sh
funcvec
cargo funcvec
```

## Usage

Run from inside a Rust project:

```sh
funcvec
```

Common variants:

```sh
funcvec --provider none --top-k 10
funcvec path/to/project --provider lexical --threshold 0.72
funcvec report path/to/project --provider nomic
funcvec eval fixtures/dupe_lab --provider nomic --threshold 0.95
```

`funcvec` downloads native Nomic model files once into a local model cache. Set
`FUNCVEC_MODEL_CACHE_DIR` to control that location.

## Providers

- `nomic`: default local embedding provider.
- `lexical`: token-overlap scoring plus lexical vectors.
- `none`: no semantic embeddings, useful for fast smoke checks.
- `ollama`: use an Ollama embedding model.
- `openai`: reserved behind explicit source-upload opt-in.

## Publishing

Run the checks:

```sh
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo publish -p funcvec-core --dry-run
cargo package -p funcvec --list
```

Publish the core crate first, wait for it to appear in the crates.io index, then
publish the CLI crate:

```sh
cargo publish -p funcvec-core
cargo publish -p funcvec
```

The CLI crate depends on `funcvec-core = "0.1.0"`, so crates.io must know about
the core crate before publishing or dry-running `funcvec`.

## License

MIT
