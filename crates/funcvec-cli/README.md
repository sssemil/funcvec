# funcvec

`funcvec` finds likely duplicate Rust functions. It discovers functions through
rust-analyzer/LSP symbols, builds function vectors, and reports candidates that
look similar enough to review and potentially merge.

## Install

```sh
cargo install funcvec
```

For local development from this repository:

```sh
cargo install --path crates/funcvec-cli
```

The install includes both `funcvec` and `cargo-funcvec`, so it can be run directly or as
a Cargo subcommand.

## Usage

From inside a Rust project:

```sh
funcvec
cargo funcvec
```

Useful variants:

```sh
funcvec --provider none --top-k 10
funcvec path/to/project --provider lexical --threshold 0.72
funcvec report path/to/project --provider nomic
funcvec eval path/to/project --provider nomic --models nomic-v1,nomic-v1.5
```

The default provider is native Nomic embeddings. It downloads model files once
into a local cache and then runs without Ollama or another daemon.

Set `FUNCVEC_MODEL_CACHE_DIR` to control the native model cache location.
