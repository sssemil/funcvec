# funcvec-core

Core library for `funcvec`.

This crate contains Rust project discovery, rust-analyzer/LSP function
extraction, source normalization, embedding providers, duplicate scoring, and
report formatting. Most users should install the CLI crate instead:

```sh
cargo install funcvec
```

The core crate is published separately because the CLI crate depends on it.
When releasing, publish `funcvec-core` before `funcvec`.

Native Nomic embeddings are behind the `native-nomic` feature to keep the
default CLI install lightweight.
