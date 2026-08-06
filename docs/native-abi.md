# Native CPU ABI

ABI identifier: `learn.cpu.csv.v1`.

Each generated library exports:

```c
const char *learn_kernel(const char *source0, const char *source1);
```

Inputs and output are comma-separated IEEE-754 decimal values. Shapes and element counts live in the native artifact. This string boundary works around Arturo 0.10’s scalar-only FFI while keeping one foreign call per fused tensor group.

Compiled artifacts are immutable schedule snapshots. The reference backend remains authoritative for fallback and parity testing.
