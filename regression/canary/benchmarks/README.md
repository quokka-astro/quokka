# Canary Benchmarks

This directory is the root for Canary-managed nightly regression benchmarks.

Each test owns a subdirectory:

```text
benchmarks/
└── <test-name>/
    └── <plotfile-or-other-gold-data>
```

The `Sedov-GPU` prototype writes its gold plotfile under:

```text
benchmarks/Sedov-GPU/
```

These files are expected to be created or updated via:

```bash
cd TestResults
canary rebaseline -k sedov_gpu .
```
