# Canary Prototype For Nightly Regressions

This directory is a prototype replacement for the legacy nightly AMReX regression harness in `extern/regression_testing`.

## Layout

```text
regression/canary/
├── README.md
├── benchmarks/
│   └── README.md
├── lib/
│   └── quokka_canary.py
└── tests/
    ├── LinearAlfvenWave-GPU.pyt
    ├── MHDBlast-GPU.pyt
    ├── RadhydroShell-GPU.pyt
    ├── RandomBlastAMR-GPU.pyt
    ├── Sedov-GPU.pyt
    ├── SedovAMR-GPU.pyt
    ├── ShockCloud-GPU.pyt
    └── Turbulence-GPU.pyt
```

## Conventions

- One `.pyt` file per nightly regression case.
- Benchmarks live under `regression/canary/benchmarks/<test-name>/`.
- Each test stages any required runtime files into Canary's execution directory.
- AMReX mesh comparisons are performed explicitly with `fcompare`.
- Particle comparisons are performed explicitly with `particle_compare`.
- Plotfile directory rebaselining is handled by a custom test flag because Canary's built-in `baseline(src, dst)` helper only copies files, not directories.

## Current Suite

The full nightly CUDA suite from [quokka-tests.ini](/Users/benwibking/amrex_codes/quokka/regression/quokka-tests.ini) is represented here:

- `Sedov-GPU`
- `SedovAMR-GPU`
- `ShockCloud-GPU`
- `RandomBlastAMR-GPU`
- `RadhydroShell-GPU`
- `MHDBlast-GPU`
- `LinearAlfvenWave-GPU`
- `Turbulence-GPU`

## Expected Environment

The prototype assumes:

- Quokka is already configured and built.
- `canary-wm` is installed in the active Python environment.
- GNU `make` is available to build the AMReX comparison tools used by the suite.
- The default Quokka build tree is `<repo>/build`.

Optional environment overrides:

- `QUOKKA_CANARY_BUILD_DIR`: absolute path to the build tree
- `QUOKKA_CANARY_MPIEXEC`: MPI launcher, default `mpirun`
- `QUOKKA_CANARY_MPIEXEC_NUMPROC_FLAG`: processor-count flag, default `-n`
- `QUOKKA_CANARY_FCOMPARE`: explicit path to `fcompare`
- `QUOKKA_CANARY_PARTICLE_COMPARE`: explicit path to `particle_compare`
- `QUOKKA_CANARY_FSNAPSHOT`: explicit path to `fsnapshot`
- `QUOKKA_CANARY_PALETTE`: optional path to an `fsnapshot` palette file; if unset, Canary uses `fsnapshot`'s default palette

## Runner

Use the host-side runner script to build the required targets, build the AMReX comparison tools, wait for an idle NVIDIA GPU, and execute the full suite in a dedicated work tree:

```bash
scripts/bash/run-canary-regression-tests.sh
```

Useful overrides:

```bash
scripts/bash/run-canary-regression-tests.sh \
  --work-dir /scratch/$USER/quokka-canary-nightly \
  --build-dir build \
  --workers 1 \
  --gpu-count 1
```

The script writes:

- `canary-run.log`
- `canary-status.json`
- `TestResults/canary-report.html`
- `TestResults/canary.json`
- `TestResults/junit.xml`
- `TestResults/canary-status.txt`

## Manual Usage

For ad hoc runs without the wrapper script, create a small Canary config file that defines the GPU resource pool and `CUDA_VISIBLE_DEVICES`, then run Canary from a disposable work tree:

```bash
cat > /tmp/quokka-canary.yaml <<'EOF'
workspace:
  view: TestResults
environment:
  set:
    CUDA_VISIBLE_DEVICES: "%(gpu_ids)s"
resource_pool:
  gpus: 1
EOF

mkdir -p /tmp/quokka-canary-work
cd /tmp/quokka-canary-work
QUOKKA_CANARY_BUILD_DIR=/path/to/quokka/build \
canary -f /tmp/quokka-canary.yaml run -w --workers=1 /path/to/quokka/regression/canary/tests
```

If the benchmark is missing or the output differs, Canary should mark the test as `DIFFED`. Rebaseline from the latest results view:

```bash
cd TestResults
canary rebaseline -k sedov_gpu .
```

## Next Steps

- Decide whether nightly reporting should stay as Canary HTML/JSON/JUnit, move to CDash, or keep a custom GitHub Pages history view.
- Decide whether benchmark plotfiles should live directly in the repository or in a separate benchmark checkout that the nightly host manages.
