# Radiation coupling verification

`radiation_coupling/` contains 48 unique Rocq source modules: 36 gray/core
modules and 12 multigroup extensions. The shared files are byte-identical to
the previously checked sources. `source_sha256.json` pins their content.
No compiled objects, machine-specific logs, or historical copies are committed.

## Run

Install Rocq Platform 9.1 with Coquelicot and Flocq (the checked environment was
Rocq 9.1.0 / OCaml 4.14.2). Then, from the repository root:

```sh
python3 verification/radiation_coupling/check.py
```

Use `--coqc /path/to/coqc --coqchk /path/to/coqchk` when they are not on PATH.
The checker creates a fresh snapshot under `build/verification`, verifies the
source manifest, rejects unfinished/custom assumption declarations, compiles
in dependency order, and runs independent kernel checking. Both logs are saved.
The proofs use classical real analysis, classical choice, and the libraries'
standard foundations; absence of custom axioms does not mean axiom-free.

## Scope

- `BlackBox/EpsilonDriver.v`: gray nested driver acceptance and conditional
  termination, with explicit initialization and primitive arithmetic premises.
- `MultiGroup/AcceptedMultigroupAccuracy.v`: constant-opacity accepted-result
  component bounds for the specified positive rounded graph.
- `MultiGroup/AcceptedVariableAccuracy.v` and `VariableGroupOutputs.v`:
  variable-opacity margins and accepted-result bounds under whole-domain
  slope, sensitivity, value-error, and positivity premises.
- `MultiGroup/VariableOpacityCounterexample.v`: multiple physical roots from
  explicit band-value premises. Physical integral estimates remain analytic.

These are **not** a C++ source/compiled-binary refinement theorem, an accuracy
certificate for Quokka's Planck table, or a GPU runtime validation. The live
adapter's input formation and the full IMEX update are outside the theorem.
The simulation uses only double precision; Rocq, exact real arithmetic, and
high-precision Python references are offline verification tools.

Read [the explainer](../docs/markdown/radiation_solver_accuracy.md) for the
algorithm, symbols, flowchart, conditions, and finite error budgets. Build the
standalone C++ accuracy tests with `cmake -S tests/radiation_coupling -B
build/coupling-kernel`; run them with CTest. The Python tests require `mpmath`.
The `NestedRadCoupling` problem exercises the live single/multigroup entry points.
