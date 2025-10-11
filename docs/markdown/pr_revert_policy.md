# Pull Request Revert Policy

!!! warning "Purpose"
    Reverts protect users, safeguard maintainability, and preserve scientific integrity when a merged change causes regressions that cannot wait for a fix-forward. They are not punitive—they provide a fast, low-risk way to restore a reliable baseline so contributors can investigate and reintroduce the change safely.

---

## When to Revert

Trigger a revert only for serious issues:

!!! danger "Typical revert triggers"
    - **Major correctness bugs** (e.g., broken conservation, highly unstable solver, incorrect physics equations).
    - **Significant performance regressions** on maintained benchmarks (typically >10–15% slower without an acceptable trade-off).
    - **Broken builds or CI** on supported targets (CPU, CUDA, HIP) that cannot be repaired quickly.
    - **Incompatible output-format changes** that break existing plotfile readers.
    - **Unreproducible outputs or race conditions** exposed by deterministic reruns or sanitizer tooling.
    - **Security or licensing concerns** (e.g., unsafe dependency, incompatible license).

---

## Revert Workflow

1. **Open an issue** summarising impact, affected versions, and how to reproduce. Label it `regression`, `revert-candidate`, and add `backport-needed` if release branches are affected.
2. **Prepare the revert PR**:
   * Title it `Revert: #<original-PR> <original title>`.
   * Use `git revert <merge-commit-sha>` when possible to keep history clean.
   * Keep the change **minimal**—no refactors or unrelated fixes.
   * Add a **failing test** that reproduces the problem (fails on `development`, passes after the revert).
3. **Explain clearly** in the PR body: user impact, scope, why revert beats a quick fix-forward, and the plan to reintroduce safely if needed.
4. **Review & merge quickly**—aim for ≤3 business days once consensus forms. One maintainer approval is enough for low-risk internal fixes; require **two approvals** when the original change was high risk or broadly user-facing.

---

## After the Revert

!!! note "Post-merge checklist"
    - File a follow-up issue to **fix forward**: root cause, new tests, ADR if architecture changes are involved.
    - Note the revert in the **changelog** and backport to supported release branches if users rely on the affected feature.
    - For `ParmParse` or problem-file interface changes, publish a **migration guide** before attempting re-introduction.

---

## Guiding Principles

!!! tip "Guiding principles"
    - Prefer a **temporary revert plus a clear follow-up plan** over leaving the code broken.
    - Always attach **evidence** (inputs, logs, benchmarks) to justify the revert.
    - Keep the revert PR **surgical** and traceable to the original change.
    - Treat reverts as a collaborative safety measure; focus on the fix, not assigning blame.
