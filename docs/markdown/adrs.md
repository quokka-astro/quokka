# Architecture Decision Records (ADRs)

!!! note "Purpose"
    ADRs document important technical decisions so future contributors understand the problem we solved, the options we evaluated, and how to revisit the choice if assumptions change.

## When to Author an ADR

Write an ADR whenever a change has broad impact, is hard to reverse, or triggers sustained debate.

!!! warning "Common triggers"
    - Adjusting `ParmParse` options beyond trivial additions (renames, removals, behaviour changes).
    - Any risk to the output file format (typically rejected; document how compatibility is preserved if discussed).
    - Modifying the problem-file interface (new hooks, migration plans, deprecations).
    - Shifts in accuracy/performance trade-offs (limiters, fluxes, numerical precision), or non-trivial dependency/toolchain updates.

---

## ADR Paths

### PR-Initiated ADR (resolving review disagreements)

1. Distil blocking PR comments into 3–5 explicit questions.
2. Create `docs/adrs/ADR-xxxx-title.md` and link it from the PR.
3. Assign 2–3 maintainers (include numerics + GPU expertise) as reviewers.
4. Provide evidence: options/trade-offs, CPU + CUDA/HIP benchmarks, accuracy/tolerance notes, portability, migration impact.
5. Time-box discussion to **≤10 business days**; schedule an optional 30–45 minute design call if needed.
6. Decide and tag the ADR: **Accepted / Rejected / Superseded / Amended**; reference PRs/issues.
7. Apply the outcome: update the PR, CI, documentation, and deprecation/migration notes.

### Proactive ADR (not tied to a PR)

1. Open an issue labeled `type:ADR-proposal` outlining Problem, Goals/Non-goals, Affected modules, and Risks.
2. Draft the ADR in `docs/adrs/ADR-xxxx-title.md`; include a short rollout roadmap (milestones, owners).
3. Set roles: **Driver** (author), **Reviewers** (numerics + GPU/AMReX), **Consulted** (downstream users if affected).
4. Add prototypes or microbenchmarks when helpful (CPU + CUDA/HIP).
5. Leave a **≤21 day** comment window (to accommodate academic schedules) or close earlier with explicit sign-off.
6. Decide and define the rollout: update CI coverage, add tests, document migrations (especially for problem interfaces or `ParmParse` adjustments).
7. Maintain the ADR lifecycle: revisit annually or when assumptions change, preferring supersession over silent drift.

---

## ADR Template

```markdown
# ADR-XXXX: <Short Title>
Date: <YYYY-MM-DD> • Status: Proposed | Accepted | Rejected | Superseded | Amended

## Context
Problem, constraints (MPI, CUDA/HIP), affected modules and users.
Touches: ParmParse options? Output format? Problem-file interface?

## Options
- Option A: …
- Option B: …
(Include small benchmarks: CPU, CUDA, HIP; accuracy/tolerance notes.)

## Decision
Chosen option and rationale.

## Consequences
Performance/accuracy impact, portability (CUDA/HIP), maintenance cost,
migration plan (deprecations, updates for problem-file interfaces; NO output-format changes).

## Rollout & Testing
CI matrix updates (CUDA/HIP/CPU), new tests, deprecations, docs updates.

## Links
PRs, issues, design notes, prior ADRs.
```

---

## Ending PR Deadlocks with ADRs

Move prolonged or subjective debates into an ADR. After a decision:

* Mark PR threads **resolved** or **superseded by ADR**.
* Merge only after the ADR is **Accepted** and CI/tests reflect the choice.
