# Radiation solver: detailed C++ control flow

These diagrams expand the [algorithm overview and error analysis](radiation_solver_accuracy.md). They follow `mgsolve::solve`, `evaluate`, `inner`, and `safeguard` in `src/radiation/nested/multigroup_solver.hpp`. Each rectangle names an operation; each diamond labels a test. Terminal nodes name the C++ status or stopping reason. Calls to **Evaluate** and **Inner** refer to the diagrams below. A failed call returns its status immediately to its caller.

The physical notation remains that of the explainer: dust temperature is \\(T_{\rm d}\\), gas temperature is \\(T\\), and the adjusted input gas temperature is \\(T^{(0)}\\). Diagram labels use the exact C++ names where this makes a condition easier to locate: `x = T_d`, `t = T`, `p.T = T^(0)`, `p.A = C_v`, `p.D = D`, `p.h = H`, and `p.chi = C` (the last three are the explainer's calligraphic symbols). `M` and `H` inside an evaluation are the separate emission and absorption sums; they are not `p.h`.

The large diagrams retain full-size labels. Scroll within a diagram to follow a path; the numbered panels connect through named calls.

<style>
.solver-flow { overflow: auto; max-height: 85vh; border: 1px solid var(--table-border-color); padding: 0.5rem; margin-block: 1rem; }
.solver-flow svg { min-width: var(--flow-width); max-width: none !important; }
</style>

## Exact tests used in the diagrams

| Test | C++ condition and meaning |
| --- | --- |
| `normal(y)` | `min_normal <= y <= max_double`: positive, finite, normal binary64. |
| `nonnegative(y)` | `y == 0` or `normal(y)`. Negative, subnormal, infinite, and NaN values fail. |
| `guard(R,k)` | Returns −1 for `R < 1-k*u`, +1 for `R > 1+k*u`, and 0 otherwise. The endpoints belong to the acceptance window; `u = 2^-53`. This test is applied after the evaluator's range checks. |
| `narrow(lo,hi,k)` | Both endpoints must be positive normal and ordered. Equal endpoints pass. Otherwise the computed gap must be positive finite and the computed `gap/lo` must be normal and at most `k*u`. |
| `adjacent(lo,hi)` | The positive binary64 bit ranks differ by at most one. There is no represented interior value. |
| `meets(certificate,tol)` | Dust, gas, and **every** radiation-group relative allowance must be at most `tol`. Exact-zero groups carry an absolute-zero sentinel. |
| **Accept** | `accepted_conditional` when `contract.certified` is true; otherwise `accepted_estimated`. This describes the caller's contract, not a runtime proof that the callback satisfies it. |
| **Range failure** | A checked graph operation produces a non-normal value, except an explicitly permitted exact zero. A zero produced by underflow is rejected. Derivatives are proposal-only and do not pass through this graph check. |

## 1. Entry checks and the constant zero-emission shortcut

<div class="solver-flow" style="--flow-width: 1602px">

```mermaid
flowchart TD
    A["solve: begin"] --> B{"Fast-math enabled, or host rounding not nearest?"}
    B -->|Yes| UM["unsupported_model"]
    B -->|No| C{"Inputs and options valid? See checklist below"}
    C -->|No| IV["invalid_input"]
    C -->|Yes| D{"Contract checks pass? See checklist below"}
    D -->|No| MC["missing_contract"]
    D -->|Yes| E["x = clamp input gas temperature to domain"]
    E --> F["Evaluate x; groups_only = zero-emission declaration or N == 0"]
    F -->|Failure| ER["Return evaluation status"]
    F -->|Success| G{"Zero-emission declaration or N == 0?"}
    G -->|No| H{"M == 0?"}
    H -->|Yes| MC
    H -->|No| OUT["Enter outer loop: diagram 2"]
    G -->|Yes| I{"Variable opacity or M != 0?"}
    I -->|Yes| UM
    I -->|No| J{"H > 0?"}
    J -->|Yes| K["t = p.T + H/p.A; x = t + H/(p.D*sqrt t)"]
    J -->|No| L["t = p.T; x = t"]
    K --> M{"Checked arithmetic valid?"}
    L --> M
    M -->|No| RF["range_failure"]
    M -->|Yes| N{"x inside allowed domain?"}
    N -->|No| NB["no_bracket"]
    N -->|Yes| O["Evaluate x again with groups_only = true"]
    O -->|Failure| ER
    O -->|Success| P["Store temperatures, radiation, gas energy = p.A*t"]
    P --> Q{"Gas energy arithmetic valid?"}
    Q -->|No| RF
    Q -->|Yes| R["Build direct certificate; stop = direct_absorption if H > 0, otherwise zero_exchange"]
    R --> S{"All component allowances meet tolerance?"}
    S -->|Yes| OK["Accept"]
    S -->|No| TU["tolerance_unavailable"]
```

</div>

**Input checks:** `p.A`, `p.D`, `p.T`, `p.h`, `p.chi`, `x_min`, and `x_max` must be positive normal; `x_min <= x_max`; tolerance must be positive finite; each of the three iteration limits must be in `[1,65536]`; each input group energy must be zero or positive normal. Compile-time checks require IEEE binary64 and at most 1024 groups. Device execution assumes nearest rounding; only the host path queries the rounding mode.

**Contract checks, in execution order:** either `certified` or `allow_estimated` must be true; the margin must be positive normal and at most one (exactly one for constant opacity); a certified contract must declare a root in the domain and coefficient log-error budgets in `[0,0]` for constant opacity or `[0,8u]` for variable opacity, plus a band budget in `[0,8u]`; every sensitivity must be nonnegative finite. Estimated mode skips the root-declaration and callback-budget checks, but still checks margin and sensitivities.

The direct path uses dust-coordinate log allowance `27.5*lambda_upper` and gas allowance `18*lambda_upper` when `H > 0`; for zero exchange these become zero and `lambda_upper`. It still constructs group allowances and calls `meets`. This shortcut does not depend on `allow_residual` or `allow_width`, and cannot be selected from a sampled zero emission alone.

## 2. Outer dust-temperature search and stopping

<div class="solver-flow" style="--flow-width: 2373px">

```mermaid
flowchart TD
    A["Start with initial evaluation; lo = hi = 0; bracket = false; save first_sign"] --> L{"Loop fuel remains? k below max_bracket + max_outer"}
    L -->|No| IL["iteration_limit"]
    L -->|Yes| B["Store current state and ratio"]
    B --> C{"e.sign == 0?"}
    C -->|Yes| D["Residual coordinate allowance: 207*lambda_upper/margin constant, 223*lambda_upper/margin variable"]
    D --> E{"allow_residual and all component allowances meet tolerance?"}
    E -->|Yes| AR["Accept; stop = residual"]
    E -->|No| F{"True bracket and allow_width and narrow lo,hi,16?"}
    F -->|No| TU["tolerance_unavailable; do not invent a sign"]
    F -->|Yes| W["Build width certificate: coordinate allowance 32*lambda_upper"]
    C -->|No| G{"True bracket and allow_width and narrow lo,hi,16?"}
    G -->|Yes| W
    W --> X{"All component allowances meet tolerance?"}
    X -->|Yes| AW["Accept; stop = width"]
    X -->|No| TU
    G -->|No| U{"e.sign negative?"}
    U -->|Yes| UL["lo = x"]
    U -->|No: positive| UH["hi = x"]
    UL --> V{"Both endpoints positive?"}
    UH --> V
    V -->|Yes| BB["Set bracket = true; increment outer_iterations"]
    BB --> O{"outer_iterations exceeds max_outer?"}
    O -->|Yes| IL
    O -->|No| P{"Endpoints adjacent?"}
    P -->|Yes| PL["precision_limit"]
    P -->|No| S["Newton proposal or rank midpoint: diagram 5"]
    V -->|No| K{"k at least max_bracket?"}
    K -->|Yes| IL
    K -->|No| DIR{"first_sign negative?"}
    DIR -->|Yes| UP["Double x, capped at x_max; avoid overflow"]
    DIR -->|No| DN["Halve x, bounded below by x_min"]
    UP --> SAME{"next == x?"}
    DN --> SAME
    SAME -->|Yes| NB["no_bracket: domain endpoint reached"]
    SAME -->|No| SET["x = next"]
    S --> EV["Evaluate new x: diagram 3"]
    SET --> EV
    EV -->|Failure| ER["Return evaluation status"]
    EV -->|Success| M{"M == 0?"}
    M -->|Yes| MC["missing_contract"]
    M -->|No| INC["Increment k"]
    INC --> L
```

</div>

The ordering matters. Acceptance uses the current bracket **before** the current reliable sign updates an endpoint. An ambiguous ratio can only use residual acceptance or an already adequate width certificate. If neither is available, it returns `tolerance_unavailable` immediately. Width acceptance requires the component budgets, not just a small temperature interval.

## 3. One trial evaluation

<div class="solver-flow" style="--flow-width: 1280px">

```mermaid
flowchart TD
    A["evaluate at x"] --> B{"Opacity callback returns true?"}
    B -->|No| OF["oracle_failure"]
    B -->|Yes| C{"Each alpha, p, B is zero or positive normal, and zero value exactly matches its structural-zero flag?"}
    C -->|No| OF
    C -->|Yes| D["Per group: compute positive emission, absorption, and reconstructed radiation"]
    D --> DD{"Callback supplies derivatives?"}
    DD -->|Yes| DE["Accumulate emission and absorption derivatives for proposals only"]
    DD -->|No| SUM["Reduce each positive sum pairwise; carry any unpaired leaf"]
    DE --> SUM
    SUM --> E{"Checked arithmetic valid?"}
    E -->|No| RF["range_failure"]
    E -->|Yes| F{"groups_only?"}
    F -->|Yes| GO["Return successful group evaluation; no inner solve"]
    F -->|No| I["Inner gas solve: diagram 4"]
    I -->|Failure| ER["Return inner status"]
    I -->|Success| J["Gas energy = p.A*t"]
    J --> BR{"x at least p.T?"}
    BR -->|Yes: heating or equality| HE["numerator = q + M; denominator = H"]
    BR -->|No: cooling| CO["numerator = M; denominator = q + H"]
    HE --> Z{"Denominator zero?"}
    CO --> Z
    Z -->|No| DIV["ratio = numerator / denominator"]
    Z -->|Yes| NZ{"Numerator zero?"}
    NZ -->|Yes| ZERO["ratio = 1; sign = 0"]
    NZ -->|No| INF["ratio = infinity; sign = +1; no division"]
    DIV --> R{"Checked arithmetic valid?"}
    ZERO --> R
    INF --> R
    R -->|No| RF
    R -->|Yes| FIN{"ratio != infinity?"}
    FIN -->|Yes| GUARD["sign = guard ratio,128"]
    FIN -->|No| DER
    GUARD --> DER{"Derivatives supplied and branch denominator positive?"}
    DER -->|Yes| NEW["Compute ratio derivative for a Newton proposal"]
    DER -->|No| OK["Return successful evaluation"]
    NEW --> OK
```

</div>

The coefficient validation repeats for every group. A structural-zero flag asserts zero throughout the domain; the evaluator only checks consistency at the current sample. Derivative flags affect speed, not acceptance. Bad derivatives can produce an unusable proposal, which the safeguard discards.

## 4. Inner gas solve: branch selection, initialization, and iteration

<div class="solver-flow" style="--flow-width: 2086px">

```mermaid
flowchart TD
    A["inner: fixed trial x"] --> EQ{"x == p.T?"}
    EQ -->|Yes| EXACT["Return t = p.T, q = 0"]
    EQ -->|No| COOL{"x below p.T?"}
    COOL -->|No| HEAT["Select heating chart"]
    COOL -->|Yes| HALF["half = p.T/2; evaluate strong chart at half"]
    HALF --> HG{"half normal and evaluation valid?"}
    HG -->|No| RF["range_failure"]
    HG -->|Yes| HS{"guard K,16 at half"}
    HS -->|Zero| ACCEPT["Return successful inner t,q"]
    HS -->|Negative| STRONG["Select strong cooling; lo = x; hi = half"]
    HS -->|Positive| WEAK["Select weak cooling"]
    HEAT --> SEED["hi = abs(x-p.T) * (1+32u); cap at half for weak cooling"]
    WEAK --> SEED
    SEED --> VALID{"Arithmetic valid and hi normal?"}
    VALID -->|No| RF
    VALID -->|Yes| TOP["Evaluate chosen chart at hi"]
    TOP -->|Invalid| RF
    TOP -->|Valid| TS{"guard K,16"}
    TS -->|Zero| ACCEPT
    TS -->|Negative| NB["no_bracket"]
    TS -->|Positive| INIT["lo = hi; begin lower-endpoint search"]
    INIT --> FUEL{"Fewer than max_bracket halvings?"}
    FUEL -->|No| IL["iteration_limit"]
    FUEL -->|Yes| LOW["Halve lo; require normal lo; evaluate chart"]
    LOW -->|Invalid| RF
    LOW -->|Valid| LS{"guard K,16"}
    LS -->|Zero| ACCEPT
    LS -->|Positive| MOVE["hi = lo; count halving"]
    MOVE --> FUEL
    LS -->|Negative| START["z = rank midpoint lo,hi"]
    STRONG --> START
    START --> IFUEL{"Fewer than max_inner iterations?"}
    IFUEL -->|No| IL
    IFUEL -->|Yes| EV["Evaluate selected chart at z"]
    EV -->|Invalid| RF
    EV -->|Valid| STOP{"guard K,16 == 0 OR narrow lo,hi,4?"}
    STOP -->|Yes| ACCEPT
    STOP -->|No| ADJ{"Endpoints adjacent?"}
    ADJ -->|Yes| PL["precision_limit"]
    ADJ -->|No| SIGN{"Negative sign XOR strong chart?"}
    SIGN -->|Yes| UPLO["lo = z"]
    SIGN -->|No| UPHI["hi = z"]
    UPLO --> NEXT["Newton proposal or rank midpoint: diagram 5; count iteration"]
    UPHI --> NEXT
    NEXT --> IFUEL
```

</div>

The chart balances are defined in the explainer. Heating and weak cooling solve for the temperature increment or decrement; strong cooling solves for gas temperature itself. The strong balance decreases, so its endpoint-update signs are reversed. The C++ implementation tests the strong chart at `p.T/2` for **every** cooling trial, even when an analytic branch test could avoid that evaluation. The direct equality return and initialization-window returns are inner successes, not final outer acceptance. `max_inner` limits the iteration loop; evaluations during chart selection and lower-endpoint search also contribute to the reported inner count.

## 5. Shared Newton safeguard

<div class="solver-flow" style="--flow-width: 748px">

```mermaid
flowchart TD
    A["Propose y - (ratio-1)/ratio_derivative"] --> B{"use_newton?"}
    B -->|No| M["Return rank midpoint"]
    B -->|Yes| C{"Proposal positive finite normal?"}
    C -->|No| M
    C -->|Yes| D{"Strictly inside bracket, with each rank distance at least floor span/4?"}
    D -->|No| M
    D -->|Yes| N["Return Newton proposal"]
```

</div>

This is a midpoint of **binary64 ranks**, not the arithmetic mean of the temperatures. The safeguard is also used by the inner loop. A zero, infinite, or NaN derivative need not abort the solve: its proposal is rejected if it fails these tests. The caller handles exhausted precision and iteration budgets.

## Live-update failure handling

`SolveNestedRadiationCoupling` maps the simulation state into this kernel. Compile-time assertions restrict the enabled path to its supported ideal-gas, thermal-dust model. The multigroup caller also asserts that external dust heating is zero. After the kernel returns, the adapter requires an accepted status, gas temperature at or above its floor, and each group energy at or above its radiation floor. A failed check triggers an AMReX assertion; it does not project the solution onto a floor or silently call the old solver. Successful results proceed through the existing flux update and state storage.
