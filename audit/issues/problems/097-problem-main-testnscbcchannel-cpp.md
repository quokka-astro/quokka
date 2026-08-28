# problem_main(): pressure diagnostic reconstructs `Eint` as `Egas - xmom^2/(2 rho)` (`src/problems/NscbcChannel/testNscbcChannel.cpp:172-180`), omitting `y/z` kinetic energy even though `v_inflow` and `w_inflow` are configurable

## Summary
pressure diagnostic reconstructs `Eint` as `Egas - xmom^2/(2 rho)` (`src/problems/NscbcChannel/testNscbcChannel.cpp:172-180`), omitting `y/z` kinetic energy even though `v_inflow` and `w_inflow` are configurable. This biases diagnostic pressure/error checks when transverse inflow velocity is nonzero.

## Severity
`High`

## Affected File
`src/problems/NscbcChannel/testNscbcChannel.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1295`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
