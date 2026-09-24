# QuokkaSimulation<ExactProblem>::setInitialConditionsOnGrid(...): this `ExactProblem` specialization writes state using `RadSystem<MGProblem>::...` indices throughout (`src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp:286-295`) instead of `RadSystem<ExactProblem>::...`

## Summary
this `ExactProblem` specialization writes state using `RadSystem<MGProblem>::...` indices throughout (`src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp:286-295`) instead of `RadSystem<ExactProblem>::...`. Because `MGProblem` and `ExactProblem` have different radiation-group layouts, this can write wrong components / go out of bounds.

## Severity
`High`

## Affected File
`src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ExactProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1779`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
