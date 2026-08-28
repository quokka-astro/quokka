# problem_main(): `AMREX_ASSERT_WITH_MESSAGE(boost_vel_x != NAN, ...)` (`src/problems/ParticleSink/testParticleSink.cpp:167-169`) is ineffective because comparisons with `NaN` are always true

## Summary
`AMREX_ASSERT_WITH_MESSAGE(boost_vel_x != NAN, ...)` (`src/problems/ParticleSink/testParticleSink.cpp:167-169`) is ineffective because comparisons with `NaN` are always true. Missing `boost_vel_x` input is not reliably detected.

## Severity
`High`

## Affected File
`src/problems/ParticleSink/testParticleSink.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1460`
- Finding tags: correctness

## Proposed Patch
- Replace NaN comparisons with `amrex::Math::isfinite(...)` / `std::isfinite(...)` checks (or `x == x` if required on device) and assert on non-finite values.
