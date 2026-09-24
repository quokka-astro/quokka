# PhysicsParticleRegister::readSFH(...): returns `last_time` by overwriting with each parsed entry (`src/particles/PhysicsParticles.hpp:1136-1175`, assignment at `:1165`) rather than tracking the maximum time across particle types/histories

## Summary
returns `last_time` by overwriting with each parsed entry (`src/particles/PhysicsParticles.hpp:1136-1175`, assignment at `:1165`) rather than tracking the maximum time across particle types/histories. With multiple formation particle types, the returned restart time can depend on iteration order.

## Severity
`Medium`

## Affected File
`src/particles/PhysicsParticles.hpp`

## Affected Function / Symbol
`PhysicsParticleRegister::readSFH(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:524`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
