# Task Plan: GravBonnorEbertSphere Test Problem

## Goal
Create a new test problem that initializes an exact Bonnor-Ebert sphere with typical star formation parameters. Validate stability at critical density and collapse with overdensity.

## Status: COMPLETE

## Phases

### Phase 1: Research — COMPLETE
- [x] Study existing gravity/self-gravity test problems for patterns
- [x] Study the Bonnor-Ebert sphere physics and Lane-Emden equation
- [x] Identify how self-gravity is enabled and configured
- [x] Review existing problem structure (CMakeLists, input files, etc.)

### Phase 2: Implementation — COMPLETE
- [x] Create `src/problems/GravBonnorEbertSphere/` directory
- [x] Implement `testGravBonnorEbertSphere.cpp` with Lane-Emden solver and IC setup
- [x] Create `CMakeLists.txt` for the problem
- [x] Create `inputs/GravBonnorEbertSphere.toml` input file
- [x] Add overdensity parameter for collapse test

### Phase 3: Validation — COMPLETE
- [x] Build the test
- [x] Run with critical density (stable case) — density changed by -1.7%
- [x] Run with overdensity (collapse case) — density increased by +4.2%
- [x] Verify results and document

### Phase 4: Documentation — COMPLETE
- [x] Create PR.md
- [x] Summarize changes

## Decisions
- Based on ParticleSink problem (stripped particles and MHD)
- Used gamma=1.001 (nearly isothermal) instead of true isothermal EOS
- Overdensity applied as density multiplier on equilibrium profile, keeping equilibrium pressure — this breaks hydrostatic balance so gravity wins
- Reflecting BCs with domain ~4x sphere radius
- Lane-Emden solved with RK4 (10000 points), profile interpolated onto 3D grid via GPU kernel
