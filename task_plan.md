# Task Plan: GravBonnorEbertSphere Test Problem

## Goal
Create a new test problem that initializes an exact Bonnor-Ebert sphere with typical star formation parameters. Validate stability at critical density and collapse with overdensity.

## Status: PLANNING

## Phases

### Phase 1: Research — ACTIVE
- [ ] Study existing gravity/self-gravity test problems for patterns
- [ ] Study the Bonnor-Ebert sphere physics and Lane-Emden equation
- [ ] Identify how self-gravity is enabled and configured
- [ ] Review existing problem structure (CMakeLists, input files, etc.)

### Phase 2: Implementation — PENDING
- [ ] Create `src/problems/GravBonnorEbertSphere/` directory
- [ ] Implement `testGravBonnorEbertSphere.cpp` with Lane-Emden solver and IC setup
- [ ] Create `CMakeLists.txt` for the problem
- [ ] Create `inputs/GravBonnorEbertSphere.in` input file
- [ ] Add overdensity parameter for collapse test

### Phase 3: Validation — PENDING
- [ ] Build the test
- [ ] Run with critical density (stable case)
- [ ] Run with overdensity (collapse case)
- [ ] Verify results and document

### Phase 4: Documentation — PENDING
- [ ] Create PR.md
- [ ] Summarize changes

## Decisions
- (none yet)
