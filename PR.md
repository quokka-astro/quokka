# Add PRIORITY parameter to quokka_add_problem

## Summary

Added an optional `PRIORITY` parameter to the `quokka_add_problem()` CMake function to control test execution order. Tests with higher priority values run first, with default tests having the lowest priority (0).

## Changes

- **ProblemHelpers.cmake**: Added `PRIORITY` parameter that maps to CMake's `COST` property
  - Default priority: 0 (lowest)
  - Higher values run first
  - Tests with the same priority run in alphabetical order

- **ParticleSF/CMakeLists.txt**: Set priority 100 for ParticleSF, 99 for ParticleSF2
- **RadhydroBB/CMakeLists.txt**: Set priority 50 using `quokka_add_problem(... PRIORITY 50)`

## Usage

```cmake
# High priority test (runs first)
quokka_add_problem(JOB_NAME MyImportantTest PRIORITY 100)

# Medium priority test
quokka_add_problem(JOB_NAME MyMediumTest PRIORITY 50)

# Default priority test (runs last, in alphabetical order)
quokka_add_problem(JOB_NAME MyStandardTest)
```

## Validation

Tested with `ctest` showing correct execution order:
1. ParticleSF (priority 100)
2. ParticleSF2 (priority 99)
3. RadhydroBB (priority 50)
4. Other tests in alphabetical order (priority 0)
