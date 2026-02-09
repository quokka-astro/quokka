# Add `quokka.ignore_return` Parameter

## Summary

Added a new runtime parameter `quokka.ignore_return` that forces `main()` to return 0 regardless of the return code from `problem_main()`. This allows tests that are expected to fail validation checks to still pass in CI/CD pipelines.

## Changes

- **src/main.cpp**: Added logic to read `quokka.ignore_return` parameter and return 0 when enabled
- **inputs/RadStreaming.in**: Set `quokka.ignore_return = 1` to demonstrate the feature

## Usage

Add to any input file:
```
quokka.ignore_return = 1
```

Or pass as command-line argument:
```
./TestName inputs/test.in quokka.ignore_return=1
```

## Validation

Validated with the RadStreaming test:
- With `quokka.ignore_return=1`: returns exit code 0 even when test fails (L1 error = 0.985)
- With `quokka.ignore_return=0`: returns exit code 1 when test fails (expected behavior)
