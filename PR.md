# Add `ignore_return` Parameter

## Summary

Added a new runtime parameter `ignore_return` that forces `main()` to return 0 regardless of the return code from `problem_main()`. This allows tests that are expected to fail validation checks to still pass in CI/CD pipelines.

## Changes

- **src/main.cpp**: Added logic to read `ignore_return` parameter and return 0 when enabled
- **inputs/RadStreaming.in**: Set `ignore_return = 1` to demonstrate the feature

## Usage

Add to any input file:
```
ignore_return = 1
```

Or pass as command-line argument:
```
./TestName inputs/test.in ignore_return=1
```
