# problem_main(): comment says the output should include `tNew_[0]` in the filename, but `matplotlibcpp::save(std::format("./radhydro_pulse_grey_temperature.pdf", sim2.tNew_[0]))` has no `{}` placeholder (`src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp:372-373`)

## Summary
comment says the output should include `tNew_[0]` in the filename, but `matplotlibcpp::save(std::format("./radhydro_pulse_grey_temperature.pdf", sim2.tNew_[0]))` has no `{}` placeholder (`src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp:372-373`). The time argument is ignored and the filename is constant.

## Severity
`High`

## Affected File
`src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1756`
- Finding tags: diagnostics/output correctness

## Proposed Patch
- Add a `{}` placeholder (or equivalent formatted token) to the filename format string so the timestep value is actually embedded in the saved filename.
