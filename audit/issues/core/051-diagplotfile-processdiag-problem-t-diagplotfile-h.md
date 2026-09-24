# DiagPlotfile::processDiag<problem_t>(...): when `QUOKKA_USE_OPENPMD` is enabled and `field_names` filtering is requested, the OpenPMD path writes the unfiltered `varnames` + `mf_cc_ptr` (`src/io/DiagPlotfile.H:127`) instead of `varnames_out` + `mf_cc_out_ptr` used by the AMReX plotfile path (`src/io/DiagPlotfile.H:136`)

## Summary
when `QUOKKA_USE_OPENPMD` is enabled and `field_names` filtering is requested, the OpenPMD path writes the unfiltered `varnames` + `mf_cc_ptr` (`src/io/DiagPlotfile.H:127`) instead of `varnames_out` + `mf_cc_out_ptr` used by the AMReX plotfile path (`src/io/DiagPlotfile.H:136`). `field_names` is silently ignored for OpenPMD output.

## Severity
`Low`

## Affected File
`src/io/DiagPlotfile.H`

## Affected Function / Symbol
`DiagPlotfile::processDiag<problem_t>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:684`
- Finding tags: conditional

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
