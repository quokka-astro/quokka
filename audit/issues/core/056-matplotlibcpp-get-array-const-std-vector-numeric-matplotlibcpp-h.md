# matplotlibcpp::get_array(const std::vector<Numeric>&): in the NumPy path for unsupported element types (`NPY_NOTYPE`), it builds a local temporary `std::vector<double> vd` and returns `PyArray_SimpleNewFromData(..., vd.data())` (`src/util/matplotlibcpp.h:316-320`)

## Summary
in the NumPy path for unsupported element types (`NPY_NOTYPE`), it builds a local temporary `std::vector<double> vd` and returns `PyArray_SimpleNewFromData(..., vd.data())` (`src/util/matplotlibcpp.h:316-320`). The returned NumPy array then points to freed stack storage after the function returns (dangling pointer / use-after-free).

## Severity
`High`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::get_array(const std::vector<Numeric>&)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:808`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
