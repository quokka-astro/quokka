# read_dataset(...): opens an HDF5 dataspace with `H5Dget_space(...)` (`src/turbulence/TurbDataReader.cpp:26`) but never calls `H5Sclose(dspace)`, leaking an HDF5 handle per dataset read

## Summary
opens an HDF5 dataspace with `H5Dget_space(...)` (`src/turbulence/TurbDataReader.cpp:26`) but never calls `H5Sclose(dspace)`, leaking an HDF5 handle per dataset read.

## Severity
`Medium`

## Affected File
`src/turbulence/TurbDataReader.cpp`

## Affected Function / Symbol
`read_dataset(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:532`
- Finding tags: resource leak

## Proposed Patch
- Close the dataspace handle with `H5Sclose(dspace)` on all paths (preferably via a small RAII wrapper for HDF5 IDs).
