# read_dataset(...): reads `ndims` but unconditionally indexes `dims[0..2]` when constructing the 3D table (`src/turbulence/TurbDataReader.cpp:27-30`, `:49`)

## Summary
reads `ndims` but unconditionally indexes `dims[0..2]` when constructing the 3D table (`src/turbulence/TurbDataReader.cpp:27-30`, `:49`). Malformed/non-3D datasets can trigger out-of-bounds access.

## Severity
`High`

## Affected File
`src/turbulence/TurbDataReader.cpp`

## Affected Function / Symbol
`read_dataset(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:533`
- Finding tags: robustness

## Proposed Patch
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.

## Why This Is a Bug
The function records `ndims` from HDF5 but immediately indexes `dims[0]`, `dims[1]`, and `dims[2]` unconditionally. If the file contains a malformed dataset or a non-3D dataset, `dims` can be shorter than 3 entries and the code reads out of bounds while constructing the `Table3D` shape.

## Complete Code Patch
```diff
diff --git a/src/turbulence/TurbDataReader.cpp b/src/turbulence/TurbDataReader.cpp
--- a/src/turbulence/TurbDataReader.cpp
+++ b/src/turbulence/TurbDataReader.cpp
@@
 	hid_t const dspace = H5Dget_space(dset_id);
 	const int ndims = H5Sget_simple_extent_ndims(dspace);
+	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ndims == 3, "TurbDataReader expects a 3D HDF5 dataset");
 	std::vector<hsize_t> dims(ndims);
 	H5Sget_simple_extent_dims(dspace, dims.data(), nullptr);
```
