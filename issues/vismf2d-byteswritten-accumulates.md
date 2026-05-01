# 2D VisMF writers accumulate `bytesWritten` across file groups

Severity: High

## Explanation

Both custom 2D VisMF writers declare `bytesWritten` once before the `amrex::NFilesIter` loop:

```cpp
amrex::Long bytesWritten(0);
...
for (; nfi.ReadyToWrite(); ++nfi) {
	...
	bytesWritten += ...
	...
	allFabData = std::make_unique<std::vector<char>>(bytesWritten);
	...
	nfi.Stream().write(allFabData->data(), bytesWritten);
}
```

The same pattern exists in:

- `src/io/projection.cpp::VisMF2D`
- `src/io/DiagFramePlane.cpp::DiagFramePlane::VisMF2D`

`NFilesIter` can require a rank to write more than one file group. In that case the second iteration starts with the previous iteration's byte count, allocates a larger buffer than the current FAB payload, and writes `bytesWritten` bytes even though only the current `writePosition` bytes were populated. That can produce corrupt MultiFab data files with stale or zero-filled trailing bytes.

## Patch

```diff
diff --git a/src/io/projection.cpp b/src/io/projection.cpp
--- a/src/io/projection.cpp
+++ b/src/io/projection.cpp
@@
-	amrex::Long bytesWritten(0);
-
 	std::string const filePrefix(a_mf_name + "_D_");
@@
 	}
 	for (; nfi.ReadyToWrite(); ++nfi) {
+		amrex::Long bytesWritten(0);
 		int const whichRDBytes(whichRD->numBytes());
 		int nFABs(0);
 		amrex::Long writeDataItems(0);
diff --git a/src/io/DiagFramePlane.cpp b/src/io/DiagFramePlane.cpp
--- a/src/io/DiagFramePlane.cpp
+++ b/src/io/DiagFramePlane.cpp
@@
-	amrex::Long bytesWritten(0);
-
 	std::string const filePrefix(a_mf_name + "_D_");
@@
 	}
 	for (; nfi.ReadyToWrite(); ++nfi) {
+		amrex::Long bytesWritten(0);
 		int const whichRDBytes(whichRD->numBytes());
 		int nFABs(0);
 		amrex::Long writeDataItems(0);
```
