# DataTable

`quokka::DataTable<Ndim, Nout, oob_policy>` is a generic n-dimensional interpolation table that supports multiple simultaneous outputs on the same coordinate grid. It is GPU-compatible: data is mirrored in pinned host memory and device memory, and a `DataTableGpuConst` view struct is passed into GPU kernels for zero-overhead interpolation.

**Source:** `src/util/DataTable.hpp`

## Template parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Ndim` | Number of input dimensions (1–4) | — |
| `Nout` | Number of output values per grid point | `1` |
| `oob_policy` | Out-of-bounds handling: `OutOfBounds::clamp` or `OutOfBounds::fail` | `clamp` |

## Coordinate spacing

Each input dimension has an independent spacing type, set per-dimension when the table is constructed or read:

| Spacing | Stored coordinate | Use case |
|---------|-------------------|----------|
| `SpacingType::linear` | physical value | uniformly spaced grids |
| `SpacingType::log` | natural log | log-spaced grids (standard) |
| `SpacingType::fast_log` | `FastMath::lg(x)` (bit-level log₂ approx) | hot paths on GPU |

Callers always pass **physical** values to `interpolate`; the spacing transform is handled internally.

## Construction

### From a CSV file

```cpp
auto table = quokka::DataTable<2, 3>::CSVReader("path/to/table.csv", SpacingType::linear);
```

CSV header format (lines 1–9 followed by data):
```
Ndim
Nx[0], Nx[1], ...
Nout
input_name_0, input_name_1, ...
output_name_0, output_name_1, ...
input_unit_0, input_unit_1, ...
output_unit_0, output_unit_1, ...
xlo[0], xlo[1], ...
xhi[0], xhi[1], ...
spacing[0], spacing[1], ...   # linear / log / fast_log
<data rows>
```

### From an HDF5 file (recommended)

```cpp
auto table = quokka::DataTable<2, 5>::H5Reader("path/to/table.h5", "tab1");
```

The HDF5 group `tab1` must contain:
- Dataset `data` with shape `[Nout, Nx0, Nx1, ...]` in C row-major order.
- Attributes `Ndim` (int32), `Nout` (int32), `Nx` (int32[Ndim]), `xlo` (float64[Ndim]), `xhi` (float64[Ndim]), `spacing` (fixed-length byte strings).
- Optional attributes `input_names`, `output_names`, `input_units`, `output_units`.

See [Cooling module](cooling_module.md) for the full attribute specification used by the cooling tables.

### From in-memory arrays

```cpp
std::array<amrex::Vector<amrex::Real>, 2> coords = {rho_vec, eint_vec};
std::array<amrex::Vector<amrex::Vector<amrex::Real>>, 2> data = {cool_2d, temp_2d};
quokka::DataTable<2, 2> table(coords, data);
```

## Interpolation

### On the GPU (inside `AMREX_GPU_DEVICE` lambdas)

```cpp
// Obtain a GPU-safe view from the host table (done once, outside kernels)
auto gpu_table = my_datatable.const_tables();  // device-backed

// Inside a ParallelFor kernel:
std::array<amrex::Real, 2> point = {rho, eint};  // physical values

// All outputs at once (most efficient; cost computed once per point)
std::array<amrex::Real, Nout> vals = gpu_table.interpolate(point);

// Single output by index
amrex::Real T = gpu_table.interpolate_single(point, 1 /*TEMPERATURE_IDX*/);
```

### On the host

```cpp
auto host_table = my_datatable.const_tables_host();  // pinned-memory view
auto vals = host_table.interpolate(point);
```

## Out-of-bounds behavior

With the default `OutOfBounds::clamp` policy, coordinates outside `[xlo, xhi]` are silently clamped to the nearest boundary before interpolation (nearest-neighbor extrapolation). With `OutOfBounds::fail`, the code aborts via `AMREX_ALWAYS_ASSERT_WITH_MESSAGE`.

## MPI

All data is read on the I/O processor and broadcast automatically to non-I/O ranks by `H5Reader` and `CSVReader`. No special MPI handling is needed at the call site.

## Key API

| Method | Description |
|--------|-------------|
| `H5Reader(file, group)` | Static factory: read from HDF5 group. |
| `CSVReader(file, output_spacing)` | Static factory: read from CSV file. |
| `const_tables()` | Return `DataTableGpuConst` view backed by device memory. |
| `const_tables_host()` | Return `DataTableGpuConst` view backed by pinned host memory. |
| `size(dim)` | Grid size along dimension `dim`. |
| `coord_xlo()` / `coord_xhi()` | Physical coordinate bounds (before any log transform). |
| `is_initialized()` | True after successful construction. |

## Example: reading a 2D, 5-output cooling table

```cpp
#include "util/DataTable.hpp"

quokka::DataTable<2, 5> tbl = quokka::DataTable<2, 5>::H5Reader(
    "extern/cooling/CloudyData_UVB=HM2012_resampled.h5", "tab1");

// Pass the GPU view to a kernel
auto gpu_tbl = tbl.const_tables();
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
    std::array<amrex::Real, 2> pt = {rho(i,j,k), eint(i,j,k)};
    auto vals = gpu_tbl.interpolate(pt);
    // vals[0] = cooling rate, vals[1] = T, vals[2] = cs, ...
});
```

## Writing a Python table for `H5Reader`

```python
import h5py, numpy as np

with h5py.File("my_table.h5", "w") as f:
    tab1 = f.create_group("tab1")
    tab1.attrs.create("Ndim",    np.int32(2))
    tab1.attrs.create("Nout",    np.int32(1))
    tab1.attrs.create("Nx",      np.array([nx0, nx1], dtype=np.int32))
    tab1.attrs.create("xlo",     np.array([xlo0, xlo1]))
    tab1.attrs.create("xhi",     np.array([xhi0, xhi1]))
    tab1.attrs.create("spacing", np.array(["fast_log", "fast_log"], dtype="S"))
    tab1.attrs.create("input_names",  np.array(["rho", "eint"], dtype="S"))
    tab1.attrs.create("output_names", np.array(["my_output"], dtype="S"))
    tab1.attrs.create("input_units",  np.array(["g/cm^3", "erg/g"], dtype="S"))
    tab1.attrs.create("output_units", np.array(["..."], dtype="S"))
    # data shape: [Nout, Nx0, Nx1]
    tab1.create_dataset("data", data=my_data[np.newaxis, :, :])
```

Fixed-length byte strings (`dtype="S"`) are required; variable-length HDF5 strings are not supported by `H5Reader`.
