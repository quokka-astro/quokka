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
// All outputs linear (default):
auto table = quokka::DataTable<2, 5>::H5Reader("path/to/table.h5", "tab1");

// Per-output transforms — declared at the call site, not stored in the HDF5 file:
auto table = quokka::DataTable<2, 5>::H5Reader("path/to/table.h5", "tab1",
    {SpacingType::linear, SpacingType::fast_log, SpacingType::fast_log,
     SpacingType::fast_log, SpacingType::fast_log});
```

The HDF5 group `tab1` must contain:
- Dataset `data` with shape `[Nout, Nx0, Nx1, ...]` in C row-major order.
- Attributes `Ndim` (int32), `Nout` (int32), `Nx` (int32[Ndim]), `xlo` (float64[Ndim]), `xhi` (float64[Ndim]), `spacing` (fixed-length string[Ndim]).
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

With the default `OutOfBounds::clamp` policy, coordinates outside `[xlo, xhi]` are silently clamped to the nearest boundary before interpolation (nearest-neighbor extrapolation). With `OutOfBounds::fail`, the code aborts via `AMREX_ALWAYS_ASSERT_WITH_MESSAGE` when running on CPUs.

## MPI

All data is read on the I/O processor and broadcast automatically to non-I/O ranks by `H5Reader` and `CSVReader`. No special MPI handling is needed at the call site.

## Output transforms

Each output can independently use a different transform for interpolation. The transform type is declared at the C++ call site and is **not** stored in the HDF5 file (it is a property of the interpolation, not of the data).

| Transform | In HDF5 file | Internal buffer | Recovered on interpolate |
|-----------|-------------|-----------------|--------------------------|
| `linear` | physical value | physical value | buffer value |
| `fast_log` | physical value | `FastMath::inverse_pow2(physical)` | `FastMath::pow2(buffer)` |
| `log` | physical value | `ln(physical)` | `exp(buffer)` |

HDF5 files always store **raw physical values** regardless of transform. When `H5Reader` loads a `fast_log` output, it applies `FastMath::inverse_pow2` (Newton iteration) element-wise at load time so that subsequent bilinear interpolation happens in log space. No transform is needed in the Python table-generation scripts. See [Cooling module](cooling_module.md) for a concrete example.

### Why `fast_log` is as accurate as a true log–exp pair

`fast_pow2` and naive `fast_log2` are each a few-percent approximation to `2^x` and `log₂(x)`, respectively. Their direct composition is **not** the identity: `fast_pow2(fast_log2(q)) ≠ q`. So if the table stored `fast_log2(q)`, each grid point would carry a noticeable reconstruction error.

`FastMath::inverse_pow2` avoids that issue. It uses Newton iteration to solve `fast_pow2(v) = q` to machine precision, so it is the numerical inverse of `fast_pow2` (not an approximation to `log₂`). This gives two key properties:

1. **Grid points are exact.** The table stores `inverse_pow2(q_i)`, and lookup returns `fast_pow2(inverse_pow2(q_i)) = q_i` to machine precision.

2. **Off-grid accuracy is resolution-limited.** Between grid points, error is set by interpolation resolution, not by the standalone approximation error of `fast_pow2` versus `2^x`. Interpolating in `inverse_pow2` space and mapping back with `fast_pow2` has the same `O(h²)` interpolation order as a standard log-exp workflow.

The general principle is that interpolation is accurate when a smooth monotone transform and its true inverse are applied consistently. In this case, `(inverse_pow2, fast_pow2)` is that paired transform. How well either function approximates `2^x` or `log₂(x)` is secondary; Newton iteration is used to enforce machine-precision invertibility of the composition.

### Python reference implementations

Two functions from `src/math/FastMath.hpp` are useful when generating a `fast_log`-spaced grid in Python.

```python
import math

def fast_log2(x: float) -> float:
    """Maps a physical value to fast-log2 space.
    Uses frexp to split x = m * 2^e (m in [0.5, 1)), then approximates log2(m) as 2*(m-1)."""
    m, e = math.frexp(x)
    return 2.0 * (m - 1.0) + e

def inverse_fast_log2(y: float) -> float:
    """Inverse of fast_log2: maps a fast-log2-space value back to a physical value."""
    flr = math.floor(y)
    remainder = y - flr
    return math.ldexp(0.5 * (remainder + 1.0), flr + 1)
```

To build a uniformly-spaced grid in fast-log2 space:

```python
import numpy as np

v_min = fast_log2(x_min)
v_max = fast_log2(x_max)
v_grid = np.linspace(v_min, v_max, n)         # uniform in fast-log2 space
x_grid = np.vectorize(inverse_fast_log2)(v_grid)  # physical values at each grid point
```

These correspond to `FastMath::lg` and `FastMath::pow2` in C++. Python table-generation scripts should store **raw physical values** in the HDF5 file; `H5Reader` handles the internal transform at load time.

## Key API

| Method | Description |
|--------|-------------|
| `H5Reader(file, group, output_transforms)` | Static factory: read from HDF5 group. `output_transforms` defaults to all `linear`. |
| `CSVReader(file, output_transform)` | Static factory: read from CSV file; same transform applied to all outputs. |
| `const_tables()` | Return `DataTableGpuConst` view backed by device memory. |
| `const_tables_host()` | Return `DataTableGpuConst` view backed by pinned host memory. |
| `size(dim)` | Grid size along dimension `dim`. |
| `coord_xlo()` / `coord_xhi()` | Physical coordinate bounds (before any log transform). |
| `is_initialized()` | True after successful construction. |

## Example: reading a 2D, 5-output cooling table

```cpp
#include "util/DataTable.hpp"

// Cooling rate (index 0) is linear; T/cs/P/S (indices 1-4) are fast_log.
quokka::DataTable<2, 5> tbl = quokka::DataTable<2, 5>::H5Reader(
    "extern/cooling/CloudyData_UVB=HM2012_resampled.h5", "tab1",
    {quokka::SpacingType::linear,   // cooling rate — can be negative
     quokka::SpacingType::fast_log, // temperature
     quokka::SpacingType::fast_log, // sound speed
     quokka::SpacingType::fast_log, // pressure
     quokka::SpacingType::fast_log  // entropy
    });

// Pass the GPU view to a kernel
auto gpu_tbl = tbl.const_tables();
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
    std::array<amrex::Real, 2> pt = {rho(i,j,k), eint(i,j,k)};

    // All outputs at once — coordinate work done once, cheapest per value.
    auto vals = gpu_tbl.interpolate(pt);
    // vals[0] = cooling rate (raw), vals[1] = T (K), vals[2] = cs (cm/s), ...
    // fast_pow2 back-transform is applied automatically for indices 1-4

    // Single output by index — useful when only one quantity is needed.
    amrex::Real const T = gpu_tbl.interpolate_single(pt, 1 /*TEMPERATURE_IDX*/);
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
