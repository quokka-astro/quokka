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
| `TransformType::linear` | physical value | uniformly spaced grids |
| `TransformType::log` | natural log | log-spaced grids (standard) |
| `TransformType::fast_log` | `FastMath::lg(x)` (bit-level log₂ approx) | hot paths on GPU |

Callers always pass **physical** values to `interpolate`; the spacing transform is handled internally.

## Construction

### From a CSV file

```cpp
auto table = quokka::DataTable<2, 3>::CSVReader("path/to/table.csv", TransformType::linear);
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
    {TransformType::linear, TransformType::fast_log, TransformType::fast_log,
     TransformType::fast_log, TransformType::fast_log});
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

HDF5 files always store **raw physical values** regardless of transform. When `H5Reader` loads a `fast_log` output, it applies `FastMath::inverse_pow2` (Newton iteration) element-wise at load time so that subsequent n-linear interpolation happens in the transformed space used by `FastMath::pow2`. No transform is needed in the Python table-generation scripts. See [Cooling module](cooling_module.md) for a concrete example.

### Why `inverse_pow2` is used for `fast_log` outputs

Using `FastMath::lg(q)` directly for stored outputs would not be invertible under this pipeline, because in general `FastMath::pow2(FastMath::lg(q)) != q`.

Instead, for `TransformType::fast_log` outputs, `H5Reader` stores `FastMath::inverse_pow2(q)` in the internal buffer and interpolation returns `FastMath::pow2(...)` of the interpolated value. `FastMath::inverse_pow2` is computed by Newton iteration to solve `FastMath::pow2(v) = q` with a relative residual tolerance (`1e-15` in the current implementation), so it acts as a numerical inverse of `FastMath::pow2` rather than as an approximation to `log2`.

This gives the behavior we want:

1. **At grid points:** reconstruction uses an inverse-consistent pair (`inverse_pow2` then `pow2`), so values are recovered to solver tolerance.
2. **Between grid points:** error is dominated by multilinear interpolation in transform space and table resolution, not by an avoidable mismatch like `pow2(lg(q))`.

This is analogous in spirit to standard log-space interpolation, but it is not mathematically identical to exact `ln/exp` interpolation.

### Python reference implementations

Two functions are needed when generating a `fast_log`-spaced grid in Python. `fast_log2` maps physical values to fast-log2 space; `inverse_fast_log2` inverts it via root-finding, giving the physical values that correspond to regularly-spaced fast-log2 coordinates.

```python
import numpy as np
from scipy.optimize import brentq

def fast_log2(x):
    """Fast approximation of log2(x). Mirrors FastMath::lg in C++."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("fast_log2 undefined for x <= 0")
    mantissa, exponent = np.frexp(x)   # x = mantissa * 2^exponent, mantissa in [0.5, 1)
    return 2.0 * (mantissa - 1.0) + exponent

def inverse_fast_log2(y):
    """Inverse of fast_log2: find x such that fast_log2(x) == y, to machine precision.
    Uses Brent's method with an optional Newton-Raphson refinement step."""
    y = np.asarray(y, dtype=float)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    eps = np.finfo(np.float64).eps
    result = np.empty_like(y)
    for i, y_val in enumerate(y.flat):
        x_guess = 2.0 ** y_val
        x_sol = brentq(lambda x: fast_log2(x) - y_val,
                       x_guess * 0.5, x_guess * 2.0,
                       xtol=eps * x_guess, rtol=4 * eps, maxiter=1000)
        # Newton-Raphson refinement if needed
        for _ in range(5):
            dx = -(fast_log2(x_sol) - y_val) * x_sol * np.log(2.0)
            x_new = x_sol + dx
            if abs(fast_log2(x_new) - y_val) < abs(fast_log2(x_sol) - y_val):
                x_sol = x_new
        result.flat[i] = x_sol
    return result[0] if scalar else result
```

To build a uniformly-spaced grid in fast-log2 space:

```python
v_grid = np.linspace(fast_log2(x_min), fast_log2(x_max), n)  # uniform in fast-log2 space
x_grid = inverse_fast_log2(v_grid)                            # physical values at each point
```

Python table-generation scripts should store **raw physical values** in the HDF5 file; `H5Reader` handles the internal transform at load time.

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
    {quokka::TransformType::linear,   // cooling rate — can be negative
     quokka::TransformType::fast_log, // temperature
     quokka::TransformType::fast_log, // sound speed
     quokka::TransformType::fast_log, // pressure
     quokka::TransformType::fast_log  // entropy
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
