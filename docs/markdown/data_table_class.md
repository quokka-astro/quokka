QUOKKA provides a `DataTable` class for efficient multi-dimentional table interpolation. The class provides GPU-accelerated n-dimensional (1D-4D) table interpolation with support for multiple coordinate spacing types and automatic logarithmic transformations.

The `DataTable` class accepts reading from files of two data types: CSV and HDF5. 

### CSV Data Format

CSV data files for table interpolation follow this structure (see `inputs/lum_demo_2groups_fastlog.csv` for an example):

- `Ndim` (integer) — number of input dimensions
- `Nx` (comma-separated integers) — number of grid points for each input dimension  
- `Nout` (integer) — number of output dimensions
- `input_names` (comma-separated strings, Ndim entries) — names of input quantities (e.g., "age", "mass")
- `output_names` (comma-separated strings, Nout entries) — names of output quantities (e.g., "luminosity_group_0", "luminosity_group_1")
- `input_units` (comma-separated strings, Ndim entries) — units of input quantities (empty strings for dimensionless)
- `output_units` (comma-separated strings, Nout entries) — units of output quantities (empty strings for dimensionless)
- `xlo` (comma-separated reals, Ndim entries) — lower bounds of input ranges (values in physical units, not logarithms)
- `xhi` (comma-separated reals, Ndim entries) — upper bounds of input ranges (values in physical units, not logarithms)
- `spacing` (comma-separated strings, Ndim entries) — spacing type for each dimension: `linear`, `log`, or `fast_log`
- **Remaining lines**: `ydata` — output data in row-major order
  - For 2D tables: `Nx[1]` rows × `Nx[0]` columns (last dimension varies fastest)
  - For 3D tables: `(Nx[2] × Nx[1])` rows × `Nx[0]` columns
  - For 4D tables: `(Nx[3] × Nx[2] × Nx[1])` rows × `Nx[0]` columns

**Note**: When `spacing[dim]` is `log` or `fast_log`, the values in `xlo` and `xhi` should still be in physical units (not logarithms). The table infrastructure handles the log transformation internally.







## Radiation feedback from stars via 2D table interpolation

### Description

This PR implements a comprehensive 2D table interpolation infrastructure for stellar radiation feedback, enabling stochastic stellar population particles to interpolate `(age, mass)` to multi-group stellar luminosity. The implementation provides GPU-accelerated n-dimensional (1D-4D) table interpolation with support for multiple coordinate spacing types and automatic logarithmic transformations. The data format follows the discussion in #1243 and currently supports CSV input, with future plans to extend `H5Reader` and migrate the cooling module to this infrastructure.

### Key Features

**DataTable Infrastructure (`src/util/DataTable.hpp`)**:

- Extended `DataTable` to support multiple output dimensions (template parameter `Nout`)
- Implemented three coordinate spacing types: `linear`, `log`, and `fast_log` for both input coordinates and output values
- Automatic coordinate transformation: input values are transformed to log10 space for interpolation when `log` or `fast_log` spacing is specified
- Automatic output transformation: interpolated values are converted back from log10 space when output spacing is `log` or `fast_log`

**Particle Radiation System (`src/particles/`)**:t

- New `particle_radiation.hpp`: defines `LuminosityTables` class and `LuminosityUpdate` class for stellar luminosity evolution
- New `particle_update.hpp`: traits-based system for particle property updates via `ParticlePropertyUpdateTraits`
- Specialization for `StochasticStellarPop` particles that updates luminosity from tables using `(age, mass)` interpolation
- Support for multi-group radiation with automatic conversion between physical units (years, solar masses, erg/s) and code units

**Simulation Integration (`src/simulation.hpp`, `src/QuokkaSimulation.hpp`)**:
- Runtime parameters `particles.use_luminosity_table` and `particles.rad_table` for enabling and specifying luminosity table files
- Runtime parameter `particles.rad_table_output_spacing` for controlling output value storage (`linear`, `log`, or `fast_log`)
- Automatic table loading and validation during initialization
- Metadata validation: enforces input names ("age", "mass"), units ("year", "Msun"), and output units ("erg/s")

**Test Problem (`src/problems/ParticleRadiation/`)**:
- New `particle_radiation.cpp`: test problem for radiation from stellar particles with multi-group radiation transport
- Tests energy conservation by tracking total gas + radiation energy over multiple timesteps
- Validates luminosity interpolation against expected values for different spacing types
- Three test configurations:
  - `ParticleRadiation`: baseline test with linear spacing
  - `ParticleRadiationLog`: test with logarithmic coordinate spacing
  - `ParticleRadiationFastlog`: test with fast_log spacing (using FastMath approximations)

**Example Data Files** (`inputs/`):
- `lum_demo_2groups.csv`: Linear spacing example
- `lum_demo_2groups_log.csv`: Logarithmic spacing example  
- `lum_demo_2groups_fastlog.csv`: Fast logarithmic spacing example

### CSV Data Format

CSV data files for table interpolation follow this structure (see `inputs/lum_demo_2groups_fastlog.csv` for an example):

- `Ndim` (integer) — number of input dimensions
- `Nx` (comma-separated integers) — number of grid points for each input dimension  
- `Nout` (integer) — number of output dimensions
- `input_names` (comma-separated strings, Ndim entries) — names of input quantities (e.g., "age", "mass")
- `output_names` (comma-separated strings, Nout entries) — names of output quantities (e.g., "luminosity_group_0", "luminosity_group_1")
- `input_units` (comma-separated strings, Ndim entries) — units of input quantities (empty strings for dimensionless)
- `output_units` (comma-separated strings, Nout entries) — units of output quantities (empty strings for dimensionless)
- `xlo` (comma-separated reals, Ndim entries) — lower bounds of input ranges (values in physical units, not logarithms)
- `xhi` (comma-separated reals, Ndim entries) — upper bounds of input ranges (values in physical units, not logarithms)
- `spacing` (comma-separated strings, Ndim entries) — spacing type for each dimension: `linear`, `log`, or `fast_log`
- **Remaining lines**: `ydata` — output data in row-major order
  - For 2D tables: `Nx[1]` rows × `Nx[0]` columns (last dimension varies fastest)
  - For 3D tables: `(Nx[2] × Nx[1])` rows × `Nx[0]` columns
  - For 4D tables: `(Nx[3] × Nx[2] × Nx[1])` rows × `Nx[0]` columns

**Note**: When `spacing[dim]` is `log` or `fast_log`, the values in `xlo` and `xhi` should still be in physical units (not logarithms). The table infrastructure handles the log transformation internally.

### Cleanup

- Removed obsolete `RadParticle` and `RadParticle2D` test problems (superseded by unified `ParticleRadiation` test)
- Updated `Advection`, `Advection2D`, and `AdvectionSemiellipse` tests to include `nGroups` constant for consistency

### Related issues
Are there any GitHub issues that are fixed by this pull request? Add a link to them here.

### Checklist
_Before this pull request can be reviewed, all of these tasks should be completed. Denote completed tasks with an `x` inside the square brackets `[ ]` in the Markdown source below:_
- [x] I have added a description (see above).
- [x] I have added a link to any related issues (if applicable; see above).
- [x] I have read the [Contributing Guide](https://github.com/quokka-astro/quokka/blob/development/CONTRIBUTING.md).
- [x] I have added tests for any new physics that this PR adds to the code.
- [x] *(For quokka-astro org members)* I have manually triggered the GPU tests with the magic comment `/azp run`.