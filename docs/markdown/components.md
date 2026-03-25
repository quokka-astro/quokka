# State Variable Component Indices

Quokka stores all cell-centred (cc) conserved quantities in a single `MultiFab` and all face-centred (fc) quantities in a separate per-dimension `MultiFab` array. The component layout within these arrays is controlled by `Physics_Traits`, `Physics_NumVars`, and `Physics_Indices`.

## Configuration structs

### `Physics_NumVars`

Defined in `src/physics_numVars.hpp`. Holds hard-coded variable counts per module.


| Constant              | Value                                 | Meaning                                                                                             |
| --------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `numHydroVars`        | 6                                     | Conserved hydro variables: density, 3 momenta, total energy, internal energy                        |
| `numRadVarsPerGroup`  | 4                                     | Radiation variables per photon group: energy density, 3 flux components                             |
| `numDustVarsPerGroup` | 4                                     | Dust variables per dust group: density, 3 momenta                                                   |
| `numMHDVars_per_dim`  | 1                                     | Face-centred MHD variables per spatial dimension (the magnetic field component normal to that face) |
| `numMHDVars_tot`      | `AMREX_SPACEDIM * numMHDVars_per_dim` | Total face-centred MHD variables across all dimensions                                              |


### `Physics_Traits<problem_t>`

Defined in `src/physics_info.hpp`. User-specialized per problem to enable/disable physics modules and set group counts.


| Field                                                                           | Meaning                                                                             |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `is_hydro_enabled`                                                              | Enable hydrodynamics                                                                |
| `is_radiation_enabled`                                                          | Enable radiation transport                                                          |
| `is_dust_enabled`                                                               | Enable dust dynamics                                                                |
| `is_mhd_enabled`                                                                | Enable MHD (face-centred magnetic fields)                                           |
| `is_self_gravity_enabled`                                                       | Enable self-gravity                                                                 |
| `numMassScalars`                                                                | Number of mass scalars (advected proportional to density)                           |
| `numPassiveScalars`                                                             | Total number of passive scalars (must be >= `numMassScalars`)                       |
| `nGroups`                                                                       | Number of radiation photon groups                                                   |
| `nDustGroups`                                                                   | Number of dust groups                                                               |
| `unit_system`                                                                   | Unit system: `CGS`, `CONSTANTS`, or `CUSTOM`                                        |
| `c_light`, `radiation_constant`, `boltzmann_constant`, `gravitational_constant` | Physical constants (set automatically for CGS, user-defined for other unit systems) |
| `unit_length`, `unit_mass`, `unit_time`, `unit_temperature`                     | Code unit definitions (only meaningful for `CUSTOM` unit system)                    |


## Cell-centred state vector layout

All cell-centred conserved variables are packed into a single `MultiFab` (`state_new_cc_`). The total number of components is `Physics_Indices<problem_t>::nvarTotal_cc`.

### `Physics_Indices<problem_t>`

Defined in `src/physics_info.hpp`. Computes starting indices and total component counts.


| Constant            | Definition                                                             | Meaning                                                                      |
| ------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `nvarTotal_cc_adv`  | 1                                                                      | Number of cc variables for a pure advection problem (no hydro, no radiation) |
| `nvarTotal_cc`      | (see below)                                                            | Total number of cell-centred components in the state vector                  |
| `hydroFirstIndex`   | 0                                                                      | Starting index of hydro variables                                            |
| `pscalarFirstIndex` | `numHydroVars` (= 6)                                                   | Starting index of passive scalars                                            |
| `dustFirstIndex`    | `pscalarFirstIndex + numPassiveScalars`                                | Starting index of dust variables                                             |
| `radFirstIndex`     | `dustFirstIndex + numDustVarsPerGroup * nDustGroups * is_dust_enabled` | Starting index of radiation variables                                        |
| `nvarPerDim_fc`     | `numMHDVars_per_dim * is_mhd_enabled`                                  | Number of face-centred variables per dimension                               |
| `nvarTotal_fc`      | `AMREX_SPACEDIM * nvarPerDim_fc`                                       | Total face-centred variables                                                 |
| `mhdFirstIndex`     | 0                                                                      | Starting index of MHD variables within each face-centred `MultiFab`          |


`**nvarTotal_cc` calculation:**

- If neither hydro nor radiation is enabled: `nvarTotal_cc = nvarTotal_cc_adv` (= 1, for pure advection).
- Otherwise: `nvarTotal_cc = numHydroVars + numPassiveScalars + numDustVarsPerGroup * nDustGroups * is_dust_enabled + numRadVarsPerGroup * nGroups * is_radiation_enabled`.

## Cell-centred component map

The cell-centred state vector is laid out contiguously as:

```
Index:  0                          6               6+Nps           dustFirstIndex        radFirstIndex
        |--- Hydro (6) ------------|-- Scalars -----|--- Dust ------|- Radiation ---------|
```

### Hydro block (6 variables, starting at `hydroFirstIndex = 0`)

Defined by `HydroSystem<problem_t>::consVarIndex`:


| Index | Name                   | Quantity                                          |
| ----- | ---------------------- | ------------------------------------------------- |
| 0     | `density_index`        | Gas mass density                                  |
| 1     | `x1Momentum_index`     | x-momentum                                        |
| 2     | `x2Momentum_index`     | y-momentum                                        |
| 3     | `x3Momentum_index`     | z-momentum                                        |
| 4     | `energy_index`         | Total energy density                              |
| 5     | `internalEnergy_index` | Auxiliary internal energy (dual energy formalism) |


### Passive scalar block (`numPassiveScalars` variables, starting at `pscalarFirstIndex = 6`)


| Index            | Name            | Quantity                                            |
| ---------------- | --------------- | --------------------------------------------------- |
| 6                | `scalar0_index` | First passive scalar                                |
| 6+1 .. 6+Nms-1   |                 | Additional mass scalars (advected with density)     |
| 6+Nms .. 6+Nps-1 |                 | Additional passive scalars (advected independently) |


where `Nms = numMassScalars` and `Nps = numPassiveScalars`. Mass scalars are the first `numMassScalars` entries of the passive scalar block; they are advected proportional to the density field.

### Dust block (`numDustVarsPerGroup * nDustGroups` variables, starting at `dustFirstIndex`)

Only present when `is_dust_enabled = true`. For each dust group `g` (0-indexed):


| Offset within group | Name                   | Quantity        |
| ------------------- | ---------------------- | --------------- |
| 0                   | `dustDensity_index`    | Dust density    |
| 1                   | `x1DustMomentum_index` | Dust x-momentum |
| 2                   | `x2DustMomentum_index` | Dust y-momentum |
| 3                   | `x3DustMomentum_index` | Dust z-momentum |


Variables for group `g` start at `dustFirstIndex + g * numDustVarsPerGroup`.

### Radiation block (`numRadVarsPerGroup * nGroups` variables, starting at `radFirstIndex`)

Only present when `is_radiation_enabled = true`. For each photon group `g` (0-indexed):


| Offset within group | Name              | Quantity                    |
| ------------------- | ----------------- | --------------------------- |
| 0                   | `radEnergy_index` | Radiation energy density    |
| 1                   | `x1RadFlux_index` | Radiation flux, x-component |
| 2                   | `x2RadFlux_index` | Radiation flux, y-component |
| 3                   | `x3RadFlux_index` | Radiation flux, z-component |


Variables for group `g` start at `radFirstIndex + g * numRadVarsPerGroup`.

Note: `RadSystem` defines `nstartHyperbolic_ = radFirstIndex` and `nvarHyperbolic_ = numRadVarsPerGroup * nGroups`. Its internal `nvar_` includes all preceding variables: `nvar_ = nstartHyperbolic_ + nvarHyperbolic_` (equal to `nvarTotal_cc`).

## Face-centred state vector layout

Face-centred variables are stored in a `std::array<MultiFab, AMREX_SPACEDIM>` (`state_new_fc_`), one `MultiFab` per spatial dimension. Each contains `nvarPerDim_fc` components.

### MHD block (1 variable per dimension, starting at `mhdFirstIndex = 0`)

Only present when `is_mhd_enabled = true`. Defined by `MHDSystem<problem_t>::varIndex_perDim`:


| Index (per dim) | Name           | Quantity                                            |
| --------------- | -------------- | --------------------------------------------------- |
| 0               | `bfield_index` | Normal component of the magnetic field on that face |


The face-centred storage follows the Constrained Transport (CT) convention: `state_new_fc_[0]` stores Bx on x-faces, `state_new_fc_[1]` stores By on y-faces, and `state_new_fc_[2]` stores Bz on z-faces (in 3D).

## Per-system variable counts

Each physics system class also defines its own `nvar_` for internal use:


| Class         | `nvar_`                                                                                  | Meaning                                                               |
| ------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `HydroSystem` | `numHydroVars + numPassiveScalars + numDustVarsPerGroup * nDustGroups * is_dust_enabled` | All cc variables managed by the hydro solver (hydro + scalars + dust) |
| `RadSystem`   | `radFirstIndex + numRadVarsPerGroup * nGroups`                                           | All cc variables up to and including radiation (= `nvarTotal_cc`)     |
| `MHDSystem`   | `nvar_per_dim_` = 1 per dimension, `nvar_tot_` = `AMREX_SPACEDIM` total                  | Face-centred magnetic field components                                |


## Example: typical hydro + radiation problem

For a problem with `numMassScalars = 1`, `numPassiveScalars = 2` (1 mass scalar + 1 additional passive scalar), `is_dust_enabled = false`, `nGroups = 2`, `is_radiation_enabled = true`:

```
Cell-centred layout (16 components total):
  [0]  density
  [1]  x1Momentum
  [2]  x2Momentum
  [3]  x3Momentum
  [4]  energy
  [5]  internalEnergy
  [6]  scalar0 (mass scalar)
  [7]  scalar1 (passive scalar)
  [8]  radEnergy (group 0)
  [9]  x1RadFlux (group 0)
  [10] x2RadFlux (group 0)
  [11] x3RadFlux (group 0)
  [12] radEnergy (group 1)
  [13] x1RadFlux (group 1)
  [14] x2RadFlux (group 1)
  [15] x3RadFlux (group 1)
```

