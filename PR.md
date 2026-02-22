# Add Variable Selection to DiagPlotfile

Adds `field_names` and `fc_dirs` parameters to `DiagPlotfile` (the `quokka.plt.*` diagnostic), allowing users to select a subset of variables to write to plotfiles.

## Changes

- `src/io/DiagPlotfile.H`: Added `m_varNames` and `m_fcDirs` member variables; updated `processDiag` template to filter cc output and select fc directions
- `src/io/DiagPlotfile.cpp`: Added reading of `field_names` and `fc_dirs` from input file in `init()`
- `inputs/ParticleSinkFormation.in`: Added `field_names` and `fc_dirs` to demonstrate the feature

## Behavior

- **`field_names`**: Selects which cell-centered variables (including cell-averaged face-centered quantities like `x-BField`) appear in the main plotfile. Empty (default) = all variables.
- **`fc_dirs`**: Selects which spatial directions (`x`, `y`, `z`) to include in `fc_vars/` output. When MHD is enabled (fc variables exist), the folder `fc_vars/` is always written — `fc_dirs` only controls which directions are included. Empty/Unspecified (default) = all directions.

## Example

```ini
quokka.plt.field_names = gasDensity x-GasMomentum y-GasMomentum z-GasMomentum temperature
quokka.plt.fc_dirs = x y   # omit z-BField from fc_vars
```

