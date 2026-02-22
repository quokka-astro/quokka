# Add Variable Selection to DiagPlotfile and Fix Initial Snapshot

Two improvements to the diagnostics system (`quokka.diagnostics`):

## Changes

- `src/io/DiagPlotfile.H`: Added `m_varNames` and `m_fcDirs` member variables; updated `processDiag` template to filter cc output and select fc directions
- `src/io/DiagPlotfile.cpp`: Added reading of `field_names` and `fc_dirs` from input file in `init()`
- `src/io/DiagBase.cpp`: Fixed initial snapshot — all diagnostics now output at step 0 when any interval is active, matching the behaviour of `plotfile_interval`/`plottime_interval`
- `inputs/ParticleSinkFormation.in`: Added `field_names` and `fc_dirs` to demonstrate the feature

## Behavior

### Variable selection (`DiagPlotfile`)

- **`field_names`**: Selects which cell-centered variables (including cell-averaged face-centered quantities like `x-BField`) appear in the main plotfile. Empty (default) = all variables.
- **`fc_dirs`**: Selects which spatial directions (`x`, `y`, `z`) to include in `fc_vars/` output. When MHD is enabled (fc variables exist), `fc_vars/` is always written — `fc_dirs` only controls which directions are included. Empty/Unspecified (default) = all directions.

### Initial snapshot fix (`DiagBase`)

All diagnostics now write an initial snapshot at step 0 whenever `int` or `time_int` is positive. Previously, time-based diagnostics (`time_int`) skipped the initial state.

## Example

```ini
quokka.plt.field_names = gasDensity x-GasMomentum y-GasMomentum z-GasMomentum temperature
quokka.plt.fc_dirs = x y   # omit z-BField from fc_vars
```
