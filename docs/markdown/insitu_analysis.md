# In-situ analysis

*In-situ analysis* refers to analyzing simulations as they are running using Quokka's built-in runtime diagnostics.

## Diagnostics

Most of Quokka's diagnostics are adapted from the implementation included in the *Pele* suite of AMReX-based combustion codes. (See the documentation for [PeleLMeX diagnostics](https://amrex-combustion.github.io/PeleLMeX/manual/html/LMeXControls.html#run-time-diagnostics) for an explanation of the original implementation.)

There are three built-in diagnostics that can be configured to output at periodic intervals while the simulation is running:

- axis-aligned 2D projections
- axis-aligned 2D slices, and
- N-dimensional probability distribution functions (PDFs).

### 2D Projections

This diagnostic outputs 2D axis-aligned projections as AMReX plotfiles prefixed with ``proj`` and is configured using the ``ProjectionPlot`` diagnostic type. Only one direction is supported per diagnostic; configure multiple diagnostics to output multiple directions.

!!! Note
    Filters are not supported for ProjectionPlot diagnostics and will be ignored with a warning.

!!! Note
    Projection outputs are line integrals along the projection axis, using code length units. Each cell value is multiplied by ``dx[dir]`` and summed, so the output units are *(field units) × length*.

    Examples: ``nH`` (cm^-3) → projected ``nH`` (cm^-2) column density; ``pressure`` (K cm^-3) → projected pressure (K cm^-2). Avoid projecting per-cell extensive quantities like ``mass = rho * dV`` unless you intentionally want an extra length factor.

**AMR Structure Preservation**: As of recent updates, 2D projections now preserve the full AMR structure from the 3D simulation, maintaining higher resolution where needed instead of averaging down to level 0. This provides better detail but may result in larger output files.

This diagnostic is configured entirely with runtime parameters. Use ``field_names`` to select which plotfile/derived variables to project along the selected direction.

*Example input file configuration (defaults shown as inline comments):*

``` ini
quokka.diagnostics = proj_x proj_z        # enable two diagnostics with independent settings

quokka.proj_x.type = ProjectionPlot       # required: selects 2D projections
quokka.proj_x.file = proj/x/plt           # output path prefix (subdir + "pltNNNNNNN")
quokka.proj_x.int = 200                   # output cadence (coarse steps)
quokka.proj_x.normal = 0                  # 0==x, 1==y, 2==z (default: 0)
quokka.proj_x.field_names = nH temperature # fields to project (plotfile/derived vars)

quokka.proj_z.type = ProjectionPlot
quokka.proj_z.file = proj/z/plt
quokka.proj_z.int = 200
quokka.proj_z.normal = 2
quokka.proj_z.field_names = nH
```

### 2D Slices

!!! Note
    This is based on the *DiagFramePlane* diagnostic from PelePhysics, and the same runtime parameters should apply here without modification. The output format is also the same as that produced by the *Pele* codes.

This outputs 2D slices of the simulation as AMReX plotfiles that can be further examined using, e.g., VisIt or yt.

*Example input file configuration:*

``` ini
quokka.diagnostics = slice_z           # Space-separated name(s) of diagnostics (arbitrary)
quokka.slice_z.type = DiagFramePlane   # Diagnostic type (others may be added in the future)
quokka.slice_z.file = slicez_plt       # Output file prefix (should end in "plt")
quokka.slice_z.normal = 2              # Plane normal (0 == x, 1 == y, 2 == z)
quokka.slice_z.center = 2.4688e20      # Coordinate in the normal direction
quokka.slice_z.int    = 10             # Output interval (in number of coarse steps)

# The problem must output these derived variable(s)
derived_vars = temperature
# List of variables to include in output
quokka.slice_z.field_names = gasDensity gasInternalEnergy temperature
```

### Histograms/PDFs

!!! Note
    This is based on the *DiagPDF* diagnostic from PelePhysics, but significant changes have been made to both the runtime parameters and the output format in order to support N-dimensional histograms, log-spaced binning, and histogramming by mass.

This adds histogram outputs (as fixed-width text files) at fixed timestep intervals as the simulation evolves. The quantity accumulated in each bin is the total mass, volume, or cell count summed over all cells not covered by refined grids over all AMR levels. If unspecified in the input parameters, the default is to accumulate the volume in each bin.

By default, the bins extend over the full range of the data at a given timestep. The *range* parameter can instead specify the minimum and maximum extent for the bins. Bins can be optionally log-spaced by setting *log_spaced_bins = 1*.

Normalization of the output is left up to the user.

*Example input file configuration:*

``` ini
quokka.hist_temp.type = DiagPDF                         # Diagnostic type
quokka.hist_temp.file = PDFTempDens                     # Output file prefix
quokka.hist_temp.int  = 10                              # Output cadence (in number of coarse steps)
quokka.hist_temp.weight_by = mass                       # (Optional, default: volume) Accumulate: mass, volume, cell_counts
quokka.hist_temp.var_names = temperature gasDensity     # Variable(s) of interest (compute a N-D histogram)

quokka.hist_temp.temperature.nBins = 20                 # temperature: Number of bins
quokka.hist_temp.temperature.log_spaced_bins = 1        # temperature: (Optional, default: 0) Use log-spaced bins
quokka.hist_temp.temperature.range = 1e3 1e7            # temperature: (Optional, default: data range) Specify min/max of bins

quokka.hist_temp.gasDensity.nBins = 5                   # gasDensity: Number of bins
quokka.hist_temp.gasDensity.log_spaced_bins = 1         # gasDensity: (Optional, default: 0) Use log-spaced bins
quokka.hist_temp.gasDensity.range = 1e-29 1e-23         # gasDensity: (Optional, default: data range) Specify min/max of bins
```

*Filters (based on any variables, not necessary those used for the histogram) can be optionally added:*

``` ini
quokka.hist_temp.filters = dense                       # (Optional) List of filters
quokka.hist_temp.dense.field_name = gasDensity         # Filter field
quokka.hist_temp.dense.value_greater = 1e-25           # Filters: value_greater, value_less, value_inrange
```
