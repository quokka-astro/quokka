# In-situ analysis

*In-situ analysis* refers to analyzing the simulations as they are running. There are two options: using the *runtime diagnostics* that are built-in to Quokka, and volume rendering using *Ascent*, a third-party library.

## Diagnostics

Most of Quokka's diagnostics are adapted from the implementation included in the *Pele* suite of AMReX-based combustion codes. (See the documentation for [PeleLMeX diagnostics](https://amrex-combustion.github.io/PeleLMeX/manual/html/LMeXControls.html#run-time-diagnostics) for an explanation of the original implementation.)

There are three built-in diagnostics that can be configured to output at periodic intervals while the simulation is running:

- axis-aligned 2D projections
- axis-aligned 2D slices, and
- N-dimensional probability distribution functions (PDFs).

### 2D Projections

This diagnostic outputs 2D axis-aligned projections as AMReX plotfiles prefixed with ``proj``.

Currently, using this diagnostic requires implementing a custom function in the problem generator for your simulation. (In the future, this diagnostic may be improved so that it can be configured entirely with runtime parameters.)

The problem generator must call ``computePlaneProjection(F const &user_f, const int dir)`` where ``user_f`` is a lambda function that returns the value to project and ``dir`` is the axis along which the projection is taken.

*Example problem generator implementation:*

```cpp
template <> auto RadhydroSimulation<ShockCloud>::ComputeProjections(const int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
  // compute density projection
  std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;
  proj["nH"] = computePlaneProjection<amrex::ReduceOpSum>(
      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
        Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
        return (quokka::cooling::cloudy_H_mass_fraction * rho) / m_H;
      },
      dir);
  return proj;
}
```

*Example input file configuration:*

``` ini
projection_interval = 200
projection.dirs = x z
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
quokka.slice_z.interpolation = Linear  # Interpolation type: Linear (default: Linear)

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

## Volume Rendering (Ascent)

!!! Warning
    Ascent should only be used for volume rendering.  Other visualization features are not expected to work correctly, since we do **not** pass ghost cells to Ascent.

Ascent allows you to generate raytraced volume renderings (saved as a sequence of PNG images) while the simulation is running.

![](media/volume_render_sphere.png)

*A volume rendering of the `SphericalCollapse` problem.*

### Compiling Ascent on an HPC cluster

1.  Download Spack and activate it in your environment.
2.  Run `spack external find`.
3.  Make sure there are entries listed for `slurm` and `mpi` in your `~/.spack/packages.yaml` file.
4.  Add [buildable: False](https://spack.readthedocs.io/en/latest/build_settings.html#external-packages) to these entries.
5.  Run `spack fetch --dependencies ascent@develop`
6.  On a dedicated compute node, run `spack install ascent@develop`

If you are running your simulation on GPU nodes, you should add either `+cuda` or `+rocm` to the Spack spec, e.g. `spack install ascent@develop+cuda`, depending on whether you are running on NVIDIA or AMD GPUs, respectively.

### Compiling Quokka with Ascent support

1.  Load Ascent: `spack load ascent`
2.  Add `-DAMReX_ASCENT=ON -DAMReX_CONDUIT=ON` to your CMake options.
3.  Compile your problem, e.g.: `ninja -j4 test_hydro3d_blast`

### Running with Ascent

Add `ascent_interval = N` to your ParmParse input file, where `N` is the number of coarse timesteps between Ascent outputs.

### Customizing the rendering

Add an [ascent_actions.yaml file](https://ascent.readthedocs.io/en/latest/Actions/Actions.html) to the simulation working directory. This example actions file will create a volume rendering with the given camera parameters:

```yaml
- action: "add_extracts"
  extracts:
    my_volume_extract:
      type: "volume"
      params:
        field: "gasDensity"
        filename: "volume%05d"
        image_width: 512
        image_height: 512
        camera:
          look_at: [0.5, 0.5, 0.5]
          position: [-1.2, 0.499, 0.501]
          up: [0.0, 0.0, 1.0]
          fov: 60.0
          xpan: 0.0
          ypan: 0.0
          zoom: 1.0
          azimuth: 0.0
          elevation: 0.0
          near_plane: 0.01
          far_plane: 10.0
```

!!! Warning
    The `ascent_actions.yaml` file can be edited while the simulation is running, and the updated parameters will be used for subsequent renders. If invalid values are given, renders may stop working without notice.
