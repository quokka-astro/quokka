# Adaptive Mesh Refinement Guidelines

Quokka relies on the Berger-Collela AMR algorithm [@berger1989local] provided by AMReX to create a hierarchy of nested grids. This page distills practical advice for planning and tuning AMR runs so the mesh tracks the physics you care about without wasting resolution elsewhere.

## Understanding the Refinement Workflow

AMReX walks through a predictable sequence each time it builds or adjusts the mesh hierarchy. Cells are tagged inside `ErrorEst`, those tags are coarsened according to `blocking_factor/ref_ratio`, clustered by the Berger-Rigoutsos algorithm [@berger1991algorithm], and then promoted to new grids that honor `blocking_factor` and `max_grid_size`. The resulting hierarchy is averaged down so coarse levels remain consistent with fine ones. Keeping this loop in mind helps interpret why the mesh responds the way it does when the problem evolves or when you revisit your refinement logic.

In Quokka, AMReX still calls an `ErrorEst` function, but this is now a thin wrapper around `refineGrid` that also invokes `refineGridsAroundParticles` to tag cells surrounding particles. The user should therefore define `refineGrid` instead of `ErrorEst`; the wrapper will automatically call `refineGrid` and then add particle-based tags on top.

## Setting AMR Parameters

Several inputs shape how aggressively the hierarchy responds to refinement tags:

| Parameter | Description | Typical Values | Performance Impact |
|-----------|-------------|----------------|-------------------|
| `blocking_factor` | Minimum grid dimension; grids must be integer multiples of this value | 8-32 | Large values improve GPU throughput but can force broad refinement |
| `grid_efficiency` | Minimum fraction of tagged cells in each grid | 0.7-1.0 | Higher values reduce wasted work yet may spawn extra boxes |
| `max_grid_size` | Maximum dimension of a grid patch | 64-128 | Smaller sizes improve load balance; larger sizes cut communication |
| `ref_ratio` | Refinement ratio between adjacent levels | 2 or 4 | Larger ratios sharpen features faster but deepen the hierarchy |

Choose these controls as a set: a generous `blocking_factor` demands tighter tagging or higher `grid_efficiency` to avoid carpet refinement, while a smaller blocking factor allows surgical meshes but can stress GPUs. When in doubt, run short experiments that sweep one parameter at a time and compare the resulting plotfile headers to see how many grids are created per level.

## Cell-Centered AMR Interpolation (Slope Limiters)

Quokka uses AMReX's cell-centered conservative linear interpolation for coarse-to-fine fills. The runtime switch `amr_interpolation_method` is an `AMREX_ENUM` with these options:

- `piecewise_constant`: `mf_pc_interp`
- `linear_ll`: `mf_lincc_interp` (LL-limited linear)
- `linear_mc`: `mf_cell_cons_interp` (MC-limited linear)
- `linear_ll_minmax` (default): `mf_linear_slope_minmax_interp`

AMReX also provides two related slope limiter variants that are not currently wired into the Quokka runtime parameter:

- **LL slope** (`mf_lincc_interp`): computes per-direction scaling factors from the minimum ratio of limited to unlimited slopes across *all components*, then applies those scalings to every component. This coupling preserves linear combinations of state variables during interpolation.
- **MC slope** (`mf_cell_cons_interp`): computes a per-component monotonic-central slope and applies a local `alpha` limiter based on nearby extrema and the refinement ratio. Limiting is component-wise, without cross-component coupling.
- **Min/max slope-vector limiting** (`mf_linear_slope_minmax_interp`): starts from LL slopes but also limits the *slope vector* so the reconstructed fine values do not create new extrema. In contrast, `mf_lincc_interp` only limits each slope *component* via the shared per-direction factors.
- **When is vector limiting enough?** The min/max slope-vector limiter is what directly enforces “no new extrema” for each component. The extra LL per-direction scaling is primarily about *cross-component consistency* (preserving linear combinations across components). Using both is more conservative; if your only goal is to avoid new extrema, vector limiting alone is sufficient.

See `extern/amrex/Src/AmrCore/AMReX_MFInterp_1D_C.H`, `extern/amrex/Src/AmrCore/AMReX_MFInterp_2D_C.H`, and `extern/amrex/Src/AmrCore/AMReX_MFInterp_3D_C.H` for the exact limiter kernels. If you want LL or MC slopes selectable at runtime, Quokka would need to extend `getAmrInterpolaterCellCentered()` in `src/simulation.hpp`.

## Performance and Scaling Considerations

GPU runs typically favor `blocking_factor ≥ 32` so each patch keeps SMs busy; values below 32 almost always break memory coalescing across a warp and leave expensive kernels underutilized even when occupancy looks fine. CPU multigrid solves prefer at least 8. If you push those limits, guard your memory footprint by trimming the number of refinement levels or enlarging `max_grid_size` so patches do not proliferate. Also watch how frequently you trigger `regrid` calls—rapid regridding magnifies the cost of tag evaluation and repartitioning, so coarse control logic or hysteresis inside `refineGrid` can pay dividends on very dynamic problems.

## Observing and Adjusting the Hierarchy

Inspect the grid structure early in a run by opening the plotfile header or by enabling verbose regridding logs in AMReX. When the layout diverges from expectations, compare the tagged volume to the resulting grid inventory—discrepancies usually trace back to either tag dilution during coarsening or to efficiency limits that extend each patch. The spherical collapse study documented in [GitHub issue #978](https://github.com/quokka-astro/quokka/issues/978) is a useful reminder that large blocking factors can still trigger whole-domain refinement if tags are sparse, so monitor that scenario whenever you upscale a production run.

### Inspecting grids with `amrex_fboxinfo`

AMReX ships a lightweight reporter named `amrex_fboxinfo` that prints the box layout stored in a plotfile. Make sure your CMake cache was configured with `-DAMReX_PLOTFILE_TOOLS=ON`, then build the utility once per build tree with `cmake --build build --target fboxinfo`. The resulting executable lives at `build/extern/amrex/Tools/Plotfile/amrex_fboxinfo`. Point it at any plotfile to see how much of the domain each AMR level occupies and how many boxes AMReX created:

```
$ build/extern/amrex/Tools/Plotfile/amrex_fboxinfo plt00040
 plotfile: plt00040
 level   0: number of boxes =      1, volume = 100.00%, number of cells = 16384
          maximum zones =     128 x     128
 level   1: number of boxes =      4, volume =  12.50%, number of cells =  8192
          maximum zones =     256 x     256
```

The summary lines are the quickest health check: `volume` reports the percentage of the level-`ℓ` domain covered by refined grids, so values creeping toward 100% warn that your refinement criteria are effectively forcing a uniform mesh. The `maximum zones` line prints the physical extents of each level’s valid domain; large jumps confirm that refinement ratios and blocking factors coarsen or refine by the factors you expect. Add `--full` to list every box and verify the coordinates land on multiples of `blocking_factor`, or `--gridfile` to emit a grid-description file for AMReX regression tools. When you only need the number of active refinement levels (for log parsing, for example), pass `--levels` to suppress the rest of the report.

### Example: Berger-Rigoutsos in Practice

![Tagged cells and Berger-Rigoutsos boxes](media/amr_grid_refinement.svg)

Compile Quokka with `-DAMReX_SPACEDIM=2`, then run the HydroBlast2D driver with the bundled `inputs/blast2d_amr_example.in`. That input sets `amr.n_cell = 128 128 4`, `amr.ref_ratio = 2 2 1`, `amr.blocking_factor = 32 32 4`, `amr.grid_eff = 0.8`, and `amr.n_error_buf = 1`. The third entry in each list is ignored in 2D, so the effective blocking factor is 32. The pressure-gradient trigger in `refineGrid` tags a thin ring around the blast front. AMReX coarsens those tags by 16 cells before clustering, so the Berger-Rigoutsos pass finds four rectangles that tile the annulus: two long bands `(lo, hi) = (64,96)→(191,127)` and `(64,128)→(191,159)` plus two shorter bridges `(96,64)→(159,95)` and `(96,160)→(159,191)`. Running `amrex_fboxinfo --full plt00040` on the resulting plotfile shows the same coordinates, confirming that the level-one patches are multiples of 32 cells in each direction and fully envelop the tagged region after refinement. Comparing those boxes with the tags in the plotfile (as sketched above) is a quick way to verify that the clustering step is behaving the way you expect.

## Troubleshooting Patterns

Start with the simplest levers. Reduce `blocking_factor` temporarily to verify the tagging logic, then restore the larger production value once you confirm the tags are reasonable. Next, nudge `grid_efficiency` upward until the refined region stabilizes without spawning many more boxes. If the hierarchy thrashes between steps, consider caching diagnostic arrays that record why cells were tagged; a quick visualization of those masks often pinpoints whether gradients, thresholds, or boundary effects are misbehaving.

## Further Reading

For deeper dives into refinement theory and AMReX implementation details, consult the original Berger & Colella paper [@berger1989local], the Berger-Rigoutsos clustering method [@berger1991algorithm], and the [AMReX-Astro Castro regridding notes](https://amrex-astro.github.io/Castro/docs/regridding.html). The [AMReX source documentation](https://github.com/AMReX-Codes/amrex) also clarifies lower-level options that Quokka exposes through its input files.
