# Checkpoint and restart

A checkpoint holds everything needed to continue a simulation: the state on
every AMR level, the grid hierarchy, timesteps, and all particle populations.
Checkpoints are written every `checkpoint_interval` coarse steps or every
`checkpointtime_interval` in simulation time, with filenames starting with
`checkpoint_prefix` (default `chk`). After each checkpoint, the symlink
`last_chk` is updated to point at it. See [Runtime parameters](parameters.md)
for these and related options.

To restart, set `restartfile` to a checkpoint (for example
`restartfile=chk0000010` or `restartfile=last_chk`). Restart detects the
checkpoint format automatically.

If a particle population enabled in the restarted run (for example tracers
turned on with `do_tracers=1`) is not in the checkpoint, Quokka prints a
warning and starts that population empty.

## openPMD checkpoints

Build with `-DQUOKKA_OPENPMD=ON`, then set these top-level input parameters:

```toml
checkpoint_format = "openpmd"
checkpoint_interval = 10
```

Each checkpoint is written as `chk0000010.bp`, with simulation metadata
(including particle history) in `chk0000010.yaml`, the same as an openPMD
plotfile. The checkpoints form a file-based openPMD series whose iteration
number is the coarse step. Restart with `restartfile=chk0000010` (or
`restartfile=chk0000010.bp` or `restartfile=last_chk`). Supply the same problem,
physics, physical domain, and refinement ratios as the original run. The restart
can use a different MPI rank count or grid decomposition and supports the
existing uniform-refinement restart path. `checkpoint_format` controls
subsequent writes; it does not select the format to read. The default remains
`amrex`, so an openPMD restart can write native AMReX checkpoints.

The layout follows openPMD plotfiles. Each state variable is a mesh per level
named as in plotfiles (for example `gasDensity` and `gasDensity_lvl1`), holding
only valid cells. Face-centered variables (for example `x_BField`) are staggered
meshes whose `position` is 0 along the face normal. A face shared by two patches
is a single dataset element, so both patches restore the same value. Particle populations use the
plotfile species, including packed AMReX ID/creation-rank values in `id`.
Iteration attributes prefixed with `quokka_` hold the extra state needed to
restart: the patch layout of each level, step counters, timestep sizes and times
for every level, field names, and each population's schema, count, and next-ID
counter (also for empty populations, which have no species). Ghost zones are
rebuilt on restart. Particles are redistributed onto the restart hierarchy.
Plotfiles lack these attributes, so they cannot be used as checkpoints.

Mesh data is copied from device memory into pinned host buffers for the I/O
backend, packing one patch at a time when the state has ghost zones. Restart
reads each patch through a host buffer, then copies it to device memory.
Particle reads stage each rank's share of a population in host memory before
redistribution. Mesh and particle floating-point data stored as `float` or
`double` is converted on read to the build's precision.

## ADIOS2 aggregation

openPMD output (checkpoints and plotfiles) goes through the ADIOS2 BP5 engine.
`openpmd.aggregation` controls whether BP5 aggregates writes:

| Parameter                       | Type    | Default  | Description |
|---------------------------------|---------|----------|-------------|
| `openpmd.aggregation`           | String  | `"auto"` | `on` keeps BP5's default aggregation (one aggregator per node, writing one subfile per node). `off` makes every rank write its own subfile. `auto` turns aggregation on from `openpmd.aggregation_min_nodes` nodes upward. |
| `openpmd.aggregation_min_nodes` | Integer | `32`     | Node count from which `auto` aggregates. |

The choice is printed once at the first openPMD output.

Without aggregation, writes are faster on a single node: about 2.4 times faster
on one CUDA node in our tests, matching native AMReX checkpoints. But the
number of files then grows with the rank count. The default threshold of 32
nodes is 256 ranks at 8 ranks per node, which matches native AMReX's default
of at most 256 files per MultiFab. This threshold is provisional. Measure both
modes on your machine with `scripts/slurm/frontier-openpmd-restart.submit`,
which compares them on Frontier.
