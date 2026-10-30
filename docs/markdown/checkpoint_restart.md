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

## Native particle restarts at scale

AMReX reads native particle checkpoints with at most `particles.nreaders` ranks
(default 64), so each reading rank holds `1/nreaders` of every particle
population before the particles are redistributed. On large runs with many
particles this can exhaust GPU memory: a 4096-rank tracer restart on Frontier
ran out of memory in `ParticleContainer::ReadParticles()`. Set
`particles.nreaders` to the number of ranks or higher (AMReX caps it at the
rank count) so every rank reads its share:

```
particles.nreaders = 1000000
```

openPMD checkpoints don't need this, because every rank reads its own share of
each particle population.

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

| Parameter             | Type   | Default | Description |
|-----------------------|--------|---------|-------------|
| `openpmd.aggregation` | String | `"off"` | `off` makes every rank write its own subfile. `on` keeps BP5's default aggregation (one aggregator per node, writing one subfile per node). |

The choice is printed once at the first openPMD output.

Without aggregation, the number of files grows with the rank count, but writes
and reads are usually faster. On Frontier (8 ranks per node),
`scripts/slurm/frontier-openpmd-restart.submit` gave these bandwidths for a
4096³ grid on 512 nodes and a 4096² × 8192 grid on 1024 nodes:

| Nodes | Output                         | Aggregation off | Aggregation on |
|-------|--------------------------------|-----------------|----------------|
| 512   | checkpoint write               | 2263 GB/s       | 673 GB/s       |
| 512   | plotfile write                 | 2043 GB/s       | 706 GB/s       |
| 1024  | checkpoint write               | 1190 GB/s       | 1372 GB/s      |
| 1024  | plotfile write                 | 2162 GB/s       | 1362 GB/s      |
| 1024  | checkpoint read (on 512 nodes) | 299 GB/s        | 43 GB/s        |

Read bandwidth varied by several times between identical runs. Measure both
modes on your machine with that script before switching to `on`.
