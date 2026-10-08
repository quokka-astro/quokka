# ffieldlines

Trace magnetic field lines through a Quokka (AMReX) plotfile and write them as
VTK PolyData for ParaView and VisIt.

Each half-line (forward and backward from a seed) is an AMReX particle. It is
integrated with RK4 in arc length, `dx/ds = ±B/|B|`. B and the sampled gas
fields are interpolated trilinearly from cell-centered plotfile data on the
finest AMR level that covers each point.

## Build

This is a standalone CMake project that builds AMReX from Quokka's
`extern/amrex` submodule. It is not part of the Quokka problem build.

```sh
git submodule update --init extern/amrex
cmake -S tools/ffieldlines -B build/ffieldlines -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/ffieldlines
ctest --test-dir build/ffieldlines --output-on-failure   # needs python3 + numpy
```

MPI is on by default (`-DAMReX_MPI=OFF` to disable). OpenMP is off by default
(`-DAMReX_OMP=ON` to enable). v1 targets CPUs.

## Usage

```sh
mpiexec -n 8 build/ffieldlines/ffieldlines \
    --periodic 0 0 0 \
    --seed-disk 0 0 0  0 0 1  3.086e22 500 \
    --max-length 3.086e23 \
    -o lines.vtp \
    plt0000020
```

`--periodic` is required because plotfiles do not record domain periodicity.

Seeds are given in one of three ways:

- `--seeds FILE`: a text file with one `x y z` per line;
- `--seed-line X0 Y0 Z0 X1 Y1 Z1 N`;
- `--seed-disk CX CY CZ NX NY NZ R N`: evenly spread points on a disk.

At most 10,000 seeds are accepted. Run `ffieldlines --help` for all options.

The defaults read Quokka's variable names: `x-BField y-BField z-BField`,
`gasDensity`, `x-GasMomentum y-GasMomentum z-GasMomentum`, and `temperature`.
Temperature is only present if the run listed it in `derived_vars`. Use
`--temperature none` to skip it.

## Output

`.vtp` (default) is VTK XML PolyData with raw appended binary. `--format vtk`
writes legacy binary VTK instead. Each line is one polyline cell; a line that
wraps through a periodic boundary is split into several cells (pieces).

| point array | meaning |
|---|---|
| `arc_length` | signed arc length from the seed (negative on the backward half) |
| `density`, `temperature` | interpolated plotfile values |
| `B_mag`, `v_mag` | `|B|` and `|v|`, with `v = m/ρ` formed from interpolated `m` and `ρ` at the point |
| `v_dot_B` | `v · B` |
| `cos_vB` | `v·B / (|v||B|)`, NaN where either magnitude vanishes |

| cell array | meaning |
|---|---|
| `seed_id` | index of the seed |
| `piece_index` | piece number within a line split at periodic boundaries |
| `status_forward`, `status_backward` | why each half stopped (below); `-1` if not traced |
| `length_forward`, `length_backward` | arc length of each half |

Status codes:

| code | name | meaning |
|---|---|---|
| 1 | `max_length` | reached `--max-length` |
| 2 | `max_steps` | reached `--max-steps` |
| 3 | `weak_field` | reached a point with `|B| <= --b-min` |
| 4 | `domain_exit` | left the domain through a non-periodic face |
| 5 | `closed_loop` | returned to within `--loop-tolerance` cells of its seed |
| 6 | `max_sweeps` | hit the `--max-sweeps` safety cap |
| 7 | `bad_sample` | could not interpolate the field |

A closed forward loop already contains the whole line, so its backward half is
not drawn.

The file also stores `TimeValue`/`TIME`/`CYCLE` field data. The command line
and the plotfile's `metadata.yaml` (units, git hashes) are kept in an XML
comment at the top of the `.vtp`.

In ParaView, color by `cos_vB` and apply a `Tube` filter. In VisIt, use a
Pseudocolor plot of `cos_vB`.

## Design notes

- **Integrating only where data is owned.** A particle integrates only while
  its cell is in the valid region of its own grid and is not covered by a finer
  level. Otherwise it stops until the next `Redistribute()` moves it to the
  owning grid, level, and rank.
  - The output therefore does not depend on the grid decomposition, the number
    of MPI ranks, or `--steps-per-sweep`; the tests check this bit for bit.
  - RK4 stages stay within half a cell of the step start, so one ghost cell
    covers every stencil. Two are used for margin.
- **Ghost cells.** These are filled once at startup:
  - same-level and periodic neighbors via `FillPatchSingleLevel` and
    `FillPatchTwoLevels`;
  - coarse/fine boundaries with `cell_bilinear_interp`;
  - first-order extrapolation across non-periodic domain faces.
- **Recording.** Each recorded point carries `(seed, direction, step)`.
  Records are gathered on the I/O rank, sorted by that key, and joined into
  polylines, so the result is independent of task order.
- **Memory.** All levels of the 8 traced components are held in memory, spread
  across ranks (`--dry-run` reports the total). Gathered output is capped by
  `--max-gather-gb`. Increase `--output-every` if you hit the cap.

Face-centered B (`pltNNNNN/fc_vars/`) is not used yet. Interpolating it
component-wise along each face normal would keep `∇·B = 0` inside cells. That
is planned as a follow-up.
