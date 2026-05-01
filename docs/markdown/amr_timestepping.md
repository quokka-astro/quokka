# AMR Timestep Selection

This note describes a timestep selection algorithm for AMR simulations that contain both hyperbolic and explicitly integrated parabolic operators. The goal is to choose the coarse timestep and the AMR subcycling factors so that every level uses the largest stable timestep allowed by its own physics, subject to a cap on the number of subcycles between adjacent levels.

The current AMR driver already stores one timestep per level in `dt_` and one integer subcycling factor per level in `nsubsteps`. The recursive advance only requires these subcycling factors to be positive integers. Therefore the timestep policy can choose `nsubsteps[lev]` dynamically instead of fixing it to the spatial refinement ratio.

## Per-level timestep limits

For each active AMR level `l`, compute a single-level stable timestep limit

```text
tau[l] = min(tau_hyperbolic[l],
             tau_parabolic[l],
             tau_particles[l],
             tau_other_explicit[l],
             change_max * old_dt[l])
```

where unavailable physics components report an infinite limit. The hyperbolic limit is the usual CFL condition. For an explicit scalar diffusion operator, a conservative parabolic limit has the form

```text
tau_parabolic = cfl_parabolic / max_cells(sum_d chi_d / dx_d^2)
```

where `chi_d` is the directional diffusivity and `cfl_parabolic` is chosen for the actual time integrator and stencil. More general parabolic operators should expose their own timestep estimator and contribute it to the minimum above.

These `tau[l]` values are local stability limits for a single update on level `l`. The AMR scheduler then chooses a synchronized set of level timesteps

```text
dt[l] = dt[0] / (nsubsteps[1] * ... * nsubsteps[l])
```

with `dt[l] <= tau[l]` on every level.

## Subcycle cap

Let

```text
N[l] = MaxRefRatio(l - 1)^2,  l >= 1
```

be the maximum allowed number of subcycles between level `l - 1` and level `l`. This cap matches the parabolic scaling of an explicit diffusion timestep under refinement. For example, a refinement ratio of 2 permits up to 4 fine steps per coarse step, and a refinement ratio of 4 permits up to 16 fine steps per coarse step.

Define the cumulative maximum subcycling product

```text
Q[0] = 1
Q[l] = N[1] * ... * N[l].
```

Then the finest possible timestep on level `l`, given a coarse timestep `dt0`, is `dt0 / Q[l]`. Stability therefore requires

```text
dt0 <= Q[l] * tau[l]
```

for every level.

## Algorithm

First compute all per-level timestep limits `tau[l]`. Then choose the largest feasible coarse timestep:

```text
dt0 = tau[0]
Q = 1

for l = 1..finest_level:
    Q *= N[l]
    dt0 = min(dt0, Q * tau[l])
```

Apply root-level user controls to `dt0`, including `max_dt`, `initial_dt` or `init_shrink`, `constant_dt`, and the final `stop_time` clamp. If `constant_dt` is set and exceeds the feasible bound above, the run should abort or emit a fatal diagnostic, because no amount of fine-level subcycling can make level 0 stable.

After the final `dt0` is known, compute the minimum cumulative subcycling product required at each level. Let

```text
need[l] = minimum required value of P[l] = nsubsteps[1] * ... * nsubsteps[l]
```

after level `l` has been assigned. The last level requires

```text
need[L] = dt0 / tau[L]
```

where `L = finest_level`. Moving from fine to coarse,

```text
need[l] = max(dt0 / tau[l], need[l + 1] / N[l + 1]).
```

The first term makes level `l` stable. The second term ensures that all finer levels can still be made stable without exceeding the remaining subcycle caps.

Finally assign the smallest feasible integer factors from coarse to fine:

```text
nsubsteps[0] = 1
dt[0] = dt0
P = 1

for l = 1..finest_level:
    nsubsteps[l] = ceil_with_roundoff_tolerance(need[l] / P)
    nsubsteps[l] = max(1, nsubsteps[l])
    assert(nsubsteps[l] <= N[l])

    P *= nsubsteps[l]
    dt[l] = dt0 / P
    assert(dt[l] <= tau[l] * tolerance)
```

The roundoff-tolerant ceiling should avoid turning values such as `4.000000000000001` into 5 because of floating-point noise. One practical rule is to round down to the nearest integer when the relative difference from that integer is smaller than a small tolerance, then otherwise use `ceil`.

If `do_subcycle == 0`, no dynamic factors are used:

```text
dt0 = min_l tau[l]
nsubsteps[l] = 1
dt[l] = dt0
```

with the same root-level user controls applied afterward, subject to stability checks.

## Behavior in limiting cases

For a purely hyperbolic operator with spatial refinement ratio `r`,

```text
tau[l] ~= tau[l - 1] / r.
```

The algorithm chooses `nsubsteps[l] ~= r`, so it recovers the usual AMR time-subcycling pattern.

For a purely parabolic explicit operator,

```text
tau[l] ~= tau[l - 1] / r^2.
```

The algorithm chooses `nsubsteps[l] ~= r^2`, which is exactly the maximum allowed by the cap. The coarse level can therefore use its natural coarse parabolic timestep while the fine level uses its natural fine parabolic timestep.

For mixed hyperbolic and parabolic operators, the optimal ratio can vary by level and by solution state. The same rule chooses the smallest integer factor that keeps the level stable and leaves enough remaining subcycling capacity for finer levels.

## Optimality proof

Consider an AMR hierarchy with levels `0..L`. Let

```text
P[0] = 1
P[l] = nsubsteps[1] * ... * nsubsteps[l].
```

An admissible synchronized timestep schedule must satisfy

```text
1 <= nsubsteps[l] <= N[l],
dt[l] = dt0 / P[l],
dt[l] <= tau[l]
```

for every level `l`.

### Largest possible coarse timestep

Because each subcycling factor is capped,

```text
P[l] <= Q[l] = N[1] * ... * N[l].
```

Stability on level `l` requires

```text
dt0 / P[l] <= tau[l],
```

so every admissible schedule must obey

```text
dt0 <= P[l] * tau[l] <= Q[l] * tau[l].
```

This bound holds for every level, including `l = 0` where `Q[0] = 1`. Therefore no stable schedule can use

```text
dt0 > min_l Q[l] * tau[l].
```

The algorithm sets `dt0` to this minimum, subject only to user-imposed root caps such as `max_dt`, `initial_dt`, and `stop_time`. Thus the selected coarse timestep is the largest stable coarse timestep compatible with the subcycle caps.

### Minimum stable subcycling factors

Fix the selected `dt0`. Define `need[l]` as the smallest cumulative subcycling product `P[l]` that allows levels `l..L` to be completed stably without violating any cap. At the finest level,

```text
need[L] = dt0 / tau[L],
```

because no finer cap remains. For an intermediate level `l`, two conditions are necessary:

1. Level `l` itself must be stable:

```text
P[l] >= dt0 / tau[l].
```

2. Level `l + 1` and all finer levels must remain feasible. Since `nsubsteps[l + 1] <= N[l + 1]`, the largest cumulative product reachable on the next level is `P[l] * N[l + 1]`. Therefore

```text
P[l] * N[l + 1] >= need[l + 1],
```

or

```text
P[l] >= need[l + 1] / N[l + 1].
```

Combining these necessary conditions gives

```text
need[l] >= max(dt0 / tau[l], need[l + 1] / N[l + 1]).
```

The algorithm defines `need[l]` to be exactly this maximum. This is also sufficient: if `P[l] >= need[l]`, then level `l` is stable, and there exists a capped choice of `nsubsteps[l + 1]` that reaches at least `need[l + 1]`; induction completes the argument for all finer levels.

During the forward pass, levels `0..l-1` have fixed cumulative product `P[l - 1]`. The algorithm chooses

```text
nsubsteps[l] = ceil(need[l] / P[l - 1]).
```

This is the smallest integer that makes `P[l] >= need[l]`. Any smaller integer would make level `l` unstable or make some finer level impossible to stabilize under the cap. Any larger integer would reduce `dt[l]` unnecessarily. Therefore, after the maximal feasible `dt0` is chosen, each level receives the largest stable timestep compatible with all finer-level stability requirements and the cap `nsubsteps[l] <= N[l]`.

## Implementation notes

- Compute and apply the final `stop_time` clamp before assigning fine-level subcycling factors. Shortening `dt0` can reduce the required number of subcycles.
- Recompute the dynamic `nsubsteps` after regridding if new levels are created or if the hierarchy changes before the next timestep.
- Keep diagnostic output that identifies which level and which operator set `tau[l]`; this is important for mixed hyperbolic/parabolic runs where the limiting operator may change over time.
- The algorithm assumes the parabolic operators are explicit. Implicit parabolic solves should report either no explicit parabolic restriction or a restriction based on accuracy rather than linear stability.
