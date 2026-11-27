#!/usr/bin/env python3
"""Demonstrate capped AMR timestep selection for advection and diffusion.

The examples use a one-dimensional scalar equation with explicit advection and
diffusion operators,

    u_t + a u_x = D u_xx,

on a nested AMR hierarchy.  The per-level stable timestep is

    tau[l] = min(cfl_adv * dx[l] / |a|,
                 cfl_diff * dx[l]**2 / D).

The AMR scheduler then chooses the largest stable coarse timestep and the
smallest stable integer subcycling factors, with

    nsubsteps[l] <= ref_ratio[l - 1]**2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class LevelLimit:
    level: int
    dx: float
    advection_dt: float
    diffusion_dt: float

    @property
    def tau(self) -> float:
        return min(self.advection_dt, self.diffusion_dt)

    @property
    def limiter(self) -> str:
        if self.advection_dt < self.diffusion_dt:
            return "advection"
        if self.diffusion_dt < self.advection_dt:
            return "diffusion"
        return "tie"


@dataclass(frozen=True)
class Schedule:
    dt0: float
    dt: List[float]
    nsubsteps: List[int]
    caps: List[int]
    need: List[float]
    coarse_limiter_level: int


@dataclass(frozen=True)
class Scenario:
    name: str
    speed: float
    diffusivity: float
    expected_limiters: Sequence[str]


def ceil_with_tolerance(value: float, rel_tol: float = 1.0e-12, abs_tol: float = 1.0e-14) -> int:
    """Return ceil(value), but do not round up values that are already integral."""
    if value <= 0.0:
        return 0

    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=rel_tol, abs_tol=abs_tol):
        return int(nearest)
    return math.ceil(value)


def compute_level_limits(
    *,
    dx0: float,
    ref_ratios: Sequence[int],
    speed: float,
    diffusivity: float,
    cfl_adv: float,
    cfl_diff: float,
) -> List[LevelLimit]:
    """Compute advection, diffusion, and combined timestep limits on all levels."""
    if dx0 <= 0.0:
        raise ValueError("dx0 must be positive.")
    if speed <= 0.0:
        raise ValueError("speed must be positive for this demo.")
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive for this demo.")
    if cfl_adv <= 0.0 or cfl_diff <= 0.0:
        raise ValueError("CFL numbers must be positive.")
    if any(r < 1 for r in ref_ratios):
        raise ValueError("refinement ratios must be positive integers.")

    limits: List[LevelLimit] = []
    dx = dx0
    for level in range(len(ref_ratios) + 1):
        limits.append(
            LevelLimit(
                level=level,
                dx=dx,
                advection_dt=cfl_adv * dx / abs(speed),
                diffusion_dt=cfl_diff * dx * dx / diffusivity,
            )
        )
        if level < len(ref_ratios):
            dx /= ref_ratios[level]

    return limits


def select_capped_amr_schedule(
    tau: Sequence[float],
    ref_ratios: Sequence[int],
    *,
    max_dt: float = math.inf,
    constant_dt: float | None = None,
) -> Schedule:
    """Select dt0 and nsubsteps with nsubsteps[l] <= ref_ratio[l - 1]**2."""
    nlevels = len(tau)
    if nlevels == 0:
        raise ValueError("At least one AMR level is required.")
    if len(ref_ratios) != nlevels - 1:
        raise ValueError("ref_ratios must contain one entry between each adjacent level.")
    if any(limit <= 0.0 or not math.isfinite(limit) for limit in tau):
        raise ValueError("All timestep limits must be positive finite values.")
    if max_dt <= 0.0:
        raise ValueError("max_dt must be positive.")

    caps = [1] + [ratio * ratio for ratio in ref_ratios]

    feasible_dt0 = min(tau[0], max_dt)
    coarse_limiter_level = 0
    cumulative_cap = 1
    for level in range(1, nlevels):
        cumulative_cap *= caps[level]
        candidate = cumulative_cap * tau[level]
        if candidate < feasible_dt0:
            feasible_dt0 = candidate
            coarse_limiter_level = level

    if constant_dt is not None:
        if constant_dt <= 0.0:
            raise ValueError("constant_dt must be positive.")
        if constant_dt > feasible_dt0 and not math.isclose(constant_dt, feasible_dt0, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError(
                f"constant_dt={constant_dt:.16e} exceeds the largest feasible stable dt0={feasible_dt0:.16e}."
            )
        dt0 = constant_dt
    else:
        dt0 = feasible_dt0

    need = [1.0] * nlevels
    need[-1] = dt0 / tau[-1]
    for level in range(nlevels - 2, 0, -1):
        need[level] = max(dt0 / tau[level], need[level + 1] / caps[level + 1])

    nsubsteps = [1] * nlevels
    dt = [dt0] + [math.nan] * (nlevels - 1)
    cumulative_product = 1
    for level in range(1, nlevels):
        factor = max(1, ceil_with_tolerance(need[level] / cumulative_product))
        if factor > caps[level]:
            raise RuntimeError(
                f"Internal error: level {level} needs {factor} substeps, exceeding cap {caps[level]}."
            )
        nsubsteps[level] = factor
        cumulative_product *= factor
        dt[level] = dt0 / cumulative_product

    for level, (dt_level, tau_level) in enumerate(zip(dt, tau)):
        if dt_level > tau_level and not math.isclose(dt_level, tau_level, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise RuntimeError(f"Internal error: level {level} has unstable dt={dt_level} > tau={tau_level}.")

    return Schedule(
        dt0=dt0,
        dt=dt,
        nsubsteps=nsubsteps,
        caps=caps,
        need=need,
        coarse_limiter_level=coarse_limiter_level,
    )


def format_float(value: float) -> str:
    return f"{value:.6e}"


def print_table(limits: Sequence[LevelLimit], schedule: Schedule) -> None:
    header = (
        "lev  dx            dt_adv        dt_diff       limiter    tau           "
        "cap  nsub  dt_level      dt/tau"
    )
    print(header)
    print("-" * len(header))
    for limit, cap, nsub, dt_level in zip(limits, schedule.caps, schedule.nsubsteps, schedule.dt):
        print(
            f"{limit.level:3d}  "
            f"{format_float(limit.dx):>12}  "
            f"{format_float(limit.advection_dt):>12}  "
            f"{format_float(limit.diffusion_dt):>12}  "
            f"{limit.limiter:<9}  "
            f"{format_float(limit.tau):>12}  "
            f"{cap:3d}  "
            f"{nsub:4d}  "
            f"{format_float(dt_level):>12}  "
            f"{dt_level / limit.tau:7.4f}"
        )


def assert_limiters(name: str, observed: Iterable[str], expected: Sequence[str]) -> None:
    observed_list = list(observed)
    if observed_list != list(expected):
        raise AssertionError(f"{name}: expected limiters {list(expected)}, got {observed_list}")


def run_scenario(
    scenario: Scenario,
    *,
    dx0: float,
    ref_ratios: Sequence[int],
    cfl_adv: float,
    cfl_diff: float,
) -> None:
    limits = compute_level_limits(
        dx0=dx0,
        ref_ratios=ref_ratios,
        speed=scenario.speed,
        diffusivity=scenario.diffusivity,
        cfl_adv=cfl_adv,
        cfl_diff=cfl_diff,
    )
    assert_limiters(scenario.name, (limit.limiter for limit in limits), scenario.expected_limiters)

    schedule = select_capped_amr_schedule([limit.tau for limit in limits], ref_ratios)

    print(f"\n{scenario.name}")
    print(f"speed = {scenario.speed:g}, diffusivity = {scenario.diffusivity:g}")
    print(f"dt0 = {format_float(schedule.dt0)}; coarse dt bound set by level {schedule.coarse_limiter_level}")
    print_table(limits, schedule)


def main() -> int:
    dx0 = 1.0
    ref_ratios = [2, 2, 2]
    cfl_adv = 0.5
    cfl_diff = 0.45

    scenarios = [
        Scenario(
            name="Case 1: every level is advection-limited",
            speed=1.0,
            diffusivity=0.02,
            expected_limiters=["advection", "advection", "advection", "advection"],
        ),
        Scenario(
            name="Case 2: every level is diffusion-limited",
            speed=1.0,
            diffusivity=2.0,
            expected_limiters=["diffusion", "diffusion", "diffusion", "diffusion"],
        ),
        Scenario(
            name="Case 3: coarse levels are advection-limited, fine levels are diffusion-limited",
            speed=1.0,
            diffusivity=0.3,
            expected_limiters=["advection", "advection", "diffusion", "diffusion"],
        ),
    ]

    print("Capped AMR timestep demo for u_t + a u_x = D u_xx")
    print(f"dx0 = {dx0:g}, ref_ratios = {ref_ratios}, cfl_adv = {cfl_adv:g}, cfl_diff = {cfl_diff:g}")
    print("Subcycle cap: nsubsteps[l] <= ref_ratio[l - 1]^2")

    for scenario in scenarios:
        run_scenario(scenario, dx0=dx0, ref_ratios=ref_ratios, cfl_adv=cfl_adv, cfl_diff=cfl_diff)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
