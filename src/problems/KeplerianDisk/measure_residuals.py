#!/usr/bin/env python3
"""Run and analyze KeplerianDisk one-step residuals (2D, single MPI rank).

Requires NumPy and Matplotlib. Raw data and command/log provenance remain in --output.
All rates are in code units. Positive radial momentum is outward.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def fields(data):
    x, y, rho, mass, mx, my, defect = data.T
    r = np.hypot(x, y)
    return r, np.arctan2(y, x), {
        "density_rate": mass,
        "radial_acceleration": (x * mx + y * my) / (r * rho),
        "normalized_Lz_rate": (x * my - y * mx) / rho,
        "torque_defect": defect,
    }


def profile(data):
    r, phi, values = fields(data)
    h = np.diff(np.unique(data[:, 0])).min()
    # Fit jointly to avoid leakage of a constant into m=4 from Cartesian sampling.
    basis = np.array([np.ones_like(phi), np.cos(4 * phi), np.sin(4 * phi),
                      np.cos(8 * phi), np.sin(8 * phi)]).T
    rows = []
    for lower in np.arange(0.5, 1.5 - h, 2 * h):
        mask = (r >= lower) & (r < lower + 2 * h)
        row = {"radius": float(lower + h), "count": int(mask.sum())}
        for name, value in values.items():
            coeff = np.linalg.lstsq(basis[mask], value[mask], rcond=None)[0]
            row[name] = {"mean": float(value[mask].mean()),
                         "a0": float(coeff[0]), "cos4": float(coeff[1]),
                         "sin4": float(coeff[2]), "amp4": float(np.hypot(*coeff[1:3])),
                         "rms": float(np.sqrt(np.mean(value[mask] ** 2)))}
        rows.append(row)
    return rows


def summarize(data):
    r, _, values = fields(data)
    h = np.diff(np.unique(data[:, 0])).min()
    ring = (r > 0.7) & (r < 1.3)
    result = {name + "_rms": float(np.sqrt(np.mean(value[ring] ** 2)))
              for name, value in values.items()}
    # Explicit rotation check on the square, cell-centered mesh.
    n = len(np.unique(data[:, 0]))
    order = np.lexsort((data[:, 0], data[:, 1]))
    result["rotation90_max_error"] = {
        name: float(np.max(np.abs(value[order].reshape(n, n) - np.rot90(value[order].reshape(n, n)))))
        for name, value in values.items()
    }
    # A common annulus at both resolutions for comparing harmonic coefficients.
    peak = (r >= 0.9375) & (r < 1.0625)
    phi = np.arctan2(data[:, 1], data[:, 0])
    basis = np.array([np.ones_like(phi), np.cos(4 * phi), np.sin(4 * phi),
                      np.cos(8 * phi), np.sin(8 * phi)]).T
    result["peak_annulus"] = {}
    for name, value in values.items():
        coeff = np.linalg.lstsq(basis[peak], value[peak], rcond=None)[0]
        result["peak_annulus"][name] = {"mean": float(value[peak].mean()), "a0": float(coeff[0]),
                                         "amp4": float(np.hypot(*coeff[1:3]))}
    result["mass_rate_r_lt_core"] = float(data[r < 0.25, 3].sum() * h**2)
    result["mass_rate_r_lt_1"] = float(data[r < 1, 3].sum() * h**2)
    result["Lz_rate_r_lt_1"] = float((data[:, 0] * data[:, 5] - data[:, 1] * data[:, 4])[r < 1].sum() * h**2)
    result["torque_defect_r_lt_1"] = float(data[r < 1, 6].sum() * h**2)
    result["outward_face_Lz_flux_r_lt_1"] = result["torque_defect_r_lt_1"] - result["Lz_rate_r_lt_1"]
    return result


def self_check():
    x, y = np.meshgrid((np.arange(128) + 0.5) / 32 - 2,
                       (np.arange(128) + 0.5) / 32 - 2)
    x, y = x.ravel(), y.ravel()
    phi = np.arctan2(y, x)
    rho = np.ones_like(x)
    q = 2 + 3 * np.cos(4 * phi) - 4 * np.sin(4 * phi)
    manufactured = np.column_stack([x, y, rho, q, x * 0, y * 0, x * 0])
    for row in profile(manufactured):
        np.testing.assert_allclose([row["density_rate"][key] for key in ("a0", "cos4", "sin4", "amp4")],
                                   [2, 3, -4, 5], atol=1.e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--input", type=Path, default=Path("inputs/KeplerianDisk.toml"))
    parser.add_argument("--output", type=Path, default=Path("build/2d/keplerian-residuals"))
    args = parser.parse_args()
    self_check()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = [(n, k, dt) for n in (128, 256) for k in (0., 0.1)
               for dt in (0.002 * 128 / n, 0.001 * 128 / n, 0.0005 * 128 / n, 0.00025 * 128 / n)]
    results = []
    for n, k, dt in configs:
        label = f"n{n}-k{k:g}-dt{dt:g}"
        directory = output / label
        if args.executable:
            directory.mkdir(exist_ok=True)
            cmd = [str(args.executable.resolve()), str(args.input.resolve()),
                   f"amr.n_cell={n} {n}", "max_timesteps=1", f"constant_dt={dt}",
                   f"stop_time={dt}", "hydro.reconstruction_order=5", "hydro.use_dual_energy=1",
                   f"hydro.artificial_viscosity_coefficient={k}", "disk.residual_prefix=residual",
                   "plotfile_interval=-1", "checkpoint_interval=-1", "suppress_output=1"]
            (directory / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            with (directory / "run.log").open("w") as log:
                subprocess.run(cmd, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
        spatial = np.loadtxt(directory / "residual-spatial.txt")
        step = np.loadtxt(directory / "residual-step.txt")
        np.testing.assert_array_equal(spatial[:, :3], step[:, :3])
        assert np.isfinite(spatial).all() and np.isfinite(step).all()
        history = np.loadtxt(directory / "keplerian_disk_history.txt")
        assert history.shape == (2, 9), history.shape
        np.testing.assert_allclose(history[-1, 0], dt, rtol=1.e-12)
        # Independent check against existing whole-domain and inner-region history.
        h = 4 / n
        mass_rate = step[:, 3].sum() * h**2
        Lz_rate = (step[:, 0] * step[:, 5] - step[:, 1] * step[:, 4]).sum() * h**2
        np.testing.assert_allclose([mass_rate, Lz_rate], (history[1, 1:3] - history[0, 1:3]) / dt, atol=2.e-9, rtol=1.e-6)
        ss, ts = summarize(spatial), summarize(step)
        np.testing.assert_allclose([ts["mass_rate_r_lt_1"], ts["Lz_rate_r_lt_1"]],
                                   (history[1, 7:9] - history[0, 7:9]) / dt, atol=2.e-9, rtol=1.e-6)
        # No face data accompany the full step; do not misreport its zero placeholder.
        for key in ("torque_defect_r_lt_1", "outward_face_Lz_flux_r_lt_1", "torque_defect_rms"):
            ts.pop(key)
        for key in ("rotation90_max_error", "peak_annulus"):
            ts[key].pop("torque_defect")
        delta = step.copy()
        delta[:, 3:6] -= spatial[:, 3:6]
        delta[:, 6] = 0
        difference = summarize(delta)
        for key in ("torque_defect_r_lt_1", "outward_face_Lz_flux_r_lt_1", "torque_defect_rms"):
            difference.pop(key)
        for key in ("rotation90_max_error", "peak_annulus"):
            difference[key].pop("torque_defect")
        record = {"n": n, "K": k, "dt": dt, "spatial": ss, "step": ts,
                  "step_minus_spatial": difference}
        results.append(record)
        for kind, data in (("spatial", spatial), ("step", step)):
            (directory / f"{kind}-profile.json").write_text(json.dumps(profile(data), indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    keys = ("density_rate", "radial_acceleration", "normalized_Lz_rate")
    chosen = [rec for rec in results if rec["n"] == 128 and rec["K"] == 0.1]
    for rec in chosen:
        label = f'n128-k0.1-dt{rec["dt"]:g}'
        rows = json.loads((output / label / "step-profile.json").read_text())
        for i, key in enumerate(keys):
            for j, stat in enumerate(("a0", "amp4")):
                axes[i, j].plot([row["radius"] for row in rows], [row[key][stat] for row in rows], label=f'dt={rec["dt"]:g}')
    rows = json.loads((output / 'n128-k0.1-dt0.002' / 'spatial-profile.json').read_text())
    for i, key in enumerate(keys):
        for j, stat in enumerate(("a0", "amp4")):
            ax = axes[i, j]
            ax.plot([row["radius"] for row in rows], [row[key][stat] for row in rows], 'k--', label='spatial limit')
            ax.set_ylabel(key.replace('_', ' '))
            ax.grid(alpha=0.25)
    axes[0, 0].set_title('Axisymmetric coefficient (signed)')
    axes[0, 1].set_title('Fourfold amplitude')
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Radius')
    for ax in axes[2]:
        ax.set_ylabel('Angular momentum density rate / rho0')
    fig.suptitle('Kepler disc equilibrium residuals: 128², xPPM, K=0.1')
    fig.tight_layout()
    fig.savefig(output / "residual-profiles.png", dpi=170)
    fig.savefig(output / "residual-profiles.pdf")
    plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
