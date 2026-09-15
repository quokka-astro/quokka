#!/usr/bin/env python3
"""Average velocity spectra from periodic, uniform-grid Turbulence plotfiles.

FFT normalization is forward (1/Ncells); E_v(k) is the shell SUM of
0.5 * |FFT(v - volume_mean(v))|**2 over all three velocity components.
Thus summing ALL shells recovers 0.5 * <|v - <v>|**2> (Parseval).
No density weighting is applied. Shell width is one fundamental box mode.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quokka-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "quokka-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt


def spectrum(ds):
    dims = np.asarray(ds.domain_dimensions)
    if ds.index.max_level != 0 or not np.all(dims == dims[0]):
        raise ValueError("A uniform cubic grid is required")
    if not np.allclose(ds.domain_width, ds.domain_width[0]):
        raise ValueError("A cubic domain is required")
    grid = ds.covering_grid(0, ds.domain_left_edge, dims)
    rho = np.asarray(grid["boxlib", "gasDensity"])
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("Invalid density")
    modes = np.meshgrid(*(np.fft.fftfreq(n) * n for n in dims), indexing="ij")
    shells = np.floor(np.sqrt(sum(k * k for k in modes)) + 0.5).astype(int)
    power = np.zeros(tuple(dims))
    variance = 0.0
    mean_velocity = []
    for component in "xyz":
        v = np.asarray(grid["boxlib", f"{component}-GasMomentum"]) / rho
        mean_velocity.append(float(v.mean()))
        v = v - v.mean()
        variance += float(0.5 * np.mean(v * v))
        power += 0.5 * np.abs(np.fft.fftn(v, norm="forward")) ** 2
    energy = np.bincount(shells.ravel(), weights=power.ravel())
    np.testing.assert_allclose(energy.sum(), variance, rtol=2e-12, atol=1e-16)
    if not np.all(np.isfinite(energy)):
        raise ValueError("Invalid spectrum")
    return energy, {"time": float(ds.current_time), "half_velocity_variance": variance,
                    "rms_velocity_fluctuation": float(np.sqrt(2 * variance)),
                    "mean_velocity": mean_velocity,
                    "parseval_relative_error": float(abs(energy.sum() - variance) / variance),
                    "resolution": dims.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--eddy-time", type=float, default=5.0)
    parser.add_argument("--methods", nargs="+", choices=("PLM", "PPM", "xPPM"), default=["PLM", "PPM", "xPPM"])
    parser.add_argument("--output-stem", default="velocity_spectra")
    args = parser.parse_args()
    yt.set_log_level(40)
    targets = np.arange(31, 41) / 10 * args.eddy_time
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    summary = {}
    arrays = {}
    colors = {"PLM": "#0072B2", "PPM": "#D55E00", "xPPM": "#009E73"}
    for name in args.methods:
        color = colors[name]
        candidates = []
        for path in sorted((args.run_dir / name).glob("plt[0-9]*")):
            if not (path / "Header").is_file():
                continue
            ds = yt.load(str(path))
            t = float(ds.current_time)
            if targets[0] - 1e-8 <= t <= targets[-1] + 1e-8:
                candidates.append((t, path))
        chosen = []
        records = []
        spectra = []
        for target in targets:
            if not candidates:
                raise RuntimeError(f"{name}: no averaging-window outputs")
            t, path = min(candidates, key=lambda item: abs(item[0] - target))
            if abs(t - target) > 0.01 * args.eddy_time or path in chosen:
                raise RuntimeError(f"{name}: missing unique snapshot near t={target}")
            chosen.append(path)
            energy, record = spectrum(yt.load(str(path)))
            record.update(plotfile=str(path), target_time=float(target))
            spectra.append(energy)
            records.append(record)
        spectra = np.stack(spectra)
        mean = spectra.mean(axis=0)
        std = spectra.std(axis=0, ddof=1)
        # Exclude shells touching the per-axis Nyquist limit from the figure.
        # Full spectra, including corner modes, remain in the NPZ artifact.
        k = np.arange(1, records[0]["resolution"][0] // 2)
        ax.loglog(k, mean[k], color=color, label=name, linewidth=2)
        ax.fill_between(k, np.maximum(mean[k] - std[k], 1e-30), mean[k] + std[k],
                        color=color, alpha=0.12, linewidth=0)
        arrays.update({f"{name}_snapshots": spectra, f"{name}_mean": mean,
                       f"{name}_std": std, f"{name}_times": np.array([r["time"] for r in records])})
        summary[name] = records
    kref = np.arange(4, 21)
    ref = arrays[f"{args.methods[-1]}_mean"][5] * (kref / 5.0) ** (-5.0 / 3.0)
    ax.loglog(kref, ref, "--", color="0.4", linewidth=1, label=r"$k^{-5/3}$ reference")
    ax.axvspan(2, 3, color="0.5", alpha=0.12)
    ax.set(xlabel=r"Box wavenumber $kL/(2\pi)$", ylabel=r"Velocity spectrum $E_v(k)$",
           title=r"Isothermal turbulence: $64^3$, target $\mathcal{M}_{\rm rms}=0.2$" + "\n"
           + r"Mean over $3.1$–$4.0\,t_{\rm eddy}$; bands: snapshot standard deviation",
           xlim=(1, 32))
    ax.legend(frameon=False)
    ax.grid(which="major", alpha=0.15)
    for suffix in ("png", "pdf"):
        fig.savefig(args.run_dir / f"{args.output_stem}.{suffix}", dpi=200)
    np.savez_compressed(args.run_dir / f"{args.output_stem}.npz", **arrays)
    (args.run_dir / f"{args.output_stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({name: {"snapshots": len(records), "first_time": records[0]["time"],
                           "last_time": records[-1]["time"],
                           "mean_rms_velocity": float(np.mean([r["rms_velocity_fluctuation"] for r in records]))}
                      for name, records in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
