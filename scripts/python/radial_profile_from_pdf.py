import argparse
import csv
import math
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import numba as nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - optional accel
    HAVE_NUMBA = False

def float_if_possible(element: str):
    try:
        return float(element)
    except ValueError:
        return element


def read_header(filename: str) -> Dict[str, List]:
    header = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                tokens = line[1:].split()
                key = tokens[0][:-1]
                values = [float_if_possible(val) for val in tokens[1:]]
                header[key] = values
    return header


def read_inputs_param(filename: str, key: str):
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.split("#", 1)[0].strip()
                if not stripped:
                    continue
                if "=" not in stripped:
                    continue
                lhs, rhs = stripped.split("=", 1)
                if lhs.strip() == key:
                    return float(rhs.strip())
    except OSError:
        return None
    return None


def read_inputs_param_str(filename: str, key: str):
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.split("#", 1)[0].strip()
                if not stripped:
                    continue
                if "=" not in stripped:
                    continue
                lhs, rhs = stripped.split("=", 1)
                if lhs.strip() == key:
                    return rhs.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def bilinear_interp(r_grid, z_grid, values, r_val, z_val):
    r = float(np.clip(r_val, r_grid[0], r_grid[-1]))
    z = float(np.clip(z_val, z_grid[0], z_grid[-1]))
    i_r = int(np.searchsorted(r_grid, r, side="right")) - 1
    i_z = int(np.searchsorted(z_grid, z, side="right")) - 1
    i_r = max(0, min(i_r, len(r_grid) - 2))
    i_z = max(0, min(i_z, len(z_grid) - 2))
    r0 = r_grid[i_r]
    r1 = r_grid[i_r + 1]
    z0 = z_grid[i_z]
    z1 = z_grid[i_z + 1]
    fr = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)
    fz = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
    v00 = values[i_z, i_r]
    v10 = values[i_z, i_r + 1]
    v01 = values[i_z + 1, i_r]
    v11 = values[i_z + 1, i_r + 1]
    return (1 - fr) * (1 - fz) * v00 + fr * (1 - fz) * v10 + (1 - fr) * fz * v01 + fr * fz * v11


if HAVE_NUMBA:

    @nb.njit(cache=True)
    def interp1d_linear(x, xp, fp):
        if x <= xp[0]:
            return fp[0]
        if x >= xp[-1]:
            return fp[-1]
        idx = np.searchsorted(xp, x, side="right") - 1
        if idx < 0:
            idx = 0
        if idx > xp.size - 2:
            idx = xp.size - 2
        x0 = xp[idx]
        x1 = xp[idx + 1]
        f0 = fp[idx]
        f1 = fp[idx + 1]
        if x1 == x0:
            return f0
        t = (x - x0) / (x1 - x0)
        return f0 + t * (f1 - f0)

    @nb.njit(cache=True)
    def bilinear_interp_numba(r_grid, z_grid, values, r_val, z_val):
        r = r_val
        z = z_val
        if r < r_grid[0]:
            r = r_grid[0]
        if r > r_grid[-1]:
            r = r_grid[-1]
        if z < z_grid[0]:
            z = z_grid[0]
        if z > z_grid[-1]:
            z = z_grid[-1]
        i_r = np.searchsorted(r_grid, r, side="right") - 1
        i_z = np.searchsorted(z_grid, z, side="right") - 1
        if i_r < 0:
            i_r = 0
        if i_r > r_grid.size - 2:
            i_r = r_grid.size - 2
        if i_z < 0:
            i_z = 0
        if i_z > z_grid.size - 2:
            i_z = z_grid.size - 2
        r0 = r_grid[i_r]
        r1 = r_grid[i_r + 1]
        z0 = z_grid[i_z]
        z1 = z_grid[i_z + 1]
        fr = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)
        fz = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
        v00 = values[i_z, i_r]
        v10 = values[i_z, i_r + 1]
        v01 = values[i_z + 1, i_r]
        v11 = values[i_z + 1, i_r + 1]
        return (1.0 - fr) * (1.0 - fz) * v00 + fr * (1.0 - fz) * v10 + (1.0 - fr) * fz * v01 + fr * fz * v11

def disk_density_spherical_avg(radius_kpc, r_scale_kpc, z_scale_kpc, rho0_cgs):
    kpc_cm = 1.0e3 * 3.085677581e18
    r_cm = np.asarray(radius_kpc) * kpc_cm
    r_scale_cm = r_scale_kpc * kpc_cm
    z_scale_cm = z_scale_kpc * kpc_cm

    theta = np.linspace(0.0, math.pi, 512)
    sin_t = np.sin(theta)
    abs_cos_t = np.abs(np.cos(theta))
    r_cm_col = r_cm[:, None]
    rho = rho0_cgs * np.exp(-(r_cm_col * sin_t) / r_scale_cm) * np.exp(-(r_cm_col * abs_cos_t) / z_scale_cm)

    # Spherical average: (1/2) * ∫ rho(r,theta) sin(theta) dtheta
    integrand = rho * sin_t
    avg = 0.5 * np.trapz(integrand, theta, axis=1)
    return avg


def disk_pressure_midplane(radius_kpc, r_scale_kpc, rho0_cgs, t_disk_k):
    m_p = 1.67262192369e-24
    mu = 0.61
    rho_mid = rho0_cgs * np.exp(-np.asarray(radius_kpc) / r_scale_kpc)
    return rho_mid * t_disk_k / (mu * m_p)


def find_weight_column(columns: List[str]) -> str:
    for col in columns:
        if col.endswith("_sum"):
            return col
    raise ValueError("Could not find histogram weight column ending in '_sum'.")


def extract_bin_edges(data, var_name: str, n_bins: int):
    idx_col = data[var_name + "_idx"].astype(int)
    min_col = data[var_name + "_min"]
    max_col = data[var_name + "_max"]
    mins = np.full(n_bins, np.nan)
    maxs = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = idx_col == i
        if np.any(mask):
            mins[i] = min_col[mask][0]
            maxs[i] = max_col[mask][0]
    return mins, maxs


def bin_centers(mins, maxs, is_log: bool):
    if is_log:
        return np.sqrt(mins * maxs)
    return 0.5 * (mins + maxs)


def compute_profiles(weights, other_centers, radius_axis: int):
    if radius_axis == 0:
        sum_w = weights.sum(axis=1)
        mean_vals = np.full(weights.shape[0], np.nan)
        median_vals = np.full(weights.shape[0], np.nan)
        for i in range(weights.shape[0]):
            w = weights[i, :]
            total = sum_w[i]
            if total > 0.0:
                mean_vals[i] = np.sum(w * other_centers) / total
                cdf = np.cumsum(w)
                idx = int(np.searchsorted(cdf, 0.5 * total))
                if idx < len(other_centers):
                    median_vals[i] = other_centers[idx]
        return sum_w, mean_vals, median_vals

    sum_w = weights.sum(axis=0)
    mean_vals = np.full(weights.shape[1], np.nan)
    median_vals = np.full(weights.shape[1], np.nan)
    for i in range(weights.shape[1]):
        w = weights[:, i]
        total = sum_w[i]
        if total > 0.0:
            mean_vals[i] = np.sum(w * other_centers) / total
            cdf = np.cumsum(w)
            idx = int(np.searchsorted(cdf, 0.5 * total))
            if idx < len(other_centers):
                median_vals[i] = other_centers[idx]
    return sum_w, mean_vals, median_vals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute radial mean/median profiles from 2D DiagPDF outputs by marginalizing the PDF."
        )
    )
    parser.add_argument("filenames", nargs="+", help="DiagPDF *.dat files.")
    parser.add_argument(
        "--radius-name",
        default="radius_sph",
        help="Variable name to treat as radius (default: radius_sph).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_radial_profile.csv",
        help="Suffix to append to each output filename.",
    )
    parser.add_argument(
        "--profile-file",
        default="vcirc_with_halo.csv",
        help="Optional input profile file to overplot (default: vcirc_with_halo.csv).",
    )
    parser.add_argument(
        "--inputs-file",
        default="DiskGalaxy.in",
        help="Input file with disk parameters to compute analytical disk profile (default: DiskGalaxy.in).",
    )
    parser.add_argument(
        "--four-panel",
        action="store_true",
        help="Make a 2x2 panel plot from four input PDFs instead of per-file plots.",
    )
    args = parser.parse_args()

    unit_labels = {
        "radius_sph": "kpc",
        "gasDensity": "g cm^-3",
        "temperature": "K",
        "bfield_strength": "microgauss",
        "radial_velocity": "km s^-1",
        "pressure": "K cm^-3",
    }

    profile_data = None
    try:
        prof = np.loadtxt(args.profile_file, comments="#")
        if prof.ndim == 1:
            prof = prof.reshape(1, -1)
        if prof.shape[1] >= 5:
            profile_data = {
                "radius": prof[:, 0],
                "vcirc": prof[:, 1],
                "rho": prof[:, 2],
                "velr": prof[:, 3],
                "temp": prof[:, 4],
            }
    except OSError:
        profile_data = None

    disk_profile = None
    disk_mass = read_inputs_param(args.inputs_file, "agora_galaxy.disk_gas_mass_Msun")
    disk_rscale = read_inputs_param(args.inputs_file, "agora_galaxy.disk_Rscale_kpc")
    disk_zscale = read_inputs_param(args.inputs_file, "agora_galaxy.disk_zscale_kpc")
    disk_temp = read_inputs_param(args.inputs_file, "agora_galaxy.disk_temperature")

    if (
        disk_mass is not None
        and disk_rscale is not None
        and disk_zscale is not None
        and disk_temp is not None
        and disk_mass > 0.0
        and disk_rscale > 0.0
        and disk_zscale > 0.0
        and disk_temp > 0.0
    ):
        M_solar = 1.98847e33
        kpc_cm = 1.0e3 * 3.085677581e18
        rho0 = (disk_mass * M_solar) / (4.0 * math.pi * (disk_rscale * kpc_cm) ** 2 * (disk_zscale * kpc_cm))
        disk_profile = {
            "rho0": rho0,
            "r_scale": disk_rscale,
            "z_scale": disk_zscale,
            "temp": disk_temp,
        }

    def format_label(name: str) -> str:
        label = name
        if label in unit_labels:
            label = f"{label} [{unit_labels[label]}]"
        return label

    def load_profile(filename: str):
        header = read_header(filename)
        var_names = header.get("variables", [])
        if len(var_names) != 2:
            raise ValueError(f"{filename}: expected 2D histogram, found variables={var_names}")
        if args.radius_name not in var_names:
            raise ValueError(f"{filename}: radius variable '{args.radius_name}' not found in {var_names}")

        is_log = [bool(v) for v in header.get("is_log_spaced", [0, 0])]
        data = np.genfromtxt(filename, names=True, skip_header=4)
        columns = list(data.dtype.names)
        weight_col = find_weight_column(columns)

        xvar, yvar = var_names
        x_idx = data[xvar + "_idx"].astype(int)
        y_idx = data[yvar + "_idx"].astype(int)
        nx = int(x_idx.max()) + 1
        ny = int(y_idx.max()) + 1

        weights = np.zeros((nx, ny))
        weights[x_idx, y_idx] = data[weight_col]

        x_mins, x_maxs = extract_bin_edges(data, xvar, nx)
        y_mins, y_maxs = extract_bin_edges(data, yvar, ny)
        x_centers = bin_centers(x_mins, x_maxs, is_log[0])
        y_centers = bin_centers(y_mins, y_maxs, is_log[1])

        if args.radius_name == xvar:
            radius_mins, radius_maxs = x_mins, x_maxs
            radius_centers = x_centers
            other_name = yvar
            other_centers = y_centers
            radius_axis = 0
        else:
            radius_mins, radius_maxs = y_mins, y_maxs
            radius_centers = y_centers
            other_name = xvar
            other_centers = x_centers
            radius_axis = 1

        sum_w, mean_vals, median_vals = compute_profiles(weights, other_centers, radius_axis)
        return {
            "header": header,
            "radius_min": radius_mins,
            "radius_max": radius_maxs,
            "radius_center": radius_centers,
            "weight_col": weight_col,
            "other_name": other_name,
            "mean": mean_vals,
            "median": median_vals,
            "sum_w": sum_w,
        }

    def plot_profile(ax, prof):
        r = prof["radius_center"]
        y = prof["mean"]
        ymed = prof["median"]
        other = prof["other_name"]
        r_min = prof["radius_min"][0]

        valid = np.isfinite(r) & np.isfinite(y)
        if np.any(valid) and np.any(y[valid] <= 0.0):
            ax.semilogx(r, y, label="mean")
            ax.semilogx(r, ymed, label="median")
        else:
            ax.loglog(r, y, label="mean")
            ax.loglog(r, ymed, label="median")

        if profile_data is not None:
            if other == "gasDensity":
                ax.loglog(profile_data["radius"], profile_data["rho"], label="input rho", linestyle="--")
            elif other == "temperature":
                ax.loglog(profile_data["radius"], profile_data["temp"], label="input T", linestyle="--")
            elif other == "radial_velocity":
                ax.semilogx(profile_data["radius"], -profile_data["velr"] / 1.0e5, label="input v_r", linestyle="--")
            elif other == "pressure":
                m_p = 1.67262192369e-24
                mu = 0.61
                pressure = profile_data["rho"] * profile_data["temp"] / (mu * m_p)
                ax.loglog(profile_data["radius"], pressure, label="input P/k_B", linestyle="--")

        if other in ("gasDensity", "pressure"):
            y_floor_candidates = []
            for vals in (y, ymed):
                vals = vals[np.isfinite(vals)]
                vals = vals[vals > 0.0]
                if vals.size > 0:
                    y_floor_candidates.append(np.min(vals))
            if profile_data is not None:
                vals = profile_data["rho"] if other == "gasDensity" else None
                if other == "pressure":
                    if profile_data is not None:
                        m_p = 1.67262192369e-24
                        mu = 0.61
                        vals = profile_data["rho"] * profile_data["temp"] / (mu * m_p)
                if vals is not None:
                    vals = vals[np.isfinite(vals)]
                    vals = vals[vals > 0.0]
                    if vals.size > 0:
                        y_floor_candidates.append(np.min(vals))
            if y_floor_candidates:
                ax.set_ylim(bottom=min(y_floor_candidates))

        if other == "gasDensity" and disk_profile is not None:
            analytic_rho = disk_density_spherical_avg(
                r,
                disk_profile["r_scale"],
                disk_profile["z_scale"],
                disk_profile["rho0"],
            )
            ax.loglog(r, analytic_rho, label="analytic disk rho", linestyle=":")

        if other == "pressure" and disk_profile is not None:
            analytic_p_mid = disk_pressure_midplane(
                r,
                disk_profile["r_scale"],
                disk_profile["rho0"],
                disk_profile["temp"],
            )
            ax.loglog(r, analytic_p_mid, label="disk midplane P/k_B", linestyle="-.")

        ax.set_xlabel(format_label(args.radius_name))
        ax.set_ylabel(format_label(other))
        ax.set_xlim(left=r_min)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()

    if args.four_panel:
        if len(args.filenames) != 4:
            raise ValueError("Four-panel mode requires exactly 4 input files.")
        profiles = [load_profile(f) for f in args.filenames]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, prof in zip(axes.flat, profiles):
            plot_profile(ax, prof)
        fig.tight_layout()
        fig.savefig("radial_profiles_4panel.png", dpi=200)
        plt.close(fig)
        return

    for filename in args.filenames:
        prof = load_profile(filename)

        header = read_header(filename)
        var_names = header.get("variables", [])
        if len(var_names) != 2:
            raise ValueError(f"{filename}: expected 2D histogram, found variables={var_names}")
        if args.radius_name not in var_names:
            raise ValueError(f"{filename}: radius variable '{args.radius_name}' not found in {var_names}")

        is_log = [bool(v) for v in header.get("is_log_spaced", [0, 0])]
        data = np.genfromtxt(filename, names=True, skip_header=4)
        columns = list(data.dtype.names)
        weight_col = find_weight_column(columns)

        xvar, yvar = var_names
        x_idx = data[xvar + "_idx"].astype(int)
        y_idx = data[yvar + "_idx"].astype(int)
        nx = int(x_idx.max()) + 1
        ny = int(y_idx.max()) + 1

        weights = np.zeros((nx, ny))
        weights[x_idx, y_idx] = data[weight_col]

        x_mins, x_maxs = extract_bin_edges(data, xvar, nx)
        y_mins, y_maxs = extract_bin_edges(data, yvar, ny)
        x_centers = bin_centers(x_mins, x_maxs, is_log[0])
        y_centers = bin_centers(y_mins, y_maxs, is_log[1])

        if args.radius_name == xvar:
            radius_mins, radius_maxs = x_mins, x_maxs
            radius_centers = x_centers
            other_name = yvar
            other_centers = y_centers
            radius_axis = 0
        else:
            radius_mins, radius_maxs = y_mins, y_maxs
            radius_centers = y_centers
            other_name = xvar
            other_centers = x_centers
            radius_axis = 1

        sum_w, mean_vals, median_vals = compute_profiles(weights, other_centers, radius_axis)

        out_name = filename + args.output_suffix
        cycle = header.get("cycle", [math.nan])[0]
        time = header.get("time", [math.nan])[0]

        with open(out_name, "w", newline="", encoding="utf-8") as f:
            f.write(f"# cycle: {cycle}\n")
            f.write(f"# time: {time}\n")
            writer = csv.writer(f)
            writer.writerow(
                [
                    "radius_min",
                    "radius_max",
                    "radius_center",
                    weight_col,
                    f"{other_name}_mean",
                    f"{other_name}_median",
                ]
            )
            for i in range(len(radius_centers)):
                writer.writerow(
                    [
                        radius_mins[i],
                        radius_maxs[i],
                        radius_centers[i],
                        sum_w[i],
                        mean_vals[i],
                        median_vals[i],
                    ]
                )

        plot_name = filename + args.output_suffix + ".png"
        fig, ax = plt.subplots()
        plot_profile(ax, prof)
        fig.tight_layout()
        fig.savefig(plot_name, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
