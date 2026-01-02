#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_inputs_param(filename: Path, key: str):
    try:
        with filename.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.split("#", 1)[0].strip()
                if not stripped or "=" not in stripped:
                    continue
                lhs, rhs = stripped.split("=", 1)
                if lhs.strip() == key:
                    return float(rhs.strip())
    except OSError:
        return None
    return None


def read_halo_profile(path: Path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError("Halo profile must have at least 5 columns (R, vcirc, rho, velr, T).")
    return data[:, 0], data[:, 2], data[:, 4]


def bisection_root(func, lo, hi, tol=1.0e-8, max_iter=200):
    f_lo = func(lo)
    f_hi = func(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return None
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if not np.isfinite(f_mid):
            return None
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def main():
    parser = argparse.ArgumentParser(
        description="Compute the P_disk == P_halo surface and a smooth blend for DiskGalaxy ICs."
    )
    parser.add_argument("--inputs", type=Path, default=Path("DiskGalaxy.in"))
    parser.add_argument("--halo-profile", type=Path, default=Path("vcirc_with_halo.csv"))
    parser.add_argument("--output-surface", type=Path, default=Path("disk_halo_pressure_surface.csv"))
    parser.add_argument("--output-blend", type=Path, default=Path("disk_halo_blend_grid.csv"))
    parser.add_argument("--nR", type=int, default=200)
    parser.add_argument("--nZ", type=int, default=200)
    parser.add_argument("--delta-dex", type=float, default=0.15, help="Smoothing width in dex for tanh blend.")
    parser.add_argument("--no-blend-grid", action="store_true")
    args = parser.parse_args()

    disk_mass = read_inputs_param(args.inputs, "agora_galaxy.disk_gas_mass_Msun")
    disk_rscale = read_inputs_param(args.inputs, "agora_galaxy.disk_Rscale_kpc")
    disk_zscale = read_inputs_param(args.inputs, "agora_galaxy.disk_zscale_kpc")
    disk_temp = read_inputs_param(args.inputs, "agora_galaxy.disk_temperature")
    if disk_mass is None or disk_rscale is None or disk_zscale is None or disk_temp is None:
        raise RuntimeError("Missing disk parameters in inputs file.")

    r_table_kpc, rho_halo, temp_halo = read_halo_profile(args.halo_profile)
    mu = 0.61
    m_p = 1.67262192369e-24
    halo_p_kb = rho_halo * temp_halo / (mu * m_p)

    r_min = np.min(r_table_kpc)
    r_max = np.max(r_table_kpc)
    z_max_kpc = 2.0 * disk_zscale

    M_solar = 1.98847e33
    kpc_cm = 1.0e3 * 3.085677581e18
    rho0 = (disk_mass * M_solar) / (4.0 * math.pi * (disk_rscale * kpc_cm) ** 2 * (disk_zscale * kpc_cm))

    def p_disk_kb(R_kpc, z_kpc):
        R = np.asarray(R_kpc)
        z = np.asarray(z_kpc)
        return rho0 * np.exp(-R / disk_rscale) * np.exp(-np.abs(z) / disk_zscale) * disk_temp / (mu * m_p)

    def p_halo_kb(r_kpc):
        return np.interp(np.asarray(r_kpc), r_table_kpc, halo_p_kb, left=halo_p_kb[0], right=halo_p_kb[-1])

    # Find outermost midplane intersection.
    f_vals = p_disk_kb(r_table_kpc, 0.0) - halo_p_kb
    sign = np.sign(f_vals)
    idx_changes = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if idx_changes.size == 0:
        raise RuntimeError("No midplane intersection found.")
    idx = idx_changes[-1]
    r0_lo = r_table_kpc[idx]
    r0_hi = r_table_kpc[idx + 1]

    def f_midplane(r_kpc):
        return p_disk_kb(r_kpc, 0.0) - p_halo_kb(r_kpc)

    r0 = bisection_root(f_midplane, r0_lo, r0_hi)
    if r0 is None:
        raise RuntimeError("Failed to bracket midplane intersection.")

    # Compute surface z(R) for R in [0, r0].
    r_vals = np.linspace(0.0, r0, args.nR)
    z_vals = np.full_like(r_vals, np.nan)
    p_disk_vals = np.full_like(r_vals, np.nan)
    p_halo_vals = np.full_like(r_vals, np.nan)

    for i, R in enumerate(r_vals):
        def f_z(z_kpc):
            return p_disk_kb(R, z_kpc) - p_halo_kb(math.sqrt(R * R + z_kpc * z_kpc))

        f0 = f_z(0.0)
        fz = f_z(z_max_kpc)
        if f0 == 0.0:
            z_vals[i] = 0.0
        elif f0 * fz > 0.0:
            z_vals[i] = np.nan
        else:
            z_root = bisection_root(f_z, 0.0, z_max_kpc)
            z_vals[i] = z_root if z_root is not None else np.nan
        r_sph = math.sqrt(R * R + (z_vals[i] if np.isfinite(z_vals[i]) else 0.0) ** 2)
        p_disk_vals[i] = p_disk_kb(R, z_vals[i]) if np.isfinite(z_vals[i]) else np.nan
        p_halo_vals[i] = p_halo_kb(r_sph)

    with args.output_surface.open("w", encoding="utf-8") as handle:
        handle.write("# R_kpc z_kpc r_kpc Pdisk_kB Phalo_kB log10_Pdisk_over_Phalo\n")
        for R, z, pd, ph in zip(r_vals, z_vals, p_disk_vals, p_halo_vals):
            if np.isfinite(z) and pd > 0.0 and ph > 0.0:
                log_ratio = math.log10(pd / ph)
            else:
                log_ratio = math.nan
            handle.write(f"{R:.6e} {z:.6e} {math.sqrt(R*R + (z if np.isfinite(z) else 0.0)**2):.6e} "
                         f"{pd:.6e} {ph:.6e} {log_ratio:.6e}\n")

    if not args.no_blend_grid:
        r_grid = np.linspace(0.0, 2.0 * r0, args.nR)
        z_grid = np.linspace(0.0, z_max_kpc, args.nZ)
        delta = args.delta_dex
        blend = np.zeros((args.nZ, args.nR))
        for iz, z in enumerate(z_grid):
            for ir, R in enumerate(r_grid):
                pd = p_disk_kb(R, z)
                ph = p_halo_kb(math.sqrt(R * R + z * z))
                if pd > 0.0 and ph > 0.0:
                    if delta <= 0.0:
                        f = 1.0 if pd >= ph else 0.0
                    else:
                        s = math.log10(pd / ph) / delta
                        f = 0.5 * (1.0 + math.tanh(s))
                else:
                    f = 0.0
                blend[iz, ir] = f

        with args.output_blend.open("w", encoding="utf-8") as handle:
            handle.write("2\n")
            handle.write(f"{args.nR}, {args.nZ}\n")
            handle.write("1\n")
            handle.write("R_kpc,z_kpc\n")
            handle.write("f_disk\n")
            handle.write("kpc,kpc\n")
            handle.write("1\n")
            handle.write(f"{r_grid[0]:.6e}, {z_grid[0]:.6e}\n")
            handle.write(f"{r_grid[-1]:.6e}, {z_grid[-1]:.6e}\n")
            handle.write("linear,linear\n")
            for iz in range(args.nZ):
                row = ", ".join(f"{blend[iz, ir]:.8e}" for ir in range(args.nR))
                handle.write(row + "\n")

        fig, ax = plt.subplots(figsize=(6, 5))
        mesh = ax.pcolormesh(r_grid, z_grid, blend, shading="auto", vmin=0.0, vmax=1.0)
        ax.contour(r_grid, z_grid, blend, levels=[0.5], colors="k", linewidths=1.0)
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel("z [kpc]")
        ax.set_title("Disk Blend Fraction")
        fig.colorbar(mesh, ax=ax, label="f_disk")
        fig.tight_layout()
        fig.savefig("disk_halo_blend_grid.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
