"""
makeplots_slicez.py — diagnostics/plots for Quokka native z-slice plotfiles

This is a sibling of makeplots_fast.py, adapted for plotfiles produced by
Quokka's built-in slice output (set in the inputs file via, e.g.):

    quokka.slice_z.field_names = Bphi x-BField y-BField z-BField \
                                  x-GasMomentum y-GasMomentum z-GasMomentum \
                                  plasma_beta gasDensity

Those plotfiles are already a single-cell-thick 2D slab at the z-midplane
(domain_dimensions[2] == 1), so this script does NOT do any of the things
makeplots_fast.py needed for full 3D plotfiles:

  - No yt.SlicePlot / FRB machinery — the midplane data is read directly
    with ds.covering_grid() at the finest level present in the slice file
    and squeezed to a plain 2D array. This is exact cell data (no
    fixed-resolution-buffer interpolation) and is essentially free, since
    the slice file is tiny compared to the parent 3D plotfile.
  - No chunked/MPI-distributed volume averaging, no divB, no rotation-curve
    circular_velocity panel — none of that is meaningful for (or dumped
    into) a single z=const slice.
  - Only the XY (midplane) view exists — there is no XZ/YZ edge-on slice
    to show, because the slice file never had that data to begin with.

What IS kept from makeplots_fast.py, translated to the 2D case:
  - The velocity derived fields (x/y/z-Velocity, velocity_mag) built from
    GasMomentum/gasDensity, same as before.
  - The Bphi_reconstructed derived field (from x-BField, y-BField, x, y),
    for the analytic-vs-reconstructed Bphi consistency check.
  - The same FIXED color-scale constants, so slice-file plots are directly
    comparable to each other AND to the corresponding midplane panels
    produced by makeplots_fast.py.
  - The dead-zone/mask-width geometry (from tests/input/Aphi_2d_meta.txt)
    used to mask the unphysical near-axis plasma-beta values.
  - The turbulent-residual velocity-fluctuation panel (analytic
    rotation-curve baseline subtracted exactly, not via a binned mean).
  - The "hydro" switch that skips every B-field-dependent diagnostic —
    now checked against BOTH the path/run-tag substring AND the actual
    field list of the slice file itself (a slice file can simply omit the
    B-field entries from quokka.slice_z.field_names even for an MHD run,
    so checking the field list is the more reliable signal here).
  - Output filenames tagged with an optional run tag + the plotfile
    timestep, same convention as makeplots_fast.py (prefixed "slicez_" so
    outputs from the two scripts never collide in the same directory).

Batch/MPI model (different from makeplots_fast.py):
  Since each slice file is small and self-contained, there is nothing to
  gain from distributing the work WITHIN one file across ranks. Instead,
  when given several slice plotfiles at once (e.g. a whole directory of
  timesteps), this script distributes the FILES round-robin across MPI
  ranks — each rank independently loads, processes, and saves its own
  subset of timesteps. Every rank's output files have distinct names
  (tagged by timestep), so there is no rank-0 gating needed for saving;
  only the startup banner is rank-0-only.

Launch:
    # one slice file, no MPI needed:
    python3 makeplots_slicez.py plots/mhddisk_vel0_slicez0000001 --tag vel0

    # many slice files, distributed round-robin across ranks:
    srun -N 1 -n 16 -c 2 python3 makeplots_slicez.py \
        plots/mhddisk_vel0_slicez* --tag vel0

    # optional zoom on a specific (x, y) location [kpc], e.g. an SN site:
    python3 makeplots_slicez.py plots/mhddisk_vel0_slicez0000001 \
        --tag vel0 --sn-xy 1.2 -0.4 --sn-width 0.5
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import yt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1
is_root = (rank == 0)

yt.set_log_level("warning")

kpc = 3.085677581e21


def rprint(*args, **kwargs):
    if is_root:
        print(*args, **kwargs)


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Plot diagnostics from Quokka z-slice plotfiles.")
parser.add_argument("plotfiles", nargs="+",
                     help="One or more slice_z plotfile paths (shell-expanded globs are fine).")
parser.add_argument("--tag", default="", help="Run tag folded into every output filename.")
parser.add_argument("--sn-xy", nargs=2, type=float, default=None, metavar=("X_KPC", "Y_KPC"),
                     help="Optional (x, y) center [kpc] for a zoomed velocity panel, e.g. an SN site.")
parser.add_argument("--sn-width", type=float, default=0.5,
                     help="Zoom width [kpc] around --sn-xy (default 0.5 kpc = 500 pc).")
args = parser.parse_args()

RUN_TAG = args.tag
do_sn_zoom = args.sn_xy is not None
if do_sn_zoom:
    SN_X_KPC, SN_Y_KPC = args.sn_xy
    SN_WIDTH_KPC = args.sn_width

# Round-robin distribute the plotfiles across ranks.
my_plotfiles = args.plotfiles[rank::size]
rprint(f"Run tag: {RUN_TAG!r}" if RUN_TAG else "Run tag: (none given)")
rprint(f"{len(args.plotfiles)} slice plotfile(s) total, distributed across {size} rank(s).")

# ============================================================
# FIXED AXES — same constants as makeplots_fast.py, for direct comparability
# ============================================================
BFIELD_COMP_VMAX = 5e-6
BFIELD_MAG_VMIN = 0.0
BFIELD_MAG_VMAX = 5e-6

BPHI_XY_VMAX = 1e-6
BPHI_LINTHRESH_FRACTION = 1e-2

DENS_LOG_VMIN = -28.0
DENS_LOG_VMAX = -20.0

VMAX_VEL = 2e7
LINTHRESH = 1e5

# ============================================================
# Dead-zone / mask geometry (from the seed-field table metadata)
# ============================================================
dead_zone_kpc = None
mask_width_kpc = None
try:
    meta = {}
    with open("tests/input/Aphi_2d_meta.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            meta[key.strip()] = val.split("#")[0].split("[")[0].strip()
    nR = int(meta.get("nR", meta.get("seed_nR")))
    Rmax_cm = float(meta.get("Rmax_cm", meta.get("seed_Rmax", meta.get("cyl_Rmax_cm"))))
    dR_table_kpc = Rmax_cm / nR / kpc
    dead_zone_kpc = 2.0 * dR_table_kpc
    mask_width_kpc = 4.0 * dead_zone_kpc
    rprint(f"  dead_zone_kpc  = {dead_zone_kpc:.4f} kpc")
    rprint(f"  mask_width_kpc = {mask_width_kpc:.4f} kpc")
except (FileNotFoundError, KeyError, TypeError, ValueError):
    rprint("  WARNING: could not read tests/input/Aphi_2d_meta.txt — "
           "plasma-beta plots will be shown unmasked.")


def tag(name, timestep_str):
    base, dot, ext = name.rpartition(".")
    suffix_parts = [p for p in (RUN_TAG, timestep_str) if p]
    suffix = "_".join(suffix_parts)
    if dot:
        return f"{base}_{suffix}.{ext}"
    return f"{name}_{suffix}"


def get_plane(ds, field):
    """
    Read a field's midplane data directly (no SlicePlot/FRB needed — the
    slice plotfile is already a single-cell-thick 2D slab). Returns a
    plain 2D numpy array oriented (row=y, col=x), matching imshow's
    default convention.
    """
    max_level = ds.index.max_level
    dims_full = ds.domain_dimensions * (2 ** max_level)
    cg = ds.covering_grid(level=max_level, left_edge=ds.domain_left_edge, dims=dims_full)
    data = cg[field].v
    # dims_full[2] should be 1 for a z-slice; squeeze it out.
    data2d = data[:, :, 0] if data.ndim == 3 else data
    return data2d.T  # (nx, ny) -> (row=y, col=x)


def process_one(plotfile_path):
    rprint(f"\n[rank {rank}] Loading {plotfile_path} ...")
    ds = yt.load(
        plotfile_path,
        units_override={"length_unit": (1.0, "cm"), "time_unit": (1.0, "s"), "mass_unit": (1.0, "g")},
    )

    field_names_present = {f[1] for f in ds.field_list}
    IS_HYDRO = (
        ("hydro" in plotfile_path.lower())
        or ("hydro" in RUN_TAG.lower())
        or ("x-BField" not in field_names_present)
    )
    if IS_HYDRO:
        print(f"[rank {rank}] {plotfile_path}: hydro/no-B-field slice -> "
              f"skipping B-field, plasma-beta, and Bphi diagnostics.")

    # Extracts the ####### digits from e.g. "slicez_8e6_plt0061351" -> "0061351".
    # Falls back to trailing digits, then to the first digit run, if a
    # plotfile is ever named without the "plt" convention.
    basename = os.path.basename(plotfile_path.rstrip("/"))
    m = re.search(r"plt(\d+)", basename)
    if m is None:
        m = re.search(r"(\d+)$", basename)
    if m is None:
        m = re.search(r"(\d+)", basename)
    timestep_str = m.group(1) if m else "unknown"

    width_cm = float(ds.domain_width[0].v)
    width_kpc = width_cm / kpc
    extent_kpc = [-width_kpc / 2, width_kpc / 2, -width_kpc / 2, width_kpc / 2]
    t_myr = float(ds.current_time.v) / 3.15576e13

    # ---- derived fields: velocity components + magnitude ----
    def _vel_comp(field, data, comp):
        p = data[("boxlib", f"{comp}-GasMomentum")]
        rho = data[("boxlib", "gasDensity")]
        return (p / rho) * data.ds.quan(1.0, "cm/s")

    for comp in ["x", "y", "z"]:
        ds.add_field(
            name=("boxlib", f"{comp}-Velocity"),
            function=lambda field, data, c=comp: _vel_comp(field, data, c),
            sampling_type="cell", units="cm/s",
        )

    def _vel_mag(field, data):
        vx = data[("boxlib", "x-Velocity")].v
        vy = data[("boxlib", "y-Velocity")].v
        vz = data[("boxlib", "z-Velocity")].v
        return np.sqrt(vx**2 + vy**2 + vz**2) * data.ds.quan(1.0, "cm/s")

    ds.add_field(name=("boxlib", "velocity_mag"), function=_vel_mag,
                 sampling_type="cell", units="cm/s")

    if not IS_HYDRO:
        def _Bphi_reconstructed(field, data):
            Bx = data[("boxlib", "x-BField")].v
            By = data[("boxlib", "y-BField")].v
            x = data[("index", "x")].v
            y = data[("index", "y")].v
            r2 = x**2 + y**2
            r = np.where(r2 > 0, np.sqrt(r2), 1.0)
            return (By * x - Bx * y) / r * data.ds.quan(1.0, "G")

        ds.add_field(name=("boxlib", "Bphi_reconstructed"), function=_Bphi_reconstructed,
                     sampling_type="cell", units="G")

    # ---- pull all midplane arrays we need, once ----
    dens = get_plane(ds, ("boxlib", "gasDensity"))
    vx = get_plane(ds, ("boxlib", "x-Velocity"))
    vy = get_plane(ds, ("boxlib", "y-Velocity"))
    vz = get_plane(ds, ("boxlib", "z-Velocity"))
    vmag = get_plane(ds, ("boxlib", "velocity_mag"))
    xgrid = get_plane(ds, ("index", "x")) / kpc
    ygrid = get_plane(ds, ("index", "y")) / kpc
    R_grid = np.sqrt(xgrid**2 + ygrid**2)

    if not IS_HYDRO:
        Bx = get_plane(ds, ("boxlib", "x-BField"))
        By = get_plane(ds, ("boxlib", "y-BField"))
        Bz = get_plane(ds, ("boxlib", "z-BField"))
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        beta = get_plane(ds, ("boxlib", "plasma_beta"))
        bphi_table = get_plane(ds, ("boxlib", "Bphi"))
        bphi_recon = get_plane(ds, ("boxlib", "Bphi_reconstructed"))

    # ============================================================
    # Density (midplane)
    # ============================================================
    log_dens = np.log10(np.where(dens > 0, dens, 1e-300))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(log_dens, origin="lower", extent=extent_kpc, cmap="viridis",
                    vmin=DENS_LOG_VMIN, vmax=DENS_LOG_VMAX, interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}(\rho\ [\mathrm{g\ cm^{-3}}])$")
    ax.set_title(f"gasDensity — midplane, t = {t_myr:.1f} Myr", fontsize=10)
    ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
    fig.tight_layout()
    fname = tag("density_slicez.png", timestep_str)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rank {rank}] Saved: {fname}")

    # ============================================================
    # B-field 4-panel (midplane only)
    # ============================================================
    if not IS_HYDRO:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        panels = [(Bx, r"$B_x$", -BFIELD_COMP_VMAX, BFIELD_COMP_VMAX, "RdBu_r"),
                  (By, r"$B_y$", -BFIELD_COMP_VMAX, BFIELD_COMP_VMAX, "RdBu_r"),
                  (Bz, r"$B_z$", -BFIELD_COMP_VMAX, BFIELD_COMP_VMAX, "RdBu_r"),
                  (Bmag, r"$|B|$", BFIELD_MAG_VMIN, BFIELD_MAG_VMAX, "inferno")]
        for ax, (data, label, vmin, vmax, cmap) in zip(axes, panels):
            im = ax.imshow(data, origin="lower", extent=extent_kpc, cmap=cmap,
                            vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
            plt.colorbar(im, ax=ax, label="G")
            ax.set_title(f"{label} — midplane", fontsize=10)
            ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
        fig.suptitle(f"Magnetic field (slice_z) — t = {t_myr:.1f} Myr", fontsize=13)
        fig.tight_layout()
        fname = tag("Bfield_4panel_slicez.png", timestep_str)
        fig.savefig(fname, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"[rank {rank}] Saved: {fname}")

        # ---- plasma beta (masked) ----
        log_beta = np.log10(np.where(beta > 0, beta, 1e-300))
        if mask_width_kpc is not None:
            mask = R_grid < mask_width_kpc
            log_beta = np.where(mask, np.nan, log_beta)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        im = ax.imshow(log_beta, origin="lower", extent=extent_kpc, cmap="magma",
                        vmin=-2, vmax=6, interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax, label=r"$\log_{10}(\beta)$")
        ax.set_title(f"Plasma beta — midplane, t = {t_myr:.1f} Myr", fontsize=10)
        ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
        if mask_width_kpc is not None:
            ax.add_patch(plt.Circle((0, 0), mask_width_kpc, color="white", fill=False, lw=0.8, ls="--"))
        fig.tight_layout()
        fname = tag("plasma_beta_slicez.png", timestep_str)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[rank {rank}] Saved: {fname}")

        # ---- Bphi comparison (analytic table vs grid-reconstructed) ----
        linthresh = BPHI_XY_VMAX * BPHI_LINTHRESH_FRACTION
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, label in zip(axes, [bphi_table, bphi_recon],
                                    ["Analytic Table Bphi", "Grid Reconstructed Bphi"]):
            im = ax.imshow(data, origin="lower", extent=extent_kpc, cmap="RdBu_r",
                            norm=SymLogNorm(linthresh=linthresh, vmin=-BPHI_XY_VMAX,
                                             vmax=BPHI_XY_VMAX, base=10),
                            interpolation="nearest", aspect="equal")
            plt.colorbar(im, ax=ax, label=r"$B_\phi$ [G]", format="%.1e")
            ax.set_title(f"{label} — midplane", fontsize=10)
            ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
        fig.suptitle(f"$B_\\phi$ comparison (slice_z) — t = {t_myr:.1f} Myr", fontsize=12)
        fig.tight_layout()
        fname = tag("Bphi_comparison_slicez.png", timestep_str)
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[rank {rank}] Saved: {fname}")

    # ============================================================
    # Velocity 4-panel (midplane)
    # ============================================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vel_panels = [(vx, r"$v_x$"), (vy, r"$v_y$"), (vz, r"$v_z$"), (vmag, r"$|v|$")]
    for ax, (data, label) in zip(axes, vel_panels):
        if label == r"$|v|$":
            norm = LogNorm(vmin=1e5, vmax=VMAX_VEL); cmap = "magma"
        else:
            norm = SymLogNorm(linthresh=LINTHRESH, linscale=1.0, vmin=-VMAX_VEL, vmax=VMAX_VEL, base=10)
            cmap = "RdBu_r"
        im = ax.imshow(data, origin="lower", extent=extent_kpc, cmap=cmap, norm=norm,
                        interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax, label="cm/s")
        ax.set_title(f"{label} — midplane", fontsize=10)
        ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
    fig.suptitle(f"Velocity (slice_z) — t = {t_myr:.1f} Myr", fontsize=13)
    fig.tight_layout()
    fname = tag("velocity_4panel_slicez.png", timestep_str)
    fig.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[rank {rank}] Saved: {fname}")

    # ============================================================
    # Optional zoom around an (x, y) site, e.g. an SN injection point
    # ============================================================
    if do_sn_zoom:
        nx, ny = vx.shape[1], vx.shape[0]
        dx_kpc = width_kpc / nx
        dy_kpc = width_kpc / ny
        cx = int((SN_X_KPC - extent_kpc[0]) / dx_kpc)
        cy = int((SN_Y_KPC - extent_kpc[2]) / dy_kpc)
        half_px_x = max(1, int(0.5 * SN_WIDTH_KPC / dx_kpc))
        half_px_y = max(1, int(0.5 * SN_WIDTH_KPC / dy_kpc))
        sl_x = slice(max(cx - half_px_x, 0), min(cx + half_px_x, nx))
        sl_y = slice(max(cy - half_px_y, 0), min(cy + half_px_y, ny))
        extent_zoom = [SN_X_KPC - half_px_x * dx_kpc, SN_X_KPC + half_px_x * dx_kpc,
                       SN_Y_KPC - half_px_y * dy_kpc, SN_Y_KPC + half_px_y * dy_kpc]

        zoom_panels = [(vx[sl_y, sl_x], r"$v_x$"), (vy[sl_y, sl_x], r"$v_y$"),
                       (vz[sl_y, sl_x], r"$v_z$"), (vmag[sl_y, sl_x], r"$|v|$")]
        comp_stack = np.concatenate([zoom_panels[0][0].ravel(), zoom_panels[1][0].ravel(),
                                      zoom_panels[2][0].ravel()])
        finite = comp_stack[np.isfinite(comp_stack)]
        zoom_vmax = max(np.percentile(np.abs(finite), 99.5), 1.0) if finite.size else 1e5
        zoom_linthresh = max(zoom_vmax * 1e-3, 1.0)
        mag_finite = zoom_panels[3][0][np.isfinite(zoom_panels[3][0]) & (zoom_panels[3][0] > 0)]
        mag_vmin = np.percentile(mag_finite, 1) if mag_finite.size else 1e3
        mag_vmax = np.percentile(mag_finite, 99.5) if mag_finite.size else zoom_vmax

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (data, label) in zip(axes, zoom_panels):
            if label == r"$|v|$":
                norm = LogNorm(vmin=max(mag_vmin, 1.0), vmax=max(mag_vmax, mag_vmin * 10))
                cmap = "magma"
            else:
                norm = SymLogNorm(linthresh=zoom_linthresh, linscale=1.0,
                                   vmin=-zoom_vmax, vmax=zoom_vmax, base=10)
                cmap = "RdBu_r"
            im = ax.imshow(data, origin="lower", extent=extent_zoom, cmap=cmap, norm=norm,
                            interpolation="nearest", aspect="equal")
            plt.colorbar(im, ax=ax, label="cm/s")
            ax.plot(SN_X_KPC, SN_Y_KPC, marker="+", color="lime", markersize=10, mew=1.5)
            ax.set_title(f"{label} — zoom", fontsize=10)
            ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
        fig.suptitle(f"Zoomed velocity around ({SN_X_KPC:.2f}, {SN_Y_KPC:.2f}) kpc, "
                     f"width={SN_WIDTH_KPC*1e3:.0f} pc — t = {t_myr:.3f} Myr", fontsize=12)
        fig.tight_layout()
        fname = tag("velocity_zoom_slicez.png", timestep_str)
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[rank {rank}] Saved: {fname}")

    # ============================================================
    # Velocity fluctuations (turbulent residual, midplane)
    # ============================================================
    R_safe = np.where(R_grid > 0, R_grid, 1.0)
    vR = (vx * xgrid + vy * ygrid) / R_safe
    vphi = (vy * xgrid - vx * ygrid) / R_safe

    Rc_kpc = 2.0
    Rc_cm = Rc_kpc * kpc
    cs_disk = 7.0e5
    Mc = float(ds.parameters.get("mhd_galaxy.Mc", 30.0))
    vc_cms = Mc * cs_disk
    R_grid_cm = R_grid * kpc
    vrot_analytic = vc_cms * R_grid_cm / np.sqrt(R_grid_cm**2 + Rc_cm**2)

    dvphi = vphi - vrot_analytic
    dvR = vR
    dvz = vz
    axis_mask = R_grid < (dead_zone_kpc if dead_zone_kpc is not None else 0.0)
    dvR = np.where(axis_mask, np.nan, dvR)
    dvphi = np.where(axis_mask, np.nan, dvphi)
    dvz = np.where(axis_mask, np.nan, dvz)

    R_bins = np.linspace(0.0, width_kpc / 2, 60)
    R_mid = 0.5 * (R_bins[:-1] + R_bins[1:])
    sigma_turb = np.zeros(len(R_mid))
    for i, (rlo, rhi) in enumerate(zip(R_bins[:-1], R_bins[1:])):
        m = (R_grid >= rlo) & (R_grid < rhi)
        if m.sum() > 4:
            sigma_turb[i] = np.sqrt(np.nanmean(dvphi[m]**2 + dvR[m]**2 + dvz[m]**2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fluct_vmax = np.nanpercentile(np.abs(np.concatenate([dvR.ravel(), dvphi.ravel(), dvz.ravel()])), 99)
    for ax, data, label in zip(axes.flat[:3], [dvR, dvphi, dvz],
                                [r"$\delta v_R$", r"$\delta v_\phi$", r"$\delta v_z$"]):
        im = ax.imshow(data, origin="lower", extent=extent_kpc, cmap="RdBu_r",
                        vmin=-fluct_vmax, vmax=fluct_vmax, interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax, label="cm/s")
        ax.set_title(f"{label} (turbulent residual)", fontsize=10)
        ax.set_xlabel("x [kpc]"); ax.set_ylabel("y [kpc]")
    ax4 = axes.flat[3]
    ax4.plot(R_mid, sigma_turb / 1e5, lw=2)
    ax4.set_xlabel("R [kpc]"); ax4.set_ylabel(r"$\sigma_{turb}$ [km/s]")
    ax4.set_title("Turbulent velocity dispersion vs R (midplane)")
    if dead_zone_kpc is not None:
        ax4.axvline(dead_zone_kpc, color="red", ls=":", lw=1, label=f"Dead zone ({dead_zone_kpc:.2f} kpc)")
        ax4.legend(fontsize=8)
    fig.suptitle(f"Velocity fluctuations (slice_z) — t = {t_myr:.1f} Myr", fontsize=13)
    fig.tight_layout()
    fname = tag("velocity_fluctuations_slicez.png", timestep_str)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rank {rank}] Saved: {fname}")

    ds.index.clear_all_data()


# ============================================================
# Main
# ============================================================
for pf in my_plotfiles:
    process_one(pf)

if comm is not None:
    comm.Barrier()
rprint("\nAll slice_z plots complete.")
