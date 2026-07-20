"""
makeplots_fast.py — memory-efficient AMReX/Quokka plotfile diagnostics (CPU/MPI)
Fixes:
  - all_data() volume averages replaced with chunked z-slab accumulation,
    now explicitly distributed round-robin across MPI ranks (see the
    "Volume averages" section) instead of every rank redundantly processing
    every slab.
  - covering_grid per level freed immediately after use
  - yt.enable_parallelism() for the collective yt calls (SlicePlot,
    ProjectionPlot, covering_grid): these are called identically by every
    rank and yt internally distributes the underlying grid/chunk IO across
    all ranks in COMM_WORLD, gathering the result back to rank 0.
  - Figure creation / savefig / summary printing is guarded to rank 0 only,
    since only one rank should touch each output file. Ranks other than 0
    still participate in the collective yt calls above (required for yt's
    internal parallelism to work), they just skip drawing/saving.
  - All output filenames are tagged with an optional run tag (from the
    command line) followed by the plotfile timestep (plt#######), e.g.
    velocity_12panel_vel0_0000001.png
  - FIXED color-scale ranges (see FIXED AXES block below) so that plots from
    different timesteps / different runs are directly comparable
  - Slice planes are re-oriented and correctly labeled: XY, XZ, YZ each show
    the expected pair of axes (horizontal, vertical) instead of everything
    being mislabeled "x"/"y"
  - Added velocity 12-panel (vx, vy, vz, |v|; derived from momentum/density)
    and an optional zoomed velocity panel around a specific point (e.g. an
    SN injection site), plus a numeric net-momentum cross-check in that box.

Launch (CPU nodes, no GPU involved anywhere in this script):
    srun -N 1 -n 32 -c 4 python makeplots_fast.py plots/mhddisk_vel0_plt0000001 vel0
or, outside Slurm (e.g. interactive salloc / local testing):
    mpirun -n 8 python makeplots_fast.py plots/mhddisk_vel0_plt0000001 vel0
Single-rank / no MPI at all also works (size=1 falls back to a serial run):
    python3 makeplots_fast.py plots/mhddisk_vel0_plt0000001 vel0

Positional command-line arguments (all optional, positional, in order):
    1) plotfile_path — which plotfile to load (required to override the
       hardcoded default; passed straight to yt.load)
    2) run_tag       — optional free-form label folded into every output
       filename, right before the timestep number. It does NOT need to
       match anything in the plotfile path/name.
    3) SN_X_KPC, 4) SN_Y_KPC, 5) SN_Z_KPC — optional center (in kpc, box
       coordinates) for a zoomed velocity panel around a specific point
       (e.g. an SN injection site). All three must be given together for
       the zoom section to run; if omitted, the zoom section is skipped.
    6) SN_WIDTH_KPC — optional zoom width in kpc (default 0.5 kpc = 500 pc)

    velocity_12panel.png  -> velocity_12panel_vel0_0000001.png   (run_tag="vel0")
    velocity_12panel.png  -> velocity_12panel_0000001.png        (no run_tag given)
"""

import gc
import re
import sys
import time
import yt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
is_root = (rank == 0)

# ── MPI setup ─────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
is_root = (rank == 0)

t_start = time.time()

def rprint(*args, **kwargs):
    """Print only from rank 0 — use for anything that isn't per-rank debugging."""
    if is_root:
        print(*args, **kwargs)

# ── Command-line arguments ───────────────────────────────────────────────────
# Usage: python3 makeplots_fast.py <plotfile_path> <run_tag> [SN_X SN_Y SN_Z [SN_WIDTH]]
# e.g.:  python3 makeplots_fast.py plots/mhddisk_vel0_plt0000001 vel0
# The two are independent: <plotfile_path> says which plotfile to load,
# <run_tag> is just a label folded into output filenames. They don't have to
# match (e.g. the tag can be a short label even if the plotfile path is long
# or doesn't follow the mhddisk_<tag>_plt####### convention).
# All ranks are launched with identical argv (mpirun/srun broadcast the same
# command line to every rank), so reading sys.argv directly here is safe and
# requires no MPI broadcast.
RUN_TAG = sys.argv[2] if len(sys.argv) > 2 else ""
rprint(f"Run tag: {RUN_TAG!r}" if RUN_TAG else "Run tag: (none given)")

# Optional: zoom in on a specific SN injection site instead of (or in
# addition to) the full-domain plots above. All four are optional;
# if SN_X_KPC is not given, the zoomed section below is skipped
# entirely. Units: kpc for position and width.
SN_X_KPC     = float(sys.argv[3]) if len(sys.argv) > 3 else None
SN_Y_KPC     = float(sys.argv[4]) if len(sys.argv) > 4 else None
SN_Z_KPC     = float(sys.argv[5]) if len(sys.argv) > 5 else None
SN_WIDTH_KPC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5  # 500 pc default
do_sn_zoom = (SN_X_KPC is not None) and (SN_Y_KPC is not None) and (SN_Z_KPC is not None)
if do_sn_zoom:
    rprint(
        f"SN zoom target: ({SN_X_KPC:.3f}, {SN_Y_KPC:.3f}, {SN_Z_KPC:.3f}) kpc, "
        f"width={SN_WIDTH_KPC:.3f} kpc"
    )
else:
    rprint("SN zoom target: (none given, skipping)")

# yt's own parallelism distributes the grid/chunk IO of a single collective
# call (SlicePlot, ProjectionPlot, covering_grid) across every rank in
# COMM_WORLD, then gathers the assembled result back to rank 0. All ranks
# must call these functions together (SPMD) for this to work.
yt.enable_parallelism()

yt.set_log_level("warning")

# ============================================================
# FIXED AXES — locked color-scale ranges for cross-run comparison
# ============================================================
# 12-panel B field: components are diverging (+-), |B| is sequential (0..max)
BFIELD_COMP_VMAX = 2e-7          # Bx, By, Bz shown on [-VMAX, +VMAX]
BFIELD_MAG_VMIN  = 0.0           # |B| shown on [VMIN, VMAX]
BFIELD_MAG_VMAX  = 2e-7

# Bphi 2-way comparison: XY (midplane) panels get a wider range than the
# XZ / YZ (edge-on) panels, since Bphi is concentrated near the midplane.
BPHI_XY_VMAX    = 1e-7
BPHI_EDGE_VMAX  = 1e-8
BPHI_LINTHRESH_FRACTION = 1e-2   # linthresh = VMAX * this fraction

# Density slices: floor is the simulation's density floor, ceiling is
# roughly the disk midplane peak.
DENS_LOG_VMIN = -34.0            # log10(rho_floor [g/cm^3])
DENS_LOG_VMAX = -22.0            # log10(rho_max   [g/cm^3])

# div B slices: symmetric linear range (NOT log)
DIVB_VMAX = 5e-31

# Column-density (z-projection) panel: normalized to Sigma/Sigma_c0 on a log
# color scale from 10^-1 to 10^0, matching Arora et al. (2025, A&A 695, A155)
# Fig. 2/3 style, where Sigma_c0 = Sigma(R=0, t=0) is the INITIAL central
# surface density of the disc. Set SIGMA_C0_GCM2 explicitly (in g/cm^2) once
# you know it -- e.g. run this script once on your t=0 plotfile, read off the
# "Central column density (this snapshot)" value it prints, and hardcode that
# here -- so that every snapshot you plot afterwards is normalized by the SAME
# reference value as the galaxy evolves. If left as None, the script falls
# back to normalizing each snapshot by its OWN central pixel; that reproduces
# the single-frame appearance of Fig. 2/3 but is not consistent across
# different times of the same run (the norm would silently drift/shift
# snapshot to snapshot as the central density itself evolves).
SIGMA_C0_GCM2 = None
SIGMA_PROJ_VMIN = 1e-1   # Sigma/Sigma_c0 lower bound (matches Fig. 2/3 colorbar)
SIGMA_PROJ_VMAX = 1e0    # Sigma/Sigma_c0 upper bound (matches Fig. 2/3 colorbar)

# Velocity 12-panel + zoom: components use SymLogNorm about a linear core,
# magnitude uses LogNorm. VMAX_VEL/LINTHRESH set the FIXED full-domain
# scale; the zoomed SN panel below auto-scales instead (see that section).
VMAX_VEL = 2e7   # example 200 km/s in cm/s
LINTHRESH = 1e5

# ============================================================
# Load
# ============================================================
PLOTFILE = sys.argv[1] if len(sys.argv) > 1 else "BField_results/mhddisk3_8nodeb_plt0060000"
rprint(f"Plotfile: {PLOTFILE}")

ds = yt.load(
    PLOTFILE,
    units_override={
        "length_unit": (1.0, "cm"),
        "time_unit":   (1.0, "s"),
        "mass_unit":   (1.0, "g"),
    }
)

# ── Timestep tag for output filenames ─────────────────────────────────────────
# Extracts the ####### digits from e.g. "...plt0044000" -> "0044000"
_m = re.search(r"plt(\d+)", PLOTFILE)
timestep_str = _m.group(1) if _m else "unknown"
rprint(f"Timestep tag: {timestep_str}")

def tag(name):
    """Insert the run tag (if given) and timestep tag before the file extension.

    Examples (RUN_TAG="vel0", timestep_str="0000001"):
        tag("velocity_12panel.png") -> "velocity_12panel_vel0_0000001.png"
    Examples (RUN_TAG="", timestep_str="0000001"):
        tag("velocity_12panel.png") -> "velocity_12panel_0000001.png"
    """
    base, dot, ext = name.rpartition(".")
    suffix_parts = [p for p in (RUN_TAG, timestep_str) if p]
    suffix = "_".join(suffix_parts)
    if dot:
        return f"{base}_{suffix}.{ext}"
    return f"{name}_{suffix}"

kpc   = 3.085677581e21
t_myr = float(ds.current_time.v) / (3.15576e13)

rprint(f"Fields: {ds.field_list}")
rprint(f"Domain dims : {ds.domain_dimensions}")
rprint(f"Max AMR level: {ds.index.max_level}")
rprint(f"Time: {t_myr:.3f} Myr")
rprint(f"MPI ranks: {size}")

width_cm  = float(ds.domain_width[0].v)
width_kpc = width_cm / kpc
extent_kpc = [-width_kpc/2, width_kpc/2, -width_kpc/2, width_kpc/2]

# ── Aphi table metadata ───────────────────────────────────────────────────────
meta = {}
with open("Aphi_2d_meta.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            val_clean = val.split("#")[0].split("[")[0].strip()
            meta[key.strip()] = val_clean

nR       = int(meta.get("nR",     meta.get("seed_nR")))
nz       = int(meta.get("nz",     meta.get("seed_nz")))
Rmax_cm  = float(meta.get("Rmax_cm", meta.get("seed_Rmax", meta.get("cyl_Rmax_cm"))))
Lz_cm    = float(meta.get("Lz_cm",   meta.get("seed_Lz",   meta.get("cyl_Lz_cm"))))
dR_seed  = float(meta.get("dR_fine_cm", Rmax_cm / nR))
dz_seed  = float(meta.get("dz_fine_cm", Lz_cm  / nz))

rprint(f"Table shape       : {nR} x {nz}")
rprint(f"dR_table          = {dR_seed/3.086e21:.3f} kpc")
rprint(f"dz_table          = {dz_seed/3.086e18:.1f} pc")
rprint(f"Rmax              = {Rmax_cm/3.086e21:.1f} kpc")
rprint(f"Lz                = {Lz_cm/3.086e18:.0f} pc")

RES = 800

# ============================================================
# Helpers
# ============================================================
def get_slice(normal, field, res=RES):
    slc = yt.SlicePlot(ds, normal, field, center="c", width=(width_cm, "cm"))
    slc.set_buff_size(res)
    slc.render()
    return slc.frb[field].v


# yt's plot-axis convention for a slice normal to axis N shows a fixed
# cyclic pair of the remaining two axes as (horizontal, vertical):
#   normal='z' -> horizontal='x', vertical='y'   (XY)
#   normal='x' -> horizontal='y', vertical='z'   (YZ)
#   normal='y' -> horizontal='z', vertical='x'   (native), i.e. NOT "XZ"
# The frb array is returned in (row=vertical, col=horizontal) order, same
# as imshow's default. For normal='y' we transpose so that the panel is
# actually oriented X horizontal / Z vertical (matching the "XZ" label),
# instead of silently plotting Z horizontal / X vertical.
_PLANE_INFO = {
    "z": dict(xlabel="x [kpc]", ylabel="y [kpc]", title="XY (z=0)", transpose=False),
    "x": dict(xlabel="y [kpc]", ylabel="z [kpc]", title="YZ (x=0)", transpose=False),
    "y": dict(xlabel="x [kpc]", ylabel="z [kpc]", title="XZ (y=0)", transpose=True),
}

def get_slice_xy(normal, field, res=RES):
    """
    Like get_slice, but returns data re-oriented and labeled consistently:
      normal='z' -> XY plane, horizontal=x, vertical=y
      normal='x' -> YZ plane, horizontal=y, vertical=z
      normal='y' -> XZ plane, horizontal=x, vertical=z
    Returns (data, xlabel, ylabel, title_suffix).
    """
    info = _PLANE_INFO[normal]
    data = get_slice(normal, field, res=res)
    if info["transpose"]:
        data = data.T
    return data, info["xlabel"], info["ylabel"], info["title"]

def get_slice_zoom(normal, field, center_cm, width_cm_zoom, res=RES):
    slc = yt.SlicePlot(
        ds,
        normal,
        field,
        center=center_cm,
        width=(width_cm_zoom, "cm"),
    )
    slc.set_buff_size(res)
    slc.render()
    return slc.frb[field].v


def get_slice_xy_zoom(normal, field, center_cm, width_cm_zoom, res=RES):
    """
    Zoomed version of get_slice_xy().
    """
    info = _PLANE_INFO[normal]
    data = get_slice_zoom(
        normal,
        field,
        center_cm,
        width_cm_zoom,
        res=res,
    )
    if info["transpose"]:
        data = data.T
    return data, info["xlabel"], info["ylabel"], info["title"]

def get_proj(field, weight=None, res=RES):
    proj = yt.ProjectionPlot(ds, "z", field,
                             weight_field=weight, center="c",
                             width=(width_cm, "cm"))
    proj.set_buff_size(res)
    proj.render()
    return proj.frb[field].v


def make_norm(data, linthresh_fraction=1e-2):
    finite   = data[np.isfinite(data)]
    vmax     = np.percentile(np.abs(finite), 99)
    vmax     = vmax if vmax > 0 else 1.0
    linthresh = linthresh_fraction * vmax
    return SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)


def make_norm_pos(data):
    finite = data[np.isfinite(data) & (data > 0)]
    if len(finite) == 0:
        return LogNorm(vmin=1e-10, vmax=1.0)
    vmin = np.percentile(finite, 1)
    vmax = np.percentile(finite, 99)
    vmin = vmin if vmin > 0 else vmax * 1e-6
    return LogNorm(vmin=vmin, vmax=vmax)


# ============================================================
# Derived field: Bphi reconstructed
# ============================================================
def _Bphi_reconstructed(field, data):
    Bx = data[("boxlib", "x-BField")].v
    By = data[("boxlib", "y-BField")].v
    x  = data[("index",  "x")].v
    y  = data[("index",  "y")].v
    r2 = x**2 + y**2
    r  = np.where(r2 > 0, np.sqrt(r2), 1.0)
    return (By * x - Bx * y) / r * data.ds.quan(1.0, "G")

ds.add_field(
    name=("boxlib", "Bphi_reconstructed"),
    function=_Bphi_reconstructed,
    sampling_type="cell",
    units="G",
)

# ============================================================
# Derived fields: Velocity components and magnitude
# ============================================================
def _vel_comp(field, data, comp):
    # Retrieve as arrays
    p = data[("boxlib", f"{comp}-GasMomentum")]
    rho = data[("boxlib", "gasDensity")]

    # Perform division and explicitly multiply by the velocity unit
    # to ensure the result is a unyt_array with units 'cm/s'
    return (p / rho) * data.ds.quan(1.0, "cm/s")

for comp in ['x', 'y', 'z']:
    ds.add_field(
        name=("boxlib", f"{comp}-Velocity"),
        function=lambda field, data, c=comp: _vel_comp(field, data, c),
        sampling_type="cell",
        units="cm/s",
    )

def _vel_mag(field, data):
    # Retrieve the velocity component fields
    vx = data[("boxlib", "x-Velocity")]
    vy = data[("boxlib", "y-Velocity")]
    vz = data[("boxlib", "z-Velocity")]

    # Calculate magnitude: (v_x^2 + v_y^2 + v_z^2)^(0.5)
    # By extracting .v, we work with raw numpy arrays,
    # then attach the correct unit 'cm/s' at the end.
    mag_val = np.sqrt(vx.v**2 + vy.v**2 + vz.v**2)
    return mag_val * data.ds.quan(1.0, "cm/s")

ds.add_field(
    name=("boxlib", "velocity_mag"),
    function=_vel_mag,
    sampling_type="cell",
    units="cm/s",
)

# ============================================================
# B-field 12-panel
# ============================================================
rprint("\n--- B-field 12-panel ---")

normals     = ["z", "y", "x"]

# Read all slices up front — one open per (field, normal).
# get_slice_xy -> get_slice is a COLLECTIVE yt call: every rank must call it
# (yt distributes the grid IO internally), so this loop runs on all ranks.
slices = {}
plane_meta = {}
for field in ["x-BField", "y-BField", "z-BField"]:
    for normal in normals:
        rprint(f"  Slicing {field} {normal}...")
        data, xlabel, ylabel, title = get_slice_xy(normal, ("boxlib", field))
        slices[(field, normal)] = data
        plane_meta[normal] = (xlabel, ylabel, title)

for normal in normals:
    Bx = slices[("x-BField", normal)]
    By = slices[("y-BField", normal)]
    Bz = slices[("z-BField", normal)]
    slices[("Bmag", normal)] = np.sqrt(Bx**2 + By**2 + Bz**2)

# Only rank 0 draws and saves — every rank has identical `slices` data
# (yt gathers the collective read results to all ranks), but only one
# rank should touch the output file.
if is_root:
    row_fields = ["x-BField", "y-BField", "z-BField", "Bmag"]
    row_labels  = [r"$B_x$", r"$B_y$", r"$B_z$", r"$|B|$"]

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    for row, (field, rlabel) in enumerate(zip(row_fields, row_labels)):
        for col, normal in enumerate(normals):
            ax   = axes[row, col]
            data = slices[(field, normal)]
            xlabel, ylabel, plabel = plane_meta[normal]
            if row == 3:
                norm = plt.Normalize(vmin=BFIELD_MAG_VMIN, vmax=BFIELD_MAG_VMAX)
                cmap = "inferno"
            else:
                norm = plt.Normalize(vmin=-BFIELD_COMP_VMAX, vmax=BFIELD_COMP_VMAX)
                cmap = "RdBu_r"
            im = ax.pcolormesh(
                np.linspace(extent_kpc[0], extent_kpc[1], data.shape[1]+1),
                np.linspace(extent_kpc[2], extent_kpc[3], data.shape[0]+1),
                data, norm=norm, cmap=cmap, shading="flat", rasterized=True,
            )
            plt.colorbar(im, ax=ax, label="G")
            ax.set_title(f"{rlabel} — {plabel}", fontsize=9)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_aspect("equal")

    fig.suptitle(f"Magnetic field slices — t = {t_myr:.1f} Myr", fontsize=13)
    fig.tight_layout()
    fig.savefig(tag("Bfield_12panel_ytb.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    rprint(f"Saved: {tag('Bfield_12panel_ytb.png')}")

# ============================================================
# Density
# ============================================================
rprint("\n--- Density ---")

normals_3 = ["z", "y", "x"]

# Collect the collective slice data on all ranks first, draw on rank 0 only.
density_slice_data = {}
for normal in normals_3:
    data, xlabel, ylabel, title = get_slice_xy(normal, ("boxlib", "gasDensity"))
    density_slice_data[normal] = (data, xlabel, ylabel, title)

if is_root:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, normal in zip(axes, normals_3):
        data, xlabel, ylabel, title = density_slice_data[normal]
        log_data = np.log10(np.where(data > 0, data, 1e-300))
        im = ax.imshow(log_data, origin="lower", extent=extent_kpc,
                       cmap="viridis", vmin=DENS_LOG_VMIN, vmax=DENS_LOG_VMAX,
                       interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax,
                     label=r"$\log_{10}(\rho\ [\mathrm{g\ cm^{-3}}])$")
        ax.set_title(f"gasDensity — {title}", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    fig.suptitle(f"Density slices — t = {t_myr:.1f} Myr", fontsize=12)
    fig.tight_layout()
    fig.savefig(tag("density_slices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    rprint(f"Saved: {tag('density_slices.png')}")

rprint("  Column density projection...")
proj_data = get_proj(("boxlib", "gasDensity"))  # collective — all ranks call this
# get_proj integrates gasDensity along the sightline (no weight_field), so
# proj_data is already a surface density Sigma in g/cm^2 -- exactly the
# quantity plotted (normalized) in Fig. 2/3 of Arora et al. (2025).

if is_root:
    # Central pixel is our proxy for Sigma(R=0) at this snapshot. Reported
    # every run so you can read it off once from your t=0 plotfile and
    # hardcode it into SIGMA_C0_GCM2 above for consistent normalization
    # across snapshots/times.
    ny, nx = proj_data.shape
    sigma_center_this_snapshot = proj_data[ny // 2, nx // 2]
    rprint(f"  Central column density (this snapshot) = {sigma_center_this_snapshot:.3e} g/cm^2")

    if SIGMA_C0_GCM2 is not None:
        sigma_c0 = SIGMA_C0_GCM2
    else:
        sigma_c0 = sigma_center_this_snapshot
        rprint("  WARNING: SIGMA_C0_GCM2 not set -- normalizing by this "
               "snapshot's own central column density. Set SIGMA_C0_GCM2 "
               "to the t=0 value above for consistent normalization across "
               "different snapshots/times.")

    sigma_ratio = proj_data / sigma_c0

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im2 = ax2.imshow(
        sigma_ratio,
        origin="lower", extent=extent_kpc, cmap="viridis",
        norm=LogNorm(vmin=SIGMA_PROJ_VMIN, vmax=SIGMA_PROJ_VMAX),
        interpolation="nearest", aspect="equal",
    )
    plt.colorbar(im2, ax=ax2, label=r"$\Sigma/\Sigma_{c0}$")
    ax2.set_title(f"Column density (z-projection) — t = {t_myr:.1f} Myr")
    ax2.set_xlabel("x [kpc]"); ax2.set_ylabel("y [kpc]")
    fig2.tight_layout()
    fig2.savefig(tag("density_projection3.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    rprint(f"Saved: {tag('density_projection3.png')}")

# ============================================================
# Plasma beta slices  (locked — unchanged)
# ============================================================
rprint("\n--- Plasma beta (masked) ---")

dR_table_kpc  = Rmax_cm / nR / kpc
dead_zone_kpc = 2.0 * dR_table_kpc
mask_width_kpc = 4.0 * dead_zone_kpc

rprint(f"  dead_zone_kpc  = {dead_zone_kpc:.4f} kpc")
rprint(f"  mask_width_kpc = {mask_width_kpc:.4f} kpc")

# Collective reads on all ranks first
beta_slice_data = {}
for normal in normals_3:
    beta_slice_data[normal] = get_slice_xy(normal, ("boxlib", "plasma_beta"))

if is_root:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, normal in zip(axes, normals_3):
        data, xlabel, ylabel, title = beta_slice_data[normal]
        log_data = np.log10(np.where(data > 0, data, 1e-300))

        px = np.linspace(-width_kpc/2, width_kpc/2, data.shape[1])
        py = np.linspace(-width_kpc/2, width_kpc/2, data.shape[0])
        XX, YY = np.meshgrid(px, py)

        if normal == "z":
            mask = np.sqrt(XX**2 + YY**2) < mask_width_kpc
        else:
            mask = (np.abs(XX) < mask_width_kpc) | (np.abs(YY) < mask_width_kpc)

        log_data_masked = np.where(mask, np.nan, log_data)
        im = ax.imshow(log_data_masked, origin="lower", extent=extent_kpc,
                       cmap="magma", vmin=-2, vmax=6,
                       interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax, label=r"$\log_{10}(\beta)$")
        ax.set_title(f"Plasma beta — {title}", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

        if normal == "z":
            ax.add_patch(plt.Circle((0, 0), mask_width_kpc,
                                    color="white", fill=False, lw=0.8, ls="--"))
        else:
            for val in [-mask_width_kpc, mask_width_kpc]:
                ax.axvline(val, color="white", lw=0.8, ls="--")
                ax.axhline(val, color="white", lw=0.8, ls="--")

    fig.suptitle(f"Plasma beta slices (axis-masked) — t = {t_myr:.1f} Myr", fontsize=12)
    fig.tight_layout()
    fig.savefig(tag("plasma_beta_slices_masked.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    rprint(f"Saved: {tag('plasma_beta_slices_masked.png')}")

# ============================================================
# Volume-averaged plasma beta — CHUNKED Z-SLAB LOOP, MPI-DISTRIBUTED
# ============================================================
rprint("\n--- Volume averages (chunked z-slab, memory-efficient) ---")

max_level = ds.index.max_level
dims_full = ds.domain_dimensions * (2 ** max_level)
n_slabs   = max(128, size)
slab_nz   = max(1, dims_full[2] // n_slabs)

rho_transition = 1e-28
dead_zone_cm   = 2.0 * (Rmax_cm / nR)

acc = {k: {"sum_bv": 0.0, "sum_v": 0.0, "sum_brhov": 0.0, "sum_rhov": 0.0}
       for k in ("all", "disk", "disk_clean")}
acc_inner = {"sum_brhov": 0.0, "sum_rhov": 0.0}

rng            = np.random.default_rng(42 + rank)
RESERVOIR_N    = 1_000_000 # Reduced slightly to ensure memory safety
beta_reservoir = np.empty(RESERVOIR_N, dtype=np.float32)
reservoir_fill = 0
reservoir_full = False

LE = ds.domain_left_edge.v
dx = ds.domain_width.v / dims_full

rprint(f"  Finest-level dims: {dims_full}")
rprint(f"  n_slabs={n_slabs}, slab_nz={slab_nz} cells, {size} rank(s)")

my_slab_indices = list(range(rank, n_slabs, size))

for slab_idx in my_slab_indices:
    z0_cell = slab_idx * slab_nz
    z1_cell = min(z0_cell + slab_nz, dims_full[2])
    if z0_cell >= dims_full[2]: continue

    slab_left  = [LE[0], LE[1], LE[2] + z0_cell * dx[2]]
    slab_right = [LE[0] + dims_full[0]*dx[0], LE[1] + dims_full[1]*dx[1], LE[2] + z1_cell * dx[2]]
    region = ds.box(slab_left, slab_right)

    for chunk in region.chunks([("boxlib", "plasma_beta"), ("boxlib", "gasDensity"), 
                                ("index", "x"), ("index", "y"), ("index", "cell_volume")], "io"):
        
        beta  = chunk[("boxlib", "plasma_beta")].v.ravel()
        rho   = chunk[("boxlib", "gasDensity")].v.ravel()
        x     = chunk[("index", "x")].v.ravel()
        y     = chunk[("index", "y")].v.ravel()
        vol   = chunk[("index", "cell_volume")].v.ravel()
        R     = np.sqrt(x**2 + y**2)

        mask_disk = rho > rho_transition
        mask_disk_clean = mask_disk & (R > dead_zone_cm)
        mask_inner = mask_disk & (R < 5.0 * kpc)

        for key, mask in [("all", np.ones(len(beta), dtype=bool)), 
                          ("disk", mask_disk), ("disk_clean", mask_disk_clean)]:
            b, v, r = beta[mask], vol[mask], rho[mask]
            acc[key]["sum_bv"]    += np.sum(b * v)
            acc[key]["sum_v"]     += np.sum(v)
            acc[key]["sum_brhov"] += np.sum(b * r * v)
            acc[key]["sum_rhov"]  += np.sum(r * v)

        # Reservoir sampling
        disk_beta = beta[mask_disk].astype(np.float32)
        n_new = len(disk_beta)
        if n_new > 0:
            if not reservoir_full:
                space = RESERVOIR_N - reservoir_fill
                take = min(n_new, space)
                beta_reservoir[reservoir_fill:reservoir_fill + take] = disk_beta[:take]
                reservoir_fill += take
                if reservoir_fill == RESERVOIR_N: reservoir_full = True
            else:
                idx = rng.integers(0, RESERVOIR_N, size=n_new)
                beta_reservoir[idx] = disk_beta

        del beta, rho, x, y, vol, R, mask_disk, mask_disk_clean, mask_inner
    gc.collect()

comm.Barrier()
rprint("  All ranks finished. Reducing data...")

# GATHER AND PERCENTILES
all_reservoirs = comm.gather(beta_reservoir[:reservoir_fill], root=0)

if is_root:
    if all_reservoirs and any(len(arr) > 0 for arr in all_reservoirs):
        master = np.concatenate(all_reservoirs)
        rprint("  Disk beta percentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"    {p}th: {np.percentile(master, p):.3e}")
    else:
        rprint("  No disk cells found in any rank.")

# ============================================================
# div B — per AMR level (free covering_grid immediately)
# ============================================================
rprint("\n--- div B ---")

# covering_grid objects don't support .chunks() in this yt version
# (YTDataSelectorNotImplemented). divB is already per-level, non-interpolated
# data (differenced directly from the level's own face data), so iterate the
# real AMR grid patches at that level instead -- also naturally chunked
# (each patch is blocking-factor sized) and distributes cleanly across ranks.
for lev in range(ds.index.max_level + 1):
    level_grids = ds.index.select_grids(lev)
    my_grids = level_grids[rank::size]

    max_val = -np.inf
    sum_val = 0.0
    count = 0

    for g in my_grids:
        data = g[("boxlib", "divB")].v
        abs_data = np.abs(data)

        if abs_data.size > 0:
            max_val = max(max_val, np.max(abs_data))
        sum_val += np.sum(abs_data)
        count += data.size

        del data, abs_data

    # Gather results from all MPI ranks to rank 0
    global_max = comm.allreduce(max_val, op=MPI.MAX)
    global_sum = comm.allreduce(sum_val, op=MPI.SUM)
    global_count = comm.allreduce(count, op=MPI.SUM)

    if is_root:
        mean_val = global_sum / global_count if global_count > 0 else 0.0
        rprint(f"  Level {lev}:  max |divB| = {global_max:.3e}  mean |divB| = {mean_val:.3e}")

    gc.collect()

rprint("  Plotting raw divB slices...")
divB_panels = {}
divB_meta = {}
for normal in ["z", "y", "x"]:
    data, xlabel, ylabel, title = get_slice_xy(normal, ("boxlib", "divB"))
    divB_panels[normal] = data
    divB_meta[normal] = (xlabel, ylabel, title)

if is_root:
    all_divB_max = max(np.max(np.abs(d)) for d in divB_panels.values())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, normal in zip(axes, normals_3):
        data = divB_panels[normal]
        xlabel, ylabel, title = divB_meta[normal]
        im = ax.imshow(data, origin="lower", extent=extent_kpc,
                       cmap="RdBu_r", vmin=-DIVB_VMAX, vmax=DIVB_VMAX,
                       interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax, label=r"$\nabla\cdot B$")
        ax.set_title(f"div B — {title}", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    fig.suptitle(
        f"div B — t = {t_myr:.1f} Myr\nmax |divB| (this run) = {all_divB_max:.2e}  |  scale fixed to ±{DIVB_VMAX:.1e}",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(tag("divB_slices_raw.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    rprint(f"Saved: {tag('divB_slices_raw.png')}")

# ============================================================
# Bphi comparison (2-way)
# ============================================================
rprint("\n--- Bphi Comparison ---")

bphi_fields  = [("boxlib", "Bphi"), ("boxlib", "Bphi_reconstructed")]
bphi_labels  = ["Analytic Table Bphi", "Grid Reconstructed Bphi"]

# Collective reads on all ranks first
bphi_slice_data = {}
for field_tuple in bphi_fields:
    for normal in normals_3:
        bphi_slice_data[(field_tuple, normal)] = get_slice_xy(normal, field_tuple)

if is_root:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (field_tuple, rlabel) in enumerate(zip(bphi_fields, bphi_labels)):
        for col, normal in enumerate(normals_3):
            ax   = axes[row, col]
            data, xlabel, ylabel, title = bphi_slice_data[(field_tuple, normal)]

            vmax      = BPHI_XY_VMAX if normal == "z" else BPHI_EDGE_VMAX
            linthresh = vmax * BPHI_LINTHRESH_FRACTION

            im = ax.imshow(
                data, origin="lower", extent=extent_kpc, cmap="RdBu_r",
                norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10),
                interpolation="nearest", aspect="equal",
            )
            plt.colorbar(im, ax=ax, label=r"$B_\phi$ [G]", format="%.1e")
            ax.set_title(f"{rlabel} — {title}", fontsize=9)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    fig.suptitle(f"$B_\\phi$ Initial Condition Comparison — t = {t_myr:.1f} Myr",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(tag("Bphi_2way_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    rprint(f"Saved: {tag('Bphi_2way_comparison.png')}")

bphi_xy  = get_slice("z", ("boxlib", "Bphi"))  # collective — all ranks call this
bmag_xy  = slices[("Bmag", "z")]

if is_root:
    bphi_max = np.percentile(np.abs(bphi_xy[np.isfinite(bphi_xy)]), 99)
    bmag_rms = np.sqrt(np.mean(bmag_xy**2))
    bmag_max = np.percentile(bmag_xy[np.isfinite(bmag_xy)], 99)

    print("====================================================")
    print(f"Bphi 99th percentile (midplane)   : {bphi_max:.3e} G")
    print(f"|B|  rms             (midplane)   : {bmag_rms:.3e} G")
    print(f"|B|  99th percentile (midplane)   : {bmag_max:.3e} G")
    print(f"Bphi/|B|_rms                      : {bphi_max/bmag_rms:.3e}  (target: < 1e-2)")
    print(f"Bphi/|B|_max                      : {bphi_max/bmag_max:.3e}  (target: < 1e-2)")
    print("====================================================")

    n_cell    = ds.domain_dimensions[0]
    Lbox_cm   = float(ds.domain_width[0].v)
    dx_level0 = Lbox_cm / n_cell
    print(f"dR_table/dx_L0 = {dR_seed/dx_level0:.2f}  (want < 0.5)")
    print(f"dz_table/dx_L0 = {dz_seed/dx_level0:.2f}")

# ============================================================
# Velocity 12-panel
# ============================================================
rprint("\n--- Velocity 12-panel ---")

vel_fields = ["x-Velocity", "y-Velocity", "z-Velocity", "velocity_mag"]
vel_labels = [r"$v_x$", r"$v_y$", r"$v_z$", r"$|v|$"]
normals = ["z", "y", "x"]

# Collect slices (Collective call)
vel_slices = {}
for field in vel_fields:
    for normal in normals:
        rprint(f"  Slicing {field} {normal}...")
        data, xlabel, ylabel, title = get_slice_xy(normal, ("boxlib", field))
        vel_slices[(field, normal)] = (data, xlabel, ylabel, title)

if is_root:
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))

    for row, (field, rlabel) in enumerate(zip(vel_fields, vel_labels)):
        for col, normal in enumerate(normals):
            ax = axes[row, col]
            data, xlabel, ylabel, plabel = vel_slices[(field, normal)]

            if field == "velocity_mag":
                # For magnitude, use LogNorm (exclude near-zero values to avoid artifacts)
                norm = LogNorm(vmin=1e5, vmax=VMAX_VEL)
                cmap = "magma"
            else:
                # For components (vx, vy, vz), use SymLogNorm for positive/negative range
                norm = SymLogNorm(linthresh=LINTHRESH, linscale=1.0,
                                vmin=-VMAX_VEL, vmax=VMAX_VEL, base=10)
                cmap = "RdBu_r"

            im = ax.pcolormesh(
                np.linspace(extent_kpc[0], extent_kpc[1], data.shape[1]+1),
                np.linspace(extent_kpc[2], extent_kpc[3], data.shape[0]+1),
                data, norm=norm, cmap=cmap, shading="flat", rasterized=True,
            )
            plt.colorbar(im, ax=ax, label="cm/s")
            ax.set_title(f"{rlabel} — {plabel}", fontsize=9)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_aspect("equal")

    fig.suptitle(f"Velocity slices — t = {t_myr:.1f} Myr", fontsize=13)
    fig.tight_layout()
    fig.savefig(tag("velocity_12panel.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    rprint(f"Saved: {tag('velocity_12panel.png')}")

# ============================================================
# Zoomed velocity 12-panel around the SN injection site
# ============================================================
# This is what actually lets you judge whether a deposition looks like a
# "proper ejection": the full-domain plots above are ~20 kpc across, so a
# few-cell SN kernel is invisible as anything but a single pixel. This
# section re-centers on the debug target with a much smaller width so you
# can see the kernel's internal structure -- is it roughly isotropic, does
# it fall off smoothly with radius, is there an obvious one-sided lopsided
# component -- and reports the numeric momentum/mass sums in the box as an
# independent cross-check against the [SN feedback conservation check] /
# [SN_VEL] printouts already emitted by the C++ code.
if do_sn_zoom:
    rprint("\n--- Zoomed velocity panel around SN injection site ---")

    center_cm = ds.arr(
        [SN_X_KPC * kpc, SN_Y_KPC * kpc, SN_Z_KPC * kpc], "cm"
    )
    width_cm_zoom = SN_WIDTH_KPC * kpc
    extent_zoom_kpc = [-SN_WIDTH_KPC / 2, SN_WIDTH_KPC / 2,
                       -SN_WIDTH_KPC / 2, SN_WIDTH_KPC / 2]

    # Collective reads on all ranks first (same SPMD requirement as the
    # full-domain slices above -- every rank must call get_slice_xy_zoom).
    zoom_slices = {}
    for field in vel_fields:
        for normal in normals:
            rprint(f"  [zoom] Slicing {field} {normal}...")
            data, xlabel, ylabel, title = get_slice_xy_zoom(
                normal, ("boxlib", field), center_cm, width_cm_zoom
            )
            zoom_slices[(field, normal)] = (data, xlabel, ylabel, title)

    if is_root:
        fig, axes = plt.subplots(4, 3, figsize=(16, 18))

        # Auto-scale rather than reuse the domain-wide FIXED VMAX_VEL: the
        # circular-velocity scale that's appropriate for a 20 kpc disk plot
        # would either clip a strong SN kick or wash out a weak one at this
        # much smaller scale. Percentile-based limits let the actual kernel
        # structure show up.
        all_zoom_data = np.concatenate([
            zoom_slices[(f, n)][0].ravel() for f in vel_fields for n in normals
            if f != "velocity_mag"
        ])
        finite_zoom = all_zoom_data[np.isfinite(all_zoom_data)]
        zoom_vmax = np.percentile(np.abs(finite_zoom), 99.5) if finite_zoom.size else 1e5
        zoom_vmax = max(zoom_vmax, 1.0)  # guard against a degenerate all-zero box
        zoom_linthresh = max(zoom_vmax * 1e-3, 1.0)

        mag_data_all = np.concatenate([
            zoom_slices[("velocity_mag", n)][0].ravel() for n in normals
        ])
        finite_mag = mag_data_all[np.isfinite(mag_data_all) & (mag_data_all > 0)]
        mag_vmin = np.percentile(finite_mag, 1) if finite_mag.size else 1e3
        mag_vmax = np.percentile(finite_mag, 99.5) if finite_mag.size else zoom_vmax

        for row, (field, rlabel) in enumerate(zip(vel_fields, vel_labels)):
            for col, normal in enumerate(normals):
                ax = axes[row, col]
                data, xlabel, ylabel, plabel = zoom_slices[(field, normal)]

                if field == "velocity_mag":
                    norm = LogNorm(vmin=max(mag_vmin, 1.0), vmax=max(mag_vmax, mag_vmin * 10))
                    cmap = "magma"
                else:
                    norm = SymLogNorm(linthresh=zoom_linthresh, linscale=1.0,
                                      vmin=-zoom_vmax, vmax=zoom_vmax, base=10)
                    cmap = "RdBu_r"

                im = ax.pcolormesh(
                    np.linspace(extent_zoom_kpc[0], extent_zoom_kpc[1], data.shape[1] + 1),
                    np.linspace(extent_zoom_kpc[2], extent_zoom_kpc[3], data.shape[0] + 1),
                    data, norm=norm, cmap=cmap, shading="flat", rasterized=True,
                )
                plt.colorbar(im, ax=ax, label="cm/s")
                ax.set_title(f"{rlabel} — {plabel}", fontsize=9)
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
                ax.set_aspect("equal")
                # Mark the injection site itself for reference.
                ax.plot(0, 0, marker="+", color="lime", markersize=10, mew=1.5)

        fig.suptitle(
            f"Zoomed velocity around SN site "
            f"({SN_X_KPC:.2f}, {SN_Y_KPC:.2f}, {SN_Z_KPC:.2f}) kpc, "
            f"width={SN_WIDTH_KPC*1e3:.0f} pc — t = {t_myr:.3f} Myr",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(tag("velocity_zoom_sn.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        rprint(f"Saved: {tag('velocity_zoom_sn.png')}")

    # ------------------------------------------------------------
    # Numeric cross-check: net mass/momentum in a small box around the
    # target. A properly isotropic, momentum-conserving kick should show
    # |sum(px)|, |sum(py)|, |sum(pz)| each small relative to the RMS
    # per-cell momentum magnitude in the box -- a large net vector in one
    # direction indicates a lopsided deposit (e.g. clipped at a domain/box
    # edge, or the Sx/Sy/Sz re-centering not fully cancelling). This uses
    # a direct region select rather than the chunked slab loop above,
    # since the box here is tiny (a few cells at the finest level) --
    # run this with a single rank (or accept it only being exact on
    # rank 0) if using MPI, since ds.box() region sums below are not
    # wrapped in the same explicit round-robin distribution as the
    # volume-average section.
    if is_root:
        half_width_cm = 0.5 * width_cm_zoom
        box_left = center_cm.v - half_width_cm
        box_right = center_cm.v + half_width_cm
        region = ds.box(ds.arr(box_left, "cm"), ds.arr(box_right, "cm"))

        rho_box = region[("boxlib", "gasDensity")].v
        vol_box = region[("index", "cell_volume")].v
        vx_box = region[("boxlib", "x-Velocity")].v
        vy_box = region[("boxlib", "y-Velocity")].v
        vz_box = region[("boxlib", "z-Velocity")].v

        mass_box = rho_box * vol_box
        px_box = mass_box * vx_box
        py_box = mass_box * vy_box
        pz_box = mass_box * vz_box

        total_mass = np.sum(mass_box)
        total_px, total_py, total_pz = np.sum(px_box), np.sum(py_box), np.sum(pz_box)
        net_p_mag = np.sqrt(total_px**2 + total_py**2 + total_pz**2)
        rms_cell_p = np.sqrt(np.mean(px_box**2 + py_box**2 + pz_box**2))

        Msun = 1.989e33
        print("====================================================")
        print(f"SN zoom-box numeric check @ ({SN_X_KPC:.2f},{SN_Y_KPC:.2f},{SN_Z_KPC:.2f}) kpc, "
              f"width={SN_WIDTH_KPC*1e3:.0f} pc, N_cells={rho_box.size}")
        print(f"  total mass in box       = {total_mass/Msun:.3e} Msun")
        print(f"  net momentum vector     = ({total_px:.3e}, {total_py:.3e}, {total_pz:.3e}) g cm/s")
        print(f"  |net momentum|          = {net_p_mag:.3e} g cm/s")
        print(f"  RMS per-cell |p|        = {rms_cell_p:.3e} g cm/s")
        if rms_cell_p > 0:
            print(f"  |net p| / RMS per-cell  = {net_p_mag/rms_cell_p:.3e}  "
                  f"(near 0 => isotropic; near/above 1 => lopsided)")
        print(f"  max |v| in box          = {np.max(np.sqrt(vx_box**2+vy_box**2+vz_box**2)):.3e} cm/s")
        print("====================================================")


# ============================================================
# Rotation curve diagnostic
# ============================================================
rprint("\n--- Rotation curve diagnostic ---")

Rc_kpc  = 2.0
Rc_cm   = Rc_kpc * 1.0e3 * 3.085677581e18
cs_disk = 7.0e5
Mc      = float(ds.parameters.get("mhd_galaxy.Mc", 30.0))
vc_cms  = Mc * cs_disk
vc_kms  = vc_cms / 1.0e5

rprint(f"  vc = {vc_kms:.1f} km/s  (Mc={Mc}, cs_disk={cs_disk/1e5:.1f} km/s)")

vcirc_data = get_slice("z", ("boxlib", "circular_velocity"))  # collective — all ranks call this

if is_root:
    R_kpc_arr   = np.linspace(0.0, 10.0, 500)
    R_cm_arr    = R_kpc_arr * kpc
    vrot_correct = vc_cms * R_cm_arr / np.sqrt(R_cm_arr**2 + Rc_cm**2)  / 1.0e5
    vrot_buggy   = vc_cms * R_cm_arr / np.sqrt(R_cm_arr**2 + Rc_kpc**2) / 1.0e5

    px = np.linspace(-width_kpc/2, width_kpc/2, vcirc_data.shape[1])
    py = np.linspace(-width_kpc/2, width_kpc/2, vcirc_data.shape[0])
    XX, YY = np.meshgrid(px, py)
    R_grid = np.sqrt(XX**2 + YY**2)

    R_bins    = np.linspace(0.0, 10.0, 80)
    R_mid     = 0.5 * (R_bins[:-1] + R_bins[1:])
    vcirc_med = np.zeros(len(R_mid))
    vcirc_p16 = np.zeros(len(R_mid))
    vcirc_p84 = np.zeros(len(R_mid))

    for idx_r, (rlo, rhi) in enumerate(zip(R_bins[:-1], R_bins[1:])):
        mask = (R_grid >= rlo) & (R_grid < rhi) & (R_grid > dead_zone_kpc)
        if mask.sum() > 4:
            vals           = vcirc_data[mask]
            vcirc_med[idx_r] = np.median(vals)
            vcirc_p16[idx_r] = np.percentile(vals, 16)
            vcirc_p84[idx_r] = np.percentile(vals, 84)

    valid = vcirc_med != 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(R_kpc_arr, vrot_correct, lw=2,        label="Correct  (Rc = 2 kpc)")
    axes[0].plot(R_kpc_arr, vrot_buggy,   lw=2, ls="--", label="Buggy (Rc = 2 cm ≈ 0)")
    axes[0].axvline(Rc_kpc, color="gray", ls=":", lw=1, label=f"Rc = {Rc_kpc} kpc")
    axes[0].set_xlabel("R [kpc]"); axes[0].set_ylabel("$v_{rot}$ [km/s]")
    axes[0].set_title("Analytic rotation curve: correct vs buggy IC")
    axes[0].legend(); axes[0].set_xlim(0, 10); axes[0].set_ylim(0, vc_kms * 1.15)

    axes[1].fill_between(R_mid[valid], vcirc_p16[valid], vcirc_p84[valid],
                         alpha=0.3, label="16–84th pct")
    axes[1].plot(R_mid[valid], vcirc_med[valid], lw=2, label="Median (sim)")
    axes[1].plot(R_kpc_arr, vrot_correct, lw=2, ls="--",
                 label="Analytic (correct)", color="green")
    axes[1].axvline(Rc_kpc,        color="gray", ls=":", lw=1,
                    label=f"Rc = {Rc_kpc} kpc")
    axes[1].axvline(dead_zone_kpc, color="red",  ls=":", lw=1,
                    label=f"Dead zone ({dead_zone_kpc:.2f} kpc)")
    axes[1].set_xlabel("R [kpc]"); axes[1].set_ylabel("$v_{circ}$ [km/s]")
    axes[1].set_title("Sim circular velocity vs analytic (midplane)")
    axes[1].legend(fontsize=8); axes[1].set_xlim(0, 10); axes[1].set_ylim(bottom=0)

    fig.suptitle(f"Rotation curve diagnostic — t = {t_myr:.1f} Myr", fontsize=12)
    fig.tight_layout()
    fig.savefig(tag("rotation_curve_diagnostic.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    rprint(f"Saved: {tag('rotation_curve_diagnostic.png')}")

comm.Barrier()
elapsed = time.time() - t_start
rprint(f"\nAll plots complete. Wall time: {elapsed:.1f} s  ({size} rank(s))")