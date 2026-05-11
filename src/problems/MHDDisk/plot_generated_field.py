import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# ── load metadata ──────────────────────────────────────────────────
meta = {}
with open("Aphi_2d_meta.txt") as f:
    for line in f:
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

nR   = int(meta["nR"])
nz   = int(meta["nz"])
Rmax = float(meta["Rmax_cm"])
Lz   = float(meta["Lz_cm"])
Rmin = float(meta["Rmin_cm"])

kpc      = 3.085677581e21
to_gauss = 1.0  # B is rms-normalised; physical scaling applied separately

Rbox_kpc = 10.0
Lz_kpc   = Lz / 2 / kpc
Rbox     = Rbox_kpc * kpc

# ── reconstruct cylindrical grids ─────────────────────────────────
dR    = (Rmax - Rmin) / nR
dz    = Lz / nz
R_cyl = Rmin + (np.arange(nR) + 0.5) * dR
z_cyl = -Lz/2 + (np.arange(nz) + 0.5) * dz

# ── load RA = R*A_phi and recompute cylindrical B ─────────────────
RA    = np.fromfile("Aphi_2d.bin", dtype=np.float64).reshape(nR, nz)
R_col = R_cyl[:, np.newaxis]

# Br = -(1/R) d(RA)/dz
Br_cyl          = np.empty_like(RA)
Br_cyl[:, 1:-1] = -(RA[:, 2:] - RA[:, :-2]) / (2 * dz * R_col)
Br_cyl[:, 0]    = -(RA[:, 1]  - RA[:, 0])   / (dz * R_cyl)
Br_cyl[:, -1]   = -(RA[:, -1] - RA[:, -2])  / (dz * R_cyl)
Br_cyl[0, :]    = 0.0

# Bz = (1/R) d(RA)/dR
Bz_cyl          = np.empty_like(RA)
Bz_cyl[1:-1]    = (RA[2:] - RA[:-2]) / (2 * dR * R_col[1:-1])
Bz_cyl[-1]      = (RA[-1] - RA[-2])  / (dR  * R_cyl[-1])
Bz_cyl[0]       = 2.0 * Bz_cyl[1] - Bz_cyl[2]

Br_cyl *= to_gauss
Bz_cyl *= to_gauss

# ── build 3D Cartesian volume ──────────────────────────────────────
nCart    = 512
n_slab   = 5   # half-width of slab average in cells (±n_slab//2)

x_1d  = np.linspace(-Rbox, Rbox, nCart)
y_1d  = np.linspace(-Rbox, Rbox, nCart)
z_1d  = np.linspace(-Lz/2, Lz/2, nCart)

X, Y, Z3 = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

R3   = np.sqrt(X**2 + Y**2)
phi3 = np.arctan2(Y, X)

R3_clamped = np.clip(R3, R_cyl[0], R_cyl[-1])
Z3_clamped = np.clip(Z3, z_cyl[0], z_cyl[-1])

iR = np.clip(np.searchsorted(R_cyl, R3_clamped) - 1, 0, nR - 2)
iz = np.clip(np.searchsorted(z_cyl, Z3_clamped) - 1, 0, nz - 2)

wR = np.clip((R3_clamped - R_cyl[iR]) / dR, 0, 1)
wz = np.clip((Z3_clamped - z_cyl[iz]) / dz, 0, 1)

def bilinear(F):
    return (F[iR,   iz  ] * (1-wR) * (1-wz)
          + F[iR+1, iz  ] *    wR  * (1-wz)
          + F[iR,   iz+1] * (1-wR) *    wz
          + F[iR+1, iz+1] *    wR  *    wz)

Br3 = bilinear(Br_cyl)
Bz3 = bilinear(Bz_cyl)

Bx3 = Br3 * np.cos(phi3)
By3 = Br3 * np.sin(phi3)

outside = np.abs(Z3) > Lz/2
Bx3[outside] = 0.0
By3[outside] = 0.0
Bz3[outside] = 0.0

Bmag3 = np.sqrt(Bx3**2 + By3**2 + Bz3**2)

x_kpc = x_1d / kpc
y_kpc = y_1d / kpc
z_kpc = z_1d / kpc

imid  = nCart // 2
iLo   = max(0,     imid - n_slab // 2)
iHi   = min(nCart, imid + n_slab // 2 + 1)

# ── slab-averaged XY slices (breaks Bessel phase coherence) ───────
def xy_slab(F):
    """Average over a thin z-slab centred on z=0."""
    return F[:, :, iLo:iHi].mean(axis=2)

def xz_slab(F):
    """Average over a thin y-slab centred on y=0."""
    return F[:, iLo:iHi, :].mean(axis=1)

def yz_slab(F):
    """Average over a thin x-slab centred on x=0."""
    return F[iLo:iHi, :, :].mean(axis=0)

# ── colour normalisation ───────────────────────────────────────────
def make_norm(data, linthresh_fraction=1e-2):
    finite    = data[np.isfinite(data)]
    vmax      = np.percentile(np.abs(finite), 99)
    if vmax == 0:
        vmax = 1.0
    linthresh = linthresh_fraction * vmax
    return SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

def make_norm_pos(data):
    from matplotlib.colors import LogNorm
    finite = data[np.isfinite(data) & (data > 0)]
    vmin   = np.percentile(finite, 1)
    vmax   = np.percentile(finite, 99)
    return LogNorm(vmin=vmin, vmax=vmax)

# ── plotting ───────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(16, 18))

def plot_panel(ax, Xax, Yax, field, title, xlabel, ylabel, positive=False):
    norm = make_norm_pos(field) if positive else make_norm(field)
    cmap = "inferno" if positive else "RdBu_r"
    im   = ax.pcolormesh(Xax, Yax, field.T,
                         norm=norm, cmap=cmap)
    plt.colorbar(im, ax=ax, label="normalised")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

plane_labels = [
    ("XY (z≈0)", x_kpc, y_kpc, "x [kpc]", "y [kpc]"),
    ("XZ (y≈0)", x_kpc, z_kpc, "x [kpc]", "z [kpc]"),
    ("YZ (x≈0)", y_kpc, z_kpc, "y [kpc]", "z [kpc]"),
]

components = [
    (xy_slab(Bx3), xz_slab(Bx3), yz_slab(Bx3), r"$B_x$"),
    (xy_slab(By3), xz_slab(By3), yz_slab(By3), r"$B_y$"),
    (xy_slab(Bz3), xz_slab(Bz3), yz_slab(Bz3), r"$B_z$"),
    (xy_slab(Bmag3), xz_slab(Bmag3), yz_slab(Bmag3), r"$|B|$"),
]

# streamplot uses the slab-averaged in-plane components
stream_slices = [
    (xy_slab(Bx3), xy_slab(By3)),   # XY plane: Bx, By
    (xz_slab(Bx3), xz_slab(Bz3)),   # XZ plane: Bx, Bz
    (yz_slab(By3), yz_slab(Bz3)),   # YZ plane: By, Bz
]

slab_kpc = n_slab * (z_1d[1] - z_1d[0]) / kpc
print(f"Slab thickness: {n_slab} cells = ±{slab_kpc/2:.2f} kpc")

for row, (s_xy, s_xz, s_yz, clabel) in enumerate(components):
    for col, slc in enumerate([s_xy, s_xz, s_yz]):
        _, Xax, Yax, xl, yl = plane_labels[col]
        positive = (row == 3)
        plot_panel(axes[row, col], Xax, Yax, slc,
                   f"{clabel} — {plane_labels[col][0]}", xl, yl,
                   positive=positive)

        if row == 3:
            u, v  = stream_slices[col]
            speed = np.sqrt(u**2 + v**2)
            with np.errstate(invalid='ignore', divide='ignore'):
                un = np.where(speed > 0, u / speed, 0.0)
                vn = np.where(speed > 0, v / speed, 0.0)
            axes[row, col].streamplot(
                Xax, Yax,
                un.T, vn.T,
                color="white",
                linewidth=0.6,
                density=1.5,
                arrowsize=0.7,
            )

plt.suptitle(
    f"Full 3D magnetic field slices — slab average ±{slab_kpc/2:.2f} kpc",
    fontsize=13)
plt.tight_layout()
plt.savefig("Bfield_12panel_py.png", dpi=300, bbox_inches="tight")
plt.savefig("Bfield_12panel_py.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("Saved Bfield_12panel_py.png / .pdf")
