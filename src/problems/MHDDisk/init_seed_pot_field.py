import numpy as np
from scipy.special import j1, jn_zeros
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════

nR_coarse, nz_coarse = 64, 128
levels               = 3
SEED                 = 42

# Physical constants
G_grav = 6.674e-8       # cm^3 g^-1 s^-2
kpc    = 3.085677581e21 # cm

# Disk parameters — must match EOS_Traits and SimulationData in testMHDDisk.cpp
cs_disk  = 7.0e5        # cm/s — matches quokka::EOS_Traits<MHDGalaxy>::cs_disk
Sigma0   = 0.01169      # g/cm^2 — from preCalculateInitialConditions output
Rd_kpc   = 3.0          # disk scale length, matches anonymous namespace
alpha_p  = 2.0          # surface density shape parameter
beta_p   = 0.5          # surface density shape parameter

# Taper widths
inner_taper_kpc = 0.5   # ramp up over this width from axis
outer_taper_kpc = 0.5   # roll off over this width at outer edge

# Seed field strength
B0 = 4.0*0.127e-6           # Gauss — set for plasma beta ~ 1000 in disk midplane
oversample           = 4.  # factor by which to oversample the spectral grid relative to the output grid

# ══════════════════════════════════════════════════════════════════
#  Disk scale height (matches diskDensityAnalytic in testMHDDisk.cpp)
#
#  Sigma(R) = Sigma0 * exp(-R/Rd - beta * exp(-alpha * R/Rd))
#  H(R)     = cs^2 / (pi * G * Sigma(R))
#
#  We use R=Rd as the representative radius since that is where
#  most of the disk mass sits and the scale height is well-defined.
# ══════════════════════════════════════════════════════════════════

def disk_scale_height(Sigma0, cs, R_kpc, Rd_kpc, alpha_p, beta_p):
    """
    Isothermal disk scale height H = cs^2 / (pi G Sigma(R)).
    Matches the formula in diskDensityAnalytic in testMHDDisk.cpp.
    """
    x     = R_kpc / Rd_kpc
    Sigma = Sigma0 * np.exp(-x - beta_p * np.exp(-alpha_p * x))
    H     = cs**2 / (np.pi * G_grav * Sigma)
    return H, Sigma

# ══════════════════════════════════════════════════════════════════
#  Physical k helpers  (operate in dimensionless units: Rmax=1)
# ══════════════════════════════════════════════════════════════════

def physical_kmin(H_nd):
    """
    Largest turbulent scale = full disk thickness = 2H
    (one scale height above and below midplane).
    lambda_max = 2H  =>  kmin = 2*pi / lambda_max = pi / H
    """
    return np.pi / H_nd


def physical_kmax(nR_coarse, nz_coarse, Rmax_nd, Lz_nd, levels):
    """Nyquist diagonal k on the fine grid, dimensionless units."""
    factor = 2 ** levels
    dR     = (Rmax_nd / nR_coarse) / factor
    dz     = (Lz_nd   / nz_coarse) / factor
    return np.sqrt((np.pi / dR)**2 + (np.pi / dz)**2)

# ══════════════════════════════════════════════════════════════════
#  Spectral modes  (all dimensionless)
# ══════════════════════════════════════════════════════════════════

def init_modes(nR_coarse, nz_coarse, Rmax_nd, Lz_nd, seed=0):
    M      = nR_coarse // 2
    N      = nz_coarse // 2

    # Radial: zeros of J1 → kR such that J1(kR*Rmax)=0 exactly
    jzeros = jn_zeros(1, M)
    kR     = jzeros / Rmax_nd          # dimensionless

    # Axial: integer Fourier modes (no jitter — jitter breaks orthogonality
    # and can produce kz*z values that overflow float64 for large z)
    n_idx = np.arange(-N, N + 1)
    n_idx = n_idx[n_idx != 0]
    kz    = 2 * np.pi * n_idx / Lz_nd

    kR_grid, kz_grid = np.meshgrid(kR, kz, indexing='ij')
    k = np.sqrt(kR_grid**2 + kz_grid**2)

    return kR, kz, k

# ══════════════════════════════════════════════════════════════════
#  Random coefficients with Kolmogorov power law
# ══════════════════════════════════════════════════════════════════

def init_coeffs(k, kmin, kmax, seed):
    rng   = np.random.default_rng(seed)
    coeff = np.zeros_like(k, dtype=np.complex128)

    mask = (k >= kmin) & (k < kmax)
    rand = rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(mask.sum())

    # Pure Kolmogorov — no kR correction
    alpha = 11.0 / 6.0
    rand *= k[mask] ** (-alpha)
    rand /= np.sqrt(np.mean(np.abs(rand)**2))

    coeff[mask] = rand
    return coeff

# ══════════════════════════════════════════════════════════════════
#  Field construction  (dimensionless grid)
# ══════════════════════════════════════════════════════════════════

def evaluate_Aphi(R_nd, z_nd, kR, kz, coeff, batch_size=500):
    Aphi = np.zeros((len(R_nd), len(z_nd)))

    kR_flat    = np.repeat(kR, len(kz))
    kz_flat    = np.tile(kz, len(kR))
    coeff_flat = coeff.ravel()

    active     = np.abs(coeff_flat) > 0
    kR_flat    = kR_flat[active]
    kz_flat    = kz_flat[active]
    coeff_flat = coeff_flat[active]
    n_modes    = len(coeff_flat)
    print(f"  Active modes: {n_modes}")

    for i in range(0, n_modes, batch_size):
        sl   = slice(i, i + batch_size)
        kR_b = kR_flat[sl]
        kz_b = kz_flat[sl]
        c_b  = coeff_flat[sl]

        J = j1(kR_b[None, :] * R_nd[:, None])   # (nR, batch)

        # Normalise each Bessel column by its cylindrical L2 norm
        # ||J1(kR*R)||^2 = int_0^1 J1(kR*R)^2 R dR  (discrete sum)
        J_norm = np.sqrt(np.sum(J**2 * R_nd[:, None], axis=0) * (R_nd[1] - R_nd[0]))
        J_norm = np.where(J_norm > 0, J_norm, 1.0)
        J /= J_norm[None, :]                         # (nR, batch), now unit cylindrical norm

        cos_z = np.cos(kz_b[:, None] * z_nd[None, :])
        sin_z = np.sin(kz_b[:, None] * z_nd[None, :])
        Z     = np.real(c_b)[:, None] * cos_z - np.imag(c_b)[:, None] * sin_z
        Aphi += J @ Z

    Aphi /= np.sqrt(n_modes)
    return Aphi

# ══════════════════════════════════════════════════════════════════
#  Tapers
# ══════════════════════════════════════════════════════════════════

def apply_axis_taper(Aphi, R_nd, dR_nd, n_cells=8):
    """
    Smooth cubic ramp from 0 at R=0 to 1 at R = n_cells*dR.
    Enforces A_phi -> 0 at the axis so that R*A_phi -> 0 there.
    """
    x     = R_nd / (n_cells * dR_nd)
    taper = np.where(x < 1.0, 3*x**2 - 2*x**3, 1.0)
    return Aphi * taper[:, np.newaxis]

def apply_outer_taper(Aphi, R_nd, Rmax_nd, width_cells=8):
    """
    Cosine roll-off to zero over the outer width_cells cells.
    Enforces A_phi -> 0 at R=Rmax so that R*A_phi -> 0 there.
    """
    dR    = R_nd[1] - R_nd[0]
    width = width_cells * dR
    x     = (Rmax_nd - R_nd) / width
    taper = np.where(x >= 1.0, 1.0,
            np.where(x <= 0.0, 0.0,
                     0.5 * (1.0 - np.cos(np.pi * x))))
    return Aphi * taper[:, np.newaxis]

# ══════════════════════════════════════════════════════════════════
#  Curl:  input is  RA = R * A_phi  (already multiplied)
#         Br = -(1/R) d(RA)/dz
#         Bz =  (1/R) d(RA)/dR
#  This form makes the flux integral telescope exactly:
#      2pi int Bz R dR = 2pi [RA]_0^Rmax = 0
# ══════════════════════════════════════════════════════════════════

def curl_Aphi(RA, R, dR, dz):
    """
    RA  : array (nR, nz) =  R * A_phi
    R   : 1-D array (nR,)
    Returns Br, Bz each of shape (nR, nz).
    """
    R_col = R[:, None]          # (nR, 1)  for 2-D interior ops

    # ── Br = -(1/R) d(RA)/dz ────────────────────────────────────
    Br = np.empty_like(RA)
    Br[:, 1:-1] = -(RA[:, 2:] - RA[:, :-2]) / (2 * dz * R_col)
    Br[:, 0]    = -(RA[:, 1]  - RA[:, 0])   / (dz * R)      # R is 1-D (nR,)
    Br[:, -1]   = -(RA[:, -1] - RA[:, -2])  / (dz * R)      # R is 1-D (nR,)
    Br[0, :]    = 0.0           # axis regularity: A_phi=0 => Br=0

    # ── Bz = (1/R) d(RA)/dR ─────────────────────────────────────
    Bz = np.empty_like(RA)
    # interior: central difference
    Bz[1:-1] = (RA[2:] - RA[:-2]) / (2 * dR * R_col[1:-1])
    # outer boundary: one-sided
    Bz[-1]   = (RA[-1] - RA[-2])  / (dR * R[-1])
    # axis: extrapolate Bz[0] linearly from the first two interior values
    # (avoids dividing small RA[1] by tiny dR which amplifies taper noise)
    Bz[0]    = 2.0 * Bz[1] - Bz[2]

    return Br, Bz

# ══════════════════════════════════════════════════════════════════
#  Save binary + metadata
# ══════════════════════════════════════════════════════════════════

def save_outputs(RA_phys, Br, Bz,
                 nR, nz, nR_coarse, nz_coarse, levels, oversample, SEED,
                 Rmax, Lz, dR, dz, kmin_nd, kmax_nd, Rmax_nd,
                 H_phys_cm, B0_gauss=1e-9, stem="Aphi_2d"):
    import os

    bin_path  = stem + ".bin"
    meta_path = stem + "_meta.txt"

    RA_phys.astype(np.float64).tofile(bin_path)
    print(f"Saved binary : {bin_path}  ({os.path.getsize(bin_path)/1e6:.1f} MB)")

    # B0_HL is the physical scale: multiply normalised B by B0_HL to get Gauss (HL)
    B0_HL    = B0_gauss / (4.0 * np.pi) ** 0.5
    # rms of normalised B is 1 by construction; physical rms = B0_HL
    rms_B_HL = B0_HL

    lines = [
        f"seed              = {SEED}",
        f"nR                = {int(nR)}",
        f"nz                = {int(nz)}",
        f"nR_coarse         = {int(nR_coarse)}",
        f"nz_coarse         = {int(nz_coarse)}",
        f"amr_levels        = {int(levels)}",
        f"oversample        = {oversample}",
        f"Rmin_cm           = {0.0:e}",
        f"Rmax_cm           = {Rmax:e}",
        f"Lz_cm             = {Lz:e}",
        f"Rmax_kpc          = {Rmax/kpc:.6f}",
        f"Lz_kpc            = {Lz/kpc:.6f}",
        f"dR_fine_cm        = {dR:e}",
        f"dz_fine_cm        = {dz:e}",
        f"H_disk_cm         = {H_phys_cm:e}  [scale height at R=Rd, used for kmin]",
        f"H_disk_pc         = {H_phys_cm/kpc*1e3:.1f}",
        f"kmin_nd           = {kmin_nd:e}  [dimensionless, Rmax=1]",
        f"kmax_nd           = {kmax_nd:e}  [dimensionless, Rmax=1]",
        f"kmin_phys_cm-1    = {kmin_nd/Rmax:e}",
        f"kmax_phys_cm-1    = {kmax_nd/Rmax:e}",
        f"kmax_over_kmin    = {kmax_nd/kmin_nd:.2f}",
        f"alpha_Aphi        = {11/6:.6f}  (|A_k| ~ k^-alpha, dimensionless k)",
        f"spectrum          = E_B(k) ~ k^(-5/3)  [Kolmogorov]",
        f"B0_gauss          = {B0_gauss:e}  [Gaussian; multiply normalised B to get physical]",
        f"B0_HL             = {B0_HL:e}  [Heaviside-Lorentz; used internally]",
        f"rms_B_HL          = {rms_B_HL:e}  [= B0_HL, since stored B is rms-normalised to 1]",
        f"layout            = C-order float64, shape ({nR}, {nz})",
        f"stored_field      = RA = R * A_phi(R,z)  [cm * normalised_A_phi]",
        f"units_note        = B fields are normalised: rms(Br^2+Bz^2)=1. Scale by B0_HL for physical Gauss (HL).",
        f"Br_formula        = -(1/R) * d(RA)/dz",
        f"Bz_formula        =  (1/R) * d(RA)/dR",
        f"Bphi              = 0  (exact, axisymmetric construction)",
        f"net_vertical_flux = 0  (RA=0 at R=0 and R=Rmax by construction)",
        f"axis_bc           = RA[0,:]=0  (hard zero + smoothstep taper over {60} cells)",
        f"outer_bc          = RA[-1,:]=~0 (cosine taper over {60} cells)",
        f"basis_R           = J_1(j_{{1,m}} R / Rmax)  [zeros of J1 at Rmax]",
        f"basis_z           = cos(2*pi*n*z/Lz), sin(2*pi*n*z/Lz)",
        f"indexing          = row-major: element [i,j] = i*nz + j  (i=R index, j=z index)",
        f"R_cell_centre     = (i + 0.5) * dR_fine_cm  for i in 0..nR-1",
        f"z_cell_centre     = -Lz_cm/2 + (j + 0.5) * dz_fine_cm  for j in 0..nz-1",
    ]

    with open(meta_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved metadata: {meta_path}")
    return bin_path, meta_path

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Lz   = 20.0 * kpc              # physical box height
    Rmax = 10.0 * np.sqrt(2) * kpc # physical outer radius — box diagonal / 2

    # ── Disk scale height at R=Rd ─────────────────────────────────
    # Uses the same formula as diskDensityAnalytic in testMHDDisk.cpp:
    #   H(R) = cs^2 / (pi * G * Sigma(R))
    # Evaluated at R=Rd as the most representative disk radius.
    H_phys, Sigma_Rd = disk_scale_height(
        Sigma0, cs_disk, Rd_kpc, Rd_kpc, alpha_p, beta_p
    )
    print(f"Disk scale height at R=Rd ({Rd_kpc} kpc): {H_phys/kpc*1e3:.0f} pc")
    print(f"Surface density at R=Rd: {Sigma_Rd:.4e} g/cm^2")

    # ── work in dimensionless units: length scale = Rmax ─────────
    Rmax_nd = 1.0
    Lz_nd   = Lz / Rmax            # ≈ 1.414...

    H_nd    = H_phys / Rmax        # dimensionless scale height at R=Rd

    # ── fine physical grid (defined FIRST so modes match finest cells) ──
    factor = 2 ** levels
    nz     = nz_coarse * factor * oversample
    nR     = int(np.ceil(nz * np.sqrt(2)))
    print(f"Output grid: nR={nR}, nz={nz}")

    dR = Rmax / nR
    dz = Lz   / nz

    # Dimensionless grids for spectral evaluation
    R_nd = (np.arange(nR) + 0.5) / nR * Rmax_nd   # in [0, Rmax_nd]
    z_nd = (-0.5 + (np.arange(nz) + 0.5) / nz) * Lz_nd

    # Physical grids (same points, different units)
    R = R_nd * Rmax
    z = z_nd * Rmax

    dR_nd = Rmax_nd / nR
    dz_nd = Lz_nd   / nz

    # ── spectral modes at FINE grid resolution ───────────────────
    kR, kz, k = init_modes(nR_coarse * factor, nz_coarse * factor, Rmax_nd, Lz_nd, seed=SEED)

    kmin_nd = physical_kmin(H_nd)

    # kmax = one-cell wavelength on the AMR grid (not the table)
    # lambda_min = dx_grid  =>  k = 2*pi/dx_grid
    dR_grid_nd = Rmax_nd / (nR_coarse * factor)
    dz_grid_nd = Lz_nd   / (nz_coarse * factor)
    kmax_nd    = np.sqrt((2.0 * np.pi / dR_grid_nd)**2 + (2.0 * np.pi / dz_grid_nd)**2)

    print(f"k range (nd): [{k.min():.3e}, {k.max():.3e}]")
    print(f"kmin (nd)   = {kmin_nd:.3e}  (lambda_max = 2H = {2*H_phys/kpc*1e3:.0f} pc)")
    print(f"kmax (nd)   = {kmax_nd:.3e}  (fine Nyquist diagonal)")
    print(f"kmax/kmin   = {kmax_nd/kmin_nd:.1f}")

    # ── spectral coefficients ────────────────────────────────────
    coeff = init_coeffs(k, kmin_nd, kmax_nd, SEED)

    # Zero out modes below the disk-thickness fundamental (avoids DC leakage)
    coeff[k < kmin_nd] = 0

    # ── evaluate A_phi on dimensionless grid ─────────────────────
    print("Evaluating A_phi...")
    Aphi = evaluate_Aphi(R_nd, z_nd, kR, kz, coeff)

    # Normalise to O(1) rms before tapers (avoids float arithmetic issues)
    Aphi /= np.sqrt(np.mean(Aphi**2))
    print(f"  Aphi rms after normalisation: {np.sqrt(np.mean(Aphi**2)):.3f}  (should be 1)")

    # ── boundary tapers ──────────────────────────────────────────
    inner_taper_cells = max(4, int(inner_taper_kpc * kpc / dR))
    outer_taper_cells = max(4, int(outer_taper_kpc * kpc / dR))

    print(f"  inner taper: {inner_taper_cells} cells = {inner_taper_cells*dR/kpc:.3f} kpc")
    print(f"  outer taper: {outer_taper_cells} cells = {outer_taper_cells*dR/kpc:.3f} kpc")

    Aphi = apply_axis_taper(Aphi, R_nd, dR_nd, n_cells=inner_taper_cells)
    Aphi = apply_outer_taper(Aphi, R_nd, Rmax_nd, width_cells=outer_taper_cells)

    # Hard-zero the axis half-cell (J1(0)=0; taper ramp starts at R=0 not R[0])
    Aphi[0, :] = 0.0

    print(f"  Aphi[0]  mean |.|: {np.mean(np.abs(Aphi[0,  :])):.3e}  (should be 0)")
    print(f"  Aphi[-1] mean |.|: {np.mean(np.abs(Aphi[-1, :])):.3e}  (should be ~0)")

    # ── radial rms equalisation on A_phi BEFORE curl ─────────────
    # Applied here so that RA[0]=RA[-1]=0 is preserved exactly after
    # rescaling, keeping div-B=0 and net flux cancellation analytic.
    Aphi_rms_R      = np.sqrt(np.mean(Aphi**2, axis=1))          # (nR,)
    Aphi_rms_smooth = uniform_filter1d(Aphi_rms_R, size=max(1, nR//50))
    floor           = 0.01 * Aphi_rms_smooth.max()
    Aphi_rms_smooth = np.where(Aphi_rms_smooth > floor, Aphi_rms_smooth, floor)
    Aphi           /= Aphi_rms_smooth[:, None]   # equalise each R-row

    # Re-enforce hard zeros that the division may have perturbed by the floor
    Aphi[0, :]  = 0.0
    Aphi[-1, :] *= 0.0   # outer taper already ~0; make exact for flux guarantee

    # Re-normalise globally to rms=1
    Aphi /= np.sqrt(np.mean(Aphi**2))
    print(f"  Aphi rms after equalisation: {np.sqrt(np.mean(Aphi**2)):.3f}  (should be 1)")

    # ── form  RA = R * A_phi  (exactly once) ─────────────────────
    RA = R_nd[:, None] * Aphi      # dimensionless RA

    print(f"  RA at R=0:    {np.mean(np.abs(RA[0,  :])):.3e}  (should be 0)")
    print(f"  RA at Rmax:   {np.mean(np.abs(RA[-1, :])):.3e}  (should be ~0)")

    # ── curl  (dimensionless derivatives) ────────────────────────
    Br_nd, Bz_nd = curl_Aphi(RA, R_nd, dR_nd, dz_nd)

    # ── verify net flux ───────────────────────────────────────────
    net_flux_raw = np.mean(np.trapezoid(Bz_nd * R_nd[:, None], R_nd, axis=0))
    print(f"  Net flux (raw, nd):  {net_flux_raw:.3e}  (target: 0)")

    # ── convert to physical units then normalise B ───────────────
    Br = Br_nd / Rmax
    Bz = Bz_nd / Rmax

    rms  = np.sqrt(np.mean(Br**2 + Bz**2))
    Br  /= rms
    Bz  /= rms
    RA_phys = RA * Rmax / rms   # keep RA consistent for saving

    print(f"  Br rms: {np.sqrt(np.mean(Br**2)):.4f}")
    print(f"  Bz rms: {np.sqrt(np.mean(Bz**2)):.4f}")
    print(f"  Total rms B: {np.sqrt(np.mean(Br**2 + Bz**2)):.4f}")

    # ── final flux check ──────────────────────────────────────────
    net_flux_nd = np.mean(np.trapezoid(Bz_nd * R_nd[:, None], R_nd, axis=0))
    print(f"  Net flux (nd, post-normalise check): {net_flux_nd:.3e}  (target: 0)")

    Bz_rms_profile = np.sqrt(np.mean(Bz[:80]**2, axis=1))
    print(f"  Bz[0:80:8] rms = {Bz_rms_profile[::8]}")

    Bphi = np.zeros_like(Br)
    print(f"  Bphi max (should be 0): {np.max(np.abs(Bphi)):.3e}")

    # ── save ─────────────────────────────────────────────────────
    save_outputs(
        RA_phys, Br, Bz,
        nR=nR, nz=nz, 
        nR_coarse=nR_coarse, nz_coarse=nz_coarse,
        levels=levels, oversample=oversample, SEED=SEED,
        Rmax=Rmax, Lz=Lz, dR=dR, dz=dz,
        kmin_nd=kmin_nd, kmax_nd=kmax_nd, Rmax_nd=Rmax_nd,
        H_phys_cm=H_phys,
        B0_gauss=B0, stem="Aphi_2d",
    )

    # ══════════════════════════════════════════════════════════════
    #  Diagnostic plots
    # ══════════════════════════════════════════════════════════════

    kpc_ext = [0, Rmax / kpc, -Lz / (2 * kpc), Lz / (2 * kpc)]

    # ── B-field components ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, B, lbl in zip(axes[:2], [Br, Bz], [r"$B_R$", r"$B_z$"]):
        vlim = np.percentile(np.abs(B), 99)
        im = ax.imshow(B.T, origin="lower", extent=kpc_ext,
                       aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        plt.colorbar(im, ax=ax, label=f"{lbl} [normalised]")
        ax.set_title(lbl)
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel("z [kpc]")

    im2 = axes[2].imshow(Bphi.T, origin="lower", extent=kpc_ext,
                         aspect="auto", cmap="RdBu_r", vmin=-1e-10, vmax=1e-10)
    plt.colorbar(im2, ax=axes[2], label=r"$B_\phi$")
    axes[2].set_title(r"$B_\phi$ (analytic zero)")
    axes[2].set_xlabel("R [kpc]")
    axes[2].set_ylabel("z [kpc]")

    plt.suptitle(r"Magnetic field from curl of $A_\phi$", fontsize=13)
    plt.tight_layout()
    plt.savefig("B_components.png", dpi=150)
    plt.show()
    print("Saved B_components.png")

    # ── A_phi spectrum ────────────────────────────────────────────
    k_flat = k.ravel()
    P_flat = np.abs(coeff.ravel())**2

    bins  = np.geomspace(kmin_nd, kmax_nd, 40)
    k_mid = 0.5 * (bins[:-1] + bins[1:])
    P     = np.array([
        np.mean(P_flat[(k_flat >= bins[i]) & (k_flat < bins[i+1])])
        if np.any((k_flat >= bins[i]) & (k_flat < bins[i+1])) else 0.0
        for i in range(len(bins)-1)
    ])

    valid  = P > 0
    k_plot = k_mid[valid]
    P_plot = P[valid]

    # ── Convert RA → Aphi ─────────────────────────────
    R_safe = R.copy()
    R_safe[0] = R_safe[1]   # avoid divide-by-zero at axis
    Aphi_plot = RA_phys / R_safe[:, None]
    Aphi_plot[0, :] = 0.0  # enforce regularity at axis

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im = ax[0].imshow(
        Aphi_plot.T,
        origin="lower",
        extent=[0, Rmax/kpc, -Lz/(2*kpc), Lz/(2*kpc)],
        aspect="auto",
        cmap="RdBu_r"
    )
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title(r"$A_\phi$ (normalised)")
    ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel("z [kpc]")

    if len(P_plot) > 1:
        mid  = len(P_plot) // 2
        norm = P_plot[mid] / k_plot[mid]**(-11/3)
        ax[1].loglog(k_plot, P_plot,              'o-', label="Measured")
        ax[1].loglog(k_plot, norm * k_plot**(-11/3), '--', label=r"$k^{-11/3}$")
        ax[1].axvline(kmin_nd, color='g', ls=':', label=f"kmin (λ=2H={2*H_phys/kpc*1e3:.0f} pc)")
        ax[1].set_xlabel("k (dimensionless)")
        ax[1].set_ylabel(r"$|A_k|^2$")
        ax[1].legend()

    plt.tight_layout()
    plt.savefig("Aphi_2d_check.png", dpi=150)
    plt.show()

    # ── magnetic energy spectrum ──────────────────────────────────
    kR_grid, kz_grid = np.meshgrid(kR, kz, indexing='ij')
    E_mode  = (kR_grid**2 + kz_grid**2) * np.abs(coeff)**2
    k_flat2 = np.sqrt(kR_grid**2 + kz_grid**2).ravel()
    E_flat  = E_mode.ravel()

    k_bins  = np.geomspace(kmin_nd, kmax_nd, 50)
    kB_plot, EB_plot = [], []
    for i in range(len(k_bins)-1):
        mask = (k_flat2 >= k_bins[i]) & (k_flat2 < k_bins[i+1])
        if np.any(mask):
            EB_plot.append(np.mean(E_flat[mask]))
            kB_plot.append(0.5*(k_bins[i]+k_bins[i+1]))

    kB_plot = np.array(kB_plot)
    EB_plot = np.array(EB_plot)

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(EB_plot) > 1:
        mid  = len(EB_plot) // 2
        norm = EB_plot[mid] / kB_plot[mid]**(-5/3)
        ax.loglog(kB_plot, EB_plot,                'o-', label="Measured")
        ax.loglog(kB_plot, norm * kB_plot**(-5/3), '--', label=r"$k^{-5/3}$")
        ax.axvline(kmin_nd, color='g', ls=':', label=f"kmin (λ=2H={2*H_phys/kpc*1e3:.0f} pc)")
    ax.set_xlabel("k (dimensionless)")
    ax.set_ylabel(r"$E_B(k)$")
    ax.set_title("Magnetic Energy Spectrum")
    ax.legend()
    plt.tight_layout()
    plt.savefig("Bspectrum_2d_check.png", dpi=150)
    plt.show()

    # ── radial rms profiles ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    B_rms_vs_R = np.sqrt(np.mean(Br**2 + Bz**2, axis=1))
    axes[0].plot(R / kpc, B_rms_vs_R)
    axes[0].set_xlabel("R [kpc]")
    axes[0].set_ylabel("rms |B| (normalised)")
    axes[0].set_title("Total rms B vs R")

    axes[1].plot(R/kpc, np.sqrt(np.mean(Br**2, axis=1)), label=r'$B_R$')
    axes[1].plot(R/kpc, np.sqrt(np.mean(Bz**2, axis=1)), label=r'$B_z$')
    axes[1].set_xlabel("R [kpc]")
    axes[1].set_ylabel("rms (normalised)")
    axes[1].set_title("Component rms vs R")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("B_rms_profiles.png", dpi=150)
    plt.show()

    # ── net flux per z-slice ──────────────────────────────────────
    flux_per_z = 2 * np.pi * np.trapezoid(Bz_nd * R_nd[:, None], R_nd, axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_nd * Rmax / kpc, flux_per_z)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("z [kpc]")
    ax.set_ylabel(r"$\Phi_z(z)$ [dimensionless]")
    ax.set_title("Net vertical flux per z-slice (should be ~0)")
    plt.tight_layout()
    plt.savefig("net_flux_check.png", dpi=150)
    plt.show()

    print(f"\nFinal summary:")
    print(f"  Scale height at R=Rd: {H_phys/kpc*1e3:.0f} pc")
    print(f"  lambda_max = 2H:      {2*H_phys/kpc*1e3:.0f} pc")
    print(f"  kmin (nd):            {kmin_nd:.3e}")
    print(f"  kmax (nd):            {kmax_nd:.3e}")
    print(f"  kmax/kmin:            {kmax_nd/kmin_nd:.1f}")
    print(f"  Br rms  = {np.sqrt(np.mean(Br**2)):.4f}")
    print(f"  Bz rms  = {np.sqrt(np.mean(Bz**2)):.4f}")
    print(f"  |B| rms = {np.sqrt(np.mean(Br**2+Bz**2)):.4f}")
    print(f"  Net flux mean = {np.mean(flux_per_z):.3e}  (dimensionless)")
    print(f"  Net flux std  = {np.std(flux_per_z):.3e}  (dimensionless)")
    print(f"  B0_gauss = {B0:.3e} G  =>  B0_HL = {B0/(4*np.pi)**0.5:.3e} G (HL)")
