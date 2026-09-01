import numpy as np
from scipy.special import j1, jn_zeros
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════

nR_coarse, nz_coarse = 128 , 256
levels               = 0
SEED = np.random.SeedSequence().entropy
print(f"Using random seed: {SEED}")

rng = np.random.default_rng(SEED)
padding              = 2    # ghost cells added beyond domain on each side

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
inner_taper_kpc = 0.6   # ramp up over this width from axis
outer_taper_kpc = 0.5   # ramp up over this width from axis

# Seed field strength
B0 = 4.0*0.127e-6           # Gauss — set for plasma beta ~ 1000 in disk midplane
B0 = B0 / np.sqrt(1000.0 / 3.0)  # ~ 2.8e-8 G
oversample           = 32.  # factor by which to oversample the spectral grid relative to the output grid
n_grid_cells = 4 

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


def physical_kmax(nR, nz, Rmax_nd, Lz_nd, n_cells=4):
    dR_fine = Rmax_nd / nR
    dz_fine = Lz_nd   / nz
    return np.sqrt((2*np.pi / (n_cells * dR_fine))**2 +
                   (2*np.pi / (n_cells * dz_fine))**2)


# ══════════════════════════════════════════════════════════════════
#  Spectral modes  (all dimensionless)
# ══════════════════════════════════════════════════════════════════

def init_modes(nR_coarse, nz_coarse, Rmax_nd, Lz_nd):
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

def init_coeffs(k, kmin, kmax, rng):
    coeff = np.zeros_like(k, dtype=np.complex128)

    mask = (k >= kmin) & (k < kmax)

    # Generate Gaussian random variables
    rand = (
        rng.standard_normal(mask.sum())
        + 1j * rng.standard_normal(mask.sum())
    )

    # Apply power law
    alpha = 11.0 / 6.0
    power_law = k[mask] ** (-alpha)
    rand *= power_law

    # Normalize by the realized total power
    total_power = np.sum(np.abs(rand)**2)
    rand /= np.sqrt(total_power)

    coeff[mask] = rand

    return coeff

def plot_coeff_spectrum(k, coeff, kmin_nd, kmax_nd, H_phys, kpc, out="Bspectrum_coeffs.png"):
    """
    Plot |B_k|^2 ~ k^2 |A_k|^2 directly from spectral coefficients.
    This is the ground-truth spectrum, unaffected by resampling.
    """
    k_flat = k.ravel()
    # B spectrum: k^2 * |A_k|^2
    EB_flat = (k_flat**2) * np.abs(coeff.ravel())**2

    bins   = np.geomspace(kmin_nd * 0.99, kmax_nd * 1.01, 50)
    k_mid, E_bin = [], []
    for i in range(len(bins)-1):
        m = (k_flat >= bins[i]) & (k_flat < bins[i+1]) & (EB_flat > 0)
        if m.any():
            k_mid.append(0.5*(bins[i]+bins[i+1]))
            E_bin.append(np.mean(EB_flat[m]))

    k_mid = np.array(k_mid)
    E_bin = np.array(E_bin)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(k_mid, E_bin, 'o-', ms=4, label=r"$k^2|A_k|^2 \propto E_B(k)$")

    # fit slope in middle third
    lo, hi = len(k_mid)//3, 2*len(k_mid)//3
    if hi > lo + 2:
        slope, _ = np.polyfit(np.log(k_mid[lo:hi]), np.log(E_bin[lo:hi]), 1)
        k_ref = k_mid[(lo+hi)//2]
        E_ref = E_bin[(lo+hi)//2]
        norm  = E_ref / k_ref**(-5/3)
        ax.loglog(k_mid, norm * k_mid**(-5/3), '--',
                  label=rf"$k^{{-5/3}}$ (slope: {slope:.2f})")

    ax.axvline(kmin_nd, color='g', ls=':', label=f"kmin (2H={2*H_phys/kpc*1e3:.0f} pc)")
    ax.axvline(kmax_nd, color='r', ls=':', label="kmax (Nyquist)")
    ax.set_xlabel("k (dimensionless)")
    ax.set_ylabel(r"$E_B(k)$ [arb.]")
    ax.set_title("Magnetic energy spectrum from spectral coefficients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

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
    if width_cells == 0:
        return Aphi
    dR    = R_nd[1] - R_nd[0]
    width = width_cells * dR
    x     = (Rmax_nd - R_nd) / width
    taper = np.where(x >= 1.0, 1.0,
            np.where(x <= 0.0, 0.0,
                     0.5 * (1.0 - np.cos(np.pi * x))))
    return Aphi * taper[:, np.newaxis]

def apply_z_taper(Aphi, z_nd, Lz_nd, width_cells=8):
    """
    Cosine roll-off to zero over the top and bottom width_cells.
    """
    dz = z_nd[1] - z_nd[0]
    width = width_cells * dz
    z_max = Lz_nd / 2.0
    
    # Distance from top and bottom boundaries
    dist_top = z_max - z_nd
    dist_bot = z_nd + z_max
    
    # Taper function
    def taper_func(dist):
        x = dist / width
        return np.where(x >= 1.0, 1.0,
               np.where(x <= 0.0, 0.0,
                        0.5 * (1.0 - np.cos(np.pi * x))))
    
    taper = taper_func(dist_top) * taper_func(dist_bot)
    return Aphi * taper[np.newaxis, :]

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
#
#  The C++ face-var initialisation should use stem_Aphi.bin so that
#  the Cartesian projection  Ax = -Aphi*y/R,  Ay = Aphi*x/R  uses
#  the exact face geometry for x/R and y/R, avoiding the 1/R^2
#  amplification of interpolation error near the axis that occurs
#  when the RA table is used.
# ══════════════════════════════════════════════════════════════════

def save_outputs(Aphi_phys,
                 nR, nz, nR_coarse, nz_coarse, levels, oversample, SEED,
                 Rmax, Lz, dR, dz, kmin_nd, kmax_nd, Rmax_nd,
                 H_phys_cm, B0_gauss=1e-9, stem="Aphi_2d"):
    import os

    aphi_bin_path = stem + "_Aphi.bin"
    meta_path     = stem + "_meta.txt"

    # ── write Aphi table (used for face-centred B initialisation) ────────
    # Aphi_phys has units of [cm * normalised_A_phi / cm] = [normalised_A_phi]
    # i.e. it is RA_phys / R_physical.  Multiply by B0_HL * Rmax in C++ to
    # get physical A_phi in Gauss*cm (HL units).
    Aphi_phys.astype(np.float64).tofile(aphi_bin_path)
    print(f"Saved Aphi binary: {aphi_bin_path}  ({os.path.getsize(aphi_bin_path)/1e6:.1f} MB)")

    # sanity: Aphi should be 0 at the axis and smooth everywhere
    print(f"  Aphi[0,:]  mean |.|: {np.mean(np.abs(Aphi_phys[0,  :])):.3e}  (should be 0)")
    print(f"  Aphi[-1,:] mean |.|: {np.mean(np.abs(Aphi_phys[-1, :])):.3e}  (should be ~0)")
    print(f"  Aphi rms           : {np.sqrt(np.mean(Aphi_phys**2)):.4f}")
    print(f"  max |Aphi|         : {np.max(np.abs(Aphi_phys)):.4f}")

    # ── metadata ──────────────────────────────────────────────────────────
    B0_HL    = B0_gauss / (4.0 * np.pi) ** 0.5
    rms_B_HL = B0_HL

    lines = [
        f"seed_seed         = {SEED}",
        f"seed_nR           = {int(nR)}",
        f"seed_nz           = {int(nz)}",
        f"seed_nR_coarse    = {int(nR_coarse)}",
        f"seed_nz_coarse    = {int(nz_coarse)}",
        f"seed_amr_levels   = {int(levels)}",
        f"seed_oversample   = {oversample}",
        f"seed_Rmin         = {0.0:e}",
        f"seed_Rmax         = {Rmax:e}",
        f"seed_Lz           = {Lz:e}",
        f"seed_B0_HL        = {B0_HL:e}",
        f"seed_H_disk_cm    = {H_phys_cm:e}",
        f"seed_kmin_nd      = {kmin_nd:e}",
        f"seed_kmax_nd      = {kmax_nd:e}",
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
        f"layout            = C-order float64, shape ({int(nR)}, {int(nz)})",
        f"",
        f"# ── Aphi table ({aphi_bin_path}) ──────────────────────────────────",
        f"stored_field_Aphi = A_phi(R,z)  [normalised scalar, no R factor]",
        f"Aphi_usage        = face-centred B initialisation via Cartesian curl",
        f"Aphi_to_physical  = multiply by B0_HL * Rmax_cm to get A_phi in Gauss*cm (HL)",
        f"Aphi_axis_bc      = Aphi[0,:] = 0  (enforced; smooth at axis)",
        f"Aphi_outer_bc     = Aphi[-1,:] ~ 0 (cosine taper)",
        f"Ax_formula        = Ax(x,y,z) = -Aphi(R,z) * y/R   [Cartesian component]",
        f"Ay_formula        = Ay(x,y,z) =  Aphi(R,z) * x/R   [Cartesian component]",
        f"Az_formula        = Az = 0  (exact, axisymmetric construction)",
        f"Bx_face_formula   = [Ay(x_i,y_cc,z_lo) - Ay(x_i,y_cc,z_hi)] / dz",
        f"By_face_formula   = [Ax(x_cc,y_j,z_hi) - Ax(x_cc,y_j,z_lo)] / dz",
        f"Bz_face_formula   = [Ay(x_hi,y_cc,z_k)-Ay(x_lo,y_cc,z_k)]/dx - [Ax(x_cc,y_hi,z_k)-Ax(x_cc,y_lo,z_k)]/dy",
        f"",
        f"# ── shared metadata ─────────────────────────────────────────────",
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

    print(f"Saved metadata   : {meta_path}")
    return aphi_bin_path, meta_path

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Lz   = 20.0 * kpc              # physical box height
    Rmax = 10.0 * kpc #* np.sqrt(2)             # physical outer radius — box diagonal / 2

    # ── Disk scale height at R=Rd ─────────────────────────────────
    H_phys, Sigma_Rd = disk_scale_height(
        Sigma0, cs_disk, Rd_kpc, Rd_kpc, alpha_p, beta_p
    )
    print(f"Disk scale height at R=Rd ({Rd_kpc} kpc): {H_phys/kpc*1e3:.0f} pc")
    print(f"Surface density at R=Rd: {Sigma_Rd:.4e} g/cm^2")

    # ── work in dimensionless units: length scale = Rmax ─────────
    Rmax_nd = 1.0
    Lz_nd   = Lz / Rmax
    H_nd    = H_phys / Rmax

    # ── fine physical grid ────────────────────────────────────────
    # nR_sim/nz_sim are the simulation domain cell counts.
    # nR/nz add padding cells on each side so the C++ face stencils
    # at the domain boundary have valid neighbours in the table.
    factor  = 2 ** levels
    nz_sim  = int(nz_coarse * factor * oversample)
    nR_sim  = int(np.ceil(nz_sim * Rmax / Lz))
    nR      = nR_sim + 2 * padding
    nz      = nz_sim + 2 * padding
    print(f"Sim grid:    nR_sim={nR_sim}, nz_sim={nz_sim}")
    print(f"Table grid:  nR={nR}, nz={nz}  (padding={padding} each side)")

    # Cell spacing defined by the sim grid — padding cells inherit the same spacing
    dR = Rmax / nR_sim
    dz = Lz   / nz_sim

    dR_nd = Rmax_nd / nR_sim
    dz_nd = Lz_nd   / nz_sim

    # Coordinate arrays: padding cells sit outside [0, Rmax] x [-Lz/2, Lz/2]
    # so they are naturally zero after the ghost-cell zeroing step below.
    R_nd = (np.arange(nR) - padding + 0.5) * dR_nd
    z_nd = (np.arange(nz) - padding + 0.5) * dz_nd - Lz_nd / 2.0

    # Physical grids
    R = R_nd * Rmax
    z = z_nd * Rmax

    # Padded Rmax/Lz: extend domain by padding cells on each side so that
    # sample_bicubic's  dR = Rmax_table / nR_table  recovers the correct dR.
    Rmax_padded = Rmax + padding * dR
    Lz_padded   = Lz   + 2 * padding * dz

    # ── spectral modes ────────────────────────────────────────────
    kR, kz, k = init_modes(nR_coarse * factor, nz_coarse * factor, Rmax_nd, Lz_nd)

    kmin_nd = physical_kmin(H_nd)
    kmax_nd = physical_kmax(nR_sim, nz_sim, Rmax_nd, Lz_nd, n_cells=n_grid_cells)

    print(f"k range (nd): [{k.min():.3e}, {k.max():.3e}]")
    print(f"kmin (nd)   = {kmin_nd:.3e}  (lambda_max = 2H = {2*H_phys/kpc*1e3:.0f} pc)")
    print(f"kmax (nd)   = {kmax_nd:.3e}  (fine Nyquist diagonal)")
    print(f"kmax/kmin   = {kmax_nd/kmin_nd:.1f}")

    # ── spectral coefficients ─────────────────────────────────────
    coeff = init_coeffs(k, kmin_nd, kmax_nd, rng)
    coeff[k < kmin_nd] = 0
    plot_coeff_spectrum(k, coeff, kmin_nd, kmax_nd, H_phys, kpc)

    # ── evaluate A_phi ────────────────────────────────────────────
    print("Evaluating A_phi...")
    Aphi = evaluate_Aphi(R_nd, z_nd, kR, kz, coeff)

    Aphi /= np.sqrt(np.mean(Aphi**2))
    print(f"  Aphi rms after normalisation: {np.sqrt(np.mean(Aphi**2)):.3f}  (should be 1)")

    # ── boundary tapers ───────────────────────────────────────────
    inner_taper_cells = max(4, int(inner_taper_kpc * kpc / dR))
    outer_taper_cells = max(4, int(outer_taper_kpc * kpc / dR))
    #z_taper_cells     = max(4, int(z_taper_kpc * kpc / dz)) 

    print(f"  inner taper: {inner_taper_cells} cells = {inner_taper_cells*dR/kpc:.3f} kpc")
    print(f"  outer taper: {outer_taper_cells} cells = {outer_taper_cells*dR/kpc:.3f} kpc")
    #print(f"  z taper: {z_taper_cells} cells = {z_taper_cells*dz/kpc:.3f} kpc")

    Aphi = apply_axis_taper(Aphi, R_nd, dR_nd, n_cells=inner_taper_cells)
    Aphi = apply_outer_taper(Aphi, R_nd, Rmax_nd, width_cells=outer_taper_cells)

    # Zero padding cells that lie outside the physical domain.
    # The tapers drive Aphi to zero at the domain edges; this makes
    # the ghost cells explicitly zero so the C++ stencil sees a clean
    # roll-off rather than whatever the spectral basis left there.
    Aphi[R_nd < 0, :]                   = 0.0
    Aphi[R_nd > Rmax_nd, :]             = 0.0
    Aphi[:, np.abs(z_nd) > Lz_nd / 2]  = 0.0

    #print(f"  Aphi[0]  mean |.|: {np.mean(np.abs(Aphi[0,  :])):.3e}  (should be 0)")
    #print(f"  Aphi[-1] mean |.|: {np.mean(np.abs(Aphi[-1, :])):.3e}  (should be ~0)")

    # ── disk envelope: shapes rms_B(R) ~ sqrt(Sigma(R)) ─────────────
    # Since rms_Br(R) ~ Aphi(R)*kz_turb, multiplying Aphi by sqrt(Sigma(R))
    # directly gives rms_B(R) ~ sqrt(Sigma(R)) after the curl.
    # A linear ramp over the axis taper zone forces Aphi->0 at R=0
    # (Sigma is finite at the axis for this profile, so it can't do it alone).
    # x_env      = R_nd * Rmax / kpc / Rd_kpc
    # Sigma_env  = np.exp(-x_env - beta_p * np.exp(-alpha_p * x_env))
    # Sigma_Rd_v = np.exp(-1.0 - beta_p * np.exp(-alpha_p))
    # ramp_env   = np.minimum(R_nd / (inner_taper_cells * dR_nd), 1.0)
    # Aphi_env   = np.sqrt(Sigma_env / Sigma_Rd_v) * ramp_env
    # Aphi_env  /= Aphi_env[np.argmin(np.abs(R_nd - Rd_kpc * kpc / Rmax))]  # =1 at Rd
    # Aphi      *= Aphi_env[:, None]

    #Aphi[0, :]   = 0.0
    #Aphi[-1, :] *= 0.0

    #Aphi /= np.sqrt(np.mean(Aphi**2))
    #print(f"  Aphi rms after disk envelope: {np.sqrt(np.mean(Aphi**2)):.3f}  (should be 1)")

    # ── low-pass filter: sigma = 1 table cell ────────────────────
    from scipy.ndimage import gaussian_filter
    #Aphi = gaussian_filter(Aphi, sigma=1.0)
    #Aphi[0, :]   = 0.0
    #Aphi[-1, :] *= 0.0
    rms_after = np.sqrt(np.mean(Aphi**2))
    #print(f"  Aphi rms after smoothing: {rms_after:.3f}")
    #Aphi /= rms_after

    # ── Radial RMS equalization ───────────────────────────────────
    # The Bessel basis concentrates power near the axis. Divide each
    # radial ring by its own rms(z) so that the *curl* rms is roughly
    # flat in R before the global normalisation step below.
    # Smoothing the profile prevents the 1/rms amplification of axis
    # noise where only a few J1 modes contribute.
    from scipy.ndimage import uniform_filter1d
    rms_profile = np.sqrt(np.mean(Aphi**2, axis=1))
    rms_profile = uniform_filter1d(rms_profile, size=max(1, len(rms_profile)//50))
    rms_profile = np.maximum(rms_profile, 0.1 * rms_profile[len(rms_profile)//4:].mean())
    Aphi /= rms_profile[:, None]
    Aphi[R_nd <= 0, :] = 0.0

    # ── form RA = R * A_phi ───────────────────────────────────────
    RA = R_nd[:, None] * Aphi

    print(f"  RA at R=0:    {np.mean(np.abs(RA[0,  :])):.3e}  (should be 0)")
    print(f"  RA at Rmax:   {np.mean(np.abs(RA[-1, :])):.3e}  (should be ~0)")

    # ── curl ──────────────────────────────────────────────────────
    Br_nd, Bz_nd = curl_Aphi(RA, R_nd, dR_nd, dz_nd)

    # ── verify net flux ───────────────────────────────────────────
    net_flux_raw = np.mean(np.trapezoid(Bz_nd * R_nd[:, None], R_nd, axis=0))
    print(f"  Net flux (raw, nd):  {net_flux_raw:.3e}  (target: 0)")

    # ── normalise by curl rms so that rms(B_nd) = 1 ──────────────
    # This is the contract the C++ beta formula relies on: B_phys_rms = B0_scale.
    # Must be done AFTER equalization so both steps are self-consistent.
    rms_nd = np.sqrt(np.mean(Br_nd**2 + Bz_nd**2))
    print(f"  curl rms before normalisation: {rms_nd:.4f}")
    Aphi  /= rms_nd
    Br_nd /= rms_nd
    Bz_nd /= rms_nd

    Aphi_norm                    = Aphi
    Aphi_norm[R_nd <= 0, :]     = 0.0

    # RA_norm derived from the clean Aphi_norm (not the other way around)
    RA_norm         = R_nd[:, None] * Aphi_norm    
    Br = Br_nd
    Bz = Bz_nd    
    print(f"  RA_norm   rms: {np.sqrt(np.mean(RA_norm**2)):.4f}")
    print(f"  Aphi_norm rms: {np.sqrt(np.mean(Aphi_norm**2)):.4f}  (should be O(1))")
    print(f"  Aphi_norm[0]  mean |.|: {np.mean(np.abs(Aphi_norm[0,:])):.3e}  (should be 0)")

    # Verify curl recovers unit-rms B
    Br_check, Bz_check = curl_Aphi(RA_norm, R_nd, dR_nd, dz_nd)
    print(f"  curl(RA_norm) rms: {np.sqrt(np.mean(Br_check**2+Bz_check**2)):.4f}  (should be 1.0)")

    grad_rms = np.sqrt(np.mean(np.diff(Aphi_norm, axis=0)**2)) / dR_nd
    print(f"  grad rms of Aphi_norm (nd): {grad_rms:.4f}  (should be ~1)")

    Aphi_phys = Aphi_norm
    RA_phys   = RA_norm * Rmax

    # Radial Gaussian smooth (sigma=1 table cell) to suppress Bessel-mode
    # ringing that imprints as concentric Bphi rings in the XY plane.
    # Applied in R only so the z-structure and curl normalisation are unaffected.
    from scipy.ndimage import gaussian_filter1d
    Aphi_phys = gaussian_filter1d(Aphi_phys, sigma=1.0, axis=0)

    # Re-zero ghost cells and axis after smoothing.
    Aphi_phys[R_nd <= 0, :]                  = 0.0
    Aphi_phys[R_nd >= Rmax_nd, :]            = 0.0
    Aphi_phys[:, np.abs(z_nd) >= Lz_nd / 2] = 0.0

    print(f"  Aphi_phys rms: {np.sqrt(np.mean(Aphi_phys**2)):.4e} cm  (should be ~Rmax/nR ~ {Rmax/nR:.3e} cm)")

    print(f"  Br rms: {np.sqrt(np.mean(Br**2)):.4f}")
    print(f"  Bz rms: {np.sqrt(np.mean(Bz**2)):.4f}")
    print(f"  Total rms B: {np.sqrt(np.mean(Br**2 + Bz**2)):.4f}")

    # consistency check
    max_err = np.max(np.abs(R_nd[:, None] * Aphi_norm - RA_norm))
    print(f"  Consistency check R_nd*Aphi_norm == RA_norm: max error = {max_err:.3e}  (should be ~0 except row 0)")
    print(np.percentile(np.abs(Aphi_phys), [50, 95, 99, 99.9, 100]))

    Bz_rms_profile = np.sqrt(np.mean(Bz[:80]**2, axis=1))
    print(f"  Bz[0:80:8] rms = {Bz_rms_profile[::8]}")

    Bphi = np.zeros_like(Br)
    print(f"  Bphi max (should be 0): {np.max(np.abs(Bphi)):.3e}")

    # ── save ──────────────────────────────────────────────────────
    save_outputs( Aphi_phys, 
        nR=nR, nz=nz,
        nR_coarse=nR_coarse, nz_coarse=nz_coarse,
        levels=levels, oversample=oversample, SEED=SEED,
        Rmax=Rmax_padded, Lz=Lz_padded, dR=dR, dz=dz,
        kmin_nd=kmin_nd, kmax_nd=kmax_nd, Rmax_nd=Rmax_nd,
        H_phys_cm=H_phys,
        B0_gauss=B0, stem="Aphi_2d",
    )

    # Save Br and Bz tables directly instead of/in addition to Aphi
    # Br_phys = Br   # already unit-rms normalised dimensionless
    # Bz_phys = Bz

    # Br_phys.astype(np.float64).tofile("Br_2d.bin")
    # Bz_phys.astype(np.float64).tofile("Bz_2d.bin")

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

    # ── Aphi map ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(
        Aphi_phys.T,
        origin="lower",
        extent=[0, Rmax/kpc, -Lz/(2*kpc), Lz/(2*kpc)],
        aspect="auto",
        cmap="RdBu_r"
    )
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title(r"$A_\phi$ (normalised, saved to Aphi_2d_Aphi.bin)")
    axes[0].set_xlabel("R [kpc]")
    axes[0].set_ylabel("z [kpc]")

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

    if len(P_plot) > 1:
        mid  = len(P_plot) // 2
        norm = P_plot[mid] / k_plot[mid]**(-11/3)
        axes[1].loglog(k_plot, P_plot,              'o-', label="Measured")
        axes[1].loglog(k_plot, norm * k_plot**(-11/3), '--', label=r"$k^{-11/3}$")
        axes[1].axvline(kmin_nd, color='g', ls=':', label=f"kmin (λ=2H={2*H_phys/kpc*1e3:.0f} pc)")
        axes[1].set_xlabel("k (dimensionless)")
        axes[1].set_ylabel(r"$|A_k|^2$")
        axes[1].legend()

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
    print(f"\nOutput files:")
    print(f"  Aphi_2d_Aphi.bin  — Aphi table          (face-centred B init)")
    print(f"  Aphi_2d_meta.txt  — metadata for both")

    def check_parity(field_2d, label):
    # Check if field at y_max reflects properly
    # This assumes field_2d is centered correctly
        is_symmetric = np.allclose(field_2d[:, -1], field_2d[:, -2])
        print(f"Parity check for {label}: {'Symmetric' if is_symmetric else 'Antisymmetric'}")

    check_parity(Aphi, "Vector Potential A_phi")
