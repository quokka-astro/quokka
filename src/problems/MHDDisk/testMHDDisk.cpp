//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDDisk.cpp
/// \brief Defines a simulation using disk galaxy initial conditions.
///

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

namespace
{
constexpr double alpha_profile = 2.0;
constexpr double beta_profile = 0.5;
constexpr double q_flatten = 0.7;
constexpr double rho_transition = 1.0e-28;
constexpr double target_beta_seed = 1.0e3;
constexpr double axis_fallback_cells = 1.0;
constexpr double turb_target_Mach = 0.5;

constexpr double r_K_factor = 2.0;		// physical kernel radius, in units of dx (paper default 3.0)
constexpr int omega_subsamples = 4;		// n_sub per dimension for boundary-cell overlap quadrature
constexpr int initial_condition_subsamples = 4; // per dimension samples. 4 means 4³ samples
constexpr int refine_grid_subsamples = 4;	// per dimension samples. 4 means 4³ samples

constexpr int eq_table_nR = 512;		  // radii in the vertical-equilibrium table, R = 0 .. domain corner
constexpr int eq_table_nz = 512;		  // heights per radius, |z| = 0 .. zmax(R)
constexpr double eq_zmax_scale_heights = 30.0; // zmax(R) in units of the smaller of the self-gravity and halo scale heights
constexpr int eq_rk4_substeps = 2;		  // RK4 steps per table interval
constexpr int eq_bisection_steps = 40;		  // bisection on ln(rho_mid); the bracket is < 2 e-folds wide
constexpr int eq_pull_terms = 12;		  // terms of the exponential-sum disk pull, |c_n| < 1e-12 after 12
} // namespace

struct MHDGalaxy {};

static_assert(AMREX_SPACEDIM == 3, "MHD disk galaxy problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<MHDGalaxy> {
	static constexpr double gamma = 1.0001;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct HydroSystem_Traits<MHDGalaxy> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Particle_Traits<MHDGalaxy> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
};

template <> struct Physics_Traits<MHDGalaxy> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 0;
	static constexpr bool is_mhd_enabled = true;
};

template <> struct SimulationData<MHDGalaxy> {
	amrex::Real Rc;
	amrex::Real Rd;
	amrex::Real Rmax;
	amrex::Real Rcutoff;
	amrex::Real Q_mean;
	amrex::Real Mc = NAN; // one will be set
	amrex::Real vc = NAN; // the other will be calculated
	amrex::Real Sigma0;
	int subtract_disk_gravity = 0;
	int vertical_equilibrium_solve = 1; // 1: solved vertical equilibrium + modified-profile disk pull; 0: legacy sech^2 x halo + Freeman
	int rotation_on_cylinders = 0; // 1: disk rotates at the midplane v_phi(R) at every z, with the gas pull subtracted at the midplane
	amrex::Real eq_Rmax{};		     // [cm] outer radius of the equilibrium tables (domain corner)
	amrex::Gpu::DeviceVector<amrex::Real> eq_lnrho_device;	     // ln(rho) [eq_table_nR][eq_table_nz], see equilibriumDensity
	amrex::Gpu::DeviceVector<amrex::Real> eq_zmax_device;	     // [cm] table height per radius
	amrex::Gpu::DeviceVector<amrex::Real> eq_dlnrho_mid_dR_device; // [1/cm] d ln(rho_mid) / dR per radius
	// measured gas self-gravity g_R(R, |z|) from measure_gas_pull.py (mhd_galaxy.gas_pull_file), see gasPullTable
	amrex::Gpu::DeviceVector<amrex::Real> gas_pull_device;
	int gas_pull_nR = 0;
	int gas_pull_nz = 0;
	amrex::Real gas_pull_Rmax{};
	amrex::Real gas_pull_z0{};
	amrex::Real gas_pull_smax{};
	amrex::Real rho_cgm;
	amrex::Real rho_mid;
	amrex::Real T_disk = NAN;  // one will be set
	amrex::Real cs_disk = NAN; // the other will be calculated
	amrex::Real cs_eq = NAN;   // effective sound speed for the initial equilibrium: cs_disk, or with magnetic pressure folded in
	amrex::Real T_cgm = NAN;   // for both disk
	amrex::Real cs_cgm = NAN;  // and for cgm

	// 2D Cylindrical potential field variables, read from metadata file
	std::size_t seed_nR{};
	std::size_t seed_nz{};
	amrex::Real seed_Rmax{};
	amrex::Real seed_Lz{};
	amrex::Real seed_B0_HL{};
	std::string seed_str; // magnetic seed

	amrex::Gpu::DeviceVector<amrex::Real> Aphi_device;

	// Supernova feedback parameters
	amrex::Real sn_jeans_J;
	amrex::Real sn_momentum;
	amrex::Real star_formation_efficiency;
	amrex::Real sn_cluster_momentum_exponent;
	amrex::Real sn_mass_per_event_msun;
	amrex::Real sn_start_time = 0.0; // [s] no SN feedback before this simulation time
	amrex::Long sn_trigger_count_cumulative = 0; // cumulative no. of cells that crossed the Jeans trigger
						     // (secondary diagnostic only; not checkpoint-persistent)

	amrex::Vector<long long> turb_seeds; // NOLINT(google-runtime-int)
					     // seeds recorded by fieldgen_mpi when generating turb_v{x,y,z}; printed for reproducibility only

	// Owning GPU storage for the vx/vy/vz turbulence cubes loaded from binary files.
	// The generator writes double-precision arrays with dimensions
	// turb_nx x turb_ny x turb_nz. These device vectors are wrapped in
	// Array4/Table views during initialization sampling.
	amrex::Gpu::DeviceVector<amrex::Real> turb_vx_device;
	amrex::Gpu::DeviceVector<amrex::Real> turb_vy_device;
	amrex::Gpu::DeviceVector<amrex::Real> turb_vz_device;
	amrex::Real turb_rescale_factor{};

	int turb_nx{};
	int turb_ny{};
	int turb_nz{};

	// cylinder dimensions
	amrex::Real refine_Rcyl{};
	amrex::Real refine_Hcyl{};
	amrex::Real refine_shrink_Rcyl{};
	amrex::Real refine_shrink_Hcyl{};
	amrex::Real refine_Rcyl_inner{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto vcircAnalytic(double R, double z, double vc, double Rc) -> double
{
	const double D = std::sqrt(R * R + Rc * Rc + (z / q_flatten) * (z / q_flatten));
	return vc * R / D;
}

// Epicyclic frequency for Toomre condition matching
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto kappa(double R, double vc, double Rc) -> double
{
#define Ω(R) vcircAnalytic(R, 0.0, vc, Rc) / (R)
	// const auto Ω = [=] AMREX_GPU_DEVICE(double R, double vc, double Rc) { return vcircAnalytic(R, 0.0, vc, Rc) / R; };
	const double ε = 0.5e-5 * Rc;
	const double dΩdR = (Ω(R + ε) - Ω(R - ε)) / (2.0 * ε);		       // numeric derivative of Ω
	const double κ = std::sqrt(4.0 * Ω(R) * Ω(R) + 2.0 * R * Ω(R) * dΩdR); // sqrt yields nan if κ negative
	return κ;
#undef Ω
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto surfaceDensityProfile(double R, double Rc, double Rd, double Sigma0) -> double
{
	const double x = R / Rd;
	return Sigma0 * std::exp(-x - beta_profile * std::exp(-alpha_profile * x));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto diskDensityAnalytic(double R, double z, double Rc, double Rd, double Sigma0, double vc, double cs) -> double
{
	const double Sigma = surfaceDensityProfile(R, Rc, Rd, Sigma0);
	if (Sigma <= 0.0) {
		return 0.0;
	}

	const double H = cs * cs / (M_PI * C::Gconst * Sigma);

	const double sech = 1.0 / std::cosh(z / H);
	const double disk_factor = (sech * sech) / (2.0 * H);

	const double denom = R * R + Rc * Rc;
	const double halo_factor = pow(1.0 + (z * z) / (q_flatten * q_flatten * denom), -(vc / cs) * (vc / cs) / 2.0);

	return Sigma * disk_factor * halo_factor;
}

// Modified Bessel functions I0, I1, K0, K1 for x > 0 (Abramowitz & Stegun 9.8.1-9.8.8, ~1e-7),
// usable in device code, unlike std::cyl_bessel_i/k.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto besselI0(double x) -> double
{
	if (x < 3.75) {
		const double t = (x / 3.75) * (x / 3.75);
		return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.360768e-1 + t * 0.45813e-2)))));
	}
	const double t = 3.75 / x;
	return (std::exp(x) / std::sqrt(x)) *
	       (0.39894228 + t * (0.1328592e-1 + t * (0.225319e-2 + t * (-0.157565e-2 + t * (0.916281e-2 + t * (-0.2057706e-1 +
		t * (0.2635537e-1 + t * (-0.1647633e-1 + t * 0.392377e-2))))))));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto besselI1(double x) -> double
{
	if (x < 3.75) {
		const double t = (x / 3.75) * (x / 3.75);
		return x * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (0.2658733e-1 + t * (0.301532e-2 + t * 0.32411e-3))))));
	}
	const double t = 3.75 / x;
	const double tail = 0.2282967e-1 + t * (-0.2895312e-1 + t * (0.1787654e-1 - t * 0.420059e-2));
	return (std::exp(x) / std::sqrt(x)) * (0.39894228 + t * (-0.3988024e-1 + t * (-0.362018e-2 + t * (0.163801e-2 + t * (-0.1031555e-1 + t * tail)))));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto besselK0(double x) -> double
{
	if (x <= 2.0) {
		const double t = x * x / 4.0;
		return (-std::log(x / 2.0) * besselI0(x)) +
		       (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.3488590e-1 + t * (0.262698e-2 + t * (0.10750e-3 + t * 0.74e-5))))));
	}
	const double t = 2.0 / x;
	return (std::exp(-x) / std::sqrt(x)) *
	       (1.25331414 + t * (-0.7832358e-1 + t * (0.2189568e-1 + t * (-0.1062446e-1 + t * (0.587872e-2 + t * (-0.251540e-2 + t * 0.53208e-3))))));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto besselK1(double x) -> double
{
	if (x <= 2.0) {
		const double t = x * x / 4.0;
		return (std::log(x / 2.0) * besselI1(x)) +
		       (1.0 / x) * (1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897 + t * (-0.1919402e-1 + t * (-0.110404e-2 + t * (-0.4686e-4)))))));
	}
	const double t = 2.0 / x;
	return (std::exp(-x) / std::sqrt(x)) *
	       (1.25331414 + t * (0.23498619 + t * (-0.3655620e-1 + t * (0.1504268e-1 + t * (-0.780353e-2 + t * (0.325614e-2 + t * (-0.68245e-3)))))));
}

// Inward midplane pull V_g^2/R of a thin exponential (Freeman) disk, Wang et al. (2010) eq. 29:
// V_g^2 = 4 pi G Sigma0 Rd y^2 [I0K0 - I1K1], y = R/(2 Rd)  =>  V_g^2/R = pi G Sigma0 (R/Rd) [I0K0 - I1K1].
// This is the disk term Arora et al. subtract from phi_dm (eq. A.5; their eq. A.2 prints V_g^2 as "phi_g").
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto freemanDiskPull(double R, double Sigma0, double Rd) -> double
{
	const double y = R / (2.0 * Rd);
	if (y < 1.0e-8) {
		return 0.0;
	}
	return M_PI * C::Gconst * Sigma0 * (R / Rd) * (besselI0(y) * besselK0(y) - besselI1(y) * besselK1(y));
}

// Inward midplane pull of a thin disk with the modified profile of surfaceDensityProfile. Expanding
// exp(-beta e^{-alpha x}) = sum_n (-beta)^n / n! e^{-alpha n x} writes Sigma as a sum of exponential disks,
// Sigma = Sigma0 sum_n c_n e^{-(1 + alpha n) R / Rd} with c_n = (-beta)^n / n!, so the pull is the same sum of
// Freeman terms. Exact for the thin disk up to the truncation at eq_pull_terms.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto surfaceDensityPull(double R, double Sigma0, double Rd) -> double
{
	double pull = 0.0;
	double c_n = 1.0;
	for (int n = 0; n < eq_pull_terms; ++n) {
		pull += freemanDiskPull(R, Sigma0 * c_n, Rd / (1.0 + alpha_profile * n));
		c_n *= -beta_profile / static_cast<double>(n + 1);
	}
	return pull;
}

// psi(R, z) - psi(R, 0) for the flattened log potential psi = vc^2 / 2 ln(R^2 + Rc^2 + z^2 / q^2) of vcircAnalytic
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto haloPotentialRise(double R, double z, double vc, double Rc) -> double
{
	return 0.5 * vc * vc * std::log1p((z / q_flatten) * (z / q_flatten) / (R * R + Rc * Rc));
}

// Density of the solved vertical equilibrium: ln(rho) bilinear in (R, |z| / zmax(R)) from the table built by
// solveVerticalEquilibrium, row-major [nR][nz]. Clamped at the table edges, where the density is negligible
// (|z| = zmax is eq_zmax_scale_heights scale heights up; R = Rmax is the domain corner).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto equilibriumDensity(const amrex::Real *lnrho, const amrex::Real *zmax, int nR, int nz, double Rmax, double R,
								  double z) -> double
{
	const double fR = std::min(R / Rmax, 1.0) * (nR - 1);
	const int i0 = std::min(static_cast<int>(fR), nR - 2);
	const double tR = fR - i0;
	auto row = [=](int i) -> double {
		const double fz = std::min(std::abs(z) / zmax[i], 1.0) * (nz - 1);
		const int j0 = std::min(static_cast<int>(fz), nz - 2);
		const double tz = fz - j0;
		return ((1.0 - tz) * lnrho[(i * nz) + j0]) + (tz * lnrho[(i * nz) + j0 + 1]);
	};
	return std::exp(((1.0 - tR) * row(i0)) + (tR * row(i0 + 1)));
}

// Linear interpolation in R of a radial table on nR points over [0, Rmax], clamped at Rmax
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpRadial(const amrex::Real *table, int nR, double Rmax, double R) -> double
{
	const double fR = std::min(R / Rmax, 1.0) * (nR - 1);
	const int i0 = std::min(static_cast<int>(fR), nR - 2);
	const double tR = fR - i0;
	return ((1.0 - tR) * table[i0]) + (tR * table[i0 + 1]);
}

// Measured gas self-gravity g_R(R, |z|) (< 0 inward), bilinear in (R, asinh(|z| / z0)) on the table written by
// measure_gas_pull.py: nodes R_i = i Rmax / (nR - 1), |z|_j = z0 sinh(j smax / (nz - 1)), row-major [nR][nz], clamped.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto gasPullTable(const amrex::Real *table, int nR, int nz, double Rmax, double z0, double smax, double R,
							    double z) -> double
{
	const double fR = std::min(R / Rmax, 1.0) * (nR - 1);
	const int i0 = std::min(static_cast<int>(fR), nR - 2);
	const double tR = fR - i0;
	const double fz = std::min(std::asinh(std::abs(z) / z0) / smax, 1.0) * (nz - 1);
	const int j0 = std::min(static_cast<int>(fz), nz - 2);
	const double tz = fz - j0;
	auto at = [=](int i, int j) { return table[(i * nz) + j]; };
	return ((1.0 - tR) * (((1.0 - tz) * at(i0, j0)) + (tz * at(i0, j0 + 1)))) + (tR * (((1.0 - tz) * at(i0 + 1, j0)) + (tz * at(i0 + 1, j0 + 1))));
}

// Read the table written by measure_gas_pull.py (two comment lines, "nR nz Rmax_cm z0_cm smax", nR * nz values)
inline void loadGasPullTable(const std::string &path, SimulationData<MHDGalaxy> &ud)
{
	std::ifstream f(path);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.is_open(), ("Cannot open gas_pull_file " + path).c_str());
	std::string line;
	while (f.peek() == '#' && std::getline(f, line)) {
	}
	f >> ud.gas_pull_nR >> ud.gas_pull_nz >> ud.gas_pull_Rmax >> ud.gas_pull_z0 >> ud.gas_pull_smax;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f && ud.gas_pull_nR >= 2 && ud.gas_pull_nz >= 2, ("Bad header in gas_pull_file " + path).c_str());
	std::vector<amrex::Real> host(static_cast<std::size_t>(ud.gas_pull_nR) * ud.gas_pull_nz);
	for (auto &v : host) {
		f >> v;
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<bool>(f), ("gas_pull_file " + path + " has fewer values than nR * nz").c_str());
	ud.gas_pull_device.resize(host.size());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), ud.gas_pull_device.begin());
	amrex::Gpu::synchronize();
}

// Isothermal vertical hydrostatic equilibrium at radius R (the vertical part of Wang et al. 2010), in the halo
// potential plus the gas's own gravity in the slab approximation:
//   cs^2 d ln(rho) / dz = -d psi / dz - d Phi_gas / dz,   d Phi_gas / dz = 4 pi G M(z),   M(z) = int_0^z rho dz'
// so ln(rho) = ln(rho_mid) - [psi(R, z) - psi(R, 0) + Phi_gas(z)] / cs^2. rho_mid is found by bisection so that the
// column 2 M(zmax) equals Sigma. Fills ln_rho on nz points over [0, zmax] and returns zmax.
inline auto solveVerticalEquilibrium(double R, double Sigma, double vc, double Rc, double cs, int nz, std::vector<double> &ln_rho) -> double
{
	const double cs2 = cs * cs;
	const double fourPiG = 4.0 * M_PI * C::Gconst;
	const double H_sg = cs2 / (M_PI * C::Gconst * Sigma);			// sech^2 scale height, self-gravity only
	const double h_halo = q_flatten * std::sqrt((R * R) + (Rc * Rc)) * cs / vc; // Gaussian width, halo only
	const double zmax = eq_zmax_scale_heights * std::min(H_sg, h_halo);
	const double h = zmax / (nz - 1) / eq_rk4_substeps;

	// RK4 on (Phi_gas, M) from the midplane; returns the column and optionally stores ln(rho)
	auto column = [&](double ln_rho_mid, bool store) -> double {
		auto rho_at = [&](double z, double phi) { return std::exp(ln_rho_mid - ((haloPotentialRise(R, z, vc, Rc) + phi) / cs2)); };
		double phi = 0.0;
		double M = 0.0;
		double z = 0.0;
		if (store) {
			ln_rho[0] = ln_rho_mid;
		}
		for (int j = 1; j < nz; ++j) {
			for (int s = 0; s < eq_rk4_substeps; ++s) {
				const double k1p = fourPiG * M;
				const double k1m = rho_at(z, phi);
				const double k2p = fourPiG * (M + (0.5 * h * k1m));
				const double k2m = rho_at(z + (0.5 * h), phi + (0.5 * h * k1p));
				const double k3p = fourPiG * (M + (0.5 * h * k2m));
				const double k3m = rho_at(z + (0.5 * h), phi + (0.5 * h * k2p));
				const double k4p = fourPiG * (M + (h * k3m));
				const double k4m = rho_at(z + h, phi + (h * k3p));
				phi += h / 6.0 * (k1p + (2.0 * k2p) + (2.0 * k3p) + k4p);
				M += h / 6.0 * (k1m + (2.0 * k2m) + (2.0 * k3m) + k4m);
				z += h;
			}
			if (store) {
				ln_rho[j] = ln_rho_mid - ((haloPotentialRise(R, z, vc, Rc) + phi) / cs2);
			}
		}
		return 2.0 * M;
	};

	// Each force alone gives a lower midplane density than both together, and their sum an upper bound
	const double rho_sg = M_PI * C::Gconst * Sigma * Sigma / (2.0 * cs2);
	const double rho_halo = Sigma / (std::sqrt(2.0 * M_PI) * h_halo);
	double lo = std::log(0.5 * std::max(rho_sg, rho_halo));
	double hi = std::log(2.0 * (rho_sg + rho_halo));
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(column(lo, false) < Sigma && column(hi, false) > Sigma, "vertical equilibrium: rho_mid bracket failed");
	for (int it = 0; it < eq_bisection_steps; ++it) {
		const double mid = 0.5 * (lo + hi);
		(column(mid, false) < Sigma ? lo : hi) = mid;
	}
	column(0.5 * (lo + hi), true);
	return zmax;
}

// 1D and 2D Interpolation Operators for Cylindrical A_phi Evaluation

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cubic_interp(double p0, double p1, double p2, double p3, double t) -> double
{
	const double a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
	const double a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
	const double a2 = -0.5 * p0 + 0.5 * p2;
	const double a3 = p1;
	return ((a0 * t + a1) * t + a2) * t + a3;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sample_bicubic(const amrex::Real *table, int nR, int nz, double Rmax, double Lz, double R, double z) -> double
{
	const double zmin = -0.5 * Lz;

	// Matches Python cell-centered spacing delta definition
	const double dR = Rmax / static_cast<double>(nR);
	const double dz = Lz / static_cast<double>(nz);

	// Exact Boundary Zeroing out
	if (R < 0.0 || R >= Rmax || z <= zmin || z >= 0.5 * Lz) {
		return 0.0;
	}
	if (R < 1e-12 * Rmax) {
		return 0.0;
	}

	// Map to cell-centered coordinates
	const double fR = (R / dR) - 0.5;
	const double fz = ((z - zmin) / dz) - 0.5;

	int i = static_cast<int>(std::floor(fR));
	int j = static_cast<int>(std::floor(fz));

	// Fallback to bilinear interpolation near the axis to prevent numerical overshoot
	if (i < 2) {
		i = amrex::max(0, amrex::min(i, nR - 2));
		j = amrex::max(0, amrex::min(j, nz - 2));
		const double tR = fR - static_cast<double>(i);
		const double tZ = fz - static_cast<double>(j);
		auto idx = [&](int ii, int jj) -> double { return table[ii * nz + jj]; };
		return (1.0 - tR) * (1.0 - tZ) * idx(i, j) + tR * (1.0 - tZ) * idx(i + 1, j) + (1.0 - tR) * tZ * idx(i, j + 1) + tR * tZ * idx(i + 1, j + 1);
	}

	const double tR = fR - static_cast<double>(i);
	const double tZ = fz - static_cast<double>(j);

	auto idx = [&](int ii, int jj) -> double {
		ii = amrex::max(0, amrex::min(ii, nR - 1));
		jj = amrex::max(0, amrex::min(jj, nz - 1));
		return table[ii * nz + jj];
	};

	std::array<double, 4> col{};
	for (int m = -1; m <= 2; ++m) {
		col[m + 1] = cubic_interp(idx(i - 1, j + m), idx(i, j + m), idx(i + 1, j + m), idx(i + 2, j + m), tR);
	}
	return cubic_interp(col[0], col[1], col[2], col[3], tZ);
}

AMREX_GPU_HOST_DEVICE
inline auto interpolate_turbulence(const amrex::Real *table, int nx, int ny, int nz, amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real
{
	x = amrex::max(0.0, amrex::min(x, static_cast<amrex::Real>(nx - 1)));
	y = amrex::max(0.0, amrex::min(y, static_cast<amrex::Real>(ny - 1)));
	z = amrex::max(0.0, amrex::min(z, static_cast<amrex::Real>(nz - 1)));

	const int i0 = static_cast<int>(x);
	const int j0 = static_cast<int>(y);
	const int k0 = static_cast<int>(z);

	const int i1 = amrex::min(i0 + 1, nx - 1);
	const int j1 = amrex::min(j0 + 1, ny - 1);
	const int k1 = amrex::min(k0 + 1, nz - 1);

	const amrex::Real fx = x - i0;
	const amrex::Real fy = y - j0;
	const amrex::Real fz = z - k0;

	// Row-major (C-order) with k fastest-varying, matching fieldgen_mpi/fieldgen3.c's
	// writeData(): the loop nest is i (outer) -> j -> k (contiguous fwrite of `ngrid`
	// doubles), so the on-disk/flat-buffer layout is [i][j][k], not AMReX's Array4
	// convention (i fastest). Indexing this buffer via amrex::Array4 -- as earlier
	// code here did -- silently transposes the i and k axes on read.
	auto idx = [ny, nz](int i, int j, int k) -> std::size_t { return (static_cast<std::size_t>(i) * ny + j) * nz + k; };

	auto c000 = table[idx(i0, j0, k0)];
	auto c100 = table[idx(i1, j0, k0)];
	auto c010 = table[idx(i0, j1, k0)];
	auto c110 = table[idx(i1, j1, k0)];
	auto c001 = table[idx(i0, j0, k1)];
	auto c101 = table[idx(i1, j0, k1)];
	auto c011 = table[idx(i0, j1, k1)];
	auto c111 = table[idx(i1, j1, k1)];

	return c000 * (1 - fx) * (1 - fy) * (1 - fz) + c100 * fx * (1 - fy) * (1 - fz) + c010 * (1 - fx) * fy * (1 - fz) + c110 * fx * fy * (1 - fz) +
	       c001 * (1 - fx) * (1 - fy) * fz + c101 * fx * (1 - fy) * fz + c011 * (1 - fx) * fy * fz + c111 * fx * fy * fz;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto get_taper_factor(double x, double y, double z, double Rmax, double Lz, double dx, double dy, double dz) -> double
{
	const double R = std::sqrt(x * x + y * y);
	const double absZ = std::abs(z);

	// Taper parameters: adjust n_taper to change the steepness of the fall-off
	const double n_taper = 4.0;
	const double R_taper_start = Rmax - n_taper * amrex::max(dx, dy);
	const double Z_taper_start = 0.5 * Lz - n_taper * dz;

	double taper = 1.0;

	// Smooth taper for outer R and Z boundaries
	if (R > R_taper_start) {
		taper *= 0.5 * (1.0 - std::cos(M_PI * (Rmax - R) / (Rmax - R_taper_start)));
	}
	if (absZ > Z_taper_start) {
		taper *= 0.5 * (1.0 - std::cos(M_PI * (0.5 * Lz - absZ) / (0.5 * Lz - Z_taper_start)));
	}

	// Hard mask for extreme domain overflow to prevent NaN/Inf
	if (R >= Rmax || absZ >= 0.5 * Lz) {
		return 0.0;
	}

	// Taper near axis to avoid singularity
	const double R_axis_thresh = 1e-6 * dx;
	if (R < R_axis_thresh) {
		return 0.0;
	}

	return taper;
}

inline int getNoCells(const QuokkaSimulation<MHDGalaxy> &sim)
{
	long cells = 0;
	for (int level = 0; level <= sim.finestLevel(); ++level)
		for (amrex::MFIter iter(sim.state_new_cc_[level]); iter.isValid(); ++iter)
			cells += iter.validbox().numPts();
	return cells;
}

inline auto load_bin_to_device(const std::string &path, std::size_t n_expect) -> amrex::Gpu::DeviceVector<amrex::Real>
{
	// Use amrex::Real so this remains compatible if you change precision
	std::vector<amrex::Real> host(n_expect);
	std::ifstream f(path, std::ios::binary);

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f, ("Cannot open " + path).c_str());
	const std::size_t total_bytes = n_expect * sizeof(amrex::Real);
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, cppcoreguidelines-narrowing-conversions)
	f.read(reinterpret_cast<char *>(host.data()), static_cast<std::streamsize>(total_bytes));

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f, ("Error reading " + path).c_str());

	// Allocate on device and copy
	amrex::Gpu::DeviceVector<amrex::Real> dev(n_expect);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), dev.begin());

	// Synchronize to ensure data is ready before proceeding
	amrex::Gpu::synchronize();

	amrex::Print() << "Loaded " << path << " (" << n_expect << " elements)\n";
	return dev;
}

inline auto load_turb_seeds(const std::string &path) -> amrex::Vector<long long>
{
	amrex::Vector<long long> seeds; // NOLINT(google-runtime-int)
	std::ifstream f(path);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f, ("Cannot open turbulence seed file " + path).c_str());
	std::string line;
	while (std::getline(f, line)) {
		const std::size_t start = line.find_first_not_of(" \t\r\n");
		if (start == std::string::npos) {
			continue;
		}
		const std::size_t end = line.find_last_not_of(" \t\r\n");
		line = line.substr(start, end - start + 1);
		if (line.empty() || line[0] == '#') {
			continue;
		}
		seeds.push_back(std::stoll(line));
	}
	return seeds;
}

// Fraction of the cell offset (di,dj,dk) from the trigger cell that overlaps a sphere
// of radius rK centred on the trigger cell, used to weight the SN kernel deposit.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto cellSphereOverlapFraction(double di, double dj, double dk, double dxu, double dyu, double dzu, double rK) -> double
{
	const double x0 = di * dxu - 0.5 * dxu;
	const double x1 = di * dxu + 0.5 * dxu;
	const double y0 = dj * dyu - 0.5 * dyu;
	const double y1 = dj * dyu + 0.5 * dyu;
	const double z0 = dk * dzu - 0.5 * dzu;
	const double z1 = dk * dzu + 0.5 * dzu;
	const double cx = amrex::max(x0, amrex::min(0.0, x1));
	const double cy = amrex::max(y0, amrex::min(0.0, y1));
	const double cz = amrex::max(z0, amrex::min(0.0, z1));
	// Fast path: nearest point of the cell box to the origin is already outside rK,
	// so the cell can't overlap the sphere at all.
	if (cx * cx + cy * cy + cz * cz > rK * rK) {
		return 0.0;
	}

	const double fx = amrex::max(std::abs(x0), std::abs(x1));
	const double fy = amrex::max(std::abs(y0), std::abs(y1));
	const double fz = amrex::max(std::abs(z0), std::abs(z1));
	// Fast path: farthest point of the cell box from the origin is still inside rK,
	// so the cell is fully covered -- skip the subsampled quadrature below.
	if (fx * fx + fy * fy + fz * fz <= rK * rK) {
		return 1.0;
	}

	constexpr int n_sub = omega_subsamples;
	int count = 0;
	for (int a = 0; a < n_sub; ++a) {
		const double xp = x0 + (a + 0.5) * (dxu / n_sub);
		for (int b = 0; b < n_sub; ++b) {
			const double yp = y0 + (b + 0.5) * (dyu / n_sub);
			for (int c = 0; c < n_sub; ++c) {
				const double zp = z0 + (c + 0.5) * (dzu / n_sub);
				if (xp * xp + yp * yp + zp * zp <= rK * rK) {
					++count;
				}
			}
		}
	}
	return static_cast<double>(count) / static_cast<double>(n_sub * n_sub * n_sub);
}

template <> void QuokkaSimulation<MHDGalaxy>::preCalculateInitialConditions()
{
	amrex::Print() << std::format("{:6.3f}M cells\n", amrex::Real(getNoCells(*this)) / 1e6);

	// run only once
	static bool was_called = false;
	if (was_called) {
		return;
	}
	was_called = true;

	amrex::ParmParse const pp("mhd_galaxy");
	// clang-format off
	pp.get("Rc_kpc",      userData_.Rc);	  userData_.Rc      *= 1.0e3 * C::parsec;
	pp.get("Rd_kpc",      userData_.Rd);	  userData_.Rd      *= 1.0e3 * C::parsec;
	pp.get("Rmax_kpc",    userData_.Rmax);	  userData_.Rmax    *= 1.0e3 * C::parsec;
	pp.get("Rcutoff_kpc", userData_.Rcutoff); userData_.Rcutoff *= 1.0e3 * C::parsec;
	pp.query("vc_kms",    userData_.vc);      userData_.vc      *= 1e5;
	pp.query("cs_disk_kms",userData_.cs_disk);userData_.cs_disk *= 1e5;
	pp.query("cs_cgm_kms",userData_.cs_cgm);  userData_.cs_cgm  *= 1e5;
	pp.query("Mc", userData_.Mc);
	pp.query("T_disk", userData_.T_disk);
	pp.query("T_cgm", userData_.T_cgm);
	pp.get("Q_mean", userData_.Q_mean);
	pp.query("subtract_disk_gravity", userData_.subtract_disk_gravity);
	pp.query("vertical_equilibrium_solve", userData_.vertical_equilibrium_solve);
	pp.query("rotation_on_cylinders", userData_.rotation_on_cylinders);
	pp.get("sn_jeans_J", userData_.sn_jeans_J);
	pp.get("sn_momentum", userData_.sn_momentum);
	pp.get("star_formation_efficiency", userData_.star_formation_efficiency);
	pp.get("sn_mass_per_event_msun", userData_.sn_mass_per_event_msun);
	pp.get("sn_cluster_momentum_exponent", userData_.sn_cluster_momentum_exponent);
	constexpr amrex::Real seconds_per_myr = 3.15576e13;
	pp.query("sn_start_time_myr", userData_.sn_start_time); userData_.sn_start_time *= seconds_per_myr;
	pp.get("refine_Rcyl_kpc",                  userData_.refine_Rcyl);        userData_.refine_Rcyl        *= 1.0e3 * C::parsec;
	pp.get("refine_Hcyl_kpc",                  userData_.refine_Hcyl);        userData_.refine_Hcyl        *= 1.0e3 * C::parsec;
	pp.get("refine_Rcyl_shrink_per_level_kpc", userData_.refine_shrink_Rcyl); userData_.refine_shrink_Rcyl *= 1.0e3 * C::parsec;
	pp.get("refine_Hcyl_shrink_per_level_kpc", userData_.refine_shrink_Hcyl); userData_.refine_shrink_Hcyl *= 1.0e3 * C::parsec;
	pp.get("refine_Rcyl_inner_kpc",            userData_.refine_Rcyl_inner);  userData_.refine_Rcyl_inner  *= 1.0e3 * C::parsec;
	// clang-format on

	constexpr double gamma = quokka::EOS_Traits<MHDGalaxy>::gamma;
	constexpr double µ = quokka::EOS_Traits<MHDGalaxy>::mean_molecular_weight;

	// set cs or T from either T or cs, to ensure both are present
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isnan(userData_.T_disk) != std::isnan(userData_.cs_disk),
					 "Must set exactly one of mhd_galaxy.T_disk or mhd_galaxy.cs_disk");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isnan(userData_.T_cgm) != std::isnan(userData_.cs_cgm),
					 "Must set exactly one of mhd_galaxy.T_cgm or mhd_galaxy.cs_cgm");

	if (std::isnan(userData_.cs_disk)) {
		userData_.cs_disk = std::sqrt(gamma * C::k_B * userData_.T_disk / µ);
	} else {
		userData_.T_disk = userData_.cs_disk * userData_.cs_disk * µ / (gamma * C::k_B);
	}
	if (std::isnan(userData_.cs_cgm)) {
		userData_.cs_cgm = std::sqrt(gamma * C::k_B * userData_.T_cgm / µ);
	} else {
		userData_.T_cgm = userData_.cs_cgm * userData_.cs_cgm * µ / (gamma * C::k_B);
	}

	const double cs_disk = userData_.cs_disk;
	const double cs_cgm = userData_.cs_cgm;

	// set vc or Mc from either Mc or vc, to ensure both are present
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isnan(userData_.vc) != std::isnan(userData_.Mc), "Must set exactly one of mhd_galaxy.vc_kms or mhd_galaxy.Mc");
	if (std::isnan(userData_.vc)) {
		userData_.vc = userData_.Mc * userData_.cs_disk;
	} else {
		userData_.Mc = userData_.vc / userData_.cs_disk;
	}
	const double vc = userData_.vc;

	// Magnetic pressure in the initial equilibrium (cf. Arora et al.): approximate the Lorentz force by the
	// gradient of a magnetic pressure at uniform plasma beta, P_mag = P_th / beta, with the same beta
	// definition as the seed normalisation below, beta = (rho cs^2 / gamma) / (B^2 / 2). Then the total
	// pressure is rho cs_eq^2 with cs_eq^2 = cs^2 (1 + 1 / (gamma beta)), which replaces cs^2 in the vertical
	// structure and the radial pressure term. Tension is neglected. Note the random A_phi seed is not
	// B ~ sqrt(rho), so this matches its actual magnetic pressure only on average.
	int magnetic_pressure_balance = 0;
	pp.query("magnetic_pressure_balance", magnetic_pressure_balance);
	userData_.cs_eq = userData_.cs_disk;
	if (magnetic_pressure_balance != 0) {
		userData_.cs_eq = userData_.cs_disk * std::sqrt(1.0 + 1.0 / (gamma * target_beta_seed));
	}
	amrex::Print() << "Equilibrium sound speed cs_eq = " << userData_.cs_eq / 1.0e5 << " km/s (thermal cs_disk = " << userData_.cs_disk / 1.0e5
		       << " km/s, magnetic pressure " << (magnetic_pressure_balance != 0 ? "included" : "not included") << ")\n";

	const double Rc = userData_.Rc;
	const double Rd = userData_.Rd;
	const double Rmax = userData_.Rmax;

	// Sigma0 via Simpson integration of Toomre Q condition, accurate to 10 figures for the default functions
	auto integrand = [=](double R) -> double { return kappa(R, vc, Rc) * cs_disk / (M_PI * C::Gconst * surfaceDensityProfile(R, Rc, Rd, 1.0)); };
	constexpr int N = 100000;
	static_assert(N % 2 == 0);
	const double h = Rmax / N;
	double integral = integrand(1e-20 * Rmax) + integrand(Rmax);
	for (int i = 1; i < N; ++i) {
		integral += (i % 2 == 0 ? 2.0 : 4.0) * integrand(i * h);
	}
	integral *= h / 3.0;
	// set densityProfile factor such that the integral equals Q_mean
	userData_.Sigma0 = integral / (userData_.Q_mean * Rmax);
	userData_.rho_cgm = rho_transition * (cs_disk * cs_disk) / (cs_cgm * cs_cgm); // rho_cgm such that equal pressure at boundary (~3.54e-32)
	userData_.rho_cgm = std::max(userData_.rho_cgm, dustDensityFloor_);	      // cgm density is at least the density floor

	// Vertical equilibrium table: solve each column so that it holds exactly Sigma(R), in the same forces the run
	// applies (full halo g_z plus live self-gravity), out to the domain corner. Every rank builds the same table.
	std::vector<double> ln_rho_mid(eq_table_nR, 0.0);
	if (userData_.vertical_equilibrium_solve != 0) {
		const auto lo = geom[0].ProbLoArray();
		const auto hi = geom[0].ProbHiArray();
		const double Rx = std::max(std::abs(lo[0]), std::abs(hi[0]));
		const double Ry = std::max(std::abs(lo[1]), std::abs(hi[1]));
		userData_.eq_Rmax = std::sqrt((Rx * Rx) + (Ry * Ry));
		const double eq_Rmax = userData_.eq_Rmax;
		const double dR = eq_Rmax / (eq_table_nR - 1);

		std::vector<amrex::Real> lnrho_host(static_cast<std::size_t>(eq_table_nR) * eq_table_nz);
		std::vector<amrex::Real> zmax_host(eq_table_nR);
		std::vector<amrex::Real> dlnrho_host(eq_table_nR);
		std::vector<double> ln_rho_col(eq_table_nz);
		for (int i = 0; i < eq_table_nR; ++i) {
			const double R = i * dR;
			const double Sigma = surfaceDensityProfile(R, Rc, Rd, userData_.Sigma0);
			zmax_host[i] = solveVerticalEquilibrium(R, Sigma, vc, Rc, userData_.cs_eq, eq_table_nz, ln_rho_col);
			std::copy(ln_rho_col.begin(), ln_rho_col.end(), lnrho_host.begin() + static_cast<std::ptrdiff_t>(i) * eq_table_nz);
			ln_rho_mid[i] = ln_rho_col[0];
		}
		// d ln(rho_mid) / dR: zero on the axis by symmetry, central differences inside, one-sided at the edge
		dlnrho_host[0] = 0.0;
		for (int i = 1; i < eq_table_nR - 1; ++i) {
			dlnrho_host[i] = (ln_rho_mid[i + 1] - ln_rho_mid[i - 1]) / (2.0 * dR);
		}
		dlnrho_host[eq_table_nR - 1] = (ln_rho_mid[eq_table_nR - 1] - ln_rho_mid[eq_table_nR - 2]) / dR;

		userData_.eq_lnrho_device.resize(lnrho_host.size());
		userData_.eq_zmax_device.resize(zmax_host.size());
		userData_.eq_dlnrho_mid_dR_device.resize(dlnrho_host.size());
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, lnrho_host.begin(), lnrho_host.end(), userData_.eq_lnrho_device.begin());
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, zmax_host.begin(), zmax_host.end(), userData_.eq_zmax_device.begin());
		amrex::Gpu::copy(amrex::Gpu::hostToDevice, dlnrho_host.begin(), dlnrho_host.end(), userData_.eq_dlnrho_mid_dR_device.begin());
		amrex::Gpu::synchronize();

		amrex::Print() << "Vertical equilibrium solved on " << eq_table_nR << " x " << eq_table_nz << " (R < " << eq_Rmax / (1.0e3 * C::parsec)
			       << " kpc), columns normalised to Sigma(R):\n";
		for (const double R_kpc : {1.0, 2.0, 4.0, 8.0}) {
			const double R = R_kpc * 1.0e3 * C::parsec;
			const int i = static_cast<int>(std::lround(R / dR));
			const double zmax = zmax_host[i];
			const double dz = zmax / (eq_table_nz - 1);
			double col = 0.0;
			double z2 = 0.0;
			for (int j = 0; j < eq_table_nz; ++j) {
				const double w = (j == 0 || j == eq_table_nz - 1) ? 0.5 : 1.0;
				const double rho = std::exp(lnrho_host[(static_cast<std::size_t>(i) * eq_table_nz) + j]);
				col += w * rho * dz;
				z2 += w * rho * (j * dz) * (j * dz) * dz;
			}
			amrex::Print() << "  R = " << R_kpc << " kpc: rho_mid = " << std::exp(ln_rho_mid[i]) << " g/cm^3, z_rms = " << std::sqrt(z2 / col) / C::parsec
				       << " pc, column / Sigma = " << 2.0 * col / surfaceDensityProfile(i * dR, Rc, Rd, userData_.Sigma0) << "\n";
		}
	} else {
		amrex::Print() << "Vertical profile: legacy sech^2 x halo factor (vertical_equilibrium_solve = 0), columns not normalised to Sigma(R)\n";
	}

	// Optional measured gas pull (measure_gas_pull.py): replaces the analytic disk pull in addStrangSplitSources
	std::string gas_pull_file;
	pp.query("gas_pull_file", gas_pull_file);
	if (!gas_pull_file.empty()) {
		loadGasPullTable(gas_pull_file, userData_);
		amrex::Print() << "Gas self-gravity pull: measured table " << gas_pull_file << " (" << userData_.gas_pull_nR << " x " << userData_.gas_pull_nz
			       << ", R < " << userData_.gas_pull_Rmax / (1.0e3 * C::parsec) << " kpc)\n";
		std::vector<amrex::Real> host(userData_.gas_pull_device.size());
		amrex::Gpu::copy(amrex::Gpu::deviceToHost, userData_.gas_pull_device.begin(), userData_.gas_pull_device.end(), host.begin());
		for (const double R_kpc : {2.0, 4.0, 8.0}) {
			const double R = R_kpc * 1.0e3 * C::parsec;
			const double measured = -gasPullTable(host.data(), userData_.gas_pull_nR, userData_.gas_pull_nz, userData_.gas_pull_Rmax,
							      userData_.gas_pull_z0, userData_.gas_pull_smax, R, 0.0);
			amrex::Print() << "  midplane pull at R = " << R_kpc << " kpc: measured / analytic thin disk = "
				       << measured / surfaceDensityPull(R, userData_.Sigma0, Rd) << "\n";
		}
	} else if (userData_.subtract_disk_gravity != 0) {
		amrex::Print() << "Gas self-gravity pull: analytic thin disk ("
			       << (userData_.vertical_equilibrium_solve != 0 ? "modified profile Sigma(R)" : "Freeman exponential, Sigma0") << ")\n";
	}

	//-------//
	// A_phi // - Load 2D Cylindrical A_phi Potential Table
	//-------//
	std::string aphi_meta_file;
	std::string aphi_data_file;
	pp.get("aphi_meta_file", aphi_meta_file);
	pp.get("aphi_data_file", aphi_data_file);

	std::ifstream meta_file(aphi_meta_file);
	if (!meta_file.is_open()) {
		amrex::Abort("Could not open 2D seed field metadata file: " + aphi_meta_file);
	}

	std::string line;
	while (std::getline(meta_file, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		const std::size_t eq_pos = line.find('=');
		if (eq_pos == std::string::npos) {
			continue;
		}
		std::string key = line.substr(0, eq_pos);
		while (!key.empty() && (std::isspace(key.back()) != 0)) {
			key.pop_back();
		}
		const std::size_t start = key.find_first_not_of(" \t");
		if (start != std::string::npos) {
			key = key.substr(start);
		}
		std::string val_str = line.substr(eq_pos + 1);
		const std::size_t first_num = val_str.find_first_not_of(" \t");
		if (first_num != std::string::npos) {
			val_str = val_str.substr(first_num);
		}
		const std::size_t end_num = val_str.find_first_of(" \t#[]");
		if (end_num != std::string::npos) {
			val_str = val_str.substr(0, end_num);
		}

		try {
			if (key == "seed_nR") {
				userData_.seed_nR = std::stoul(val_str);
			} else if (key == "seed_nz") {
				userData_.seed_nz = std::stoul(val_str);
			} else if (key == "seed_Rmax") {
				userData_.seed_Rmax = std::stod(val_str);
			} else if (key == "seed_Lz") {
				userData_.seed_Lz = std::stod(val_str);
			} else if (key == "seed_seed") {
				userData_.seed_str = val_str;
			}
		} catch (const std::exception &e) {
			std::string msg;
			msg += "Error parsing '";
			msg += key;
			msg += "' = '";
			msg += val_str;
			msg += "' in ";
			msg += aphi_meta_file;
			msg += ": ";
			msg += e.what();
			amrex::Abort(msg);
		}
	}
	meta_file.close();

	if (userData_.seed_str.empty()) {
		amrex::Print() << "WARNING: no 'seed_seed' key found while parsing " << aphi_meta_file << "\n";
	}

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(userData_.seed_nR > 0 && userData_.seed_nz > 0 && userData_.seed_Rmax > 0.0 && userData_.seed_Lz > 0.0,
					 "Error parsing cylindrical vector potential meta variables from init_seed_pot_field "
					 "(seed_nR/seed_nz/seed_Rmax/seed_Lz must all be present and positive).");

	const std::size_t total_elements = userData_.seed_nR * userData_.seed_nz;
	userData_.Aphi_device = load_bin_to_device(aphi_data_file, total_elements);

	amrex::Print() << "Loaded 2D Cylindrical Aphi Table cleanly. Map Size: " << userData_.seed_nR << " x " << userData_.seed_nz << "\n";

	// Derive B0 from target plasma beta at the disk midplane (R=Rd, z=0).
	// The stored Aphi table is dimensionless with rms(curl_nd) = 1 in units of 1/Rmax.
	// Physical B = aphi_nd * B0_scale / Rmax * curl_nd, so B_rms = B0_scale / Rmax.
	// beta = (rho cs^2 / gamma) / (B_rms^2 / 2)
	// => B_rms = cs * sqrt(2 rho_mid / (gamma * beta))
	// => B0_scale = B_rms * Rmax
	{
		const double Sigma_Rd = surfaceDensityProfile(Rd, Rc, Rd, userData_.Sigma0);
		const double rho_mid = userData_.vertical_equilibrium_solve != 0
					   ? std::exp(interpRadial(ln_rho_mid.data(), eq_table_nR, userData_.eq_Rmax, Rd))
					   : (M_PI * C::Gconst * Sigma_Rd * Sigma_Rd) / (2.0 * userData_.cs_eq * userData_.cs_eq);
		userData_.rho_mid = rho_mid;
		const double B_rms_HL = cs_disk * std::sqrt(2.0 * rho_mid / (gamma * target_beta_seed));
		userData_.seed_B0_HL = B_rms_HL * userData_.seed_Rmax;
		amrex::Print() << "Seed field: target_beta=" << target_beta_seed << "  rho_mid=" << rho_mid << "  B_rms_HL=" << B_rms_HL
			       << "  B0_scale=" << userData_.seed_B0_HL << " G*cm (HL)\n";
	}

	//---------------//
	// Turb Sampling //
	//---------------//
	std::string turb_vx_file;
	std::string turb_vy_file;
	std::string turb_vz_file;

	pp.get("turb_vx_file", turb_vx_file);
	pp.get("turb_vy_file", turb_vy_file);
	pp.get("turb_vz_file", turb_vz_file);

	// assume nturb is cubic, i.e. has equal x, y, and z dimensions
	// get number of cells in file
	const std::size_t n_turb = std::filesystem::file_size(turb_vx_file) / sizeof(amrex::Real);
	// take cube root to get sidelength
	const auto n_turb_side = static_cast<std::size_t>(std::lround(std::cbrt(static_cast<double>(n_turb))));

	userData_.turb_nx = static_cast<int>(n_turb_side);
	userData_.turb_ny = static_cast<int>(n_turb_side);
	userData_.turb_nz = static_cast<int>(n_turb_side);

	userData_.turb_vx_device = load_bin_to_device(turb_vx_file, n_turb);
	userData_.turb_vy_device = load_bin_to_device(turb_vy_file, n_turb);
	userData_.turb_vz_device = load_bin_to_device(turb_vz_file, n_turb);

	std::string turb_seed_file;
	pp.query("turb_seed_file", turb_seed_file);
	if (!turb_seed_file.empty()) {
		userData_.turb_seeds = load_turb_seeds(turb_seed_file);

		amrex::Print() << "Turbulence seed file: " << turb_seed_file << " (seeds read = " << userData_.turb_seeds.size() << "):";
		for (const auto turb_seed : userData_.turb_seeds) {
			amrex::Print() << " " << turb_seed;
		}
		amrex::Print() << "\n";
	}

	const double turb_rescale = turb_target_Mach * cs_disk;
	userData_.turb_rescale_factor = turb_rescale;

	amrex::Print() << "Turbulence loaded from binary files:\n"
		       << " vx = " << turb_vx_file << "\n"
		       << " vy = " << turb_vy_file << "\n"
		       << " vz = " << turb_vz_file << "\n"
		       << " cube size = " << userData_.turb_nx << " x " << userData_.turb_ny << " x " << userData_.turb_nz << "\n"
		       << "Velocity scale = " << turb_rescale / 1.0e5 << " km/s\n";

	amrex::Print() << "MHDGalaxy init complete\n"
		       << "Mc=" << userData_.Mc << " Q=" << userData_.Q_mean << " Sigma0=" << userData_.Sigma0
		       << " Seed=" << (userData_.seed_str.empty() ? std::string("<not found>") : userData_.seed_str)
		       << "\nsn_mass_per_event_msun=" << userData_.sn_mass_per_event_msun
		       << " sn_cluster_momentum_exponent=" << userData_.sn_cluster_momentum_exponent
		       << " sn_start_time_myr=" << userData_.sn_start_time / 3.15576e13 << "\n"
		       << "M_solar=" << C::M_solar << "\n";
}

// Set initial conditions on the grid by evaluating the analytic disk density and velocity profiles at cell centers,
// and calculating the local magnetic energy at cell centers by taking the curl of the analytically
// sampled vector potential A_phi at the surrounding staggered Yee mesh nodes.

template <> void QuokkaSimulation<MHDGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const double vc = userData_.vc;
	const bool rotation_on_cylinders = (userData_.rotation_on_cylinders != 0);
	const double Rc = userData_.Rc;
	const double Rd = userData_.Rd;
	const double Rcutoff = userData_.Rcutoff;
	const double Sigma0 = userData_.Sigma0;
	const double cs_disk = userData_.cs_disk;
	const double cs_eq = userData_.cs_eq;
	const double cs_cgm = userData_.cs_cgm;
	const double rho_cgm = userData_.rho_cgm;
	constexpr double gamma = quokka::EOS_Traits<MHDGalaxy>::gamma;

	const bool eq_solve = (userData_.vertical_equilibrium_solve != 0);
	const amrex::Real *eq_lnrho = userData_.eq_lnrho_device.data();
	const amrex::Real *eq_zmax = userData_.eq_zmax_device.data();
	const double eq_Rmax = userData_.eq_Rmax;

	const double B0_scale = userData_.seed_B0_HL;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(B0_scale > 0.0, "Beta-derived seed field strength must be positive.");

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const auto prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;

	const auto dxvec = grid_elem.dx_;
	const int subsample = initial_condition_subsamples;
	const double dx = dxvec[0] / static_cast<double>(subsample);
	const double dy = dxvec[1] / static_cast<double>(subsample);
	const double dz = dxvec[2] / static_cast<double>(subsample);

	const int turb_nx = userData_.turb_nx;
	const int turb_ny = userData_.turb_ny;
	const int turb_nz = userData_.turb_nz;

	// Raw pointers to the device buffers loaded in preCalculateInitialConditions
	// (row-major [turb_nx][turb_ny][turb_nz], k fastest -- see interpolate_turbulence).
	const amrex::Real *turb_vx = userData_.turb_vx_device.data();
	const amrex::Real *turb_vy = userData_.turb_vy_device.data();
	const amrex::Real *turb_vz = userData_.turb_vz_device.data();

	const double turb_rescale = userData_.turb_rescale_factor;

	const double turb_xmin = geom[0].ProbLoArray()[0];
	const double turb_ymin = geom[0].ProbLoArray()[1];
	const double turb_zmin = geom[0].ProbLoArray()[2];

	const double turb_Lx = geom[0].ProbHiArray()[0] - geom[0].ProbLoArray()[0];
	const double turb_Ly = geom[0].ProbHiArray()[1] - geom[0].ProbLoArray()[1];
	const double turb_Lz = geom[0].ProbHiArray()[2] - geom[0].ProbLoArray()[2];

	const double turb_dx = turb_Lx / static_cast<double>(turb_nx - 1);
	const double turb_dy = turb_Ly / static_cast<double>(turb_ny - 1);
	const double turb_dz = turb_Lz / static_cast<double>(turb_nz - 1);

	// Cylindrical Potential Table Pointers & Parameters for GPU Lambdas
	const amrex::Real *aphi_ptr = userData_.Aphi_device.data();
	const int nR_table = static_cast<int>(userData_.seed_nR);
	const int nz_table = static_cast<int>(userData_.seed_nz);
	const double Rmax_table = userData_.seed_Rmax;
	const double Lz_table = userData_.seed_Lz;

	const double dR_table = Rmax_table / static_cast<double>(nR_table);
	const double axis_dead_zone = axis_fallback_cells * dR_table;

	// True physical domain extents (level-independent), used to force exact B=0
	// at the real simulation boundary regardless of whether seed_Rmax/seed_Lz
	// (the table's taper calibration) are matched to the actual domain size.
	const auto prob_lo_dom = geom[0].ProbLoArray();
	const auto prob_hi_dom = geom[0].ProbHiArray();
	constexpr double boundary_tol = 1.0e-6; // relative tolerance, scaled by dx below

	// physical potential with hard boundary guard and smooth tapering, sampled directly in the node lambdas
	// to ensure consistency with the interpolated values used for curl calculation.
	auto get_Aphi_physical = [=] AMREX_GPU_DEVICE(double x_val, double y_val, double z_val) -> double {
		const double R_val = std::sqrt(x_val * x_val + y_val * y_val);
		const double aphi_nd = sample_bicubic(aphi_ptr, nR_table, nz_table, Rmax_table, Lz_table, R_val, z_val);
		return aphi_nd * B0_scale;
	};
	auto get_Ax = [=] AMREX_GPU_DEVICE(double x_e, double y_e, double z_e) -> double {
		const double R_e = std::sqrt(x_e * x_e + y_e * y_e);
		// Force zero within 2.0*dx to prevent 1/R amplification of interpolation noise near the axis.
		if (R_e < axis_dead_zone) {
			return 0.0;
		}
		const double taper = get_taper_factor(x_e, y_e, z_e, Rmax_table, Lz_table, dx, dy, dz);
		const double Aphi = get_Aphi_physical(x_e, y_e, z_e);
		return -Aphi * (y_e / R_e) * taper;
	};

	auto get_Ay = [=] AMREX_GPU_DEVICE(double x_e, double y_e, double z_e) -> double {
		const double R_e = std::sqrt(x_e * x_e + y_e * y_e);
		// Force zero within 2.0*dx to prevent 1/R amplification of interpolation noise near the axis.
		if (R_e < axis_dead_zone) {
			return 0.0;
		}
		const double taper = get_taper_factor(x_e, y_e, z_e, Rmax_table, Lz_table, dx, dy, dz);
		const double Aphi = get_Aphi_physical(x_e, y_e, z_e);
		return Aphi * (x_e / R_e) * taper;
	};

	auto sampleSubcell = [=] AMREX_GPU_DEVICE(int i, int j, int k, double &rho_samples, double &px_samples, double &py_samples, double &pz_samples,
						  double &Etot_samples, double &Eint_samples) {
		const double x = prob_lo[0] + (i + 0.5) * dx;
		const double y = prob_lo[1] + (j + 0.5) * dy;
		const double z = prob_lo[2] + (k + 0.5) * dz;
		const double R = std::sqrt(x * x + y * y + 1e-200);

		double rho_disc_raw = 0.0;
		if (R < Rcutoff) {
			rho_disc_raw = eq_solve ? equilibriumDensity(eq_lnrho, eq_zmax, eq_table_nR, eq_table_nz, eq_Rmax, R, z)
						: diskDensityAnalytic(R, z, Rc, Rd, Sigma0, vc, cs_eq);
		}
		const bool in_disk = (rho_disc_raw > rho_transition);
		const double rho = in_disk ? amrex::max(rho_disc_raw, rho_transition * 1e-6) : rho_cgm;
		const double cs = in_disk ? cs_disk : cs_cgm;

		// An isothermal (barotropic) disk in vertical hydrostatic equilibrium rotates on cylinders: the drop of the halo's
		// radial pull above the plane is offset by the radial pressure gradient at fixed z (Wang et al. 2010, sec. 2)
		const double vrot = vcircAnalytic(R, rotation_on_cylinders ? 0.0 : z, vc, Rc);
		double vx = 0.0;
		double vy = 0.0;
		double vz = 0.0;
		if (in_disk) {
			vx = -vrot * y / R;
			vy = vrot * x / R;
		}

		double dvx_pert = 0.0;
		double dvy_pert = 0.0;
		double dvz_pert = 0.0;

		if (in_disk) {

			const double tx = (x - turb_xmin) / turb_dx;
			const double ty = (y - turb_ymin) / turb_dy;
			const double tz = (z - turb_zmin) / turb_dz;

			dvx_pert = interpolate_turbulence(turb_vx, turb_nx, turb_ny, turb_nz, tx, ty, tz) * turb_rescale;
			dvy_pert = interpolate_turbulence(turb_vy, turb_nx, turb_ny, turb_nz, tx, ty, tz) * turb_rescale;
			dvz_pert = interpolate_turbulence(turb_vz, turb_nx, turb_ny, turb_nz, tx, ty, tz) * turb_rescale;
		}
		vx += dvx_pert;
		vy += dvy_pert;
		vz += dvz_pert;

		const double pressure = rho * cs * cs;
		const double Eint = pressure / (gamma - 1.0);
		const double Ekin = 0.5 * rho * (vx * vx + vy * vy + vz * vz);

		const double x_node_lo = prob_lo[0] + (i + 0) * dx;
		const double x_node_hi = prob_lo[0] + (i + 1) * dx;
		const double y_node_lo = prob_lo[1] + (j + 0) * dy;
		const double y_node_hi = prob_lo[1] + (j + 1) * dy;
		const double z_node_lo = prob_lo[2] + (k + 0) * dz;
		const double z_node_hi = prob_lo[2] + (k + 1) * dz;

		const double Ay_hi_left = get_Ay(x_node_lo, y, z_node_hi);
		const double Ay_lo_left = get_Ay(x_node_lo, y, z_node_lo);
		const double Ay_hi_right = get_Ay(x_node_hi, y, z_node_hi);
		const double Ay_lo_right = get_Ay(x_node_hi, y, z_node_lo);
		double Bx_face_left = -(Ay_hi_left - Ay_lo_left) / dz;
		double Bx_face_right = -(Ay_hi_right - Ay_lo_right) / dz;
		if (std::abs(x_node_lo - prob_lo_dom[0]) < boundary_tol * dx || std::abs(x_node_lo - prob_hi_dom[0]) < boundary_tol * dx) {
			Bx_face_left = 0.0;
		}
		if (std::abs(x_node_hi - prob_lo_dom[0]) < boundary_tol * dx || std::abs(x_node_hi - prob_hi_dom[0]) < boundary_tol * dx) {
			Bx_face_right = 0.0;
		}
		const double Bx_cc = 0.5 * (Bx_face_left + Bx_face_right);

		const double Ax_hi_bot = get_Ax(x, y_node_lo, z_node_hi);
		const double Ax_lo_bot = get_Ax(x, y_node_lo, z_node_lo);
		const double Ax_hi_top = get_Ax(x, y_node_hi, z_node_hi);
		const double Ax_lo_top = get_Ax(x, y_node_hi, z_node_lo);
		double By_face_bot = (Ax_hi_bot - Ax_lo_bot) / dz;
		double By_face_top = (Ax_hi_top - Ax_lo_top) / dz;
		if (std::abs(y_node_lo - prob_lo_dom[1]) < boundary_tol * dy || std::abs(y_node_lo - prob_hi_dom[1]) < boundary_tol * dy) {
			By_face_bot = 0.0;
		}
		if (std::abs(y_node_hi - prob_lo_dom[1]) < boundary_tol * dy || std::abs(y_node_hi - prob_hi_dom[1]) < boundary_tol * dy) {
			By_face_top = 0.0;
		}
		const double By_cc = 0.5 * (By_face_bot + By_face_top);

		const double Ay_r_cc = get_Ay(x_node_hi, y, z);
		const double Ay_l_cc = get_Ay(x_node_lo, y, z);
		const double Ax_t_cc = get_Ax(x, y_node_hi, z);
		const double Ax_b_cc = get_Ax(x, y_node_lo, z);
		const double Bz_cc = ((Ay_r_cc - Ay_l_cc) / dx) - ((Ax_t_cc - Ax_b_cc) / dy);

		const double Emag = 0.5 * (Bx_cc * Bx_cc + By_cc * By_cc + Bz_cc * Bz_cc);

		rho_samples += rho;
		px_samples += rho * vx;
		py_samples += rho * vy;
		pz_samples += rho * vz;
		Etot_samples += Ekin + Eint + Emag;
		Eint_samples += Eint;
	};

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const auto n_samples = static_cast<double>(subsample * subsample * subsample);
		double rho_samples = 0.0;
		double px_samples = 0.0;
		double py_samples = 0.0;
		double pz_samples = 0.0;
		double Etot_samples = 0.0;
		double Eint_samples = 0.0;

		for (int a = 0; a < subsample; ++a) {
			for (int b = 0; b < subsample; ++b) {
				for (int c = 0; c < subsample; ++c) {
					sampleSubcell(subsample * i + a, subsample * j + b, subsample * k + c, rho_samples, px_samples, py_samples, pz_samples,
						      Etot_samples, Eint_samples);
				}
			}
		}

		// set cell to average of subcells
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::density_index) = rho_samples / n_samples;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) = px_samples / n_samples;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) = py_samples / n_samples;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) = pz_samples / n_samples;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::energy_index) = Etot_samples / n_samples;
		state_cc(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index) = Eint_samples / n_samples;
	});
}

template <> void QuokkaSimulation<MHDGalaxy>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<amrex::Real> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const auto dx = grid_elem.dx_;
	const auto prob_lo = grid_elem.prob_lo_;

	const double B0_scale = userData_.seed_B0_HL;

	// Cylindrical Potential Table Pointers & Parameters
	const amrex::Real *aphi_ptr = userData_.Aphi_device.data();
	const int nR_table = static_cast<int>(userData_.seed_nR);
	const int nz_table = static_cast<int>(userData_.seed_nz);
	const double Rmax_table = userData_.seed_Rmax;
	const double Lz_table = userData_.seed_Lz;
	const double dR_table = Rmax_table / static_cast<double>(nR_table);
	const double axis_dead_zone = axis_fallback_cells * dR_table;

	// True physical domain extents (level-independent), used to force exact B=0
	// at the real simulation boundary regardless of whether seed_Rmax/seed_Lz
	// (the table's taper calibration) are matched to the actual domain size.
	const auto prob_lo_dom = geom[0].ProbLoArray();
	const auto prob_hi_dom = geom[0].ProbHiArray();
	constexpr double boundary_tol = 1.0e-6; // relative tolerance, scaled by dx below

	// magnetic potential with hard boundary guard
	auto get_Aphi_physical = [=] AMREX_GPU_DEVICE(double x_val, double y_val, double z_val) -> double {
		const double R_val = std::sqrt(x_val * x_val + y_val * y_val);
		const double aphi_nd = sample_bicubic(aphi_ptr, nR_table, nz_table, Rmax_table, Lz_table, R_val, z_val);
		return aphi_nd * B0_scale;
	};
	// cartesian mapping with deadzone
	auto get_Ax_node = [=] AMREX_GPU_DEVICE(double x_n, double y_n, double z_n) -> double {
		const double R = std::sqrt(x_n * x_n + y_n * y_n);
		// axis deadzone: R < axis_fallback_cells * dR_table (currently 1*dR_table)
		if (R < axis_dead_zone) {
			return 0.0;
		}
		const double taper = get_taper_factor(x_n, y_n, z_n, Rmax_table, Lz_table, dx[0], dx[1], dx[2]);
		const double Aphi = get_Aphi_physical(x_n, y_n, z_n);
		return -Aphi * (y_n / R) * taper;
	};

	auto get_Ay_node = [=] AMREX_GPU_DEVICE(double x_n, double y_n, double z_n) -> double {
		const double R = std::sqrt(x_n * x_n + y_n * y_n);
		// axis deadzone: R < axis_fallback_cells * dR_table (currently 1*dR_table)
		if (R < axis_dead_zone) {
			return 0.0;
		}
		const double taper = get_taper_factor(x_n, y_n, z_n, Rmax_table, Lz_table, dx[0], dx[1], dx[2]);
		const double Aphi = get_Aphi_physical(x_n, y_n, z_n);
		return Aphi * (x_n / R) * taper;
	};

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		double B_face = 0.0;

		if (dir == quokka::direction::x) {
			// Bx lives at x-faces: index (i) represents the x-node plane.
			// Python samples Ay_node at the x-node, averages over y, and differentiates over z.
			const double xf = prob_lo[0] + i * dx[0];
			const double yj = prob_lo[1] + j * dx[1];
			const double yjp = prob_lo[1] + (j + 1) * dx[1];
			const double zk = prob_lo[2] + k * dx[2];
			const double zkp = prob_lo[2] + (k + 1) * dx[2];

			const double Ay_j_kp = get_Ay_node(xf, yj, zkp);
			const double Ay_jp_kp = get_Ay_node(xf, yjp, zkp);
			const double Ay_j_k = get_Ay_node(xf, yj, zk);
			const double Ay_jp_k = get_Ay_node(xf, yjp, zk);

			const double Ay_xface_kp = 0.5 * (Ay_j_kp + Ay_jp_kp);
			const double Ay_xface_k = 0.5 * (Ay_j_k + Ay_jp_k);

			// Bx = -dAy/dz
			B_face = -(Ay_xface_kp - Ay_xface_k) / dx[2];

			// Hard-zero enforcement at the true x-domain boundary (normal component
			// on x-faces), independent of seed-table taper calibration.
			if (std::abs(xf - prob_lo_dom[0]) < boundary_tol * dx[0] || std::abs(xf - prob_hi_dom[0]) < boundary_tol * dx[0]) {
				B_face = 0.0;
			}

		} else if (dir == quokka::direction::y) {
			// By lives at y-faces: index (j) represents the y-node plane.
			// Python samples Ax_node at the y-node, averages over x, and differentiates over z.
			const double xi = prob_lo[0] + i * dx[0];
			const double xip = prob_lo[0] + (i + 1) * dx[0];
			const double yf = prob_lo[1] + j * dx[1];
			const double zk = prob_lo[2] + k * dx[2];
			const double zkp = prob_lo[2] + (k + 1) * dx[2];

			const double Ax_i_kp = get_Ax_node(xi, yf, zkp);
			const double Ax_ip_kp = get_Ax_node(xip, yf, zkp);
			const double Ax_i_k = get_Ax_node(xi, yf, zk);
			const double Ax_ip_k = get_Ax_node(xip, yf, zk);

			const double Ax_yface_kp = 0.5 * (Ax_i_kp + Ax_ip_kp);
			const double Ax_yface_k = 0.5 * (Ax_i_k + Ax_ip_k);

			// By = dAx/dz
			B_face = (Ax_yface_kp - Ax_yface_k) / dx[2];

			// Hard-zero enforcement at the true y-domain boundary (normal component
			// on y-faces), independent of seed-table taper calibration.
			if (std::abs(yf - prob_lo_dom[1]) < boundary_tol * dx[1] || std::abs(yf - prob_hi_dom[1]) < boundary_tol * dx[1]) {
				B_face = 0.0;
			}

		} else {
			// Bz lives at z-faces: index (k) represents the z-node plane.
			// Bz = dAy/dx - dAx/dy using the exact node-level cross-averages.
			const double xi = prob_lo[0] + i * dx[0];
			const double xip = prob_lo[0] + (i + 1) * dx[0];
			const double yj = prob_lo[1] + j * dx[1];
			const double yjp = prob_lo[1] + (j + 1) * dx[1];
			const double zf = prob_lo[2] + k * dx[2];

			// dAy/dx term: difference over x, then average over y
			const double Ay_ip_j = get_Ay_node(xip, yj, zf);
			const double Ay_i_j = get_Ay_node(xi, yj, zf);
			const double Ay_ip_jp = get_Ay_node(xip, yjp, zf);
			const double Ay_i_jp = get_Ay_node(xi, yjp, zf);

			const double dAy_dx_j = (Ay_ip_j - Ay_i_j) / dx[0];
			const double dAy_dx_jp = (Ay_ip_jp - Ay_i_jp) / dx[0];
			const double dAy_dx_cc = 0.5 * (dAy_dx_j + dAy_dx_jp);

			// dAx/dy term: difference over y, then average over x
			const double Ax_i_jp = get_Ax_node(xi, yjp, zf);
			const double Ax_i_j = get_Ax_node(xi, yj, zf);
			const double Ax_ip_jp = get_Ax_node(xip, yjp, zf);
			const double Ax_ip_j = get_Ax_node(xip, yj, zf);

			const double dAx_dy_i = (Ax_i_jp - Ax_i_j) / dx[1];
			const double dAx_dy_ip = (Ax_ip_jp - Ax_ip_j) / dx[1];
			const double dAx_dy_cc = 0.5 * (dAx_dy_i + dAx_dy_ip);

			B_face = dAy_dx_cc - dAx_dy_cc;

			// Hard-zero enforcement at the true z-domain boundary (normal component
			// on z-faces), independent of seed-table taper calibration.
			if (std::abs(zf - prob_lo_dom[2]) < boundary_tol * dx[2] || std::abs(zf - prob_hi_dom[2]) < boundary_tol * dx[2]) {
				B_face = 0.0;
			}
		}
		state_fc(i, j, k, 0) = B_face;
	});
}

void cleanup(QuokkaSimulation<MHDGalaxy> &sim)
{
	amrex::Gpu::streamSynchronize();
	sim.userData_.turb_vx_device = {};
	sim.userData_.turb_vy_device = {};
	sim.userData_.turb_vz_device = {};
	sim.userData_.turb_seeds = {};
	sim.userData_.Aphi_device = {};
}

template <> void QuokkaSimulation<MHDGalaxy>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real /*time*/, amrex::Real dt_lev)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const double vc = userData_.vc;
	const bool rotation_on_cylinders = (userData_.rotation_on_cylinders != 0);
	const double Rc = userData_.Rc;
	const bool subtract_disk_gravity = (userData_.subtract_disk_gravity != 0);
	const double Sigma0 = userData_.Sigma0;
	const double Rd = userData_.Rd;
	const double cs_eq = userData_.cs_eq;
	const bool eq_solve = (userData_.vertical_equilibrium_solve != 0);
	const amrex::Real *eq_dlnrho_mid_dR = userData_.eq_dlnrho_mid_dR_device.data();
	const double eq_Rmax = userData_.eq_Rmax;
	const bool use_pull_table = (userData_.gas_pull_nR > 0);
	const amrex::Real *pull_table = userData_.gas_pull_device.data();
	const int pull_nR = userData_.gas_pull_nR;
	const int pull_nz = userData_.gas_pull_nz;
	const double pull_Rmax = userData_.gas_pull_Rmax;
	const double pull_z0 = userData_.gas_pull_z0;
	const double pull_smax = userData_.gas_pull_smax;
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			const double y = prob_lo[1] + (j + 0.5) * dx[1];
			const double z = prob_lo[2] + (k + 0.5) * dx[2];
			const double R = std::sqrt(x * x + y * y + 1e-200);

			const double rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
			const double px = state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
			const double py = state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
			const double pz = state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
			const double Ekin_old = 0.5 * (px * px + py * py + pz * pz) / rho;

			// -grad(psi), the flattened log potential (Arora et al. eq. A.3)
			const double v = vcircAnalytic(R, z, vc, Rc);
			const double g_R_halo = -v * v / R;
			// phi_dm = psi - Phi_disk + cs^2 R/Rd (eq. A.5): add back the disk's own pull (outward) and the
			// pressure-gradient term (inward); the live self-gravity supplies the disk's pull once
			double g_R = g_R_halo;
			if (subtract_disk_gravity) {
				// add back (outward) the gas's own pull, which the live self-gravity supplies: measured from a
				// t = 0 snapshot at this (R, |z|) if a table is given, else the thin-disk pull at the midplane --
				// of the gas actually set up (columns hold Sigma(R)) or, legacy, of a Sigma0 exponential
				double disk_pull = 0.0;
				// (rotation_on_cylinders: the midplane pull at every z, so halo + live gravity + pressure balance v_phi(R, 0))
				if (use_pull_table) {
					disk_pull = -gasPullTable(pull_table, pull_nR, pull_nz, pull_Rmax, pull_z0, pull_smax, R, rotation_on_cylinders ? 0.0 : z);
				} else if (eq_solve) {
					disk_pull = surfaceDensityPull(R, Sigma0, Rd);
				} else {
					disk_pull = freemanDiskPull(R, Sigma0, Rd);
				}
				// remove the pressure support: cs^2 d ln(rho_mid)/dR (< 0, inward) of the solved profile, or an
				// exponential's -cs^2/Rd
				const double pressure_term =
				    eq_solve ? cs_eq * cs_eq * interpRadial(eq_dlnrho_mid_dR, eq_table_nR, eq_Rmax, R) : -cs_eq * cs_eq / Rd;
				g_R += disk_pull + pressure_term;
			}
			const double gx = g_R * x / R;
			const double gy = g_R * y / R;
			const double gz = g_R_halo * z / R / (q_flatten * q_flatten); // vertical force unchanged


			const double px_new = px + dt_lev * rho * gx;
			const double py_new = py + dt_lev * rho * gy;
			const double pz_new = pz + dt_lev * rho * gz;
			const double Ekin_new = 0.5 * (px_new * px_new + py_new * py_new + pz_new * pz_new) / rho;

			state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) = px_new;
			state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) = py_new;
			state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) = pz_new;
			state(i, j, k, HydroSystem<MHDGalaxy>::energy_index) += Ekin_new - Ekin_old;
		});
	}
}

template <> void QuokkaSimulation<MHDGalaxy>::refineGrid(const int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	const amrex::Real refine_Rcyl = userData_.refine_Rcyl;
	const amrex::Real refine_Hcyl = userData_.refine_Hcyl;
	const amrex::Real Rcyl_inner = userData_.refine_Rcyl_inner;

	// Shrink the refinement cylinder at each successive level, floored at 30% of the
	// base size, so finer levels progressively focus on the disk core instead of all
	// sharing the same footprint out to refine_Rcyl/refine_Hcyl.
	const amrex::Real Rcyl_lev = amrex::max(refine_Rcyl - userData_.refine_shrink_Rcyl * lev, 0.3 * refine_Rcyl, Rcyl_inner);
	const amrex::Real Hcyl_lev = amrex::max(refine_Hcyl - userData_.refine_shrink_Hcyl * lev, 0.3 * refine_Hcyl);

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real x0 = prob_lo[0] + i * dx[0];
		const amrex::Real y0 = prob_lo[1] + j * dx[1];
		const amrex::Real z0 = prob_lo[2] + k * dx[2];
		const amrex::Real x1 = x0 + dx[0];
		const amrex::Real y1 = y0 + dx[1];
		const amrex::Real z1 = z0 + dx[2];

		auto tagIfInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			const auto R = std::sqrt(x * x + y * y);
			const auto H = std::abs(z);
			if (Rcyl_inner <= R && R < Rcyl_lev && H < Hcyl_lev) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};

		constexpr int subsamples = refine_grid_subsamples;
		for (int a = 0; a < subsamples; ++a) {
			const amrex::Real x = std::lerp(x0, x1, a / (subsamples - 1));
			for (int b = 0; b < subsamples; ++b) {
				const amrex::Real y = std::lerp(y0, y1, b / (subsamples - 1));
				for (int c = 0; c < subsamples; ++c) {
					const amrex::Real z = std::lerp(z0, z1, c / (subsamples - 1));
					tagIfInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<MHDGalaxy>::computeAfterTimestep()
{
	if (!(userData_.sn_jeans_J > 0.0)) {
		return;
	}
	// delay feedback so the initial disk can settle before cells are tested against the Jeans trigger
	if (tNew_[0] < userData_.sn_start_time) {
		return;
	}

	constexpr double MSUN = C::M_solar;
	constexpr double KM_S = 1.0e5;
	constexpr int stencil_radius = 2;		 // rK in units of dx
	constexpr int kernel_ghost = stencil_radius + 1; // +1 cell of margin beyond rK's reach,
							 // for both the deposit loop bound and
							 // the mask/delta MultiFab ghost width
	constexpr double v_terminal_max = 1000.0 * KM_S; // cap on radial kick velocity, cm/s

	const double sn_jeans_J = userData_.sn_jeans_J;
	const double sn_momentum_ref = userData_.sn_momentum; // calibration coefficient
	const double cluster_exponent = userData_.sn_cluster_momentum_exponent;
	// Floor guards against divide-by-zero / a runaway N_SN if sn_mass_per_event_msun
	// is misconfigured to (near) zero.
	const double M_cluster_per_SN = amrex::max(userData_.sn_mass_per_event_msun, 1.0e-3) * MSUN;
	const double sfe = userData_.star_formation_efficiency;

	// Summed into the base AMRSimulation::sn_count_/sn_count_cumulative_ (not a
	// userData_ field) so it survives checkpoint/restart; see sn_trigger_count_cumulative
	// in SimulationData<MHDGalaxy> for why that counter isn't treated the same way.
	amrex::Long total_sn_events_this_step = 0;

	for (int lev = 0; lev <= finest_level; ++lev) {
		auto &state = state_new_cc_[lev];
		auto const &state_fc = state_new_fc_[lev];

		// Fill ghost zones before reading neighbor data below.
		const auto time = tNew_[lev];
		fillBoundaryConditions(state, state, lev, time, quokka::centering::cc, quokka::direction::na, InterpHookNone, InterpHookNone,
				       FillPatchType::fillpatch_function);
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			fillBoundaryConditions(state_new_fc_[lev][idim], state_new_fc_[lev][idim], lev, time, quokka::centering::fc,
					       static_cast<quokka::direction>(idim), InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);
		}

		const auto dx = geom[lev].CellSizeArray();
		const double dx_max = amrex::max(dx[0], amrex::max(dx[1], dx[2]));
		const double vol = dx[0] * dx[1] * dx[2];
		const double rK = r_K_factor * dx[0]; // physical kernel radius (assumes isotropic dx)

		amrex::iMultiFab mask_valid(state.boxArray(), state.DistributionMap(), 1, 0);
		if (lev < finest_level) {
			mask_valid = amrex::makeFineMask(state.boxArray(), state.DistributionMap(), state_new_cc_[lev + 1].boxArray(), refRatio(lev), 1, 0);
		} else {
			mask_valid.setVal(1);
		}
		amrex::iMultiFab mask(state.boxArray(), state.DistributionMap(), 1, kernel_ghost);
		mask.setVal(1);
		amrex::iMultiFab::Copy(mask, mask_valid, 0, 0, 1, 0);
		mask.FillBoundary(geom[lev].periodicity());

		// delta components: 0-2 = summed recentering (rho_nb * vCOM - p_nb) over the kernels touching the
		// cell, 3-5 = summed radial kicks, 6 = number of kernels touching the cell. Pass 2 applies
		// recentering / count + kicks, so a cell in n overlapping kernels is recentered once (onto the
		// mean of their vCOM) rather than n times: summing n recenterings computed from the same old
		// state gives p -> (1 - n) p + ..., which reverses (n = 2) or amplifies (n >= 3) the momentum
		// every step. Pure momentum injection -- no density change, no separately-tracked energy delta.
		constexpr int n_delta_comp = 7;
		amrex::MultiFab delta(state.boxArray(), state.DistributionMap(), n_delta_comp, kernel_ghost);
		delta.setVal(0.0);

		// --- Pass 1: gather kernel sums + deposit momentum (direct, unlimited) ---
		amrex::Gpu::DeviceScalar<int> d_sn_events_lev(0);
		amrex::Gpu::DeviceScalar<int> d_triggered_lev(0);
		amrex::Gpu::DeviceScalar<int> d_capped_lev(0);
		amrex::Gpu::DeviceScalar<double> d_resid_x(0.0);
		amrex::Gpu::DeviceScalar<double> d_resid_y(0.0);
		amrex::Gpu::DeviceScalar<double> d_resid_z(0.0);
		int *p_sn_events_lev = d_sn_events_lev.dataPtr();
		int *p_triggered_lev = d_triggered_lev.dataPtr();
		int *p_capped_lev = d_capped_lev.dataPtr();
		double *p_resid_x = d_resid_x.dataPtr();
		double *p_resid_y = d_resid_y.dataPtr();
		double *p_resid_z = d_resid_z.dataPtr();

		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();
			auto const &s = state.const_array(mfi);
			auto d = delta.array(mfi);
			auto const &mask_arr = mask.const_array(mfi);

			const std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc{state_fc[0].const_array(mfi), state_fc[1].const_array(mfi),
												  state_fc[2].const_array(mfi)};

			const auto slo = state[mfi].box().smallEnd();
			const auto shi = state[mfi].box().bigEnd();
			const auto dlo = delta[mfi].box().smallEnd();
			const auto dhi = delta[mfi].box().bigEnd();

			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				// Skip cells covered by a finer level
				if (mask_arr(i, j, k) == 0) {
					return;
				}

				const double rho = s(i, j, k, HydroSystem<MHDGalaxy>::density_index);

				const double cs = HydroSystem<MHDGalaxy>::ComputeIsothermalSoundSpeed(s, i, j, k, &fab_fc);
				const double plasma_beta = HydroSystem<MHDGalaxy>::ComputePlasmaBeta(s, i, j, k, &fab_fc); // returns inf when mhd off
				const double beta_safe = amrex::max(plasma_beta, 1.0e-10);
				const double cs_eff_sq = cs * cs * (1.0 + 0.74 / beta_safe);
				const double rho_J = M_PI * cs_eff_sq / (C::Gconst * sn_jeans_J * sn_jeans_J * (dx_max * dx_max));

				if (rho <= rho_J) {
					return;
				} // Jeans trigger

				// Star mass formed this cell -> number of IMF-averaged SN events.
				const double M_cell = rho * vol;
				const double M_stars = sfe * M_cell;
				const int N_SN = amrex::max(1, static_cast<int>(std::floor(M_stars / M_cluster_per_SN)));
				amrex::Gpu::Atomic::Add(p_triggered_lev, 1);
				amrex::Gpu::Atomic::Add(p_sn_events_lev, N_SN);

				// --- Gather kernel sums: fractional-overlap-weighted mass + momentum ---
				double omega_sum = 0.0;
				double mass_sum = 0.0;
				double momx_sum = 0.0;
				double momy_sum = 0.0;
				double momz_sum = 0.0;

				for (int di = -kernel_ghost; di <= kernel_ghost; ++di) {
					for (int dj = -kernel_ghost; dj <= kernel_ghost; ++dj) {
						for (int dk = -kernel_ghost; dk <= kernel_ghost; ++dk) {
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] || kk < slo[2] || kk > shi[2]) {
								continue;
							}
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] || kk < dlo[2] || kk > dhi[2]) {
								continue;
							}
							if (mask_arr(ii, jj, kk) == 0) {
								continue;
							}

							const double omega = cellSphereOverlapFraction(static_cast<double>(di), static_cast<double>(dj),
												       static_cast<double>(dk), dx[0], dx[1], dx[2], rK);
							if (omega <= 0.0) {
								continue;
							}

							const double rho_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::density_index);
							omega_sum += omega;
							mass_sum += omega * rho_nb;
							momx_sum += omega * s(ii, jj, kk, HydroSystem<MHDGalaxy>::x1Momentum_index);
							momy_sum += omega * s(ii, jj, kk, HydroSystem<MHDGalaxy>::x2Momentum_index);
							momz_sum += omega * s(ii, jj, kk, HydroSystem<MHDGalaxy>::x3Momentum_index);
						}
					}
				}

				if (omega_sum == 0.0 || mass_sum <= 0.0) {
					return;
				}

				// Mass-weighted center-of-mass velocity of the kernel gas (simulation frame).
				const double vCOMx = momx_sum / mass_sum;
				const double vCOMy = momy_sum / mass_sum;
				const double vCOMz = momz_sum / mass_sum;

				// Radial momentum magnitude -- plain calibration, no ambient-density scaling.
				const double p_terminal = sn_momentum_ref * MSUN * KM_S * std::pow(static_cast<double>(N_SN), cluster_exponent);
				const double p_radial_mag = p_terminal / vol;

				// --- Deposit pass: recenter each kernel cell's momentum onto vCOM and
				// add the outward radial kick, both directly in the simulation frame. ---
				for (int di = -kernel_ghost; di <= kernel_ghost; ++di) {
					for (int dj = -kernel_ghost; dj <= kernel_ghost; ++dj) {
						for (int dk = -kernel_ghost; dk <= kernel_ghost; ++dk) {
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] || kk < slo[2] || kk > shi[2]) {
								continue;
							}
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] || kk < dlo[2] || kk > dhi[2]) {
								continue;
							}
							if (mask_arr(ii, jj, kk) == 0) {
								continue;
							}

							const double omega = cellSphereOverlapFraction(static_cast<double>(di), static_cast<double>(dj),
												       static_cast<double>(dk), dx[0], dx[1], dx[2], rK);
							if (omega <= 0.0) {
								continue;
							}

							const double rho_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::density_index);
							const double px_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::x1Momentum_index);
							const double py_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::x2Momentum_index);
							const double pz_nb = s(ii, jj, kk, HydroSystem<MHDGalaxy>::x3Momentum_index);

							const double w_mom = omega * rho_nb / mass_sum;

							double pradx = 0.0;
							double prady = 0.0;
							double pradz = 0.0;
							if (di != 0 || dj != 0 || dk != 0) {
								const double rx = di * dx[0];
								const double ry = dj * dx[1];
								const double rz = dk * dx[2];
								const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
								const double ex = rx / r;
								const double ey = ry / r;
								const double ez = rz / r;

								double dp = p_radial_mag * w_mom;

								// Caps the kick velocity so low-density kernel cells don't tank the
								// CFL timestep; applied per-cell, so it's not momentum-conserving.
								// The residual (below) tracks how much momentum this discards.
								const double v_kick = dp / rho_nb;
								if (v_kick > v_terminal_max) {
									const double dp_uncapped = dp;
									dp = rho_nb * v_terminal_max;
									amrex::Gpu::Atomic::Add(p_capped_lev, 1);

									const double dresid = dp_uncapped - dp;
									amrex::Gpu::Atomic::Add(p_resid_x, dresid * ex);
									amrex::Gpu::Atomic::Add(p_resid_y, dresid * ey);
									amrex::Gpu::Atomic::Add(p_resid_z, dresid * ez);
								}

								pradx = dp * ex;
								prady = dp * ey;
								pradz = dp * ez;
							}

							// New momentum = rho_nb * vCOM + radial kick. The recentering delta and the
							// kick are accumulated separately, with a kernel count, so that pass 2 can
							// apply the recentering once per cell however many kernels overlap it.
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 0), rho_nb * vCOMx - px_nb);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 1), rho_nb * vCOMy - py_nb);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 2), rho_nb * vCOMz - pz_nb);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 3), pradx);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 4), prady);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 5), pradz);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 6), 1.0);
						}
					}
				}
			});
		}
		amrex::Gpu::streamSynchronize();

		int n_sn_events_lev = d_sn_events_lev.dataValue();
		int n_triggered_lev = d_triggered_lev.dataValue();
		int n_capped_lev = d_capped_lev.dataValue();
		double resid_x_lev = d_resid_x.dataValue();
		double resid_y_lev = d_resid_y.dataValue();
		double resid_z_lev = d_resid_z.dataValue();
		amrex::ParallelAllReduce::Sum(n_sn_events_lev, amrex::ParallelContext::CommunicatorSub());
		amrex::ParallelAllReduce::Sum(n_triggered_lev, amrex::ParallelContext::CommunicatorSub());
		amrex::ParallelAllReduce::Sum(n_capped_lev, amrex::ParallelContext::CommunicatorSub());
		amrex::ParallelAllReduce::Sum(resid_x_lev, amrex::ParallelContext::CommunicatorSub());
		amrex::ParallelAllReduce::Sum(resid_y_lev, amrex::ParallelContext::CommunicatorSub());
		amrex::ParallelAllReduce::Sum(resid_z_lev, amrex::ParallelContext::CommunicatorSub());

		total_sn_events_this_step += n_sn_events_lev;
		userData_.sn_trigger_count_cumulative += static_cast<amrex::Long>(n_triggered_lev);

		if (n_sn_events_lev > 0) {
			const double resid_mag = std::sqrt(resid_x_lev * resid_x_lev + resid_y_lev * resid_y_lev + resid_z_lev * resid_z_lev);
			amrex::Print() << "  [lev " << lev << "] SN events this step: " << n_sn_events_lev << " (" << n_triggered_lev << " triggered cells, "
				       << n_capped_lev << " kernel-cell kicks capped, "
				       << "residual |p| = " << resid_mag << " g cm/s"
				       << " [" << resid_x_lev << ", " << resid_y_lev << ", " << resid_z_lev << "])\n";
		}

		delta.SumBoundary(geom[lev].periodicity());

		// --- Pass 2: apply directly. Eint (and therefore temperature) held fixed;
		// only momentum (and the kinetic energy it implies) changes. ---
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();
			auto s = state.array(mfi);
			auto const &d = delta.const_array(mfi);

			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double n_kernels = d(i, j, k, 6);
				if (n_kernels <= 0.0) {
					return;
				}
				// recenter once onto the mean vCOM of the overlapping kernels, then add every kick
				const double dpx = d(i, j, k, 0) / n_kernels + d(i, j, k, 3);
				const double dpy = d(i, j, k, 1) / n_kernels + d(i, j, k, 4);
				const double dpz = d(i, j, k, 2) / n_kernels + d(i, j, k, 5);

				const double rho = s(i, j, k, HydroSystem<MHDGalaxy>::density_index);
				const double px_old = s(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
				const double py_old = s(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
				const double pz_old = s(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
				const double Eint_old = s(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index);
				const double Etot_old = s(i, j, k, HydroSystem<MHDGalaxy>::energy_index);

				const double Ekin_old = 0.5 * (px_old * px_old + py_old * py_old + pz_old * pz_old) / rho;
				const double Emag = Etot_old - Ekin_old - Eint_old;

				const double px_new = px_old + dpx;
				const double py_new = py_old + dpy;
				const double pz_new = pz_old + dpz;
				const double Ekin_new = 0.5 * (px_new * px_new + py_new * py_new + pz_new * pz_new) / rho;

				// Internal energy (and thus temperature) held fixed -- pure momentum injection.
				const double Eint_new = Eint_old;
				const double Etot_new = Eint_new + Ekin_new + Emag;

				s(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) = px_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) = py_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) = pz_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index) = Eint_new;
				s(i, j, k, HydroSystem<MHDGalaxy>::energy_index) = Etot_new;
			});
		}
		amrex::Gpu::streamSynchronize();
	}

	sn_count_ = static_cast<int>(total_sn_events_this_step);
	sn_count_cumulative_ += static_cast<int>(total_sn_events_this_step);

	if (sn_count_ > 0) {
		amrex::Print() << "SN events this step: " << sn_count_ << " | cumulative: " << sn_count_cumulative_
			       << " (trigger cells, cumulative: " << userData_.sn_trigger_count_cumulative << ")\n";
	}

	AverageDown();
}

template <>
void QuokkaSimulation<MHDGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
						    amrex::MultiFab const &state_cc, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	const double cs_disk = userData_.cs_disk;
	const double cs_cgm = userData_.cs_cgm;

	const int ncomp = ncomp_cc_in;
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();

	if (dname == "gpot") {
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "pressure") {
		auto const &state_arrs = state_cc.const_arrays();
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const double rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
			const double cs = (rho > rho_transition) ? cs_disk : cs_cgm;
			out_arrs[bx](i, j, k, ncomp) = rho * cs * cs;
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "radius_sph") {
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			const double y = prob_lo[1] + (j + 0.5) * dx[1];
			const double z = prob_lo[2] + (k + 0.5) * dx[2];
			out_arrs[bx](i, j, k, ncomp) = std::sqrt(x * x + y * y + z * z) / C::parsec / 1.0e3;
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "radial_velocity") {
		auto const &state_arrs = state_cc.const_arrays();
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const double rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
			const double vx = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) / rho;
			const double vy = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) / rho;
			const double vz = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) / rho;
			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			const double y = prob_lo[1] + (j + 0.5) * dx[1];
			const double z = prob_lo[2] + (k + 0.5) * dx[2];
			const double r = std::sqrt(x * x + y * y + z * z);
			out_arrs[bx](i, j, k, ncomp) = (r > 0.0) ? ((x * vx + y * vy + z * vz) / r) / 1.0e5 : 0.0;
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "circular_velocity") {
		auto const &state_arrs = state_cc.const_arrays();
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const double rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
			const double vx = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) / rho;
			const double vy = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) / rho;
			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			const double y = prob_lo[1] + (j + 0.5) * dx[1];
			const double r_cyl = std::sqrt(x * x + y * y);
			out_arrs[bx](i, j, k, ncomp) = (r_cyl > 0.0) ? ((x * vy - y * vx) / r_cyl) / 1.0e5 : 0.0;
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "mach") {
		auto const &state_arrs = state_cc.const_arrays();
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const double rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
			const double momx = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
			const double momy = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
			const double momz = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
			const double cs = (rho > rho_transition) ? cs_disk : cs_cgm;
			const double v2 = (momx * momx + momy * momy + momz * momz) / (rho * rho);
			out_arrs[bx](i, j, k, ncomp) = std::sqrt(v2) / cs;
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "plasma_beta") {
		auto const &state_arrs = state_cc.const_arrays();
		auto const &Bx_arrs = state_fc[0].const_arrays();
		auto const &By_arrs = state_fc[1].const_arrays();
		auto const &Bz_arrs = state_fc[2].const_arrays();
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const double rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
			const double cs = (rho > rho_transition) ? cs_disk : cs_cgm;
			const double Pgas = rho * cs * cs;
			const double Bx_cc = 0.5 * (Bx_arrs[bx](i, j, k) + Bx_arrs[bx](i + 1, j, k));
			const double By_cc = 0.5 * (By_arrs[bx](i, j, k) + By_arrs[bx](i, j + 1, k));
			const double Bz_cc = 0.5 * (Bz_arrs[bx](i, j, k) + Bz_arrs[bx](i, j, k + 1));
			const double Pmag = amrex::max(0.5 * (Bx_cc * Bx_cc + By_cc * By_cc + Bz_cc * Bz_cc), 1.0e-30);
			out_arrs[bx](i, j, k, ncomp) = Pgas / Pmag;
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "divB") {
		const double idx = 1.0 / dx[0];
		const double idy = 1.0 / dx[1];
		const double idz = 1.0 / dx[2];
		auto const &Bx_arrs = state_fc[0].const_arrays();
		auto const &By_arrs = state_fc[1].const_arrays();
		auto const &Bz_arrs = state_fc[2].const_arrays();
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			out_arrs[bx](i, j, k, ncomp) = (Bx_arrs[bx](i + 1, j, k) - Bx_arrs[bx](i, j, k)) * idx +
						       (By_arrs[bx](i, j + 1, k) - By_arrs[bx](i, j, k)) * idy +
						       (Bz_arrs[bx](i, j, k + 1) - Bz_arrs[bx](i, j, k)) * idz;
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	if (dname == "Bphi") {
		auto const &Bx_arrs = state_fc[0].const_arrays();
		auto const &By_arrs = state_fc[1].const_arrays();
		auto out_arrs = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			const double y = prob_lo[1] + (j + 0.5) * dx[1];
			const double R2 = x * x + y * y;
			if (R2 < 1e-20) {
				out_arrs[bx](i, j, k, ncomp) = 0.0;
				return;
			}
			const double Bx_cc = 0.5 * (Bx_arrs[bx](i, j, k) + Bx_arrs[bx](i + 1, j, k));
			const double By_cc = 0.5 * (By_arrs[bx](i, j, k) + By_arrs[bx](i, j + 1, k));
			out_arrs[bx](i, j, k, ncomp) = (By_cc * x - Bx_cc * y) / std::sqrt(R2);
		});
		amrex::Gpu::streamSynchronize();
		return;
	}

	// default case, set field NaN
	auto out_arrs = mf.arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { out_arrs[bx](i, j, k, ncomp) = NAN; });
	amrex::Gpu::streamSynchronize();
}

template <> auto QuokkaSimulation<MHDGalaxy>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	// One-time check of the measured gas-pull table (mhd_galaxy.gas_pull_file) against this run's own gravity.
	// The first statistics call comes right after the initial Poisson solve, so phi is the live gas potential of
	// the initial disk. A table measured from a different setup (disk, cs, vc, grid, refinement) fails here.
	static bool gas_pull_checked = false;
	if (!gas_pull_checked && userData_.gas_pull_nR > 0) {
		gas_pull_checked = true;
		if (tNew_[0] > 0.0) {
			amrex::Print() << "Gas pull table check skipped: restart at t > 0, the disk has evolved since the table's snapshot\n";
		} else {
			const amrex::Real *pull_table = userData_.gas_pull_device.data();
			const int pull_nR = userData_.gas_pull_nR;
			const int pull_nz = userData_.gas_pull_nz;
			const double pull_Rmax = userData_.gas_pull_Rmax;
			const double pull_z0 = userData_.gas_pull_z0;
			const double pull_smax = userData_.gas_pull_smax;
			const double Sigma0 = userData_.Sigma0;
			const double Rd = userData_.Rd;
			const amrex::Real R_lo = 0.5 * 1.0e3 * C::parsec;
			const amrex::Real R_hi = 12.0 * 1.0e3 * C::parsec;
			// mass-weighted: sum m (g_tab - g)^2, sum m g^2, sum m (g_tab - g), sum m |g|, and the same first two for
			// the analytic midplane pull, for reference
			std::array<amrex::Real, 6> sums{};
			for (int lev = 0; lev <= finest_level; ++lev) {
				const auto prob_lo = geom[lev].ProbLoArray();
				const auto dx = geom[lev].CellSizeArray();
				const amrex::Real vol = dx[0] * dx[1] * dx[2];
				const int ng = phi[lev].nGrow();
				amrex::iMultiFab mask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(), 1, 0);
				if (lev < finest_level) {
					mask = amrex::makeFineMask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(),
								   state_new_cc_[lev + 1].boxArray(), refRatio(lev), 1, 0);
				} else {
					mask.setVal(1);
				}
				auto const &mask_arrs = mask.const_arrays();
				auto const &state_arrs = state_new_cc_[lev].const_arrays();
				auto const &phi_arrs = phi[lev].const_arrays();
				auto r = amrex::ParReduce(
				    amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
						    amrex::ReduceOpSum>{},
				    amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>{}, state_new_cc_[lev],
				    amrex::IntVect(0), 1,
				    [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int /*n*/) noexcept
				    -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
					    auto const &p = phi_arrs[bx];
					    // central differences only where both neighbours are valid cells of this box (no ghost cells needed)
					    const bool interior = i - 1 >= p.begin[0] + ng && i + 1 < p.end[0] - ng && j - 1 >= p.begin[1] + ng &&
								  j + 1 < p.end[1] - ng && k - 1 >= p.begin[2] + ng && k + 1 < p.end[2] - ng;
					    const amrex::Real rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
					    const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
					    const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
					    const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
					    const amrex::Real R = std::sqrt(x * x + y * y);
					    if (mask_arrs[bx](i, j, k) == 0 || !interior || rho <= rho_transition || R < R_lo || R > R_hi) {
						    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
					    }
					    const amrex::Real gx = -0.5 * (p(i + 1, j, k) - p(i - 1, j, k)) / dx[0];
					    const amrex::Real gy = -0.5 * (p(i, j + 1, k) - p(i, j - 1, k)) / dx[1];
					    const amrex::Real g = (x * gx + y * gy) / R; // live g_R, < 0 inward
					    const amrex::Real d = gasPullTable(pull_table, pull_nR, pull_nz, pull_Rmax, pull_z0, pull_smax, R, z) - g;
					    const amrex::Real d_ana = -surfaceDensityPull(R, Sigma0, Rd) - g;
					    const amrex::Real m = rho * vol;
					    return {m * d * d, m * g * g, m * d, m * std::abs(g), m * d_ana * d_ana, 0.0};
				    });
				sums[0] += amrex::get<0>(r);
				sums[1] += amrex::get<1>(r);
				sums[2] += amrex::get<2>(r);
				sums[3] += amrex::get<3>(r);
				sums[4] += amrex::get<4>(r);
			}
			amrex::ParallelAllReduce::Sum(sums.data(), static_cast<int>(sums.size()), amrex::ParallelContext::CommunicatorSub());
			const amrex::Real rms = (sums[1] > 0.0) ? std::sqrt(sums[0] / sums[1]) : 0.0;
			const amrex::Real bias = (sums[3] > 0.0) ? sums[2] / sums[3] : 0.0;
			const amrex::Real rms_ana = (sums[1] > 0.0) ? std::sqrt(sums[4] / sums[1]) : 0.0;
			amrex::Real tolerance = 0.03;
			amrex::ParmParse const pp("mhd_galaxy");
			pp.query("gas_pull_tolerance", tolerance);
			amrex::Print() << "Gas pull table check against the live gravity (disk gas, 0.5 < R < 12 kpc, mass-weighted): rms error = " << 100.0 * rms
				       << "%, bias = " << 100.0 * bias << "% (analytic midplane pull would be " << 100.0 * rms_ana << "% rms); tolerance "
				       << 100.0 * tolerance << "%\n";
			if (rms > tolerance) {
				amrex::Abort("gas_pull_file does not match this run's gravity (rms error above mhd_galaxy.gas_pull_tolerance): it was measured "
					     "from a different setup. Regenerate it with measure_gas_pull.py from a t = 0 snapshot of this setup, or unset "
					     "mhd_galaxy.gas_pull_file to use the analytic pull.");
			}
		}
	}
	const amrex::Real R_min = 2.0 * 1.0e3 * C::parsec;
	const amrex::Real R_max = 8.0 * 1.0e3 * C::parsec;
	const amrex::Real z_max = 0.5 * 1.0e3 * C::parsec;

	// Threshold for "field-bearing" cells: well below seed B_rms (~1e-6 G), well above
	// the CGM's essentially-zero field, so the normalized divB ratio isn't dominated
	// by near-zero-B halo cells dividing tiny residuals into huge ratios.
	constexpr amrex::Real Bmag_threshold = 1.0e-9;

	amrex::Real total_E_tot = 0.0;
	amrex::Real total_E_tor = 0.0;
	amrex::Real total_E_pol = 0.0;

	amrex::Real divB_max_global = 0.0;
	amrex::Real divB_sumsq_global = 0.0;
	amrex::Real divB_norm_sumsq_global = 0.0;
	amrex::Real divB_norm_ncells_global = 0.0;

	using FaceStateArray = std::array<amrex::Array4<const Real>, AMREX_SPACEDIM>;

	const amrex::Real mean_density = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    return state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
	    });
	stats["mean_density"] = mean_density / geom[0].ProbSize();

	const amrex::Real disk_mass = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
		    return (rho > rho_transition) ? rho : static_cast<amrex::Real>(0.0);
	    });

	const amrex::Real disk_volume = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
		    return (rho > rho_transition) ? static_cast<amrex::Real>(1.0) : static_cast<amrex::Real>(0.0);
	    });

	const amrex::Real mean_disk_density = (disk_volume > 0.0) ? (disk_mass / disk_volume) : static_cast<amrex::Real>(1.0);

	stats["mean_disk_density"] = mean_disk_density;
	stats["disk_mass"] = disk_mass / C::M_solar;

	const amrex::Real sigma_vol = computeVolumeIntegral(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
		    if (rho <= rho_transition) {
			    return static_cast<amrex::Real>(0.0);
		    }
		    const amrex::Real eta = std::log(rho / mean_disk_density);
		    return eta * eta;
	    });

	stats["sigma_eta"] = (disk_volume > 0.0) ? std::sqrt(sigma_vol / disk_volume) : static_cast<amrex::Real>(0.0);

	for (int lev = 0; lev <= finest_level; ++lev) {
		const auto &geom_lev = geom[lev];
		auto const &state_fc = state_new_fc_[lev];
		auto const &state_fc_x = state_fc[0].const_arrays();
		auto const &state_fc_y = state_fc[1].const_arrays();
		auto const &state_fc_z = state_fc[2].const_arrays();
		const auto prob_lo = geom_lev.ProbLoArray();
		const auto dx = geom_lev.CellSizeArray();
		const amrex::Real vol = dx[0] * dx[1] * dx[2];
		const amrex::Real idx = 1.0 / dx[0];
		const amrex::Real idy = 1.0 / dx[1];
		const amrex::Real idz = 1.0 / dx[2];
		const amrex::Real dx_min = amrex::min(dx[0], amrex::min(dx[1], dx[2]));

		amrex::iMultiFab mask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(), 1, 0);
		if (lev < finest_level) {
			// covered cells -> 0, uncovered -> 1
			mask = amrex::makeFineMask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(), state_new_cc_[lev + 1].boxArray(),
						   refRatio(lev), 1, 0);
		} else {
			mask.setVal(1);
		}
		auto const &mask_arrs = mask.const_arrays();

		// Combined reduction: [0]=E_tot, [1]=E_tor, [2]=E_pol (annulus-masked),
		//                      [3]=max|divB| (whole domain), [4]=sum(divB^2 * vol) (whole domain),
		//                      [5]=sum(divB_norm^2) (field-bearing cells only),
		//                      [6]=field-bearing cell count
		auto level_result =
		    amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpMax, amrex::ReduceOpSum,
						     amrex::ReduceOpSum, amrex::ReduceOpSum>{},
				     amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>{},
				     state_new_cc_[lev], amrex::IntVect(0), 1,
				     [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int /*n*/) noexcept
					 -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
					     if (mask_arrs[bx](i, j, k) == 0) {
						     return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
					     }

					     const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
					     const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
					     const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
					     const amrex::Real R = std::sqrt(x * x + y * y);

					     const amrex::Real Bx_cc = 0.5 * (state_fc_x[bx](i, j, k) + state_fc_x[bx](i + 1, j, k));
					     const amrex::Real By_cc = 0.5 * (state_fc_y[bx](i, j, k) + state_fc_y[bx](i, j + 1, k));
					     const amrex::Real Bz_cc = 0.5 * (state_fc_z[bx](i, j, k) + state_fc_z[bx](i, j, k + 1));

					     // --- divB diagnostics: raw max/RMS computed everywhere (whole domain), so
					     //     boundary/grid-decomposition artifacts outside the disk are still caught ---
					     const amrex::Real divB = (state_fc_x[bx](i + 1, j, k) - state_fc_x[bx](i, j, k)) * idx +
								      (state_fc_y[bx](i, j + 1, k) - state_fc_y[bx](i, j, k)) * idy +
								      (state_fc_z[bx](i, j, k + 1) - state_fc_z[bx](i, j, k)) * idz;

					     const amrex::Real divB_abs = std::abs(divB);
					     const amrex::Real divB_sq_vol = divB * divB * vol;

					     // --- normalized divB: restricted to field-bearing cells (Option A), so
					     //     near-zero-B CGM cells don't blow up the ratio with tiny residuals ---
					     const amrex::Real Bmag = std::sqrt(Bx_cc * Bx_cc + By_cc * By_cc + Bz_cc * Bz_cc);
					     const bool has_field = (Bmag > Bmag_threshold);

					     const amrex::Real divB_norm = has_field ? (divB_abs * dx_min / Bmag) : static_cast<amrex::Real>(0.0);
					     const amrex::Real divB_norm_sq = has_field ? (divB_norm * divB_norm) : static_cast<amrex::Real>(0.0);
					     const amrex::Real norm_cell_count = has_field ? static_cast<amrex::Real>(1.0) : static_cast<amrex::Real>(0.0);

					     // --- annulus-restricted toroidal/poloidal energy stats (unchanged) ---
					     amrex::Real e_tot = 0.0;
					     amrex::Real e_tor = 0.0;
					     amrex::Real e_pol = 0.0;
					     if (R >= R_min && R <= R_max && std::abs(z) <= z_max) {
						     const amrex::Real invR = (R > 0.0) ? 1.0 / R : 0.0;
						     const amrex::Real cos_phi = x * invR;
						     const amrex::Real sin_phi = y * invR;

						     const amrex::Real Bphi = -Bx_cc * sin_phi + By_cc * cos_phi;
						     const amrex::Real Br = Bx_cc * cos_phi + By_cc * sin_phi;

						     e_tot = 0.5 * (Bx_cc * Bx_cc + By_cc * By_cc + Bz_cc * Bz_cc) * vol;
						     e_tor = 0.5 * (Bphi * Bphi) * vol;
						     e_pol = 0.5 * (Br * Br + Bz_cc * Bz_cc) * vol;
					     }

					     return {e_tot, e_tor, e_pol, divB_abs, divB_sq_vol, divB_norm_sq, norm_cell_count};
				     });

		total_E_tot += amrex::get<0>(level_result);
		total_E_tor += amrex::get<1>(level_result);
		total_E_pol += amrex::get<2>(level_result);

		divB_max_global = amrex::max(divB_max_global, amrex::get<3>(level_result));
		divB_sumsq_global += amrex::get<4>(level_result);
		divB_norm_sumsq_global += amrex::get<5>(level_result);
		divB_norm_ncells_global += amrex::get<6>(level_result);
	}

	std::array<amrex::Real, 3> global_sums = {total_E_tot, total_E_tor, total_E_pol};
	amrex::ParallelAllReduce::Sum(global_sums.data(), 3, amrex::ParallelContext::CommunicatorSub());

	stats["energy_Btot_annulus"] = global_sums[0];
	stats["energy_Btor_annulus"] = global_sums[1];
	stats["energy_Bpol_annulus"] = global_sums[2];

	amrex::Real divB_max_reduced = divB_max_global;
	amrex::ParallelAllReduce::Max(divB_max_reduced, amrex::ParallelContext::CommunicatorSub());

	std::array<amrex::Real, 3> divB_sums = {divB_sumsq_global, divB_norm_sumsq_global, divB_norm_ncells_global};
	amrex::ParallelAllReduce::Sum(divB_sums.data(), 3, amrex::ParallelContext::CommunicatorSub());

	const amrex::Real total_volume = geom[0].ProbSize();
	stats["divB_max"] = divB_max_reduced;
	stats["divB_rms"] = std::sqrt(divB_sums[0] / total_volume);
	stats["divB_rms_normalized"] = (divB_sums[2] > 0.0) ? std::sqrt(divB_sums[1] / divB_sums[2]) : static_cast<amrex::Real>(0.0);
	stats["sn_count_cumulative"] = static_cast<amrex::Real>(sn_count_cumulative_);
	stats["sn_trigger_count_cumulative"] = static_cast<amrex::Real>(userData_.sn_trigger_count_cumulative);

	// --- Equilibrium diagnostics: mass-weighted kinematics and shape of the disk gas in
	//     R_eq_min <= R <= R_max. A disk in rotational + hydrostatic balance keeps vR, vz ~ 0
	//     and z_rms, R_mean fixed; drifts in these measure the gravity/rotation imbalance. ---
	const amrex::Real R_eq_min = 1.0 * 1.0e3 * C::parsec; // excludes the axis, where 1/R terms are noisy
	std::array<amrex::Real, 7> eq_sums{}; // M, M vR, M vR^2, M vz^2, M vphi, M z^2, M R
	for (int lev = 0; lev <= finest_level; ++lev) {
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
		const amrex::Real vol = dx[0] * dx[1] * dx[2];

		amrex::iMultiFab mask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(), 1, 0);
		if (lev < finest_level) {
			mask = amrex::makeFineMask(state_new_cc_[lev].boxArray(), state_new_cc_[lev].DistributionMap(), state_new_cc_[lev + 1].boxArray(),
						   refRatio(lev), 1, 0);
		} else {
			mask.setVal(1);
		}
		auto const &mask_arrs = mask.const_arrays();
		auto const &state_arrs = state_new_cc_[lev].const_arrays();

		auto level_result =
		    amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
						     amrex::ReduceOpSum, amrex::ReduceOpSum>{},
				     amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>{},
				     state_new_cc_[lev], amrex::IntVect(0), 1,
				     [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int /*n*/) noexcept
					 -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
					     if (mask_arrs[bx](i, j, k) == 0) {
						     return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
					     }
					     const amrex::Real rho = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::density_index);
					     const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
					     const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
					     const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
					     const amrex::Real R = std::sqrt(x * x + y * y);
					     if (rho <= rho_transition || R < R_eq_min || R > R_max) {
						     return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
					     }
					     const amrex::Real vx = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) / rho;
					     const amrex::Real vy = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) / rho;
					     const amrex::Real vz = state_arrs[bx](i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) / rho;
					     const amrex::Real vR = (vx * x + vy * y) / R;
					     const amrex::Real vphi = (-vx * y + vy * x) / R;
					     const amrex::Real m = rho * vol;
					     return {m, m * vR, m * vR * vR, m * vz * vz, m * vphi, m * z * z, m * R};
				     });
		eq_sums[0] += amrex::get<0>(level_result);
		eq_sums[1] += amrex::get<1>(level_result);
		eq_sums[2] += amrex::get<2>(level_result);
		eq_sums[3] += amrex::get<3>(level_result);
		eq_sums[4] += amrex::get<4>(level_result);
		eq_sums[5] += amrex::get<5>(level_result);
		eq_sums[6] += amrex::get<6>(level_result);
	}
	amrex::ParallelAllReduce::Sum(eq_sums.data(), static_cast<int>(eq_sums.size()), amrex::ParallelContext::CommunicatorSub());

	const amrex::Real eq_mass = eq_sums[0];
	const amrex::Real eq_inv_mass = (eq_mass > 0.0) ? 1.0 / eq_mass : 0.0;
	constexpr amrex::Real km_s = 1.0e5;
	stats["eq_disk_mass"] = eq_mass / C::M_solar;					  // Msun
	stats["eq_vR_mean"] = eq_sums[1] * eq_inv_mass / km_s;				  // km/s, net inflow (<0) / outflow (>0)
	stats["eq_vR_rms"] = std::sqrt(eq_sums[2] * eq_inv_mass) / km_s;		  // km/s
	stats["eq_vz_rms"] = std::sqrt(eq_sums[3] * eq_inv_mass) / km_s;		  // km/s
	stats["eq_vphi_mean"] = eq_sums[4] * eq_inv_mass / km_s;			  // km/s
	stats["eq_z_rms"] = std::sqrt(eq_sums[5] * eq_inv_mass) / C::parsec;		  // pc, vertical thickness
	stats["eq_R_mean"] = eq_sums[6] * eq_inv_mass / (1.0e3 * C::parsec);		  // kpc, radial redistribution
	return stats;
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<MHDGalaxy>(quokka::BCType::reflecting);

	const int nvars_fc = Physics_Indices<MHDGalaxy>::nvarTotal_fc;
	const int nvars_per_dim_fc = Physics_Indices<MHDGalaxy>::nvarPerDim_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		const int component_dir = (nvars_per_dim_fc > 0) ? (icomp / nvars_per_dim_fc) : 0;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			const int bc_type = (component_dir == idim) ? amrex::BCType::reflect_even : amrex::BCType::reflect_odd;
			BCs_fc[icomp].setLo(idim, bc_type);
			BCs_fc[icomp].setHi(idim, bc_type);
		}
	}

	QuokkaSimulation<MHDGalaxy> sim(BCs_cc, BCs_fc);
	sim.preCalculateInitialConditions();
	sim.setInitialConditions();
	cleanup(sim);
	sim.evolve();
	return 0;
}
