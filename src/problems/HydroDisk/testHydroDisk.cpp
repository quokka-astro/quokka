//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDiskGalaxy.cpp
/// \brief Defines a simulation using disk galaxy initial conditions.
///

#include <cmath>
#include <optional>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Parser.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "math/quadrature.hpp"
#include "particles/particle_types.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/DataTable.hpp"

namespace
{
	constexpr double keV_in_ergs = 1000.0 * C::ev2erg; 	// ergs == 1 keV
	constexpr double seconds_per_year = 3.15576e7;
	constexpr double Rd_kpc = 3.0; 						// disk scale length in kpc
	constexpr double Rc_kpc = 2.0; 						// rotation curve core radius in kpc
	constexpr double Rd = Rd_kpc * 1.0e3 * C::parsec;
	constexpr double Rc = Rc_kpc * 1.0e3 * C::parsec;
	constexpr double alpha_profile = 2.0;               // Eq. 1 shape parameter
	constexpr double beta_profile  = 0.5;               // Eq. 1 shape parameter
	constexpr double q_flatten     = 0.7;               // Binney & Tremaine flattening (Arora+25 Eq. A.3)
	constexpr double T_cgm         = 1.0e7;             // CGM temperature [K]
	constexpr double rho_transition = 1.0e-28; 			// g/cm^3, disc-CGM interface density
	constexpr double Rmax_kpc = 8.0;
	constexpr double Rmax = Rmax_kpc * 1.0e3 * C::parsec;

	constexpr double refine_Rcyl_kpc = 6.0;
	constexpr double refine_Hcyl_pc  = 100.0;
	constexpr double refine_Rcyl     = refine_Rcyl_kpc * 1.0e3 * C::parsec;
	constexpr double refine_Hcyl     = refine_Hcyl_pc  * C::parsec;
}

struct HDGalaxy {
};

static_assert(AMREX_SPACEDIM == 3, "Hydro disk galaxy problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<HDGalaxy> {
	static constexpr double gamma = 1.;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double cs_isothermal = 7.0e5; // 7 km/s in cm/s
};

template <> struct HydroSystem_Traits<HDGalaxy> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Particle_Traits<HDGalaxy> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
};

template <> struct Physics_Traits<HDGalaxy> {
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 0;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr int nGroups = 1;
};

template <> struct SimulationData<HDGalaxy> {
	// Primary dimensionless parameters (Table 1 of Arora+25)
	amrex::Real Q_mean{};
	amrex::Real Mc{};
	amrex::Real vc{};       // saturated circular velocity [cm/s]
	amrex::Real Sigma0{};   // surface density normalisation [g/cm^2]
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto surfaceDensityProfile(double R, double Sigma0) -> double
{
	const double x = R / Rd;
	return Sigma0 * std::exp(-x - beta_profile * std::exp(-alpha_profile * x));
}

/// Arora+25 Appendix A, Eq. A.2:
///   phi_g(R) = 4*pi*G*Sigma0*Rd * y^2 * [I0(y)*K0(y) - I1(y)*K1(y)],  y = R/(2*Rd)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto dPhiGas_dR(double R, double Sigma0) -> double
{
	const double y = R / (2.0 * Rd);

	// I0 via Abramowitz & Stegun 9.8.1 / 9.8.2
	auto bessel_I0 = [](double x) -> double {
		x = std::abs(x);
		if (x <= 3.75) {
			const double t = (x / 3.75) * (x / 3.75);
			return 1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492
			     + t*(0.2659732 + t*(0.0360768 + t*0.0045813)))));
		}
		const double t = 3.75 / x;
		return (std::exp(x) / std::sqrt(x)) *
		       (0.39894228 + t*(0.01328592 + t*(0.00225319 + t*(-0.00157565
		     + t*(0.00916281 + t*(-0.02057706 + t*(0.02635537
		     + t*(-0.01647633 + t*0.00392377))))))));
	};

	// I1 via A&S 9.8.3 / 9.8.4
	auto bessel_I1 = [](double x) -> double {
		const double sign = (x >= 0.0) ? 1.0 : -1.0;
		x = std::abs(x);
		if (x <= 3.75) {
			const double t = (x / 3.75) * (x / 3.75);
			return sign * x * (0.5 + t*(0.87890594 + t*(0.51498869 + t*(0.15084934
			     + t*(0.02658733 + t*(0.00301532 + t*0.00032411))))));
		}
		const double t = 3.75 / x;
		return sign * (std::exp(x) / std::sqrt(x)) *
		       (0.39894228 + t*(-0.03988024 + t*(-0.00362018 + t*(0.00163801
		     + t*(-0.01031555 + t*(0.02282967 + t*(-0.02895312
		     + t*(0.01787654 - t*0.00420059))))))));
	};

	// K0 via A&S 9.8.5 / 9.8.6
	auto bessel_K0 = [&bessel_I0](double x) -> double {
		if (x <= 2.0) {
			const double t = (x / 2.0) * (x / 2.0);
			return -std::log(x / 2.0) * bessel_I0(x)
			     + (-0.57721566 + t*(0.42278420 + t*(0.23069756 + t*(0.03488590
			     + t*(0.00262698 + t*(0.00010750 + t*0.0000074))))));
		}
		const double t = 2.0 / x;
		return (std::exp(-x) / std::sqrt(x)) *
		       (1.25331414 + t*(-0.07832358 + t*(0.02189568 + t*(-0.01062446
		     + t*(0.00587872 + t*(-0.00251540 + t*0.00053208))))));
	};

	// K1 via A&S 9.8.7 / 9.8.8
	auto bessel_K1 = [&bessel_I1](double x) -> double {
		if (x <= 2.0) {
			const double t = (x / 2.0) * (x / 2.0);
			return std::log(x / 2.0) * bessel_I1(x)
			     + (1.0 / x) * (1.0 + t*(0.15443144 + t*(-0.67278579 + t*(-0.18156897
			     + t*(-0.01919402 + t*(-0.00110404 + t*(-0.00004686)))))));
		}
		const double t = 2.0 / x;
		return (std::exp(-x) / std::sqrt(x)) *
		       (1.25331414 + t*(0.23498619 + t*(-0.03655620 + t*(0.01504268
		     + t*(-0.00780353 + t*(0.00325614 - t*0.00068245))))));
	};

	const double I0v = bessel_I0(y);
	const double I1v = bessel_I1(y);
	const double K0v = bessel_K0(y);
	const double K1v = bessel_K1(y);

	// d(phi_g)/dR = 2*pi*G*Sigma0 * 2y * [I0*K0 + y*(I1*K0 - I0*K1)]
	const double dfdy = 2.0 * y * (I0v*K0v + y*(I1v*K0v - I0v*K1v));
	return 2.0 * M_PI * C::Gconst * Sigma0 * dfdy;
}

template <> void QuokkaSimulation<HDGalaxy>::preCalculateInitialConditions()
{
	amrex::ParmParse const pp("hd_galaxy");

	pp.get("Mc", userData_.Mc);
	pp.get("Q_mean", userData_.Q_mean);

	userData_.vc = userData_.Mc * quokka::EOS_Traits<HDGalaxy>::cs_isothermal;
	const double vc = userData_.vc;
	constexpr double cs = quokka::EOS_Traits<HDGalaxy>::cs_isothermal;

	auto integrand = [=](double R) {
		const double Omega = vc / std::sqrt(R * R + Rc * Rc);
		const double dOmegadR = -vc * R / std::pow(R * R + Rc * Rc, 1.5);
		const double kappa = std::sqrt(std::max(4.0 * Omega * Omega + 2.0 * R * Omega * dOmegadR, 0.0));
		const double fR = surfaceDensityProfile(R, 1.0); // Sigma0=1 since it cancels out
		return kappa * cs / (M_PI * C::Gconst * fR);
	};

	const double integral = quad_1d(integrand, 1.0, Rmax);
	userData_.Sigma0 = integral / (userData_.Q_mean * Rmax);

	amrex::Print() << "HDGalaxy:"
	               << " Q_mean = " << userData_.Q_mean
	               << ", Mc = " << userData_.Mc
	               << ", vc = " << vc / 1.0e5 << " km/s"
	               << ", Sigma0 = " << userData_.Sigma0 << " g/cm^2\n";
}

template <> void QuokkaSimulation<HDGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const double vc = userData_.vc;
	const double Sigma0 = userData_.Sigma0;
	constexpr double cs = quokka::EOS_Traits<HDGalaxy>::cs_isothermal;

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
		const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
		const amrex::Real R = std::sqrt(x * x + y * y);

		const double Sigma_R = surfaceDensityProfile(R, Sigma0);
		const double h = cs * cs / (M_PI * C::Gconst * Sigma_R);
		const double rho_mid = Sigma_R / (std::sqrt(2.0 * M_PI) * h);
		const double rho_disc = rho_mid * std::exp(-0.5 * z * z / (h * h));

		const double rho = std::max(rho_disc, rho_transition);

		const double vrot = vc * R / std::sqrt(R * R + Rc * Rc);
		double vx = 0.0;
		double vy = 0.0;
		const double vz = 0.0;
		if (R > 0.0) {
			vx = -vrot * y / R;
			vy =  vrot * x / R;
		}

		const double momx = rho * vx;
		const double momy = rho * vy;
		const double momz = rho * vz;
		const double Ekin = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		const double Eint = rho * cs * cs;
		const double Etot = Ekin + Eint;

		state_cc(i, j, k, HydroSystem<HDGalaxy>::density_index)       = rho;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index)    = momx;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index)    = momy;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index)    = momz;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::energy_index)         = Etot;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<HDGalaxy>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real /*time*/, amrex::Real dt_lev)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx = geom[lev].CellSizeArray();
	const amrex::Real dt = dt_lev;

	const double vc     = userData_.vc;
	const double Sigma0 = userData_.Sigma0;
	constexpr double cs = quokka::EOS_Traits<HDGalaxy>::cs_isothermal;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			const double y = prob_lo[1] + (j + 0.5) * dx[1];
			const double z = prob_lo[2] + (k + 0.5) * dx[2];
			const double R2 = x*x + y*y;
			const double R  = std::sqrt(R2);

			// Flattened denominator D = R^2 + Rc^2 + (z/q)^2  (Arora+25 Eq. A.3)
			const double z_over_q = z / q_flatten;
			const double D = R2 + Rc*Rc + z_over_q * z_over_q;

			// Radial acceleration from dark matter halo + pressure correction:
			//   -d(psi)/dR = -vc^2 * R / D           (halo, Arora+25 Eq. A.3)
			//   -cs^2 / Rd                            (pressure correction, Eq. A.5)
			// Note: gas self-gravity -d(phi_g)/dR is handled by the Poisson solver.
			double g_R = 0.0;
			if (R > 0.0) {
				g_R = (-vc*vc * R / D) + (-cs*cs / Rd);
			}

			// Vertical acceleration:
			//   -d(psi)/dz = -vc^2 * (z/q^2) / D         (halo, Eq. A.3)
			//   d(phi_g)/dz = 0                           (midplane approximation, Appendix A)
			const double g_z = -vc*vc * z / (q_flatten * q_flatten * D);

			// Project radial acceleration into Cartesian components
			double gx = 0.0;
			double gy = 0.0;
			if (R > 0.0) {
				gx = g_R * x / R;
				gy = g_R * y / R;
			}

			const double rho   = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
			const double x1mom = state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index);
			const double x2mom = state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index);
			const double x3mom = state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index);
			const double Egas  = state(i, j, k, HydroSystem<HDGalaxy>::energy_index);

			// Conserve internal energy across the momentum kick
			const double Ekin = 0.5 * (x1mom*x1mom + x2mom*x2mom + x3mom*x3mom) / rho;
			const double Eint = Egas - Ekin;

			const double x1mom_new = x1mom + dt * rho * gx;
			const double x2mom_new = x2mom + dt * rho * gy;
			const double x3mom_new = x3mom + dt * rho * g_z;

			state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index) = x3mom_new;

			// Update total energy, keeping internal energy fixed
			const double Ekin_new = 0.5 * (x1mom_new*x1mom_new + x2mom_new*x2mom_new + x3mom_new*x3mom_new) / rho;
			const double Egas_new = Ekin_new + Eint;
			state(i, j, k, HydroSystem<HDGalaxy>::energy_index) = Egas_new;
		});
	}
}


template <> void QuokkaSimulation<HDGalaxy>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real x0 = prob_lo[0] + (i * dx[0]);
		const amrex::Real y0 = prob_lo[1] + (j * dx[1]);
		const amrex::Real z0 = prob_lo[2] + (k * dx[2]);
		const amrex::Real x1 = prob_lo[0] + ((i + 1) * dx[0]);
		const amrex::Real y1 = prob_lo[1] + ((j + 1) * dx[1]);
		const amrex::Real z1 = prob_lo[2] + ((k + 1) * dx[2]);

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			const amrex::Real R = std::sqrt(x * x + y * y);
			if ((R < refine_Rcyl) && (std::abs(z) < refine_Hcyl)) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};
		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfPointInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<HDGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k);
		});
		amrex::Gpu::streamSynchronize();
	}

	if (dname == "pressure") {
		const int ncomp = ncomp_cc_in;
		constexpr double cs = quokka::EOS_Traits<HDGalaxy>::cs_isothermal;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
				output(i, j, k, ncomp) = rho * cs * cs / C::k_B;
			});
		}
	}

	if (dname == "radius_sph") {
		const int ncomp = ncomp_cc_in;
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
				const amrex::Real r_cm = std::sqrt(x * x + y * y + z * z);
				output(i, j, k, ncomp) = r_cm / 3.08567758e21;
			});
		}
	}

	if (dname == "radial_velocity") {
		const int ncomp = ncomp_cc_in;
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
				const amrex::Real vx = state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index) / rho;
				const amrex::Real vy = state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index) / rho;
				const amrex::Real vz = state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index) / rho;
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
				const amrex::Real r_cm = std::sqrt(x * x + y * y + z * z);
				output(i, j, k, ncomp) = (r_cm > 0.0) ? ((x * vx + y * vy + z * vz) / r_cm) / 1.0e5 : 0.0;
			});
		}
	}

	if (dname == "circular_velocity") {
		const int ncomp = ncomp_cc_in;
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
				const amrex::Real vx = state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index) / rho;
				const amrex::Real vy = state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index) / rho;
				const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				const amrex::Real r_cyl = std::sqrt(x * x + y * y);
				output(i, j, k, ncomp) = (r_cyl > 0.0) ? ((x * vy - y * vx) / r_cyl) / 1.0e5 : 0.0;
			});
		}
	}
}

template <> auto QuokkaSimulation<HDGalaxy>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	const amrex::Real mean_Sigma = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		return state(i, j, k, HydroSystem<HDGalaxy>::density_index);
	});
	stats["mean_density"] = mean_Sigma;

	const amrex::Real sigma_eta_sq = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
		const amrex::Real eta = std::log(rho / mean_Sigma);
		return eta * eta;
	});
	stats["sigma_eta"] = std::sqrt(sigma_eta_sq);

	const amrex::Real disk_mass = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		return state(i, j, k, HydroSystem<HDGalaxy>::density_index);
	});
	stats["disk_mass"] = disk_mass / C::M_solar;

	return stats;
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<HDGalaxy>(quokka::BCType::foextrap);

	QuokkaSimulation<HDGalaxy> sim(BCs_cc);
	sim.setInitialConditions();
	sim.evolve();

	return 0;
}