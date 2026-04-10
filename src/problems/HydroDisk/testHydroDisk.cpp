//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroDisk.cpp
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
	static constexpr double gamma = 1.0001;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double T_cgm =  1.0e7;// K, already defined in anonymous namespace
	static constexpr double cs_cgm = gcem::sqrt(gamma * C::k_B * T_cgm / mean_molecular_weight);
	static constexpr double cs_disk = 7.0e5; // disk sound speed [cm/s]
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
	amrex::Real rho_cgm{};
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

	constexpr double eps = 1e-12;
	if (std::abs(y) < eps) {
		return 0.0;
	}

	// I0 via Abramowitz & Stegun 9.8.1 / 9.8.2
	auto bessel_I0 = [](double x) -> double {
		x = std::abs(x);
		if (x <= 3.75) {
			const double t_sq = (x / 3.75) * (x / 3.75);
			return 1.0 + t_sq*(3.5156229 + t_sq*(3.0899424 + t_sq*(1.2067492
			     + t_sq*(0.2659732 + t_sq*(0.0360768 + t_sq*0.0045813)))));
		}
		const double t_inv = 3.75 / x;
		return (std::exp(x) / std::sqrt(x)) *
		       (0.39894228 + t_inv*(0.01328592 
			 + t_inv*(0.00225319 + t_inv*(-0.00157565
		     + t_inv*(0.00916281 + t_inv*(-0.02057706 
			 + t_inv*(0.02635537 + t_inv*(-0.01647633 
			 					 + t_inv*0.00392377))))))));
	};

	// I1 via A&S 9.8.3 / 9.8.4
	auto bessel_I1 = [](double x) -> double {
		const double sign = (x >= 0.0) ? 1.0 : -1.0;
		x = std::abs(x);
		if (x <= 3.75) {
			const double t_sq = (x / 3.75) * (x / 3.75);
			return sign * x *
			    (0.5 + t_sq*( 0.87890594 + t_sq*( 0.51498869 
			  		 + t_sq*( 0.15084934 + t_sq*( 0.02658733 
			 		 + t_sq*( 0.00301532 + t_sq*( 0.00032411)))))));
		}
		const double t_inv = 3.75 / x;
		return sign * (std::exp(x) / std::sqrt(x)) *
		       (0.39894228 + t_inv*(-0.03988024 + t_inv*(-0.00362018 + t_inv*( 0.00163801
		    	 + t_inv*(-0.01031555 + t_inv*( 0.02282967 + t_inv*(-0.02895312
		    	 + t_inv*( 0.01787654 - t_inv*( 0.00420059)))))))));
	};

	// K0 via A&S 9.8.5 / 9.8.6
	auto bessel_K0 = [&bessel_I0](double x) -> double {
		if (x <= 2.0) {
			const double t_sq = (x / 2.0) * (x / 2.0);
			return -std::log(x / 2.0) * bessel_I0(x) + (-0.57721566 
				 + t_sq*(0.42278420 + t_sq*(0.23069756 
				 + t_sq*(0.03488590 + t_sq*(0.00262698 
				 + t_sq*(0.00010750 + t_sq*(0.00000740)))))));
		}
		const double t_inv = 2.0 / x;
		return (std::exp(-x) / std::sqrt(x)) *
		       (1.25331414 + t_inv*(-0.07832358 
			  + t_inv*(0.02189568 + t_inv*(-0.01062446
		      + t_inv*(0.00587872 + t_inv*(-0.00251540 
								  + t_inv*( 0.00053208)))))));
	};

	// K1 via A&S 9.8.7 / 9.8.8
	auto bessel_K1 = [&bessel_I1](double x) -> double {
		if (x <= 2.0) {
			const double t_sq = (x / 2.0) * (x / 2.0);
			return (1.0 / x) * 
				(x * std::log(x / 2.0) * bessel_I1(x) +  (1.0 + t_sq*(0.15443144 
				 + t_sq*(-0.67278579 + t_sq*(-0.18156897
			     + t_sq*(-0.01919402 + t_sq*(-0.00110404 
									 + t_sq*(-0.00004686))))))));
		}
		const double t_inv = 2.0 / x;
		return (std::exp(-x) / std::sqrt(x)) *
		       		(1.25331414 + t_inv*(0.23498619 
				+ t_inv*(-0.03655620 + t_inv*( 0.01504268
		  	    + t_inv*(-0.00780353 + t_inv*( 0.00325614 
									 + t_inv*(-0.00068245)))))));
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
    pp.get("Mc",     userData_.Mc);
    pp.get("Q_mean", userData_.Q_mean);

    constexpr double cs_disk = quokka::EOS_Traits<HDGalaxy>::cs_disk;
    constexpr double cs_cgm  = quokka::EOS_Traits<HDGalaxy>::cs_cgm;

    userData_.vc = userData_.Mc * cs_disk;
    const double vc = userData_.vc;

    // Toomre Q integral uses cs_disk
    auto integrand = [=](double R) -> double {
        const double D = R * R + Rc * Rc;
		const double sqrtD = std::sqrt(D);
		const double Omega    = vc / sqrtD;
		const double dOmegadR = -vc * R / (D * sqrtD);
        const double kappa    = std::sqrt(std::max(4.0 * Omega * Omega + 2.0 * R * Omega * dOmegadR, 0.0));
        const double fR       = surfaceDensityProfile(R, 1.0);
        return kappa * cs_disk / (M_PI * C::Gconst * fR);
    };

	constexpr int N = 1000;
	static_assert(N % 2 == 0, "Simpson's rule requires even N");
	const double a = 0.0;
	const double b = Rmax;
	const double h = (b - a) / N;

	double integral = integrand(a) + integrand(b);
	for (int i = 1; i < N; ++i) {
		const double R = a + i * h;
		integral += (i % 2 == 0 ? 2.0 : 4.0) * integrand(R);
	}
	integral *= h / 3.0;

	userData_.Sigma0 = integral / (userData_.Q_mean * Rmax);

    // Pressure matching: P = cs_disk^2 * rho_transition = cs_cgm^2 * rho_cgm
    userData_.rho_cgm = rho_transition * (cs_disk * cs_disk) / (cs_cgm * cs_cgm);

    amrex::Print() << "HDGalaxy:"
                   << " Q_mean = "    << userData_.Q_mean
                   << ", Mc = "       << userData_.Mc
                   << ", vc = "       << vc / 1.0e5        << " km/s"
                   << ", cs_disk = "  << cs_disk / 1.0e5   << " km/s"
                   << ", cs_cgm = "   << cs_cgm  / 1.0e5   << " km/s"
                   << ", Sigma0 = "   << userData_.Sigma0   << " g/cm^2"
                   << ", rho_cgm = "  << userData_.rho_cgm  << " g/cm^3\n";
}

template <> void QuokkaSimulation<HDGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::Print() << "Setting initial conditions for HDGalaxy problem on grid with index range " << grid_elem.indexRange_ << "\n";
	
	const double vc      = userData_.vc;
	const double Sigma0  = userData_.Sigma0;
	const double cs_disk = quokka::EOS_Traits<HDGalaxy>::cs_disk;
	const double cs_cgm  = quokka::EOS_Traits<HDGalaxy>::cs_cgm;
	const double rho_cgm = userData_.rho_cgm;
	constexpr double gamma = quokka::EOS_Traits<HDGalaxy>::gamma;

	const amrex::Box &indexRange                                = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx      = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc                       = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double y = prob_lo[1] + (j + 0.5) * dx[1];
		const double z = prob_lo[2] + (k + 0.5) * dx[2];
		const double R = std::sqrt(x * x + y * y);

		// Disk vertical structure
		const double Sigma_R  = surfaceDensityProfile(R, Sigma0);
		const double h_disk   = cs_disk * cs_disk / (M_PI * C::Gconst * Sigma_R);
		const double rho_mid  = Sigma_R / (std::sqrt(2.0 * M_PI) * h_disk);
		const double rho_disc = rho_mid * std::exp(-0.5 * z * z / (h_disk * h_disk));

		// Two-phase assignment
		const bool in_disk = (rho_disc > rho_transition);
		const double rho   = std::max(in_disk ? rho_disc : rho_cgm, rho_cgm);
		const double cs    = in_disk ? cs_disk : cs_cgm;

		// Rotation velocity from radial force balance (Arora+25 Eq. A.4)
		// vrot^2/R = -d(phi_dm)/dR - (cs^2/rho) * d(rho)/dR
		// Using exact logarithmic derivative of the surface density profile
		const double z_over_q = z / q_flatten;
		const double D = R*R + Rc*Rc + z_over_q * z_over_q;

		double vrot = 0.0;
		if (in_disk) {
			const double dlnSigma_dR = (1.0 / Rd) * (-1.0 + alpha_profile * beta_profile * std::exp(-alpha_profile * R / Rd));
			const double vrot_sq = std::max(
				vc*vc * R*R / D                        // DM halo: -d(phi_dm)/dR * R
				+ cs_disk*cs_disk * R * dlnSigma_dR    // pressure gradient correction
				+ R * dPhiGas_dR(R, Sigma0)            // gas self-gravity via Bessel functions
				, 0.0);
			vrot = std::sqrt(vrot_sq);
		}

		// Velocity components — CGM is at rest
		double vx = 0.0;
		double vy = 0.0;
		const double vz = 0.0;
		if (in_disk && R > 0.0) {
			vx = -vrot * y / R;
			vy =  vrot * x / R;
		}

		const double pressure = cs * cs * rho;
		const double momx     = rho * vx;
		const double momy     = rho * vy;
		const double momz     = rho * vz;
		const double Ekin     = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
		const double Eint     = pressure / (gamma - 1.0);
		const double Etot     = Ekin + Eint;

		state_cc(i, j, k, HydroSystem<HDGalaxy>::density_index)        = rho;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index)     = momx;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index)     = momy;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index)     = momz;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::energy_index)         = Etot;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<HDGalaxy>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real /*time*/, amrex::Real dt_lev)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx     = geom[lev].CellSizeArray();
	const amrex::Real dt = dt_lev;

	const double vc      = userData_.vc;
	constexpr double cs_disk = quokka::EOS_Traits<HDGalaxy>::cs_disk;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const double x  = prob_lo[0] + (i + 0.5) * dx[0];
			const double y  = prob_lo[1] + (j + 0.5) * dx[1];
			const double z  = prob_lo[2] + (k + 0.5) * dx[2];
			const double R2 = x*x + y*y;
			const double R  = std::sqrt(R2);

			const double rho   = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
			const double x1mom = state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index);
			const double x2mom = state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index);
			const double x3mom = state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index);
			const double Egas  = state(i, j, k, HydroSystem<HDGalaxy>::energy_index);

			// Flattened denominator D = R^2 + Rc^2 + (z/q)^2  (Arora+25 Eq. A.3)
			const double z_over_q = z / q_flatten;
			const double D = R2 + Rc*Rc + z_over_q * z_over_q;

			const bool in_disk = (rho > rho_transition);
			double g_R = 0.0;
			if (R > 0.0) {
				g_R = -vc*vc * R / D;  // dark matter halo
				if (in_disk) {
					const double dlnSigma_dR = (1.0/Rd) * (-1.0 + alpha_profile * beta_profile * std::exp(-alpha_profile * R / Rd));
					g_R += cs_disk * cs_disk * dlnSigma_dR;
				}
			}

			// Vertical acceleration from dark matter halo (Arora+25 Eq. A.3)
			const double g_z = -vc*vc * z / (q_flatten * q_flatten * D);

			// Project radial acceleration into Cartesian components
			double gx = 0.0;
			double gy = 0.0;
			if (R > 0.0) {
				gx = g_R * x / R;
				gy = g_R * y / R;
			}

			// Conserve internal energy across the momentum kick
			const double Ekin = 0.5 * (x1mom*x1mom + x2mom*x2mom + x3mom*x3mom) / rho;
			const double Eint = Egas - Ekin;

			const double x1mom_new = x1mom + dt * rho * gx;
			const double x2mom_new = x2mom + dt * rho * gy;
			const double x3mom_new = x3mom + dt * rho * g_z;

			state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index) = x3mom_new;

			// Update total energy keeping internal energy fixed
			const double Ekin_new = 0.5 * (x1mom_new*x1mom_new + x2mom_new*x2mom_new + x3mom_new*x3mom_new) / rho;
			state(i, j, k, HydroSystem<HDGalaxy>::energy_index) = Ekin_new + Eint;
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
	constexpr double cs_disk = quokka::EOS_Traits<HDGalaxy>::cs_disk;
	constexpr double cs_cgm  = quokka::EOS_Traits<HDGalaxy>::cs_cgm;
	const int ncomp          = ncomp_cc_in;
	const auto prob_lo       = geom[lev].ProbLoArray();
	const auto dx            = geom[lev].CellSizeArray();

	if (dname == "gpot") {
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k);  //cm^2/s^2
		});
		amrex::Gpu::streamSynchronize();
	}

	if (dname == "pressure") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
				const double cs  = (rho > rho_transition) ? cs_disk : cs_cgm;
				output(i, j, k, ncomp) = rho * cs * cs;  //dyne/cm^2
			});
		}
	}

	if (dname == "radius_sph") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double x    = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
				const double y    = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
				const double z    = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
				const double r_cm = std::sqrt(x * x + y * y + z * z);
				output(i, j, k, ncomp) = r_cm / C::parsec / 1.0e3;  // kpc
			});
		}
	}

	if (dname == "radial_velocity") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho  = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
				const double vx   = state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index) / rho;
				const double vy   = state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index) / rho;
				const double vz   = state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index) / rho;
				const double x    = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
				const double y    = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
				const double z    = prob_lo[2] + (static_cast<double>(k) + 0.5) * dx[2];
				const double r_cm = std::sqrt(x * x + y * y + z * z);
				output(i, j, k, ncomp) = (r_cm > 0.0) ? ((x * vx + y * vy + z * vz) / r_cm) / 1.0e5 : 0.0;  // km/s
			});
		}
	}

	if (dname == "circular_velocity") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho   = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
				const double vx    = state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index) / rho;
				const double vy    = state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index) / rho;
				const double x     = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
				const double y     = prob_lo[1] + (static_cast<double>(j) + 0.5) * dx[1];
				const double r_cyl = std::sqrt(x * x + y * y);
				output(i, j, k, ncomp) = (r_cyl > 0.0) ? ((x * vy - y * vx) / r_cyl) / 1.0e5 : 0.0; // km/s
			});
		}
	}

	if (dname == "mach") {
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state  = state_new_cc_[lev].const_array(iter);
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double rho  = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
				const double momx = state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index);
				const double momy = state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index);
				const double momz = state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index);
				const double cs   = (rho > rho_transition) ? cs_disk : cs_cgm;
				const double v2   = (momx * momx + momy * momy + momz * momz) / (rho * rho);
				output(i, j, k, ncomp) = std::sqrt(v2) / cs;  // Mach number (unitless)
			});
		}
	}
}

template <> auto QuokkaSimulation<HDGalaxy>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	// Volume-averaged mean density over whole box
	const amrex::Real mean_density = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		return state(i, j, k, HydroSystem<HDGalaxy>::density_index);
	});
	stats["mean_density"] = mean_density / geom[0].ProbSize();  // g/cm³;

	// Disk mass (rho integrated over disk cells)
	const amrex::Real disk_mass = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
		return (rho > rho_transition) ? rho : amrex::Real(0.0); //M☉
	});
	stats["disk_mass"] = disk_mass / C::M_solar;

	// Disk volume (cm^3) — needed to convert rho integral to mean density
	const amrex::Real disk_volume = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
		return (rho > rho_transition) ? amrex::Real(1.0) : amrex::Real(0.0);  //M☉
	});

	// Mass-weighted mean disk density: <rho> = disk_mass / disk_volume
	// This is a host-side scalar — safe to capture by value into the next kernel
	const amrex::Real mean_disk_density = (disk_volume > 0.0) ? (disk_mass / disk_volume) : amrex::Real(1.0);  // g/cm³
	stats["mean_disk_density"] = mean_disk_density; // g/cm³;
	stats["disk_mass"] = disk_mass / C::M_solar;  // convert to solar masses after

	// Volume-weighted log-density variance over disk cells:
	// sigma_eta^2 = (1/V_disk) * int_{disk} [ln(rho/<rho>)]^2 dV
	const amrex::Real sigma_eta_sq_times_vol = computeVolumeIntegral(
		[=] AMREX_GPU_DEVICE(int i, int j, int k,
		amrex::Array4<const amrex::Real> const &state) noexcept {
		const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
		if (rho <= rho_transition) { return amrex::Real(0.0); }
		const amrex::Real eta = std::log(rho / mean_disk_density);
		return eta * eta;
	});

	stats["sigma_eta"] = (disk_volume > 0.0) ? std::sqrt(sigma_eta_sq_times_vol / disk_volume) : amrex::Real(0.0);

	return stats;
}

auto problem_main() -> int
{
    auto BCs_cc = quokka::BC<HDGalaxy>(quokka::BCType::foextrap);
    QuokkaSimulation<HDGalaxy> sim(BCs_cc);
    sim.preCalculateInitialConditions();
    sim.setInitialConditions();
    sim.evolve();

    return 0;
}