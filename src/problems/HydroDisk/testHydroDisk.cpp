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
	constexpr double refine_Rcyl_kpc = 8.0;
	constexpr double refine_Hcyl_pc  = 600.0;
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

template <> struct Particle_Traits<HDGalaxy> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
};

template <> struct Physics_Traits<HDGalaxy> : DefaultPhysicsTraits {
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

	amrex::Real sn_jeans_J;
	amrex::Real sn_momentum;
	amrex::Real sn_remnant_fraction;
	amrex::Real sn_ejecta_mass{};   // Mej, grams
};


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto surfaceDensityProfile(double R, double Sigma0) -> double
{
	const double x = R / Rd;
	return Sigma0 * std::exp(-x - beta_profile * std::exp(-alpha_profile * x));
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double diskDensityAnalytic(double R, double z,
                           double Sigma0,
                           double vc,
                           double cs)
{
    const double Sigma = surfaceDensityProfile(R, Sigma0);
    if (Sigma <= 0.0) {
        return 0.0;
    }

    // Isothermal disk scale height
    const double H = cs*cs / (M_PI * C::Gconst * Sigma);

    // Midplane density from exact normalization: Sigma = 2 rho0 H
    const double rho0 =
        (M_PI * C::Gconst * Sigma * Sigma) / (2.0 * cs*cs);

	// Disk self-gravity (sech^2 profile)
	const double sech = 1.0 / std::cosh(z / H);
	const double disk_factor = sech * sech;

    // Vertical confinement from flattened halo (exact ΔΦ)
    const double denom = R*R + Rc*Rc;
    const double halo_factor =
        pow(
            1.0 + (z*z) / (q_flatten*q_flatten * denom),
            -vc*vc / (2.0 * cs*cs)
        );

    return rho0 * disk_factor * halo_factor;
}

template <> void QuokkaSimulation<HDGalaxy>::preCalculateInitialConditions()
{
	amrex::Print() << "Pre-calculating HDGalaxy initial conditions...\n";
	amrex::ParmParse const pp("hd_galaxy");
    pp.get("Mc",     userData_.Mc);
    pp.get("Q_mean", userData_.Q_mean);
	pp.query("sn_jeans_J", userData_.sn_jeans_J);
	pp.query("sn_momentum", userData_.sn_momentum);	
	pp.query("sn_remnant_fraction", userData_.sn_remnant_fraction);
	double Mej_msun = 10.0; // default, matches Kim & Ostriker test value
	pp.query("sn_ejecta_mass_msun", Mej_msun);
	userData_.sn_ejecta_mass = Mej_msun * C::M_solar;


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
	amrex::Print() << "Setting HDGalaxy initial conditions on grid...\n";
	const double vc      = userData_.vc;
	const double Sigma0  = userData_.Sigma0;
	const double cs_disk = quokka::EOS_Traits<HDGalaxy>::cs_disk;
	const double cs_cgm  = quokka::EOS_Traits<HDGalaxy>::cs_cgm;
	const double rho_cgm = userData_.rho_cgm;
	constexpr double gamma = quokka::EOS_Traits<HDGalaxy>::gamma;

	const amrex::Box &indexRange                                = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx       = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc                        = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double y = prob_lo[1] + (j + 0.5) * dx[1];
		const double z = prob_lo[2] + (k + 0.5) * dx[2];
		const double R = std::sqrt(x * x + y * y);
		const double rho_disc_raw = diskDensityAnalytic(R, z, Sigma0, vc, cs_disk);
		const double Sigma_R = surfaceDensityProfile(R, Sigma0);

		// 5. Two-phase assignment (Disk vs CGM)

		const bool in_disk = (rho_disc_raw > rho_transition);

		const double rho =
			in_disk ? amrex::max(rho_disc_raw, rho_transition*1e-6)
					: rho_cgm;

		const double cs  = in_disk ? cs_disk : cs_cgm;


		// 6. Rotation velocity (Arora+25 Eq. A.4)

		double vrot = 0.0;
		if (R > 0.0) {
			vrot = vc * R / std::sqrt(R*R + Rc*Rc);
		}


		// 7. Velocity components
		double vx = 0.0;
		double vy = 0.0;
		if (in_disk && R > 0.0) {
			vx = -vrot * y / R;
			vy =  vrot * x / R;
		}

		// 8. Conserved variables
		const double pressure = rho * cs * cs;
		const double Eint     = pressure / (gamma - 1.0);
		const double Ekin     = 0.5 * rho * (vx * vx + vy * vy);
		
		state_cc(i, j, k, HydroSystem<HDGalaxy>::density_index)        = rho;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index)     = rho * vx;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index)     = rho * vy;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index)     = 0.0;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::energy_index)         = Ekin + Eint;
		state_cc(i, j, k, HydroSystem<HDGalaxy>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<HDGalaxy>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real /*time*/, amrex::Real dt_lev)
{
	amrex::Print() << "Adding HDGalaxy Strang split sources...\n";
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx      = geom[lev].CellSizeArray();
	const amrex::Real dt = dt_lev;
	const double vc     = userData_.vc;

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

			// 1. Flattened potential denominator (Arora+25 Eq. A.3)
			const double z_over_q = z / q_flatten;
			const double D = R2 + Rc*Rc + z_over_q * z_over_q;

			// 2. Radial Acceleration (g_R)
			// We apply the background potential (DM) and subtract the initial gas potential
			// because the Poisson solver is already providing the live gas gravity.
			double g_R = 0.0;
			if (R > 0.0) {
				g_R = -(vc*vc * R / D);  // pure DM halo only
			}

			// 3. Vertical Acceleration (g_z)
			// This provides the vertical "squeezing" from the DM halo.
			const double g_z = -vc*vc * z / (q_flatten * q_flatten * D);

			// 4. Project radial acceleration into Cartesian components
			double gx = 0.0;
			double gy = 0.0;
			if (R > 0.0) {
				gx = g_R * x / R;
				gy = g_R * y / R;
			}

			// 5. Update Momenta (Momentum Kick)
			const double x1mom_new = x1mom + dt * rho * gx;
			const double x2mom_new = x2mom + dt * rho * gy;
			const double x3mom_new = x3mom + dt * rho * g_z;

			state(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index) = x3mom_new;

			// 6. Update Total Energy (Conserve Internal Energy)
			// This prevents the "source term heating" that can happen with large gravity steps
			const double Ekin_old = 0.5 * (x1mom*x1mom + x2mom*x2mom + x3mom*x3mom) / rho;
			const double Eint = Egas - Ekin_old;
			const double Ekin_new = 0.5 * (x1mom_new*x1mom_new + x2mom_new*x2mom_new + x3mom_new*x3mom_new) / rho;
			
			state(i, j, k, HydroSystem<HDGalaxy>::energy_index) = Ekin_new + Eint;
		});
	}
}


template <> void QuokkaSimulation<HDGalaxy>::refineGrid(
	int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx      = geom[lev].CellSizeArray();
	const auto tag     = tags.arrays();

	amrex::ParmParse pp("mhd_galaxy");
	amrex::Real shrink_kpc = 1.0;
	amrex::Real shrink_pc  = 50.0;
	pp.query("refine_shrink_per_level_kpc", shrink_kpc);
	pp.query("refine_shrink_per_level_pc",  shrink_pc);

	const amrex::Real margin_R = static_cast<amrex::Real>(lev) * shrink_kpc * 1.0e3 * C::parsec;
	const amrex::Real margin_H = static_cast<amrex::Real>(lev) * shrink_pc  * C::parsec;

	const amrex::Real Rcyl_lev = amrex::max(static_cast<amrex::Real>(refine_Rcyl) - margin_R,
	                                         static_cast<amrex::Real>(0.3 * refine_Rcyl));
	const amrex::Real Hcyl_lev = amrex::max(static_cast<amrex::Real>(refine_Hcyl) - margin_H,
	                                         static_cast<amrex::Real>(0.3 * refine_Hcyl));

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real x0 = prob_lo[0] + i * dx[0];
		const amrex::Real y0 = prob_lo[1] + j * dx[1];
		const amrex::Real z0 = prob_lo[2] + k * dx[2];
		const amrex::Real x1 = x0 + dx[0];
		const amrex::Real y1 = y0 + dx[1];
		const amrex::Real z1 = z0 + dx[2];

		auto tagIfInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			if (std::sqrt(x*x + y*y) < Rcyl_lev && std::abs(z) < Hcyl_lev) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};
		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<HDGalaxy>::computeAfterTimestep()
{
	if (!(userData_.sn_jeans_J > 0.0)) {
		return;
	}

	constexpr double MSUN = C::M_solar;
	constexpr double KM_S = 1.0e5;
	constexpr double Esn = 1.0e51;          // erg, fixed SN energy (Sec 3.2.1)
	constexpr double mu_H = 1.4 * C::m_u;   // mass per H nucleus, standard He abundance
	constexpr int stencil_radius = 2;       // rK in units of dx (paper default is 3dx; adjust as needed)

	const double Mej = userData_.sn_ejecta_mass;              // ejecta mass, Eq 17
	const double sn_jeans_J = userData_.sn_jeans_J;
	const double sn_momentum_ref = userData_.sn_momentum;     // calibration coefficient for Eq 20



	for (int lev = 0; lev <= finest_level; ++lev) {
		auto &state = state_new_cc_[lev];

		// Fill ghost zones before reading neighbor data below. Hydro-only: cell-centered
		// state only, no face-centered fields to fill.
		const auto time = tNew_[lev];
		fillBoundaryConditions(state, state, lev, time, quokka::centering::cc, quokka::direction::na,
		                        InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);

		const auto dx        = geom[lev].CellSizeArray();
		const double dx_max  = amrex::max(dx[0], amrex::max(dx[1], dx[2]));
		const double vol     = dx[0] * dx[1] * dx[2];

		// Fine mask: covered cells -> 0, uncovered (real, finest-resolution) cells -> 1.
		amrex::iMultiFab mask(state.boxArray(), state.DistributionMap(), 1, 0);
		if (lev < finest_level) {
			mask = amrex::makeFineMask(state.boxArray(), state.DistributionMap(),
			                            state_new_cc_[lev + 1].boxArray(), refRatio(lev), 1, 0);
		} else {
			mask.setVal(1);
		}


		// delta components: 0=drho, 1=dpx, 2=dpy, 3=dpz, 4=dE.
		// Everything here is the UNLIMITED deposit (paper Sec 2.2, Steps 1-2); the
		// analytic limiter (Eq 21, Step 3) is applied once, after SumBoundary, in the
		// apply pass below -- exactly matching the paper's separation of concerns.
		amrex::MultiFab delta(state.boxArray(), state.DistributionMap(), 5, stencil_radius);
		delta.setVal(0.0);

		// --- Pass 1: deposit (unlimited) mass/momentum/energy contributions ---
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();
			auto const &s = state.const_array(mfi);
			auto d = delta.array(mfi);
			auto const &mask_arr = mask.const_array(mfi);

			const auto slo = state[mfi].box().smallEnd();
			const auto shi = state[mfi].box().bigEnd();
			const auto dlo = delta[mfi].box().smallEnd();
			const auto dhi = delta[mfi].box().bigEnd();

			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				// Skip cells covered by a finer level
				if (mask_arr(i, j, k) == 0) { return; }

				const double rho = s(i, j, k, HydroSystem<HDGalaxy>::density_index);

				// Hydro-only Jeans trigger: no magnetic pressure support, so cs_eff is
				// just the thermal sound speed (no plasma-beta correction term).
				const double cs = HydroSystem<HDGalaxy>::ComputeSoundSpeed(s, i, j, k);
				const double cs_eff_sq = cs * cs;
				const double rho_J = M_PI * cs_eff_sq / (C::Gconst * sn_jeans_J * sn_jeans_J * (dx_max * dx_max));
				if (rho <= rho_J) { return; } // Jeans trigger, unchanged from before

				const double px = s(i,j,k,HydroSystem<HDGalaxy>::x1Momentum_index);
				const double py = s(i,j,k,HydroSystem<HDGalaxy>::x2Momentum_index);
				const double pz = s(i,j,k,HydroSystem<HDGalaxy>::x3Momentum_index);
				const double vx_c = px / rho;
				const double vy_c = py / rho;
				const double vz_c = pz / rho;

				// --- Gather kernel sums: cell count, gas mass, momentum (for vCOM), Eq 17-18 ---
				int    N_kernel   = 0;
				double mass_sum   = 0.0; // Sum rho_ijk over kernel (incl. source cell)
				double momx_sum   = 0.0;
				double momy_sum   = 0.0;
				double momz_sum   = 0.0;
				double neighbor_mass_sum = 0.0; // excludes center; used for mass-weighted w_mom

				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rho_nb = s(ii, jj, kk, HydroSystem<HDGalaxy>::density_index);
							++N_kernel;
							mass_sum += rho_nb;
							momx_sum += s(ii, jj, kk, HydroSystem<HDGalaxy>::x1Momentum_index);
							momy_sum += s(ii, jj, kk, HydroSystem<HDGalaxy>::x2Momentum_index);
							momz_sum += s(ii, jj, kk, HydroSystem<HDGalaxy>::x3Momentum_index);
							if (di != 0 || dj != 0 || dk != 0) { neighbor_mass_sum += rho_nb; }
						}
					}
				}
				if (N_kernel == 0) { return; }
				const double Vsnr_local = static_cast<double>(N_kernel) * vol;

				// Eq 17-18: total swept-up + ejecta mass, and mass-weighted vCOM.
				// Simplification: ejecta velocity v_ej taken equal to the source cell's
				// own velocity (we do not track an actual star particle here).
				const double Msnr = mass_sum * vol + Mej;
				const double vCOMx = (momx_sum * vol + Mej * vx_c) / Msnr;
				const double vCOMy = (momy_sum * vol + Mej * vy_c) / Msnr;
				const double vCOMz = (momz_sum * vol + Mej * vz_c) / Msnr;

				// nH_amb, shell-formation mass (Sec 3.2.1). M_sf is rescaled by
				// (p_terminal_ref / p_terminal_canonical)^2 so that the kinetic energy
				// p_terminal^2/(2 M_sf) stays invariant under changes to the runtime
				// sn_momentum_ref parameter -- matches the real reference implementation
				// (SNFeedbackUtils::depositThermalKineticMomentumSNR)
				const double nH_amb = Msnr / (mu_H * Vsnr_local);
				constexpr double M_sf_canonical = 1679.0 * MSUN;   // at nH=1 cm^-3, calibrated to p_ref_canonical
				constexpr double p_ref_canonical = 2.8e5 * MSUN * KM_S; // Kim & Ostriker (2015) canonical terminal momentum
				const double p_ratio = (sn_momentum_ref * MSUN * KM_S) / p_ref_canonical;
				const double M_sf = M_sf_canonical * std::pow(amrex::max(nH_amb, 1.0e-8), -0.26) * p_ratio * p_ratio;
				const double R_M  = Msnr / M_sf;

				// Eq 20: terminal momentum (extensive, g*cm/s)
				const double p_terminal = sn_momentum_ref * MSUN * KM_S * std::pow(amrex::max(nH_amb, 1.0e-8), -0.17);

				// MC regime only (Eq 19, R_M > 1): full terminal momentum, no thermal-only
				// or Sedov-Taylor branching. R_M is retained purely as a diagnostic.
				const double p_radial_mag = p_terminal / vol;

				// Full weight omega_ijk = 1 applied to every included kernel cell; the
				// 1/N_kernel normalization is already folded into Vsnr_local = N_kernel*vol,
				// so summing this over all N_kernel cells gives exactly Mej (mass conserved).
				const double drho_ej_cell = Mej / Vsnr_local;
				const bool have_mass_weight = (neighbor_mass_sum > 0.0);

				// --- Pass A: raw radial kicks, accumulate net vector S for re-centering.
				// Even though p_radial is isotropic in the continuum limit (paper: "the
				// cross term cancels when summing over all cells in the stencil"), a
				// discrete/domain-clipped stencil is not exactly isotropic, so we still
				// need the same re-centering trick as before to guarantee zero net thrust. ---
				double Sx = 0.0;
				double Sy = 0.0;
				double Sz = 0.0;
				double Lraw = 0.0;
				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di == 0 && dj == 0 && dk == 0) { continue; }
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rx = di * dx[0];
							const double ry = dj * dx[1];
							const double rz = dk * dx[2];
							const double r = std::sqrt(rx*rx + ry*ry + rz*rz);
							if (r == 0.0) { continue; }
							const double rho_nb = s(ii, jj, kk, HydroSystem<HDGalaxy>::density_index);
							const double w_mom = have_mass_weight ? (rho_nb / neighbor_mass_sum)
							                                       : (1.0 / static_cast<double>(N_kernel - 1));
							const double ex = rx / r;
							const double ey = ry / r;
							const double ez = rz / r;
							const double dp = p_radial_mag * w_mom;
							Sx += dp * ex; Sy += dp * ey; Sz += dp * ez;
							Lraw += dp;
						}
					}
				}

				double Lprime = 0.0;
				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di == 0 && dj == 0 && dk == 0) { continue; }
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rx = di * dx[0];
							const double ry = dj * dx[1];
							const double rz = dk * dx[2];
							const double r = std::sqrt(rx*rx + ry*ry + rz*rz);
							if (r == 0.0) { continue; }
							const double rho_nb = s(ii, jj, kk, HydroSystem<HDGalaxy>::density_index);
							const double w_mom = have_mass_weight ? (rho_nb / neighbor_mass_sum)
							                                       : (1.0 / static_cast<double>(N_kernel - 1));
							const double ex = rx / r;
							const double ey = ry / r;
							const double ez = rz / r;
							const double dp = p_radial_mag * w_mom;
							const double cx = dp * ex - Sx * w_mom;
							const double cy = dp * ey - Sy * w_mom;
							const double cz = dp * ez - Sz * w_mom;
							Lprime += std::sqrt(cx*cx + cy*cy + cz*cz);
						}
					}
				}
				const double rescale = (Lprime > 0.0) ? (Lraw / Lprime) : 0.0;

				// --- Pass B: deposit Eq 17 into every kernel cell (mass, base momentum
				// from velocity-smoothing, corrected radial kick, and energy including
				// the vCOM . p_radial cross term). All UNLIMITED -- limiter applied later. ---
				for (int di = -stencil_radius; di <= stencil_radius; ++di) {
					for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
						for (int dk = -stencil_radius; dk <= stencil_radius; ++dk) {
							if (di*di + dj*dj + dk*dk > stencil_radius * stencil_radius) { continue; }
							const int ii = i + di;
							const int jj = j + dj;
							const int kk = k + dk;
							if (ii < slo[0] || ii > shi[0] || jj < slo[1] || jj > shi[1] ||
								kk < slo[2] || kk > shi[2]) { continue; }
							if (ii < dlo[0] || ii > dhi[0] || jj < dlo[1] || jj > dhi[1] ||
								kk < dlo[2] || kk > dhi[2]) { continue; }

							const double rho_ijk = s(ii, jj, kk, HydroSystem<HDGalaxy>::density_index);
							const double px_ijk  = s(ii, jj, kk, HydroSystem<HDGalaxy>::x1Momentum_index);
							const double py_ijk  = s(ii, jj, kk, HydroSystem<HDGalaxy>::x2Momentum_index);
							const double pz_ijk  = s(ii, jj, kk, HydroSystem<HDGalaxy>::x3Momentum_index);
							const double rho_new_ijk = rho_ijk + drho_ej_cell;

							// Eq 17, momentum row: rho_new*vCOM - rho_old*v_old, plus the
							// (re-centered) radial kick for non-center cells.
							double pradx = 0.0;
							double prady = 0.0;
							double pradz = 0.0;
							if (di != 0 || dj != 0 || dk != 0) {
								const double rx = di * dx[0];
								const double ry = dj * dx[1];
								const double rz = dk * dx[2];
								const double r = std::sqrt(rx*rx + ry*ry + rz*rz);
								const double rho_nb = rho_ijk;
								const double w_mom = have_mass_weight ? (rho_nb / neighbor_mass_sum)
								                                       : (1.0 / static_cast<double>(N_kernel - 1));
								const double ex = rx / r;
								const double ey = ry / r;
								const double ez = rz / r;
								const double dp = p_radial_mag * w_mom;
								pradx = (dp * ex - Sx * w_mom) * rescale;
								prady = (dp * ey - Sy * w_mom) * rescale;
								pradz = (dp * ez - Sz * w_mom) * rescale;
							}

							const double dpx = rho_new_ijk * vCOMx - px_ijk + pradx;
							const double dpy = rho_new_ijk * vCOMy - py_ijk + prady;
							const double dpz = rho_new_ijk * vCOMz - pz_ijk + pradz;

							// Eq 17, energy row, with omega_ijk = 1 (see mass-deposit comment
							// above): (Esn+Ekin_ej)/Vsnr_local + vCOM . p_radial. Summed over
							// all N_kernel cells this adds exactly (Esn+Ekin_ej) total, plus
							// the vCOM.p_radial cross term which nets to ~0 by construction.
							// Ekin_ej (ejecta kinetic energy) is set to 0 here: we do not track
							// a separate ejecta launch velocity distinct from the source cell.
							constexpr double Ekin_ej = 0.0;
							const double dE_ijk = (Esn + Ekin_ej) / Vsnr_local
								+ (vCOMx * pradx + vCOMy * prady + vCOMz * pradz);

							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 0), drho_ej_cell);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 1), dpx);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 2), dpy);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 3), dpz);
							amrex::Gpu::Atomic::Add(&d(ii, jj, kk, 4), dE_ijk);
						}
					}
				}
			});
		}
		amrex::Gpu::streamSynchronize();
		delta.SumBoundary(geom[lev].periodicity()); // Step 2: inter-rank buffer summation

		// --- Pass 2 / Step 3-5: apply the analytic limiter (Eq 21) once per cell using
		// the fully-summed delta, then commit to state. Hydro-only: no Emag term, so
		// Etot = Ekin + Eint exactly. ---
		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();
			auto s = state.array(mfi);
			auto const &d = delta.const_array(mfi);

			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const double drho = d(i, j, k, 0);
				const double dE    = d(i, j, k, 4);
				if (drho == 0.0 && dE == 0.0) { return; }

				const double rho_old  = s(i, j, k, HydroSystem<HDGalaxy>::density_index);
				const double px_old   = s(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index);
				const double py_old   = s(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index);
				const double pz_old   = s(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index);
				const double Eint_old = s(i, j, k, HydroSystem<HDGalaxy>::internalEnergy_index);

				const double rho_new = rho_old + drho;

				// Internal-energy floor: hold SPECIFIC internal energy fixed and scale by
				// the new density, i.e. the added ejecta mass carries its proportional
				// share of internal energy too. Matches the real reference implementation
				// (SNFeedbackUtils::addCompositeBufferToState's e_int_new_tmp), rather than
				// holding the absolute Eint_old fixed as we had been doing, which implicitly
				// assumed the added mass carries zero extra internal energy.
				const double Eint_floor = (Eint_old / rho_old) * rho_new;

				const double dpx = d(i, j, k, 1);
				const double dpy = d(i, j, k, 2);
				const double dpz = d(i, j, k, 3);

				// Eq 21 limiter: find largest lambda in [0,1] s.t.
				//   Eint_floor + |p_old + lambda*dp|^2 / (2 rho_new) <= Etot_old + dE
				// Mass and energy are always added in full (matches paper: Delta e is fixed);
				// only the momentum *vector* is rescaled. Hydro-only: Etot_old is read
				// directly from the energy slot (no Emag to subtract out first).
				const double Etot_old = s(i, j, k, HydroSystem<HDGalaxy>::energy_index);
				const double a = dpx*dpx + dpy*dpy + dpz*dpz;
				const double b = 2.0 * (px_old*dpx + py_old*dpy + pz_old*dpz);
				const double c = (px_old*px_old + py_old*py_old + pz_old*pz_old)
					- 2.0 * rho_new * (Etot_old + dE - Eint_floor);

				double lambda = 1.0;
				if (a > 0.0) {
					// Feasible at lambda=1 iff a+b+c <= 0; else solve a*l^2+b*l+c=0 for
					// the largest root in [0,1].
					if (a + b + c > 0.0) {
						const double disc = b*b - 4.0*a*c;
						if (disc <= 0.0) {
							lambda = 0.0; // no physically valid nonzero kick this step
						} else {
							const double sq = std::sqrt(disc);
							const double l1 = (-b + sq) / (2.0*a);
							const double l2 = (-b - sq) / (2.0*a);
							const double lmax = amrex::max(l1, l2);
							lambda = amrex::max(0.0, amrex::min(1.0, lmax));
						}
					}
				}

				const double px_new = px_old + lambda * dpx;
				const double py_new = py_old + lambda * dpy;
				const double pz_new = pz_old + lambda * dpz;

				const double Etot_new = Etot_old + dE; // full energy always added (unscaled)
				const double Ekin_new = 0.5 * (px_new*px_new + py_new*py_new + pz_new*pz_new) / rho_new;
				const double Eint_new = Etot_new - Ekin_new; // absorbs any un-kicked energy as heat (no Emag term)

				s(i, j, k, HydroSystem<HDGalaxy>::density_index)        = rho_new;
				s(i, j, k, HydroSystem<HDGalaxy>::x1Momentum_index)     = px_new;
				s(i, j, k, HydroSystem<HDGalaxy>::x2Momentum_index)     = py_new;
				s(i, j, k, HydroSystem<HDGalaxy>::x3Momentum_index)     = pz_new;
				s(i, j, k, HydroSystem<HDGalaxy>::internalEnergy_index) = Eint_new;
				s(i, j, k, HydroSystem<HDGalaxy>::energy_index)         = Etot_new;
			});
		}
		amrex::Gpu::streamSynchronize();
	}
	AverageDown(); 
}

template <>
void QuokkaSimulation<HDGalaxy>::ComputeDerivedVar(
    int lev, std::string const &dname, amrex::MultiFab &mf,
    const int ncomp_cc_in,
    amrex::MultiFab const &state_cc,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
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

    using FaceStateArray = std::array<amrex::Array4<const Real>, AMREX_SPACEDIM>;

    const amrex::Real mean_density = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            return state(i, j, k, HydroSystem<HDGalaxy>::density_index);
        });
    stats["mean_density"] = mean_density / geom[0].ProbSize();

    const amrex::Real disk_mass = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
            return (rho > rho_transition) ? rho : static_cast<amrex::Real>(0.0);
        });

    const amrex::Real disk_volume = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
            return (rho > rho_transition) ? static_cast<amrex::Real>(1.0) : static_cast<amrex::Real>(0.0);
        });

    const amrex::Real mean_disk_density =
        (disk_volume > 0.0) ? (disk_mass / disk_volume) : static_cast<amrex::Real>(1.0);

    stats["mean_disk_density"] = mean_disk_density;
    stats["disk_mass"]         = disk_mass / C::M_solar;

    const amrex::Real sigma_vol = computeVolumeIntegral(
        [=] AMREX_GPU_DEVICE(int i, int j, int k,
                              amrex::Array4<const amrex::Real> const &state, FaceStateArray const & /*state_fc*/) noexcept {
            const amrex::Real rho = state(i, j, k, HydroSystem<HDGalaxy>::density_index);
            if (rho <= rho_transition) { return static_cast<amrex::Real>(0.0); }
            const amrex::Real eta = std::log(rho / mean_disk_density);
            return eta * eta;
        });

    stats["sigma_eta"] = (disk_volume > 0.0)
                             ? std::sqrt(sigma_vol / disk_volume)
                             : static_cast<amrex::Real>(0.0);
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
