//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_shadow.cpp
/// \brief Defines a 2D test problem for radiation in the transport regime.
///

#include "test_radiation_shadow.hpp"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_IntVect.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "radiation_system.hpp"
#include "test_radhydro_shock_cgs.hpp"
#include <limits>
#include <tuple>

struct ShadowProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double sigma0 = 0.1;	     // cm^-1 (opacity)
constexpr double rho_bg = 1.0e-3;    // g cm^-3 (matter density)
constexpr double rho_clump = 1.0;    // g cm^-3 (matter density)
constexpr double T_hohlraum = 1740.; // K
constexpr double T_initial = 290.;   // K

constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;  // cm s^-1

template <> struct RadSystem_Traits<ShadowProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = c;
	static constexpr double radiation_constant = a_rad;
	static constexpr double mean_molecular_mass = 1.0e5 * hydrogen_mass_cgs_;
	static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = 0.;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<ShadowProblem>::ComputeOpacity(const double rho,
								    const double /*Tgas*/) -> double
{
	const amrex::Real sigma = sigma0 * std::pow(rho / rho_bg, 2);
	//	    sigma0 * std::pow(Tgas / T_initial, -3.5) * std::pow(rho / rho_bg, 2);

	const amrex::Real kappa = sigma / rho; // specific opacity [cm^2 g^-1]
	return kappa;
}

#if 0
template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<ShadowProblem>::ComputeOpacityTempDerivative(const double rho,
										  const double Tgas)
    -> double
{
	const amrex::Real dsigma_dTgas = sigma0 * (-3.5) * std::pow(Tgas / T_initial, -4.5) *
					 std::pow(rho / rho_bg, 2) / T_initial;

	const amrex::Real dkappa_dTgas = dsigma_dTgas / rho; // specific opacity [cm^2 g^-1]
	return dkappa_dTgas;
}
#endif

#if 0
template <>
AMREX_GPU_DEVICE auto RadSystem<ShadowProblem>::ComputeEddingtonFactor(double /*f_in*/) -> double
{
	return 1.0; // streaming
	// return (1./3.); // Eddington approximation
}
#endif

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
RadhydroSimulation<ShadowProblem>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
    amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
    int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	// const amrex::Real *dx = geom.CellSize();
	// amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = geom.ProbLoArray();
	const amrex::Box &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();

	if (i < lo[0]) {
		// free-streaming boundary condition
		const double Egas = consVar(lo[0], j, k, RadSystem<ShadowProblem>::gasEnergy_index);
		const double rho = consVar(lo[0], j, k, RadSystem<ShadowProblem>::gasDensity_index);
		const double px =
		    consVar(lo[0], j, k, RadSystem<ShadowProblem>::x1GasMomentum_index);
		const double py =
		    consVar(lo[0], j, k, RadSystem<ShadowProblem>::x2GasMomentum_index);
		const double pz =
		    consVar(lo[0], j, k, RadSystem<ShadowProblem>::x3GasMomentum_index);

		const double E_inc = a_rad * std::pow(T_hohlraum, 4);
		const double Fx_bdry = c * E_inc; // free-streaming (F/cE == 1)
		const double Fy_bdry = 0.;
		const double Fz_bdry = 0.;

		const amrex::Real Fnorm =
		    std::sqrt(Fx_bdry * Fx_bdry + Fy_bdry * Fy_bdry + Fz_bdry * Fz_bdry);
		AMREX_ASSERT((Fnorm / (c * E_inc)) <= 1.0); // flux-limiting condition

		// x1 left side boundary (streaming)
		consVar(i, j, k, RadSystem<ShadowProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<ShadowProblem>::x1RadFlux_index) = Fx_bdry;
		consVar(i, j, k, RadSystem<ShadowProblem>::x2RadFlux_index) = Fy_bdry;
		consVar(i, j, k, RadSystem<ShadowProblem>::x3RadFlux_index) = Fz_bdry;

		// extrapolated/outflow boundary for gas variables
		consVar(i, j, k, RadSystem<ShadowProblem>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<ShadowProblem>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<ShadowProblem>::x1GasMomentum_index) = px;
		consVar(i, j, k, RadSystem<ShadowProblem>::x2GasMomentum_index) = py;
		consVar(i, j, k, RadSystem<ShadowProblem>::x3GasMomentum_index) = pz;
	}
}

template <> void RadhydroSimulation<ShadowProblem>::setInitialConditions()
{
	auto prob_lo = simGeometry_.ProbLoArray();
	auto dx = dx_;

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
			amrex::Real const xc = 0.5;
			amrex::Real const yc = 0.0;
			amrex::Real const x0 = 0.1;
			amrex::Real const y0 = 0.06;

			amrex::Real const Delta =
			    10. * (std::pow((x - xc) / x0, 2) + std::pow((y - yc) / y0, 2) - 1.0);
			amrex::Real const rho =
			    rho_bg + (rho_clump - rho_bg) / (1.0 + std::exp(Delta));

			amrex::Real const Erad = a_rad * std::pow(T_initial, 4);
			amrex::Real const Egas =
			    RadSystem<ShadowProblem>::ComputeEgasFromTgas(rho, T_initial);

			state(i, j, k, RadSystem<ShadowProblem>::radEnergy_index) = Erad;
			state(i, j, k, RadSystem<ShadowProblem>::x1RadFlux_index) = 0;
			state(i, j, k, RadSystem<ShadowProblem>::x2RadFlux_index) = 0;
			state(i, j, k, RadSystem<ShadowProblem>::x3RadFlux_index) = 0;

			state(i, j, k, RadSystem<ShadowProblem>::gasEnergy_index) = Egas;
			state(i, j, k, RadSystem<ShadowProblem>::gasDensity_index) = rho;
			state(i, j, k, RadSystem<ShadowProblem>::x1GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<ShadowProblem>::x2GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<ShadowProblem>::x3GasMomentum_index) = 0.;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

auto problem_main() -> int
{
	// N.B. The matter-energy exchange used to exceed numerical precision in IEEE double for
	// this problem.

	// Problem parameters
	constexpr int max_timesteps = 20000;
	constexpr double CFL_number = 0.4;
	constexpr int nx = 560; // 280;
	constexpr int ny = 160; // 80;

	constexpr double Lx = 1.0;	     // cm
	constexpr double Ly = 0.24;	     // cm
	constexpr double max_time = 1.0e-10; // 10.0 * (Lx / c); // s

	amrex::IntVect gridDims{AMREX_D_DECL(nx, ny, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.), amrex::Real(0.0))},       // NOLINT
	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Ly / 2.0), amrex::Real(1.0))}}; // NOLINT

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<ShadowProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	constexpr int nvars = 9;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);	 // left x1 -- streaming
		boundaryConditions[n].setHi(0, amrex::BCType::foextrap); // right x1 -- extrapolate
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) { // reflect lower
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
			} else {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
			}
			// extrapolate upper
			boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Print radiation epsilon ("stiffness parameter" from Su & Olson).
	// (if epsilon is smaller than machine epsilon, garbage results ensue.)

	const auto dt_CFL = CFL_number * std::min(Lx / nx, Ly / ny) / c;
	const auto c_v = RadSystem_Traits<ShadowProblem>::boltzmann_constant /
			 (RadSystem_Traits<ShadowProblem>::mean_molecular_mass *
			  (RadSystem_Traits<ShadowProblem>::gamma - 1.0));
	const auto epsilon =
	    4.0 * a_rad * (T_initial * T_initial * T_initial) * sigma0 * (c * dt_CFL) / c_v;
	const auto machine_epsilon = std::numeric_limits<amrex::Real>::epsilon();
	amrex::Print() << "radiation epsilon (stiffness parameter) = " << epsilon << "\n"
		       << "machine epsilon = " << machine_epsilon << "\n";

	// Problem initialization
	RadhydroSimulation<ShadowProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.outputAtInterval_ = true;
	sim.plotfileInterval_ = 50; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
