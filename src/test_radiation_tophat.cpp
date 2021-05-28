//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_Config.H"
#include "AMReX_IntVect.H"
#include "radiation_system.hpp"
#include "test_radiation_marshak_cgs.hpp"

auto main(int argc, char **argv) -> int
{
	// Initialization (copied from ExaWind)

	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {
		amrex::ParmParse pp("amrex");
		// Set the defaults so that we throw an exception instead of attempting
		// to generate backtrace files. However, if the user has explicitly set
		// these options in their input files respect those settings.
		if (!pp.contains("throw_exception")) {
			pp.add("throw_exception", 1);
		}
		if (!pp.contains("signal_handling")) {
			pp.add("signal_handling", 0);
		}
	});

	int result = 0;

	{ // objects must be destroyed before amrex::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_radiation_marshak_cgs();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct TophatProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// "Tophat" pipe flow test (Gentile 2001)
constexpr double kelvin_to_eV = 8.617385e-5;

constexpr double kappa_wall = 200.0;			 // cm^2 g^-1 (specific opacity)
constexpr double rho_wall = 10.0;			 // g cm^-3 (matter density)
constexpr double kappa_pipe = 20.0;			 // cm^2 g^-1 (specific opacity)
constexpr double rho_pipe = 0.01;			 // g cm^-3 (matter density)
constexpr double T_hohlraum = 500. / kelvin_to_eV;	 // K [== 500 eV]
constexpr double T_initial = 50. / kelvin_to_eV;	 // K [== 50 eV]
constexpr double c_v = (1.0e15 * 1.0e-6 * kelvin_to_eV); // erg g^-1 K^-1

constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;  // cm s^-1

template <> struct RadSystem_Traits<TophatProblem> {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = radiation_constant_cgs_;
	static constexpr double mean_molecular_mass = hydrogen_mass_cgs_;
	static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = 0.;
	static constexpr bool do_marshak_left_boundary = false;
	static constexpr double T_marshak_left = T_hohlraum;
};

template <>
auto RadSystem<TophatProblem>::ComputeOpacity(const double /*rho*/, const double /*Tgas*/) -> double
{
	// this should be position-dependent for this problem
	return kappa_wall;
}

template <>
auto RadSystem<TophatProblem>::ComputeTgasFromEgas(const double rho, const double Egas) -> double
{
	return Egas / (rho * c_v);
}

template <>
auto RadSystem<TophatProblem>::ComputeEgasFromTgas(const double rho, const double Tgas) -> double
{
	return rho * c_v * Tgas;
}

template <>
auto RadSystem<TophatProblem>::ComputeEgasTempDerivative(const double rho, const double Tgas)
    -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	return rho * c_v;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
RadiationSimulation<TophatProblem>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
    amrex::GeometryData const & geom, const Real /*time*/, const amrex::BCRec *bcr,
    int /*bcomp*/, int /*orig_comp*/)
{
	if (!((bcr->lo(0) == amrex::BCType::ext_dir) || (bcr->hi(0) == amrex::BCType::ext_dir))) {
		return;
	}

#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	const amrex::Real* dx = geom.CellSize();
	const amrex::Real* prob_lo = geom.ProbLo();
	const amrex::Real* prob_hi = geom.ProbHi();
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];

	if (i < 0) {
		// Marshak boundary condition
		double T_H = NAN;
		if (std::abs(y - y0) < 0.5) {
			T_H = T_hohlraum;
		} else {
			T_H = 0.0;
		}

		const double E_inc = a_rad * std::pow(T_H, 4);
		const double E_0 = consVar(0, j, k, RadSystem<TophatProblem>::radEnergy_index);
		const double F_0 = consVar(0, j, k, RadSystem<TophatProblem>::x1RadFlux_index);

		// const double E_1 = consVar(1, j, k,
		// RadSystem<TophatProblem>::radEnergy_index); const double F_1 = consVar(1, j,
		// k, RadSystem<TophatProblem>::x1RadFlux_index);

		// use PPM stencil at interface to solve for F_rad in the ghost zones
		// const double F_bdry = 0.5 * c * E_inc - (7. / 12.) * (c * E_0 + 2.0 * F_0) +
		//		      (1. / 12.) * (c * E_1 + 2.0 * F_1);

		// use value at interface to solve for F_rad in the ghost zones
		const double F_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * F_0);

		AMREX_ASSERT(std::abs(F_bdry / (c * E_inc)) < 1.0);

		// x1 left side boundary (Marshak)
		consVar(i, j, k, RadSystem<TophatProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<TophatProblem>::x1RadFlux_index) = F_bdry;
		consVar(i, j, k, RadSystem<TophatProblem>::x2RadFlux_index) = 0.;
		consVar(i, j, k, RadSystem<TophatProblem>::x3RadFlux_index) = 0.;
	} else {
		// right-side boundary -- constant
		const double Erad = a_rad * std::pow(T_initial, 4);

		consVar(i, j, k, RadSystem<TophatProblem>::radEnergy_index) = Erad;
		consVar(i, j, k, RadSystem<TophatProblem>::x1RadFlux_index) = 0;
		consVar(i, j, k, RadSystem<TophatProblem>::x2RadFlux_index) = 0;
		consVar(i, j, k, RadSystem<TophatProblem>::x3RadFlux_index) = 0;
	}
}

template <> void RadiationSimulation<TophatProblem>::setInitialConditions()
{
	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const double Egas =
			    RadSystem<TophatProblem>::ComputeEgasFromTgas(rho_wall, T_initial);
			const double Erad = a_rad * std::pow(T_initial, 4);

			state(i, j, k, RadSystem<TophatProblem>::radEnergy_index) = Erad;
			state(i, j, k, RadSystem<TophatProblem>::x1RadFlux_index) = 0;
			state(i, j, k, RadSystem<TophatProblem>::x2RadFlux_index) = 0;
			state(i, j, k, RadSystem<TophatProblem>::x3RadFlux_index) = 0;

			state(i, j, k, RadSystem<TophatProblem>::gasEnergy_index) = Egas;
			state(i, j, k, RadSystem<TophatProblem>::gasDensity_index) = rho_wall;
			state(i, j, k, RadSystem<TophatProblem>::x1GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<TophatProblem>::x2GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<TophatProblem>::x3GasMomentum_index) = 0.;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

auto testproblem_radiation_marshak_cgs() -> int
{
	// Problem parameters
	const int max_timesteps = 10000;
	const double CFL_number = 0.4;
	const int nx = 1400;
	const int ny = 800;

	const double Lx = 7.0;		 // cm
	const double Ly = 4.0;		 // cm
	const double max_time = 1.0e-10; // s

	amrex::IntVect gridDims{AMREX_D_DECL(nx, 4, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))}, // NOLINT
	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Ly), amrex::Real(1.0))}};  // NOLINT

	constexpr int nvars = 9;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);	 // left x1 -- Marshak
		boundaryConditions[n].setHi(0, amrex::BCType::foextrap); // right x1 -- extrapolate
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			boundaryConditions[n].setLo(
			    i, amrex::BCType::foextrap); // extrapolate all others
			boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadiationSimulation<TophatProblem> sim(gridDims, boxSize, boundaryConditions, nvars);
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.outputAtInterval_ = true;
	sim.plotfileInterval_ = 100; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
