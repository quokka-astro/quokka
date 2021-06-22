//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_IntVect.H"
#include "AMReX_REAL.H"
#include "radiation_system.hpp"
#include "test_radiation_beam.hpp"
#include <tuple>

auto main(int argc, char **argv) -> int
{
	// Initialization (copied from ExaWind)

	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() { // NOLINT
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

		result = testproblem_radiation_beam();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct BeamProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double kelvin_to_eV = 8.617385e-5;
constexpr double kappa0 = 0.01;		 // cm^2 g^-1 (specific opacity)
constexpr double rho0 = 1.0;		 // g cm^-3 (matter density)
constexpr double T_hohlraum = 1000.; // K
constexpr double T_initial = 300.;	 // K
constexpr double c_v = (1.0e15 * 1.0e-6 * kelvin_to_eV); // erg g^-1 K^-1

constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;  // cm s^-1

template <> struct RadSystem_Traits<BeamProblem> {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = radiation_constant_cgs_;
	static constexpr double mean_molecular_mass = hydrogen_mass_cgs_;
	static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = 0.;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputeOpacity(const double rho,
								  const double /*Tgas*/) -> double
{
	return kappa0;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputeTgasFromEgas(const double rho,
								       const double Egas) -> double
{
	return Egas / (rho * c_v);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputeEgasFromTgas(const double rho,
								       const double Tgas) -> double
{
	return rho * c_v * Tgas;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputeEgasTempDerivative(const double rho,
									     const double /*Tgas*/)
    -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	return rho * c_v;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
RadiationSimulation<BeamProblem>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
    amrex::GeometryData const &geom, const Real /*time*/, const amrex::BCRec * /*bcr*/,
    int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	const amrex::Real *dx = geom.CellSize();
	const amrex::Real *prob_lo = geom.ProbLo();
	const amrex::Box &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::Real const y0 = 0.;
	amrex::Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];

	if (i < lo[0]) {
		// streaming boundary condition
		double E_inc = NAN;

		const double E_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::radEnergy_index);
		const double Fx_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::x1RadFlux_index);
		const double Fy_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::x2RadFlux_index);
		const double Fz_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::x3RadFlux_index);

		const double Egas = consVar(lo[0], j, k, RadSystem<BeamProblem>::gasEnergy_index);
		const double rho = consVar(lo[0], j, k, RadSystem<BeamProblem>::gasDensity_index);
		const double px = consVar(lo[0], j, k, RadSystem<BeamProblem>::x1GasMomentum_index);
		const double py = consVar(lo[0], j, k, RadSystem<BeamProblem>::x2GasMomentum_index);
		const double pz = consVar(lo[0], j, k, RadSystem<BeamProblem>::x3GasMomentum_index);

		double Fx_bdry = NAN;
		double Fy_bdry = NAN;
		double Fz_bdry = NAN;

		const double y_min = 0.125;
		const double y_max = 0.25;

		if ((y >= y_min) && (y <= y_max)) {
			E_inc = a_rad * std::pow(T_hohlraum, 4);
			Fx_bdry = (1.0/std::sqrt(2.0)) * c * E_inc;
			Fy_bdry = (1.0/std::sqrt(2.0)) * c * E_inc;
			Fz_bdry = 0.;
		} else {
			// reflecting/absorbing boundary
			E_inc = E_0;
			Fx_bdry = -Fx_0;
			Fy_bdry = Fy_0;
			Fz_bdry = Fz_0;
		}
		const amrex::Real Fnorm =
		    std::sqrt(Fx_bdry * Fx_bdry + Fy_bdry * Fy_bdry + Fz_bdry * Fz_bdry);
		AMREX_ASSERT((Fnorm / (c * E_inc)) < 1.0); // flux-limiting condition

		// x1 left side boundary
		consVar(i, j, k, RadSystem<BeamProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<BeamProblem>::x1RadFlux_index) = Fx_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x2RadFlux_index) = Fy_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x3RadFlux_index) = Fz_bdry;

		// extrapolated/outflow boundary for gas variables
		consVar(i, j, k, RadSystem<BeamProblem>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<BeamProblem>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<BeamProblem>::x1GasMomentum_index) = px;
		consVar(i, j, k, RadSystem<BeamProblem>::x2GasMomentum_index) = py;
		consVar(i, j, k, RadSystem<BeamProblem>::x3GasMomentum_index) = pz;
	}
}

template <> void RadiationSimulation<BeamProblem>::setInitialConditions()
{
	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const double Erad = a_rad * std::pow(T_initial, 4);
			const double rho = rho0;
			const double Egas =
			    RadSystem<BeamProblem>::ComputeEgasFromTgas(rho, T_initial);

			state(i, j, k, RadSystem<BeamProblem>::radEnergy_index) = Erad;
			state(i, j, k, RadSystem<BeamProblem>::x1RadFlux_index) = 0;
			state(i, j, k, RadSystem<BeamProblem>::x2RadFlux_index) = 0;
			state(i, j, k, RadSystem<BeamProblem>::x3RadFlux_index) = 0;

			state(i, j, k, RadSystem<BeamProblem>::gasEnergy_index) = Egas;
			state(i, j, k, RadSystem<BeamProblem>::gasDensity_index) = rho;
			state(i, j, k, RadSystem<BeamProblem>::x1GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<BeamProblem>::x2GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<BeamProblem>::x3GasMomentum_index) = 0.;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

auto testproblem_radiation_beam() -> int
{
	// Problem parameters
	const int max_timesteps = 10000;
	const double CFL_number = 0.2;
	const int nx = 128;
	const int ny = 128;
	const double Lx = 2.0;		 // cm
	const double Ly = 2.0;		 // cm
	const double max_time = 3.0 * (Lx/c); // s

	amrex::IntVect gridDims{AMREX_D_DECL(nx, ny, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.), amrex::Real(0.), amrex::Real(0.))},	 // NOLINT
	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Ly), amrex::Real(1.0))}}; // NOLINT

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<BeamProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<BeamProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<BeamProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<BeamProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<BeamProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<BeamProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	constexpr int nvars = 9;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);	 // left x1 -- Marshak
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

	// Problem initialization
	RadiationSimulation<BeamProblem> sim(gridDims, boxSize, boundaryConditions, nvars);
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.outputAtInterval_ = true;
	sim.plotfileInterval_ = 20; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
