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
#include "radiation_system.hpp"
#include "test_radiation_tophat.hpp"
#include <tuple>

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
auto RadSystem<TophatProblem>::ComputeOpacity(const double rho, const double /*Tgas*/) -> double
{
	amrex::Real kappa = 0.;
	if (rho == rho_pipe) {
		kappa = kappa_pipe;
	} else if (rho == rho_wall) {
		kappa = kappa_wall;
	} else {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(true, "opacity not defined!");
	}
	return kappa;
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
auto RadSystem<TophatProblem>::ComputeEgasTempDerivative(const double rho, const double /*Tgas*/)
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
    amrex::GeometryData const &geom, const Real /*time*/, const amrex::BCRec *bcr, int /*bcomp*/,
    int /*orig_comp*/)
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

	const amrex::Real *dx = geom.CellSize();
	const amrex::Real *prob_lo = geom.ProbLo();
	//const amrex::Real *prob_hi = geom.ProbHi();
	const amrex::Box &box = geom.Domain();

	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	amrex::Real const y0 = 0.;
	amrex::Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];

	if (i < lo[0]) {
		// Marshak boundary condition
		double E_inc = NAN;

		const double E_0 = consVar(lo[0], j, k, RadSystem<TophatProblem>::radEnergy_index);
		const double Fx_0 = consVar(lo[0], j, k, RadSystem<TophatProblem>::x1RadFlux_index);
		//const double Fy_0 = consVar(lo[0], j, k, RadSystem<TophatProblem>::x2RadFlux_index);
		//const double Fz_0 = consVar(lo[0], j, k, RadSystem<TophatProblem>::x3RadFlux_index);

		double Fx_bdry = NAN;
		double Fy_bdry = NAN;
		double Fz_bdry = NAN;

		if (std::abs(y - y0) < 0.5) {
			E_inc = a_rad * std::pow(T_hohlraum, 4);
			Fx_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * Fx_0);
			Fy_bdry = 0.;
			Fz_bdry = 0.;
		} else {
			// reflecting boundary (usually works best)
			E_inc = E_0;
			Fx_bdry = -Fx_0;
			Fy_bdry = 0.;
			Fz_bdry = 0.;

			// extrapolated boundary
			//E_inc = E_0;
			//Fx_bdry = Fx_0;
			//Fy_bdry = 0.;
			//Fz_bdry = 0.;

			// Marshak boundary (does not work)
			//E_inc = a_rad * std::pow(T_initial, 4);
			//Fx_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * Fx_0);
			//Fy_bdry = 0.;
			//Fz_bdry = 0.;
		}
		const amrex::Real Fnorm = std::sqrt(Fx_bdry*Fx_bdry + Fy_bdry*Fy_bdry + Fz_bdry*Fz_bdry);
		AMREX_ASSERT((Fnorm / (c * E_inc)) < 1.0); // flux-limiting condition

		// x1 left side boundary (Marshak)
		consVar(i, j, k, RadSystem<TophatProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<TophatProblem>::x1RadFlux_index) = Fx_bdry;
		consVar(i, j, k, RadSystem<TophatProblem>::x2RadFlux_index) = Fy_bdry;
		consVar(i, j, k, RadSystem<TophatProblem>::x3RadFlux_index) = Fz_bdry;
	}
}

template <> void RadiationSimulation<TophatProblem>::setInitialConditions()
{
	auto prob_lo = simGeometry_.ProbLo();
	//auto prob_hi = simGeometry_.ProbHi();
	auto dx = dx_;

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const double Erad = a_rad * std::pow(T_initial, 4);
			double rho = rho_wall;

			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];

			bool inside_region1 =
			    ((((x > 0.) && (x <= 2.5)) || ((x > 4.5) && (x < 7.0))) &&
			     (std::abs(y) < 0.5));
			bool inside_region2 =
			    ((((x > 2.5) && (x < 3.0)) || ((x > 4.) && (x <= 4.5))) &&
			     (std::abs(y) < 1.5));
			bool inside_region3 = (((x > 3.0) && (x < 4.0)) &&
					       ((std::abs(y) > 1.0) && (std::abs(y) < 1.5)));

			if (inside_region1 || inside_region2 || inside_region3) {
				rho = rho_pipe;
			}

			const double Egas =
			    RadSystem<TophatProblem>::ComputeEgasFromTgas(rho, T_initial);

			state(i, j, k, RadSystem<TophatProblem>::radEnergy_index) = Erad;
			state(i, j, k, RadSystem<TophatProblem>::x1RadFlux_index) = 0;
			state(i, j, k, RadSystem<TophatProblem>::x2RadFlux_index) = 0;
			state(i, j, k, RadSystem<TophatProblem>::x3RadFlux_index) = 0;

			state(i, j, k, RadSystem<TophatProblem>::gasEnergy_index) = Egas;
			state(i, j, k, RadSystem<TophatProblem>::gasDensity_index) = rho;
			state(i, j, k, RadSystem<TophatProblem>::x1GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<TophatProblem>::x2GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<TophatProblem>::x3GasMomentum_index) = 0.;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

namespace quokka
{
template <>
AMREX_GPU_HOST_DEVICE bool
CheckSymmetryArray<TophatProblem>(amrex::Array4<const amrex::Real> const &arr,
				  amrex::Box const &indexRange, const int ncomp)
{
#if 0
	amrex::Long asymmetry = 0;
	amrex::GpuArray<int, 3> lo = indexRange.loVect3d();
	auto [nx, ny, nz] = indexRange.hiVect3d().arr;
	AMREX_ASSERT(lo[0] == 0);
	AMREX_ASSERT(lo[1] == 0);
	AMREX_ASSERT(lo[2] == 0);

	int j0 = ny / 2;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			for (int k = 0; k < nz; ++k) {
				for (int n = 0; n < ncomp; ++n) {
					const amrex::Real comp_upper = arr(i, j, k, n);
					int j_reflect = j0 - (j - j0 + 1);
					amrex::Real comp_lower = arr(i, j_reflect, k, n);

					if ((n == RadSystem<TophatProblem>::x2RadFlux_index) ||
					    (n == RadSystem<TophatProblem>::x2GasMomentum_index)) {
						comp_lower *= -1.0;
					}

					if (comp_upper != comp_lower) {
						amrex::Print()
						    << i << "," << j << "," << k << "," << n
						    << comp_upper << comp_lower << "\n";
						asymmetry++;
					}
				}
			}
		}
	}
	if (asymmetry == 0) {
		// amrex::Print() << "no symmetry violations.\n";
	}
	AMREX_ASSERT_WITH_MESSAGE(asymmetry == 0, "y-midplane symmetry check failed!");
#endif
	return true;
}

template <>
AMREX_GPU_HOST_DEVICE bool CheckSymmetry<TophatProblem>(amrex::FArrayBox const &arr,
							amrex::Box const &indexRange,
							const int ncomp)
{
	return CheckSymmetryArray<TophatProblem>(arr.const_array(), indexRange, ncomp);
}
} // namespace quokka

template <> void RadiationSimulation<TophatProblem>::computeAfterTimestep()
{
#if 0
	// copy all FABs to a local FAB across the entire domain
	amrex::BoxArray localBoxes(domain_);
	amrex::DistributionMapping localDistribution(localBoxes, 1);
	amrex::MultiFab state_mf(localBoxes, localDistribution, ncomp_, 0);
	state_mf.ParallelCopy(state_new_);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &state = state_mf.array(0);

		amrex::Long asymmetry = 0;
		auto nx = nx_;
		auto ny = ny_;
		auto nz = nz_;
		auto ncomp = ncomp_;
		int j0 = ny / 2;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				for (int k = 0; k < nz; ++k) {
					for (int n = 0; n < ncomp; ++n) {
						const amrex::Real comp_upper = state(i, j, k, n);
						int j_reflect = j0 - (j - j0 + 1);
						amrex::Real comp_lower = state(i, j_reflect, k, n);

						if ((n ==
						     RadSystem<TophatProblem>::x2RadFlux_index) ||
						    (n ==
						     RadSystem<
							 TophatProblem>::x2GasMomentum_index)) {
							comp_lower *= -1.0;
						}

						if (comp_upper != comp_lower) {
							// amrex::Print()
							//    << i << ", " << j << ", " << k << ", "
							//    << n << ", "
							//    << comp_upper << ", " << comp_lower <<
							//    "\n";
							asymmetry++;
							// AMREX_ASSERT(false);
						}
					}
				}
			}
		}
		// AMREX_ASSERT_WITH_MESSAGE(asymmetry == 0, "y-midplane symmetry check failed!");
	}
#endif
}

auto testproblem_radiation_marshak_cgs() -> int
{
	// Problem parameters
	const int max_timesteps = 10000;
	const double CFL_number = 0.1;
	const int nx = 140;
	const int ny = 40; // 80;

	const double Lx = 7.0;		 // cm
	const double Ly = 4.0;		 // cm
	const double max_time = 3.0e-10; // s

	amrex::IntVect gridDims{AMREX_D_DECL(nx, ny, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.), amrex::Real(0.0))},       // NOLINT
	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Ly / 2.0), amrex::Real(1.0))}}; // NOLINT

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<TophatProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TophatProblem>::x3GasMomentum_index) && (dim == 2)) {
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
				//boundaryConditions[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
				//boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);
			}
			// extrapolate upper
			boundaryConditions[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadiationSimulation<TophatProblem> sim(gridDims, boxSize, boundaryConditions, nvars);
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.outputAtInterval_ = true;
	sim.plotfileInterval_ = 1; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
