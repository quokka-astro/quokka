//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_box.hpp"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_IntVect.H"
#include "radiation_system.hpp"
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

struct BoxProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double kelvin_to_eV = 8.617385e-5;

// "Tophat" pipe flow test (Gentile 2001)
constexpr double kappa_pipe = 20.0;			 // cm^2 g^-1 (specific opacity)
constexpr double rho_pipe = 0.01;			 // g cm^-3 (matter density)
constexpr double T_hohlraum = 500. / kelvin_to_eV;	 // K [== 500 eV]
constexpr double T_initial = 50. / kelvin_to_eV;	 // K [== 50 eV]
constexpr double c_v = (1.0e15 * 1.0e-6 * kelvin_to_eV); // erg g^-1 K^-1

constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;  // cm s^-1

template <> struct RadSystem_Traits<BoxProblem> {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = radiation_constant_cgs_;
	static constexpr double mean_molecular_mass = hydrogen_mass_cgs_;
	static constexpr double boltzmann_constant = boltzmann_constant_cgs_;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = 0.;
};

template <>
auto RadSystem<BoxProblem>::ComputeOpacity(const double /*rho*/, const double /*Tgas*/) -> double
{
	amrex::Real kappa = kappa_pipe;
	return kappa;
}

template <>
auto RadSystem<BoxProblem>::ComputeTgasFromEgas(const double rho, const double Egas) -> double
{
	return Egas / (rho * c_v);
}

template <>
auto RadSystem<BoxProblem>::ComputeEgasFromTgas(const double rho, const double Tgas) -> double
{
	return rho * c_v * Tgas;
}

template <>
auto RadSystem<BoxProblem>::ComputeEgasTempDerivative(const double rho, const double /*Tgas*/)
    -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	return rho * c_v;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
RadiationSimulation<BoxProblem>::setCustomBoundaryConditions(
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

	// const amrex::Real *dx = geom.CellSize();
	// const amrex::Real *prob_lo = geom.ProbLo();
	const amrex::Box &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	double E_0 = NAN;
	double Fx_0 = NAN;
	double Fy_0 = NAN;
	double Fz_0 = NAN;

	if (i < lo[0]) {
		E_0 = consVar(lo[0], j, k, RadSystem<BoxProblem>::radEnergy_index);
		Fx_0 = consVar(lo[0], j, k, RadSystem<BoxProblem>::x1RadFlux_index);
		Fy_0 = consVar(lo[0], j, k, RadSystem<BoxProblem>::x2RadFlux_index);
		Fz_0 = consVar(lo[0], j, k, RadSystem<BoxProblem>::x3RadFlux_index);
	} else if (j < lo[1]) {
		E_0 = consVar(i, lo[1], k, RadSystem<BoxProblem>::radEnergy_index);
		Fx_0 = consVar(i, lo[1], k, RadSystem<BoxProblem>::x1RadFlux_index);
		Fy_0 = consVar(i, lo[1], k, RadSystem<BoxProblem>::x2RadFlux_index);
		Fz_0 = consVar(i, lo[1], k, RadSystem<BoxProblem>::x3RadFlux_index);
	} else if (i > hi[0]) {
		E_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::radEnergy_index);
		Fx_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::x1RadFlux_index);
		Fy_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::x2RadFlux_index);
		Fz_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::x3RadFlux_index);
	} else if (j > hi[1]) {
		E_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::radEnergy_index);
		Fx_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::x1RadFlux_index);
		Fy_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::x2RadFlux_index);
		Fz_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::x3RadFlux_index);
	} else {
		return;
	}

	double E_inc = NAN;
	double Fx_bdry = NAN;
	double Fy_bdry = NAN;
	double Fz_bdry = NAN;

	if ((i < lo[0]) || (j < lo[1])) {
		// lower and left-side walls
		E_inc = a_rad * std::pow(T_hohlraum, 4);
		Fx_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * Fx_0);
		Fy_bdry = 0.;
		Fz_bdry = 0.;
	} else { // upper and right-side walls
		// extrapolated boundary
		E_inc = E_0;
		Fx_bdry = Fx_0;
		Fy_bdry = Fy_0;
		Fz_bdry = Fz_0;
	}

	const amrex::Real Fnorm =
	    std::sqrt(Fx_bdry * Fx_bdry + Fy_bdry * Fy_bdry + Fz_bdry * Fz_bdry);
	AMREX_ASSERT((Fnorm / (c * E_inc)) < 1.0); // flux-limiting condition

	consVar(i, j, k, RadSystem<BoxProblem>::radEnergy_index) = E_inc;
	consVar(i, j, k, RadSystem<BoxProblem>::x1RadFlux_index) = Fx_bdry;
	consVar(i, j, k, RadSystem<BoxProblem>::x2RadFlux_index) = Fy_bdry;
	consVar(i, j, k, RadSystem<BoxProblem>::x3RadFlux_index) = Fz_bdry;
}

template <> void RadiationSimulation<BoxProblem>::setInitialConditions()
{
	// auto prob_lo = simGeometry_.ProbLo();
	// auto prob_hi = simGeometry_.ProbHi();
	// auto dx = dx_;

	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const double Erad = a_rad * std::pow(T_initial, 4);
			double rho = rho_pipe;
			const double Egas =
			    RadSystem<BoxProblem>::ComputeEgasFromTgas(rho, T_initial);

			state(i, j, k, RadSystem<BoxProblem>::radEnergy_index) = Erad;
			state(i, j, k, RadSystem<BoxProblem>::x1RadFlux_index) = 0;
			state(i, j, k, RadSystem<BoxProblem>::x2RadFlux_index) = 0;
			state(i, j, k, RadSystem<BoxProblem>::x3RadFlux_index) = 0;

			state(i, j, k, RadSystem<BoxProblem>::gasEnergy_index) = Egas;
			state(i, j, k, RadSystem<BoxProblem>::gasDensity_index) = rho;
			state(i, j, k, RadSystem<BoxProblem>::x1GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<BoxProblem>::x2GasMomentum_index) = 0.;
			state(i, j, k, RadSystem<BoxProblem>::x3GasMomentum_index) = 0.;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

// based on:
// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template <class T>
auto isEqualToMachinePrecision(T x, T y, int ulp = 7) ->
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
{
	// the machine epsilon has to be scaled to the magnitude of the values used
	// and multiplied by the desired precision in ULPs (units in the last place)
	// [Note: 7 ULP * epsilon() ~= 1.554e-15]
	return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
	       // unless the result is subnormal
	       || std::fabs(x - y) < std::numeric_limits<T>::min();
}

template <> void RadiationSimulation<BoxProblem>::computeAfterTimestep()
{
	// copy all FABs to a local FAB across the entire domain
	amrex::BoxArray localBoxes(domain_);
	amrex::DistributionMapping localDistribution(localBoxes, 1);
	amrex::MultiFab state_mf(localBoxes, localDistribution, ncomp_, 0);
	state_mf.ParallelCopy(state_new_);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &state = state_mf.array(0);
		auto prob_lo = simGeometry_.ProbLo();
		auto dx = dx_;

		amrex::Long asymmetry = 0;
		auto nx = nx_;
		auto ny = ny_;
		auto nz = nz_;
		auto ncomp = ncomp_;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				for (int k = 0; k < nz; ++k) {
					amrex::Real const x =
					    prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
					amrex::Real const y =
					    prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
					for (int n = 0; n < ncomp; ++n) {
						const amrex::Real comp_upper = state(i, j, k, n);
						// reflect across x/y diagonal
						const amrex::Real comp_lower = state(j, i, k, n);
						const amrex::Real average =
						    std::fabs(comp_upper + comp_lower);
						const amrex::Real residual =
						    std::abs(comp_upper - comp_lower) / average;
						
						if (!isEqualToMachinePrecision(comp_upper,
									      comp_lower)) {
							amrex::Print()
							    << i << ", " << j << ", " << k << ", "
							    << n << ", " << comp_upper << ", "
							    << comp_lower << " " << residual
							    << "\n";
							amrex::Print() << "x = " << x << "\n";
							amrex::Print() << "y = " << y << "\n";
							asymmetry++;
							AMREX_ASSERT_WITH_MESSAGE(
							    false, "x/y not symmetric!");
						}
					}
				}
			}
		}
		AMREX_ASSERT_WITH_MESSAGE(asymmetry == 0, "x/y not symmetric!");
	}
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

	constexpr int nvars = 9;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		// x
		boundaryConditions[n].setLo(0, amrex::BCType::ext_dir);	 // left x1 -- Marshak
		boundaryConditions[n].setHi(0, amrex::BCType::foextrap); // right x1 -- extrapolate
		// y
		boundaryConditions[n].setLo(1, amrex::BCType::ext_dir);	 // left x2 -- Marshak
		boundaryConditions[n].setHi(1, amrex::BCType::foextrap); // right x2 -- extrapolate
	}

	// Problem initialization
	RadiationSimulation<BoxProblem> sim(gridDims, boxSize, boundaryConditions, nvars);
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.outputAtInterval_ = true;
	sim.plotfileInterval_ = 10; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
