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
#include "test_radhydro_shock_cgs.hpp"
#include <csignal>
#include <limits>
#include <tuple>

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
AMREX_GPU_HOST_DEVICE auto RadSystem<BoxProblem>::ComputeOpacity(const double /*rho*/,
								 const double /*Tgas*/) -> double
{
	amrex::Real kappa = kappa_pipe;
	return kappa;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<BoxProblem>::ComputeTgasFromEgas(const double rho,
								      const double Egas) -> double
{
	return Egas / (rho * c_v);
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<BoxProblem>::ComputeEgasFromTgas(const double rho,
								      const double Tgas) -> double
{
	return rho * c_v * Tgas;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<BoxProblem>::ComputeEgasTempDerivative(const double rho,
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
RadhydroSimulation<BoxProblem>::setCustomBoundaryConditions(
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

	const amrex::Box &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	bool left = (i < lo[0]);
	bool right = (i > hi[0]);
	bool lower = (j < lo[1]);
	bool upper = (j > hi[1]);

	if (!(left || right || lower || upper)) {
		return;
	}

	const double E_inc = a_rad * std::pow(T_hohlraum, 4);
	double E_bdry = NAN;
	double Fx_bdry = 0.;
	double Fy_bdry = 0.;
	double Fz_bdry = 0.;

	if (left) { // left boundary
		const double E_0 = consVar(lo[0], j, k, RadSystem<BoxProblem>::radEnergy_index);
		const double Fx_0 = consVar(lo[0], j, k, RadSystem<BoxProblem>::x1RadFlux_index);
		// Marshak boundary -- left wall
		E_bdry = E_inc;
		Fx_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * Fx_0);
	}
	if (lower) { // lower boundary
		const double E_0 = consVar(i, lo[1], k, RadSystem<BoxProblem>::radEnergy_index);
		const double Fy_0 = consVar(i, lo[1], k, RadSystem<BoxProblem>::x2RadFlux_index);
		// Marshak boundary -- lower wall
		E_bdry = E_inc;
		Fy_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * Fy_0);
	}
	if (right) { // right boundary
		const double E_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::radEnergy_index);
		const double Fx_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::x1RadFlux_index);
		const double Fy_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::x2RadFlux_index);
		const double Fz_0 = consVar(hi[0], j, k, RadSystem<BoxProblem>::x3RadFlux_index);
		// extrapolated boundary -- upper and right-side walls
		E_bdry = E_0;
		Fx_bdry = Fx_0;
		Fy_bdry = Fy_0;
		Fz_bdry = Fz_0;
	}
	if (upper) { // upper boundary
		const double E_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::radEnergy_index);
		const double Fx_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::x1RadFlux_index);
		const double Fy_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::x2RadFlux_index);
		const double Fz_0 = consVar(i, hi[1], k, RadSystem<BoxProblem>::x3RadFlux_index);
		// extrapolated boundary -- upper and right-side walls
		E_bdry = E_0;
		Fx_bdry = Fx_0;
		Fy_bdry = Fy_0;
		Fz_bdry = Fz_0;
	}

	const amrex::Real Fnorm =
	    std::sqrt(Fx_bdry * Fx_bdry + Fy_bdry * Fy_bdry + Fz_bdry * Fz_bdry);
	AMREX_ASSERT((Fnorm / (c * E_bdry)) < 1.0); // flux-limiting condition

	consVar(i, j, k, RadSystem<BoxProblem>::radEnergy_index) = E_bdry;
	consVar(i, j, k, RadSystem<BoxProblem>::x1RadFlux_index) = Fx_bdry;
	consVar(i, j, k, RadSystem<BoxProblem>::x2RadFlux_index) = Fy_bdry;
	consVar(i, j, k, RadSystem<BoxProblem>::x3RadFlux_index) = Fz_bdry;
}

template <> void RadhydroSimulation<BoxProblem>::setInitialConditions()
{
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

namespace quokka
{
template <>
AMREX_GPU_HOST_DEVICE auto
CheckSymmetryArray<BoxProblem>(amrex::Array4<const amrex::Real> const &arr,
			       amrex::Box const &indexRange, const int ncomp,
			       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) -> bool
{
	amrex::Long asymmetry = 0;
	amrex::GpuArray<int, 3> prob_lo = indexRange.loVect3d();
	auto nx = indexRange.hiVect3d()[0] + 1;
	auto ny = indexRange.hiVect3d()[1] + 1;
	auto nz = indexRange.hiVect3d()[2] + 1;
	AMREX_ASSERT(prob_lo[0] == 0);
	AMREX_ASSERT(prob_lo[1] == 0);
	AMREX_ASSERT(prob_lo[2] == 0);
	auto state = arr;

	// std::raise(SIGINT); // break

	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			for (int k = 0; k < nz; ++k) {
				amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
				amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];

				for (int n = 0; n < ncomp; ++n) {
					const amrex::Real comp_upper = state(i, j, k, n);

					// reflect across x/y diagonal
					int n_lower = n;
					if (n == RadSystem<BoxProblem>::x1RadFlux_index) {
						n_lower = RadSystem<BoxProblem>::x2RadFlux_index;
					} else if (n == RadSystem<BoxProblem>::x2RadFlux_index) {
						n_lower = RadSystem<BoxProblem>::x1RadFlux_index;
					} else if (n ==
						   RadSystem<BoxProblem>::x1GasMomentum_index) {
						n_lower =
						    RadSystem<BoxProblem>::x2GasMomentum_index;
					} else if (n ==
						   RadSystem<BoxProblem>::x2GasMomentum_index) {
						n_lower =
						    RadSystem<BoxProblem>::x1GasMomentum_index;
					}
					amrex::Real comp_lower = state(j, i, k, n_lower);

					const amrex::Real average =
					    std::fabs(comp_upper + comp_lower);
					const amrex::Real residual =
					    std::abs(comp_upper - comp_lower) / average;

					if (comp_upper != comp_lower) {
#ifndef AMREX_USE_GPU
						amrex::Print()
						    << i << ", " << j << ", " << k << ", " << n
						    << ", " << comp_upper << ", " << comp_lower
						    << " " << residual << "\n";
						amrex::Print() << "x = " << x << "\n";
						amrex::Print() << "y = " << y << "\n";
#endif
						asymmetry++;
						AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
						    false, "x/y not symmetric!");
					}
				}
			}
		}
	}
	return true;
}
} // namespace quokka

auto problem_main() -> int
{
	// Problem parameters
	const int max_timesteps = 1000;
	const double CFL_number = 0.4;
	const int nx = 100;
	const int ny = 100;

	const double Lx = 1.0;		 // cm
	const double Ly = 1.0;		 // cm
	const double max_time = 1.0e-11; // s

	amrex::IntVect gridDims{AMREX_D_DECL(nx, ny, 4)};
	amrex::RealBox boxSize{
	    {AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.), amrex::Real(0.0))}, // NOLINT
	    {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Ly), amrex::Real(1.0))}}; // NOLINT

	constexpr int nvars = 9;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		// x
		boundaryConditions[n].setLo(0, amrex::BCType::ext_dir); // left x1 -- Marshak
		boundaryConditions[n].setHi(0, amrex::BCType::ext_dir); // right x1 -- extrapolate
		// y
		boundaryConditions[n].setLo(1, amrex::BCType::ext_dir); // left x2 -- Marshak
		boundaryConditions[n].setHi(1, amrex::BCType::ext_dir); // right x2 -- extrapolate
	}

	// Problem initialization
	RadhydroSimulation<BoxProblem> sim(gridDims, boxSize, boundaryConditions);
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
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
