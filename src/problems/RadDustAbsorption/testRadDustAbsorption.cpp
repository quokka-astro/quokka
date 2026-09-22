//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRadDustAbsorption.cpp
/// \brief Defines a test problem for dust-absorption-only radiation bands.
///
/// A beam enters a uniform, static slab through the left boundary. The two groups are
/// dust-absorption-only bands (RadSystem_Traits::dust_absorption_only = true), so they are absorbed and
/// push on the gas but never heat it. The test pins the three properties that define the band type:
///
///   1. the beam is attenuated as exp(-rho kappa_g x), group by group, including the kappa = 0 limit;
///   2. the gas internal energy is unchanged, even though the absorbed energy exceeds it ~100-fold;
///   3. the gas still gains the momentum the absorbed radiation carried.
///
/// Check 2 is the defining property. If the absorbed energy were delivered to the gas as it is for a
/// thermal band, the internal energy would rise by roughly two orders of magnitude, so the test
/// separates the two behaviours by a wide margin rather than by a tolerance.

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include <cmath>
#include <format>

struct DustAbsorptionProblem {};

constexpr double c = 1.0;     // speed of light
constexpr double rho0 = 10.0; // gas density; large enough that the gas stays effectively static (v/c ~ 1e-6)
constexpr double Lx = 2.0;    // slab thickness

// Group 0 is absorbing, group 1 is transparent. The transparent band exercises the tau = 0 limit of the
// analytic band update, where the backward-Euler denominator is exactly 1.
constexpr double kappa0 = 0.1;
constexpr double kappa1 = 0.0;
constexpr double tau0 = rho0 * kappa0 * Lx; // = 2

constexpr double Frad0 = 2.0e-7; // incident flux, per group
constexpr double initial_Erad = 1.0e-12;
constexpr double tmax = 100.0; // 50 light-crossing times, so the slab is in steady state

// Gas internal energy, chosen so that the radiation absorbed over the run exceeds it ~100-fold. That
// ratio is what makes check 2 decisive.
constexpr double initial_Egas = 1.0e-7;

template <> struct quokka::EOS_Traits<DustAbsorptionProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DustAbsorptionProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2;
	// face-centred
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = 1.0;
};

template <> struct RadSystem_Traits<DustAbsorptionProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr double energy_unit = 1.0;
	static constexpr amrex::GpuArray<double, 3> radBoundaries = {0.1, 1.0, 10.0};
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool dust_absorption_only = true;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DustAbsorptionProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, 3> /*rad_boundaries*/,
												  const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, 3>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, 3>, 2> exponents_and_values{};
	for (int i = 0; i < 3; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	exponents_and_values[1][0] = kappa0;
	exponents_and_values[1][1] = kappa1;
	exponents_and_values[1][2] = kappa1;
	return exponents_and_values;
}

template <> void QuokkaSimulation<DustAbsorptionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int g = 0; g < Physics_Traits<DustAbsorptionProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = initial_Erad;
			state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}

		state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::x3GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::gasEnergy_index) = initial_Egas;
		state_cc(i, j, k, RadSystem<DustAbsorptionProblem>::gasInternalEnergy_index) = initial_Egas;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustAbsorptionProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
								  int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								  const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	constexpr int nvar = Physics_Indices<DustAbsorptionProblem>::nvarTotal_cc;

	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};

	// a fully beamed source along +x: |F| = c E
	for (int g = 0; g < Physics_Traits<DustAbsorptionProblem>::nGroups; ++g) {
		low_bdr_cells[RadSystem<DustAbsorptionProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad0 / c;
		low_bdr_cells[RadSystem<DustAbsorptionProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad0;
		low_bdr_cells[RadSystem<DustAbsorptionProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0.;
		low_bdr_cells[RadSystem<DustAbsorptionProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0.;
	}

	low_bdr_cells[RadSystem<DustAbsorptionProblem>::gasDensity_index] = rho0;
	low_bdr_cells[RadSystem<DustAbsorptionProblem>::x1GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionProblem>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionProblem>::x3GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionProblem>::gasEnergy_index] = initial_Egas;
	low_bdr_cells[RadSystem<DustAbsorptionProblem>::gasInternalEnergy_index] = initial_Egas;

	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
}

auto problem_main() -> int
{
	constexpr int nvars = RadSystem<DustAbsorptionProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<DustAbsorptionProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = 0.8;
	sim.maxTimesteps_ = 500000;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// 1. attenuation, group by group
	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		const double x = position[i];
		for (int g = 0; g < Physics_Traits<DustAbsorptionProblem>::nGroups; ++g) {
			const double kappa = (g == 0) ? kappa0 : kappa1;
			const double Erad_exact = (Frad0 / c) * std::exp(-rho0 * kappa * x);
			const double Erad = values.at(RadSystem<DustAbsorptionProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g)[i];
			err_norm += std::abs(Erad - Erad_exact);
			sol_norm += std::abs(Erad_exact);
		}
	}
	const double rel_err_norm = err_norm / sol_norm;

	// 2. the gas internal energy must be unchanged: the absorbed energy goes to the dust, not the gas
	double max_eint_change = 0.;
	for (int i = 0; i < nx; ++i) {
		const double Eint = values.at(RadSystem<DustAbsorptionProblem>::gasInternalEnergy_index)[i];
		max_eint_change = std::max(max_eint_change, std::abs(Eint - initial_Egas) / initial_Egas);
	}
	// For scale: had the absorbed energy been delivered to the gas, this ratio would be ~100.
	const double absorbed_per_volume = Frad0 * (1.0 - std::exp(-tau0)) * tmax / Lx;
	const double heating_if_thermal = absorbed_per_volume / initial_Egas;

	// 3. the gas must still gain the momentum the absorbed radiation carried
	double momentum = 0.;
	const double dx = Lx / nx;
	for (int i = 0; i < nx; ++i) {
		momentum += values.at(RadSystem<DustAbsorptionProblem>::x1GasMomentum_index)[i] * dx;
	}
	// Steady-state deposition rate, integrated over the run. The beam takes Lx/c to fill the slab, so the
	// measured value falls short of this by about Lx/(c tmax) = 2%.
	const double momentum_exact = (Frad0 / c) * (1.0 - std::exp(-tau0)) * tmax;
	const double momentum_ratio = momentum / momentum_exact;

	amrex::Print() << "Relative L1 norm of Erad = " << rel_err_norm << '\n';
	amrex::Print() << "Max relative change in gas internal energy = " << max_eint_change << " (would be " << heating_if_thermal
		       << " if the absorbed energy heated the gas)\n";
	amrex::Print() << "Gas momentum / analytic = " << momentum_ratio << '\n';

	int status = 0;
	if (!(rel_err_norm < 0.02)) {
		amrex::Print() << "ERROR: the beam is not attenuated as exp(-rho kappa x).\n";
		status = 1;
	}
	if (!(max_eint_change < 0.01)) {
		amrex::Print() << "ERROR: the gas internal energy changed; a dust-absorption band must not heat the gas.\n";
		status = 1;
	}
	if (!((momentum_ratio > 0.95) && (momentum_ratio < 1.005))) {
		amrex::Print() << "ERROR: the gas did not receive the momentum of the absorbed radiation.\n";
		status = 1;
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
