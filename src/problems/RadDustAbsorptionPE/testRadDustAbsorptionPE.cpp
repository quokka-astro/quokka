//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRadDustAbsorptionPE.cpp
/// \brief Defines a test problem for photoelectric heating by dust-absorption bands.
///
/// The same uniform slab as RadDustAbsorption, but with a non-zero photoelectric yield on the absorbing
/// band. The gas is now heated, at a rate that is a fixed fraction of the energy the band absorbs:
///
///     dE_int/dt = epsilon * c * rho * kappa * E_g = epsilon * rho * kappa * F_0 * exp(-rho kappa x)
///
/// in the steady-state slab. The test compares the whole profile against that expression, not merely
/// that the gas got hotter. Because epsilon = 0.01, a profile 100 times larger is exactly what the
/// thermal-band behaviour would produce, so pinning the amplitude pins both that the photoelectric
/// heating is right and that the rest of the absorbed energy still does not reach the gas.
///
/// The transparent second band has epsilon = 0, which pins that a zero-yield band contributes nothing.

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

struct DustAbsorptionPEProblem {};

constexpr double c = 1.0;     // speed of light
constexpr double rho0 = 10.0; // gas density
constexpr double Lx = 2.0;    // slab thickness

constexpr double kappa0 = 0.1; // absorbing band
constexpr double kappa1 = 0.0; // transparent band
constexpr double tau0 = rho0 * kappa0 * Lx;

// Photoelectric yield: the fraction of the absorbed band energy returned to the gas. Only the absorbing
// band has one; the transparent band is left at zero.
constexpr double epsilon0 = 0.01;

constexpr double Frad0 = 2.0e-7; // incident flux, per band
constexpr double initial_Erad = 1.0e-12;
constexpr double tmax = 100.0; // 50 light-crossing times, so the slab is in steady state

// Sized so the photoelectric heating roughly doubles the internal energy over the run: large enough to
// measure precisely, small enough that the heating stays a perturbation on the radiation budget.
constexpr double initial_Egas = 1.0e-7;

template <> struct quokka::EOS_Traits<DustAbsorptionPEProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DustAbsorptionPEProblem> : DefaultPhysicsTraits {
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

template <> struct RadSystem_Traits<DustAbsorptionPEProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr double energy_unit = 1.0;
	static constexpr amrex::GpuArray<double, 3> radBoundaries = {0.1, 1.0, 10.0};
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool dust_absorption_only = true;
	static constexpr amrex::GpuArray<double, 2> pe_heating_efficiency = {epsilon0, 0.0};
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DustAbsorptionPEProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, 3> /*rad_boundaries*/,
												    const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, 3>, 2>
{
	// Note that the opacity is independent of Tgas. The dust-absorption band solver evaluates the opacity
	// once, at the start-of-step gas temperature, and does not revise it for the temperature change the
	// photoelectric heating produces within the step.
	amrex::GpuArray<amrex::GpuArray<double, 3>, 2> exponents_and_values{};
	for (int i = 0; i < 3; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	exponents_and_values[1][0] = kappa0;
	exponents_and_values[1][1] = kappa1;
	exponents_and_values[1][2] = kappa1;
	return exponents_and_values;
}

template <> void QuokkaSimulation<DustAbsorptionPEProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int g = 0; g < Physics_Traits<DustAbsorptionPEProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = initial_Erad;
			state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}

		state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::x3GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::gasEnergy_index) = initial_Egas;
		state_cc(i, j, k, RadSystem<DustAbsorptionPEProblem>::gasInternalEnergy_index) = initial_Egas;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustAbsorptionPEProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
								    int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								    const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	constexpr int nvar = Physics_Indices<DustAbsorptionPEProblem>::nvarTotal_cc;

	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};

	// a fully beamed source along +x: |F| = c E
	for (int g = 0; g < Physics_Traits<DustAbsorptionPEProblem>::nGroups; ++g) {
		low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad0 / c;
		low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad0;
		low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0.;
		low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0.;
	}

	low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::gasDensity_index] = rho0;
	low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::x1GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::x3GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::gasEnergy_index] = initial_Egas;
	low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::gasInternalEnergy_index] = initial_Egas;

	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
}

auto problem_main() -> int
{
	constexpr int nvars = RadSystem<DustAbsorptionPEProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<DustAbsorptionPEProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = 0.8;
	sim.maxTimesteps_ = 500000;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// 1. the beam is still attenuated as exp(-rho kappa x)
	double erad_err = 0.;
	double erad_sol = 0.;
	for (int i = 0; i < nx; ++i) {
		const double x = position[i];
		for (int g = 0; g < Physics_Traits<DustAbsorptionPEProblem>::nGroups; ++g) {
			const double kappa = (g == 0) ? kappa0 : kappa1;
			const double Erad_exact = (Frad0 / c) * std::exp(-rho0 * kappa * x);
			const double Erad = values.at(RadSystem<DustAbsorptionPEProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g)[i];
			erad_err += std::abs(Erad - Erad_exact);
			erad_sol += std::abs(Erad_exact);
		}
	}
	const double erad_rel_err = erad_err / erad_sol;

	// 2. the gas is heated at exactly the photoelectric rate. In the steady-state slab the local heating
	// rate is epsilon * c * rho * kappa * E_g = epsilon * rho * kappa * Frad0 * exp(-rho kappa x), and a
	// cell starts accumulating it once the beam front reaches it at t = x / c.
	double eint_err = 0.;
	double eint_sol = 0.;
	for (int i = 0; i < nx; ++i) {
		const double x = position[i];
		const double dEint_exact = epsilon0 * rho0 * kappa0 * Frad0 * std::exp(-rho0 * kappa0 * x) * (tmax - x / c);
		const double dEint = values.at(RadSystem<DustAbsorptionPEProblem>::gasInternalEnergy_index)[i] - initial_Egas;
		eint_err += std::abs(dEint - dEint_exact);
		eint_sol += std::abs(dEint_exact);
	}
	const double eint_rel_err = eint_err / eint_sol;

	// 3. the gas still receives the momentum of the absorbed radiation
	double momentum = 0.;
	const double dx = Lx / nx;
	for (int i = 0; i < nx; ++i) {
		momentum += values.at(RadSystem<DustAbsorptionPEProblem>::x1GasMomentum_index)[i] * dx;
	}
	const double momentum_exact = (Frad0 / c) * (1.0 - std::exp(-tau0)) * tmax;
	const double momentum_ratio = momentum / momentum_exact;

	amrex::Print() << "Relative L1 norm of Erad = " << erad_rel_err << '\n';
	amrex::Print() << "Relative L1 norm of the photoelectric heating profile = " << eint_rel_err << '\n';
	amrex::Print() << "Gas momentum / analytic = " << momentum_ratio << '\n';

	int status = 0;
	if (!(erad_rel_err < 0.02)) {
		amrex::Print() << "ERROR: the beam is not attenuated as exp(-rho kappa x).\n";
		status = 1;
	}
	if (!(eint_rel_err < 0.02)) {
		amrex::Print() << "ERROR: the gas heating does not match the photoelectric rate.\n";
		status = 1;
	}
	if (!((momentum_ratio > 0.95) && (momentum_ratio < 1.005))) {
		amrex::Print() << "ERROR: the gas did not receive the momentum of the absorbed radiation.\n";
		status = 1;
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
