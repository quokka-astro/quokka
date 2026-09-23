//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRadDustAbsorptionPE.cpp
/// \brief Defines a test problem for photoelectric heating by dust-absorption bands.
///
/// A beam of interstellar far-ultraviolet radiation enters a uniform, static slab of cold molecular gas
/// through the left boundary. Both bands are dust-absorption bands with a photoelectric efficiency, so
/// the gas is heated at
///
///     dE_int/dt = epsilon * R * n_H * sum_g E_g ,   R = 1.33e-24 / 5.29e-14 cm^3 s^-1
///
/// following Bate & Keto (2015), Eq. 26. The test compares the whole internal-energy profile against
/// that expression rather than merely checking that the gas got hotter.
///
/// The problem runs in cgs with interstellar values -- n_H = 100 cm^-3, T = 20 K, epsilon = 0.05, and an
/// incident field of one Habing unit per band -- because the rate coefficient is empirical and defined
/// in cgs. The heating then roughly doubles the internal energy over the run.
///
/// The two bands differ in their opacity but share an efficiency: the first is absorbed (tau = 2 across
/// the slab), the second is transparent. That pins a property of the photoelectric effect which a
/// fraction-of-absorbed-energy model would get wrong -- the heating does not go through the dust
/// opacity, so the transparent band heats the gas just as much, and neither band's heating is taken out
/// of the radiation.

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

struct DustAbsorptionPEProblem {
};

constexpr double gamma_gas = 5. / 3.;
constexpr double n_H = 100.0;			// cm^-3, a cold interstellar cloud
constexpr double mu = C::m_u;			// g, so that n_H = rho / mu
constexpr double rho0 = n_H * mu;		// g cm^-3
constexpr double T0 = 20.0;			// K
constexpr double Lx = 2.0e19;			// cm, about 6 pc
constexpr double kappa_fuv = 2.0 / (rho0 * Lx); // cm^2 g^-1, giving tau = 2 across the slab
constexpr double kappa_lw = 0.0;		// the second band is transparent
constexpr double tau_fuv = rho0 * kappa_fuv * Lx;

// Photoelectric efficiency and the rate coefficient of Bate & Keto (2015), Eq. 26. The coefficient is
// repeated here rather than taken from RadSystem, so that the check is independent of the solver.
constexpr double epsilon0 = 0.05;
constexpr double J_ISR = 5.29e-14;	       // erg cm^-3, the reference interstellar field
constexpr double pe_rate = 1.33e-24 / J_ISR;   // cm^3 s^-1
constexpr double Frad0 = J_ISR * c_light_cgs_; // one Habing unit at the boundary, per band

constexpr double initial_Erad = 1.0e-25;
constexpr double initial_Egas = n_H * C::k_B * T0 / (gamma_gas - 1.0);
constexpr double tmax = 6.0e10; // s, about 90 light-crossing times; long enough to roughly double E_int

template <> struct quokka::EOS_Traits<DustAbsorptionPEProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = gamma_gas;
};

template <> struct Physics_Traits<DustAbsorptionPEProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 2;
	// face-centred
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<DustAbsorptionPEProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr double energy_unit = C::ev2erg;
	// far-ultraviolet and Lyman-Werner, in eV
	static constexpr amrex::GpuArray<double, 3> radBoundaries = {6.0, 11.2, 13.6};
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool dust_absorption_only = true;
	static constexpr amrex::GpuArray<double, 2> pe_heating_efficiency = {epsilon0, epsilon0};
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DustAbsorptionPEProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, 3> /*rad_boundaries*/,
												   const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, 3>, 2>
{
	// The opacity is independent of Tgas, which the dust-absorption band solver requires: it evaluates
	// the opacity once, at the start-of-step gas temperature, and does not revise it for the temperature
	// change the photoelectric heating produces within the step.
	amrex::GpuArray<amrex::GpuArray<double, 3>, 2> exponents_and_values{};
	for (int i = 0; i < 3; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	exponents_and_values[1][0] = kappa_fuv;
	exponents_and_values[1][1] = kappa_lw;
	exponents_and_values[1][2] = kappa_lw;
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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<DustAbsorptionPEProblem>::setCustomBoundaryConditions(
    const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
    const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	constexpr int nvar = Physics_Indices<DustAbsorptionPEProblem>::nvarTotal_cc;

	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};

	// a fully beamed source along +x: |F| = c E
	for (int g = 0; g < Physics_Traits<DustAbsorptionPEProblem>::nGroups; ++g) {
		low_bdr_cells[RadSystem<DustAbsorptionPEProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad0 / c_light_cgs_;
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

	// 1. each band is attenuated as exp(-rho kappa_g x); the transparent band is not attenuated at all
	double erad_err = 0.;
	double erad_sol = 0.;
	for (int i = 0; i < nx; ++i) {
		const double x = position[i];
		for (int g = 0; g < Physics_Traits<DustAbsorptionPEProblem>::nGroups; ++g) {
			const double kappa = (g == 0) ? kappa_fuv : kappa_lw;
			const double Erad_exact = (Frad0 / c_light_cgs_) * std::exp(-rho0 * kappa * x);
			const double Erad = values.at(RadSystem<DustAbsorptionPEProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g)[i];
			erad_err += std::abs(Erad - Erad_exact);
			erad_sol += std::abs(Erad_exact);
		}
	}
	const double erad_rel_err = erad_err / erad_sol;

	// 2. the gas is heated at exactly the photoelectric rate, summed over both bands. A cell starts
	// accumulating once the beam front reaches it at t = x / c.
	double eint_err = 0.;
	double eint_sol = 0.;
	double max_heating_ratio = 0.;
	for (int i = 0; i < nx; ++i) {
		const double x = position[i];
		const double Erad_sum = (Frad0 / c_light_cgs_) * (std::exp(-rho0 * kappa_fuv * x) + 1.0);
		const double dEint_exact = epsilon0 * pe_rate * n_H * Erad_sum * (tmax - x / c_light_cgs_);
		const double dEint = values.at(RadSystem<DustAbsorptionPEProblem>::gasInternalEnergy_index)[i] - initial_Egas;
		eint_err += std::abs(dEint - dEint_exact);
		eint_sol += std::abs(dEint_exact);
		max_heating_ratio = std::max(max_heating_ratio, dEint / initial_Egas);
	}
	const double eint_rel_err = eint_err / eint_sol;

	// 3. the gas still receives the momentum of the absorbed radiation. Only the absorbing band
	// contributes: the transparent one heats the gas but exerts no force on it.
	double momentum = 0.;
	const double dx = Lx / nx;
	for (int i = 0; i < nx; ++i) {
		momentum += values.at(RadSystem<DustAbsorptionPEProblem>::x1GasMomentum_index)[i] * dx;
	}
	const double momentum_exact = (Frad0 / c_light_cgs_) * (1.0 - std::exp(-tau_fuv)) * tmax;
	const double momentum_ratio = momentum / momentum_exact;

	amrex::Print() << "Relative L1 norm of Erad = " << erad_rel_err << '\n';
	amrex::Print() << "Relative L1 norm of the photoelectric heating profile = " << eint_rel_err << '\n';
	amrex::Print() << "Peak gas internal energy gain = " << max_heating_ratio << " of its initial value\n";
	amrex::Print() << "Gas momentum / analytic = " << momentum_ratio << '\n';

	int status = 0;
	if (!(erad_rel_err < 0.02)) {
		amrex::Print() << "ERROR: the bands are not attenuated as exp(-rho kappa x).\n";
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
