//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRadDustAbsorptionPPL.cpp
/// \brief Defines a test problem for dust-absorption-only radiation bands.
///
/// A beam of interstellar ultraviolet radiation enters a uniform, static slab of cold molecular gas
/// through the left boundary. The bands are dust-absorption bands, so they are absorbed and push on the
/// gas but never heat it thermally; the only energy they give the gas is photoelectric, at the rate of
/// Bate & Keto (2015), Eq. 26,
///
///     dE_int/dt = sum_g epsilon_g * R * n_H * E_g ,   R = 1.33e-24 / 5.29e-14 cm^3 s^-1 .
///
/// The problem is in cgs with interstellar values -- n_H = 100 cm^-3, T = 20 K, epsilon = 0.05, and one
/// Habing unit incident per band over about 6 pc -- because that rate coefficient is empirical and
/// defined in cgs. The heating then roughly doubles the internal energy over the run.
///
/// Three bands isolate the three behaviours that define the band type. They are chosen to separate those
/// behaviours rather than to model a real dust opacity curve:
///
///   band 0: absorbed (tau = 2), epsilon = 0.05  -- attenuation, radiation force, and heating
///   band 1: transparent,        epsilon = 0.05  -- heats the gas although nothing is absorbed
///   band 2: transparent,        epsilon = 0     -- contributes nothing at all
///
/// Band 1 matters because photoelectric heating is the photoelectric effect on grains, not a share of
/// the energy the dust absorbs: the rate does not go through the band opacity, so a transparent band
/// heats the gas exactly as much as an absorbed one carrying the same E_g. Band 2 pins that a zero
/// efficiency really does switch a band's heating off, which is how non-ultraviolet bands are labelled.
///
/// Comparing the whole internal-energy profile against the analytic photoelectric rate also pins the
/// defining property of the band type, that the absorbed energy does not reach the gas: were it
/// delivered as heat, the profile would be about an order of magnitude larger (see the printout).
///
/// This is RadDustAbsorption with a piecewise power-law opacity model instead of a piecewise constant
/// one, in the same spirit as the RadhydroPulseMGconst / RadhydroPulseMGint pair. All the power-law
/// slopes are zero, so the opacity is numerically the same as in RadDustAbsorption and every expected
/// value below is unchanged; what differs is the code path that produces the mean opacities. In
/// particular kappaF comes from ComputeDiffusionFluxMeanOpacity, whose Planck-weighted denominator
/// vanishes for a band that does not emit. The momentum check is what catches that: it reads exactly
/// zero if the non-emitting bands are not given a flux-mean opacity of their own.

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

struct DustAbsorptionPPLProblem {};

constexpr int n_groups = 3;
constexpr double gamma_gas = 5. / 3.;
constexpr double n_H = 100.0;	  // cm^-3, a cold interstellar cloud
constexpr double mu = C::m_u;	  // g, so that n_H = rho / mu
constexpr double rho0 = n_H * mu; // g cm^-3
constexpr double T0 = 20.0;	  // K
constexpr double Lx = 2.0e19;	  // cm, about 6 pc

constexpr double kappa_absorbing = 2.0 / (rho0 * Lx); // cm^2 g^-1, giving tau = 2 across the slab
constexpr double tau_absorbing = rho0 * kappa_absorbing * Lx;

// Photoelectric efficiency and the rate coefficient of Bate & Keto (2015), Eq. 26. The coefficient is
// repeated here rather than taken from RadSystem, so that the check is independent of the solver.
constexpr double epsilon0 = 0.05;
constexpr double J_ISR = 5.29e-14;	       // erg cm^-3, the reference interstellar field
constexpr double pe_rate = 1.33e-24 / J_ISR;   // cm^3 s^-1
constexpr double Frad0 = J_ISR * c_light_cgs_; // one Habing unit at the boundary, per band

constexpr double initial_Erad = 1.0e-25;
constexpr double initial_Egas = n_H * C::k_B * T0 / (gamma_gas - 1.0);
constexpr double tmax = 6.0e10; // s, about 90 light-crossing times; long enough to roughly double E_int

template <> struct quokka::EOS_Traits<DustAbsorptionPPLProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = gamma_gas;
};

template <> struct Physics_Traits<DustAbsorptionPPLProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = n_groups;
	// face-centred
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<DustAbsorptionPPLProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, n_groups + 1> radBoundaries = {5.0, 8.0, 11.2, 13.6}; // eV
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
	static constexpr bool dust_absorption_only = true;
	static constexpr amrex::GpuArray<double, n_groups> pe_heating_efficiency = {epsilon0, epsilon0, 0.0};
};

// the opacity of each band, in the order described at the top of this file
constexpr amrex::GpuArray<double, n_groups> kappa_band = {kappa_absorbing, 0.0, 0.0};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DustAbsorptionPPLProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, n_groups + 1> /*rad_boundaries*/,
												     const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, n_groups + 1>, 2>
{
	// The opacity is independent of Tgas, which the dust-absorption band solver requires: it evaluates
	// the opacity once, at the start-of-step gas temperature, and does not revise it for the temperature
	// change the photoelectric heating produces within the step.
	// kappa_band has no device storage, so copy it to a local before indexing it with a runtime index
	const amrex::GpuArray<double, n_groups> kappa = kappa_band;
	amrex::GpuArray<amrex::GpuArray<double, n_groups + 1>, 2> exponents_and_values{};
	for (int i = 0; i < n_groups + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = kappa[std::min(i, n_groups - 1)];
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<DustAbsorptionPPLProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int g = 0; g < n_groups; ++g) {
			state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = initial_Erad;
			state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
			state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0;
		}

		state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::x3GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::gasEnergy_index) = initial_Egas;
		state_cc(i, j, k, RadSystem<DustAbsorptionPPLProblem>::gasInternalEnergy_index) = initial_Egas;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<DustAbsorptionPPLProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
								     int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
								     const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	constexpr int nvar = Physics_Indices<DustAbsorptionPPLProblem>::nvarTotal_cc;

	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};

	// a fully beamed source along +x: |F| = c E
	for (int g = 0; g < n_groups; ++g) {
		low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad0 / c_light_cgs_;
		low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = Frad0;
		low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0.;
		low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g] = 0.;
	}

	low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::gasDensity_index] = rho0;
	low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::x1GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::x3GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::gasEnergy_index] = initial_Egas;
	low_bdr_cells[RadSystem<DustAbsorptionPPLProblem>::gasInternalEnergy_index] = initial_Egas;

	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
}

auto problem_main() -> int
{
	constexpr int nvars = RadSystem<DustAbsorptionPPLProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<DustAbsorptionPPLProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = 0.8;
	sim.maxTimesteps_ = 500000;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// 1. each band is attenuated as exp(-rho kappa_g x); the transparent bands are not attenuated at all
	double erad_err = 0.;
	double erad_sol = 0.;
	for (int i = 0; i < nx; ++i) {
		const double x = position[i];
		for (int g = 0; g < n_groups; ++g) {
			const double Erad_exact = (Frad0 / c_light_cgs_) * std::exp(-rho0 * kappa_band[g] * x);
			const double Erad = values.at(RadSystem<DustAbsorptionPPLProblem>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g)[i];
			erad_err += std::abs(Erad - Erad_exact);
			erad_sol += std::abs(Erad_exact);
		}
	}
	const double erad_rel_err = erad_err / erad_sol;

	// 2. the gas is heated at exactly the photoelectric rate, summed over the bands that have a non-zero
	// efficiency. A cell starts accumulating once the beam front reaches it at t = x / c.
	double eint_err = 0.;
	double eint_sol = 0.;
	double max_heating_ratio = 0.;
	for (int i = 0; i < nx; ++i) {
		const double x = position[i];
		// bands 0 and 1 heat; band 2 has zero efficiency and must contribute nothing
		const double Erad_heating = (Frad0 / c_light_cgs_) * (std::exp(-rho0 * kappa_absorbing * x) + 1.0);
		const double dEint_exact = epsilon0 * pe_rate * n_H * Erad_heating * (tmax - x / c_light_cgs_);
		const double dEint = values.at(RadSystem<DustAbsorptionPPLProblem>::gasInternalEnergy_index)[i] - initial_Egas;
		eint_err += std::abs(dEint - dEint_exact);
		eint_sol += std::abs(dEint_exact);
		max_heating_ratio = std::max(max_heating_ratio, dEint / initial_Egas);
	}
	const double eint_rel_err = eint_err / eint_sol;

	// For scale: had the absorbed energy been delivered to the gas as heat, as it is for a thermal band,
	// the gain at the illuminated face would be this instead of max_heating_ratio.
	const double heating_if_thermal = rho0 * kappa_absorbing * Frad0 * tmax / initial_Egas;

	// 3. the gas still receives the momentum of the absorbed radiation. Only the absorbing band
	// contributes: the transparent bands heat the gas but exert no force on it.
	double momentum = 0.;
	const double dx = Lx / nx;
	for (int i = 0; i < nx; ++i) {
		momentum += values.at(RadSystem<DustAbsorptionPPLProblem>::x1GasMomentum_index)[i] * dx;
	}
	const double momentum_exact = (Frad0 / c_light_cgs_) * (1.0 - std::exp(-tau_absorbing)) * tmax;
	const double momentum_ratio = momentum / momentum_exact;

	amrex::Print() << "Relative L1 norm of Erad = " << erad_rel_err << '\n';
	amrex::Print() << "Relative L1 norm of the photoelectric heating profile = " << eint_rel_err << '\n';
	amrex::Print() << "Peak gas internal energy gain = " << max_heating_ratio << " of its initial value (would be " << heating_if_thermal
		       << " if the absorbed energy heated the gas)\n";
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
