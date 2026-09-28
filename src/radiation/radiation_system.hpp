#ifndef RADIATION_SYSTEM_HPP_ // NOLINT
#define RADIATION_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.hpp
/// \brief Defines a class for solving the (1d) radiation moment equations.
///

// c++ headers

#include <array>
#include <cmath>

// library headers
#include "AMReX.H" // IWYU pragma: keep
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

// internal headers
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hyperbolic_system.hpp"
#include "math/math_impl.hpp"
#include "physics_info.hpp"
#include "radiation/planck_integral.hpp"
#include "util/valarray.hpp"

using Real = amrex::Real;

// Hyper parameters for the radiation solver
static constexpr bool include_delta_B = true;
static constexpr bool use_diffuse_flux_mean_opacity = true;
static constexpr bool special_edge_bin_slopes = false; // Use 2 and -4 as the slopes for the first and last bins, respectively
static constexpr bool include_work_term_in_source = true;

// Smallest fraction of a cell's internal energy that the beta_order = 1 work term is allowed to leave
// behind. The work done by radiation is credited to internal energy by the source term and then moved to
// kinetic energy in UpdateFlux; in a cold, strongly radiation-driven cell that transfer can exceed the
// internal energy available, and subtracting it unclamped leaves a negative internal energy. See the cap
// in UpdateFlux, which serves both single-group and multigroup radiation.
//
// Capping does not conserve energy. It is a known limitation of the lagged O(v/c) work term rather than a
// symptom of too long a radiation timestep: the work term is lagged from the previous outer iteration while
// dEkin_work is evaluated from the freshly updated momentum, so the two do not cancel exactly, and refining
// dt does not help. Measured on DTypeFront1D, the fraction of radiation cell-updates that reach the cap is
// invariant at ~27% across a fourfold refinement of dt (26.8% / 27.2% / 26.5% at 128 / 256 / 512 cells).
//
// It is nonetheless benign, which is why it is left in place and not reported at runtime. The cap can only
// bind where dEkin_work already exceeds the internal energy available, i.e. in cold cells the radiation has
// evacuated, so Egas there is minuscule and the absolute energy discarded is negligible however many cells
// are involved: DTypeFront1D closes its thermal-band energy budget to 1.0000000 with ~27% of its radiation
// cell-updates capping. Counting them would mean an atomic on one address in a large fraction of every
// radiation kernel, which is not worth paying for in a production GPU run.
//
// See https://github.com/quokka-astro/quokka/issues/2173 for the two candidate fixes.
static constexpr double work_term_min_eint_fraction = 0.1;
static const bool PPL_free_slope_st_total = false; // PPL with free slopes for all, but subject to the constraint sum_g alpha_g B_g = - sum_g B_g. Not working
						   // well -- Newton iteration convergence issue.

// physical constants in CGS units
static constexpr double c_light_cgs_ = C::c_light;	    // cgs
static constexpr double radiation_constant_cgs_ = C::a_rad; // cgs
static constexpr double inf = std::numeric_limits<double>::max();

// enum for opacity_model
enum class OpacityModel {
	single_group = 0, // user-defined opacity for each group, given as a function of density and temperature.
	piecewise_constant_opacity,
	PPL_opacity_fixed_slope_spectrum,
	PPL_opacity_full_spectrum // piecewise power-law opacity model with piecewise power-law fitting to a user-defined opacity function and on-the-fly
				  // piecewise power-law fitting to radiation energy density and flux.
};

// this struct is specialized by the user application code
//
template <typename problem_t> struct RadSystem_Traits {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 0.;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups + 1> radBoundaries = {0., inf};
	static constexpr double beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::single_group;
	static constexpr bool dust_absorption_only = false;
	static constexpr amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> pe_heating_efficiency = {};
};

// this struct is specialized by the user application code
//
template <typename problem_t> struct ISM_Traits {
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
};

// A struct to hold the results of the ComputeRadPressure function.
struct RadPressureResult {
	quokka::valarray<double, 4> F; // components of radiation pressure tensor
	double S;		       // maximum wavespeed for the radiation system
};

// A struct to hold the opacity terms for the radiation-matter energy exchange, containing the following elements:
// kappaE, kappaP, kappaF, delta_nu_kappa_B_at_edge, alpha_P, alpha_E
template <typename problem_t> struct OpacityTerms {
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> kappaE;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> kappaP;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> kappaF;
	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> delta_nu_kappa_B_at_edge; // Delta (nu * kappa * B)
	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> alpha_P;
	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> alpha_E;
};

// The per-cell state one backward-Euler coupling step is solved from. Built by AddSourceTerms once per outer iteration
// (the work term changes between them) and read by every residual evaluation. See radiation_coupling.hpp.
template <typename problem_t> struct CouplingCell {
	static constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
	double Egas0{};				   // gas internal energy at the start of the step
	quokka::valarray<double, nGroups> Erad0{}; // group radiation energies at the start of the step
	quokka::valarray<double, nGroups> Src{};   // external source over the step, radiation side (already scaled by chat/c for thermal bands)
	quokka::valarray<double, nGroups> work{};  // lagged work term over the step, radiation side
	double rho{};
	double dt{};	    // the step
	double tau_scale{}; // dt * chat * lorentz factor: what multiplies rho * kappa to make an optical depth
	double dtK{};	    // dt * K, K = dustGasCoeff * n_H^2 the collisional gas-dust coefficient; 0 without the dust model
	double Tfloor{};    // temperature floor
	double Emin{};	    // gas internal energy at the temperature floor
	amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::numMassScalars> massScalars{};
	amrex::GpuArray<double, nGroups + 1> rad_boundaries{};
	amrex::GpuArray<double, nGroups> rad_boundary_ratios{};
};

// The two coefficients of the coupling at one matter temperature: emission_g = rho kappa_P,g 4 pi B_g / c and
// absorption_g = rho kappa_E,g, with the opacities they were built from.
template <typename problem_t> struct CouplingCoefficients {
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> emission{};
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> absorption{};
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> fourPiBoverC{};
	OpacityTerms<problem_t> opacity{};
};

// The state implied by one trial value of the coupling unknown, and the residual there.
template <typename problem_t> struct CouplingSolution {
	double residual{}; // G (gas-energy unknown) or H (dust-temperature unknown) at this state
	double Egas{};
	double T_gas{};
	double T_d{}; // the temperature the radiation couples at: T_gas without dust, the dust temperature with it
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Erad{};
	int nevals{};	  // residual evaluations spent by the solve
	bool converged{}; // whether the bracket met the tolerance within the iteration budget
};

// The result of the energy exchange of one cell: gas energy, the temperatures, the group energies, the work term used,
// and the opacities at the temperature the radiation coupled at.
template <typename problem_t> struct EnergyExchangeResult {
	double Egas;							      // gas internal energy
	double T_gas;							      // gas temperature
	double T_d;							      // dust temperature
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> EradVec; // radiation energy density
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> work;    // work term
	OpacityTerms<problem_t> opacity_terms;
};

// A struct to hold the results of UpdateFlux(), containing the following elements:
// Erad, gasMomentum, Frad
template <typename problem_t> struct FluxUpdateResult {
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Erad;			   // radiation energy density
	amrex::GpuArray<double, 3> gasMomentum;							   // gas momentum
	amrex::GpuArray<amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nGroups>, 3> Frad; // radiation flux
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod_func(double a, double b) -> double
{
	return 0.5 * (sgn(a) + sgn(b)) * std::min(std::abs(a), std::abs(b));
}

// Use SFINAE (Substitution Failure Is Not An Error) to check if opacity_model is defined in RadSystem_Traits<problem_t>
template <typename problem_t, typename = void> struct RadSystem_Has_Opacity_Model : std::false_type {};

template <typename problem_t>
struct RadSystem_Has_Opacity_Model<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::opacity_model)>> : std::true_type {};

// Use SFINAE to check if dust_absorption_only is defined in RadSystem_Traits<problem_t>
template <typename problem_t, typename = void> struct RadSystem_Has_Dust_Absorption_Only : std::false_type {};

template <typename problem_t>
struct RadSystem_Has_Dust_Absorption_Only<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::dust_absorption_only)>> : std::true_type {};

// Use SFINAE to check if pe_heating_efficiency is defined in RadSystem_Traits<problem_t>
template <typename problem_t, typename = void> struct RadSystem_Has_PE_Heating_Efficiency : std::false_type {};

template <typename problem_t>
struct RadSystem_Has_PE_Heating_Efficiency<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::pe_heating_efficiency)>> : std::true_type {};

// Use SFINAE to check if ChemBands() is defined in RadSystem_Traits<problem_t> (indicates photoionization group)
template <typename problem_t, typename = void> struct RadSystem_Has_ChemBands : std::false_type {};

template <typename problem_t> struct RadSystem_Has_ChemBands<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::ChemBands())>> : std::true_type {};

// Get NChemBands (number of chemistry frequency bands) from RadSystem_Traits<problem_t>.
// Returns 0 if ChemBands() is not defined (no photoionization groups).
template <typename problem_t, typename = void> struct RadSystem_NChemBands {
	static constexpr int value = 0;
};

template <typename problem_t> struct RadSystem_NChemBands<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::ChemBands())>> {
	static constexpr int value = static_cast<int>(decltype(RadSystem_Traits<problem_t>::ChemBands())::size()) - 1;
};

template <typename problem_t, typename = void> struct RadSystem_EnergyUnit {
	static constexpr double value = C::ev2erg;
};

template <typename problem_t> struct RadSystem_EnergyUnit<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::energy_unit)>> {
	static constexpr double value = RadSystem_Traits<problem_t>::energy_unit;
};

/// Class for the radiation moment equations
///
template <typename problem_t> class RadSystem : public HyperbolicSystem<problem_t>
{
      public:
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(0.5 * std::abs(a + b), std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr int numRadVars_ = Physics_NumVars::numRadVarsPerGroup;			 // number of radiation variables for each photon group
	static constexpr int nvarHyperbolic_ = numRadVars_ * Physics_Traits<problem_t>::nGroups; // total number of radiation variables
	static constexpr int nstartHyperbolic_ = Physics_Indices<problem_t>::radFirstIndex;
	static constexpr int nvar_ = nstartHyperbolic_ + nvarHyperbolic_;

	enum gasVarIndex {
		gasDensity_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1GasMomentum_index,
		x2GasMomentum_index,
		x3GasMomentum_index,
		gasEnergy_index,
		gasInternalEnergy_index,
		scalar0_index
	};

	enum radVarIndex { radEnergy_index = nstartHyperbolic_, x1RadFlux_index, x2RadFlux_index, x3RadFlux_index };

	enum primVarIndex {
		primRadEnergy_index = 0,
		x1ReducedFlux_index,
		x2ReducedFlux_index,
		x3ReducedFlux_index,
	};

	// C++ standard does not allow constexpr to be uninitialized, even in a
	// templated class!

	static constexpr amrex::Real c_light_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return c_light_cgs_;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::c_light;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// c / c_bar = u_l / u_t
			return c_light_cgs_ / (Physics_Traits<problem_t>::unit_length / Physics_Traits<problem_t>::unit_time);
		}
	}();
	static constexpr double c_hat_ = c_light_ * RadSystem_Traits<problem_t>::c_hat_over_c;

	static constexpr double radiation_constant_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return C::a_rad;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::radiation_constant;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// a_rad / a_rad_bar = 1 / u_l * u_m / u_t^2 / u_T^4
			return C::a_rad / (1.0 / Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
					   (Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) /
					   (Physics_Traits<problem_t>::unit_temperature * Physics_Traits<problem_t>::unit_temperature *
					    Physics_Traits<problem_t>::unit_temperature * Physics_Traits<problem_t>::unit_temperature));
		}
	}();

	static constexpr int beta_order_ = RadSystem_Traits<problem_t>::beta_order;

	static constexpr bool enable_dust_gas_thermal_coupling_model_ = ISM_Traits<problem_t>::enable_dust_gas_thermal_coupling_model;

	static constexpr int nGroups_ = Physics_Traits<problem_t>::nGroups;
	// Chemical (ionizing) bands occupy the LAST NChemBands groups; the leading nGroupsThermal_ groups
	// are thermal. The thermal radiation update (blackbody emission + gas-radiation energy exchange)
	// acts only on the thermal groups; chemical bands are handled by transport, direct source injection,
	// and photochemistry. When NChemBands == 0, nGroupsThermal_ == nGroups_ and all chem-specific code
	// paths are inert.
	static constexpr int nGroupsThermal_ = nGroups_ - RadSystem_NChemBands<problem_t>::value;

	// When true, every thermal group is instead a dust-absorption-only band: it is transported, absorbed
	// by dust, and exerts radiation force on the gas, but emits nothing and gives none of the absorbed
	// energy to the gas. The absorbed energy leaves the simulation -- physically it heats the dust, which
	// re-radiates it in the infrared, and neither the dust temperature nor that emission is followed, so
	// total energy is deliberately not conserved. Gas heating (photoelectric, photodissociation) is left
	// to an external chemistry and cooling module, which needs the radiation field rather than the
	// absorbed energy. The motivating case is FUV + Lyman-Werner in a galaxy simulation.
	static constexpr bool dust_absorption_only_ = []() constexpr {
		if constexpr (RadSystem_Has_Dust_Absorption_Only<problem_t>::value) {
			return RadSystem_Traits<problem_t>::dust_absorption_only;
		} else {
			return false;
		}
	}();

	// Number of groups that emit blackbody radiation. Dust-absorption-only bands and chemical bands do
	// not, so this is the range over which the Planck emission and its temperature derivative are built.
	static constexpr int nGroupsEmitting_ = dust_absorption_only_ ? 0 : nGroupsThermal_;

	// Rate coefficient of the photoelectric heating, in cgs units, from Bate & Keto (2015), Eq. 26: the
	// standard heating rate 1.33e-24 erg s^-1 per hydrogen nucleus per unit Habing field, divided by the
	// reference interstellar radiation field energy density 5.29e-14 erg cm^-3 that defines that field.
	// Units: cm^3 s^-1, so that (coefficient * n_H * E_g) is an energy density per unit time.
	static constexpr double pe_heating_rate_coeff_ = 1.33e-24 / 5.29e-14;

	// Photoelectric efficiency of each dust-absorption band -- the dimensionless factor epsilon of the
	// standard interstellar expression, about 0.05 for cold molecular gas. The heating rate is
	//
	//     Gamma_PE = sum_g pe_heating_efficiency_[g] * pe_heating_rate_coeff_ * n_H * E_g .
	//
	// Note what this does NOT depend on: the dust opacity of the band. Photoelectric heating is the
	// photoelectric effect on grains, and the grain physics is folded into the empirical coefficient
	// above rather than taken from kappa. A band with zero opacity therefore still heats the gas if its
	// efficiency is non-zero. A consequence is that this heating is not bounded by, and is not debited
	// from, the energy the band absorbs: it adds energy to the gas that the radiation does not lose.
	//
	// A zero entry means the band drives no photoelectric heating, which is how non-ultraviolet bands are
	// labelled. Because the expression is linear in E_g, splitting one band into two and giving both the
	// same efficiency reproduces the unsplit result exactly.
	//
	// This is a compile-time constant, so it cannot depend on the local electron density or grain charge.
	// It applies to dust-absorption bands only; thermal bands have no photoelectric heating.
	static constexpr amrex::GpuArray<double, nGroups_> pe_heating_efficiency_ = []() constexpr {
		if constexpr (RadSystem_Has_PE_Heating_Efficiency<problem_t>::value) {
			return RadSystem_Traits<problem_t>::pe_heating_efficiency;
		} else {
			// value-initialization zeroes the aggregate; GpuArray::operator[] is not constexpr, so it
			// cannot be filled element by element here
			return amrex::GpuArray<double, nGroups_>{};
		}
	}();

	// True when any band has a non-zero photoelectric efficiency. Used to guard against double-counting the
	// photoelectric heating against an external cooling module.
	// GpuArray::operator[] is not constexpr, so the compile-time scans below go through the underlying
	// array member instead.
	static constexpr bool enable_dust_pe_heating_ = []() constexpr {
		for (int g = 0; g < nGroups_; ++g) {
			if (pe_heating_efficiency_.arr[g] > 0.0) {
				return true;
			}
		}
		return false;
	}();
	static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries_ = []() constexpr {
		if constexpr (nGroups_ > 1) {
			return RadSystem_Traits<problem_t>::radBoundaries;
		} else {
			amrex::GpuArray<double, 2> boundaries{0., inf};
			return boundaries;
		}
	}();

	static constexpr double Erad_floor_ = RadSystem_Traits<problem_t>::Erad_floor / nGroups_;

	static constexpr OpacityModel opacity_model_ = []() constexpr {
		if constexpr (RadSystem_Has_Opacity_Model<problem_t>::value) {
			return RadSystem_Traits<problem_t>::opacity_model;
		} else {
			return OpacityModel::single_group;
		}
	}();

	// Assertion: has to use single_group when nGroups_ == 1
	static_assert(((nGroups_ > 1 && opacity_model_ != OpacityModel::single_group) || (nGroups_ == 1 && opacity_model_ == OpacityModel::single_group)),
		      "OpacityModel::single_group MUST be used when nGroups_ == 1. If nGroups_ > 1, you MUST set opacity_model."); // NOLINT

	// Assertion: PPL_opacity_full_spectrum requires at least 3 photon groups
	static_assert(!(nGroups_ < 3 && opacity_model_ == OpacityModel::PPL_opacity_full_spectrum), // NOLINT
		      "PPL_opacity_full_spectrum requires at least 3 photon groups.");

	// Assertion: chemical (ionizing) bands, when present, occupy the last NChemBands groups; the
	// leading (nGroups_ - NChemBands) groups are thermal. Mixed thermal+chemical configurations are
	// therefore valid as long as there are no more chemical bands than groups.
	static_assert(RadSystem_NChemBands<problem_t>::value >= 0 && RadSystem_NChemBands<problem_t>::value <= nGroups_,
		      "The number of chemical radiation bands must be between 0 and the number of radiation groups.");

#ifdef PHOTOCHEMISTRY
	static_assert(RadSystem_EnergyUnit<problem_t>::value == C::ev2erg,
		      "ChemBands() is interpreted as eV by GetChemBandQuanta(); energy_unit must be C::ev2erg when PHOTOCHEMISTRY is enabled.");
#endif

	// Assertions: dust_absorption_only turns off thermal emission and the gas-radiation energy exchange
	// for every thermal group, so it is incompatible with the models that rely on either.
	static_assert(!(dust_absorption_only_ && nGroups_ == 1), // NOLINT
		      "dust_absorption_only is implemented for multigroup radiation only; it requires nGroups > 1.");

	static_assert(!(dust_absorption_only_ && enable_dust_gas_thermal_coupling_model_), // NOLINT
		      "dust_absorption_only assumes the dust is thermally decoupled from the gas, so it cannot be combined with "
		      "ISM_Traits::enable_dust_gas_thermal_coupling_model.");

	// Assertion: pe_heating_efficiency is the dimensionless efficiency factor epsilon of the standard
	// interstellar expression, a fraction between 0 and 1 (about 0.05 for cold molecular gas).
	static_assert(
	    []() constexpr {
		    for (int g = 0; g < nGroups_; ++g) {
			    if (!((pe_heating_efficiency_.arr[g] >= 0.0) && (pe_heating_efficiency_.arr[g] <= 1.0))) {
				    return false;
			    }
		    }
		    return true;
	    }(), // NOLINT
	    "Each entry of RadSystem_Traits::pe_heating_efficiency must lie between 0 and 1.");

	// Assertion: photoelectric heating from the radiation field is implemented for dust-absorption bands
	// only.
	static_assert(!(enable_dust_pe_heating_ && !dust_absorption_only_), // NOLINT
		      "RadSystem_Traits::pe_heating_efficiency applies to dust-absorption bands, so it requires dust_absorption_only = true.");

	// Assertion: pe_heating_rate_coeff_ is an empirical constant in cgs units, so the photoelectric
	// heating is only meaningful for a problem whose units are cgs. Rejecting the other unit systems
	// outright is better than silently applying a cgs number to dimensionless quantities.
	static_assert(!(enable_dust_pe_heating_ && (Physics_Traits<problem_t>::unit_system != UnitSystem::CGS)), // NOLINT
		      "RadSystem_Traits::pe_heating_efficiency uses an empirical rate coefficient in cgs units, so it requires "
		      "Physics_Traits::unit_system == UnitSystem::CGS.");

	static constexpr double mean_molecular_mass_ = ::quokka::EOS_Traits<problem_t>::mean_molecular_weight;
	static constexpr double gamma_ = ::quokka::EOS_Traits<problem_t>::gamma;

	static constexpr amrex::Real boltzmann_constant_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return C::k_B;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::boltzmann_constant;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// k_B / k_B_bar = u_l^2 * u_m / u_t^2 / u_T
			return C::k_B /
			       (Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
				(Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) / Physics_Traits<problem_t>::unit_temperature);
		}
	}();

	// static functions

#ifdef PHOTOCHEMISTRY
	AMREX_GPU_HOST_DEVICE static auto GetChemBandQuanta(int group_index) -> amrex::Real;
#endif

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange);
	static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange);

	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);

	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, double alpha, double Aex_s1_coeff,
				 double Aex_s2_coeff);

	template <FluxDir DIR>
	static void ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
				  amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
				  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool use_wavespeed_correction,
				  std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc = {});

	//! Set the user-defined radiation source terms.
	//!
	//! Both buffers belong to this hook alone and are zeroed before every call, so assign to them; the
	//! framework merges the result into any radiation that particles have already deposited.
	//!
	//! \param radEnergySource luminosity volume density of group g, in component g; unit: erg s^-1 cm^-3.
	//! \param reducedFluxSource reduced flux f = F / (c E) of the injected radiation of group g along direction n,
	//!			   in component 3 * g + n; dimensionless, and physical only if |f| <= 1 (asserted in a debug
	//!			   build). The deposited flux source is c * f * radEnergySource, so c is always the runtime
	//!			   speed of light and the injected radiation satisfies F = f c E by construction. f = 0 (the
	//!			   default) injects isotropic radiation; |f| = 1 injects fully beamed (free-streaming) radiation.
	//!			   Asking for the reduced flux rather than the flux itself makes |F| > c E unrepresentable.
	static void AddRadSource(array_t &radEnergySource, array_t &reducedFluxSource, amrex::Box const &indexRange,
				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time);

	//! Merge the source written by AddRadSource into the source that is actually deposited.
	//!
	//! Radiation from particles has already been deposited into radEnergySource, so the user source is
	//! collected in its own buffers and added here. The user gives a reduced flux, which is converted to
	//! the flux source c * f * E that the solver consumes; unit: erg cm^-2 s^-2.
	static void MergeUserRadSource(array_t &radEnergySource, array_t &radFluxSource, arrayconst_t &userEnergySource, arrayconst_t &userReducedFlux,
				       amrex::Box const &indexRange);

	AMREX_GPU_DEVICE static auto UpdateFlux(int i, int j, int k, arrayconst_t const &consPrev, EnergyExchangeResult<problem_t> &energy,
						CouplingCell<problem_t> const &cell, double gas_update_factor, double Ekin0,
						amrex::GpuArray<quokka::valarray<double, nGroups_>, 3> const &Src_flux, double Emag,
						amrex::GpuArray<double, 3> const &lorentz) -> FluxUpdateResult<problem_t>;

	static void AddSourceTerms(array_t &consVar, arrayconst_t &radEnergySource, arrayconst_t &radFluxSource, amrex::Box const &indexRange,
				   amrex::Real dt_implicit, double gas_update_factor, double dustGasCoeff, double tol, double tempFloor,
				   int *p_iteration_counter, int *p_iteration_failure_counter,
				   std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc = {});

	static void balanceMatterRadiation(arrayconst_t &consPrev, array_t &consNew, amrex::Box const &indexRange);

	// Use an additionalr template for ComputeMassScalars as the Array type is not always the same
	template <typename ArrayType>
	AMREX_GPU_DEVICE static auto ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeEddingtonFactor(double f) -> double;

	AMREX_GPU_HOST_DEVICE static auto ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const &massScalars) -> double;

	// Used for single-group RHD only. Not used for multi-group RHD.
	AMREX_GPU_HOST_DEVICE static auto ComputePlanckOpacity(double rho, double Tgas) -> Real;
	AMREX_GPU_HOST_DEVICE static auto ComputeFluxMeanOpacity(double rho, double Tgas) -> Real;
	AMREX_GPU_HOST_DEVICE static auto ComputeEnergyMeanOpacity(double rho, double Tgas) -> Real;

	// For multi-group RHD, use DefineOpacityExponentsAndLowerValues to define the opacities.
	AMREX_GPU_HOST_DEVICE static auto DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double rho, double Tgas)
	    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>;

	AMREX_GPU_HOST_DEVICE static auto ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
								  amrex::GpuArray<double, nGroups_> const &radBoundaryRatios,
								  amrex::GpuArray<double, nGroups_> const &alpha_quant) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
								  amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value)
	    -> quokka::valarray<double, nGroups_>;
	// AMREX_GPU_HOST_DEVICE static auto
	// ComputeGroupMeanOpacityWithMinusOneSlope(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value,
	// 					 amrex::GpuArray<double, nGroups_> radBoundaryRatios) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto PlanckFunction(double nu, double T) -> double;
	AMREX_GPU_HOST_DEVICE static auto
	ComputeDiffusionFluxMeanOpacity(quokka::valarray<double, nGroups_> kappaPVec, quokka::valarray<double, nGroups_> kappaEVec,
					quokka::valarray<double, nGroups_> fourPiBoverC, amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
					amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge, amrex::GpuArray<double, nGroups_ + 1> kappa_slope)
	    -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeFluxInDiffusionLimit(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double T, double vel)
	    -> amrex::GpuArray<double, nGroups_>;

	template <typename ArrayType>
	AMREX_GPU_HOST_DEVICE static auto ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
	    -> amrex::GpuArray<double, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto Solve3x3matrix(double C00, double C01, double C02, double C10, double C11, double C12, double C20, double C21,
							 double C22, double Y0, double Y1, double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>;

	AMREX_GPU_HOST_DEVICE static auto ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature)
	    -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> double;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
	    -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_DEVICE static void ComputeModelDependentKappaFAndDeltaTerms(double T, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
									      quokka::valarray<double, nGroups_> const &fourPiBoverC,
									      OpacityTerms<problem_t> &opacity_terms);

	AMREX_GPU_DEVICE static auto ComputeModelDependentKappaEAndKappaP(double T, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
									  amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
									  quokka::valarray<double, nGroups_> const &fourPiBoverC,
									  quokka::valarray<double, nGroups_> const &Erad) -> OpacityTerms<problem_t>;

	// --- the bracketed coupling solve (radiation_coupling.hpp) ---
	static constexpr int max_root_iterations_ = 100;

	AMREX_GPU_DEVICE static auto TgasOf(CouplingCell<problem_t> const &cell, double Egas) -> double;
	AMREX_GPU_DEVICE static auto TotalEnergy(CouplingCell<problem_t> const &cell) -> double;
	AMREX_GPU_DEVICE static auto ComputeCouplingCoefficients(CouplingCell<problem_t> const &cell, double T) -> CouplingCoefficients<problem_t>;
	AMREX_GPU_DEVICE static auto GroupEnergies(CouplingCell<problem_t> const &cell, CouplingCoefficients<problem_t> const &coef)
	    -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_DEVICE static auto GasCouplingState(CouplingCell<problem_t> const &cell, double Egas) -> CouplingSolution<problem_t>;
	AMREX_GPU_DEVICE static auto DustCouplingState(CouplingCell<problem_t> const &cell, double T_d) -> CouplingSolution<problem_t>;
	AMREX_GPU_DEVICE static void ApplyEnergyFloors(CouplingCell<problem_t> const &cell, CouplingSolution<problem_t> &sol);
	AMREX_GPU_DEVICE static auto SolveGasCoupling(CouplingCell<problem_t> const &cell, double tol) -> CouplingSolution<problem_t>;
	AMREX_GPU_DEVICE static auto SolveDustCoupling(CouplingCell<problem_t> const &cell, double tol) -> CouplingSolution<problem_t>;

	// --- the source-term update (source_terms.hpp) ---
	AMREX_GPU_DEVICE static auto ComputeLorentzFactors(double rho, std::array<double, 3> const &gasMtm) -> amrex::GpuArray<double, 3>;
	AMREX_GPU_DEVICE static auto ComputeOpacityTermsAt(CouplingCell<problem_t> const &cell, double T, quokka::valarray<double, nGroups_> const &Erad)
	    -> OpacityTerms<problem_t>;
	AMREX_GPU_DEVICE static auto ComputeWorkTerm(CouplingCell<problem_t> const &cell, double T, OpacityTerms<problem_t> const &opacity,
						     quokka::valarray<double, nGroups_> const &vel_times_F, double lorentz_v)
	    -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_DEVICE static auto SolveEnergyExchange(CouplingCell<problem_t> const &cell, double tol, int *p_iteration_counter,
							 int *p_iteration_failure_counter) -> EnergyExchangeResult<problem_t>;
	AMREX_GPU_DEVICE static auto SolveDustAbsorptionBands(CouplingCell<problem_t> const &cell, int *p_iteration_counter) -> EnergyExchangeResult<problem_t>;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
							     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k, int i_phys, int j_phys,
							     int k_phys, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc,
							     const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
	    -> quokka::valarray<double, nGroups_>;

	AMREX_GPU_DEVICE static auto isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool;

	AMREX_GPU_DEVICE static void amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons);

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeRadPressure(double erad_L, double Fx_L, double Fy_L, double Fz_L, double fx_L, double fy_L, double fz_L)
	    -> RadPressureResult;

	AMREX_GPU_DEVICE static auto ComputeEddingtonTensor(double fx_L, double fy_L, double fz_L) -> std::array<std::array<double, 3>, 3>;
};

// Compute radiation energy fractions for each photon group from a Planck function, given nGroups, radBoundaries, and temperature
// This function enforces that the total fraction is 1.0, no matter what are the group boundaries
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<amrex::Real, nGroups_> radEnergyFractions{};
	if constexpr (nGroups_ == 1) {
		radEnergyFractions[0] = 1.0;
		return radEnergyFractions;
	} else {
		amrex::Real const energy_unit_over_kT = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * temperature);
		amrex::Real y = NAN;
		amrex::Real previous = 0.0;
		// Only the emitting groups (the leading nGroupsEmitting_ groups) receive blackbody emission. When
		// chemical bands are present the thermal fractions are NOT renormalized: the blackbody radiation
		// above the first chemical-band boundary is simply dropped, so the fractions sum to < 1. Under
		// dust_absorption_only there are no emitting groups at all and every fraction is left at 0.
		for (int g = 0; g < nGroupsEmitting_; ++g) {
			if (g == nGroups_ - 1) {
				// no chemical bands: the last group carries all remaining blackbody, total fraction = 1.0
				y = 1.0;
			} else {
				const amrex::Real x = boundaries[g + 1] * energy_unit_over_kT;
				if (x >= 100.) { // 100. is the upper limit of x in the table
					y = 1.0;
				} else {
					y = integrate_planck_from_0_to_x(x);
				}
			}
			radEnergyFractions[g] = y - previous;
			previous = y;
		}
		// non-emitting bands (g >= nGroupsEmitting_) emit no blackbody radiation; left at 0.
		AMREX_ASSERT(sum(radEnergyFractions) < 1.0 + 1.0e-10);

		return radEnergyFractions;
	}
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const & /*massScalars*/) -> double
{
	return rho / mean_molecular_mass_;
}

// define ComputeThermalRadiation for single-group, returns the thermal radiation power = a_r * T^4
template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> Real
{
	double power = radiation_constant_ * std::pow(temperature, 4);
	// set floor
	if (power < Erad_floor_) {
		power = Erad_floor_;
	}
	return power;
}

// define ComputeThermalRadiationMultiGroup, returns the thermal radiation power for each photon group. = a_r * T^4 * radEnergyFractions
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationMultiGroup(amrex::Real temperature,
										   amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	const double power = radiation_constant_ * std::pow(temperature, 4);
	const auto radEnergyFractions = ComputePlanckEnergyFractions(boundaries, temperature);
	auto Erad_g = power * radEnergyFractions;
	// set floor on the emitting groups only; the other bands emit no blackbody radiation and are left at 0.
	for (int g = 0; g < nGroupsEmitting_; ++g) {
		if (Erad_g[g] < Erad_floor_) {
			Erad_g[g] = Erad_floor_;
		}
	}
	return Erad_g;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::Solve3x3matrix(const double C00, const double C01, const double C02, const double C10, const double C11,
								const double C12, const double C20, const double C21, const double C22, const double Y0,
								const double Y1, const double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>
{
	// Solve the 3x3 matrix equation: C * X = Y under the assumption that only the diagonal terms
	// are guaranteed to be non-zero and are thus allowed to be divided by.

	auto E11 = C11 - C01 * C10 / C00;
	auto E12 = C12 - C02 * C10 / C00;
	auto E21 = C21 - C01 * C20 / C00;
	auto E22 = C22 - C02 * C20 / C00;
	auto Z1 = Y1 - Y0 * C10 / C00;
	auto Z2 = Y2 - Y0 * C20 / C00;
	auto X2 = (Z2 - Z1 * E21 / E11) / (E22 - E12 * E21 / E11);
	auto X1 = (Z1 - E12 * X2) / E11;
	auto X0 = (Y0 - C01 * X1 - C02 * X2) / C00;

	return std::make_tuple(X0, X1, X2);
}

template <typename problem_t>
void RadSystem<problem_t>::AddRadSource(array_t &radEnergySource, array_t &reducedFluxSource, amrex::Box const &indexRange,
					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time)
{
	// Default implementation: no radiation source is added.
	// Users should override this method to set their own radiation source. Both buffers belong to this hook
	// alone and are zeroed before every call, so simply assign to them; the framework merges the result into
	// the source that particles have already deposited into.
	// This function is intentionally left blank.
}

template <typename problem_t>
void RadSystem<problem_t>::MergeUserRadSource(array_t &radEnergySource, array_t &radFluxSource, arrayconst_t &userEnergySource, arrayconst_t &userReducedFlux,
					      amrex::Box const &indexRange)
{
	const double c = c_light_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int g = 0; g < nGroups_; ++g) {
			const double Euser = userEnergySource(i, j, k, g);
			const double fx = userReducedFlux(i, j, k, 3 * g + 0);
			const double fy = userReducedFlux(i, j, k, 3 * g + 1);
			const double fz = userReducedFlux(i, j, k, 3 * g + 2);

			// Each contribution must be physical on its own. The set {(E, F) : E >= 0, |F| <= c E} is a
			// convex cone, so the sum of physical contributions is physical by the triangle inequality:
			// |F1 + F2| <= |F1| + |F2| <= c (E1 + E2). No flux limiter is needed here.
			AMREX_ASSERT(Euser >= 0.0);
			AMREX_ASSERT(fx * fx + fy * fy + fz * fz <= 1.0 + 1.0e-10);

			radEnergySource(i, j, k, g) += Euser;
			radFluxSource(i, j, k, 3 * g + 0) += c * fx * Euser;
			radFluxSource(i, j, k, 3 * g + 1) += c * fy * Euser;
			radFluxSource(i, j, k, 3 * g + 2) += c * fz * Euser;
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange)
{
	// keep radiation energy density as-is
	// convert (Fx,Fy,Fz) into reduced flux components (fx,fy,fx):
	//   F_x -> F_x / (c*E_r)

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// add reduced fluxes for each radiation group
		for (int g = 0; g < nGroups_; ++g) {
			const auto E_r = cons(i, j, k, radEnergy_index + numRadVars_ * g);
			const auto Fx = cons(i, j, k, x1RadFlux_index + numRadVars_ * g);
			const auto Fy = cons(i, j, k, x2RadFlux_index + numRadVars_ * g);
			const auto Fz = cons(i, j, k, x3RadFlux_index + numRadVars_ * g);

			// check admissibility of states
			AMREX_ASSERT(E_r > 0.0); // NOLINT

			primVar(i, j, k, primRadEnergy_index + numRadVars_ * g) = E_r;
			primVar(i, j, k, x1ReducedFlux_index + numRadVars_ * g) = Fx / (c_light_ * E_r);
			primVar(i, j, k, x2ReducedFlux_index + numRadVars_ * g) = Fy / (c_light_ * E_r);
			primVar(i, j, k, x3ReducedFlux_index + numRadVars_ * g) = Fz / (c_light_ * E_r);
		}
	});
}

#ifdef PHOTOCHEMISTRY
template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::GetChemBandQuanta(int group_index) -> amrex::Real
{
	// ChemBands() is in eV (jaff's native unit for radiation band edges);
	// convert to erg here rather than have every problem's CMakeLists
	// convert to Hz by hand.
	auto const ev_bounds = RadSystem_Traits<problem_t>::ChemBands();
	amrex::Real const ev_low = ev_bounds[group_index];
	amrex::Real const ev_high = ev_bounds[group_index + 1];

	amrex::Real const alpha = RadSystem_Traits<problem_t>::ChemBandsPowerLawIndex();

	amrex::Real ev_avg = NAN;
	if (std::isinf(ev_high)) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(alpha < 0.0, "GetChemBandQuanta: an open-topped chemistry band only has a "
							      "finite average photon energy for power_law_index < 0");
		ev_avg = ev_low * (1.0 - 1.0 / alpha);
	} else if (alpha == 0.0) {
		ev_avg = ev_low * ev_high * std::log(ev_high / ev_low) / (ev_high - ev_low);
	} else if (alpha == 1.0) {
		ev_avg = (ev_high - ev_low) / std::log(ev_high / ev_low);
	} else {
		ev_avg = ((alpha - 1.0) / alpha) * (std::pow(ev_high, alpha) - std::pow(ev_low, alpha)) /
			 (std::pow(ev_high, alpha - 1.0) - std::pow(ev_low, alpha - 1.0));
	}

	return ev_avg * C::ev2erg;
}
#endif

template <typename problem_t>
void RadSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const & /*cons*/, array_t &maxSignal, amrex::Box const &indexRange)
{
	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = c_hat_;
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool
{
	// check if the state variable 'cons' is a valid state
	bool isValid = true;
	for (int g = 0; g < nGroups_; ++g) {
		const auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];

		const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
		const auto f = Fnorm / (c_light_ * E_r);

		bool isNonNegative = (E_r > 0.);
		bool isFluxCausal = (f <= 1.);
		isValid = (isValid && isNonNegative && isFluxCausal);
	}
	return isValid;
}

template <typename problem_t> AMREX_GPU_DEVICE void RadSystem<problem_t>::amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons)
{
	constexpr amrex::Real small_number = 1.0e20 * std::numeric_limits<amrex::Real>::min();
	constexpr amrex::Real smaller_than_one = 1.0 - 20.0 * std::numeric_limits<amrex::Real>::epsilon();

	// amend the state variable 'cons' to be a valid state
	for (int g = 0; g < nGroups_; ++g) {
		auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		// If E_r is NaN or below floor, set to floor
		if (E_r < Erad_floor_) {
			cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_] = Erad_floor_;
			cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = 0.0;
			cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = 0.0;
			cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = 0.0;
			continue;
		}
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		if (Fx * Fx + Fy * Fy + Fz * Fz > (c_light_ * c_light_) * (E_r * E_r) * smaller_than_one) {
			const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
			// If Fnorm is NaN or very close to zero, set fluxes to zero
			if (Fnorm < small_number) {
				cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = 0.0;
				cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = 0.0;
				cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = 0.0;
			} else {
				cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = (Fx / Fnorm) * c_light_ * E_r * smaller_than_one;
				cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = (Fy / Fnorm) * c_light_ * E_r * smaller_than_one;
				cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = (Fz / Fnorm) * c_light_ * E_r * smaller_than_one;
			}
		}
	}
}

template <typename problem_t>
void RadSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				       amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1Flux = fluxArray[0];
	// const auto x1FluxDiffusive = fluxDiffusiveArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2Flux = fluxArray[1];
	// const auto x2FluxDiffusive = fluxDiffusiveArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3Flux = fluxArray[2];
	// const auto x3FluxDiffusive = fluxDiffusiveArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons{};

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			cons[n] = consVarOld(i, j, k, nstartHyperbolic_ + n) + (AMREX_D_TERM((dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
											     +(dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
											     +(dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))));
		}

		if (!isStateValid(cons)) {
			amendRadState(cons);
		}
		AMREX_ASSERT(isStateValid(cons));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			consVarNew(i, j, k, nstartHyperbolic_ + n) = cons[n];
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArrayOld*/,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/,
					const double alpha, const double Aex_s1_coeff, const double Aex_s2_coeff)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1FluxOld = fluxArrayOld[0];
	const auto x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2FluxOld = fluxArrayOld[1];
	const auto x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3FluxOld = fluxArrayOld[2];
	const auto x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons_new{};

		// Shu-Osher form: y^(3)* = (1-alpha)*y^n + alpha*y^(2) + dt*Aex_s1_coeff*s(y^n) + dt*Aex_s2_coeff*s(y^(2))
		// where alpha = Aim_32/Aim_22, Aex_s1_coeff = Aex_31 - alpha*Aex_21, Aex_s2_coeff = Aex_32
		// The implicit term dt*Aim_33*g(y^(3)) is handled separately in subcycleRadiationAtLevel.
		for (int n = 0; n < nvarHyperbolic_; ++n) {
			const double U_0 = U0(i, j, k, nstartHyperbolic_ + n);
			const double U_1 = U1(i, j, k, nstartHyperbolic_ + n);
			const double FxU_0 = (dt / dx) * (x1FluxOld(i, j, k, n) - x1FluxOld(i + 1, j, k, n));
			const double FxU_1 = (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n));
#if (AMREX_SPACEDIM >= 2)
			const double FyU_0 = (dt / dy) * (x2FluxOld(i, j, k, n) - x2FluxOld(i, j + 1, k, n));
			const double FyU_1 = (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n));
#endif
#if (AMREX_SPACEDIM == 3)
			const double FzU_0 = (dt / dz) * (x3FluxOld(i, j, k, n) - x3FluxOld(i, j, k + 1, n));
			const double FzU_1 = (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n));
#endif
			// save results in cons_new
			cons_new[n] = (1.0 - alpha) * U_0 + alpha * U_1 + (Aex_s1_coeff * (AMREX_D_TERM(FxU_0, +FyU_0, +FzU_0))) +
				      (Aex_s2_coeff * (AMREX_D_TERM(FxU_1, +FyU_1, +FzU_1)));
		}

		if (!isStateValid(cons_new)) {
			amendRadState(cons_new);
		}
		AMREX_ASSERT(isStateValid(cons_new));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			U_new(i, j, k, nstartHyperbolic_ + n) = cons_new[n];
		}
	});
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEddingtonFactor(double f_in) -> double
{
	// f is the reduced flux == |F|/cE.
	// compute Levermore (1984) closure [Eq. 25]
	// the is the M1 closure that is derived from Lorentz invariance
	const double f = clamp(f_in, 0., 1.); // restrict f to be within [0, 1]
	const double f_fac = std::sqrt(4.0 - 3.0 * (f * f));
	const double chi = (3.0 + 4.0 * (f * f)) / (5.0 + 2.0 * f_fac);

#if 0 // NOLINT
      // compute Minerbo (1978) closure [piecewise approximation]
      // (For unknown reasons, this closure tends to work better
      // than the Levermore/Lorentz closure on the Su & Olson 1997 test.)
	const double chi = (f < 1. / 3.) ? (1. / 3.) : (0.5 - f + 1.5 * f*f);
#endif

	return chi;
}

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>
{
	amrex::GpuArray<Real, nmscalars_> massScalars{};
	for (int n = 0; n < nmscalars_; ++n) {
		massScalars[n] = arr(i, j, k, scalar0_index + n);
	}
	return massScalars;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k, int i_phys,
								    int j_phys, int k_phys,
								    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc,
								    const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
    -> quokka::valarray<double, nGroups_>
{
	// compute interface-averaged cell optical depth

	// [By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.]

	// piecewise-constant reconstruction
	const double rho_L = consVar(i - 1, j, k, gasDensity_index);
	const double rho_R = consVar(i, j, k, gasDensity_index);

	const double x1GasMom_L = consVar(i - 1, j, k, x1GasMomentum_index);
	const double x1GasMom_R = consVar(i, j, k, x1GasMomentum_index);

	const double x2GasMom_L = consVar(i - 1, j, k, x2GasMomentum_index);
	const double x2GasMom_R = consVar(i, j, k, x2GasMomentum_index);

	const double x3GasMom_L = consVar(i - 1, j, k, x3GasMomentum_index);
	const double x3GasMom_R = consVar(i, j, k, x3GasMomentum_index);

	const double Egas_L = consVar(i - 1, j, k, gasEnergy_index);
	const double Egas_R = consVar(i, j, k, gasEnergy_index);

	auto massScalars_L = RadSystem<problem_t>::ComputeMassScalars(consVar, i - 1, j, k);
	auto massScalars_R = RadSystem<problem_t>::ComputeMassScalars(consVar, i, j, k);

	double Eint_L = NAN;
	double Eint_R = NAN;
	double Tgas_L = NAN;
	double Tgas_R = NAN;

	if constexpr (gamma_ != 1.0) {
		double Emag_L = 0.0;
		double Emag_R = 0.0;
		if constexpr (DIR == FluxDir::X1) {
			Emag_L = ComputeCellCenteredMagneticEnergy<problem_t>(i_phys - 1, j_phys, k_phys, cons_fc);
		} else if constexpr (DIR == FluxDir::X2) {
			Emag_L = ComputeCellCenteredMagneticEnergy<problem_t>(i_phys, j_phys - 1, k_phys, cons_fc);
		} else {
			Emag_L = ComputeCellCenteredMagneticEnergy<problem_t>(i_phys, j_phys, k_phys - 1, cons_fc);
		}
		Emag_R = ComputeCellCenteredMagneticEnergy<problem_t>(i_phys, j_phys, k_phys, cons_fc);
		Eint_L = ::quokka::EOS<problem_t>::ComputeEintFromEgas(rho_L, x1GasMom_L, x2GasMom_L, x3GasMom_L, Egas_L, Emag_L);
		Eint_R = ::quokka::EOS<problem_t>::ComputeEintFromEgas(rho_R, x1GasMom_R, x2GasMom_R, x3GasMom_R, Egas_R, Emag_R);
		Tgas_L = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho_L, Eint_L, massScalars_L);
		Tgas_R = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho_R, Eint_R, massScalars_R);
	}

	double dl = NAN;
	if constexpr (DIR == FluxDir::X1) {
		dl = dx[0];
	} else if constexpr (DIR == FluxDir::X2) {
		dl = dx[1];
	} else if constexpr (DIR == FluxDir::X3) {
		dl = dx[2];
	}

	quokka::valarray<double, nGroups_> optical_depths{};
	if constexpr (nGroups_ == 1) {
		const double tau_L = dl * rho_L * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_L, Tgas_L);
		const double tau_R = dl * rho_R * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_R, Tgas_R);
		optical_depths[0] = (tau_L * tau_R * 2.) / (tau_L + tau_R); // harmonic mean. Alternative: 0.5*(tau_L + tau_R)
	} else {
		const auto opacity_L = DefineOpacityExponentsAndLowerValues(group_boundaries, rho_L, Tgas_L);
		const auto opacity_R = DefineOpacityExponentsAndLowerValues(group_boundaries, rho_R, Tgas_R);
		const auto tau_L = dl * rho_L * ComputeBinCenterOpacity(group_boundaries, opacity_L);
		const auto tau_R = dl * rho_R * ComputeBinCenterOpacity(group_boundaries, opacity_R);
		optical_depths = (tau_L * tau_R * 2.) / (tau_L + tau_R); // harmonic mean. Alternative: 0.5*(tau_L + tau_R)
	}

	return optical_depths;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeEddingtonTensor(const double fx, const double fy, const double fz) -> std::array<std::array<double, 3>, 3>
{
	// Compute the radiation pressure tensor

	// AMREX_ASSERT(f < 1.0); // there is sometimes a small (<1%) flux
	// limiting violation when using P1 AMREX_ASSERT(f_R < 1.0);

	auto f = std::sqrt(fx * fx + fy * fy + fz * fz);
	std::array<amrex::Real, 3> fvec = {fx, fy, fz};

	// angle between interface and radiation flux \hat{n}
	// If direction is undefined, just drop direction-dependent
	// terms.
	std::array<amrex::Real, 3> n{};

	for (int ii = 0; ii < 3; ++ii) {
		n[ii] = (f > 0.) ? (fvec[ii] / f) : 0.;
	}

	// compute radiation pressure tensors
	const double chi = RadSystem<problem_t>::ComputeEddingtonFactor(f);

	AMREX_ASSERT((chi >= 1. / 3.) && (chi <= 1.0)); // NOLINT

	// diagonal term of Eddington tensor
	const double Tdiag = (1.0 - chi) / 2.0;

	// anisotropic term of Eddington tensor (in the direction of the
	// rad. flux)
	const double Tf = (3.0 * chi - 1.0) / 2.0;

	// assemble Eddington tensor
	std::array<std::array<double, 3>, 3> T{};

	for (int ii = 0; ii < 3; ++ii) {
		for (int jj = 0; jj < 3; ++jj) {
			const double delta_ij = (ii == jj) ? 1 : 0;
			T[ii][jj] = Tdiag * delta_ij + Tf * (n[ii] * n[jj]);
		}
	}

	return T;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeRadPressure(const double erad, const double Fx, const double Fy, const double Fz, const double fx,
							       const double fy, const double fz) -> RadPressureResult
{
	// Compute the radiation pressure tensor and the maximum signal speed and return them as a struct.

	// check that states are physically admissible
	AMREX_ASSERT(erad > 0.0);

	// Compute the Eddington tensor
	auto T = ComputeEddingtonTensor(fx, fy, fz);

	// frozen Eddington tensor approximation, following Balsara
	// (1999) [JQSRT Vol. 61, No. 5, pp. 617–627, 1999], Eq. 46.
	double Tnormal = NAN;
	if constexpr (DIR == FluxDir::X1) {
		Tnormal = T[0][0];
	} else if constexpr (DIR == FluxDir::X2) {
		Tnormal = T[1][1];
	} else if constexpr (DIR == FluxDir::X3) {
		Tnormal = T[2][2];
	}

	// compute fluxes F_L, F_R
	// T_nx, T_ny, T_nz indicate components where 'n' is the direction of the
	// face normal. F_n is the radiation flux component in the direction of the
	// face normal
	double Fn = NAN;
	double Tnx = NAN;
	double Tny = NAN;
	double Tnz = NAN;

	if constexpr (DIR == FluxDir::X1) {
		Fn = Fx;

		Tnx = T[0][0];
		Tny = T[0][1];
		Tnz = T[0][2];
	} else if constexpr (DIR == FluxDir::X2) {
		Fn = Fy;

		Tnx = T[1][0];
		Tny = T[1][1];
		Tnz = T[1][2];
	} else if constexpr (DIR == FluxDir::X3) {
		Fn = Fz;

		Tnx = T[2][0];
		Tny = T[2][1];
		Tnz = T[2][2];
	}

	AMREX_ASSERT(std::isfinite(Fn));
	AMREX_ASSERT(std::isfinite(Tnx));
	AMREX_ASSERT(std::isfinite(Tny));
	AMREX_ASSERT(std::isfinite(Tnz));

	RadPressureResult result{};
	result.F = {Fn, Tnx * erad, Tny * erad, Tnz * erad};
	// It might be possible to remove this 0.1 floor without affecting the code. I tried and only the 3D RadForce failed (causing S_L = S_R = 0.0 and F[0] =
	// NAN). Read more on https://github.com/quokka-astro/quokka/pull/582 .
	result.S = std::max(0.1, std::sqrt(Tnormal));

	return result;
}

template <typename problem_t>
template <FluxDir DIR>
void RadSystem<problem_t>::ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
					 amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool const use_wavespeed_correction,
					 std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc)
{
	quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in);
	quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);
	quokka::Array4View<amrex::Real, DIR> x1FluxDiffusive(x1FluxDiffusive_in);
	quokka::Array4View<const amrex::Real, DIR> consVar(consVar_in);

	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// interface-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
		for (int g = 0; g < nGroups_ + 1; ++g) {
			radBoundaries_g_copy[g] = radBoundaries_g[g];
		}

		// HLL solver following Toro (1998) and Balsara (2017).
		// Radiation eigenvalues from Skinner & Ostriker (2013).

		// calculate cell optical depth for each photon group
		// Similar to the asymptotic-preserving flux correction in Skinner et al. (2019). Use optionally apply it here to reduce odd-even instability.
		quokka::valarray<double, nGroups_> tau_cell{};
		if (use_wavespeed_correction) {
			tau_cell = ComputeCellOpticalDepth<DIR>(consVar, dx, i, j, k, i_in, j_in, k_in, cons_fc, radBoundaries_g_copy);
		}

		// gather left- and right- state variables
		for (int g = 0; g < nGroups_; ++g) {
			double erad_L = x1LeftState(i, j, k, primRadEnergy_index + numRadVars_ * g);
			double erad_R = x1RightState(i, j, k, primRadEnergy_index + numRadVars_ * g);

			double fx_L = x1LeftState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);
			double fx_R = x1RightState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);

			double fy_L = x1LeftState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);
			double fy_R = x1RightState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);

			double fz_L = x1LeftState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);
			double fz_R = x1RightState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);

			// compute scalar reduced flux f
			double f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
			double f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);

			// Compute "un-reduced" Fx, Fy, Fz
			double Fx_L = fx_L * (c_light_ * erad_L);
			double Fx_R = fx_R * (c_light_ * erad_R);

			double Fy_L = fy_L * (c_light_ * erad_L);
			double Fy_R = fy_R * (c_light_ * erad_R);

			double Fz_L = fz_L * (c_light_ * erad_L);
			double Fz_R = fz_R * (c_light_ * erad_R);

			// check that states are physically admissible; if not, use first-order
			// reconstruction
			if ((erad_L <= 0.) || (erad_R <= 0.) || (f_L >= 1.) || (f_R >= 1.)) {
				erad_L = consVar(i - 1, j, k, radEnergy_index + numRadVars_ * g);
				erad_R = consVar(i, j, k, radEnergy_index + numRadVars_ * g);

				Fx_L = consVar(i - 1, j, k, x1RadFlux_index + numRadVars_ * g);
				Fx_R = consVar(i, j, k, x1RadFlux_index + numRadVars_ * g);

				Fy_L = consVar(i - 1, j, k, x2RadFlux_index + numRadVars_ * g);
				Fy_R = consVar(i, j, k, x2RadFlux_index + numRadVars_ * g);

				Fz_L = consVar(i - 1, j, k, x3RadFlux_index + numRadVars_ * g);
				Fz_R = consVar(i, j, k, x3RadFlux_index + numRadVars_ * g);

				// compute primitive variables
				fx_L = Fx_L / (c_light_ * erad_L);
				fx_R = Fx_R / (c_light_ * erad_R);

				fy_L = Fy_L / (c_light_ * erad_L);
				fy_R = Fy_R / (c_light_ * erad_R);

				fz_L = Fz_L / (c_light_ * erad_L);
				fz_R = Fz_R / (c_light_ * erad_R);

				f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
				f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);
			}

			// ComputeRadPressure returns F_L_and_S_L or F_R_and_S_R
			auto [F_L, S_L] = ComputeRadPressure<DIR>(erad_L, Fx_L, Fy_L, Fz_L, fx_L, fy_L, fz_L);
			S_L *= -1.; // speed sign is -1
			auto [F_R, S_R] = ComputeRadPressure<DIR>(erad_R, Fx_R, Fy_R, Fz_R, fx_R, fy_R, fz_R);

			// correct for reduced speed of light
			F_L[0] *= c_hat_ / c_light_;
			F_R[0] *= c_hat_ / c_light_;
			for (int n = 1; n < numRadVars_; ++n) {
				F_L[n] *= c_hat_ * c_light_;
				F_R[n] *= c_hat_ * c_light_;
			}
			S_L *= c_hat_;
			S_R *= c_hat_;

			const quokka::valarray<double, numRadVars_> U_L = {erad_L, Fx_L, Fy_L, Fz_L};
			const quokka::valarray<double, numRadVars_> U_R = {erad_R, Fx_R, Fy_R, Fz_R};

			// Adjusting wavespeeds is no longer necessary with the IMEX PD-ARS scheme.
			// Read more in https://github.com/quokka-astro/quokka/pull/582
			// However, we let the user optionally apply it to reduce odd-even instability.
			quokka::valarray<double, numRadVars_> epsilon = {1.0, 1.0, 1.0, 1.0};
			if (use_wavespeed_correction) {
				// no correction for odd zones
				if ((i + j + k) % 2 == 0) {
					const double S_corr = std::min(1.0, 1.0 / tau_cell[g]); // Skinner et al.
					epsilon = {S_corr, 1.0, 1.0, 1.0};			// Skinner et al. (2019)
				}
			}

			AMREX_ASSERT(std::abs(S_L) <= c_hat_); // NOLINT
			AMREX_ASSERT(std::abs(S_R) <= c_hat_); // NOLINT

			// in the frozen Eddington tensor approximation, we are always
			// in the star region, so F = F_star
			const quokka::valarray<double, numRadVars_> F =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + epsilon * (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			// check states are valid
			AMREX_ASSERT(!std::isnan(F[0])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[1])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[2])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[3])); // NOLINT

			x1Flux(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = F[0];
			x1Flux(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[1];
			x1Flux(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[2];
			x1Flux(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[3];

			const quokka::valarray<double, numRadVars_> diffusiveF =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			x1FluxDiffusive(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[0];
			x1FluxDiffusive(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[1];
			x1FluxDiffusive(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[2];
			x1FluxDiffusive(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[3];
		} // end loop over radiation groups
	});
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> Real
{
	return NAN;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEnergyMeanOpacity(const double rho, const double Tgas) -> Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/,
										      const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = NAN;
		exponents_and_values[1][g] = NAN;
	}
	return exponents_and_values;
}

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> amrex::GpuArray<double, nGroups_>
{
	// Compute the exponents for the radiation energy density, radiation flux, radiation pressure, or Planck function.

	// Note: Could save some memory by using bin_center_previous and bin_center_current
	amrex::GpuArray<double, nGroups_> bin_center{};
	amrex::GpuArray<double, nGroups_> quant_mean{};
	amrex::GpuArray<double, nGroups_ - 1> logslopes{};
	amrex::GpuArray<double, nGroups_> exponents{};
	for (int g = 0; g < nGroups_; ++g) {
		bin_center[g] = std::sqrt(boundaries[g] * boundaries[g + 1]);
		quant_mean[g] = quant[g] / (boundaries[g + 1] - boundaries[g]);
		if (g > 0) {
			AMREX_ASSERT(bin_center[g] > bin_center[g - 1]);
			if (quant_mean[g] == 0.0 && quant_mean[g - 1] == 0.0) {
				logslopes[g - 1] = 0.0;
			} else if (quant_mean[g - 1] * quant_mean[g] <= 0.0) {
				if (quant_mean[g] > quant_mean[g - 1]) {
					logslopes[g - 1] = inf;
				} else {
					logslopes[g - 1] = -inf;
				}
			} else {
				logslopes[g - 1] = std::log(std::abs(quant_mean[g] / quant_mean[g - 1])) / std::log(bin_center[g] / bin_center[g - 1]);
			}
			AMREX_ASSERT(!std::isnan(logslopes[g - 1]));
		}
	}

	for (int g = 0; g < nGroups_; ++g) {
		if (g == 0) {
			if constexpr (!special_edge_bin_slopes) {
				exponents[g] = -1.0;
			} else {
				exponents[g] = 2.0;
			}
		} else if (g == nGroups_ - 1) {
			if constexpr (!special_edge_bin_slopes) {
				exponents[g] = -1.0;
			} else {
				exponents[g] = -4.0;
			}
		} else {
			exponents[g] = minmod_func(logslopes[g - 1], logslopes[g]);
		}
		AMREX_ASSERT(!std::isnan(exponents[g]));
	}

	if constexpr (PPL_free_slope_st_total) {
		int peak_idx = 0; // index of the peak of logslopes
		for (; peak_idx < nGroups_; ++peak_idx) {
			if (peak_idx == nGroups_ - 1) {
				peak_idx += 0;
				break;
			}
			if (exponents[peak_idx] >= 0.0 && exponents[peak_idx + 1] < 0.0) {
				break;
			}
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(peak_idx < nGroups_ - 1,
						 "Peak index not found. Here peak_index is the index at which the exponent changes its sign.");
		double quant_sum = 0.0;
		double part_sum = 0.0;
		for (int g = 0; g < nGroups_; ++g) {
			quant_sum += quant[g];
			if (g == peak_idx) {
				continue;
			}
			part_sum += exponents[g] * quant[g];
		}
		if (quant[peak_idx] > 0.0 && quant_sum > 0.0) {
			exponents[peak_idx] = (-quant_sum - part_sum) / quant[peak_idx];
			AMREX_ASSERT(!std::isnan(exponents[peak_idx]));
		}
	}
	return exponents;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto
RadSystem<problem_t>::ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
					      amrex::GpuArray<double, nGroups_> const &radBoundaryRatios, amrex::GpuArray<double, nGroups_> const &alpha_quant)
    -> quokka::valarray<double, nGroups_>
{
	amrex::GpuArray<double, nGroups_ + 1> const &alpha_kappa = kappa_expo_and_lower_value[0];
	amrex::GpuArray<double, nGroups_ + 1> const &kappa_lower = kappa_expo_and_lower_value[1];

	quokka::valarray<double, nGroups_> kappa{};
	for (int g = 0; g < nGroups_; ++g) {
		double alpha = alpha_quant[g] + 1.0;
		if (alpha > 100.) {
			kappa[g] = kappa_lower[g] * std::pow(radBoundaryRatios[g], kappa_expo_and_lower_value[0][g]);
			continue;
		}
		if (alpha < -100.) {
			kappa[g] = kappa_lower[g];
			continue;
		}
		double part1 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part1 = std::log(radBoundaryRatios[g]);
		} else {
			part1 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		alpha += alpha_kappa[g];
		double part2 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part2 = std::log(radBoundaryRatios[g]);
		} else {
			part2 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		kappa[g] = kappa_lower[g] / part1 * part2;
		AMREX_ASSERT(!std::isnan(kappa[g]));
	}
	return kappa;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::PlanckFunction(const double nu, const double T) -> double
{
	// returns 4 pi B(nu) / c
	double const coeff = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * T);
	double const x = coeff * nu;
	if (x > 100.) {
		return 0.0;
	}
	double planck_integral = NAN;
	if (x <= 1.0e-10) {
		// Taylor series
		planck_integral = x * x - x * x * x / 2.;
	} else {
		planck_integral = std::pow(x, 3) / (std::exp(x) - 1.0);
	}
	return coeff / (std::pow(PI, 4) / 15.0) * (radiation_constant_ * std::pow(T, 4)) * planck_integral;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeDiffusionFluxMeanOpacity(const quokka::valarray<double, nGroups_> kappaPVec,
										 const quokka::valarray<double, nGroups_> kappaEVec,
										 const quokka::valarray<double, nGroups_> fourPiBoverC,
										 const amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
										 const amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge,
										 const amrex::GpuArray<double, nGroups_ + 1> kappa_slope)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaF{};
	for (int g = 0; g < nGroups_; ++g) {
		// kappaF[g] = 4. / 3. * kappaPVec[g] * fourPiBoverC[g] + 1. / 3. * kappa_slope[g] * kappaPVec[g] * fourPiBoverC[g] - 1. / 3. *
		// delta_nu_kappa_B_at_edge[g];
		kappaF[g] = (kappaPVec[g] + 1. / 3. * kappaEVec[g]) * fourPiBoverC[g] +
			    1. / 3. * (kappa_slope[g] * kappaEVec[g] * fourPiBoverC[g] - delta_nu_kappa_B_at_edge[g]);
		auto const denom = 4. / 3. * fourPiBoverC[g] - 1. / 3. * delta_nu_B_at_edge[g];
		if (denom <= 0.0) {
			AMREX_ASSERT(kappaF[g] == 0.0);
			kappaF[g] = 0.0;
		} else {
			kappaF[g] /= denom;
		}
	}
	return kappaF;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
									 amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa_center{};
	for (int g = 0; g < nGroups_; ++g) {
		kappa_center[g] =
		    kappa_expo_and_lower_value[1][g] * std::pow(rad_boundaries[g + 1] / rad_boundaries[g], 0.5 * kappa_expo_and_lower_value[0][g]);
	}
	return kappa_center;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxInDiffusionLimit(const amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, const double T,
									     const double vel) -> amrex::GpuArray<double, nGroups_>
{
	double const coeff = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * T);
	amrex::GpuArray<double, nGroups_ + 1> edge_values{};
	amrex::GpuArray<double, nGroups_> flux{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		auto x = coeff * rad_boundaries[g];
		edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x) - 1. / 3. * x * (std::pow(x, 3) / (std::exp(x) - 1.0)) / gInf;
		// test: reproduce the Planck function
		// edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x);
	}
	for (int g = 0; g < nGroups_; ++g) {
		flux[g] = vel * radiation_constant_ * std::pow(T, 4) * (edge_values[g + 1] - edge_values[g]);
	}
	return flux;
}

#include "radiation/radiation_coupling.hpp" // IWYU pragma: export
#include "radiation/source_terms.hpp"	    // IWYU pragma: export

#endif // RADIATION_SYSTEM_HPP_