// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
#define NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
/// \file network.hpp
/// \brief The thermophotochemistry_handwritten network: its constants, runtime parameters, right-hand side and
/// Jacobian.

#include <cmath>
#include <limits>
#include <string_view>

#include "gcem.hpp"

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "microphysics/ode_system.hpp"

namespace quokka::networks
{

/// The runtime parameters of Thermophotochemistry, read from the `network` prefix with Microphysics' names and
/// defaults. They are defined in network.cpp, not inline here: HIP cannot register inline managed variables.
namespace thermophotochemistry
{
extern AMREX_GPU_MANAGED int recombination_switch;		  // NOLINT
extern AMREX_GPU_MANAGED int recombination_temperature_dependent; // NOLINT
extern AMREX_GPU_MANAGED int collisional_ionization_switch;	  // NOLINT
extern AMREX_GPU_MANAGED int energy_switch;			  // NOLINT
extern AMREX_GPU_MANAGED int photoheating_switch;		  // NOLINT
extern AMREX_GPU_MANAGED int KI_heating_switch;			  // NOLINT
extern AMREX_GPU_MANAGED int recombination_cooling_switch;	  // NOLINT
extern AMREX_GPU_MANAGED int KI_cooling_switch;			  // NOLINT
extern AMREX_GPU_MANAGED int ion_ff_cooling_switch;		  // NOLINT
/// cooling in mixed cells; with 0, cells with mixed_cell_lb < x_HII < mixed_cell_ub do not cool
extern AMREX_GPU_MANAGED int allow_mixed_cell_cooling; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real mixed_cell_lb;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real mixed_cell_ub;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real dust_kappa;       // NOLINT
extern AMREX_GPU_MANAGED amrex::Real small_x;	       // NOLINT
} // namespace thermophotochemistry

/// Hydrogen photoionization in one chemical band, with dust in two thermal bands.
struct Thermophotochemistry {
	static constexpr int NumSpec = 3;
	/// indices of the species in xn
	enum Species : int { H = 0, H_p, e }; // NOLINT(cppcoreguidelines-use-enum-class)

	/// mass of each species [g]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesMass() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		return {{1.673773e-24, 1.6728620616289998e-24, 9.10938371e-28}};
	}
	/// adiabatic index of each species
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesGamma() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		return {{5. / 3., 5. / 3., 5. / 3.}};
	}
	/// the equation of state the network assumes, named as jaff names it (the type of jaff's [network.eos])
	static constexpr std::string_view EOSType = "ideal";

	static constexpr int NumChemBands = 1;
	/// edges of the chemical bands [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1> { return {{13.6, 26.0}}; }
	/// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a chemical band
	static constexpr double ChemBandsPowerLawIndex = 0;

	/// the IR and the optical band, in that order
	static constexpr int NumThermalBands = 2;

	/// gray dust opacity to ionizing radiation [cm^2/g] (thermophotochemistry::dust_kappa)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto DustKappa() -> amrex::Real { return thermophotochemistry::dust_kappa; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SmallX() -> amrex::Real { return thermophotochemistry::small_x; }

	/// Read the runtime parameters (namespace thermophotochemistry).
	static void ReadParameters();

	/// The right-hand side on Quokka's burn state (thermophotochemistry::actual_rhs, below).
	template <typename BurnT>
	AMREX_GPU_HOST_DEVICE static void
	actual_rhs(const BurnT &state, amrex::GpuArray<amrex::Real, quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>::neqs> &ydot);
	/// The Jacobian of actual_rhs (thermophotochemistry::actual_jac, below).
	template <typename BurnT, class MatrixType> AMREX_GPU_HOST_DEVICE static void actual_jac(const BurnT &state, MatrixType &jac);
	/// Both at once, from one evaluation of the rate coefficients. The integrator uses it where it needs both at the
	/// same state.
	template <typename BurnT, class MatrixType>
	AMREX_GPU_HOST_DEVICE static void
	actual_rhs_and_jac(const BurnT &state, amrex::GpuArray<amrex::Real, quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>::neqs> &ydot,
			   MatrixType &jac);
};

/// The right-hand side, the Jacobian and their rate coefficients, in the namespace of the network's parameters.
namespace thermophotochemistry
{
using amrex::Real;
using amrex::literals::operator""_rt;

// Species indices (xn[]/SpeciesMass()) come from the Species enum above; ydot[]/jac() use the same indices.
// net_ienuc (the ODE system's energy_row, NumSpec) is the specific internal energy row/column.
// Radiation rows/columns follow net_ienuc: photon number density, then
// (with Microphysics_Traits::integrate_rad_flux) photon flux.
constexpr int e_idx = static_cast<int>(quokka::networks::Thermophotochemistry::Species::e);
constexpr int H_idx = static_cast<int>(quokka::networks::Thermophotochemistry::Species::H);
constexpr int Hp_idx = static_cast<int>(quokka::networks::Thermophotochemistry::Species::H_p);
constexpr int ie = e_idx;
constexpr int iH = H_idx;
constexpr int iHp = Hp_idx;
// The thermal-band and dust rows come from the ODE system (ODESystem::thermal_band_row and dust_row).
// Band 0 is the IR, band 1 the optical; the order matches group_ir / group_optical on the Quokka side, which reads
// these back into the radiation state.
static_assert(quokka::networks::Thermophotochemistry::NumThermalBands == 2,
	      "thermophotochemistry_handwritten emits into exactly two thermal bands (IR, optical)");

constexpr amrex::Real RydbergEnergy = 13.6_rt * C::ev2erg;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 3.2385396673401957e-18_rt; }

/// The mean photon energy [erg] of the ionizing band, from its edges and its power-law index. Both are compile-time
/// constants of the network, so this is evaluated once, at compile time (gcem provides the constexpr log and pow).
constexpr auto ionizing_band_photon_energy() -> amrex::Real
{
	constexpr amrex::Real ev_low = quokka::networks::Thermophotochemistry::ChemBands().arr[0];
	constexpr amrex::Real ev_high = quokka::networks::Thermophotochemistry::ChemBands().arr[1];
	constexpr amrex::Real alpha = quokka::networks::Thermophotochemistry::ChemBandsPowerLawIndex;
	constexpr bool open_topped = (ev_high == std::numeric_limits<amrex::Real>::infinity());
	static_assert(!open_topped || alpha < 0.0, "an open-topped chemistry band only has a finite average photon energy for power_law_index < 0");

	if constexpr (open_topped) {
		return ev_low * (1.0 - 1.0 / alpha) * C::ev2erg;
	} else if constexpr (alpha == 0.0) {
		return ev_low * ev_high * gcem::log(ev_high / ev_low) / (ev_high - ev_low) * C::ev2erg;
	} else if constexpr (alpha == 1.0) {
		return (ev_high - ev_low) / gcem::log(ev_high / ev_low) * C::ev2erg;
	} else {
		return ((alpha - 1.0) / alpha) * (gcem::pow(ev_high, alpha) - gcem::pow(ev_low, alpha)) /
		       (gcem::pow(ev_high, alpha - 1.0) - gcem::pow(ev_low, alpha - 1.0)) * C::ev2erg;
	}
}

constexpr amrex::Real IonizingBandPhotonEnergy = ionizing_band_photon_energy();

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_ionizing_band_photon_energy() -> Real { return IonizingBandPhotonEnergy; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionization_heating_coefficient(Real T) -> Real
{
	amrex::ignore_unused(T);
	return amrex::max(get_ionizing_band_photon_energy() - RydbergEnergy, 0.0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_KI_heating_coefficient(Real T) -> Real { return 2.0e-26; }

/// The temperature-dependent rate coefficients and their temperature derivatives. A coefficient whose process is
/// switched off is zero, and so is its derivative.
struct Rates {
	amrex::Real alpha_rec = 0.0_rt; ///< case-B recombination [cm^3 s^-1]
	amrex::Real d_alpha_rec_dT = 0.0_rt;
	amrex::Real sigma_coll = 0.0_rt; ///< collisional ionization [cm^3 s^-1] (Krumholz et al. 2007)
	amrex::Real d_sigma_coll_dT = 0.0_rt;
	amrex::Real Lambda_KI = 0.0_rt; ///< KI cooling (Vazquez-Semadeni et al. 2007)
	amrex::Real d_Lambda_KI_dT = 0.0_rt;
	amrex::Real Lambda_rec = 0.0_rt; ///< recombination cooling (Krumholz et al. 2007)
	amrex::Real d_Lambda_rec_dT = 0.0_rt;
	amrex::Real Lambda_ff = 0.0_rt; ///< ion free-free cooling (Frazer and Heitsch 2019)
	amrex::Real d_Lambda_ff_dT = 0.0_rt;
};

/// One piece c T^a of a piecewise power-law fit.
struct PowerLaw {
	amrex::Real c;
	amrex::Real a;
};

/// The piece of the Frazer and Heitsch (2019) cle_term fit that holds at temperature T. The thresholds are 10^2, 10^2.8,
/// 10^3.6, 10^4, 10^4.5, 10^5 and 10^6.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cle_fit(const Real T) -> PowerLaw
{
	// selected from the top down with conditional moves, not a chain of branches
	PowerLaw fit{7.49e-21, -0.462};
	fit = (T < 1.0e6) ? PowerLaw{1.10e-21, -0.323} : fit;
	fit = (T < 1.0e5) ? PowerLaw{1.74e-24, 0.237} : fit;
	fit = (T < 31622.776601683792) ? PowerLaw{2.67e-30, 1.529} : fit;
	fit = (T < 1.0e4) ? PowerLaw{1.08e-32, 2.127} : fit;
	fit = (T < 3981.0717055349733) ? PowerLaw{1.11e-24, -0.099} : fit;
	fit = (T < 630.957344480193) ? PowerLaw{2.34e-26, 0.500} : fit;
	fit = (T < 1.0e2) ? PowerLaw{3.47e-29, 1.915} : fit;
	return fit;
}

/// All the rate coefficients at temperature T, from one log, one sqrt and one exp per term: each power T^a is
/// exp(a log T), and each derivative reuses its coefficient (d(c T^a)/dT = a c T^a / T).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto compute_rates(const Real T) -> Rates
{
	Rates r{};
	const amrex::Real logT = std::log(T);
	const amrex::Real sqrtT = std::sqrt(T);
	const amrex::Real invT = 1.0_rt / T;

	if (thermophotochemistry::recombination_switch != 0) {
		constexpr amrex::Real coeff = 2.63e-13_rt;
		if (thermophotochemistry::recombination_temperature_dependent == 0) {
			r.alpha_rec = coeff;
		} else {
			constexpr amrex::Real log_1e4 = 9.210340371976184; // log(1e4)
			r.alpha_rec = coeff * std::exp(-0.7_rt * (logT - log_1e4));
			r.d_alpha_rec_dT = -0.7_rt * r.alpha_rec * invT;
		}
	}

	if (thermophotochemistry::collisional_ionization_switch != 0) {
		constexpr amrex::Real E_ion_over_k = 2.18e-11_rt / C::k_B;
		r.sigma_coll = 5.84e-11 * sqrtT * std::exp(-E_ion_over_k * invT);
		r.d_sigma_coll_dT = r.sigma_coll * (0.5_rt * invT + E_ion_over_k * invT * invT);
	}

	if (thermophotochemistry::energy_switch != 0) {
		if (thermophotochemistry::KI_cooling_switch != 0) {
			const amrex::Real inv_T1 = 1.0_rt / (T + 1.0e3);
			const amrex::Real a = 1.0e7 * std::exp(-118400.0 * inv_T1);
			const amrex::Real b = 1.4e-2 * sqrtT * std::exp(-92.0 * invT);
			r.Lambda_KI = 2.0e-26 * (a + b);
			r.d_Lambda_KI_dT = 2.0e-26 * (a * 118400.0 * inv_T1 * inv_T1 + b * (0.5_rt * invT + 92.0 * invT * invT));
		}
		if ((thermophotochemistry::recombination_cooling_switch != 0) && (T >= 100.0_rt)) {
			// 6.1e-10 k_B T T^-0.89 = 6.1e-10 k_B T^0.11
			r.Lambda_rec = 6.1e-10 * C::k_B * std::exp(0.11_rt * logT);
			r.d_Lambda_rec_dT = 0.11_rt * r.Lambda_rec * invT;
		}
		if (thermophotochemistry::ion_ff_cooling_switch != 0) {
			const PowerLaw fit = cle_fit(T);
			const amrex::Real cle_term = fit.c * std::exp(fit.a * logT);
			r.Lambda_ff = 1.3 * 1.427e-27 * sqrtT + cle_term;
			r.d_Lambda_ff_dT = 1.3 * 1.427e-27 * 0.5 * (sqrtT * invT) + fit.a * cle_term * invT;
		}
	}
	return r;
}

/// The right-hand side at the burn state, given the rate coefficients at its temperature.
template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
rhs_with_rates(const BurnT &state, const Rates &rates, amrex::GpuArray<Real, quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>::neqs> &ydot)
{
	using System = quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>;
	constexpr int net_ienuc = System::energy_row;
	constexpr int i_ngamma = net_ienuc + 1;
	constexpr int i_ngamma_flux = net_ienuc + 2;
	constexpr int i_ir = System::thermal_band_row;
	constexpr int i_optical = System::thermal_band_row + 1;
	constexpr int i_dust = System::dust_row;

	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	const amrex::Real sigma_coll = rates.sigma_coll;

	const amrex::Real n_e = state.xn[e_idx];
	const amrex::Real n_HI = state.xn[H_idx];
	const amrex::Real n_HII = state.xn[Hp_idx];
	const amrex::Real n_gamma = amrex::max(state.rn[0], 0.0_rt);

	// a mixed cell has mixed_cell_lb < x_HII < mixed_cell_ub, tested without forming x_HII = n_HII / n_H_total
	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	if ((thermophotochemistry::allow_mixed_cell_cooling == 0) && (n_H_total > 0.0_rt) && (n_HII > thermophotochemistry::mixed_cell_lb * n_H_total) &&
	    (n_HII < thermophotochemistry::mixed_cell_ub * n_H_total)) {
		molecular_process_switch = 0;
	}

	const amrex::Real photoionization_term = chat_sigma_photo * n_HI * n_gamma;
	const amrex::Real collisional_ionization_term = (thermophotochemistry::collisional_ionization_switch != 0) ? n_e * n_HI * sigma_coll : 0.0_rt;
	const amrex::Real recombination_term = (thermophotochemistry::recombination_switch != 0) ? rates.alpha_rec * n_e * n_HII : 0.0_rt;

	ydot[ie] = photoionization_term + collisional_ionization_term - recombination_term;
	ydot[iH] = -ydot[ie];
	ydot[iHp] = ydot[ie];
	ydot[net_ienuc] = 0.0_rt;
	const amrex::Real recombination_cooling_rate =
	    (thermophotochemistry::energy_switch != 0 && thermophotochemistry::recombination_cooling_switch != 0) ? rates.Lambda_rec * n_e * n_HII : 0.0_rt;
	const amrex::Real ion_ff_cooling_rate =
	    (thermophotochemistry::energy_switch != 0 && thermophotochemistry::ion_ff_cooling_switch != 0) ? rates.Lambda_ff * n_e * n_HII : 0.0_rt;
	const amrex::Real KI_heating_rate =
	    (thermophotochemistry::energy_switch != 0 && molecular_process_switch != 0 && thermophotochemistry::KI_heating_switch != 0)
		? get_KI_heating_coefficient(state.T) * n_HI
		: 0.0_rt;
	const amrex::Real KI_cooling_rate =
	    (thermophotochemistry::energy_switch != 0 && molecular_process_switch != 0 && thermophotochemistry::KI_cooling_switch != 0)
		? rates.Lambda_KI * n_HI * n_HI
		: 0.0_rt;
	const amrex::Real collisional_ionization_cooling_rate =
	    (thermophotochemistry::energy_switch != 0 && thermophotochemistry::collisional_ionization_switch != 0) ? RydbergEnergy * collisional_ionization_term
														   : 0.0_rt;
	const amrex::Real photoheating_rate = (thermophotochemistry::energy_switch != 0 && thermophotochemistry::photoheating_switch != 0)
						  ? get_ionization_heating_coefficient(state.T) * photoionization_term
						  : 0.0_rt;
	amrex::Real energy_source = 0.0_rt;
	energy_source += photoheating_rate;
	energy_source += KI_heating_rate;
	energy_source -= KI_cooling_rate;
	energy_source -= recombination_cooling_rate;
	energy_source -= ion_ff_cooling_rate;
	energy_source -= collisional_ionization_cooling_rate;
	ydot[net_ienuc] = energy_source / state.rho;

	amrex::Real chat_dust_kappa_rho = 0.0_rt;
	if constexpr (BurnT::traits::dust_chemical_band_absorption) {
		chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
	}
	const amrex::Real dust_absorption_term = chat_dust_kappa_rho * n_gamma;

	ydot[i_ngamma] = -photoionization_term - dust_absorption_term;
	if constexpr (BurnT::traits::integrate_rad_flux) {
		const amrex::Real n_gamma_flux = state.rn[1];
		ydot[i_ngamma_flux] = -chat_sigma_photo * n_HI * n_gamma_flux - chat_dust_kappa_rho * n_gamma_flux;
	}
	if constexpr (BurnT::traits::dust_chemical_band_absorption) {
		ydot[i_dust] = dust_absorption_term * get_ionizing_band_photon_energy();
	}

	const amrex::Real recombination_binding_energy_rate = (thermophotochemistry::energy_switch != 0) ? RydbergEnergy * recombination_term : 0.0_rt;
	ydot[i_ir] = ion_ff_cooling_rate + KI_cooling_rate;
	ydot[i_optical] = recombination_cooling_rate + recombination_binding_energy_rate;
}

/// The Jacobian at the burn state, given the rate coefficients at its temperature.
template <typename BurnT, class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void jac_with_rates(const BurnT &state, const Rates &rates, MatrixType &jac)
{
	using System = quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>;
	constexpr int net_ienuc = System::energy_row;
	constexpr int i_ngamma = net_ienuc + 1;
	constexpr int i_ngamma_flux = net_ienuc + 2;
	constexpr int i_ir = System::thermal_band_row;
	constexpr int i_optical = System::thermal_band_row + 1;
	constexpr int i_dust = System::dust_row;

	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	const amrex::Real sigma_coll = rates.sigma_coll;
	const amrex::Real d_sigma_coll_dT = rates.d_sigma_coll_dT;
	const amrex::Real alpha_rec = rates.alpha_rec;
	const amrex::Real d_alpha_rec_dT = rates.d_alpha_rec_dT;

	const amrex::Real Gamma_photo = ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::photoheating_switch != 0))
					    ? get_ionization_heating_coefficient(state.T)
					    : 0.0_rt;
	const amrex::Real Gamma_KI =
	    ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::KI_heating_switch != 0)) ? get_KI_heating_coefficient(state.T) : 0.0_rt;
	const amrex::Real Lambda_KI = rates.Lambda_KI;
	const amrex::Real d_Lambda_KI_dT = rates.d_Lambda_KI_dT;
	const amrex::Real Lambda_rec = rates.Lambda_rec;
	const amrex::Real d_Lambda_rec_dT = rates.d_Lambda_rec_dT;
	const amrex::Real Lambda_ff = rates.Lambda_ff;
	const amrex::Real d_Lambda_ff_dT = rates.d_Lambda_ff_dT;

	const amrex::Real n_e = state.xn[e_idx];
	const amrex::Real n_HI = state.xn[H_idx];
	const amrex::Real n_HII = state.xn[Hp_idx];
	const amrex::Real n_gamma = amrex::max(state.rn[0], 0.0_rt);
	const amrex::Real n_gamma_active = (state.rn[0] >= 0.0_rt) ? 1.0_rt : 0.0_rt;

	// a mixed cell has mixed_cell_lb < x_HII < mixed_cell_ub, tested without forming x_HII = n_HII / n_H_total
	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	if ((thermophotochemistry::allow_mixed_cell_cooling == 0) && (n_H_total > 0.0_rt) && (n_HII > thermophotochemistry::mixed_cell_lb * n_H_total) &&
	    (n_HII < thermophotochemistry::mixed_cell_ub * n_H_total)) {
		molecular_process_switch = 0;
	}

	const amrex::Real d_collisional_ionization_term_dT = thermophotochemistry::collisional_ionization_switch * n_e * n_HI * d_sigma_coll_dT;
	const amrex::Real d_recombination_term_dT = thermophotochemistry::recombination_switch * (d_alpha_rec_dT * n_e * n_HII);
	const amrex::Real d_KI_cooling_dT = molecular_process_switch * thermophotochemistry::KI_cooling_switch * (d_Lambda_KI_dT * n_HI * n_HI);
	const amrex::Real d_recombination_cooling_dT = thermophotochemistry::recombination_cooling_switch * (d_Lambda_rec_dT * n_e * n_HII);
	const amrex::Real d_ion_ff_cooling_dT = thermophotochemistry::ion_ff_cooling_switch * (d_Lambda_ff_dT * n_e * n_HII);

	const amrex::Real dTde = 1 / state.dedT;
	const amrex::Real inv_rho = 1 / state.rho;

	amrex::Real chat_dust_kappa_rho = 0.0_rt;
	if constexpr (BurnT::traits::dust_chemical_band_absorption) {
		chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
	}

	// the integrator zeroes jac before calling actual_jac

	jac(ie, ie) =
	    (thermophotochemistry::collisional_ionization_switch * n_HI * sigma_coll - thermophotochemistry::recombination_switch * alpha_rec * n_HII);
	jac(ie, iH) = (chat_sigma_photo * n_gamma + thermophotochemistry::collisional_ionization_switch * n_e * sigma_coll);
	jac(ie, iHp) = -thermophotochemistry::recombination_switch * alpha_rec * n_e;
	jac(ie, net_ienuc) = (d_collisional_ionization_term_dT - d_recombination_term_dT) * dTde;
	jac(ie, i_ngamma) = n_gamma_active * chat_sigma_photo * n_HI;

	jac(iH, ie) = -jac(ie, ie);
	jac(iH, iH) = -jac(ie, iH);
	jac(iH, iHp) = -jac(ie, iHp);
	jac(iH, net_ienuc) = -jac(ie, net_ienuc);
	jac(iH, i_ngamma) = -jac(ie, i_ngamma);
	jac(iHp, ie) = jac(ie, ie);
	jac(iHp, iH) = jac(ie, iH);
	jac(iHp, iHp) = jac(ie, iHp);
	jac(iHp, net_ienuc) = jac(ie, net_ienuc);
	jac(iHp, i_ngamma) = jac(ie, i_ngamma);

	jac(net_ienuc, ie) =
	    thermophotochemistry::energy_switch *
	    (-thermophotochemistry::recombination_cooling_switch * Lambda_rec * n_HII - thermophotochemistry::ion_ff_cooling_switch * Lambda_ff * n_HII -
	     RydbergEnergy * thermophotochemistry::collisional_ionization_switch * n_HI * sigma_coll) *
	    inv_rho;
	jac(net_ienuc, iH) = thermophotochemistry::energy_switch *
			     (thermophotochemistry::photoheating_switch * Gamma_photo * chat_sigma_photo * n_gamma +
			      molecular_process_switch * thermophotochemistry::KI_heating_switch * Gamma_KI -
			      2.0_rt * molecular_process_switch * thermophotochemistry::KI_cooling_switch * Lambda_KI * n_HI -
			      RydbergEnergy * thermophotochemistry::collisional_ionization_switch * n_e * sigma_coll) *
			     inv_rho;
	jac(net_ienuc, iHp) =
	    thermophotochemistry::energy_switch *
	    (-thermophotochemistry::recombination_cooling_switch * Lambda_rec * n_e - thermophotochemistry::ion_ff_cooling_switch * Lambda_ff * n_e) * inv_rho;
	jac(net_ienuc, net_ienuc) = thermophotochemistry::energy_switch *
				    (-d_recombination_cooling_dT - d_KI_cooling_dT - d_ion_ff_cooling_dT - RydbergEnergy * d_collisional_ionization_term_dT) *
				    dTde * inv_rho;
	jac(net_ienuc, i_ngamma) = thermophotochemistry::energy_switch * (thermophotochemistry::photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) *
				   n_gamma_active * inv_rho;

	jac(i_ngamma, ie) = 0.0_rt;
	jac(i_ngamma, iH) = -chat_sigma_photo * n_gamma;
	jac(i_ngamma, iHp) = 0.0_rt;
	jac(i_ngamma, net_ienuc) = 0.0_rt;
	jac(i_ngamma, i_ngamma) = -n_gamma_active * (chat_sigma_photo * n_HI + chat_dust_kappa_rho);

	if constexpr (BurnT::traits::integrate_rad_flux) {
		const amrex::Real n_gamma_flux = state.rn[1];
		jac(ie, i_ngamma_flux) = 0.0_rt;
		jac(iH, i_ngamma_flux) = -jac(ie, i_ngamma_flux);
		jac(iHp, i_ngamma_flux) = jac(ie, i_ngamma_flux);
		jac(net_ienuc, i_ngamma_flux) = 0.0_rt;
		jac(i_ngamma, i_ngamma_flux) = 0.0_rt;
		jac(i_ngamma_flux, ie) = 0.0_rt;
		jac(i_ngamma_flux, iH) = -chat_sigma_photo * n_gamma_flux;
		jac(i_ngamma_flux, iHp) = 0.0_rt;
		jac(i_ngamma_flux, net_ienuc) = 0.0_rt;
		jac(i_ngamma_flux, i_ngamma) = 0.0_rt;
		jac(i_ngamma_flux, i_ngamma_flux) = -n_HI * chat_sigma_photo - chat_dust_kappa_rho;
	}

	jac(i_ir, ie) = thermophotochemistry::ion_ff_cooling_switch * Lambda_ff * n_HII;
	jac(i_ir, iH) = molecular_process_switch * (2.0_rt * thermophotochemistry::KI_cooling_switch * Lambda_KI * n_HI);
	jac(i_ir, iHp) = thermophotochemistry::ion_ff_cooling_switch * Lambda_ff * n_e;
	jac(i_ir, net_ienuc) = (thermophotochemistry::ion_ff_cooling_switch * d_Lambda_ff_dT * n_e * n_HII + d_KI_cooling_dT) * dTde;

	const amrex::Real rydberg_rate_coeff = thermophotochemistry::energy_switch * RydbergEnergy * alpha_rec;
	const amrex::Real d_rydberg_rate_coeff_dT = thermophotochemistry::energy_switch * RydbergEnergy * d_alpha_rec_dT;
	jac(i_optical, ie) = thermophotochemistry::recombination_cooling_switch * Lambda_rec * n_HII + rydberg_rate_coeff * n_HII;
	jac(i_optical, iHp) = thermophotochemistry::recombination_cooling_switch * Lambda_rec * n_e + rydberg_rate_coeff * n_e;
	jac(i_optical, net_ienuc) =
	    (thermophotochemistry::recombination_cooling_switch * d_Lambda_rec_dT * n_e * n_HII + d_rydberg_rate_coeff_dT * n_e * n_HII) * dTde;

	if constexpr (BurnT::traits::dust_chemical_band_absorption) {
		jac(i_dust, i_ngamma) = n_gamma_active * chat_dust_kappa_rho * get_ionizing_band_photon_energy();
	}
}

} // namespace thermophotochemistry

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Thermophotochemistry::actual_rhs(const BurnT &state,
				 amrex::GpuArray<amrex::Real, quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>::neqs> &ydot)
{
	thermophotochemistry::rhs_with_rates(state, thermophotochemistry::compute_rates(state.T), ydot);
}

template <typename BurnT, class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Thermophotochemistry::actual_jac(const BurnT &state, MatrixType &jac)
{
	thermophotochemistry::jac_with_rates(state, thermophotochemistry::compute_rates(state.T), jac);
}

template <typename BurnT, class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Thermophotochemistry::actual_rhs_and_jac(
    const BurnT &state, amrex::GpuArray<amrex::Real, quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>::neqs> &ydot, MatrixType &jac)
{
	const thermophotochemistry::Rates rates = thermophotochemistry::compute_rates(state.T);
	thermophotochemistry::rhs_with_rates(state, rates, ydot);
	thermophotochemistry::jac_with_rates(state, rates, jac);
}

} // namespace quokka::networks

#endif // NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
