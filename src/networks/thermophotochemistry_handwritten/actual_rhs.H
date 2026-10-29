#ifndef actual_rhs_H
#define actual_rhs_H

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "microphysics/integrator_layout.hpp"
#include "networks/thermophotochemistry_handwritten/network.hpp"

using amrex::Real;
// The runtime parameters are qualified with the network's namespace (network.hpp).
namespace thermophotochemistry = quokka::networks::thermophotochemistry;
using namespace C;

void actual_rhs_init();

// Species indices (0-indexed, xn[]/spmasses[]) come from the Species enum in network.hpp.
// ydot()/jac() are 1-indexed, so the same species use index+1 there.
// net_ienuc (the integrator layout's energy_row, NumSpec + 1) is the specific internal energy row/column.
// Radiation rows/columns follow net_ienuc: photon number density, then
// (with Microphysics_Traits::integrate_rad_flux) photon flux.
constexpr int e_idx = static_cast<int>(quokka::networks::Thermophotochemistry::Species::e);
constexpr int H_idx = static_cast<int>(quokka::networks::Thermophotochemistry::Species::H);
constexpr int Hp_idx = static_cast<int>(quokka::networks::Thermophotochemistry::Species::H_p);
constexpr int ie = e_idx + 1;
constexpr int iH = H_idx + 1;
constexpr int iHp = Hp_idx + 1;
// The thermal-band and dust rows come from the integrator's layout (IntegratorLayout::thermal_band_row and dust_row).
// Band 0 is the IR, band 1 the optical; the order matches group_ir / group_optical on the Quokka side, which reads
// these back into the radiation state.
static_assert(quokka::networks::Thermophotochemistry::NumThermalBands == 2,
	      "thermophotochemistry_handwritten emits into exactly two thermal bands (IR, optical)");

constexpr amrex::Real RydbergEnergy = 13.6_rt * C::ev2erg;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 2.1596876715103067e-18_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionizing_band_photon_energy() -> Real
{
	auto const ev_bounds = quokka::networks::Thermophotochemistry::ChemBands();
	const amrex::Real ev_low = ev_bounds[0];
	const amrex::Real ev_high = ev_bounds[1];
	const amrex::Real alpha = quokka::networks::Thermophotochemistry::ChemBandsPowerLawIndex;

	amrex::Real ev_avg = NAN;
	if (std::isinf(ev_high)) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(alpha < 0.0, "get_ionizing_band_photon_energy: an open-topped chemistry band only has a "
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

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_coefficient(Real T) -> Real
{
	const amrex::Real coeff = 2.63e-13_rt;
	if (thermophotochemistry::recombination_temperature_dependent == 0) {
		return coeff;
	}
	return coeff * std::pow(T / 1.0e4, -0.69999999999999996);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dRecombination_coefficient_dT(Real T) -> Real
{
	if (thermophotochemistry::recombination_temperature_dependent == 0) {
		return 0.0_rt;
	}
	return -2.63e-13 * 0.7 * std::pow(T / 1.0e4, -1.69999999999999996) / 1.0e4;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_collisional_ionization_coefficient(Real T) -> Real
{
	// Krumholz et al. 2007
	return 5.84e-11 * std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dCollisional_ionization_coefficient_dT(Real T) -> Real
{
	return 5.84e-11 * 0.5 / std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T)) +
	       5.84e-11 * std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T)) * 2.18e-11_rt / (C::k_B * T * T);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionization_heating_coefficient(Real T) -> Real
{
	amrex::ignore_unused(T);
	return amrex::max(get_ionizing_band_photon_energy() - RydbergEnergy, 0.0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_KI_heating_coefficient(Real T) -> Real { return 2.0e-26; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_KI_cooling_coefficient(Real T) -> Real
{
	// Fix KI cooling function based on Vazquez-Semadeni et al. (2007).
	return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) + 1.4e-2 * std::sqrt(T) * std::exp(-92.0 / T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dKI_cooling_coefficient_dT(Real T) -> Real
{
	return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) * 118400.0 / ((T + 1.0e3) * (T + 1.0e3)) +
			  1.4e-2 * (0.5 / std::sqrt(T) * std::exp(-92.0 / T) + std::sqrt(T) * std::exp(-92.0 / T) * 92.0 / (T * T)));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_cooling_coefficient(Real T) -> Real
{
	// Krumholz et al. 2007
	if (T < 100.0_rt) {
		return 0.0_rt;
	}
	return 6.1e-10 * C::k_B * T * std::pow(T, -0.89);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dRecombination_cooling_coefficient_dT(Real T) -> Real
{
	if (T < 100.0_rt) {
		return 0.0_rt;
	}
	return 6.1e-10 * C::k_B * (std::pow(T, -0.89) + T * (-0.89) * std::pow(T, -1.89));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_cle_term(Real T) -> Real
{
	// Frazer and Heitsch 2019
	amrex::Real cle_term = 0.0_rt;
	if (T < 1.0e2) {
		cle_term = 3.47e-29 * std::pow(T, 1.915);
	} else if (T < std::pow(10.0, 2.8)) {
		cle_term = 2.34e-26 * std::pow(T, 0.500);
	} else if (T < std::pow(10.0, 3.6)) {
		cle_term = 1.11e-24 * std::pow(T, -0.099);
	} else if (T < 1.0e4) {
		cle_term = 1.08e-32 * std::pow(T, 2.127);
	} else if (T < std::pow(10.0, 4.5)) {
		cle_term = 2.67e-30 * std::pow(T, 1.529);
	} else if (T < 1.0e5) {
		cle_term = 1.74e-24 * std::pow(T, 0.237);
	} else if (T < 1.0e6) {
		cle_term = 1.10e-21 * std::pow(T, -0.323);
	} else {
		cle_term = 7.49e-21 * std::pow(T, -0.462);
	}
	return cle_term;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ion_ff_cooling_coefficient(Real T) -> Real
{
	amrex::Real cle_term = get_cle_term(T);
	return 1.3 * 1.427e-27 * std::sqrt(T) + cle_term; // Frazer and Heitsch 2019
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto d_ion_ff_cooling_coefficient_dT(Real T) -> Real
{
	amrex::Real d_cle_term_dT = 0.0_rt;
	amrex::Real alpha = 0.0_rt;
	if (T < 1.0e2) {
		alpha = 1.915;
	} else if (T < std::pow(10.0, 2.8)) {
		alpha = 0.500;
	} else if (T < std::pow(10.0, 3.6)) {
		alpha = -0.099;
	} else if (T < 1.0e4) {
		alpha = 2.127;
	} else if (T < std::pow(10.0, 4.5)) {
		alpha = 1.529;
	} else if (T < 1.0e5) {
		alpha = 0.237;
	} else if (T < 1.0e6) {
		alpha = -0.323;
	} else {
		alpha = -0.462;
	}
	d_cle_term_dT = alpha * get_cle_term(T) / T;
	return 1.3 * 1.427e-27 * 0.5 / std::sqrt(T) + d_cle_term_dT;
}

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
actual_rhs(const BurnT &state, amrex::Array1D<Real, 1, quokka::microphysics::IntegratorLayout<typename BurnT::problem_type, BurnT>::neqs> &ydot)
{
	using Layout = quokka::microphysics::IntegratorLayout<typename BurnT::problem_type, BurnT>;
	constexpr int net_ienuc = Layout::energy_row;
	constexpr int i_ngamma = net_ienuc + 1;
	constexpr int i_ngamma_flux = net_ienuc + 2;
	constexpr int i_ir = Layout::thermal_band_row;
	constexpr int i_optical = Layout::thermal_band_row + 1;
	constexpr int i_dust = Layout::dust_row;

	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	const amrex::Real sigma_coll = (thermophotochemistry::collisional_ionization_switch != 0) ? get_collisional_ionization_coefficient(state.T) : 0.0_rt;

	const amrex::Real n_e = state.xn[e_idx];
	const amrex::Real n_HI = state.xn[H_idx];
	const amrex::Real n_HII = state.xn[Hp_idx];
	const amrex::Real n_gamma = amrex::max(state.rn[0], 0.0_rt);

	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
	if ((thermophotochemistry::allow_mixed_cell_cooling == 0) && (xHII > thermophotochemistry::mixed_cell_lb) &&
	    (xHII < thermophotochemistry::mixed_cell_ub)) {
		molecular_process_switch = 0;
	}

	const amrex::Real photoionization_term = chat_sigma_photo * n_HI * n_gamma;
	const amrex::Real collisional_ionization_term = (thermophotochemistry::collisional_ionization_switch != 0) ? n_e * n_HI * sigma_coll : 0.0_rt;
	const amrex::Real recombination_term =
	    (thermophotochemistry::recombination_switch != 0) ? get_recombination_coefficient(state.T) * n_e * n_HII : 0.0_rt;

	ydot(ie) = photoionization_term + collisional_ionization_term - recombination_term;
	ydot(iH) = -ydot(ie);
	ydot(iHp) = ydot(ie);
	ydot(net_ienuc) = 0.0_rt;
	const amrex::Real recombination_cooling_rate = (thermophotochemistry::energy_switch != 0 && thermophotochemistry::recombination_cooling_switch != 0)
							   ? get_recombination_cooling_coefficient(state.T) * n_e * n_HII
							   : 0.0_rt;
	const amrex::Real ion_ff_cooling_rate = (thermophotochemistry::energy_switch != 0 && thermophotochemistry::ion_ff_cooling_switch != 0)
						    ? get_ion_ff_cooling_coefficient(state.T) * n_e * n_HII
						    : 0.0_rt;
	const amrex::Real KI_heating_rate =
	    (thermophotochemistry::energy_switch != 0 && molecular_process_switch != 0 && thermophotochemistry::KI_heating_switch != 0)
		? get_KI_heating_coefficient(state.T) * n_HI
		: 0.0_rt;
	const amrex::Real KI_cooling_rate =
	    (thermophotochemistry::energy_switch != 0 && molecular_process_switch != 0 && thermophotochemistry::KI_cooling_switch != 0)
		? get_KI_cooling_coefficient(state.T) * n_HI * n_HI
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
	ydot(net_ienuc) = energy_source / state.rho;

	amrex::Real chat_dust_kappa_rho = 0.0_rt;
	if constexpr (BurnT::traits::dust_chemical_band_absorption) {
		chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
	}
	const amrex::Real dust_absorption_term = chat_dust_kappa_rho * n_gamma;

	ydot(i_ngamma) = -photoionization_term - dust_absorption_term;
	if constexpr (BurnT::traits::integrate_rad_flux) {
		const amrex::Real n_gamma_flux = state.rn[1];
		ydot(i_ngamma_flux) = -chat_sigma_photo * n_HI * n_gamma_flux - chat_dust_kappa_rho * n_gamma_flux;
	}
	if constexpr (BurnT::traits::dust_chemical_band_absorption) {
		ydot(i_dust) = dust_absorption_term * get_ionizing_band_photon_energy();
	}

	const amrex::Real recombination_binding_energy_rate = (thermophotochemistry::energy_switch != 0) ? RydbergEnergy * recombination_term : 0.0_rt;
	ydot(i_ir) = ion_ff_cooling_rate + KI_cooling_rate;
	ydot(i_optical) = recombination_cooling_rate + recombination_binding_energy_rate;
}

template <typename BurnT, class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_jac(const BurnT &state, MatrixType &jac)
{
	using Layout = quokka::microphysics::IntegratorLayout<typename BurnT::problem_type, BurnT>;
	constexpr int net_ienuc = Layout::energy_row;
	constexpr int i_ngamma = net_ienuc + 1;
	constexpr int i_ngamma_flux = net_ienuc + 2;
	constexpr int i_ir = Layout::thermal_band_row;
	constexpr int i_optical = Layout::thermal_band_row + 1;
	constexpr int i_dust = Layout::dust_row;

	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	amrex::Real sigma_coll = 0.0_rt;
	amrex::Real d_sigma_coll_dT = 0.0_rt;
	if (thermophotochemistry::collisional_ionization_switch != 0) {
		sigma_coll = get_collisional_ionization_coefficient(state.T);
		d_sigma_coll_dT = dCollisional_ionization_coefficient_dT(state.T);
	}
	amrex::Real alpha_rec = 0.0_rt;
	amrex::Real d_alpha_rec_dT = 0.0_rt;
	if (thermophotochemistry::recombination_switch != 0) {
		alpha_rec = get_recombination_coefficient(state.T);
		d_alpha_rec_dT = dRecombination_coefficient_dT(state.T);
	}

	const amrex::Real Gamma_photo = ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::photoheating_switch != 0))
					    ? get_ionization_heating_coefficient(state.T)
					    : 0.0_rt;
	const amrex::Real Gamma_KI =
	    ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::KI_heating_switch != 0)) ? get_KI_heating_coefficient(state.T) : 0.0_rt;
	const amrex::Real Lambda_KI =
	    ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::KI_cooling_switch != 0)) ? get_KI_cooling_coefficient(state.T) : 0.0_rt;
	const amrex::Real d_Lambda_KI_dT =
	    ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::KI_cooling_switch != 0)) ? dKI_cooling_coefficient_dT(state.T) : 0.0_rt;
	const amrex::Real Lambda_rec = ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::recombination_cooling_switch != 0))
					   ? get_recombination_cooling_coefficient(state.T)
					   : 0.0_rt;
	const amrex::Real d_Lambda_rec_dT = ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::recombination_cooling_switch != 0))
						? dRecombination_cooling_coefficient_dT(state.T)
						: 0.0_rt;
	const amrex::Real Lambda_ff = ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::ion_ff_cooling_switch != 0))
					  ? get_ion_ff_cooling_coefficient(state.T)
					  : 0.0_rt;
	const amrex::Real d_Lambda_ff_dT = ((thermophotochemistry::energy_switch != 0) && (thermophotochemistry::ion_ff_cooling_switch != 0))
					       ? d_ion_ff_cooling_coefficient_dT(state.T)
					       : 0.0_rt;

	const amrex::Real n_e = state.xn[e_idx];
	const amrex::Real n_HI = state.xn[H_idx];
	const amrex::Real n_HII = state.xn[Hp_idx];
	const amrex::Real n_gamma = amrex::max(state.rn[0], 0.0_rt);
	const amrex::Real n_gamma_active = (state.rn[0] >= 0.0_rt) ? 1.0_rt : 0.0_rt;

	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
	if ((thermophotochemistry::allow_mixed_cell_cooling == 0) && (xHII > thermophotochemistry::mixed_cell_lb) &&
	    (xHII < thermophotochemistry::mixed_cell_ub)) {
		molecular_process_switch = 0;
	}

	const amrex::Real d_collisional_ionization_term_dT = thermophotochemistry::collisional_ionization_switch * n_e * n_HI * d_sigma_coll_dT;
	const amrex::Real d_recombination_term_dT = thermophotochemistry::recombination_switch * (d_alpha_rec_dT * n_e * n_HII);
	const amrex::Real d_KI_cooling_dT = molecular_process_switch * thermophotochemistry::KI_cooling_switch * (d_Lambda_KI_dT * n_HI * n_HI);
	const amrex::Real d_recombination_cooling_dT = thermophotochemistry::recombination_cooling_switch * (d_Lambda_rec_dT * n_e * n_HII);
	const amrex::Real d_ion_ff_cooling_dT = thermophotochemistry::ion_ff_cooling_switch * (d_Lambda_ff_dT * n_e * n_HII);

	const amrex::Real dTde = 1 / state.dedT;

	amrex::Real chat_dust_kappa_rho = 0.0_rt;
	if constexpr (BurnT::traits::dust_chemical_band_absorption) {
		chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
	}

	jac.zero();

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
	     RydbergEnergy * thermophotochemistry::collisional_ionization_switch * n_HI * sigma_coll) /
	    state.rho;
	jac(net_ienuc, iH) = thermophotochemistry::energy_switch *
			     (thermophotochemistry::photoheating_switch * Gamma_photo * chat_sigma_photo * n_gamma +
			      molecular_process_switch * thermophotochemistry::KI_heating_switch * Gamma_KI -
			      2.0_rt * molecular_process_switch * thermophotochemistry::KI_cooling_switch * Lambda_KI * n_HI -
			      RydbergEnergy * thermophotochemistry::collisional_ionization_switch * n_e * sigma_coll) /
			     state.rho;
	jac(net_ienuc, iHp) =
	    thermophotochemistry::energy_switch *
	    (-thermophotochemistry::recombination_cooling_switch * Lambda_rec * n_e - thermophotochemistry::ion_ff_cooling_switch * Lambda_ff * n_e) /
	    state.rho;
	jac(net_ienuc, net_ienuc) = thermophotochemistry::energy_switch *
				    (-d_recombination_cooling_dT - d_KI_cooling_dT - d_ion_ff_cooling_dT - RydbergEnergy * d_collisional_ionization_term_dT) *
				    dTde / state.rho;
	jac(net_ienuc, i_ngamma) = thermophotochemistry::energy_switch * (thermophotochemistry::photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) *
				   n_gamma_active / state.rho;

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

#endif
