#ifndef actual_rhs_H
#define actual_rhs_H

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "burn_type.H"
#include "fundamental_constants.H"
#include "integrator_data.H"
#include <actual_network.H>
#include <extern_parameters.H>
#include <jacobian_utilities.H>

using namespace amrex;
using namespace network_rp;
using namespace eos_rp;
using namespace C;

void actual_rhs_init();

// Species 0 (Index 0 in the state)         = electron
// Species 1 (Index 1 in the state)         = neutral hydrogen
// Species 2 (Index 2 in the state)         = ionized hydrogen
// Index net_ienuc(=3)                      = specific internal energy

constexpr amrex::Real RydbergEnergy = 13.6_rt * C::ev2erg;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 1.5e-18_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionizing_band_photon_energy() -> Real { return 0.5_rt * (ChemBandsHeader_[0] + ChemBandsHeader_[1]) * C::hplanck; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_coefficient(Real T) -> Real
{
	if (recombination_temperature_dependent == 0) {
		return 2.6e-13_rt;
	}
	return 2.6e-13 * std::pow(T / 1.0e4, -0.7);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dRecombination_coefficient_dT(Real T) -> Real
{
	if (recombination_temperature_dependent == 0) {
		return 0.0_rt;
	}
	return -2.6e-13 * 0.7 * std::pow(T / 1.0e4, -1.7) / 1.0e4;
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_rhs(const burn_t &state, Array1D<Real, 1, INT_NEQS> &ydot)
{
	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;

	const amrex::Real n_e = state.xn[0];
	const amrex::Real n_HI = state.xn[1];
	const amrex::Real n_HII = state.xn[2];
	const amrex::Real n_gamma =
	    amrex::max(state.rn[0], 0.0_rt); // clamp at zero: prevents sign reversal if VODE steps rn[0] slightly negative within tolerance
#ifndef SKIP_PHOTOCHEMFLUX
	const amrex::Real n_gamma_flux = state.rn[1];
#endif

	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
	if ((allow_mixed_cell_cooling == 0) && (xHII > mixed_cell_lb) && (xHII < mixed_cell_ub)) {
		molecular_process_switch = 0;
	}

	const amrex::Real photoionization_term = chat_sigma_photo * n_HI * n_gamma;
	const amrex::Real collisional_ionization_term =
	    (collisional_ionization_switch != 0) ? n_e * n_HI * get_collisional_ionization_coefficient(state.T) : 0.0_rt;
	const amrex::Real recombination_term = (recombination_switch != 0) ? get_recombination_coefficient(state.T) * n_e * n_HII : 0.0_rt;

	ydot(1) = photoionization_term + collisional_ionization_term - recombination_term;
	ydot(2) = -ydot(1);
	ydot(3) = ydot(1);
	ydot(net_ienuc) = 0.0_rt;
	const amrex::Real recombination_cooling_rate =
	    (energy_switch != 0 && recombination_cooling_switch != 0) ? get_recombination_cooling_coefficient(state.T) * n_e * n_HII : 0.0_rt;
	const amrex::Real ion_ff_cooling_rate =
	    (energy_switch != 0 && ion_ff_cooling_switch != 0) ? get_ion_ff_cooling_coefficient(state.T) * n_e * n_HII : 0.0_rt;
	const amrex::Real KI_heating_rate =
	    (energy_switch != 0 && molecular_process_switch != 0 && KI_heating_switch != 0) ? get_KI_heating_coefficient(state.T) * n_HI : 0.0_rt;
	const amrex::Real KI_cooling_rate =
	    (energy_switch != 0 && molecular_process_switch != 0 && KI_cooling_switch != 0) ? get_KI_cooling_coefficient(state.T) * n_HI * n_HI : 0.0_rt;
	const amrex::Real collisional_ionization_cooling_rate =
	    (energy_switch != 0 && collisional_ionization_switch != 0) ? RydbergEnergy * collisional_ionization_term : 0.0_rt;
	const amrex::Real photoheating_rate =
	    (energy_switch != 0 && photoheating_switch != 0) ? get_ionization_heating_coefficient(state.T) * photoionization_term : 0.0_rt;
	if (energy_switch != 0) {
		amrex::Real energy_source = 0.0_rt;
		energy_source += photoheating_rate;
		energy_source += KI_heating_rate;
		energy_source -= KI_cooling_rate;
		energy_source -= recombination_cooling_rate;
		energy_source -= ion_ff_cooling_rate;
		energy_source -= collisional_ionization_cooling_rate;
		ydot(net_ienuc) = energy_source / state.rho;
	}
	amrex::Real dust_absorption_term = 0.0_rt;
	// burn_t::dust_kappa exists only under THERMAL_DUST_PHOTOCHEMISTRY, so it is read here into a local that
	// stays zero otherwise -- mirroring the chat_dust_kappa_rho local in actual_jac below. Without this the
	// flux row would fail to compile for any problem that has photochemistry and flux but no dust.
	amrex::Real chat_dust_kappa_rho = 0.0_rt;
#ifdef THERMAL_DUST_PHOTOCHEMISTRY
	chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
	dust_absorption_term = chat_dust_kappa_rho * n_gamma;
#endif
	ydot(5) = -photoionization_term - dust_absorption_term;
#ifndef SKIP_PHOTOCHEMFLUX
	ydot(6) = -chat_sigma_photo * n_HI * n_gamma_flux - chat_dust_kappa_rho * n_gamma_flux;
#else
	amrex::ignore_unused(chat_dust_kappa_rho);
#endif
#ifdef THERMAL_DUST_PHOTOCHEMISTRY
	const amrex::Real recombination_binding_energy_rate = (energy_switch != 0) ? RydbergEnergy * recombination_term : 0.0_rt;
	ydot(net_ienuc + NumChemRadEqs + 1) = get_ion_ff_cooling_coefficient(state.T) * n_e * n_HII + KI_cooling_rate;
	ydot(net_ienuc + NumChemRadEqs + 2) = recombination_cooling_rate + recombination_binding_energy_rate;
	ydot(net_ienuc + NumChemRadEqs + NumThermalBands + 1) = dust_absorption_term * get_ionizing_band_photon_energy();
#endif
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{
	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	amrex::Real sigma_coll = 0.0_rt;
	amrex::Real d_sigma_coll_dT = 0.0_rt;
	if (collisional_ionization_switch != 0) {
		sigma_coll = get_collisional_ionization_coefficient(state.T);
		d_sigma_coll_dT = dCollisional_ionization_coefficient_dT(state.T);
	}
	amrex::Real alpha_rec = 0.0_rt;
	amrex::Real d_alpha_rec_dT = 0.0_rt;
	if (recombination_switch != 0) {
		alpha_rec = get_recombination_coefficient(state.T);
		d_alpha_rec_dT = dRecombination_coefficient_dT(state.T);
	}

	const amrex::Real Gamma_photo = ((energy_switch != 0) && (photoheating_switch != 0)) ? get_ionization_heating_coefficient(state.T) : 0.0_rt;
	const amrex::Real Gamma_KI = ((energy_switch != 0) && (KI_heating_switch != 0)) ? get_KI_heating_coefficient(state.T) : 0.0_rt;
	const amrex::Real Lambda_KI = ((energy_switch != 0) && (KI_cooling_switch != 0)) ? get_KI_cooling_coefficient(state.T) : 0.0_rt;
	const amrex::Real d_Lambda_KI_dT = ((energy_switch != 0) && (KI_cooling_switch != 0)) ? dKI_cooling_coefficient_dT(state.T) : 0.0_rt;
	const amrex::Real Lambda_rec = ((energy_switch != 0) && (recombination_cooling_switch != 0)) ? get_recombination_cooling_coefficient(state.T) : 0.0_rt;
	const amrex::Real d_Lambda_rec_dT =
	    ((energy_switch != 0) && (recombination_cooling_switch != 0)) ? dRecombination_cooling_coefficient_dT(state.T) : 0.0_rt;
	const amrex::Real Lambda_ff = ((energy_switch != 0) && (ion_ff_cooling_switch != 0)) ? get_ion_ff_cooling_coefficient(state.T) : 0.0_rt;
	const amrex::Real d_Lambda_ff_dT = ((energy_switch != 0) && (ion_ff_cooling_switch != 0)) ? d_ion_ff_cooling_coefficient_dT(state.T) : 0.0_rt;

	const amrex::Real n_e = state.xn[0];
	const amrex::Real n_HI = state.xn[1];
	const amrex::Real n_HII = state.xn[2];
	const amrex::Real n_gamma =
	    amrex::max(state.rn[0], 0.0_rt); // clamp at zero: prevents sign reversal if VODE steps rn[0] slightly negative within tolerance
	// Gate for column-5 Jacobian entries: when rn[0] < 0 the clamp makes the RHS
	// independent of rn[0], so all derivatives w.r.t. rn[0] are zero.
	const amrex::Real n_gamma_active = (state.rn[0] >= 0.0_rt) ? 1.0_rt : 0.0_rt;
#ifndef SKIP_PHOTOCHEMFLUX
	const amrex::Real n_gamma_flux = state.rn[1];
#endif

	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
	if ((allow_mixed_cell_cooling == 0) && (xHII > mixed_cell_lb) && (xHII < mixed_cell_ub)) {
		molecular_process_switch = 0;
	}

	const amrex::Real d_collisional_ionization_term_dT = collisional_ionization_switch * n_e * n_HI * d_sigma_coll_dT;
	const amrex::Real d_recombination_term_dT = recombination_switch * (d_alpha_rec_dT * n_e * n_HII);
	const amrex::Real d_KI_cooling_dT = molecular_process_switch * KI_cooling_switch * (d_Lambda_KI_dT * n_HI * n_HI);
	const amrex::Real d_recombination_cooling_dT = recombination_cooling_switch * (d_Lambda_rec_dT * n_e * n_HII);
	const amrex::Real d_ion_ff_cooling_dT = ion_ff_cooling_switch * (d_Lambda_ff_dT * n_e * n_HII);

	const amrex::Real dTde = 1 / state.dedT;

	jac.zero();

	jac(1, 1) = (collisional_ionization_switch * n_HI * sigma_coll - recombination_switch * alpha_rec * n_HII);
	jac(1, 2) = (chat_sigma_photo * n_gamma + collisional_ionization_switch * n_e * sigma_coll);
	jac(1, 3) = -recombination_switch * alpha_rec * n_e;
	jac(1, net_ienuc) = (d_collisional_ionization_term_dT - d_recombination_term_dT) * dTde;
	jac(1, 5) = n_gamma_active * chat_sigma_photo * n_HI;

	jac(2, 1) = -jac(1, 1);
	jac(2, 2) = -jac(1, 2);
	jac(2, 3) = -jac(1, 3);
	jac(2, net_ienuc) = -jac(1, net_ienuc);
	jac(2, 5) = -jac(1, 5);
	jac(3, 1) = jac(1, 1);
	jac(3, 2) = jac(1, 2);
	jac(3, 3) = jac(1, 3);
	jac(3, net_ienuc) = jac(1, net_ienuc);
	jac(3, 5) = jac(1, 5);

	// Collisional ionization debits RydbergEnergy * collisional_ionization_term from gas heat (see
	// actual_rhs); sigma_coll and d_collisional_ionization_term_dT are gated by collisional_ionization_switch
	// but not energy_switch, so that gate is applied explicitly here to match the RHS.
	jac(net_ienuc, 1) = energy_switch *
			    (-recombination_cooling_switch * Lambda_rec * n_HII - ion_ff_cooling_switch * Lambda_ff * n_HII -
			     RydbergEnergy * collisional_ionization_switch * n_HI * sigma_coll) /
			    state.rho;
	jac(net_ienuc, 2) =
	    energy_switch *
	    (photoheating_switch * Gamma_photo * chat_sigma_photo * n_gamma + molecular_process_switch * KI_heating_switch * Gamma_KI -
	     2.0_rt * molecular_process_switch * KI_cooling_switch * Lambda_KI * n_HI - RydbergEnergy * collisional_ionization_switch * n_e * sigma_coll) /
	    state.rho;
	jac(net_ienuc, 3) = energy_switch * (-recombination_cooling_switch * Lambda_rec * n_e - ion_ff_cooling_switch * Lambda_ff * n_e) / state.rho;
	jac(net_ienuc, net_ienuc) = energy_switch *
				    (-d_recombination_cooling_dT - d_KI_cooling_dT - d_ion_ff_cooling_dT - RydbergEnergy * d_collisional_ionization_term_dT) *
				    dTde / state.rho;
	jac(net_ienuc, 5) = energy_switch * (photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) * n_gamma_active / state.rho;

	amrex::Real chat_dust_kappa_rho = 0.0_rt;
#ifdef THERMAL_DUST_PHOTOCHEMISTRY
	chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
#endif

	jac(5, 1) = 0.0_rt;
	jac(5, 2) = -chat_sigma_photo * n_gamma;
	jac(5, 3) = 0.0_rt;
	jac(5, net_ienuc) = 0.0_rt;
	jac(5, 5) = -n_gamma_active * (chat_sigma_photo * n_HI + chat_dust_kappa_rho);

#ifndef SKIP_PHOTOCHEMFLUX
	jac(1, 6) = 0.0_rt;
	jac(2, 6) = -jac(1, 6);
	jac(3, 6) = jac(1, 6);
	jac(net_ienuc, 6) = 0.0_rt;
	jac(5, 6) = 0.0_rt;
	jac(6, 1) = 0.0_rt;
	jac(6, 2) = -chat_sigma_photo * n_gamma_flux;
	jac(6, 3) = 0.0_rt;
	jac(6, net_ienuc) = 0.0_rt;
	jac(6, 5) = 0.0_rt;
	jac(6, 6) = -n_HI * chat_sigma_photo - chat_dust_kappa_rho;
#endif

#ifdef THERMAL_DUST_PHOTOCHEMISTRY
	// Thermal-band rows: index 0 (row i_ir) = IR, sourced by ff_cooling; index 1
	// (row i_optical) = optical, sourced by recombination cooling. See rhs_thermal_bands().
	const int i_ir = net_ienuc + NumChemRadEqs + 1;
	const int i_optical = net_ienuc + NumChemRadEqs + 2;

	// The IR band carries ydot(i_ir) = ff_cooling_rate + KI_cooling_rate - KI_heating_rate, so its n_HI
	// dependence is d(Lambda_KI * n_HI^2)/d(n_HI) - d(Gamma_KI * n_HI)/d(n_HI). Gamma_KI is the constant
	// 2.0e-26 (get_KI_heating_coefficient), so the heating term adds nothing to the net_ienuc column below.
	jac(i_ir, 1) = ion_ff_cooling_switch * Lambda_ff * n_HII;
	jac(i_ir, 2) = molecular_process_switch * (2.0_rt * KI_cooling_switch * Lambda_KI * n_HI);
	jac(i_ir, 3) = ion_ff_cooling_switch * Lambda_ff * n_e;
	jac(i_ir, net_ienuc) = (ion_ff_cooling_switch * d_Lambda_ff_dT * n_e * n_HII + d_KI_cooling_dT) * dTde;

	// recombination_binding_energy_rate = energy_switch * RydbergEnergy * alpha_rec * n_e * n_HII (see
	// actual_rhs); alpha_rec and d_alpha_rec_dT are gated by recombination_switch but not energy_switch, so
	// that gate is applied explicitly here to match the RHS.
	const amrex::Real rydberg_rate_coeff = energy_switch * RydbergEnergy * alpha_rec;
	const amrex::Real d_rydberg_rate_coeff_dT = energy_switch * RydbergEnergy * d_alpha_rec_dT;
	jac(i_optical, 1) = recombination_cooling_switch * Lambda_rec * n_HII + rydberg_rate_coeff * n_HII;
	// No column-2 entry: ydot(i_optical) = recombination_cooling_rate + recombination_binding_energy_rate, and
	// neither term depends on n_HI. The KI heating term lives in the i_ir row above.
	jac(i_optical, 3) = recombination_cooling_switch * Lambda_rec * n_e + rydberg_rate_coeff * n_e;
	jac(i_optical, net_ienuc) = (recombination_cooling_switch * d_Lambda_rec_dT * n_e * n_HII + d_rydberg_rate_coeff_dT * n_e * n_HII) * dTde;

	// e_dust_absorbed row: d(dust_absorption_term)/d(n_gamma). Dust absorption does not
	// depend on species, T, or flux, so all other columns in this row are zero.
	const int i_dust = net_ienuc + NumChemRadEqs + NumThermalBands + 1;
	jac(i_dust, 5) = n_gamma_active * chat_dust_kappa_rho * get_ionizing_band_photon_energy();
#endif
}

#endif
