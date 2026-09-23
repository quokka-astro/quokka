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

// Species indices (0-indexed, xn[]/spmasses[]) come from the SPECIES_ENUM
// set in this problem's CMakeLists.txt, via Species::NetworkSpecies.
// ydot()/jac() are 1-indexed, so the same species use index+1 there.
// net_ienuc (= NumSpec + 1) is the specific internal energy row/column.
// Radiation rows/columns follow net_ienuc: photon number density, then
// (unless SKIP_PHOTOCHEMFLUX) photon flux.
constexpr int e_idx = static_cast<int>(Species::e);
constexpr int H_idx = static_cast<int>(Species::H);
constexpr int Hp_idx = static_cast<int>(Species::H_p);
constexpr int ie = e_idx + 1;
constexpr int iH = H_idx + 1;
constexpr int iHp = Hp_idx + 1;
constexpr int i_ngamma = net_ienuc + 1;
constexpr int i_ngamma_flux = net_ienuc + 2;
// Thermal-band rows follow the chemistry-band rows, whose count (NumChemRadEqs) already accounts for
// whether each band carries a flux variable. Band 0 is the IR, band 1 the optical; the order matches
// group_ir / group_optical on the Quokka side, which reads these back into the radiation state.
static_assert(NumThermalBands == 2, "thermophotochemistry_handwritten emits into exactly two thermal bands (IR, optical)");
constexpr int i_ir = net_ienuc + NumChemRadEqs + 1;
constexpr int i_optical = net_ienuc + NumChemRadEqs + 2;

constexpr amrex::Real RydbergEnergy = 13.6_rt * C::ev2erg;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 2.1596876715103067e-18_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionizing_band_photon_energy() -> Real
{
	auto const ev_bounds = ChemBandsHeader();
	const amrex::Real ev_low = ev_bounds[0];
	const amrex::Real ev_high = ev_bounds[1];
	const amrex::Real alpha = ChemBandsPowerLawIndex_;

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
	if (recombination_temperature_dependent == 0) {
		return coeff;
	}
	return coeff * std::pow(T / 1.0e4, -0.69999999999999996);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dRecombination_coefficient_dT(Real T) -> Real
{
	if (recombination_temperature_dependent == 0) {
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_rhs(const burn_t &state, Array1D<Real, 1, INT_NEQS> &ydot)
{
	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	const amrex::Real sigma_coll = (collisional_ionization_switch != 0) ? get_collisional_ionization_coefficient(state.T) : 0.0_rt;

	const amrex::Real n_e = state.xn[e_idx];
	const amrex::Real n_HI = state.xn[H_idx];
	const amrex::Real n_HII = state.xn[Hp_idx];
	const amrex::Real n_gamma = amrex::max(state.rn[0], 0.0_rt);
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
	const amrex::Real collisional_ionization_term = (collisional_ionization_switch != 0) ? n_e * n_HI * sigma_coll : 0.0_rt;
	const amrex::Real recombination_term = (recombination_switch != 0) ? get_recombination_coefficient(state.T) * n_e * n_HII : 0.0_rt;

	ydot(ie) = photoionization_term + collisional_ionization_term - recombination_term;
	ydot(iH) = -ydot(ie);
	ydot(iHp) = ydot(ie);
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
	amrex::Real energy_source = 0.0_rt;
	energy_source += photoheating_rate;
	energy_source += KI_heating_rate;
	energy_source -= KI_cooling_rate;
	energy_source -= recombination_cooling_rate;
	energy_source -= ion_ff_cooling_rate;
	energy_source -= collisional_ionization_cooling_rate;
	ydot(net_ienuc) = energy_source / state.rho;
	ydot(5) = -photoionization_term;
#ifndef SKIP_PHOTOCHEMFLUX
	ydot(i_ngamma_flux) = -chat_sigma_photo * n_HI * n_gamma_flux;
#endif

	const amrex::Real recombination_binding_energy_rate = (energy_switch != 0) ? RydbergEnergy * recombination_term : 0.0_rt;
	ydot(i_ir) = ion_ff_cooling_rate + KI_cooling_rate;
	ydot(i_optical) = recombination_cooling_rate + recombination_binding_energy_rate;
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

	const amrex::Real n_e = state.xn[e_idx];
	const amrex::Real n_HI = state.xn[H_idx];
	const amrex::Real n_HII = state.xn[Hp_idx];
	const amrex::Real n_gamma = amrex::max(state.rn[0], 0.0_rt);
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

	jac(ie, ie) = (collisional_ionization_switch * n_HI * sigma_coll - recombination_switch * alpha_rec * n_HII);
	jac(ie, iH) = (chat_sigma_photo * n_gamma + collisional_ionization_switch * n_e * sigma_coll);
	jac(ie, iHp) = -recombination_switch * alpha_rec * n_e;
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

	jac(net_ienuc, ie) = energy_switch *
			     (-recombination_cooling_switch * Lambda_rec * n_HII - ion_ff_cooling_switch * Lambda_ff * n_HII -
			      RydbergEnergy * collisional_ionization_switch * n_HI * sigma_coll) /
			     state.rho;
	jac(net_ienuc, iH) =
	    energy_switch *
	    (photoheating_switch * Gamma_photo * chat_sigma_photo * n_gamma + molecular_process_switch * KI_heating_switch * Gamma_KI -
	     2.0_rt * molecular_process_switch * KI_cooling_switch * Lambda_KI * n_HI - RydbergEnergy * collisional_ionization_switch * n_e * sigma_coll) /
	    state.rho;
	jac(net_ienuc, iHp) = energy_switch * (-recombination_cooling_switch * Lambda_rec * n_e - ion_ff_cooling_switch * Lambda_ff * n_e) / state.rho;
	jac(net_ienuc, net_ienuc) = energy_switch *
				    (-d_recombination_cooling_dT - d_KI_cooling_dT - d_ion_ff_cooling_dT - RydbergEnergy * d_collisional_ionization_term_dT) *
				    dTde / state.rho;
	jac(net_ienuc, 5) = energy_switch * (photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) * n_gamma_active / state.rho;

	jac(i_ngamma, ie) = 0.0_rt;
	jac(i_ngamma, iH) = -chat_sigma_photo * n_gamma;
	jac(i_ngamma, iHp) = 0.0_rt;
	jac(i_ngamma, net_ienuc) = 0.0_rt;
	jac(i_ngamma, i_ngamma) = -n_gamma_active * chat_sigma_photo * n_HI;

#ifndef SKIP_PHOTOCHEMFLUX
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
	jac(i_ngamma_flux, i_ngamma_flux) = -n_HI * chat_sigma_photo;
#endif

	jac(i_ir, ie) = ion_ff_cooling_switch * Lambda_ff * n_HII;
	jac(i_ir, iH) = molecular_process_switch * (2.0_rt * KI_cooling_switch * Lambda_KI * n_HI);
	jac(i_ir, iHp) = ion_ff_cooling_switch * Lambda_ff * n_e;
	jac(i_ir, net_ienuc) = (ion_ff_cooling_switch * d_Lambda_ff_dT * n_e * n_HII + d_KI_cooling_dT) * dTde;

	const amrex::Real rydberg_rate_coeff = energy_switch * RydbergEnergy * alpha_rec;
	const amrex::Real d_rydberg_rate_coeff_dT = energy_switch * RydbergEnergy * d_alpha_rec_dT;
	jac(i_optical, ie) = recombination_cooling_switch * Lambda_rec * n_HII + rydberg_rate_coeff * n_HII;
	jac(i_optical, iHp) = recombination_cooling_switch * Lambda_rec * n_e + rydberg_rate_coeff * n_e;
	jac(i_optical, net_ienuc) = (recombination_cooling_switch * d_Lambda_rec_dT * n_e * n_HII + d_rydberg_rate_coeff_dT * n_e * n_HII) * dTde;
}

#endif
