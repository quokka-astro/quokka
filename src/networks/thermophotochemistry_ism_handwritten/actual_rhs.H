// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef actual_rhs_H
#define actual_rhs_H

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "burn_type.H"
#include "cooling/GrackleNTTable.hpp"
#include "cooling/ISMCooling.hpp"
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
// This network emits into no thermal bands (NumThermalBands = 0): all cooling energy leaves the simulation. The dust row, if any,
// follows the chemistry-band rows. The FUV field G0 [Habing units] is state.aux[0].
static_assert(NumThermalBands == 0, "thermophotochemistry_ism_handwritten emits into no thermal bands");
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
constexpr int i_dust = net_ienuc + NumChemRadEqs + NumThermalBands + 1;
#endif

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

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ion_ff_cooling_coefficient(Real T) -> Real
{
	return 1.3 * 1.427e-27 * std::sqrt(T); // Frazer and Heitsch 2019 (without their metal-line term, which the nebular lines replace)
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto d_ion_ff_cooling_coefficient_dT(Real T) -> Real { return 1.3 * 1.427e-27 * 0.5 / std::sqrt(T); }

/// Net energy rate [erg cm^-3 s^-1] of the Lya, nebular, fine-structure, grain-recombination, PE and CR terms, as heating minus cooling,
/// with its partial derivatives with respect to T, n_HI and n_H+. The terms that depend on the electron density use
/// n_e^max (ISMCooling.hpp, ElectronDensityMax), whose dependence on n_HI, n_H+ and T is chained through; the network electron species
/// does not enter. The derivative field dne of the result is zero.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_ism_energy_terms(const burn_t &state) -> quokka::ISMCooling::Rate
{
	namespace ISM = quokka::ISMCooling;
	if (energy_switch == 0) {
		return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
	}
	const auto *table = ISM::GetGrackleNTTable();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(table != nullptr,
					 "thermophotochemistry_ism_handwritten: the Grackle (nH, T) table is not registered (ism_cooling.grackle_nT_table)");

	const amrex::Real T = state.T;
	const amrex::Real n_HI = amrex::max(state.xn[H_idx], 0.0_rt);
	const amrex::Real n_Hp = amrex::max(state.xn[Hp_idx], 0.0_rt);
	const amrex::Real G0 = state.aux[0];
	const amrex::Real Zp = ism_metallicity;

	const auto nt = ISM::LookupGrackleNT(n_HI + n_Hp, T, *table);
	auto ne = ISM::ElectronDensityMax(n_HI, n_Hp, T, nt.xe, nt.dxe_dT, nt.dxe_dnH, Zp, cr_ionization_rate_HI, get_recombination_coefficient(T),
					  dRecombination_coefficient_dT(T));
	// keep the rates finite when there is no gas at all
	constexpr amrex::Real ne_floor = 1.0e-30_rt;
	if (ne.ne < ne_floor) {
		ne = {ne_floor, 0.0_rt, 0.0_rt, 0.0_rt, 0};
	}

	ISM::Rate net{0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
	// add sign * r, with the dependence on ne chained through n_e^max
	const auto add = [&](const ISM::Rate &r, amrex::Real sign) {
		net.val += sign * r.val;
		net.dT += sign * (r.dT + r.dne * ne.dT);
		net.dnHI += sign * (r.dnHI + r.dne * ne.dnHI);
		net.dnHp += sign * (r.dnHp + r.dne * ne.dnHp);
	};
	if (lya_cooling_switch != 0) {
		add(ISM::LyaCooling(n_HI, ne.ne, T), -1.0_rt);
	}
	if (nebular_cooling_switch != 0) {
		add(ISM::NebularCooling(n_Hp, ne.ne, T, Zp), -1.0_rt);
	}
	if (fine_structure_cooling_switch != 0) {
		add(ISM::CIICooling(n_HI, n_Hp, ne.ne, T, Zp), -1.0_rt);
		add(ISM::OICooling(n_HI, ne.ne, T, Zp), -1.0_rt);
	}
	if (grain_recombination_cooling_switch != 0) {
		add(ISM::GrainRecombinationCooling(n_HI, n_Hp, ne.ne, T, G0, Zp), -1.0_rt);
	}
	if (pe_heating_switch != 0) {
		add(ISM::PEHeating(n_HI, n_Hp, ne.ne, T, G0, Zp), 1.0_rt);
	}
	if (cr_heating_switch != 0) {
		add(ISM::CRHeating(n_HI, cr_ionization_rate_HI, 0.0_rt), 1.0_rt); // y_He = 0 until the network has helium
	}
	return net;
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
	const amrex::Real collisional_ionization_cooling_rate =
	    (energy_switch != 0 && collisional_ionization_switch != 0) ? RydbergEnergy * collisional_ionization_term : 0.0_rt;
	const amrex::Real photoheating_rate =
	    (energy_switch != 0 && photoheating_switch != 0) ? get_ionization_heating_coefficient(state.T) * photoionization_term : 0.0_rt;
	amrex::Real energy_source = 0.0_rt;
	energy_source += photoheating_rate;
	energy_source -= recombination_cooling_rate;
	energy_source -= ion_ff_cooling_rate;
	energy_source -= collisional_ionization_cooling_rate;
	energy_source += get_ism_energy_terms(state).val;
	ydot(net_ienuc) = energy_source / state.rho;

	amrex::Real chat_dust_kappa_rho = 0.0_rt;
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
	chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
#endif
	const amrex::Real dust_absorption_term = chat_dust_kappa_rho * n_gamma;

	ydot(i_ngamma) = -photoionization_term - dust_absorption_term;
#ifndef SKIP_PHOTOCHEMFLUX
	ydot(i_ngamma_flux) = -chat_sigma_photo * n_HI * n_gamma_flux - chat_dust_kappa_rho * n_gamma_flux;
#endif
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
	ydot(i_dust) = dust_absorption_term * get_ionizing_band_photon_energy();
#else
	amrex::ignore_unused(dust_absorption_term);
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

	const amrex::Real d_collisional_ionization_term_dT = collisional_ionization_switch * n_e * n_HI * d_sigma_coll_dT;
	const amrex::Real d_recombination_term_dT = recombination_switch * (d_alpha_rec_dT * n_e * n_HII);
	const amrex::Real d_recombination_cooling_dT = recombination_cooling_switch * (d_Lambda_rec_dT * n_e * n_HII);
	const amrex::Real d_ion_ff_cooling_dT = ion_ff_cooling_switch * (d_Lambda_ff_dT * n_e * n_HII);

	const amrex::Real dTde = 1 / state.dedT;
	const auto ism = get_ism_energy_terms(state);

	amrex::Real chat_dust_kappa_rho = 0.0_rt;
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
	chat_dust_kappa_rho = state.c_hat * state.dust_kappa * state.rho;
#endif

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
	jac(net_ienuc, iH) = (energy_switch * (photoheating_switch * Gamma_photo * chat_sigma_photo * n_gamma -
					       RydbergEnergy * collisional_ionization_switch * n_e * sigma_coll) +
			      ism.dnHI) /
			     state.rho;
	jac(net_ienuc, iHp) =
	    (energy_switch * (-recombination_cooling_switch * Lambda_rec * n_e - ion_ff_cooling_switch * Lambda_ff * n_e) + ism.dnHp) / state.rho;
	jac(net_ienuc, net_ienuc) =
	    (energy_switch * (-d_recombination_cooling_dT - d_ion_ff_cooling_dT - RydbergEnergy * d_collisional_ionization_term_dT) + ism.dT) * dTde /
	    state.rho;
	jac(net_ienuc, i_ngamma) = energy_switch * (photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) * n_gamma_active / state.rho;

	jac(i_ngamma, ie) = 0.0_rt;
	jac(i_ngamma, iH) = -chat_sigma_photo * n_gamma;
	jac(i_ngamma, iHp) = 0.0_rt;
	jac(i_ngamma, net_ienuc) = 0.0_rt;
	jac(i_ngamma, i_ngamma) = -n_gamma_active * (chat_sigma_photo * n_HI + chat_dust_kappa_rho);

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
	jac(i_ngamma_flux, i_ngamma_flux) = -n_HI * chat_sigma_photo - chat_dust_kappa_rho;
#endif

#ifdef DUST_CHEMICAL_BAND_ABSORPTION
	jac(i_dust, i_ngamma) = n_gamma_active * chat_dust_kappa_rho * get_ionizing_band_photon_energy();
#else
	amrex::ignore_unused(chat_dust_kappa_rho);
#endif
}

#endif
