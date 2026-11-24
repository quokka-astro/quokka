// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef NETWORKS_PHOTOIONIZATION_H_CASEB_HANDWRITTEN_NETWORK_HPP_
#define NETWORKS_PHOTOIONIZATION_H_CASEB_HANDWRITTEN_NETWORK_HPP_
/// \file network.hpp
/// \brief The hand-written case B photoionization network of hydrogen.

#include <cmath>

#include "AMReX.H"
#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "util/fundamental_constants.hpp"

namespace quokka::networks
{

using amrex::literals::operator""_rt;

/// The hand-written case B photoionization network. Each chemical band integrates the photon number density and a flux
/// attenuation factor.
struct PhotoionizationHCaseBHandwritten {
	using Real = amrex::Real;

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

	/// one ionizing band, 13.6-26 eV, with a flat photon-number spectrum
	static constexpr int NumChemBands = 1;
	static constexpr int NumThermalBands = 0;
	/// the photon-number spectrum index alpha of each chemical band, n(E) ~ E^(alpha-2)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBandsPowerLawIndex() -> amrex::GpuArray<double, NumChemBands> { return {{0.0}}; }
	/// the chemical band edges [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1> { return {{13.6, 26.0}}; }

	/// whether each chemical band also integrates a flux attenuation factor
	static constexpr bool integrate_rad_flux = true;

	/// The runtime parameters, with Microphysics' names and defaults. Kernels take a copy by value.
	struct Params {
		int recombination_switch = 1;
		int recombination_temperature_dependent = 1;
		int collisional_ionization_switch = 1;
		int energy_switch = 1;
		int photoheating_switch = 1;
		int KI_heating_switch = 1;
		int recombination_cooling_switch = 1;
		int KI_cooling_switch = 1;
		int ion_ff_cooling_switch = 1;
		int allow_mixed_cell_cooling = 1;
		Real mixed_cell_lb = 0.01;
		Real mixed_cell_ub = 0.99;
		/// the floor of the species after a burn
		Real small_x = 1.e-30;

		/// Read the parameters from the `network` prefix.
		static auto Read() -> Params
		{
			Params p{};
			const amrex::ParmParse pp("network");
			pp.query("recombination_switch", p.recombination_switch);
			pp.query("recombination_temperature_dependent", p.recombination_temperature_dependent);
			pp.query("collisional_ionization_switch", p.collisional_ionization_switch);
			pp.query("energy_switch", p.energy_switch);
			pp.query("photoheating_switch", p.photoheating_switch);
			pp.query("KI_heating_switch", p.KI_heating_switch);
			pp.query("recombination_cooling_switch", p.recombination_cooling_switch);
			pp.query("KI_cooling_switch", p.KI_cooling_switch);
			pp.query("ion_ff_cooling_switch", p.ion_ff_cooling_switch);
			pp.query("allow_mixed_cell_cooling", p.allow_mixed_cell_cooling);
			pp.query("mixed_cell_lb", p.mixed_cell_lb);
			pp.query("mixed_cell_ub", p.mixed_cell_ub);
			pp.query("small_x", p.small_x);
			return p;
		}
	};

	/// The parameters of this run, read on first use. Host only.
	static auto Parameters() -> const Params &
	{
		static const Params params = Params::Read();
		return params;
	}

	static constexpr int ie = e;
	static constexpr int iH = H;
	static constexpr int iHp = H_p;
	static constexpr int ienuc = NumSpec;
	static constexpr int i_ngamma = ienuc + 1;
	static constexpr int i_ngamma_flux = ienuc + 2;

	static constexpr amrex::Real RydbergEnergy = 13.6_rt * C::ev2erg;

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_photoionization_coefficient(Real /*T*/) -> Real { return 3.2385396673401957e-18_rt; }

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_ionizing_band_photon_energy() -> Real
	{
		auto const ev_bounds = ChemBands();
		const amrex::Real ev_low = ev_bounds[0];
		const amrex::Real ev_high = ev_bounds[1];
		const amrex::Real alpha = ChemBandsPowerLawIndex()[0];

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

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_recombination_coefficient(Real T, const Params &p) -> Real
	{
		const amrex::Real coeff = 2.63e-13_rt;
		if (p.recombination_temperature_dependent == 0) {
			return coeff;
		}
		return coeff * std::pow(T / 1.0e4, -0.69999999999999996);
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto dRecombination_coefficient_dT(Real T, const Params &p) -> Real
	{
		if (p.recombination_temperature_dependent == 0) {
			return 0.0_rt;
		}
		return -2.63e-13 * 0.7 * std::pow(T / 1.0e4, -1.69999999999999996) / 1.0e4;
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_collisional_ionization_coefficient(Real T) -> Real
	{
		// Krumholz et al. 2007
		return 5.84e-11 * std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T));
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto dCollisional_ionization_coefficient_dT(Real T) -> Real
	{
		return 5.84e-11 * 0.5 / std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T)) +
		       5.84e-11 * std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T)) * 2.18e-11_rt / (C::k_B * T * T);
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_ionization_heating_coefficient(Real T) -> Real
	{
		amrex::ignore_unused(T);
		return amrex::max(get_ionizing_band_photon_energy() - RydbergEnergy, 0.0_rt);
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_KI_heating_coefficient(Real /*T*/) -> Real { return 2.0e-26; }

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_KI_cooling_coefficient(Real T) -> Real
	{
		// Fix KI cooling function based on Vazquez-Semadeni et al. (2007).
		return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) + 1.4e-2 * std::sqrt(T) * std::exp(-92.0 / T));
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto dKI_cooling_coefficient_dT(Real T) -> Real
	{
		return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) * 118400.0 / ((T + 1.0e3) * (T + 1.0e3)) +
				  1.4e-2 * (0.5 / std::sqrt(T) * std::exp(-92.0 / T) + std::sqrt(T) * std::exp(-92.0 / T) * 92.0 / (T * T)));
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_recombination_cooling_coefficient(Real T) -> Real
	{
		// Krumholz et al. 2007
		if (T < 100.0_rt) {
			return 0.0_rt;
		}
		return 6.1e-10 * C::k_B * T * std::pow(T, -0.89);
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto dRecombination_cooling_coefficient_dT(Real T) -> Real
	{
		if (T < 100.0_rt) {
			return 0.0_rt;
		}
		return 6.1e-10 * C::k_B * (std::pow(T, -0.89) + T * (-0.89) * std::pow(T, -1.89));
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_cle_term(Real T) -> Real
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

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto get_ion_ff_cooling_coefficient(Real T) -> Real
	{
		const amrex::Real cle_term = get_cle_term(T);
		return 1.3 * 1.427e-27 * std::sqrt(T) + cle_term; // Frazer and Heitsch 2019
	}

	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto d_ion_ff_cooling_coefficient_dT(Real T) -> Real
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

	template <typename BurnT, typename Vector> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void rhs(const BurnT &state, const Params &p, Vector &ydot)
	{
		const int recombination_switch = p.recombination_switch;
		const int collisional_ionization_switch = p.collisional_ionization_switch;
		const int energy_switch = p.energy_switch;
		const int photoheating_switch = p.photoheating_switch;
		const int KI_heating_switch = p.KI_heating_switch;
		const int recombination_cooling_switch = p.recombination_cooling_switch;
		const int KI_cooling_switch = p.KI_cooling_switch;
		const int ion_ff_cooling_switch = p.ion_ff_cooling_switch;
		const int allow_mixed_cell_cooling = p.allow_mixed_cell_cooling;
		const Real mixed_cell_lb = p.mixed_cell_lb;
		const Real mixed_cell_ub = p.mixed_cell_ub;

		const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
		const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
		const amrex::Real sigma_coll = (collisional_ionization_switch != 0) ? get_collisional_ionization_coefficient(state.T) : 0.0_rt;

		const amrex::Real n_e = state.xn[ie];
		const amrex::Real n_HI = state.xn[iH];
		const amrex::Real n_HII = state.xn[iHp];
		const amrex::Real n_gamma =
		    amrex::max(state.rn[0], 0.0_rt); // clamp at zero: prevents sign reversal if VODE steps rn[0] slightly negative within tolerance

		int molecular_process_switch = 1;
		const amrex::Real n_H_total = n_HI + n_HII;
		const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
		if ((allow_mixed_cell_cooling == 0) && (xHII > mixed_cell_lb) && (xHII < mixed_cell_ub)) {
			molecular_process_switch = 0;
		}

		const amrex::Real photoionization_term = chat_sigma_photo * n_HI * n_gamma;
		const amrex::Real collisional_ionization_term = (collisional_ionization_switch != 0) ? n_e * n_HI * sigma_coll : 0.0_rt;
		const amrex::Real recombination_term = (recombination_switch != 0) ? get_recombination_coefficient(state.T, p) * n_e * n_HII : 0.0_rt;

		ydot[ie] = photoionization_term + collisional_ionization_term - recombination_term;
		ydot[iH] = -ydot[ie];
		ydot[iHp] = ydot[ie];
		ydot[ienuc] = 0.0_rt;
		const amrex::Real recombination_cooling_rate =
		    (energy_switch != 0 && recombination_cooling_switch != 0) ? get_recombination_cooling_coefficient(state.T) * n_e * n_HII : 0.0_rt;
		const amrex::Real ion_ff_cooling_rate =
		    (energy_switch != 0 && ion_ff_cooling_switch != 0) ? get_ion_ff_cooling_coefficient(state.T) * n_e * n_HII : 0.0_rt;
		const amrex::Real KI_heating_rate =
		    (energy_switch != 0 && molecular_process_switch != 0 && KI_heating_switch != 0) ? get_KI_heating_coefficient(state.T) * n_HI : 0.0_rt;
		const amrex::Real KI_cooling_rate = (energy_switch != 0 && molecular_process_switch != 0 && KI_cooling_switch != 0)
							? get_KI_cooling_coefficient(state.T) * n_HI * n_HI
							: 0.0_rt;
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
		ydot[ienuc] = energy_source / state.rho;
		ydot[i_ngamma] = -photoionization_term;
		if constexpr (BurnT::vars_per_group == 2) {
			const amrex::Real n_gamma_flux = state.rn[1];
			ydot[i_ngamma_flux] = -chat_sigma_photo * n_HI * n_gamma_flux;
		}
	}

	/// The caller zeroes jac first, so only the nonzero entries are written.
	template <typename BurnT, typename MatrixType>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void jac(const BurnT &state, const Params &p, MatrixType &jac)
	{
		const int recombination_switch = p.recombination_switch;
		const int collisional_ionization_switch = p.collisional_ionization_switch;
		const int energy_switch = p.energy_switch;
		const int photoheating_switch = p.photoheating_switch;
		const int KI_heating_switch = p.KI_heating_switch;
		const int recombination_cooling_switch = p.recombination_cooling_switch;
		const int KI_cooling_switch = p.KI_cooling_switch;
		const int ion_ff_cooling_switch = p.ion_ff_cooling_switch;
		const int allow_mixed_cell_cooling = p.allow_mixed_cell_cooling;
		const Real mixed_cell_lb = p.mixed_cell_lb;
		const Real mixed_cell_ub = p.mixed_cell_ub;

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
			alpha_rec = get_recombination_coefficient(state.T, p);
			d_alpha_rec_dT = dRecombination_coefficient_dT(state.T, p);
		}

		const amrex::Real Gamma_photo = ((energy_switch != 0) && (photoheating_switch != 0)) ? get_ionization_heating_coefficient(state.T) : 0.0_rt;
		const amrex::Real Gamma_KI = ((energy_switch != 0) && (KI_heating_switch != 0)) ? get_KI_heating_coefficient(state.T) : 0.0_rt;
		const amrex::Real Lambda_KI = ((energy_switch != 0) && (KI_cooling_switch != 0)) ? get_KI_cooling_coefficient(state.T) : 0.0_rt;
		const amrex::Real d_Lambda_KI_dT = ((energy_switch != 0) && (KI_cooling_switch != 0)) ? dKI_cooling_coefficient_dT(state.T) : 0.0_rt;
		const amrex::Real Lambda_rec =
		    ((energy_switch != 0) && (recombination_cooling_switch != 0)) ? get_recombination_cooling_coefficient(state.T) : 0.0_rt;
		const amrex::Real d_Lambda_rec_dT =
		    ((energy_switch != 0) && (recombination_cooling_switch != 0)) ? dRecombination_cooling_coefficient_dT(state.T) : 0.0_rt;
		const amrex::Real Lambda_ff = ((energy_switch != 0) && (ion_ff_cooling_switch != 0)) ? get_ion_ff_cooling_coefficient(state.T) : 0.0_rt;
		const amrex::Real d_Lambda_ff_dT = ((energy_switch != 0) && (ion_ff_cooling_switch != 0)) ? d_ion_ff_cooling_coefficient_dT(state.T) : 0.0_rt;

		const amrex::Real n_e = state.xn[ie];
		const amrex::Real n_HI = state.xn[iH];
		const amrex::Real n_HII = state.xn[iHp];
		const amrex::Real n_gamma =
		    amrex::max(state.rn[0], 0.0_rt); // clamp at zero: prevents sign reversal if VODE steps rn[0] slightly negative within tolerance
		// Gate for column-5 Jacobian entries: when rn[0] < 0 the clamp makes the RHS
		// independent of rn[0], so all derivatives w.r.t. rn[0] are zero.
		const amrex::Real n_gamma_active = (state.rn[0] >= 0.0_rt) ? 1.0_rt : 0.0_rt;

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

		jac(ie, ie) = (collisional_ionization_switch * n_HI * sigma_coll - recombination_switch * alpha_rec * n_HII);
		jac(ie, iH) = (chat_sigma_photo * n_gamma + collisional_ionization_switch * n_e * sigma_coll);
		jac(ie, iHp) = -recombination_switch * alpha_rec * n_e;
		jac(ie, ienuc) = (d_collisional_ionization_term_dT - d_recombination_term_dT) * dTde;
		jac(ie, i_ngamma) = n_gamma_active * chat_sigma_photo * n_HI;

		jac(iH, ie) = -jac(ie, ie);
		jac(iH, iH) = -jac(ie, iH);
		jac(iH, iHp) = -jac(ie, iHp);
		jac(iH, ienuc) = -jac(ie, ienuc);
		jac(iH, i_ngamma) = -jac(ie, i_ngamma);
		jac(iHp, ie) = jac(ie, ie);
		jac(iHp, iH) = jac(ie, iH);
		jac(iHp, iHp) = jac(ie, iHp);
		jac(iHp, ienuc) = jac(ie, ienuc);
		jac(iHp, i_ngamma) = jac(ie, i_ngamma);

		jac(ienuc, ie) = energy_switch *
				 (-recombination_cooling_switch * Lambda_rec * n_HII - ion_ff_cooling_switch * Lambda_ff * n_HII -
				  RydbergEnergy * collisional_ionization_switch * n_HI * sigma_coll) /
				 state.rho;
		jac(ienuc, iH) = energy_switch *
				 (photoheating_switch * Gamma_photo * chat_sigma_photo * n_gamma + molecular_process_switch * KI_heating_switch * Gamma_KI -
				  2.0_rt * molecular_process_switch * KI_cooling_switch * Lambda_KI * n_HI -
				  RydbergEnergy * collisional_ionization_switch * n_e * sigma_coll) /
				 state.rho;
		jac(ienuc, iHp) = energy_switch * (-recombination_cooling_switch * Lambda_rec * n_e - ion_ff_cooling_switch * Lambda_ff * n_e) / state.rho;
		jac(ienuc, ienuc) = energy_switch *
				    (-d_recombination_cooling_dT - d_KI_cooling_dT - d_ion_ff_cooling_dT - RydbergEnergy * d_collisional_ionization_term_dT) *
				    dTde / state.rho;
		jac(ienuc, i_ngamma) = energy_switch * (photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) * n_gamma_active / state.rho;

		jac(i_ngamma, ie) = 0.0_rt;
		jac(i_ngamma, iH) = -chat_sigma_photo * n_gamma;
		jac(i_ngamma, iHp) = 0.0_rt;
		jac(i_ngamma, ienuc) = 0.0_rt;
		jac(i_ngamma, i_ngamma) = -n_gamma_active * chat_sigma_photo * n_HI;

		if constexpr (BurnT::vars_per_group == 2) {
			const amrex::Real n_gamma_flux = state.rn[1];
			jac(ie, i_ngamma_flux) = 0.0_rt;
			jac(iH, i_ngamma_flux) = -jac(ie, i_ngamma_flux);
			jac(iHp, i_ngamma_flux) = jac(ie, i_ngamma_flux);
			jac(ienuc, i_ngamma_flux) = 0.0_rt;
			jac(i_ngamma, i_ngamma_flux) = 0.0_rt;
			jac(i_ngamma_flux, ie) = 0.0_rt;
			jac(i_ngamma_flux, iH) = -chat_sigma_photo * n_gamma_flux;
			jac(i_ngamma_flux, iHp) = 0.0_rt;
			jac(i_ngamma_flux, ienuc) = 0.0_rt;
			jac(i_ngamma_flux, i_ngamma) = 0.0_rt;
			jac(i_ngamma_flux, i_ngamma_flux) = -n_HI * chat_sigma_photo;
		}
	}
};

/// PhotoionizationHCaseBHandwritten without the flux equations: the flux is attenuated algebraically after the burn,
/// by the change of the photon number density.
struct PhotoionizationHCaseBHandwrittenNoFlux : PhotoionizationHCaseBHandwritten {
	static constexpr bool integrate_rad_flux = false;
};

} // namespace quokka::networks

#endif // NETWORKS_PHOTOIONIZATION_H_CASEB_HANDWRITTEN_NETWORK_HPP_
