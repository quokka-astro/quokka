#ifndef NETWORK_PHOTOIONIZATION_H_CASEB_HANDWRITTEN_HPP_
#define NETWORK_PHOTOIONIZATION_H_CASEB_HANDWRITTEN_HPP_
/// \file network.hpp
/// \brief Hydrogen photoionization network (case B): H, H+, e- and one ionizing photon band, with photoheating,
/// recombination, collisional ionization and line/free-free cooling.
///
/// Network contract used by src/radiation/photochemistry.hpp: global NumSpec, Species::, spmasses, gammas,
/// NumChemBands, NumChemRadVarsPerGroup, NumChemRadEqs, NumThermalBands, ChemBandsHeader(), ChemBandsPowerLawIndex_;
/// and in namespace network: NumEqs, State, NetworkParams, network_rhs, network_jac.
/// Solver vector layout (0-based): species number densities, specific internal energy, then per chemical band the
/// photon number density (and the flux factor unless SKIP_PHOTOCHEMFLUX).

#include <cmath>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "photochemistry/PhotochemState.hpp"
#include "rosenbrock/dense_lu.hpp"
#include "util/fundamental_constants.hpp"

#ifdef AMREX_USE_HIP
#define QUOKKA_NETWORK_CONSTEXPR constexpr
#else
#define QUOKKA_NETWORK_CONSTEXPR
#endif

constexpr int NumSpec = 3;

namespace Species
{
enum NetworkSpecies { H = 0, H_p, e };
} // namespace Species

constexpr int NumChemBands = 1;
#ifdef SKIP_PHOTOCHEMFLUX
constexpr int NumChemRadVarsPerGroup = 1;
#else
constexpr int NumChemRadVarsPerGroup = 2;
#endif
constexpr int NumChemRadEqs = NumChemBands * NumChemRadVarsPerGroup;
constexpr int NumThermalBands = 0;

// photon-number spectrum index alpha in n(E) ~ E^(alpha-2) within each chemical band
constexpr double ChemBandsPowerLawIndex_ = 0;

/// chemical band edges [eV]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto ChemBandsHeader() -> amrex::GpuArray<double, NumChemBands + 1> { return {{13.6, 26.0}}; }

namespace
{
// particle masses [g] and adiabatic indices, ordered as Species (values from Microphysics EOS/photoionization/_parameters)
[[maybe_unused]] QUOKKA_NETWORK_CONSTEXPR AMREX_GPU_MANAGED amrex::Real spmasses[NumSpec] = {1.673773e-24, 1.6728620616289998e-24, 9.10938371e-28};
[[maybe_unused]] QUOKKA_NETWORK_CONSTEXPR AMREX_GPU_MANAGED amrex::Real gammas[NumSpec] = {5. / 3., 5. / 3., 5. / 3.};
} // namespace

namespace network
{
using amrex::Real;
using namespace amrex::literals; // NOLINT(google-build-using-namespace)

constexpr int NumEqs = NumSpec + 1 + NumChemRadEqs + NumThermalBands;
using State = quokka::photochemistry::State<NumSpec, NumChemRadEqs, NumThermalBands>;

// 0-based indices into the solver vector
constexpr int e_idx = static_cast<int>(Species::e);
constexpr int H_idx = static_cast<int>(Species::H);
constexpr int Hp_idx = static_cast<int>(Species::H_p);
constexpr int ie = e_idx;
constexpr int iH = H_idx;
constexpr int iHp = Hp_idx;
constexpr int net_ienuc = NumSpec;
constexpr int i_ngamma = net_ienuc + 1;
constexpr int i_ngamma_flux = net_ienuc + 2;

/// runtime switches, read from the "network" ParmParse prefix (defaults as in the former _parameters file)
struct NetworkParams {
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
	Real small_x = 1.0e-30; ///< floor for species number densities after the solve

	static auto read() -> NetworkParams
	{
		NetworkParams p{};
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

constexpr amrex::Real RydbergEnergy = 13.6_rt * C::ev2erg;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 3.2467944898988799e-18_rt; }

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

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_coefficient(Real T, int recombination_temperature_dependent) -> Real
{
	const amrex::Real coeff = 2.63e-13_rt;
	if (recombination_temperature_dependent == 0) {
		return coeff;
	}
	return coeff * std::pow(T / 1.0e4, -0.69999999999999996);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dRecombination_coefficient_dT(Real T, int recombination_temperature_dependent) -> Real
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void network_rhs(State const &state, NetworkParams const &p, rosenbrock::Vector<NumEqs> &ydot)
{
	[[maybe_unused]] const auto recombination_switch = p.recombination_switch;
	[[maybe_unused]] const auto collisional_ionization_switch = p.collisional_ionization_switch;
	[[maybe_unused]] const auto energy_switch = p.energy_switch;
	[[maybe_unused]] const auto photoheating_switch = p.photoheating_switch;
	[[maybe_unused]] const auto KI_heating_switch = p.KI_heating_switch;
	[[maybe_unused]] const auto recombination_cooling_switch = p.recombination_cooling_switch;
	[[maybe_unused]] const auto KI_cooling_switch = p.KI_cooling_switch;
	[[maybe_unused]] const auto ion_ff_cooling_switch = p.ion_ff_cooling_switch;
	[[maybe_unused]] const auto allow_mixed_cell_cooling = p.allow_mixed_cell_cooling;
	[[maybe_unused]] const auto mixed_cell_lb = p.mixed_cell_lb;
	[[maybe_unused]] const auto mixed_cell_ub = p.mixed_cell_ub;
	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	const amrex::Real sigma_coll = (collisional_ionization_switch != 0) ? get_collisional_ionization_coefficient(state.T) : 0.0_rt;

	const amrex::Real n_e = state.xn[e_idx];
	const amrex::Real n_HI = state.xn[H_idx];
	const amrex::Real n_HII = state.xn[Hp_idx];
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
	const amrex::Real collisional_ionization_term = (collisional_ionization_switch != 0) ? n_e * n_HI * sigma_coll : 0.0_rt;
	const amrex::Real recombination_term =
	    (recombination_switch != 0) ? get_recombination_coefficient(state.T, p.recombination_temperature_dependent) * n_e * n_HII : 0.0_rt;

	ydot[ie] = photoionization_term + collisional_ionization_term - recombination_term;
	ydot[iH] = -ydot[ie];
	ydot[iHp] = ydot[ie];
	ydot[net_ienuc] = 0.0_rt;
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
	ydot[net_ienuc] = energy_source / state.rho;
	ydot[i_ngamma] = -photoionization_term;
#ifndef SKIP_PHOTOCHEMFLUX
	ydot[i_ngamma_flux] = -chat_sigma_photo * n_HI * n_gamma_flux;
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void network_jac(State const &state, NetworkParams const &p, rosenbrock::Matrix<NumEqs> &jac)
{
	[[maybe_unused]] const auto recombination_switch = p.recombination_switch;
	[[maybe_unused]] const auto collisional_ionization_switch = p.collisional_ionization_switch;
	[[maybe_unused]] const auto energy_switch = p.energy_switch;
	[[maybe_unused]] const auto photoheating_switch = p.photoheating_switch;
	[[maybe_unused]] const auto KI_heating_switch = p.KI_heating_switch;
	[[maybe_unused]] const auto recombination_cooling_switch = p.recombination_cooling_switch;
	[[maybe_unused]] const auto KI_cooling_switch = p.KI_cooling_switch;
	[[maybe_unused]] const auto ion_ff_cooling_switch = p.ion_ff_cooling_switch;
	[[maybe_unused]] const auto allow_mixed_cell_cooling = p.allow_mixed_cell_cooling;
	[[maybe_unused]] const auto mixed_cell_lb = p.mixed_cell_lb;
	[[maybe_unused]] const auto mixed_cell_ub = p.mixed_cell_ub;
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
		alpha_rec = get_recombination_coefficient(state.T, p.recombination_temperature_dependent);
		d_alpha_rec_dT = dRecombination_coefficient_dT(state.T, p.recombination_temperature_dependent);
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

	for (int j = 0; j < NumEqs; ++j) {
		for (int i = 0; i < NumEqs; ++i) {
			jac(i, j) = 0.0_rt;
		}
	}

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
	jac(net_ienuc, i_ngamma) = energy_switch * (photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) * n_gamma_active / state.rho;

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
}

} // namespace network

#endif
