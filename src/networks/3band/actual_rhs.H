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
//
// Radiation state is split into two disjoint blocks in y()/ydot, both starting after net_ienuc:
//   [net_ienuc+1, net_ienuc+NumChemRadEqs]: chem bands (here just band 2, the ionizing band), each
//     contributing a (photon number density, flux-attenuation factor) pair -- see rad_eq_offset.
//   [net_ienuc+NumChemRadEqs+1, net_ienuc+NumChemRadEqs+NumThermalBands]: thermal bands (bands 0, 1: IR
//     and optical), each contributing a single energy-density equation, no flux slot -- their
//     emission from chemical/recombination processes is assumed isotropic, so no net flux is ever
//     added -- see thermal_eq_offset. Excluded from the burner's adaptive timestep/error control
//     (see rad_eq_in_error_control in rosenbrock_integrator.H), same as a chem band's flux slot.
//
// Band 0 = IR (dust-reprocessed, thermal group -- normal RadSystem opacity/Planck coupling)
// Band 1 = optical / non-ionizing UV (thermal group -- ditto)
// Band 2 = photoionizing (> 13.6 eV, the sole chem band: NumChemBands = 1, NumRadGroups = 3)
// Only band 2 (the chem band) couples to hydrogen photoionization below and has real absorption
// physics. Bands 0 and 1 (thermal groups) have no reaction/absorption physics of their own in
// this network; they only receive a constant placeholder energy-density injection rate, standing
// in for recombination-line/continuum feedback.
constexpr int IRBand = 0;
constexpr int OpticalBand = 1;
constexpr int IonizingBand = 2;

// band is chem-band-local (0 .. NumChemBands-1).
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto rad_eq_offset(int band) -> int { return net_ienuc + 1 + MicrophysicsNumChemRadVarsPerGroup * band; }

// band is thermal-band-local (0 .. NumThermalBands-1), i.e. the same indexing as IRBand/OpticalBand.
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto thermal_eq_offset(int band) -> int { return net_ienuc + NumChemRadEqs + 1 + band; }

// IonizingBand (global band index 2) is the sole chem band; its chem-band-local index (for
// rad_eq_offset/rn[]) is IonizingBand - NumThermalBands. IRBand/OpticalBand (global indices 0, 1)
// are already thermal-band-local (bands 0, 1 are the leading NumThermalBands groups), so they're
// used directly with thermal_eq_offset.
constexpr int IonizingChemBand = IonizingBand - NumThermalBands;

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 1.5e-18_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_coefficient(Real T) -> Real
{
	if (recombination_temperature_dependent == 0) {
		return 2.6e-13_rt;
	}
	return 2.6e-13 * (1.0 + std::pow(T / 1.0e4, -0.7) - 1.0);
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
	return 5.84e-11 * std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dCollisional_ionization_coefficient_dT(Real T) -> Real
{
	return 5.84e-11 * 0.5 / std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T)) +
	       5.84e-11 * std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T)) * 2.18e-11_rt / (C::k_B * T * T);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionization_heating_coefficient(Real T) -> Real { return 6.4e-12_rt; }

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

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ion_ff_cooling_coefficient(Real T) -> Real { return 1.4e-27 * std::sqrt(T) + 1.0e-19 * std::exp(-118348.0 / T); }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto d_ion_ff_cooling_coefficient_dT(Real T) -> Real
{
	return 1.4e-27 * 0.5 / std::sqrt(T) + 1.0e-19 * std::exp(-118348.0 / T) * 118348.0 / (T * T);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_rhs(const burn_t &state, Array1D<Real, 1, INT_NEQS> &ydot)
{
	const amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;

	const amrex::Real n_e = state.xn[0];
	const amrex::Real n_HI = state.xn[1];
	const amrex::Real n_HII = state.xn[2];
	// clamp at zero: prevents sign reversal if VODE steps rn[.] slightly negative within tolerance
	const amrex::Real n_gamma_ion = amrex::max(state.rn[MicrophysicsNumChemRadVarsPerGroup * IonizingChemBand], 0.0_rt);
	const amrex::Real n_gamma_ion_flux = state.rn[1 + MicrophysicsNumChemRadVarsPerGroup * IonizingChemBand];

	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
	if ((allow_mixed_cell_cooling == 0) && (xHII > mixed_cell_lb) && (xHII < mixed_cell_ub)) {
		molecular_process_switch = 0;
	}

	const amrex::Real photoionization_term = chat_sigma_photo * n_HI * n_gamma_ion;
	const amrex::Real collisional_ionization_term =
	    (collisional_ionization_switch != 0) ? n_e * n_HI * get_collisional_ionization_coefficient(state.T) : 0.0_rt;
	const amrex::Real recombination_term = (recombination_switch != 0) ? get_recombination_coefficient(state.T) * n_e * n_HII : 0.0_rt;

	ydot(1) = photoionization_term + collisional_ionization_term - recombination_term;
	ydot(2) = -ydot(1);
	ydot(3) = ydot(1);
	ydot(net_ienuc) = 0.0_rt;
	if (energy_switch != 0) {
		amrex::Real energy_source = 0.0_rt;
		if (photoheating_switch != 0) {
			energy_source += get_ionization_heating_coefficient(state.T) * photoionization_term;
		}
		if (molecular_process_switch != 0) {
			if (KI_cooling_switch != 0) {
				energy_source -= get_KI_cooling_coefficient(state.T) * n_HI * n_HI;
			}
		}
		if (recombination_cooling_switch != 0) {
			energy_source -= get_recombination_cooling_coefficient(state.T) * n_e * n_HII;
		}
		if (ion_ff_cooling_switch != 0) {
			energy_source -= get_ion_ff_cooling_coefficient(state.T) * n_e * n_HII;
		}
		if (KI_heating_switch != 0) {
			energy_source += get_KI_heating_coefficient(state.T) * n_HI;
		}
		ydot(net_ienuc) = energy_source / state.rho;
	}

	// Band 0 (IR) and band 1 (optical): chemically inert (no absorption), but receive a
	// constant placeholder energy-density injection rate standing in for recombination feedback.
	// No flux equation exists for thermal bands (isotropic emission, no net flux).
	const amrex::Real feedback_rate = (recombination_feedback_switch != 0) ? recombination_feedback_rate : 0.0_rt;
	ydot(thermal_eq_offset(IRBand)) = feedback_rate;
	ydot(thermal_eq_offset(OpticalBand)) = feedback_rate;

	// Band 2 (ionizing): photoionization absorption depletes both photon number and flux.
	ydot(rad_eq_offset(IonizingChemBand)) = -photoionization_term;
	ydot(rad_eq_offset(IonizingChemBand) + 1) = -chat_sigma_photo * n_HI * n_gamma_ion_flux;
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
	const amrex::Real Lambda_ion_ff = ((energy_switch != 0) && (ion_ff_cooling_switch != 0)) ? get_ion_ff_cooling_coefficient(state.T) : 0.0_rt;
	const amrex::Real d_Lambda_ion_ff_dT = ((energy_switch != 0) && (ion_ff_cooling_switch != 0)) ? d_ion_ff_cooling_coefficient_dT(state.T) : 0.0_rt;

	const amrex::Real n_e = state.xn[0];
	const amrex::Real n_HI = state.xn[1];
	const amrex::Real n_HII = state.xn[2];
	const int ion_n_idx = MicrophysicsNumChemRadVarsPerGroup * IonizingChemBand;
	const int ion_f_idx = ion_n_idx + 1;
	const amrex::Real n_gamma =
	    amrex::max(state.rn[ion_n_idx], 0.0_rt); // clamp at zero: prevents sign reversal if VODE steps rn[.] slightly negative within tolerance
	// Gate for the ionizing-band photon-number Jacobian entries: when rn[ion_n_idx] < 0 the clamp
	// makes the RHS independent of it, so all derivatives w.r.t. it are zero.
	const amrex::Real n_gamma_active = (state.rn[ion_n_idx] >= 0.0_rt) ? 1.0_rt : 0.0_rt;
	const amrex::Real n_gamma_flux = state.rn[ion_f_idx];

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
	const amrex::Real d_ion_ff_cooling_dT = ion_ff_cooling_switch * (d_Lambda_ion_ff_dT * n_e * n_HII);

	const amrex::Real dTde = 1 / state.dedT;

	jac.zero();

	// Columns/rows for the IR (band 0) and optical (band 1) thermal radiation equations are
	// omitted: their RHS is a constant feedback rate, independent of all state variables
	// (including each other and the ionizing band), so those rows/columns stay zero (set by
	// jac.zero() above).

	const int ion_n_col = rad_eq_offset(IonizingChemBand);
	const int ion_f_col = ion_n_col + 1;

	jac(1, 1) = (collisional_ionization_switch * n_HI * sigma_coll - recombination_switch * alpha_rec * n_HII);
	jac(1, 2) = (chat_sigma_photo * n_gamma + collisional_ionization_switch * n_e * sigma_coll);
	jac(1, 3) = -recombination_switch * alpha_rec * n_e;
	jac(1, net_ienuc) = (d_collisional_ionization_term_dT - d_recombination_term_dT) * dTde;
	jac(1, ion_n_col) = n_gamma_active * chat_sigma_photo * n_HI;
	jac(1, ion_f_col) = 0.0_rt;

	jac(2, 1) = -jac(1, 1);
	jac(2, 2) = -jac(1, 2);
	jac(2, 3) = -jac(1, 3);
	jac(2, net_ienuc) = -jac(1, net_ienuc);
	jac(2, ion_n_col) = -jac(1, ion_n_col);
	jac(2, ion_f_col) = -jac(1, ion_f_col);
	jac(3, 1) = jac(1, 1);
	jac(3, 2) = jac(1, 2);
	jac(3, 3) = jac(1, 3);
	jac(3, net_ienuc) = jac(1, net_ienuc);
	jac(3, ion_n_col) = jac(1, ion_n_col);
	jac(3, ion_f_col) = jac(1, ion_f_col);

	jac(net_ienuc, 1) = energy_switch * (-recombination_cooling_switch * Lambda_rec * n_HII - ion_ff_cooling_switch * Lambda_ion_ff * n_HII) / state.rho;
	jac(net_ienuc, 2) = energy_switch *
			    (photoheating_switch * Gamma_photo * chat_sigma_photo * n_gamma + KI_heating_switch * Gamma_KI -
			     2.0_rt * molecular_process_switch * KI_cooling_switch * Lambda_KI * n_HI) /
			    state.rho;
	jac(net_ienuc, 3) = energy_switch * (-recombination_cooling_switch * Lambda_rec * n_e - ion_ff_cooling_switch * Lambda_ion_ff * n_e) / state.rho;
	jac(net_ienuc, net_ienuc) = energy_switch * (-d_recombination_cooling_dT - d_KI_cooling_dT - d_ion_ff_cooling_dT) * dTde / state.rho;
	jac(net_ienuc, ion_n_col) = energy_switch * (photoheating_switch * Gamma_photo * chat_sigma_photo * n_HI) * n_gamma_active / state.rho;
	jac(net_ienuc, ion_f_col) = 0.0_rt;

	jac(ion_n_col, 1) = 0.0_rt;
	jac(ion_n_col, 2) = -chat_sigma_photo * n_gamma;
	jac(ion_n_col, 3) = 0.0_rt;
	jac(ion_n_col, net_ienuc) = 0.0_rt;
	jac(ion_n_col, ion_n_col) = -n_gamma_active * chat_sigma_photo * n_HI;
	jac(ion_n_col, ion_f_col) = 0.0_rt;

	jac(ion_f_col, 1) = 0.0_rt;
	jac(ion_f_col, 2) = -chat_sigma_photo * n_gamma_flux;
	jac(ion_f_col, 3) = 0.0_rt;
	jac(ion_f_col, net_ienuc) = 0.0_rt;
	jac(ion_f_col, ion_n_col) = 0.0_rt;
	jac(ion_f_col, ion_f_col) = -n_HI * chat_sigma_photo;
}

#endif
