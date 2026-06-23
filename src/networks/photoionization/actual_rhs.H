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
// Radiation index 0 (Index 4 in the state) = number density of photons
// Radiation index 1 (Index 5 in the state) = flux of photons normalized by the flux magnitude before the burn
// (therefore, it is 1.0 at the beginning of burn)

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto fast_sqrt_approx(Real x) -> Real
{
#if defined(AMREX_USE_CUDA) && AMREX_DEVICE_COMPILE
	return static_cast<Real>(sqrtf(static_cast<float>(x)));
#else
	return std::sqrt(x);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto fast_pow_approx(Real x, Real y) -> Real
{
#if defined(AMREX_USE_CUDA) && AMREX_DEVICE_COMPILE
	return static_cast<Real>(powf(static_cast<float>(x), static_cast<float>(y)));
#else
	return std::pow(x, y);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 1.5e-18_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_coefficient(Real T) -> Real
{
	if (recombination_temperature_dependent == 0) {
		return 2.6e-13_rt;
	}
	return 2.6e-13_rt * std::pow(T / 1.0e4_rt, -0.7_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dRecombination_coefficient_dT(Real T, Real alpha_rec) -> Real
{
	if (recombination_temperature_dependent == 0) {
		return 0.0_rt;
	}
	return -0.7_rt * alpha_rec / T;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_collisional_ionization_coefficient(Real T) -> Real
{
	return 5.84e-11 * std::sqrt(T) * std::exp(-2.18e-11_rt / (C::k_B * T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dCollisional_ionization_coefficient_dT(Real T, Real sigma_coll) -> Real
{
	return sigma_coll * (0.5_rt / T + 2.18e-11_rt / (C::k_B * T * T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionization_heating_coefficient(Real T) -> Real { return 6.4e-12_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_KI_heating_coefficient(Real T) -> Real { return 2.0e-26; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_KI_cooling_coefficient(Real T) -> Real
{
	// Fix KI cooling function based on Vazquez-Semadeni et al. (2007).
	return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) + 1.4e-2 * fast_sqrt_approx(T) * std::exp(-92.0 / T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dKI_cooling_coefficient_dT(Real T, Real exp_hot, Real sqrt_T, Real exp_cold) -> Real
{
	return 2.0e-26_rt * (1.0e7_rt * exp_hot * 118400.0_rt / ((T + 1.0e3_rt) * (T + 1.0e3_rt)) +
			     1.4e-2_rt * (0.5_rt * exp_cold / sqrt_T + sqrt_T * exp_cold * 92.0_rt / (T * T)));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_cooling_coefficient(Real T) -> Real
{
	if (T < 100.0_rt) {
		return 0.0_rt;
	}
	return 6.1e-10 * C::k_B * T * fast_pow_approx(T, -0.89);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto dRecombination_cooling_coefficient_dT(Real T) -> Real
{
	if (T < 100.0_rt) {
		return 0.0_rt;
	}
	const Real Lambda_rec = get_recombination_cooling_coefficient(T);
	return 0.11_rt * Lambda_rec / T;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ion_ff_cooling_coefficient(Real T) -> Real
{
	return 1.4e-27 * fast_sqrt_approx(T) + 1.0e-19 * std::exp(-118348.0 / T);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto d_ion_ff_cooling_coefficient_dT(Real T, Real sqrt_T, Real exp_term) -> Real
{
	return 1.4e-27_rt * 0.5_rt / sqrt_T + 1.0e-19_rt * exp_term * 118348.0_rt / (T * T);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_rhs(const burn_t &state, Array1D<Real, 1, INT_NEQS> &ydot)
{
	const Real T = state.T;
	const Real rho = state.rho;
	const amrex::Real sigma_photo = get_photoionization_coefficient(T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;

	const amrex::Real n_e = state.xn[0];
	const amrex::Real n_HI = state.xn[1];
	const amrex::Real n_HII = state.xn[2];
	const amrex::Real n_gamma =
	    amrex::max(state.rn[0], 0.0_rt); // clamp at zero: prevents sign reversal if VODE steps rn[0] slightly negative within tolerance
	const amrex::Real n_gamma_flux = state.rn[1];

	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
	if ((allow_mixed_cell_cooling == 0) && (xHII > mixed_cell_lb) && (xHII < mixed_cell_ub)) {
		molecular_process_switch = 0;
	}

	const amrex::Real photoionization_term = chat_sigma_photo * n_HI * n_gamma;
	const amrex::Real collisional_ionization_term = (collisional_ionization_switch != 0) ? n_e * n_HI * get_collisional_ionization_coefficient(T) : 0.0_rt;
	const amrex::Real recombination_term = (recombination_switch != 0) ? get_recombination_coefficient(T) * n_e * n_HII : 0.0_rt;

	ydot(1) = photoionization_term + collisional_ionization_term - recombination_term;
	ydot(2) = -ydot(1);
	ydot(3) = ydot(1);
	ydot(net_ienuc) = 0.0_rt;
	if (energy_switch != 0) {
		amrex::Real energy_source = 0.0_rt;
		if (photoheating_switch != 0) {
			energy_source += get_ionization_heating_coefficient(T) * photoionization_term;
		}
		if (molecular_process_switch != 0) {
			if (KI_heating_switch != 0) {
				energy_source += get_KI_heating_coefficient(T) * n_HI;
			}
			if (KI_cooling_switch != 0) {
				energy_source -= get_KI_cooling_coefficient(T) * n_HI * n_HI;
			}
		}
		if (recombination_cooling_switch != 0) {
			energy_source -= get_recombination_cooling_coefficient(T) * n_e * n_HII;
		}
		if (ion_ff_cooling_switch != 0) {
			energy_source -= get_ion_ff_cooling_coefficient(T) * n_e * n_HII;
		}
		ydot(net_ienuc) = energy_source / rho;
	}
	ydot(5) = -photoionization_term;
	ydot(6) = -chat_sigma_photo * n_HI * n_gamma_flux;
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{
	const Real T = state.T;
	const Real rho = state.rho;
	const amrex::Real sigma_photo = get_photoionization_coefficient(T);
	const amrex::Real chat_sigma_photo = state.c_hat * sigma_photo;
	amrex::Real sigma_coll = 0.0_rt;
	amrex::Real d_sigma_coll_dT = 0.0_rt;
	if (collisional_ionization_switch != 0) {
		sigma_coll = get_collisional_ionization_coefficient(T);
		d_sigma_coll_dT = dCollisional_ionization_coefficient_dT(T, sigma_coll);
	}
	amrex::Real alpha_rec = 0.0_rt;
	amrex::Real d_alpha_rec_dT = 0.0_rt;
	if (recombination_switch != 0) {
		alpha_rec = get_recombination_coefficient(T);
		d_alpha_rec_dT = dRecombination_coefficient_dT(T, alpha_rec);
	}

	const amrex::Real Gamma_photo = ((energy_switch != 0) && (photoheating_switch != 0)) ? get_ionization_heating_coefficient(T) : 0.0_rt;
	const amrex::Real Gamma_KI = ((energy_switch != 0) && (KI_heating_switch != 0)) ? get_KI_heating_coefficient(T) : 0.0_rt;
	amrex::Real Lambda_KI = 0.0_rt;
	amrex::Real d_Lambda_KI_dT = 0.0_rt;
	if ((energy_switch != 0) && (KI_cooling_switch != 0)) {
		const Real exp_hot = std::exp(-118400.0_rt / (T + 1.0e3_rt));
		const Real sqrt_T = fast_sqrt_approx(T);
		const Real exp_cold = std::exp(-92.0_rt / T);
		Lambda_KI = 2.0e-26_rt * (1.0e7_rt * exp_hot + 1.4e-2_rt * sqrt_T * exp_cold);
		d_Lambda_KI_dT = dKI_cooling_coefficient_dT(T, exp_hot, sqrt_T, exp_cold);
	}
	const amrex::Real Lambda_rec = ((energy_switch != 0) && (recombination_cooling_switch != 0)) ? get_recombination_cooling_coefficient(T) : 0.0_rt;
	const amrex::Real d_Lambda_rec_dT = ((energy_switch != 0) && (recombination_cooling_switch != 0)) ? dRecombination_cooling_coefficient_dT(T) : 0.0_rt;
	amrex::Real Lambda_ion_ff = 0.0_rt;
	amrex::Real d_Lambda_ion_ff_dT = 0.0_rt;
	if ((energy_switch != 0) && (ion_ff_cooling_switch != 0)) {
		const Real sqrt_T = fast_sqrt_approx(T);
		const Real exp_term = std::exp(-118348.0_rt / T);
		Lambda_ion_ff = 1.4e-27_rt * sqrt_T + 1.0e-19_rt * exp_term;
		d_Lambda_ion_ff_dT = d_ion_ff_cooling_coefficient_dT(T, sqrt_T, exp_term);
	}

	const amrex::Real n_e = state.xn[0];
	const amrex::Real n_HI = state.xn[1];
	const amrex::Real n_HII = state.xn[2];
	const amrex::Real n_gamma =
	    amrex::max(state.rn[0], 0.0_rt); // clamp at zero: prevents sign reversal if VODE steps rn[0] slightly negative within tolerance
	// Gate for column-5 Jacobian entries: when rn[0] < 0 the clamp makes the RHS
	// independent of rn[0], so all derivatives w.r.t. rn[0] are zero.
	const amrex::Real n_gamma_active = (state.rn[0] >= 0.0_rt) ? 1.0_rt : 0.0_rt;
	const amrex::Real n_gamma_flux = state.rn[1];

	int molecular_process_switch = 1;
	const amrex::Real n_H_total = n_HI + n_HII;
	const amrex::Real xHII = (n_H_total > 0.0_rt) ? (n_HII / n_H_total) : 0.0_rt;
	if ((allow_mixed_cell_cooling == 0) && (xHII > mixed_cell_lb) && (xHII < mixed_cell_ub)) {
		molecular_process_switch = 0;
	}

	const amrex::Real d_collisional_ionization_term_dT = n_e * n_HI * d_sigma_coll_dT;
	const amrex::Real d_recombination_term_dT = d_alpha_rec_dT * n_e * n_HII;
	const amrex::Real d_KI_cooling_dT = static_cast<Real>(molecular_process_switch) * d_Lambda_KI_dT * n_HI * n_HI;
	const amrex::Real d_recombination_cooling_dT = d_Lambda_rec_dT * n_e * n_HII;
	const amrex::Real d_ion_ff_cooling_dT = d_Lambda_ion_ff_dT * n_e * n_HII;

	const amrex::Real dTde = 1 / state.dedT;

	jac.zero();

	jac(1, 1) = n_HI * sigma_coll - alpha_rec * n_HII;
	jac(1, 2) = chat_sigma_photo * n_gamma + n_e * sigma_coll;
	jac(1, 3) = -alpha_rec * n_e;
	jac(1, net_ienuc) = (d_collisional_ionization_term_dT - d_recombination_term_dT) * dTde;
	jac(1, 5) = n_gamma_active * chat_sigma_photo * n_HI;
	jac(1, 6) = 0.0_rt;

	jac(2, 1) = -jac(1, 1);
	jac(2, 2) = -jac(1, 2);
	jac(2, 3) = -jac(1, 3);
	jac(2, net_ienuc) = -jac(1, net_ienuc);
	jac(2, 5) = -jac(1, 5);
	jac(2, 6) = -jac(1, 6);
	jac(3, 1) = jac(1, 1);
	jac(3, 2) = jac(1, 2);
	jac(3, 3) = jac(1, 3);
	jac(3, net_ienuc) = jac(1, net_ienuc);
	jac(3, 5) = jac(1, 5);
	jac(3, 6) = jac(1, 6);

	const Real inv_rho = 1.0_rt / rho;
	const Real molecular_switch = static_cast<Real>(molecular_process_switch);
	jac(net_ienuc, 1) = -(Lambda_rec + Lambda_ion_ff) * n_HII * inv_rho;
	jac(net_ienuc, 2) = (Gamma_photo * chat_sigma_photo * n_gamma + molecular_switch * (Gamma_KI - 2.0_rt * Lambda_KI * n_HI)) * inv_rho;
	jac(net_ienuc, 3) = -(Lambda_rec + Lambda_ion_ff) * n_e * inv_rho;
	jac(net_ienuc, net_ienuc) = (-d_recombination_cooling_dT - d_KI_cooling_dT - d_ion_ff_cooling_dT) * dTde * inv_rho;
	jac(net_ienuc, 5) = Gamma_photo * chat_sigma_photo * n_HI * n_gamma_active * inv_rho;
	jac(net_ienuc, 6) = 0.0_rt;

	jac(5, 1) = 0.0_rt;
	jac(5, 2) = -chat_sigma_photo * n_gamma;
	jac(5, 3) = 0.0_rt;
	jac(5, net_ienuc) = 0.0_rt;
	jac(5, 5) = -n_gamma_active * chat_sigma_photo * n_HI;
	jac(5, 6) = 0.0_rt;

	jac(6, 1) = 0.0_rt;
	jac(6, 2) = -chat_sigma_photo * n_gamma_flux;
	jac(6, 3) = 0.0_rt;
	jac(6, net_ienuc) = 0.0_rt;
	jac(6, 5) = 0.0_rt;
	jac(6, 6) = -n_HI * chat_sigma_photo;
}

#endif
