#ifndef actual_rhs_H
#define actual_rhs_H

#include "AMReX_BLassert.H"
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
// Radiation index 1 (Index 5 in the state) = flux of photons normalized by the flux magnitude before the burn (therefore, it is 1.0 at the beginning of burn)

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_photoionization_coefficient(Real T) -> Real { return 1.5e-18_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_coefficient(Real T) -> Real
{
	return 2.6e-13 * (1.0 + recombination_temperature_dependent * (std::pow(T / 1.0e4, -0.7) - 1.0));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_collisional_ionization_coefficient(Real T) -> Real
{
	return 5.84e-11 * std::sqrt(T) * std::exp(-6.4e-12_rt / (C::k_B * T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionization_heating_coefficient(Real T) -> Real { return 6.4e-12_rt; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_KI_heating_coefficient(Real T) -> Real { return 2.0e-26; }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_KI_cooling_coefficient(Real T) -> Real
{
	return 2.0e-26 * (1.0e7 * std::exp(-114800.0 / (T + 1.0e3)) + 14 * std::sqrt(T) * std::exp(-92.0 / T));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_recombination_cooling_coefficient(Real T) -> Real
{
	return C::k_B * T * get_recombination_coefficient(T) * (0.684 - 0.0416 * std::log(T / 1.0e4));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ion_ff_cooling_coefficient(Real T) -> Real { return 1.4e-27 * std::sqrt(T) + 1.0e-19 * std::exp(-118348.0 / T); }

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto get_ionization_fraction_scaling(Real n_HI, Real n_HII) -> Real
{
	// If sharp ionization front is assumed, scale densities by 1/x to account for
	// the fact that only 1/x of the cell is ionized instead of the full cell.
	AMREX_ASSERT(assume_sharp_ionization_front == 0); // We don't support it yet.
	if (assume_sharp_ionization_front == 1) {
		if (n_HII <= 0.0_rt) {
			return 1.0_rt;
		}
		return (n_HI + n_HII) / n_HII;
	}
	return 1.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_rhs(const burn_t &state, Array1D<Real, 1, INT_NEQS> &ydot)
{
	amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	amrex::Real sigma_coll = get_collisional_ionization_coefficient(state.T);
	amrex::Real alpha_rec = get_recombination_coefficient(state.T);

	amrex::Real Gamma_photo = get_ionization_heating_coefficient(state.T);
	amrex::Real Gamma_KI = get_KI_heating_coefficient(state.T);
	amrex::Real Lambda_KI = get_KI_cooling_coefficient(state.T);
	amrex::Real Lambda_rec = get_recombination_cooling_coefficient(state.T);
	amrex::Real Lambda_ion_ff = get_ion_ff_cooling_coefficient(state.T);

	amrex::Real c = C::c_light;
	amrex::Real chat = state.c_hat;
	amrex::Real frac = chat / c;

	amrex::Real n_e = state.xn[0];
	amrex::Real n_HI = state.xn[1];
	amrex::Real n_HII = state.xn[2];
	amrex::Real n_gamma = state.rn[0];
	amrex::Real n_gamma_flux = state.rn[1];

	// Apply sharp ionization front scaling if enabled
	amrex::Real scaling = get_ionization_fraction_scaling(n_HI, n_HII);
	amrex::Real n_e_eff = n_e * scaling;
	amrex::Real n_HII_eff = n_HII * scaling;

	amrex::Real photoionization_term = c * sigma_photo * n_HI * n_gamma;
	amrex::Real collisional_ionization_term = collisional_ionization_switch * n_e_eff * n_HI * sigma_coll / scaling;
	amrex::Real recombination_term = recombination_switch * alpha_rec * n_e_eff * n_HII_eff / scaling;
	amrex::Real photoionization_heating = photoheating_switch * Gamma_photo * photoionization_term;
	amrex::Real KI_heating = KI_heating_switch * Gamma_KI * n_HI;
	amrex::Real KI_cooling = KI_cooling_switch * Lambda_KI * n_HI * n_HI;
	amrex::Real recombination_cooling = recombination_cooling_switch * Lambda_rec * n_e_eff * n_HII_eff / scaling;
	amrex::Real ion_ff_cooling = ion_ff_cooling_switch * Lambda_ion_ff * n_e_eff * n_HII_eff / scaling;

	ydot(1) = photoionization_term * frac + collisional_ionization_term - recombination_term;
	ydot(2) = -ydot(1);
	ydot(3) = ydot(1);
	ydot(net_ienuc) = energy_switch * (photoionization_heating + KI_heating - recombination_cooling - KI_cooling - ion_ff_cooling) / state.rho;
	ydot(5) = -photoionization_term * frac;
	ydot(6) = -n_HI * c * sigma_photo * n_gamma_flux * frac;
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{
	amrex::Real sigma_photo = get_photoionization_coefficient(state.T);
	amrex::Real sigma_coll = get_collisional_ionization_coefficient(state.T);
	amrex::Real alpha_rec = get_recombination_coefficient(state.T);

	amrex::Real Gamma_photo = get_ionization_heating_coefficient(state.T);
	amrex::Real Gamma_KI = get_KI_heating_coefficient(state.T);
	amrex::Real Lambda_KI = get_KI_cooling_coefficient(state.T);
	amrex::Real Lambda_rec = get_recombination_cooling_coefficient(state.T);
	amrex::Real Lambda_ion_ff = get_ion_ff_cooling_coefficient(state.T);

	amrex::Real c = C::c_light;
	amrex::Real chat = state.c_hat;
	amrex::Real frac = chat / c;

	amrex::Real n_e = state.xn[0];
	amrex::Real n_HI = state.xn[1];
	amrex::Real n_HII = state.xn[2];
	amrex::Real n_gamma = state.rn[0];
	amrex::Real n_gamma_flux = state.rn[1];

	// Apply sharp ionization front scaling if enabled
	amrex::Real scaling = get_ionization_fraction_scaling(n_HI, n_HII);
	amrex::Real n_e_eff = n_e * scaling;
	amrex::Real n_HII_eff = n_HII * scaling;

	amrex::Real photoionization_term = c * sigma_photo * n_HI * n_gamma;
	amrex::Real collisional_ionization_term = collisional_ionization_switch * n_e_eff * n_HI * sigma_coll / scaling;
	amrex::Real recombination_term = recombination_switch * alpha_rec * n_e_eff * n_HII_eff / scaling;
	amrex::Real photoionization_heating = photoheating_switch * Gamma_photo * photoionization_term;
	amrex::Real KI_heating = KI_heating_switch * Gamma_KI * n_HI;
	amrex::Real KI_cooling = KI_cooling_switch * Lambda_KI * n_HI * n_HI;
	amrex::Real recombination_cooling = recombination_cooling_switch * Lambda_rec * n_e_eff * n_HII_eff / scaling;
	amrex::Real ion_ff_cooling = ion_ff_cooling_switch * Lambda_ion_ff * n_e_eff * n_HII_eff / scaling;

	jac.zero();

	jac(1, 1) = (collisional_ionization_term - recombination_term) / n_e_eff;
	jac(1, 2) = (photoionization_term * frac + collisional_ionization_term) / n_HI;
	jac(1, 3) = -recombination_term / n_HII_eff;
	jac(1, net_ienuc) = 0.0_rt;
	jac(1, 5) = photoionization_term * frac / n_gamma;
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

	jac(net_ienuc, 1) = -energy_switch * (recombination_cooling / n_e_eff + ion_ff_cooling / n_e_eff) / state.rho;
	jac(net_ienuc, 2) = energy_switch * (photoionization_heating / n_HI + KI_heating / n_HI - 2.0_rt * KI_cooling / n_HI) / state.rho;
	jac(net_ienuc, 3) = -energy_switch * (recombination_cooling / n_HII_eff + ion_ff_cooling / n_HII_eff) / state.rho;
	jac(net_ienuc, net_ienuc) = 0.0_rt;
	jac(net_ienuc, 5) = energy_switch * (photoionization_heating / n_gamma) / state.rho;
	jac(net_ienuc, 6) = 0.0_rt;

	jac(5, 1) = 0.0_rt;
	jac(5, 2) = -photoionization_term * frac / n_HI;
	jac(5, 3) = 0.0_rt;
	jac(5, net_ienuc) = 0.0_rt;
	jac(5, 5) = -photoionization_term * frac / n_gamma;
	jac(5, 6) = 0.0_rt;

	jac(6, 1) = 0.0_rt;
	jac(6, 2) = -c * sigma_photo * n_gamma_flux;
	jac(6, 3) = 0.0_rt;
	jac(6, net_ienuc) = 0.0_rt;
	jac(6, 5) = 0.0_rt;
	jac(6, 6) = -n_HI * c * sigma_photo;
}

#endif
