#ifndef PHOTOCHEMEOS_HPP_
#define PHOTOCHEMEOS_HPP_
/// \file PhotochemEOS.hpp
/// \brief Multi-gamma ideal gas for photochemistry: each species i is an ideal gas with its own adiabatic index gamma_i.
///
/// With number densities n_i and particle masses m_i, rho_sp = sum_i n_i m_i and the specific internal energy is
/// e = k_B T sum_i n_i / (gamma_i - 1) / rho_sp. This is Microphysics's photoionization EOS (eos_input_re and
/// eos_input_rt), without input floors.

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "util/fundamental_constants.hpp"

namespace quokka::photochemistry
{

namespace detail
{
/// sum_i n_i / (gamma_i - 1) and rho_sp = sum_i n_i m_i
template <int NSpec>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void HeatCapacitySums(amrex::GpuArray<amrex::Real, NSpec> const &n, amrex::Real const *mass, amrex::Real const *gamma,
							       amrex::Real &sum_n_over_gm1, amrex::Real &rho_sp)
{
	sum_n_over_gm1 = 0.0;
	rho_sp = 0.0;
	for (int i = 0; i < NSpec; ++i) {
		rho_sp += n[i] * mass[i];
		sum_n_over_gm1 += n[i] / (gamma[i] - 1.0);
	}
}
} // namespace detail

/// temperature [K] from the specific internal energy e [erg g^-1]
template <int NSpec>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto TemperatureFromEnergy(const amrex::Real e, amrex::GpuArray<amrex::Real, NSpec> const &n, amrex::Real const *mass,
								    amrex::Real const *gamma) -> amrex::Real
{
	amrex::Real s = NAN;
	amrex::Real rho_sp = NAN;
	detail::HeatCapacitySums<NSpec>(n, mass, gamma, s, rho_sp);
	return e * rho_sp / (s * C::k_B);
}

/// specific internal energy [erg g^-1] from the temperature T [K]
template <int NSpec>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto EnergyFromTemperature(const amrex::Real T, amrex::GpuArray<amrex::Real, NSpec> const &n, amrex::Real const *mass,
								    amrex::Real const *gamma) -> amrex::Real
{
	amrex::Real s = NAN;
	amrex::Real rho_sp = NAN;
	detail::HeatCapacitySums<NSpec>(n, mass, gamma, s, rho_sp);
	return s * C::k_B * T / rho_sp;
}

/// de/dT at constant composition [erg g^-1 K^-1]
template <int NSpec>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto dEnergy_dT(amrex::GpuArray<amrex::Real, NSpec> const &n, amrex::Real const *mass, amrex::Real const *gamma)
    -> amrex::Real
{
	amrex::Real s = NAN;
	amrex::Real rho_sp = NAN;
	detail::HeatCapacitySums<NSpec>(n, mass, gamma, s, rho_sp);
	return s * C::k_B / rho_sp;
}

} // namespace quokka::photochemistry

#endif // PHOTOCHEMEOS_HPP_
