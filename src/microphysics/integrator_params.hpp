#ifndef MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
#define MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
/// \file integrator_params.hpp
/// \brief Runtime parameters of the microphysics integrators, read from the `integrator` prefix. Ported from Microphysics'
/// integration/_parameters and integration/Rosenbrock/_parameters, with the same names and defaults. The integrator
/// switches are compile-time traits instead (microphysics_traits.hpp).

#include "AMReX_REAL.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// The runtime parameters of the integrators.
struct IntegratorParams {
	/// Relative and absolute tolerances of the species.
	amrex::Real rtol_spec = 1.e-10_rt;
	amrex::Real atol_spec = 1.e-10_rt;
	/// Relative and absolute tolerances of the energy.
	amrex::Real rtol_enuc = 1.e-10_rt;
	amrex::Real atol_enuc = 1.e-25_rt;
	/// Relative and absolute tolerances of the chemical-band photon numbers.
	amrex::Real rtol_rad_num = 1.e-10_rt;
	amrex::Real atol_rad_num = 1.e-10_rt;
	/// A burn fails if a species ends below -species_failure_tolerance.
	amrex::Real species_failure_tolerance = 1.e-2_rt;
	/// A burn fails if a chemical-band photon number ends below -radiation_failure_tolerance.
	amrex::Real radiation_failure_tolerance = 1.e-2_rt;
	/// The floor of the species and the photon numbers.
	amrex::Real SMALL_X_SAFE = 1.0e-30_rt;
	/// The largest number of steps in one burn.
	int ode_max_steps = 150000;
	/// The Rosenbrock integrator rejects a step that changes a species above X_reject_buffer * atol_spec by more than a
	/// factor of 4 either way.
	amrex::Real X_reject_buffer = 1.0_rt;
	/// Print a summary after each burn (CPU builds only).
	bool burner_verbose = false;
};

/// Read the integrator parameters. Replaces the integrator part of Microphysics' init_extern_parameters().
auto ReadIntegratorParams() -> IntegratorParams;

} // namespace quokka::microphysics

#endif // MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
