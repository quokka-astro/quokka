#ifndef MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
#define MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
/// \file integrator_params.hpp
/// \brief Runtime parameters of the microphysics integrators, read from the `integrator` prefix. Ported from Microphysics'
/// integration/_parameters and integration/Rosenbrock/_parameters, with the same names and defaults. The integrator
/// switches are compile-time traits instead (microphysics_traits.hpp).
///
/// The variables are defined in integrator_params.cpp, not inline here: HIP cannot register inline managed variables.

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::microphysics::integrator_params
{

/// Relative and absolute tolerances of the species.
extern AMREX_GPU_MANAGED amrex::Real rtol_spec; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real atol_spec; // NOLINT
/// Relative and absolute tolerances of the energy.
extern AMREX_GPU_MANAGED amrex::Real rtol_enuc; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real atol_enuc; // NOLINT
/// Relative and absolute tolerances of the chemical-band photon numbers.
extern AMREX_GPU_MANAGED amrex::Real rtol_rad_num; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real atol_rad_num; // NOLINT
/// A burn fails if a species ends below -species_failure_tolerance.
extern AMREX_GPU_MANAGED amrex::Real species_failure_tolerance; // NOLINT
/// A burn fails if a chemical-band photon number ends below -radiation_failure_tolerance.
extern AMREX_GPU_MANAGED amrex::Real radiation_failure_tolerance; // NOLINT
/// The floor of the species and the photon numbers.
extern AMREX_GPU_MANAGED amrex::Real SMALL_X_SAFE; // NOLINT
/// The largest number of steps in one burn.
extern AMREX_GPU_MANAGED int ode_max_steps; // NOLINT
/// The Rosenbrock integrator rejects a step that changes a species above X_reject_buffer * atol_spec by more than a
/// factor of 4 either way.
extern AMREX_GPU_MANAGED amrex::Real X_reject_buffer; // NOLINT
/// Print a summary after each burn (CPU builds only).
extern AMREX_GPU_MANAGED bool burner_verbose; // NOLINT

/// Read the integrator parameters. Replaces the integrator part of Microphysics' init_extern_parameters().
void readIntegratorParams();

} // namespace quokka::microphysics::integrator_params

#endif // MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
