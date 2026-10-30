#ifndef MICROPHYSICS_BURNER_HPP_
#define MICROPHYSICS_BURNER_HPP_
/// \file burner.hpp
/// \brief Burn<problem_t>, which integrates a burn state over one timestep.

#include "AMReX.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "microphysics/microphysics_traits.hpp"

// still from extern/Microphysics: the problem's CMake target picks the integrator through its include path
#include "actual_integrator.H"
#include "burn_type.H"

namespace quokka::microphysics
{

/// Integrates the burn state over dt with the integrator the problem is built with. Replaces Microphysics'
/// burner(), integrator() and integrator_wrapper(): the NSE branch is gone, and burn retry is chosen at compile
/// time by Microphysics_Traits<problem_t>::burn_retry instead of the runtime parameter integrator.use_burn_retry.
template <typename problem_t> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Burn(burn_t &state, const amrex::Real dt)
{
	actual_integrator(state, dt);

	if constexpr (Microphysics_Traits<problem_t>::burn_retry) {
		// placeholder for the retry path, which is not implemented yet
		if (!state.success) {
			amrex::Abort("quokka::microphysics::Burn: burn retry is not implemented");
		}
	}
}

} // namespace quokka::microphysics

#endif // MICROPHYSICS_BURNER_HPP_
