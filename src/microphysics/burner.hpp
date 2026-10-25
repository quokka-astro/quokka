#ifndef MICROPHYSICS_BURNER_HPP_
#define MICROPHYSICS_BURNER_HPP_
/// \file burner.hpp
/// \brief Burn<problem_t>, which integrates a burn state over one timestep.

#include "AMReX.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "microphysics/microphysics_traits.hpp"
#include "microphysics/rosenbrock/rosenbrock.hpp"

// still from extern/Microphysics
#include "burn_type.H"

namespace quokka::microphysics
{

template <typename problem_t> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void Burn(burn_t &state, const amrex::Real dt)
{
	Microphysics_Traits<problem_t>::Integrator::template Integrate<problem_t>(state, dt);

	if constexpr (Microphysics_Traits<problem_t>::burn_retry) {
		// placeholder for the retry path, which is not implemented yet
		if (!state.success) {
			amrex::Abort("quokka::microphysics::Burn: burn retry is not implemented");
		}
	}
}

} // namespace quokka::microphysics

#endif // MICROPHYSICS_BURNER_HPP_
