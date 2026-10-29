/// \file integrator_params.cpp
/// \brief Reader of the integrator runtime parameters declared in integrator_params.hpp.

#include "microphysics/integrator_params.hpp"

#include "AMReX_ParmParse.H"

namespace quokka::microphysics
{

auto ReadIntegratorParams() -> IntegratorParams
{
	IntegratorParams params{};
	const amrex::ParmParse pp("integrator");
	pp.query("rtol_spec", params.rtol_spec);
	pp.query("atol_spec", params.atol_spec);
	pp.query("rtol_enuc", params.rtol_enuc);
	pp.query("atol_enuc", params.atol_enuc);
	pp.query("rtol_rad_num", params.rtol_rad_num);
	pp.query("atol_rad_num", params.atol_rad_num);
	pp.query("species_failure_tolerance", params.species_failure_tolerance);
	pp.query("radiation_failure_tolerance", params.radiation_failure_tolerance);
	pp.query("SMALL_X_SAFE", params.SMALL_X_SAFE);
	pp.query("ode_max_steps", params.ode_max_steps);
	pp.query("X_reject_buffer", params.X_reject_buffer);
	pp.query("burner_verbose", params.burner_verbose);
	return params;
}

} // namespace quokka::microphysics
