/// \file integrator_params.cpp
/// \brief Definitions and reader of the integrator runtime parameters declared in integrator_params.hpp.

#include "microphysics/integrator_params.hpp"

#include "AMReX_ParmParse.H"

namespace quokka::microphysics::integrator_params
{

using amrex::literals::operator""_rt;

AMREX_GPU_MANAGED amrex::Real rtol_spec = 1.e-10_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real atol_spec = 1.e-10_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real rtol_enuc = 1.e-10_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real atol_enuc = 1.e-25_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real rtol_rad_num = 1.e-10_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real atol_rad_num = 1.e-10_rt;		      // NOLINT
AMREX_GPU_MANAGED amrex::Real species_failure_tolerance = 1.e-2_rt;   // NOLINT
AMREX_GPU_MANAGED amrex::Real radiation_failure_tolerance = 1.e-2_rt; // NOLINT
AMREX_GPU_MANAGED amrex::Real SMALL_X_SAFE = 1.0e-30_rt;	      // NOLINT
AMREX_GPU_MANAGED int ode_max_steps = 150000;			      // NOLINT
AMREX_GPU_MANAGED amrex::Real X_reject_buffer = 1.0_rt;		      // NOLINT
AMREX_GPU_MANAGED bool burner_verbose = false;			      // NOLINT

void readIntegratorParams()
{
	const amrex::ParmParse pp("integrator");
	pp.query("rtol_spec", rtol_spec);
	pp.query("atol_spec", atol_spec);
	pp.query("rtol_enuc", rtol_enuc);
	pp.query("atol_enuc", atol_enuc);
	pp.query("rtol_rad_num", rtol_rad_num);
	pp.query("atol_rad_num", atol_rad_num);
	pp.query("species_failure_tolerance", species_failure_tolerance);
	pp.query("radiation_failure_tolerance", radiation_failure_tolerance);
	pp.query("SMALL_X_SAFE", SMALL_X_SAFE);
	pp.query("ode_max_steps", ode_max_steps);
	pp.query("X_reject_buffer", X_reject_buffer);
	pp.query("burner_verbose", burner_verbose);
}

} // namespace quokka::microphysics::integrator_params
