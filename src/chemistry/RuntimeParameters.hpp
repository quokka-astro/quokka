#ifndef QUOKKA_CHEMISTRY_RUNTIME_PARAMETERS_HPP_
#define QUOKKA_CHEMISTRY_RUNTIME_PARAMETERS_HPP_

#include "AMReX_ParmParse.H"
#include "chemistry/ChemistryNetwork.hpp"

namespace quokka::chemistry
{

[[nodiscard]] inline auto readIntegratorOptions(amrex::Real minimum_temperature = 0.0) -> IntegratorOptions
{
	IntegratorOptions options{};
	options.minimum_temperature = minimum_temperature;
	amrex::ParmParse const parameters("integrator");
	parameters.query("rtol_spec", options.rtol_species);
	parameters.query("atol_spec", options.atol_species);
	parameters.query("rtol_enuc", options.rtol_energy);
	parameters.query("atol_enuc", options.atol_energy);
	parameters.query("rtol_rad_num", options.rtol_radiation);
	parameters.query("atol_rad_num", options.atol_radiation);
	parameters.query("species_failure_tolerance", options.species_failure_tolerance);
	parameters.query("radiation_failure_tolerance", options.radiation_failure_tolerance);
	parameters.query("SMALL_X_SAFE", options.small_state);
	parameters.query("ode_max_dt", options.maximum_timestep);
	parameters.query("MIN_TEMP", options.minimum_temperature);
	parameters.query("MAX_TEMP", options.maximum_temperature);
	parameters.query("X_reject_buffer", options.rejection_buffer);
	parameters.query("h211b_fac_min", options.controller_minimum);
	parameters.query("h211b_fac_max", options.controller_maximum);
	parameters.query("h211b_reduction_fac", options.controller_reduction);
	parameters.query("h211b_b", options.controller_b);
	parameters.query("h211b_k", options.controller_k);
	parameters.query("ode_max_steps", options.maximum_steps);
	parameters.query("rosenbrock_tableau", options.tableau);
	int jacobian = options.analytic_jacobian ? 1 : 2;
	parameters.query("jacobian", jacobian);
	options.analytic_jacobian = jacobian == 1;
	int pivoting = options.pivoting ? 1 : 0;
	parameters.query("linalg_do_pivoting", pivoting);
	options.pivoting = pivoting != 0;
	int retry = options.retry_enabled ? 1 : 0;
	parameters.query("use_burn_retry", retry);
	options.retry_enabled = retry != 0;
	int swapJacobian = options.retry_swap_jacobian ? 1 : 0;
	parameters.query("retry_swap_jacobian", swapJacobian);
	options.retry_swap_jacobian = swapJacobian != 0;
	parameters.query("retry_rtol_spec", options.retry_rtol_species);
	parameters.query("retry_atol_spec", options.retry_atol_species);
	parameters.query("retry_rtol_enuc", options.retry_rtol_energy);
	parameters.query("retry_atol_enuc", options.retry_atol_energy);
	parameters.query("retry_rtol_rad_num", options.retry_rtol_radiation);
	parameters.query("retry_atol_rad_num", options.retry_atol_radiation);
	return options;
}

} // namespace quokka::chemistry

#endif
