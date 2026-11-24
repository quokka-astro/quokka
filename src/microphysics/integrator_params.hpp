// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
#define MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
/// \file integrator_params.hpp
/// \brief Runtime parameters of the burn integrator, read from the `integrator` prefix. Ported from Microphysics'
/// integration/_parameters and integration/Rosenbrock/_parameters, with the same names and defaults.

#include <limits>
#include <string>

#include "AMReX.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

namespace quokka::microphysics
{

using amrex::literals::operator""_rt;

/// The runtime parameters of the burn integrator.
struct IntegratorParams {
	/// Relative and absolute tolerances of the species.
	amrex::Real rtol_spec = 1.e-2_rt;
	amrex::Real atol_spec = 1.e-3_rt;
	/// Relative and absolute tolerances of the specific internal energy.
	amrex::Real rtol_enuc = 1.e-2_rt;
	amrex::Real atol_enuc = 1.e+8_rt;
	/// Relative and absolute tolerances of the chemical-band photon numbers.
	amrex::Real rtol_rad_num = 1.e-2_rt;
	amrex::Real atol_rad_num = 1.e-3_rt;
	/// A burn fails if a species ends below -species_failure_tolerance.
	amrex::Real species_failure_tolerance = 1.e-3_rt;
	/// A burn fails if a chemical-band photon number ends below -radiation_failure_tolerance.
	amrex::Real radiation_failure_tolerance = 1.e-3_rt;
	/// The floor of the species, the photon numbers and the thermal-band energies.
	amrex::Real SMALL_X_SAFE = 1.0e-30_rt;
	/// The largest number of steps in one burn.
	int ode_max_steps = 150000;
	/// A step is rejected if it changes a species above X_reject_buffer * atol_spec by more than a factor of 4 either way.
	amrex::Real X_reject_buffer = 1.0_rt;
	/// Print a summary after each burn (CPU builds only).
	bool burner_verbose = false;

	/// Read the parameters from the `integrator` prefix.
	static auto Read() -> IntegratorParams
	{
		IntegratorParams p{};
		const amrex::ParmParse pp("integrator");
		pp.query("rtol_spec", p.rtol_spec);
		pp.query("atol_spec", p.atol_spec);
		pp.query("rtol_enuc", p.rtol_enuc);
		pp.query("atol_enuc", p.atol_enuc);
		pp.query("rtol_rad_num", p.rtol_rad_num);
		pp.query("atol_rad_num", p.atol_rad_num);
		pp.query("species_failure_tolerance", p.species_failure_tolerance);
		pp.query("radiation_failure_tolerance", p.radiation_failure_tolerance);
		pp.query("SMALL_X_SAFE", p.SMALL_X_SAFE);
		pp.query("ode_max_steps", p.ode_max_steps);
		pp.query("X_reject_buffer", p.X_reject_buffer);
		pp.query("burner_verbose", p.burner_verbose);

		// The integrator needs positive tolerances, with every rtol above 10 machine epsilons.
		const auto check_tolerances = [](const char *name, const amrex::Real rtol, const amrex::Real atol) {
			constexpr amrex::Real rtol_min = 10.0_rt * std::numeric_limits<amrex::Real>::epsilon();
			if (!(atol > 0.0_rt) || !(rtol > rtol_min)) {
				amrex::Abort(std::string("integrator.rtol_") + name + " must exceed 10 machine epsilons and integrator.atol_" + name +
					     " must be positive");
			}
		};
		check_tolerances("spec", p.rtol_spec, p.atol_spec);
		check_tolerances("enuc", p.rtol_enuc, p.atol_enuc);
		check_tolerances("rad_num", p.rtol_rad_num, p.atol_rad_num);
		return p;
	}

	/// The parameters of this run, read on first use. Host only; kernels take a copy by value.
	static auto Parameters() -> const IntegratorParams &
	{
		static const IntegratorParams params = Read();
		return params;
	}
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_INTEGRATOR_PARAMS_HPP_
