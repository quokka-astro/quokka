#ifndef PHOTOCHEMPARAMS_HPP_
#define PHOTOCHEMPARAMS_HPP_
/// \file PhotochemParams.hpp
/// \brief Runtime parameters of the photochemistry solve, read from the "integrator" ParmParse prefix.
///
/// Defaults are Microphysics's defaults. The struct is trivially copyable so it can be captured by value in device lambdas.

#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

namespace quokka::photochemistry
{

struct IntegratorParams {
	amrex::Real rtol_spec = 1.0e-10;		  ///< relative tolerance, species number densities
	amrex::Real atol_spec = 1.0e-10;		  ///< absolute tolerance, species number densities [cm^-3]
	amrex::Real rtol_enuc = 1.0e-10;		  ///< relative tolerance, specific internal energy
	amrex::Real atol_enuc = 1.0e-25;		  ///< absolute tolerance, specific internal energy [erg g^-1]
	amrex::Real rtol_rad_num = 1.0e-10;		  ///< relative tolerance, chemical-band photon number densities
	amrex::Real atol_rad_num = 1.0e-10;		  ///< absolute tolerance, chemical-band photon number densities [cm^-3]
	amrex::Real X_reject_buffer = 1.0;		  ///< species change limit applies above X_reject_buffer * atol_spec
	amrex::Real species_failure_tolerance = 1.0e-2;	  ///< a final species number density below -this fails the cell [cm^-3]
	amrex::Real radiation_failure_tolerance = 1.0e-2; ///< a final photon number density below -this fails the cell [cm^-3]
	int ode_max_steps = 150000;			  ///< maximum number of solver steps per cell

	static auto read() -> IntegratorParams
	{
		IntegratorParams p{};
		const amrex::ParmParse pp("integrator");
		pp.query("rtol_spec", p.rtol_spec);
		pp.query("atol_spec", p.atol_spec);
		pp.query("rtol_enuc", p.rtol_enuc);
		pp.query("atol_enuc", p.atol_enuc);
		pp.query("rtol_rad_num", p.rtol_rad_num);
		pp.query("atol_rad_num", p.atol_rad_num);
		pp.query("X_reject_buffer", p.X_reject_buffer);
		pp.query("species_failure_tolerance", p.species_failure_tolerance);
		pp.query("radiation_failure_tolerance", p.radiation_failure_tolerance);
		pp.query("ode_max_steps", p.ode_max_steps);
		return p;
	}
};

} // namespace quokka::photochemistry

#endif // PHOTOCHEMPARAMS_HPP_
