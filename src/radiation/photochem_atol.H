#ifndef PHOTOCHEM_ATOL_H_ // NOLINT
#define PHOTOCHEM_ATOL_H_

#include <cmath>

#include "AMReX.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "fundamental_constants.H"
#include "radiation/radiation_system.hpp"

#ifdef PHOTOCHEMISTRY

namespace quokka::photochemistry
{

namespace detail
{
// Returns true if any integrator.typical_* key appears in the toml
auto hasTypicalParams() -> bool
{
	amrex::ParmParse const pp("integrator");
	return pp.contains("typical_n_H") || pp.contains("typical_minimal_radiation_T") || pp.contains("desired_accuracy_on_T_at_typical_n_H") ||
	       pp.contains("spec_abundance_tol");
}

// Returns true if any integrator.atol_* key appears explicitly in the toml
auto hasRawAtolParams() -> bool
{
	amrex::ParmParse const pp("integrator");
	return pp.contains("atol_spec") || pp.contains("atol_enuc") || pp.contains("atol_rad_num");
}
} // namespace detail

// SetAtolFromPhysics: derive VODE absolute tolerances from physical scales and inject them
// into ParmParse so init_extern_parameters() picks them up.
//
// Call between readParmParse() and init_extern_parameters() in QuokkaSimulation::Initialize().
// Aborts when neither integrator.typical_* nor integrator.atol_* keys appear in the toml —
// VODE's built-in defaults (~1e-10) are unusably tight for photochemistry and will cause
// the integrator to stall.
//
// Note: Erad_floor (the M1 radiation floor) is a compile-time constexpr in RadSystem_Traits
// and is set independently in each problem's specialization. typical_minimal_radiation_T only
// controls the VODE tolerance, not Erad_floor.
//
// radiation_failure_tolerance defaults to 0.01 cm^-3 — a physically negligible photon count
// per cm^3 — and can be overridden in the toml.
template <typename problem_t> void SetAtolFromPhysics()
{
	if (!detail::hasTypicalParams() && !detail::hasRawAtolParams()) {
		amrex::Abort("No integrator tolerances specified. Either set integrator.typical_* (recommended) "
			     "or integrator.atol_* parameters. VODE's built-in defaults (~1e-10) are unusably "
			     "tight for photochemistry and will cause the integrator to stall. "
			     "See docs/markdown/photoionization.md.");
	}
	if (detail::hasTypicalParams() && detail::hasRawAtolParams()) {
		amrex::Abort("integrator.typical_* and integrator.atol_* are mutually exclusive. Remove one set.");
	}

	if (!detail::hasTypicalParams()) {
		return; // user chose hand-tuned atol_*; nothing to derive
	}

	amrex::ParmParse const pp("integrator");

	if (!pp.contains("typical_n_H")) {
		amrex::Abort("integrator.typical_n_H is required when using integrator.typical_* parameters.");
	}
	if (!pp.contains("typical_minimal_radiation_T")) {
		amrex::Abort("integrator.typical_minimal_radiation_T is required when using integrator.typical_* parameters.");
	}

	// --- Read required inputs ---
	amrex::Real typical_n_H = NAN;
	pp.get("typical_n_H", typical_n_H);

	amrex::Real typical_minimal_radiation_T = NAN;
	pp.get("typical_minimal_radiation_T", typical_minimal_radiation_T);

	// --- Read optional inputs ---
	amrex::Real desired_accuracy_on_T_at_typical_n_H = 1.0; // K
	pp.query("desired_accuracy_on_T_at_typical_n_H", desired_accuracy_on_T_at_typical_n_H);

	amrex::Real spec_abundance_tol = 1.0e-5; // fraction of n_H
	pp.query("spec_abundance_tol", spec_abundance_tol);

	// --- Physical constants ---
	const amrex::Real E_photon = RadSystem<problem_t>::GetChemBandQuanta(0); // midpoint energy of first chem band (erg)
	const amrex::Real c_v = 1.5 * C::k_B / C::m_p;				 // erg/g/K; monatomic hydrogen

	// --- atol_spec: species negligibility floor ---
	const amrex::Real atol_spec = spec_abundance_tol * typical_n_H;

	// --- atol_enuc: temperature accuracy requirement ---
	const amrex::Real atol_enuc = c_v * desired_accuracy_on_T_at_typical_n_H;

	// --- atol_rad_num: photon number density tolerance ---
	// Set to 1e-6 × the blackbody photon density at typical_minimal_radiation_T.
	// This makes photon counts below 1 ppm of the minimum meaningful radiation field
	// negligible to VODE, so dark cells (Erad ≈ Erad_floor << a*T^4) return in one BDF step.
	// typical_minimal_radiation_T is NOT Erad_floor; Erad_floor is set in RadSystem_Traits.
	const amrex::Real atol_rad_num = 1.0e-6 * C::a_rad * std::pow(typical_minimal_radiation_T, 4.0) / E_photon;

	// --- radiation_failure_tolerance ---
	// Physical guard: maximum allowed negative photon number density (cm^-3).
	// 0.01 photons/cm^3 is negligible in any astrophysical context.
	amrex::Real radiation_failure_tolerance = 1.0e-2;
	pp.query("radiation_failure_tolerance", radiation_failure_tolerance);

	// --- Inject into ParmParse (init_extern_parameters reads these) ---
	amrex::ParmParse pp_int("integrator");
	pp_int.add("atol_spec", atol_spec);
	pp_int.add("atol_enuc", atol_enuc);
	pp_int.add("atol_rad_num", atol_rad_num);
	pp_int.add("radiation_failure_tolerance", radiation_failure_tolerance);

	amrex::Print() << "SetAtolFromPhysics: atol_spec=" << atol_spec << " atol_enuc=" << atol_enuc << " atol_rad_num=" << atol_rad_num
		       << " radiation_failure_tolerance=" << radiation_failure_tolerance << "\n";
}

} // namespace quokka::photochemistry
#endif // PHOTOCHEMISTRY
#endif // PHOTOCHEM_ATOL_H_
