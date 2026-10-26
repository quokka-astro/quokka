#ifndef MICROPHYSICS_INTEGRATOR_ERRORS_HPP_
#define MICROPHYSICS_INTEGRATOR_ERRORS_HPP_
/// \file integrator_errors.hpp
/// \brief Return codes of the microphysics integrators. Ported from Microphysics' integration/integrator_data.H.

#include <cstdint>

namespace quokka::microphysics
{

enum IntegratorError : std::int8_t {
	IERR_SUCCESS = 1,
	IERR_BAD_INPUTS = -1,
	IERR_DT_UNDERFLOW = -2,
	IERR_SPRAD_CONVERGENCE = -3,
	IERR_TOO_MANY_STEPS = -4,
	IERR_TOO_MUCH_ACCURACY_REQUESTED = -5,
	IERR_CORRECTOR_CONVERGENCE = -6,
	IERR_LU_DECOMPOSITION_ERROR = -7,
	IERR_BAD_STATE_IN_CORRECTOR = -8,
	IERR_JACOBIAN_NOT_FINITE = -9
};

} // namespace quokka::microphysics

#endif // MICROPHYSICS_INTEGRATOR_ERRORS_HPP_
