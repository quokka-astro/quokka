#ifndef INTEGRATOR_DATA_H
#define INTEGRATOR_DATA_H

#include <limits>

#include <burn_type.H>

// Define the size of the ODE system that Rosenbrock will integrate

#ifndef PHOTOCHEMISTRY
constexpr int INT_NEQS = NumSpec + 1;
#else
constexpr int INT_NEQS = NumRadEqs + NumSpec + 1;
#endif


// for outputting the burn state failure, use enough precision
// to be reproducible (not guaranteed for std::fixed)
constexpr int OUTDIGITS = std::numeric_limits<amrex::Real>::max_digits10;


enum integrator_errors : std::int8_t {
    IERR_SUCCESS = 1,
    IERR_DT_UNDERFLOW = -2,
    IERR_TOO_MANY_STEPS = -4,
    IERR_TOO_MUCH_ACCURACY_REQUESTED = -5,
    IERR_LU_DECOMPOSITION_ERROR = -7
};


template<typename BurnT>
constexpr int integrator_neqs ()
{
    int int_neqs = 0;

    if constexpr (has_xn<BurnT>) {
        int_neqs += NumSpec;
#ifdef PHOTOCHEMISTRY
        int_neqs += NumRadEqs;
#endif
    }

    if constexpr (has_energy<BurnT>) {
        int_neqs += 1;
    }

    return int_neqs;
}

using IArray1D = amrex::Array1D<short, 1, INT_NEQS>;
using RArray1D = amrex::Array1D<amrex::Real, 1, INT_NEQS>;
using RArray2D = ArrayUtil::MathArray2D<jac_t, 1, INT_NEQS, 1, INT_NEQS>;

#endif
