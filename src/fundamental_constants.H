// Fundamental constants taken from NIST's 2010 CODATA recommended values

#ifndef _fundamental_constants_H_
#define _fundamental_constants_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Array.H>

namespace C
{
    // newton's gravitational constant
    constexpr amrex::Real Gconst = 6.67428e-8;  // cm^3/g/s^2
} // namespace C

#endif
