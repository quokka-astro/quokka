#ifndef FUNDAMENTAL_CONSTANTS_HPP_
#define FUNDAMENTAL_CONSTANTS_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file fundamental_constants.hpp
/// \brief Physical constants in CGS units.
///
/// The values are copied from Microphysics's constants/fundamental_constants.H, so results do not depend on which header defines them.
/// Chemistry and photochemistry builds still use Microphysics, whose headers define namespace C themselves, so there this header
/// includes Microphysics's header instead of defining the constants again.

#if defined(CHEMISTRY) || defined(PHOTOCHEMISTRY)
#include "fundamental_constants.H"
#else

#include "AMReX_REAL.H"

namespace C
{
// speed of light in vacuum
constexpr amrex::Real c_light = 29979245800.0; // cm/s

// Newton's gravitational constant
constexpr amrex::Real Gconst = 6.67428e-8; // cm^3/g/s^2

// Boltzmann's constant
constexpr amrex::Real k_B = 1.3806490000000002e-16; // erg/K

// Planck's constant
constexpr amrex::Real hplanck = 6.62607015e-27; // erg s

// convert eV to erg
constexpr amrex::Real ev2erg = 1.602176634e-12;

// mass of proton
constexpr amrex::Real m_p = 1.67262192595e-24; // g

// mass of electron
constexpr amrex::Real m_e = 9.1093837139e-28; // g

// atomic mass unit
constexpr amrex::Real m_u = 1.66053906892e-24; // g

// Stefan-Boltzmann constant
constexpr amrex::Real sigma_SB = 5.670374419184432e-05; // erg/s/cm^2/K^4

// radiation constant
constexpr amrex::Real a_rad = 4.0 * sigma_SB / c_light;

// parsec
constexpr amrex::Real parsec = 3.085677581491367e+18; // cm

// solar mass
constexpr amrex::Real M_solar = 1.9884e33; // g

// solar radius
constexpr amrex::Real R_solar = 6.957e10; // cm
} // namespace C

#endif // CHEMISTRY || PHOTOCHEMISTRY

#endif // FUNDAMENTAL_CONSTANTS_HPP_
