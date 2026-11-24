// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef FUNDAMENTAL_CONSTANTS_HPP_
#define FUNDAMENTAL_CONSTANTS_HPP_
/// \file fundamental_constants.hpp
/// \brief Physical constants in CGS units, used as namespace C.
///
/// The values are copied from Microphysics's constants/fundamental_constants.H.

#include "AMReX_REAL.H"

namespace quokka::constants
{
// speed of light in vacuum
constexpr amrex::Real c_light = 29979245800.0; // cm/s

// Newton's gravitational constant
constexpr amrex::Real Gconst = 6.67428e-8; // cm^3/g/s^2

// Boltzmann's constant
constexpr amrex::Real k_B = 1.3806490000000002e-16; // erg/K

// Planck's constant
constexpr amrex::Real hplanck = 6.62607015e-27; // erg s

// Avogadro's number
constexpr amrex::Real n_A = 6.02214076e+23; // mol^-1

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
} // namespace quokka::constants

namespace C = quokka::constants; // NOLINT(misc-unused-alias-decls): a short name for the files that include this one

#endif // FUNDAMENTAL_CONSTANTS_HPP_
