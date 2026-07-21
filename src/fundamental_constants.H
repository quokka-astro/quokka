#ifndef FUNDAMENTAL_CONSTANTS_H
#define FUNDAMENTAL_CONSTANTS_H

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

// Quokka-owned CGS physical constants. Values match the constants previously
// supplied by AMReX-Astrophysics Microphysics so this ownership change does not
// alter existing problem normalization.
namespace C
{
constexpr amrex::Real c_light = 2.99792458e10;
constexpr amrex::Real Gconst = 6.67428e-8;
constexpr amrex::Real k_B = 1.3806490000000002e-16;
constexpr amrex::Real k_MeV = 8.61733326214518e-11;
constexpr amrex::Real hbar = 1.0545718176461563e-27;
constexpr amrex::Real hplanck = 6.62607015e-27;
constexpr amrex::Real n_A = 6.02214076e23;
constexpr amrex::Real ev2erg = 1.602176634e-12;
constexpr amrex::Real MeV2eV = 1.0e6;
constexpr amrex::Real MeV2erg = 1.6021766339999998e-6;
constexpr amrex::Real MeV2gr = (MeV2eV * ev2erg) / (c_light * c_light);
constexpr amrex::Real enuc_conv2 = -n_A * c_light * c_light;
constexpr amrex::Real m_p = 1.67262192595e-24;
constexpr amrex::Real m_n = 1.6749275005600003e-24;
constexpr amrex::Real m_e = 9.1093837139e-28;
constexpr amrex::Real m_u = 1.66053906892e-24;
constexpr amrex::Real q_e = 4.803204712570263e-10;
constexpr amrex::Real sigma_SB = 5.670374419184432e-5;
constexpr amrex::Real a_rad = 4.0 * sigma_SB / c_light;
constexpr amrex::Real AU = 1.495978707e13;
constexpr amrex::Real parsec = 3.085677581491367e18;
constexpr amrex::Real Hubble_const = 32.407764868e-19;
constexpr amrex::Real M_solar = 1.9884e33;
constexpr amrex::Real R_solar = 6.957e10;

namespace Legacy
{
constexpr amrex::Real m_n = 1.67492721184e-24;
constexpr amrex::Real m_p = 1.67262163783e-24;
constexpr amrex::Real m_e = 9.1093821545e-28;
constexpr amrex::Real eV2erg = 1.60217648740e-12;
constexpr amrex::Real MeV2erg = eV2erg * 1.0e6;
constexpr amrex::Real MeV2gr = MeV2erg / (c_light * c_light);
constexpr amrex::Real n_A = 6.0221417930e23;
constexpr amrex::Real enuc_conv2 = -n_A * c_light * c_light;
} // namespace Legacy
} // namespace C

#endif
