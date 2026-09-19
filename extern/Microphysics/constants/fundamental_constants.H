#ifndef FUNDAMENTAL_CONSTANTS_H
#define FUNDAMENTAL_CONSTANTS_H

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Array.H>

//
// *** THIS FILE IS AUTO GENERATED VIA write_fundamental_constants.py ***
//
// Non-legacy constants are obtained from scipy.constants.
// Legacy constants are out-dated constants used for aprox networks only.
//

namespace C
{
    // speed of light in vacuum
    constexpr amrex::Real c_light = 29979245800.0;  // cm/s

    // newton's gravitational constant
    constexpr amrex::Real Gconst = 6.67428e-8;  // cm^3/g/s^2

    // new value; if uncommented initial models will need to be re-HSE'ed
    // constexpr amrex::Real Gconst = 6.674299999999998e-08;  // cm^3/g/s^2

    // boltzmann's constant
    constexpr amrex::Real k_B = 1.3806490000000002e-16;  // erg/K

    // Boltzmann constant in MeV/K
    constexpr amrex::Real k_MeV = 8.61733326214518e-11;  // MeV/K

    // planck's constant over 2pi
    constexpr amrex::Real hbar = 1.0545718176461563e-27;  // erg

    // planck's constant
    constexpr amrex::Real hplanck = 6.62607015e-27;  // erg s

    // avogradro's Number
    constexpr amrex::Real n_A = 6.02214076e+23;  // mol^-1

    // convert eV to erg
    constexpr amrex::Real ev2erg = 1.602176634e-12;

    // convert MeV to eV
    constexpr amrex::Real MeV2eV = 1.0e6;

    // convert MeV to erg
    constexpr amrex::Real MeV2erg = 1.6021766339999998e-06;

    // convert MeV to grams
    constexpr amrex::Real MeV2gr  = (MeV2eV * ev2erg) / (c_light * c_light);

    // conversion factor for nuclear energy generation rate
    constexpr amrex::Real enuc_conv2 = -n_A * c_light * c_light;

    // mass of proton
    constexpr amrex::Real m_p = 1.67262192595e-24;  // g

    // mass of neutron
    constexpr amrex::Real m_n = 1.6749275005600003e-24;  // g

    // mass of electron
    constexpr amrex::Real m_e = 9.1093837139e-28;  // g

    // atomic mass unit
    constexpr amrex::Real m_u = 1.66053906892e-24; // g

    // electron charge
    // NIST: q_e = 1.602176565e-19 C
    //
    // C is the SI unit Coulomb; in cgs we have the definition:
    //     1 C = 0.1 * |c_light| * 1 statC
    // where statC is the cgs unit statCoulomb; 1 statC = 1 erg^1/2 cm^1/2
    // and |c_light| is the speed of light in cgs (but without units)
    constexpr amrex::Real q_e = 4.803204712570263e-10;  // erg^1/2 cm^1/2

    // stefan-boltzmann constant
    constexpr amrex::Real sigma_SB = 5.670374419184432e-05;  // erg/s/cm^2/K^4

    // radiation constant
    constexpr amrex::Real a_rad = 4.0*sigma_SB/c_light;

    // Number of centimeters in a parsec and an AU.
    // Note that since the length of an AU is defined exactly
    // by IAU convention, the length of a parsec is also
    // defined exactly as (6.48e5 / pi) AU.
    constexpr amrex::Real AU = 14959787070000.0;  // cm
    constexpr amrex::Real parsec = 3.085677581491367e+18;  // cm

    // Hubble constant (in s^{-1}, converted from 100 (km/s)/Mpc by dividing by 3.08568025e19km/Mpc)
    constexpr amrex::Real Hubble_const = 32.407764868e-19;

    // solar mass (from http://asa.usno.navy.mil/SecK/Constants.html)
    constexpr amrex::Real M_solar = 1.9884e33;

    // solar radius
    constexpr amrex::Real R_solar = 6.957e10;

    namespace Legacy
    {
        // These are the values of the constants used in the original aprox networks
        constexpr amrex::Real m_n = 1.67492721184e-24;
        constexpr amrex::Real m_p = 1.67262163783e-24;
        constexpr amrex::Real m_e = 9.1093821545e-28;

        constexpr amrex::Real eV2erg  = 1.60217648740e-12;
        constexpr amrex::Real MeV2erg = eV2erg * 1.0e6;
        constexpr amrex::Real MeV2gr  = MeV2erg / (c_light * c_light);

        constexpr amrex::Real n_A = 6.0221417930e23;

        // conversion factor for nuclear energy generation rate
        constexpr amrex::Real enuc_conv2 = -n_A * c_light * c_light;
    }
}

#endif
