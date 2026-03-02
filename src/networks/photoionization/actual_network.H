#ifndef actual_network_H
#define actual_network_H

#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <fundamental_constants.H>
#include <network_properties.H>
#include <burn_type.H>

using namespace amrex;

void actual_network_init();

const std::string network_name = "photoionization";

namespace network
{
    constexpr amrex::Real sigma_photoionization = 1.5e-18_rt; // photoionization cross-section in cm^2
    constexpr amrex::Real E_ionization = 6.4e-12_rt; // ionization energy for hydrogen in ergs
}

namespace Rates
{

    enum NetworkRates
    {
    };

}

#endif
