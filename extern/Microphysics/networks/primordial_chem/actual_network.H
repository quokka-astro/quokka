#ifndef actual_network_H
#define actual_network_H

#include <string_view>
#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <fundamental_constants.H>
#include <network_properties.H>
#include <burn_type.H>

using namespace amrex;


constexpr std::string_view network_name = "pynucastro-cxx";

namespace network
{
}

namespace Rates
{

    enum NetworkRates
    {
    };

}


#endif
