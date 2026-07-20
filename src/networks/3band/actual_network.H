#ifndef actual_network_H
#define actual_network_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <burn_type.H>
#include <fundamental_constants.H>
#include <network_properties.H>

using namespace amrex;

void actual_network_init();

const std::string network_name = "3band";

namespace network
{
} // namespace network

namespace Rates
{

enum NetworkRates {};

}

#endif
