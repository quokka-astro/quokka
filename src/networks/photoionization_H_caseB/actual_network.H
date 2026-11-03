#ifndef actual_network_H
#define actual_network_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <string>

#include <burn_type.H>
#include <fundamental_constants.H>
#include <network_properties.H>

void actual_network_init();

const std::string network_name = "jaff-chem";

namespace network
{
}

namespace Rates
{

enum NetworkRates {};

}

void balance_charge(burn_t &state);

#endif
