#ifndef JACOBIAN_UTILITIES_H
#define JACOBIAN_UTILITIES_H

#ifndef AMREX_USE_GPU
#include <format>
#endif

#include <burn_type.H>
#include <extern_parameters.H>
#include <integrator_data.H>
#include <network_properties.H>

// Given the Jacobian element for d(spec) / dT, convert it
// to d(spec) / de.

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real temperature_to_energy_jacobian (const BurnT& state, const amrex::Real& jac_T)
{
    amrex::Real jac_e = jac_T / state.cv;

    return jac_e;
}


#ifndef AMREX_USE_GPU
template <typename BurnT, typename T>
AMREX_INLINE
void
print_jacobian (const T& jac) {

    constexpr int int_neqs = integrator_neqs<BurnT>();

    std::cout << std::format("{:9} ", "*");
    for (int i = 1; i <= NumSpec; ++i) {
        std::cout << std::format("{:9} ", short_spec_names_cxx[i-1]);
    }
    std::cout << std::format("{:9} ", "e") << std::endl;

    for (int j = 1; j <= int_neqs; ++j) {
        if (j < int_neqs) {
            std::cout << std::format("{:5} ", short_spec_names_cxx[j-1]);
        } else {
            std::cout << std::format("{:5} ", "e");
        }
        for (int i = 1; i <= int_neqs; ++i) {
            std::cout << std::format("{:9.3g} ", jac.get(i, j));
        }
        std::cout << std::endl;
    }
}
#endif

#endif
