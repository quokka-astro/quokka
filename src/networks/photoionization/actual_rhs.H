#ifndef actual_rhs_H
#define actual_rhs_H

#include "burn_type.H"
#include "fundamental_constants.H"
#include <extern_parameters.H>
#include <actual_network.H>
#include <jacobian_utilities.H>

using namespace amrex;
using namespace network_rp;
using namespace eos_rp;
using namespace C;

void actual_rhs_init ();

// Species 0 (Index 0 in the state)         = electron
// Species 1 (Index 1 in the state)         = neutral hydrogen
// Species 2 (Index 2 in the state)         = ionized hydrogen
// Index net_ienuc(=3)                      = specific internal energy
// Radiation index 0 (Index 4 in the state) = number density of photons
// Radiation index 1 (Index 5 in the state) = flux of photons normalized by the flux magnitude before the burn (therefore, it is 1.0 at the beginning of burn)

AMREX_GPU_HOST_DEVICE AMREX_INLINE
auto get_recombination_coefficient(Real T) -> Real
{
    return 2.6e-13 * std::pow((T / 1.0e4), -0.7);
}

auto get_recombination_cooling_coefficient(Real T) -> Real
{
    return C::k_B * T * get_recombination_coefficient(T) * (0.684 - 0.0416 * std::log(T / 1.0e4));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_rhs (const burn_t& state, Array1D<Real, 1, INT_NEQS>& ydot)
{
    amrex::Real alpha_rec = get_recombination_coefficient(state.T);
    amrex::Real Lambda_rec = get_recombination_cooling_coefficient(state.T);
    amrex::Real c = C::c_light; // speed of light in cm/s
    amrex::Real sigma_photo = network::sigma_photoionization; // photoionization cross-section in cm^2
    amrex::Real E_ionization = network::E_ionization; // ionization energy released for hydrogen in ergs

    // Photoionization with electrons being carried
    amrex::Real ionization_term = state.xn[1] * c * sigma_photo * state.rn[0];
    amrex::Real recombination_term = alpha_rec * state.xn[0] * state.xn[2];
    amrex::Real ionization_energy = ionization_term * E_ionization;
    amrex::Real recombination_cooling = state.xn[0] * state.xn[2] * Lambda_rec;
    ydot(1) = ionization_term - recombination_term;
    ydot(2) = -ydot(1);
    ydot(3) = ydot(1);
    ydot(net_ienuc) = (ionization_energy - recombination_cooling) / state.rho;
    ydot(5) = -ionization_term;
    ydot(6) = -state.xn[1] * c * sigma_photo;
}

template<class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_jac (const burn_t& state, MatrixType& jac)
{
    amrex::Real alpha_rec = get_recombination_coefficient(state.T);
    amrex::Real Lambda_rec = get_recombination_cooling_coefficient(state.T);
    amrex::Real c = C::c_light; // speed of light in cm/s
    amrex::Real sigma_photo = network::sigma_photoionization; // photoionization cross-section in cm^2
    amrex::Real E_ionization = network::E_ionization; // ionization energy for hydrogen in ergs
    jac.zero();

    // Jacobian entries for photoionization with electrons being carried
    jac(1, 1) = -alpha_rec * state.xn[2];
    jac(1, 2) = c * sigma_photo * state.rn[0];
    jac(1, 3) = -alpha_rec * state.xn[0];
    jac(1, net_ienuc) = 0.0_rt;
    jac(1, 5) = state.xn[1] * c * sigma_photo;
    jac(1, 6) = 0.0_rt;
    
    jac(2, 1) = -jac(1, 1); jac(2, 2) = -jac(1, 2); jac(2, 3) = -jac(1, 3); jac(2, net_ienuc) = -jac(1, net_ienuc); jac(2, 5) = -jac(1, 5); jac(2, 6) = -jac(1, 6);
    jac(3, 1) = jac(1, 1); jac(3, 2) = jac(1, 2); jac(3, 3) = jac(1, 3); jac(3, net_ienuc) = jac(1, net_ienuc); jac(3, 5) = jac(1, 5); jac(3, 6) = jac(1, 6);

    jac(net_ienuc, 1) = -state.xn[2] * Lambda_rec / state.rho;
    jac(net_ienuc, 2) = c * sigma_photo * state.rn[0] * E_ionization / state.rho;
    jac(net_ienuc, 3) = -state.xn[0] * Lambda_rec / state.rho;
    jac(net_ienuc, net_ienuc) = 0.0_rt;
    jac(net_ienuc, 5) = state.xn[1] * c * sigma_photo * E_ionization / state.rho;
    jac(net_ienuc, 6) = 0.0_rt;

    jac(5, 1) = 0.0_rt;
    jac(5, 2) = -c * sigma_photo * state.rn[0];
    jac(5, 3) = 0.0_rt;
    jac(5, net_ienuc) = 0.0_rt;
    jac(5, 5) = -state.xn[1] * c * sigma_photo;
    jac(5, 6) = 0.0_rt;

    jac(6, 1) = 0.0_rt;
    jac(6, 2) = -c * sigma_photo * state.rn[1];
    jac(6, 3) = 0.0_rt;
    jac(6, net_ienuc) = 0.0_rt;
    jac(6, 5) = 0.0_rt;
    jac(6, 6) = -c * sigma_photo * state.xn[1];
}

#endif
