#ifndef EOS_TYPE_H
#define EOS_TYPE_H

#include <AMReX.H>
#include <network.H>

struct eos_base_t {};

struct eos_t:eos_base_t {
    amrex::Real rho{};
    amrex::Real T{};
    amrex::Real p{};
    amrex::Real e{};
    amrex::Real h{};
    amrex::Real s{};
    amrex::Real xn[NumSpec]{};
#ifdef PHOTOCHEMISTRY
    amrex::Real rn[NumChemRadEqs]{};
    amrex::Real re_thermal[NumThermalBands]{};
    amrex::Real c_hat{};
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
    amrex::Real dust_kappa{};
    amrex::Real e_dust_absorbed{};
#endif
#endif
#if NAUX_NET > 0
    amrex::Real aux[NumAux]{};
#endif
    amrex::Real dpdT{};
    amrex::Real dpdr{};
    amrex::Real dedT{};
    amrex::Real dedr{};
    amrex::Real dhdT{};
    amrex::Real dhdr{};
    amrex::Real dsdT{};
    amrex::Real dsdr{};
    amrex::Real dpde{};
    amrex::Real dpdr_e{};
    amrex::Real G{}; // fundamental derivative (Thompson 1971), we use equation 2.24 of Menikoff & Plohr 1989

    amrex::Real cv{};
    amrex::Real cp{};
    amrex::Real xne{};
    amrex::Real xnp{};
    amrex::Real eta{};
    amrex::Real pele{};
    amrex::Real ppos{};
    amrex::Real mu{};
    amrex::Real mu_e{};
    amrex::Real y_e{};
    amrex::Real gam1{};
    amrex::Real cs{};

    amrex::Real abar{};
    amrex::Real zbar{};

    amrex::Real conductivity{};

#ifdef NSE_NET
    amrex::Real mu_p{};   //chemical potential of proton
    amrex::Real mu_n{};   //chemical potential of neutron
#endif
};

struct eos_re_t:eos_base_t {
    amrex::Real rho{};
    amrex::Real T{};
    amrex::Real e{};
    amrex::Real xn[NumSpec]{};
#ifdef PHOTOCHEMISTRY
    amrex::Real rn[NumChemRadEqs]{};
    amrex::Real re_thermal[NumThermalBands]{};
    amrex::Real c_hat{};
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
    amrex::Real dust_kappa{};
    amrex::Real e_dust_absorbed{};
#endif
#endif
#if NAUX_NET > 0
    amrex::Real aux[NumAux]{};
#endif
    amrex::Real dedT{};
    amrex::Real dedr{};

    amrex::Real cv{};

    amrex::Real mu{};
    amrex::Real mu_e{};
    amrex::Real y_e{};
    amrex::Real eta{};

    amrex::Real abar{};
    amrex::Real zbar{};

#ifdef NSE_NET
    amrex::Real mu_p{};   //chemical potential of proton
    amrex::Real mu_n{};   //chemical potential of neutron
#endif
};

struct eos_rep_t:eos_base_t {
    amrex::Real rho{};
    amrex::Real T{};
    amrex::Real e{};
    amrex::Real p{};
    amrex::Real xn[NumSpec]{};
#if NAUX_NET > 0
    amrex::Real aux[NumAux]{};
#endif
#ifdef PHOTOCHEMISTRY
    amrex::Real rn[NumChemRadEqs]{};
    amrex::Real re_thermal[NumThermalBands]{};
    amrex::Real c_hat{};
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
    amrex::Real dust_kappa{};
    amrex::Real e_dust_absorbed{};
#endif
#endif
    amrex::Real dedT{};
    amrex::Real dedr{};

    amrex::Real dpdT{};
    amrex::Real dpdr{};
    amrex::Real dpdr_e{};
    amrex::Real dpde{};

    amrex::Real cv{};
    amrex::Real cp{};
    amrex::Real gam1{};
    amrex::Real cs{};

    amrex::Real mu{};
    amrex::Real mu_e{};
    amrex::Real y_e{};
    amrex::Real eta{};

    amrex::Real abar{};
    amrex::Real zbar{};

#ifdef NSE_NET
    amrex::Real mu_p{};   //chemical potential of proton
    amrex::Real mu_n{};   //chemical potential of neutron
#endif
};

struct eos_rh_t:eos_rep_t {
    amrex::Real dhdr{};
    amrex::Real dhdT{};
    amrex::Real h{};
};

struct eos_extra_t:eos_t {
    amrex::Real dpdA{};
    amrex::Real dpdZ{};
    amrex::Real dedA{};
    amrex::Real dedZ{};
};

struct eos_re_extra_t:eos_re_t {
    amrex::Real dedA{};
    amrex::Real dedZ{};
};

struct eos_rep_extra_t:eos_rep_t {
    amrex::Real dpdA{};
    amrex::Real dpdZ{};
    amrex::Real dedA{};
    amrex::Real dedZ{};
};

struct chem_eos_t:eos_base_t {
    amrex::Real rho{};
    amrex::Real T{};
    amrex::Real p{};
    amrex::Real e{};
    amrex::Real dpdT{};
    amrex::Real dpdr{};
    amrex::Real dpde{};
    amrex::Real dpdr_e{};
    amrex::Real dedT{};
    amrex::Real dedr{};
    amrex::Real G{};

    amrex::Real mu{};
    amrex::Real cs{};

    amrex::Real cv{};
    amrex::Real cp{};

    amrex::Real gam1{};
};


template <typename T>
concept has_energy = requires { std::declval<T&>().e; };

template <typename T>
concept has_enthalpy = requires { std::declval<T&>().h; };

template <typename T>
concept has_entropy = requires { std::declval<T&>().s; };

template <typename T>
concept has_pressure = requires { std::declval<T&>().p; };

template <typename T>
concept has_dpdA = requires { std::declval<T&>().dpdA; };

template <typename T>
concept has_dpdZ = requires { std::declval<T&>().dpdZ; };

template <typename T>
concept has_dedA = requires { std::declval<T&>().dedA; };

template <typename T>
concept has_dedZ = requires { std::declval<T&>().dedZ; };

template <typename T>
concept has_G = requires { std::declval<T&>().G; };

template <typename T>
concept has_pele_ppos = requires { std::declval<T&>().pele; };

template <typename T>
concept has_xne_xnp = requires { std::declval<T&>().xne; };

template <typename T>
concept has_eta = requires { std::declval<T&>().eta; };

template <typename T>
concept has_conductivity = requires { std::declval<T&>().conductivity; };

template <typename T>
concept has_xn = requires { std::declval<T&>().xn; };

template <typename T>
concept has_base_variables = requires { std::declval<T&>().rho; };

template <typename T>
inline
std::ostream& print_state (std::ostream& o, T const& eos_state)
{
    if constexpr (has_base_variables<T>) {
        o << "rho = " << eos_state.rho << std::endl;
        o << "T =   " << eos_state.T << std::endl;
        o << "xn = ";
        for (int n = 0; n < NumSpec; ++n) {
            o << eos_state.xn[n] << " ";
        }
        o << std::endl;
#if NAUX_NET > 0
        o << "aux = ";
        for (int n = 0; n < NumAux; ++n) {
            o << eos_state.aux[n] << " ";
        }
        o << std::endl;
#endif
#ifdef PHOTOCHEMISTRY
        o << "rn = ";
        for (int n = 0; n < NumChemRadEqs; ++n) {
            o << eos_state.rn[n] << " ";
        }
        o << std::endl;
        o << "re_thermal = ";
        for (int n = 0; n < NumThermalBands; ++n) {
            o << eos_state.re_thermal[n] << " ";
        }
        o << std::endl;
        o << "c_hat = " << eos_state.c_hat << std::endl;
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
        o << "dust_kappa = " << eos_state.dust_kappa << std::endl;
        o << "e_dust_absorbed = " << eos_state.e_dust_absorbed << std::endl;
#endif
#endif
    }
    if constexpr (has_pressure<T>) {
        o << "p = " << eos_state.p << std::endl;
    }
    if constexpr (has_energy<T>) {
        o << "e = " << eos_state.e << std::endl;
    }
    if constexpr (has_enthalpy<T>) {
        o << "h = " << eos_state.h << std::endl;
    }
    if constexpr (has_entropy<T>) {
        o << "s = " << eos_state.s << std::endl;
    }

    if constexpr (has_pressure<T>) {
        o << "dpdT = " << eos_state.dpdT << std::endl;
        o << "dpdr = " << eos_state.dpdr << std::endl;
    }
    if constexpr (has_energy<T>) {
        o << "dedT = " << eos_state.dedT << std::endl;
        o << "dedr = " << eos_state.dedr << std::endl;
    }
    if constexpr (has_enthalpy<T>) {
        o << "dhdT = " << eos_state.dhdT << std::endl;
        o << "dhdr = " << eos_state.dhdr << std::endl;
    }
    if constexpr (has_entropy<T>) {
        o << "dsdT = " << eos_state.dsdT << std::endl;
        o << "dsdr = " << eos_state.dsdr << std::endl;
    }
    if constexpr (has_pressure<T>) {
        o << "dpde = " << eos_state.dpde << std::endl;
        o << "dpdr_e = " << eos_state.dpdr_e  << std::endl;
    }

    if constexpr (has_base_variables<T>) {
        o << "cv = " << eos_state.cv << std::endl;
    }
    if constexpr (has_pressure<T>) {
        o << "cp = " << eos_state.cp << std::endl;
    }
    if constexpr (has_xne_xnp<T>) {
        o << "xne = " << eos_state.xne << std::endl;
        o << "xnp = " << eos_state.xnp << std::endl;
    }
    if constexpr (has_eta<T>) {
        o << "eta = " << eos_state.eta << std::endl;
    }
    if constexpr (has_pele_ppos<T>) {
        o << "pele = " << eos_state.pele << std::endl;
        o << "ppos = " << eos_state.ppos << std::endl;
    }
    if constexpr (has_base_variables<T>) {
        o << "mu = " << eos_state.mu << std::endl;
        o << "mu_e = " << eos_state.mu_e << std::endl;
        o << "y_e = " << eos_state.y_e << std::endl;
    }
    if constexpr (has_pressure<T>) {
        o << "gam1 = " << eos_state.gam1 << std::endl;
        o << "cs = " << eos_state.cs << std::endl;
    }

    if constexpr (has_base_variables<T>) {
        o << "abar = " << eos_state.abar << std::endl;
        o << "zbar = " << eos_state.zbar << std::endl;
    }

    if constexpr (has_dpdA<T>) {
        o << "dpdA = " << eos_state.dpdA << std::endl;
    }
    if constexpr (has_dpdZ<T>) {
        o << "dpdZ = " << eos_state.dpdZ << std::endl;
    }
    if constexpr (has_dedA<T>) {
        o << "dedA = " << eos_state.dedA << std::endl;
    }
    if constexpr (has_dedZ<T>) {
        o << "dedZ = " << eos_state.dedZ << std::endl;
    }
#ifdef CONDUCTIVITY
    if constexpr (has_conductivity<T>) {
        o << "conductivity = " << eos_state.conductivity << std::endl;
    }
#endif

    return o;
}

template <typename T, typename U = typename std::enable_if_t<std::is_base_of_v<eos_base_t, T>>>
inline
std::ostream& operator<< (std::ostream& o, T const& eos_state)
{
    return print_state(o, eos_state);
}

enum eos_input_t : std::uint8_t {
    eos_input_rt = 0,
    eos_input_rh,
    eos_input_tp,
    eos_input_rp,
    eos_input_re,
    eos_input_ps,
    eos_input_ph,
    eos_input_th};

// these are used to allow for a generic interface to the
// root finding

namespace EOS {

    enum eos_iter : std::uint8_t {
        itemp = 1,
        idens = 2,
        iener = 3,
        ienth = 4,
        ientr = 5,
        ipres = 6};

}

#endif
