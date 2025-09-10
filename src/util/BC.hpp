#ifndef BC_HPP_
#define BC_HPP_

#include "AMReX_BC_TYPES.H"
#include "AMReX_BCRec.H"
#include "AMReX_Vector.H"
#include "hydro/hydro_system.hpp"
#include <array>
#include <radiation/radiation_system.hpp>

namespace quokka {

namespace BoundaryCondition {
    // Special boundary condition for reflecting walls
    // Uses both reflect_odd and reflect_even depending on component
    constexpr int reflecting = 8881;
} // namespace BoundaryCondition

namespace detail {

    // Check if a component is a normal momentum component in a given dimension
    template <typename problem_t>
    constexpr bool isNormalMomentumComponent(int n, int dim) {
        if ((n == HydroSystem<problem_t>::x1Momentum_index) && (dim == 0)) {
            return true;
        }
        if ((n == HydroSystem<problem_t>::x2Momentum_index) && (dim == 1)) {
            return true;
        }
        if ((n == HydroSystem<problem_t>::x3Momentum_index) && (dim == 2)) {
            return true;
        }
        return false;
    }
} // namespace detail

// Three parameter version - sets each dimension separately
template <typename problem_t>
amrex::Vector<amrex::BCRec> BC(int bc_x, int bc_y, int bc_z) {
    const int ncomp_cc = Physics_Indices<problem_t>::nvar_;
    amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
    
    std::array<int, 3> bcs = {bc_x, bc_y, bc_z};
    
    for (int n = 0; n < ncomp_cc; ++n) {
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            if (bcs[i] == BoundaryCondition::reflecting) {
                // For reflecting boundaries, use reflect_odd for normal momentum components
                // and reflect_even for all other components (including tangential momentum)
                if (detail::isNormalMomentumComponent<problem_t>(n, i)) {
                    BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
                    BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
                } else {
                    BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
                    BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
                }
            } else {
                // For all other boundaries, use the AMReX BC type directly
                BCs_cc[n].setLo(i, bcs[i]);
                BCs_cc[n].setHi(i, bcs[i]);
            }
        }
    }
    
    return BCs_cc;
}

// Single parameter version - sets all dimensions to the same boundary condition
template <typename problem_t>
amrex::Vector<amrex::BCRec> BC(int bc) {
    return BC<problem_t>(bc, bc, bc);
}

} // namespace quokka

#endif // BC_HPP_
