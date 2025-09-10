#ifndef BC_HPP_
#define BC_HPP_

#include "AMReX_BC_TYPES.H"
#include "AMReX_BCRec.H"
#include "AMReX_Vector.H"
#include "hydro/hydro_system.hpp"
#include <array>

namespace quokka {

enum class BoundaryCondition {
    reflecting,  // Uses both reflect_odd and reflect_even depending on component
    ext_dir,     // External Dirichlet boundary
    int_dir,     // Internal/periodic boundary
    foextrap,    // First-order extrapolation
    hoextrap     // Higher-order extrapolation
};

namespace detail {
    // Helper function to convert BoundaryCondition enum to AMReX BCType
    constexpr int toAMReXBCType(BoundaryCondition bc) {
        switch (bc) {
            case BoundaryCondition::ext_dir:
                return amrex::BCType::ext_dir;
            case BoundaryCondition::int_dir:
                return amrex::BCType::int_dir;
            case BoundaryCondition::foextrap:
                return amrex::BCType::foextrap;
            case BoundaryCondition::hoextrap:
                return amrex::BCType::hoextrap;
            case BoundaryCondition::reflecting:
                // This will be handled specially in the main function
                return -999; // Placeholder
            default:
                return amrex::BCType::bogus;
        }
    }

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

// Single parameter version - sets all dimensions to the same boundary condition
template <typename problem_t>
amrex::Vector<amrex::BCRec> BC(BoundaryCondition bc) {
    return BC<problem_t>(bc, bc, bc);
}

// Three parameter version - sets each dimension separately
template <typename problem_t>
amrex::Vector<amrex::BCRec> BC(BoundaryCondition bc_x, BoundaryCondition bc_y, BoundaryCondition bc_z) {
    const int ncomp_cc = Physics_Indices<problem_t>::nvarTotal_cc;
    amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
    
    std::array<BoundaryCondition, 3> bcs = {bc_x, bc_y, bc_z};
    
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
                // For non-reflecting boundaries, use the same BC type for all components
                int amrex_bc_type = detail::toAMReXBCType(bcs[i]);
                BCs_cc[n].setLo(i, amrex_bc_type);
                BCs_cc[n].setHi(i, amrex_bc_type);
            }
        }
    }
    
    return BCs_cc;
}

} // namespace quokka

#endif // BC_HPP_
