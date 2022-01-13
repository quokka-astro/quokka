#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"

#include "Gravity.hpp"

using Real = amrex::Real;

///
/// Multipole gravity data
///
// TODO(benwibking): move these into Gravity class as static member variables

AMREX_GPU_MANAGED Real multipole::volumeFactor;
AMREX_GPU_MANAGED Real multipole::parityFactor;

AMREX_GPU_MANAGED Real multipole::rmax;

AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> multipole::doSymmetricAddLo;
AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> multipole::doSymmetricAddHi;
AMREX_GPU_MANAGED bool multipole::doSymmetricAdd;

AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> multipole::doReflectionLo;
AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> multipole::doReflectionHi;

AMREX_GPU_MANAGED amrex::Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max>
    multipole::factArray;
AMREX_GPU_MANAGED amrex::Array1D<Real, 0, multipole::lnum_max> multipole::parity_q0;
AMREX_GPU_MANAGED amrex::Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max>
    multipole::parity_qC_qS;
