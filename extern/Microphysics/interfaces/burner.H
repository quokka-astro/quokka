#ifndef BURNER_H
#define BURNER_H

#include <burn_type.H>
#include <integrator.H>

#include <ArrayUtilities.H>


template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void burner (BurnT& state, amrex::Real dt)
{
    integrator(state, dt);
}

#endif
