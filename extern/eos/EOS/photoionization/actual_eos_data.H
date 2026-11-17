#ifndef actual_eos_data_H
#define actual_eos_data_H

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <network.H>

extern AMREX_GPU_MANAGED amrex::Real gammas[NumSpec];
extern AMREX_GPU_MANAGED amrex::Real spmasses[NumSpec];

#endif
