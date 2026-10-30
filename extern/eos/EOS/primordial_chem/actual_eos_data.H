#ifndef actual_eos_data_H
#define actual_eos_data_H

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <network.H>

inline AMREX_GPU_MANAGED amrex::Real gammas[NumSpec];
inline AMREX_GPU_MANAGED amrex::Real spmasses[NumSpec];

#endif
