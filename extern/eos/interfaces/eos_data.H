#ifndef EOS_DATA_H
#define EOS_DATA_H

#include <AMReX.H>
#include <AMReX_REAL.H>

// store some global limits set at initialization time

namespace EOSData
{
  inline bool initialized;

  extern AMREX_GPU_MANAGED amrex::Real mintemp;
  extern AMREX_GPU_MANAGED amrex::Real maxtemp;
  extern AMREX_GPU_MANAGED amrex::Real mindens;
  extern AMREX_GPU_MANAGED amrex::Real maxdens;
  extern AMREX_GPU_MANAGED amrex::Real minx;
  extern AMREX_GPU_MANAGED amrex::Real maxx;
  extern AMREX_GPU_MANAGED amrex::Real minye;
  extern AMREX_GPU_MANAGED amrex::Real maxye;
  extern AMREX_GPU_MANAGED amrex::Real mine;
  extern AMREX_GPU_MANAGED amrex::Real maxe;
  extern AMREX_GPU_MANAGED amrex::Real minp;
  extern AMREX_GPU_MANAGED amrex::Real maxp;
  extern AMREX_GPU_MANAGED amrex::Real mins;
  extern AMREX_GPU_MANAGED amrex::Real maxs;
  extern AMREX_GPU_MANAGED amrex::Real minh;
  extern AMREX_GPU_MANAGED amrex::Real maxh;
}

#endif
