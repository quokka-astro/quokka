#ifndef GRID_HPP_ // NOLINT
#define GRID_HPP_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_REAL.H>

enum class centering { cc=0, fc, ec };
enum class direction { na=-1, x, y, z };

struct grid {
  const amrex::Array4<double>& array;
  const amrex::Box& indexRange;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi;
  enum centering cen;
  enum direction dir;
  grid(const amrex::Array4<double>& array, const amrex::Box& indexRange,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo,
       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi,
       centering cen, direction dir)
      : array(array), indexRange(indexRange), dx(dx), prob_lo(prob_lo),
        prob_hi(prob_hi), cen(cen), dir(dir) {}
};

#endif // GRID_HPP_