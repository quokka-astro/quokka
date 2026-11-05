#ifndef EOS_FLOORS_HPP_
#define EOS_FLOORS_HPP_
/// \file EOSFloors.hpp
/// \brief The bounds the EOSs clamp their inputs to, ported from Microphysics' eos() front end (EOSData and eos_init).
/// Only the bounds of the inputs Quokka gives an EOS are kept: temperature, density, specific internal energy and
/// pressure.

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::eos_floors
{

extern AMREX_GPU_MANAGED amrex::Real min_temp; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real max_temp; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real min_dens; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real max_dens; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real min_e;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real max_e;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real min_p;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real max_p;    // NOLINT

/// Read eos.min_temp and eos.min_dens; every other bound takes its default.
void ReadParameters();

} // namespace quokka::eos_floors

#endif // EOS_FLOORS_HPP_
