#ifndef OPENPMD_HPP_ // NOLINT
#define OPENPMD_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file openPMD.hpp
///  \brief openPMD I/O for snapshots

#include <string>

// AMReX headers
#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>

// Particles
#include "particles/PhysicsParticles.hpp"

// openPMD headers
#include "openPMD/openPMD.hpp"

namespace quokka::OpenPMDOutput
{

namespace detail
{

auto getReversedVec(const amrex::IntVect &v) -> std::vector<std::uint64_t>;
auto getReversedVec(const amrex::Real *v) -> std::vector<double>;
void SetupMeshComponent(openPMD::Mesh &mesh, int /*meshLevel*/, const std::string &comp_name, amrex::Geometry &full_geom);
auto GetMeshComponentName(int meshLevel, std::string const &field_name) -> std::string;

} // namespace detail

void WriteFields(openPMD::Series &series, openPMD::Iteration &iteration, const std::vector<std::string> &varnames, int output_levels,
	amrex::Vector<const amrex::MultiFab *> &mf, amrex::Vector<amrex::Geometry> &geom);

void WriteFile(const std::vector<std::string> &varnames, int output_levels, amrex::Vector<const amrex::MultiFab *> &mf,
	amrex::Vector<amrex::Geometry> &geom, const std::string &output_basename, amrex::Real time, int file_number);

template <typename problem_t>
void WriteParticles(openPMD::Series &series, openPMD::Iteration &iteration,
	            PhysicsParticleRegister<problem_t> &particle_register, amrex::Real time);

} // namespace quokka::OpenPMDOutput

#include "openPMDParticles_impl.hpp"

#endif // OPENPMD_HPP_
