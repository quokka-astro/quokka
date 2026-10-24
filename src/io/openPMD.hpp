#ifndef OPENPMD_HPP_ // NOLINT
#define OPENPMD_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file openPMD.hpp
///  \brief openPMD I/O for snapshots

#include "particles/particle_container.hpp"
#include <array>
#include <functional>
#include <string>

// AMReX headers
#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>

// openPMD headers
#include "openPMD/openPMD.hpp"

namespace quokka::OpenPMDOutput
{

namespace detail
{

auto getReversedVec(const amrex::IntVect &v) -> std::vector<std::uint64_t>;
auto getReversedVec(const amrex::Real *v) -> std::vector<double>;
void SetupMeshComponent(openPMD::Mesh &mesh, amrex::Geometry &full_geom);
auto GetMeshComponentName(int meshLevel, std::string const &field_name) -> std::string;

} // namespace detail

void WriteFile(const std::vector<std::string> &varnames, int output_levels, const amrex::Vector<const amrex::MultiFab *> &mf,
	       const amrex::Vector<amrex::Geometry> &geom, const std::string &output_basename, amrex::Real time, int file_number,
	       const std::function<void(openPMD::Series &, openPMD::Iteration &)> &writeParticles = {});

// Restart payloads preserve patch storage (including duplicate faces) exactly.
void WriteCheckpointMultiFab(const amrex::MultiFab &mf, const std::string &path);
void ReadCheckpointMultiFab(amrex::MultiFab &mf, const std::string &path, int expectedComponents);
void WriteCheckpointParticles(const ParticleContainer &particles, const std::string &directory, const std::string &name);
void ReadCheckpointParticles(ParticleContainer &particles, const std::string &directory, const std::string &name);

// Base units in SI, ordered mass, length, time, temperature. NaN denotes unknown units.
void WriteParticles(openPMD::Series &series, openPMD::Iteration &iteration, const ParticleContainer &particles, const std::string &name,
		    const std::array<double, 4> &units);

} // namespace quokka::OpenPMDOutput

#endif // OPENPMD_HPP_