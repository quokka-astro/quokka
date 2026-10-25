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
void SetupMeshComponent(openPMD::Mesh &mesh, const amrex::Geometry &full_geom, amrex::IndexType ixType = amrex::IndexType::TheCellType());
auto GetMeshComponentName(int meshLevel, std::string const &field_name) -> std::string;
void WriteCheckpointParticleAttributes(openPMD::Iteration &iteration, const ParticleContainer &particles, const std::string &name, std::uint64_t total);

} // namespace detail

void WriteFile(const std::vector<std::string> &varnames, int output_levels, const amrex::Vector<const amrex::MultiFab *> &mf,
	       const amrex::Vector<amrex::Geometry> &geom, const std::string &output_basename, amrex::Real time, int file_number,
	       const std::function<void(openPMD::Series &, openPMD::Iteration &)> &writeParticles = {});

// Checkpoints use the plotfile layout: a file-based series (<basename>%07T.bp)
// whose iteration holds one mesh per field and level, named as in plotfiles,
// and the plotfile particle species. Face fields are staggered meshes. quokka_*
// iteration attributes hold the additional state needed to restart.
auto CreateCheckpointSeries(const std::string &basename) -> openPMD::Series;
auto OpenCheckpointSeries(const std::string &path) -> openPMD::Series;
// Returns the checkpoint's only iteration after checking its version.
auto CheckpointIteration(openPMD::Series &series) -> openPMD::Iteration;
void SetCheckpointVersion(openPMD::Iteration &iteration);
void WriteCheckpointNames(openPMD::Iteration &iteration, const std::string &key, const amrex::Vector<std::string> &names);
auto CheckpointNamesMatch(openPMD::Iteration &iteration, const std::string &key, const amrex::Vector<std::string> &names) -> bool;
// Valid data only; records the level's cell-centered BoxArray. Flushes before returning.
void WriteCheckpointMesh(openPMD::Iteration &iteration, const amrex::MultiFab &mf, const amrex::Geometry &geom, int lev,
			 const amrex::Vector<std::string> &names);
auto ReadCheckpointBoxArray(openPMD::Iteration &iteration, int lev) -> amrex::BoxArray;
// Defines mf on the saved patches (converted to ixType) with a default distribution.
void ReadCheckpointMesh(openPMD::Iteration &iteration, amrex::MultiFab &mf, const amrex::Box &cellDomain, int lev, amrex::IndexType ixType,
			const amrex::Vector<std::string> &names);
auto HasCheckpointParticles(openPMD::Iteration &iteration, const std::string &name) -> bool;
void ReadCheckpointParticles(openPMD::Iteration &iteration, ParticleContainer &particles, const std::string &name);

// Base units in SI, ordered mass, length, time, temperature. NaN denotes unknown units.
// With `restart`, also record the population's restart attributes, even when it is empty.
void WriteParticles(openPMD::Series &series, openPMD::Iteration &iteration, const ParticleContainer &particles, const std::string &name,
		    const std::array<double, 4> &units, bool restart = false);

} // namespace quokka::OpenPMDOutput

#endif // OPENPMD_HPP_