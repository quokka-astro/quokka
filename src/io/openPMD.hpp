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
#include <memory>
#include <string>

// AMReX headers
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
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
// Stores n contiguous values from data (device memory in GPU builds) as one chunk.
//
// Device pointers must not reach ADIOS2. GPU-enabled ADIOS2 treats GPU buffers as
// column-major and reverses the dimensions of multidimensional chunks
// (ornladios/ADIOS2#5241), and ADIOS2 without GPU support reads them from the CPU,
// which is very slow on AMD GPUs. As in WarpX, copy straight into a span of the
// backend's own buffer instead. Without span support, openPMD queues a plain host
// buffer from this allocator.
template <typename T>
void StoreDeviceChunk(openPMD::RecordComponent &record, const T *data, std::size_t n, const openPMD::Offset &offset, const openPMD::Extent &extent)
{
	auto view = record.storeChunk<T>(offset, extent, [](std::size_t size) { return std::shared_ptr<T>(new T[size], std::default_delete<T[]>()); });
	// The span is valid until the next storeChunk, so copy synchronously.
	auto span = view.currentBuffer();
	amrex::Gpu::dtoh_memcpy(span.data(), data, n * sizeof(T));
}
// Creates a file-based series for writing, e.g. filename = "plt%05T.bp".
auto CreateSeries(const std::string &filename) -> openPMD::Series;
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