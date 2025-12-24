#ifndef PROJECTION_HPP_ // NOLINT
#define PROJECTION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file projection.hpp
///  \brief AMReX I/O for 2D projections

#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// AMReX headers
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Orientation.H"
#include "AMReX_VisMF.H"
#include <AMReX.H>

// YAML headers
#include <yaml-cpp/yaml.h>

// Forward declarations
namespace quokka
{
template <typename problem_t> class PhysicsParticleRegister;
} // namespace quokka

namespace quokka::diagnostics
{

namespace detail
{

auto direction_to_string(amrex::Direction dir) -> std::string;
auto transform_box_to_2D(amrex::Direction const &dir, amrex::Box const &box) -> amrex::Box;
auto transform_realbox_to_2D(amrex::Direction const &dir, amrex::RealBox const &box) -> amrex::RealBox;

void printLowerDimIntVect(std::ostream &a_File, const amrex::IntVect &a_IntVect, int skipDim);
void printLowerDimBox(std::ostream &a_File, const amrex::Box &a_box, int skipDim);

void Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice,
			       const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time,
			       const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref);

void Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray, const amrex::Vector<std::string> &varnames,
			   const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time, const amrex::Vector<int> &level_steps,
			   const amrex::Vector<amrex::IntVect> &ref_ratio, const std::string &versionName, const std::string &levelPrefix,
			   const std::string &mfPrefix);

void VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name);

void Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm);

void Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr,
		   amrex::VisMF::Header::Version /*whichVersion*/, amrex::NFilesIter &nfi, int nOutFiles, MPI_Comm comm);

void write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar);

} // namespace detail

template <typename ReduceOp, typename F>
auto ComputePlaneProjection(amrex::Vector<amrex::MultiFab> const &state_new, const int finest_level, amrex::Vector<amrex::Geometry> const &geom,
			    amrex::Vector<amrex::IntVect> const &ref_ratio, const amrex::Direction dir, F const &user_f) -> amrex::Vector<amrex::MultiFab>
{
	// compute plane-parallel projection of user_f(i, j, k, state) along the given axis.
	const BL_PROFILE("quokka::DiagProjection::computePlaneProjection()");

	amrex::Vector<amrex::MultiFab> projections(finest_level + 1);

	for (int lev = 0; lev <= finest_level; ++lev) {
		auto const &level_ba = state_new[lev].boxArray();
		amrex::BoxList bl(level_ba.ixType());
		for (int i = 0; i < level_ba.size(); ++i) {
			bl.push_back(detail::transform_box_to_2D(dir, level_ba[i]));
		}
		bl.simplify();
		amrex::BoxArray ba2d(std::move(bl));
		ba2d.removeOverlap();
		amrex::DistributionMapping dm2d(ba2d);

		projections[lev].define(ba2d, dm2d, 1, 0);
		projections[lev].setVal(0.0);

		amrex::iMultiFab mask;
		if (lev == finest_level) {
			mask.define(state_new[lev].boxArray(), state_new[lev].DistributionMap(), 1, amrex::IntVect(0));
			mask.setVal(1);
		} else {
			mask = amrex::makeFineMask(state_new[lev], state_new[lev + 1], amrex::IntVect(0), ref_ratio[lev], geom[lev].periodicity(), 1, 0);
		}

		auto const &state = state_new[lev].const_arrays();
		auto const &mask_arr = mask.const_arrays();
		auto const &dx = geom[lev].CellSizeArray();

		auto plane_local = amrex::ReduceToPlane<ReduceOp, amrex::Real>(
		    static_cast<int>(dir), geom[lev].Domain(), state_new[lev],
		    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) -> amrex::Real {
			    if (mask_arr[box_no](i, j, k) == 0) {
				    return 0.0;
			    }
			    return dx[static_cast<int>(dir)] * user_f(i, j, k, state[box_no]);
		    });
		amrex::ParallelDescriptor::ReduceRealSum(plane_local.dataPtr(), static_cast<int>(plane_local.size()));

		auto const plane_arr = plane_local.const_array();
		auto const proj_arr = projections[lev].arrays();
		amrex::ParallelFor(projections[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
			if (dir == amrex::Direction::x) {
				proj_arr[bx](i, j, k) = plane_arr(0, i, j);
			} else if (dir == amrex::Direction::y) {
				proj_arr[bx](i, j, k) = plane_arr(i, 0, j);
			} else {
				proj_arr[bx](i, j, k) = plane_arr(i, j, k);
			}
		});
		amrex::Gpu::streamSynchronize();
	}

	return projections;
}

void WriteProjection(amrex::Direction dir, std::unordered_map<std::string, amrex::Vector<amrex::MultiFab>> const &proj,
		     amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Real time, int istep,
		     const std::string &basename, const YAML::Node &simulationMetadata);

// Overload with particle support
template <typename problem_t>
void WriteProjection(amrex::Direction dir, std::unordered_map<std::string, amrex::Vector<amrex::MultiFab>> const &proj,
		     amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Real time, int istep,
		     const std::string &basename, quokka::PhysicsParticleRegister<problem_t> &particleRegister, const std::vector<std::string> &particleTypes,
		     const YAML::Node &simulationMetadata)
{
	const BL_PROFILE("quokka::diagnostics::WriteProjection(with particles)");

	// First, write the projection data using the base function (includes metadata)
	WriteProjection(dir, proj, geom, ref_ratio, time, istep, basename, simulationMetadata);

	// If no particle types specified, skip particle output
	if (particleTypes.empty()) {
		return;
	}

	// Construct the plotfile name (same directory as field data)
	const std::string filename = amrex::Concatenate(basename, istep, 5);

	// Write particles using the filtered method
	particleRegister.writePlotFileFiltered(filename, particleTypes);

	amrex::Print() << "  Wrote particles to projection " << filename << "\n";
}

} // namespace quokka::diagnostics

#endif // PROJECTION_HPP_
