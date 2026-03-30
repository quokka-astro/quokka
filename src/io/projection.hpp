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
#include <unordered_map>
#include <vector>

// AMReX headers
#include "AMReX_FillPatchUtil.H"
#include "AMReX_MFInterpolater.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Orientation.H"
#include "AMReX_SPACE.H"
#include "AMReX_VisMF.H"
#include "AMReX_iMultiFab.H"
#include <AMReX.H>
#include <AMReX_BC_TYPES.H>

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

inline auto ComputePlaneProjectionFromMultiFab(const amrex::Vector<const amrex::MultiFab *> &mfs, const int finest_level,
					       amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio,
					       const amrex::Direction dir, const int comp) -> amrex::Vector<amrex::MultiFab>
{
	// compute plane-parallel projection of a single MultiFab component along the given axis.
	const BL_PROFILE("quokka::DiagProjection::computePlaneProjectionFromMultiFab()");

	amrex::Vector<amrex::MultiFab> projections(finest_level + 1);
	amrex::Vector<amrex::Geometry> geom2d(finest_level + 1);

	for (int lev = 0; lev <= finest_level; ++lev) {
		const amrex::Box box2d = detail::transform_box_to_2D(dir, geom[lev].Domain());
		const amrex::RealBox domain2d = detail::transform_realbox_to_2D(dir, geom[lev].ProbDomain());
		geom2d[lev] = amrex::Geometry(box2d, &domain2d);
	}

	for (int lev = 0; lev <= finest_level; ++lev) {
		amrex::iMultiFab mask;
		if (lev == finest_level) {
			mask.define(mfs[lev]->boxArray(), mfs[lev]->DistributionMap(), 1, amrex::IntVect(0));
			mask.setVal(1);
		} else {
			mask = amrex::makeFineMask(*mfs[lev], *mfs[lev + 1], amrex::IntVect(0), ref_ratio[lev], geom[lev].periodicity(), 1, 0);
		}

		auto const &mf_arr = mfs[lev]->const_arrays();
		auto const &mask_arr = mask.const_arrays();
		auto const &dx = geom[lev].CellSizeArray();

		auto plane_pair = amrex::ReduceToPlaneMF2Patchy<amrex::ReduceOpSum>(static_cast<int>(dir), geom[lev].Domain(), *mfs[lev],
										    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) -> amrex::Real {
											    if (mask_arr[box_no](i, j, k) == 0) {
												    return 0.0;
											    }
											    return dx[static_cast<int>(dir)] * mf_arr[box_no](i, j, k, comp);
										    });
		auto &plane_global = plane_pair.second;
		const auto &plane_ba = plane_global.boxArray();
		amrex::BoxList bl2d(plane_ba.ixType());
		for (int i = 0; i < plane_ba.size(); ++i) {
			bl2d.push_back(detail::transform_box_to_2D(dir, plane_ba[i]));
		}
		amrex::BoxArray ba2d(std::move(bl2d));
		projections[lev].define(ba2d, plane_global.DistributionMap(), 1, 0);

		auto const &src_arr = plane_global.const_arrays();
		auto const &dst_arr = projections[lev].arrays();
		for (amrex::MFIter mfi(projections[lev]); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			const int box_no = mfi.LocalIndex();
			auto const &src = src_arr[box_no];
			auto const &dst = dst_arr[box_no];
			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::IntVect dst_iv{AMREX_D_DECL(i, j, k)};
				amrex::IntVect src_iv;
				if (dir == amrex::Direction::x) {
					src_iv = amrex::IntVect{AMREX_D_DECL(0, i, j)};
#if AMREX_SPACEDIM >= 2
				} else if (dir == amrex::Direction::y) {
					src_iv = amrex::IntVect{AMREX_D_DECL(i, 0, j)};
#endif
#if AMREX_SPACEDIM == 3
				} else if (dir == amrex::Direction::z) {
					src_iv = amrex::IntVect{AMREX_D_DECL(i, j, 0)};
#endif
				} else {
					src_iv = dst_iv;
				}
				dst(dst_iv, 0) = src(src_iv, 0);
			});
		}
		amrex::Gpu::streamSynchronize();
	}

	amrex::Vector<amrex::MultiFab> projections_accum = std::move(projections);

	for (int lev = 0; lev < finest_level; ++lev) {
		amrex::IntVect rr_2d{AMREX_D_DECL(1, 1, 1)};
		if (dir == amrex::Direction::x) {
			rr_2d = amrex::IntVect{AMREX_D_DECL(ref_ratio[lev][1], ref_ratio[lev][2], 1)};
#if AMREX_SPACEDIM >= 2
		} else if (dir == amrex::Direction::y) {
			rr_2d = amrex::IntVect{AMREX_D_DECL(ref_ratio[lev][0], ref_ratio[lev][2], 1)};
#endif
#if AMREX_SPACEDIM == 3
		} else if (dir == amrex::Direction::z) {
			rr_2d = amrex::IntVect{AMREX_D_DECL(ref_ratio[lev][0], ref_ratio[lev][1], 1)};
#endif
		}

		amrex::MultiFab coarse_on_fine_layout(amrex::coarsen(projections_accum[lev + 1].boxArray(), rr_2d),
						      projections_accum[lev + 1].DistributionMap(), 1, 0);
		coarse_on_fine_layout.ParallelCopy(projections_accum[lev], 0, 0, 1, 0, 0);

		auto const &coarse_arr = coarse_on_fine_layout.const_arrays();
		auto const &fine_arr = projections_accum[lev + 1].arrays();
		for (amrex::MFIter mfi(projections_accum[lev + 1]); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			const int box_no = mfi.LocalIndex();
			auto const &coarse = coarse_arr[box_no];
			auto const &fine = fine_arr[box_no];
			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
				fine(i, j, k, 0) += coarse(amrex::coarsen(iv, rr_2d), 0);
			});
		}
	}

	return projections_accum;
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
	const std::string filename = amrex::Concatenate(basename, istep, 7);

	// Write particles using the filtered method
	particleRegister.writePlotFileFiltered(filename, particleTypes);

	amrex::Print() << "  Wrote particles to projection " << filename << "\n";
}

} // namespace quokka::diagnostics

#endif // PROJECTION_HPP_
