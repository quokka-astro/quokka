// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

#ifndef BOUNDARY_FILL_HPP_ // NOLINT
#define BOUNDARY_FILL_HPP_

/// \file boundary_fill.hpp
/// \brief Filling of physical-boundary ghost cells.
///
/// A problem fills its boundaries by specializing a host function that receives a quokka::BoundaryFill
/// and calls BoundaryFill::ForEachGhostCell() with a device lambda:
///
/// \code
/// template <> void AMRSimulation<MyProblem>::FillCustomBoundaryConditions(quokka::BoundaryFill const &bc)
/// {
/// 	const amrex::Real rho_in = ::rho_inflow; // runtime parameters are captured by value
/// 	bc.ForEachGhostCell([=] AMREX_GPU_DEVICE(int i, int j, int k, quokka::BoundaryTag const &tag) {
/// 		if (tag.side[0] == -1) { // beyond the lower x boundary
/// 			tag.fab(i, j, k, HydroSystem<MyProblem>::density_index) = rho_in;
/// 		}
/// 	});
/// }
/// \endcode
///
/// quokka::TagPhysBCFunct does the filling. It has the interface AMReX expects of a physical-BC functor
/// (amrex::PhysBCFunct), and fills all physical-boundary ghost cells of a MultiFab with one TagParallelFor
/// launch per phase (faces, then edges, then corners). The cells that are filled, their order, and the AMReX
/// built-in fill applied to each cell (foextrap, reflect_even, hoextrap, ...) are the same as for
/// amrex::GpuBndryFuncFab.

#include <array>

#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_Box.H"
#include "AMReX_FilCC_C.H"
#include "AMReX_FilFC_C.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MFIter.H"
#include "AMReX_MultiFab.H"
#include "AMReX_TagParallelFor.H"
#include "AMReX_Vector.H"

#include "grid.hpp"

namespace quokka
{

/// A rectangular region of physical-boundary ghost cells in one FAB.
struct BoundaryTag {
	amrex::Array4<amrex::Real> fab; ///< the whole FAB being filled (valid and ghost cells)
	amrex::Box bx;			///< the ghost cells in this region (in the index type of the data)
	amrex::IntVect side;		///< per direction: -1 if beyond the lower domain boundary, +1 if beyond the upper, otherwise 0
	amrex::Box domain;		///< the level's cell-centred index domain

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto box() const noexcept -> amrex::Box const & { return bx; }
};

/// Number of fill phases: faces, edges (2D and 3D), corners (3D).
constexpr int nBoundaryPhases = AMREX_SPACEDIM;

/// Describes one boundary fill (one MultiFab, one component range) to a problem's boundary hook.
class BoundaryFill
{
      public:
	/// Call \p f(i, j, k, tag) on every physical-boundary ghost cell, after the AMReX built-in fill of that cell.
	///
	/// Cells are visited in phases: first the cells beyond one domain face, then those beyond two (edges), then those
	/// beyond three (corners). A phase finishes before the next one starts, so \p f may read valid cells and ghost cells
	/// of earlier phases. Cells within a phase are filled concurrently, so \p f must not read ghost cells of the same phase.
	///
	/// Call this at most once per fill. If a hook does not call it, the built-in fill is applied on its own.
	template <typename F> void ForEachGhostCell(F const &f) const;

	/// Lower corner of the problem domain.
	[[nodiscard]] auto ProbLo() const -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> { return geom_.ProbLoArray(); }
	/// Upper corner of the problem domain.
	[[nodiscard]] auto ProbHi() const -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> { return geom_.ProbHiArray(); }
	/// Cell size of the level being filled.
	[[nodiscard]] auto CellSize() const -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> { return geom_.CellSizeArray(); }
	/// Geometry of the level being filled (host only).
	[[nodiscard]] auto Geom() const -> amrex::Geometry const & { return geom_; }
	/// Geometry data of the level being filled, by value (for device functions that take amrex::GeometryData).
	[[nodiscard]] auto GeomData() const -> amrex::GeometryData { return geom_.data(); }
	/// Time at which the boundary values are needed.
	[[nodiscard]] auto Time() const -> amrex::Real { return time_; }
	/// First component being filled.
	[[nodiscard]] auto DestComp() const -> int { return dcomp_; }
	/// Number of components being filled.
	[[nodiscard]] auto NumComp() const -> int { return ncomp_; }
	/// Boundary types of the \p n-th component being filled (host only).
	[[nodiscard]] auto HostBCRec(int n) const -> amrex::BCRec const & { return h_bcr_[n]; } // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	/// True if the data are cell-centred.
	[[nodiscard]] auto IsCellCentered() const -> bool { return ixtype_.cellCentered(); }
	/// For face-centred data, the face-normal direction; direction::na for cell-centred data.
	[[nodiscard]] auto FaceDirection() const -> quokka::direction
	{
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (ixtype_.nodeCentered(idim)) {
				return static_cast<quokka::direction>(idim);
			}
		}
		return quokka::direction::na;
	}

	/// (Constructed by TagPhysBCFunct.)
	BoundaryFill(amrex::Geometry const &geom, amrex::IndexType ixtype, amrex::Real time, int dcomp, int ncomp, amrex::BCRec const *h_bcr,
		     amrex::BCRec const *d_bcr, std::array<amrex::TagVector<BoundaryTag> const *, nBoundaryPhases> phases)
	    : geom_(geom), ixtype_(ixtype), time_(time), dcomp_(dcomp), ncomp_(ncomp), h_bcr_(h_bcr), d_bcr_(d_bcr), phases_(phases)
	{
	}

	/// True if ForEachGhostCell() has been called.
	[[nodiscard]] auto WasFilled() const -> bool { return filled_; }

      private:
	amrex::Geometry const &geom_;
	amrex::IndexType ixtype_;
	amrex::Real time_;
	int dcomp_;
	int ncomp_;
	amrex::BCRec const *h_bcr_;
	amrex::BCRec const *d_bcr_;
	std::array<amrex::TagVector<BoundaryTag> const *, nBoundaryPhases> phases_;
	mutable bool filled_ = false;
};

template <typename F> void BoundaryFill::ForEachGhostCell(F const &f) const
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!filled_, "BoundaryFill::ForEachGhostCell() must be called at most once per fill");
	filled_ = true;

	const amrex::Box domain = amrex::convert(geom_.Domain(), ixtype_); // built-in fills index the domain like the data
	const amrex::BCRec *bcr = d_bcr_;
	const int dcomp = dcomp_;
	const int ncomp = ncomp_;
	const bool cell_centered = ixtype_.cellCentered();

	for (int phase = 0; phase < nBoundaryPhases; ++phase) {
		amrex::TagVector<BoundaryTag> const &tags = *phases_.at(phase);
		if (tags.ntags == 0) {
			continue;
		}
		amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int i, int j, int k, BoundaryTag const &tag) noexcept {
			AMREX_D_PICK(amrex::ignore_unused(j, k), amrex::ignore_unused(k), (void)0);
			const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
			if (cell_centered) {
				amrex::FilccCell{}(iv, tag.fab, dcomp, ncomp, domain, bcr, 0);
			} else {
				amrex::FilfcFace{}(iv, tag.fab, dcomp, ncomp, domain, bcr, 0);
			}
			f(i, j, k, tag);
		});
	}
}

/// Fills physical-boundary ghost cells of a MultiFab using TagParallelFor.
/// \tparam Hook host-callable object with `void operator()(quokka::BoundaryFill const &) const`.
///
/// Has the interface AMReX expects of a physical-BC functor (FillPatchSingleLevel, FillPatchTwoLevels,
/// InterpFromCoarseLevel, FillPatcher).
template <typename Hook> class TagPhysBCFunct
{
      public:
	TagPhysBCFunct() = default;
	TagPhysBCFunct(amrex::Geometry const &geom, amrex::Vector<amrex::BCRec> const &bcr, Hook const &hook) : geom_(geom), bcr_(bcr), hook_(hook) {}

	void operator()(amrex::MultiFab &mf, int icomp, int ncomp, amrex::IntVect const &nghost, amrex::Real time, int bccomp);

	// For backward compatibility with amrex::PhysBCFunct
	void FillBoundary(amrex::MultiFab &mf, int icomp, int ncomp, amrex::IntVect const &nghost, amrex::Real time, int bccomp)
	{
		this->operator()(mf, icomp, ncomp, nghost, time, bccomp);
	}

      private:
	amrex::Geometry geom_;
	amrex::Vector<amrex::BCRec> bcr_;
	Hook hook_{};
};

namespace detail
{
/// Append the boundary regions of one FAB to the per-phase tag lists, replicating
/// amrex::GpuBndryFuncFab::ccfcdoit (AMReX_PhysBCFunct.H).
inline void AddBoundaryTags(amrex::Array4<amrex::Real> const &fab, amrex::Box const &bx, amrex::Geometry const &geom,
			    std::array<amrex::Vector<BoundaryTag>, nBoundaryPhases> &tags)
{
	const amrex::IntVect len = bx.length();
	const amrex::IndexType ixtype = bx.ixType();

	amrex::Box gdomain = amrex::convert(geom.Domain(), ixtype);
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		if (geom.isPeriodic(idim)) {
			gdomain.grow(idim, len[idim]);
		}
	}
	if (gdomain.contains(bx)) {
		return;
	}

	// enumerate the 3^D - 1 side vectors
	constexpr int nside = AMREX_D_TERM(3, *3, *3);
	for (int code = 0; code < nside; ++code) {
		amrex::IntVect side(0);
		int nfaces = 0;
		int c = code;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			side[idim] = (c % 3) - 1;
			c /= 3;
			nfaces += (side[idim] != 0) ? 1 : 0;
		}
		if (nfaces == 0) {
			continue;
		}
		amrex::Box region = gdomain;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (side[idim] == -1) {
				region = amrex::adjCellLo(region, idim, len[idim]);
			} else if (side[idim] == 1) {
				region = amrex::adjCellHi(region, idim, len[idim]);
			}
		}
		region = amrex::convert(region, ixtype) & bx;
		if (region.ok()) {
			tags.at(nfaces - 1).push_back(BoundaryTag{fab, region, side, geom.Domain()});
		}
	}
}
} // namespace detail

template <typename Hook>
void TagPhysBCFunct<Hook>::operator()(amrex::MultiFab &mf, int icomp, int ncomp, amrex::IntVect const &nghost, amrex::Real time, int bccomp)
{
	if (geom_.isAllPeriodic() || ncomp == 0) {
		return;
	}

	BL_PROFILE("quokka::TagPhysBCFunct::()"); // NOLINT(misc-const-correctness)

	AMREX_ASSERT(nghost.allLE(mf.nGrowVect()));
	// amrex::GpuBndryFuncFab fills node-centred data (including face-centred data in 1D) with a different
	// algorithm (nddoit), which is not replicated here.
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!mf.ixType().nodeCentered(), "quokka::TagPhysBCFunct does not support node-centred data");

	// build the per-phase region lists on the host
	std::array<amrex::Vector<BoundaryTag>, nBoundaryPhases> tag_lists;
	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const amrex::Box bx = amrex::grow(mfi.validbox(), nghost);
		detail::AddBoundaryTags(mf.array(mfi), bx, geom_, tag_lists);
	}

#ifdef AMREX_DEBUG
	// regions within one phase must not overlap (they are filled concurrently)
	for (auto const &list : tag_lists) {
		for (int a = 0; a < list.size(); ++a) {
			for (int b = a + 1; b < list.size(); ++b) {
				AMREX_ALWAYS_ASSERT(list[a].fab.p != list[b].fab.p || !list[a].bx.intersects(list[b].bx));
			}
		}
	}
#endif

	// the BCRecs for the components being filled, on the device
	// (declared before the TagVectors so that it is freed after their destructors synchronize the stream)
	amrex::Gpu::DeviceVector<amrex::BCRec> d_bcr(ncomp);
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, bcr_.begin() + bccomp, bcr_.begin() + bccomp + ncomp, d_bcr.begin());

	std::array<amrex::TagVector<BoundaryTag>, nBoundaryPhases> tag_vectors;
	std::array<amrex::TagVector<BoundaryTag> const *, nBoundaryPhases> phase_ptrs{};
	for (int phase = 0; phase < nBoundaryPhases; ++phase) {
		tag_vectors.at(phase).define(tag_lists.at(phase));
		phase_ptrs.at(phase) = &tag_vectors.at(phase);
	}

	const BoundaryFill bc(geom_, mf.ixType(), time, icomp, ncomp, bcr_.data() + bccomp, d_bcr.data(), phase_ptrs);
	hook_(bc);
	if (!bc.WasFilled()) {
		bc.ForEachGhostCell([] AMREX_GPU_DEVICE(int /*i*/, int /*j*/, int /*k*/, BoundaryTag const & /*tag*/) {});
	}

	// make sure the kernels are done before d_bcr and the tag buffers are freed
	amrex::Gpu::streamSynchronize();
}

} // namespace quokka

#endif // BOUNDARY_FILL_HPP_
