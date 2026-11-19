#ifndef BOUNDARY_FILL_HPP_ // NOLINT
#define BOUNDARY_FILL_HPP_
//==============================================================================
// Quokka -- two-moment radiation hydrodynamics on GPUs for astrophysics
// Copyright 2026 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file boundary_fill.hpp
/// \brief Physical boundary filling with TagParallelFor.
///
/// quokka::TagPhysBCFunct is a drop-in replacement for amrex::PhysBCFunct<amrex::GpuBndryFuncFab<F>>.
/// It fills the non-periodic physical-boundary ghost cells of an entire MultiFab with at most
/// one kernel launch per phase (faces, then edges, then corners), instead of up to three launches
/// (plus small host-to-device copies) per FAB.
///
/// The cells that are filled, the order in which they are filled, and the per-cell operations
/// (the AMReX built-in fill for foextrap/reflect/hoextrap/..., followed by the user fill) are the
/// same as in amrex::GpuBndryFuncFab::ccfcdoit, so results are expected to be bitwise identical.
///
/// Problems fill their boundaries by providing a host function that receives a quokka::BoundaryFill
/// and calls BoundaryFill::ForEachGhostCell() with a device lambda.

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

namespace quokka
{

/// One rectangular region of boundary ghost cells belonging to one FAB.
struct BoundaryTag {
	amrex::Array4<amrex::Real> fab; ///< the whole FAB (valid + ghost cells), so interior cells may be read
	amrex::Box bx;			///< the ghost cells to fill (in the index type of the data)
	amrex::IntVect side;		///< per direction: -1 = beyond the lower domain boundary, +1 = beyond the upper, 0 = neither

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto box() const noexcept -> amrex::Box const & { return bx; }
};

/// Number of fill phases: faces, edges (2D/3D), corners (3D).
constexpr int nBoundaryPhases = AMREX_SPACEDIM;

/// Context handed to a problem's boundary-fill hook for one MultiFab.
class BoundaryFill
{
      public:
	/// Run \p f(i, j, k, tag) on every physical-boundary ghost cell, after the AMReX built-in fill for that cell.
	///
	/// The cells are visited in phases: first the cells beyond exactly one domain face, then those beyond two
	/// (edges), then those beyond three (corners). Each phase is one kernel launch, and each phase finishes before
	/// the next begins, so \p f may read valid cells and ghost cells from earlier phases. Within a phase, the cells
	/// are filled concurrently, so \p f must not read other ghost cells of the same phase.
	///
	/// Call this exactly once per hook invocation. (If the hook does not call it, the built-in fill is applied
	/// afterwards on its own.)
	template <typename F> void ForEachGhostCell(F const &f) const;

	/// Lower corner of the problem domain, device-safe.
	[[nodiscard]] auto ProbLo() const -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> { return geom_.ProbLoArray(); }
	/// Upper corner of the problem domain, device-safe.
	[[nodiscard]] auto ProbHi() const -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> { return geom_.ProbHiArray(); }
	/// Cell size, device-safe.
	[[nodiscard]] auto CellSize() const -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> { return geom_.CellSizeArray(); }
	/// The level's domain, converted to the index type of the data being filled.
	[[nodiscard]] auto Domain() const -> amrex::Box const & { return domain_; }
	/// The level's geometry (host only).
	[[nodiscard]] auto Geom() const -> amrex::Geometry const & { return geom_; }
	/// Geometry data by value, for passing to legacy per-cell boundary functions.
	[[nodiscard]] auto GeomData() const -> amrex::GeometryData { return geom_.data(); }
	/// Simulation time at which the boundary values are needed.
	[[nodiscard]] auto Time() const -> amrex::Real { return time_; }
	/// First component of the MultiFab being filled.
	[[nodiscard]] auto DestComp() const -> int { return dcomp_; }
	/// Number of components being filled.
	[[nodiscard]] auto NumComp() const -> int { return ncomp_; }
	/// Index of the first component being filled within the full BCRec vector (AMReX "orig_comp").
	[[nodiscard]] auto OrigComp() const -> int { return orig_comp_; }
	/// Device pointer to the NumComp() BCRecs for the components being filled.
	[[nodiscard]] auto BCRecs() const -> amrex::BCRec const * { return bcr_; }
	/// Index type of the data being filled.
	[[nodiscard]] auto IndexType() const -> amrex::IndexType { return ixtype_; }
	/// True if the data are cell-centred.
	[[nodiscard]] auto IsCellCentered() const -> bool { return ixtype_.cellCentered(); }
	/// For face-centred data, the face-normal direction; -1 for cell-centred data.
	[[nodiscard]] auto FaceDirection() const -> int
	{
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (ixtype_.nodeCentered(idim)) {
				return idim;
			}
		}
		return -1;
	}
	/// The tag vectors for each phase, for hooks that launch their own kernels. Such hooks must also apply the
	/// built-in fill themselves (see ForEachGhostCell()).
	[[nodiscard]] auto PhaseTags(int phase) const -> amrex::TagVector<BoundaryTag> const & { return *phases_.at(phase); }

	BoundaryFill(amrex::Geometry const &geom, amrex::Box const &domain, amrex::IndexType ixtype, amrex::Real time, int dcomp, int ncomp, int orig_comp,
		     amrex::BCRec const *bcr, std::array<amrex::TagVector<BoundaryTag> const *, nBoundaryPhases> phases)
	    : geom_(geom), domain_(domain), ixtype_(ixtype), time_(time), dcomp_(dcomp), ncomp_(ncomp), orig_comp_(orig_comp), bcr_(bcr), phases_(phases)
	{
	}

	[[nodiscard]] auto WasFilled() const -> bool { return filled_; }

      private:
	amrex::Geometry const &geom_;
	amrex::Box domain_;
	amrex::IndexType ixtype_;
	amrex::Real time_;
	int dcomp_;
	int ncomp_;
	int orig_comp_;
	amrex::BCRec const *bcr_;
	std::array<amrex::TagVector<BoundaryTag> const *, nBoundaryPhases> phases_;
	mutable bool filled_ = false;
};

template <typename F> void BoundaryFill::ForEachGhostCell(F const &f) const
{
	filled_ = true;
	const amrex::Box domain = domain_;
	const amrex::BCRec *bcr = bcr_;
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

/// Fill boundaries with a legacy per-cell functor that has the amrex::GpuBndryFuncFab signature
/// f(iv, dest, dcomp, numcomp, geomdata, time, bcr, bcomp, orig_comp).
template <typename F> void FillWithPerCellFunctor(BoundaryFill const &bc, F const &f)
{
	const amrex::GeometryData geomdata = bc.GeomData();
	const amrex::Real time = bc.Time();
	const int dcomp = bc.DestComp();
	const int ncomp = bc.NumComp();
	const int orig_comp = bc.OrigComp();
	const amrex::BCRec *bcr = bc.BCRecs();
	bc.ForEachGhostCell([=] AMREX_GPU_DEVICE(int i, int j, int k, BoundaryTag const &tag) {
		AMREX_D_PICK(amrex::ignore_unused(j, k), amrex::ignore_unused(k), (void)0);
		f(amrex::IntVect(AMREX_D_DECL(i, j, k)), tag.fab, dcomp, ncomp, geomdata, time, bcr, 0, orig_comp);
	});
}

/// Hook adapter for a legacy per-cell functor (e.g. a fixed-value fill that does not depend on the problem).
template <typename F> struct PerCellBCHook {
	F f;
	void operator()(BoundaryFill const &bc) const { FillWithPerCellFunctor(bc, f); }
};

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
inline void AddBoundaryTags(amrex::Array4<amrex::Real> const &fab, amrex::Box const &bx, amrex::Geometry const &geom, amrex::Box const &domain,
			    std::array<amrex::Vector<BoundaryTag>, nBoundaryPhases> &tags)
{
	const amrex::IntVect len = bx.length();
	const amrex::IndexType ixtype = bx.ixType();

	amrex::Box gdomain = domain;
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
			tags.at(nfaces - 1).push_back(BoundaryTag{fab, region, side});
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

	const amrex::IndexType ixtype = mf.ixType();
	const amrex::Box domain = amrex::convert(geom_.Domain(), ixtype);

	// build the per-phase region lists on the host
	std::array<amrex::Vector<BoundaryTag>, nBoundaryPhases> tag_lists;
	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const amrex::Box bx = amrex::grow(mfi.validbox(), nghost);
		detail::AddBoundaryTags(mf.array(mfi), bx, geom_, domain, tag_lists);
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

	const BoundaryFill bc(geom_, domain, ixtype, time, icomp, ncomp, bccomp, d_bcr.data(), phase_ptrs);
	hook_(bc);
	if (!bc.WasFilled()) {
		bc.ForEachGhostCell([] AMREX_GPU_DEVICE(int /*i*/, int /*j*/, int /*k*/, BoundaryTag const & /*tag*/) {});
	}

	// make sure the kernels are done before d_bcr and the tag buffers are freed
	amrex::Gpu::streamSynchronize();
}

} // namespace quokka

#endif // BOUNDARY_FILL_HPP_
