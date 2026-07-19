//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDBitwiseICs.cpp
/// \brief Defines a test problem to make sure face-centered quantities are created correctly.
///

#include <array>
#include <cassert>
#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct MHDBitwiseICs {
};

template <> struct quokka::EOS_Traits<MHDBitwiseICs> {
	static constexpr amrex::Real gamma = 5. / 3.;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDBitwiseICs> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

AMREX_GPU_DEVICE
void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_L = prob_lo[2] + k * dx[2];
	// the following is a simple setup to ensure that every cell has a different value, for each quantity
	if (cen == quokka::centering::cc) {
		const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];
		const amrex::Real x2_C = x2_L + static_cast<amrex::Real>(0.5) * dx[1];
		const amrex::Real x3_C = x3_L + static_cast<amrex::Real>(0.5) * dx[2];
		const double radius = (x1_C + x2_C + x3_C) * (x1_C + x2_C + x3_C);
		state(i, j, k, HydroSystem<MHDBitwiseICs>::density_index) = radius;
		state(i, j, k, HydroSystem<MHDBitwiseICs>::x1Momentum_index) = x1_C;
		state(i, j, k, HydroSystem<MHDBitwiseICs>::x2Momentum_index) = x2_C;
		state(i, j, k, HydroSystem<MHDBitwiseICs>::x3Momentum_index) = x3_C;
		state(i, j, k, HydroSystem<MHDBitwiseICs>::energy_index) = radius;
		state(i, j, k, HydroSystem<MHDBitwiseICs>::internalEnergy_index) = radius;
	} else if (cen == quokka::centering::fc) {
		if (dir == quokka::direction::x) {
			state(i, j, k, MHDSystem<MHDBitwiseICs>::bfield_index) = x1_L;
		} else if (dir == quokka::direction::y) {
			state(i, j, k, MHDSystem<MHDBitwiseICs>::bfield_index) = x2_L;
		} else if (dir == quokka::direction::z) {
			state(i, j, k, MHDSystem<MHDBitwiseICs>::bfield_index) = x3_L;
		}
	}
}

template <> void QuokkaSimulation<MHDBitwiseICs>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<MHDBitwiseICs>::nvarTotal_cc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir);
	});
}

template <> void QuokkaSimulation<MHDBitwiseICs>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<MHDBitwiseICs>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir);
	});
}

template <>
void QuokkaSimulation<MHDBitwiseICs>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);

		const int ncomp_cc = Physics_Indices<MHDBitwiseICs>::nvarTotal_cc;
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp_cc; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na);
		});
	}
}

template <>
void QuokkaSimulation<MHDBitwiseICs>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);

		const int ncomp_fc = Physics_Indices<MHDBitwiseICs>::nvarPerDim_fc;
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp_fc; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir);
		});
	}
}

auto verifyPeriodicBCs(const amrex::MultiFab &mf, const std::string &label) -> int
{
	// enforce test assumptions
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3, "verifyPeriodicBCs: test requires a 3D domain.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf.boxArray().size() == 1, "verifyPeriodicBCs: test only works for a single-FAB domain.");
	// make sure any prior device work has finished
	amrex::Gpu::synchronize();
	const amrex::IntVect nghosts = mf.nGrowVect();
	const int num_comps = mf.nComp();
	const amrex::IndexType box_centering = mf.boxArray().ixType();
	const amrex::IntVect is_nodal = box_centering.toIntVect();
	int num_diffs = 0;
	for (amrex::MFIter mf_iter(mf, false); mf_iter.isValid(); ++mf_iter) {
		const amrex::Box valid_region = mf_iter.validbox();
		const amrex::Box grown_region = amrex::grow(valid_region, nghosts);
		// We will want to inspect ghost cells vs wrapped valid cells on the host, but on GPU builds
		// the MultiFab data lives on the device, so here we will need to build a host mirror of the
		// full domain (valid + ghosts cells) and let AMReX handle the transfer.
#if defined(AMREX_USE_GPU)
		// for GPU-builds: use pinned host memory so a device kernel can write into this fab
		amrex::FArrayBox host_fab(grown_region, num_comps, amrex::The_Pinned_Arena());
		// device-side copy: read device-resident `mf[mf_iter]`, write into pinned host fab
		static_cast<void>(mf[mf_iter].template copyToMem<amrex::RunOn::Device>(grown_region, 0, num_comps, host_fab.dataPtr()));
		amrex::Gpu::synchronize();
#else
		// for CPU-builds: everything is already on the host, so just use a host copy
		amrex::FArrayBox host_fab(grown_region, num_comps);
		static_cast<void>(mf[mf_iter].template copyToMem<amrex::RunOn::Host>(grown_region, 0, num_comps, host_fab.dataPtr()));
#endif
		// read-only view of the host mirror we just filled
		auto const &fab_view = host_fab.const_array();
		// index-space bounds of the valid region
		const std::array<int, 3> valid_lower_indices{valid_region.smallEnd(0), valid_region.smallEnd(1), valid_region.smallEnd(2)};
		const std::array<int, 3> valid_upper_indices{valid_region.bigEnd(0), valid_region.bigEnd(1), valid_region.bigEnd(2)};
		// raw extent of valid box
		const std::array<int, 3> valid_extent{valid_upper_indices[0] - valid_lower_indices[0] + 1, valid_upper_indices[1] - valid_lower_indices[1] + 1,
						      valid_upper_indices[2] - valid_lower_indices[2] + 1};
		// number of distinct periodic entries per axis: N for face-centred, N-1 for cell-centred (last entry wraps to first)
		const std::array<int, 3> num_valid_entries{valid_extent[0] - is_nodal[0], valid_extent[1] - is_nodal[1], valid_extent[2] - is_nodal[2]};
		AMREX_ALWAYS_ASSERT(num_valid_entries[0] > 0 && num_valid_entries[1] > 0 && num_valid_entries[2] > 0);
		// map an index in the grown region back into the equivalent periodic index in the valid region along a given axis
		auto wrap_into_valid = [&](int index, int axis) -> int {
			const int wrapped_index = index - valid_lower_indices[axis];
			if (index < valid_lower_indices[axis]) {
				return valid_lower_indices[axis] + (wrapped_index + num_valid_entries[axis]);
			}
			if (index <= valid_upper_indices[axis]) {
				return index;
			}
			return valid_lower_indices[axis] + (wrapped_index - num_valid_entries[axis]);
		};
		auto get_num_elems = [](const amrex::Box &box) -> amrex::IntVect { return {box.length(0), box.length(1), box.length(2)}; };
		const amrex::IntVect ntotal = get_num_elems(grown_region);
		const amrex::IntVect nvalid = get_num_elems(valid_region);
		amrex::Print() << "[" << label << "] dims:"
			       << " ntotal=" << ntotal << " nvalid=" << nvalid << " (nghosts=" << nghosts << ", is_nodal=" << is_nodal << ")\n";
		// Inspect every cell and check that it matches its periodic counterpart in the valid region.
		// Note, we loop over the full domain, so for some fraction of the cells, we end up comparing valid cells with them-self.
		for (amrex::BoxIterator box_iter(grown_region); box_iter.ok(); ++box_iter) {
			const amrex::IntVect this_index = box_iter();
			const int wrapped_i = wrap_into_valid(this_index[0], 0);
			const int wrapped_j = wrap_into_valid(this_index[1], 1);
			const int wrapped_k = wrap_into_valid(this_index[2], 2);
			for (int icomp = 0; icomp < num_comps; ++icomp) {
				const amrex::Real this_value = fab_view(this_index[0], this_index[1], this_index[2], icomp);
				const amrex::Real wrapped_value = fab_view(wrapped_i, wrapped_j, wrapped_k, icomp);
				if (this_value != wrapped_value) {
					++num_diffs;
					amrex::Print()
					    << "[" << label << "]"
					    << " fab=" << mf_iter.index() << " coord=(" << this_index[0] << "," << this_index[1] << "," << this_index[2] << ")"
					    << " -> wrapped=(" << wrapped_i << "," << wrapped_j << "," << wrapped_k << ")"
					    << " icomp=" << icomp << " val=" << std::setprecision(17) << this_value << " vs ref=" << std::setprecision(17)
					    << wrapped_value << " (mismatch)\n";
				}
			}
		}
	}
	amrex::ParallelDescriptor::ReduceIntSum(num_diffs);
	if (num_diffs == 0) {
		amrex::Print() << "[" << label << "]"
			       << " periodic BC check: ghost and valid cells exactly match their wrapped values.\n";
	} else {
		amrex::Print() << "[" << label << "]"
			       << " periodic BC check: found " << num_diffs << " mismatched entries between ghost/valid cells and their wrapped values.\n";
	}
	amrex::Print() << "\n";
	return num_diffs;
}

auto problem_main() -> int
{

	QuokkaSimulation<MHDBitwiseICs> sim;

	sim.setInitialConditions();

	amrex::Vector<amrex::MultiFab> const &cc_state = sim.getNewMF_cc();
	amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &fc_state = sim.getNewMF_fc();

	int num_diffs = 0;
	const int amr_level = 0;
	amrex::Print() << "\n";
	num_diffs += verifyPeriodicBCs(cc_state[amr_level], "cell-centered");
	num_diffs += verifyPeriodicBCs(fc_state[amr_level][0], "FC-x");
	num_diffs += verifyPeriodicBCs(fc_state[amr_level][1], "FC-y");
	num_diffs += verifyPeriodicBCs(fc_state[amr_level][2], "FC-z");
	const bool diff_exist = (num_diffs != 0);

	sim.computeErrorNorm();

	return static_cast<int>(diff_exist);
}
