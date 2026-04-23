//==============================================================================
// Copyright 2026 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testFS18EMFStencil.cpp
/// \brief Verifies the Felker-Stone (2017) EMF reconstruction stencil using poison values.
///
/// The test targets E_z(i+1/2,j+1/2,k) on a single z-slice. It first computes a
/// baseline EMF from fully finite data, then poisons every input outside the
/// predicted dependency footprint with NaNs and verifies that the target EMF is
/// unchanged. For high-order reconstructions, it also perturbs the predicted
/// outer cell-centered and magnetic-face rings and requires the target EMF to
/// change, proving that the support boundary is actively used. Finally, it
/// poisons and perturbs the cell-centered corner blocks that sit one cell away
/// from the central edge-centered cross, distinguishing a full-box velocity
/// stencil from a cross-shaped stencil.

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <string_view>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Gpu.H"
#include "AMReX_MFIter.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include "main.hpp"
#include "physics_info.hpp"

struct FS18EMFStencil {
};

template <> struct quokka::EOS_Traits<FS18EMFStencil> {
	static constexpr amrex::Real gamma = 5. / 3.;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
	static constexpr amrex::Real boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<FS18EMFStencil> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

namespace
{

constexpr int nghost_cc = 6;
constexpr int nghost_fc = 6;
constexpr int target_i = 8;
constexpr int target_j = 8;
constexpr int target_k = 8;

auto targetEdgeIV() -> amrex::IntVect { return amrex::IntVect(AMREX_D_DECL(target_i, target_j, target_k)); }

enum class PoisonMode {
	none,
	expected_support,
	poison_cc_corner_blocks,
	perturb_outer_cc_ring,
	perturb_cc_corner_blocks,
	perturb_outer_bfield_ring,
};

auto validCellBox() -> amrex::Box { return {amrex::IntVect(AMREX_D_DECL(0, 0, 0)), amrex::IntVect(AMREX_D_DECL(15, 15, 15))}; }

auto supportRadius(int reconstructionOrder) -> int
{
	switch (reconstructionOrder) {
		case 1:
			return 0;
		case 2:
			return 1;
		case 3:
		case 5:
			return 2;
		default:
			amrex::Abort("Unsupported reconstruction order in FS18EMFStencil test.");
	}
	return -1;
}

auto schemeName(EMFAvgScheme scheme) -> std::string_view
{
	switch (scheme) {
		case EMFAvgScheme::LondrilloDelZanna2004:
			return "LondrilloDelZanna2004";
		case EMFAvgScheme::Balsara2025:
			return "Balsara2025";
	}
	return "unknown";
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto supportBounds(int radius) -> std::array<int, 4>
{
	int const i_lo = target_i - 1 - radius;
	int const i_hi = target_i + radius;
	int const j_lo = target_j - 1 - radius;
	int const j_hi = target_j + radius;
	return {i_lo, i_hi, j_lo, j_hi};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto inCellCenteredSupportBox(int i, int j, int k, int radius) -> bool
{
	auto const bounds = supportBounds(radius);
	return (k == target_k) && (i >= bounds[0]) && (i <= bounds[1]) && (j >= bounds[2]) && (j <= bounds[3]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto inCentralEdgeCross(int i, int j, int k, int radius) -> bool
{
	if (!inCellCenteredSupportBox(i, j, k, radius)) {
		return false;
	}
	bool const in_central_columns = (i >= (target_i - 1)) && (i <= target_i);
	bool const in_central_rows = (j >= (target_j - 1)) && (j <= target_j);
	return in_central_columns || in_central_rows;
}

void fillBaselineCellCentered(amrex::MultiFab &cc_mf)
{
	cc_mf.setVal(0.0);

	for (amrex::MFIter mfi(cc_mf, false); mfi.isValid(); ++mfi) {
		amrex::Box const full_box = cc_mf[mfi].box();
		auto const state = cc_mf[mfi].array();

		amrex::ParallelFor(full_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const ii = static_cast<amrex::Real>(i);
			amrex::Real const jj = static_cast<amrex::Real>(j);
			amrex::Real const kk = static_cast<amrex::Real>(k);
			const amrex::Real rho = 1.0 + 2.0e-4 * ii * ii + 3.0e-4 * jj * jj + 1.0e-4 * kk * kk;
			const amrex::Real vx = 1.0 + 0.015 * ii + 0.010 * jj + 0.008 * kk + 8.0e-4 * ii * ii + 3.0e-4 * jj * jj + 2.0e-4 * ii * jj;
			const amrex::Real vy = 1.6 + 0.008 * ii + 0.020 * jj + 0.005 * kk + 4.0e-4 * ii * ii + 7.0e-4 * jj * jj + 3.0e-4 * ii * jj;
			const amrex::Real vz = 0.4 + 0.006 * ii + 0.004 * jj + 0.009 * kk + 2.0e-4 * ii * ii + 2.0e-4 * jj * jj;
			state(i, j, k, HydroSystem<FS18EMFStencil>::density_index) = rho;
			state(i, j, k, HydroSystem<FS18EMFStencil>::x1Momentum_index) = rho * vx;
			state(i, j, k, HydroSystem<FS18EMFStencil>::x2Momentum_index) = rho * vy;
			state(i, j, k, HydroSystem<FS18EMFStencil>::x3Momentum_index) = rho * vz;
			state(i, j, k, HydroSystem<FS18EMFStencil>::energy_index) = 20.0;
			state(i, j, k, HydroSystem<FS18EMFStencil>::internalEnergy_index) = 10.0;
		});
	}
}

void fillBaselineFaceCentered(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_mf_cVars)
{
	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		fc_mf_cVars[dir].setVal(0.0);
		for (amrex::MFIter mfi(fc_mf_cVars[dir], false); mfi.isValid(); ++mfi) {
			amrex::Box const full_box = fc_mf_cVars[dir][mfi].box();
			auto const state = fc_mf_cVars[dir][mfi].array();

			amrex::ParallelFor(full_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real const ii = static_cast<amrex::Real>(i);
				amrex::Real const jj = static_cast<amrex::Real>(j);
				amrex::Real const kk = static_cast<amrex::Real>(k);
				amrex::Real const base = 1.0 + static_cast<amrex::Real>(dir);
				amrex::Real const bval =
				    base + 0.012 * ii + 0.018 * jj + 0.007 * kk + 7.0e-4 * ii * ii + 9.0e-4 * jj * jj + 2.0e-4 * kk * kk + 1.5e-4 * ii * jj;
				state(i, j, k, MHDSystem<FS18EMFStencil>::bfield_index) = bval;
			});
		}
	}
}

void fillBaselineWavespeeds(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_mf_fspds)
{
	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		fc_mf_fspds[dir].setVal(0.0);
		for (amrex::MFIter mfi(fc_mf_fspds[dir], false); mfi.isValid(); ++mfi) {
			amrex::Box const full_box = fc_mf_fspds[dir][mfi].box();
			auto const state = fc_mf_fspds[dir][mfi].array();

			amrex::ParallelFor(full_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real const ii = static_cast<amrex::Real>(i);
				amrex::Real const jj = static_cast<amrex::Real>(j);
				amrex::Real const kk = static_cast<amrex::Real>(k);
				amrex::Real const offset = 0.25 * static_cast<amrex::Real>(dir);
				state(i, j, k, 0) = 3.0 + offset + 0.006 * ii + 0.014 * jj + 0.005 * kk + 3.0e-4 * ii * ii + 5.0e-4 * jj * jj;
				state(i, j, k, 1) = 4.0 + offset + 0.013 * ii + 0.007 * jj + 0.004 * kk + 5.0e-4 * ii * ii + 3.0e-4 * jj * jj;
			});
		}
	}
}

void poisonCellCenteredOutsideSupport(amrex::MultiFab &cc_mf, int radius)
{
	amrex::Real const poison = std::numeric_limits<amrex::Real>::quiet_NaN();

	for (amrex::MFIter mfi(cc_mf, false); mfi.isValid(); ++mfi) {
		amrex::Box const full_box = cc_mf[mfi].box();
		auto const state = cc_mf[mfi].array();
		int const ncomp = cc_mf.nComp();

		amrex::ParallelFor(full_box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
			bool const keep = inCellCenteredSupportBox(i, j, k, radius);
			if (!keep) {
				state(i, j, k, n) = poison;
			}
		});
	}
}

void poisonCellCenteredCornerBlocks(amrex::MultiFab &cc_mf, int radius)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(radius > 0, "Corner-block poison requires radius > 0.");
	amrex::Real const poison = std::numeric_limits<amrex::Real>::quiet_NaN();

	for (amrex::MFIter mfi(cc_mf, false); mfi.isValid(); ++mfi) {
		amrex::Box const full_box = cc_mf[mfi].box();
		auto const state = cc_mf[mfi].array();
		int const ncomp = cc_mf.nComp();

		amrex::ParallelFor(full_box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
			bool const in_box = inCellCenteredSupportBox(i, j, k, radius);
			bool const in_cross = inCentralEdgeCross(i, j, k, radius);
			if (in_box && !in_cross) {
				state(i, j, k, n) = poison;
			}
		});
	}
}

void poisonBfieldOutsideSupport(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_mf_cVars, int radius)
{
	amrex::Real const poison = std::numeric_limits<amrex::Real>::quiet_NaN();
	int const i_lo = target_i - 1 - radius;
	int const i_hi = target_i + radius;
	int const j_lo = target_j - 1 - radius;
	int const j_hi = target_j + radius;

	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		for (amrex::MFIter mfi(fc_mf_cVars[dir], false); mfi.isValid(); ++mfi) {
			amrex::Box const full_box = fc_mf_cVars[dir][mfi].box();
			auto const state = fc_mf_cVars[dir][mfi].array();

			amrex::ParallelFor(full_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				bool keep = false;
				if (dir == 0) {
					keep = (k == target_k) && (i == target_i) && (j >= j_lo) && (j <= j_hi);
				} else if (dir == 1) {
					keep = (k == target_k) && (j == target_j) && (i >= i_lo) && (i <= i_hi);
				}
				if (!keep) {
					state(i, j, k, MHDSystem<FS18EMFStencil>::bfield_index) = poison;
				}
			});
		}
	}
}

void poisonWavespeedsOutsideSupport(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_mf_fspds, EMFAvgScheme scheme)
{
	amrex::Real const poison = std::numeric_limits<amrex::Real>::quiet_NaN();

	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		for (amrex::MFIter mfi(fc_mf_fspds[dir], false); mfi.isValid(); ++mfi) {
			amrex::Box const full_box = fc_mf_fspds[dir][mfi].box();
			auto const state = fc_mf_fspds[dir][mfi].array();
			int const ncomp = fc_mf_fspds[dir].nComp();

			amrex::ParallelFor(full_box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
				bool keep = false;
				if (scheme == EMFAvgScheme::LondrilloDelZanna2004) {
					if (dir == 0) {
						keep = (k == target_k) && (j == target_j) && (i == target_i || i == (target_i + 1));
					} else if (dir == 1) {
						keep = (k == target_k) && (i == target_i) && (j == target_j || j == (target_j + 1));
					}
				} else if (scheme == EMFAvgScheme::Balsara2025) {
					if (dir == 0) {
						keep = (k == target_k) && (i == target_i) && (j == target_j || j == (target_j + 1));
					} else if (dir == 1) {
						keep = (k == target_k) && (j == target_j) && (i == target_i || i == (target_i + 1));
					}
				}

				if (!keep) {
					state(i, j, k, n) = poison;
				}
			});
		}
	}
}

void perturbCellCenteredOuterRing(amrex::MultiFab &cc_mf, int radius)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(radius > 0, "Cell-centered outer-ring perturbation requires radius > 0.");

	auto const bounds = supportBounds(radius);
	int const i_lo = bounds[0];
	int const i_hi = bounds[1];
	int const j_lo = bounds[2];
	int const j_hi = bounds[3];
	int const inner_i_lo = i_lo + 1;
	int const inner_i_hi = i_hi - 1;
	int const inner_j_lo = j_lo + 1;
	int const inner_j_hi = j_hi - 1;

	for (amrex::MFIter mfi(cc_mf, false); mfi.isValid(); ++mfi) {
		amrex::Box const full_box = cc_mf[mfi].box();
		auto const state = cc_mf[mfi].array();

		amrex::ParallelFor(full_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			bool const in_support = (k == target_k) && (i >= i_lo) && (i <= i_hi) && (j >= j_lo) && (j <= j_hi);
			bool const in_inner = (i >= inner_i_lo) && (i <= inner_i_hi) && (j >= inner_j_lo) && (j <= inner_j_hi);
			if (in_support && !in_inner) {
				state(i, j, k, HydroSystem<FS18EMFStencil>::x1Momentum_index) +=
				    0.5 + 0.03 * static_cast<amrex::Real>(i) - 0.02 * static_cast<amrex::Real>(j);
				state(i, j, k, HydroSystem<FS18EMFStencil>::x2Momentum_index) +=
				    -0.4 + 0.01 * static_cast<amrex::Real>(i) + 0.04 * static_cast<amrex::Real>(j);
			}
		});
	}
}

void perturbCellCenteredCornerBlocks(amrex::MultiFab &cc_mf, int radius)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(radius > 0, "Corner-block perturbation requires radius > 0.");

	for (amrex::MFIter mfi(cc_mf, false); mfi.isValid(); ++mfi) {
		amrex::Box const full_box = cc_mf[mfi].box();
		auto const state = cc_mf[mfi].array();

		amrex::ParallelFor(full_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			bool const in_box = inCellCenteredSupportBox(i, j, k, radius);
			bool const in_cross = inCentralEdgeCross(i, j, k, radius);
			if (in_box && !in_cross) {
				state(i, j, k, HydroSystem<FS18EMFStencil>::x1Momentum_index) +=
				    0.9 + 0.05 * static_cast<amrex::Real>(i) - 0.03 * static_cast<amrex::Real>(j);
				state(i, j, k, HydroSystem<FS18EMFStencil>::x2Momentum_index) +=
				    -0.8 + 0.02 * static_cast<amrex::Real>(i) + 0.06 * static_cast<amrex::Real>(j);
			}
		});
	}
}

void perturbBfieldOuterRing(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_mf_cVars, int radius)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(radius > 0, "Magnetic outer-ring perturbation requires radius > 0.");

	int const i_lo = target_i - 1 - radius;
	int const i_hi = target_i + radius;
	int const j_lo = target_j - 1 - radius;
	int const j_hi = target_j + radius;
	int const inner_i_lo = i_lo + 1;
	int const inner_i_hi = i_hi - 1;
	int const inner_j_lo = j_lo + 1;
	int const inner_j_hi = j_hi - 1;

	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		for (amrex::MFIter mfi(fc_mf_cVars[dir], false); mfi.isValid(); ++mfi) {
			amrex::Box const full_box = fc_mf_cVars[dir][mfi].box();
			auto const state = fc_mf_cVars[dir][mfi].array();

			amrex::ParallelFor(full_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				bool in_support = false;
				bool in_inner = false;
				if (dir == 0) {
					in_support = (k == target_k) && (i == target_i) && (j >= j_lo) && (j <= j_hi);
					in_inner = (j >= inner_j_lo) && (j <= inner_j_hi);
				} else if (dir == 1) {
					in_support = (k == target_k) && (j == target_j) && (i >= i_lo) && (i <= i_hi);
					in_inner = (i >= inner_i_lo) && (i <= inner_i_hi);
				}

				if (in_support && !in_inner) {
					state(i, j, k, MHDSystem<FS18EMFStencil>::bfield_index) +=
					    0.7 + 0.02 * static_cast<amrex::Real>(i) + 0.01 * static_cast<amrex::Real>(j);
				}
			});
		}
	}
}

auto readSingleValue(amrex::MultiFab const &mf, amrex::IntVect const &iv) -> amrex::Real
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf.boxArray().size() == 1, "FS18EMFStencil test expects a single-FAB MultiFab.");

	for (amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi) {
		amrex::Box const valid_box = mfi.validbox();
		if (!valid_box.contains(iv)) {
			continue;
		}

#if defined(AMREX_USE_GPU)
		amrex::FArrayBox host_fab(valid_box, mf.nComp(), amrex::The_Pinned_Arena());
		static_cast<void>(mf[mfi].template copyToMem<amrex::RunOn::Device>(valid_box, 0, mf.nComp(), host_fab.dataPtr()));
		amrex::Gpu::synchronize();
#else
		amrex::FArrayBox host_fab(valid_box, mf.nComp());
		static_cast<void>(mf[mfi].template copyToMem<amrex::RunOn::Host>(valid_box, 0, mf.nComp(), host_fab.dataPtr()));
#endif
		return host_fab.const_array()(iv[0], iv[1], iv[2], 0);
	}

	amrex::Abort("Target edge index is outside the EMF MultiFab.");
	return 0.0;
}

void assertAllEMFsAreFinite(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_mf_emf_components, int reconstructionOrder, EMFAvgScheme scheme)
{
	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
		    !ec_mf_emf_components[dir].contains_nan(0, 1),
		    std::format("Finite baseline EMF contains NaNs in component {} for order {} / {}.", dir, reconstructionOrder, schemeName(scheme)));
	}
}

auto computeTargetEz(int reconstructionOrder, EMFAvgScheme scheme, PoisonMode poisonMode) -> amrex::Real
{
	amrex::Box const box_cc = validCellBox();
	amrex::BoxArray const ba_cc(box_cc);
	amrex::DistributionMapping const dm(ba_cc);

	amrex::MultiFab cc_mf_cVars(ba_cc, dm, Physics_Indices<FS18EMFStencil>::nvarTotal_cc, nghost_cc);

	std::array<amrex::MultiFab, AMREX_SPACEDIM> fc_mf_cVars;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fc_mf_fspds;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> ec_mf_emf_components;

	for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
		amrex::BoxArray ba_fc = amrex::convert(ba_cc, amrex::IntVect::TheDimensionVector(dir));
		fc_mf_cVars[dir].define(ba_fc, dm, Physics_Indices<FS18EMFStencil>::nvarPerDim_fc, nghost_fc);
		fc_mf_fspds[dir].define(ba_fc, dm, 2, nghost_fc);

		amrex::IntVect const edge_type =
		    amrex::IntVect::TheDimensionVector((dir + 1) % AMREX_SPACEDIM) + amrex::IntVect::TheDimensionVector((dir + 2) % AMREX_SPACEDIM);
		amrex::BoxArray ba_ec = amrex::convert(ba_cc, edge_type);
		ec_mf_emf_components[dir].define(ba_ec, dm, 1, 0);
		ec_mf_emf_components[dir].setVal(0.0);
	}

	fillBaselineCellCentered(cc_mf_cVars);
	fillBaselineFaceCentered(fc_mf_cVars);
	fillBaselineWavespeeds(fc_mf_fspds);
	amrex::Gpu::synchronize();

	if (poisonMode != PoisonMode::none) {
		int const radius = supportRadius(reconstructionOrder);
		if (poisonMode == PoisonMode::expected_support) {
			poisonCellCenteredOutsideSupport(cc_mf_cVars, radius);
			poisonBfieldOutsideSupport(fc_mf_cVars, radius);
			poisonWavespeedsOutsideSupport(fc_mf_fspds, scheme);
			amrex::Gpu::synchronize();
		} else if (poisonMode == PoisonMode::poison_cc_corner_blocks) {
			poisonCellCenteredCornerBlocks(cc_mf_cVars, radius);
			amrex::Gpu::synchronize();
		} else if (poisonMode == PoisonMode::perturb_outer_cc_ring) {
			perturbCellCenteredOuterRing(cc_mf_cVars, radius);
			amrex::Gpu::synchronize();
		} else if (poisonMode == PoisonMode::perturb_cc_corner_blocks) {
			perturbCellCenteredCornerBlocks(cc_mf_cVars, radius);
			amrex::Gpu::synchronize();
		} else if (poisonMode == PoisonMode::perturb_outer_bfield_ring) {
			perturbBfieldOuterRing(fc_mf_cVars, radius);
			amrex::Gpu::synchronize();
		}
	}

	MHDSystem<FS18EMFStencil>::ComputeEMF_FelkerStone2017(ec_mf_emf_components, cc_mf_cVars, fc_mf_cVars, fc_mf_fspds, reconstructionOrder,
							      SlopeLimiter::minmod, scheme);
	amrex::Gpu::synchronize();

	if (poisonMode == PoisonMode::none) {
		assertAllEMFsAreFinite(ec_mf_emf_components, reconstructionOrder, scheme);
	}

	return readSingleValue(ec_mf_emf_components[2], targetEdgeIV());
}

void checkCase(int reconstructionOrder, EMFAvgScheme scheme)
{
	amrex::Real const baseline = computeTargetEz(reconstructionOrder, scheme, PoisonMode::none);
	amrex::Real const poisoned = computeTargetEz(reconstructionOrder, scheme, PoisonMode::expected_support);

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(baseline),
					 std::format("Baseline E_z is not finite for order {} / {}.", reconstructionOrder, schemeName(scheme)));
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(poisoned),
					 std::format("Poisoned E_z is not finite for order {} / {}.", reconstructionOrder, schemeName(scheme)));

	amrex::Real const scale = std::max<amrex::Real>(1.0, std::abs(baseline));
	amrex::Real const tolerance = 128.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
	amrex::Real const diff = std::abs(poisoned - baseline);

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    diff <= tolerance,
	    std::format("Poisoning outside the predicted support changed E_z for order {} / {}: baseline = {}, poisoned = {}, |diff| = {}, tol = {}",
			reconstructionOrder, schemeName(scheme), baseline, poisoned, diff, tolerance));

	amrex::Print() << "Verified expected support for reconstruction order " << reconstructionOrder << " with averaging scheme " << schemeName(scheme)
		       << ": E_z = " << poisoned << "\n";

	if (supportRadius(reconstructionOrder) == 0) {
		return;
	}

	amrex::Real const poisoned_corners = computeTargetEz(reconstructionOrder, scheme, PoisonMode::poison_cc_corner_blocks);
	amrex::Real const poisoned_corner_diff = std::abs(poisoned_corners - baseline);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    !std::isfinite(poisoned_corners) || (poisoned_corner_diff > tolerance),
	    std::format("Poisoning the corner blocks one cell off the central edge cross did not affect E_z for order {} / {}. Baseline = {}, poisoned = {}, "
			"|diff| = {}, tol = {}. This suggests a cross-shaped stencil rather than the documented full box.",
			reconstructionOrder, schemeName(scheme), baseline, poisoned_corners, poisoned_corner_diff, tolerance));

	amrex::Real const perturbed_corners = computeTargetEz(reconstructionOrder, scheme, PoisonMode::perturb_cc_corner_blocks);
	amrex::Real const corner_diff = std::abs(perturbed_corners - baseline);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    corner_diff > tolerance,
	    std::format("Perturbing the corner blocks one cell off the central edge cross did not change E_z for order {} / {}. Baseline = {}, perturbed = {}, "
			"|diff| = {}. This suggests a cross-shaped stencil rather than the documented full box.",
			reconstructionOrder, schemeName(scheme), baseline, perturbed_corners, corner_diff));

	amrex::Real const perturbed_cc = computeTargetEz(reconstructionOrder, scheme, PoisonMode::perturb_outer_cc_ring);
	amrex::Real const cc_diff = std::abs(perturbed_cc - baseline);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    cc_diff > tolerance,
	    std::format("Perturbing the predicted outer cell-centered ring did not change E_z for order {} / {}. Baseline = {}, perturbed = {}, |diff| = {}",
			reconstructionOrder, schemeName(scheme), baseline, perturbed_cc, cc_diff));

	amrex::Real const perturbed_b = computeTargetEz(reconstructionOrder, scheme, PoisonMode::perturb_outer_bfield_ring);
	amrex::Real const b_diff = std::abs(perturbed_b - baseline);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    b_diff > tolerance,
	    std::format("Perturbing the predicted outer magnetic-face ring did not change E_z for order {} / {}. Baseline = {}, perturbed = {}, |diff| = {}",
			reconstructionOrder, schemeName(scheme), baseline, perturbed_b, b_diff));

	amrex::Print() << "Verified that cells one step off the central edge cross are active for reconstruction order " << reconstructionOrder
		       << " with averaging scheme " << schemeName(scheme) << ", so the velocity stencil is the full box.\n";
	amrex::Print() << "Verified active use of the predicted support boundary for reconstruction order " << reconstructionOrder << " with averaging scheme "
		       << schemeName(scheme) << ".\n";
}

} // namespace

auto problem_main() -> int
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3, "FS18EMFStencil test requires AMREX_SPACEDIM == 3.");

	amrex::Print() << "\nChecking Felker-Stone 2017 EMF stencil dependencies with poison values...\n";

	std::array<int, 4> const reconstruction_orders = {1, 2, 3, 5};
	std::array<EMFAvgScheme, 2> const averaging_schemes = {EMFAvgScheme::LondrilloDelZanna2004, EMFAvgScheme::Balsara2025};

	for (auto const order : reconstruction_orders) {
		for (auto const scheme : averaging_schemes) {
			checkCase(order, scheme);
		}
	}

	amrex::Print() << "FS18 EMF stencil poison test passed.\n\n";
	return 0;
}
