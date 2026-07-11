//==============================================================================
// Copyright 2025 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDResistiveEnergyFluxKernel.cpp
/// \brief Unit/algorithm test for MHDSystem::AddResistiveEnergyFlux().
///
/// This test never constructs a QuokkaSimulation or calls evolve(): it builds a
/// static, hand-chosen B-field snapshot directly, calls AddResistiveEnergyFlux()
/// once, and compares the result against values computed independently (see the
/// Python replica used to derive them) at a few grid points. Because nothing is
/// time-integrated, there is no risk of the check being confounded by any
/// self-consistent dynamics (e.g. the initial condition need not be in force
/// balance); this isolates the discrete flux formula itself, which is the piece
/// that a full-simulation test cannot discriminate (AddResistiveEnergyFlux only
/// ever adds a flux-difference correction, so its net contribution to the
/// domain-integrated energy is forced to be correct by conservation regardless
/// of whether the formula itself is right).
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"

#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include "physics_info.hpp"

struct MHDResistiveEnergyFluxKernel {
};

template <> struct quokka::EOS_Traits<MHDResistiveEnergyFluxKernel> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDResistiveEnergyFluxKernel> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr ResistivityModel resistivity_model = ResistivityModel::constant;
};

auto problem_main() -> int
{
	// synthetic field: B_x = 0, B_y = amp_y*sin(k*x), B_z = amp_z*cos(k*x) (varies only along x).
	// Both cross terms of (J x B)_x = J_y*B_z - J_z*B_y are nonzero, so this exercises the full
	// four-edge averaging formula, not just the degenerate case that hid the original bug.
	constexpr double amp_y = 0.3;
	constexpr double amp_z = 0.4;
	constexpr double k_mode = 2.0 * M_PI;
	constexpr double resistivity = 0.5;
	constexpr double dx_val = 0.1;

	const amrex::Box domain_box(amrex::IntVect(0, 0, 0), amrex::IntVect(15, 7, 7));
	amrex::BoxArray const ba_cc(domain_box);
	amrex::DistributionMapping const dm(ba_cc);
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{dx_val, dx_val, dx_val};
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo{0.0, 0.0, 0.0};

	const int nvars_fc = Physics_Indices<MHDResistiveEnergyFluxKernel>::nvarPerDim_fc;
	const int nvars_cc = Physics_Indices<MHDResistiveEnergyFluxKernel>::nvarTotal_cc;
	constexpr int bfield_index = MHDSystem<MHDResistiveEnergyFluxKernel>::bfield_index;
	constexpr int energy_idx = HydroSystem<MHDResistiveEnergyFluxKernel>::energy_index;

	std::array<amrex::MultiFab, AMREX_SPACEDIM> bfield_fc;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux_fc;
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_fc = amrex::convert(ba_cc, amrex::IntVect::TheDimensionVector(idim));
		bfield_fc[idim] = amrex::MultiFab(ba_fc, dm, nvars_fc, 0);
		flux_fc[idim] = amrex::MultiFab(ba_fc, dm, nvars_cc, 0);
		bfield_fc[idim].setVal(0.0);
		flux_fc[idim].setVal(0.0);
	}

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		for (amrex::MFIter mfi(bfield_fc[idim]); mfi.isValid(); ++mfi) {
			const amrex::Box &box_fc = mfi.validbox();
			auto const &b_arr = bfield_fc[idim].array(mfi);
			amrex::ParallelFor(box_fc, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				// every face array here is cell-centred in x (the field only varies with x),
				// so the x-coordinate formula is the same regardless of which direction idim is.
				const double x = prob_lo[0] + (i + 0.5) * dx[0];
				if (idim == 1) {
					b_arr(i, j, k, bfield_index) = amp_y * std::sin(k_mode * x);
				} else if (idim == 2) {
					b_arr(i, j, k, bfield_index) = amp_z * std::cos(k_mode * x);
				}
				// idim == 0 (B_x) stays zero.
			});
		}
	}

	MHDSystem<MHDResistiveEnergyFluxKernel>::AddResistiveEnergyFlux(flux_fc, bfield_fc, dx, resistivity);

	// expected values computed independently (not by re-running this code) by replicating
	// AddResistiveEnergyFlux's discrete formula in Python for this exact field and grid.
	struct ExpectedPoint {
		int i, j, k;
		double flux_eta;
	};
	const std::array<ExpectedPoint, 3> expected{{
	    {2, 3, 3, 6.046101299219208e-02},
	    {4, 3, 3, -9.782797401561581e-02},
	    {6, 3, 3, 9.782797401561583e-02},
	}};

	constexpr double reltol = 1.0e-10;
	bool all_ok = true;
	auto const &flux_arr = flux_fc[0].array(0);
	for (const auto &pt : expected) {
		const double computed = flux_arr(pt.i, pt.j, pt.k, energy_idx);
		const double rel_err = std::abs(computed - pt.flux_eta) / std::abs(pt.flux_eta);
		amrex::Print() << "(i,j,k)=(" << pt.i << "," << pt.j << "," << pt.k << "): computed=" << computed << " expected=" << pt.flux_eta
			       << " rel_err=" << rel_err << "\n";
		if (!(rel_err < reltol)) {
			all_ok = false;
		}
	}

	int status = 1;
	if (all_ok) {
		status = 0;
		amrex::Print() << "test passed\n";
	} else {
		amrex::Print() << "test failed\n";
	}
	return status;
}
