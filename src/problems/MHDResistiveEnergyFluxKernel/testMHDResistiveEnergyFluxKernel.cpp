//==============================================================================
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDResistiveEnergyFluxKernel.cpp
/// \brief Defines a unit test for MHDSystem::AddResistiveEnergyFlux(); never evolves a simulation.
///

#include <cmath>

#include "AMReX_Arena.H"
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

	// indexing: fcw_mf_bfields[3: world-direction of the face normal]
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fcw_mf_bfields;
	// indexing: fcw_mf_fluxes[3: world-direction of the face normal]
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fcw_mf_fluxes;
	for (int wcomp0 = 0; wcomp0 < AMREX_SPACEDIM; ++wcomp0) {
		auto ba_fc = amrex::convert(ba_cc, amrex::IntVect::TheDimensionVector(wcomp0));
		// 1 ghost cell: AddResistiveEnergyFlux reads the B-field up to one cell beyond the
		// current index in transverse directions (four-edge EMF averaging).
		fcw_mf_bfields[wcomp0] = amrex::MultiFab(ba_fc, dm, nvars_fc, 1);
		// pinned arena: fluxes are read directly from host below, after the device kernel
		// writes them; the default GPU arena is not host-accessible.
		fcw_mf_fluxes[wcomp0] = amrex::MultiFab(ba_fc, dm, nvars_cc, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
		fcw_mf_bfields[wcomp0].setVal(0.0);
		fcw_mf_fluxes[wcomp0].setVal(0.0);
	}

	for (int wcomp0 = 0; wcomp0 < AMREX_SPACEDIM; ++wcomp0) {
		for (amrex::MFIter mfi(fcw_mf_bfields[wcomp0]); mfi.isValid(); ++mfi) {
			// fill the ghost cell too: the analytic field is well-defined at any x, and
			// AddResistiveEnergyFlux reads one cell beyond the valid box transversely.
			const amrex::Box box_fc = amrex::grow(mfi.validbox(), 1);
			auto const &fc_a4_bfield = fcw_mf_bfields[wcomp0].array(mfi);
			amrex::ParallelFor(box_fc, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				// every face array here is cell-centred in x (the field only varies with x),
				// so the x-coordinate formula is the same regardless of which direction wcomp0 is.
				const double x = prob_lo[0] + (i + 0.5) * dx[0];
				if (wcomp0 == 1) {
					fc_a4_bfield(i, j, k, bfield_index) = amp_y * std::sin(k_mode * x);
				} else if (wcomp0 == 2) {
					fc_a4_bfield(i, j, k, bfield_index) = amp_z * std::cos(k_mode * x);
				}
				// wcomp0 == 0 (B_x) stays zero.
			});
		}
	}

	MHDSystem<MHDResistiveEnergyFluxKernel>::AddResistiveEnergyFlux(fcw_mf_fluxes, fcw_mf_bfields, dx, resistivity);

	// expected values computed independently (not by re-running this code) by replicating
	// AddResistiveEnergyFlux's discrete formula in Python for this exact field and grid.
	struct ExpectedPoint {
		int i, j, k;
		double flux_eta;
	};
	const std::array<ExpectedPoint, 3> expected{{
	    {.i = 2, .j = 3, .k = 3, .flux_eta = 6.046101299219208e-02},
	    {.i = 4, .j = 3, .k = 3, .flux_eta = -9.782797401561581e-02},
	    {.i = 6, .j = 3, .k = 3, .flux_eta = 9.782797401561583e-02},
	}};

	constexpr double reltol = 1.0e-10;
	bool all_ok = true;
	auto const &fc_a4_flux = fcw_mf_fluxes[0].array(0);
	for (const auto &pt : expected) {
		const double computed = fc_a4_flux(pt.i, pt.j, pt.k, energy_idx);
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
