//==============================================================================
// ABOUTME: Test program for Gaussian random vector field generator
// ABOUTME: Validates FewModesFT implementation with simple test case
//==============================================================================
/// \file test_few_modes_ft.cpp
/// \brief Test for FewModesFT Gaussian random vector field generator

#include <array>
#include <cmath>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_RealBox.H"

#include "../../util/FewModesFT.hpp"

auto problem_main() -> int
{
	// Problem parameters
	constexpr int num_modes = 10;
	constexpr amrex::Real k_peak = 2.0;
	constexpr amrex::Real sol_weight = 0.5; // 50% solenoidal
	constexpr amrex::Real t_corr = 1.0;
	constexpr uint32_t rseed = 12345;

	// Grid parameters
	constexpr int n_cell = 64;
	constexpr int max_grid_size = 32;
	constexpr amrex::Real prob_lo = 0.0;
	constexpr amrex::Real prob_hi = 1.0;

	// Set up domain
	amrex::IntVect const domain_lo(0, 0, 0);
	amrex::IntVect const domain_hi(n_cell - 1, n_cell - 1, n_cell - 1);
	amrex::Box const domain(domain_lo, domain_hi);

	amrex::RealBox const real_box({prob_lo, prob_lo, prob_lo}, {prob_hi, prob_hi, prob_hi});

	amrex::Array<int, AMREX_SPACEDIM> is_periodic = {1, 1, 1};

	amrex::Geometry const geom(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

	// Create BoxArray and DistributionMapping
	amrex::BoxArray ba(domain);
	ba.maxSize(max_grid_size);
	amrex::DistributionMapping const dm(ba);

	// Create MultiFab for the vector field (3 components)
	amrex::MultiFab mf(ba, dm, 3, 0);

	// Generate random wave vectors
	auto k_vec = quokka::util::MakeRandomModes(num_modes, k_peak, rseed);

	// Create FewModesFT object
	quokka::util::FewModesFT few_modes_ft("test", num_modes, k_vec, k_peak, sol_weight, t_corr, rseed, ba, dm);

	// Set up phases
	few_modes_ft.SetPhases(geom);

	// Generate the random field
	constexpr amrex::Real dt = 0.1;
	few_modes_ft.Generate(mf, dt);

	// Compute some statistics
	std::array<amrex::Real, 3> mean_field = {0.0, 0.0, 0.0};
	std::array<amrex::Real, 3> rms_field = {0.0, 0.0, 0.0};
	const amrex::Real total_cells = static_cast<amrex::Real>(geom.Domain().numPts());

	for (int n = 0; n < 3; ++n) {
		mean_field[n] = mf.sum(n, 1) / total_cells;
		rms_field[n] = mf.norm2(n) / std::sqrt(total_cells);
	}

	amrex::Print() << "FewModesFT Test Results:\n";
	amrex::Print() << "Number of modes: " << num_modes << "\n";
	amrex::Print() << "Peak wavenumber: " << k_peak << "\n";
	amrex::Print() << "Solenoidal weight: " << sol_weight << "\n";
	amrex::Print() << "Field statistics:\n";
	for (int n = 0; n < 3; ++n) {
		amrex::Print() << "  Component " << n << ": mean = " << mean_field[n] << ", RMS = " << rms_field[n] << "\n";
	}

	// Write plotfile
	amrex::Vector<std::string> const varnames = {"vx", "vy", "vz"};
	amrex::WriteSingleLevelPlotfile("plt_few_modes_ft", mf, varnames, geom, 0.0, 0);

	amrex::Print() << "Test completed successfully. Output written to plt_few_modes_ft.\n";

	return 0;
}
