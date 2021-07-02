//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file main.cpp
/// \brief The main() function for simulations.
///

#include "AMReX.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "main.hpp"

auto main(int argc, char **argv) -> int
{
	// Initialization (copied from ExaWind)

	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {
		amrex::ParmParse pp("amrex");
		// Set the defaults so that we throw an exception instead of attempting
		// to generate backtrace files. However, if the user has explicitly set
		// these options in their input files respect those settings.
		if (!pp.contains("throw_exception")) {
			pp.add("throw_exception", 1);
		}
		if (!pp.contains("signal_handling")) {
			pp.add("signal_handling", 0);
		}
	});

	amrex::Real start_time = amrex::ParallelDescriptor::second();

	int result = 0;
	{ // objects must be destroyed before amrex::finalize, so enter new
	  // scope here to do that automatically

		result = problem_main();

	} // destructors must be called before amrex::Finalize()

	// compute elapsed time
	amrex::Real elapsed_sec = amrex::ParallelDescriptor::second() - start_time;
	const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
	amrex::ParallelDescriptor::ReduceRealMax(elapsed_sec, IOProc);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		//const double zone_cycles = cycleCount_ * (nx_[0] * nx_[1] * nx_[2]);
		//const double microseconds_per_update = 1.0e6 * elapsed_sec / zone_cycles;
		//const double megaupdates_per_second = 1.0 / microseconds_per_update;
		//amrex::Print() << "Performance figure-of-merit: " << microseconds_per_update
		//	       << " μs/zone-update [" << megaupdates_per_second << " Mupdates/s]\n";
		amrex::Print() << "elapsed time: " << elapsed_sec << " seconds.\n";
	}

	amrex::Finalize();

	return result;
}