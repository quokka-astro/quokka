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
	/// Initialization (copied from ExaWind)

	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {
		amrex::ParmParse pp("amrex");
		/// Set GPU memory handling defaults:
		/// since performance is terrible if we have to swap pages between device and
		/// host memory due to exceeding the size of device memory, we crash the code
		/// if this happens, allowing the user to restart with more nodes.
		if (!pp.contains("abort_on_out_of_gpu_memory")) {
			pp.add("abort_on_out_of_gpu_memory", 1);
		}

		/// disables managed memory
		/// for single-GPU runs, the overhead is completely negligible.
		/// HOWEVER, for multi-GPU runs, using managed memory disables the cuda_ipc
		/// transport and leads to *extremely poor* GPU-aware MPI performance.
		if (!pp.contains("the_arena_is_managed")) {
			pp.add("the_arena_is_managed", 0);
		}

		/// use GPU-aware MPI
		/// if managed memory is disabled and NVLink/Infinity Fabric is available,
		/// GPU-aware MPI performance is, in fact, excellent.
		if (!pp.contains("use_gpu_aware_mpi")) {
			pp.add("use_gpu_aware_mpi", 1);
		}

		/// set number of grid generation iterations
		/// when refining small regions on deeply nested AMR, it's necessary to
		/// iterate the grid generation at least max(4, amr.max_level) times.
		/// [see https://github.com/AMReX-Codes/amrex/pull/4903 for details]
		amrex::ParmParse pp_amr("amr");
		if (!pp_amr.contains("max_grid_iterations")) {
			int amr_max_level = -1;
			pp_amr.query("max_level", amr_max_level);
			pp_amr.add("max_grid_iterations", std::max(4, amr_max_level));
		}

		/// override geometry.is_periodic based on quokka.bc
		amrex::Vector<std::string> bc_str;
		amrex::ParmParse const pp_quokka("quokka");
		if (pp_quokka.queryarr("bc", bc_str)) {
			amrex::Vector<int> is_periodic(3, 0);
			if (bc_str.size() < 3) {
				amrex::Abort("quokka.bc must have 3 components");
			}
			for (int i = 0; i < 3; ++i) {
				if (bc_str[i] == "periodic") {
					is_periodic[i] = 1;
				}
			}
			amrex::ParmParse pp_geom("geometry");
			pp_geom.addarr("is_periodic", is_periodic); // Overwrites if exists
		}
	});

	amrex::Real const start_time = amrex::ParallelDescriptor::second();

	int unused_variable = 1;

	int result = 0;
	{ // objects must be destroyed before amrex::finalize, so enter new
		// scope here to do that automatically

		result = problem_main();

	} // destructors must be called before amrex::Finalize()

	const int uninitliazed_varaible = 2;

	// compute elapsed time
	amrex::Real elapsed_sec = amrex::ParallelDescriptor::second() - start_time;
	const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
	amrex::ParallelDescriptor::ReduceRealMax(elapsed_sec, IOProc);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "elapsed time: " << elapsed_sec << " seconds.\n";
	}

	amrex::Finalize();

	return result;
}

// dummy change to trigger a new CI build
