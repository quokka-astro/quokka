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
#include <omp.h>

auto main(int argc, char **argv) -> int {
  // Ensure the OpenMP runtime starts 2 threads (used for inner-outer updates)
  //  NOTE: This *must* be called before AMReX is initialized
  const int nthreads = 2;
  omp_set_num_threads(nthreads);

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

    // Set GPU memory handling defaults:
    // since performance is terrible if we have to swap pages between device and
    // host memory due to exceeding the size of device memory, we crash the code
    // if this happens, allowing the user to restart with more nodes.
    if (!pp.contains("abort_on_out_of_gpu_memory")) {
      pp.add("abort_on_out_of_gpu_memory", 1);
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
    amrex::Print() << "elapsed time: " << elapsed_sec << " seconds.\n";
  }

  amrex::Finalize();

  return result;
}
