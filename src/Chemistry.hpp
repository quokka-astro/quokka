#ifndef CHEMISTRY_HPP_ // NOLINT
#define CHEMISTRY_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CloudyCooling.hpp
/// \brief Defines methods for interpolating cooling rates from Cloudy tables.
///

#include <limits>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"

#include "burn_type.H"
#include "burner.H"
#include "eos.H"
#include "extern_parameters.H"

namespace quokka::chemistry
{
template <typename problem_t> void computeChemistry(amrex::MultiFab &mf, const Real dt_in)
{
	BL_PROFILE("computeChemistry()")

	const Real grav_constant = 6.674e-8;
	const Real dt = dt_in;


	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab nsubstepsMF(ba, dmap, 1, 0);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real Eint = state(i, j, k, HydroSystem<problem_t>::internalEnergy_index);

			Real chem[NumSpec] = {-1.0};
			Real inmfracs[NumSpec] = {-1.0};

			for (int nn = 0; nn < NumSpec; ++nn) {
				chem[nn] = state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn);
			}

			// do chemistry using microphysics
			int nsteps = 1000;
			// replace below with call to microphysics

			burn_t chemstate;

			for (int n = 0; n < nsteps; n++) {

				for (int nn = 0; nn < NumSpec; ++nn) {
					inmfracs[nn] = chem[nn] * rho / spmasses[nn];
					chemstate.xn[nn] = inmfracs[nn];
				}

				// stop the test if dt is very small
				if (dt < 10) {
					break;
				}

				// stop the test if we have reached very high densities
				if (rho > 3e-6) {
					break;
				}

				// input the scaled density in burn state
				chemstate.rho = rho;
				chemstate.e = Eint;

				// call the EOS to set initial internal energy e
				eos(eos_input_re, chemstate);

				// do the actual integration
				burner(chemstate, dt);

				// ensure positivity and normalize
				Real inmfracs[NumSpec] = {-1.0};
				Real insum = 0.0_rt;
				for (int nn = 0; nn < NumSpec; ++nn) {
					chemstate.xn[nn] = amrex::max(chemstate.xn[nn], small_x);
					inmfracs[nn] = spmasses[nn] * chemstate.xn[nn] / chemstate.rho;
					insum += inmfracs[nn];
				}

				for (int nn = 0; nn < NumSpec; ++nn) {
					inmfracs[nn] /= insum;
					// update the number densities with conserved mass fractions
					chemstate.xn[nn] = inmfracs[nn] * chemstate.rho / spmasses[nn];
				}

				// update the number density of electrons due to charge conservation
				chemstate.xn[0] = -chemstate.xn[3] - chemstate.xn[7] + chemstate.xn[1] + chemstate.xn[12] + chemstate.xn[6] + chemstate.xn[4] +
						  chemstate.xn[9] + 2.0 * chemstate.xn[11];

				// reconserve mass fractions post charge conservation
				insum = 0;
				for (int nn = 0; nn < NumSpec; ++nn) {
					chemstate.xn[nn] = amrex::max(chemstate.xn[nn], small_x);
					inmfracs[nn] = spmasses[nn] * chemstate.xn[nn] / chemstate.rho;
					insum += inmfracs[nn];
				}

				for (int nn = 0; nn < NumSpec; ++nn) {
					inmfracs[nn] /= insum;
					// update the number densities with conserved mass fractions
					chemstate.xn[nn] = inmfracs[nn] * chemstate.rho / spmasses[nn];
				}

				// get the updated T
				eos(eos_input_re, chemstate);

				state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = chemstate.e;

				for (int nn = 0; nn < NumSpec; ++n) {
					state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) = inmfracs[nn];
				}
			}

			int nmin = nsubstepsMF.min(0);
			int nmax = nsubstepsMF.max(0);
			Real navg = static_cast<Real>(nsubstepsMF.sum(0)) / static_cast<Real>(nsubstepsMF.boxArray().numPts());
			amrex::Print() << fmt::format("\tChemistry substeps (per cell): min {}, avg {}, max {}\n", nmin, navg, nmax);
		});
	}
} // namespace quokka::chemistry

} // namespace quokka::chemistry
#endif // CHEMISTRY_HPP_
