#ifndef CHEMISTRY_HPP_ // NOLINT
#define CHEMISTRY_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Chemistry.hpp
/// \brief Defines methods for integrating primordial chemical network using Microphysics
///

#include <array>

#include "AMReX.H"
#include "AMReX_GpuQualifiers.H"

#include "hydro_system.hpp"
#include "radiation_system.hpp"

#ifdef PRIMORDIAL_CHEM
#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"

namespace quokka::chemistry
{

AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, Real dt);

template <typename problem_t> void computeChemistry(amrex::MultiFab &mf, const Real dt, const Real max_density_allowed, const Real min_density_allowed)
{

	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab burnstepsMF(ba, dmap, 1, 0);

	const BL_PROFILE("Chemistry::computeChemistry()");
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &burnsteps = burnstepsMF.array(iter);

		if (dt < 0) {
			amrex::Abort("Cannot do chemistry with dt < 0!");
		}

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real xmom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const Real ymom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const Real zmom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const Real Ener = state(i, j, k, HydroSystem<problem_t>::energy_index);
			const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener);

			std::array<Real, NumSpec> chem = {-1.0};
			std::array<Real, NumSpec> inmfracs = {-1.0};
			Real insum = 0.0_rt;

			for (int nn = 0; nn < NumSpec; ++nn) {
				chem[nn] = state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) /
					   rho; // state has partial densities, so divide by rho to get mass fractions
			}

			// do chemistry using microphysics

			burn_t chemstate;

			for (int nn = 0; nn < NumSpec; ++nn) {
				inmfracs[nn] = chem[nn] * rho / spmasses[nn];
				chemstate.xn[nn] = inmfracs[nn];
			}

			// dont do chemistry in cells with densities below the minimum density specified
			if (rho < min_density_allowed) {
				return;
			}

			// stop the test if we have reached very high densities
			if (rho > max_density_allowed) {
				amrex::Abort("Density exceeded max_density_allowed!");
			}

			// input density and eint in burn state
			// Microphysics needs specific eint
			chemstate.rho = rho;
			chemstate.e = Eint / rho;

			// call the EOS to set initial internal energy e
			eos(eos_input_re, chemstate);

			// do the actual integration
			// do it in .cpp so that it is not built at compile time for all tests
			// which would otherwise slow down compilation due to the large RHS file
			chemburner(chemstate, dt);
			burnsteps(i, j, k) = chemstate.success;

			if (std::isnan(chemstate.xn[0]) || std::isnan(chemstate.rho)) {
				amrex::Abort("Burner returned NAN");
			}

			// ensure positivity and normalize
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
			// TODO(psharda): generalize this to other chem networks
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

			// get the updated specific eint
			eos(eos_input_rt, chemstate);

			// get dEint
			// Quokka uses rho*eint
			const Real dEint = (chemstate.e * chemstate.rho) - Eint;
			state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += dEint;
			state(i, j, k, HydroSystem<problem_t>::energy_index) += dEint;

			for (int nn = 0; nn < NumSpec; ++nn) {
				state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) = inmfracs[nn] * rho; // scale by rho to return partial densities
			}
		});
	}

	int burnmin = burnstepsMF.min(0);
	if (burnmin == 0) {
		amrex::Abort("VODE integration was unsuccessful!");
	}
}

} // namespace quokka::chemistry
#endif
#endif // CHEMISTRY_HPP_
