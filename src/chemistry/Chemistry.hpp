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

// The definition of computeChemistry, declared in QuokkaSimulation.hpp. A problem with chemistry includes this header;
// problems without it do not, because the definition uses spmasses, which only the chemistry EOSs declare.

#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"

#include "actual_eos_data.H"
#include "eos.H"
#include "extern_parameters.H"
#include "hydro/EOSMicrophysics.hpp"
#include "microphysics/burn_state.hpp"
#include "microphysics/burner.hpp"

namespace quokka::chemistry
{

template <typename problem_t>
auto computeChemistry(amrex::MultiFab &mf, const Real dt, const Real max_density_allowed, const Real min_density_allowed,
		      const quokka::microphysics::IntegratorParams integrator_params) -> bool
{
	using Network = typename Microphysics_Traits<problem_t>::Network;

	// Until the EOS reads its species from the network (the EOS port), it sizes its composition by the species table on
	// the include path (network_properties.H), so the problem's network must match that table.
	static_assert(Network::NumSpec == NumSpec, "The problem's Network does not match the species table of the EOS it is built with.");

	// Start off by assuming a successful burn.
	int burn_success = 1;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	auto *p_num_failed = d_num_failed.data();

	int num_failed = 0;

	const BL_PROFILE("Chemistry::computeChemistry()");
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real xmom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const Real ymom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const Real zmom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const Real Ener = state(i, j, k, HydroSystem<problem_t>::energy_index);
			static_assert(!Physics_Traits<problem_t>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
			const Real Eint = ::quokka::EOS<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener, 0.0);

			std::array<Real, Network::NumSpec> chem = {-1.0};
			std::array<Real, Network::NumSpec> inmfracs = {-1.0};
			Real insum = 0.0_rt;

			for (int nn = 0; nn < Network::NumSpec; ++nn) {
				chem[nn] = state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) /
					   rho; // state has partial densities, so divide by rho to get mass fractions
			}

			// do chemistry using microphysics

			quokka::microphysics::BurnState<problem_t> chemstate;
			chemstate.success = true;
			int burn_failed = 0;

			for (int nn = 0; nn < Network::NumSpec; ++nn) {
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
			quokka::microphysics::Burn<problem_t>(chemstate, dt, integrator_params);

			if (std::isnan(chemstate.xn[0]) || std::isnan(chemstate.rho)) {
				amrex::Abort("Burner returned NAN");
			}

			if (!chemstate.success) {
				burn_failed = 1;
			}

			if (burn_failed) {
				amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
			}

			// ensure positivity and normalize
			for (int nn = 0; nn < Network::NumSpec; ++nn) {
				chemstate.xn[nn] = amrex::max(chemstate.xn[nn], Network::SmallX());
				inmfracs[nn] = spmasses[nn] * chemstate.xn[nn] / chemstate.rho;
				insum += inmfracs[nn];
			}

			for (int nn = 0; nn < Network::NumSpec; ++nn) {
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

			for (int nn = 0; nn < Network::NumSpec; ++nn) {
				state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) = inmfracs[nn] * rho; // scale by rho to return partial densities
			}
		});

#if defined(AMREX_USE_HIP)
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
	}

	num_failed = *(d_num_failed.copyToHost());

	burn_success = !num_failed;
	amrex::ParallelDescriptor::ReduceIntMin(burn_success);

	if (!burn_success) {
		// amrex::Abort("Burn failed in the integrator. Aborting.");
		amrex::Print() << "\t>> WARNING: Unsuccessful burn. Retrying hydro step."
			       << "\n";
	}

	return burn_success;
}

} // namespace quokka::chemistry

#endif // CHEMISTRY_HPP_
