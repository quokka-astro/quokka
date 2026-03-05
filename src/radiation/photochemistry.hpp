#ifndef PHOTOCHEMISTRY_HPP_ // NOLINT
#define PHOTOCHEMISTRY_HPP_

#include <array>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"

#include "radiation/radiation_system.hpp"

#ifdef PHOTOCHEMISTRY
#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "physics_numVars.hpp"
#include "network_properties.H"

namespace quokka::photochemistry
{
AMREX_GPU_DEVICE void photochem_burner(burn_t &photochemstate, Real dt);

template <typename problem_t> auto computePhotoChemistry(amrex::MultiFab &mf, const Real dt, const int stage, const Real max_density_allowed, const Real min_density_allowed) -> bool
{
	AMREX_ASSERT(stage==1 || stage==2);
    // Start off by assuming a successful burn.
    int photochem_burn_success = 1;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	auto *p_num_failed = d_num_failed.data();

    int num_failed = 0;

	auto dt_stage = dt / stage;
	auto energy_update_factor = stage / 1.0_rt;

	auto ChemActiveRadFreqBounds_ = RadSystem_Traits<problem_t>::ChemActiveRadFreqBounds;

    const BL_PROFILE("PhotoChemistry::computePhotoChemistry()");
    for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
        const amrex::Box &indexRange = iter.validbox();
        auto const &state = mf.array(iter);

        amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const Real rho = state(i, j, k, RadSystem<problem_t>::gasDensity_index);
            const Real xmom = state(i, j, k, RadSystem<problem_t>::x1GasMomentum_index);
            const Real ymom = state(i, j, k, RadSystem<problem_t>::x2GasMomentum_index);
            const Real zmom = state(i, j, k, RadSystem<problem_t>::x3GasMomentum_index);
            const Real Ener = state(i, j, k, RadSystem<problem_t>::gasEnergy_index);
            const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener);

			Real quanta_energy = 0.0_rt;
			burn_t photochemstate;
            photochemstate.success = true;
            int burn_failed = 0;
            for (int nn = 0; nn < NumSpec; ++nn) {
                photochemstate.xn[nn] = state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) / spmasses[nn];
            }
			for (int nn = 0; nn < NumChemActiveRadGroups; ++nn) {
				quanta_energy = RadSystem<problem_t>::GetRadiationGroupQuanta(ChemActiveRadFreqBounds_[nn], ChemActiveRadFreqBounds_[nn + 1]);
				photochemstate.rn[0 + MicrophysicsNumRadVarsPerGroup * nn] = state(i, j, k, RadSystem<problem_t>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * nn) / quanta_energy;
				// // TODO(james471): Add check for isotropy
				photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] = 1.0_rt;
			}
			photochemstate.rho = rho;
			photochemstate.e = Eint / rho;

            // dont do photochemistry in cells with densities below the minimum density specified
			// if (rho < min_density_allowed) {
			// 	return;
			// }
            // stop the test if we have reached very high densities
			// if (rho > max_density_allowed) {
			// 	amrex::Abort("Density exceeded max_density_allowed!");
			// }

			// call the EOS to set the temperature
			eos(eos_input_re, photochemstate);

			// do the actual integration
			// do it in .cpp so that it is not built at compile time for all tests
			// which would otherwise slow down compilation due to the large RHS file
			photochem_burner(photochemstate, dt_stage);

			if (std::isnan(photochemstate.xn[0]) || std::isnan(photochemstate.rho)) {
				amrex::Abort("Burner returned NAN");
			}

			if (!photochemstate.success) {
				burn_failed = 1;
			}

			if (burn_failed) {
				amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
			}

			// get the updated specific eint
			eos(eos_input_re, photochemstate);
			
			for (int nn = 0; nn < NumSpec; ++nn) {
				state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) = photochemstate.xn[nn] * spmasses[nn]; 
			}
			for (int nn = 0; nn < NumChemActiveRadGroups; ++nn) {
				quanta_energy = RadSystem<problem_t>::GetRadiationGroupQuanta(ChemActiveRadFreqBounds_[nn], ChemActiveRadFreqBounds_[nn + 1]);
				state(i, j, k, RadSystem<problem_t>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * nn) = 
					photochemstate.rn[0 + MicrophysicsNumRadVarsPerGroup * nn] * quanta_energy;
				state(i, j, k, RadSystem<problem_t>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * nn) = 
					photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] * state(i, j, k, RadSystem<problem_t>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * nn);
				state(i, j, k, RadSystem<problem_t>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * nn) = 
					photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] * state(i, j, k, RadSystem<problem_t>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * nn);
				state(i, j, k, RadSystem<problem_t>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * nn) = 
					photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] * state(i, j, k, RadSystem<problem_t>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * nn);
			}
			// Quokka uses rho*eint
			const Real dEint = (photochemstate.e * photochemstate.rho) - Eint;
			state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index) += dEint * energy_update_factor;
			state(i, j, k, RadSystem<problem_t>::gasEnergy_index) += dEint * energy_update_factor;
        });

#if defined(AMREX_USE_HIP)
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
    }

	num_failed = *(d_num_failed.copyToHost());

	photochem_burn_success = !num_failed;
	amrex::ParallelDescriptor::ReduceIntMin(photochem_burn_success);

	if (!photochem_burn_success) {
		amrex::Abort("Burn failed in VODE. Aborting.");
	}

    return photochem_burn_success;
}

} // namespace quokka::photochemistry
#endif

#endif