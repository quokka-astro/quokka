#ifndef PHOTOCHEMISTRY_HPP_ // NOLINT
#define PHOTOCHEMISTRY_HPP_

#include <array>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"

#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"

#ifdef PHOTOCHEMISTRY
#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network_properties.H"
#include "physics_numVars.hpp"

namespace quokka::photochemistry
{
AMREX_GPU_DEVICE void photochem_burner(burn_t &photochemstate, Real dt);

template <typename problem_t>
auto computePhotoChemistry(amrex::MultiFab &mf, const Real dt, const int stage, const Real max_density_allowed, const Real min_density_allowed) -> bool
{
	AMREX_ASSERT(stage == 1 || stage == 2);
	// Start off by assuming a successful burn.
	int photochem_burn_success = 1;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	auto *p_num_failed = d_num_failed.data();

	int num_failed = 0;

	auto dt_stage = dt / static_cast<Real>(stage);
	auto energy_update_factor = static_cast<Real>(stage);

	const int firstChemIndex = RadSystem<problem_t>::radEnergy_index +
				   RadSystem<problem_t>::numRadVars_ * (RadSystem<problem_t>::nGroups_ - RadSystem_NChemBands<problem_t>::value);
	const int firstChemFxIndex = firstChemIndex + 1;
	const int firstChemFyIndex = firstChemFxIndex + 1;
	const int firstChemFzIndex = firstChemFyIndex + 1;

	amrex::GpuArray<Real, NumChemBands> chemBandQuanta{};
	amrex::GpuArray<Real, NumChemBands> invChemBandQuanta{};
	for (int nn = 0; nn < NumChemBands; ++nn) {
		chemBandQuanta[nn] = RadSystem<problem_t>::GetChemBandQuanta(nn);
		invChemBandQuanta[nn] = 1.0_rt / chemBandQuanta[nn];
	}

	const BL_PROFILE("PhotoChemistry::computePhotoChemistry()");
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, RadSystem<problem_t>::gasDensity_index);
			// dont do photochemistry in cells with densities below the minimum density specified
			if (rho < min_density_allowed) {
				return;
			}
			// stop the test if we have reached very high densities
			if (rho > max_density_allowed) {
				amrex::Abort("Density exceeded max_density_allowed!");
			}

			const Real xmom = state(i, j, k, RadSystem<problem_t>::x1GasMomentum_index);
			const Real ymom = state(i, j, k, RadSystem<problem_t>::x2GasMomentum_index);
			const Real zmom = state(i, j, k, RadSystem<problem_t>::x3GasMomentum_index);
			const Real Ener = state(i, j, k, RadSystem<problem_t>::gasEnergy_index);
			const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener);

			burn_t photochemstate;
			photochemstate.success = true;
			int burn_failed = 0;
			photochemstate.c_hat = RadSystem_Traits<problem_t>::c_hat_over_c * C::c_light;
			for (int nn = 0; nn < NumSpec; ++nn) {
				photochemstate.xn[nn] = state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) / spmasses[nn];
			}
			for (int nn = 0; nn < NumChemBands; ++nn) {
				photochemstate.rn[0 + MicrophysicsNumRadVarsPerGroup * nn] =
				    state(i, j, k, firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn) * invChemBandQuanta[nn];
			}
			photochemstate.rho = rho;
			photochemstate.e = Eint / rho;

			// call the EOS to set the temperature
			eos(eos_input_re, photochemstate);

			// Save initial photon density for algebraic flux attenuation
			const Real n_gamma_initial = photochemstate.rn[0];

			// do the actual integration
			// do it in .cpp so that it is not built at compile time for all tests
			// which would otherwise slow down compilation due to the large RHS file
			photochem_burner(photochemstate, dt_stage);

			if (std::isnan(photochemstate.xn[0]) || std::isnan(photochemstate.rho) || std::isnan(photochemstate.rn[0])) {
				amrex::Abort("Burner returned NAN");
			}

			if (!photochemstate.success) {
				burn_failed = 1;
			}

			if (burn_failed) {
				amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
			}

			// Ensure positivity
			for (double &nn : photochemstate.xn) {
				nn = amrex::max(nn, small_x);
			}
			for (int nn = 0; nn < NumChemBands; nn += 1) {
				// TODO (james471): Ensure that flux doesn't deviate from the corresponding energy density.
				photochemstate.rn[static_cast<std::size_t>(nn) * MicrophysicsNumRadVarsPerGroup] =
				    amrex::max(photochemstate.rn[static_cast<std::size_t>(nn) * MicrophysicsNumRadVarsPerGroup], small_x);
			}

			// Compute algebraic flux attenuation. The flux ODE dy(6)/dt =
			// -chat*sigma*n_HI*y(6) has the same attenuation factor as
			// the photon density equation dy(5)/dt = -chat*sigma*n_HI*y(5),
			// so flux_attenuation = n_gamma_final / n_gamma_initial.
			const Real flux_attenuation = (n_gamma_initial > 0.0_rt) ? (photochemstate.rn[0] / n_gamma_initial) : 1.0_rt;

			// get the updated specific eint
			eos(eos_input_re, photochemstate);

			for (int nn = 0; nn < NumSpec; ++nn) {
				state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) = photochemstate.xn[nn] * spmasses[nn];
			}
			for (int nn = 0; nn < NumChemBands; ++nn) {
				state(i, j, k, firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    photochemstate.rn[0 + MicrophysicsNumRadVarsPerGroup * nn] * chemBandQuanta[nn];
				state(i, j, k, firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    flux_attenuation * state(i, j, k, firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn);
				state(i, j, k, firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    flux_attenuation * state(i, j, k, firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn);
				state(i, j, k, firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    flux_attenuation * state(i, j, k, firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn);
			}
			// Quokka uses rho*eint
			const Real dEint = (photochemstate.e * photochemstate.rho) - Eint;
			state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index) += dEint * energy_update_factor;
			state(i, j, k, RadSystem<problem_t>::gasEnergy_index) += dEint * energy_update_factor;
		});

#ifdef AMREX_USE_HIP
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
	}

	num_failed = *(d_num_failed.copyToHost());

	photochem_burn_success = num_failed == 0;
	amrex::ParallelDescriptor::ReduceIntMin(photochem_burn_success);

	if (!photochem_burn_success) {
		amrex::Abort("Burn failed in VODE. Aborting.");
	}

	return photochem_burn_success;
}

} // namespace quokka::photochemistry
#endif

#endif
