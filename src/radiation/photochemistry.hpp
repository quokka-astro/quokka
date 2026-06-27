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
auto computePhotoChemistry(amrex::MultiFab &mf, std::array<amrex::MultiFab const *, AMREX_SPACEDIM> const &fc_mfs, const Real dt, const int stage,
			   const Real max_density_allowed, const Real min_density_allowed) -> bool
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

	// Gate the O(v/c) radiation-pressure work term on beta_order==1 (the same
	// compile-time switch used in the regular RHD source terms, see
	// radiation_system.hpp:41 and source_terms_multi_group.hpp:455-577) and on
	// hydrodynamics being enabled. The work term activates only for problems that
	// have requested O(v/c) radiation coupling; problems with beta_order==0
	// (e.g. DTypeFront, OneZonePhotoionization) are unaffected.
	constexpr bool do_vc_work = (RadSystem_Traits<problem_t>::beta_order == 1) && Physics_Traits<problem_t>::is_hydro_enabled;

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

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			cons_fc[0] = fc_mfs[0]->const_array(iter);
#if (AMREX_SPACEDIM >= 2)
			cons_fc[1] = fc_mfs[1]->const_array(iter);
#endif
#if (AMREX_SPACEDIM == 3)
			cons_fc[2] = fc_mfs[2]->const_array(iter);
#endif
		}

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
			const Real Emag = ComputeCellCenteredMagneticEnergy<problem_t>(i, j, k, cons_fc);

			const Real Eint = ::quokka::EOS<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener, Emag);

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
				photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] = 1.0_rt;
			}

			// Cache the pre-burn chem-band radiation flux (energy units) so the O(v/c) work
			// term can use the VODE-attenuated flux difference -(F_after - F_before) to
			// deposit the absorbed photon momentum to the gas.
			amrex::GpuArray<amrex::GpuArray<Real, 3>, NumChemBands> frad_before{};
			if constexpr (do_vc_work) {
				for (int nn = 0; nn < NumChemBands; ++nn) {
					frad_before[nn][0] = state(i, j, k, firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn);
					frad_before[nn][1] = state(i, j, k, firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn);
					frad_before[nn][2] = state(i, j, k, firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn);
				}
			}
			photochemstate.rho = rho;
			photochemstate.e = Eint / rho;

			// call the EOS to set the temperature
			eos(eos_input_re, photochemstate);

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

			// get the updated specific eint
			eos(eos_input_re, photochemstate);

			for (int nn = 0; nn < NumSpec; ++nn) {
				state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) = photochemstate.xn[nn] * spmasses[nn];
			}
			for (int nn = 0; nn < NumChemBands; ++nn) {
				state(i, j, k, firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    photochemstate.rn[0 + MicrophysicsNumRadVarsPerGroup * nn] * chemBandQuanta[nn];
				state(i, j, k, firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] *
				    state(i, j, k, firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn);
				state(i, j, k, firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] *
				    state(i, j, k, firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn);
				state(i, j, k, firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] *
				    state(i, j, k, firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn);
			}
			// Quokka uses rho*eint
			const Real dEint = (photochemstate.e * photochemstate.rho) - Eint;
			state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index) += dEint * energy_update_factor;
			state(i, j, k, RadSystem<problem_t>::gasEnergy_index) += dEint * energy_update_factor;

			// O(v/c) radiation-pressure work term: deposit the photon momentum absorbed
			// during the burn to the gas. The matching RHD source-term path removes the
			// kinetic-energy gain from internal energy so that the gas total energy remains
			// consistent with the updated momentum.
			if constexpr (do_vc_work) {
				const Real c_light_local = C::c_light;
				const Real inv_c2 = 1.0_rt / (c_light_local * c_light_local);

				// dP = -(F_after - F_before) / c^2, summed over chem bands. The absorbed
				// photon momentum is tied to the physical photon flux F = c E; the reduced
				// speed of light only changes the absorption rate used by the chemistry
				// solve.
				Real dPx = 0.0_rt;
				Real dPy = 0.0_rt;
				Real dPz = 0.0_rt;
				for (int nn = 0; nn < NumChemBands; ++nn) {
					const Real Fx_after = state(i, j, k, firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn);
					const Real Fy_after = state(i, j, k, firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn);
					const Real Fz_after = state(i, j, k, firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn);
					dPx += -(Fx_after - frad_before[nn][0]) * inv_c2;
					dPy += -(Fy_after - frad_before[nn][1]) * inv_c2;
					dPz += -(Fz_after - frad_before[nn][2]) * inv_c2;
				}

				const Real xmom_new = xmom + dPx;
				const Real ymom_new = ymom + dPy;
				const Real zmom_new = zmom + dPz;
				state(i, j, k, RadSystem<problem_t>::x1GasMomentum_index) = xmom_new;
				state(i, j, k, RadSystem<problem_t>::x2GasMomentum_index) = ymom_new;
				state(i, j, k, RadSystem<problem_t>::x3GasMomentum_index) = zmom_new;

				// Ekin per volume = sum_i (mom_i^2) / (2 * rho). Decrease internal energy
				// by dEkin so the kinetic gain is balanced, then recompute total gas energy
				// from the updated conserved fields.
				const Real Ekin_before = (xmom * xmom + ymom * ymom + zmom * zmom) / (2.0_rt * rho);
				const Real Ekin_after = (xmom_new * xmom_new + ymom_new * ymom_new + zmom_new * zmom_new) / (2.0_rt * rho);
				const Real dEkin = Ekin_after - Ekin_before;
				state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index) -= dEkin * energy_update_factor;
				state(i, j, k, RadSystem<problem_t>::gasEnergy_index) = ::quokka::EOS<problem_t>::ComputeEgasFromEint(
				    rho, xmom_new, ymom_new, zmom_new, state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index), Emag);
			}
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
