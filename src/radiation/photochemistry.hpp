#ifndef PHOTOCHEMISTRY_HPP_ // NOLINT
#define PHOTOCHEMISTRY_HPP_

#include <array>
#include <format>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"

#include "chemistry/ChemistryNetwork.hpp"
#include "chemistry/RuntimeParameters.hpp"
#include "networks/photoionization/PhotoionizationNetwork.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"

#ifdef PHOTOCHEMISTRY
#include "physics_numVars.hpp"

namespace quokka::photochemistry
{
constexpr int NumSpec = quokka::chemistry::PhotoionizationNetwork::species_count;
constexpr int NumChemBands = quokka::chemistry::PhotoionizationNetwork::chemistry_band_count;
// Only relevant when the O(v/c) work term is active (see do_vc_work below). If true, the work-term kinetic
// energy the gas gains from the absorbed-photon momentum kick is debited from the chem-band photon field
// (photon energy density reduced by (c / chat) * dEkin, and each band's flux rescaled by the same factor to
// preserve the M1 closure |F| <= c E), strictly conserving the gas+radiation energy. If false (default), the
// photon field is left unchanged after the burn: the gas still gains the kinetic energy, but it is not removed
// from the radiation, so the ionizing photon number density is conserved at the cost of gas+radiation energy
// conservation. Conserving the ionizing photon budget is the more important property for correct
// ionization-front propagation, so it is the default.
static constexpr bool debit_work_term_from_radiation = false;

AMREX_GPU_DEVICE auto photochem_burner(quokka::chemistry::IntegratorState<quokka::chemistry::PhotoionizationNetwork::variable_count> &state, Real dt,
				       quokka::chemistry::PhotoionizationNetwork const &network, quokka::chemistry::IntegratorOptions const &options)
    -> quokka::chemistry::IntegratorDiagnostics;

template <typename problem_t>
auto computePhotoChemistry(amrex::MultiFab &mf, std::array<amrex::MultiFab const *, AMREX_SPACEDIM> const &fc_mfs, const Real dt, const int stage,
			   const Real max_density_allowed, const Real min_density_allowed, const Real minimum_temperature) -> bool
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

	// The O(v/c) radiation-pressure work term is gated on beta_order>=1 && is_hydro_enabled; the condition is
	// inlined inside the device lambda's if constexpr below to avoid NVCC first-capturing a local constexpr.

	amrex::GpuArray<Real, NumChemBands> chemBandQuanta{};
	amrex::GpuArray<Real, NumChemBands> invChemBandQuanta{};
	for (int nn = 0; nn < NumChemBands; ++nn) {
		chemBandQuanta[nn] = RadSystem<problem_t>::GetChemBandQuanta(nn);
		invChemBandQuanta[nn] = 1.0_rt / chemBandQuanta[nn];
	}
	const auto integratorOptions = quokka::chemistry::readIntegratorOptions(minimum_temperature);
	const quokka::chemistry::PhotoionizationNetwork chemistryNetwork{quokka::chemistry::readPhotoionizationParameters()};

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

			quokka::chemistry::IntegratorState<quokka::chemistry::PhotoionizationNetwork::variable_count> photochemstate{};
			int burn_failed = 0;
			photochemstate.reduced_speed_of_light = RadSystem_Traits<problem_t>::c_hat_over_c * C::c_light;
			for (int nn = 0; nn < NumSpec; ++nn) {
				photochemstate.values[nn] =
				    state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) / quokka::chemistry::PhotoionizationNetwork::species_masses[nn];
			}
			for (int nn = 0; nn < NumChemBands; ++nn) {
				photochemstate.values[quokka::chemistry::PhotoionizationNetwork::photon_number +
						      quokka::chemistry::PhotoionizationNetwork::radiation_variables_per_group * nn] =
				    state(i, j, k, firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn) * invChemBandQuanta[nn];
				photochemstate.values[quokka::chemistry::PhotoionizationNetwork::photon_flux_factor +
						      quokka::chemistry::PhotoionizationNetwork::radiation_variables_per_group * nn] = 1.0_rt;
			}
			photochemstate.density = rho;
			photochemstate.values[quokka::chemistry::PhotoionizationNetwork::energy] = Eint / rho;

			// do the actual integration
			// do it in .cpp so that it is not built at compile time for all tests
			// which would otherwise slow down compilation due to the large RHS file
			const auto integrationDiagnostics = photochem_burner(photochemstate, dt_stage, chemistryNetwork, integratorOptions);

			if (std::isnan(photochemstate.values[0]) || std::isnan(photochemstate.density) ||
			    std::isnan(photochemstate.values[quokka::chemistry::PhotoionizationNetwork::photon_number])) {
				amrex::Abort("Burner returned NAN");
			}

			if (!integrationDiagnostics.succeeded()) {
				burn_failed = static_cast<int>(integrationDiagnostics.status);
			}

			if (burn_failed) {
				if (amrex::Gpu::Atomic::CAS(p_num_failed, 0, burn_failed) == 0) {
					printf("photochemistry failure: status=%d dt=%g reached=%g suggested_dt=%g steps=%d rejected=%d rho=%g ne=%g nHI=%g "
					       "nHII=%g e=%g "
					       "photons=%g\n",
					       burn_failed, dt_stage, integrationDiagnostics.reached_time, integrationDiagnostics.suggested_timestep,
					       integrationDiagnostics.steps, integrationDiagnostics.rejected_steps, photochemstate.density,
					       photochemstate.values[quokka::chemistry::PhotoionizationNetwork::electron],
					       photochemstate.values[quokka::chemistry::PhotoionizationNetwork::neutral_hydrogen],
					       photochemstate.values[quokka::chemistry::PhotoionizationNetwork::ionized_hydrogen],
					       photochemstate.values[quokka::chemistry::PhotoionizationNetwork::energy],
					       photochemstate.values[quokka::chemistry::PhotoionizationNetwork::photon_number]);
				}
			}

			// Ensure positivity
			for (int nn = 0; nn < NumSpec; ++nn) {
				photochemstate.values[nn] = amrex::max(photochemstate.values[nn], integratorOptions.small_state);
			}
			for (int nn = 0; nn < NumChemBands; nn += 1) {
				// TODO (james471): Ensure that flux doesn't deviate from the corresponding energy density.
				const int radiationIndex = quokka::chemistry::PhotoionizationNetwork::photon_number +
							   nn * quokka::chemistry::PhotoionizationNetwork::radiation_variables_per_group;
				photochemstate.values[radiationIndex] = amrex::max(photochemstate.values[radiationIndex], integratorOptions.small_state);
			}

			for (int nn = 0; nn < NumSpec; ++nn) {
				state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) =
				    photochemstate.values[nn] * quokka::chemistry::PhotoionizationNetwork::species_masses[nn];
			}
			// Update the chem-band photon number density, attenuate the flux by the burn's rn[1] factor, and
			// accumulate the momentum the gas absorbs from the O(v/c) work term: dP = -(F_after - F_before) / c^2,
			// summed over chem bands. The absorbed photon momentum is tied to the physical photon flux F = c E; the
			// reduced speed of light only changes the absorption rate used by the chemistry solve. dMom is computed
			// here unconditionally (cheap) and only applied when do_vc_work is true -- computing it outside the
			// `if constexpr (do_vc_work)` block avoids NVCC's "cannot first-capture in constexpr-if" restriction.
			const Real inv_c2 = 1.0_rt / (C::c_light * C::c_light);
			Real dMomX = 0.0_rt;
			Real dMomY = 0.0_rt;
			Real dMomZ = 0.0_rt;
			// per-band momentum-kick magnitudes, used only to split the optional work-term energy debit across bands
			[[maybe_unused]] amrex::GpuArray<Real, NumChemBands> dMomMag{};
			[[maybe_unused]] Real dMomMagSum = 0.0_rt;
			for (int nn = 0; nn < NumChemBands; ++nn) {
				const int fxIdx = firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn;
				const int fyIdx = firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn;
				const int fzIdx = firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn;
				const Real flux_factor = photochemstate.values[quokka::chemistry::PhotoionizationNetwork::photon_flux_factor +
									       quokka::chemistry::PhotoionizationNetwork::radiation_variables_per_group * nn];
				state(i, j, k, firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn) =
				    photochemstate.values[quokka::chemistry::PhotoionizationNetwork::photon_number +
							  quokka::chemistry::PhotoionizationNetwork::radiation_variables_per_group * nn] *
				    chemBandQuanta[nn];
				const Real FxOld = state(i, j, k, fxIdx);
				const Real FyOld = state(i, j, k, fyIdx);
				const Real FzOld = state(i, j, k, fzIdx);
				// dp = (F_before - F_after) / c^2 = (1 - flux_factor) * F_before / c^2
				const Real dpx = (1.0_rt - flux_factor) * FxOld * inv_c2;
				const Real dpy = (1.0_rt - flux_factor) * FyOld * inv_c2;
				const Real dpz = (1.0_rt - flux_factor) * FzOld * inv_c2;
				dMomX += dpx;
				dMomY += dpy;
				dMomZ += dpz;
				if constexpr (debit_work_term_from_radiation) {
					dMomMag[nn] = std::sqrt(dpx * dpx + dpy * dpy + dpz * dpz);
					dMomMagSum += dMomMag[nn];
				}
				state(i, j, k, fxIdx) = flux_factor * FxOld;
				state(i, j, k, fyIdx) = flux_factor * FyOld;
				state(i, j, k, fzIdx) = flux_factor * FzOld;
			}

			// Quokka uses rho*eint
			const Real dEint = (photochemstate.values[quokka::chemistry::PhotoionizationNetwork::energy] * photochemstate.density) - Eint;
			state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index) += dEint * energy_update_factor;
			state(i, j, k, RadSystem<problem_t>::gasEnergy_index) += dEint * energy_update_factor;

			// O(v/c) radiation-pressure work term: apply the absorbed photon momentum (dMom, computed above) to the
			// gas. This changes the kinetic energy of the updated momentum only; the auxiliary internal energy is
			// unaffected. Gated on beta_order>=1 && hydro, so beta_order==0 problems are untouched.
			if constexpr ((RadSystem_Traits<problem_t>::beta_order >= 1) && Physics_Traits<problem_t>::is_hydro_enabled) {
				const Real xmom_new = xmom + dMomX;
				const Real ymom_new = ymom + dMomY;
				const Real zmom_new = zmom + dMomZ;
				state(i, j, k, RadSystem<problem_t>::x1GasMomentum_index) = xmom_new;
				state(i, j, k, RadSystem<problem_t>::x2GasMomentum_index) = ymom_new;
				state(i, j, k, RadSystem<problem_t>::x3GasMomentum_index) = zmom_new;

				// Recompute total gas energy from the (unchanged) internal energy and the updated momentum so the
				// absorbed momentum appears only as kinetic energy.
				state(i, j, k, RadSystem<problem_t>::gasEnergy_index) = ::quokka::EOS<problem_t>::ComputeEgasFromEint(
				    rho, xmom_new, ymom_new, zmom_new, state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index), Emag);

				// Optionally enforce strict gas+radiation energy conservation: remove the work-term energy
				// (c / chat) * dEkin from the chem-band photon field, split across bands in proportion to each band's
				// momentum contribution, and rescale each band's flux by the same factor its energy density changes so
				// the reduced flux |F| / (c E) is preserved (keeps the M1 closure F / c <= E from being violated by the
				// debit). Disabled by default: conserving the ionizing photon number density is more important than
				// strict energy conservation for correct ionization-front propagation (see debit_work_term_from_radiation).
				if constexpr (debit_work_term_from_radiation) {
					if (dMomMagSum > 0.0_rt) {
						const Real dEkin = (xmom_new * xmom_new + ymom_new * ymom_new + zmom_new * zmom_new -
								    (xmom * xmom + ymom * ymom + zmom * zmom)) /
								   (2.0_rt * rho);
						const Real E_debit = (C::c_light / photochemstate.reduced_speed_of_light) * dEkin;
						for (int nn = 0; nn < NumChemBands; ++nn) {
							const int eIdx = firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const int fxIdx = firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const int fyIdx = firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const int fzIdx = firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const Real E_old = state(i, j, k, eIdx); // > 0: rn[0] was clamped to small_x above
							const Real E_new = amrex::max(E_old - E_debit * (dMomMag[nn] / dMomMagSum),
										      integratorOptions.small_state * chemBandQuanta[nn]);
							state(i, j, k, eIdx) = E_new;
							const Real flux_rescale = E_new / E_old;
							state(i, j, k, fxIdx) *= flux_rescale;
							state(i, j, k, fyIdx) *= flux_rescale;
							state(i, j, k, fzIdx) *= flux_rescale;
						}
					}
				}
			}
		});

#ifdef AMREX_USE_HIP
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
	}

	num_failed = *(d_num_failed.copyToHost());
	amrex::ParallelDescriptor::ReduceIntMin(num_failed);

	photochem_burn_success = num_failed == 0;

	if (!photochem_burn_success) {
		amrex::Abort(std::format("Photochemistry integration failed with status {}.", num_failed));
	}

	return photochem_burn_success;
}

} // namespace quokka::photochemistry
#endif

#endif
