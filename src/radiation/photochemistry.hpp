#ifndef PHOTOCHEMISTRY_HPP_ // NOLINT
#define PHOTOCHEMISTRY_HPP_

#include <array>
#include <iostream>
#include <limits>

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
#include "integrator_data.H"
#include "network_properties.H"
#include "physics_numVars.hpp"

namespace quokka::photochemistry
{
AMREX_GPU_DEVICE void photochem_burner(burn_t &photochemstate, Real dt);

namespace detail
{
inline AMREX_GPU_MANAGED unsigned long long d_total_burns = 0;
inline AMREX_GPU_MANAGED unsigned long long d_total_steps = 0;
inline AMREX_GPU_MANAGED unsigned long long d_total_rhs = 0;
inline AMREX_GPU_MANAGED unsigned long long d_total_jac = 0;
} // namespace detail

struct PhotochemCounterTotals {
	unsigned long long burns{};
	unsigned long long steps{};
	unsigned long long rhs{};
	unsigned long long jac{};
};

inline auto getPhotochemCounterTotals() -> PhotochemCounterTotals
{
	return PhotochemCounterTotals{detail::d_total_burns, detail::d_total_steps, detail::d_total_rhs, detail::d_total_jac};
}

template <typename problem_t>
auto computePhotoChemistry(amrex::MultiFab &mf, const Real dt, const int stage, const Real max_density_allowed, const Real min_density_allowed) -> bool
{
	AMREX_ASSERT(stage == 1 || stage == 2);
	// Start off by assuming a successful burn.
	int photochem_burn_success = 1;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	auto *p_num_failed = d_num_failed.data();
	amrex::Gpu::Buffer<int> d_error_code({std::numeric_limits<int>::max()});
	auto *p_error_code = d_error_code.data();
	amrex::Gpu::Buffer<int> d_species_low({0});
	auto *p_species_low = d_species_low.data();
	amrex::Gpu::Buffer<int> d_species_high({0});
	auto *p_species_high = d_species_high.data();
	amrex::Gpu::Buffer<int> d_rad_low({0});
	auto *p_rad_low = d_rad_low.data();

	int num_failed = 0;
	int error_code = IERR_SUCCESS;
	int species_low = 0;
	int species_high = 0;
	int rad_low = 0;

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
				photochemstate.rn[1 + MicrophysicsNumRadVarsPerGroup * nn] = 1.0_rt;
			}
			photochemstate.rho = rho;
			photochemstate.e = Eint / rho;

			// call the EOS to set the temperature
			eos(eos_input_re, photochemstate);

			// do the actual integration
			// do it in .cpp so that it is not built at compile time for all tests
			// which would otherwise slow down compilation due to the large RHS file
			photochem_burner(photochemstate, dt_stage);

			amrex::Gpu::Atomic::Add(&detail::d_total_burns, 1ULL);
			amrex::Gpu::Atomic::Add(&detail::d_total_steps, static_cast<unsigned long long>(photochemstate.n_step));
			amrex::Gpu::Atomic::Add(&detail::d_total_rhs, static_cast<unsigned long long>(photochemstate.n_rhs));
			amrex::Gpu::Atomic::Add(&detail::d_total_jac, static_cast<unsigned long long>(photochemstate.n_jac));

			if (std::isnan(photochemstate.xn[0]) || std::isnan(photochemstate.rho) || std::isnan(photochemstate.rn[0])) {
				amrex::Abort("Burner returned NAN");
			}

			if (!photochemstate.success) {
				burn_failed = 1;
				for (int nn = 0; nn < NumSpec; ++nn) {
					if (photochemstate.xn[nn] < -integrator_rp::species_failure_tolerance) {
						amrex::Gpu::Atomic::Add(p_species_low, 1);
					}
					if (!integrator_rp::use_number_densities && photochemstate.xn[nn] > 1.0_rt + integrator_rp::species_failure_tolerance) {
						amrex::Gpu::Atomic::Add(p_species_high, 1);
					}
				}
				for (int nn = 0; nn < NumChemBands; ++nn) {
					const int rad_num_index = MicrophysicsNumRadVarsPerGroup * nn;
					if (photochemstate.rn[rad_num_index] < -integrator_rp::radiation_failure_tolerance) {
						amrex::Gpu::Atomic::Add(p_rad_low, 1);
					}
				}
			}

			if (burn_failed) {
				amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
				amrex::Gpu::Atomic::Min(p_error_code, static_cast<int>(photochemstate.error_code));
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
		});

#ifdef AMREX_USE_HIP
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
	}

	num_failed = *(d_num_failed.copyToHost());
	error_code = *(d_error_code.copyToHost());
	if (error_code == std::numeric_limits<int>::max()) {
		error_code = IERR_SUCCESS;
	}
	species_low = *(d_species_low.copyToHost());
	species_high = *(d_species_high.copyToHost());
	rad_low = *(d_rad_low.copyToHost());

	photochem_burn_success = num_failed == 0;
	amrex::ParallelDescriptor::ReduceIntMin(photochem_burn_success);
	amrex::ParallelDescriptor::ReduceIntMin(error_code);
	amrex::ParallelDescriptor::ReduceIntSum(species_low);
	amrex::ParallelDescriptor::ReduceIntSum(species_high);
	amrex::ParallelDescriptor::ReduceIntSum(rad_low);

	if (!photochem_burn_success) {
		amrex::Abort("Photochemistry burn failed with integrator error_code = " + std::to_string(error_code) +
			     ", species_low = " + std::to_string(species_low) + ", species_high = " + std::to_string(species_high) +
			     ", rad_low = " + std::to_string(rad_low) + ". Aborting.");
	}

	return photochem_burn_success;
}

} // namespace quokka::photochemistry
#endif

#endif
