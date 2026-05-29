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
#include "AMReX_MultiFabUtil.H"

#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"

#ifdef CHEMISTRY
#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"

namespace quokka::chemistry
{

AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, Real dt);
void chemburnerHost(burn_t &chemstate, Real dt);

template <typename problem_t>
void replayFailedBurnOnCpu(amrex::Vector<amrex::Real> const &cell_values, amrex::IntVect const &failed_index, const Real dt, const int lev, const Real time,
			   char const *source_stage)
{
	if (cell_values.empty()) {
		return;
	}

	const Real rho = cell_values[HydroSystem<problem_t>::density_index];
	const Real xmom = cell_values[HydroSystem<problem_t>::x1Momentum_index];
	const Real ymom = cell_values[HydroSystem<problem_t>::x2Momentum_index];
	const Real zmom = cell_values[HydroSystem<problem_t>::x3Momentum_index];
	const Real Ener = cell_values[HydroSystem<problem_t>::energy_index];
	const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener);

	burn_t replay_state;
	replay_state.success = true;
	replay_state.debug_replay = true;
	replay_state.debug_replay_max_logs = 200;
	replay_state.i = failed_index[0];
#if AMREX_SPACEDIM >= 2
	replay_state.j = failed_index[1];
#endif
#if AMREX_SPACEDIM == 3
	replay_state.k = failed_index[2];
#endif

	for (int nn = 0; nn < NumSpec; ++nn) {
		replay_state.xn[nn] = cell_values[HydroSystem<problem_t>::scalar0_index + nn] / spmasses[nn];
	}

	replay_state.rho = rho;
	replay_state.e = Eint / rho;
	eos(eos_input_re, replay_state);

	amrex::AllPrint() << "\t>> CPU replay of failed burn begins: level = " << lev << ", stage = " << source_stage << ", time = " << time << ", dt = " << dt
			  << ", cell = " << failed_index << ", rho = " << replay_state.rho << ", T = " << replay_state.T << ", e = " << replay_state.e << "\n";
	chemburnerHost(replay_state, dt);
	amrex::AllPrint() << "\t>> CPU replay of failed burn ends: success = " << replay_state.success << ", error_code = " << replay_state.error_code
			  << ", n_step = " << replay_state.n_step << ", n_rhs = " << replay_state.n_rhs << ", n_jac = " << replay_state.n_jac
			  << ", burn_time_reached = " << replay_state.time << ", T = " << replay_state.T << ", e = " << replay_state.e << "\n";
}

template <typename problem_t>
auto computeChemistry(amrex::MultiFab &mf, const Real dt, const Real max_density_allowed, const Real min_density_allowed, const int burn_failure_verbose,
		      const int burn_failure_cpu_replay, const int lev, const Real time, char const *source_stage) -> bool
{

	// Start off by assuming a successful burn.
	int burn_success = 1;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	auto *p_num_failed = d_num_failed.data();

	int num_failed = 0;
	amrex::Gpu::Buffer<int> d_first_failed_cell({0, 0, 0, 0, 0, 0, 0});
	amrex::Gpu::Buffer<Real> d_first_failed_burn_state({0.0_rt, 0.0_rt, 0.0_rt});
	auto *p_first_failed_cell = d_first_failed_cell.data();
	auto *p_first_failed_burn_state = d_first_failed_burn_state.data();

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
			chemstate.success = true;
			int burn_failed = 0;

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

			if (std::isnan(chemstate.xn[0]) || std::isnan(chemstate.rho)) {
				amrex::Abort("Burner returned NAN");
			}

			if (!chemstate.success) {
				burn_failed = 1;
			}

			if (burn_failed) {
				const int previous_failures = amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
				if (previous_failures == 0) {
					p_first_failed_cell[0] = i;
					p_first_failed_cell[1] = j;
					p_first_failed_cell[2] = k;
					p_first_failed_cell[3] = chemstate.error_code;
					p_first_failed_cell[4] = chemstate.n_step;
					p_first_failed_cell[5] = chemstate.n_rhs;
					p_first_failed_cell[6] = chemstate.n_jac;
					p_first_failed_burn_state[0] = chemstate.time;
					p_first_failed_burn_state[1] = chemstate.T;
					p_first_failed_burn_state[2] = chemstate.e;
				}
				return;
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

#if defined(AMREX_USE_HIP)
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
	}

	num_failed = *(d_num_failed.copyToHost());
	auto *h_first_failed_cell = d_first_failed_cell.copyToHost();
	auto *h_first_failed_burn_state = d_first_failed_burn_state.copyToHost();
	int global_num_failed = num_failed;
	amrex::ParallelDescriptor::ReduceIntSum(global_num_failed);

	burn_success = !num_failed;
	amrex::ParallelDescriptor::ReduceIntMin(burn_success);

	if (!burn_success) {
		// amrex::Abort("Burn failed in chemistry integrator. Aborting.");
		amrex::Print() << "\t>> WARNING: Unsuccessful burn. Retrying hydro step. Failed cells = " << global_num_failed << "\n";

		if (num_failed > 0) {
			const amrex::IntVect failed_index{AMREX_D_DECL(h_first_failed_cell[0], h_first_failed_cell[1], h_first_failed_cell[2])};
			const int error_code = h_first_failed_cell[3];
			amrex::AllPrint() << "\t>> Burn failure summary: level = " << lev << ", stage = " << source_stage << ", time = " << time
					  << ", dt = " << dt << ", cell = " << failed_index << ", error_code = " << error_code << "\n";
			amrex::AllPrint() << "\t>> Failed burn integrator diagnostics: n_step = " << h_first_failed_cell[4]
					  << ", n_rhs = " << h_first_failed_cell[5] << ", n_jac = " << h_first_failed_cell[6]
					  << ", burn_time_reached = " << h_first_failed_burn_state[0] << ", T = " << h_first_failed_burn_state[1]
					  << ", e = " << h_first_failed_burn_state[2] << "\n";

			amrex::Vector<amrex::Real> const cell_values = amrex::get_cell_data(mf, failed_index);
			if (!cell_values.empty()) {
				amrex::AllPrint() << "\t>> Pre-burn state components:";
				for (int n = 0; n < static_cast<int>(cell_values.size()); ++n) {
					amrex::AllPrint() << " [" << n << "]=" << cell_values[n];
				}
				amrex::AllPrint() << "\n";
			}

			if (burn_failure_cpu_replay != 0 && !cell_values.empty()) {
				replayFailedBurnOnCpu<problem_t>(cell_values, failed_index, dt, lev, time, source_stage);
			}

			if (burn_failure_verbose != 0) {
				if (!cell_values.empty()) {
					const Real rho = cell_values[HydroSystem<problem_t>::density_index];
					const Real xmom = cell_values[HydroSystem<problem_t>::x1Momentum_index];
					const Real ymom = cell_values[HydroSystem<problem_t>::x2Momentum_index];
					const Real zmom = cell_values[HydroSystem<problem_t>::x3Momentum_index];
					const Real Ener = cell_values[HydroSystem<problem_t>::energy_index];
					const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener);
					const Real Eint_aux = cell_values[HydroSystem<problem_t>::internalEnergy_index];

					amrex::AllPrint() << "\t>> Burn failure detail:"
							  << " rho = " << rho << ", xmom = " << xmom << ", ymom = " << ymom << ", zmom = " << zmom
							  << ", Etot = " << Ener << ", Eint = " << Eint << ", Eint_aux = " << Eint_aux << "\n";

					for (int nn = 0; nn < NumSpec; ++nn) {
						const Real partial_density = cell_values[HydroSystem<problem_t>::scalar0_index + nn];
						const Real mass_fraction = partial_density / rho;
						const Real number_density = partial_density / spmasses[nn];
						amrex::AllPrint() << "\t   species[" << nn << "]: partial_density = " << partial_density
								  << ", mass_fraction = " << mass_fraction << ", number_density = " << number_density << "\n";
					}
				}
			}
		}
	}

	return burn_success;
}

} // namespace quokka::chemistry
#endif
#endif // CHEMISTRY_HPP_
