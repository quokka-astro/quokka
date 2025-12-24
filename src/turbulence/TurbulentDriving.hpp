#ifndef TURBULENTDRIVING_HPP
#define TURBULENTDRIVING_HPP

#include "AMReX.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArray.H"
#include "AMReX_FabFactory.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuControl.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_iMultiFab.H"

#include "TurbGenEx.h"
#include "fmt/base.h"
#include "fmt/core.h"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "radiation/radiation_system.hpp"
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace quokka::turbulence
{
template <typename problem_t> auto calculate_dispersion(amrex::MultiFab &state) -> amrex::GpuArray<amrex::Real, 3>;

template <typename problem_t> class turbulentDriving
{
      private:
	TurbGenEx tg;
	bool updated = false;
	amrex::GpuArray<amrex::Real, 3> disp = {-1.0, -1.0, -1.0};

	void update(const amrex::Real &time, amrex::MultiFab &state)
	{
		disp = quokka::turbulence::calculate_dispersion<problem_t>(state);
		updated = time == 0 ? tg.check_for_update(time) : tg.check_for_update(time, disp.data());
	}

      public:
	turbulentDriving() = default;
	explicit turbulentDriving(const std::map<std::string, std::string> &turb_params) { tg.init_driving(turb_params); }

	auto applyDriving(amrex::MultiFab &state, const amrex::Real time, const amrex::Real dt_in,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes) -> bool
	{
		update(time, state);
		const amrex::Real dt = dt_in;

		for (amrex::MFIter mf(state); mf.isValid(); ++mf) {
			const amrex::Box &bx = mf.validbox();
			auto const &data = state.array(mf);

			amrex::FArrayBox axFab(bx, AMREX_SPACEDIM, amrex::The_Async_Arena());
			amrex::Array4<amrex::Real> const ax = axFab.array();

			tg.get_turb_vector_unigrid(axFab, cellSizes);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = data(i, j, k, HydroSystem<problem_t>::density_index);

				amrex::Real dE = 0;

				for (int m = 0; m < AMREX_SPACEDIM; m++) {
					const amrex::Real vel = data(i, j, k, HydroSystem<problem_t>::x1Momentum_index + m) / rho;
					const amrex::Real dMom = ax(i, j, k, m) * dt * rho;

					data(i, j, k, HydroSystem<problem_t>::x1Momentum_index + m) += dMom;
					dE += vel * dMom + dMom * dMom / (2 * rho);
				}

				data(i, j, k, HydroSystem<problem_t>::energy_index) += dE;
			});
		}

		amrex::Gpu::streamSynchronize();
		return true;
	}
};

// Function to calculate the mass weighted velocity dispersion in the computational domain
template <typename problem_t> auto calculate_dispersion(amrex::MultiFab &state) -> amrex::GpuArray<amrex::Real, 3>
{
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
			 amrex::ReduceOpSum>
	    reduce_op;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &data = state.array(mfi);

		reduce_op.eval(bx, reduce_data,
			       [=] AMREX_GPU_DEVICE(int i, int j, int k)
				   -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
				       const amrex::Real rho = data(i, j, k, HydroSystem<problem_t>::density_index);
				       const amrex::Real px = data(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				       const amrex::Real py = data(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				       const amrex::Real pz = data(i, j, k, HydroSystem<problem_t>::x3Momentum_index);

				       const amrex::Real vx = px / rho;
				       const amrex::Real vy = py / rho;
				       const amrex::Real vz = pz / rho;

				       return {rho, px, py, pz, px * vx, py * vy, pz * vz};
			       });
	}

	auto [sum_rho, sum_px, sum_py, sum_pz, sum_pvx, sum_pvy, sum_pvz] = reduce_data.value();

	amrex::GpuArray<amrex::Real, 7> reduce_vec = {sum_rho, sum_px, sum_py, sum_pz, sum_pvx, sum_pvy, sum_pvz};
	amrex::ParallelDescriptor::ReduceRealSum(reduce_vec.data(), 7);

	const amrex::Real total_rho = reduce_vec[0];
	const amrex::Real total_px = reduce_vec[1];
	const amrex::Real total_py = reduce_vec[2];
	const amrex::Real total_pz = reduce_vec[3];
	const amrex::Real total_pvx = reduce_vec[4];
	const amrex::Real total_pvy = reduce_vec[5];
	const amrex::Real total_pvz = reduce_vec[6];

	const amrex::Real v_avg_x = total_px / total_rho;
	const amrex::Real v_avg_y = total_py / total_rho;
	const amrex::Real v_avg_z = total_pz / total_rho;

	// Compute dispersion using the identity: Var(X) = E[X^2] - (E[X])^2
	const amrex::Real dispx = std::sqrt(std::max(0.0, (total_pvx / total_rho) - (v_avg_x * v_avg_x)));
	const amrex::Real dispy = std::sqrt(std::max(0.0, (total_pvy / total_rho) - (v_avg_y * v_avg_y)));
	const amrex::Real dispz = std::sqrt(std::max(0.0, (total_pvz / total_rho) - (v_avg_z * v_avg_z)));

	return {dispx, dispy, dispz};
}
} // namespace quokka::turbulence

#endif // TURBULENTDRIVING_HPP
