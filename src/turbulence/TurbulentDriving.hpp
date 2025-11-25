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

#include "../extern/turbulence/plugins/AMReX/TurbGenEx.h"
#include "fmt/base.h"
#include "fmt/core.h"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "radiation/radiation_system.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace quokka::turbulence
{

template <typename problem_t> class turbulentDriving
{
      private:
	TurbGenEx tg{};
	bool updated = false;
	amrex::Gpu::DeviceVector<amrex::Real> disp = {-1.0, -1.0, -1.0};
	std::array<double, 3> host_disp = {-1.0, -1.0, -1.0};

	void update(const amrex::Real &time, amrex::MultiFab &state)
	{
		calculate_dispersion(state);
		updated = time == 0 ? tg.check_for_update(time) : tg.check_for_update(time, host_disp.data());
	}

      public:
	turbulentDriving() = default;
	explicit turbulentDriving(const std::string &fp) { tg.init_driving(fp); }

	auto computeDriving(amrex::MultiFab &state, const amrex::Real time, const amrex::Real dt_in,
			    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes) -> bool
	{
		update(time, state);
		const amrex::Real dt = dt_in;

		for (amrex::MFIter mf(state); mf.isValid(); ++mf) {
			const amrex::Box &bx = mf.validbox();
			auto const &data = state.array(mf);

			amrex::FArrayBox forceFieldFab(bx, AMREX_SPACEDIM, amrex::The_Async_Arena());
			amrex::Array4<amrex::Real> const forceField = forceFieldFab.array();

			tg.get_turb_vector_unigrid(forceFieldFab, cellSizes);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = data(i, j, k, HydroSystem<problem_t>::density_index);

				amrex::Real dE = 0;

				for (int m = 0; m < AMREX_SPACEDIM; m++) {
					const amrex::Real dMom = forceField(i, j, k, m) * dt;

					data(i, j, k, HydroSystem<problem_t>::x1Momentum_index + m) += dMom;
					dE += dMom * dMom / (2 * rho);
				}

				data(i, j, k, HydroSystem<problem_t>::energy_index) += dE;
			});
		}

		amrex::Gpu::streamSynchronize();
		return true;
	}

	void calculate_dispersion(amrex::MultiFab &state)
	{
		amrex::Real sum_rho = state.sum(HydroSystem<problem_t>::density_index, false);
		amrex::Real sum_px = state.sum(HydroSystem<problem_t>::x1Momentum_index, false);
		amrex::Real sum_py = state.sum(HydroSystem<problem_t>::x2Momentum_index, false);
		amrex::Real sum_pz = state.sum(HydroSystem<problem_t>::x3Momentum_index, false);
		amrex::GpuArray<amrex::Real, 3> const v_avg = {sum_px / sum_rho, sum_py / sum_rho, sum_pz / sum_rho};

		amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
		amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);

		for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			auto const &data = state.array(mfi);

			reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real> {
				amrex::Real rho = data(i, j, k, HydroSystem<problem_t>::density_index);
				amrex::Real vx = data(i, j, k, HydroSystem<problem_t>::x1Momentum_index) / rho;
				amrex::Real vy = data(i, j, k, HydroSystem<problem_t>::x2Momentum_index) / rho;
				amrex::Real vz = data(i, j, k, HydroSystem<problem_t>::x3Momentum_index) / rho;

				return {rho * (vx - v_avg[0]) * (vx - v_avg[0]), rho * (vy - v_avg[1]) * (vy - v_avg[1]),
					rho * (vz - v_avg[2]) * (vz - v_avg[2])};
			});
		}

		auto stdd = reduce_data.value();
		amrex::Real dispx = amrex::get<0>(stdd);
		amrex::Real dispy = amrex::get<1>(stdd);
		amrex::Real dispz = amrex::get<2>(stdd);

		amrex::ParallelDescriptor::ReduceRealSum(dispx);
		amrex::ParallelDescriptor::ReduceRealSum(dispy);
		amrex::ParallelDescriptor::ReduceRealSum(dispz);

		disp = {std::sqrt(dispx / sum_rho), std::sqrt(dispy / sum_rho), std::sqrt(dispz / sum_rho)};
		amrex::Gpu::copy(amrex::Gpu::deviceToHost, disp.begin(), disp.end(), host_disp.begin());
	}
};
} // namespace quokka::turbulence

#endif // TURBULENTDRIVING_HPP
