//==============================================================================
// Copyright 2025 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testOrszagTang.cpp
/// \brief
///   This problem is based on the implementation here:
///   https://github.com/PrincetonUniversity/athena/blob/master/src/pgen/orszag_tang.cpp.
///	  (Phil Hopkins made several typos on this page, do not use:
///	  https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html)
///

#include <cmath>
#include <format>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct OrszagTang {
};

template <> struct quokka::EOS_Traits<OrszagTang> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<OrszagTang> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

constexpr double B0 = 1.0 / gcem::sqrt(4.0 * PI);

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto A_z(double x, double y) -> double
{
	return B0 / (4.0 * M_PI) * (std::cos(4.0 * M_PI * x) - 2.0 * std::cos(2.0 * M_PI * y));
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_x(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1];
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_y(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0];
};

template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	constexpr double gamma_gas = quokka::EOS_Traits<OrszagTang>::gamma;
	constexpr double rho0 = 25. / (36. * M_PI);
	constexpr double P0 = 5. / (12. * M_PI);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		const double vx = std::sin(2 * M_PI * y);
		const double vy = -std::sin(2 * M_PI * x);

		const double Bx = 0.5 * (B_x(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_x(x + 0.5 * dx[0], y - 0.5 * dx[1], dx));
		const double By = 0.5 * (B_y(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_y(x - 0.5 * dx[0], y + 0.5 * dx[1], dx));

		const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy);
		const double Eint = P0 / (gamma_gas - 1.0);
		const double Emag = 0.5 * (Bx * Bx + By * By);

		state_cc(i, j, k, HydroSystem<OrszagTang>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x1Momentum_index) = rho0 * vx;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x2Momentum_index) = rho0 * vy;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<OrszagTang>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<OrszagTang>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + (i * dx[0]);
		const double yL = prob_lo[1] + (j * dx[1]);

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = B_x(xL, yL, dx);
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = B_y(xL, yL, dx);
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = 0;
		}
	});
}

template <> void QuokkaSimulation<OrszagTang>::computeAfterTimestep()
{
	constexpr int lev = 0;
	constexpr int bidx = Physics_Indices<OrszagTang>::mhdFirstIndex;

	const amrex::Box domain = Geom(lev).Domain();
	const int nx = domain.length(0);
	const int ny = domain.length(1);

	// x-face Bx: rot180 maps (i,j,k) -> (nx-i, ny-1-j, k), Bx is odd
	amrex::Real max_res_x = 0.;
	amrex::Real max_abs_x = 0.;
	{
		const amrex::MultiFab &Bx_mf = state_new_fc_[lev][0];
		for (amrex::MFIter mfi(Bx_mf); mfi.isValid(); ++mfi) {
			auto const &Bx = Bx_mf.const_array(mfi);
			const amrex::Box &box = mfi.validbox();
			amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_op;
			amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
			using ReduceTuple = typename decltype(reduce_data)::Type;
			reduce_op.eval(box, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
				const amrex::Real bx = Bx(i, j, k, bidx);
				const amrex::Real bx_rot = Bx(nx - i, ny - 1 - j, k, bidx);
				return {std::abs(bx + bx_rot), std::abs(bx)};
			});
			auto [res, ab] = reduce_data.value();
			max_res_x = std::max(max_res_x, res);
			max_abs_x = std::max(max_abs_x, ab);
		}
	}

	// y-face By: rot180 maps (i,j,k) -> (nx-1-i, ny-j, k), By is odd
	amrex::Real max_res_y = 0.;
	amrex::Real max_abs_y = 0.;
	{
		const amrex::MultiFab &By_mf = state_new_fc_[lev][1];
		for (amrex::MFIter mfi(By_mf); mfi.isValid(); ++mfi) {
			auto const &By = By_mf.const_array(mfi);
			const amrex::Box &box = mfi.validbox();
			amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_op;
			amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
			using ReduceTuple = typename decltype(reduce_data)::Type;
			reduce_op.eval(box, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
				const amrex::Real by = By(i, j, k, bidx);
				const amrex::Real by_rot = By(nx - 1 - i, ny - j, k, bidx);
				return {std::abs(by + by_rot), std::abs(by)};
			});
			auto [res, ab] = reduce_data.value();
			max_res_y = std::max(max_res_y, res);
			max_abs_y = std::max(max_abs_y, ab);
		}
	}

	amrex::ParallelDescriptor::ReduceRealMax(max_res_x);
	amrex::ParallelDescriptor::ReduceRealMax(max_abs_x);
	amrex::ParallelDescriptor::ReduceRealMax(max_res_y);
	amrex::ParallelDescriptor::ReduceRealMax(max_abs_y);

	const amrex::Real rel_x = (max_abs_x > 0.) ? max_res_x / max_abs_x : 0.;
	const amrex::Real rel_y = (max_abs_y > 0.) ? max_res_y / max_abs_y : 0.;

	// second pass: locate the (i,j) cell attaining max_res_x / max_res_y, for tracing the seed back to its source.
	// packs (i,j) into a double score, valid only at cells matching the max residual (reversible since nx,ny <<
	// 2^26, well within double's exact-integer range). inlined (not a helper lambda) because nvcc's extended
	// __device__ lambdas cannot capture a locally-defined lambda type.
	amrex::Real loc_score_x = -1.;
	{
		const amrex::MultiFab &Bx_mf = state_new_fc_[lev][0];
		for (amrex::MFIter mfi(Bx_mf); mfi.isValid(); ++mfi) {
			auto const &Bx = Bx_mf.const_array(mfi);
			const amrex::Box &box = mfi.validbox();
			amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
			amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
			using ReduceTuple = typename decltype(reduce_data)::Type;
			reduce_op.eval(box, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
				const amrex::Real bx = Bx(i, j, k, bidx);
				const amrex::Real bx_rot = Bx(nx - i, ny - 1 - j, k, bidx);
				const amrex::Real res = std::abs(bx + bx_rot);
				return {(res >= max_res_x) ? (static_cast<double>(i) * 100000.0 + static_cast<double>(j)) : -1.};
			});
			loc_score_x = std::max(loc_score_x, amrex::get<0>(reduce_data.value()));
		}
	}
	amrex::Real loc_score_y = -1.;
	{
		const amrex::MultiFab &By_mf = state_new_fc_[lev][1];
		for (amrex::MFIter mfi(By_mf); mfi.isValid(); ++mfi) {
			auto const &By = By_mf.const_array(mfi);
			const amrex::Box &box = mfi.validbox();
			amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
			amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
			using ReduceTuple = typename decltype(reduce_data)::Type;
			reduce_op.eval(box, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
				const amrex::Real by = By(i, j, k, bidx);
				const amrex::Real by_rot = By(nx - 1 - i, ny - j, k, bidx);
				const amrex::Real res = std::abs(by + by_rot);
				return {(res >= max_res_y) ? (static_cast<double>(i) * 100000.0 + static_cast<double>(j)) : -1.};
			});
			loc_score_y = std::max(loc_score_y, amrex::get<0>(reduce_data.value()));
		}
	}
	amrex::ParallelDescriptor::ReduceRealMax(loc_score_x);
	amrex::ParallelDescriptor::ReduceRealMax(loc_score_y);
	const int ix = static_cast<int>(loc_score_x / 100000.0);
	const int jx = static_cast<int>(loc_score_x) % 100000;
	const int iy = static_cast<int>(loc_score_y / 100000.0);
	const int jy = static_cast<int>(loc_score_y) % 100000;

	amrex::Print() << std::format("[rot180] step={} Bx={:.4e} By={:.4e} argmax_x=({},{}) argmax_y=({},{})\n", cycleCount_, rel_x, rel_y, ix, jx, iy, jy);
}

auto problem_main() -> int
{
	QuokkaSimulation<OrszagTang> sim;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
