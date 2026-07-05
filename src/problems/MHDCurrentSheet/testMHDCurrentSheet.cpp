//==============================================================================
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDCurrentSheet.cpp
/// \brief Sharp, rot180-symmetric current-sheet bench for localising floating-point
///   symmetry-breaking in the MHD reconstruction. Bx(y) = B0 tanh(sin(2 pi y)/delta)
///   gives sharp current sheets (odd under rot180); vy = V0 sin(2 pi x) (odd under
///   rot180) makes the EMF nonzero so the reconstruction is exercised at the sharp
///   gradient from step 1. Density and pressure are uniform; the field is div-free
///   (Bx depends on y only, By = 0). Not an equilibrium: it is a one-step symmetry
///   probe, not a physics run.
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

struct MHDCurrentSheet {
};

template <> struct quokka::EOS_Traits<MHDCurrentSheet> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDCurrentSheet> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

constexpr double B0 = 1.0;
constexpr double rho0 = 1.0;
constexpr double P0 = 1.0;
constexpr double V0 = 1.0;
constexpr double delta = 0.05; // sheet half-width; smaller is sharper

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto Bx_of_y(double y) -> double { return B0 * std::tanh(std::sin(2.0 * M_PI * y) / delta); }

template <> void QuokkaSimulation<MHDCurrentSheet>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	constexpr double gamma_gas = quokka::EOS_Traits<MHDCurrentSheet>::gamma;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		const double vy = V0 * std::tanh(std::sin(2.0 * M_PI * x) / delta); // sharp shear, odd under rot180
		const double Bx = Bx_of_y(y);					    // sharp sheet, odd under rot180; By = 0

		const double Ekin = 0.5 * rho0 * (vy * vy);
		const double Eint = P0 / (gamma_gas - 1.0);
		const double Emag = 0.5 * (Bx * Bx);

		state_cc(i, j, k, HydroSystem<MHDCurrentSheet>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<MHDCurrentSheet>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<MHDCurrentSheet>::x2Momentum_index) = rho0 * vy;
		state_cc(i, j, k, HydroSystem<MHDCurrentSheet>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<MHDCurrentSheet>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<MHDCurrentSheet>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<MHDCurrentSheet>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			// x-face: nodal in x, cell-centred in y -> use y = (j + 0.5) dy
			const double y = prob_lo[1] + ((j + 0.5) * dx[1]);
			state_fc(i, j, k, Physics_Indices<MHDCurrentSheet>::mhdFirstIndex) = Bx_of_y(y);
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<MHDCurrentSheet>::mhdFirstIndex) = 0;
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<MHDCurrentSheet>::mhdFirstIndex) = 0;
		}
	});
}

template <> void QuokkaSimulation<MHDCurrentSheet>::computeAfterTimestep()
{
	constexpr int lev = 0;
	constexpr int bidx = Physics_Indices<MHDCurrentSheet>::mhdFirstIndex;

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
	amrex::Print() << std::format("[rot180] step={} Bx={:.4e} By={:.4e}\n", cycleCount_, rel_x, rel_y);
}

auto problem_main() -> int
{
	QuokkaSimulation<MHDCurrentSheet> sim;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
