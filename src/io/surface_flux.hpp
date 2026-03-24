#ifndef SURFACE_FLUX_HPP
#define SURFACE_FLUX_HPP

#include <array>
#include <cmath>

#include "AMReX_Array4.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelReduce.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "AMReX_iMultiFab.H"
#include "hydro/hydro_system.hpp"
#include "math/spherical_geometry.hpp"

namespace quokka::diagnostics
{

struct SurfaceFluxes {
	amrex::Real mass_flux{0.0};
	amrex::Real hydro_energy_flux{0.0};
	amrex::Real mhd_energy_flux{0.0};
	amrex::Real passive_scalar_flux{0.0};
};

template <typename problem_t>
auto computeSphericalSurfaceFluxes(amrex::Vector<amrex::MultiFab> const &state_cc,
				   amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &state_fc,
				   amrex::Vector<amrex::Geometry> const &geoms, amrex::Vector<amrex::iMultiFab> const &flux_mask,
				   amrex::Real flux_sphere_radius) -> SurfaceFluxes
{
	SurfaceFluxes result{};

	for (int lev = 0; lev < state_cc.size(); ++lev) {
		const auto prob_lo = geoms[lev].ProbLoArray();
		const auto dx = geoms[lev].CellSizeArray();
		auto const &state = state_cc[lev].const_arrays();
		auto const &mask = flux_mask[lev].const_arrays();

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			auto const &state_fc_x = state_fc[lev][0].const_arrays();
			auto const &state_fc_y = state_fc[lev][1].const_arrays();
			auto const &state_fc_z = state_fc[lev][2].const_arrays();

			auto const level_flux = amrex::ParReduce(
			    amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>{},
			    amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real>{}, state_cc[lev], amrex::IntVect(0),
			    [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
				    if (mask[bx](i, j, k) == 0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    const amrex::Real x0 = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
				    const amrex::Real y0 = prob_lo[1] + static_cast<amrex::Real>(j) * dx[1];
				    const amrex::Real z0 = prob_lo[2] + static_cast<amrex::Real>(k) * dx[2];
				    const amrex::Real x1 = x0 + dx[0];
				    const amrex::Real y1 = y0 + dx[1];
				    const amrex::Real z1 = z0 + dx[2];

				    const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				    const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				    const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
				    const amrex::Real r = std::sqrt(x * x + y * y + z * z);

				    const amrex::Real rho = state[bx](i, j, k, HydroSystem<problem_t>::density_index);
				    if (r <= 0.0 || rho <= 0.0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    const amrex::Real momx = state[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				    const amrex::Real momy = state[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				    const amrex::Real momz = state[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
				    const amrex::Real vx = momx / rho;
				    const amrex::Real vy = momy / rho;
				    const amrex::Real vz = momz / rho;
				    const amrex::Real vr = (x * momx + y * momy + z * momz) / (rho * r);
				    const amrex::Real rhat_x = x / r;
				    const amrex::Real rhat_y = y / r;
				    const amrex::Real rhat_z = z / r;

				    const amrex::Real mass_flux_density = rho * vr;
				    const amrex::Real energy_density = state[bx](i, j, k, HydroSystem<problem_t>::energy_index);
				    amrex::Real scalar_density = 0.0;
				    if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
					    scalar_density = state[bx](i, j, k, HydroSystem<problem_t>::scalar0_index);
				    }

				    std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
					AMREX_D_DECL(state_fc_x[bx], state_fc_y[bx], state_fc_z[bx])};
				    const amrex::Real Pgas = HydroSystem<problem_t>::ComputePressure(state[bx], i, j, k, &cons_fc);
				    const amrex::Real Emag = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &cons_fc);
				    const amrex::Real Ehydro = energy_density - Emag;

				    const amrex::Real bx1_m = cons_fc[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
				    const amrex::Real bx1_p = cons_fc[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
				    const amrex::Real bx2_m = cons_fc[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
				    const amrex::Real bx2_p = cons_fc[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
				    const amrex::Real bx3_m = cons_fc[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
				    const amrex::Real bx3_p = cons_fc[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
				    const amrex::Real Bx = 0.5 * (bx1_m + bx1_p);
				    const amrex::Real By = 0.5 * (bx2_m + bx2_p);
				    const amrex::Real Bz = 0.5 * (bx3_m + bx3_p);
				    const amrex::Real Bdotv = vx * Bx + vy * By + vz * Bz;
				    const amrex::Real Br = rhat_x * Bx + rhat_y * By + rhat_z * Bz;

				    const amrex::Real hydro_energy_flux_density = (Ehydro + Pgas) * vr;
				    const amrex::Real mhd_energy_flux_density = (energy_density + Pgas + Emag) * vr - Bdotv * Br;
				    const amrex::Real area = quokka::math::sphericalSectionAreaInCell(flux_sphere_radius, x0, x1, y0, y1, z0, z1);
				    if (area <= 0.0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    return {mass_flux_density * area, hydro_energy_flux_density * area, mhd_energy_flux_density * area,
					    scalar_density * vr * area};
			    });

			result.mass_flux += amrex::get<0>(level_flux);
			result.hydro_energy_flux += amrex::get<1>(level_flux);
			result.mhd_energy_flux += amrex::get<2>(level_flux);
			result.passive_scalar_flux += amrex::get<3>(level_flux);
		} else {
			auto const level_flux = amrex::ParReduce(
			    amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>{},
			    amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real>{}, state_cc[lev], amrex::IntVect(0),
			    [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
				    if (mask[bx](i, j, k) == 0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    const amrex::Real x0 = prob_lo[0] + static_cast<amrex::Real>(i) * dx[0];
				    const amrex::Real y0 = prob_lo[1] + static_cast<amrex::Real>(j) * dx[1];
				    const amrex::Real z0 = prob_lo[2] + static_cast<amrex::Real>(k) * dx[2];
				    const amrex::Real x1 = x0 + dx[0];
				    const amrex::Real y1 = y0 + dx[1];
				    const amrex::Real z1 = z0 + dx[2];

				    const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				    const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
				    const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
				    const amrex::Real r = std::sqrt(x * x + y * y + z * z);

				    const amrex::Real rho = state[bx](i, j, k, HydroSystem<problem_t>::density_index);
				    if (r <= 0.0 || rho <= 0.0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    const amrex::Real momx = state[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				    const amrex::Real momy = state[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				    const amrex::Real momz = state[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
				    const amrex::Real vr = (x * momx + y * momy + z * momz) / (rho * r);

				    const amrex::Real mass_flux_density = rho * vr;
				    const amrex::Real energy_density = state[bx](i, j, k, HydroSystem<problem_t>::energy_index);
				    amrex::Real scalar_density = 0.0;
				    if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
					    scalar_density = state[bx](i, j, k, HydroSystem<problem_t>::scalar0_index);
				    }

				    const amrex::Real Pgas = HydroSystem<problem_t>::ComputePressure(state[bx], i, j, k);
				    const amrex::Real hydro_energy_flux_density = (energy_density + Pgas) * vr;
				    const amrex::Real area = quokka::math::sphericalSectionAreaInCell(flux_sphere_radius, x0, x1, y0, y1, z0, z1);
				    if (area <= 0.0) {
					    return {0.0, 0.0, 0.0, 0.0};
				    }

				    const amrex::Real energy_flux = hydro_energy_flux_density * area;
				    return {mass_flux_density * area, energy_flux, energy_flux, scalar_density * vr * area};
			    });

			result.mass_flux += amrex::get<0>(level_flux);
			result.hydro_energy_flux += amrex::get<1>(level_flux);
			result.mhd_energy_flux += amrex::get<2>(level_flux);
			result.passive_scalar_flux += amrex::get<3>(level_flux);
		}
	}

	std::array<amrex::Real, 4> reduced = {result.mass_flux, result.hydro_energy_flux, result.mhd_energy_flux, result.passive_scalar_flux};
	amrex::ParallelAllReduce::Sum(reduced.data(), reduced.size(), amrex::ParallelContext::CommunicatorSub());

	result.mass_flux = reduced[0];
	result.hydro_energy_flux = reduced[1];
	result.mhd_energy_flux = reduced[2];
	result.passive_scalar_flux = reduced[3];

	return result;
}

} // namespace quokka::diagnostics

#endif
