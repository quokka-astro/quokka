#ifndef ELECTRON_CONDUCTION_HPP_ // NOLINT
#define ELECTRON_CONDUCTION_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ElectronConduction.hpp
/// \brief Explicit flux-limited electron thermal conduction update.

#include <array>
#include <cmath>
#include <limits>

#include "AMReX_Array4.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "hydro/hydro_system.hpp"

namespace quokka::conduction
{

struct ElectronConductionParams {
	amrex::Real conductivity_prefactor = 3.e34; 
	amrex::Real flux_limiter_phi = 0.1;
	amrex::Real saturation_factor = 5.0;
	amrex::Real min_temperature = 0.0;
};

template <typename problem_t> class ElectronConduction
{
      public:
	static void ComputeExplicit(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, amrex::Geometry const &geom,
				    amrex::Real dt, ElectronConductionParams const &params)
	{
		static_assert(Physics_Traits<problem_t>::is_hydro_enabled, "Electron conduction requires hydro to be enabled.");
		
		if ((dt <= 0.0) || (params.conductivity_prefactor <= 0.0)) {
			return;
		}

		if constexpr (HydroSystem<problem_t>::is_eos_isothermal()) {
			amrex::ignore_unused(state_fc, geom, params);
			return;
		}

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.nGrow() >= 1, "Electron conduction requires at least 1 ghost cell.");

		const auto dx = geom.CellSizeArray();
		const amrex::Real flux_limiter_phi = params.flux_limiter_phi;
		const amrex::Real saturation_factor = params.saturation_factor;
		const amrex::Real t_min = params.min_temperature;
		const amrex::Real small = std::numeric_limits<amrex::Real>::min();

		amrex::MultiFab temperature(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
		temperature.setVal(0.0);
		amrex::MultiFab conductivity(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
		conductivity.setVal(0.0);
		amrex::MultiFab saturated_flux(state.boxArray(), state.DistributionMap(), 1, state.nGrow());
		saturated_flux.setVal(0.0);

		auto const &state_x0 = state.const_arrays();
		auto const &state_fc_x0 = state_fc[0].const_arrays();
#if AMREX_SPACEDIM >= 2
		auto const &state_fc_x1 = state_fc[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
		auto const &state_fc_x2 = state_fc[2].const_arrays();
#endif
		auto temperature_arr = temperature.arrays();
		auto conductivity_arr = conductivity.arrays();
		auto saturated_flux_arr = saturated_flux.arrays();
		amrex::IntVect ng = amrex::IntVect(AMREX_D_DECL(2, 2, 2));

		amrex::ParallelFor(state, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_state_fc{};
			if (Physics_Traits<problem_t>::is_mhd_enabled) {
				local_state_fc[0] = state_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
				local_state_fc[1] = state_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
				local_state_fc[2] = state_fc_x2[bx];
#endif
			}

			auto const &cons = state_x0[bx];
			const amrex::Real rho = cons(i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real Eint = HydroSystem<problem_t>::ComputeInternalEnergy(cons, i, j, k, &local_state_fc);

			amrex::Real Tgas = NAN;
			amrex::Real Pgas = NAN;
			amrex::Real cs_iso = NAN;
			if constexpr (HydroSystem<problem_t>::nmscalars_ > 0) {
				amrex::GpuArray<amrex::Real, HydroSystem<problem_t>::nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
				Tgas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);
				Pgas = quokka::EOS<problem_t>::ComputePressure(rho, Eint, massScalars);
				cs_iso = quokka::EOS<problem_t>::ComputeIsothermalSoundSpeed(rho, Pgas);
			} else {
				Tgas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint);
				Pgas = quokka::EOS<problem_t>::ComputePressure(rho, Eint);
				cs_iso = quokka::EOS<problem_t>::ComputeIsothermalSoundSpeed(rho, Pgas);
			}

			const amrex::Real Tuse = amrex::max(Tgas, t_min);
			const amrex::Real kappa = params.conductivity_prefactor ;//* std::pow(Tuse, 2.5);
			const amrex::Real qsat = amrex::max(saturation_factor * flux_limiter_phi * rho * std::pow(cs_iso, 3), small);

			temperature_arr[bx](i, j, k) = Tuse;
			conductivity_arr[bx](i, j, k) = kappa;
			saturated_flux_arr[bx](i, j, k) = qsat;
		});

		std::array<amrex::MultiFab, AMREX_SPACEDIM> heat_flux;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::BoxArray ba_face = amrex::convert(state.boxArray(), amrex::IntVect::TheDimensionVector(idim));
			heat_flux[idim].define(ba_face, state.DistributionMap(), 1, 0);
			heat_flux[idim].setVal(0.0);
		}

		auto const &temp = temperature.const_arrays();
		auto const &kappa = conductivity.const_arrays();
		auto const &qsat = saturated_flux.const_arrays();
		auto flux_x = heat_flux[0].arrays();
		amrex::ParallelFor(heat_flux[0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real gradT = (temp[bx](i , j, k) - temp[bx](i - 1, j, k) ) / dx[0] ;
			const amrex::Real kappa_face = 0.5 * (kappa[bx](i, j, k) + kappa[bx](i - 1, j, k));
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real q_sat_face = 0.5 * (qsat[bx](i, j, k) + qsat[bx](i - 1, j, k));
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_x[bx](i, j, k) =  q_classical   / limiter;
		});

#if AMREX_SPACEDIM >= 2
		auto flux_y = heat_flux[1].arrays();
		amrex::ParallelFor(heat_flux[1], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real gradT = (temp[bx](i, j, k) - temp[bx](i, j - 1, k)) / dx[1];
			const amrex::Real kappa_face = 0.5 * (kappa[bx](i, j, k) + kappa[bx](i, j - 1, k));
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real q_sat_face = 0.5 * (qsat[bx](i, j, k) + qsat[bx](i, j - 1, k));
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_y[bx](i, j, k) =   q_classical  / limiter;
		});
#endif

#if AMREX_SPACEDIM == 3
		auto flux_z = heat_flux[2].arrays();
		amrex::ParallelFor(heat_flux[2], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			const amrex::Real gradT = (temp[bx](i, j, k) - temp[bx](i, j, k - 1)) / dx[2];
			const amrex::Real kappa_face = 0.5 * (kappa[bx](i, j, k) + kappa[bx](i, j, k - 1));
			const amrex::Real q_classical = -kappa_face * gradT;
			const amrex::Real q_sat_face = 0.5 * (qsat[bx](i, j, k) + qsat[bx](i, j, k - 1));
			const amrex::Real limiter = 1.0 + std::abs(q_classical) / amrex::max(q_sat_face, small);
			flux_z[bx](i, j, k) = q_classical  / limiter;
		});
#endif

		auto state_out = state.arrays();
		auto const &flux_x_const = heat_flux[0].const_arrays();
#if AMREX_SPACEDIM >= 2
		auto const &flux_y_const = heat_flux[1].const_arrays();
#endif
#if AMREX_SPACEDIM == 3
		auto const &flux_z_const = heat_flux[2].const_arrays();
#endif

		amrex::ParallelFor(state, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> local_state_fc{};
			if (Physics_Traits<problem_t>::is_mhd_enabled) {
				local_state_fc[0] = state_fc_x0[bx];
#if AMREX_SPACEDIM >= 2
				local_state_fc[1] = state_fc_x1[bx];
#endif
#if AMREX_SPACEDIM == 3
				local_state_fc[2] = state_fc_x2[bx];
#endif
			}

			const amrex::Real rho = state_out[bx](i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real px = state_out[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const amrex::Real py = state_out[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const amrex::Real pz = state_out[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const amrex::Real Egas_old = state_out[bx](i, j, k, HydroSystem<problem_t>::energy_index);

			const amrex::Real Ekin = 0.5 * (px * px + py * py + pz * pz) / rho;
			// const amrex::Real Emag = HydroSystem<problem_t>::ComputeMagneticEnergy(i, j, k, &local_state_fc);
			const amrex::Real Eint_old = HydroSystem<problem_t>::ComputeInternalEnergy(state_out[bx], i, j, k, &local_state_fc);
			amrex::Real div_flux = (flux_x_const[bx](i + 1, j, k) - flux_x_const[bx](i, j, k)) / dx[0];
#if AMREX_SPACEDIM >= 2
			div_flux += (flux_y_const[bx](i, j + 1, k) - flux_y_const[bx](i, j, k)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
			div_flux += (flux_z_const[bx](i, j, k + 1) - flux_z_const[bx](i, j, k)) / dx[2];
#endif

			amrex::Real Eint_new = Eint_old - dt * div_flux;
			// if(i==185 && j==120 && k==122) {
			// 	amrex::Print() << "i, j, k: " << i << " " << j << " " << k << std::endl;
			// 	amrex::Print() << "Eint_old: " << Eint_old << std::endl;
			// 	amrex::Print() << "div_flux: " << div_flux << std::endl;
			// 	amrex::Print() << "Eint_new: " << Eint_new << std::endl;
			// 	amrex::Print() << "flux_x at i: " << flux_x_const[bx](i, j, k) << std::endl;
			// 	amrex::Print() << "flux_y at j: " << flux_y_const[bx](i, j, k) << std::endl;
			// 	amrex::Print() << "flux_z at k: " << flux_z_const[bx](i, j, k) << std::endl;
			// }
			if(Eint_new < 0.0) {
				amrex::Real Tmin = 10; 
				Eint_new = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, Tmin);
				// amrex::Print() << "Warning: Negative internal energy at i, j, k: " << i << " " << j << " " << k << std::endl;
			}
			state_out[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Eint_new + Ekin; // + Emag;
			state_out[bx](i, j, k, HydroSystem<problem_t>::internalEnergy_index) = Eint_new;

		});
	}
};

} // namespace quokka::conduction

#endif // ELECTRON_CONDUCTION_HPP_

