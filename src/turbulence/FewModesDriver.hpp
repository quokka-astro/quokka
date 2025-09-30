#ifndef QUOKKA_TURBULENCE_FEWMODESDRIVER_HPP_
#define QUOKKA_TURBULENCE_FEWMODESDRIVER_HPP_

#include <string>

#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "hydro/hydro_system.hpp"

namespace quokka::turbulence
{

struct FewModesDriverParameters {
	std::string prefix{"few_modes"};
	int numModes{10};
	amrex::Real kPeak{2.0};
	amrex::Real solenoidalWeight{0.5};
	amrex::Real correlationTime{1.0};
	int randomSeed{12345};
	amrex::Real initialDensity{1.0};
	amrex::Real initialPressure{1.0};
	amrex::Real forceAmplitude{0.0};
};

[[nodiscard]] auto GetFewModesDriverParameters() -> FewModesDriverParameters const &;
void ResetFewModesDriver();
[[nodiscard]] auto FewModesDriverEnabled() -> bool;

namespace detail
{
struct FewModesDriverState;

[[nodiscard]] auto Params() -> FewModesDriverParameters const &;
[[nodiscard]] auto Enabled(FewModesDriverParameters const &params) -> bool;
[[nodiscard]] auto StateForLevel(int lev, amrex::MultiFab &state_mf, amrex::Geometry const &geom) -> FewModesDriverState &;
[[nodiscard]] auto Acceleration(FewModesDriverState &state) -> amrex::MultiFab &;
void GenerateAcceleration(FewModesDriverState &state, amrex::Real dt);
void ResetContext();
} // namespace detail

template <typename problem_t> void ApplyFewModesDriver(int lev, amrex::MultiFab &state_mf, amrex::Geometry const &geom, amrex::Real dt)
{
	static_assert(Physics_Traits<problem_t>::is_hydro_enabled, "FewModes driver requires hydro to be enabled");

	if (dt <= 0.0) {
		return;
	}

	auto const &params = detail::Params();
	if (!detail::Enabled(params)) {
		return;
	}
	auto &state = detail::StateForLevel(lev, state_mf, geom);
	detail::GenerateAcceleration(state, dt);
	auto const accel = detail::Acceleration(state).const_arrays();
	auto const state_arrays = state_mf.arrays();

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = state_arrays[bx](i, j, k, HydroSystem<problem_t>::density_index);
		amrex::Real px = state_arrays[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		amrex::Real py = state_arrays[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		amrex::Real pz = state_arrays[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const amrex::Real KE0 = (px * px + py * py + pz * pz) / (2.0 * rho);

		const amrex::Real ax = accel[bx](i, j, k, 0);
		const amrex::Real ay = accel[bx](i, j, k, 1);
		const amrex::Real az = accel[bx](i, j, k, 2);

		px += dt * rho * ax;
		py += dt * rho * ay;
		pz += dt * rho * az;

		const amrex::Real KE1 = (px * px + py * py + pz * pz) / (2.0 * rho);
		const amrex::Real dKE = KE1 - KE0;

		state_arrays[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index) = px;
		state_arrays[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index) = py;
		state_arrays[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index) = pz;
		state_arrays[bx](i, j, k, HydroSystem<problem_t>::energy_index) += dKE;
	});
	amrex::Gpu::streamSynchronize();
}

} // namespace quokka::turbulence

#endif // QUOKKA_TURBULENCE_FEWMODESDRIVER_HPP_
