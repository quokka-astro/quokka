#include "turbulence/FewModesDriver.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "AMReX.H"
#include "AMReX_ParmParse.H"

#include "util/FewModesFT.hpp"

namespace quokka::turbulence
{

namespace detail
{

struct FewModesDriverState {
	std::unique_ptr<quokka::util::FewModesFT> driver;
	std::unique_ptr<amrex::MultiFab> acceleration;
	std::vector<std::vector<amrex::Real>> modes;
};

struct FewModesDriverContext {
	FewModesDriverParameters params{};
	std::vector<FewModesDriverState> levelStates;
	bool parsed{false};
};

auto context() -> FewModesDriverContext &
{
	static FewModesDriverContext ctx;
	return ctx;
}

void ResetContext()
{
	auto &ctx = context();
	ctx.levelStates.clear();
	ctx.parsed = false;
	ctx.params = FewModesDriverParameters{};
}

void RegisterFinalizeCleanup()
{
	static bool registered = false;
	if (!registered) {
		amrex::ExecOnFinalize([]() { ResetContext(); });
		registered = true;
	}
}

auto Params() -> FewModesDriverParameters const &
{
	auto &ctx = context();
	if (!ctx.parsed) {
		RegisterFinalizeCleanup();

		ctx.params = FewModesDriverParameters{};
		amrex::ParmParse const pp("fewmodesft");
		pp.query("prefix", ctx.params.prefix);
		pp.query("num_modes", ctx.params.numModes);
		pp.query("k_peak", ctx.params.kPeak);
		pp.query("solenoidal_weight", ctx.params.solenoidalWeight);
		pp.query("t_corr", ctx.params.correlationTime);
		pp.query("random_seed", ctx.params.randomSeed);
		pp.query("rho0", ctx.params.initialDensity);
		pp.query("p0", ctx.params.initialPressure);
		pp.query("force_amplitude", ctx.params.forceAmplitude);

		ctx.parsed = true;
	}
	return ctx.params;
}

auto Enabled(FewModesDriverParameters const &params) -> bool { return (params.numModes > 0) && (params.forceAmplitude != 0.0); }

auto StateForLevel(int lev, amrex::MultiFab &state_mf, amrex::Geometry const &geom) -> FewModesDriverState &
{
	auto &ctx = context();
	const auto requiredSize = static_cast<std::size_t>(lev + 1);
	if (ctx.levelStates.size() < requiredSize) {
		ctx.levelStates.resize(requiredSize);
	}

	FewModesDriverState &state = ctx.levelStates.at(static_cast<std::size_t>(lev));
	const auto &ba = state_mf.boxArray();
	const auto &dm = state_mf.DistributionMap();
	const bool needs_allocation = (!state.acceleration) || (state.acceleration->boxArray() != ba) || (state.acceleration->DistributionMap() != dm);

	if (needs_allocation) {
		auto const &params = ctx.params;
		const auto level_seed = static_cast<uint32_t>(params.randomSeed + lev);
		state.modes = quokka::util::MakeRandomModes(params.numModes, params.kPeak, level_seed);
		state.driver = std::make_unique<quokka::util::FewModesFT>(params.prefix, params.numModes, state.modes, params.kPeak, params.solenoidalWeight,
									  params.correlationTime, level_seed);
		state.driver->SetPhases(geom);
		state.acceleration = std::make_unique<amrex::MultiFab>(ba, dm, AMREX_SPACEDIM, 0);
		state.acceleration->setVal(0.0);
	}

	return state;
}

auto Acceleration(FewModesDriverState &state) -> amrex::MultiFab &
{
	AMREX_ALWAYS_ASSERT(state.acceleration != nullptr);
	return *state.acceleration;
}

void GenerateAcceleration(FewModesDriverState &state, amrex::Real dt)
{
	AMREX_ALWAYS_ASSERT(state.driver != nullptr);
	AMREX_ALWAYS_ASSERT(state.acceleration != nullptr);

	auto &ctx = context();
	state.driver->Generate(*state.acceleration, dt);
	state.acceleration->mult(ctx.params.forceAmplitude, 0, AMREX_SPACEDIM);
}

} // namespace detail

auto GetFewModesDriverParameters() -> FewModesDriverParameters const & { return detail::Params(); }

void ResetFewModesDriver() { detail::ResetContext(); }

auto FewModesDriverEnabled() -> bool
{
	auto const &params = detail::Params();
	return detail::Enabled(params);
}

} // namespace quokka::turbulence
