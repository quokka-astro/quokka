//==============================================================================
// FewModesFT test problem with operator-split stochastic forcing
//==============================================================================

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"
#include "util/FewModesFT.hpp"

static_assert(AMREX_SPACEDIM == 3, "FewModesFT forcing test requires 3D");

namespace
{

struct ForcingParameters {
	std::string prefix{"few_modes"};
	int numModes{10};
	amrex::Real kPeak{2.0};
	amrex::Real solenoidalWeight{0.5};
	amrex::Real correlationTime{1.0};
	int randomSeed{12345};
	amrex::Real initialDensity{1.0};
	amrex::Real initialPressure{1.0};
	amrex::Real forceAmplitude{1.0};
};

struct ForcingState {
	std::unique_ptr<quokka::util::FewModesFT> driver;
	std::unique_ptr<amrex::MultiFab> acceleration;
	std::vector<std::vector<amrex::Real>> modes;
	bool phasesInitialized{false};
};

struct ForcingContext {
	ForcingParameters params;
	std::vector<ForcingState> levelForcing;
	bool parametersParsed{false};
};

auto forcing_context() -> ForcingContext &
{
	static ForcingContext context;
	return context;
}

void clear_forcing_state();

void ensure_finalize_cleanup_registered()
{
	static bool registered = false;
	if (!registered) {
		amrex::ExecOnFinalize([] { clear_forcing_state(); });
		registered = true;
	}
}

void parse_parameters()
{
	auto &context = forcing_context();
	if (context.parametersParsed) {
		return;
	}
	ensure_finalize_cleanup_registered();
	const amrex::ParmParse pp("fewmodesft");
	auto &forcingParams = context.params;
	pp.query("prefix", forcingParams.prefix);
	pp.query("num_modes", forcingParams.numModes);
	pp.query("k_peak", forcingParams.kPeak);
	pp.query("solenoidal_weight", forcingParams.solenoidalWeight);
	pp.query("t_corr", forcingParams.correlationTime);
	pp.query("random_seed", forcingParams.randomSeed);
	pp.query("rho0", forcingParams.initialDensity);
	pp.query("p0", forcingParams.initialPressure);
	pp.query("force_amplitude", forcingParams.forceAmplitude);
	context.parametersParsed = true;
}

auto ensure_forcing_state(int lev, amrex::MultiFab &state_mf, amrex::Geometry const &geom) -> ForcingState &
{
	auto &levelForcing = forcing_context().levelForcing;
	const auto requiredSize = static_cast<std::size_t>(lev) + 1;
	if (requiredSize > levelForcing.size()) {
		levelForcing.resize(requiredSize);
	}

	ForcingState &forcing = levelForcing[static_cast<std::size_t>(lev)];
	const auto &ba = state_mf.boxArray();
	const auto &dm = state_mf.DistributionMap();
	const bool needs_allocation = !forcing.acceleration || forcing.acceleration->boxArray() != ba || forcing.acceleration->DistributionMap() != dm;

	if (needs_allocation) {
		auto const &forcingParams = forcing_context().params;
		const auto level_seed = static_cast<uint32_t>(forcingParams.randomSeed + lev);
		forcing.modes = quokka::util::MakeRandomModes(forcingParams.numModes, forcingParams.kPeak, level_seed);
		forcing.driver = std::make_unique<quokka::util::FewModesFT>(forcingParams.prefix, forcingParams.numModes, forcing.modes, forcingParams.kPeak,
									    forcingParams.solenoidalWeight, forcingParams.correlationTime, level_seed);
		forcing.driver->SetPhases(geom);
		forcing.phasesInitialized = true;
		forcing.acceleration = std::make_unique<amrex::MultiFab>(ba, dm, AMREX_SPACEDIM, 0);
		forcing.acceleration->setVal(0.0);
	}

	return forcing;
}

void clear_forcing_state() { forcing_context().levelForcing.clear(); }

} // namespace

struct FewModesFTProblem {
};

template <> struct quokka::EOS_Traits<FewModesFTProblem> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<FewModesFTProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<FewModesFTProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<FewModesFTProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	parse_parameters();

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto &forcingParams = forcing_context().params;
	const auto rho0 = forcingParams.initialDensity;
	const auto P0 = forcingParams.initialPressure;
	const auto gamma = quokka::EOS_Traits<FewModesFTProblem>::gamma;
	const auto eint0 = P0 / (gamma - 1.0);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::energy_index) = eint0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::internalEnergy_index) = eint0;
	});
}

template <> void QuokkaSimulation<FewModesFTProblem>::computeAfterLevelAdvance(int lev, amrex::Real /*time*/, amrex::Real dt_lev, int /*ncycle*/)
{
	parse_parameters();
	auto &state_mf = state_new_cc_[lev];
	auto const &geom_lev = this->Geom(lev);
	ForcingState &forcing = ensure_forcing_state(lev, state_mf, geom_lev);

	AMREX_ALWAYS_ASSERT(forcing.driver != nullptr);
	AMREX_ALWAYS_ASSERT(forcing.acceleration != nullptr);
	AMREX_ALWAYS_ASSERT(forcing.phasesInitialized);

	forcing.driver->Generate(*forcing.acceleration, dt_lev);

	const amrex::Real amplitude = forcing_context().params.forceAmplitude;
	forcing.acceleration->mult(amplitude, 0, AMREX_SPACEDIM);

	const auto accel = forcing.acceleration->const_arrays();
	const auto state = state_mf.arrays();

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = state[bx](i, j, k, HydroSystem<FewModesFTProblem>::density_index);
		amrex::Real px = state[bx](i, j, k, HydroSystem<FewModesFTProblem>::x1Momentum_index);
		amrex::Real py = state[bx](i, j, k, HydroSystem<FewModesFTProblem>::x2Momentum_index);
		amrex::Real pz = state[bx](i, j, k, HydroSystem<FewModesFTProblem>::x3Momentum_index);
		const amrex::Real KE0 = (px * px + py * py + pz * pz) / (2.0 * rho);

		const amrex::Real ax = accel[bx](i, j, k, 0);
		const amrex::Real ay = accel[bx](i, j, k, 1);
		const amrex::Real az = accel[bx](i, j, k, 2);

		px += dt_lev * rho * ax;
		py += dt_lev * rho * ay;
		pz += dt_lev * rho * az;

		const amrex::Real KE1 = (px * px + py * py + pz * pz) / (2.0 * rho);
		const amrex::Real dKE = KE1 - KE0;

		state[bx](i, j, k, HydroSystem<FewModesFTProblem>::x1Momentum_index) = px;
		state[bx](i, j, k, HydroSystem<FewModesFTProblem>::x2Momentum_index) = py;
		state[bx](i, j, k, HydroSystem<FewModesFTProblem>::x3Momentum_index) = pz;
		state[bx](i, j, k, HydroSystem<FewModesFTProblem>::energy_index) += dKE;
	});
	amrex::Gpu::streamSynchronize();
}

template <> auto QuokkaSimulation<FewModesFTProblem>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	const amrex::Real mach_sq_integral =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<FewModesFTProblem>::density_index);
		    if (rho <= 0.0) {
			    return amrex::Real(0.0);
		    }

		    const amrex::Real px = state(i, j, k, HydroSystem<FewModesFTProblem>::x1Momentum_index);
		    const amrex::Real py = state(i, j, k, HydroSystem<FewModesFTProblem>::x2Momentum_index);
		    const amrex::Real pz = state(i, j, k, HydroSystem<FewModesFTProblem>::x3Momentum_index);
		    const amrex::Real inv_rho = 1.0 / rho;
		    const amrex::Real vx = px * inv_rho;
		    const amrex::Real vy = py * inv_rho;
		    const amrex::Real vz = pz * inv_rho;
		    const amrex::Real speed_sq = vx * vx + vy * vy + vz * vz;

		    const amrex::Real cs = HydroSystem<FewModesFTProblem>::ComputeSoundSpeed(state, i, j, k);
		    if (cs <= 0.0) {
			    return amrex::Real(0.0);
		    }

		    return speed_sq / (cs * cs);
	    });

	const amrex::Geometry &geom0 = this->Geom(0);
	const amrex::Real *prob_lo = geom0.ProbLo();
	const amrex::Real *prob_hi = geom0.ProbHi();
	const amrex::Real total_volume = (prob_hi[0] - prob_lo[0]) * (prob_hi[1] - prob_lo[1]) * (prob_hi[2] - prob_lo[2]);

	amrex::Real mach_rms = 0.0;
	if (total_volume > 0.0) {
		mach_rms = std::sqrt(mach_sq_integral / total_volume);
	}

	stats["mach_rms"] = mach_rms;
	return stats;
}

auto problem_main() -> int
{
	parse_parameters();
	clear_forcing_state();

	auto BCs_cc = quokka::BC<FewModesFTProblem>(quokka::BCType::int_dir);
	QuokkaSimulation<FewModesFTProblem> sim(BCs_cc);

	sim.setInitialConditions();
	sim.evolve();

	amrex::Print() << "FewModesFT forcing run complete." << '\n';
	return 0;
}
