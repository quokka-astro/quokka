//==============================================================================
// FewModesFT test problem with operator-split stochastic forcing
//==============================================================================

#include <cmath>
#include <map>

#include "AMReX.H"
#include "AMReX_Array4.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "turbulence/FewModesDriver.hpp"
#include "util/BC.hpp"

static_assert(AMREX_SPACEDIM == 3, "FewModesFT forcing test requires 3D");

struct FewModesFTProblem {
};

template <> struct quokka::EOS_Traits<FewModesFTProblem> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0;
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
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	auto const &forcingParams = quokka::turbulence::GetFewModesDriverParameters();
	const auto rho0 = forcingParams.initialDensity;
	constexpr auto eint0 = 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::energy_index) = eint0;
		state_cc(i, j, k, HydroSystem<FewModesFTProblem>::internalEnergy_index) = eint0;
	});
}

template <> auto QuokkaSimulation<FewModesFTProblem>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	std::map<std::string, amrex::Real> stats;

	const amrex::Real cs_iso = quokka::EOS_Traits<FewModesFTProblem>::cs_isothermal;
	const amrex::Real mach_sq_integral =
	    computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const amrex::Real> const &state) noexcept {
		    const amrex::Real rho = state(i, j, k, HydroSystem<FewModesFTProblem>::density_index);
		    if (rho <= 0.0) {
			    return static_cast<amrex::Real>(0.0);
		    }

		    const amrex::Real px = state(i, j, k, HydroSystem<FewModesFTProblem>::x1Momentum_index);
		    const amrex::Real py = state(i, j, k, HydroSystem<FewModesFTProblem>::x2Momentum_index);
		    const amrex::Real pz = state(i, j, k, HydroSystem<FewModesFTProblem>::x3Momentum_index);
		    const amrex::Real inv_rho = 1.0 / rho;
		    const amrex::Real vx = px * inv_rho;
		    const amrex::Real vy = py * inv_rho;
		    const amrex::Real vz = pz * inv_rho;
		    const amrex::Real speed_sq = vx * vx + vy * vy + vz * vz;
		    if (cs_iso <= 0.0) {
			    return static_cast<amrex::Real>(0.0);
		    }

		    return speed_sq / (cs_iso * cs_iso);
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
	quokka::turbulence::ResetFewModesDriver();
	if (!quokka::turbulence::FewModesDriverEnabled()) {
		amrex::Abort("FewModesFT turbulent driving requires a non-zero fewmodesft.force_amplitude.");
	}

	auto BCs_cc = quokka::BC<FewModesFTProblem>(quokka::BCType::int_dir);
	QuokkaSimulation<FewModesFTProblem> sim(BCs_cc);

	sim.setInitialConditions();
	sim.evolve();

	amrex::Print() << "FewModesFT forcing run complete." << '\n';
	return 0;
}
