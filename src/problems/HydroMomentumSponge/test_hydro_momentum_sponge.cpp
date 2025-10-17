//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_momentum_sponge.cpp
/// \brief Defines a 1D test problem to verify the momentum sponge accuracy.
///

#include <cmath>

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct MomentumSpongeProblem {
};

template <> struct quokka::EOS_Traits<MomentumSpongeProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MomentumSpongeProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

namespace
{
constexpr amrex::Real rho0 = 1.0;
constexpr amrex::Real P0 = 1.0;
constexpr amrex::Real vx0 = 1.0;
} // namespace

template <> void QuokkaSimulation<MomentumSpongeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	amrex::ignore_unused(dx, prob_lo);

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<MomentumSpongeProblem>::nvarTotal_cc;
	const amrex::Real gamma = quokka::EOS_Traits<MomentumSpongeProblem>::gamma;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::ignore_unused(i, j, k);

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		amrex::Real const eint = P0 / (gamma - 1.0);
		amrex::Real const Etot = eint + static_cast<amrex::Real>(0.5) * rho0 * vx0 * vx0;

		state_cc(i, j, k, HydroSystem<MomentumSpongeProblem>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<MomentumSpongeProblem>::x1Momentum_index) = rho0 * vx0;
		state_cc(i, j, k, HydroSystem<MomentumSpongeProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MomentumSpongeProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MomentumSpongeProblem>::energy_index) = Etot;
		state_cc(i, j, k, HydroSystem<MomentumSpongeProblem>::internalEnergy_index) = eint;
	});
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<MomentumSpongeProblem>(quokka::BCType::int_dir);

	QuokkaSimulation<MomentumSpongeProblem> sim(BCs_cc);

	if (!sim.densitySpongeConfig_.enabled) {
		amrex::Abort("Momentum sponge test requires sponge.enable_density_sponge = 1.");
	}

	amrex::Real const tau = sim.densitySpongeConfig_.timescale;
	if (tau <= 0.0) {
		amrex::Abort("Momentum sponge test requires sponge.timescale > 0.");
	}

	sim.cflNumber_ = 0.1;
	sim.stopTime_ = static_cast<amrex::Real>(0.3) * tau;
	sim.maxTimesteps_ = 100000;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	auto [positions, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0.5);
	const int nx = static_cast<int>(positions.size());

	amrex::Real avgVelocity = 0.0;
	for (int i = 0; i < nx; ++i) {
		amrex::Real const rho = values.at(HydroSystem<MomentumSpongeProblem>::density_index)[i];
		amrex::Real const px = values.at(HydroSystem<MomentumSpongeProblem>::x1Momentum_index)[i];
		avgVelocity += px / rho;
	}
	avgVelocity /= static_cast<amrex::Real>(nx);

	amrex::Real const finalTime = sim.tNew_[0];
	amrex::Real const expectedVelocity = vx0 * std::exp(-finalTime / tau);
	amrex::Real const absError = std::abs(avgVelocity - expectedVelocity);
	amrex::Real const relError = absError / std::abs(expectedVelocity);

	amrex::Print() << "Momentum sponge results:\n";
	amrex::Print() << "  final time      = " << finalTime << '\n';
	amrex::Print() << "  numerical vx    = " << avgVelocity << '\n';
	amrex::Print() << "  analytic vx     = " << expectedVelocity << '\n';
	amrex::Print() << "  absolute error  = " << absError << '\n';
	amrex::Print() << "  relative error  = " << relError << '\n';

	int status = 0;
	constexpr amrex::Real relTol = 1.0e-3;
	if (relError > relTol) {
		status = 1;
	}

	return status;
}
