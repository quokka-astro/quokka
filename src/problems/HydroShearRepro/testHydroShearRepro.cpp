//==============================================================================
// Copyright 2026 Quokka developers.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// Two-dimensional perturbed, oblique shear for nonlinear reconstruction tests.
#include "QuokkaSimulation.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <string>

struct HydroShearRepro {};
template <> struct quokka::EOS_Traits<HydroShearRepro> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};
template <> struct Physics_Traits<HydroShearRepro> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
};
// Scalar-bearing policy fixture keeps the hydrodynamic reproducer unchanged.
struct ShearScalarReconstruction {};
template <> struct Physics_Traits<ShearScalarReconstruction> : Physics_Traits<HydroShearRepro> {
	static constexpr int numPassiveScalars = 2;
};

template <> struct SimulationData<HydroShearRepro> {
	std::ofstream history;
};

namespace shear
{
using Hydro = HydroSystem<HydroShearRepro>;
struct Parameters {
	double amplitude = 0.1;
	double pressure = 0.01;
	double bulkX = 1.;
	double bulkY = 1.;
	double densityAmplitude = 0.;
	double sharpness = 0.;
	double perturbation = 0.;
	int modeX = 1;
	int modeY = 1;
};
Parameters parameters; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

AMREX_GPU_HOST_DEVICE auto initialProfile(double x, double y, Parameters const &p) -> amrex::GpuArray<double, 3>
{
	const double phase = 2. * std::numbers::pi * (p.modeX * x + p.modeY * y);
	const double norm = std::sqrt(static_cast<double>(p.modeX * p.modeX + p.modeY * p.modeY));
	double wave = std::sin(phase);
	if (p.sharpness > 0.) {
		wave = std::tanh(p.sharpness * wave) / std::tanh(p.sharpness);
	}
	const double transversePhase = 2. * std::numbers::pi * (-p.modeY * x + p.modeX * y);
	const double perturbation = p.perturbation * std::sin(transversePhase);
	return {1. + p.densityAmplitude * wave, p.bulkX - p.amplitude * p.modeY / norm * wave + perturbation * p.modeX / norm,
		p.bulkY + p.amplitude * p.modeX / norm * wave + perturbation * p.modeY / norm};
}
} // namespace shear

template <> void QuokkaSimulation<HydroShearRepro>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const auto dx = grid_elem.dx_;
	const auto lo = grid_elem.prob_lo_;
	const auto state = grid_elem.array_;
	const auto p = shear::parameters;
	amrex::ParallelFor(grid_elem.indexRange_, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto q = shear::initialProfile(lo[0] + (i + 0.5) * dx[0], lo[1] + (j + 0.5) * dx[1], p);
		const double eint = p.pressure / (quokka::EOS_Traits<HydroShearRepro>::gamma - 1.);
		state(i, j, k, shear::Hydro::density_index) = q[0];
		state(i, j, k, shear::Hydro::x1Momentum_index) = q[0] * q[1];
		state(i, j, k, shear::Hydro::x2Momentum_index) = q[0] * q[2];
		state(i, j, k, shear::Hydro::x3Momentum_index) = 0.;
		state(i, j, k, shear::Hydro::energy_index) = eint + 0.5 * q[0] * (q[1] * q[1] + q[2] * q[2]);
		state(i, j, k, shear::Hydro::internalEnergy_index) = eint;
	});
}

template <> void QuokkaSimulation<HydroShearRepro>::computeAfterTimestep()
{
	const auto &mf = state_new_cc_[0];
	amrex::MultiFab diagnostics(mf.boxArray(), mf.DistributionMap(), 6, 0);
	const auto result = diagnostics.arrays();
	const auto state = mf.const_arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const double rho = state[bx](i, j, k, shear::Hydro::density_index);
		const double mx = state[bx](i, j, k, shear::Hydro::x1Momentum_index);
		const double my = state[bx](i, j, k, shear::Hydro::x2Momentum_index);
		const double mz = state[bx](i, j, k, shear::Hydro::x3Momentum_index);
		const double energy = state[bx](i, j, k, shear::Hydro::energy_index);
		const double kinetic = 0.5 * (mx * mx + my * my + mz * mz) / rho;
		constexpr double gm1 = quokka::EOS_Traits<HydroShearRepro>::gamma - 1.;
		result[bx](i, j, k, 0) = rho;
		result[bx](i, j, k, 1) = gm1 * state[bx](i, j, k, shear::Hydro::internalEnergy_index);
		result[bx](i, j, k, 2) = energy;
		result[bx](i, j, k, 3) = kinetic;
		result[bx](i, j, k, 4) = std::sqrt(2. * kinetic / rho);
		// Informational only: raw E-KE can lose accuracy at high Mach number.
		result[bx](i, j, k, 5) = gm1 * (energy - kinetic);
	});
	const auto dx = geom[0].CellSizeArray();
	const double volume = dx[0] * dx[1];
	const double minRho = diagnostics.min(0);
	const double minAuxPressure = diagnostics.min(1);
	const double mass = diagnostics.sum(0) * volume;
	const double energy = diagnostics.sum(2) * volume;
	const double kinetic = diagnostics.sum(3) * volume;
	const double maxSpeed = diagnostics.max(4);
	const double minRawPressure = diagnostics.min(5);
	double maxSignal = shear::Hydro::maxSignalSpeedLocal(mf, state_new_fc_[0]);
	amrex::ParallelDescriptor::ReduceRealMax(maxSignal);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!userData_.history.is_open()) {
			const amrex::ParmParse pp("shear");
			std::string filename = "shear_history.txt";
			pp.query("history_file", filename);
			userData_.history.open(filename);
			AMREX_ALWAYS_ASSERT(userData_.history.good());
			userData_.history << "# time min_rho min_aux_pressure mass total_energy kinetic_energy max_speed min_raw_pressure max_signal\n";
			userData_.history << std::setprecision(17);
		}
		userData_.history << tNew_[0] << ' ' << minRho << ' ' << minAuxPressure << ' ' << mass << ' ' << energy << ' ' << kinetic << ' ' << maxSpeed
				  << ' ' << minRawPressure << ' ' << maxSignal << '\n';
		userData_.history.flush();
	}
}

// Exercise the captured failure stencil and an admissible shifted stencil on the
// same execution backend as reconstruction. Signed components must stay unchanged.
void checkPositiveReconstruction()
{
	const amrex::Box box(amrex::IntVect(AMREX_D_DECL(0, 0, 0)), amrex::IntVect(AMREX_D_DECL(4, 1, 0)));
	constexpr int nvars = 8;
	using PositivePolicy = HydroSystem<ShearScalarReconstruction>::PositivePrimitiveReconstruction;
	static_assert(PositivePolicy{}(6) && PositivePolicy{}(7));
	static_assert(!PositivePolicy{}(8)); // Do not classify the following dust components as passive scalars.
	amrex::FArrayBox values(box, nvars);
	amrex::FArrayBox left(box, nvars);
	amrex::FArrayBox right(box, nvars);
	amrex::FArrayBox signedLeft(box, nvars);
	amrex::FArrayBox signedRight(box, nvars);
	const auto q = values.array();
	const amrex::GpuArray<double, 5> stencil{0.02234486969211808, 0.0013623592269931292, 0.00014408067275252544, 0.00007585230405546217,
						 1.2840798703969565};
	amrex::ParallelFor(box, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { q(i, j, k, n) = stencil[i] + 2.0 * j; });
	const quokka::Array4View<amrex::Real const, FluxDir::X1> input(values.const_array());
	const quokka::Array4View<amrex::Real, FluxDir::X1> l(left.array());
	const quokka::Array4View<amrex::Real, FluxDir::X1> r(right.array());
	const quokka::Array4View<amrex::Real, FluxDir::X1> sl(signedLeft.array());
	const quokka::Array4View<amrex::Real, FluxDir::X1> sr(signedRight.array());
	const amrex::Box donors(amrex::IntVect(AMREX_D_DECL(2, 0, 0)), amrex::IntVect(AMREX_D_DECL(2, 1, 0)));
	amrex::ParallelFor(donors, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		HyperbolicSystem<HydroShearRepro>::ReconstructStatesPPM_EP<FluxDir::X1>(input, sl, sr, n, i, j, k);
		HyperbolicSystem<HydroShearRepro>::ReconstructStatesPPM_EP<FluxDir::X1>(input, l, r, n, i, j, k, 0, 0, PositivePolicy{});
		const bool positive = PositivePolicy{}(n);
		if (j == 0) {
			AMREX_ALWAYS_ASSERT(sl(i + 1, j, k, n) < 0.0);
		}
		if (positive && j == 0) {
			AMREX_ALWAYS_ASSERT(l(i + 1, j, k, n) == stencil[3]);
			AMREX_ALWAYS_ASSERT(r(i, j, k, n) == sr(i, j, k, n));
		} else {
			AMREX_ALWAYS_ASSERT(l(i + 1, j, k, n) == sl(i + 1, j, k, n));
			AMREX_ALWAYS_ASSERT(r(i, j, k, n) == sr(i, j, k, n));
		}
	});
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	const amrex::ParmParse pp("shear");
	bool checkReconstruction = false;
	pp.query("check_reconstruction", checkReconstruction);
	if (checkReconstruction) {
		checkPositiveReconstruction();
		return 0;
	}
	pp.query("amplitude", shear::parameters.amplitude);
	pp.query("pressure", shear::parameters.pressure);
	pp.query("bulk_x", shear::parameters.bulkX);
	pp.query("bulk_y", shear::parameters.bulkY);
	pp.query("mode_x", shear::parameters.modeX);
	pp.query("mode_y", shear::parameters.modeY);
	pp.query("density_amplitude", shear::parameters.densityAmplitude);
	pp.query("sharpness", shear::parameters.sharpness);
	pp.query("perturbation", shear::parameters.perturbation);
	AMREX_ALWAYS_ASSERT(shear::parameters.pressure > 0. && std::abs(shear::parameters.densityAmplitude) < 1.);
	AMREX_ALWAYS_ASSERT(shear::parameters.modeX != 0 || shear::parameters.modeY != 0);
	const int ncomp = Physics_Indices<HydroShearRepro>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> bcs(ncomp);
	for (auto &bc : bcs) {
		for (int dim = 0; dim < 2; ++dim) {
			bc.setLo(dim, amrex::BCType::int_dir);
			bc.setHi(dim, amrex::BCType::int_dir);
		}
	}
	QuokkaSimulation<HydroShearRepro> sim(bcs);
	int maxLevel = 0;
	amrex::ParmParse("amr").query("max_level", maxLevel);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(maxLevel == 0, "HydroShearRepro requires amr.max_level=0.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.geom[0].isAllPeriodic() && sim.geom[0].ProbLength(0) == 1. && sim.geom[0].ProbLength(1) == 1.,
					 "HydroShearRepro requires a periodic unit square.");
	sim.setInitialConditions();
	sim.computeAfterTimestep();
	sim.evolve();
	return sim.tNew_[0] >= sim.stopTime_ ? 0 : 1;
}
