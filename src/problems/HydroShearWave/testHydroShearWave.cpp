//==============================================================================
// Copyright 2026 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydroShearWave.cpp
/// \brief A single-mode transverse shear flow, decaying under physical shear viscosity.
///

#include "hydro/hydro_system.hpp"
#include <format>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"

struct ShearWaveProblem {
};

template <> struct quokka::EOS_Traits<ShearWaveProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ShearWaveProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr ViscosityModel viscosity_model = ViscosityModel::constant; // shear defaults to 0; no-op unless set
};

constexpr double rho0 = 1.0;   // background density
constexpr double P0 = 1.0;     // background pressure
constexpr double amp = 1.0e-6; // velocity perturbation amplitude

// shear decay rate: shear_viscosity*k^2/rho0, for the single hardcoded mode k=2*pi below; zero (no
// decay) unless hydro.shear_viscosity is set
AMREX_GPU_MANAGED double shear_decay_rate = 0.0; // NOLINT
// which velocity component varies (0=x1,1=x2,2=x3), and which spatial coordinate it varies along;
// must differ from each other, else div(v) != 0 and this is no longer a pure-shear flow
AMREX_GPU_MANAGED int shear_flow_axis = 0; // NOLINT
AMREX_GPU_MANAGED int shear_grad_axis = 1; // NOLINT

AMREX_GPU_DEVICE void computeShearSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real time)
{
	amrex::GpuArray<int, 3> const cell_idx{i, j, k};
	const amrex::Real pos_C = prob_lo[shear_grad_axis] + (cell_idx[shear_grad_axis] + static_cast<amrex::Real>(0.5)) * dx[shear_grad_axis];
	const amrex::Real v = amp * std::sin(2.0 * M_PI * pos_C) * std::exp(-shear_decay_rate * time);

	const double Eint = P0 / (quokka::EOS_Traits<ShearWaveProblem>::gamma - 1.0);
	const double Ekin = 0.5 * rho0 * v * v;

	state(i, j, k, HydroSystem<ShearWaveProblem>::density_index) = rho0;
	state(i, j, k, HydroSystem<ShearWaveProblem>::x1Momentum_index) = (shear_flow_axis == 0) ? rho0 * v : 0.0;
	state(i, j, k, HydroSystem<ShearWaveProblem>::x2Momentum_index) = (shear_flow_axis == 1) ? rho0 * v : 0.0;
	state(i, j, k, HydroSystem<ShearWaveProblem>::x3Momentum_index) = (shear_flow_axis == 2) ? rho0 * v : 0.0;
	state(i, j, k, HydroSystem<ShearWaveProblem>::energy_index) = Eint + Ekin;
	state(i, j, k, HydroSystem<ShearWaveProblem>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<ShearWaveProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<ShearWaveProblem>::nvarTotal_cc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused components with zeros
		}
		computeShearSolution(i, j, k, state_cc, dx, prob_lo, 0.0);
	});
}

// Sets shear_decay_rate from hydro.shear_viscosity, and flow_axis/grad_axis from
// setup.shear_flow_axis/setup.shear_grad_axis. Zero decay when hydro.shear_viscosity is absent,
// recovering the undamped shear flow.
void configureShearViscousParameters()
{
	{
		amrex::ParmParse const pp("setup");
		pp.query("shear_flow_axis", shear_flow_axis);
		pp.query("shear_grad_axis", shear_grad_axis);
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(shear_flow_axis >= 0 && shear_flow_axis < AMREX_SPACEDIM, "setup.shear_flow_axis must be a valid spatial axis.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(shear_grad_axis >= 0 && shear_grad_axis < AMREX_SPACEDIM, "setup.shear_grad_axis must be a valid spatial axis.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(shear_flow_axis != shear_grad_axis,
					 "setup.shear_flow_axis must differ from setup.shear_grad_axis (else div(v) != 0).");

	double shearViscosity = 0.0;
	amrex::ParmParse const hpp("hydro");
	hpp.query("shear_viscosity", shearViscosity);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(shearViscosity >= 0.0, "hydro.shear_viscosity must be non-negative.");
	constexpr double k_magn = 2.0 * M_PI; // single hardcoded mode, box length 1
	shear_decay_rate = shearViscosity * k_magn * k_magn / rho0;
	if (shearViscosity > 0.0) {
		amrex::Print() << "Hydro shear wave (viscous): flow_axis=" << shear_flow_axis << " grad_axis=" << shear_grad_axis
			       << " decay_rate=" << shear_decay_rate << "\n";
	}
}

// fills every cell of mf with the analytic shear solution at the given time
void fillShearSolutionState(amrex::MultiFab &mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real time)
{
	const int ncomp = mf.nComp();
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				state(i, j, k, n) = 0; // fill unused components with zeros
			}
			computeShearSolution(i, j, k, state, dx, prob_lo, time);
		});
	}
}

auto problem_main() -> int
{
	configureShearViscousParameters();

	double error_tol = 1.0e-8;
	{
		amrex::ParmParse const pp("setup");
		pp.query("error_tol", error_tol);
	}

	const int ncomp_cc = Physics_Indices<ShearWaveProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<ShearWaveProblem> sim(BCs_cc);
	// idealized test in non-physical CGS units: disable the default 5 K temperature floor
	sim.tempFloor_ = 0.0;
	sim.setInitialConditions();
	sim.evolve();

	// extract along grad_axis, at the midplane of the other two -- any slice there gives the same profile
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], shear_grad_axis, 0.5);
	const int ny_final = static_cast<int>(position.size());

	// analytic solution at the final simulation time
	amrex::MultiFab exactState(sim.boxArray(0), sim.DistributionMap(0), QuokkaSimulation<ShearWaveProblem>::nvars_, 0);
	fillShearSolutionState(exactState, sim.geom[0].CellSizeArray(), sim.geom[0].ProbLoArray(), sim.tNew_[0]);
	auto [pos_exact, val_exact] = fextract(exactState, sim.geom[0], shear_grad_axis, 0.5);

	amrex::Real err_sq = 0.;
	for (int n = 0; n < QuokkaSimulation<ShearWaveProblem>::nvars_; ++n) {
		if (n == HydroSystem<ShearWaveProblem>::internalEnergy_index) {
			continue;
		}
		amrex::Real dU_k = 0.;
		for (int i = 0; i < ny_final; ++i) {
			const amrex::Real U_k0 = val_exact.at(n)[i];
			const amrex::Real U_k1 = values.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(ny_final);
		}
		err_sq += dU_k * dU_k;
	}
	amrex::Real epsilon = std::sqrt(err_sq);
	// fextract only gathers the full comparison to the IO processor; broadcast so every rank agrees
	amrex::ParallelDescriptor::Bcast(&epsilon, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	amrex::Print() << std::format("\nrun_sim error norm = {:.6e}  (tol = {:.6e})\n", static_cast<double>(epsilon), error_tol);

	int status = 0;
	if (epsilon > error_tol) {
		status = 1;
	}
	return status;
}
