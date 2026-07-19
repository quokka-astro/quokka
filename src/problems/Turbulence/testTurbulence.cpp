#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "turbulence/TurbulentDriving.hpp"
#include "util/BC.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

struct TurbulentBox {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Physics_Traits<TurbulentBox> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr amrex::Real gravitational_constant = 1.0;
};

template <> struct quokka::EOS_Traits<TurbulentBox> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<TurbulentBox> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct SimulationData<TurbulentBox> {
	std::vector<double> t_vec_;
	std::vector<double> Disp3d_vec_;
};

template <> void QuokkaSimulation<TurbulentBox>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<TurbulentBox>::density_index) = 1.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::energy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::scalar0_index) = 1.0;
	});
}

template <> void QuokkaSimulation<TurbulentBox>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2; // gradient refinement threshold
	const amrex::Real rho_min = 0.1;       // minimum density for refinement

	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const int n = HydroSystem<TurbulentBox>::density_index;
		amrex::Real const rho = state[bx](i, j, k, n);
		amrex::Real const rho_xplus = state[bx](i + 1, j, k, n);
		amrex::Real const rho_xminus = state[bx](i - 1, j, k, n);
		amrex::Real const rho_yplus = state[bx](i, j + 1, k, n);
		amrex::Real const rho_yminus = state[bx](i, j - 1, k, n);
		amrex::Real const rho_zplus = state[bx](i, j, k + 1, n);
		amrex::Real const rho_zminus = state[bx](i, j, k - 1, n);

		amrex::Real const del_x = 0.5 * (rho_xplus - rho_xminus);
		amrex::Real const del_y = 0.5 * (rho_yplus - rho_yminus);
		amrex::Real const del_z = 0.5 * (rho_zplus - rho_zminus);

		amrex::Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y + del_z * del_z) / rho;

		if ((gradient_indicator > eta_threshold) && (rho > rho_min)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<TurbulentBox>::computeAfterTimestep()
{
	auto disp = quokka::turbulence::calculate_dispersion<TurbulentBox>(state_new_cc_[0]);
	const amrex::Real disp3d = std::sqrt(disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);
		userData_.Disp3d_vec_.push_back(disp3d);
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<TurbulentBox>(quokka::BCType::int_dir,	 // x: periodic
					       quokka::BCType::int_dir,	 // y: periodic
					       quokka::BCType::int_dir); // z: periodic

	QuokkaSimulation<TurbulentBox> sim(BCs_cc);

	sim.setInitialConditions();

	// Main time loop
	sim.evolve();

	// Check solution validity
	int status = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!sim.userData_.Disp3d_vec_.empty()) {
			const auto disp_last = sim.userData_.Disp3d_vec_.back();
			const double target_vdisp = std::stod(sim.turbParams_["target_vdisp"]);
			const double rel_error = std::abs(target_vdisp - disp_last) / target_vdisp;
			const double err_tol = 0.075;

			amrex::Print() << "\n" << "Target velocity dispersion: " << target_vdisp << "\n";
			amrex::Print() << "Last calculated velocity dispersion: " << disp_last << "\n";
			amrex::Print() << "Relative error: " << rel_error << "\n\n";

			if ((rel_error > err_tol) || std::isnan(rel_error)) {
				status = 1;
			}

#if HAVE_PYTHON
			// Plot dispersion
			const std::vector<double> &time = sim.userData_.t_vec_;
			const std::vector<double> &disp3d = sim.userData_.Disp3d_vec_;

			matplotlibcpp::clf();
			std::map<std::string, std::string> Vdisp_args;
			Vdisp_args["label"] = "Velocity dispersion vs time";
			Vdisp_args["linestyle"] = "-";
			Vdisp_args["color"] = "C1";
			matplotlibcpp::plot(time, disp3d, Vdisp_args);
			matplotlibcpp::xlabel("t (dimensionless)");
			matplotlibcpp::ylabel("vdisp (dimensionless)");
			matplotlibcpp::legend();
			matplotlibcpp::tight_layout();
			matplotlibcpp::save("./Turbulence_vdisp.pdf");
#endif

		} else {
			amrex::Print() << "Error: Dispersion vector is empty!\n";
			status = 1;
		}
	}

	return status;
}
