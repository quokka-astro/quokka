#include "AMReX_BLassert.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_ParmParse.H"
#include "particles/particle_types.hpp"
#include "QuokkaSimulation.hpp"
#include "dust/dust_system.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "turbulence/TurbulentDriving.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct TurbulentBox {
}; // dummy type to allow compile-type polymorphism via template specialization
template <> struct Particle_Traits<TurbulentBox> {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Sink;
};
// ============================================================
// Physics traits
// ============================================================

template <> struct Physics_Traits<TurbulentBox> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;

	// Keep MHD closed
	static constexpr bool is_mhd_enabled = false;

	// Keep self-gravity opened as in your original file
	static constexpr bool is_self_gravity_enabled = true;

	// Turn on dust
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 3;

	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr int nGroups = 1;

	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr amrex::Real gravitational_constant = C::Gconst;
};

// ============================================================
// EOS traits
// ============================================================

template <> struct quokka::EOS_Traits<TurbulentBox> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double cs_isothermal = 1.9e4; // cm/s
	static constexpr double mean_molecular_weight = 2.33 * C::m_u;
};

template <> struct HydroSystem_Traits<TurbulentBox> {
	static constexpr bool reconstruct_eint = true;
};

// ============================================================
// Simulation data
// ============================================================

template <> struct SimulationData<TurbulentBox> {
	std::vector<double> t_vec_;
	std::vector<double> Disp3d_vec_;

	// gas
	std::vector<double> gas_vx_vec_;

	// dust group 0
	std::vector<double> dust0_vx_vec_;
	std::vector<double> dust0_to_gas_vec_;

	// dust group 1
	std::vector<double> dust1_vx_vec_;
	std::vector<double> dust1_to_gas_vec_;

	// dust group 2
	std::vector<double> dust2_vx_vec_;
	std::vector<double> dust2_to_gas_vec_;
};

// ============================================================
// Problem parameters
// ============================================================

namespace ProblemParams
{
// Initial gas state
constexpr amrex::Real nH0 = 50.0;  // cm^-3
constexpr amrex::Real T0 = 6000.0; // K

// Dust-to-gas mass ratios
constexpr amrex::Real eps_d0 = 0.002;
constexpr amrex::Real eps_d1 = 0.003;
constexpr amrex::Real eps_d2 = 0.005;

// Grain material density and grain radii
constexpr amrex::Real rho_gr = 3.0; // g cm^-3
constexpr amrex::Real a0 = 1.0e-6;  // cm, 0.01 micron
constexpr amrex::Real a1 = 1.0e-5;  // cm, 0.10 micron
constexpr amrex::Real a2 = 3.0e-5;  // cm, 0.30 micron
} // namespace ProblemParams

// ============================================================
// Derived variables
//
// Current QuokkaSimulation.hpp expects the newer interface:
//
// ComputeDerivedVar(
//     int lev,
//     std::string const &dname,
//     amrex::MultiFab &mf,
//     const int ncomp,
//     amrex::MultiFab const &state_cc,
//     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &state_fc
// ) const
// ============================================================

template <>
void QuokkaSimulation<TurbulentBox>::ComputeDerivedVar(
    int /*lev*/, std::string const &dname, amrex::MultiFab &mf,
    const int ncomp_cc_in, amrex::MultiFab const &state_cc,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto const &state = state_cc.const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const rho =
			    state[bx](i, j, k, HydroSystem<TurbulentBox>::density_index);

			amrex::Real const Eint =
			    state[bx](i, j, k, HydroSystem<TurbulentBox>::internalEnergy_index);

			output[bx](i, j, k, ncomp) =
			    quokka::EOS<TurbulentBox>::ComputeTgasFromEint(rho, Eint);
		});
	}
}

// ============================================================
// Dust stopping time
//
// Important change:
// old version: DustDrag<TurbulentBox>::ComputeReciprocalStoppingTime
// new version: DustSources<TurbulentBox>::ComputeReciprocalStoppingTime
//
// Use Quokka official Kwok/Epstein-like helper.
// ============================================================

template <>
AMREX_GPU_HOST_DEVICE auto
DustSources<TurbulentBox>::ComputeReciprocalStoppingTime(
    amrex::Real rho_g,
    amrex::GpuArray<amrex::Real, Physics_Traits<TurbulentBox>::nDustGroups> rho_d,
    amrex::GpuArray<amrex::Real, Physics_Traits<TurbulentBox>::nDustGroups> rel_vel_mag,
    double cs)
    -> amrex::GpuArray<amrex::Real, Physics_Traits<TurbulentBox>::nDustGroups>
{
	constexpr int nDust = Physics_Traits<TurbulentBox>::nDustGroups;
	static_assert(nDust == 3, "This problem is set up for exactly 3 dust groups.");

	const amrex::GpuArray<amrex::Real, nDust> grain_radius = {
	    ProblemParams::a0,
	    ProblemParams::a1,
	    ProblemParams::a2
	};

	const amrex::GpuArray<amrex::Real, nDust> grain_density = {
	    ProblemParams::rho_gr,
	    ProblemParams::rho_gr,
	    ProblemParams::rho_gr
	};

	return DustSources<TurbulentBox>::ComputeReciprocalStoppingTimeKwok(
	    rho_g,
	    rho_d,
	    rel_vel_mag,
	    cs,
	    grain_radius,
	    grain_density,
	    true);
}

// ============================================================
// Dust charge-to-mass ratio
//
// MHD is turned off in this file, so charged-dust Lorentz coupling
// should be disabled. Return zero charge-to-mass ratio for all groups.
// ============================================================

template <>
AMREX_GPU_HOST_DEVICE auto
DustSources<TurbulentBox>::ComputeDustChargeToMassRatio()
    -> amrex::GpuArray<amrex::Real, Physics_Traits<TurbulentBox>::nDustGroups>
{
	constexpr int nDust = Physics_Traits<TurbulentBox>::nDustGroups;
	amrex::GpuArray<amrex::Real, nDust> xi{};

	for (int g = 0; g < nDust; ++g) {
		xi[g] = 0.0;
	}

	return xi;
}

// ============================================================
// Initial conditions
// ============================================================

template <>
void QuokkaSimulation<TurbulentBox>::setInitialConditionsOnGrid(
    quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	constexpr amrex::Real gamma = quokka::EOS_Traits<TurbulentBox>::gamma;
	constexpr amrex::Real mu = quokka::EOS_Traits<TurbulentBox>::mean_molecular_weight;

	// gas density
	const amrex::Real rho0 = ProblemParams::nH0 * C::m_p;

	// gas velocity
	const amrex::Real vx0 = 0.0;
	const amrex::Real vy0 = 0.0;
	const amrex::Real vz0 = 0.0;

	// pressure and energy density
	const amrex::Real P0 = rho0 * C::k_B * ProblemParams::T0 / mu;
	const amrex::Real Eint0 = P0 / (gamma - 1.0);
	const amrex::Real Ekin0 =
	    0.5 * rho0 * (vx0 * vx0 + vy0 * vy0 + vz0 * vz0);

	// MHD is off, so no magnetic energy is included.
	const amrex::Real Etot0 = Eint0 + Ekin0;

	// dust densities
	const amrex::Real rho_d0 = ProblemParams::eps_d0 * rho0;
	const amrex::Real rho_d1 = ProblemParams::eps_d1 * rho0;
	const amrex::Real rho_d2 = ProblemParams::eps_d2 * rho0;

	constexpr int ncomp_cc = Physics_Indices<TurbulentBox>::nvarTotal_cc;
	constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Initialize all cell-centered variables to zero first.
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		// gas
		state_cc(i, j, k, HydroSystem<TurbulentBox>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x1Momentum_index) = rho0 * vx0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x2Momentum_index) = rho0 * vy0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x3Momentum_index) = rho0 * vz0;

		state_cc(i, j, k, HydroSystem<TurbulentBox>::energy_index) = Etot0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::internalEnergy_index) = Eint0;

		// passive scalar
		state_cc(i, j, k, HydroSystem<TurbulentBox>::scalar0_index) = 1.0;

		if constexpr (Physics_Traits<TurbulentBox>::is_dust_enabled) {
			// dust group 0
			state_cc(i, j, k, HydroSystem<TurbulentBox>::dustDensity_index) =
			    rho_d0;
			state_cc(i, j, k, HydroSystem<TurbulentBox>::x1DustMomentum_index) =
			    rho_d0 * vx0;
			state_cc(i, j, k, HydroSystem<TurbulentBox>::x2DustMomentum_index) =
			    rho_d0 * vy0;
			state_cc(i, j, k, HydroSystem<TurbulentBox>::x3DustMomentum_index) =
			    rho_d0 * vz0;

			// dust group 1
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::dustDensity_index + numDustVars) =
			    rho_d1;
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::x1DustMomentum_index + numDustVars) =
			    rho_d1 * vx0;
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::x2DustMomentum_index + numDustVars) =
			    rho_d1 * vy0;
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::x3DustMomentum_index + numDustVars) =
			    rho_d1 * vz0;

			// dust group 2
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::dustDensity_index + 2 * numDustVars) =
			    rho_d2;
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::x1DustMomentum_index + 2 * numDustVars) =
			    rho_d2 * vx0;
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::x2DustMomentum_index + 2 * numDustVars) =
			    rho_d2 * vy0;
			state_cc(i, j, k,
			         HydroSystem<TurbulentBox>::x3DustMomentum_index + 2 * numDustVars) =
			    rho_d2 * vz0;
		}
	});
}

// ============================================================
// AMR refinement
// ============================================================

// ============================================================
// AMR refinement
// Pure Truelove / Jeans-length criterion only.
// No additional density gate.
// ============================================================

template <>
void QuokkaSimulation<TurbulentBox>::refineGrid(
    int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// AMR strategy:
	//
	// Use only the Truelove / Jeans-length criterion on all levels:
	//
	//     lambda_J < N_J * dx
	//
	// where
	//
	//     lambda_J^2 = pi * c_s^2 / (G * rho)
	//
	// If this condition is satisfied, the local Jeans length is not
	// resolved by N_J cells on the current level, so the cell is tagged
	// for refinement.
	//
	// No extra density threshold is used.

	constexpr amrex::Real gamma = quokka::EOS_Traits<TurbulentBox>::gamma;
	constexpr amrex::Real mu =
	    quokka::EOS_Traits<TurbulentBox>::mean_molecular_weight; // g per particle

	// Truelove number. N_J = 4 is the classical minimum.
	// You can increase this to 8, 16, or 32 if you want more conservative refinement.
	constexpr amrex::Real N_J = 4.0;

	// Avoid division by zero or invalid density.
	constexpr amrex::Real small_rho = 1.0e-100;

	// Read temperature floor from inputs; default = 10 K.
	// This avoids using a temperature below your physical/numerical floor
	// when estimating the Jeans length.
	amrex::Real temperature_floor = 10.0;
	{
		amrex::ParmParse pp;
		pp.query("temperature_floor", temperature_floor);
	}

	// Current level cell size.
	const auto dx = Geom(lev).CellSizeArray();
	const amrex::Real dx_min = amrex::min(dx[0], amrex::min(dx[1], dx[2]));

	// Jeans length must be resolved by at least N_J cells.
	const amrex::Real jeans_limit_sq = (N_J * dx_min) * (N_J * dx_min);

	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(state_new_cc_[lev],
	                   [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		                   const amrex::Real rho =
		                       state[bx](i, j, k,
		                                 HydroSystem<TurbulentBox>::density_index);

		                   const amrex::Real Eint =
		                       state[bx](i, j, k,
		                                 HydroSystem<TurbulentBox>::internalEnergy_index);

		                   if (rho <= small_rho) {
			                   return;
		                   }

		                   // Temperature from EOS, consistent with ComputeDerivedVar("temperature").
		                   amrex::Real Tgas =
		                       quokka::EOS<TurbulentBox>::ComputeTgasFromEint(rho, Eint);

		                   // Do not allow the Jeans estimate to use temperature below the floor.
		                   Tgas = amrex::max(Tgas, temperature_floor);

		                   // Thermal sound speed squared:
		                   //
		                   //     c_s^2 = gamma * k_B * T / mu
		                   //
		                   const amrex::Real cs2 = gamma * C::k_B * Tgas / mu;

		                   // Jeans length squared:
		                   //
		                   //     lambda_J^2 = pi * c_s^2 / (G * rho)
		                   //
		                   const amrex::Real lambdaJ_sq =
		                       M_PI * cs2 / (C::Gconst * rho);

		                   const bool jeans_unresolved =
		                       (lambdaJ_sq < jeans_limit_sq);

		                   // Pure Truelove criterion on every level.
		                   // No density gate.
		                   if (jeans_unresolved) {
			                   tag[bx](i, j, k) = amrex::TagBox::SET;
		                   }
	                   });

	amrex::Gpu::streamSynchronize();
}
// ============================================================
// Diagnostics after timestep
// ============================================================

template <>
void QuokkaSimulation<TurbulentBox>::computeAfterTimestep()
{
	auto disp =
	    quokka::turbulence::calculate_dispersion<TurbulentBox>(state_new_cc_[0]);
	const amrex::Real disp3d =
	    std::sqrt(disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]);

	auto [_, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr double rho_floor_diag = 1.0e-40;

		userData_.t_vec_.push_back(tNew_[0]);
		userData_.Disp3d_vec_.push_back(disp3d);

		// gas velocity
		const double rho_g_raw =
		    values.at(HydroSystem<TurbulentBox>::density_index)[0];
		const double rho_g = std::max(rho_g_raw, rho_floor_diag);

		const double mom_gx =
		    values.at(HydroSystem<TurbulentBox>::x1Momentum_index)[0];
		const double vx_g = mom_gx / rho_g;
		userData_.gas_vx_vec_.push_back(vx_g);

		if constexpr (Physics_Traits<TurbulentBox>::is_dust_enabled) {
			constexpr int numDustVars = Physics_NumVars::numDustVarsPerGroup;

			// group 0
			{
				const double rho_d_raw =
				    values.at(HydroSystem<TurbulentBox>::dustDensity_index)[0];
				const double rho_d = std::max(rho_d_raw, rho_floor_diag);

				const double mom_dx =
				    values.at(HydroSystem<TurbulentBox>::x1DustMomentum_index)[0];

				const double vx_d = mom_dx / rho_d;
				userData_.dust0_vx_vec_.push_back(vx_d);
				userData_.dust0_to_gas_vec_.push_back(rho_d_raw / rho_g);
			}

			// group 1
			{
				const double rho_d_raw =
				    values.at(HydroSystem<TurbulentBox>::dustDensity_index +
				              numDustVars)[0];
				const double rho_d = std::max(rho_d_raw, rho_floor_diag);

				const double mom_dx =
				    values.at(HydroSystem<TurbulentBox>::x1DustMomentum_index +
				              numDustVars)[0];

				const double vx_d = mom_dx / rho_d;
				userData_.dust1_vx_vec_.push_back(vx_d);
				userData_.dust1_to_gas_vec_.push_back(rho_d_raw / rho_g);
			}

			// group 2
			{
				const double rho_d_raw =
				    values.at(HydroSystem<TurbulentBox>::dustDensity_index +
				              2 * numDustVars)[0];
				const double rho_d = std::max(rho_d_raw, rho_floor_diag);

				const double mom_dx =
				    values.at(HydroSystem<TurbulentBox>::x1DustMomentum_index +
				              2 * numDustVars)[0];

				const double vx_d = mom_dx / rho_d;
				userData_.dust2_vx_vec_.push_back(vx_d);
				userData_.dust2_to_gas_vec_.push_back(rho_d_raw / rho_g);
			}
		}
	}
}

// ============================================================
// Main
// ============================================================

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<TurbulentBox>(
	    quokka::BCType::int_dir,  // x periodic
	    quokka::BCType::int_dir,  // y periodic
	    quokka::BCType::int_dir); // z periodic

	// MHD is off, so only cell-centered boundary conditions are needed.
	QuokkaSimulation<TurbulentBox> sim(BCs_cc);

	sim.setInitialConditions();

	// main evolution
	sim.evolve();

	// check solution validity
	int status = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!sim.userData_.Disp3d_vec_.empty()) {
			const auto disp_last = sim.userData_.Disp3d_vec_.back();
			const double target_vdisp = std::stod(sim.turbParams_["target_vdisp"]);
			const double rel_error =
			    std::abs(target_vdisp - disp_last) / target_vdisp;
			const double err_tol = 0.075;

			amrex::Print() << "\n"
			               << "Target velocity dispersion: " << target_vdisp
			               << "\n";
			amrex::Print() << "Last calculated velocity dispersion: " << disp_last
			               << "\n";
			amrex::Print() << "Relative error: " << rel_error << "\n\n";

			if ((rel_error > err_tol) || std::isnan(rel_error)) {
				status = 1;
			}

#if HAVE_PYTHON
			// -------------------------
			// Plot gas velocity dispersion
			// -------------------------
			{
				const std::vector<double> &time = sim.userData_.t_vec_;
				const std::vector<double> &disp3d = sim.userData_.Disp3d_vec_;

				matplotlibcpp::clf();
				std::map<std::string, std::string> Vdisp_args;
				Vdisp_args["label"] = "Velocity dispersion vs time";
				Vdisp_args["linestyle"] = "-";
				Vdisp_args["color"] = "C1";
				matplotlibcpp::plot(time, disp3d, Vdisp_args);
				matplotlibcpp::xlabel("t");
				matplotlibcpp::ylabel("vdisp");
				matplotlibcpp::legend();
				matplotlibcpp::tight_layout();
				matplotlibcpp::save("./Turbulence_vdisp.pdf");
			}

			// -------------------------
			// Plot gas/dust x-velocity
			// -------------------------
			if constexpr (Physics_Traits<TurbulentBox>::is_dust_enabled) {
				const std::vector<double> &time = sim.userData_.t_vec_;

				matplotlibcpp::clf();
				matplotlibcpp::plot(
				    time, sim.userData_.gas_vx_vec_,
				    {{"label", "gas vx"}, {"color", "k"}, {"linestyle", "-"}});

				matplotlibcpp::plot(
				    time, sim.userData_.dust0_vx_vec_,
				    {{"label", "dust group 0 vx"},
				     {"color", "r"},
				     {"linestyle", "-"}});

				matplotlibcpp::plot(
				    time, sim.userData_.dust1_vx_vec_,
				    {{"label", "dust group 1 vx"},
				     {"color", "b"},
				     {"linestyle", "-"}});

				matplotlibcpp::plot(
				    time, sim.userData_.dust2_vx_vec_,
				    {{"label", "dust group 2 vx"},
				     {"color", "g"},
				     {"linestyle", "-"}});

				matplotlibcpp::xlabel("t");
				matplotlibcpp::ylabel("vx");
				matplotlibcpp::legend();
				matplotlibcpp::tight_layout();
				matplotlibcpp::save("./Turbulence_dust_vx.pdf");

				// dust-to-gas ratio
				matplotlibcpp::clf();
				matplotlibcpp::plot(
				    time, sim.userData_.dust0_to_gas_vec_,
				    {{"label", "dust0/gas"}, {"color", "r"}, {"linestyle", "-"}});

				matplotlibcpp::plot(
				    time, sim.userData_.dust1_to_gas_vec_,
				    {{"label", "dust1/gas"}, {"color", "b"}, {"linestyle", "-"}});

				matplotlibcpp::plot(
				    time, sim.userData_.dust2_to_gas_vec_,
				    {{"label", "dust2/gas"}, {"color", "g"}, {"linestyle", "-"}});

				matplotlibcpp::xlabel("t");
				matplotlibcpp::ylabel("rho_d / rho_g");
				matplotlibcpp::legend();
				matplotlibcpp::tight_layout();
				matplotlibcpp::save("./Turbulence_dust_to_gas.pdf");
			}
#endif
		} else {
			amrex::Print() << "Error: Dispersion vector is empty!\n";
			status = 1;
		}
	}

	return status;
}
