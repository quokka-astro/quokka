//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include <limits>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "RadhydroSimulation.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_radhydro_shell.hpp"

struct ShellProblem {
};

constexpr double a_rad = 7.5646e-15;  // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;   // cm s^-1
constexpr double cs0 = 2.0e5; // (2 km/s) [cm s^-1]
constexpr double chat = 260. * cs0; // cm s^-1
constexpr double k_B = 1.380658e-16;  // erg K^-1
constexpr double m_H = 1.6726231e-24; // mass of hydrogen atom [g]
constexpr double gamma_gas = 1.00001; // approximate isothermal EOS

template <> struct RadSystem_Traits<ShellProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double mean_molecular_mass = 2.2*m_H;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = gamma_gas;
	static constexpr double Erad_floor = 0.;
};

template <> struct EOS_Traits<ShellProblem> {
	static constexpr double gamma = gamma_gas;
};

constexpr amrex::Real Msun = 2.0e33; // g
constexpr amrex::Real parsec_in_cm = 3.086e18; // cm

constexpr amrex::Real specific_luminosity = 2000.; // erg s^-1 g^-1
constexpr amrex::Real GMC_mass = 1.0e6 * Msun; // g
constexpr amrex::Real epsilon = 0.5; // dimensionless
constexpr amrex::Real M_shell = (1 - epsilon)*GMC_mass; // g
constexpr amrex::Real L_star = GMC_mass * specific_luminosity; // erg s^-1

constexpr amrex::Real r_0 = 5.0 * parsec_in_cm; // cm 
constexpr amrex::Real rho_0 = M_shell / ((4./3.) * M_PI * r_0*r_0*r_0); // g cm^-3
constexpr amrex::Real P_0 = rho_0 * cs0*cs0; // erg cm^-3
constexpr amrex::Real kappa0 = 20.0; // specific opacity [cm^2 g^-1]
constexpr amrex::Real Trad_0 = 10.0; // Kelvins
constexpr amrex::Real t0 = r_0 / cs0; // seconds

constexpr double c_v = k_B / ((2.2*m_H) * (gamma_gas - 1.0));

template <>
void RadSystem<ShellProblem>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange,
												 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
												 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
												 amrex::Real /*time*/)
{
	// set point-like radiation source
	amrex::Real const x0 = 0.;
	amrex::Real const y0 = 0.;
	amrex::Real const z0 = 0.;

	amrex::Real sigma = 0.0625 * r_0; // cannot be defined in terms of dx when using AMR!!
	amrex::Real normalisation = (4.0*M_PI/c) * L_star / std::pow(2.0*M_PI*sigma*sigma, 1.5);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) +
										std::pow(z - z0, 2));
		
		radEnergy(i,j,k) = normalisation * std::exp(-r*r/(2.0*sigma*sigma));
	});
}

template <>
auto RadSystem<ShellProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> double
{
	return 0.;
}

template <>
auto RadSystem<ShellProblem>::ComputeRosselandOpacity(const double /*rho*/, const double /*Tgas*/) -> double
{
	return kappa0;
}

template <> void RadhydroSimulation<ShellProblem>::setInitialConditionsAtLevel(int lev)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();

	amrex::Real const x0 = 0.;
	amrex::Real const y0 = 0.;
	amrex::Real const z0 = 0.;

	for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = state_new_[lev].array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
			amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
			amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) +
							std::pow(z - z0, 2));

			double rho = rho_0;
			double vx = 0.;
			double vy = 0.;
			double vz = 0.;
			double P = P_0;

			AMREX_ASSERT(!std::isnan(vx));
			AMREX_ASSERT(!std::isnan(vy));
			AMREX_ASSERT(!std::isnan(vz));
			AMREX_ASSERT(!std::isnan(rho));
			AMREX_ASSERT(!std::isnan(P));

			const auto v_sq = vx * vx + vy * vy + vz * vz;
			const auto gamma = HydroSystem<ShellProblem>::gamma_;

			state(i, j, k, HydroSystem<ShellProblem>::density_index) = rho;
			state(i, j, k, HydroSystem<ShellProblem>::x1Momentum_index) = rho * vx;
			state(i, j, k, HydroSystem<ShellProblem>::x2Momentum_index) = rho * vy;
			state(i, j, k, HydroSystem<ShellProblem>::x3Momentum_index) = rho * vz;
			state(i, j, k, HydroSystem<ShellProblem>::energy_index) =
			    P / (gamma - 1.) + 0.5 * rho * v_sq;
			
			state(i, j, k, RadSystem<ShellProblem>::radEnergy_index) = a_rad * std::pow(Trad_0, 4);
			state(i, j, k, RadSystem<ShellProblem>::x1RadFlux_index) = 0.;
			state(i, j, k, RadSystem<ShellProblem>::x2RadFlux_index) = 0.;
			state(i, j, k, RadSystem<ShellProblem>::x3RadFlux_index) = 0.;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

template <>
void RadhydroSimulation<ShellProblem>::computeAfterLevelAdvance(int lev, amrex::Real /*time*/,
								 amrex::Real /*dt_lev*/, int /*iteration*/, int /*ncycle*/)
{
	amrex::Real const rho_floor = 1.0e-10 * rho_0;
	amrex::Real const P_floor = 1.0e-10 * P_0;

	// enforce density floor to prevent vacuum creation
	for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_[lev].array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const rho = state(i, j, k, RadSystem<ShellProblem>::gasDensity_index);
			amrex::Real const vx1 = state(i, j, k, RadSystem<ShellProblem>::x1GasMomentum_index) / rho;
			amrex::Real const vx2 = state(i, j, k, RadSystem<ShellProblem>::x2GasMomentum_index) / rho;
			amrex::Real const vx3 = state(i, j, k, RadSystem<ShellProblem>::x3GasMomentum_index) / rho;
			amrex::Real const Etot = state(i, j, k, RadSystem<ShellProblem>::gasEnergy_index);

			amrex::Real rho_new = rho;
			if (rho < rho_floor) {
				rho_new = rho_floor;
			}

			// recompute gas energy (to prevent P < 0)
			amrex::Real const Eint = Etot - 0.5 * rho * (vx1*vx1 + vx2*vx2 + vx3*vx3);
			amrex::Real const P = Eint * (gamma_gas - 1.);
			amrex::Real P_new = P;
			if (P < P_floor) {
				P_new = P_floor;
			}
			amrex::Real const Etot_new = P_new / (gamma_gas - 1.) + 
										 0.5 * rho_new * (vx1*vx1 + vx2*vx2 + vx3*vx3);

			state(i, j, k, RadSystem<ShellProblem>::gasDensity_index) = rho_new;
			state(i, j, k, RadSystem<ShellProblem>::gasEnergy_index) = Etot_new;
		});
	}
}

template <>
void RadhydroSimulation<ShellProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags,
						amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2; // gradient refinement threshold
	const amrex::Real P_min = P_0;      // minimum pressure for refinement

	for (amrex::MFIter mfi(state_new_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const P =
			    HydroSystem<ShellProblem>::ComputePressure(state, i, j, k);
			amrex::Real const P_xplus =
			    HydroSystem<ShellProblem>::ComputePressure(state, i + 1, j, k);
			amrex::Real const P_xminus =
			    HydroSystem<ShellProblem>::ComputePressure(state, i - 1, j, k);
			amrex::Real const P_yplus =
			    HydroSystem<ShellProblem>::ComputePressure(state, i, j + 1, k);
			amrex::Real const P_yminus =
			    HydroSystem<ShellProblem>::ComputePressure(state, i, j - 1, k);

			amrex::Real const del_x =
			    std::max(std::abs(P_xplus - P), std::abs(P - P_xminus));
			amrex::Real const del_y =
			    std::max(std::abs(P_yplus - P), std::abs(P - P_yminus));

			amrex::Real const gradient_indicator =
			    std::max(del_x, del_y) / std::max(P, P_min);

			if (gradient_indicator > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// This problem can only be run in 3D
	static_assert(AMREX_SPACEDIM == 3);

	auto isNormalComp = [=](int n, int dim) {
		// it is critical to reflect both the radiation and gas momenta!
		if ((n == RadSystem<ShellProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int nvars = RadhydroSimulation<ShellProblem>::nvarTotal_;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_odd);
				boundaryConditions[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				boundaryConditions[n].setLo(i, amrex::BCType::reflect_even);
				boundaryConditions[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	RadhydroSimulation<ShellProblem> sim(boundaryConditions);
	sim.is_hydro_enabled_ = true;
	sim.is_radiation_enabled_ = true;
	sim.stopTime_ = 0.02 * t0;
	sim.cflNumber_ = 0.2;
	sim.initDt_ = 1.0e9; // seconds
	sim.maxDt_ = 1.0e10; // seconds
	sim.maxTimesteps_ = 5000;
	sim.reconstructionOrder_ = 2; // 1 == donor cell, 2 == PLM
	sim.integratorOrder_ = 2; // RK2
	sim.checkpointInterval_ = 500;
	sim.plotfileInterval_ = 100;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}