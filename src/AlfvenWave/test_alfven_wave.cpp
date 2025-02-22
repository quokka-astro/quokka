//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_fc_quantities.cpp
/// \brief Defines a test problem to make sure face-centred quantities are created correctly.
///

#include <cassert>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "test_alfven_wave.hpp"

struct AlfvenWave {
};

template <> struct quokka::EOS_Traits<AlfvenWave> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<AlfvenWave> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups // TODO(Neco): shold this be zero?
};

// constants
constexpr double sound_speed = 1.0;
constexpr double gamma = 5. / 3.;

// we have set up the problem so that:
// the direction of wave propogation, vec(k), is aligned with the x1-direction
// the background magnetic field sits in the x1-x2 plane

// k = 2 pi / wave_length. note: wave_length should be an integer, because of periodic BCs + the requirement that the magnetic field be continuous. also, the box length = 1, so |k| in [1, inf)
constexpr double num_modes = 2;
constexpr double k_amplitude = 2 * M_PI * num_modes;

// background states
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma;
constexpr double bg_mag_amplitude = 1.0;

// input perturbation: remember, the linear regime is only valid when this perturbation is small
constexpr double delta_b = 1e-6;

// alignment between the background magnetic field and the direction of wave propogation (in the x1-x2 plane). recall that hat(k) = (1, 0, 0) and hat(delta_u) = (0, 1, 0)
constexpr double theta_degrees = 0.0; // degrees
const double cos_theta = std::cos(theta_degrees * M_PI / 180.0);
const double sin_theta = std::sin(theta_degrees * M_PI / 180.0);

// magnetic field properties
const double alfven_speed = bg_mag_amplitude / std::sqrt(bg_density);
const double bg_mag_x1 = bg_mag_amplitude * cos_theta;
const double bg_mag_x2 = bg_mag_amplitude * sin_theta;

const double omega = std::sqrt(std::pow(alfven_speed,2) * std::pow(k_amplitude,2) * std::pow(cos_theta,2));

AMREX_GPU_DEVICE double computeMagneticVectorPotential_x(double x1, double x2, double x3, double time) {
  return 0.0;
}

AMREX_GPU_DEVICE double computeMagneticVectorPotential_y(double x1, double x2, double x3, double time) {
  return -bg_mag_amplitude * delta_b / k_amplitude * std::sin(omega * time - k_amplitude * x1);
}

AMREX_GPU_DEVICE double computeMagneticVectorPotential_z(double x1, double x2, double x3, double time) {
  return bg_mag_amplitude * x2;
}

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, double time)
{
  const amrex::Real x1_L = prob_lo[0] + i * dx[0];
  const amrex::Real x2_L = prob_lo[1] + j * dx[1];
  const amrex::Real x3_L = prob_lo[2] + k * dx[2];

  const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];
  const amrex::Real x2_C = x2_L + static_cast<amrex::Real>(0.5) * dx[1];
  const amrex::Real x3_C = x3_L + static_cast<amrex::Real>(0.5) * dx[2];
  
  if (cen == quokka::centering::cc) {
    const double cos_wave_C = std::cos(omega * time - k_amplitude * x1_C);

    const double density = bg_density;
    const double pressure = bg_pressure;
    const double x1vel = 0;
    const double x2vel = 0;
    const double x3vel = -omega * delta_b / (sound_speed * k_amplitude * cos_theta) * cos_wave_C;

    // magnetic field at the center of the cell
    const double x1mag = bg_mag_x1;
    const double x2mag = bg_mag_x2;
    const double x3mag = bg_mag_amplitude * delta_b * cos_wave_C;
  
    const double velocity_magnitude = std::sqrt(std::pow(x1vel, 2) + std::pow(x2vel, 2) + std::pow(x3vel, 2));
    const double momentum = density * velocity_magnitude;
    const double Ekin = 0.5 * std::pow(momentum, 2) / density;
    const double Emag = 0.5 * (x1mag * x1mag + x2mag * x2mag + x3mag * x3mag);
    const double Eint = pressure / (gamma - 1);
    const double Etot = Ekin + Emag + Eint;

    state(i, j, k, HydroSystem<AlfvenWave>::density_index) = density;
    state(i, j, k, HydroSystem<AlfvenWave>::x1Momentum_index) = x1vel * density;
    state(i, j, k, HydroSystem<AlfvenWave>::x2Momentum_index) = x2vel * density;
    state(i, j, k, HydroSystem<AlfvenWave>::x3Momentum_index) = x3vel * density;
    state(i, j, k, HydroSystem<AlfvenWave>::energy_index) = Etot;
    state(i, j, k, HydroSystem<AlfvenWave>::internalEnergy_index) = Eint;
  } else if (cen == quokka::centering::fc) {
    // // method 1: magnetic field at the center of the cell-face
    // const double x1mag = bg_mag_x1;
    // const double x2mag = bg_mag_x2;
    // const double x3mag = bg_mag_amplitude * delta_b * std::cos(omega * time - k_amplitude * x1_L);
    // method 2: average magnetic field across the cell-face
    // const double x1mag = bg_mag_x1;
    // const double x2mag = bg_mag_x2;
    // const double x3mag = bg_mag_amplitude * delta_b / (k_amplitude * dx[0]) * (
    //   std::sin(omega * time - k_amplitude * x1_L) - std::sin(omega * time - k_amplitude * (x1_L + dx[0]))
    // );
    // method 3: vector potential
    const double x1mag = 
      (computeMagneticVectorPotential_z(x1_L, x2_L+dx[1], x3_L+dx[2]/2, time) - computeMagneticVectorPotential_z(x1_L, x2_L, x3_L+dx[2]/2, time)) / dx[1] - 
      (computeMagneticVectorPotential_y(x1_L, x2_L+dx[1]/2, x3_L+dx[2], time) - computeMagneticVectorPotential_y(x1_L, x2_L+dx[1]/2, x3_L, time)) / dx[2];
    const double x2mag =
      (computeMagneticVectorPotential_x(x1_L+dx[0]/2, x2_L, x3_L+dx[2], time) - computeMagneticVectorPotential_x(x1_L+dx[0]/2, x2_L, x3_L, time)) / dx[2] -
      (computeMagneticVectorPotential_z(x1_L+dx[0], x2_L, x3_L+dx[2]/2, time) - computeMagneticVectorPotential_z(x1_L, x2_L, x3_L+dx[2]/2, time)) / dx[0];
    const double x3mag =
      (computeMagneticVectorPotential_y(x1_L+dx[0], x2_L+dx[1]/2, x3_L, time) - computeMagneticVectorPotential_y(x1_L, x2_L+dx[1]/2, x3_L, time)) / dx[0] -
      (computeMagneticVectorPotential_x(x1_L+dx[0]/2, x2_L+dx[2], x3_L, time) - computeMagneticVectorPotential_x(x1_L+dx[2]/2, x2_L, x3_L, time)) / dx[1];
    if      (dir == quokka::direction::x) {state(i, j, k, MHDSystem<AlfvenWave>::bfield_index) = x1mag;}
    else if (dir == quokka::direction::y) {state(i, j, k, MHDSystem<AlfvenWave>::bfield_index) = x2mag;}
    else if (dir == quokka::direction::z) {state(i, j, k, MHDSystem<AlfvenWave>::bfield_index) = x3mag;}
  }
}

template <> void RadhydroSimulation<AlfvenWave>::setInitialConditionsOnGrid_cc(quokka::grid grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
  const quokka::centering cen = grid_elem.cen_;
  const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<AlfvenWave>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void RadhydroSimulation<AlfvenWave>::setInitialConditionsOnGrid_fc(quokka::grid grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
  const quokka::centering cen = grid_elem.cen_;
  const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<AlfvenWave>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void RadhydroSimulation<AlfvenWave>::computeReferenceSolution_cc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::cc, quokka::direction::na, 0);
		});
	}
}

template <>
void RadhydroSimulation<AlfvenWave>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			computeWaveSolution(i, j, k, stateExact, dx, prob_lo, quokka::centering::fc, dir, 0);
		});
	}
}

auto problem_main() -> int
{
	const int nvars_cc = Physics_Indices<AlfvenWave>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars_cc);
	for (int icomp = 0; icomp < nvars_cc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_cc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_cc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	const int nvars_fc = Physics_Indices<AlfvenWave>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	RadhydroSimulation<AlfvenWave> sim(BCs_cc, BCs_fc);
  sim.computeReferenceSolution_ = true;
	sim.setInitialConditions();
	sim.evolve();

  // Compute test success condition
	int status = 0;
	const double error_tol = 0.002;
	if (sim.errorNorm_ > error_tol) {
		status = 1;
	}

	return status;
}
