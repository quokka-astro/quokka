//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave.cpp
/// \brief Defines a test problem for a linear hydro wave.
///

#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "test_hydro_wave_fc.hpp"
#include "util/fextract.hpp"

struct HydroWaveFC {
};

template <> struct quokka::EOS_Traits<HydroWaveFC> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct Physics_Traits<HydroWaveFC> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// constants
constexpr double sound_speed = 1.0;
constexpr double gamma_gas = quokka::EOS_Traits<HydroWaveFC>::gamma;

// background states
constexpr double bg_density = 1.0;
constexpr double bg_pressure = sound_speed * sound_speed * bg_density / gamma_gas;


// k = 2 pi / wave length
// box length = 1, so |k| in [1, inf)
constexpr double num_modes = 1;
constexpr double k_amplitude = 2 * M_PI * num_modes;

// input perturbation: choose to do this via the relative denisty field in [0, 1]. remember, the linear regime is valid when this perturbation is small
constexpr double amp = 1e-6;
const double omega = k_amplitude * sound_speed;


////////////////////////////////////
AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir,
					  double time)
{
	const amrex::Real x1_L = prob_lo[0] + i * dx[0];
	const amrex::Real x2_L = prob_lo[1] + j * dx[1];
	const amrex::Real x3_L = prob_lo[2] + k * dx[2];

	const amrex::Real x1_C = x1_L + static_cast<amrex::Real>(0.5) * dx[0];
	const amrex::Real x2_C = x2_L + static_cast<amrex::Real>(0.5) * dx[1];
	const amrex::Real x3_C = x3_L + static_cast<amrex::Real>(0.5) * dx[2];

	if (cen == quokka::centering::cc) {
		const double cos_wave_C = std::cos(omega * time - k_amplitude * x1_C);

		const double density = bg_density + bg_density * amp * cos_wave_C;
		const double pressure = bg_pressure + bg_pressure * gamma_gas * amp * cos_wave_C;
		const double x1vel = sound_speed * amp * cos_wave_C;
		const double x2vel = 0.0;
		const double x3vel = 0.0;

		const double velocity_magnitude = std::sqrt(std::pow(x1vel, 2) + std::pow(x2vel, 2) + std::pow(x3vel, 2));
		const double momentum = density * velocity_magnitude;
		const double Ekin = 0.5 * std::pow(momentum, 2) / density;
		const double Eint = pressure / (gamma_gas - 1);
		const double Etot = Ekin + Eint;

		state(i, j, k, HydroSystem<HydroWaveFC>::density_index) = density;
		state(i, j, k, HydroSystem<HydroWaveFC>::x1Momentum_index) = x1vel * density;
		state(i, j, k, HydroSystem<HydroWaveFC>::x2Momentum_index) = x2vel * density;
		state(i, j, k, HydroSystem<HydroWaveFC>::x3Momentum_index) = x3vel * density;
		state(i, j, k, HydroSystem<HydroWaveFC>::energy_index) = Etot;
		state(i, j, k, HydroSystem<HydroWaveFC>::internalEnergy_index) = Eint;
	}
}

template <> void QuokkaSimulation<HydroWaveFC>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_cc = Physics_Indices<HydroWaveFC>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_cc, dx, prob_lo, cen, dir, 0);
	});
}

template <> void QuokkaSimulation<HydroWaveFC>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::centering cen = grid_elem.cen_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<HydroWaveFC>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state_fc, dx, prob_lo, cen, dir, 0);
	});
}

template <>
void QuokkaSimulation<HydroWaveFC>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
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
void QuokkaSimulation<HydroWaveFC>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir)
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
	const int nvars_cc = Physics_Indices<HydroWaveFC>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars_cc);
	for (int icomp = 0; icomp < nvars_cc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_cc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_cc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	const int nvars_fc = Physics_Indices<HydroWaveFC>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<HydroWaveFC> sim(BCs_cc, BCs_fc);
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