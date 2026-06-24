//==============================================================================
// Copyright 2025 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDResistiveDiffusion.cpp
/// \brief 1D resistive MHD diffusion test with analytic solution.
///
/// Initial condition: b_z = b_amp * sin(k*x), v = 0, uniform rho and pressure.
/// Analytic solution: b_z(x,t) = b_amp * exp(-eta*k^2*t) * sin(k*x).
/// The total energy gains a resistive Poynting flux correction; comparing eint
/// against the analytic prediction checks that AddResistiveEnergyFlux is working.
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"
#include "util/fextract.hpp"

struct MHDResistiveDiffusion {
};

template <> struct quokka::EOS_Traits<MHDResistiveDiffusion> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDResistiveDiffusion> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr ResistivityModel resistivity_model = ResistivityModel::constant;
};

// physical parameters
constexpr double rho_0 = 1.0;
constexpr double pressure_0 = 0.01;
constexpr double b_amp = 0.1;
constexpr double gamma_gas = ::quokka::EOS_Traits<MHDResistiveDiffusion>::gamma;
constexpr double eint_0 = pressure_0 / (gamma_gas - 1.0);

// wavevector: one mode in a unit box
AMREX_GPU_MANAGED double k_mode = 0.0;	// NOLINT
AMREX_GPU_MANAGED double eta_val = 0.0; // NOLINT

template <> void QuokkaSimulation<MHDResistiveDiffusion>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp = Physics_Indices<MHDResistiveDiffusion>::nvarTotal_cc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp; ++n) {
			state(i, j, k, n) = 0.0;
		}

		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double b_z = b_amp * std::sin(k_mode * x);
		const double emag = 0.5 * b_z * b_z;
		const double etot = eint_0 + emag;

		state(i, j, k, HydroSystem<MHDResistiveDiffusion>::density_index) = rho_0;
		state(i, j, k, HydroSystem<MHDResistiveDiffusion>::x1Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<MHDResistiveDiffusion>::x2Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<MHDResistiveDiffusion>::x3Momentum_index) = 0.0;
		state(i, j, k, HydroSystem<MHDResistiveDiffusion>::energy_index) = etot;
		state(i, j, k, HydroSystem<MHDResistiveDiffusion>::internalEnergy_index) = eint_0;
	});
}

template <> void QuokkaSimulation<MHDResistiveDiffusion>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<amrex::Real> &state = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const int ncomp = Physics_Indices<MHDResistiveDiffusion>::nvarPerDim_fc;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp; ++n) {
			state(i, j, k, n) = 0.0;
		}
		if (dir == quokka::direction::z) {
			// z-face: staggered in z, cell-centred in x; x = prob_lo[0] + (i+0.5)*dx[0]
			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			state(i, j, k, MHDSystem<MHDResistiveDiffusion>::bfield_index) = b_amp * std::sin(k_mode * x);
		}
	});
}

template <>
void QuokkaSimulation<MHDResistiveDiffusion>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const double t = tNew_[0];
	const double decay = std::exp(-eta_val * k_mode * k_mode * t);
	const double decay_sq = decay * decay;
	const int ncomp = ref.nComp();

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state_ref = ref.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				state_ref(i, j, k, n) = 0.0;
			}

			const double x = prob_lo[0] + (i + 0.5) * dx[0];
			const double sin_kx = std::sin(k_mode * x);
			const double b_z_ref = b_amp * decay * sin_kx;
			const double emag_ref = 0.5 * b_z_ref * b_z_ref;

			// spatially uniform mean: exact from Joule heating integral <eta*|J|^2>
			const double eint_ref = eint_0 + (b_amp * b_amp / 4.0) * (1.0 - decay_sq);
			const double etot_ref = eint_ref + emag_ref;

			state_ref(i, j, k, HydroSystem<MHDResistiveDiffusion>::density_index) = rho_0;
			state_ref(i, j, k, HydroSystem<MHDResistiveDiffusion>::x1Momentum_index) = 0.0;
			state_ref(i, j, k, HydroSystem<MHDResistiveDiffusion>::x2Momentum_index) = 0.0;
			state_ref(i, j, k, HydroSystem<MHDResistiveDiffusion>::x3Momentum_index) = 0.0;
			state_ref(i, j, k, HydroSystem<MHDResistiveDiffusion>::energy_index) = etot_ref;
			state_ref(i, j, k, HydroSystem<MHDResistiveDiffusion>::internalEnergy_index) = eint_ref;
		});
	}
}

template <>
void QuokkaSimulation<MHDResistiveDiffusion>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
									  quokka::direction const dir)
{
	const double t = tNew_[0];
	const double decay = std::exp(-eta_val * k_mode * k_mode * t);
	const int ncomp = ref.nComp();

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state_ref = ref.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				state_ref(i, j, k, n) = 0.0;
			}
			if (dir == quokka::direction::z) {
				// z-face: cell-centred in x; x = prob_lo[0] + (i+0.5)*dx[0]
				const double x = prob_lo[0] + (i + 0.5) * dx[0];
				state_ref(i, j, k, MHDSystem<MHDResistiveDiffusion>::bfield_index) = b_amp * decay * std::sin(k_mode * x);
			}
		});
	}
}

auto problem_main() -> int
{
	{
		amrex::ParmParse const mhd_pp("mhd");
		mhd_pp.get("resistivity", eta_val);
	}
	k_mode = 2.0 * M_PI; // one mode in a unit-length box

	QuokkaSimulation<MHDResistiveDiffusion> sim;
	sim.setInitialConditions();
	sim.evolve();

	// Check 1: combined error norm validates B_z decay (induction solver + resistivity param)
	constexpr double combined_tol = 0.01;
	const amrex::Real error_norm = sim.computeErrorNorm();
	const bool combined_ok = (error_norm < combined_tol);

	// Check 2: E_tot relative error validates AddResistiveEnergyFlux (use_dual_energy=0
	// means gasInternalEnergy is never updated; E_tot is the discriminating quantity).
	const auto comp_errors = sim.computeComponentErrors();
	amrex::Real etot_rel_err = NAN;
	for (const auto &[name, abs_err, rel_err] : comp_errors) {
		if (name == "gasEnergy") {
			etot_rel_err = rel_err;
			break;
		}
	}
	constexpr double etot_tol = 0.02;
	const bool etot_ok = (!std::isnan(etot_rel_err) && etot_rel_err < etot_tol);

	amrex::Print() << "Combined error norm = " << error_norm << " (tol " << combined_tol << ")\n";
	amrex::Print() << "E_tot relative error = " << etot_rel_err << " (tol " << etot_tol << ")\n";

	int status = 1;
	if (combined_ok && etot_ok) {
		status = 0;
		amrex::Print() << "test passed\n";
	} else {
		amrex::Print() << "test failed\n";
	}
	return status;
}
