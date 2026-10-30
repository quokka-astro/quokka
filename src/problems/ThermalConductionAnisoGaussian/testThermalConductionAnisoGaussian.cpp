//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testThermalConductionAniso.cpp
/// \brief Defines a test problem for anisotropic thermal conduction.
///
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "hydro/hydro_system.hpp"
#include <cmath>

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

/* Gaussian temperature perturbation in x and y, diffusing in a uniform field B = (Bx0, 0, 0).
The conductivity tensor is diagonal in this frame: kappa_par along x and kappa_perp along y (and z), so the x and y
profiles diffuse independently with D_par = kappa_par (gamma - 1) / (n k_B) and D_perp = kappa_perp (gamma - 1) / (n k_B),
n = rho / mu. kappaPar and kappaPerp (erg cm^-1 s^-1 K^-1) are read from the input file (conduction.*).
The profile is uniform in z.
*/
static_assert(AMREX_SPACEDIM == 3, "This problem is only supported in 3D.");

constexpr double Eint0 = 2.505e-8;	     // Gaussian peak (equivalent to T = 2.e8 K)
constexpr double Efloor = 2.505e-11;	     // equivalent to T = 2.e6 K
const double rho0 = 0.1;		     // 1/cm^3
constexpr double Lref = 7.714e+17;	     // quarter box length, fixes region of refinement
constexpr double sigma = 2.410685615625e+17; // width of the initial Gaussian, in cm (amr2-branch value)
constexpr double Bx0 = 1.e-6;		     // uniform field along x, in Quokka's code units (E_mag = B^2 / 2, i.e. B_Gauss / sqrt(4 pi))
struct ThermalConductionAnisoProblem {};

template <> struct quokka::EOS_Traits<ThermalConductionAnisoProblem> {
	static constexpr double gamma = 2.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<ThermalConductionAnisoProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ThermalConductionAnisoProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_mhd_enabled = true;
	static constexpr ConductionModel conduction_model = ConductionModel::constant;
	static constexpr ConductionGeometry conduction_geometry = ConductionGeometry::anisotropic;
};

/// Cell average over [xlow, xhigh] of a 1D Gaussian of variance sigma2 that conserves the integral of exp(-x^2 / (2 sigma^2)),
/// i.e. its peak is sigma / sqrt(sigma2) (=1 at t = 0).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto gaussianCellAverage(amrex::Real xlow, amrex::Real xhigh, amrex::Real sigma2) -> amrex::Real
{
	const amrex::Real erf_low = std::erf(xlow / std::sqrt(2.0 * sigma2));
	const amrex::Real erf_high = std::erf(xhigh / std::sqrt(2.0 * sigma2));
	return (sigma * std::sqrt(M_PI / 2.0)) * (erf_high - erf_low) / (xhigh - xlow);
}

template <> void QuokkaSimulation<ThermalConductionAnisoProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Real rho = rho0 * C::m_p;	    // g/cm^3
	const amrex::Real sigma2_t = sigma * sigma; // t = 0

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real xlow = prob_lo[0] + i * dx[0];
		const amrex::Real xhigh = prob_lo[0] + (i + 1) * dx[0];
		const amrex::Real ylow = prob_lo[1] + j * dx[1];
		const amrex::Real yhigh = prob_lo[1] + (j + 1) * dx[1];
		const amrex::Real Eint = Efloor + Eint0 * gaussianCellAverage(xlow, xhigh, sigma2_t) * gaussianCellAverage(ylow, yhigh, sigma2_t);

		// Magnetic energy of the uniform field B = (Bx0, 0, 0); matches HydroSystem::ComputeMagneticEnergy (0.5 B^2)
		const amrex::Real Emag = 0.5 * Bx0 * Bx0;

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::energy_index) = Eint + Emag;
		state_cc(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::internalEnergy_index) = Eint;
	});
}

template <> void QuokkaSimulation<ThermalConductionAnisoProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	const int ncomp_fc = Physics_Indices<ThermalConductionAnisoProblem>::nvarPerDim_fc;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0.0; // fill unused quantities with zeros
		}
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, MHDSystem<ThermalConductionAnisoProblem>::bfield_index) = Bx0;

		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, MHDSystem<ThermalConductionAnisoProblem>::bfield_index) = 0.0;
		}
	});
}

template <>
void QuokkaSimulation<ThermalConductionAnisoProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real t = tNew_[0];
	const amrex::Real rho = rho0 * C::m_p; // g/cm^3
	// x is along B (kappa_par), y is across it (kappa_perp)
	const amrex::Real n = rho / quokka::EOS_Traits<ThermalConductionAnisoProblem>::mean_molecular_weight;
	const amrex::Real D_par = conductivityParams_.kappa0_par * (quokka::EOS_Traits<ThermalConductionAnisoProblem>::gamma - 1.0) / (n * C::k_B);
	const amrex::Real D_perp = conductivityParams_.kappa0_perp * (quokka::EOS_Traits<ThermalConductionAnisoProblem>::gamma - 1.0) / (n * C::k_B);
	const amrex::Real sigma2_x = sigma * sigma + 2.0 * D_par * t;
	const amrex::Real sigma2_y = sigma * sigma + 2.0 * D_perp * t;
	const amrex::Real Emag = 0.5 * Bx0 * Bx0;

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const xlow = prob_lo[0] + i * dx[0];
			amrex::Real const xhigh = prob_lo[0] + (i + 1) * dx[0];
			amrex::Real const ylow = prob_lo[1] + j * dx[1];
			amrex::Real const yhigh = prob_lo[1] + (j + 1) * dx[1];
			amrex::Real const Eint_exact = Efloor + Eint0 * gaussianCellAverage(xlow, xhigh, sigma2_x) * gaussianCellAverage(ylow, yhigh, sigma2_y);

			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			stateExact(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::energy_index) = Eint_exact + Emag;
			stateExact(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::internalEnergy_index) = Eint_exact;
			stateExact(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::x1Momentum_index) = 0.0;
			stateExact(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::x3Momentum_index) = 0.;
		});
	}
	amrex::Gpu::streamSynchronize();
}

template <>
void QuokkaSimulation<ThermalConductionAnisoProblem>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
										  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
										  quokka::direction const dir)
{
	amrex::ignore_unused(dx, prob_lo);
	// conduction does not evolve B, so the exact solution is the initial uniform field B = (Bx0, 0, 0)
	const amrex::Real B_exact = (dir == quokka::direction::x) ? Bx0 : 0.0;
	const int ncomp_fc = Physics_Indices<ThermalConductionAnisoProblem>::nvarPerDim_fc;

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp_fc; ++n) {
				stateExact(i, j, k, n) = 0.0; // fill unused quantities with zeros
			}
			stateExact(i, j, k, MHDSystem<ThermalConductionAnisoProblem>::bfield_index) = B_exact;
		});
	}
	amrex::Gpu::streamSynchronize();
}

auto runConductionTest() -> double
{
	// 2 D_par t = 0.71 sigma^2 for D_par = 4.396303164750053e28 cm^2/s, i.e. kappaPar = D_par n k_B / (gamma - 1) = 0.6113916490935586e12
	constexpr double max_time = 469054.0075444166;

	// Domain (geometry.*), resolution (amr.n_cell) and refinement (amr.max_level) are read from the input file.

	// Setup boundary conditions
	constexpr int ncomp_cc = Physics_Indices<ThermalConductionAnisoProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
			BCs_cc[n].setLo(dir, amrex::BCType::foextrap);
			BCs_cc[n].setHi(dir, amrex::BCType::foextrap);
		}
	}

	// The field is uniform, B = (Bx0, 0, 0), so the ghost faces must copy it. reflect_odd on the tangential components
	// (a conducting wall) would set Bx -> -Bx in the y/z ghost cells, zeroing the corner-averaged bhat on those walls
	// and suppressing parallel conduction in the boundary planes.
	const int nvars_fc = Physics_Indices<ThermalConductionAnisoProblem>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::foextrap);
			BCs_fc[icomp].setHi(idim, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<ThermalConductionAnisoProblem> sim(BCs_cc, BCs_fc);

	sim.cflNumber_ = 0.3;
	sim.stopTime_ = max_time;

	// set initial conditions
	sim.setInitialConditions();

	sim.evolve();

	// The combined computeErrorNorm() mixes components with very different units (its denominator is dominated by |Bx|),
	// so test the relative L1 error of the internal energy, which is the quantity conduction evolves.
	double eint_rel_err = NAN;
	for (const auto &[name, abs_err, rel_err, ref_norm] : sim.computeComponentErrors()) {
		if (name == "gasInternalEnergy") {
			eint_rel_err = rel_err;
		}
	}
	return eint_rel_err;
}

template <>
void QuokkaSimulation<ThermalConductionAnisoProblem>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
									amrex::MultiFab const &state_cc,
									amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc) const
{
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_cc.const_array(iter);
			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
			    AMREX_D_DECL(state_fc[0].const_array(iter), state_fc[1].const_array(iter), state_fc[2].const_array(iter))};
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<ThermalConductionAnisoProblem>::density_index);
				Real const Eint = HydroSystem<ThermalConductionAnisoProblem>::ComputeInternalEnergy(state, i, j, k, &cons_fc);
				Real const Tgas = quokka::EOS<ThermalConductionAnisoProblem>::ComputeTgasFromEint(rho, Eint);
				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

auto problem_main() -> int
{
	// Single run against a pre-computed reference error norm; the grid is set by the input file.
	double const error_norm = runConductionTest();
	constexpr amrex::Real estimated_error = 4.560056e-04; // estimated from 128^2X8 run
	amrex::Real const delta = std::abs(error_norm - estimated_error) / estimated_error;

	amrex::Print() << std::format("gasInternalEnergy relative L1 error = {:.6e} (expected = {:.6e})\n", error_norm, estimated_error);
	bool const passed = (delta <= 1.e-04 || error_norm < estimated_error);

	if (passed) {
		amrex::Print() << "\n✓ Thermal conduction (anisotropic) test PASSED\n";
		return 0;
	}
	amrex::Print() << "\n✗ Thermal conduction (anisotropic) test FAILED\n";
	return 1;
}
