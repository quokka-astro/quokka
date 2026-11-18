// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testOrionRestart.cpp
/// \brief Restart Quokka from an ORION2 driven-turbulence checkpoint converted by
///        scripts/python/orion2_to_quokka_checkpoint.py.
///
/// The converter rescales the scale-free ORION2 snapshot to physical units (box length,
/// volume-mean density and mass-weighted velocity dispersion given on its command line) and
/// writes a native Quokka checkpoint, recording the target values in metadata.yaml. This
/// problem restarts from that checkpoint, checks that the composite AMR state reproduces the
/// recorded targets and is divergence free, evolves a few steps, and checks mass conservation.
///
/// Cell-centred layout expected from the converter (see Physics_Indices):
///   0 rho, 1-3 momentum, 4 total energy, 5 internal energy, 6 passive scalar (ORION2 tracer1),
///   7 Erad, 8-10 Frad. Face-centred: one B component per direction.

#include <cmath>
#include <format>

#include "AMReX_BCRec.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParallelReduce.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "util/BC.hpp"

struct OrionRestart {}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double T_gas = 10.0;		// K; isothermal gas temperature
constexpr double mu_gas = 2.33;		// mean molecular weight of molecular gas
constexpr double kappa0 = 0.1;		// cm^2 g^-1; constant grey opacity
constexpr double rho_default = 1.0e-20; // g cm^-3; uniform density used only when starting without a checkpoint

template <> struct quokka::EOS_Traits<OrionRestart> {
	static constexpr double gamma = 1.0;
	static constexpr double mean_molecular_weight = mu_gas * C::m_u;
	static constexpr double cs_isothermal = 18890.3; // = sqrt(k_B * T_gas / (mu_gas * m_u)) in cm s^-1
};

template <> struct RadSystem_Traits<OrionRestart> {
	// reduced speed of light (30 km/s): radiation substeps must stay affordable on a pc-scale grid
	static constexpr double c_hat_over_c = 1.0e-4;
	static constexpr double Erad_floor = 0.0;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<OrionRestart> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
	static constexpr bool is_radiation_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr int nDustGroups = 0;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 1; // ORION2 tracer1
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct HydroSystem_Traits<OrionRestart> {
	static constexpr bool reconstruct_eint = false; // isothermal: no internal energy to reconstruct
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<OrionRestart>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<OrionRestart>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

// Only used when no restartfile is given: a uniform, unmagnetised, static state.
template <> void QuokkaSimulation<OrionRestart>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double Erad0 = C::a_rad * std::pow(T_gas, 4);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.0;
		}
		state_cc(i, j, k, HydroSystem<OrionRestart>::density_index) = rho_default;
		state_cc(i, j, k, RadSystem<OrionRestart>::radEnergy_index) = Erad0;
	});
}

template <> void QuokkaSimulation<OrionRestart>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<OrionRestart>::mhdFirstIndex) = 0.0; });
}

// Refine where the cell size exceeds jeans_number times the thermal Jeans length sqrt(pi cs^2 / (G rho)),
// the Truelove criterion used by ORION2 turbulence runs. jeans_number is the runtime parameter orion.jeans_number.
template <> void QuokkaSimulation<OrionRestart>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	double jeans_number = 0.125; // 8 cells per Jeans length
	const amrex::ParmParse pp("orion");
	pp.query("jeans_number", jeans_number);

	const double dx = geom[lev].CellSizeArray()[0];
	constexpr double cs = quokka::EOS_Traits<OrionRestart>::cs_isothermal;
	// dx > J * lambda_J  <=>  rho > pi cs^2 J^2 / (G dx^2)
	const double rho_refine = M_PI * cs * cs * jeans_number * jeans_number / (C::Gconst * dx * dx);

	auto const &state = state_new_cc_[lev].const_arrays();
	auto const &tag = tags.arrays();
	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		if (state[bx](i, j, k, HydroSystem<OrionRestart>::density_index) > rho_refine) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

namespace
{
struct CompositeDiagnostics {
	double mass = 0.0;	   // total gas mass
	double mean_density = 0.0; // mass / box volume
	double vdisp = 0.0;	   // mass-weighted 3D velocity dispersion about the mass-weighted mean velocity
	double max_divb = 0.0;	   // max over levels of dx * max|div B| / max|B|
};

// Composite (non-overlapping) sums over the AMR hierarchy: each level contributes only the cells not covered by a finer level.
auto computeCompositeDiagnostics(QuokkaSimulation<OrionRestart> &sim) -> CompositeDiagnostics
{
	constexpr int rho_idx = HydroSystem<OrionRestart>::density_index;
	constexpr int mx_idx = HydroSystem<OrionRestart>::x1Momentum_index;
	constexpr int my_idx = HydroSystem<OrionRestart>::x2Momentum_index;
	constexpr int mz_idx = HydroSystem<OrionRestart>::x3Momentum_index;
	constexpr int b_idx = Physics_Indices<OrionRestart>::mhdFirstIndex;

	double mass = 0.0;
	double px = 0.0;
	double py = 0.0;
	double pz = 0.0;
	double two_ekin = 0.0;
	double max_divb = 0.0;

	for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
		const amrex::MultiFab &state = sim.state_new_cc_[lev];
		const auto dx = sim.Geom(lev).CellSizeArray();
		const double vol = dx[0] * dx[1] * dx[2];

		// weight = 1 where the cell is not covered by the next finer level, 0 otherwise
		amrex::iMultiFab weight(state.boxArray(), state.DistributionMap(), 1, 0);
		if (lev < sim.finestLevel()) {
			weight = amrex::makeFineMask(state.boxArray(), state.DistributionMap(), sim.boxArray(lev + 1), sim.refRatio(lev), 1, 0);
		} else {
			weight.setVal(1);
		}

		auto const &s = state.const_arrays();
		auto const &w = weight.const_arrays();
		auto sums =
		    amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>{},
				     amrex::TypeList<double, double, double, double, double>{}, state, amrex::IntVect(0),
				     [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) -> amrex::GpuTuple<double, double, double, double, double> {
					     const double wt = static_cast<double>(w[bx](i, j, k));
					     const double rho = s[bx](i, j, k, rho_idx);
					     const double mx = s[bx](i, j, k, mx_idx);
					     const double my = s[bx](i, j, k, my_idx);
					     const double mz = s[bx](i, j, k, mz_idx);
					     return {wt * rho, wt * mx, wt * my, wt * mz, wt * (mx * mx + my * my + mz * mz) / rho};
				     });
		mass += vol * amrex::get<0>(sums);
		px += vol * amrex::get<1>(sums);
		py += vol * amrex::get<2>(sums);
		pz += vol * amrex::get<3>(sums);
		two_ekin += vol * amrex::get<4>(sums);

		// divergence of the face-centred field on this level
		amrex::MultiFab divB(state.boxArray(), state.DistributionMap(), 1, 0);
		const amrex::Array<amrex::MultiFab const *, AMREX_SPACEDIM> bfield{
		    AMREX_D_DECL(&sim.state_new_fc_[lev][0], &sim.state_new_fc_[lev][1], &sim.state_new_fc_[lev][2])};
		amrex::computeDivergence(divB, bfield, sim.Geom(lev));
		double bmax = 0.0;
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			bmax = std::max(bmax, sim.state_new_fc_[lev][idim].norm0(b_idx, 0));
		}
		if (bmax > 0.0) {
			max_divb = std::max(max_divb, divB.norm0(0, 0) * dx[0] / bmax);
		}
	}

	amrex::ParallelDescriptor::ReduceRealSum({mass, px, py, pz, two_ekin});

	const auto prob_lo = sim.Geom(0).ProbLoArray();
	const auto prob_hi = sim.Geom(0).ProbHiArray();
	const double box_volume = (prob_hi[0] - prob_lo[0]) * (prob_hi[1] - prob_lo[1]) * (prob_hi[2] - prob_lo[2]);

	CompositeDiagnostics diag;
	diag.mass = mass;
	diag.mean_density = mass / box_volume;
	diag.vdisp = std::sqrt(std::max(0.0, two_ekin / mass - (px * px + py * py + pz * pz) / (mass * mass)));
	diag.max_divb = max_divb;
	return diag;
}

// relative difference test; returns true on success
auto checkClose(const std::string &name, double value, double expected, double rtol) -> bool
{
	const double rel = std::abs(value - expected) / std::abs(expected);
	const bool ok = rel <= rtol;
	amrex::Print() << std::format("  {:<28s} = {:.10e} (expected {:.10e}, rel. err. {:.3e}, tol {:.1e}) {}\n", name, value, expected, rel, rtol,
				      ok ? "OK" : "FAIL");
	return ok;
}
} // namespace

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<OrionRestart>(quokka::BCType::int_dir);

	constexpr int nvars_fc = Physics_Indices<OrionRestart>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<OrionRestart> sim(BCs_cc, BCs_fc);

	double rtol = 1.0e-10;	    // tolerance for reproducing the converter's targets
	double divb_tol = 1.0e-10;  // tolerance on dx |div B| / max|B|
	double mass_rtol = 1.0e-10; // tolerance on composite mass conservation during evolve
	const amrex::ParmParse pp("orion");
	pp.query("rtol", rtol);
	pp.query("divb_tol", divb_tol);
	pp.query("mass_rtol", mass_rtol);

	sim.setInitialConditions();

	bool success = true;
	const auto diag0 = computeCompositeDiagnostics(sim);
	amrex::Print() << std::format("\nOrionRestart: initial state ({} AMR levels, t = {:.6e} s)\n", sim.finestLevel() + 1, sim.tNew_[0]);

	// targets recorded by the converter (absent when starting without a checkpoint)
	const YAML::Node &meta = sim.simulationMetadata_;
	if (meta["orion2_target_mean_density"] && meta["orion2_target_vdisp"] && meta["orion2_target_length"]) {
		const auto prob_lo = sim.Geom(0).ProbLoArray();
		const auto prob_hi = sim.Geom(0).ProbHiArray();
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			success &=
			    checkClose(std::format("box length [{}]", idim), prob_hi[idim] - prob_lo[idim], meta["orion2_target_length"].as<double>(), rtol);
		}
		success &= checkClose("composite mean density", diag0.mean_density, meta["orion2_target_mean_density"].as<double>(), rtol);
		success &= checkClose("composite mass-weighted vdisp", diag0.vdisp, meta["orion2_target_vdisp"].as<double>(), rtol);
	} else {
		amrex::Print() << "  no ORION2 conversion metadata found: skipping target checks\n";
		amrex::Print() << std::format("  composite mean density = {:.10e}, vdisp = {:.10e}\n", diag0.mean_density, diag0.vdisp);
	}
	const bool divb_ok = diag0.max_divb <= divb_tol;
	amrex::Print() << std::format("  max dx |div B| / max|B|      = {:.3e} (tol {:.1e}) {}\n", diag0.max_divb, divb_tol, divb_ok ? "OK" : "FAIL");
	success &= divb_ok;

	sim.evolve();

	const auto diag1 = computeCompositeDiagnostics(sim);
	amrex::Print() << std::format("\nOrionRestart: final state ({} AMR levels, t = {:.6e} s)\n", sim.finestLevel() + 1, sim.tNew_[0]);
	success &= checkClose("composite mass", diag1.mass, diag0.mass, mass_rtol);
	amrex::Print() << std::format("  composite mass-weighted vdisp = {:.10e} (initial {:.10e})\n", diag1.vdisp, diag0.vdisp);
	const bool divb_ok1 = diag1.max_divb <= divb_tol;
	amrex::Print() << std::format("  max dx |div B| / max|B|      = {:.3e} (tol {:.1e}) {}\n", diag1.max_divb, divb_tol, divb_ok1 ? "OK" : "FAIL");
	success &= divb_ok1;

	amrex::Print() << (success ? "\nOrionRestart: PASSED\n" : "\nOrionRestart: FAILED\n");
	return success ? 0 : 1;
}
