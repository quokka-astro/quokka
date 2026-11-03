//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDDiode.cpp
/// \brief Tests the MHD diode (outflow, no-inflow) boundary condition on the x boundaries.
///
/// The field is oblique and divergence-free, with a non-zero normal component at both x boundaries, and the density is
/// non-uniform near the walls, so that a copy-and-flip ghost fill and a mirror ghost fill give different face states.
/// - diode.flow = "wall": converging flow, v_x = -v0 x, so every boundary column is inflow. The mass flux through the
///   boundary must vanish, so the total mass is conserved to round-off.
/// - diode.flow = "mixed": v_x = v0 sin(2 pi y), so half of each boundary is outflow and half is inflow. The total mass
///   must never increase.
/// In both cases, div B must vanish to round-off in every valid and ghost cell after the boundary fill.

#include <cmath>
#include <limits>
#include <string>

#include "AMReX_BC_TYPES.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Reduce.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct MHDDiode {};

template <> struct quokka::EOS_Traits<MHDDiode> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDDiode> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

namespace
{
constexpr double P0 = 1.0;
constexpr double v0 = 0.5;
constexpr double shear = 0.3;
constexpr double Bx0 = 0.4;
constexpr double By0 = 0.3;
constexpr double A0 = 0.1; // amplitude of the vector potential perturbation

bool mixed_flow = false; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// vector potential A_z(x, y) of the field perturbation, evaluated on cell edges so that the face field is exactly divergence-free
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto vectorPotential(double x, double y) -> double
{ // (the x^2 dependence gives both B_x and dB_y/dx non-zero values at the walls)
	return 0.5 * A0 * x * x * std::sin(2.0 * M_PI * y);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto density(double x, double y) -> double { return 1.0 + 0.5 * x * x + 0.2 * std::sin(2.0 * M_PI * y); }
} // namespace

template <> void QuokkaSimulation<MHDDiode>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const int ncomp_cc = Physics_Indices<MHDDiode>::nvarTotal_cc;
	const bool mixed = mixed_flow;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x_L = prob_lo[0] + i * dx[0];
		const double y_L = prob_lo[1] + j * dx[1];
		const double x = x_L + 0.5 * dx[0];
		const double y = y_L + 0.5 * dx[1];
		const double gamma = quokka::EOS_Traits<MHDDiode>::gamma;

		const double rho = density(x, y);
		const double vx = mixed ? v0 * std::sin(2.0 * M_PI * y) : -v0 * x;
		const double vy = mixed ? 0.0 : shear * x;

		// cell-averaged field from the face values set in setInitialConditionsOnGridFaceVars
		const double bx_m = Bx0 + (vectorPotential(x_L, y_L + dx[1]) - vectorPotential(x_L, y_L)) / dx[1];
		const double bx_p = Bx0 + (vectorPotential(x_L + dx[0], y_L + dx[1]) - vectorPotential(x_L + dx[0], y_L)) / dx[1];
		const double by_m = By0 - (vectorPotential(x_L + dx[0], y_L) - vectorPotential(x_L, y_L)) / dx[0];
		const double by_p = By0 - (vectorPotential(x_L + dx[0], y_L + dx[1]) - vectorPotential(x_L, y_L + dx[1])) / dx[0];
		const double bx = 0.5 * (bx_m + bx_p);
		const double by = 0.5 * (by_m + by_p);
		const double Emag = 0.5 * (bx * bx + by * by);

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<MHDDiode>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<MHDDiode>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<MHDDiode>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<MHDDiode>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<MHDDiode>::energy_index) = P0 / (gamma - 1.) + 0.5 * rho * (vx * vx + vy * vy) + Emag;
		state_cc(i, j, k, HydroSystem<MHDDiode>::internalEnergy_index) = P0 / (gamma - 1.);
	});
}

template <> void QuokkaSimulation<MHDDiode>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x_L = prob_lo[0] + i * dx[0];
		const double y_L = prob_lo[1] + j * dx[1];
		double b = 0.;
		if (dir == quokka::direction::x) {
			b = Bx0 + (vectorPotential(x_L, y_L + dx[1]) - vectorPotential(x_L, y_L)) / dx[1];
		} else if (dir == quokka::direction::y) {
			b = By0 - (vectorPotential(x_L + dx[0], y_L) - vectorPotential(x_L, y_L)) / dx[0];
		}
		state_fc(i, j, k, Physics_Indices<MHDDiode>::mhdFirstIndex) = b;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDDiode>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
						     amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
						     int /*orig_comp*/)
{
	setDiodeBCLo<0>(iv, consVar, geom);
	setDiodeBCHi<0>(iv, consVar, geom);
}

template <> auto AMRSimulation<MHDDiode>::isMHDDiodeBoundary(int dir, int /*side*/) -> bool { return dir == 0; }

namespace
{
amrex::Real initial_mass = NAN;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
amrex::Real previous_mass = NAN;				      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
amrex::Real max_mass_err = 0.;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
amrex::Real max_mass_gain = 0.;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
amrex::Real max_divB = 0.;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
amrex::Real min_inflow_mom = std::numeric_limits<amrex::Real>::max(); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace

template <> void QuokkaSimulation<MHDDiode>::computeAfterTimestep()
{
	auto const dx = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	// total mass
	amrex::Real const mass = state_new_cc_[0].sum(HydroSystem<MHDDiode>::density_index) * vol;
	max_mass_err = std::max(max_mass_err, std::abs(mass - initial_mass) / initial_mass);
	max_mass_gain = std::max(max_mass_gain, (mass - previous_mass) / initial_mass);

	// smallest inflow momentum over the boundary columns (positive: every column is inflow)
	auto const domain = geom[0].Domain();
	auto const &st = state_new_cc_[0].const_arrays();
	amrex::Real min_mom = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMin>{}, amrex::TypeList<amrex::Real>{}, state_new_cc_[0], amrex::IntVect(0),
					       [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
						       if (i == domain.smallEnd(0)) {
							       return {st[box](i, j, k, HydroSystem<MHDDiode>::x1Momentum_index)};
						       }
						       if (i == domain.bigEnd(0)) {
							       return {-st[box](i, j, k, HydroSystem<MHDDiode>::x1Momentum_index)};
						       }
						       return {std::numeric_limits<amrex::Real>::max()};
					       });
	amrex::ParallelDescriptor::ReduceRealMin(min_mom);
	min_inflow_mom = std::min(min_inflow_mom, min_mom);
	previous_mass = mass;

	// fill the ghost cells exactly as before a hydro update, then measure div B in all valid and ghost cells
	amrex::MultiFab cc(state_new_cc_[0].boxArray(), state_new_cc_[0].DistributionMap(), state_new_cc_[0].nComp(), nghost_cc_);
	amrex::MultiFab::Copy(cc, state_new_cc_[0], 0, 0, cc.nComp(), 0);
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fc;
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		fc[idim].define(state_new_fc_[0][idim].boxArray(), state_new_fc_[0][idim].DistributionMap(), state_new_fc_[0][idim].nComp(), nghost_fc_);
		amrex::MultiFab::Copy(fc[idim], state_new_fc_[0][idim], 0, 0, fc[idim].nComp(), 0);
	}
	fillBoundaryConditions(cc, cc, 0, tNew_[0], quokka::centering::cc, quokka::direction::na, InterpHookNone, InterpHookNone);
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		fillBoundaryConditions(fc[idim], fc[idim], 0, tNew_[0], quokka::centering::fc, static_cast<quokka::direction>(idim), InterpHookNone,
				       InterpHookNone, FillPatchType::fillpatch_function);
	}
	applyMHDDiodeBC(cc, fc, 0);

	constexpr int bIdx = Physics_Indices<MHDDiode>::mhdFirstIndex;
	auto const &bx = fc[0].const_arrays();
	auto const &by = fc[1].const_arrays();
	auto const &bz = fc[2].const_arrays();
	amrex::Real divB = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpMax>{}, amrex::TypeList<amrex::Real>{}, cc, cc.nGrowVect(),
					    [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
						    amrex::Real const div = (bx[box](i + 1, j, k, bIdx) - bx[box](i, j, k, bIdx)) / dx[0] +
									    (by[box](i, j + 1, k, bIdx) - by[box](i, j, k, bIdx)) / dx[1] +
									    (bz[box](i, j, k + 1, bIdx) - bz[box](i, j, k, bIdx)) / dx[2];
						    return {std::abs(div)};
					    });
	amrex::ParallelDescriptor::ReduceRealMax(divB);
	// normalise by |B| / dx
	max_divB = std::max(max_divB, divB * dx[0] / std::sqrt(Bx0 * Bx0 + By0 * By0));
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("diode");
	std::string flow = "wall";
	pp.query("flow", flow);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(flow == "wall" || flow == "mixed", "diode.flow must be \"wall\" or \"mixed\"");
	mixed_flow = (flow == "mixed");

	// diode in x (cell-centred: ext_dir with setDiodeBC; face-centred: foextrap placeholder, overwritten by applyMHDDiodeBC)
	using BT = quokka::BCType::mathematicalBndryTypes;
	auto BCs_cc = quokka::BC_cc<MHDDiode>(BT::ext_dir, BT::periodic, BT::periodic);
	auto BCs_fc = quokka::BC_fc<MHDDiode>(BT::foextrap, BT::periodic, BT::periodic);

	QuokkaSimulation<MHDDiode> sim(BCs_cc, BCs_fc);
	sim.setInitialConditions();
	{
		auto const dx = sim.Geom(0).CellSizeArray();
		initial_mass = sim.state_new_cc_[0].sum(HydroSystem<MHDDiode>::density_index) * (dx[0] * dx[1] * dx[2]);
		previous_mass = initial_mass;
	}
	sim.evolve();

	amrex::Print() << "\nMHDDiode (" << flow << "): max |dM|/M = " << max_mass_err << ", max mass gain per step = " << max_mass_gain
		       << ", max dx |div B| / |B| (valid + ghost) = " << max_divB << "\n";

	constexpr double divB_tol = 1.0e-12;
	constexpr double mass_tol = 1.0e-12;
	bool pass = (max_divB < divB_tol);
	if (mixed_flow) {
		pass = pass && (max_mass_gain < mass_tol);
	} else {
		// the wall case is only a valid test while every boundary column stays inflow
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(min_inflow_mom > 0., "a boundary column became outflow; reduce stop_time");
		pass = pass && (max_mass_err < mass_tol);
	}
	amrex::Print() << (pass ? "Test passed.\n" : "Test FAILED.\n");
	return pass ? 0 : 1;
}
