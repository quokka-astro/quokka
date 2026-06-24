//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testHydrostaticAtmosphere.cpp
/// \brief Unit test for a hydrostatic exponential atmosphere density floor.
///

#include <cmath>

#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"

struct HydrostaticAtmosphereProblem {
};

template <> struct SimulationData<HydrostaticAtmosphereProblem> {
	amrex::Real atmosphere_scale_height = NAN;
};

template <> struct quokka::EOS_Traits<HydrostaticAtmosphereProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<HydrostaticAtmosphereProblem> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
};

constexpr amrex::Real kTgasInit = 1.0;
constexpr amrex::Real kRhoInitFactor = 5.0e-3;
AMREX_GPU_MANAGED amrex::Real g_base_density_floor = NAN; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real g_scale_height = NAN;	  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<HydrostaticAtmosphereProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
									 int numcomp, amrex::GeometryData const &geom, const amrex::Real /*time*/,
									 const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();

	amrex::Real const *dx = geom.CellSize();
	amrex::Real const *prob_lo = geom.ProbLo();

	amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	amrex::Real const rho_atm = g_base_density_floor * std::exp(-x / g_scale_height);
	amrex::Real const rho_init = kRhoInitFactor * rho_atm;
	amrex::Real const Eint_init = quokka::EOS<HydrostaticAtmosphereProblem>::ComputeEintFromTgas(rho_init, kTgasInit);

	for (int n = 0; n < numcomp; ++n) {
		consVar(i, j, k, n) = 0.;
	}

	consVar(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::density_index) = rho_init;
	consVar(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::energy_index) = Eint_init;
	consVar(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::internalEnergy_index) = Eint_init;
}

template <> void QuokkaSimulation<HydrostaticAtmosphereProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<HydrostaticAtmosphereProblem>::nvarTotal_cc;

	amrex::Real const base_density_floor = densityFloor_;
	amrex::Real const scale_height = userData_.atmosphere_scale_height;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const rho_atm = base_density_floor * std::exp(-x / scale_height);
		amrex::Real const rho_init = kRhoInitFactor * rho_atm;
		amrex::Real const Eint_init = quokka::EOS<HydrostaticAtmosphereProblem>::ComputeEintFromTgas(rho_init, kTgasInit);

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}

		state_cc(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::density_index) = rho_init;
		state_cc(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::energy_index) = Eint_init;
		state_cc(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::internalEnergy_index) = Eint_init;
	});
}

template <>
void QuokkaSimulation<HydrostaticAtmosphereProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
									      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	amrex::Real const base_density_floor = densityFloor_;
	amrex::Real const scale_height = userData_.atmosphere_scale_height;
	const int ncomp_cc = ref.nComp();

	if (useDensityFloorParser_) {
		auto const density_floor_parser = densityFloorParserExe_.value(); // NOLINT(bugprone-unchecked-optional-access)
		for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &state_ref = ref.array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
				amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
#else
				amrex::Real const y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
				amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
#else
				amrex::Real const z = 0.0;
#endif
				amrex::Real const rho_atm = base_density_floor * std::exp(-x / scale_height);
				amrex::Real const rho_floor = density_floor_parser(x, y, z, base_density_floor);
				amrex::Real const rho_init = kRhoInitFactor * rho_atm;
				amrex::Real const Eint_init = quokka::EOS<HydrostaticAtmosphereProblem>::ComputeEintFromTgas(rho_init, kTgasInit);

				for (int n = 0; n < ncomp_cc; ++n) {
					state_ref(i, j, k, n) = 0.;
				}

				state_ref(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::density_index) = rho_floor;
				state_ref(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::energy_index) = Eint_init;
				state_ref(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::internalEnergy_index) = Eint_init;
			});
		}
	} else {
		auto const density_floor_func = [this] AMREX_GPU_HOST_DEVICE(amrex::Real x, amrex::Real y, amrex::Real z,
									     amrex::Real base_floor) -> amrex::Real {
			return densityFloor(x, y, z, base_floor);
		};
		for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &state_ref = ref.array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
				amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
#else
				amrex::Real const y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
				amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
#else
				amrex::Real const z = 0.0;
#endif
				amrex::Real const rho_atm = base_density_floor * std::exp(-x / scale_height);
				amrex::Real const rho_floor = density_floor_func(x, y, z, base_density_floor);
				amrex::Real const rho_init = kRhoInitFactor * rho_atm;
				amrex::Real const Eint_init = quokka::EOS<HydrostaticAtmosphereProblem>::ComputeEintFromTgas(rho_init, kTgasInit);

				for (int n = 0; n < ncomp_cc; ++n) {
					state_ref(i, j, k, n) = 0.;
				}

				state_ref(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::density_index) = rho_floor;
				state_ref(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::energy_index) = Eint_init;
				state_ref(i, j, k, HydroSystem<HydrostaticAtmosphereProblem>::internalEnergy_index) = Eint_init;
			});
		}
	}
}

auto problem_main() -> int
{
	amrex::ParmParse const pp;
	amrex::Real base_density_floor = 0.0;
	if (pp.query("density_floor", base_density_floor) == 0) {
		amrex::Print() << "density_floor must be set for HydrostaticAtmosphere test.\n";
		return 1;
	}
	if (base_density_floor <= 0.0) {
		amrex::Print() << "density_floor must be positive for HydrostaticAtmosphere test.\n";
		return 1;
	}

	amrex::Real scale_height = 0.0;
	if (pp.query("atmosphere_scale_height", scale_height) == 0) {
		amrex::Print() << "atmosphere_scale_height must be set for HydrostaticAtmosphere test.\n";
		return 1;
	}
	if (scale_height <= 0.0) {
		amrex::Print() << "atmosphere_scale_height must be positive for HydrostaticAtmosphere test.\n";
		return 1;
	}

	std::string density_floor_expr;
	pp.query("density_floor_expr", density_floor_expr);
	if (density_floor_expr.empty()) {
		amrex::Print() << "density_floor_expr must be set for HydrostaticAtmosphere test.\n";
		return 1;
	}

	g_base_density_floor = base_density_floor;
	g_scale_height = scale_height;

	QuokkaSimulation<HydrostaticAtmosphereProblem> sim;
	sim.userData_.atmosphere_scale_height = scale_height;
	sim.plotfileInterval_ = -1;

	sim.setInitialConditions();
	sim.FixupState(0);

	amrex::Real const error_norm = sim.computeErrorNorm(false);
	amrex::Real const tol = 1.0e-12;
	int status = 0;
	if (!(error_norm <= tol)) {
		amrex::Print() << "Density floor error norm = " << error_norm << "\n";
		status = 1;
	}

	return status;
}
