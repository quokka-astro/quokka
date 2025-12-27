//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDensityFloorUnit.cpp
/// \brief Unit test for spatially varying density floors.
///

#include "AMReX_Geometry.H"
#include "AMReX_Math.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Parser.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"

#include "eos.H"
#include "extern_parameters.H"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"

struct DensityFloorUnitProblem {
};

template <> struct quokka::EOS_Traits<DensityFloorUnitProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<DensityFloorUnitProblem> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr int nDustGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

auto problem_main() -> int
{
	init_extern_parameters();
	amrex::Real small_temp = 1.0e-10;
	amrex::Real small_dens = 1.0e-100;
	eos_init(small_temp, small_dens);

	amrex::ParmParse pp;
	amrex::Real base_density_floor = 0.0;
	pp.query("density_floor", base_density_floor);

	std::string density_floor_expr;
	pp.query("density_floor_expr", density_floor_expr);
	if (density_floor_expr.empty()) {
		amrex::Print() << "density_floor_expr must be set for DensityFloorUnit test.\n";
		return 1;
	}

	amrex::Parser parser(density_floor_expr);
	parser.registerVariables({"x", "y", "z", "base_density_floor"});
	auto const parser_exe = parser.compile<4>();

	constexpr int nx = 4;
	constexpr int ny = 1;
	constexpr int nz = 1;
	amrex::IntVect const dom_lo(AMREX_D_DECL(0, 0, 0));
	amrex::IntVect const dom_hi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
	amrex::Box const domain(dom_lo, dom_hi);
	amrex::RealBox const real_box({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)});
	amrex::Array<int, AMREX_SPACEDIM> const is_periodic{AMREX_D_DECL(0, 0, 0)};
	amrex::Geometry const geom(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

	amrex::BoxArray ba(domain);
	ba.maxSize(domain.size());
	amrex::DistributionMapping const dm(ba);
	int const ncomp = Physics_Indices<DensityFloorUnitProblem>::nvarTotal_cc;
	amrex::MultiFab state(ba, dm, ncomp, 0);

	state.setVal(0.0);

	amrex::Real const rho_init = 0.5 * base_density_floor;
	amrex::Real const Tgas_init = 1.0;
	amrex::Real const Eint_init = quokka::EOS<DensityFloorUnitProblem>::ComputeEintFromTgas(rho_init, Tgas_init);
	state.setVal(rho_init, HydroSystem<DensityFloorUnitProblem>::density_index, 1, 0);
	state.setVal(Eint_init, HydroSystem<DensityFloorUnitProblem>::energy_index, 1, 0);
	state.setVal(Eint_init, HydroSystem<DensityFloorUnitProblem>::internalEnergy_index, 1, 0);

	auto const density_floor_func = [=] AMREX_GPU_HOST_DEVICE(amrex::Real x, amrex::Real y, amrex::Real z,
							  amrex::Real base_floor) -> amrex::Real { return parser_exe(x, y, z, base_floor); };

	HydroSystem<DensityFloorUnitProblem>::EnforceLimits(base_density_floor, 0.0, state, geom.data(), density_floor_func);

	amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const *const prob_lo = geom.ProbLo();
	auto const *const dx = geom.CellSize();

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		amrex::Box const &bx = mfi.validbox();
		auto const &data = state.array(mfi);

		reduce_op.eval(bx, reduce_data,
			       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
				       amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
				       amrex::Real const y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
#else
				       amrex::Real const y = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
				       amrex::Real const z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
#else
				       amrex::Real const z = 0.0;
#endif
				       amrex::Real const expected = parser_exe(x, y, z, base_density_floor);
				       amrex::Real const actual = data(i, j, k, HydroSystem<DensityFloorUnitProblem>::density_index);
				       return {amrex::Math::abs(actual - expected)};
			       });
	}

	auto [max_err] = reduce_data.value();
	amrex::ParallelDescriptor::ReduceRealMax(max_err);

	amrex::Real const tol = 1.0e-12;
	int status = 0;
	if (!(max_err <= tol)) {
		amrex::Print() << "Max density floor error = " << max_err << "\n";
		status = 1;
	}

	return status;
}
