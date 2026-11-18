// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testMHDQuirk.cpp
/// \brief Defines a test problem for the odd-even decoupling instability.
///

#include "hydro/hydro_system.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuAsyncArray.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"

using Real = amrex::Real;

struct MHDQuirk {};

template <> struct quokka::EOS_Traits<MHDQuirk> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<MHDQuirk> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<MHDQuirk> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
};

constexpr Real dl = 3.692;
constexpr Real ul = -0.625;
constexpr Real pl = 26.85;
constexpr Real dr = 1.0;
constexpr Real ur = -5.0;
constexpr Real pr = 0.6;
int ishock_g = 0; // NOLINT; set from the computed shock index in setInitialConditionsOnGrid; host-only, never dereferenced on device

// post-shock state, and the x2 component of the magnetic field on either side of the shock; set from
// problem.inverse_plasma_beta in problem_main(), where the default of zero keeps the states above
// with no magnetic field
AMREX_GPU_MANAGED Real post_shock_density = dl;	       // NOLINT
AMREX_GPU_MANAGED Real post_shock_velocity = ul;       // NOLINT
AMREX_GPU_MANAGED Real post_shock_pressure = pl;       // NOLINT
AMREX_GPU_MANAGED Real post_shock_magnetic_field = 0.; // NOLINT
AMREX_GPU_MANAGED Real pre_shock_magnetic_field = 0.;  // NOLINT

// for a pre-shock plasma beta of 1 / inverse_plasma_beta, solve the jump conditions of a
// perpendicular shock (magnetic field along x2) that moves at the same speed as the shock with no magnetic field
void setShockStates(Real const inverse_plasma_beta)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(inverse_plasma_beta >= 0., "problem.inverse_plasma_beta must be non-negative");
	if (inverse_plasma_beta == 0.) {
		return;
	}
	constexpr Real gamma = quokka::EOS_Traits<MHDQuirk>::gamma;
	constexpr Real shock_speed = (dl * ul - dr * ur) / (dl - dr);
	constexpr Real pre_shock_frame_velocity = shock_speed - ur;
	constexpr Real mass_flux = dr * pre_shock_frame_velocity;
	Real const b_pre = std::sqrt(2. * pr * inverse_plasma_beta);
	auto const compute_post_shock_pressure = [=](Real const compression) {
		return pr + 0.5 * b_pre * b_pre * (1. - compression * compression) + mass_flux * pre_shock_frame_velocity * (1. - 1. / compression);
	};
	auto const compute_energy_flux_jump = [=](Real const compression) {
		Real const post_shock_frame_velocity = pre_shock_frame_velocity / compression;
		Real const pre_shock_flux = mass_flux * (0.5 * pre_shock_frame_velocity * pre_shock_frame_velocity + gamma / (gamma - 1.) * pr / dr) +
					    pre_shock_frame_velocity * b_pre * b_pre;
		Real const post_shock_flux = mass_flux * (0.5 * post_shock_frame_velocity * post_shock_frame_velocity +
							  gamma / (gamma - 1.) * compute_post_shock_pressure(compression) / (compression * dr)) +
					     post_shock_frame_velocity * compression * compression * b_pre * b_pre;
		return pre_shock_flux - post_shock_flux;
	};
	// bisect on the compression ratio, which is bounded above by the strong-shock limit
	Real compression_lo = 1. + 1.e-6;
	Real compression_hi = (gamma + 1.) / (gamma - 1.) - 1.e-9;
	for (int iter = 0; iter < 200; ++iter) {
		Real const compression_mid = 0.5 * (compression_lo + compression_hi);
		if (compute_energy_flux_jump(compression_lo) * compute_energy_flux_jump(compression_mid) <= 0.) {
			compression_hi = compression_mid;
		} else {
			compression_lo = compression_mid;
		}
	}
	Real const compression = 0.5 * (compression_lo + compression_hi);
	post_shock_density = compression * dr;
	post_shock_velocity = shock_speed - pre_shock_frame_velocity / compression;
	post_shock_pressure = compute_post_shock_pressure(compression);
	post_shock_magnetic_field = compression * b_pre;
	pre_shock_magnetic_field = b_pre;
}

// index of the last cell centre to the left of the initial shock position
auto computeShockIndex(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) -> int
{
	Real const xshock = 0.4;
	int ishock = 0;
	for (ishock = 0; (prob_lo[0] + dx[0] * (ishock + static_cast<Real>(0.5))) < xshock; ++ishock) {
	}
	return ishock - 1;
}

template <> void QuokkaSimulation<MHDQuirk>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	int const ishock = computeShockIndex(grid_elem.dx_, grid_elem.prob_lo_);
	// only the x2 component of the magnetic field is non-zero, and it varies with x0 alone, so it is
	// divergence-free
	bool const is_x2_face = (grid_elem.dir_ == quokka::direction::z);
	Real const b_post = post_shock_magnetic_field;
	Real const b_pre = pre_shock_magnetic_field;

	const int ncomp_fc = Physics_Indices<MHDQuirk>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0;
		}
		if (is_x2_face) {
			if (i <= ishock) {
				state_fc(i, j, k, Physics_Indices<MHDQuirk>::mhdFirstIndex) = b_post;
			} else {
				state_fc(i, j, k, Physics_Indices<MHDQuirk>::mhdFirstIndex) = b_pre;
			}
		}
	});
}

template <> void QuokkaSimulation<MHDQuirk>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	int const ishock = computeShockIndex(dx, prob_lo);
	ishock_g = ishock;
	amrex::Print() << "ishock = " << ishock << "\n";

	Real const rho_post = post_shock_density;
	Real const vx_post = post_shock_velocity;
	Real const P_post = post_shock_pressure;
	Real const emag_post = 0.5 * post_shock_magnetic_field * post_shock_magnetic_field;
	Real const emag_pre = 0.5 * pre_shock_magnetic_field * pre_shock_magnetic_field;
	Real const dd = rho_post - 0.135;
	Real const ud = vx_post + 0.219;
	Real const pd = P_post - 1.31;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double vx = NAN;
		double const vy = 0.;
		double const vz = 0.;
		double rho = NAN;
		double P = NAN;
		double emag = NAN;

		if (i <= ishock) {
			rho = rho_post;
			vx = vx_post;
			P = P_post;
			emag = emag_post;
		} else {
			rho = dr;
			vx = ur;
			P = pr;
			emag = emag_pre;
		}

		if ((i == ishock) && (j % 2 == 0)) {
			rho = dd;
			vx = ud;
			P = pd;
		}

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(vz));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto v_sq = vx * vx + vy * vy + vz * vz;

		state_cc(i, j, k, HydroSystem<MHDQuirk>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<MHDQuirk>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<MHDQuirk>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<MHDQuirk>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<MHDQuirk>::energy_index) = quokka::EOS<MHDQuirk>::ComputeEintFromPres(rho, P) + 0.5 * rho * v_sq + emag;
		state_cc(i, j, k, HydroSystem<MHDQuirk>::internalEnergy_index) = quokka::EOS<MHDQuirk>::ComputeEintFromPres(rho, P);
	});
}

auto getDeltaEntropyVector() -> std::vector<Real> &
{
	static std::vector<Real> delta_s_vector;
	return delta_s_vector;
}

template <> void QuokkaSimulation<MHDQuirk>::computeAfterTimestep()
{
	// it should be sufficient to examine a single row on level 0
	// (no AMR should be used for this problem, and the odd-even decoupling will
	// manifest in every row along the shock, if it happens)
	//
	// the box containing the target cell is not necessarily owned by the IOProcessor, so every
	// rank searches its own local boxes and the result is reduced across ranks; ilo/jlo/klo are
	// a fixed target (not derived per-box), since the search must always target the same cell
	const auto &state_fc = state_new_fc_[0];
	amrex::MultiFab const &mf_state = state_new_cc_[0];
	int const ilo = ishock_g;
	int const jlo = 0;
	int const klo = 0;
	amrex::IntVect const cell{AMREX_D_DECL(ilo, jlo, klo)};

	Real local_s = AMREX_REAL_LOWEST;
	for (amrex::MFIter mfi(mf_state); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		if (!bx.contains(cell)) {
			continue;
		}

		int const box_no = mfi.index();
		auto const &state = mf_state.const_array(box_no);
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
		    AMREX_D_DECL(state_fc[0].const_array(box_no), state_fc[1].const_array(box_no), state_fc[2].const_array(box_no))};
		amrex::Box const single_cell_box = amrex::makeSingleCellBox(ilo, jlo, klo);
		Real host_s = NAN;
		amrex::AsyncArray async_s(&host_s, 1);
		Real *s = async_s.data();

		amrex::launch(single_cell_box, [=] AMREX_GPU_DEVICE(amrex::Box const &tbx) {
			amrex::GpuArray<int, 3> const idx = tbx.loVect3d();
			int const i = idx[0];
			int const j = idx[1];
			int const k = idx[2];
			Real const dodd = state(i, j + 1, k, HydroSystem<MHDQuirk>::density_index);
			Real const podd = HydroSystem<MHDQuirk>::ComputePressure(state, i, j + 1, k, &cons_fc);
			Real const deven = state(i, j, k, HydroSystem<MHDQuirk>::density_index);
			Real const peven = HydroSystem<MHDQuirk>::ComputePressure(state, i, j, k, &cons_fc);

			// the 'entropy function' s == P / rho^gamma
			const Real gamma = quokka::EOS_Traits<MHDQuirk>::gamma;
			Real const sodd = podd / std::pow(dodd, gamma);
			Real const seven = peven / std::pow(deven, gamma);
			s[0] = std::abs(sodd - seven);
		});

		async_s.copyToHost(&host_s, 1);
		local_s = host_s;
		break;
	}

	// exactly one rank owns the target cell; broadcast its result to the rest
	amrex::ParallelDescriptor::ReduceRealMax(local_s);
	AMREX_ALWAYS_ASSERT(local_s > AMREX_REAL_LOWEST);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		getDeltaEntropyVector().push_back(local_s);
	}
}

template <> void QuokkaSimulation<MHDQuirk>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &delta_s_vector = getDeltaEntropyVector();
		AMREX_ALWAYS_ASSERT(!delta_s_vector.empty());

		// the seed perturbation is large by construction right after it is introduced; a scheme
		// that damps it (rather than amplifies it) should show a small delta_s by late time, even
		// though delta_s starts large. Only the tail of the run distinguishes damping from growth.
		constexpr Real tail_fraction = 0.2;
		auto const tail_size = std::max<std::size_t>(1, static_cast<std::size_t>(tail_fraction * static_cast<Real>(delta_s_vector.size())));
		auto const tail_begin = delta_s_vector.end() - static_cast<std::ptrdiff_t>(tail_size);
		const Real max_delta_s = *std::max_element(tail_begin, delta_s_vector.end());

		if (max_delta_s > 0.06) {
			amrex::Print() << "The scheme suffers from the Carbuncle phenomenon : late-time max delta s = " << max_delta_s << "\n\n";
			amrex::Abort("Carbuncle detected!");
		} else {
			amrex::Print() << "The scheme looks stable against the Carbuncle phenomenon : "
					  "late-time max delta s = "
				       << max_delta_s << "\n\n";
		}
	}
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDQuirk>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
						     amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
						     int /*orig_comp*/)
{
	// Number of variables
	constexpr int nvar = Physics_Indices<MHDQuirk>::nvarTotal_cc;
	const auto gamma = quokka::EOS_Traits<MHDQuirk>::gamma;

	// Left state
	amrex::GpuArray<amrex::Real, nvar> low_bdr_cells{};
	low_bdr_cells[RadSystem<MHDQuirk>::gasEnergy_index] = post_shock_pressure / (gamma - 1.) +
							      0.5 * post_shock_density * post_shock_velocity * post_shock_velocity +
							      0.5 * post_shock_magnetic_field * post_shock_magnetic_field;
	low_bdr_cells[RadSystem<MHDQuirk>::gasInternalEnergy_index] = post_shock_pressure / (gamma - 1.);
	low_bdr_cells[RadSystem<MHDQuirk>::gasDensity_index] = post_shock_density;
	low_bdr_cells[RadSystem<MHDQuirk>::x1GasMomentum_index] = post_shock_density * post_shock_velocity;
	low_bdr_cells[RadSystem<MHDQuirk>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<MHDQuirk>::x3GasMomentum_index] = 0.;

	// Right state
	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	high_bdr_cells[RadSystem<MHDQuirk>::gasEnergy_index] =
	    pr / (gamma - 1.) + 0.5 * dr * ur * ur + 0.5 * pre_shock_magnetic_field * pre_shock_magnetic_field;
	high_bdr_cells[RadSystem<MHDQuirk>::gasInternalEnergy_index] = pr / (gamma - 1.);
	high_bdr_cells[RadSystem<MHDQuirk>::gasDensity_index] = dr;
	high_bdr_cells[RadSystem<MHDQuirk>::x1GasMomentum_index] = dr * ur;
	high_bdr_cells[RadSystem<MHDQuirk>::x2GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<MHDQuirk>::x3GasMomentum_index] = 0.;

	// Apply boundary conditions
	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
	setConstantDirichletBCHi<0>(iv, consVar, geom, high_bdr_cells);
}

template <>
template <quokka::direction dir>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MHDQuirk>::setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar_fc, int /*dcomp*/,
							    int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
							    const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
	// {x0-face, x1-face, x2-face} values of the magnetic field on each side of the domain
	amrex::GpuArray<amrex::Real, 3> const low_bdr_values = {0., 0., post_shock_magnetic_field};
	amrex::GpuArray<amrex::Real, 3> const high_bdr_values = {0., 0., pre_shock_magnetic_field};
	setConstantDirichletBCFaceVarLo<0, dir, 3>(iv, consVar_fc, geom, low_bdr_values);
	setConstantDirichletBCFaceVarHi<0, dir, 3>(iv, consVar_fc, geom, high_bdr_values);
}

auto problem_main() -> int
{
	// Boundary conditions: ext_dir in x, periodic in y and z
	auto BCs_cc = quokka::BC<MHDQuirk>(quokka::BCType::ext_dir,  // x: outflow
					   quokka::BCType::int_dir,  // y: periodic
					   quokka::BCType::int_dir); // z: periodic

	const int nvars_fc = Physics_Indices<MHDQuirk>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		// x: Dirichlet (the magnetic field can differ on either side of the shock); y, z: periodic
		BCs_fc[icomp].setLo(0, amrex::BCType::ext_dir);
		BCs_fc[icomp].setHi(0, amrex::BCType::ext_dir);
		for (int idim = 1; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	// pre-shock inverse plasma beta; the default of zero sets no magnetic field
	Real inverse_plasma_beta = 0.;
	amrex::ParmParse const pp("problem");
	pp.query("inverse_plasma_beta", inverse_plasma_beta);
	setShockStates(inverse_plasma_beta);

	// Problem initialization
	QuokkaSimulation<MHDQuirk> sim(BCs_cc, BCs_fc);

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
