//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testMHDQuirk.cpp
/// \brief Defines a test problem for the odd-even decoupling instability.
///

#include "hydro/hydro_system.hpp"
#include <algorithm>
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

struct MHDQuirk {
};

template <> struct quokka::EOS_Traits<MHDQuirk> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<MHDQuirk> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<MHDQuirk> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr Real dl = 3.692;
constexpr Real ul = -0.625;
constexpr Real pl = 26.85;
constexpr Real dr = 1.0;
constexpr Real ur = -5.0;
constexpr Real pr = 0.6;
constexpr int ishock_g = 0;

template <> void QuokkaSimulation<MHDQuirk>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_fc = Physics_Indices<MHDQuirk>::nvarPerDim_fc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_fc; ++n) {
			state_fc(i, j, k, n) = 0; // fill all b-field quantities with zeros
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

	Real const xshock = 0.4;
	int ishock = 0;
	for (ishock = 0; (prob_lo[0] + dx[0] * (ishock + static_cast<Real>(0.5))) < xshock; ++ishock) {
	}
	ishock--;
	amrex::Print() << "ishock = " << ishock << "\n";

	Real const dd = dl - 0.135;
	Real const ud = ul + 0.219;
	Real const pd = pl - 1.31;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double vx = NAN;
		double const vy = 0.;
		double const vz = 0.;
		double rho = NAN;
		double P = NAN;

		if (i <= ishock) {
			rho = dl;
			vx = ul;
			P = pl;
		} else {
			rho = dr;
			vx = ur;
			P = pr;
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
		state_cc(i, j, k, HydroSystem<MHDQuirk>::energy_index) = quokka::EOS<MHDQuirk>::ComputeEintFromPres(rho, P) + 0.5 * rho * v_sq;
		state_cc(i, j, k, HydroSystem<MHDQuirk>::internalEnergy_index) = quokka::EOS<MHDQuirk>::ComputeEintFromPres(rho, P);
	});
}

auto getDeltaEntropyVector() -> std::vector<Real> &
{
	static std::vector<Real> delta_s_vec;
	return delta_s_vec;
}

template <> void QuokkaSimulation<MHDQuirk>::computeAfterTimestep()
{
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// it should be sufficient examine a single box on level 0
		// (no AMR should be used for this problem, and the odd-even decoupling will
		// manifest in every row along the shock, if it happens)
		const auto &state_fc = state_new_fc_[0];
		amrex::MultiFab const &mf_state = state_new_cc_[0];
		int box_no = -1;
		int const ilo = ishock_g;
		int jlo = 0;
		int klo = 0;
		for (amrex::MFIter mfi(mf_state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			amrex::GpuArray<int, 3> box_lo = bx.loVect3d();
			jlo = box_lo[1];
			klo = box_lo[2];
			amrex::IntVect const cell{AMREX_D_DECL(ilo, jlo, klo)};
			if (bx.contains(cell)) {
				box_no = mfi.index();
				break;
			}
		}

		AMREX_ALWAYS_ASSERT(box_no != -1);
		auto const &state = mf_state.const_array(box_no);
		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const cons_fc{
		    AMREX_D_DECL(state_fc[0].const_array(box_no), state_fc[1].const_array(box_no), state_fc[2].const_array(box_no))};
		amrex::Box const bx = amrex::makeSingleCellBox(ilo, jlo, klo);
		Real host_s = NAN;
		amrex::AsyncArray async_s(&host_s, 1);
		Real *s = async_s.data();

		amrex::launch(bx, [=] AMREX_GPU_DEVICE(amrex::Box const &tbx) {
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
		getDeltaEntropyVector().push_back(host_s);
	}
}

template <> void QuokkaSimulation<MHDQuirk>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &deltas_vec = getDeltaEntropyVector();
		const Real deltas = *std::max_element(deltas_vec.begin(), deltas_vec.end());

		if (deltas > 0.06) {
			amrex::Print() << "The scheme suffers from the Carbuncle phenomenon : max delta s = " << deltas << "\n\n";
			amrex::Abort("Carbuncle detected!");
		} else {
			amrex::Print() << "The scheme looks stable against the Carbuncle phenomenon : "
					  "max delta s = "
				       << deltas << "\n\n";
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
	low_bdr_cells[RadSystem<MHDQuirk>::gasEnergy_index] = pl / (gamma - 1.) + 0.5 * dl * ul * ul;
	low_bdr_cells[RadSystem<MHDQuirk>::gasInternalEnergy_index] = pl / (gamma - 1.);
	low_bdr_cells[RadSystem<MHDQuirk>::gasDensity_index] = dl;
	low_bdr_cells[RadSystem<MHDQuirk>::x1GasMomentum_index] = dl * ul;
	low_bdr_cells[RadSystem<MHDQuirk>::x2GasMomentum_index] = 0.;
	low_bdr_cells[RadSystem<MHDQuirk>::x3GasMomentum_index] = 0.;

	// Right state
	amrex::GpuArray<amrex::Real, nvar> high_bdr_cells{};
	high_bdr_cells[RadSystem<MHDQuirk>::gasEnergy_index] = pr / (gamma - 1.) + 0.5 * dr * ur * ur;
	high_bdr_cells[RadSystem<MHDQuirk>::gasInternalEnergy_index] = pr / (gamma - 1.);
	high_bdr_cells[RadSystem<MHDQuirk>::gasDensity_index] = dr;
	high_bdr_cells[RadSystem<MHDQuirk>::x1GasMomentum_index] = dr * ur;
	high_bdr_cells[RadSystem<MHDQuirk>::x2GasMomentum_index] = 0.;
	high_bdr_cells[RadSystem<MHDQuirk>::x3GasMomentum_index] = 0.;

	// Apply boundary conditions
	setConstantDirichletBCLo<0>(iv, consVar, geom, low_bdr_cells);
	setConstantDirichletBCHi<0>(iv, consVar, geom, high_bdr_cells);
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
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir); // periodic
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<MHDQuirk> sim(BCs_cc, BCs_fc);

	sim.reconstructionOrder_ = 2; // PLM
	sim.stopTime_ = 0.4;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 2000;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
