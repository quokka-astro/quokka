//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_box.cpp
/// \brief Defines a test problem for Cloudy cooling.
///
#include <random>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "CloudyCooling.hpp"
#include "ODEIntegrate.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "radiation_system.hpp"
#include "test_cooling.hpp"

using amrex::Real;

struct CoolingTest {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double m_H = quokka::hydrogen_mass_cgs;
constexpr double seconds_in_year = 3.154e7;

template <> struct quokka::EOS_Traits<CoolingTest> {
	static constexpr double gamma = 5. / 3.; // default value
	static constexpr double mean_molecular_weight = quokka::hydrogen_mass_cgs;
	static constexpr double boltzmann_constant = quokka::boltzmann_constant_cgs;
};

template <> struct Physics_Traits<CoolingTest> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numPassiveScalars = 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};

constexpr double Tgas0 = 6000.;	      // K
constexpr amrex::Real T_floor = 10.0; // K
constexpr double rho0 = 0.6 * m_H;    // g cm^-3

template <> struct SimulationData<CoolingTest> {
	cloudy_tables cloudyTables;
	std::vector<Real> t_vec_;
	std::vector<Real> Tgas_vec_;
};

template <> void RadhydroSimulation<CoolingTest>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real rho = 0.12 * m_H; // g cm^-3
		Real xmom = 0;
		Real ymom = 0;
		Real zmom = 0;
		Real const P = 4.0e4 * quokka::boltzmann_constant_cgs; // erg cm^-3
		Real Eint = (quokka::EOS_Traits<CoolingTest>::gamma - 1.) * P;
		Real const Egas = RadSystem<CoolingTest>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		state_cc(i, j, k, RadSystem<CoolingTest>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<CoolingTest>::gasInternalEnergy_index) = Eint;
		state_cc(i, j, k, RadSystem<CoolingTest>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<CoolingTest>::x1GasMomentum_index) = xmom;
		state_cc(i, j, k, RadSystem<CoolingTest>::x2GasMomentum_index) = ymom;
		state_cc(i, j, k, RadSystem<CoolingTest>::x3GasMomentum_index) = zmom;
	});
}

struct ODEUserData {
	amrex::Real rho{};
	cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
{
	// unpack user_data
	auto *udata = static_cast<ODEUserData *>(user_data);
	Real rho = udata->rho;
	cloudyGpuConstTables &tables = udata->tables;

	// compute temperature (implicit solve, depends on composition)
	Real Eint = y_data[0];
	Real T = ComputeTgasFromEgas(rho, Eint, quokka::EOS_Traits<CoolingTest>::gamma, tables);

	// compute cooling function
	y_rhs[0] = cloudy_cooling_function(rho, T, tables);
	return 0;
}

void computeCooling(amrex::MultiFab &mf, const Real dt_in, cloudy_tables &cloudyTables)
{
	BL_PROFILE("RadhydroSimulation::computeCooling()")

	const Real dt = dt_in;
	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4; // not recommended to change this

	auto tables = cloudyTables.const_tables();

	// loop over all cells in MultiFab mf
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<CoolingTest>::density_index);
			const Real x1Mom = state(i, j, k, HydroSystem<CoolingTest>::x1Momentum_index);
			const Real x2Mom = state(i, j, k, HydroSystem<CoolingTest>::x2Momentum_index);
			const Real x3Mom = state(i, j, k, HydroSystem<CoolingTest>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<CoolingTest>::energy_index);

			Real Eint = RadSystem<CoolingTest>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);

			ODEUserData user_data{rho, tables};
			quokka::valarray<Real, 1> y = {Eint};
			quokka::valarray<Real, 1> abstol = {reltol_floor * ComputeEgasFromTgas(rho, T_floor, quokka::EOS_Traits<CoolingTest>::gamma, tables)};

			// do integration with RK2 (Heun's method)
			int steps_taken = 0;
			rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, steps_taken);

			const Real Eint_new = y[0];
			const Real dEint = Eint_new - Eint;

			state(i, j, k, HydroSystem<CoolingTest>::energy_index) += dEint;
			state(i, j, k, HydroSystem<CoolingTest>::internalEnergy_index) += dEint;
		});
	}
}

template <> void RadhydroSimulation<CoolingTest>::computeAfterLevelAdvance(int lev, amrex::Real /*time*/, amrex::Real dt_lev, int /*ncycle*/)
{
	// compute operator split physics
	computeCooling(state_new_cc_[lev], dt_lev, userData_.cloudyTables);
}

template <> void RadhydroSimulation<CoolingTest>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		const Real Etot = values.at(RadSystem<CoolingTest>::gasEnergy_index)[0];
		const Real x1GasMom = values.at(RadSystem<CoolingTest>::x1GasMomentum_index)[0];
		const Real x2GasMom = values.at(RadSystem<CoolingTest>::x2GasMomentum_index)[0];
		const Real x3GasMom = values.at(RadSystem<CoolingTest>::x3GasMomentum_index)[0];
		const Real rho = values.at(RadSystem<CoolingTest>::gasDensity_index)[0];

		const Real Eint = RadSystem<CoolingTest>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Etot);
		const Real T = ComputeTgasFromEgas(rho, Eint, quokka::EOS_Traits<CoolingTest>::gamma, userData_.cloudyTables.const_tables());

		userData_.t_vec_.push_back(tNew_[0]);
		userData_.Tgas_vec_.push_back(T);
	}
}

template <> void RadhydroSimulation<CoolingTest>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { tag(i, j, k) = amrex::TagBox::SET; });
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.25;
	// double max_time = 7.5e4 * seconds_in_year; // 75 kyr
	// const int max_timesteps = 2e4;

	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<CoolingTest>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::int_dir); // periodic
		BCs_cc[n].setHi(0, amrex::BCType::int_dir);
	}

	RadhydroSimulation<CoolingTest> sim(BCs_cc);

	// Standard PPM gives unphysically enormous temperatures when used for
	// this problem (e.g., ~1e14 K or higher), but can be fixed by
	// reconstructing the temperature instead of the pressure
	sim.reconstructionOrder_ = 3; // PLM
	sim.cflNumber_ = CFL_number;

	// Read Cloudy tables
	readCloudyData(sim.userData_.cloudyTables);

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// save time series
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// save solution values to csv file
		std::ofstream csvfile;
		csvfile.open("cooling_box_output.csv");
		csvfile << "# time temperature\n";
		const int nx = static_cast<int>(sim.userData_.t_vec_.size());
		for (int i = 0; i < nx; ++i) {
			csvfile << sim.userData_.t_vec_.at(i) << " ";
			csvfile << sim.userData_.Tgas_vec_.at(i) << "\n";
		}
		csvfile.close();
	}

	// Cleanup and exit
	int status = 0;
	return status;
}
