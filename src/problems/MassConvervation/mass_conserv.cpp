/// \file mass_conserv.cpp
/// \brief Defines a test problem for mass conservation.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "util/fextract.hpp"

#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "mass_conserv.hpp"

struct TheProblem {
};

constexpr double mass_loc = 0.5001;
constexpr double mass_mass = 1.0e-2;
constexpr double initial_Tgas = 1.0;
constexpr double CV = 1.5;
constexpr double initial_rho = 1.0;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const auto prob_lo = grid_elem.prob_lo_;
	const auto dx = grid_elem.dx_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + (i * dx[0]);
		const double yL = prob_lo[1] + (j * dx[1]);
		const double zL = prob_lo[2] + (k * dx[2]);
		double rho = initial_rho;
		if (xL <= mass_loc && (xL + dx[0]) > mass_loc && yL <= mass_loc && (yL + dx[1]) > mass_loc && zL <= mass_loc && (zL + dx[2]) > mass_loc) {
			rho = mass_mass / (AMREX_D_TERM(dx[0], *dx[1], *dx[2]));
		}
		const double Egas = rho * CV * initial_Tgas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
	});
}

template <> void QuokkaSimulation<TheProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto prob_lo = geom[lev].ProbLoArray();
		const auto dx = geom[lev].CellSizeArray();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<TheProblem>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
			// refine in the left half (x < 0.5). Use 0.4 to keep off the boundary
			// if (x >= 0.7) {
			if (x <= 0.3) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{

	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<TheProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<TheProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<TheProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<TheProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// // Boundary conditions
	// constexpr int nvars = RadSystem<TheProblem>::nvar_;
	// amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	// for (int n = 0; n < nvars; ++n) {
	// 	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
	// 		BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
	// 		BCs_cc[n].setHi(i, amrex::BCType::int_dir);
	// 	}
	// }

	// Problem parameters
	const double tmax = 1.0;

	// Problem initialization
	QuokkaSimulation<TheProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;

	// initialize
	sim.setInitialConditions();

	// get total mass
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const total_mass = sim.state_new_cc_[0].sum(HydroSystem<TheProblem>::density_index) * vol;
	amrex::Print() << "Initial total mass: " << total_mass << "\n";

	// evolve
	sim.evolve();

	// get total mass
	amrex::Real const total_mass_final = sim.state_new_cc_[0].sum(HydroSystem<TheProblem>::density_index) * vol;
	amrex::Print() << "Final total mass: " << total_mass_final << "\n";

	const double rel_err = std::abs(total_mass_final - total_mass) / total_mass;
	amrex::Print() << "Relative error: " << rel_err << "\n";

	// check if mass is conserved
	if (rel_err > 1.0e-10) {
		amrex::Print() << "Mass is not conserved!\n";
		return 1;
	}

	return 0;
}
