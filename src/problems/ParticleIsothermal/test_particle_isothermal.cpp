/// \file test_particle_isothermal.cpp
/// \brief Defines a test problem for isothermal accretion.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "util/fextract.hpp"
#include <gcem.hpp>
#include <iomanip>
#include <fstream>

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

using amrex::Real;

struct AccretionProblem {
};

// from dimentionless units to cgs units
constexpr double t0 = 1.0e12;
constexpr double t_end = 5.0e12;
constexpr double cs0 = 0.2 * 1.0e5; // 0.2 km/s to cm/s
constexpr double temp0 = 10.0; // K, used for estimating internal energy
constexpr double mu = 2.33 * C::m_p;
constexpr double e0 = 1.0 / mu * C::k_B * temp0; // thermal energy per unit mass
constexpr double G = C::Gconst;
constexpr double unit_rho = 1.0 / (4 * M_PI * G * t0 * t0);
constexpr double unit_m = cs0 * cs0 * cs0 * t0 / G;
constexpr double unit_v = cs0;
constexpr double unit_l = cs0 * t0;

template <> struct Particle_Traits<AccretionProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Sink;
};

template <> struct quokka::EOS_Traits<AccretionProblem> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = cs0;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<AccretionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<AccretionProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// x values (dimensionless radius)
	const amrex::Gpu::DeviceVector<double> x_isothermal = {
			0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000
	};

	// alpha values (density parameter)
	const amrex::Gpu::DeviceVector<double> alpha_isothermal = {
			71.500, 27.800, 16.400, 11.500, 8.760, 7.090, 5.950, 5.140, 4.520, 4.040, 3.660, 3.350, 3.080, 2.860, 2.670, 2.500, 2.350, 2.220, 2.100, 2.000
	};

	// neg_v values (negative velocity)
	const amrex::Gpu::DeviceVector<double> neg_v_isothermal = {
			5.440, 3.470, 2.580, 2.050, 1.680, 1.400, 1.180, 1.010, 0.861, 0.735, 0.625, 0.528, 0.442, 0.363, 0.291, 0.225, 0.163, 0.106, 0.051, 0.000
	};

	// m values (mass parameter)
	const amrex::Gpu::DeviceVector<double> m_isothermal = {
			0.981, 0.993, 1.010, 1.030, 1.050, 1.080, 1.120, 1.160, 1.200, 1.250, 1.300, 1.360, 1.420, 1.490, 1.560, 1.640, 1.720, 1.810, 1.900, 2.000
	};

	const double par_center = 0.0;

	// set initial conditions
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	auto const &x_isothermal_ptr = x_isothermal.dataPtr();
	auto const &alpha_isothermal_ptr = alpha_isothermal.dataPtr();
	auto const &neg_v_isothermal_ptr = neg_v_isothermal.dataPtr();
	auto const &m_isothermal_ptr = m_isothermal.dataPtr();
	const int array_size = static_cast<int>(x_isothermal.size());

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// compute x,y,z relative to the particle position
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0] - par_center;
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1] - par_center;
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2] - par_center;
		const Real r = std::sqrt(x * x + y * y + z * z);

		const Real xx = r / unit_l;

		// interpolate alpha_isothermal, neg_v_isothermal and m_isothermal at xx
		const Real alpha = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, alpha_isothermal_ptr, array_size);
		const Real neg_v = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, neg_v_isothermal_ptr, array_size);
		const Real m = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, m_isothermal_ptr, array_size);

		const Real rho = alpha * unit_rho;
		const Real v = - neg_v * unit_v;
		Real vx = 0.0;
		Real vy = 0.0;
		Real vz = 0.0;
		if (r > 0.0) {
			vx = v * x / r;
			vy = v * y / r;
			vz = v * z / r;
		}

		const Real Eint = rho * e0;
		const Real Ekin = 0.5 * rho * v * v;
		const Real Etot = Eint + Ekin;

		state_cc(i, j, k, HydroSystem<AccretionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::energy_index) = Etot;
	});
}

auto problem_main() -> int
{

	const int ncomp_cc = Physics_Indices<AccretionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// // periodic boundaries
			// BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			// BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			// octant symmetry
			// FOextrap
			for (int i = 0; i < AMREX_SPACEDIM; ++i) {
				BCs_cc[n].setLo(i, amrex::BCType::foextrap);
				BCs_cc[n].setHi(i, amrex::BCType::foextrap);
			}
			// if (isNormalComp(n, i)) {
			// 	BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
			// 	BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			// } else {
			// 	BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
			// 	BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			// }
		}
	}

	// Problem initialization
	QuokkaSimulation<AccretionProblem> sim(BCs_cc);
	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	// sim.initDt_ = 3.0e10;	      // ~1 kyr
	sim.tempFloor_ = 10.0; // K
	sim.stopTime_ = t_end;

	// initialize
	sim.setInitialConditions();

	auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);

	// evolve
	sim.evolve();

	// get cell density as a function of x
	auto [position, values1] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);

	// get total gas mass of the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	// get total gas mass of the final state
	amrex::Real const m_gas_final = sim.state_new_cc_[0].sum(HydroSystem<AccretionProblem>::density_index) * vol;

	// const double m_tot_init = m_gas_init + m_stars_init;
	// const double m_tot_final = m_gas_final + m_stars_final;

	const int nx = static_cast<int>(position.size());
	std::vector<double> x(nx);
	std::vector<double> rho(nx);
	std::vector<double> rho1(nx);

	for (int i = 0; i < nx; ++i) {
		x[i] = position[i] / unit_l;
		rho[i] = values0.at(HydroSystem<AccretionProblem>::density_index)[i] / unit_rho;
		rho1[i] = values1.at(HydroSystem<AccretionProblem>::density_index)[i] / unit_rho;
	}

	// Mass will not be conserved because of the open boundary conditions
	// // check mass conservation
	// const double rel_error_total_mass = std::abs(m_tot_final - m_tot_init) / m_tot_init;
	// amrex::Print() << "rel_error_total_mass = " << rel_error_total_mass << "\n";

#ifdef HAVE_PYTHON
	// plot density profile at beginning and end
	matplotlibcpp::clf();
	std::map<std::string, std::string> rho_args;
	rho_args["label"] = "Initial";
	rho_args["color"] = "red";
	matplotlibcpp::plot(x, rho, rho_args);
	std::map<std::string, std::string> rho1_args;
	rho1_args["label"] = "Final";
	rho1_args["color"] = "blue";
	matplotlibcpp::plot(x, rho1, rho1_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("Density / rho0");
	matplotlibcpp::legend();
	matplotlibcpp::save("particle_isothermal_density_profile.png");
#endif

}
