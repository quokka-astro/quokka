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

constexpr bool par_in_cell_center = true;
const int sink_write_interval = 1;
const double sphere_radius = 2.0e16; // cm
const double dx_fixed = sphere_radius / 64.0;

// from dimentionless units to cgs units
constexpr double cs0 = 0.2 * 1.0e5; // 0.2 km/s to cm/s
constexpr double temp0 = 10.0; // K, used for estimating internal energy
constexpr double mu = 2.33 * C::m_p;
constexpr double e0 = 1.0 / mu * C::k_B * temp0; // thermal energy per unit mass
constexpr double G = C::Gconst;
constexpr double t0 = 1.0e12;
double t_end = t0;
constexpr double unit_rho = 1.0 / (4 * M_PI * G * t0 * t0);
constexpr double unit_m = cs0 * cs0 * cs0 * t0 / G;
constexpr double unit_v = cs0;
constexpr double unit_l = cs0 * t0;
constexpr double mass_star = 0.975 * unit_m;
std::string sink_file = "../inputs/sink.txt";  // NOLINT

constexpr double rho_floor = 1.0e-10 * unit_rho;

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

template <> struct SimulationData<AccretionProblem> {
	std::vector<Real> time;
	std::vector<Real> Mstar;
	int step_counter = 0;  // Counter for tracking timesteps
};

template <> void QuokkaSimulation<AccretionProblem>::computeAfterTimestep()
{
	// Increment step counter
	userData_.step_counter++;

	// Check if we should write data this step
	if (userData_.step_counter % sink_write_interval == 0) {
		// Get particle data using the physics particle descriptor
		const int finest_level = finestLevel();
		const auto &real_data = particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(finest_level).first;

		if (amrex::ParallelDescriptor::IOProcessor()) {
			Real Mstar = 0.0;
			const int mass_index = 3;
			for (const auto &p : real_data) {
				Mstar += p[mass_index];
			}

			// Store data in memory
			userData_.time.push_back(tNew_[0]);
			userData_.Mstar.push_back(Mstar);

			// // Write data to file
			// std::ofstream outfile;
			// outfile.open(sink_output_file, std::ios_base::app); // Append mode
			// if (outfile.is_open()) {
			// 	outfile << std::scientific << std::setprecision(14) 
			// 		<< tNew_[0] << "\t" << Mstar << "\n";
			// 	outfile.close();
			// }
		}
	}
}

template <> void QuokkaSimulation<AccretionProblem>::createInitialSinkParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	SinkParticles->SetVerbose(1);
	SinkParticles->InitFromAsciiFile(sink_file, nreal_extra, nullptr);

	const int max_lev = max_level;

	// For the test problem in the Sink Particle paper, we want to set max_lev to 2.
	// AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_lev == 2, "amx_lev is not 2");

	// const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[max_lev].CellSizeArray();

	// manually set particle mass to M_star_in_Msun * C::M_solar
	for (auto &kv : SinkParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

			auto *pdata = particle_array().data();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.rdata(0) = mass_star;
				if (par_in_cell_center) {
					p.pos(0) = 0.5 * dx_fixed;
					p.pos(1) = 0.5 * dx_fixed;
					p.pos(2) = 0.5 * dx_fixed;
				} else {
					p.pos(0) = 0.0;
					p.pos(1) = 0.0;
					p.pos(2) = 0.0;
				}
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<AccretionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const amrex::Real dx = geom[lev].CellSizeArray()[0];

	const auto &prob_lo = geom[lev].ProbLoArray();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx;
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx;
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx;
		const Real r = std::sqrt(x * x + y * y + z * z);
		if (r < 0.5 * sphere_radius) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

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
	// const amrex::Gpu::DeviceVector<double> m_isothermal = {
	// 		0.981, 0.993, 1.010, 1.030, 1.050, 1.080, 1.120, 1.160, 1.200, 1.250, 1.300, 1.360, 1.420, 1.490, 1.560, 1.640, 1.720, 1.810, 1.900, 2.000
	// };

	const double par_center = par_in_cell_center ? 0.5 * dx_fixed : 0.0;

	// set initial conditions
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	auto const &x_isothermal_ptr = x_isothermal.dataPtr();
	auto const &alpha_isothermal_ptr = alpha_isothermal.dataPtr();
	auto const &neg_v_isothermal_ptr = neg_v_isothermal.dataPtr();
	// auto const &m_isothermal_ptr = m_isothermal.dataPtr();
	const int array_size = static_cast<int>(x_isothermal.size());

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// compute x,y,z relative to the particle position
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0] - par_center;
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1] - par_center;
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2] - par_center;
		const Real r = std::sqrt(x * x + y * y + z * z);

		const Real xx = r / unit_l;

		// interpolate alpha_isothermal, neg_v_isothermal and m_isothermal at xx
		Real alpha = 0.0;
		Real neg_v = 0.0; 
		if (xx >= 1.0) {
			alpha = 2.0 / (xx * xx);
			neg_v = 0.0;
		} else {
			alpha = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, alpha_isothermal_ptr, array_size);
			neg_v = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, neg_v_isothermal_ptr, array_size);
		}
		// const Real m = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, m_isothermal_ptr, array_size);

		const Real rho = std::max(alpha * unit_rho, rho_floor);
		const Real u = - neg_v * unit_v;
		Real vx = 0.0;
		Real vy = 0.0;
		Real vz = 0.0;
		if (r / dx_fixed  > 1.0e-10) {
			vx = u * x / r;
			vy = u * y / r;
			vz = u * z / r;
		}

		const Real Eint = rho * e0;
		const Real Ekin = 0.5 * rho * u * u;
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

	// read t_end from parameter file
	amrex::ParmParse pp("problem");
	pp.get("t_end", t_end);

	// Problem initialization
	QuokkaSimulation<AccretionProblem> sim(BCs_cc);
	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	// sim.initDt_ = 3.0e10;	      // ~1 kyr
	sim.tempFloor_ = 10.0; // K
	sim.stopTime_ = t_end - t0;

	// initialize
	sim.setInitialConditions();

	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.time.push_back(0.0);
		sim.userData_.Mstar.push_back(mass_star);
	}

	// get cell density as a function of x
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	const int nx = static_cast<int>(position.size());

	const Real par_x = par_in_cell_center ? 0.5 * dx_fixed : 0.0;
	const Real par_y = par_in_cell_center ? 0.5 * dx_fixed : 0.0;
	const Real par_z = par_in_cell_center ? 0.5 * dx_fixed : 0.0;

	std::vector<double> x(nx);
	std::vector<double> alpha(nx);
	std::vector<double> v_abs(nx);
	std::vector<double> rho(nx);
	std::vector<double> u(nx);
	std::vector<double> physical_x(nx);

	for (int i = 0; i < nx; ++i) {
		const double x_coor = position[i] - par_x;
		const double y_coor = 0.5 * dx_fixed - par_y;
		const double z_coor = 0.5 * dx_fixed - par_z;
		const double r = std::sqrt(x_coor * x_coor + y_coor * y_coor + z_coor * z_coor);
		physical_x[i] = x_coor;
		const Real rho_i = values.at(HydroSystem<AccretionProblem>::density_index)[i];
		const Real u_i = values.at(HydroSystem<AccretionProblem>::x1Momentum_index)[i] / rho_i;
		const Real alpha_i = rho_i / unit_rho;
		const Real v_i = u_i / unit_v;

		x[i] = r / unit_l;

		rho[i] = rho_i;
		u[i] = u_i;
		alpha[i] = alpha_i;
		v_abs[i] = physical_x[i] < 0.0 ? v_i : -v_i;
	}

	// save initial data to file
	std::ofstream fstream;
	fstream.open("particle_isothermal_data_initial.csv");
	fstream << "# t0 = " << t0 << "\n";
	fstream << "# t = " << 0.0 << "\n";
	fstream << "# x, alpha, v_abs, pos, rho, u";
	for (int i = 0; i < nx; ++i) {
		fstream << '\n';
		fstream << std::scientific << std::setprecision(14) << x[i] << ", " << alpha[i] << ", " << v_abs[i] << ", " << physical_x[i] << ", " << rho[i] << ", " << u[i];
	}
	fstream.close();

	// evolve
	sim.evolve();

	amrex::Print() << "Initial particle mass = " << mass_star << "\n";
	amrex::Print() << "Final particle mass = " << sim.userData_.Mstar.back() << "\n";

	const double t_end_real = sim.tNew_[0] + t0;
	const double unit_rho_1 = 1.0 / (4 * M_PI * G * t_end_real * t_end_real);
	const double unit_m_1 = cs0 * cs0 * cs0 * t_end_real / G;
	const double unit_v_1 = cs0;
	const double unit_l_1 = cs0 * t_end_real;

	// compute reference solution
	//
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
	// const amrex::Gpu::DeviceVector<double> m_isothermal = {
	// 		0.981, 0.993, 1.010, 1.030, 1.050, 1.080, 1.120, 1.160, 1.200, 1.250, 1.300, 1.360, 1.420, 1.490, 1.560, 1.640, 1.720, 1.810, 1.900, 2.000
	// };
	const int array_size = static_cast<int>(x_isothermal.size());
	auto const &x_isothermal_ptr = x_isothermal.dataPtr();
	auto const &alpha_isothermal_ptr = alpha_isothermal.dataPtr();
	auto const &neg_v_isothermal_ptr = neg_v_isothermal.dataPtr();

	// get cell density as a function of x
	auto [position1, values1] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);

	// get total gas mass of the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	// get total gas mass of the final state
	amrex::Real const m_gas_final = sim.state_new_cc_[0].sum(HydroSystem<AccretionProblem>::density_index) * vol;

	// const double m_tot_init = m_gas_init + m_stars_init;
	// const double m_tot_final = m_gas_final + m_stars_final;

	std::vector<double> alpha_ref(nx);
	std::vector<double> v_ref(nx);

	for (int i = 0; i < nx; ++i) {
		const double x_coor = position1[i] - par_x;
		const double y_coor = 0.5 * dx_fixed - par_y;
		const double z_coor = 0.5 * dx_fixed - par_z;
		const double r = std::sqrt(x_coor * x_coor + y_coor * y_coor + z_coor * z_coor);
		physical_x[i] = x_coor;
		x[i] = r / unit_l_1;
		const Real rho_i = values1.at(HydroSystem<AccretionProblem>::density_index)[i];
		const Real u_i = values1.at(HydroSystem<AccretionProblem>::x1Momentum_index)[i] / rho_i;
		const Real alpha_i = rho_i / unit_rho_1;
		const Real v_i = u_i / unit_v_1;

		rho[i] = rho_i;
		u[i] = u_i;
		alpha[i] = alpha_i;
		v_abs[i] = physical_x[i] < 0.0 ? v_i : -v_i;

		// interpolate alpha_isothermal, neg_v_isothermal and m_isothermal at xx
		if (x[i] >= 1.0) {
			alpha_ref[i] = 2.0 / (x[i] * x[i]);
			v_ref[i] = 0.0;
		} else {
			alpha_ref[i] = interpolate_value<BoundaryPolicy::Clamp>(x[i], x_isothermal_ptr, alpha_isothermal_ptr, array_size);
			v_ref[i] = interpolate_value<BoundaryPolicy::Clamp>(x[i], x_isothermal_ptr, neg_v_isothermal_ptr, array_size);
		}
	}

	// Mass will not be conserved because of the open boundary conditions
	// // check mass conservation
	// const double rel_error_total_mass = std::abs(m_tot_final - m_tot_init) / m_tot_init;
	// amrex::Print() << "rel_error_total_mass = " << rel_error_total_mass << "\n";

	// save positional data to file
	std::ofstream fstream_final;
	fstream_final.open("particle_isothermal_data_final.csv");
	fstream_final << "# t0 = " << t0 << "\n";
	fstream_final << "# t = " << sim.tNew_[0] << "\n";
	fstream_final << "# x, alpha, v_abs, alpha_ref, v_ref, pos, rho, u";
	for (int i = 0; i < nx; ++i) {
		fstream_final << '\n';
		fstream_final << std::scientific << std::setprecision(14) << x[i] << ", " << alpha[i] << ", " << v_abs[i] << ", " << alpha_ref[i] << ", " << v_ref[i] << ", " << physical_x[i] << ", " << rho[i] << ", " << u[i];
	}
	fstream_final.close();

	// save temporal data to file
	std::ofstream fstream_temporal;
	fstream_temporal.open("particle_isothermal_data_temporal.csv");
	fstream_temporal << "# t0 = " << t0 << "\n";
	fstream_temporal << "# t, Mstar";
	for (int i = 0; i < sim.userData_.time.size(); ++i) {
		fstream_temporal << '\n';
		fstream_temporal << std::scientific << std::setprecision(14) << sim.userData_.time[i] << ", " << sim.userData_.Mstar[i];
	}
	fstream_temporal.close();

#ifdef HAVE_PYTHON
	// plot density profile at end
	matplotlibcpp::clf();
	std::map<std::string, std::string> rho_args;
	rho_args["color"] = "red";
	rho_args["linestyle"] = "solid";
	rho_args["label"] = "simulation";
	matplotlibcpp::plot(physical_x, alpha, rho_args);
	std::map<std::string, std::string> alpha_ref_args;
	alpha_ref_args["color"] = "blue";
	alpha_ref_args["linestyle"] = "dashed";
	alpha_ref_args["label"] = "reference";
	matplotlibcpp::plot(physical_x, alpha_ref, alpha_ref_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("alpha");
	// title: t = t_end - t0
	matplotlibcpp::title("t = " + std::to_string(sim.tNew_[0]));
	matplotlibcpp::xlim(-1.0 * sphere_radius, 1.0 * sphere_radius);
	matplotlibcpp::legend();
	matplotlibcpp::save("particle_isothermal_alpha_profile.png");
	// plot velocity profile at end
	matplotlibcpp::clf();
	std::map<std::string, std::string> neg_v_args;
	neg_v_args["color"] = "red";
	neg_v_args["label"] = "simulation";
	matplotlibcpp::plot(physical_x, v_abs, neg_v_args);
	std::map<std::string, std::string> neg_v_ref_args;
	neg_v_ref_args["color"] = "blue";
	neg_v_ref_args["linestyle"] = "dashed";
	neg_v_ref_args["label"] = "reference";
	matplotlibcpp::plot(physical_x, v_ref, neg_v_ref_args);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("neg_v");
	matplotlibcpp::xlim(-1.0 * sphere_radius, 1.0 * sphere_radius);
	matplotlibcpp::title("t = " + std::to_string(sim.tNew_[0]));
	matplotlibcpp::legend();
	matplotlibcpp::save("particle_isothermal_neg_v_profile.png");
#endif

	return 0;
}
