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
#include <fstream>
#include <gcem.hpp>
#include <iomanip>

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

using amrex::Real;

struct AccretionProblem {
};

static constexpr bool par_in_cell_center = true;
const int sink_write_interval = 1;
const double sphere_radius = 2.0e16; // cm
const double dx_fixed = sphere_radius / 64.0;

// from dimentionless units to cgs units
constexpr double cs0 = 0.2 * 1.0e5; // 0.2 km/s to cm/s
constexpr double temp0 = 10.0;	    // K, used for estimating internal energy
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
std::string sink_file = "../inputs/sink.txt"; // NOLINT

constexpr double rho_floor = 1.0e-10 * unit_rho;

// x values (dimensionless radius)
const std::vector<double> x_isothermal = {
	1.000000e-03, 1.078972e-03, 1.164181e-03, 1.256119e-03, 1.355318e-03, 1.462351e-03, 1.577836e-03, 1.702441e-03, 1.836887e-03, 1.981950e-03, 2.138469e-03, 2.307349e-03, 2.489566e-03, 2.686173e-03, 2.898306e-03, 3.127192e-03, 3.374153e-03, 3.640618e-03, 3.928126e-03, 4.238339e-03, 4.573051e-03, 4.934195e-03, 5.323860e-03, 5.744297e-03, 6.197938e-03, 6.687403e-03, 7.215523e-03, 7.785349e-03, 8.400176e-03, 9.063558e-03, 9.779328e-03, 1.055162e-02, 1.138491e-02, 1.228400e-02, 1.325410e-02, 1.430081e-02, 1.543017e-02, 1.664873e-02, 1.796352e-02, 1.938214e-02, 2.091279e-02, 2.256432e-02, 2.434628e-02, 2.626896e-02, 2.834348e-02, 3.058183e-02, 3.299695e-02, 3.560280e-02, 3.841443e-02, 4.144811e-02, 4.472136e-02, 4.825311e-02, 5.206377e-02, 5.617536e-02, 6.061166e-02, 6.539831e-02, 7.056296e-02, 7.613548e-02, 8.214808e-02, 8.863550e-02, 9.563525e-02, 1.031878e-01, 1.113368e-01, 1.201293e-01, 1.296162e-01, 1.398523e-01, 1.508967e-01, 1.628134e-01, 1.756711e-01, 1.895443e-01, 2.045130e-01, 2.206639e-01, 2.380902e-01, 2.568928e-01, 2.771802e-01, 2.990698e-01, 3.226880e-01, 3.481714e-01, 3.756673e-01, 4.053346e-01, 4.373448e-01, 4.718830e-01, 5.091486e-01, 5.493573e-01, 5.927413e-01, 6.395515e-01, 6.900583e-01, 7.445538e-01, 8.033530e-01, 8.667956e-01, 9.352484e-01, 1.009107e+00, 1.088799e+00, 1.174784e+00, 1.267559e+00, 1.367661e+00, 1.475668e+00, 1.592205e+00, 1.717946e+00, 1.853616e+00, 2.000000e+00
};

// alpha values (density parameter)
const std::vector<double> alpha_isothermal = {
	2.220883e+04, 1.982301e+04, 1.769390e+04, 1.579389e+04, 1.409828e+04, 1.258508e+04, 1.123465e+04, 1.002945e+04, 8.953846e+03, 7.993895e+03, 7.137143e+03, 6.372482e+03, 5.689998e+03, 5.080848e+03, 4.537139e+03, 4.051829e+03, 3.618635e+03, 3.231949e+03, 2.886769e+03, 2.578629e+03, 2.303546e+03, 2.057965e+03, 1.838715e+03, 1.642963e+03, 1.468185e+03, 1.312126e+03, 1.172775e+03, 1.048337e+03, 9.372101e+02, 8.379647e+02, 7.493253e+02, 6.701535e+02, 5.994334e+02, 5.362581e+02, 4.798187e+02, 4.293929e+02, 3.843362e+02, 3.440733e+02, 3.080909e+02, 2.759306e+02, 2.471836e+02, 2.214846e+02, 1.985079e+02, 1.779625e+02, 1.595888e+02, 1.431550e+02, 1.284541e+02, 1.153014e+02, 1.035320e+02, 9.299862e+01, 8.356977e+01, 7.512806e+01, 6.756866e+01, 6.079795e+01, 5.473231e+01, 4.929709e+01, 4.442559e+01, 4.005824e+01, 3.614182e+01, 3.262879e+01, 2.947667e+01, 2.664750e+01, 2.410737e+01, 2.182595e+01, 1.977616e+01, 1.793375e+01, 1.627708e+01, 1.478676e+01, 1.344547e+01, 1.223772e+01, 1.114961e+01, 1.016875e+01, 9.284012e+00, 8.485438e+00, 7.764110e+00, 7.112035e+00, 6.522046e+00, 5.987716e+00, 5.503281e+00, 5.063567e+00, 4.663931e+00, 4.300207e+00, 3.968657e+00, 3.665929e+00, 3.389019e+00, 3.135242e+00, 2.902204e+00, 2.687773e+00, 2.490071e+00, 2.307447e+00, 2.138475e+00, 1.963108e+00, 1.686863e+00, 1.449087e+00, 1.244777e+00, 1.069256e+00, 9.184772e-01, 7.889563e-01, 6.776981e-01, 5.821282e-01, 5.000350e-01
};

// neg_v values (negative velocity)
const std::vector<double> neg_v_isothermal = {
	4.393831e+01, 4.228426e+01, 4.069148e+01, 3.915768e+01, 3.768067e+01, 3.625831e+01, 3.488855e+01, 3.356943e+01, 3.229905e+01, 3.107559e+01, 2.989728e+01, 2.876243e+01, 2.766942e+01, 2.661667e+01, 2.560268e+01, 2.462600e+01, 2.368521e+01, 2.277898e+01, 2.190602e+01, 2.106506e+01, 2.025491e+01, 1.947442e+01, 1.872247e+01, 1.799798e+01, 1.729993e+01, 1.662733e+01, 1.597921e+01, 1.535466e+01, 1.475280e+01, 1.417276e+01, 1.361374e+01, 1.307493e+01, 1.255559e+01, 1.205499e+01, 1.157241e+01, 1.110718e+01, 1.065865e+01, 1.022620e+01, 9.809227e+00, 9.407145e+00, 9.019400e+00, 8.645457e+00, 8.284800e+00, 7.936933e+00, 7.601382e+00, 7.277687e+00, 6.965409e+00, 6.664125e+00, 6.373429e+00, 6.092931e+00, 5.822254e+00, 5.561040e+00, 5.308943e+00, 5.065630e+00, 4.830782e+00, 4.604094e+00, 4.385271e+00, 4.174033e+00, 3.970107e+00, 3.773235e+00, 3.583168e+00, 3.399665e+00, 3.222497e+00, 3.051444e+00, 2.886292e+00, 2.726838e+00, 2.572884e+00, 2.424240e+00, 2.280721e+00, 2.142149e+00, 2.008350e+00, 1.879153e+00, 1.754392e+00, 1.633902e+00, 1.517518e+00, 1.405078e+00, 1.296416e+00, 1.191366e+00, 1.089757e+00, 9.914143e-01, 8.961579e-01, 8.038001e-01, 7.141451e-01, 6.269881e-01, 5.421144e-01, 4.592977e-01, 3.782997e-01, 2.988683e-01, 2.207357e-01, 1.436150e-01, 6.719382e-02, 6.749651e-04, 3.028416e-04, 2.116846e-04, 1.622080e-04, 1.295612e-04, 1.058054e-04, 8.744707e-05, 7.266235e-05, 6.040116e-05, 5.000127e-05
};

// // m values (mass parameter)
// const std::vector<double> m_isothermal = {
// 	9.758410e-01, 9.758428e-01, 9.758448e-01, 9.758471e-01, 9.758496e-01, 9.758524e-01, 9.758556e-01, 9.758591e-01, 9.758631e-01, 9.758675e-01, 9.758726e-01, 9.758782e-01, 9.758845e-01, 9.758915e-01, 9.758995e-01, 9.759084e-01, 9.759183e-01, 9.759295e-01, 9.759421e-01, 9.759562e-01, 9.759720e-01, 9.759898e-01, 9.760097e-01, 9.760320e-01, 9.760571e-01, 9.760853e-01, 9.761169e-01, 9.761524e-01, 9.761923e-01, 9.762370e-01, 9.762873e-01, 9.763438e-01, 9.764073e-01, 9.764786e-01, 9.765587e-01, 9.766488e-01, 9.767501e-01, 9.768640e-01, 9.769920e-01, 9.771361e-01, 9.772982e-01, 9.774806e-01, 9.776859e-01, 9.779171e-01, 9.781775e-01, 9.784709e-01, 9.788015e-01, 9.791742e-01, 9.795944e-01, 9.800686e-01, 9.806036e-01, 9.812077e-01, 9.818901e-01, 9.826611e-01, 9.835327e-01, 9.845187e-01, 9.856344e-01, 9.868978e-01, 9.883292e-01, 9.899519e-01, 9.917927e-01, 9.938823e-01, 9.962561e-01, 9.989546e-01, 1.002025e+00, 1.005520e+00, 1.009504e+00, 1.014048e+00, 1.019236e+00, 1.025164e+00, 1.031944e+00, 1.039709e+00, 1.048609e+00, 1.058821e+00, 1.070552e+00, 1.084042e+00, 1.099573e+00, 1.117474e+00, 1.138130e+00, 1.161990e+00, 1.189581e+00, 1.221521e+00, 1.258530e+00, 1.301454e+00, 1.351279e+00, 1.409160e+00, 1.476442e+00, 1.554694e+00, 1.645743e+00, 1.751713e+00, 1.875070e+00, 2.018582e+00, 2.177928e+00, 2.349884e+00, 2.535430e+00, 2.735636e+00, 2.951657e+00, 3.184741e+00, 3.436235e+00, 3.707592e+00, 4.000380e+00
// };

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
	int step_counter = 0; // Counter for tracking timesteps
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
	// Convert std::vector to GPU-accessible data for device code
	amrex::Gpu::DeviceVector<double> x_isothermal_gpu;
	amrex::Gpu::DeviceVector<double> alpha_isothermal_gpu;
	amrex::Gpu::DeviceVector<double> neg_v_isothermal_gpu;

	x_isothermal_gpu.resize(x_isothermal.size());
	alpha_isothermal_gpu.resize(alpha_isothermal.size());
	neg_v_isothermal_gpu.resize(neg_v_isothermal.size());

	amrex::Gpu::copy(amrex::Gpu::hostToDevice, x_isothermal.begin(), x_isothermal.end(), x_isothermal_gpu.begin());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, alpha_isothermal.begin(), alpha_isothermal.end(), alpha_isothermal_gpu.begin());
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, neg_v_isothermal.begin(), neg_v_isothermal.end(), neg_v_isothermal_gpu.begin());

	// m values (mass parameter)
	// const amrex::Gpu::DeviceVector<double> m_isothermal = {
	// 		0.981,
	// 0.993, 1.010, 1.030, 1.050, 1.080, 1.120, 1.160, 1.200, 1.250, 1.300, 1.360, 1.420, 1.490, 1.560, 1.640, 1.720, 1.810, 1.900, 2.000
	// };

	const double par_center = par_in_cell_center ? 0.5 * dx_fixed : 0.0;

	// set initial conditions
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	auto const &x_isothermal_ptr = x_isothermal_gpu.dataPtr();
	auto const &alpha_isothermal_ptr = alpha_isothermal_gpu.dataPtr();
	auto const &neg_v_isothermal_ptr = neg_v_isothermal_gpu.dataPtr();
	// auto const &m_isothermal_ptr = m_isothermal.dataPtr();
	const int array_size = static_cast<int>(x_isothermal_gpu.size());

	const auto rho_floor_ = rho_floor;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// compute x,y,z relative to the particle position
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0] - par_center;
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1] - par_center;
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2] - par_center;
		const Real r = std::sqrt(x * x + y * y + z * z);

		Real xx = r / unit_l;

		// interpolate alpha_isothermal, neg_v_isothermal and m_isothermal at xx
		Real alpha = 0.0;
		Real neg_v = 0.0;
		if (xx >= 1.0) {
			alpha = 2.0 / (xx * xx);
			neg_v = 0.0;
		} else {
			if (r < 0.5 * dx[0]) {
				xx = 0.5 * dx[0] / unit_l;
			}
			alpha = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, alpha_isothermal_ptr, array_size);
			neg_v = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, neg_v_isothermal_ptr, array_size);
		}
		// const Real m = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, m_isothermal_ptr, array_size);

		const Real rho = std::max(alpha * unit_rho, rho_floor_);
		const Real u = -neg_v * unit_v;
		Real vx = 0.0;
		Real vy = 0.0;
		Real vz = 0.0;
		if (r / dx_fixed > 1.0e-10) {
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
	pp.get("sink_file", sink_file);

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
		fstream << std::scientific << std::setprecision(14) << x[i] << ", " << alpha[i] << ", " << v_abs[i] << ", " << physical_x[i] << ", " << rho[i]
			<< ", " << u[i];
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
	const int array_size = static_cast<int>(x_isothermal.size());
	auto const &x_isothermal_ptr = x_isothermal.data();
	auto const &alpha_isothermal_ptr = alpha_isothermal.data();
	auto const &neg_v_isothermal_ptr = neg_v_isothermal.data();

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
			double xx = x[i];
			if (r < 0.5 * dx0[0]) {
				xx = 0.5 * dx0[0] / unit_l_1;
			}
			alpha_ref[i] = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, alpha_isothermal_ptr, array_size);
			v_ref[i] = interpolate_value<BoundaryPolicy::Clamp>(xx, x_isothermal_ptr, neg_v_isothermal_ptr, array_size);
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
		fstream_final << std::scientific << std::setprecision(14) << x[i] << ", " << alpha[i] << ", " << v_abs[i] << ", " << alpha_ref[i] << ", "
			      << v_ref[i] << ", " << physical_x[i] << ", " << rho[i] << ", " << u[i];
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
	matplotlibcpp::xlim(-2.0 * sphere_radius, 2.0 * sphere_radius);
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
	matplotlibcpp::xlim(-2.0 * sphere_radius, 2.0 * sphere_radius);
	matplotlibcpp::title("t = " + std::to_string(sim.tNew_[0]));
	matplotlibcpp::legend();
	matplotlibcpp::save("particle_isothermal_neg_v_profile.png");
#endif

	return 0;
}
