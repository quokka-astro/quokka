/// \file testSN.cpp
/// \brief Defines a test problem for supernova feedback.
/// In this test, two supernovae explode and in the end the gas temperature and velocity is checked for
/// Galilean invariance between a rest frame and a boost frame.

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "math/interpolate.hpp"
#include "util/fextract.hpp"
#include <format>
#include <fstream>
#include <iomanip>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "particles/particle_types.hpp"
#include "util/BC.hpp"

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct SNProblem {
};

static bool refine_half_domain = false; // NOLINT

static double max_Eint_global = 0.0;	       // NOLINT
static std::vector<double> max_Eint_history{}; // NOLINT
static std::vector<double> t_history{};	       // NOLINT

static std::string SN_particles_file = "";	    // NOLINT
static std::string coolingTableType_ = "resampled"; // NOLINT

constexpr double mu = 1.0 * C::m_u;
// constexpr double mu = 1.295 * C::m_u; // neutral gas
constexpr double gamma_ = 5. / 3.;
const double CV = 1. / (gamma_ - 1.) / mu * C::k_B;
const double cloudy_H_mass_fraction = 1.0 / (1.0 + 0.1 * 3.971);
const double year = 3.15576e+07; // in seconds
constexpr double B0 = 1.0e-7;	 // uniform background field for MHD variant

static double n_amb = 1.0; // ambient density (g cm^-3) // NOLINT

template <> struct Particle_Traits<SNProblem> : DefaultParticleTraits {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Test;
};

template <> struct quokka::EOS_Traits<SNProblem> {
	static constexpr double gamma = gamma_;
	static constexpr double mean_molecular_weight = mu;
	using EOSBackend = quokka::EOSTabulated<SNProblem>;
};

template <> struct HydroSystem_Traits<SNProblem> {
	static constexpr bool reconstruct_eint = true; // need to reconstruct temperature
};

template <> struct Physics_Traits<SNProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
	// face-centred
	static constexpr bool is_mhd_enabled = true;
};

template <> struct SimulationData<SNProblem> {
	AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity{0.0, 0.0, 0.0};
};

template <> void QuokkaSimulation<SNProblem>::createInitialTestParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 7; // mass vx vy vz birth_time death_time lum
	TestParticles->SetVerbose(0);
	TestParticles->InitFromAsciiFile(SN_particles_file, nreal_extra, nullptr);

	// Loop over all particle at all levels and set first integer component to SNProgenitor
	for (int lev = 0; lev <= TestParticles->finestLevel(); ++lev) {
		auto &particles = TestParticles->GetParticles(lev);

		for (auto &kv : particles) {
			auto &particle_array = kv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();
			auto *pdata = particle_array().data();
			const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity = userData_.boost_velocity;

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.idata(0) = static_cast<int>(quokka::StellarEvolutionStage::SNProgenitor);
				p.rdata(quokka::TestParticleVxIdx) += boost_velocity[0];
				p.rdata(quokka::TestParticleVyIdx) += boost_velocity[1];
				p.rdata(quokka::TestParticleVzIdx) += boost_velocity[2];
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<SNProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	std::vector<double> const n_amb_list = {0.1,	     0.12589254, 0.15848932,  0.19952623,  0.25118864,	0.31622777,  0.39810717,  0.50118723,
						0.63095734,  0.79432823, 1.,	      1.25892541,  1.58489319,	1.99526231,  2.51188643,  3.16227766,
						3.98107171,  5.01187234, 6.30957344,  7.94328235,  10.,		12.58925412, 15.84893192, 19.95262315,
						25.11886432, 31.6227766, 39.81071706, 50.11872336, 63.09573445, 79.43282347, 100.};
	std::vector<double> const T_amb_list = {5802.24013246, 5654.93628944, 5453.30579455, 5213.67068722, 4869.63642975, 4149.54099069, 3907.79953306,
						3583.30375005, 3075.83243575, 1040.16133059, 718.24569694,  664.12747722,  609.83449913,  549.88018839,
						476.83284222,  377.69995731,  359.34890804,  340.22591449,  318.36683039,  289.56818245,  245.82521174,
						236.69671661,  227.20779715,  215.57381444,  200.14862878,  174.5847715,   168.91080447,  162.96050543,
						155.58880555,  144.9653022,   128.3144137};

	// manual clamp
	Real T_amb = NAN;
	if (n_amb <= n_amb_list.front()) {
		T_amb = T_amb_list.front();
	} else if (n_amb >= n_amb_list.back()) {
		T_amb = T_amb_list.back();
	} else {
		T_amb = interpolate_value(n_amb, n_amb_list.data(), T_amb_list.data(), static_cast<int>(n_amb_list.size()));
	}

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const double rho_bg = n_amb * C::m_u / cloudy_H_mass_fraction;
	const double E0 = CV * T_amb * rho_bg;
	const double rho = rho_bg;
	const double rho_e = E0;
	const double Emag = 0.5 * B0 * B0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boost_velocity = userData_.boost_velocity;
	const double v2 = (userData_.boost_velocity[0] * userData_.boost_velocity[0]) + (userData_.boost_velocity[1] * userData_.boost_velocity[1]) +
			  (userData_.boost_velocity[2] * userData_.boost_velocity[2]);

	amrex::Real initial_scalar_density = 0.0;
	if constexpr (Physics_Traits<SNProblem>::numPassiveScalars > 0) {
		const amrex::Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
		initial_scalar_density = 1.0e-6 * quokka::scalar_yield_per_SN / cell_vol;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<SNProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<SNProblem>::x1Momentum_index) = rho * boost_velocity[0];
		state_cc(i, j, k, HydroSystem<SNProblem>::x2Momentum_index) = rho * boost_velocity[1];
		state_cc(i, j, k, HydroSystem<SNProblem>::x3Momentum_index) = rho * boost_velocity[2];
		state_cc(i, j, k, HydroSystem<SNProblem>::energy_index) = rho_e + Emag + 0.5 * rho * v2;
		state_cc(i, j, k, HydroSystem<SNProblem>::internalEnergy_index) = rho_e;

		const auto initial_scalar_density_d = initial_scalar_density;

		// Initialize passive scalar field
		if constexpr (Physics_Traits<SNProblem>::numPassiveScalars > 0) {
			state_cc(i, j, k, HydroSystem<SNProblem>::scalar0_index) = initial_scalar_density_d;
		}
	});
}

template <> void QuokkaSimulation<SNProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<SNProblem>::mhdFirstIndex) = B_val; });
}

template <> void QuokkaSimulation<SNProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement: static mesh refinement for the whole domain (if refine_half_domain is false) or for x > 0 (if refine_half_domain is true)

	auto const &dx = geom[lev].CellSizeArray();
	auto const &plo = geom[lev].ProbLoArray();
	auto const &phi = geom[lev].ProbHiArray();
	const bool refine_half_domain_ = refine_half_domain;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const double x_frac = ((i + 0.5) * dx[0]) / (phi[0] - plo[0]);
			const double y_frac = ((j + 0.5) * dx[1]) / (phi[1] - plo[1]);
			const double z_frac = ((k + 0.5) * dx[2]) / (phi[2] - plo[2]);
			if (!refine_half_domain_ || (x_frac >= 0.7 && x_frac <= 0.8 && y_frac >= 0.3 && y_frac <= 0.7 && z_frac >= 0.3 && z_frac <= 0.7)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <> void QuokkaSimulation<SNProblem>::computeAfterTimestep()
{
	// find the maximum temperature in the state_new_cc_[0]
	const double max_internal_energy_density = state_new_cc_[0].max(HydroSystem<SNProblem>::internalEnergy_index);
	max_Eint_global = std::max(max_Eint_global, max_internal_energy_density);
	max_Eint_history.push_back(max_internal_energy_density);
	t_history.push_back(tNew_[0]);
}

auto problem_main() -> int
{
	// get n_amb from the input file
	amrex::ParmParse const pp("problem");
	pp.query("n_amb", n_amb);
	pp.query("SN_particles_file", SN_particles_file);
	pp.query("refine_half_domain", refine_half_domain);
	double boost_vel_x = 1.0e8;
	pp.query("boost_vel_x", boost_vel_x);

	amrex::ParmParse const cpp("cooling");
	cpp.query("cooling_table_type", coolingTableType_);

	// Problem initialization
	QuokkaSimulation<SNProblem> sim;

	// initialize
	sim.setInitialConditions();

	// Compute initial scalar quantities for validation
	amrex::Real total_scalar_init = 0.0;
	amrex::Real initial_scalar_density = 0.0;
	if constexpr (Physics_Traits<SNProblem>::numPassiveScalars > 0) {
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx0 = sim.geom[0].CellSizeArray();
		const amrex::Real vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
		total_scalar_init = sim.state_new_cc_[0].sum(HydroSystem<SNProblem>::scalar0_index) * vol;
		initial_scalar_density = 1.0e-6 * quokka::scalar_yield_per_SN / vol;
	}

	// evolve
	sim.evolve();

	// Validate passive scalar conservation and peak enhancement
	int scalar_validation_status = 0;
	if constexpr (Physics_Traits<SNProblem>::numPassiveScalars > 0) {
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx0 = sim.geom[0].CellSizeArray();
		const amrex::Real vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
		const amrex::Real total_scalar_final = sim.state_new_cc_[0].sum(HydroSystem<SNProblem>::scalar0_index) * vol;
		const amrex::Real peak_scalar = sim.state_new_cc_[0].max(HydroSystem<SNProblem>::scalar0_index);

		// Expected: initial + 2 SNe × scalar_yield_per_SN (there are 2 SNe in the test)
		const int num_sn = 2;
		const amrex::Real expected_total_scalar = total_scalar_init + num_sn * quokka::scalar_yield_per_SN;
		const amrex::Real scalar_conservation_error = std::abs(total_scalar_final - expected_total_scalar) / expected_total_scalar;

		// Check peak enhancement (should be > 10× initial density)
		const amrex::Real peak_enhancement_factor = peak_scalar / initial_scalar_density;

		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Print() << "\n=== Passive Scalar Validation ===\n";
			amrex::Print() << "Initial total scalar: " << total_scalar_init << "\n";
			amrex::Print() << "Final total scalar: " << total_scalar_final << "\n";
			amrex::Print() << "Expected total scalar: " << expected_total_scalar << "\n";
			amrex::Print() << "Conservation error: " << scalar_conservation_error << "\n";
			amrex::Print() << "Peak scalar density: " << peak_scalar << "\n";
			amrex::Print() << "Initial scalar density: " << initial_scalar_density << "\n";
			amrex::Print() << "Peak enhancement factor: " << peak_enhancement_factor << "\n";

			// Validate conservation (< 1e-10 relative error)
			if (scalar_conservation_error > 1.0e-10) {
				amrex::Print() << "FAIL: Scalar conservation error too large!\n";
				scalar_validation_status = 1;
			} else {
				amrex::Print() << "PASS: Scalar conservation\n";
			}

			// Validate peak enhancement (> 10× initial)
			if (peak_enhancement_factor < 10.0) {
				amrex::Print() << "FAIL: Peak scalar enhancement too small!\n";
				scalar_validation_status = 1;
			} else {
				amrex::Print() << "PASS: Peak scalar enhancement\n";
			}
		}
	}

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0, true);
	const int nx = static_cast<int>(position.size());

	std::vector<double> T(nx);
	std::vector<double> x(nx);
	std::vector<double> vx(nx);

	// extract the temperature and vx profile along the x axis at the center
	if (amrex::ParallelDescriptor::IOProcessor()) {
		for (int i = 0; i < nx; ++i) {
			const double rho = values.at(HydroSystem<SNProblem>::density_index)[i];
			const double Eint = values.at(HydroSystem<SNProblem>::internalEnergy_index)[i];
			const double vx_val = values.at(HydroSystem<SNProblem>::x1Momentum_index)[i] / rho;
			T[i] = Eint / (rho * CV); // simplified, but good enough for the purpose
			x[i] = position[i];
			vx[i] = vx_val;
		}
	}

	QuokkaSimulation<SNProblem> sim2;
	// set boosted velocity
	sim2.userData_.boost_velocity = {boost_vel_x, 0.0, 0.0};

	// initialize
	sim2.setInitialConditions();

	// evolve
	sim2.evolve();

	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0, true);

	// compute the spatial shift due to the boost velocity
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo2 = sim2.geom[0].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi2 = sim2.geom[0].ProbHiArray();
	const double dx2 = (prob_hi2[0] - prob_lo2[0]) / static_cast<double>(nx);
	const double move = boost_vel_x * sim2.tNew_[0];
	const int n_p = static_cast<int>(move / dx2);
	const int half = static_cast<int>(nx / 2.0);
	const double drift = move - static_cast<double>(n_p) * dx2;
	const int shift = n_p - static_cast<int>((n_p + half) / nx) * nx;

	std::vector<double> T2(nx);
	std::vector<double> x2(nx);
	std::vector<double> vx2_rel(nx);

	int status = 0;

	// validate Galilean invariance and optionally plot profiles
	if (amrex::ParallelDescriptor::IOProcessor()) {
		double v_value_norm = 0.0;
		double v_err_norm = 0.0;
		double T_value_norm = 0.0;
		double T_err_norm = 0.0;
		for (int i = 0; i < nx; ++i) {
			// compute the remapped index to account for spatial shift
			int index_ = 0;
			if (shift >= 0) {
				if (i < shift) {
					index_ = nx - shift + i;
				} else {
					index_ = i - shift;
				}
			} else {
				if (i <= nx - 1 + shift) {
					index_ = i - shift;
				} else {
					index_ = i - (nx + shift);
				}
			}
			const double rho = values2.at(HydroSystem<SNProblem>::density_index)[i];
			const double Eint = values2.at(HydroSystem<SNProblem>::internalEnergy_index)[i];
			const double vx_val = values2.at(HydroSystem<SNProblem>::x1Momentum_index)[i] / rho;
			T2[index_] = Eint / (rho * CV); // simplified, but good enough for the purpose
			x2[i] = position2[i] - drift;
			vx2_rel[index_] = vx_val - boost_vel_x;
			v_value_norm += std::abs(vx_val); // use raw vx_val to account for the large boost velocity
			v_err_norm += std::abs(vx2_rel[index_] - vx[index_]);
			T_value_norm += std::abs(T[index_]);
			T_err_norm += std::abs(T2[index_] - T[index_]);
		}
		const double v_rel_err_norm = v_err_norm / v_value_norm;
		const double T_rel_err_norm = T_err_norm / T_value_norm;
		const double v_rel_err_tol = 0.01;
		const double T_rel_err_tol = 0.15;
		amrex::Print() << std::format("Relative L1 norm for vx = {}, tolerance = {}\n", v_rel_err_norm, v_rel_err_tol);
		amrex::Print() << std::format("Relative L1 norm for T = {}, tolerance = {}\n", T_rel_err_norm, T_rel_err_tol);
		if (!(v_rel_err_norm < v_rel_err_tol) || !(T_rel_err_norm < T_rel_err_tol)) {
			amrex::Print() << "FAIL: Velocity or T error not within tolerance\n";
			status = 1;
		}

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		matplotlibcpp::plot(x, T, {{"label", "base"}, {"color", "C0"}});
		matplotlibcpp::plot(x2, T2, {{"label", "boosted"}, {"color", "C1"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x (cm)");
		matplotlibcpp::ylabel("T (K)");
		matplotlibcpp::title(std::format("time t = {:.4g}", sim2.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(std::format("sn_temperature_profile_n0_{:.1g}.pdf", n_amb));

		matplotlibcpp::clf();
		matplotlibcpp::plot(x, vx, {{"label", "base"}, {"color", "C0"}});
		matplotlibcpp::plot(x2, vx2_rel, {{"label", "boosted"}, {"color", "C1"}, {"linestyle", "--"}});
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x (cm)");
		matplotlibcpp::ylabel("vx (cm/s)");
		matplotlibcpp::title(std::format("time t = {:.4g}", sim2.tNew_[0]));
		matplotlibcpp::save(std::format("sn_velocity_profile_n0_{:.1g}_boost_vel_{:.1g}.pdf", n_amb, boost_vel_x));
#endif
	}

	// Return combined status (Galilean invariance + scalar validation)
	return status + scalar_validation_status;
}
