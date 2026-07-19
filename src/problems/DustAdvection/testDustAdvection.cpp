/// \file testDustAdvection.cpp
/// \brief Defines a test problem for dust transport with drag force
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <format>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct DustAdvection {
};

constexpr double initial_Egas = 1.0e-9;
constexpr double rho = 1.0;
constexpr double v0 = 5.0;
constexpr double dust_v0 = 5.0;

// Gaussian parameters
constexpr double rho_bg = 1.0;
constexpr double A = 1.0;     // amplitude
constexpr double sigma = 0.1; // width
constexpr double Lx = 1.0;    // domain size
constexpr double xc = 0.5;    // domain center

template <> struct quokka::EOS_Traits<DustAdvection> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DustAdvection> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<DustAdvection>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Egas0 = initial_Egas;
	const auto vx0 = v0;	      // gas velocity
	const auto vx_dust = dust_v0; // dust velocity

	// get geometry information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = Geom(0).CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = Geom(0).ProbLoArray();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

		// Gaussian + background for gas
		amrex::Real const rho_gas_local = rho_bg + A * std::exp(-((x - xc) * (x - xc)) / (2.0 * sigma * sigma));
		amrex::Real const kinetic_energy = 0.5 * rho_gas_local * vx0 * vx0;
		state_cc(i, j, k, HydroSystem<DustAdvection>::density_index) = rho_gas_local;
		state_cc(i, j, k, HydroSystem<DustAdvection>::energy_index) = Egas0 + kinetic_energy;
		state_cc(i, j, k, HydroSystem<DustAdvection>::internalEnergy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<DustAdvection>::x1Momentum_index) = rho_gas_local * vx0;
		state_cc(i, j, k, HydroSystem<DustAdvection>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustAdvection>::x3Momentum_index) = 0.;

		// Compute dust values before constexpr-if to ensure proper capture
		// Gaussian + background for dust
		amrex::Real const rho_dust_local = rho_bg + A * std::exp(-((x - xc) * (x - xc)) / (2.0 * sigma * sigma));
		// Reference vx_dust before constexpr-if to ensure proper capture
		amrex::Real const vx_dust_local = vx_dust;

		if constexpr (Physics_Traits<DustAdvection>::is_dust_enabled) {
			state_cc(i, j, k, HydroSystem<DustAdvection>::dustDensity_index) = rho_dust_local;
			state_cc(i, j, k, HydroSystem<DustAdvection>::x1DustMomentum_index) = rho_dust_local * vx_dust_local;
			state_cc(i, j, k, HydroSystem<DustAdvection>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustAdvection>::x3DustMomentum_index) = 0.;
		}
	});
}

template <>
void QuokkaSimulation<DustAdvection>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const auto Egas0 = initial_Egas;
	const auto vx0 = v0;
	const auto vx_dust = dust_v0;
	const amrex::Real t = tNew_[0];

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

			// exact gas density (shifted by v0 * t)
			amrex::Real x_gas_initial = x - vx0 * t;
			x_gas_initial = std::fmod(x_gas_initial, Lx);
			if (x_gas_initial < 0.0) {
				x_gas_initial += Lx;
			}
			amrex::Real const rho_gas_exact = rho_bg + A * std::exp(-((x_gas_initial - xc) * (x_gas_initial - xc)) / (2.0 * sigma * sigma));

			// exact dust density (shifted by dust_v0 * t)
			amrex::Real x_dust_initial = x - vx_dust * t;
			x_dust_initial = std::fmod(x_dust_initial, Lx);
			if (x_dust_initial < 0.0) {
				x_dust_initial += Lx;
			}
			amrex::Real const rho_dust_exact = rho_bg + A * std::exp(-((x_dust_initial - xc) * (x_dust_initial - xc)) / (2.0 * sigma * sigma));

			// clear all components
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			// fill gas components
			stateExact(i, j, k, HydroSystem<DustAdvection>::density_index) = rho_gas_exact;
			stateExact(i, j, k, HydroSystem<DustAdvection>::energy_index) = Egas0 + 0.5 * rho_gas_exact * vx0 * vx0;
			stateExact(i, j, k, HydroSystem<DustAdvection>::internalEnergy_index) = Egas0;
			stateExact(i, j, k, HydroSystem<DustAdvection>::x1Momentum_index) = rho_gas_exact * vx0;
			stateExact(i, j, k, HydroSystem<DustAdvection>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<DustAdvection>::x3Momentum_index) = 0.;

			// fill dust components
			if constexpr (Physics_Traits<DustAdvection>::is_dust_enabled) {
				stateExact(i, j, k, HydroSystem<DustAdvection>::dustDensity_index) = rho_dust_exact;
				stateExact(i, j, k, HydroSystem<DustAdvection>::x1DustMomentum_index) = rho_dust_exact * vx_dust;
				stateExact(i, j, k, HydroSystem<DustAdvection>::x2DustMomentum_index) = 0.;
				stateExact(i, j, k, HydroSystem<DustAdvection>::x3DustMomentum_index) = 0.;
			}
		});
	}

#ifdef HAVE_PYTHON
	// plot results
	auto [position, values] = fextract(state_new_cc_[0], geom[0], 0, 0.0);
	auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.0);
	const int nx = static_cast<int>(position.size());

	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> vx_gas_sim(nx);
		std::vector<double> vx_gas_exact(nx);
		std::vector<double> rho_gas_sim(nx);
		std::vector<double> rho_gas_exact(nx);
		std::vector<double> vx_dust_sim(nx);
		std::vector<double> vx_dust_exact(nx);
		std::vector<double> rho_dust_sim(nx);
		std::vector<double> rho_dust_exact(nx);

		for (int i = 0; i < nx; ++i) {
			xs.at(i) = position[i];

			// gas
			rho_gas_sim.at(i) = values.at(HydroSystem<DustAdvection>::density_index)[i];
			vx_gas_sim.at(i) = values.at(HydroSystem<DustAdvection>::x1Momentum_index)[i] / rho_gas_sim.at(i);

			rho_gas_exact.at(i) = val_exact.at(HydroSystem<DustAdvection>::density_index)[i];
			vx_gas_exact.at(i) = val_exact.at(HydroSystem<DustAdvection>::x1Momentum_index)[i] / rho_gas_exact.at(i);

			// dust
			rho_dust_sim.at(i) = values.at(HydroSystem<DustAdvection>::dustDensity_index)[i];
			vx_dust_sim.at(i) = values.at(HydroSystem<DustAdvection>::x1DustMomentum_index)[i] / rho_dust_sim.at(i);

			rho_dust_exact.at(i) = val_exact.at(HydroSystem<DustAdvection>::dustDensity_index)[i];
			vx_dust_exact.at(i) = val_exact.at(HydroSystem<DustAdvection>::x1DustMomentum_index)[i] / rho_dust_exact.at(i);
		}

		// plot density
		matplotlibcpp::clf();
		std::map<std::string, std::string> rho_gas_args;
		std::map<std::string, std::string> rho_gas_exact_args;
		std::map<std::string, std::string> rho_dust_args;
		std::map<std::string, std::string> rho_dust_exact_args;

		rho_gas_args = {{"label", "gas density (numerical)"}, {"color", "r"}, {"linestyle", "-"}};
		rho_gas_exact_args = {{"label", "gas density (exact)"}, {"color", "r"}, {"linestyle", "--"}};
		rho_dust_args = {{"label", "dust density (numerical)"}, {"color", "b"}, {"linestyle", "-."}};
		rho_dust_exact_args = {{"label", "dust density (exact)"}, {"color", "b"}, {"linestyle", ":"}};

		matplotlibcpp::plot(xs, rho_gas_sim, rho_gas_args);
		matplotlibcpp::plot(xs, rho_gas_exact, rho_gas_exact_args);
		matplotlibcpp::plot(xs, rho_dust_sim, rho_dust_args);
		matplotlibcpp::plot(xs, rho_dust_exact, rho_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(std::format("t = {:.4f}", tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_density.pdf");

		// plot velocity
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 6.0);
		std::map<std::string, std::string> vx_gas_args;
		std::map<std::string, std::string> vx_gas_exact_args;
		std::map<std::string, std::string> vx_dust_args;
		std::map<std::string, std::string> vx_dust_exact_args;

		vx_gas_args = {{"label", "gas velocity (numerical)"}, {"color", "r"}, {"linestyle", "-"}};
		vx_gas_exact_args = {{"label", "gas velocity (exact)"}, {"color", "r"}, {"linestyle", "--"}};
		vx_dust_args = {{"label", "dust velocity (numerical)"}, {"color", "b"}, {"linestyle", "-."}};
		vx_dust_exact_args = {{"label", "dust velocity (exact)"}, {"color", "b"}, {"linestyle", ":"}};

		matplotlibcpp::plot(xs, vx_gas_sim, vx_gas_args);
		matplotlibcpp::plot(xs, vx_gas_exact, vx_gas_exact_args);
		matplotlibcpp::plot(xs, vx_dust_sim, vx_dust_args);
		matplotlibcpp::plot(xs, vx_dust_exact, vx_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(std::format("t = {:.4f}", tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_velocity.pdf");
	}
#endif
}

auto problem_main() -> int
{
	// problem parameters
	const double CFL_number = 0.8;

	// problem initialization
	QuokkaSimulation<DustAdvection> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const double rel_err_tol = 0.03;
	int status = 0;
	const double error_norm = sim.computeErrorNorm();
	if (error_norm > rel_err_tol) {
		status = 1;
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
