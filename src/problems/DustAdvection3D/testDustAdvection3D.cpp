/// \file testDustAdvection3D.cpp
/// \brief Defines a 3D test problem for dust transport with drag force
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <format>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct DustAdvection3D {
};

constexpr double initial_Egas = 1.0e-9;
constexpr double v0 = 5.0;
constexpr double dust_v0 = 5.0;

// Gaussian parameters
constexpr double rho_bg = 1.0;
constexpr double A = 1.0;     // amplitude
constexpr double sigma = 0.1; // width
constexpr double xc = 0.5;    // domain center
constexpr double yc = 0.5;
constexpr double zc = 0.5;
constexpr double Lx = 1.0; // domain size
constexpr double Ly = 1.0;
constexpr double Lz = 1.0;

template <> struct quokka::EOS_Traits<DustAdvection3D> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DustAdvection3D> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> void QuokkaSimulation<DustAdvection3D>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Egas0 = initial_Egas;
	const auto v_gas = v0;	     // gas velocity in all directions
	const auto v_dust = dust_v0; // dust velocity in all directions

	// get geometry information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = Geom(0).CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = Geom(0).ProbLoArray();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];

		// 3D Gaussian + background for gas
		double const r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc) + (z - zc) * (z - zc);
		amrex::Real const rho_gas_local = rho_bg + A * std::exp(-r2 / (2.0 * sigma * sigma));

		state_cc(i, j, k, HydroSystem<DustAdvection3D>::density_index) = rho_gas_local;
		state_cc(i, j, k, HydroSystem<DustAdvection3D>::energy_index) = Egas0 + 0.5 * rho_gas_local * (v_gas * v_gas + v_gas * v_gas + v_gas * v_gas);
		state_cc(i, j, k, HydroSystem<DustAdvection3D>::internalEnergy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<DustAdvection3D>::x1Momentum_index) = rho_gas_local * v_gas;
		state_cc(i, j, k, HydroSystem<DustAdvection3D>::x2Momentum_index) = rho_gas_local * v_gas;
		state_cc(i, j, k, HydroSystem<DustAdvection3D>::x3Momentum_index) = rho_gas_local * v_gas;

		// 3D Gaussian + background for dust
		amrex::Real const rho_dust_local = rho_bg + A * std::exp(-r2 / (2.0 * sigma * sigma));
		amrex::Real const v_dust_local = v_dust;

		if constexpr (Physics_Traits<DustAdvection3D>::is_dust_enabled) {
			state_cc(i, j, k, HydroSystem<DustAdvection3D>::dustDensity_index) = rho_dust_local;
			state_cc(i, j, k, HydroSystem<DustAdvection3D>::x1DustMomentum_index) = rho_dust_local * v_dust_local;
			state_cc(i, j, k, HydroSystem<DustAdvection3D>::x2DustMomentum_index) = rho_dust_local * v_dust_local;
			state_cc(i, j, k, HydroSystem<DustAdvection3D>::x3DustMomentum_index) = rho_dust_local * v_dust_local;
		}
	});
}

template <>
void QuokkaSimulation<DustAdvection3D>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const auto Egas0 = initial_Egas;
	const auto v_gas = v0;
	const auto v_dust = dust_v0;
	const amrex::Real t = tNew_[0];

	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
			amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];

			// exact gas density (shifted by v_gas * t in all directions)
			double x_gas_initial = std::fmod(x - v_gas * t, Lx);
			double y_gas_initial = std::fmod(y - v_gas * t, Ly);
			double z_gas_initial = std::fmod(z - v_gas * t, Lz);

			if (x_gas_initial < 0.0) {
				x_gas_initial += Lx;
			}
			if (y_gas_initial < 0.0) {
				y_gas_initial += Ly;
			}
			if (z_gas_initial < 0.0) {
				z_gas_initial += Lz;
			}

			const double r2_gas = (x_gas_initial - xc) * (x_gas_initial - xc) + (y_gas_initial - yc) * (y_gas_initial - yc) +
					      (z_gas_initial - zc) * (z_gas_initial - zc);
			amrex::Real const rho_gas_exact = rho_bg + A * std::exp(-r2_gas / (2.0 * sigma * sigma));

			// exact dust density (shifted by v_dust * t in all directions)
			double x_dust_initial = std::fmod(x - v_dust * t, Lx);
			double y_dust_initial = std::fmod(y - v_dust * t, Ly);
			double z_dust_initial = std::fmod(z - v_dust * t, Lz);

			if (x_dust_initial < 0.0) {
				x_dust_initial += Lx;
			}
			if (y_dust_initial < 0.0) {
				y_dust_initial += Ly;
			}
			if (z_dust_initial < 0.0) {
				z_dust_initial += Lz;
			}

			const double r2_dust = (x_dust_initial - xc) * (x_dust_initial - xc) + (y_dust_initial - yc) * (y_dust_initial - yc) +
					       (z_dust_initial - zc) * (z_dust_initial - zc);
			amrex::Real const rho_dust_exact = rho_bg + A * std::exp(-r2_dust / (2.0 * sigma * sigma));

			// clear all components
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			// fill gas components
			stateExact(i, j, k, HydroSystem<DustAdvection3D>::density_index) = rho_gas_exact;
			stateExact(i, j, k, HydroSystem<DustAdvection3D>::energy_index) =
			    Egas0 + 0.5 * rho_gas_exact * (v_gas * v_gas + v_gas * v_gas + v_gas * v_gas);
			stateExact(i, j, k, HydroSystem<DustAdvection3D>::internalEnergy_index) = Egas0;
			stateExact(i, j, k, HydroSystem<DustAdvection3D>::x1Momentum_index) = rho_gas_exact * v_gas;
			stateExact(i, j, k, HydroSystem<DustAdvection3D>::x2Momentum_index) = rho_gas_exact * v_gas;
			stateExact(i, j, k, HydroSystem<DustAdvection3D>::x3Momentum_index) = rho_gas_exact * v_gas;

			// fill dust components
			if constexpr (Physics_Traits<DustAdvection3D>::is_dust_enabled) {
				stateExact(i, j, k, HydroSystem<DustAdvection3D>::dustDensity_index) = rho_dust_exact;
				stateExact(i, j, k, HydroSystem<DustAdvection3D>::x1DustMomentum_index) = rho_dust_exact * v_dust;
				stateExact(i, j, k, HydroSystem<DustAdvection3D>::x2DustMomentum_index) = rho_dust_exact * v_dust;
				stateExact(i, j, k, HydroSystem<DustAdvection3D>::x3DustMomentum_index) = rho_dust_exact * v_dust;
			}
		});
	}

#ifdef HAVE_PYTHON
	auto [x_pos, x_vals] = fextract(state_new_cc_[0], geom[0], 0, 0.5);
	auto [x_pos_exact, x_vals_exact] = fextract(ref, geom[0], 0, 0.5);
	auto [y_pos, y_vals] = fextract(state_new_cc_[0], geom[0], 1, 0.5);
	auto [y_pos_exact, y_vals_exact] = fextract(ref, geom[0], 1, 0.5);
	auto [z_pos, z_vals] = fextract(state_new_cc_[0], geom[0], 2, 0.5);
	auto [z_pos_exact, z_vals_exact] = fextract(ref, geom[0], 2, 0.5);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// x directionss
		const int nx = static_cast<int>(x_pos.size());
		std::vector<double> vx_sim(nx);
		std::vector<double> vx_exact(nx);
		std::vector<double> rho_gas_sim_x(nx);
		std::vector<double> rho_gas_exact_x(nx);
		std::vector<double> vx_dust_sim(nx);
		std::vector<double> vx_dust_exact(nx);
		std::vector<double> rho_dust_sim_x(nx);
		std::vector<double> rho_dust_exact_x(nx);

		for (int i = 0; i < nx; ++i) {
			rho_gas_sim_x[i] = x_vals[HydroSystem<DustAdvection3D>::density_index][i];
			vx_sim[i] = x_vals[HydroSystem<DustAdvection3D>::x1Momentum_index][i] / rho_gas_sim_x[i];
			rho_gas_exact_x[i] = x_vals_exact[HydroSystem<DustAdvection3D>::density_index][i];
			vx_exact[i] = x_vals_exact[HydroSystem<DustAdvection3D>::x1Momentum_index][i] / rho_gas_exact_x[i];

			rho_dust_sim_x[i] = x_vals[HydroSystem<DustAdvection3D>::dustDensity_index][i];
			vx_dust_sim[i] = x_vals[HydroSystem<DustAdvection3D>::x1DustMomentum_index][i] / rho_dust_sim_x[i];
			rho_dust_exact_x[i] = x_vals_exact[HydroSystem<DustAdvection3D>::dustDensity_index][i];
			vx_dust_exact[i] = x_vals_exact[HydroSystem<DustAdvection3D>::x1DustMomentum_index][i] / rho_dust_exact_x[i];
		}

		// y direction
		const int ny = static_cast<int>(y_pos.size());
		std::vector<double> vy_sim(ny);
		std::vector<double> vy_exact(ny);
		std::vector<double> rho_gas_sim_y(ny);
		std::vector<double> rho_gas_exact_y(ny);
		std::vector<double> vy_dust_sim(ny);
		std::vector<double> vy_dust_exact(ny);
		std::vector<double> rho_dust_sim_y(ny);
		std::vector<double> rho_dust_exact_y(ny);

		for (int j = 0; j < ny; ++j) {
			rho_gas_sim_y[j] = y_vals[HydroSystem<DustAdvection3D>::density_index][j];
			vy_sim[j] = y_vals[HydroSystem<DustAdvection3D>::x2Momentum_index][j] / rho_gas_sim_y[j];
			rho_gas_exact_y[j] = y_vals_exact[HydroSystem<DustAdvection3D>::density_index][j];
			vy_exact[j] = y_vals_exact[HydroSystem<DustAdvection3D>::x2Momentum_index][j] / rho_gas_exact_y[j];

			rho_dust_sim_y[j] = y_vals[HydroSystem<DustAdvection3D>::dustDensity_index][j];
			vy_dust_sim[j] = y_vals[HydroSystem<DustAdvection3D>::x2DustMomentum_index][j] / rho_dust_sim_y[j];
			rho_dust_exact_y[j] = y_vals_exact[HydroSystem<DustAdvection3D>::dustDensity_index][j];
			vy_dust_exact[j] = y_vals_exact[HydroSystem<DustAdvection3D>::x2DustMomentum_index][j] / rho_dust_exact_y[j];
		}

		// z direction
		const int nz = static_cast<int>(z_pos.size());
		std::vector<double> vz_sim(nz);
		std::vector<double> vz_exact(nz);
		std::vector<double> rho_gas_sim_z(nz);
		std::vector<double> rho_gas_exact_z(nz);
		std::vector<double> vz_dust_sim(nz);
		std::vector<double> vz_dust_exact(nz);
		std::vector<double> rho_dust_sim_z(nz);
		std::vector<double> rho_dust_exact_z(nz);

		for (int k = 0; k < nz; ++k) {
			rho_gas_sim_z[k] = z_vals[HydroSystem<DustAdvection3D>::density_index][k];
			vz_sim[k] = z_vals[HydroSystem<DustAdvection3D>::x3Momentum_index][k] / rho_gas_sim_z[k];
			rho_gas_exact_z[k] = z_vals_exact[HydroSystem<DustAdvection3D>::density_index][k];
			vz_exact[k] = z_vals_exact[HydroSystem<DustAdvection3D>::x3Momentum_index][k] / rho_gas_exact_z[k];

			rho_dust_sim_z[k] = z_vals[HydroSystem<DustAdvection3D>::dustDensity_index][k];
			vz_dust_sim[k] = z_vals[HydroSystem<DustAdvection3D>::x3DustMomentum_index][k] / rho_dust_sim_z[k];
			rho_dust_exact_z[k] = z_vals_exact[HydroSystem<DustAdvection3D>::dustDensity_index][k];
			vz_dust_exact[k] = z_vals_exact[HydroSystem<DustAdvection3D>::x3DustMomentum_index][k] / rho_dust_exact_z[k];
		}

		// common styles
		std::map<std::string, std::string> const rho_gas_args{{"label", "gas density (numerical)"}, {"color", "r"}, {"linestyle", "-"}};
		std::map<std::string, std::string> const rho_gas_exact_args{{"label", "gas density (exact)"}, {"color", "r"}, {"linestyle", "--"}};
		std::map<std::string, std::string> const rho_dust_args{{"label", "dust density (numerical)"}, {"color", "b"}, {"linestyle", "-."}};
		std::map<std::string, std::string> const rho_dust_exact_args{{"label", "dust density (exact)"}, {"color", "b"}, {"linestyle", ":"}};

		std::map<std::string, std::string> const v_gas_args{{"label", "gas velocity (numerical)"}, {"color", "r"}, {"linestyle", "-"}};
		std::map<std::string, std::string> const v_gas_exact_args{{"label", "gas velocity (exact)"}, {"color", "r"}, {"linestyle", "--"}};
		std::map<std::string, std::string> const v_dust_args{{"label", "dust velocity (numerical)"}, {"color", "b"}, {"linestyle", "-."}};
		std::map<std::string, std::string> const v_dust_exact_args{{"label", "dust velocity (exact)"}, {"color", "b"}, {"linestyle", ":"}};

		// plot x
		matplotlibcpp::clf();
		matplotlibcpp::plot(x_pos, rho_gas_sim_x, rho_gas_args);
		matplotlibcpp::plot(x_pos, rho_gas_exact_x, rho_gas_exact_args);
		matplotlibcpp::plot(x_pos, rho_dust_sim_x, rho_dust_args);
		matplotlibcpp::plot(x_pos, rho_dust_exact_x, rho_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(std::format("t = {:.4f}", t));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_3d_density_x.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 6.0);
		matplotlibcpp::plot(x_pos, vx_sim, v_gas_args);
		matplotlibcpp::plot(x_pos, vx_exact, v_gas_exact_args);
		matplotlibcpp::plot(x_pos, vx_dust_sim, v_dust_args);
		matplotlibcpp::plot(x_pos, vx_dust_exact, v_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(std::format("t = {:.4f}", t));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_3d_velocity_x.pdf");

		// plot y
		matplotlibcpp::clf();
		matplotlibcpp::plot(y_pos, rho_gas_sim_y, rho_gas_args);
		matplotlibcpp::plot(y_pos, rho_gas_exact_y, rho_gas_exact_args);
		matplotlibcpp::plot(y_pos, rho_dust_sim_y, rho_dust_args);
		matplotlibcpp::plot(y_pos, rho_dust_exact_y, rho_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("y");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(std::format("t = {:.4f}", t));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_3d_density_y.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 6.0);
		matplotlibcpp::plot(y_pos, vy_sim, v_gas_args);
		matplotlibcpp::plot(y_pos, vy_exact, v_gas_exact_args);
		matplotlibcpp::plot(y_pos, vy_dust_sim, v_dust_args);
		matplotlibcpp::plot(y_pos, vy_dust_exact, v_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("y");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(std::format("t = {:.4f}", t));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_3d_velocity_y.pdf");

		// plot z
		matplotlibcpp::clf();
		matplotlibcpp::plot(z_pos, rho_gas_sim_z, rho_gas_args);
		matplotlibcpp::plot(z_pos, rho_gas_exact_z, rho_gas_exact_args);
		matplotlibcpp::plot(z_pos, rho_dust_sim_z, rho_dust_args);
		matplotlibcpp::plot(z_pos, rho_dust_exact_z, rho_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("z");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(std::format("t = {:.4f}", t));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_3d_density_z.pdf");

		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 6.0);
		matplotlibcpp::plot(z_pos, vz_sim, v_gas_args);
		matplotlibcpp::plot(z_pos, vz_exact, v_gas_exact_args);
		matplotlibcpp::plot(z_pos, vz_dust_sim, v_dust_args);
		matplotlibcpp::plot(z_pos, vz_dust_exact, v_dust_exact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("z");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(std::format("t = {:.4f}", t));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_advection_3d_velocity_z.pdf");
	}
#endif
}

auto problem_main() -> int
{
	// problem parameters
	const double CFL_number = 0.3;

	// problem initialization
	QuokkaSimulation<DustAdvection3D> sim;

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
