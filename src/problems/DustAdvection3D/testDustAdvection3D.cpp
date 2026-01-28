/// \file testDustAdvection3D.cpp
/// \brief Defines a 3D test problem for dust transport with drag force
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <fmt/format.h>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct DustAdvection3D {
};

constexpr double initial_Egas = 1.0e-9;
constexpr double rho = 1.0;
constexpr double v0 = 5.0;
constexpr double dust_v0 = 5.0;

template <> struct quokka::EOS_Traits<DustAdvection3D> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DustAdvection3D> {
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = true;
	static constexpr int nDustGroups = 1; // number of dust groups
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
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

	// Gaussian parameters
	const double rho_bg = 1.0;
	const double A = 1.0;	  // amplitude
	const double sigma = 0.1; // width
	const double xc = 0.5;	  // domain center (assuming Lx = Ly = Lz = 1.0)
	const double yc = 0.5;
	const double zc = 0.5;

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
		state_cc(i, j, k, HydroSystem<DustAdvection3D>::energy_index) = Egas0;
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

auto problem_main() -> int
{
	// problem parameters
	const double Lx = 1.0;
	const double Ly = 1.0;
	const double Lz = 1.0;
	const double CFL_number = 0.3;

	// Gaussian parameters
	const double rho_bg = 1.0;
	const double A = 1.0;
	const double sigma = 0.1;
	const double xc = 0.5;
	const double yc = 0.5;
	const double zc = 0.5;

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
	int status = 0;
	auto [x_pos, x_vals] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	auto [y_pos, y_vals] = fextract(sim.state_new_cc_[0], sim.Geom(0), 1, 0.0);
	auto [z_pos, z_vals] = fextract(sim.state_new_cc_[0], sim.Geom(0), 2, 0.0);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// X direction (fixed y and z at center)
		const int nx = static_cast<int>(x_pos.size());

		std::vector<double> vx_sim(nx);
		std::vector<double> vx_exact(nx);
		std::vector<double> vx_dust_sim(nx);
		std::vector<double> vx_dust_exact(nx);
		std::vector<double> rho_dust_sim_x(nx);
		std::vector<double> rho_dust_exact_x(nx);
		std::vector<double> rho_gas_exact_x(nx);
		std::vector<double> rho_gas_sim_x(nx);

		for (int i = 0; i < nx; ++i) {
			const double x = x_pos[i];
			const double t = sim.tNew_[0];

			// exact gas density (shifted by v0 * t in all directions)
			double x_gas_initial = std::fmod(x - v0 * t, Lx);
			double y_gas_initial = std::fmod(0.0 - v0 * t, Ly);
			double z_gas_initial = std::fmod(0.0 - v0 * t, Lz);

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
			rho_gas_exact_x[i] = rho_bg + A * std::exp(-r2_gas / (2.0 * sigma * sigma));

			// exact dust density (shifted by dust_v0 * t in all directions)
			double x_dust_initial = std::fmod(x - dust_v0 * t, Lx);
			double y_dust_initial = std::fmod(0.0 - dust_v0 * t, Ly);
			double z_dust_initial = std::fmod(0.0 - dust_v0 * t, Lz);

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
			rho_dust_exact_x[i] = rho_bg + A * std::exp(-r2_dust / (2.0 * sigma * sigma));

			vx_exact[i] = v0;
			vx_dust_exact[i] = dust_v0;

			// get numerical values from fextract results
			const double density = x_vals[HydroSystem<DustAdvection3D>::density_index][i];
			const double momentum_x = x_vals[HydroSystem<DustAdvection3D>::x1Momentum_index][i];
			const double dust_density = x_vals[HydroSystem<DustAdvection3D>::dustDensity_index][i];
			const double dust_momentum_x = x_vals[HydroSystem<DustAdvection3D>::x1DustMomentum_index][i];

			vx_sim[i] = momentum_x / density;
			vx_dust_sim[i] = dust_momentum_x / dust_density;
			rho_dust_sim_x[i] = dust_density;
			rho_gas_sim_x[i] = density;
		}

		// Y direction (fixed x and z at center)
		const int ny = static_cast<int>(y_pos.size());

		std::vector<double> vy_sim(ny);
		std::vector<double> vy_exact(ny);
		std::vector<double> vy_dust_sim(ny);
		std::vector<double> vy_dust_exact(ny);
		std::vector<double> rho_dust_sim_y(ny);
		std::vector<double> rho_dust_exact_y(ny);
		std::vector<double> rho_gas_exact_y(ny);
		std::vector<double> rho_gas_sim_y(ny);

		for (int j = 0; j < ny; ++j) {
			const double y = y_pos[j];
			const double t = sim.tNew_[0];

			// exact gas density (shifted by v0 * t in all directions)
			double x_gas_initial = std::fmod(0.0 - v0 * t, Lx);
			double y_gas_initial = std::fmod(y - v0 * t, Ly);
			double z_gas_initial = std::fmod(0.0 - v0 * t, Lz);

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
			rho_gas_exact_y[j] = rho_bg + A * std::exp(-r2_gas / (2.0 * sigma * sigma));

			// exact dust density (shifted by dust_v0 * t in all directions)
			double x_dust_initial = std::fmod(0.0 - dust_v0 * t, Lx);
			double y_dust_initial = std::fmod(y - dust_v0 * t, Ly);
			double z_dust_initial = std::fmod(0.0 - dust_v0 * t, Lz);

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
			rho_dust_exact_y[j] = rho_bg + A * std::exp(-r2_dust / (2.0 * sigma * sigma));

			vy_exact[j] = v0;
			vy_dust_exact[j] = dust_v0;

			// get numerical values from fextract results
			const double density = y_vals[HydroSystem<DustAdvection3D>::density_index][j];
			const double momentum_y = y_vals[HydroSystem<DustAdvection3D>::x2Momentum_index][j];
			const double dust_density = y_vals[HydroSystem<DustAdvection3D>::dustDensity_index][j];
			const double dust_momentum_y = y_vals[HydroSystem<DustAdvection3D>::x2DustMomentum_index][j];

			vy_sim[j] = momentum_y / density;
			vy_dust_sim[j] = dust_momentum_y / dust_density;
			rho_dust_sim_y[j] = dust_density;
			rho_gas_sim_y[j] = density;
		}

		// Z direction (fixed x and y at center)
		const int nz = static_cast<int>(z_pos.size());

		std::vector<double> vz_sim(nz);
		std::vector<double> vz_exact(nz);
		std::vector<double> vz_dust_sim(nz);
		std::vector<double> vz_dust_exact(nz);
		std::vector<double> rho_dust_sim_z(nz);
		std::vector<double> rho_dust_exact_z(nz);
		std::vector<double> rho_gas_exact_z(nz);
		std::vector<double> rho_gas_sim_z(nz);

		for (int k = 0; k < nz; ++k) {
			const double z = z_pos[k];
			const double t = sim.tNew_[0];

			// exact gas density (shifted by v0 * t in all directions)
			double x_gas_initial = std::fmod(0.0 - v0 * t, Lx);
			double y_gas_initial = std::fmod(0.0 - v0 * t, Ly);
			double z_gas_initial = std::fmod(z - v0 * t, Lz);

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
			rho_gas_exact_z[k] = rho_bg + A * std::exp(-r2_gas / (2.0 * sigma * sigma));

			// exact dust density (shifted by dust_v0 * t in all directions)
			double x_dust_initial = std::fmod(0.0 - dust_v0 * t, Lx);
			double y_dust_initial = std::fmod(0.0 - dust_v0 * t, Ly);
			double z_dust_initial = std::fmod(z - dust_v0 * t, Lz);

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
			rho_dust_exact_z[k] = rho_bg + A * std::exp(-r2_dust / (2.0 * sigma * sigma));

			vz_exact[k] = v0;
			vz_dust_exact[k] = dust_v0;

			// get numerical values from fextract results
			const double density = z_vals[HydroSystem<DustAdvection3D>::density_index][k];
			const double momentum_z = z_vals[HydroSystem<DustAdvection3D>::x3Momentum_index][k];
			const double dust_density = z_vals[HydroSystem<DustAdvection3D>::dustDensity_index][k];
			const double dust_momentum_z = z_vals[HydroSystem<DustAdvection3D>::x3DustMomentum_index][k];

			vz_sim[k] = momentum_z / density;
			vz_dust_sim[k] = dust_momentum_z / dust_density;
			rho_dust_sim_z[k] = dust_density;
			rho_gas_sim_z[k] = density;
		}

		// X direction errors
		double err_norm_x = 0.;
		double sol_norm_x = 0.;
		for (int i = 0; i < nx; ++i) {
			err_norm_x += std::abs(vx_sim[i] - vx_exact[i]);
			sol_norm_x += std::abs(vx_exact[i]);
		}
		const double rel_err_norm_x = err_norm_x / sol_norm_x;

		double err_norm_dust_rho_x = 0.;
		double sol_norm_dust_rho_x = 0.;
		for (int i = 0; i < nx; ++i) {
			err_norm_dust_rho_x += std::abs(rho_dust_sim_x[i] - rho_dust_exact_x[i]);
			sol_norm_dust_rho_x += std::abs(rho_dust_exact_x[i]);
		}
		const double rel_err_norm_dust_rho_x = err_norm_dust_rho_x / sol_norm_dust_rho_x;

		// Y direction errors
		double err_norm_y = 0.;
		double sol_norm_y = 0.;
		for (int i = 0; i < ny; ++i) {
			err_norm_y += std::abs(vy_sim[i] - vy_exact[i]);
			sol_norm_y += std::abs(vy_exact[i]);
		}
		const double rel_err_norm_y = err_norm_y / sol_norm_y;

		double err_norm_dust_rho_y = 0.;
		double sol_norm_dust_rho_y = 0.;
		for (int i = 0; i < ny; ++i) {
			err_norm_dust_rho_y += std::abs(rho_dust_sim_y[i] - rho_dust_exact_y[i]);
			sol_norm_dust_rho_y += std::abs(rho_dust_exact_y[i]);
		}
		const double rel_err_norm_dust_rho_y = err_norm_dust_rho_y / sol_norm_dust_rho_y;

		// Z direction errors
		double err_norm_z = 0.;
		double sol_norm_z = 0.;
		for (int i = 0; i < nz; ++i) {
			err_norm_z += std::abs(vz_sim[i] - vz_exact[i]);
			sol_norm_z += std::abs(vz_exact[i]);
		}
		const double rel_err_norm_z = err_norm_z / sol_norm_z;

		double err_norm_dust_rho_z = 0.;
		double sol_norm_dust_rho_z = 0.;
		for (int i = 0; i < nz; ++i) {
			err_norm_dust_rho_z += std::abs(rho_dust_sim_z[i] - rho_dust_exact_z[i]);
			sol_norm_dust_rho_z += std::abs(rho_dust_exact_z[i]);
		}
		const double rel_err_norm_dust_rho_z = err_norm_dust_rho_z / sol_norm_dust_rho_z;

		const double rel_err_tol = 0.03;
		if ((rel_err_norm_x > rel_err_tol) || (rel_err_norm_dust_rho_x > rel_err_tol) || (rel_err_norm_y > rel_err_tol) ||
		    (rel_err_norm_dust_rho_y > rel_err_tol) || (rel_err_norm_z > rel_err_tol) || (rel_err_norm_dust_rho_z > rel_err_tol)) {
			status = 1;
		}

		amrex::Print() << "X direction:" << '\n';
		amrex::Print() << "  Relative L1 norm for gas x velocity = " << rel_err_norm_x << '\n';
		amrex::Print() << "  Relative L1 norm for dust density   = " << rel_err_norm_dust_rho_x << '\n';
		amrex::Print() << "Y direction:" << '\n';
		amrex::Print() << "  Relative L1 norm for gas y velocity = " << rel_err_norm_y << '\n';
		amrex::Print() << "  Relative L1 norm for dust density   = " << rel_err_norm_dust_rho_y << '\n';
		amrex::Print() << "Z direction:" << '\n';
		amrex::Print() << "  Relative L1 norm for gas z velocity = " << rel_err_norm_z << '\n';
		amrex::Print() << "  Relative L1 norm for dust density   = " << rel_err_norm_dust_rho_z << '\n';

#ifdef HAVE_PYTHON
		// X direction density
		matplotlibcpp::clf();
		std::map<std::string, std::string> rho_gas_args;
		std::map<std::string, std::string> rho_gas_exact_args;
		std::map<std::string, std::string> rho_dust_args;
		std::map<std::string, std::string> rho_dust_exact_args;

		rho_gas_args["label"] = "gas density (numerical)";
		rho_gas_args["color"] = "r";
		rho_gas_args["linestyle"] = "-";

		rho_gas_exact_args["label"] = "gas density (exact)";
		rho_gas_exact_args["color"] = "r";
		rho_gas_exact_args["linestyle"] = "--";

		rho_dust_args["label"] = "dust density (numerical)";
		rho_dust_args["color"] = "b";
		rho_dust_args["linestyle"] = "-.";

		rho_dust_exact_args["label"] = "dust density (exact)";
		rho_dust_exact_args["color"] = "b";
		rho_dust_exact_args["linestyle"] = ":";

		matplotlibcpp::plot(x_pos, rho_gas_sim_x, rho_gas_args);
		matplotlibcpp::plot(x_pos, rho_gas_exact_x, rho_gas_exact_args);
		matplotlibcpp::plot(x_pos, rho_dust_sim_x, rho_dust_args);
		matplotlibcpp::plot(x_pos, rho_dust_exact_x, rho_dust_exact_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_drag_density_x.pdf");

		// X direction velocity
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 6.0);

		std::map<std::string, std::string> vx_gas_args;
		std::map<std::string, std::string> vx_gas_exact_args;
		std::map<std::string, std::string> vx_dust_args;
		std::map<std::string, std::string> vx_dust_exact_args;

		vx_gas_args["label"] = "gas velocity (numerical)";
		vx_gas_args["color"] = "r";
		vx_gas_args["linestyle"] = "-";

		vx_gas_exact_args["label"] = "gas velocity (exact)";
		vx_gas_exact_args["color"] = "r";
		vx_gas_exact_args["linestyle"] = "--";

		vx_dust_args["label"] = "dust velocity (numerical)";
		vx_dust_args["color"] = "b";
		vx_dust_args["linestyle"] = "-.";

		vx_dust_exact_args["label"] = "dust velocity (exact)";
		vx_dust_exact_args["color"] = "b";
		vx_dust_exact_args["linestyle"] = ":";

		matplotlibcpp::plot(x_pos, vx_sim, vx_gas_args);
		matplotlibcpp::plot(x_pos, vx_exact, vx_gas_exact_args);
		matplotlibcpp::plot(x_pos, vx_dust_sim, vx_dust_args);
		matplotlibcpp::plot(x_pos, vx_dust_exact, vx_dust_exact_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_drag_velocity_x.pdf");

		// Y direction density
		matplotlibcpp::clf();
		matplotlibcpp::plot(y_pos, rho_gas_sim_y, rho_gas_args);
		matplotlibcpp::plot(y_pos, rho_gas_exact_y, rho_gas_exact_args);
		matplotlibcpp::plot(y_pos, rho_dust_sim_y, rho_dust_args);
		matplotlibcpp::plot(y_pos, rho_dust_exact_y, rho_dust_exact_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("y");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_drag_density_y.pdf");

		// Y direction velocity
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 6.0);

		std::map<std::string, std::string> vy_gas_args;
		std::map<std::string, std::string> vy_gas_exact_args;
		std::map<std::string, std::string> vy_dust_args;
		std::map<std::string, std::string> vy_dust_exact_args;

		vy_gas_args["label"] = "gas velocity (numerical)";
		vy_gas_args["color"] = "r";
		vy_gas_args["linestyle"] = "-";

		vy_gas_exact_args["label"] = "gas velocity (exact)";
		vy_gas_exact_args["color"] = "r";
		vy_gas_exact_args["linestyle"] = "--";

		vy_dust_args["label"] = "dust velocity (numerical)";
		vy_dust_args["color"] = "b";
		vy_dust_args["linestyle"] = "-.";

		vy_dust_exact_args["label"] = "dust velocity (exact)";
		vy_dust_exact_args["color"] = "b";
		vy_dust_exact_args["linestyle"] = ":";

		matplotlibcpp::plot(y_pos, vy_sim, vy_gas_args);
		matplotlibcpp::plot(y_pos, vy_exact, vy_gas_exact_args);
		matplotlibcpp::plot(y_pos, vy_dust_sim, vy_dust_args);
		matplotlibcpp::plot(y_pos, vy_dust_exact, vy_dust_exact_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("y");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_drag_velocity_y.pdf");

		// Z direction density
		matplotlibcpp::clf();
		matplotlibcpp::plot(z_pos, rho_gas_sim_z, rho_gas_args);
		matplotlibcpp::plot(z_pos, rho_gas_exact_z, rho_gas_exact_args);
		matplotlibcpp::plot(z_pos, rho_dust_sim_z, rho_dust_args);
		matplotlibcpp::plot(z_pos, rho_dust_exact_z, rho_dust_exact_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("z");
		matplotlibcpp::ylabel("density");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_drag_density_z.pdf");

		// Z direction velocity
		matplotlibcpp::clf();
		matplotlibcpp::ylim(0.0, 6.0);

		std::map<std::string, std::string> vz_gas_args;
		std::map<std::string, std::string> vz_gas_exact_args;
		std::map<std::string, std::string> vz_dust_args;
		std::map<std::string, std::string> vz_dust_exact_args;

		vz_gas_args["label"] = "gas velocity (numerical)";
		vz_gas_args["color"] = "r";
		vz_gas_args["linestyle"] = "-";

		vz_gas_exact_args["label"] = "gas velocity (exact)";
		vz_gas_exact_args["color"] = "r";
		vz_gas_exact_args["linestyle"] = "--";

		vz_dust_args["label"] = "dust velocity (numerical)";
		vz_dust_args["color"] = "b";
		vz_dust_args["linestyle"] = "-.";

		vz_dust_exact_args["label"] = "dust velocity (exact)";
		vz_dust_exact_args["color"] = "b";
		vz_dust_exact_args["linestyle"] = ":";

		matplotlibcpp::plot(z_pos, vz_sim, vz_gas_args);
		matplotlibcpp::plot(z_pos, vz_exact, vz_gas_exact_args);
		matplotlibcpp::plot(z_pos, vz_dust_sim, vz_dust_args);
		matplotlibcpp::plot(z_pos, vz_dust_exact, vz_dust_exact_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("z");
		matplotlibcpp::ylabel("velocity");
		matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dust_drag_velocity_z.pdf");
#endif // HAVE_PYTHON

		amrex::Print() << "Finished." << '\n';
	}
	return status;
}
