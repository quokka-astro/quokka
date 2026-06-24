/// \file testDustyShock.cpp
/// \brief Dust–gas drag test with analytic comparison.

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <format>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

constexpr double shock_position_init = 4.0;
constexpr double rho_gas_left = 1.0;
constexpr double vel_gas_left = 2.0;
constexpr double rho_gas_right = 8.0;
constexpr double vel_gas_right = 0.25;
constexpr double rho_dust_left = 1.0;
constexpr double vel_dust_left = 2.0;
constexpr double cs_isothermal = 1.0;
constexpr double drag_coefficient = 1.0; // K1 in the analytic solution

struct DustyShock {
};

template <> struct quokka::EOS_Traits<DustyShock> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // only used when gamma = 1
};

template <> struct Physics_Traits<DustyShock> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numPassiveScalars = 0;
	static constexpr bool is_dust_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustyShock>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d,
										  amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, nDustGroups_> alpha{};
	for (int g = 0; g < nDustGroups_; ++g) {
		alpha[g] = 1.0 / rho_d[g];
	}
	return alpha;
}

template <> void QuokkaSimulation<DustyShock>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = Geom(0).CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = Geom(0).ProbLoArray();

	// shock tube initial conditions
	const double shock_position = shock_position_init;
	const double rho_left = rho_gas_left;
	const double u_left = vel_gas_left;
	const double rho_right = rho_gas_right;
	const double u_right = vel_gas_right;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		const bool left = (x < shock_position);

		double const rho = left ? rho_left : rho_right;
		double const u = left ? u_left : u_right;
		double const E = 0.5 * rho * u * u; // kinetic-only (isothermal setup)

		state_cc(i, j, k, HydroSystem<DustyShock>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<DustyShock>::energy_index) = E;
		state_cc(i, j, k, HydroSystem<DustyShock>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<DustyShock>::x1Momentum_index) = rho * u;
		state_cc(i, j, k, HydroSystem<DustyShock>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustyShock>::x3Momentum_index) = 0.;

		if constexpr (Physics_Traits<DustyShock>::is_dust_enabled) {
			state_cc(i, j, k, HydroSystem<DustyShock>::dustDensity_index) = rho;
			state_cc(i, j, k, HydroSystem<DustyShock>::x1DustMomentum_index) = rho * u;
			state_cc(i, j, k, HydroSystem<DustyShock>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustyShock>::x3DustMomentum_index) = 0.;
		}
	});
}

namespace
{

auto solve_quadratic_root_in_0_1(double a, double b, double c) -> double
{
	double const disc = b * b - 4.0 * a * c;
	if (disc < 0.0) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	double const r1 = (-b + std::sqrt(disc)) / (2.0 * a);
	double const r2 = (-b - std::sqrt(disc)) / (2.0 * a);
	bool const ok1 = (r1 > 0.0 && r1 < 1.0);
	bool const ok2 = (r2 > 0.0 && r2 < 1.0);
	if (ok1 && ok2) {
		return std::min(r1, r2);
	}
	if (ok1) {
		return r1;
	}
	if (ok2) {
		return r2;
	}
	if (r1 > 0.0) {
		return r1;
	}
	if (r2 > 0.0) {
		return r2;
	}
	return std::numeric_limits<double>::quiet_NaN();
}

auto linear_interpolate(const std::vector<double> &x, const std::vector<double> &y, double xi) -> double
{
	// assume x is sorted ascending
	if (xi <= x.front()) {
		return y.front();
	}
	if (xi >= x.back()) {
		return y.back();
	}
	auto it = std::upper_bound(x.begin(), x.end(), xi);
	int const idx = static_cast<int>(std::distance(x.begin(), it)) - 1;
	double const x0 = x[idx];
	double const x1 = x[idx + 1];
	double const y0 = y[idx];
	double const y1 = y[idx + 1];
	double const t = (xi - x0) / (x1 - x0);
	return y0 + t * (y1 - y0);
}

} // namespace

auto problem_main() -> int
{
	const double Lx = 40.0;
	const double CFL_number = 0.4;

	QuokkaSimulation<DustyShock> sim;
	sim.reconstructionOrder_ = 2;
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;

	sim.setInitialConditions();
	sim.evolve();

	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// extract numerical solution slices
		std::vector<double> rho_g_num(nx);
		std::vector<double> u_g_num(nx);
		std::vector<double> rho_d_num(nx);
		std::vector<double> u_d_num(nx);
		for (int i = 0; i < nx; ++i) {
			double const rho = values.at(HydroSystem<DustyShock>::density_index)[i];
			double const mom = values.at(HydroSystem<DustyShock>::x1Momentum_index)[i];
			double const rho_d = values.at(HydroSystem<DustyShock>::dustDensity_index)[i];
			double const mom_d = values.at(HydroSystem<DustyShock>::x1DustMomentum_index)[i];
			rho_g_num[i] = rho;
			u_g_num[i] = (rho != 0.0) ? mom / rho : 0.0;
			rho_d_num[i] = rho_d;
			u_d_num[i] = (rho_d != 0.0) ? mom_d / rho_d : 0.0;
		}

		// locate shock position from numerical density gradient
		std::vector<double> drho(nx > 1 ? nx - 1 : 0);
		for (int i = 0; i < nx - 1; ++i) {
			drho[i] = std::abs(rho_g_num[i + 1] - rho_g_num[i]);
		}
		int const shock_idx = (drho.empty()) ? 0 : static_cast<int>(std::distance(drho.begin(), std::max_element(drho.begin(), drho.end())));
		double const shock_pos_numeric = position[shock_idx];

		// analytic solution parameters
		const double v_s = vel_gas_left;
		const double c_s = cs_isothermal;
		const double M = v_s / c_s;
		const double epsilon = 1.0;
		const double K1 = drag_coefficient;
		const double rho_d_left = rho_dust_left;
		const double v_d_left = vel_dust_left;
		const double K_over_rho_v = K1 / (rho_d_left * v_d_left);

		// prepare analytic x grid
		const int n_analytic = 1000;
		std::vector<double> x_an(n_analytic);
		for (int i = 0; i < n_analytic; ++i) {
			x_an[i] = (Lx * static_cast<double>(i)) / static_cast<double>(n_analytic - 1);
		}

		// compute analytic omega_d and omega_g
		std::vector<double> omega_d(n_analytic, 1.0);
		std::vector<double> omega_g(n_analytic, 1.0);
		// find analytic index corresponding to numeric shock position
		int shock_idx_an = 0;
		for (int i = 0; i < n_analytic; ++i) {
			if (x_an[i] >= shock_pos_numeric) {
				shock_idx_an = i;
				break;
			}
		}

		// integrate to the right of the shock using RK4
		for (int i = shock_idx_an; i < n_analytic - 1; ++i) {
			double const x0 = x_an[i];
			double const x1 = x_an[i + 1];
			double const h = x1 - x0;
			double const w0 = omega_d[i];

			auto compute_omega_g = [&](double od) -> double {
				double const a = 1.0;
				double const b = epsilon * (od - 1.0) - 1.0 / (M * M) - 1.0;
				double const c = 1.0 / (M * M);
				return solve_quadratic_root_in_0_1(a, b, c);
			};

			auto deriv = [&](double /*xval*/, double wval) -> double {
				double const og = compute_omega_g(wval);
				if (!std::isfinite(og)) {
					return 0.0;
				}
				return K_over_rho_v * (og - wval);
			};

			double const k1 = deriv(x0, w0);
			double const k2 = deriv(x0 + 0.5 * h, w0 + 0.5 * h * k1);
			double const k3 = deriv(x0 + 0.5 * h, w0 + 0.5 * h * k2);
			double const k4 = deriv(x1, w0 + h * k3);

			double const wnext = w0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
			omega_d[i + 1] = wnext;

			double og_next = compute_omega_g(wnext);
			if (!std::isfinite(og_next)) {
				og_next = omega_g[i];
			}
			omega_g[i + 1] = og_next;
		}

		// build analytic density & velocity profiles from omega arrays
		std::vector<double> rho_g_an(n_analytic);
		std::vector<double> rho_d_an(n_analytic);
		std::vector<double> u_g_an(n_analytic);
		std::vector<double> u_d_an(n_analytic);
		for (int i = 0; i < n_analytic; ++i) {
			if (x_an[i] < shock_pos_numeric) {
				rho_g_an[i] = 1.0;
				rho_d_an[i] = rho_d_left;
				u_g_an[i] = v_s;
				u_d_an[i] = v_s;
			} else {
				double og = omega_g[i];
				double od = omega_d[i];
				if (!std::isfinite(og)) {
					og = 1.0;
				}
				if (!std::isfinite(od)) {
					od = 1.0;
				}
				u_g_an[i] = og * v_s;
				u_d_an[i] = od * v_s;
				double const denom_g = (og * v_s == 0.0) ? 1e-12 : (og * v_s);
				double const denom_d = (od * v_s == 0.0) ? 1e-12 : (od * v_s);
				rho_g_an[i] = 1.0 * v_d_left / denom_g;
				rho_d_an[i] = rho_d_left * v_d_left / denom_d;
			}
		}
#ifdef HAVE_PYTHON
		using matplotlibcpp::clf;
		using matplotlibcpp::legend;
		using matplotlibcpp::plot;
		using matplotlibcpp::save;
		using matplotlibcpp::tight_layout;
		using matplotlibcpp::title;
		using matplotlibcpp::xlabel;
		using matplotlibcpp::ylabel;

		// plotting: restrict to x in [0,20]
		std::vector<double> x_plot;
		std::vector<double> rho_g_plot;
		std::vector<double> rho_d_plot;
		std::vector<double> u_g_plot;
		std::vector<double> u_d_plot;
		for (int i = 0; i < nx; ++i) {
			if (position[i] >= 0.0 && position[i] <= 20.0) {
				x_plot.push_back(position[i]);
				rho_g_plot.push_back(rho_g_num[i]);
				rho_d_plot.push_back(rho_d_num[i]);
				u_g_plot.push_back(u_g_num[i]);
				u_d_plot.push_back(u_d_num[i]);
			}
		}

		// analytic curves limited to x in [0,20]
		std::vector<double> x_an_plot;
		std::vector<double> rho_g_an_plot;
		std::vector<double> rho_d_an_plot;
		std::vector<double> u_g_an_plot;
		std::vector<double> u_d_an_plot;
		for (int i = 0; i < n_analytic; ++i) {
			if (x_an[i] >= 0.0 && x_an[i] <= 20.0) {
				x_an_plot.push_back(x_an[i]);
				rho_g_an_plot.push_back(rho_g_an[i]);
				rho_d_an_plot.push_back(rho_d_an[i]);
				u_g_an_plot.push_back(u_g_an[i]);
				u_d_an_plot.push_back(u_d_an[i]);
			}
		}

		// density plot
		clf();
		plot(x_plot, rho_g_plot, {{"label", "Gas Density (Numerical)"}, {"marker", "o"}, {"linestyle", "None"}, {"markersize", "2"}});
		plot(x_plot, rho_d_plot, {{"label", "Dust Density (Numerical)"}, {"marker", "o"}, {"linestyle", "None"}, {"markersize", "2"}});
		plot(x_an_plot, rho_g_an_plot, {{"label", "Gas Density (Analytic)"}, {"linestyle", "-"}, {"color", "black"}});
		plot(x_an_plot, rho_d_an_plot, {{"label", "Dust Density (Analytic)"}, {"linestyle", "--"}, {"color", "black"}});
		xlabel("x");
		ylabel("Density");
		title(std::format("Density Comparison at t = {:.4f}", sim.tNew_[0]));
		legend();
		tight_layout();
		save("dusty_shock_density.pdf");

		// velocity plot
		clf();
		plot(x_plot, u_g_plot, {{"label", "Gas Velocity (Numerical)"}, {"marker", "o"}, {"linestyle", "None"}, {"markersize", "2"}});
		plot(x_plot, u_d_plot, {{"label", "Dust Velocity (Numerical)"}, {"marker", "o"}, {"linestyle", "None"}, {"markersize", "2"}});
		plot(x_an_plot, u_g_an_plot, {{"label", "Gas Velocity (Analytic)"}, {"linestyle", "-"}, {"color", "black"}});
		plot(x_an_plot, u_d_an_plot, {{"label", "Dust Velocity (Analytic)"}, {"linestyle", "--"}, {"color", "black"}});
		xlabel("x");
		ylabel("Velocity");
		title(std::format("Velocity Comparison at t = {:.4f}", sim.tNew_[0]));
		legend();
		tight_layout();
		save("dusty_shock_velocity.pdf");
#endif // HAVE_PYTHON

		// interpolate analytic solution to numerical x positions, then compute relative L1 norms
		std::vector<double> rho_g_an_at_num(nx);
		std::vector<double> rho_d_an_at_num(nx);
		std::vector<double> u_g_an_at_num(nx);
		std::vector<double> u_d_an_at_num(nx);
		for (int i = 0; i < nx; ++i) {
			double const xi = position[i];
			rho_g_an_at_num[i] = linear_interpolate(x_an, rho_g_an, xi);
			rho_d_an_at_num[i] = linear_interpolate(x_an, rho_d_an, xi);
			u_g_an_at_num[i] = linear_interpolate(x_an, u_g_an, xi);
			u_d_an_at_num[i] = linear_interpolate(x_an, u_d_an, xi);
		}

		auto relative_L1 = [&](const std::vector<double> &num, const std::vector<double> &ana) -> double {
			double err = 0.0;
			double sol = 0.0;
			for (size_t i = 0; i < num.size(); ++i) {
				err += std::abs(num[i] - ana[i]);
				sol += std::abs(ana[i]);
			}
			if (sol == 0.0) {
				return (err == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
			}
			return err / sol;
		};

		double const err_rho_g = relative_L1(rho_g_num, rho_g_an_at_num);
		double const err_rho_d = relative_L1(rho_d_num, rho_d_an_at_num);
		double const err_u_g = relative_L1(u_g_num, u_g_an_at_num);
		double const err_u_d = relative_L1(u_d_num, u_d_an_at_num);

		amrex::Print() << "Relative L1 norm (gas density)  = " << err_rho_g << "\n";
		amrex::Print() << "Relative L1 norm (dust density) = " << err_rho_d << "\n";
		amrex::Print() << "Relative L1 norm (gas velocity) = " << err_u_g << "\n";
		amrex::Print() << "Relative L1 norm (dust velocity)= " << err_u_d << "\n";

		const double tol = 0.01;
		if (err_rho_g > tol || err_rho_d > tol || err_u_g > tol || err_u_d > tol) {
			status = 1;
		}
		amrex::Print() << (status == 0 ? "Test PASSED.\n" : "Test FAILED.\n");
	}
	return status;
}