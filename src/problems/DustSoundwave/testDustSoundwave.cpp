/// \file testDustSoundwave.cpp
/// \brief Defines a test problem for dust transport terms and dust drag force
///

#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include <format>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

#include <cmath>
#include <complex>

struct DustSoundwave {
};

// parameters for soundwave damping test
constexpr double A = 1e-4;
constexpr double rho_g0 = 1.0;
constexpr double rho_d0 = 2.24;
constexpr double omega_r = 1.915896;
constexpr double omega_i = -4.410541;
constexpr double re_rho_g = 1.0;
constexpr double im_rho_g = 0.0;
constexpr double re_u_g = -0.701960;
constexpr double im_u_g = -0.304924;
constexpr double re_rho_d = 0.165251;
constexpr double im_rho_d = -1.247801;
constexpr double re_u_d = -0.221645;
constexpr double im_u_d = 0.368534;

auto real_part_analytic(double t, double re, double im) -> double
{
	std::complex<double> const hat(re, im);
	double const magnitude = std::abs(hat);
	double const phase = std::arg(hat);
	double const damp = std::exp(-omega_r * t);
	double const osc = -omega_i * t;
	return magnitude * damp * std::cos(osc + phase);
}

auto v_gas_analytic(double t) -> double { return real_part_analytic(t, re_u_g, im_u_g); }

auto rho_gas_analytic(double t) -> double { return real_part_analytic(t, re_rho_g, im_rho_g); }

auto v_dust_analytic(double t) -> double { return real_part_analytic(t, re_u_d, im_u_d); }

auto rho_dust_analytic(double t) -> double { return real_part_analytic(t, re_rho_d, im_rho_d); }

template <> struct SimulationData<DustSoundwave> {
	std::vector<double> t_vec_;
	std::vector<double> v_gas_vec_;
	std::vector<double> v_dust_vec_;
	std::vector<double> rho_gas_vec_;
	std::vector<double> rho_dust_vec_;
};
;

template <> struct quokka::EOS_Traits<DustSoundwave> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // only used when gamma = 1
};

const double cs = quokka::EOS_Traits<DustSoundwave>::cs_isothermal;

template <> struct Physics_Traits<DustSoundwave> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_dust_enabled = true;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <>
AMREX_GPU_HOST_DEVICE auto DustSources<DustSoundwave>::ComputeReciprocalStoppingTime(amrex::Real /*rho_g*/,
										     amrex::GpuArray<amrex::Real, nDustGroups_> /*rho_d*/,
										     amrex::GpuArray<amrex::Real, nDustGroups_> /*rel_vel_mag*/, double /*cs*/)
    -> amrex::GpuArray<amrex::Real, nDustGroups_>
{
	amrex::GpuArray<amrex::Real, 1> alpha{};
	alpha[0] = 2.5;
	return alpha;
}

template <> void QuokkaSimulation<DustSoundwave>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// parameters for soundwave damping test
	const double L = 1.0;		  // domain length
	const double kk = 2.0 * M_PI / L; // wave number k=2π/L

	const double rho_g0 = 1.0;  // gas base density
	const double rho_d0 = 2.24; // dust base density

	const double A_rho = A * rho_g0; // density perturbation amplitude A*ρ_g^0
	const double A_vel = A * cs;	 // velocity perturbation amplitude A*c_s

	const double Re_rho_g = 1.0;	 // Re(δρ_g^)
	const double Im_rho_g = 0.0;	 // Im(δρ_g^)
	const double Re_u_g = -0.701960; // Re(δu_g^)
	const double Im_u_g = -0.304924; // Im(δu_g^)

	const double Re_rho_d = 0.165251;  // Re(δρ_d^)
	const double Im_rho_d = -1.247801; // Im(δρ_d^)
	const double Re_u_d = -0.221645;   // Re(δu_d^)
	const double Im_u_d = 0.368534;	   // Im(δu_d^)

	// get geometry information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = Geom(0).CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = Geom(0).ProbLoArray();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

		// density perturbation: δρ_g = A_rho[Re(δρ_g^)cos(kx) - Im(δρ_g^)sin(kx)]
		double const drho_g = A_rho * (Re_rho_g * cos(kk * x) - Im_rho_g * sin(kk * x));
		amrex::Real const rho_gas_local = rho_g0 + drho_g;

		// velocity perturbation: δv_g = A_vel[Re(δv_g^)cos(kx) - Im(δv_g^)sin(kx)]
		double const du_g = A_vel * (Re_u_g * cos(kk * x) - Im_u_g * sin(kk * x));

		state_cc(i, j, k, HydroSystem<DustSoundwave>::density_index) = rho_gas_local;
		state_cc(i, j, k, HydroSystem<DustSoundwave>::x1Momentum_index) = rho_gas_local * du_g;
		state_cc(i, j, k, HydroSystem<DustSoundwave>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustSoundwave>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<DustSoundwave>::energy_index) = 0.5 * rho_gas_local * (du_g * du_g);
		state_cc(i, j, k, HydroSystem<DustSoundwave>::internalEnergy_index) = 0.0;

		// compute dust values before constexpr-if to ensure proper capture
		// density perturbation: δρ_d = A_rho[Re(δρ_d^)cos(kx) - Im(δρ_d^)sin(kx)]
		double const drho_d = A_rho * (Re_rho_d * cos(kk * x) - Im_rho_d * sin(kk * x));
		amrex::Real const rho_dust_local = rho_d0 + drho_d;

		// velocity perturbation: δv_d = A_vel[Re(δv_d^)cos(kx) - Im(δv_d^)sin(kx)]
		double const du_d = A_vel * (Re_u_d * cos(kk * x) - Im_u_d * sin(kk * x));

		if constexpr (Physics_Traits<DustSoundwave>::is_dust_enabled) {
			state_cc(i, j, k, HydroSystem<DustSoundwave>::dustDensity_index) = rho_dust_local;
			state_cc(i, j, k, HydroSystem<DustSoundwave>::x1DustMomentum_index) = rho_dust_local * du_d;
			state_cc(i, j, k, HydroSystem<DustSoundwave>::x2DustMomentum_index) = 0.;
			state_cc(i, j, k, HydroSystem<DustSoundwave>::x3DustMomentum_index) = 0.;
		}
	});
}

template <> void QuokkaSimulation<DustSoundwave>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.0);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);
		const double density = values.at(HydroSystem<DustSoundwave>::density_index)[0];
		const double mom_x = values.at(HydroSystem<DustSoundwave>::x1Momentum_index)[0];
		const double dust_density = values.at(HydroSystem<DustSoundwave>::dustDensity_index)[0];
		const double dust_mom_x = values.at(HydroSystem<DustSoundwave>::x1DustMomentum_index)[0];
		double const v_gas = mom_x / density;
		double const rho_gas = density;
		double const v_dust = dust_mom_x / dust_density;
		double const rho_dust = dust_density;
		userData_.v_gas_vec_.push_back(v_gas);
		userData_.rho_gas_vec_.push_back(rho_gas);
		userData_.v_dust_vec_.push_back(v_dust);
		userData_.rho_dust_vec_.push_back(rho_dust);
	}
}

auto problem_main() -> int
{
	// problem parameters
	const double CFL_number = 0.4;

	// problem initialization
	QuokkaSimulation<DustSoundwave> sim;

	sim.reconstructionOrder_ = 3;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.plotfileInterval_ = -1;
	sim.cflNumber_ = CFL_number;

	// initialize
	sim.setInitialConditions();

	// store initial values for t=0 plotting

	auto [position, val_ini] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.t_vec_.push_back(0.0);
		const double density = val_ini.at(HydroSystem<DustSoundwave>::density_index)[0];
		const double mom_x = val_ini.at(HydroSystem<DustSoundwave>::x1Momentum_index)[0];
		const double dust_density = val_ini.at(HydroSystem<DustSoundwave>::dustDensity_index)[0];
		const double dust_mom_x = val_ini.at(HydroSystem<DustSoundwave>::x1DustMomentum_index)[0];
		double const v_gas = mom_x / density;
		double const rho_gas = density;
		double const v_dust = dust_mom_x / dust_density;
		double const rho_dust = dust_density;
		sim.userData_.v_gas_vec_.push_back(v_gas);
		sim.userData_.rho_gas_vec_.push_back(rho_gas);
		sim.userData_.v_dust_vec_.push_back(v_dust);
		sim.userData_.rho_dust_vec_.push_back(rho_dust);
	}

	// evolve
	sim.evolve();

	// use time series data from SimulationData
	std::vector<double> &t_vec = sim.userData_.t_vec_;
	std::vector<double> &v_gas_vec = sim.userData_.v_gas_vec_;
	std::vector<double> &v_dust_vec = sim.userData_.v_dust_vec_;
	std::vector<double> &rho_gas_vec = sim.userData_.rho_gas_vec_;
	std::vector<double> &rho_dust_vec = sim.userData_.rho_dust_vec_;

	// compute normalized numerical values
	std::vector<double> norm_v_gas(t_vec.size());
	std::vector<double> norm_rho_gas(t_vec.size());
	std::vector<double> norm_v_dust(t_vec.size());
	std::vector<double> norm_rho_dust(t_vec.size());
	for (size_t i = 0; i < t_vec.size(); ++i) {
		norm_v_gas[i] = v_gas_vec[i] / A;
		norm_rho_gas[i] = (rho_gas_vec[i] - rho_g0) / (A * rho_g0);
		norm_v_dust[i] = v_dust_vec[i] / A;
		norm_rho_dust[i] = (rho_dust_vec[i] - rho_d0) / A / rho_d0;
	}

	// compute exact normalized values at simulation times
	std::vector<double> norm_v_gas_exact(t_vec.size());
	std::vector<double> norm_rho_gas_exact(t_vec.size());
	std::vector<double> norm_v_dust_exact(t_vec.size());
	std::vector<double> norm_rho_dust_exact(t_vec.size());
	for (size_t i = 0; i < t_vec.size(); ++i) {
		double const t = t_vec[i];
		norm_v_gas_exact[i] = v_gas_analytic(t);
		norm_rho_gas_exact[i] = rho_gas_analytic(t);
		norm_v_dust_exact[i] = v_dust_analytic(t);
		norm_rho_dust_exact[i] = rho_dust_analytic(t) / rho_d0;
	}

	// dense points for analytic plotting
	const size_t n_dense_points = 1000;
	std::vector<double> t_dense(n_dense_points);
	std::vector<double> norm_v_gas_dense(n_dense_points);
	std::vector<double> norm_rho_gas_dense(n_dense_points);
	std::vector<double> norm_v_dust_dense(n_dense_points);
	std::vector<double> norm_rho_dust_dense(n_dense_points);
	double const t_max = t_vec.empty() ? 0.0 : t_vec.back();
	for (size_t i = 0; i < n_dense_points; ++i) {
		double const tt = t_max * static_cast<double>(i) / (n_dense_points - 1);
		t_dense[i] = tt;
		norm_v_gas_dense[i] = v_gas_analytic(tt);
		norm_rho_gas_dense[i] = rho_gas_analytic(tt);
		norm_v_dust_dense[i] = v_dust_analytic(tt);
		norm_rho_dust_dense[i] = rho_dust_analytic(tt) / rho_d0;
	}

	// relative L1 error function
	auto rel_err = [](const std::vector<double> &sim, const std::vector<double> &exact) {
		double err = 0.0;
		double sol = 0.0;
		for (size_t i = 0; i < sim.size(); ++i) {
			err += std::abs(sim[i] - exact[i]);
			sol += std::abs(exact[i]);
		}
		return (sol > 0.0) ? err / sol : 0.0;
	};

	double const rel_err_v_gas = rel_err(norm_v_gas, norm_v_gas_exact);
	double const rel_err_rho_gas = rel_err(norm_rho_gas, norm_rho_gas_exact);
	double const rel_err_v_dust = rel_err(norm_v_dust, norm_v_dust_exact);
	double const rel_err_rho_dust = rel_err(norm_rho_dust, norm_rho_dust_exact);

	amrex::Print() << "Relative L1 norm for gas velocity = " << rel_err_v_gas << '\n';
	amrex::Print() << "Relative L1 norm for gas density = " << rel_err_rho_gas << '\n';
	amrex::Print() << "Relative L1 norm for dust velocity = " << rel_err_v_dust << '\n';
	amrex::Print() << "Relative L1 norm for dust density = " << rel_err_rho_dust << '\n';

	int status = 0;
	const double rel_err_tol = 0.03;
	if ((rel_err_v_gas > rel_err_tol) || (rel_err_rho_gas > rel_err_tol) || (rel_err_v_dust > rel_err_tol) || (rel_err_rho_dust > rel_err_tol)) {
		status = 1;
	}

#ifdef HAVE_PYTHON
	const int plot_stride = 100;

	// downsample function
	auto downsample = [&](const std::vector<double> &vec) {
		std::vector<double> out;
		out.reserve(vec.size() / plot_stride + 2);
		for (size_t i = 0; i < vec.size(); i += plot_stride) {
			out.push_back(vec[i]);
		}
		// make sure to include the last point if not already included
		if (!vec.empty() && (vec.size() - 1) % plot_stride != 0) {
			out.push_back(vec.back());
		}
		return out;
	};

	auto t_vec_plot = downsample(t_vec);
	auto norm_v_gas_plot = downsample(norm_v_gas);
	auto norm_rho_gas_plot = downsample(norm_rho_gas);
	auto norm_v_dust_plot = downsample(norm_v_dust);
	auto norm_rho_dust_plot = downsample(norm_rho_dust);

	matplotlibcpp::clf();
	matplotlibcpp::plot(t_vec_plot, norm_v_gas_plot,
			    {{"label", "gas (numerical)"}, {"color", "blue"}, {"linestyle", "None"}, {"marker", "o"}, {"markersize", "4"}});
	matplotlibcpp::plot(t_dense, norm_v_gas_dense, {{"label", "gas (analytic)"}, {"color", "blue"}, {"linestyle", "--"}});
	matplotlibcpp::plot(t_vec_plot, norm_v_dust_plot,
			    {{"label", "dust (numerical)"}, {"color", "red"}, {"linestyle", "None"}, {"marker", "o"}, {"markersize", "4"}});
	matplotlibcpp::plot(t_dense, norm_v_dust_dense, {{"label", "dust (analytic)"}, {"color", "red"}, {"linestyle", "--"}});
	matplotlibcpp::legend();
	matplotlibcpp::xlabel("Time");
	matplotlibcpp::ylabel(R"($\delta u/(A c_s)$)");
	matplotlibcpp::title(std::format("Velocity Evolution", plot_stride));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./dust_soundwave_velocity.pdf");

	matplotlibcpp::clf();
	matplotlibcpp::plot(t_vec_plot, norm_rho_gas_plot,
			    {{"label", "gas (numerical)"}, {"color", "blue"}, {"linestyle", "None"}, {"marker", "o"}, {"markersize", "4"}});
	matplotlibcpp::plot(t_dense, norm_rho_gas_dense, {{"label", "gas (analytic)"}, {"color", "blue"}, {"linestyle", "--"}});
	matplotlibcpp::plot(t_vec_plot, norm_rho_dust_plot,
			    {{"label", "dust (numerical)"}, {"color", "red"}, {"linestyle", "None"}, {"marker", "o"}, {"markersize", "4"}});
	matplotlibcpp::plot(t_dense, norm_rho_dust_dense, {{"label", "dust (analytic)"}, {"color", "red"}, {"linestyle", "--"}});
	matplotlibcpp::legend();
	matplotlibcpp::xlabel("Time");
	matplotlibcpp::ylabel(R"($\delta \rho/(A \rho^0)$)");
	matplotlibcpp::title(std::format("Density Evolution", plot_stride));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./dust_soundwave_density.pdf");
#endif // HAVE_PYTHON

	amrex::Print() << "Finished." << '\n';
	return status;
}
