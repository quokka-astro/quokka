//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testStromgrenSphere.cpp
/// \brief Defines a test problem for the static Stromgren sphere with no temperature dependence.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <limits>
#include <string>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct StromgrenSphere {
};

constexpr double c_hat = C::c_light / 1.0;

template <> struct quokka::EOS_Traits<StromgrenSphere> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StromgrenSphere> {
	static constexpr bool is_self_gravity_enabled = false;
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1; // number of dust groups
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gravitational_constant = C::Gconst;
	static constexpr double c_light = C::c_light;
	static constexpr double radiation_constant = C::a_rad;
};

template <> struct RadSystem_Traits<StromgrenSphere> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = 1e-99;
	static constexpr int beta_order = 0;
	AMREX_GPU_HOST_DEVICE static constexpr amrex::GpuArray<double, NumChemActiveRadGroups + 1> ChemActiveRadFreqBounds()
	{
		return ChemActiveRadFreqBoundsHeader_;
	}
};

template <>
void RadSystem<StromgrenSphere>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse pp("stromgen");
	amrex::Real Q = 1.0e49_rt;
	pp.query("Q", Q);

	const amrex::Real sigma_star = 1.5 * dx[0];
	const int nCells = 3;
	const amrex::Real L_star = Q * RadSystem<StromgrenSphere>::GetChemActiveRadiationGroupQuanta(0) / 8.0_rt;
	const amrex::Real x0 = 0.0_rt;
	const amrex::Real y0 = 0.0_rt;
	const amrex::Real z0 = 0.0_rt;
	amrex::Real sum = 0.0_rt;
	for (int i = 0; i <= nCells; ++i) {
		for (int j = 0; j <= nCells; ++j) {
			for (int k = 0; k <= nCells; ++k) {
				amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
				amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
				amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
				amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
				sum += std::exp(-(r * r) / (2.0 * sigma_star * sigma_star)) * dx[0] * dx[1] * dx[2] /
				       (std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5));
			}
		}
	}
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		if ((i <= nCells) && (j <= nCells) && (k <= nCells)) {
			amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
			amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
			amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
			amrex::Real w_i = std::exp(-(r * r) / (2.0 * sigma_star * sigma_star)) / (std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5));
			amrex::Real val = L_star * w_i / sum;
			radEnergy(i, j, k) = val;
		} else {
			radEnergy(i, j, k) = 0.0_rt;
		}
	});
}

template <> struct SimulationData<StromgrenSphere> {
	amrex::Real small_temp;
	amrex::Real small_dens;
	amrex::Real temperature;
	amrex::Real primary_species_1;
	amrex::Real primary_species_2;
	amrex::Real primary_species_3;
	amrex::Real Q;
	amrex::Real tend;
	int recombination_switch;
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> r50_vec_;
	amrex::Vector<amrex::Real> r16_vec_;
	amrex::Vector<amrex::Real> r84_vec_;
	std::ofstream output_file_;
};

template <> void QuokkaSimulation<StromgrenSphere>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species and temperature
	amrex::ParmParse const pp("stromgen");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e4;
	userData_.tend = 1000.0_rt;
	userData_.primary_species_1 = 0.0e0_rt;
	userData_.primary_species_2 = 1.0e2_rt;
	userData_.primary_species_3 = 0.0e0_rt;
	userData_.Q = 1.0e49_rt;
	userData_.recombination_switch = 0;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("tend", userData_.tend);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("Q", userData_.Q);

	amrex::ParmParse const pp2("network");
	pp2.query("recombination_switch", userData_.recombination_switch);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::string filename = fmt::format("stromgren_sphere_radii.csv");
		userData_.output_file_.open(filename);
		userData_.output_file_ << "time,r16,r50,r84\n";
	}
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphere>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StromgrenSphere>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> void QuokkaSimulation<StromgrenSphere>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	for (int n = 1; n <= NumSpec; ++n) {
		switch (n) {
			case 1:
				numdens[n - 1] = userData_.primary_species_1;
				break;
			case 2:
				numdens[n - 1] = userData_.primary_species_2;
				break;
			case 3:
				numdens[n - 1] = userData_.primary_species_3;
				break;
			default:
				amrex::Abort("Cannot initialize number density for chem specie");
				break;
		}
	}

	state.T = userData_.temperature;
	// find the density in g/cm^3
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n]; // spmasses contains the masses of all species, defined in EOS
	}
	state.rho = rhotot;

	// call the EOS to set initial internal energy e
	eos(eos_input_rt, state);
	const auto Egas0 = state.e * rhotot;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<StromgrenSphere>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<StromgrenSphere>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = 1.e-99_rt;
			state_cc(i, j, k, RadSystem<StromgrenSphere>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<StromgrenSphere>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<StromgrenSphere>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<StromgrenSphere>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<StromgrenSphere>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<StromgrenSphere>::computeAfterTimestep()
{
	if (userData_.recombination_switch == 0) {
		return;
	}
	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	const amrex::Real xmax = std::max(std::abs(prob_lo[0]), std::abs(prob_hi[0]));
	const amrex::Real ymax = std::max(std::abs(prob_lo[1]), std::abs(prob_hi[1]));
	const amrex::Real zmax = std::max(std::abs(prob_lo[2]), std::abs(prob_hi[2]));
	const amrex::Real rmax = std::sqrt(xmax * xmax + ymax * ymax + zmax * zmax);

	const int n_bins =
	    3 * static_cast<int>(CountCells(lev)); // The test is meant to be run with a small number of cells, so this narrowing static cast is fine.
	using hist_t = unsigned long long;
	amrex::Gpu::DeviceVector<hist_t> d_hist(n_bins, static_cast<hist_t>(0));

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &state = state_new_cc_[lev].const_array(mfi);
		auto *hist_ptr = d_hist.data();

		amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real n_HI = state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + 1) / spmasses[1];
			const amrex::Real n_HII = state(i, j, k, HydroSystem<StromgrenSphere>::scalar0_index + 2) / spmasses[2];
			const amrex::Real denom = n_HI + n_HII;
			if (denom <= 0.0_rt) {
				return;
			}

			const amrex::Real x_HII = n_HII / denom;
			if ((x_HII < 0.01_rt) || (x_HII > 0.99_rt)) {
				return;
			}

			const amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5_rt) * dx[0];
			const amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1];
			const amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5_rt) * dx[2];
			const amrex::Real r = std::sqrt(x * x + y * y + z * z);
			if (rmax <= 0.0_rt) {
				return;
			}

			int ibin = static_cast<int>((r / rmax) * static_cast<amrex::Real>(n_bins));
			ibin = amrex::max(0, amrex::min(ibin, n_bins - 1));
			amrex::Gpu::Atomic::AddNoRet(&hist_ptr[ibin], static_cast<hist_t>(1));
		});
	}

	amrex::Gpu::streamSynchronize();

	amrex::Gpu::HostVector<hist_t> h_hist(n_bins);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_hist.begin(), d_hist.end(), h_hist.begin());

	const int num_local = n_bins;
	auto num_local_vec = amrex::ParallelDescriptor::Gather(num_local, amrex::ParallelDescriptor::IOProcessorNumber());

	amrex::Vector<int> recvcnt;
	amrex::Vector<int> disp;
	amrex::Vector<hist_t> all_hist;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		recvcnt.resize(num_local_vec.size());
		disp.resize(num_local_vec.size());
		int ntot = 0;
		disp[0] = 0;
		for (int i = 0, n = static_cast<int>(num_local_vec.size()); i < n; ++i) {
			ntot += num_local_vec[i];
			recvcnt[i] = num_local_vec[i];
			if (i + 1 < n) {
				disp[i + 1] = disp[i] + num_local_vec[i];
			}
		}
		all_hist.resize(ntot);
	} else {
		recvcnt.resize(1);
		disp.resize(1);
		all_hist.resize(1);
	}

	static hist_t static_hist = 0;
	const hist_t *send_ptr = h_hist.empty() ? &static_hist : h_hist.data();
	hist_t *recv_ptr = all_hist.empty() ? &static_hist : all_hist.data();
	amrex::ParallelDescriptor::Gatherv(send_ptr, num_local, recv_ptr, recvcnt, disp, amrex::ParallelDescriptor::IOProcessorNumber());

	amrex::Real r16 = std::numeric_limits<amrex::Real>::quiet_NaN();
	amrex::Real r50 = std::numeric_limits<amrex::Real>::quiet_NaN();
	amrex::Real r84 = std::numeric_limits<amrex::Real>::quiet_NaN();

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Vector<hist_t> global_hist(n_bins, static_cast<hist_t>(0));
		const int n_ranks = static_cast<int>(num_local_vec.size());
		for (int rank = 0; rank < n_ranks; ++rank) {
			const int offset = disp[rank];
			for (int b = 0; b < n_bins; ++b) {
				global_hist[b] += all_hist[offset + b];
			}
		}

		hist_t total_count = 0;
		for (int b = 0; b < n_bins; ++b) {
			total_count += global_hist[b];
		}

		if (total_count == 0) {
			amrex::Print() << fmt::format("t = {:e}: no cells with 0.01 <= x_HII <= 0.99", tNew_[lev]) << '\n';
		} else {
			auto percentile_from_hist = [=](amrex::Vector<hist_t> const &hist, amrex::Real q) -> amrex::Real {
				const amrex::Real target = q * static_cast<amrex::Real>(total_count - 1);
				amrex::Real cum = 0.0_rt;
				for (int b = 0; b < n_bins; ++b) {
					const amrex::Real next_cum = cum + static_cast<amrex::Real>(hist[b]);
					if (target < next_cum) {
						const amrex::Real bin_lo = (static_cast<amrex::Real>(b) / static_cast<amrex::Real>(n_bins)) * rmax;
						const amrex::Real bin_hi = (static_cast<amrex::Real>(b + 1) / static_cast<amrex::Real>(n_bins)) * rmax;
						if (hist[b] <= 1) {
							return 0.5_rt * (bin_lo + bin_hi);
						}
						const amrex::Real frac = (target - cum) / static_cast<amrex::Real>(hist[b]);
						return bin_lo + frac * (bin_hi - bin_lo);
					}
					cum = next_cum;
				}
				amrex::Abort(fmt::format("Error in percentile_from_hist: target {} exceeds total count {}", target, total_count));
				return rmax;
			};

			r16 = percentile_from_hist(global_hist, 0.16_rt);
			r50 = percentile_from_hist(global_hist, 0.50_rt);
			r84 = percentile_from_hist(global_hist, 0.84_rt);
		}
	}

	amrex::ParallelDescriptor::Bcast(&r16, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	amrex::ParallelDescriptor::Bcast(&r50, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	amrex::ParallelDescriptor::Bcast(&r84, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	userData_.t_vec_.push_back(tNew_[lev]);
	userData_.r16_vec_.push_back(r16);
	userData_.r50_vec_.push_back(r50);
	userData_.r84_vec_.push_back(r84);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << tNew_[lev] << ',' << r16 << ',' << r50 << ',' << r84 << '\n';
	}
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.3;
	const double dt_max = 1e99;
	const int max_timesteps = 5000000;

	// Problem initialization
	QuokkaSimulation<StromgrenSphere> sim;

	// initialize
	sim.setInitialConditions();
	sim.stopTime_ = sim.userData_.tend;
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	int status = 0;
	if (sim.userData_.recombination_switch == 0) {
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
		const amrex::Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
		const amrex::Real quanta = RadSystem<StromgrenSphere>::GetChemActiveRadiationGroupQuanta(0);
		const amrex::Real n_photon_sum0 = sim.state_new_cc_[0].sum(RadSystem<StromgrenSphere>::radEnergy_index) / quanta;
		const amrex::Real n_e_sum0 = sim.state_new_cc_[0].sum(HydroSystem<StromgrenSphere>::scalar0_index + 0) / C::m_e;

		sim.evolve();

		const amrex::Real error_tol = 1e-6;
		const amrex::Real n_photon_sum = sim.state_new_cc_[0].sum(RadSystem<StromgrenSphere>::radEnergy_index) / quanta;
		const amrex::Real n_e_sum = sim.state_new_cc_[0].sum(HydroSystem<StromgrenSphere>::scalar0_index + 0) / C::m_e;
		amrex::Print() << fmt::format("Final total number of photons: {0}", n_photon_sum) << '\n';
		amrex::Print() << fmt::format("Final total number of electrons: {0}", n_e_sum) << '\n';
		const amrex::Real n_photon_added = sim.userData_.Q * sim.userData_.tend / 8.0_rt;
		const amrex::Real n_photon_inflight = (n_photon_sum - n_photon_sum0) * cell_vol;
		const amrex::Real n_electron_added = (n_e_sum - n_e_sum0) * cell_vol;
		const amrex::Real photon_absorbed = n_electron_added;

		amrex::Real photon_err = std::abs((n_photon_added - (n_photon_inflight + photon_absorbed)) / n_photon_added);
		if (photon_err > error_tol) {
			amrex::Print() << fmt::format("Test failed: Relative photon conservation error is {0} (tolerance: {1})", photon_err, error_tol) << '\n';
			status = 1;
		} else {
			amrex::Print() << fmt::format("Test passed: Relative photon conservation error is {0} (tolerance: {1})", photon_err, error_tol) << '\n';
		}
	} else {
		sim.evolve();

		if (amrex::ParallelDescriptor::IOProcessor()) {
			const amrex::Real n_HI0 = sim.userData_.primary_species_2;
			const amrex::Real alpha_B = 2.6e-13;
			const amrex::Real r_s = std::pow((3.0_rt * sim.userData_.Q) / (4.0_rt * M_PI * alpha_B * n_HI0 * n_HI0), 1.0_rt / 3.0_rt);
			const amrex::Real t_rec = 1.0_rt / (alpha_B * n_HI0);
			const amrex::Real cell_size = sim.geom[0].CellSizeArray()[0];
			const amrex::Real error_tol = 2.0_rt * cell_size;
			for (int i = 0; i < sim.userData_.t_vec_.size(); ++i) {
				const amrex::Real r50_numerical = sim.userData_.r50_vec_[i];
				const amrex::Real r16_numerical = sim.userData_.r16_vec_[i];
				const amrex::Real r84_numerical = sim.userData_.r84_vec_[i];
				const amrex::Real analytical_radius = r_s * std::pow(1.0_rt - std::exp(-sim.userData_.t_vec_[i] / t_rec), 1.0_rt / 3.0_rt);
				const amrex::Real upper_bound = analytical_radius + error_tol;
				const amrex::Real lower_bound = analytical_radius - error_tol;
				if ((r50_numerical < lower_bound) || (r50_numerical > upper_bound)) {
					amrex::Print()
					    << fmt::format("Test failed at t = {0}: Numerical r50 = {1} is not within [{2}, {3}] (analytical r50 = {4})",
							   sim.userData_.t_vec_[i], r50_numerical, lower_bound, upper_bound, analytical_radius)
					    << '\n';
					status = 1;
				}
			}
			if (status == 0) {
				amrex::Print()
				    << "Test passed: Numerical ionization front radii are within error tolerance of analytical solution at all output times."
				    << '\n';
			}
		}
	}

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return status;
}
