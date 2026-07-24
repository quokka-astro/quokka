/// \file testDTypeFront1D.cpp
/// \brief Defines a 1D planar D-type ionization-front test problem.
///
/// A constant ionizing photon flux F [photons cm^-2 s^-1] is injected at the
/// x = 0 boundary into a uniform, initially neutral hydrogen slab. The gas
/// ionizes, heats, over-pressurizes, and drives a planar D-type front. The
/// numerical front position (effective ionized length) is compared against the
/// analytic planar D-type solution
///
///   x_i(t) = x_S * [1 + (5/4) * sqrt(4/3) * c_i * t / x_S]^(4/5),
///
/// the planar (momentum-conserving) analog of the spherical Spitzer law. See
/// ~/superpowers/quokka/adr-DTypeFront1D/adr-DTypeFront1D.md for the derivation.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

struct DTypeFront1D {
};

// reduced speed of light (same choice as the 3D DTypeFront problem)
constexpr double c_hat = C::c_light / 1000.0;

// Mean energy of an ionizing photon in the single chemistry band [3.29e15, 1.5e16] Hz (see
// CMakeLists.txt CHEM_BANDS); this matches RadSystem::GetChemBandQuanta(0).
constexpr double E_photon = 0.5 * (3.29e15 + 1.50e16) * C::hplanck; // erg
// Radiation energy-density floor. Rather than the arbitrary blackbody-at-0.01-K value used by the 3D
// DTypeFront, this is a physically meaningful, negligible photon-number density (1e-10 cm^-3, vs the
// ~hundreds cm^-3 of the ionizing field) converted to a radiation energy density. Dark cells are
// initialized to exactly this floor (see setInitialConditionsOnGrid), following the best practice of
// RadStreaming / RadhydroShockMultigroup instead of seeding an unphysical 1e-99.
constexpr double Erad_floor_ = 1.0e-10 * E_photon; // erg cm^-3

template <> struct quokka::EOS_Traits<DTypeFront1D> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DTypeFront1D> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
};

template <> struct RadSystem_Traits<DTypeFront1D> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	// beta_order = 0: no O(v/c) photochemistry radiation-momentum kick. The analytic law is pure
	// thermal-pressure, so the physics must match (radiation pressure is subdominant here anyway).
	static constexpr int beta_order = 0;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
};

template <> struct SimulationData<DTypeFront1D> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real flux{}; // ionizing photon flux F [photons cm^-2 s^-1] injected at x = 0
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> xeff_vec_;
	std::ofstream output_file_;
};

namespace
{

// Effective ionized length: x_eff = integral_0^L (1 - x_HI) dx = sum_cells (1 - x_HI) * dx.
auto compute_effective_length(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real n_HI = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 1) / spmasses[1];
		const amrex::Real n_HII = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 2) / spmasses[2];
		const amrex::Real denom = n_HI + n_HII;
		if (denom <= 0.0_rt) {
			return 0.0_rt;
		}
		const amrex::Real x_HI = n_HI / denom;
		return cell_length * (1.0_rt - x_HI);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total_ionized_length = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total_ionized_length, amrex::ParallelContext::CommunicatorSub());
	return total_ionized_length;
}

// Median temperature of ionized ("cavity") cells, x_HII > 0.90. Gathered to the IO processor.
auto compute_cavity_median_temperature(amrex::MultiFab const &state_mf) -> double
{
	std::vector<double> cavity_temps;

	for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();

		// In GPU builds, MultiFab data resides on device; copy to pinned host memory before CPU access.
		amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
		static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
		amrex::Gpu::synchronize();

		const auto state = host_fab.const_array();

		amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
			const amrex::Real rho = state(i, j, k, HydroSystem<DTypeFront1D>::density_index);
			const amrex::Real Eint = state(i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index);
			const amrex::Real n_HI_cell = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 1) / spmasses[1];
			const amrex::Real n_HII_cell = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 2) / spmasses[2];
			const amrex::Real denom = n_HI_cell + n_HII_cell;
			if (denom <= 0.0_rt) {
				return;
			}
			const amrex::Real x_HII = n_HII_cell / denom;
			if (x_HII <= 0.90_rt) {
				return;
			}

			burn_t bstate;
			for (int nn = 0; nn < NumSpec; ++nn) {
				bstate.xn[nn] = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + nn) / spmasses[nn];
			}
			bstate.rho = rho;
			bstate.e = Eint / rho;
			bstate.T = 1.0e4; // initial guess
			eos(eos_input_re, bstate);
			cavity_temps.push_back(bstate.T);
		});
	}

	// Gather all cavity temperatures to the IO processor and compute the median there.
	const int num_local = static_cast<int>(cavity_temps.size());
	auto num_local_vec = amrex::ParallelDescriptor::Gather(num_local, amrex::ParallelDescriptor::IOProcessorNumber());

	amrex::Vector<int> recvcnt;
	amrex::Vector<int> disp;
	std::vector<double> all_temps;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		recvcnt.resize(num_local_vec.size());
		disp.resize(num_local_vec.size());
		int ntot = 0;
		disp[0] = 0;
		for (int r = 0, n = static_cast<int>(num_local_vec.size()); r < n; ++r) {
			recvcnt[r] = num_local_vec[r];
			ntot += num_local_vec[r];
			if (r + 1 < n) {
				disp[r + 1] = disp[r] + num_local_vec[r];
			}
		}
		all_temps.resize(ntot);
	} else {
		recvcnt.resize(1);
		disp.resize(1);
		all_temps.resize(1);
	}

	static double static_val = 0.0;
	const double *send_ptr = cavity_temps.empty() ? &static_val : cavity_temps.data();
	double *recv_ptr = all_temps.empty() ? &static_val : all_temps.data();
	amrex::ParallelDescriptor::Gatherv(send_ptr, num_local, recv_ptr, recvcnt, disp, amrex::ParallelDescriptor::IOProcessorNumber());

	double T_median = 0.0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const int ntot = static_cast<int>(all_temps.size());
		if (ntot > 0) {
			std::sort(all_temps.begin(), all_temps.end());
			T_median = (ntot % 2 == 0) ? 0.5 * (all_temps[ntot / 2 - 1] + all_temps[ntot / 2]) : all_temps[ntot / 2];
		}
	}
	amrex::ParallelDescriptor::Bcast(&T_median, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	return T_median;
}

} // namespace

template <>
void RadSystem<DTypeFront1D>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	// Planar ionizing photon flux injected in the first cell (adjacent to x = 0). radEnergy is a
	// luminosity volume density [erg s^-1 cm^-3]: F * E_gamma is the energy flux [erg cm^-2 s^-1],
	// and dividing by the cell width dx[0] gives the volumetric injection rate.
	amrex::ParmParse const pp("photoionize");
	amrex::Real flux = 1.0e11_rt;
	pp.query("flux", flux);

	const amrex::Real E_gamma = RadSystem<DTypeFront1D>::GetChemBandQuanta(0);
	const amrex::Real src = flux * E_gamma / dx[0];

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { radEnergy(i, j, k) = (i == 0) ? src : 0.0_rt; });
}

template <> void QuokkaSimulation<DTypeFront1D>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species, temperature, and flux
	amrex::ParmParse const pp("photoionize");
	userData_.small_temp = 1e-2;
	userData_.small_dens = 1e-60;
	userData_.temperature = 1.0e2;
	userData_.primary_species_1 = 1.0e-10_rt;
	userData_.primary_species_2 = 1.0e2_rt;
	userData_.primary_species_3 = 1.0e-10_rt;
	userData_.flux = 1.0e11_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("flux", userData_.flux);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_.open("dtype_front_1d_length.csv");
		userData_.output_file_ << "time,x_effective\n";
	}
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFront1D>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFront1D>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 0.0_rt;
}

template <> void QuokkaSimulation<DTypeFront1D>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	std::array<Real, NumSpec> numdens = {-1.0};
	numdens[0] = userData_.primary_species_1;
	numdens[1] = userData_.primary_species_2;
	numdens[2] = userData_.primary_species_3;

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
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DTypeFront1D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_floor_;
			state_cc(i, j, k, RadSystem<DTypeFront1D>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront1D>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
			state_cc(i, j, k, RadSystem<DTypeFront1D>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0_rt;
		}
		state_cc(i, j, k, RadSystem<DTypeFront1D>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::gasDensity_index) = rhotot;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::x1GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::x2GasMomentum_index) = 0.0_rt;
		state_cc(i, j, k, RadSystem<DTypeFront1D>::x3GasMomentum_index) = 0.0_rt;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + nn) =
			    state.xn[nn] * spmasses[nn]; // scalar indices carry partial densities instead of number densities
		}
	});
}

template <> void QuokkaSimulation<DTypeFront1D>::computeAfterTimestep()
{
	const int lev = 0;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const amrex::Real x_effective = compute_effective_length(state_new_cc_[lev], dx);
	const amrex::Real t = tNew_[lev];
	userData_.xeff_vec_.push_back(x_effective);
	userData_.t_vec_.push_back(t);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << t << ',' << x_effective << '\n';
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.3;

	// Problem initialization
	QuokkaSimulation<DTypeFront1D> sim;

	// initialize
	sim.setInitialConditions();
	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.plotfileInterval_ = -1;

	sim.evolve();

	int status = 0;

	// Measured ionized-region temperature (fixed, end-of-run cavity median).
	const double T_i = compute_cavity_median_temperature(sim.state_new_cc_[0]);

	// Analytic planar D-type reference solution, evaluated with the measured T_i.
	//   alpha_B(T_i) = 2.6e-13 (T_i/1e4)^-0.7
	//   c_i          = sqrt(2 k_B T_i / m_H)             (mu = 1/2, ionized H)
	//   x_S          = F / (alpha_B n_0^2)               (fixed Stromgren length)
	//   x_i(t)       = x_S [1 + (5/4) sqrt(4/3) c_i t / x_S]^(4/5)   (momentum-conserving)
	const double n_0 = sim.userData_.primary_species_2; // initial neutral-H number density
	const double F = sim.userData_.flux;
	const double alpha_B = 2.6e-13 * std::pow(T_i / 1.0e4, -0.7);
	const double c_i = std::sqrt(2.0 * C::k_B * T_i / C::m_p);
	const double x_S = F / (alpha_B * n_0 * n_0);
	const double K = std::sqrt(4.0 / 3.0);

	auto x_analytic = [=](double t_shifted) -> double {
		if (t_shifted <= 0.0) {
			return x_S;
		}
		return x_S * std::pow(1.0 + (5.0 / 4.0) * K * c_i * t_shifted / x_S, 4.0 / 5.0);
	};

	// Formation-phase time offset: t_form = first time x_eff reaches x_S (linear interpolation).
	double t_form = 0.0;
	{
		const auto &tv = sim.userData_.t_vec_;
		const auto &xv = sim.userData_.xeff_vec_;
		bool found = false;
		for (size_t i = 1; i < tv.size(); ++i) {
			if (xv[i] >= x_S) {
				const double x0 = xv[i - 1];
				const double x1 = xv[i];
				const double frac = (x1 > x0) ? (x_S - x0) / (x1 - x0) : 0.0;
				t_form = tv[i - 1] + frac * (tv[i] - tv[i - 1]);
				found = true;
				break;
			}
		}
		if (!found && !tv.empty()) {
			// Front never reached x_S; leave t_form = 0 (the check below will flag the mismatch).
			t_form = 0.0;
		}
	}

	// Check 1: numerical effective length vs analytic D-type front at end of simulation.
	if (!sim.userData_.xeff_vec_.empty()) {
		const double t_end = sim.userData_.t_vec_.back();
		const double x_eff_end = sim.userData_.xeff_vec_.back();
		const double x_ref = x_analytic(t_end - t_form);
		const double percent_diff = std::abs(x_eff_end - x_ref) / x_ref * 100.0;
		const double tol_percent = 15.0;

		amrex::Print() << "Measured ionized-cavity temperature T_i = " << T_i << " K\n";
		amrex::Print() << "Stromgren length x_S = " << x_S << " cm, c_i = " << c_i << " cm/s, t_form = " << t_form << " s\n";
		amrex::Print() << "Analytic D-type front length: " << x_ref << " cm\n";
		amrex::Print() << "Effective ionized length:     " << x_eff_end << " cm\n";
		amrex::Print() << "Difference: " << percent_diff << " percent (tolerance: " << tol_percent << " percent)\n";

		if (percent_diff > tol_percent) {
			amrex::Print() << "Test FAILED: ionized length differs from analytic D-type solution by more than " << tol_percent << " percent.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: ionized length matches analytic D-type solution within " << tol_percent << " percent.\n";
		}
	}

	// Check 2: lightweight sanity check that the measured ionized-region temperature is physical.
	if (T_i < 5.0e3 || T_i > 2.0e4) {
		amrex::Print() << "Test FAILED: measured ionized-cavity temperature " << T_i << " K is outside the physical range [5000, 20000] K.\n";
		status = 1;
	} else {
		amrex::Print() << "Test passed: ionized-cavity temperature " << T_i << " K is within the physical range [5000, 20000] K.\n";
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		const int n = static_cast<int>(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> t_Myr(n);
		std::vector<amrex::Real> x_eff_pc(n);
		std::vector<amrex::Real> x_ref_pc(n);
		for (int i = 0; i < n; ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			x_eff_pc[i] = sim.userData_.xeff_vec_[i] / cm_per_pc;
			x_ref_pc[i] = x_analytic(sim.userData_.t_vec_[i] - t_form) / cm_per_pc;
		}
		matplotlibcpp::clf();
		std::map<std::string, std::string> numerical_args;
		numerical_args["label"] = "numerical";
		numerical_args["color"] = "C0";
		std::map<std::string, std::string> analytical_args;
		analytical_args["label"] = "analytic (planar D-type)";
		analytical_args["color"] = "k";
		analytical_args["linestyle"] = "--";
		matplotlibcpp::plot(t_Myr, x_eff_pc, numerical_args);
		matplotlibcpp::plot(t_Myr, x_ref_pc, analytical_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("ionized length (pc)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dtype_front_1d_length.pdf");
	}
#endif

	amrex::Print() << "Finished." << '\n';
	return status;
}
