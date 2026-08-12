/// \file testDTypeFront1D.cpp
/// \brief Defines a 1D planar test of a beamed radiation source injected through SetRadSource.
///
/// A constant photon flux F [photons cm^-2 s^-1] is injected into the first cell (adjacent to x = 0) of a
/// uniform, cold, neutral hydrogen slab. Both the radiation energy source and the companion radiation flux
/// source are set, with flux = c * E, so the injected radiation is fully beamed along +x. Only the thermal
/// band (group 0) is fed; the ionizing band (group 1) is left dark, which isolates thermal-band transport
/// from the photochemistry.
///
/// With zero thermal opacity the injected radiation free-streams at the reduced speed of light, giving a
/// top-hat profile whose analytic solution is
///
///   E_gamma(x, t) = F * E_photon / c   for x < chat * t,   0 otherwise.
///
/// The test checks the front position (chat * t), the plateau energy budget, and that the ionizing band
/// stays at the radiation floor.

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

// Mean energy of a photon in the injected band. This is the mean energy of the single chemistry band
// [3.29e15, 1.5e16] Hz (see CMakeLists.txt CHEM_BANDS), retained as the luminosity normalization so the
// injected luminosity is unchanged from the photoionizing version of this problem.
constexpr double E_photon = 0.5 * (3.29e15 + 1.50e16) * C::hplanck; // erg
// Radiation energy-density floor. This is a physically meaningful, negligible photon-number density
// (1e-10 cm^-3, vs the ~hundreds cm^-3 of the injected beam) converted to a radiation energy density. Dark
// cells are initialized to exactly this floor (see setInitialConditionsOnGrid), following the best practice
// of RadStreaming / RadhydroShockMultigroup instead of seeding an unphysical 1e-99.
constexpr double Erad_floor_ = 1.0e-10 * E_photon; // erg cm^-3

// Gray opacity of the thermal band [cm^2 g^-1], set at runtime from photoionize.kappa_thermal.
// Managed memory so the device-side opacity function can read it.
AMREX_GPU_MANAGED double kappa_thermal = 0.0; // NOLINT
// Temperature above which the thermal-band opacity is destroyed (dust sublimation in ionized gas).
AMREX_GPU_MANAGED double T_dust_destroy = 0.0; // NOLINT

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
	// 2 radiation groups: group 0 = thermal (non-ionizing), group 1 = ionizing (the chemistry band).
	// Chemistry bands must be the last groups; see radiation_system.hpp.
	static constexpr int nGroups = 2;
};

template <> struct RadSystem_Traits<DTypeFront1D> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	// beta_order = 0: no O(v/c) radiation-momentum kick, so the beam free-streams without dragging the gas.
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::hplanck; // radBoundaries below are frequencies in Hz
	// Group frequency boundaries [Hz]: group 0 = thermal (non-ionizing) [2.5e15, 3.29e15], group 1 =
	// ionizing [3.29e15, 1.5e16]. The last group coincides with the chemistry band (ChemBands below).
	//
	// The thermal band sits just below the Lyman edge on purpose. Opacity in Quokka is pure absorption,
	// so a band with non-zero opacity also *emits* the local blackbody. Placing the band deep on the Wien
	// tail of the gas temperature keeps the emission term tractable when kappa_thermal is turned on, while
	// the cold neutral gas (100 K) emits nothing at all and can absorb freely.
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFront1D>::nGroups + 1> radBoundaries{2.5e15, 3.29e15, 1.50e16};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
};

template <> struct SimulationData<DTypeFront1D> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real flux{}; // photon flux F [photons cm^-2 s^-1] injected at x = 0
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> xfront_vec_;
	std::ofstream output_file_;
};

namespace
{

// Plateau radiation energy density of a free-streaming beam carrying photon flux F: E = F * E_photon / c.
// Note this is independent of the reduced speed of light.
auto compute_plateau_erad(amrex::Real flux) -> amrex::Real { return flux * E_photon / C::c_light; }

// Position of the radiation front: the right edge of the outermost cell whose thermal-band radiation energy
// density exceeds Erad_threshold.
auto compute_front_position(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real Erad_threshold) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];
	const amrex::Real threshold = Erad_threshold;
	const int erad_index = RadSystem<DTypeFront1D>::radEnergy_index;

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		if (state[box_no](i, j, k, erad_index) < threshold) {
			return 0.0_rt;
		}
		return x_lo + static_cast<amrex::Real>(i + 1) * cell_length;
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real x_front = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Max(x_front, amrex::ParallelContext::CommunicatorSub());
	return x_front;
}

// Domain-integrated radiation energy of group g: sum_cells Erad_g * dx  [erg cm^-2 in 1D].
auto compute_group_total_erad(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, int g) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const int erad_index = RadSystem<DTypeFront1D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g;

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		return cell_length * state[box_no](i, j, k, erad_index);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total, amrex::ParallelContext::CommunicatorSub());
	return total;
}

} // namespace

template <>
void RadSystem<DTypeFront1D>::SetRadSource(array_t &radEnergy, array_t &radFlux, const amrex::Box &indexRange,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	// Planar photon flux injected in the first cell (adjacent to x = 0). radEnergy is a luminosity volume
	// density [erg s^-1 cm^-3]: F * E_photon is the energy flux [erg cm^-2 s^-1], and dividing by the cell
	// width dx[0] gives the volumetric injection rate.
	//
	// The companion flux source is set to c * (energy source), i.e. the injected radiation has a reduced flux
	// of unity and therefore free-streams along +x instead of spreading isotropically.
	//
	// Only the thermal band (group 0) is fed. The ionizing band (group 1) is left dark so that this problem
	// exercises thermal-band transport in isolation from the photochemistry.
	amrex::ParmParse const pp("photoionize");
	amrex::Real flux = 1.0e11_rt;
	pp.query("flux", flux);

	const amrex::Real src_thermal = flux * E_photon / dx[0];

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			const amrex::Real src = ((g == 0) && (i == 0)) ? src_thermal : 0.0_rt;
			radEnergy(i, j, k, g) = src;
			radFlux(i, j, k, 3 * g + 0) = C::c_light * src;
			radFlux(i, j, k, 3 * g + 1) = 0.0_rt;
			radFlux(i, j, k, 3 * g + 2) = 0.0_rt;
		}
	});
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
	pp.query("kappa_thermal", kappa_thermal);   // gray opacity of the thermal band [cm^2 g^-1]
	pp.query("T_dust_destroy", T_dust_destroy); // dust-destruction temperature [K]; 0 disables
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
		userData_.output_file_.open("dtype_front_1d_front.csv");
		userData_.output_file_ << "time,x_front\n";
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

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFront1D>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/, const double Tgas)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	// Group 0 (thermal) carries a dust-like gray opacity, zero by default so the injected beam free-streams;
	// group 1 (ionizing) has none — its interaction with the gas would be photoionization, handled by the
	// photochemistry network rather than by this opacity.
	//
	// The opacity is destroyed above T_dust_destroy, mimicking dust sublimation in hot gas. Opacity here is
	// pure absorption, so opaque gas also emits its local blackbody; dust-free hot gas neither absorbs nor
	// emits in this band, which is both the correct physics and what keeps the energy solve well behaved.
	const double kappa_0 = (T_dust_destroy > 0.0) ? kappa_thermal * std::exp(-Tgas / T_dust_destroy) : kappa_thermal;
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = (i == 0) ? kappa_0 : 0.0;
	}
	return exponents_and_values;
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
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	const amrex::Real Erad_threshold = 0.5 * compute_plateau_erad(userData_.flux);
	const amrex::Real x_front = compute_front_position(state_new_cc_[lev], dx, prob_lo, Erad_threshold);
	const amrex::Real t = tNew_[lev];
	userData_.xfront_vec_.push_back(x_front);
	userData_.t_vec_.push_back(t);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << t << ',' << x_front << '\n';
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

	const double F = sim.userData_.flux;
	const double t_end = sim.userData_.t_vec_.back();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
	const double Lx = sim.geom[0].ProbHiArray()[0] - sim.geom[0].ProbLoArray()[0];

	// Analytic free-streaming solution: a top-hat of height F * E_photon / c reaching x = chat * t.
	const double Erad_plateau = compute_plateau_erad(F);
	auto x_analytic = [=](double t_now) -> double { return c_hat * t_now; };

	// Check 1: the beamed source (flux = c * E) must free-stream at the reduced speed of light. Without the
	// flux source the radiation would spread isotropically at chat / sqrt(3), so this check is what verifies
	// that SetRadSource actually injected a directed flux.
	{
		const double x_front = sim.userData_.xfront_vec_.back();
		const double x_ref = x_analytic(t_end);
		const double percent_diff = std::abs(x_front - x_ref) / x_ref * 100.0;
		const double tol_percent = 5.0;

		amrex::Print() << "Free-streaming front position: " << x_front << " cm\n";
		amrex::Print() << "Analytic front position:      " << x_ref << " cm (chat * t)\n";
		amrex::Print() << "Difference: " << percent_diff << " percent (tolerance: " << tol_percent << " percent)\n";

		if (x_ref >= Lx) {
			amrex::Print() << "Test FAILED: the analytic front has left the domain; reduce stop_time.\n";
			status = 1;
		} else if (percent_diff > tol_percent) {
			amrex::Print() << "Test FAILED: radiation front differs from chat * t by more than " << tol_percent << " percent.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: radiation front free-streams at chat within " << tol_percent << " percent.\n";
		}
	}

	// Check 2: energy budget of the thermal band. With zero opacity the band is transparent and conserves
	// energy exactly, so the domain-integrated Erad must equal the injected amount. Together with Check 1
	// (which fixes the width of the top-hat) this also pins down its height, F * E_photon / c.
	{
		const double E_therm = compute_group_total_erad(sim.state_new_cc_[0], dx, 0);
		// The thermal group carries the code's internal chat/c source factor (see source_terms_multi_group.hpp).
		const double injected_thermal = (c_hat / C::c_light) * F * E_photon * t_end;
		const double therm_frac = E_therm / injected_thermal;
		const double tau_dom = kappa_thermal * sim.userData_.primary_species_2 * spmasses[1] * Lx; // domain optical depth

		amrex::Print() << "Thermal band integrated Erad: " << E_therm << " (injected " << injected_thermal << ", tau_dom = " << tau_dom << ")\n";
		amrex::Print() << "Plateau Erad (analytic):      " << Erad_plateau << " erg cm^-3\n";

		if (std::abs(therm_frac - 1.0) > 0.03) {
			amrex::Print() << "Test FAILED: transparent thermal band energy is " << therm_frac << " of injected (expected 1 within 3%).\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: transparent thermal band conserves energy (Erad/injected = " << therm_frac << ").\n";
		}
	}

	// Check 3: the ionizing band receives no source at all, so it must stay at the radiation floor.
	{
		const double E_ion = compute_group_total_erad(sim.state_new_cc_[0], dx, 1);
		const double E_ion_floor = Erad_floor_ * Lx;
		const double ion_frac = E_ion / E_ion_floor;

		amrex::Print() << "Ionizing band integrated Erad: " << E_ion << " (floor " << E_ion_floor << ")\n";

		if (ion_frac > 1.01) {
			amrex::Print() << "Test FAILED: unsourced ionizing band rose above the radiation floor (Erad/floor = " << ion_frac << ").\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: unsourced ionizing band stays at the radiation floor (Erad/floor = " << ion_frac << ").\n";
		}
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		const int n = static_cast<int>(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> t_Myr(n);
		std::vector<amrex::Real> x_front_pc(n);
		std::vector<amrex::Real> x_ref_pc(n);
		for (int i = 0; i < n; ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			x_front_pc[i] = sim.userData_.xfront_vec_[i] / cm_per_pc;
			x_ref_pc[i] = x_analytic(sim.userData_.t_vec_[i]) / cm_per_pc;
		}
		matplotlibcpp::clf();
		std::map<std::string, std::string> numerical_args;
		numerical_args["label"] = "numerical";
		numerical_args["color"] = "C0";
		std::map<std::string, std::string> analytical_args;
		analytical_args["label"] = "analytic (free-streaming)";
		analytical_args["color"] = "k";
		analytical_args["linestyle"] = "--";
		matplotlibcpp::plot(t_Myr, x_front_pc, numerical_args);
		matplotlibcpp::plot(t_Myr, x_ref_pc, analytical_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("radiation front position (pc)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dtype_front_1d_front.pdf");
	}
#endif

	amrex::Print() << "Finished." << '\n';
	return status;
}
