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
// #include "radiation/radiation_system.hpp"
#include "radiation/radiation_dust_system.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
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

constexpr double nu_IR_lo = 0;
constexpr double nu_IR_optical = 4.3e14;
constexpr double nu_ion_lo = 3.29e15;
constexpr double nu_ion_hi = 1.50e16;

// Mean energy of an ionizing photon in the single chemistry band [3.29e15, 1.5e16] Hz (see
// CMakeLists.txt CHEM_BANDS); this matches RadSystem::GetChemBandQuanta(0).
constexpr double E_photon = 0.5 * (nu_ion_lo + nu_ion_hi) * C::hplanck; // erg
// Radiation energy-density floor. Rather than the arbitrary blackbody-at-0.01-K value used by the 3D
// DTypeFront, this is a physically meaningful, negligible photon-number density (1e-10 cm^-3, vs the
// ~hundreds cm^-3 of the ionizing field) converted to a radiation energy density. Dark cells are
// initialized to exactly this floor (see setInitialConditionsOnGrid), following the best practice of
// RadStreaming / RadhydroShockMultigroup instead of seeding an unphysical 1e-99.
constexpr double Erad_floor_ = 1.0e-10 * E_photon; // erg cm^-3

constexpr int IRBand = 0;
constexpr int OpticalBand = 1;
constexpr int IonizingBand = 2;

template <> struct quokka::EOS_Traits<DTypeFront1D> {
	static constexpr double mean_molecular_weight = C::m_p;
	static constexpr double gamma = 5. / 3.;
};

template <> struct ISM_Traits<DTypeFront1D> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
};

template <> struct Physics_Traits<DTypeFront1D> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// 3 radiation groups: group 0 = IR, group 1 = optical/UV (both thermal), group 2 = ionizing (the
	// chemistry band). Chemistry bands must be the last groups; see radiation_system.hpp.
	static constexpr int nGroups = 3;
};

template <> struct RadSystem_Traits<DTypeFront1D> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 1;
	static constexpr double energy_unit = C::hplanck; // radBoundaries below are frequencies in Hz
	// Group frequency boundaries [Hz]: group 0 = IR [3e13, 4.3e14], group 1 = optical/UV [4.3e14,
	// 3.29e15], group 2 = ionizing [3.29e15, 1.5e16]. The last group coincides with the chemistry
	// band (ChemBands below). The two thermal groups are transparent (zero opacity), so the exact IR/
	// optical boundaries are cosmetic.
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFront1D>::nGroups + 1> radBoundaries{nu_IR_lo, nu_IR_optical, nu_ion_lo, nu_ion_hi};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
};

AMREX_GPU_MANAGED double kappa_IR = 0.0;      // NOLINT
AMREX_GPU_MANAGED double kappa_optical = 0.0; // NOLINT

template <> struct SimulationData<DTypeFront1D> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real ion_flux{};	    // ionizing photon flux F [photons cm^-2 s^-1] injected at x = 0
	amrex::Real optical_flux{}; // optical photon flux F [photons cm^-2 s^-1] injected at x = 0
	amrex::Real IR_flux{};	    // IR photon flux F [photons cm^-2 s^-1] injected at x = 0
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> leff_vec_;
	amrex::Vector<amrex::Real> E_optical_vec_; // domain-integrated optical-band Erad [erg cm^-2] vs. time
	amrex::Vector<amrex::Real> E_IR_vec_;	   // domain-integrated IR-band Erad [erg cm^-2] vs. time
	// Time-integrated gas blackbody emission into each thermal band, accumulated in computeAfterTimestep via
	// trapezoidal integration of compute_group_emission_rate. [erg cm^-2], running total vs. time.
	amrex::Vector<amrex::Real> E_optical_emitted_vec_;
	amrex::Vector<amrex::Real> E_IR_emitted_vec_;
	amrex::Real last_rate_ir_{};	  // most recent gas emission rate into the IR band [erg cm^-2 s^-1]
	amrex::Real last_rate_optical_{}; // most recent gas emission rate into the optical band [erg cm^-2 s^-1]
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

// Domain-integrated instantaneous gas blackbody emission rate into group g: sum_cells c_hat * rho * kappa_g *
// a_rad * T_gas^4 * planckFraction_g(T_gas) * dx  [erg cm^-2 s^-1 in 1D]. This is the same emission term the
// radiation-matter coupling Newton solve uses (c_hat * dt * rho * kappaP * fourPiBoverC in
// source_terms_multi_group.hpp), evaluated directly from the gas state rather than backed out of the solve.
// At kappa_IR = kappa_optical = 1e-10 this rate is tiny per unit volume and time, but integrating it over the
// full run (t_end ~ 1e5-1e6 yr) and the ionized-cavity volume is what the plot's "expected" curve below is
// missing: the beamed source alone does not account for the gas's own thermal emission once photoionization
// heats it.
auto compute_group_emission_rate(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, int g) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real chat = c_hat;
	amrex::Real kappa_g = 0.0_rt;
	if (g == IRBand) {
		kappa_g = kappa_IR;
	} else if (g == OpticalBand) {
		kappa_g = kappa_optical;
	}
	const auto radBoundaries = RadSystem_Traits<DTypeFront1D>::radBoundaries;

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real rho = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::density_index);
		const amrex::Real Eint = state[box_no](i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index);

		burn_t bstate;
		for (int nn = 0; nn < NumSpec; ++nn) {
			bstate.xn[nn] = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + nn) / spmasses[nn];
		}
		bstate.rho = rho;
		bstate.e = Eint / rho;
		bstate.T = 1.0e4; // initial guess
		eos(eos_input_re, bstate);

		const auto Erad_g_thermal = RadSystem<DTypeFront1D>::ComputeThermalRadiationMultiGroup(bstate.T, radBoundaries);
		return cell_length * chat * rho * kappa_g * Erad_g_thermal[g];
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total, amrex::ParallelContext::CommunicatorSub());
	return total;
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
void RadSystem<DTypeFront1D>::SetRadSource(array_t &radEnergy, array_t &radFluxSource, const amrex::Box &indexRange,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	// Planar photon flux injected in the first cell (adjacent to x = 0). radEnergy is a luminosity
	// volume density [erg s^-1 cm^-3]: F * E_gamma is the energy flux [erg cm^-2 s^-1], and dividing by
	// the cell width dx[0] gives the volumetric injection rate.
	//
	// The SAME luminosity density is injected into all three groups (IR, optical, ionizing). The code
	// applies its own thermal-vs-chemical rescaling (thermal groups carry an internal chat/c factor;
	// see source_terms_multi_group.hpp), which is intentional: an identical radEnergySource across
	// groups corresponds to a physically identical luminosity in each band.
	amrex::ParmParse const pp("photoionize");
	amrex::Real ion_flux = 1.0e11_rt;
	pp.query("ion_flux", ion_flux);
	amrex::Real optical_flux = 1.0e11_rt;
	pp.query("optical_flux", optical_flux);
	amrex::Real IR_flux = 0.0_rt;
	pp.query("IR_flux", IR_flux);

	const amrex::Real E_ion = RadSystem<DTypeFront1D>::GetChemBandQuanta(0);
	const amrex::Real E_optical = 0.5_rt * (nu_IR_optical + nu_ion_lo) * C::hplanck; // mean photon energy of the optical band [erg]
	const amrex::Real E_IR = 0.5_rt * (nu_IR_lo + nu_IR_optical) * C::hplanck;	 // mean photon energy of the IR band [erg]
	const amrex::Real src_ion = ion_flux * E_ion / dx[0];
	const amrex::Real src_optical = optical_flux * E_optical / dx[0];
	const amrex::Real src_IR = IR_flux * E_IR / dx[0];

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			amrex::Real src = 0.0_rt;
			if (i == 0) {
				src = (g == OpticalBand) ? src_optical : ((g == IonizingBand) ? src_ion : ((g == IRBand) ? src_IR : 0.0_rt));
			}
			radEnergy(i, j, k, g) = src;
			// UpdateFlux() applies the same c_hat-scaled momentum-transfer physics to every group,
			// but the ionizing band is decoupled from that machinery (its Src is zeroed and injected
			// directly; see the Src_chem handling in AddSourceTermsMultiGroup). A nonzero flux source
			// here for the ionizing band would be absorbed and turned into a spurious gas momentum
			// kick scaled by c/c_hat, so leave it at zero and only inject via radEnergy above.
			radFluxSource(i, j, k, 3 * g + 0) = 0.0_rt; //(g == IonizingBand) ? 0.0_rt : C::c_light * src;
			radFluxSource(i, j, k, 3 * g + 1) = 0.0_rt;
			radFluxSource(i, j, k, 3 * g + 2) = 0.0_rt;
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
	userData_.ion_flux = 1.0e11_rt;
	userData_.optical_flux = 1.0e11_rt;
	userData_.IR_flux = 1.0e11_rt;
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("ion_flux", userData_.ion_flux);
	pp.query("optical_flux", userData_.optical_flux);
	pp.query("IR_flux", userData_.IR_flux);
	pp.query("kappa_IR", kappa_IR);
	pp.query("kappa_optical", kappa_optical);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_.open("dtype_front_1d_length.csv");
		userData_.output_file_ << "time,x_effective\n";
	}
}

// Hydrogen nucleus number density, n_H = n_HI + n_HII. Must be specialized: the base implementation
// returns rho / mean_molecular_weight, which for this problem (mean_molecular_weight = 1.0) is off by
// 1/m_p ~ 6e23. n_H enters the dust-gas coupling as coeff_n ~ n_H^2, so the base version leaves the
// dust-gas Newton solve degenerate (see ComputeDustTemperatureBateKeto).
template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFront1D>::ComputeNumberDensityH(double /*rho*/, amrex::GpuArray<Real, nmscalars_> const &massScalars) -> double
{
	return massScalars[1] / spmasses[1] + massScalars[2] / spmasses[2];
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFront1D>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = 0.0;
	}
	exponents_and_values[1][IRBand] = kappa_IR;
	exponents_and_values[1][OpticalBand] = kappa_optical;
	exponents_and_values[1][IonizingBand] = 0.0_rt;
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
	const amrex::Real x_effective = compute_effective_length(state_new_cc_[lev], dx);
	const amrex::Real t = tNew_[lev];
	userData_.leff_vec_.push_back(x_effective);
	userData_.t_vec_.push_back(t);
	userData_.E_IR_vec_.push_back(compute_group_total_erad(state_new_cc_[lev], dx, IRBand));
	userData_.E_optical_vec_.push_back(compute_group_total_erad(state_new_cc_[lev], dx, OpticalBand));

	// Trapezoidal time-integration of the gas blackbody emission rate into each thermal band (see
	// compute_group_emission_rate), giving a running total comparable to E_IR_vec_/E_optical_vec_.
	const amrex::Real rate_ir = compute_group_emission_rate(state_new_cc_[lev], dx, IRBand);
	const amrex::Real rate_optical = compute_group_emission_rate(state_new_cc_[lev], dx, OpticalBand);
	if (userData_.t_vec_.size() == 1) {
		userData_.E_IR_emitted_vec_.push_back(0.0_rt);
		userData_.E_optical_emitted_vec_.push_back(0.0_rt);
	} else {
		const amrex::Real dt_step = t - userData_.t_vec_[userData_.t_vec_.size() - 2];
		userData_.E_IR_emitted_vec_.push_back(userData_.E_IR_emitted_vec_.back() + 0.5_rt * (rate_ir + userData_.last_rate_ir_) * dt_step);
		userData_.E_optical_emitted_vec_.push_back(userData_.E_optical_emitted_vec_.back() +
							   0.5_rt * (rate_optical + userData_.last_rate_optical_) * dt_step);
	}
	userData_.last_rate_ir_ = rate_ir;
	userData_.last_rate_optical_ = rate_optical;

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
	const double F_1 = sim.userData_.ion_flux;
	const double F_2 = sim.userData_.optical_flux;
	const double epsilon_1 = RadSystem<DTypeFront1D>::GetChemBandQuanta(0);
	const double epsilon_2 = 0.5 * (nu_IR_optical + nu_ion_lo) * C::hplanck; // mean photon energy of the optical band [erg]
	const double alpha_B = 2.6e-13 * std::pow(T_i / 1.0e4, -0.7);
	const double mu = 0.5;
	const amrex::Real c_i = std::sqrt(C::k_B * T_i / (mu * C::m_p));
	const double x_S = F_1 / (alpha_B * n_0 * n_0);
	const double rho_0 = n_0 * spmasses[1];
	const double l_ch = 4 * C::c_light * C::c_light * C::k_B * C::k_B * T_i * T_i * F_1 / (std::pow(F_1 * epsilon_1 + F_2 * epsilon_2, 2) * alpha_B);
	const double t_ref = std::sqrt(rho_0 * std::pow(l_ch, 2.0) * C::c_light / (F_1 * epsilon_1 + F_2 * epsilon_2));
	const double zi = std::pow(2 * C::c_light * C::k_B * T_i * n_0 / (F_1 * epsilon_1 + F_2 * epsilon_2), 2.0);
	const double K = std::sqrt(4.0 / 3.0);

	auto l_analytic = [=](double t) -> double {
		const double tau = t / t_ref;
		const double x_rad = tau;
		const double x_gas = std::pow((25.0 / 12.0), 2.0 / 5.0) * std::pow(tau, 4.0 / 5.0);
		const double x = std::pow(std::pow(x_rad, 5.0 / 4.0) + std::pow(x_gas, 5.0 / 4.0), 4.0 / 5.0);
		const double l = l_ch * x;
		return l;
	};

	std::cout << "n_0 = " << n_0 << " cm^-3, F_1 = " << F_1 << " cm^-2 s^-1, F_2 = " << F_2 << " cm^-2 s^-1\n";
	std::cout << "Measured ionized-cavity temperature T_i = " << T_i << " K\n";
	std::cout << "Stromgren length x_S = " << x_S << " cm, c_i = " << c_i << " cm/s\n";
	std::cout << "Characteristic length l_ch = " << l_ch << " cm, characteristic time t_ref = " << t_ref << " s\n";
	std::cout << "Value of zi = " << zi << std::endl;

	auto x_analytic_old = [=](double t_shifted) -> double {
		if (t_shifted <= 0.0) {
			return x_S;
		}
		return x_S * std::pow(1.0 + (5.0 / 4.0) * K * c_i * t_shifted / x_S, 4.0 / 5.0);
	};

	// Formation-phase time offset: t_form = first time l_eff reaches l_S (linear interpolation).
	double t_form = 0.0;
	{
		const auto &tv = sim.userData_.t_vec_;
		const auto &xv = sim.userData_.leff_vec_;
		bool found = false;
		for (int i = 1; i < static_cast<int>(tv.size()); ++i) {
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
	if (!sim.userData_.leff_vec_.empty()) {
		const double t_end = sim.userData_.t_vec_.back();
		const double l_eff_end = sim.userData_.leff_vec_.back();
		const double l_ref = l_analytic(t_end);
		const double percent_diff = (l_eff_end - l_ref) / l_ref * 100.0;
		const double tol_percent = 15.0;

		amrex::Print() << "Measured ionized-cavity temperature T_i = " << T_i << " K\n";
		amrex::Print() << "Stromgren length x_S = " << x_S << " cm, c_i = " << c_i << " cm/s\n";
		amrex::Print() << "Analytic D-type front length: " << l_ref << " cm\n";
		amrex::Print() << "Effective ionized length:     " << l_eff_end << " cm\n";
		amrex::Print() << "Difference: " << percent_diff << " percent (tolerance: " << tol_percent << " percent)\n";

		if (std::abs(percent_diff) > tol_percent) {
			amrex::Print() << "Test FAILED: ionized length differs from analytic D-type solution by more than " << tol_percent << " percent.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: ionized length matches analytic D-type solution within " << tol_percent << " percent.\n";
		}
	}
	// if (!sim.userData_.leff_vec_.empty()) {
	// 	const double t_end = sim.userData_.t_vec_.back();
	// 	const double l_eff_end = sim.userData_.leff_vec_.back();
	// 	const double x_ref = x_analytic(t_end - t_form);
	// 	const double percent_diff = (l_eff_end - x_ref) / x_ref * 100.0;
	// 	const double tol_percent = 15.0;

	// 	amrex::Print() << "Measured ionized-cavity temperature T_i = " << T_i << " K\n";
	// 	amrex::Print() << "Stromgren length x_S = " << x_S << " cm, c_i = " << c_i << " cm/s, t_form = " << t_form << " s\n";
	// 	amrex::Print() << "Analytic D-type front length: " << x_ref << " cm\n";
	// 	amrex::Print() << "Effective ionized length:     " << l_eff_end << " cm\n";
	// 	amrex::Print() << "Difference: " << percent_diff << " percent (tolerance: " << tol_percent << " percent)\n";

	// 	if (percent_diff > tol_percent) {
	// 		amrex::Print() << "Test FAILED: ionized length differs from analytic D-type solution by more than " << tol_percent << " percent.\n";
	// 		status = 1;
	// 	} else {
	// 		amrex::Print() << "Test passed: ionized length matches analytic D-type solution within " << tol_percent << " percent.\n";
	// 	}
	// }

	// Check 2: lightweight sanity check that the measured ionized-region temperature is physical.
	if (T_i < 5.0e3 || T_i > 2.0e4) {
		amrex::Print() << "Test FAILED: measured ionized-cavity temperature " << T_i << " K is outside the physical range [5000, 20000] K.\n";
		status = 1;
	} else {
		amrex::Print() << "Test passed: ionized-cavity temperature " << T_i << " K is within the physical range [5000, 20000] K.\n";
	}

	// Checks 3 & 4: the two extra thermal bands (group 0 = IR, group 1 = optical) with zero opacity.
	{
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
		const double E_gamma = RadSystem<DTypeFront1D>::GetChemBandQuanta(0);
		const double t_end = sim.userData_.t_vec_.back();
		const double E_ir = compute_group_total_erad(sim.state_new_cc_[0], dx, 0);
		const double E_opt = compute_group_total_erad(sim.state_new_cc_[0], dx, 1);
		const double E_ion = compute_group_total_erad(sim.state_new_cc_[0], dx, 2);

		// Radiation energy injected per unit area over the run. The chemical band receives F*E_gamma*t;
		// thermal groups carry the code's internal chat/c source factor (see source_terms_multi_group.hpp).
		const double injected_chem = sim.userData_.ion_flux * E_gamma * t_end;
		const double injected_thermal = (c_hat / C::c_light) * injected_chem;

		amrex::Print() << "Thermal band (IR)  integrated Erad: " << E_ir << " ; (optical): " << E_opt << " ; ionizing: " << E_ion << '\n';
		amrex::Print() << "Injected: thermal (chat/c * F*E*t) = " << injected_thermal << " ; chem (F*E*t) = " << injected_chem << '\n';

		// Check 3: the two thermal bands have identical setup (same source, zero opacity) -> identical Erad.
		// Disabled: IR and optical bands are now injected using their own bands' mean photon energy
		// (E_IR, E_optical) rather than a shared E_gamma, so they are no longer expected to be identical.
		// const double band_rel_diff = (E_ir + E_opt > 0.0) ? std::abs(E_ir - E_opt) / (0.5 * (E_ir + E_opt)) : 0.0;
		// if (band_rel_diff > 1.0e-6) {
		// 	amrex::Print() << "Test FAILED: IR and optical thermal bands differ by " << band_rel_diff << " (should be identical).\n";
		// 	status = 1;
		// } else {
		// 	amrex::Print() << "Test passed: IR and optical thermal bands are identical (relative difference " << band_rel_diff << ").\n";
		// }

		// Check 4a: with zero opacity and reflecting boundaries (no escape), each thermal band conserves
		// energy, so its integrated Erad equals the injected amount.
		// Disabled: injected_thermal is derived from E_gamma (the ionizing band's photon energy), which no
		// longer matches the IR band's own injection energy (E_IR).
		// const double thermal_cons = std::abs(E_ir - injected_thermal) / injected_thermal;
		// if (thermal_cons > 0.03) {
		// 	amrex::Print() << "Test FAILED: thermal-band energy " << E_ir << " differs from injected " << injected_thermal << " by "
		// 		       << 100.0 * thermal_cons << "% (tolerance 3%).\n";
		// 	status = 1;
		// } else {
		// 	amrex::Print() << "Test passed: thermal-band energy conserved to " << 100.0 * thermal_cons << "% of injected (tolerance 3%).\n";
		// }

		// Check 4b: the ionizing band is depleted by photoionization, so its integrated Erad is below the
		// amount injected (unlike the transparent thermal bands).
		const double ion_frac = E_ion / injected_chem;
		if (ion_frac >= 1.0) {
			amrex::Print() << "Test FAILED: ionizing band is not depleted (Erad/injected = " << ion_frac << " >= 1).\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: ionizing band depleted by photoionization (Erad/injected = " << ion_frac << " < 1).\n";
		}
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		const int n = static_cast<int>(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> t_Myr(n);
		std::vector<amrex::Real> l_eff_pc(n);
		std::vector<amrex::Real> l_ref_pc(n);
		std::vector<amrex::Real> l_ref_old_pc(n);
		for (int i = 0; i < n; ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			l_eff_pc[i] = sim.userData_.leff_vec_[i] / cm_per_pc;
			l_ref_pc[i] = l_analytic(sim.userData_.t_vec_[i]) / cm_per_pc;
			l_ref_old_pc[i] = x_analytic_old(sim.userData_.t_vec_[i] - t_form) / cm_per_pc;
		}
		matplotlibcpp::clf();
		std::map<std::string, std::string> numerical_args;
		numerical_args["label"] = "numerical";
		numerical_args["color"] = "C0";
		std::map<std::string, std::string> analytical_args;
		analytical_args["label"] = "analytic (planar D-type, KM09)";
		analytical_args["color"] = "k";
		analytical_args["linestyle"] = "--";
		std::map<std::string, std::string> analytical_args_old;
		analytical_args_old["label"] = "analytic (planar D-type, Spitzer)";
		analytical_args_old["color"] = "green";
		analytical_args_old["linestyle"] = ":";
		matplotlibcpp::plot(t_Myr, l_eff_pc, numerical_args);
		matplotlibcpp::plot(t_Myr, l_ref_pc, analytical_args);
		matplotlibcpp::plot(t_Myr, l_ref_old_pc, analytical_args_old);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("ionized length (pc)");
		matplotlibcpp::title(std::format("beta_order = {}", RadSystem_Traits<DTypeFront1D>::beta_order));
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(std::format("./dtype_front_1d_length_beta_order{}.pdf", RadSystem_Traits<DTypeFront1D>::beta_order));

		// Domain-integrated optical/IR band energy vs. time, compared against the no-absorption
		// expectation from the injected source. Both thermal bands are injected with the same
		// volumetric luminosity density but carry the code's internal chat/c rescaling (see
		// source_terms_multi_group.hpp), so with zero opacity (no absorption) the expected
		// integrated energy is E_expected(t) = (chat/c) * flux * E_band * t.
		const double E_optical_photon = 0.5 * (nu_IR_optical + nu_ion_lo) * C::hplanck; // mean photon energy of the optical band [erg]
		const double E_IR_photon = 0.5 * (nu_IR_lo + nu_IR_optical) * C::hplanck;	// mean photon energy of the IR band [erg]
		std::vector<amrex::Real> E_optical(n);
		std::vector<amrex::Real> E_IR(n);
		std::vector<amrex::Real> E_optical_expected(n);
		std::vector<amrex::Real> E_IR_expected(n);
		std::vector<amrex::Real> E_optical_expected_plus_emitted(n);
		std::vector<amrex::Real> E_IR_expected_plus_emitted(n);
		for (int i = 0; i < n; ++i) {
			E_optical[i] = sim.userData_.E_optical_vec_[i];
			E_IR[i] = sim.userData_.E_IR_vec_[i];
			E_optical_expected[i] = (c_hat / C::c_light) * sim.userData_.optical_flux * E_optical_photon * sim.userData_.t_vec_[i];
			E_IR_expected[i] = (c_hat / C::c_light) * sim.userData_.IR_flux * E_IR_photon * sim.userData_.t_vec_[i];
			E_optical_expected_plus_emitted[i] = E_optical_expected[i] + sim.userData_.E_optical_emitted_vec_[i];
			E_IR_expected_plus_emitted[i] = E_IR_expected[i] + sim.userData_.E_IR_emitted_vec_[i];
		}
		matplotlibcpp::clf();
		std::map<std::string, std::string> optical_args;
		optical_args["label"] = "E_optical (numerical)";
		optical_args["color"] = "C1";
		std::map<std::string, std::string> optical_expected_args;
		optical_expected_args["label"] = "E_optical (injected, no absorption)";
		optical_expected_args["color"] = "C3";
		optical_expected_args["linestyle"] = "--";
		std::map<std::string, std::string> optical_expected_plus_emitted_args;
		optical_expected_plus_emitted_args["label"] = "E_optical (injected + gas emission (T_d = T_gas))";
		optical_expected_plus_emitted_args["color"] = "C3";
		optical_expected_plus_emitted_args["linestyle"] = ":";
		std::map<std::string, std::string> ir_args;
		ir_args["label"] = "E_IR (numerical)";
		ir_args["color"] = "C2";
		std::map<std::string, std::string> ir_expected_args;
		ir_expected_args["label"] = "E_IR (injected, no absorption)";
		ir_expected_args["color"] = "C4";
		ir_expected_args["linestyle"] = "--";
		std::map<std::string, std::string> ir_expected_plus_emitted_args;
		ir_expected_plus_emitted_args["label"] = "E_IR (injected + gas emission (T_d = T_gas))";
		ir_expected_plus_emitted_args["color"] = "C4";
		ir_expected_plus_emitted_args["linestyle"] = ":";
		matplotlibcpp::plot(t_Myr, E_optical, optical_args);
		matplotlibcpp::plot(t_Myr, E_optical_expected, optical_expected_args);
		// matplotlibcpp::plot(t_Myr, E_optical_expected_plus_emitted, optical_expected_plus_emitted_args);
		matplotlibcpp::plot(t_Myr, E_IR, ir_args);
		matplotlibcpp::plot(t_Myr, E_IR_expected, ir_expected_args);
		// matplotlibcpp::plot(t_Myr, E_IR_expected_plus_emitted, ir_expected_plus_emitted_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("domain-integrated radiation energy (erg cm^-2)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./dtype_front_1d_thermal_bands.pdf");
	}
#endif

	amrex::Print() << "Finished." << '\n';
	return status;
}
