/// \file testDTypeFront1D.cpp
/// \brief Defines a 1D planar test of dust reprocessing a beamed optical source into the IR.
///
/// There are three radiation groups: IR (group 0), optical (group 1) and an ionizing chemistry band
/// (group 2). Constant photon fluxes are injected in the first cell (adjacent to x = 0) of a uniform, cold,
/// dusty hydrogen slab: photoionize.flux into the OPTICAL band and photoionize.flux_ion into the ionizing
/// band. Both the radiation energy source and the companion radiation flux source are set, with
/// flux = c * E, so both inflows are fully beamed along +x. The IR band receives no source at all.
///
/// The two sourced bands are scaled differently inside the solver -- a thermal group's source is multiplied
/// by chat/c and a chemistry band's is not -- so the shipped fluxes differ by exactly c/chat = 1000 and
/// deliver equal energy. The ionizing band is transparent to the dust opacity, so photoionization is the
/// only process that removes it, and it ionizes the slab as it advances.
///
/// A separate dust temperature is solved for, with the gas-dust collisional coupling switched off
/// (radiation.dust_gas_interaction_coeff = 0), so the dust sits at radiative equilibrium and exchanges no
/// energy with the gas. Radiation momentum is still deposited, so the beam does accelerate the gas.
///
/// The dust opacity is gray within each band and set at runtime (photoionize.kappa1 for the IR,
/// photoionize.kappa2 for the optical), with the optical opacity much the larger, as for real dust. The
/// chain the test exercises is therefore:
///
///   beamed optical inflow -> absorbed by dust -> dust heats to radiative equilibrium -> re-emitted as IR
///
/// Opacity in Quokka is pure absorption, so an opaque group also emits its share of the local blackbody,
/// which is what supplies the re-emission. The dust settles where absorption balances emission,
/// a * T^4 = E_IR + (kappa_opt / kappa_IR) * E_optical, giving ~130 K for the shipped parameters (the dust
/// temperature is internal to the solver and is not stored in the state, so the checks below do not read it
/// directly). At that temperature h*nu/(k*T) = 37 at the IR/optical boundary, so the Planck function has
/// nothing left above the boundary and essentially all of the re-emission lands in the IR band rather than
/// back in the optical one. Behind the light front the optical band is then in pure attenuation,
///
///   E_optical(x) = (F * E_photon / c) * exp(-kappa_opt * rho * x),
///
/// and the energy it loses reappears in the IR. The IR is nearly transparent and escapes.
///
/// The test checks that the two thermal bands together conserve the energy injected into the optical one
/// (photoionization drains the ionizing band only, so it is budgeted separately), that the optical light
/// front sits near chat * t, that the radiation leaves the injection cell beamed (reduced flux of unity,
/// the only check here that exercises the flux source), that the dust reprocessed a substantial fraction of
/// the beam into the IR with the survivors following exp(-tau), and that the ionizing band's photon budget
/// balances against the ionized column it produced.

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "radiation/radiation_dust_system.hpp" // for the separate dust-temperature solver (see ISM_Traits below)
#include "radiation/radiation_system.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <cmath>
#include <fstream>
#include <map>
#include <numbers>
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

// Group indices. Group 0 is the IR band, group 1 the optical band, group 2 the ionizing chemistry band;
// chemistry bands must come last (see radiation_system.hpp).
constexpr int group_ir = 0;
constexpr int group_optical = 1;
constexpr int group_ionizing = 2;

// Fraction of the unattenuated beam energy density used to locate the optical light front. It has to sit
// below exp(-tau) at the front (~0.14 for the shipped opacity) so the threshold finds the light front and
// not the dust absorption depth, and far enough above the radiation floor to be unambiguous.
constexpr double front_threshold_fraction = 0.05;

// Gray dust opacities of the two thermal groups [cm^2 g^-1], set at runtime from photoionize.kappa1 (IR)
// and photoionize.kappa2 (optical). Both default to zero, i.e. a transparent domain. The ionizing band is
// always transparent to this gray opacity; it couples to the gas through photochemistry instead. Managed
// memory so the device-side opacity function can read them.
AMREX_GPU_MANAGED double kappa1 = 0.0; // NOLINT
AMREX_GPU_MANAGED double kappa2 = 0.0; // NOLINT

// Old dust-destruction knob, kept for reference:
// // Temperature above which the thermal-band opacity is destroyed (dust sublimation in ionized gas).
// AMREX_GPU_MANAGED double T_dust_destroy = 0.0; // NOLINT

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
	// 3 radiation groups: groups 0 and 1 = thermal (non-ionizing), group 2 = ionizing (the chemistry band).
	// Chemistry bands must be the last groups; see radiation_system.hpp.
	static constexpr int nGroups = 3;
};

template <> struct RadSystem_Traits<DTypeFront1D> {
	static constexpr double c_hat_over_c = c_hat / C::c_light;
	static constexpr double Erad_floor = Erad_floor_;
	// beta_order = 0: drop the O(v/c) terms in the radiation-matter coupling. Note this does not switch off
	// radiation pressure -- momentum from absorbed radiation is still deposited in the gas.
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::hplanck; // radBoundaries below are frequencies in Hz
	// Group frequency boundaries [Hz]: group 0 = IR (below 1e14 Hz, i.e. longward of 3 um), group 1 =
	// optical (1e14 Hz to the Lyman edge), group 2 = the ionizing chemistry band, which starts at the Lyman
	// edge (3.29e15 Hz) to match ChemBands below.
	//
	// The outermost two boundaries are deliberately set far outside the range that carries any energy, and
	// should be read as 0 and infinity. They are not physical band edges: ComputePlanckEnergyFractions
	// accumulates the Planck integral from zero, so group 0 receives the whole blackbody below
	// radBoundaries[1] no matter what radBoundaries[0] says, and emission above radBoundaries[2] is dropped
	// rather than assigned to the chemistry band, so radBoundaries[3] never enters the emission budget.
	//
	// The IR/optical split at 1e14 Hz is what makes the reprocessing clean. The dust settles at ~130 K here
	// (see the header comment), where h*nu/(k*T) = 37 at the split, so the Planck function has nothing left
	// above it: essentially all re-emission lands in the IR group and none of it back into the optical one.
	static constexpr amrex::GpuArray<double, Physics_Traits<DTypeFront1D>::nGroups + 1> radBoundaries{1.0e8, 1.0e14, 3.29e15, 1.0e19};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
};

template <> struct ISM_Traits<DTypeFront1D> {
	// Solve for a separate dust temperature rather than assuming T_dust == T_gas. With
	// radiation.dust_gas_interaction_coeff = 0 in the input file the gas-dust collisional term vanishes, so
	// the solver takes its decoupled branch (dust_model == 2 in radiation_dust_system.hpp): the dust
	// temperature is fixed purely by radiative equilibrium with the local radiation field, and no energy is
	// exchanged with the gas at all.
	//
	// Decoupled here means thermally decoupled only. Radiation momentum is a separate channel and is still
	// deposited, so the beam drives the gas: it reaches ~8e6 cm/s and evacuates the cells nearest the source
	// by a factor of a few hundred in density. The gas temperature therefore still varies widely, through
	// compression and expansion rather than through radiative heating.
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
};

template <> struct SimulationData<DTypeFront1D> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real flux{};	// optical photon flux [photons cm^-2 s^-1] injected at x = 0
	amrex::Real flux_ion{}; // ionizing photon flux [photons cm^-2 s^-1] injected at x = 0
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> xfront_vec_;
	std::ofstream output_file_;
};

namespace
{

// Plateau radiation energy density of a free-streaming beam carrying photon flux F: E = F * E_photon / c.
// Note this is independent of the reduced speed of light.
auto compute_plateau_erad(amrex::Real flux) -> amrex::Real { return flux * E_photon / C::c_light; }

// Position of the radiation front of group g: the right edge of the outermost cell whose radiation energy
// density exceeds Erad_threshold.
auto compute_front_position(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real Erad_threshold, int g) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];
	const amrex::Real threshold = Erad_threshold;
	const int erad_index = RadSystem<DTypeFront1D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g;

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

// Reduced flux f = F_x / (c * Erad) of group g in the injection cell (i == 0). Free-streaming radiation has
// f = 1; isotropic radiation has f = 0.
auto compute_injection_reduced_flux(amrex::MultiFab const &state_mf, int g) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const int erad_index = RadSystem<DTypeFront1D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g;
	const int frad_index = RadSystem<DTypeFront1D>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g;

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		if (i != 0) {
			return 0.0_rt;
		}
		return state[box_no](i, j, k, frad_index) / (C::c_light * state[box_no](i, j, k, erad_index));
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real reduced_flux = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Max(reduced_flux, amrex::ParallelContext::CommunicatorSub());
	return reduced_flux;
}

// Ionized hydrogen column: sum_cells n_HII * dx  [cm^-2 in 1D].
auto compute_ionized_column(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real mass_HII = spmasses[2];

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		return cell_length * state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 2) / mass_HII;
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real column = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(column, amrex::ParallelContext::CommunicatorSub());
	return column;
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
	// The optical and ionizing bands are fed; the IR band starts dark and is filled purely by the dust's own
	// thermal re-emission of the absorbed optical light.
	//
	// The two sourced bands take different internal scalings: a thermal group's source is multiplied by
	// chat/c, a chemistry band's is not (see source_terms_multi_group.hpp). The shipped fluxes differ by
	// exactly that factor of c/chat = 1000, so the two bands receive the same injected energy and the
	// energy budget below would be off by three orders of magnitude if either scaling were wrong.
	amrex::ParmParse const pp("photoionize");
	amrex::Real flux = 1.0e11_rt;
	pp.query("flux", flux);
	amrex::Real flux_ion = 0.0_rt;
	pp.query("flux_ion", flux_ion);

	const amrex::Real src_optical = flux * E_photon / dx[0];
	const amrex::Real src_ionizing = flux_ion * E_photon / dx[0];

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			amrex::Real src = 0.0_rt;
			if (i == 0) {
				if (g == group_optical) {
					src = src_optical;
				} else if (g == group_ionizing) {
					src = src_ionizing;
				}
			}
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
	userData_.flux_ion = 0.0_rt;
	pp.query("kappa1", kappa1); // gray opacity of the thermal band, group 0 [cm^2 g^-1]
	pp.query("kappa2", kappa2); // gray opacity of the optical band, group 1 [cm^2 g^-1]
	// Old dust-destruction knob:
	//
	// pp.query("T_dust_destroy", T_dust_destroy); // dust-destruction temperature [K]; 0 disables
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("flux", userData_.flux);
	pp.query("flux_ion", userData_.flux_ion);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_.open("dtype_front_1d_front.csv");
		userData_.output_file_ << "time,x_front\n";
	}
}

// ComputePlanckOpacity / ComputeFluxMeanOpacity are only consulted by the single-group solver. This problem
// runs with nGroups = 3, so the group opacities come from DefineOpacityExponentsAndLowerValues below and
// these specializations are dead code:
//
// template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFront1D>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
// {
//	return 0.0_rt;
// }
//
// template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DTypeFront1D>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
// {
//	return 0.0_rt;
// }

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFront1D>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	// Each thermal group carries its own constant gray opacity; the ionizing (chemistry) band is left
	// transparent. The trailing entry (i == nGroups_) is the unused upper band edge.
	//
	// Old temperature-dependent form, which destroyed the opacity above T_dust_destroy to mimic dust
	// sublimation in hot gas. Opacity here is pure absorption, so opaque gas also emits its local blackbody;
	// dust-free hot gas neither absorbs nor emits in this band, which keeps the energy solve well behaved:
	//
	// const double kappa_0 = (T_dust_destroy > 0.0) ? kappa_thermal * std::exp(-Tgas / T_dust_destroy) : kappa_thermal;
	const amrex::GpuArray<double, nGroups_> kappa_g{kappa1, kappa2, 0.0};
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = (i < nGroups_) ? kappa_g[i] : 0.0;
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
	// Track the optical band, which carries the beam. The threshold is well below the unattenuated plateau
	// because dust absorption thins the beam as it advances: at the light front the optical energy density
	// is down by exp(-tau) ~ 0.14, so a 0.5 threshold would report the absorption depth rather than the
	// front. See front_threshold_fraction.
	const amrex::Real Erad_threshold = front_threshold_fraction * compute_plateau_erad(userData_.flux);
	const amrex::Real x_front = compute_front_position(state_new_cc_[lev], dx, prob_lo, Erad_threshold, group_optical);
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
	const double F_ion = sim.userData_.flux_ion;
	const double t_end = sim.userData_.t_vec_.back();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
	const double Lx = sim.geom[0].ProbHiArray()[0] - sim.geom[0].ProbLoArray()[0];

	// Analytic free-streaming solution: a top-hat of height F * E_photon / c reaching x = chat * t.
	auto x_analytic = [=](double t_now) -> double { return c_hat * t_now; };

	// Check 1: conservation of energy in the two thermal bands. Reflecting boundaries lose nothing, and with
	// the gas thermally decoupled the dust only shuffles energy between the IR and optical groups, so their
	// combined domain-integrated Erad must equal the energy injected into the optical band. Photoionization
	// removes energy from the ionizing band only, which is why that band is excluded here and gets its own
	// budget in Check 5. This is the primary check on the source-term accounting.
	{
		double E_thermal = 0.0;
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			const double E_g = compute_group_total_erad(sim.state_new_cc_[0], dx, g);
			amrex::Print() << "Group " << g << " integrated Erad: " << E_g << "\n";
			if (g != group_ionizing) {
				E_thermal += E_g;
			}
		}
		// A thermal group's source carries the code's internal chat/c factor (see source_terms_multi_group.hpp).
		// Only the optical group is sourced among the thermal ones; both start at the radiation floor.
		const double injected = (c_hat / C::c_light) * F * E_photon * t_end + 2.0 * Erad_floor_ * Lx;
		const double energy_frac = E_thermal / injected;
		const double tol = 0.01;

		amrex::Print() << "Thermal bands (IR + optical): " << E_thermal << " (injected " << injected << ")\n";
		amrex::Print() << "Opacities: kappa1 = " << kappa1 << ", kappa2 = " << kappa2 << " cm^2/g\n";

		if (std::abs(energy_frac - 1.0) > tol) {
			amrex::Print() << "Test FAILED: thermal-band energy is " << energy_frac << " of injected (expected 1 within " << tol * 100.0 << "%).\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: thermal-band energy is conserved (Erad/injected = " << energy_frac << ").\n";
		}
	}

	// Check 2: the beam must propagate at close to the reduced speed of light. This is a sanity check on
	// transport, NOT a test of the flux source -- Check 3 is that. Deleting the flux source entirely moves
	// the measured front only from 3.6% to 10.1% behind chat * t, because the M1 closure lets the leading
	// edge of an isotropic pulse go free-streaming anyway; it does not crawl at chat / sqrt(3). The
	// tolerance also has to absorb the isotropization caused by the gray opacity, which re-emits absorbed
	// beam energy in all directions and so drags the front a few percent back.
	{
		const double x_front = sim.userData_.xfront_vec_.back();
		const double x_ref = x_analytic(t_end);
		const double percent_diff = std::abs(x_front - x_ref) / x_ref * 100.0;
		const double tol_percent = 10.0;

		amrex::Print() << "Radiation front position: " << x_front << " cm\n";
		amrex::Print() << "Analytic front position:  " << x_ref << " cm (chat * t; an isotropic source would give " << x_ref / std::numbers::sqrt3
			       << ")\n";
		amrex::Print() << "Difference: " << percent_diff << " percent (tolerance: " << tol_percent << " percent)\n";

		if (x_ref >= Lx) {
			amrex::Print() << "Test FAILED: the analytic front has left the domain; reduce stop_time.\n";
			status = 1;
		} else if (percent_diff > tol_percent) {
			amrex::Print() << "Test FAILED: radiation front differs from chat * t by more than " << tol_percent << " percent.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: beamed source propagates at chat within " << tol_percent << " percent.\n";
		}
	}

	// Check 3: the reduced flux of the optical band in the injection cell. This is the check that actually
	// exercises the flux source; neither of the checks above does. SetRadSource sets
	// radFluxSource = c * radEnergySource, so the injected radiation must arrive fully beamed, f = 1. Here
	// that is exact rather than approximate, because the ~130 K dust re-emits into the IR band and puts
	// nothing isotropic back into the optical one to dilute it.
	//
	// The upper bound matters because the M1 flux limiter in ConservedToPrimitive only acts during
	// reconstruction and does not repair the conserved state, so an over-large flux source leaves an
	// unphysical f > 1 sitting in the state array that neither the energy budget nor the front position
	// would reveal. Measured: 1 - f = 4e-15 as written, f = 0.09 with the flux source deleted, and f = 1000
	// with it scaled up by c / chat.
	{
		const double reduced_flux = compute_injection_reduced_flux(sim.state_new_cc_[0], group_optical);
		const double f_min = 0.9;
		const double f_max = 1.0 + 1.0e-6;

		amrex::Print() << "Reduced flux at the injection cell: 1 - f = " << 1.0 - reduced_flux << " (f expected in [" << f_min << ", " << f_max
			       << "])\n";

		if (reduced_flux < f_min) {
			amrex::Print() << "Test FAILED: injected radiation is not beamed (f = " << reduced_flux
				       << "); the flux source is missing or too small.\n";
			status = 1;
		} else if (reduced_flux > f_max) {
			amrex::Print() << "Test FAILED: injected radiation exceeds the free-streaming limit (f = " << reduced_flux
				       << "); the flux source is too large.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: SetRadSource injected free-streaming radiation (f = " << reduced_flux << ").\n";
		}
	}

	// Check 4: dust reprocessing. Nothing is injected into the IR band, so every erg in it arrived there by
	// being absorbed out of the optical beam and thermally re-emitted by the dust. Behind the light front
	// the optical band is in steady state and simply attenuates, E_opt(x) = (F E_photon / c) exp(-kappa_opt
	// rho x), because the ~130 K dust emits nothing back into the optical; integrating that to the front
	// gives a closed form for what should be left unprocessed.
	{
		const double E_ir = compute_group_total_erad(sim.state_new_cc_[0], dx, group_ir);
		const double E_opt = compute_group_total_erad(sim.state_new_cc_[0], dx, group_optical);
		const double rho_0 = sim.userData_.primary_species_2 * spmasses[1]; // initial neutral-H mass density
		const double alpha_opt = rho_0 * kappa2;			    // optical absorption coefficient [cm^-1]
		const double tau_front = alpha_opt * x_analytic(t_end);
		const double E_opt_ref = compute_plateau_erad(F) * (1.0 - std::exp(-tau_front)) / alpha_opt;
		const double reprocessed = E_ir / (E_ir + E_opt);
		const double opt_frac = E_opt / E_opt_ref;

		amrex::Print() << "IR band " << E_ir << ", optical band " << E_opt << " (analytic unprocessed " << E_opt_ref << ")\n";
		amrex::Print() << "Optical depth to the front: " << tau_front << ", reprocessed fraction: " << reprocessed << "\n";

		if (!(alpha_opt > 0.0)) {
			// With kappa2 = 0 the attenuated-beam reference is 0/0. Both comparisons below would then be
			// false against the resulting NaN and the check would report a pass, so reject it here.
			amrex::Print() << "Test FAILED: photoionize.kappa2 must be positive for the attenuated-beam check.\n";
			status = 1;
		} else if (reprocessed < 0.25) {
			amrex::Print() << "Test FAILED: too little of the optical beam was reprocessed into the IR (" << reprocessed << ").\n";
			status = 1;
		} else if (std::abs(opt_frac - 1.0) > 0.10) {
			amrex::Print() << "Test FAILED: surviving optical energy is " << opt_frac
				       << " of the attenuated-beam solution (expected 1 within 10%).\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: dust reprocessed " << reprocessed
				       << " of the beam into the IR, and the surviving optical matches exp(-tau) (ratio " << opt_frac << ").\n";
		}
	}

	// Check 5: photon budget of the ionizing band. It is transparent to the dust opacity, so photoionization
	// is the only thing that removes it: every photon absorbed ionizes one hydrogen atom, and the ones that
	// have not since recombined are still visible as the ionized column. Photon conservation therefore reads
	//
	//   injected = still in the field + ionized column + recombinations,
	//
	// and since the recombinations are not tracked here this is enforced as the two inequalities below: the
	// band must be measurably depleted, and the absorbed photons must at least cover the ionized column.
	//
	// The budget inequality alone is weak -- it has a factor of ten of headroom here, and it does NOT catch
	// a mis-scaled chemistry-band source, because fewer photons simply produce a proportionally smaller
	// ionized column and the inequality still holds (verified: scaling the source by chat/c passes it). The
	// surviving fraction is what catches that. The ionization front stalls at its Stromgren column well
	// inside the light-travel distance chat * t, so a good fraction of the injected photons are always still
	// in flight; mis-scaling the source by c/chat = 1000 collapses the surviving fraction from 0.20 to
	// 8e-5. Hence the lower bound as well as the upper one.
	{
		const double E_ion = compute_group_total_erad(sim.state_new_cc_[0], dx, group_ionizing);
		const double injected_ion = F_ion * E_photon * t_end + Erad_floor_ * Lx;
		const double ion_frac = E_ion / injected_ion;
		const double n_injected = injected_ion / E_photon;
		const double n_absorbed = (injected_ion - E_ion) / E_photon;
		const double column_HII = compute_ionized_column(sim.state_new_cc_[0], dx);

		amrex::Print() << "Ionizing band integrated Erad: " << E_ion << " (injected " << injected_ion << ", surviving fraction " << ion_frac << ")\n";
		amrex::Print() << "Ionizing photons: " << n_injected << " injected, " << n_absorbed << " absorbed; ionized column " << column_HII
			       << " cm^-2 (recombinations account for the rest)\n";

		const double ion_frac_min = 0.02;

		if (ion_frac >= 1.0) {
			amrex::Print() << "Test FAILED: ionizing band was not depleted by photoionization (surviving fraction " << ion_frac << ").\n";
			status = 1;
		} else if (ion_frac < ion_frac_min) {
			amrex::Print() << "Test FAILED: ionizing band is almost entirely gone (surviving fraction " << ion_frac << " < " << ion_frac_min
				       << "); too few photons were injected, or they are being over-absorbed.\n";
			status = 1;
		} else if (n_absorbed < column_HII) {
			amrex::Print() << "Test FAILED: ionized column " << column_HII << " cm^-2 exceeds the " << n_absorbed
				       << " photons absorbed from the ionizing band.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: ionizing photon budget is consistent (" << n_absorbed << " absorbed >= " << column_HII
				       << " ionized).\n";
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
