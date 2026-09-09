/// \file testDTypeFront1D.cpp
/// \brief Defines a 1D planar H II region test: a central ionizing source drives a D-type ionization front
/// into a uniform neutral slab, while dust reprocesses the accompanying optical light into the IR.
///
/// There are three radiation groups: IR (group 0), optical (group 1) and an ionizing chemistry band
/// (group 2). Constant photon fluxes are injected in the two cells straddling the middle of a uniform, cold,
/// dusty hydrogen slab: photoionize.flux_optical into the OPTICAL band and photoionize.flux_ion into the
/// ionizing band. photoionize.flux_ir optionally injects the IR band directly as well, though it defaults to
/// zero: the IR band is normally filled only by the dust's own re-emission of the absorbed optical light. Both the radiation energy source and the companion
/// radiation flux source are set, the latter as a reduced flux of -1 on the left of the source and +1 on the right, so each half of the slab is injected fully
/// beamed away from the centre. The two wings are mirror images, so the source injects zero net momentum while each wing carries the outward momentum its
/// luminosity implies; photoionize.flux is the photon flux delivered to EACH side. The IR band receives no source at all.
///
/// Putting the source in the middle rather than against a boundary keeps both fronts away from the walls.
/// Both domain boundaries are reflecting, and the run is stopped well before either front arrives, so
/// nothing is reflected and nothing escapes.
///
/// The two sourced bands are scaled differently inside the solver -- a thermal group's source is multiplied
/// by chat/c and a chemistry band's is not -- so the shipped fluxes differ by exactly c/chat = 1000 and
/// deliver equal energy. The ionizing band is transparent to the dust opacity, so photoionization is the
/// only process that removes it, and it ionizes the slab as it advances.
///
/// A separate dust temperature is solved for, with the gas-dust collisional coupling switched off
/// (radiation.dust_gas_interaction_coeff = 0), so the dust sits at radiative equilibrium and exchanges no
/// energy with the gas. Radiation momentum is still deposited, so the radiation does accelerate the gas and
/// contributes to driving the front.
///
/// The dust opacity is gray within each band and set at runtime (photoionize.kappa_ir for the IR,
/// photoionize.kappa_optical for the optical), with the optical opacity much the larger, as for real dust. The
/// chain the test exercises is therefore:
///
///   optical source -> absorbed by dust -> dust heats to radiative equilibrium -> re-emitted as IR
///
/// Opacity in Quokka is pure absorption, so an opaque group also emits its share of the local blackbody,
/// which is what supplies the re-emission.
///
/// The test makes three checks:
///
///   1. The gas temperature in the ionized cavity and in the undisturbed neutral gas ahead of the front each
///      match the equilibrium temperature of the corresponding heating/cooling balance of the chemical
///      network. The neutral comparison is made per cell against the equilibrium temperature of that cell's
///      own density, since -- unlike the ionized balance -- the neutral one is density-dependent.
///   2. The measured D-type front position matches a numerically integrated thin-shell solution that is
///      driven by BOTH the ionized-gas pressure and the radiation pressure of the two sourced bands. The
///      closed-form Spitzer 4/5 law, which carries the gas term only, is recorded alongside it for
///      reference, so the gap between the two curves shows what the radiation pressure is worth.
///   3. Optical light lost from the beam is conserved into the IR band: the two together are budgeted
///      against what the source injected minus what the attenuated-beam solution says should still be in the
///      optical band. Dust self-emission back into the optical band is negligible at the dust's radiative
///      equilibrium temperature and is neglected on the reference side of the budget (see Check 3 below).

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
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
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

// Mean photon energy of each injected band, taken as the arithmetic mean of the band's frequency edges times
// hplanck. The IR/optical edges come from radBoundaries below (1e8, 1e14, 3.29e15 Hz); the ionizing band uses
// its true edges [3.29e15, 1.5e16] Hz (see CMakeLists.txt CHEM_BANDS), NOT radBoundaries[3], which is a
// nominal "read as infinity" placeholder (see the radBoundaries comment) rather than the chemistry band's
// actual upper edge.
constexpr double eps_ir = 0.5 * (1.0e8 + 1.0e14) * C::hplanck;	  // erg
constexpr double eps_opt = 0.5 * (1.0e14 + 3.29e15) * C::hplanck; // erg
constexpr double eps_ion = 0.5 * (3.29e15 + 8.0e15) * C::hplanck; // erg
// Radiation energy-density floor. This is a physically meaningful, negligible photon-number density
// (1e-10 cm^-3, vs the ~hundreds cm^-3 of the injected beam) converted to a radiation energy density. Dark
// cells are initialized to exactly this floor (see setInitialConditionsOnGrid), following the best practice
// of RadStreaming / RadhydroShockMultigroup instead of seeding an unphysical 1e-99.
constexpr double Erad_floor_ = 1.0e-10 * eps_ion; // erg cm^-3

// Group indices. Group 0 is the IR band, group 1 the optical band, group 2 the ionizing chemistry band;
// chemistry bands must come last (see radiation_system.hpp).
constexpr int group_ir = 0;
constexpr int group_optical = 1;
constexpr int group_ionizing = 2;

// Fraction of the unattenuated beam energy density used to locate the optical light front. It has to sit
// below exp(-tau) at the front so the threshold finds the light front and not the dust absorption depth, and
// far enough above the radiation floor to be unambiguous. Only used for the diagnostic trace written to the
// CSV; no check depends on it.
constexpr double front_threshold_fraction = 0.05;

// Gray dust opacities of the two thermal groups [cm^2 g^-1], set at runtime from photoionize.kappa_ir (IR)
// and photoionize.kappa_optical (optical). Both default to zero, i.e. a transparent domain. The ionizing band
// is always transparent to this gray opacity; it couples to the gas through photochemistry instead. Managed
// memory so the device-side opacity function can read them.
AMREX_GPU_MANAGED double kappa_ir = 0.0;      // NOLINT
AMREX_GPU_MANAGED double kappa_optical = 0.0; // NOLINT

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
	// beta_order = 1: keep the O(v/c) terms in the radiation-matter coupling, including the work term.
	static constexpr int beta_order = 1;
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
	// The IR/optical split at 1e14 Hz is what makes the reprocessing clean: the dust is cold enough that the
	// Planck function has nothing left above the split, so essentially all re-emission lands in the IR group
	// and none of it back into the optical one.
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
	// deposited, so the beam drives the gas. That is what the radiation-pressure term of the front ODE below
	// accounts for.
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
	// This is the one place the network's THERMAL_DUST_PHOTOCHEMISTRY macro is translated into a Quokka
	// trait. The macro has to reach the reaction network, which is compiled through Microphysics and so
	// cannot see problem_t; everything on the Quokka side reads the trait instead. The two must agree,
	// hence the binding here rather than a hard-coded true.
	static constexpr bool dust_chemical_band_absorption =
#ifdef THERMAL_DUST_PHOTOCHEMISTRY
	    true;
#else
	    false;
#endif
};

template <> struct SimulationData<DTypeFront1D> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real flux_optical{}; // optical photon flux [photons cm^-2 s^-1] injected per side
	amrex::Real flux_ion{};	    // ionizing photon flux [photons cm^-2 s^-1] injected per side
	amrex::Real T_ionized{};    // ionized-gas temperature of the analytic D-type solution [K]; computed, not read
	amrex::Vector<amrex::Real> t_vec_;
	amrex::Vector<amrex::Real> xfront_vec_;	  // optical light front, as a distance from the source [cm]
	amrex::Vector<amrex::Real> xshell_vec_;	  // measured dense-shell position, as a distance from the source [cm]
	amrex::Vector<amrex::Real> xspitzer_vec_; // closed-form planar D-type (gas pressure only) at the same times [cm]
	amrex::Vector<amrex::Real> xeff_vec_;	  // ionization-fraction-weighted effective ionized length, +x side [cm]
	amrex::Vector<amrex::Real> xode_vec_;	  // numerically integrated D-type front position, incl. radiation pressure [cm]
	// Running state of the front ODE, advanced one simulation timestep at a time in computeAfterTimestep.
	// u = l * ldot is the integration variable paired with l; see integrate_front.
	amrex::Real l_ode_last_t_{}; // time the stored ODE state corresponds to [s]
	amrex::Real l_ode_last_l_{}; // front position at that time [cm]
	amrex::Real l_ode_last_u_{}; // u = l * ldot at that time [cm^2 s^-1]
	// Total-energy conservation check: the baseline gas + binding energy at t = 0, and the running total of
	// radiation energy injected by AddRadSource since then. Both are domain integrals [erg cm^-2 in 1D].
	amrex::Real energy_initial_{};		 // gas internal + H binding energy at t = 0 (radiation starts at the floor)
	amrex::Real energy_injected_{};		 // running integral of the injected radiation luminosity, all bands combined
	amrex::Real energy_injected_ir_{};	 // running integral of the injected IR-band luminosity
	amrex::Real energy_injected_optical_{};	 // running integral of the injected optical-band luminosity
	amrex::Real energy_injected_ionizing_{}; // running integral of the injected ionizing-band luminosity
	amrex::Real energy_last_t_{};		 // time through which energy_injected_ has been accumulated [s]
	amrex::Real flux_ir{};			 // IR photon flux [photons cm^-2 s^-1] injected per side; mirrors AddRadSource's photoionize.flux_ir
	std::ofstream output_file_;
	std::ofstream energy_output_file_; // per-step energy budget, written only when c_hat == c (see computeAfterTimestep)
};

namespace
{

// Plateau radiation energy density of a free-streaming beam carrying photon flux F in the optical band:
// E = F * eps_opt / c. Note this is independent of the reduced speed of light.
auto compute_plateau_erad(amrex::Real flux, amrex::Real eps) -> amrex::Real { return flux * eps / C::c_light; }

// Position of the outward-going radiation front of group g: the right edge of the outermost cell whose
// radiation energy density exceeds Erad_threshold. The source is symmetric about the middle of the domain,
// so this is the +x front and the caller measures it relative to the source.
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

// Ionization-fraction-weighted effective ionized length on the +x side of the source, as a distance from the
// source: x_eff = integral_{x_source}^{L} (1 - x_HI) dx = sum_cells (1 - x_HI) * dx. Restricted to the +x
// half so it is comparable to compute_shell_position and the reference curves, which are also measured from
// the source outward. Unlike the shell position, this does not require a density peak to have formed, so it
// is well defined from t = 0 and gives a smoother trace than the single-cell shell finder. This is the
// quantity the front-radius check below compares against the integrated solution.
auto compute_effective_length(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real x_source) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
		if (x <= x_source) {
			return 0.0_rt;
		}
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

// Position of the dense shocked shell on the +x side of the source, returned as a distance from the source.
// In a D-type front the neutral gas swept up ahead of the ionization front piles into a dense shell, so the
// cell holding the largest gas density tracks the shock; the thin-shell approximation underlying the
// reference solutions takes that shock to sit essentially on top of the ionization front. Recorded as a
// diagnostic alongside the effective length.
//
// This takes two passes because AMReX's reductions return an extremum, not the location of one. The first
// pass takes the maximum density over the +x half of the domain, the second the SMALLEST x among the cells
// attaining it. The tie-break only matters before the shell has formed, when the slab is still uniform and
// every cell attains rho_max: reporting the innermost of them puts the shell on the source, which is the
// sensible reading, rather than parking it at the far edge of the domain.
auto compute_shell_position(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real x_source) -> amrex::Real
{
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];

	// Pass 1: the largest gas density outward of the source.
	amrex::Real rho_max = 0.0;
	{
		amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
		amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
		auto const state = state_mf.const_arrays();

		reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
			const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
			if (x <= x_source) {
				return 0.0_rt;
			}
			return state[box_no](i, j, k, HydroSystem<DTypeFront1D>::density_index);
		});

		auto const &hv = reduce_data.value(reduce_op);
		rho_max = amrex::get<0>(hv);
		amrex::ParallelAllReduce::Max(rho_max, amrex::ParallelContext::CommunicatorSub());
	}

	// Pass 2: the innermost cell attaining that density.
	amrex::Real x_shell = 0.0;
	{
		amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
		amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
		auto const state = state_mf.const_arrays();
		const amrex::Real threshold = rho_max;
		const amrex::Real x_none = std::numeric_limits<amrex::Real>::max();

		reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
			const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
			if (x <= x_source || state[box_no](i, j, k, HydroSystem<DTypeFront1D>::density_index) < threshold) {
				return x_none;
			}
			return x;
		});

		auto const &hv = reduce_data.value(reduce_op);
		x_shell = amrex::get<0>(hv);
		amrex::ParallelAllReduce::Min(x_shell, amrex::ParallelContext::CommunicatorSub());
	}

	return x_shell - x_source;
}

// Domain-integrated radiation energy of group g: sum_cells Erad_g * dx  [erg cm^-2 in 1D]. Used by the
// dust-reprocessing check to budget the optical and IR bands against each other.
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

// Domain-integrated radiation energy summed over every group [erg cm^-2 in 1D].
auto compute_total_erad(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::Real total = 0.0_rt;
	for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
		total += compute_group_total_erad(state_mf, dx, g);
	}
	return total;
}

// Domain-integrated gas internal energy plus the chemical (binding) energy stored in ionized hydrogen
// [erg cm^-2 in 1D]. Ionizing a hydrogen atom banks 13.6 eV in the H+/e- pair, so n_HII * 13.6 eV is the
// energy the gas is holding chemically rather than thermally; the network debits exactly this amount from
// the photon that did the ionizing (see get_ionization_heating_coefficient in actual_rhs.H, which credits
// the gas only with the photoelectron's excess kinetic energy) and returns it on recombination. Counting it
// here is what closes the budget: without it, ionization looks like an energy sink.
//
// Dust carries no heat capacity in this problem, so it stores nothing and needs no term here -- whatever it
// absorbs is re-emitted into the thermal bands within the same step.
auto compute_gas_plus_binding_energy(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real binding_energy = 13.6 * C::ev2erg;

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real Eint = state[box_no](i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index);
		const amrex::Real n_HII = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 2) / spmasses[2];
		return cell_length * (Eint + n_HII * binding_energy);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total, amrex::ParallelContext::CommunicatorSub());
	return total;
}

// Domain-integrated gas internal energy alone (no binding energy) [erg cm^-2 in 1D].
auto compute_gas_internal_energy(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		return cell_length * state[box_no](i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total, amrex::ParallelContext::CommunicatorSub());
	return total;
}

// Domain-integrated binding energy held up in unbinding (ionizing) hydrogen: n_HII * 13.6 eV [erg cm^-2 in
// 1D]. See compute_gas_plus_binding_energy for why this term belongs in the energy budget.
auto compute_binding_energy(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real binding_energy = 13.6 * C::ev2erg;

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real n_HII = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 2) / spmasses[2];
		return cell_length * n_HII * binding_energy;
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total, amrex::ParallelContext::CommunicatorSub());
	return total;
}

// Domain-integrated gas kinetic energy, 0.5 * rho * v^2 [erg cm^-2 in 1D]. 1D planar, so only the x-momentum
// component carries kinetic energy.
auto compute_kinetic_energy(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real rho = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::density_index);
		const amrex::Real px = state[box_no](i, j, k, HydroSystem<DTypeFront1D>::x1Momentum_index);
		return cell_length * 0.5_rt * px * px / rho;
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real total = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(total, amrex::ParallelContext::CommunicatorSub());
	return total;
}

// Photoionization-equilibrium temperatures of the ionized and neutral gas, obtained from the same
// heating/cooling balances the photoionization network itself integrates. Taken from the 3D DTypeFront
// problem so the two D-type tests normalize their reference temperatures identically.
//
// Recombination cooling, per recombination [erg cm^3 s^-1].
auto lambda_rec(double T) -> double
{
	if (T < 100.0) {
		return 0.0;
	}
	return 6.1e-10 * 1.380649e-16 * T * std::pow(T, -0.89);
}

// Ion free-free (+ CLE) cooling, per (electron, ion) pair [erg cm^3 s^-1]. Matches
// get_ion_ff_cooling_coefficient in actual_rhs.H (Frazer & Heitsch 2019).
auto get_cle_term(double T) -> double
{
	if (T < 1.0e2) {
		return 3.47e-29 * std::pow(T, 1.915);
	}
	if (T < std::pow(10.0, 2.8)) {
		return 2.34e-26 * std::pow(T, 0.500);
	}
	if (T < std::pow(10.0, 3.6)) {
		return 1.11e-24 * std::pow(T, -0.099);
	}
	if (T < 1.0e4) {
		return 1.08e-32 * std::pow(T, 2.127);
	}
	if (T < std::pow(10.0, 4.5)) {
		return 2.67e-30 * std::pow(T, 1.529);
	}
	if (T < 1.0e5) {
		return 1.74e-24 * std::pow(T, 0.237);
	}
	if (T < 1.0e6) {
		return 1.10e-21 * std::pow(T, -0.323);
	}
	return 7.49e-21 * std::pow(T, -0.462);
}

auto lambda_ff(double T) -> double { return 1.3 * 1.427e-27 * std::sqrt(T) + get_cle_term(T); }

// Koyama & Inutsuka cooling function of the neutral gas [erg cm^3 s^-1].
auto lambda_KI(double T) -> double { return 2.0e-26 * (1.0e7 * std::exp(-118400.0 / (T + 1.0e3)) + 1.4e-2 * std::sqrt(T) * std::exp(-92.0 / T)); }

// Net volumetric heating minus cooling of fully ionized gas at temperature T and electron density n_e
// [erg cm^-3 s^-1]. In ionization equilibrium every recombination is balanced by a photoionization, so the
// photoionization rate per unit volume is alpha_B * n_e^2 and each one deposits the mean excess energy
// epsilon = eps_ion - 13.6 eV, the photoelectron's kinetic energy after paying the ionization potential (see
// get_ionization_heating_coefficient in actual_rhs.H, which this mirrors). Collisional ionization is not
// included in this balance: k_coll/alpha_B ~ 5e-5 at the cavity's equilibrium temperature (~8000 K), and the
// cavity is highly ionized (n_HI tiny), so its contribution to both the ionization and energy balance is
// negligible here -- adding it would also break the density-independence this function relies on (every
// remaining term scales as n_e^2; collisional ionization scales as n_e*n_HI instead).
auto net_energy_ionized(double T, double n_e) -> double
{
	const double alpha_B = 2.6e-13 * std::pow(T / 1.0e4, -0.7);
	const double epsilon = eps_ion - 13.6 * C::ev2erg;
	// alpha_B * n_e^2 = n_gamma
	const double photoheating = alpha_B * n_e * n_e * epsilon;
	const double recombination_cooling = n_e * n_e * lambda_rec(T);
	const double ff_cooling = n_e * n_e * lambda_ff(T);
	// Assume KI heating and cooling are negligible in the cavity since the neutral fraction is low.
	const double KI_heating = 0.0;
	const double KI_cooling = 0.0;
	return photoheating - recombination_cooling - ff_cooling + KI_heating - KI_cooling;
}

// Net volumetric heating minus cooling of neutral gas at temperature T and neutral density n_HI
// [erg cm^-3 s^-1]. No ionizing photons reach it, so the balance is purely the KI photoelectric heating
// against the KI cooling curve.
auto net_energy_neutral(double T, double n_HI) -> double
{
	const double photoheating = 0.0;
	const double KI_heating = n_HI * 2e-26;
	const double KI_cooling = n_HI * n_HI * lambda_KI(T);
	const double recombination_cooling = 0.0;
	const double ion_ff_cooling = 0.0;
	return photoheating + KI_heating - recombination_cooling - KI_cooling - ion_ff_cooling;
}

// Temperature at which net_energy_neutral vanishes, by bisection.
auto compute_equilibrium_temperature_neutral(double n_HI) -> double
{
	double T_lo = 1;
	double T_hi = 1000;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(net_energy_neutral(T_lo, n_HI) > 0.0 && net_energy_neutral(T_hi, n_HI) < 0.0,
					 "compute_equilibrium_temperature_neutral: brackets do not straddle a root");
	int const max_iter = 10000;
	for (int iter = 0; iter < max_iter; ++iter) {
		const double T_mid = 0.5 * (T_lo + T_hi);
		if (net_energy_neutral(T_mid, n_HI) > 0.0) {
			T_lo = T_mid;
		} else {
			T_hi = T_mid;
		}
		if ((T_hi - T_lo) < 1e-2) {
			break;
		}
	}
	return 0.5 * (T_lo + T_hi);
}

// Temperature at which net_energy_ionized vanishes, by bisection. The balance is density-independent (every
// term scales as n_e^2), so the result depends on n_e only through the assertion bracket; it is nonetheless
// passed through to keep the call site explicit about which gas is being equilibrated.
auto compute_equilibrium_temperature_ionized(double n_e) -> double
{
	double T_lo = 1000.0;
	double T_hi = 1.0e5;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(net_energy_ionized(T_lo, n_e) > 0.0 && net_energy_ionized(T_hi, n_e) < 0.0,
					 "compute_equilibrium_temperature_ionized: brackets do not straddle a root");
	int const max_iter = 10000;
	for (int iter = 0; iter < max_iter; ++iter) {
		const double T_mid = 0.5 * (T_lo + T_hi);
		if (net_energy_ionized(T_mid, n_e) > 0.0) {
			T_lo = T_mid;
		} else {
			T_hi = T_mid;
		}
		if ((T_hi - T_lo) < 1.0) {
			break;
		}
	}
	return 0.5 * (T_lo + T_hi);
}

auto recombination_coefficient(amrex::Real T_i) -> amrex::Real { return 2.6e-13 * std::pow(T_i / 1.0e4, -0.7); }

auto ionized_sound_speed(amrex::Real T_i) -> amrex::Real { return std::sqrt(C::k_B * T_i / (0.5_rt * C::m_p)); }

auto stromgren_column(amrex::Real flux_ion, amrex::Real n_0, amrex::Real T_i) -> amrex::Real { return flux_ion / (recombination_coefficient(T_i) * n_0 * n_0); }

// Planar (1D) analog of the Spitzer D-type expansion law, evaluated at time t. Gas pressure only.
auto spitzer_planar_position(amrex::Real t, amrex::Real flux_ion, amrex::Real n_0, amrex::Real T_i) -> amrex::Real
{
	const amrex::Real c_i = ionized_sound_speed(T_i);
	const amrex::Real x_St = stromgren_column(flux_ion, n_0, T_i);
	return x_St * std::pow(1.0_rt + (5.0_rt / 4.0_rt) * c_i * t / x_St, 4.0_rt / 5.0_rt);
}

// Numerically integrate the planar D-type front ODE including radiation pressure,
// d(l * ldot)/dt = sqrt(l_s / l) * c_s^2  +  (F_ion * eps_ion + F_opt * eps_opt) / (rho_0 * c),
// Integrated as the first-order system for y = (l, u) with u = l * ldot:
// dl/dt = u / l,     du/dt = sqrt(l_s / l) * c_s^2 + Xi.
// The natural start is the end of the R-type phase, l = l_s moving at c_s
auto integrate_front(amrex::Real dt_target, amrex::Real l0, amrex::Real u0, amrex::Real l_s, amrex::Real c_s, amrex::Real Xi) -> amrex::GpuArray<amrex::Real, 2>
{
	if (dt_target <= 0.0_rt) {
		return {l0, u0};
	}

	const amrex::Real l_floor = 1.0e-10_rt * l_s; // guard against a division by zero in an RK stage
	auto rhs = [&](amrex::GpuArray<amrex::Real, 2> const &y) -> amrex::GpuArray<amrex::Real, 2> {
		const amrex::Real l = std::max(y[0], l_floor);
		return {y[1] / l, std::sqrt(l_s / l) * c_s * c_s + Xi};
	};

	int N = 256;
	const int max_iters = 10;
	const amrex::Real tol = 1.0e-6_rt * std::max(l_s, 1.0_rt);
	amrex::GpuArray<amrex::Real, 2> y_prev{l0, u0};

	for (int iter = 0; iter < max_iters; ++iter) {
		const amrex::Real dt = dt_target / static_cast<amrex::Real>(N);
		amrex::GpuArray<amrex::Real, 2> y{l0, u0};

		for (int step = 0; step < N; ++step) {
			const auto k1 = rhs(y);
			const auto k2 = rhs({y[0] + 0.5_rt * dt * k1[0], y[1] + 0.5_rt * dt * k1[1]});
			const auto k3 = rhs({y[0] + 0.5_rt * dt * k2[0], y[1] + 0.5_rt * dt * k2[1]});
			const auto k4 = rhs({y[0] + dt * k3[0], y[1] + dt * k3[1]});
			y[0] += (dt / 6.0_rt) * (k1[0] + 2.0_rt * k2[0] + 2.0_rt * k3[0] + k4[0]);
			y[1] += (dt / 6.0_rt) * (k1[1] + 2.0_rt * k2[1] + 2.0_rt * k3[1] + k4[1]);
			y[0] = std::max(y[0], 0.0_rt);
		}

		if (iter > 0 && std::abs(y[0] - y_prev[0]) < tol) {
			return y;
		}
		y_prev = y;
		N *= 2;
	}

	amrex::Abort("integrate_front failed to converge within max_iters for dt=" + std::to_string(dt_target));
	return y_prev; // unreachable
}

constexpr const char *therm_suffix = ISM_Traits<DTypeFront1D>::dust_chemical_band_absorption ? "_THERM_ON" : "_THERM_OFF";

} // namespace

template <>
void RadSystem<DTypeFront1D>::AddRadSource(array_t &radEnergy, array_t &reducedFlux, const amrex::Box &indexRange,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	amrex::ParmParse const pp("photoionize");
	amrex::Real flux_optical = 1.0e11_rt;
	pp.query("flux_optical", flux_optical);
	amrex::Real flux_ion = 0.0_rt;
	pp.query("flux_ion", flux_ion);
	amrex::Real flux_ir = 0.0_rt;
	pp.query("flux_ir", flux_ir);
	int source_cells = 1;
	pp.query("source_cells", source_cells); // cells per side occupied by the source slab
	int beamed = 1;
	pp.query("beamed", beamed); // 1 = each wing injected beamed outward, 0 = isotropic

	const auto n_cells = static_cast<amrex::Real>(source_cells);
	const amrex::Real src_ir = flux_ir * eps_ir / (n_cells * dx[0]);
	const amrex::Real src_optical = flux_optical * eps_opt / (n_cells * dx[0]);
	const amrex::Real src_ionizing = flux_ion * eps_ion / (n_cells * dx[0]);

	// A cell belongs to the source slab when its centre lies within source_cells cell widths of the middle of
	// the domain, which selects exactly source_cells cells per side and keeps the source symmetric at any
	// resolution with an even cell count.
	const amrex::Real x_source = 0.5_rt * (prob_lo[0] + prob_hi[0]);
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];
	const amrex::Real half_width = n_cells * cell_length;
	// A reduced flux of unit magnitude is fully beamed; zero leaves the injection isotropic.
	const amrex::Real beam_factor = (beamed != 0) ? 1.0_rt : 0.0_rt;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
		const bool in_source = std::abs(x - x_source) < half_width;
		// Outward is -x on the left of the source and +x on the right.
		const amrex::Real outward = (x > x_source) ? 1.0_rt : -1.0_rt;
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			amrex::Real src = 0.0_rt;
			if (in_source) {
				if (g == group_ir) {
					src = src_ir;
				} else if (g == group_optical) {
					src = src_optical;
				} else if (g == group_ionizing) {
					src = src_ionizing;
				}
			}
			radEnergy(i, j, k, g) = src;
			reducedFlux(i, j, k, 3 * g + 0) = (src > 0.0_rt) ? outward * beam_factor : 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 1) = 0.0_rt;
			reducedFlux(i, j, k, 3 * g + 2) = 0.0_rt;
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
	userData_.flux_optical = 1.0e11_rt;
	userData_.flux_ion = 0.0_rt;
	userData_.flux_ir = 0.0_rt;
	pp.query("kappa_ir", kappa_ir);
	pp.query("kappa_optical", kappa_optical);
	pp.query("small_temp", userData_.small_temp);
	pp.query("small_dens", userData_.small_dens);
	pp.query("temperature", userData_.temperature);
	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("flux_optical", userData_.flux_optical);
	pp.query("flux_ion", userData_.flux_ion);
	pp.query("flux_ir", userData_.flux_ir);

	userData_.T_ionized = compute_equilibrium_temperature_ionized(userData_.primary_species_2);
	amrex::Print() << "Photoionization-equilibrium temperature of the ionized gas: " << userData_.T_ionized << " K\n";

	{
		const amrex::Real l_s = stromgren_column(userData_.flux_ion, userData_.primary_species_2, userData_.T_ionized);
		const amrex::Real c_s = ionized_sound_speed(userData_.T_ionized);
		userData_.l_ode_last_t_ = 0.0_rt;
		userData_.l_ode_last_l_ = l_s;
		userData_.l_ode_last_u_ = l_s * c_s;
		amrex::Print() << "Stromgren column l_s = " << l_s << " cm, ionized sound speed c_s = " << c_s << " cm/s\n";
	}

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_.open(std::string("dtype_front_1d_front") + therm_suffix + ".csv");
		userData_.output_file_ << "time,x_front,x_shell,x_spitzer,x_eff,x_ode,E_opt_tot,E_ir_tot\n";
		if (RadSystem_Traits<DTypeFront1D>::c_hat_over_c == 1.0) {
			userData_.energy_output_file_.open("energy.csv");
			userData_.energy_output_file_ << "time,E_internal,E_kinetic,E_rad_ir,E_rad_optical,E_rad_ionizing,E_injected_ir,E_injected_optical,"
							 "E_injected_ionizing,E_binding,E_lag,E_total,abs_error,rel_error\n";
		}
	}
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DTypeFront1D>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	// Each thermal group carries its own constant gray opacity; the ionizing (chemistry) band is left
	// transparent. The trailing entry (i == nGroups_) is the unused upper band edge.
	const amrex::GpuArray<double, nGroups_> kappa_g{kappa_ir, kappa_optical, 0.0};
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
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	// Track the optical band. The threshold is well below the unattenuated plateau because dust absorption
	// thins the light as it advances: at the front the optical energy density is down by exp(-tau), so a 0.5
	// threshold would report the absorption depth rather than the front. See front_threshold_fraction.
	const amrex::Real Erad_threshold = front_threshold_fraction * compute_plateau_erad(userData_.flux_optical, eps_opt);
	const amrex::Real x_source = 0.5 * (prob_lo[0] + prob_hi[0]);
	// Distance travelled from the source, not an absolute position: the source sits at the middle of the
	// domain and the +x front is the one compute_front_position reports.
	const amrex::Real x_front = compute_front_position(state_new_cc_[lev], dx, prob_lo, Erad_threshold, group_optical) - x_source;
	const amrex::Real t = tNew_[lev];
	userData_.xfront_vec_.push_back(x_front);
	userData_.t_vec_.push_back(t);

	const amrex::Real x_shell = compute_shell_position(state_new_cc_[lev], dx, prob_lo, x_source);
	const amrex::Real x_spitzer = spitzer_planar_position(t, userData_.flux_ion, userData_.primary_species_2, userData_.T_ionized);
	const amrex::Real x_eff = compute_effective_length(state_new_cc_[lev], dx, prob_lo, x_source);

	amrex::Real x_ode = std::numeric_limits<amrex::Real>::quiet_NaN();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const amrex::Real n_0 = userData_.primary_species_2;
		const amrex::Real rho_0 = n_0 * spmasses[1];
		const amrex::Real l_s = stromgren_column(userData_.flux_ion, n_0, userData_.T_ionized);
		const amrex::Real c_s = ionized_sound_speed(userData_.T_ionized);
		const amrex::Real T_i = userData_.T_ionized;
		// Energy released per recombination: the 13.6 eV binding energy plus the mean kinetic energy carried
		// off (lambda_rec/alpha_B). The network emits this into the OPTICAL band, not the ionizing one --
		// ydot(net_ienuc + NumChemRadEqs + 2) = recombination_cooling_rate + recombination_binding_energy_rate
		// in actual_rhs.H -- so it is grouped with flux_optical here.
		//
		// Xi lumps every absorbed band together and applies no per-band attenuation, so this regrouping does
		// not change Xi's value. It is written this way to match where the energy actually goes, so that the
		// 1D and 3D problems agree; the 3D ODE does attenuate the optical band, and there the placement
		// changes the answer.
		//
		// flux_ion stands in for the recombination rate, which is approximate in two ways, both of which
		// overestimate the re-emission and both of which act on a term worth ~2% of the total luminosity:
		// ionizations balance recombinations only in global equilibrium, not while the front is still eating
		// into neutral gas; and dust competes with hydrogen for ionizing photons (ydot(5) subtracts both
		// photoionization_term and dust_absorption_term), so with network.dust_kappa = 1000 a real fraction of
		// the ionizing photons are absorbed by dust and never produce a recombination at all. Left
		// uncorrected on purpose -- see the fuller note in compute_driving_luminosities in the 3D problem.
		const amrex::Real eps_rec = 13.6 * C::ev2erg + lambda_rec(T_i) / recombination_coefficient(T_i);
		const amrex::Real Xi = (userData_.flux_ion * eps_ion + userData_.flux_optical * eps_opt + userData_.flux_ion * eps_rec) / (rho_0 * C::c_light);

		amrex::Real dt_ode = t - userData_.l_ode_last_t_;
		if (dt_ode < 0.0_rt) {
			// time went backwards or was reset; restart the integration from the R-type endpoint
			userData_.l_ode_last_t_ = 0.0_rt;
			userData_.l_ode_last_l_ = l_s;
			userData_.l_ode_last_u_ = l_s * c_s;
			dt_ode = t;
		}
		const auto y = integrate_front(dt_ode, userData_.l_ode_last_l_, userData_.l_ode_last_u_, l_s, c_s, Xi);
		userData_.l_ode_last_t_ = t;
		userData_.l_ode_last_l_ = y[0];
		userData_.l_ode_last_u_ = y[1];
		x_ode = y[0];
	}
	amrex::ParallelDescriptor::Bcast(&x_ode, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	userData_.xshell_vec_.push_back(x_shell);
	userData_.xspitzer_vec_.push_back(x_spitzer);
	userData_.xeff_vec_.push_back(x_eff);
	userData_.xode_vec_.push_back(x_ode);

	const amrex::Real E_opt_tot = compute_group_total_erad(state_new_cc_[lev], dx, group_optical);
	const amrex::Real E_ir_tot = compute_group_total_erad(state_new_cc_[lev], dx, group_ir);

	// Accumulate the radiation energy AddRadSource has injected. Each band injects flux * eps per unit area
	// per side, and the source is symmetric about the middle of the domain, so both wings count. The thermal
	// (IR, optical) bands are scaled by c_hat / c on the way in -- see the src_scale line in
	// AddSourceTermsMultiGroup, which applies that factor to every group below nGroupsThermal_ and leaves the
	// ionizing chem band unscaled -- so the accounting has to apply the same factor to match what the state
	// actually received.
	const amrex::Real dt_step = t - userData_.energy_last_t_;
	if (dt_step > 0.0_rt) {
		const amrex::Real cscale = RadSystem_Traits<DTypeFront1D>::c_hat_over_c;
		// Thermal bands (IR, optical) are scaled by c_hat / c on the way in; the ionizing chem band is not.
		// See the src_scale line in AddSourceTermsMultiGroup.
		const amrex::Real power_per_area_ir = 2.0_rt * cscale * userData_.flux_ir * eps_ir;
		const amrex::Real power_per_area_optical = 2.0_rt * cscale * userData_.flux_optical * eps_opt;
		const amrex::Real power_per_area_ionizing = 2.0_rt * userData_.flux_ion * eps_ion;
		userData_.energy_injected_ir_ += power_per_area_ir * dt_step;
		userData_.energy_injected_optical_ += power_per_area_optical * dt_step;
		userData_.energy_injected_ionizing_ += power_per_area_ionizing * dt_step;
		userData_.energy_injected_ += (power_per_area_ir + power_per_area_optical + power_per_area_ionizing) * dt_step;
	}
	userData_.energy_last_t_ = t;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.output_file_ << t << ',' << x_front << ',' << x_shell << ',' << x_spitzer << ',' << x_eff << ',' << x_ode << ',' << E_opt_tot << ','
				       << E_ir_tot << '\n';
	}

	// Per-step energy budget, written only when c_hat == c: at reduced speed of light the radiation and gas
	// clocks run at different rates, so a snapshot total is not meaningful as an instantaneous energy budget.
	if (RadSystem_Traits<DTypeFront1D>::c_hat_over_c == 1.0) {
		const amrex::Real E_internal = compute_gas_internal_energy(state_new_cc_[lev], dx);
		const amrex::Real E_kinetic = compute_kinetic_energy(state_new_cc_[lev], dx);
		const amrex::Real E_binding = compute_binding_energy(state_new_cc_[lev], dx);
		const amrex::Real E_rad_ir = compute_group_total_erad(state_new_cc_[lev], dx, group_ir);
		const amrex::Real E_rad_optical = compute_group_total_erad(state_new_cc_[lev], dx, group_optical);
		const amrex::Real E_rad_ionizing = compute_group_total_erad(state_new_cc_[lev], dx, group_ionizing);
		// Dust heating used to be deposited at the end of a radiation subcycle and consumed at the start of
		// the next one, leaving one substep's worth in flight that the budget had to add back. Photochemistry
		// now runs at the top of a subcycle and the source terms consume the deposit within that same
		// subcycle, so nothing is ever in flight. Kept as a named zero so the CSV column stays put.
		const amrex::Real E_lag = 0.0_rt;
		const amrex::Real E_total = E_internal + E_kinetic + E_binding + E_rad_ir + E_rad_optical + E_rad_ionizing + E_lag;
		const amrex::Real E_expected = userData_.energy_initial_ + userData_.energy_injected_;
		const amrex::Real abs_error = E_total - E_expected;
		const amrex::Real rel_error = (E_expected != 0.0_rt) ? std::abs(abs_error) / std::abs(E_expected) : std::abs(abs_error);

		if (amrex::ParallelDescriptor::IOProcessor()) {
			userData_.energy_output_file_ << std::setprecision(17) << t << ',' << E_internal << ',' << E_kinetic << ',' << E_rad_ir << ','
						      << E_rad_optical << ',' << E_rad_ionizing << ',' << userData_.energy_injected_ir_ << ','
						      << userData_.energy_injected_optical_ << ',' << userData_.energy_injected_ionizing_ << ',' << E_binding
						      << ',' << E_lag << ',' << E_total << ',' << abs_error << ',' << rel_error << '\n';
		}
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

	// Baseline for the total-energy conservation check below. Taken after setInitialConditions so the grid
	// holds the real initial state. The radiation bands start at Erad_floor_ (a negligible but nonzero seed),
	// so they are included here rather than assumed to be zero.
	{
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx0 = sim.geom[0].CellSizeArray();
		sim.userData_.energy_initial_ = compute_gas_plus_binding_energy(sim.state_new_cc_[0], dx0) + compute_total_erad(sim.state_new_cc_[0], dx0);
		sim.userData_.energy_injected_ = 0.0_rt;
		sim.userData_.energy_injected_ir_ = 0.0_rt;
		sim.userData_.energy_injected_optical_ = 0.0_rt;
		sim.userData_.energy_injected_ionizing_ = 0.0_rt;
		sim.userData_.energy_last_t_ = 0.0_rt;
	}

	sim.evolve();

	int status = 0;

	const double t_end = sim.userData_.t_vec_.back();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = sim.geom[0].CellSizeArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
	const double Lx = sim.geom[0].ProbHiArray()[0] - prob_lo[0];
	// The source sits at the middle of the domain and radiates both ways, so each front has Lx / 2 to travel.
	const double x_source = 0.5 * (prob_lo[0] + sim.geom[0].ProbHiArray()[0]);
	const double half_Lx = 0.5 * Lx;

	// Final-profile dump: per-cell density, velocity, ionization fraction, and band energy densities along
	// the j = k = 0 pencil, written once at the end of the run. Used to locate the shell's internal structure
	// (ionization front, density maximum, forward shock) against the thin-shell ODE position x_ode when
	// diagnosing the shell-position check below.
	{
		amrex::MultiFab const &state_mf = sim.state_new_cc_[0];
		const int nx = sim.geom[0].Domain().length(0);
		constexpr int ncols = 6; // rho, vx, x_HII, E_ir, E_opt, E_ion
		std::vector<double> profile(static_cast<std::size_t>(nx) * ncols, 0.0);

		for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();

			// In GPU builds, MultiFab data resides on device; copy to pinned host memory before CPU access.
			amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
			static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
			amrex::Gpu::synchronize();

			const auto state = host_fab.const_array();

			amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
				if (j != 0 || k != 0) {
					return; // one pencil is enough: the problem is planar
				}
				const amrex::Real rho = state(i, j, k, HydroSystem<DTypeFront1D>::density_index);
				const amrex::Real vx = state(i, j, k, HydroSystem<DTypeFront1D>::x1Momentum_index) / rho;
				const amrex::Real n_HI_cell = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 1) / spmasses[1];
				const amrex::Real n_HII_cell = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 2) / spmasses[2];
				const amrex::Real denom = n_HI_cell + n_HII_cell;
				const amrex::Real x_HII = (denom > 0.0_rt) ? (n_HII_cell / denom) : 0.0_rt;
				const int erad0 = RadSystem<DTypeFront1D>::radEnergy_index;
				const std::size_t row = static_cast<std::size_t>(i) * ncols;
				profile[row + 0] = rho;
				profile[row + 1] = vx;
				profile[row + 2] = x_HII;
				profile[row + 3] = state(i, j, k, erad0 + Physics_NumVars::numRadVarsPerGroup * group_ir);
				profile[row + 4] = state(i, j, k, erad0 + Physics_NumVars::numRadVarsPerGroup * group_optical);
				profile[row + 5] = state(i, j, k, erad0 + Physics_NumVars::numRadVarsPerGroup * (Physics_Traits<DTypeFront1D>::nGroups - 1));
			});
		}

		amrex::ParallelDescriptor::ReduceRealSum(profile.data(), static_cast<int>(profile.size()));
		if (amrex::ParallelDescriptor::IOProcessor()) {
			std::ofstream profile_file(std::string("dtype_front_1d_profile") + therm_suffix + ".csv");
			profile_file << "x,rho,vx,xHII,E_ir,E_opt,E_ion\n";
			for (int i = 0; i < nx; ++i) {
				const double x = prob_lo[0] + (static_cast<double>(i) + 0.5) * dx[0];
				profile_file << x;
				for (int c = 0; c < ncols; ++c) {
					profile_file << ',' << profile[static_cast<std::size_t>(i) * ncols + c];
				}
				profile_file << '\n';
			}
		}
	}

	// Check 1: gas temperature in the ionized cavity and in the undisturbed neutral gas. Each must sit at the
	{
		const double ne_eq = sim.userData_.primary_species_2;
		const double T_ion_eq = compute_equilibrium_temperature_ionized(ne_eq);
		const double n_HI_init = sim.userData_.primary_species_2;
		const double T_neu_eq = compute_equilibrium_temperature_neutral(n_HI_init);

		amrex::MultiFab const &state_mf = sim.state_new_cc_[0];

		// Collect temperatures per region: cavity (x_HII > 90%), neutral (x_HI > 99.99%).
		const double v_quiescent = 0.05 * ionized_sound_speed(sim.userData_.T_ionized);
		std::vector<double> cavity_temps;
		std::vector<double> neutral_temps;

		for (amrex::MFIter mfi(state_mf); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();

			// In GPU builds, MultiFab data resides on device; copy to pinned host memory before CPU access.
			amrex::FArrayBox host_fab(box, state_mf.nComp(), amrex::The_Pinned_Arena());
			static_cast<void>(state_mf[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, state_mf.nComp(), host_fab.dataPtr()));
			amrex::Gpu::synchronize();

			const auto state = host_fab.const_array();

			amrex::LoopOnCpu(box, [&](int i, int j, int k) noexcept {
				const amrex::Real x_cell = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
				const amrex::Real dist_from_source = std::abs(x_cell - x_source);
				const amrex::Real rho = state(i, j, k, HydroSystem<DTypeFront1D>::density_index);
				const amrex::Real Eint = state(i, j, k, RadSystem<DTypeFront1D>::gasInternalEnergy_index);
				const amrex::Real n_HI_cell = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 1) / spmasses[1];
				const amrex::Real n_HII_cell = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + 2) / spmasses[2];
				const amrex::Real denom = n_HI_cell + n_HII_cell;
				if (denom <= 0.0_rt) {
					return;
				}
				const amrex::Real x_HII = n_HII_cell / denom;
				const amrex::Real x_HI = n_HI_cell / denom;

				burn_t bstate;
				for (int nn = 0; nn < NumSpec; ++nn) {
					bstate.xn[nn] = state(i, j, k, HydroSystem<DTypeFront1D>::scalar0_index + nn) / spmasses[nn];
				}
				bstate.rho = rho;
				bstate.e = Eint / rho;
				bstate.T = 1.0e4; // initial guess
				eos(eos_input_re, bstate);
				const double T_cell = bstate.T;

				if (x_HII > 0.90_rt) {
					cavity_temps.push_back(T_cell);
				}
				// Quiescent neutral gas only: see the comment above. The gas the front has set in motion is
				// adiabatically cooled and out of thermal equilibrium, and |vx| is what separates it from the
				// gas still sitting on the KI balance.
				const amrex::Real vx = state(i, j, k, HydroSystem<DTypeFront1D>::x1Momentum_index) / rho;
				if (x_HI > 0.9999_rt && std::abs(vx) < v_quiescent) {
					neutral_temps.push_back(T_cell);
				}
			});
		}

		auto compute_median_and_check = [&](std::vector<double> &local_temps, double T_analytical, const char *region_name, const char *quantity_name,
						    const char *unit) {
			const int num_local = static_cast<int>(local_temps.size());
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
			const double *send_ptr = local_temps.empty() ? &static_val : local_temps.data();
			double *recv_ptr = all_temps.empty() ? &static_val : all_temps.data();
			amrex::ParallelDescriptor::Gatherv(send_ptr, num_local, recv_ptr, recvcnt, disp, amrex::ParallelDescriptor::IOProcessorNumber());

			if (amrex::ParallelDescriptor::IOProcessor()) {
				const int ntot = static_cast<int>(all_temps.size());
				if (ntot == 0) {
					amrex::Print() << "Warning: no " << region_name << " cells found.\n";
					return;
				}
				std::sort(all_temps.begin(), all_temps.end());
				const double T_median = (ntot % 2 == 0) ? 0.5 * (all_temps[ntot / 2 - 1] + all_temps[ntot / 2]) : all_temps[ntot / 2];
				const double rel_err = std::abs(T_median - T_analytical) / T_analytical;
				if (rel_err > 0.05) {
					amrex::Print() << "Test FAILED: " << region_name << " median " << quantity_name << " " << T_median << unit
						       << " differs from analytical equilibrium " << T_analytical << unit << " by " << 100.0 * rel_err
						       << "% (tolerance: 5%).\n";
					status = 1;
				} else {
					amrex::Print() << "Test passed: " << region_name << " median " << quantity_name << " " << T_median << unit
						       << " is within 5% of analytical equilibrium " << T_analytical << unit << " (" << ntot << " cells).\n";
				}
			}
		};

		compute_median_and_check(cavity_temps, T_ion_eq, "cavity", "temperature", " K");
		compute_median_and_check(neutral_temps, T_neu_eq, "neutral", "temperature", " K");
	}

	// Check 2: the D-type front radius against the numerically integrated thin-shell solution that carries
	{
		const double x_front = sim.userData_.xeff_vec_.back();
		const double x_shell = sim.userData_.xshell_vec_.back();
		const double x_ode = sim.userData_.xode_vec_.back();
		const double cell_diff = (x_front - x_ode) / dx[0];
		const double shell_cell_diff = (x_shell - x_ode) / dx[0];

		const double tol_cells = 3.0;

		amrex::Print() << "Integrated solution (gas + radiation pressure):   " << x_ode << " cm\n";

		if (!(x_ode > 0.0)) {
			amrex::Print() << "Test FAILED: the integrated front solution is not positive; check photoionize.flux_ion.\n";
			status = 1;
		} else if (x_ode >= half_Lx) {
			amrex::Print() << "Test FAILED: the integrated front has left the domain; reduce stop_time.\n";
			status = 1;
		} else if (std::abs(cell_diff) > tol_cells) {
			amrex::Print() << "Test FAILED: D-type I front differs from the integrated radiation + gas pressure solution by more than " << tol_cells
				       << " cells (" << cell_diff << " cells).\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: D-type I front matches the integrated radiation + gas pressure solution within " << tol_cells
				       << " cells (" << cell_diff << " cells).\n";
		}

		amrex::Print() << "Numerical max-density shell position: " << x_shell << " cm\n";

		if (std::abs(shell_cell_diff) > tol_cells) {
			amrex::Print() << "Test FAILED: max-density shell differs from the integrated radiation + gas pressure solution by more than "
				       << tol_cells << " cells (" << shell_cell_diff << " cells).\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: max-density shell matches the integrated radiation + gas pressure solution within " << tol_cells
				       << " cells (" << shell_cell_diff << " cells).\n";
		}
	}

	// Check 3: total energy conservation if c_hat = c. Includes gas kinetic energy: radiation momentum
	// deposition does real work on the gas (see dEkin_work in AddSourceTermsMultiGroup), converting some of the
	// injected radiation energy into bulk gas motion, so a budget that omits KE is not actually checking total
	// energy conservation.
	if (RadSystem_Traits<DTypeFront1D>::c_hat_over_c == 1.0) {
		// No dust-heating energy is ever in flight: photochemistry runs at the top of a radiation subcycle
		// and AddSourceTerms consumes its deposit in the two IMEX stages of that same subcycle. Kept as a
		// named zero so the printed budget keeps its line.
		const amrex::Real lag_energy = 0.0_rt;
		const amrex::Real E_rad_final = compute_total_erad(sim.state_new_cc_[0], dx);
		const amrex::Real E_gas_final = compute_gas_plus_binding_energy(sim.state_new_cc_[0], dx);
		const amrex::Real E_kin_final = compute_kinetic_energy(sim.state_new_cc_[0], dx);
		const amrex::Real E_final = E_rad_final + E_gas_final + E_kin_final + lag_energy;
		const amrex::Real E_expected = sim.userData_.energy_initial_ + sim.userData_.energy_injected_;

		const amrex::Real abs_err = E_final - E_expected;
		const amrex::Real rel_err = (E_expected != 0.0) ? std::abs(abs_err) / std::abs(E_expected) : std::abs(abs_err);

		amrex::Print() << "Energy budget [erg cm^-2]:\n"
			       << "  initial (gas + binding + radiation) = " << sim.userData_.energy_initial_ << "\n"
			       << "  injected by the source              = " << sim.userData_.energy_injected_ << "\n"
			       << "  expected final                      = " << E_expected << "\n"
			       << "  actual final radiation              = " << E_rad_final << "\n"
			       << "  actual final gas + binding          = " << E_gas_final << "\n"
			       << "  actual final kinetic                = " << E_kin_final << "\n"
			       << "  lagged dust heating (in flight)     = " << lag_energy << "\n"
			       << "  actual final total                  = " << E_final << "\n";

		const amrex::Real tol_energy = 1.0e-2;
		if (rel_err > tol_energy) {
			amrex::Print() << "Test FAILED: total energy is not conserved; final total differs from initial + injected by a fraction of " << rel_err
				       << " (tolerance: " << tol_energy << ").\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: total energy is conserved to a fraction of " << rel_err << " (tolerance: " << tol_energy << ").\n";
		}
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		const auto n = static_cast<int>(sim.userData_.t_vec_.size());
		std::vector<amrex::Real> t_Myr(n);
		std::vector<amrex::Real> x_shell_pc(n);
		std::vector<amrex::Real> x_spitzer_pc(n);
		std::vector<amrex::Real> x_eff_pc(n);
		std::vector<amrex::Real> x_ode_pc(n);
		for (int i = 0; i < n; ++i) {
			t_Myr[i] = sim.userData_.t_vec_[i] / seconds_per_Myr;
			x_shell_pc[i] = sim.userData_.xshell_vec_[i] / cm_per_pc;
			x_spitzer_pc[i] = sim.userData_.xspitzer_vec_[i] / cm_per_pc;
			x_eff_pc[i] = sim.userData_.xeff_vec_[i] / cm_per_pc;
			x_ode_pc[i] = sim.userData_.xode_vec_[i] / cm_per_pc;
		}
		matplotlibcpp::clf();
		std::map<std::string, std::string> shell_args;
		shell_args["label"] = "max-density shell";
		shell_args["color"] = "C0";
		std::map<std::string, std::string> eff_args;
		eff_args["label"] = "effective ionized length";
		eff_args["color"] = "C1";
		std::map<std::string, std::string> spitzer_args;
		spitzer_args["label"] = "analytic (planar D-type, 4/5 law)";
		spitzer_args["color"] = "k";
		spitzer_args["linestyle"] = "--";
		std::map<std::string, std::string> ode_args;
		ode_args["label"] = "ODE (gas + radiation pressure)";
		ode_args["color"] = "k";
		ode_args["linestyle"] = ":";
		matplotlibcpp::plot(t_Myr, x_shell_pc, shell_args);
		matplotlibcpp::plot(t_Myr, x_eff_pc, eff_args);
		matplotlibcpp::plot(t_Myr, x_spitzer_pc, spitzer_args);
		matplotlibcpp::plot(t_Myr, x_ode_pc, ode_args);
		matplotlibcpp::xlabel("time (Myr)");
		matplotlibcpp::ylabel("front position from source (pc)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(std::string("./dtype_front_1d_shell") + therm_suffix + ".pdf");
	}
#endif

	{
		const amrex::Real E_ir_final = compute_group_total_erad(sim.state_new_cc_[0], dx, group_ir);
		amrex::Real dust_kappa = 0.0;
		amrex::ParmParse pp("network");
		pp.query("dust_kappa", dust_kappa);
		amrex::Print() << "Total IR band energy [erg cm^-2]: " << E_ir_final << '\n';
		amrex::Print() << "dust_kappa [cm^2 g^-1]: " << dust_kappa << '\n';
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
