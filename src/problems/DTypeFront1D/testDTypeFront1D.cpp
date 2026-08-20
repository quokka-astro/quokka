/// \file testDTypeFront1D.cpp
/// \brief Defines a 1D planar H II region test: a central ionizing source drives an ionization front out to
/// its Stromgren column, while dust reprocesses the accompanying optical light into the IR.
///
/// There are three radiation groups: IR (group 0), optical (group 1) and an ionizing chemistry band
/// (group 2). Constant photon fluxes are injected in the two cells straddling the middle of a uniform, cold,
/// dusty hydrogen slab: photoionize.flux into the OPTICAL band and photoionize.flux_ion into the ionizing
/// band. Both the radiation energy source and the companion radiation flux source are set, the latter as
/// -c * E on the left of the source and +c * E on the right, so each half of the slab is injected fully
/// beamed away from the centre. The two wings are mirror images, so the source injects zero net momentum
/// while each wing carries the outward momentum its luminosity implies; photoionize.flux is the photon flux
/// delivered to EACH side. The IR band receives no source at all.
///
/// Putting the source in the middle rather than against a boundary is what keeps the budgets below clean.
/// Both domain boundaries are reflecting, and a reflecting wall is a momentum source -- it turns radiation
/// around, and the reduced speed of light amplifies the bookkeeping value of what it turns around by c/chat.
/// With the source at the centre and the run stopped well before either front arrives, neither wall is ever
/// reached, so nothing is reflected and nothing escapes. Beaming each wing outward at injection is the other
/// half of that: an isotropic source injects no momentum at all, leaving the momentum budget below to measure
/// how fast the M1 closure can beam it instead of what the solver did with it.
///
/// The two sourced bands are scaled differently inside the solver -- a thermal group's source is multiplied
/// by chat/c and a chemistry band's is not -- so the shipped fluxes differ by exactly c/chat = 1000 and
/// deliver equal energy. The ionizing band is transparent to the dust opacity, so photoionization is the
/// only process that removes it, and it ionizes the slab as it advances.
///
/// A separate dust temperature is solved for, with the gas-dust collisional coupling switched off
/// (radiation.dust_gas_interaction_coeff = 0), so the dust sits at radiative equilibrium and exchanges no
/// energy with the gas. Radiation momentum is still deposited, so the radiation does accelerate the gas.
///
/// The dust opacity is gray within each band and set at runtime (photoionize.kappa1 for the IR,
/// photoionize.kappa2 for the optical), with the optical opacity much the larger, as for real dust. The
/// chain the test exercises is therefore:
///
///   isotropic optical source -> absorbed by dust -> dust heats to radiative equilibrium -> re-emitted as IR
///
/// Opacity in Quokka is pure absorption, so an opaque group also emits its share of the local blackbody,
/// which is what supplies the re-emission. The dust settles where absorption balances emission,
/// a * T^4 = E_IR + (kappa_opt / kappa_IR) * E_optical, giving ~130 K for the shipped parameters (the dust
/// temperature is internal to the solver and is not stored in the state, so the checks below do not read it
/// directly). At that temperature h*nu/(k*T) = 37 at the IR/optical boundary, so the Planck function has
/// nothing left above the boundary and essentially all of the re-emission lands in the IR band rather than
/// back in the optical one. Behind each light front the optical band is then in pure attenuation,
///
///   E_optical(x) = (F * E_photon / c) * exp(-kappa_opt * rho * |x - x_c|),
///
/// and the energy it loses reappears in the IR. The IR is nearly transparent and fills the domain.
///
/// The test checks that the two thermal bands together conserve the energy injected into the optical one
/// (photoionization drains the ionizing band only, so it is budgeted separately), that the momentum the two
/// wings carry outward plus what they have handed to the gas accounts for the beamed momentum of the
/// injected luminosity while the signed total stays exactly zero, that the optical light front sits near
/// chat * t from the source, that the dust reprocessed a substantial fraction of the optical light into the
/// IR with the survivors following exp(-tau), and that the ionizing band's photon budget balances against
/// the ionized column it produced, which is the Stromgren column for this slab.

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
	// beta_order = 1: keep the O(v/c) terms in the radiation-matter coupling, including the work term. The
	// outward-momentum budget in problem_main is the check that needs them; with beta_order = 0 the radiation
	// force is still applied, but the work done by that force on the moving gas is dropped.
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

// Ionized hydrogen column: sum_cells n_HII * dx  [cm^-2]. The slab is uniform across y and z, so in more than
// one dimension the domain sum counts the same column once per transverse cell; transverse_cells divides that
// back out and makes the result the per-unit-area column in any dimensionality.
auto compute_ionized_column(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::Real transverse_cells)
    -> amrex::Real
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
	return column / transverse_cells;
}

// Domain-integrated radiation energy of group g: sum_cells Erad_g * dx  [erg cm^-2], per unit area; see
// compute_ionized_column for transverse_cells.
auto compute_group_total_erad(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, int g, amrex::Real transverse_cells)
    -> amrex::Real
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
	return total / transverse_cells;
}

// Domain-integrated x-momentum of the gas [g cm^-1 s^-1 in 1D]. With outward set, each cell is signed by
// sgn(x - x_source) so the result measures momentum directed away from the source; without it the plain
// signed sum is returned, which the mirror symmetry of the problem forces to zero.
auto compute_gas_momentum(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real x_source, bool outward, amrex::Real transverse_cells)
    -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
		const amrex::Real sign = (!outward || x > x_source) ? 1.0_rt : -1.0_rt;
		return sign * cell_length * state[box_no](i, j, k, RadSystem<DTypeFront1D>::x1GasMomentum_index);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real momentum = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(momentum, amrex::ParallelContext::CommunicatorSub());
	return momentum / transverse_cells;
}

// Domain-integrated x-momentum of the radiation field: sum_cells sign * w_g * F_x,g * dx for group g, with the same sign convention as compute_gas_momentum.
// The weight w_g turns a radiation flux into the momentum the solver actually trades with the gas, and it is not the same for the two kinds of band. A thermal
// group uses w = 1 / (c * chat), the pairing in UpdateFlux (source_terms_multi_group.hpp), while a chemistry band uses w = 1 / c^2, the pairing in
// computePhotoChemistry (photochemistry.hpp). They differ because a chemistry band's energy density is deliberately inflated by c / chat so that chat *
// n_photon reproduces the physical photon flux and the ionization rate comes out right; its momentum weight divides that factor back out. Under either weight
// an injected photon flux Phi carries the physical momentum flux Phi * E_photon / c, which is what makes the single budget below meaningful across both kinds
// of band.
auto compute_group_rad_momentum(amrex::MultiFab const &state_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::Real x_source, bool outward, int g,
				amrex::Real transverse_cells) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const amrex::Real cell_length = dx[0];
	const amrex::Real x_lo = prob_lo[0];
	const int frad_index = RadSystem<DTypeFront1D>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g;
	const amrex::Real weight = (g == group_ionizing) ? 1.0_rt / (C::c_light * C::c_light) : 1.0_rt / (C::c_light * c_hat);

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real x = x_lo + (static_cast<amrex::Real>(i) + 0.5_rt) * cell_length;
		const amrex::Real sign = (!outward || x > x_source) ? 1.0_rt : -1.0_rt;
		return sign * cell_length * weight * state[box_no](i, j, k, frad_index);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real momentum = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Sum(momentum, amrex::ParallelContext::CommunicatorSub());
	return momentum / transverse_cells;
}

// Largest reduced flux |F| / (c E) of group g anywhere in the domain. The M1 closure admits no state with
// |F| > c E, so this is a physical invariant rather than a tuned tolerance: a value above one means the
// radiation flux and the radiation energy have been updated inconsistently. It is what detects a flux source
// applied to a group whose energy source was dropped, which is how a transparent sourced group behaved in the
// dust solvers before the injection loop there was added (radiation_dust_system.hpp).
auto compute_max_reduced_flux(amrex::MultiFab const &state_mf, int g) -> amrex::Real
{
	amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
	amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
	auto const state = state_mf.const_arrays();
	const int erad_index = RadSystem<DTypeFront1D>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g;
	const int frad_index = RadSystem<DTypeFront1D>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g;

	reduce_op.eval(state_mf, amrex::IntVect(0), reduce_data, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
		const amrex::Real erad = state[box_no](i, j, k, erad_index);
		if (!(erad > 0.0_rt)) {
			return 0.0_rt;
		}
		return std::abs(state[box_no](i, j, k, frad_index)) / (C::c_light * erad);
	});

	auto const &hv = reduce_data.value(reduce_op);
	amrex::Real max_f = amrex::get<0>(hv);
	amrex::ParallelAllReduce::Max(max_f, amrex::ParallelContext::CommunicatorSub());
	return max_f;
}

} // namespace

template <>
void RadSystem<DTypeFront1D>::AddRadSource(array_t &radEnergy, array_t &reducedFlux, const amrex::Box &indexRange,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	// Planar photon source straddling the middle of the domain, occupying photoionize.source_cells cells on
	// each side of it. radEnergy is a luminosity volume density [erg s^-1 cm^-3]: the slab emits
	// 2 * F * E_photon per unit area per unit time, half of it to each side, and dividing by the slab width
	// 2 * n * dx[0] gives the volumetric rate F * E_photon / (n * dx[0]) in each source cell. So
	// photoionize.flux is the photon flux delivered to EACH side, whatever the slab is widened to.
	//
	// The flux source makes each half of the slab emit outward instead of isotropically: the cells left of
	// centre are given a reduced flux of -1 and those right of centre +1, which is the fully beamed
	// (free-streaming, |F| = c E) injection documented on AddRadSource. The two wings are
	// mirror images, so the source still injects zero *net* momentum, but each wing now arrives carrying the
	// outward momentum F * E_photon / c per unit area and time that a beam of that luminosity must have.
	// That is what makes the outward-momentum budget in problem_main a conservation law rather than a
	// statement about how fast M1 can beam an isotropic source (see the discussion there). Setting
	// photoionize.beamed = 0 restores the isotropic source and is useful for exactly that comparison.
	//
	// The optical and ionizing bands are fed; the IR band receives no source at all and is filled purely by
	// the dust's own thermal re-emission of the absorbed optical light.
	//
	// The two sourced bands take different internal scalings: a thermal group's source is multiplied by
	// chat/c, a chemistry band's is not (see source_terms_multi_group.hpp), and the flux source is scaled to
	// match its own energy source. The hook takes a reduced flux rather than a flux, so a reduced flux of
	// unit magnitude means "beamed" for either kind of band and |F| > c E is unrepresentable.
	// The shipped fluxes differ by exactly that factor of c/chat = 1000, so the two bands receive the same
	// injected energy and the energy budget in problem_main would be off by three orders of magnitude if
	// either scaling were wrong.
	amrex::ParmParse const pp("photoionize");
	amrex::Real flux = 1.0e11_rt;
	pp.query("flux", flux);
	amrex::Real flux_ion = 0.0_rt;
	pp.query("flux_ion", flux_ion);
	int source_cells = 1;
	pp.query("source_cells", source_cells); // cells per side occupied by the source slab
	int beamed = 1;
	pp.query("beamed", beamed); // 1 = each wing injected beamed outward, 0 = isotropic

	const auto n_cells = static_cast<amrex::Real>(source_cells);
	const amrex::Real src_optical = flux * E_photon / (n_cells * dx[0]);
	const amrex::Real src_ionizing = flux_ion * E_photon / (n_cells * dx[0]);

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
				if (g == group_optical) {
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
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	// Track the optical band. The threshold is well below the unattenuated plateau because dust absorption
	// thins the light as it advances: at the front the optical energy density is down by exp(-tau), so a 0.5
	// threshold would report the absorption depth rather than the front. See front_threshold_fraction.
	const amrex::Real Erad_threshold = front_threshold_fraction * compute_plateau_erad(userData_.flux);
	const amrex::Real x_source = 0.5 * (prob_lo[0] + prob_hi[0]);
	// Distance travelled from the source, not an absolute position: the source sits at the middle of the
	// domain and the +x front is the one compute_front_position reports.
	const amrex::Real x_front = compute_front_position(state_new_cc_[lev], dx, prob_lo, Erad_threshold, group_optical) - x_source;
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
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
	const double Lx = sim.geom[0].ProbHiArray()[0] - prob_lo[0];
	// The source sits at the middle of the domain and radiates both ways, so each front has Lx / 2 to travel.
	const double x_source = 0.5 * (prob_lo[0] + sim.geom[0].ProbHiArray()[0]);
	const double half_Lx = 0.5 * Lx;
	// The slab is uniform across y and z, so in 2D/3D every transverse column repeats the 1D solution. The
	// domain sums below would then count it once per transverse cell; dividing by that count keeps every
	// budget a per-unit-area quantity and lets the same tolerances apply in any dimensionality.
	const amrex::Box &domain_box = sim.geom[0].Domain();
	const double transverse_cells = static_cast<double>(AMREX_D_TERM(1, *domain_box.length(1), *domain_box.length(2)));

	// Analytic free-streaming solution: a pair of top-hats of height F * E_photon / c reaching
	// |x - x_source| = chat * t.
	auto x_analytic = [=](double t_now) -> double { return c_hat * t_now; };

	// Check 1: conservation of energy in the two thermal bands, and of outward linear momentum. The energy
	// part first. No radiation ever reaches a boundary, and with
	// the gas thermally decoupled the dust only shuffles energy between the IR and optical groups, so their
	// combined domain-integrated Erad must equal the energy injected into the optical band. Photoionization
	// removes energy from the ionizing band only, which is why that band is excluded here and gets its own
	// budget in Check 4. This is the primary check on the source-term accounting.
	{
		double E_thermal = 0.0;
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			const double E_g = compute_group_total_erad(sim.state_new_cc_[0], dx, g, transverse_cells);
			amrex::Print() << "Group " << g << " integrated Erad: " << E_g << "\n";
			if (g != group_ionizing) {
				E_thermal += E_g;
			}
		}
		// A thermal group's source carries the code's internal chat/c factor (see source_terms_multi_group.hpp).
		// Only the optical group is sourced among the thermal ones; both start at the radiation floor.
		// The slab emits F * E_photon per unit area per unit time to EACH side, hence the leading factor of two.
		const double injected = 2.0 * (c_hat / C::c_light) * F * E_photon * t_end + 2.0 * Erad_floor_ * Lx;
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

		// Momentum. Every change the solver makes to a radiation flux is paired with an equal and opposite kick
		// to the gas momentum, so the two sourced bands plus the gas must still hold what the source injected.
		// The source is mirror-symmetric, so the signed total momentum is zero and says nothing on its own;
		// what carries the information is the OUTWARD momentum, each cell signed by sgn(x - x_source). Each
		// wing is injected beamed (see AddRadSource), so it arrives already carrying F * E_photon / c per unit
		// time and area, and the two together inject 2 * (F + F_ion) * E_photon / c. Note the absence of any
		// chat factor. The energy budget above is scaled by chat / c and this one is not, so the two are
		// independent statements, and this is the only check that exercises the radiation force. The gas ends
		// up carrying about a third of the injected momentum, so dropping the gas momentum kick, or mis-scaling
		// it by c / chat, misses by far more than the tolerance.
		//
		// The beaming is what makes this a conservation law rather than a statement about the M1 closure. An
		// isotropic source injects no net momentum at all: the outward momentum then has to be *generated* by
		// transport as M1 beams the two wings, which it does only gradually, leaving the budget short and
		// strongly resolution-dependent. Run with photoionize.beamed = 0 to see it -- the ratio falls to 0.78
		// and drifts with resolution. Beamed, it is 1.0038 at the shipped 128 cells and 1.0041 at 1024, i.e.
		// flat to 4e-4 over an eightfold refinement, which is why a 1% tolerance is meaningful here.
		//
		// Widening the source slab does not help either: photoionize.source_cells = 1, 2, 4, 8 gives 1.0038,
		// 1.0083, 1.0105, 1.0118 beamed and 0.78, 0.74, 0.63, 0.47 isotropic. One cell per side is both the
		// tightest and the least intrusive, since a wide slab also pushes the measured light front outward.
		//
		// The IR band is excluded, for the same reason the ionizing band is excluded from the energy budget
		// above: nothing injects momentum into it. The dust creates the IR by re-emission, which is isotropic
		// and carries no net momentum, and the outward flux it subsequently develops is generated by transport
		// down the radiation pressure gradient rather than by the source. Outward momentum is not conserved
		// under transport -- only the signed total is -- so that term has no place in a budget against what was
		// injected. It is not small either: including it would add 9.2% here, because the reservoir
		// F / (c * chat) is inflated by c / chat relative to the physical E / c.
		//
		// The residual +0.4% is the IR the dust reabsorbs: a fraction tau_IR = rho * kappa1 * chat * t_end of
		// the outward IR flux is absorbed and lands in the gas momentum. The tolerance covers it.
		//
		// The signed total is checked alongside, and much more tightly: the source injects zero net momentum,
		// nothing reaches a boundary, and every radiation-gas exchange is equal and opposite, so the signed sum
		// over the gas and all three bands must vanish to round-off. It is what catches an asymmetry between
		// the two wings.
		{
			double p_out_beamed = 0.0;
			double p_signed_total = 0.0;
			for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
				const double p_out = compute_group_rad_momentum(sim.state_new_cc_[0], dx, prob_lo, x_source, true, g, transverse_cells);
				p_signed_total += compute_group_rad_momentum(sim.state_new_cc_[0], dx, prob_lo, x_source, false, g, transverse_cells);
				amrex::Print() << "Group " << g << " outward momentum: " << p_out << "\n";
				if (g != group_ir) {
					p_out_beamed += p_out;
				}
			}
			const double p_gas_out = compute_gas_momentum(sim.state_new_cc_[0], dx, prob_lo, x_source, true, transverse_cells);
			p_signed_total += compute_gas_momentum(sim.state_new_cc_[0], dx, prob_lo, x_source, false, transverse_cells);
			const double p_injected = 2.0 * (F + F_ion) * E_photon * t_end / C::c_light;
			const double p_frac = (p_gas_out + p_out_beamed) / p_injected;
			const double tol_p = 0.01;
			// The signed total is a round-off quantity; it measures 1e-16 of the injected scale here.
			const double tol_symmetry = 1.0e-10;

			amrex::Print() << "Outward momentum: gas " << p_gas_out << " + sourced bands " << p_out_beamed << " = " << p_gas_out + p_out_beamed
				       << " (injected " << p_injected << ", ratio " << p_frac << ", gas share " << p_gas_out / (p_gas_out + p_out_beamed)
				       << ")\n";
			amrex::Print() << "Signed total momentum: " << p_signed_total << ", i.e. " << std::abs(p_signed_total) / p_injected
				       << " of injected (zero by symmetry)\n";

			if (std::abs(p_frac - 1.0) > tol_p) {
				amrex::Print() << "Test FAILED: outward linear momentum is " << p_frac << " of injected (expected 1 within " << tol_p * 100.0
					       << "%).\n";
				status = 1;
			} else if (std::abs(p_signed_total) / p_injected > tol_symmetry) {
				amrex::Print() << "Test FAILED: the two wings are not mirror images; signed total momentum is "
					       << std::abs(p_signed_total) / p_injected << " of injected.\n";
				status = 1;
			} else {
				amrex::Print() << "Test passed: outward linear momentum is conserved (p/injected = " << p_frac << ").\n";
			}
		}
	}

	// Check 2: the light must propagate outward at close to the reduced speed of light. This is a sanity check
	// on transport. Each wing is injected beamed, so it free-streams from the start and its front tracks
	// chat * t rather than the chat / sqrt(3) an isotropic pulse would start at. The measured front lands one
	// cell beyond chat * t at the shipped resolution, so the tolerance mostly covers the quantization of the
	// front position (one cell is 2.6% of the front distance here; the difference falls to 0.7% by 1024
	// cells) and the drag from the gray opacity re-emitting absorbed energy isotropically.
	{
		const double x_front = sim.userData_.xfront_vec_.back();
		const double x_ref = x_analytic(t_end);
		const double percent_diff = std::abs(x_front - x_ref) / x_ref * 100.0;
		const double tol_percent = 10.0;

		amrex::Print() << "Radiation front distance from the source: " << x_front << " cm\n";
		amrex::Print() << "Analytic front distance:  " << x_ref << " cm (chat * t; an isotropic source would give " << x_ref / std::numbers::sqrt3
			       << ")\n";
		amrex::Print() << "Difference: " << percent_diff << " percent (tolerance: " << tol_percent << " percent)\n";

		if (x_ref >= half_Lx) {
			amrex::Print() << "Test FAILED: the analytic front has left the domain; reduce stop_time.\n";
			status = 1;
		} else if (percent_diff > tol_percent) {
			amrex::Print() << "Test FAILED: radiation front differs from chat * t by more than " << tol_percent << " percent.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: the source propagates outward at chat within " << tol_percent << " percent.\n";
		}
	}

	// Check 3: dust reprocessing. Nothing is injected into the IR band, so every erg in it arrived there by
	// being absorbed out of the optical light and thermally re-emitted by the dust. Behind each front the
	// optical band is in steady state and simply attenuates, E_opt(x) = (F E_photon / c) exp(-kappa_opt rho
	// |x - x_source|), because the ~130 K dust emits nothing back into the optical; integrating that out to
	// both fronts gives a closed form for what should be left unprocessed, hence the factor of two.
	{
		const double E_ir = compute_group_total_erad(sim.state_new_cc_[0], dx, group_ir, transverse_cells);
		const double E_opt = compute_group_total_erad(sim.state_new_cc_[0], dx, group_optical, transverse_cells);
		const double rho_0 = sim.userData_.primary_species_2 * spmasses[1]; // initial neutral-H mass density
		const double alpha_opt = rho_0 * kappa2;			    // optical absorption coefficient [cm^-1]
		const double tau_front = alpha_opt * x_analytic(t_end);
		const double E_opt_ref = 2.0 * compute_plateau_erad(F) * (1.0 - std::exp(-tau_front)) / alpha_opt;
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
			amrex::Print() << "Test FAILED: too little of the optical light was reprocessed into the IR (" << reprocessed << ").\n";
			status = 1;
		} else if (std::abs(opt_frac - 1.0) > 0.10) {
			amrex::Print() << "Test FAILED: surviving optical energy is " << opt_frac
				       << " of the attenuated-beam solution (expected 1 within 10%).\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: dust reprocessed " << reprocessed
				       << " of the optical light into the IR, and the surviving optical matches exp(-tau) (ratio " << opt_frac << ").\n";
		}
	}

	// Check 4: photon budget of the ionizing band. It is transparent to the dust opacity, so photoionization
	// is the only thing that removes it: every photon absorbed ionizes one hydrogen atom, and the ones that
	// have not since recombined are still visible as the ionized column. Photon conservation therefore reads
	//
	//   injected = still in the field + ionized column + recombinations,
	//
	// and since the recombinations are not tracked here this is enforced as the two inequalities below: the
	// band must be measurably depleted, and the absorbed photons must at least cover the ionized column summed
	// over both ionization fronts.
	//
	// The budget inequality alone is weak -- it has a factor of ten of headroom here, and it does NOT catch
	// a mis-scaled chemistry-band source, because fewer photons simply produce a proportionally smaller
	// ionized column and the inequality still holds (verified: scaling the source by chat/c passes it). The
	// surviving fraction is what catches that. Each ionization front stalls at its Stromgren column well
	// inside the light-travel distance chat * t, so a good fraction of the injected photons are always still
	// in flight; mis-scaling the source by c/chat = 1000 collapses the surviving fraction from 0.20 to
	// 8e-5. Hence the lower bound as well as the upper one.
	{
		const double E_ion = compute_group_total_erad(sim.state_new_cc_[0], dx, group_ionizing, transverse_cells);
		// As in the energy budget above, the slab feeds both sides, hence the factor of two.
		const double injected_ion = 2.0 * F_ion * E_photon * t_end + Erad_floor_ * Lx;
		const double ion_frac = E_ion / injected_ion;
		const double n_injected = injected_ion / E_photon;
		const double n_absorbed = (injected_ion - E_ion) / E_photon;
		const double column_HII = compute_ionized_column(sim.state_new_cc_[0], dx, transverse_cells);

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

	// Check 5: the M1 reduced-flux invariant, |F| <= c E, in every cell of every group. This is not a
	// tuned tolerance but a property the closure cannot violate, so any excess means the radiation flux and
	// the radiation energy were updated inconsistently -- which is exactly what happens if a flux source is
	// applied to a group whose energy source was dropped. The dust solvers used to drop the energy source of
	// a transparent (tau == 0) group while UpdateFlux applied its flux source regardless; that produced
	// |F| / (c E) ~ 1e12 here. Run with photoionize.kappa2 = 0 to exercise that path: it makes the optical
	// band transparent while the dust solver is still active, and this check is what catches a regression.
	{
		double max_f = 0.0;
		for (int g = 0; g < Physics_Traits<DTypeFront1D>::nGroups; ++g) {
			const double f_g = compute_max_reduced_flux(sim.state_new_cc_[0], g);
			amrex::Print() << "Group " << g << " max reduced flux |F|/(cE): " << f_g << "\n";
			max_f = std::max(max_f, f_g);
		}
		// The beamed bands sit at exactly 1; allow only round-off above it.
		const double max_f_tol = 1.0 + 1.0e-6;

		if (max_f > max_f_tol) {
			amrex::Print() << "Test FAILED: reduced flux |F|/(cE) reaches " << max_f << ", which the M1 closure forbids.\n";
			status = 1;
		} else {
			amrex::Print() << "Test passed: reduced flux stays within the M1 bound (max " << max_f << ").\n";
		}
	}

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr amrex::Real seconds_per_Myr = 3.15576e13;
		constexpr amrex::Real cm_per_pc = 3.085677581491367e18;

		const auto n = static_cast<int>(sim.userData_.t_vec_.size());
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
