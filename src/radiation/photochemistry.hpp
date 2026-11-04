#ifndef PHOTOCHEMISTRY_HPP_ // NOLINT
#define PHOTOCHEMISTRY_HPP_

#include <array>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"

#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"

#ifdef PHOTOCHEMISTRY
#include "network.hpp"
#include "photochemistry/PhotochemEOS.hpp"
#include "photochemistry/PhotochemParams.hpp"
#include "photochemistry/PhotochemState.hpp"
#include "physics_numVars.hpp"
#include "rosenbrock/rosenbrock.hpp"

namespace quokka::photochemistry
{
using namespace amrex::literals; // NOLINT(google-build-using-namespace)

// Only relevant when the O(v/c) work term is active (see do_vc_work below). If true, the work-term kinetic
// energy the gas gains from the absorbed-photon momentum kick is debited from the chem-band photon field
// (photon energy density reduced by (c / chat) * dEkin, and each band's flux rescaled by the same factor to
// preserve the M1 closure |F| <= c E), strictly conserving the gas+radiation energy. If false (default), the
// photon field is left unchanged after the burn: the gas still gains the kinetic energy, but it is not removed
// from the radiation, so the ionizing photon number density is conserved at the cost of gas+radiation energy
// conservation. Conserving the ionizing photon budget is the more important property for correct
// ionization-front propagation, so it is the default.
static constexpr bool debit_work_term_from_radiation = false;

// Bounds outside which the network is frozen (RHS and Jacobian set to zero), as in Microphysics: the EOS minimum
// temperature and integrator.MAX_TEMP. (Under Microphysics the photochemistry problems also called eos_init with their
// own small_temp = 1e-2 K; at physical temperatures the choice has no effect.)
static constexpr Real photochem_min_temp = 1.0e-10;
static constexpr Real photochem_max_temp = 1.0e11;
// floor applied to photon number densities, thermal-band energies and species inside every RHS call (Microphysics SMALL_X_SAFE)
static constexpr Real photochem_small_x_safe = 1.0e-30;

namespace detail
{
// solver-vector offsets (0-based)
constexpr int i_enuc = NumSpec;
constexpr int i_rad0 = NumSpec + 1;
constexpr int i_thermal0 = i_rad0 + NumChemRadEqs;
constexpr int i_dust = i_thermal0 + NumThermalBands; // valid only with DUST_CHEMICAL_BAND_ABSORPTION
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
static_assert(i_dust < network::NumEqs, "DUST_CHEMICAL_BAND_ABSORPTION needs a network with a dust-absorption slot in its solver vector "
					"(e.g. thermophotochemistry_handwritten)");
#endif

/// The ODE system for one cell: wraps the network RHS and Jacobian with Microphysics's clean_state clamps, the EOS
/// call and the temperature bounds (integration/integrator_rhs_strang.H, integrator_type_strang.H).
struct PhotochemSystem {
	network::State state; // rho, c_hat and dust_kappa are fixed; the rest is unpacked from y
	network::NetworkParams params;

	AMREX_GPU_HOST_DEVICE static void clean(rosenbrock::Vector<network::NumEqs> &y)
	{
		for (int n = 0; n < NumSpec; ++n) {
			y[n] = amrex::max(y[n], photochem_small_x_safe);
		}
		for (int n = i_rad0; n < i_rad0 + NumChemRadEqs; n += NumChemRadVarsPerGroup) {
			y[n] = amrex::max(y[n], photochem_small_x_safe);
		}
		for (int n = i_thermal0; n < i_thermal0 + NumThermalBands; ++n) {
			y[n] = amrex::max(y[n], photochem_small_x_safe);
		}
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
		y[i_dust] = amrex::max(y[i_dust], 0.0_rt);
#endif
	}

	AMREX_GPU_HOST_DEVICE void unpack(rosenbrock::Vector<network::NumEqs> const &y)
	{
		for (int n = 0; n < NumSpec; ++n) {
			state.xn[n] = y[n];
		}
		state.e = y[i_enuc];
		for (int n = 0; n < NumChemRadEqs; ++n) {
			state.rn[n] = y[i_rad0 + n];
		}
		for (int n = 0; n < NumThermalBands; ++n) {
			state.re_thermal[n] = y[i_thermal0 + n];
		}
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
		state.e_dust_absorbed = y[i_dust];
#endif
		state.T = TemperatureFromEnergy<NumSpec>(state.e, state.xn, spmasses, gammas);
		state.dedT = dEnergy_dT<NumSpec>(state.xn, spmasses, gammas);
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE auto in_temperature_bounds() const -> bool
	{
		return (state.T > photochem_min_temp) && (state.T < photochem_max_temp);
	}

	AMREX_GPU_HOST_DEVICE void rhs(Real /*t*/, rosenbrock::Vector<network::NumEqs> &y, rosenbrock::Vector<network::NumEqs> &ydot)
	{
		clean(y);
		unpack(y);
		if (!in_temperature_bounds()) {
			for (int n = 0; n < network::NumEqs; ++n) {
				ydot[n] = 0.0;
			}
			return;
		}
		network::network_rhs(state, params, ydot);
	}

	AMREX_GPU_HOST_DEVICE void jac(Real /*t*/, rosenbrock::Vector<network::NumEqs> const &y, rosenbrock::Matrix<network::NumEqs> &J)
	{
		unpack(y);
		if (!in_temperature_bounds()) {
			for (int j = 0; j < network::NumEqs; ++j) {
				for (int i = 0; i < network::NumEqs; ++i) {
					J(i, j) = 0.0;
				}
			}
			return;
		}
		network::network_jac(state, params, J);
	}
};

/// Solver settings: which components enter the error norm, their tolerances, and the validity limits, as in
/// Microphysics's Rosenbrock integrator (rtol_for/atol_for, valid_integrator_state, species_change_valid).
inline auto MakeSolverSettings(IntegratorParams const &ip) -> rosenbrock::Settings<network::NumEqs>
{
	constexpr Real inf = std::numeric_limits<Real>::infinity();
	rosenbrock::Settings<network::NumEqs> s{};
	for (int n = 0; n < network::NumEqs; ++n) {
		s.atol[n] = 1.0;
		s.rtol[n] = 1.0;
		s.in_error_norm[n] = false;
		s.min_value[n] = -inf;
		s.change_floor[n] = inf;
	}
	for (int n = 0; n < NumSpec; ++n) {
		s.atol[n] = ip.atol_spec;
		s.rtol[n] = ip.rtol_spec;
		s.in_error_norm[n] = true;
		s.min_value[n] = -ip.atol_spec;
		s.change_floor[n] = ip.X_reject_buffer * ip.atol_spec;
	}
	s.atol[i_enuc] = ip.atol_enuc;
	s.rtol[i_enuc] = ip.rtol_enuc;
	s.in_error_norm[i_enuc] = true;
	s.min_value[i_enuc] = std::numeric_limits<Real>::min(); // the specific internal energy must stay > 0
	for (int n = i_rad0; n < i_rad0 + NumChemRadEqs; n += NumChemRadVarsPerGroup) {
		s.atol[n] = ip.atol_rad_num;
		s.rtol[n] = ip.rtol_rad_num;
		s.in_error_norm[n] = true;
		s.min_value[n] = -ip.atol_rad_num;
	}
	s.max_steps = ip.ode_max_steps;
	return s;
}
} // namespace detail

template <typename problem_t>
auto computePhotoChemistry(amrex::MultiFab &mf, std::array<amrex::MultiFab const *, AMREX_SPACEDIM> const &fc_mfs, const Real dt,
			   const Real max_density_allowed, const Real min_density_allowed, amrex::MultiFab &dustHeatingSource, amrex::MultiFab &radEnergySource)
    -> bool
{
	amrex::ignore_unused(dustHeatingSource, radEnergySource);

	// Start off by assuming a successful burn.
	int photochem_burn_success = 1;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	auto *p_num_failed = d_num_failed.data();

	int num_failed = 0;

	const int firstChemIndex = RadSystem<problem_t>::radEnergy_index +
				   RadSystem<problem_t>::numRadVars_ * (RadSystem<problem_t>::nGroups_ - RadSystem_NChemBands<problem_t>::value);
	const int firstChemFxIndex = firstChemIndex + 1;
	const int firstChemFyIndex = firstChemFxIndex + 1;
	const int firstChemFzIndex = firstChemFyIndex + 1;

	static_assert(!RadSystem<problem_t>::thermal_band_photochemistry_ || NumThermalBands == RadSystem<problem_t>::nGroupsThermal_,
		      "NumThermalBands (set for this network in NetworkRegistry.cmake) must equal RadSystem<problem_t>::nGroupsThermal_ "
		      "(nGroups_ - NChemBands) when ISM_Traits::thermal_band_photochemistry is true.");
#ifndef DUST_CHEMICAL_BAND_ABSORPTION
	static_assert(!RadSystem<problem_t>::dust_chemical_band_absorption_,
		      "ISM_Traits::dust_chemical_band_absorption is true but DUST_CHEMICAL_BAND_ABSORPTION is not defined: "
		      "the network does not compute e_dust_absorbed, so Q_dust would always be zero.");
#endif

	// The O(v/c) radiation-pressure work term is gated on beta_order>=1 && is_hydro_enabled; the condition is
	// inlined inside the device lambda's if constexpr below to avoid NVCC first-capturing a local constexpr.

	amrex::GpuArray<Real, NumChemBands> chemBandQuanta{};
	amrex::GpuArray<Real, NumChemBands> invChemBandQuanta{};
	for (int nn = 0; nn < NumChemBands; ++nn) {
		chemBandQuanta[nn] = RadSystem<problem_t>::GetChemBandQuanta(nn);
		invChemBandQuanta[nn] = 1.0_rt / chemBandQuanta[nn];
	}

	// copied to a local so the device lambda below captures it by value
	const amrex::GpuArray<double, RadSystem<problem_t>::nGroups_ + 1> radBoundaries = RadSystem<problem_t>::radBoundaries_;

	// Cells the burn skips (rho < min_density_allowed) must read zero, not the previous substep's deposit.
	if constexpr (RadSystem<problem_t>::dust_chemical_band_absorption_) {
		dustHeatingSource.setVal(0.0);
	}

	static const IntegratorParams integratorParams = IntegratorParams::read();
	static const network::NetworkParams networkParams = network::NetworkParams::read();
	const rosenbrock::Settings<network::NumEqs> solverSettings = detail::MakeSolverSettings(integratorParams);
	const IntegratorParams ip = integratorParams;
	const network::NetworkParams np = networkParams;
	const Real small_x = np.small_x;

	const BL_PROFILE("PhotoChemistry::computePhotoChemistry()");
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::Array4<amrex::Real> dustHeatingSource_arr{};
		if constexpr (RadSystem<problem_t>::dust_chemical_band_absorption_) {
			dustHeatingSource_arr = dustHeatingSource.array(iter);
		}
		amrex::Array4<amrex::Real> radEnergySource_arr{};
		if constexpr (RadSystem<problem_t>::thermal_band_photochemistry_) {
			radEnergySource_arr = radEnergySource.array(iter);
		}

		std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> cons_fc{};
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			cons_fc[0] = fc_mfs[0]->const_array(iter);
#if (AMREX_SPACEDIM >= 2)
			cons_fc[1] = fc_mfs[1]->const_array(iter);
#endif
#if (AMREX_SPACEDIM == 3)
			cons_fc[2] = fc_mfs[2]->const_array(iter);
#endif
		}

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, RadSystem<problem_t>::gasDensity_index);
			// dont do photochemistry in cells with densities below the minimum density specified
			if (rho < min_density_allowed) {
				return;
			}
			// stop the test if we have reached very high densities
			if (rho > max_density_allowed) {
				amrex::Abort("Density exceeded max_density_allowed!");
			}

			const Real xmom = state(i, j, k, RadSystem<problem_t>::x1GasMomentum_index);
			const Real ymom = state(i, j, k, RadSystem<problem_t>::x2GasMomentum_index);
			const Real zmom = state(i, j, k, RadSystem<problem_t>::x3GasMomentum_index);
			const Real Ener = state(i, j, k, RadSystem<problem_t>::gasEnergy_index);
			const Real Emag = ComputeCellCenteredMagneticEnergy<problem_t>(i, j, k, cons_fc);

			const Real Eint = ::quokka::EOS<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener, Emag);

			amrex::ignore_unused(radBoundaries, radEnergySource_arr);
			if constexpr (RadSystem<problem_t>::nGroupsThermal_ == 0) {
				// With no thermal groups the thermal IMEX stages are skipped, so check the chem-band opacities here.
				const auto massScalars = RadSystem<problem_t>::ComputeMassScalars(state, i, j, k);
				const Real Tgas = ::quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Eint, massScalars);
				RadSystem<problem_t>::AssertChemBandOpacitiesAreZero(radBoundaries, rho, Tgas);
			}

			detail::PhotochemSystem sys{};
			sys.params = np;
			network::State &photochemstate = sys.state;
			int burn_failed = 0;
			photochemstate.c_hat = RadSystem_Traits<problem_t>::c_hat_over_c * C::c_light;
			for (int nn = 0; nn < NumSpec; ++nn) {
				photochemstate.xn[nn] = state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) / spmasses[nn];
			}
#ifdef SKIP_PHOTOCHEMFLUX
			amrex::GpuArray<Real, NumChemBands> n_gamma_initial{};
#endif
			for (int nn = 0; nn < NumChemBands; ++nn) {
				const Real n_gamma = state(i, j, k, firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn) * invChemBandQuanta[nn];
				photochemstate.rn[0 + NumChemRadVarsPerGroup * nn] = n_gamma;
#ifdef SKIP_PHOTOCHEMFLUX
				n_gamma_initial[nn] = n_gamma;
#else
				photochemstate.rn[1 + NumChemRadVarsPerGroup * nn] = 1.0_rt;
#endif
			}
			amrex::GpuArray<Real, amrex::max(NumThermalBands, 1)> re_thermal_initial{};
			if constexpr (RadSystem<problem_t>::thermal_band_photochemistry_) {
				for (int nn = 0; nn < NumThermalBands; ++nn) {
					re_thermal_initial[nn] = state(i, j, k, RadSystem<problem_t>::radEnergy_index + RadSystem<problem_t>::numRadVars_ * nn);
					photochemstate.re_thermal[nn] = re_thermal_initial[nn];
				}
			}
			photochemstate.rho = rho;
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
			photochemstate.dust_kappa = np.dust_kappa;
			photochemstate.e_dust_absorbed = 0.0_rt;
#endif

			// temperature from the EOS, then e from T, as the Microphysics EOS calls with eos_input_re and (in integrator_setup) eos_input_rt did
			photochemstate.T = TemperatureFromEnergy<NumSpec>(Eint / rho, photochemstate.xn, spmasses, gammas);
			photochemstate.e = EnergyFromTemperature<NumSpec>(photochemstate.T, photochemstate.xn, spmasses, gammas);

			// pack the solver vector and integrate
			rosenbrock::Vector<network::NumEqs> y{};
			for (int nn = 0; nn < NumSpec; ++nn) {
				y[nn] = photochemstate.xn[nn];
			}
			y[detail::i_enuc] = photochemstate.e;
			for (int nn = 0; nn < NumChemRadEqs; ++nn) {
				y[detail::i_rad0 + nn] = photochemstate.rn[nn];
			}
			for (int nn = 0; nn < NumThermalBands; ++nn) {
				y[detail::i_thermal0 + nn] = photochemstate.re_thermal[nn];
			}
#ifdef DUST_CHEMICAL_BAND_ABSORPTION
			y[detail::i_dust] = 0.0_rt;
#endif
			const rosenbrock::Result result = rosenbrock::integrate<network::NumEqs>(sys, y, dt, solverSettings);

			// final-state checks (Microphysics integrator_cleanup)
			if (result.status != rosenbrock::Status::Success) {
				burn_failed = 1;
			}
			for (int nn = 0; nn < NumSpec; ++nn) {
				if (y[nn] < -ip.species_failure_tolerance) {
					burn_failed = 1;
				}
			}
			for (int nn = detail::i_rad0; nn < detail::i_rad0 + NumChemRadEqs; nn += NumChemRadVarsPerGroup) {
				if (y[nn] < -ip.radiation_failure_tolerance) {
					burn_failed = 1;
				}
			}
			sys.unpack(y);

			if (std::isnan(photochemstate.xn[0]) || std::isnan(photochemstate.rho) || std::isnan(photochemstate.rn[0])) {
				amrex::Abort("Burner returned NAN");
			}

			if (burn_failed) {
				amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
			}

			// Ensure positivity
			for (double &nn : photochemstate.xn) {
				nn = amrex::max(nn, small_x);
			}
			for (int nn = 0; nn < NumChemBands; nn += 1) {
				// TODO (james471): Ensure that flux doesn't deviate from the corresponding energy density.
				photochemstate.rn[static_cast<std::size_t>(nn) * NumChemRadVarsPerGroup] =
				    amrex::max(photochemstate.rn[static_cast<std::size_t>(nn) * NumChemRadVarsPerGroup], small_x);
			}

			for (int nn = 0; nn < NumSpec; ++nn) {
				state(i, j, k, RadSystem<problem_t>::scalar0_index + nn) = photochemstate.xn[nn] * spmasses[nn];
			}
			// Update the chem-band photon number density, attenuate the flux by the burn's rn[1] factor or algebraically using the photon number
			// density if SKIP_PHOTOCHEMFLUX is defined, and accumulate the momentum the gas absorbs from the O(v/c) work term: dP = -(F_after -
			// F_before) / c^2, summed over chem bands. The absorbed photon momentum is tied to the physical photon flux F = c E; the reduced speed
			// of light only changes the absorption rate used by the chemistry solve. dMom is computed here unconditionally (cheap) and only applied
			// when do_vc_work is true -- computing it outside the `if constexpr (do_vc_work)` block avoids NVCC's "cannot first-capture in
			// constexpr-if" restriction.
			const Real inv_c2 = 1.0_rt / (C::c_light * C::c_light);
			Real dMomX = 0.0_rt;
			Real dMomY = 0.0_rt;
			Real dMomZ = 0.0_rt;
			// per-band momentum-kick magnitudes, used only to split the optional work-term energy debit across bands
			[[maybe_unused]] amrex::GpuArray<Real, NumChemBands> dMomMag{};
			[[maybe_unused]] Real dMomMagSum = 0.0_rt;
			for (int nn = 0; nn < NumChemBands; ++nn) {
				const int fxIdx = firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn;
				const int fyIdx = firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn;
				const int fzIdx = firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn;
				const Real n_gamma_final = photochemstate.rn[0 + NumChemRadVarsPerGroup * nn];
				state(i, j, k, firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn) = n_gamma_final * chemBandQuanta[nn];
				const Real FxOld = state(i, j, k, fxIdx);
				const Real FyOld = state(i, j, k, fyIdx);
				const Real FzOld = state(i, j, k, fzIdx);
#ifdef SKIP_PHOTOCHEMFLUX
				// flux is not carried in photochemstate; derive the attenuation factor from the change in n_gamma
				const Real flux_factor = (n_gamma_initial[nn] > 0.0_rt) ? (n_gamma_final / n_gamma_initial[nn]) : 0.0_rt;
#else
				const Real flux_factor = photochemstate.rn[1 + NumChemRadVarsPerGroup * nn];
#endif
				// dp = (F_before - F_after) / c^2 = (1 - flux_factor) * F_before / c^2
				const Real dpx = (1.0_rt - flux_factor) * FxOld * inv_c2;
				const Real dpy = (1.0_rt - flux_factor) * FyOld * inv_c2;
				const Real dpz = (1.0_rt - flux_factor) * FzOld * inv_c2;
				dMomX += dpx;
				dMomY += dpy;
				dMomZ += dpz;
				if constexpr (debit_work_term_from_radiation) {
					dMomMag[nn] = std::sqrt(dpx * dpx + dpy * dpy + dpz * dpz);
					dMomMagSum += dMomMag[nn];
				}
				state(i, j, k, fxIdx) = flux_factor * FxOld;
				state(i, j, k, fyIdx) = flux_factor * FyOld;
				state(i, j, k, fzIdx) = flux_factor * FzOld;
			}

			if constexpr (RadSystem<problem_t>::thermal_band_photochemistry_) {
				// Flux of thermal band is unchanged after burn since we assume isotropic emission.
				for (int nn = 0; nn < NumThermalBands; ++nn) {
					radEnergySource_arr(i, j, k, nn) += (photochemstate.re_thermal[nn] - re_thermal_initial[nn]) / dt;
				}
			}

#ifdef DUST_CHEMICAL_BAND_ABSORPTION
			static_assert(RadSystem<problem_t>::dust_chemical_band_absorption_,
				      "DUST_CHEMICAL_BAND_ABSORPTION is defined but ISM_Traits::dust_chemical_band_absorption is false: "
				      "the dust-absorbed photon energy would be silently dropped.");
			dustHeatingSource_arr(i, j, k) = photochemstate.e_dust_absorbed / dt;
#endif

			// Quokka uses rho*eint
			const Real dEint = (photochemstate.e * photochemstate.rho) - Eint;
			state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index) += dEint;
			state(i, j, k, RadSystem<problem_t>::gasEnergy_index) += dEint;

			// O(v/c) radiation-pressure work term: apply the absorbed photon momentum (dMom, computed above) to the
			// gas. This changes the kinetic energy of the updated momentum only; the auxiliary internal energy is
			// unaffected. Gated on beta_order>=1 && hydro, so beta_order==0 problems are untouched.
			if constexpr ((RadSystem_Traits<problem_t>::beta_order >= 1) && Physics_Traits<problem_t>::is_hydro_enabled) {
				const Real xmom_new = xmom + dMomX;
				const Real ymom_new = ymom + dMomY;
				const Real zmom_new = zmom + dMomZ;
				state(i, j, k, RadSystem<problem_t>::x1GasMomentum_index) = xmom_new;
				state(i, j, k, RadSystem<problem_t>::x2GasMomentum_index) = ymom_new;
				state(i, j, k, RadSystem<problem_t>::x3GasMomentum_index) = zmom_new;

				// Recompute total gas energy from the (unchanged) internal energy and the updated momentum so the
				// absorbed momentum appears only as kinetic energy.
				state(i, j, k, RadSystem<problem_t>::gasEnergy_index) = ::quokka::EOS<problem_t>::ComputeEgasFromEint(
				    rho, xmom_new, ymom_new, zmom_new, state(i, j, k, RadSystem<problem_t>::gasInternalEnergy_index), Emag);

				// Optionally enforce strict gas+radiation energy conservation: remove the work-term energy
				// (c / chat) * dEkin from the chem-band photon field, split across bands in proportion to each band's
				// momentum contribution, and rescale each band's flux by the same factor its energy density changes so
				// the reduced flux |F| / (c E) is preserved (keeps the M1 closure F / c <= E from being violated by the
				// debit). Disabled by default: conserving the ionizing photon number density is more important than
				// strict energy conservation for correct ionization-front propagation (see debit_work_term_from_radiation).
				if constexpr (debit_work_term_from_radiation) {
					if (dMomMagSum > 0.0_rt) {
						const Real dEkin = (xmom_new * xmom_new + ymom_new * ymom_new + zmom_new * zmom_new -
								    (xmom * xmom + ymom * ymom + zmom * zmom)) /
								   (2.0_rt * rho);
						const Real E_debit = (C::c_light / photochemstate.c_hat) * dEkin;
						for (int nn = 0; nn < NumChemBands; ++nn) {
							const int eIdx = firstChemIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const int fxIdx = firstChemFxIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const int fyIdx = firstChemFyIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const int fzIdx = firstChemFzIndex + Physics_NumVars::numRadVarsPerGroup * nn;
							const Real E_old = state(i, j, k, eIdx); // > 0: rn[0] was clamped to small_x above
							const Real E_new =
							    amrex::max(E_old - E_debit * (dMomMag[nn] / dMomMagSum), small_x * chemBandQuanta[nn]);
							state(i, j, k, eIdx) = E_new;
							const Real flux_rescale = E_new / E_old;
							state(i, j, k, fxIdx) *= flux_rescale;
							state(i, j, k, fyIdx) *= flux_rescale;
							state(i, j, k, fzIdx) *= flux_rescale;
						}
					}
				}
			}
		});

#ifdef AMREX_USE_HIP
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
	}

	num_failed = *(d_num_failed.copyToHost());

	photochem_burn_success = num_failed == 0;
	amrex::ParallelDescriptor::ReduceIntMin(photochem_burn_success);

	if (!photochem_burn_success) {
		amrex::Abort("Photochemistry solve failed in the Rosenbrock integrator. Aborting.");
	}

	return photochem_burn_success;
}

} // namespace quokka::photochemistry
#endif

#endif
