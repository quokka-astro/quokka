#ifndef RADIATION_BASE_HPP_
#define RADIATION_BASE_HPP_

#include <array>
#include <cmath>

// library headers
#include "AMReX.H" // IWYU pragma: keep
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"
// #include "AMReX_IParser_Y.H"
// #include "AMReX_IntVect.H"
#include "AMReX_REAL.H"

// internal headers
#include "hydro/EOS.hpp"
#include "hyperbolic_system.hpp"
#include "math/math_impl.hpp"
#include "physics_info.hpp"
#include "radiation/planck_integral.hpp"
#include "util/valarray.hpp"

using Real = amrex::Real;

// Hyper parameters for the radiation solver
static constexpr bool include_delta_B = true;
static constexpr bool use_diffuse_flux_mean_opacity = true;
static constexpr bool special_edge_bin_slopes = false;	    // Use 2 and -4 as the slopes for the first and last bins, respectively
static constexpr bool force_rad_floor_in_iteration = false; // force radiation energy density to be positive (and above the floor value) in the Newton iteration
static constexpr bool include_work_term_in_source = true;

static const int max_ite_to_update_alpha_E = 5; // Apply to the PPL_opacity_full_spectrum only. Only update alpha_E for the first max_ite_to_update_alpha_E
// iterations of the Newton iteration
static constexpr bool enable_dE_constrain = true;
static constexpr bool use_D_as_base = false;
static const bool PPL_free_slope_st_total = false; // PPL with free slopes for all, but subject to the constraint sum_g alpha_g B_g = - sum_g B_g. Not working
						   // well -- Newton iteration convergence issue.

// Time integration scheme
// IMEX PD-ARS
static constexpr double IMEX_a22 = 1.0;
static constexpr double IMEX_a32 = 0.5; // 0 < IMEX_a32 <= 0.5
// SSP-RK2 + implicit radiation-matter exchange
// static constexpr double IMEX_a22 = 0.0;
// static constexpr double IMEX_a32 = 0.0;

// physical constants in CGS units
static constexpr double c_light_cgs_ = C::c_light;	    // cgs
static constexpr double radiation_constant_cgs_ = C::a_rad; // cgs
static constexpr double inf = std::numeric_limits<double>::max();

// enum for opacity_model
enum class OpacityModel {
	single_group = 0, // user-defined opacity for each group, given as a function of density and temperature.
	piecewise_constant_opacity,
	PPL_opacity_fixed_slope_spectrum,
	PPL_opacity_full_spectrum // piecewise power-law opacity model with piecewise power-law fitting to a user-defined opacity function and on-the-fly
				  // piecewise power-law fitting to radiation energy density and flux.
};

// this struct is specialized by the user application code
//
template <typename problem_t> struct RadSystem_Traits {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = radiation_constant_cgs_;
	static constexpr double Erad_floor = 0.;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups + 1> radBoundaries = {0., inf};
	static constexpr double beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::single_group;
};

// A struct to hold the results of the ComputeRadPressure function.
struct RadPressureResult {
	quokka::valarray<double, 4> F; // components of radiation pressure tensor
	double S;		       // maximum wavespeed for the radiation system
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod_func(double a, double b) -> double
{
	return 0.5 * (sgn(a) + sgn(b)) * std::min(std::abs(a), std::abs(b));
}

// Use SFINAE (Substitution Failure Is Not An Error) to check if opacity_model is defined in RadSystem_Traits<problem_t>
template <typename problem_t, typename = void> struct RadSystem_Has_Opacity_Model : std::false_type {
};

template <typename problem_t>
struct RadSystem_Has_Opacity_Model<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::opacity_model)>> : std::true_type {
};

/// Class for the radiation moment equations
///
template <typename problem_t> class RadSystem : public HyperbolicSystem<problem_t>
{
      public:
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(0.5 * std::abs(a + b), std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr int numRadVars_ = Physics_NumVars::numRadVars;				 // number of radiation variables for each photon group
	static constexpr int nvarHyperbolic_ = numRadVars_ * Physics_Traits<problem_t>::nGroups; // total number of radiation variables
	static constexpr int nstartHyperbolic_ = Physics_Indices<problem_t>::radFirstIndex;
	static constexpr int nvar_ = nstartHyperbolic_ + nvarHyperbolic_;

	enum gasVarIndex {
		gasDensity_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1GasMomentum_index,
		x2GasMomentum_index,
		x3GasMomentum_index,
		gasEnergy_index,
		gasInternalEnergy_index,
		scalar0_index
	};

	enum radVarIndex { radEnergy_index = nstartHyperbolic_, x1RadFlux_index, x2RadFlux_index, x3RadFlux_index };

	enum primVarIndex {
		primRadEnergy_index = 0,
		x1ReducedFlux_index,
		x2ReducedFlux_index,
		x3ReducedFlux_index,
	};

	// C++ standard does not allow constexpr to be uninitialized, even in a
	// templated class!
	static constexpr double c_light_ = RadSystem_Traits<problem_t>::c_light;
	static constexpr double c_hat_ = RadSystem_Traits<problem_t>::c_hat;
	static constexpr double radiation_constant_ = RadSystem_Traits<problem_t>::radiation_constant;

	static constexpr int beta_order_ = RadSystem_Traits<problem_t>::beta_order;

	static constexpr bool enable_dust_gas_thermal_coupling_model_ = RadSystem_Traits<problem_t>::enable_dust_gas_thermal_coupling_model;

	static constexpr int nGroups_ = Physics_Traits<problem_t>::nGroups;
	static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries_ = []() constexpr {
		if constexpr (nGroups_ > 1) {
			return RadSystem_Traits<problem_t>::radBoundaries;
		} else {
			amrex::GpuArray<double, 2> boundaries{0., inf};
			return boundaries;
		}
	}();

	static constexpr double Erad_floor_ = RadSystem_Traits<problem_t>::Erad_floor / nGroups_;

	static constexpr OpacityModel opacity_model_ = []() constexpr {
		if constexpr (RadSystem_Has_Opacity_Model<problem_t>::value) {
			return RadSystem_Traits<problem_t>::opacity_model;
		} else {
			return OpacityModel::single_group;
		}
	}();

	// Assertion: has to use single_group when nGroups_ == 1
	static_assert(((nGroups_ > 1 && opacity_model_ != OpacityModel::single_group) || (nGroups_ == 1 && opacity_model_ == OpacityModel::single_group)),
		      "OpacityModel::single_group MUST be used when nGroups_ == 1. If nGroups_ > 1, you MUST set opacity_model."); // NOLINT

	// Assertion: PPL_opacity_full_spectrum requires at least 3 photon groups
	static_assert(!(nGroups_ < 3 && opacity_model_ == OpacityModel::PPL_opacity_full_spectrum), // NOLINT
		      "PPL_opacity_full_spectrum requires at least 3 photon groups.");

	static constexpr double mean_molecular_mass_ = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
	static constexpr double boltzmann_constant_ = quokka::EOS_Traits<problem_t>::boltzmann_constant;
	static constexpr double gamma_ = quokka::EOS_Traits<problem_t>::gamma;

	// static functions

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange);
	static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange);

	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);

	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR>
	static void ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
				  amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
				  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool use_wavespeed_correction);

	static void SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
				       amrex::Real time);

	static void AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt, int stage,
					     double dustGasCoeff, int *p_iteration_counter, int *num_failed_coupling, int *num_failed_dust,
					     int *p_num_failed_outer_ite);

	static void AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt, int stage,
					      double dustGasCoeff, int *p_iteration_counter, int *num_failed_coupling, int *num_failed_dust,
					      int *p_num_failed_outer_ite);

	static void balanceMatterRadiation(arrayconst_t &consPrev, array_t &consNew, amrex::Box const &indexRange);

	// Use an additionalr template for ComputeMassScalars as the Array type is not always the same
	template <typename ArrayType>
	AMREX_GPU_DEVICE static auto ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeEddingtonFactor(double f) -> double;

	// Used for single-group RHD only. Not used for multi-group RHD.
	AMREX_GPU_HOST_DEVICE static auto ComputePlanckOpacity(double rho, double Tgas) -> Real;
	AMREX_GPU_HOST_DEVICE static auto ComputeFluxMeanOpacity(double rho, double Tgas) -> Real;
	AMREX_GPU_HOST_DEVICE static auto ComputeEnergyMeanOpacity(double rho, double Tgas) -> Real;

	// For multi-group RHD, use DefineOpacityExponentsAndLowerValues to define the opacities.
	AMREX_GPU_HOST_DEVICE static auto DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double rho,
									       double Tgas) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>;

	AMREX_GPU_HOST_DEVICE static auto ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
								  amrex::GpuArray<double, nGroups_> const &radBoundaryRatios,
								  amrex::GpuArray<double, nGroups_> const &alpha_quant) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto
	ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
				amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value) -> quokka::valarray<double, nGroups_>;
	// AMREX_GPU_HOST_DEVICE static auto
	// ComputeGroupMeanOpacityWithMinusOneSlope(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value,
	// 					 amrex::GpuArray<double, nGroups_> radBoundaryRatios) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeEintFromEgas(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Etot) -> double;
	AMREX_GPU_HOST_DEVICE static auto ComputeEgasFromEint(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Eint) -> double;
	AMREX_GPU_HOST_DEVICE static auto PlanckFunction(double nu, double T) -> double;
	AMREX_GPU_HOST_DEVICE static auto
	ComputeDiffusionFluxMeanOpacity(quokka::valarray<double, nGroups_> kappaPVec, quokka::valarray<double, nGroups_> kappaEVec,
					quokka::valarray<double, nGroups_> fourPiBoverC, amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
					amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge,
					amrex::GpuArray<double, nGroups_ + 1> kappa_slope) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeFluxInDiffusionLimit(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double T,
								      double vel) -> amrex::GpuArray<double, nGroups_>;

	template <typename ArrayType>
	AMREX_GPU_HOST_DEVICE static auto
	ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> amrex::GpuArray<double, nGroups_>;

	AMREX_GPU_HOST_DEVICE static void SolveLinearEqs(double a00, const quokka::valarray<double, nGroups_> &a0i,
							 const quokka::valarray<double, nGroups_> &ai0, const quokka::valarray<double, nGroups_> &aii,
							 const double &y0, const quokka::valarray<double, nGroups_> &yi, double &x0,
							 quokka::valarray<double, nGroups_> &xi);

	AMREX_GPU_HOST_DEVICE static auto Solve3x3matrix(double C00, double C01, double C02, double C10, double C11, double C12, double C20, double C21,
							 double C22, double Y0, double Y1, double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>;

	AMREX_GPU_HOST_DEVICE static auto ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries,
								       amrex::Real temperature) -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
	    -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> double;

	AMREX_GPU_HOST_DEVICE static auto
	ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
							amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real;

	AMREX_GPU_HOST_DEVICE static auto
	ComputeDustTemperature(double T_gas, double T_d_init, double rho, quokka::valarray<double, nGroups_> const &Erad, double dustGasCoeff,
			       amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries = amrex::GpuArray<double, nGroups_ + 1>{},
			       amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios = amrex::GpuArray<double, nGroups_>{}) -> double;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto
	ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k,
				const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries) -> quokka::valarray<double, nGroups_>;

	AMREX_GPU_DEVICE static auto isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool;

	AMREX_GPU_DEVICE static void amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons);

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeRadPressure(double erad_L, double Fx_L, double Fy_L, double Fz_L, double fx_L, double fy_L,
							double fz_L) -> RadPressureResult;

	AMREX_GPU_DEVICE static auto ComputeEddingtonTensor(double fx_L, double fy_L, double fz_L) -> std::array<std::array<double, 3>, 3>;
};

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>
{
	amrex::GpuArray<Real, nmscalars_> massScalars;
	for (int n = 0; n < nmscalars_; ++n) {
		massScalars[n] = arr(i, j, k, scalar0_index + n);
	}
	return massScalars;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEintFromEgas(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
								     const double Etot) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Eint = Etot - Ekin;
	AMREX_ASSERT_WITH_MESSAGE(Eint > 0., "Gas internal energy is not positive!");
	return Eint;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEgasFromEint(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
								     const double Eint) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Etot = Eint + Ekin;
	return Etot;
}

#endif // RADIATION_BASE_HPP_