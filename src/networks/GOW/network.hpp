#ifndef JAFF_NETWORK_HPP_
#define JAFF_NETWORK_HPP_
/// \file network.hpp
/// \brief A jaff-generated network: its constants, runtime parameters,
/// right-hand side and Jacobian. A Quokka problem names the network type with
/// Microphysics_Traits::Network.

#include <cmath> // INFINITY, for an open top band edge
#include <string_view>

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "microphysics/ode_system.hpp"

// The table lookups the expressions call (functions named *interp*, and their
// derivatives *_partial_N) and LoadTables(), which reads the tables, are
// written by hand in tables.hpp next to this file, in the network's namespace.
// $$JAFF SUB ninterp
#if 9 > 0
#include "tables.hpp"
#endif
// $$JAFF END

/// The jaff network's own namespace, named after the network, so that several
/// jaff networks can be linked into one executable.
// $$JAFF SUB label
namespace quokka::networks::GOW
{
// $$JAFF END

// The runtime parameters of Network, read from the `network` prefix. They are
// defined in network.cpp, not inline here: HIP cannot register inline managed
// variables.

/// floor of the species and radiation after a burn, and of the mass scalars
extern AMREX_GPU_MANAGED amrex::Real small_x; // NOLINT
// The free scalars of the network's expressions, each read from network.<name>.
// $$JAFF REPEAT idx, parameter IN parameters
extern AMREX_GPU_MANAGED amrex::Real Zd;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real crate; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real d2g;   // NOLINT
extern AMREX_GPU_MANAGED amrex::Real gradv; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real vdisp; // NOLINT
// $$JAFF END

// Network symbols mapped onto the burn state, shared by the RHS and the
// Jacobian. jaff's energy row is the volumetric dE/dt; Quokka integrates the
// specific internal energy e = E / rho, so the functions below divide that row
// by state.rho. jaff emits a density and a flux moment per band, interleaved in
// state.rn, so the problem must integrate the photon flux. The column densities
// ncol_<species> are constant during a burn: the problem sets state.ncol
// before it, and the Jacobian has no derivative with respect to them.
// $$JAFF DEFINE quokka {
// REPLACE {
// "radeden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
// "photden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
// "rflux\[\s*(\d+)\s*\]" -> "state.rn[2*\1+1]"
// "nden\[\s*(\d+)\s*\]" -> "state.xn[\1]"
// "\bncol_(\w+)" -> "state.ncol[ncol_\1]"
// "tgas" -> "state.T"
// "c_hat" -> "state.c_hat"
// }
// THERMAL { MODE -> dedt  ENERGY -> volumetric }
// RADIATION { ENABLED -> True }
// }

// $$JAFF SUB label
/// The jaff network GOW.
// $$JAFF END
struct Network {
	// $$JAFF SUB nspec
	static constexpr int NumSpec = 18;
	// $$JAFF END
	/// indices of the species in xn
	enum Species : int { // NOLINT(cppcoreguidelines-use-enum-class)
		// $$JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign
		h = 0,
		hj = 1,
		ek = 2,
		h2 = 3,
		h2j = 4,
		he = 5,
		hej = 6,
		c = 7,
		cj = 8,
		co = 9,
		hcoj = 10,
		o = 11,
		si = 12,
		sij = 13,
		chx = 14,
		ohx = 15,
		h3j = 16,
		oj = 17,
		// $$JAFF END
	};

	/// mass of each species [g]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesMass() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		// $$JAFF REPEAT specie_mass IN specie_masses
		return {{1.673773e-24, 1.6728620616289998e-24, 9.10938371e-28, 3.347546e-24, 3.3466350616289996e-24, 6.646473e-24, 6.645562061629e-24,
			 1.9944730000000003e-23, 1.9943819061629003e-23, 4.651236000000001e-23, 4.8185222061629005e-23, 2.6567630000000005e-23, 4.663707e-23,
			 4.6636159061629004e-23, 2.1618503000000003e-23, 2.8241403000000005e-23, 5.020408061629e-24, 2.656671906162901e-23}};
		// $$JAFF END
	}
	/// adiabatic index of each species, from the network's [network.eos]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesGamma() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		// $$JAFF REPEAT specie_gamma IN specie_gammas
		return {{1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.4, 1.4, 1.6666666666666667, 1.6666666666666667, 1.6666666666666667,
			 1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.6666666666666667,
			 1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.6666666666666667}};
		// $$JAFF END
	}
	/// the equation of state the network assumes, as jaff names it (the type of
	/// [network.eos]); Quokka maps it to one of its EOSs
	// $$JAFF SUB eos_type
	static constexpr std::string_view EOSType = "multi_gamma";
	// $$JAFF END

	// $$JAFF SUB nbands
	static constexpr int NumChemBands = 2;
	/// actual_jac and actual_rhs_and_jac write every entry of the matrix (postprocess_network.py)
	static constexpr bool jacobian_writes_all_entries = true;
	// $$JAFF END
	/// edges of the chemical bands [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1>
	{
		// $$JAFF REPEAT radiation_boundary IN radiation_boundaries
		return {{6.0, 11.2, 13.6}};
		// $$JAFF END
	}
	/// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a
	/// chemical band; Quokka takes one alpha for all bands
	// $$JAFF GET profile_index IN 0
	static constexpr double ChemBandsPowerLawIndex = 0;
	// $$JAFF END
	// $$JAFF REPEAT idx, profile_index IN profile_indices
	static_assert(0 == ChemBandsPowerLawIndex, "band 0: Quokka takes one profile index for all chemical bands");
	static_assert(0 == ChemBandsPowerLawIndex, "band 1: Quokka takes one profile index for all chemical bands");
	// $$JAFF END

	static constexpr int NumThermalBands = 0;

	// $$JAFF SUB ncolumns
	static constexpr int NumColumns = 4;
	// $$JAFF END
	/// index in the burn state's ncol of each column density [cm^-2] the
	/// expressions read
	enum Column : int { // NOLINT(cppcoreguidelines-use-enum-class)
		// $$JAFF REPEAT idx, column_density IN column_densities
		ncol_C = 0,
		ncol_CO = 1,
		ncol_H = 2,
		ncol_H2 = 3,
		// $$JAFF END
	};

	/// floor of the species and radiation after a burn, and of the mass scalars
	/// (small_x)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SmallX() -> amrex::Real { return small_x; }

	/// Read the runtime parameters.
	static void ReadParameters();

	/// The right-hand side on Quokka's burn state: d/dt of the species, the
	/// specific internal energy and the radiation.
	template <typename BurnT>
	AMREX_GPU_HOST_DEVICE static void
	actual_rhs(const BurnT &state, amrex::GpuArray<amrex::Real, quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>::neqs> &ydot)
	{
		static_assert(BurnT::traits::integrate_rad_flux, "a jaff network needs Microphysics_Traits::integrate_rad_flux");
		// $$JAFF REPEAT idx, rhs, cse IN rhses { USE quokka }
		const amrex::Real lnT = std::log(state.T);
		const amrex::Real cse0 = 6.3999999999999996e-10 * state.xn[0] * state.xn[4];
		const amrex::Real cse1 = -cse0;
		const amrex::Real cse2 = std::sqrt(state.T);
		const amrex::Real cse3 = cse2 / state.xn[2];
		const amrex::Real cse4 = 0.00011320754716981131 * cse3 * state.xn[3];
		const amrex::Real cse5 = std::min(10000000000.0, cse4);
		const amrex::Real cse6 = 1.0 / (cse5 + 1);
		const amrex::Real cse7 = cse6 * state.xn[17] * state.xn[3];
		const amrex::Real cse8 = 1.6000000000000001e-9 * cse7;
		const amrex::Real cse9 = cse5 * cse8;
		const amrex::Real cse10 = 1.99e-9 * std::exp((-0.19) * lnT) * state.xn[11] * state.xn[16];
		const amrex::Real cse11 = cse10 * cse6;
		const amrex::Real cse12 = state.c_hat * state.rn[2 * 0];
		const amrex::Real cse13 = 1.7141251340095377e-18 * cse12;
		const amrex::Real cse14 = state.c_hat * state.rn[2 * 1];
		const amrex::Real cse15 = 4.1730320353928637e-18 * cse14;
		const amrex::Real cse16 = (cse13 + cse15) * state.xn[15];
		const amrex::Real cse17 = 3.5000000000000002e-11 * state.xn[11] * state.xn[15];
		const amrex::Real cse18 = 1.0 / state.T;
		const amrex::Real cse19 = 7.9513696852843427e-10 * std::exp((-0.33900000000000002) * lnT) * std::exp(0.11 * cse18) * state.xn[15] * state.xn[7];
		const amrex::Real cse20 = M_LN10;
		const amrex::Real cse21 = 1.0 / cse20;
		const amrex::Real cse22 = 2 * state.xn[3];
		const amrex::Real cse23 = cse22 + state.xn[0] + state.xn[10] + state.xn[14] + state.xn[15] + 3 * state.xn[16] + state.xn[1] + 2 * state.xn[4];
		const amrex::Real cse24 = std::log((1.0 / 10000.0) * state.T);
		const amrex::Real cse25 = (1.0 / (cse20 * cse20));
		const amrex::Real cse26 = (cse24 * cse24) * cse25;
		const amrex::Real cse27 = 1.0 / cse23;
		const amrex::Real cse28 = cse27 * state.xn[3];
		const amrex::Real cse29 = cse27 * state.xn[0];
		const amrex::Real cse30 = std::exp(2.302585092994046 * (0.41599999999999998 * cse21 * cse24 + 0.32700000000000001 * cse26 - 3)) * cse29 +
					  std::exp(2.302585092994046 * (1.3 * cse21 * cse24 - 1.6200000000000001 * cse26 - 4.8449999999999998)) * cse28;
		const amrex::Real cse31 = cse23 * ((cse30 < 1.0e-50) ? (1.0e-50) : (cse30));
		const amrex::Real cse32 = cse21 / (cse31 + 1);
		const amrex::Real cse33 = cse31 * cse32;
		const amrex::Real cse34 = state.T > 700.0;
		const amrex::Real cse35 =
		    ((cse34) ? (std::exp(2.302585092994046 * (cse32 * std::log(5.9959999999999997e-30 * std::exp((4.1881000000000004) * lnT) *
									       std::pow(6.7610000000000002e-6 * state.T + 1, -5.6881000000000004) *
									       std::exp(-54657.400000000001 * cse18)) +
							      cse33 * std::log(1.3000000000000001e-9 * std::exp(-53300 * cse18)))))
			     : (0.0)) *
		    ((state.xn[3]) * (state.xn[3]));
		const amrex::Real cse36 = std::max(1.0e-35, 2.0000000000000002e-15 * state.ncol[ncol_H2]);
		const amrex::Real cse37 = 0.035000000000000003 * (1.0 / std::sqrt(cse36 + 1)) * std::exp(-0.00084999999999999995 * std::sqrt(cse36 + 1.0)) +
					  0.96499999999999997 * (1.0 / ((cse36 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0) *
									(cse36 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0)));
		const amrex::Real cse38 = cse12 * cse37;
		const amrex::Real cse39 = cse14 * cse37;
		const amrex::Real cse40 = 1.2216858566394234e-27 * cse38 + 3.6749360303422164e-18 * cse39;
		const amrex::Real cse41 = (7.5e-10 * std::exp((-0.45800000000000002) * lnT) + 4.9899999999999997e-11 * std::exp((0.40500000000000003) * lnT)) *
					  state.xn[0] * state.xn[17];
		const amrex::Real cse42 =
		    ((cse34) ? (std::exp(2.302585092994046 * (cse32 * std::log(6.6699999999999996e-12 * cse2 * std::exp(-63590 * cse18 - 1)) +
							      cse33 * std::log(3.5199999999999998e-9 * std::exp(-43900 * cse18)))))
			     : (0.0)) *
		    state.xn[0];
		const amrex::Real cse43 = std::exp((-0.52000000000000002) * lnT);
		const amrex::Real cse44 = cse43 * state.xn[16] * state.xn[2];
		const amrex::Real cse45 = cse23 * d2g;
		const amrex::Real cse46 =
		    (4.0000000000000001e-10 * std::exp((0.0066899999999999998) * lnT) + 1.1000000000000001e-11 * std::exp((0.51700000000000002) * lnT)) *
		    std::exp(-227 * cse18) * state.xn[11] * state.xn[1];
		const amrex::Real cse47 = std::exp((-1.3) * lnT);
		const amrex::Real cse48 = std::exp(-23 * cse18);
		const amrex::Real cse49 = cse47 * cse48 * state.xn[3] * state.xn[8];
		const amrex::Real cse50 = 4.2188032355736613e-18 * cse12;
		const amrex::Real cse51 = 1.168047934563505e-18 * cse14;
		const amrex::Real cse52 = (cse50 + cse51) * state.xn[14];
		const amrex::Real cse53 = 7.0000000000000004e-11 * state.xn[11] * state.xn[14];
		const amrex::Real cse54 = cse52 + cse53;
		const amrex::Real cse55 = (26.218500000000002 * std::sqrt(3) * std::sqrt(cse18) + 0.62) * state.xn[15];
		const amrex::Real cse56 = 1.3500000000000001e-9 * cse55 * state.xn[6];
		const amrex::Real cse57 = state.xn[3] * state.xn[6];
		const amrex::Real cse58 = 1.2599999999999999e-13 * cse57 * std::exp(-22.5 * cse18);
		const amrex::Real cse59 = cse56 + cse58;
		const amrex::Real cse60 = 2.8142984982406731e-11 * std::exp((0.26000000000000001) * lnT) * state.xn[0] * state.xn[14];
		const amrex::Real cse61 =
		    1.7599999999999999e-9 * std::exp((0.042000000000000003) * lnT) * std::exp(-2.145922746781116e-5 * state.T) * state.xn[3] * state.xn[4];
		const amrex::Real cse62 = -cse60 + cse61;
		const amrex::Real cse63 = crate * state.xn[9];
		const amrex::Real cse64 = 6.5199999999999996 * cse63;
		const amrex::Real cse65 = 2.2999999999999998 * cse28 + 1.5 * cse29;
		const amrex::Real cse66 = crate * cse65;
		const amrex::Real cse67 = cse66 * state.xn[0];
		const amrex::Real cse68 = std::log(8.6173000000000003e-5 * state.T);
		const amrex::Real cse69 = state.xn[0] * state.xn[2];
		const amrex::Real cse70 =
		    cse69 *
		    ((cse34)
			 ? (5.9078447946382801e-70 * std::exp((13.536555999999999) * lnT) *
			    std::exp(-2.0391498499999999e-6 * (cse68 * cse68 * cse68 * cse68 * cse68 * cse68 * cse68 * cse68) +
				     0.000111954395 * (cse68 * cse68 * cse68 * cse68 * cse68 * cse68 * cse68) -
				     0.00263197617 * (cse68 * cse68 * cse68 * cse68 * cse68 * cse68) +
				     0.034825597700000002 * (cse68 * cse68 * cse68 * cse68 * cse68) - 0.28770560000000001 * (cse68 * cse68 * cse68 * cse68) +
				     1.56315498 * (cse68 * cse68 * cse68) - 5.7393287500000003 * (cse68 * cse68)))
			 : (0.0));
		const amrex::Real cse71 = 1.05849862170431e-5 * std::exp((-0.64000000000000001) * lnT) * state.xn[10] * state.xn[2];
		const amrex::Real cse72 = 4.8813573300130879e-6 * (cse18 * std::sqrt(cse18)) *
					  std::pow(114.81344434488582 * std::pow(cse18, 0.40699999999999997) + 1, -2.242) * state.xn[1] * state.xn[2];
		const amrex::Real cse73 = 144.3803103740824 * state.rn[2 * 0] + 220.5700827961663 * state.rn[2 * 1];
		const amrex::Real cse74 = state.c_hat * cse73;
		const amrex::Real cse75 = cse3 * cse74;
		const amrex::Real ln_cse75 = std::log(cse75);
		const amrex::Real cse76 = 5.670589618368584e-11 * cse75;
		const amrex::Real ln_cse76 = std::log(cse76);
		const amrex::Real cse77 = std::log(state.T);
		const amrex::Real cse78 =
		    1.2250000000000001e-11 * cse45 * state.xn[1] /
		    (6.1318063394789369e-20 * std::exp((1.3779999999999999) * ln_cse75) *
			 (508.69999999999999 * std::exp((-1.102e-5 * cse77 - 0.4723) * ln_cse76) * std::exp((0.015859999999999999) * lnT) + 1) +
		     1);
		const amrex::Real cse79 = -cse64 - cse67 - cse70 + cse71 + cse72 + cse78;
		const amrex::Real cse80 = cse41 - cse46;
		const amrex::Real cse81 = 1.9917698086432708e-17 * cse12;
		const amrex::Real cse82 = 1.9088222220940885e-17 * cse14;
		const amrex::Real cse83 = (cse81 + cse82) * state.xn[12];
		const amrex::Real cse84 = cse22 * cse66;
		const amrex::Real cse85 = 1.4595000290776864e-10 * std::exp((-0.62) * lnT) * state.xn[13] * state.xn[2];
		const amrex::Real cse86 = 2.1659999999999999e-12 * cse45 * state.xn[13] /
					  (3.5685736926834518e-27 * std::exp((1.8740000000000001) * ln_cse75) *
					       (43750.0 * std::exp((-7.538e-5 * cse77 - 0.89639999999999997) * ln_cse76) * std::exp((1.635e-6) * lnT) + 1) +
					   1);
		const amrex::Real cse87 = 16800.0 * crate * cse28 * state.xn[12];
		const amrex::Real cse88 =
		    (1.0537934368240927e-9 * std::pow(cse18, 0.00231) +
		     std::exp((-1.5) * lnT) *
			 (0.00015100000000000001 * std::exp(-8110.0 * cse18) + 1.31e-7 * std::exp(-26.600000000000001 * cse18) +
			  3.4e-8 * std::exp(-7.6200000000000001 * cse18) + 6.9699999999999997e-9 * std::exp(-1.3799999999999999 * cse18))) *
		    state.xn[16] * state.xn[7];
		const amrex::Real cse89 = cse27 * state.xn[2];
		const amrex::Real cse90 = ((cse89 < 1.0e-50) ? (10000000000.0) : (cse4));
		const amrex::Real cse91 = cse10 * cse90 / (cse90 + 1);
		const amrex::Real cse92 = cse11 + 1.3007021411393064e-6 * cse44 + cse88 + cse91;
		const amrex::Real cse93 = cse21 * cse77;
		const amrex::Real cse94 =
		    -1.1000000000000001 * crate * state.xn[5] +
		    5.5720000000000001e-12 * cse45 * state.xn[6] /
			(1.0246966183388124e-22 * std::exp((1.512) * ln_cse75) *
			     (5115.0 * std::exp((-5.4939999999999998e-7 * cse77 - 0.49559999999999998) * ln_cse76) * std::exp((3.9029999999999999e-7) * lnT) +
			      1) +
			 1) +
		    9.9999999999999994e-12 * (1.0 / std::sqrt(state.T)) *
			(cse93 * (cse93 * (0.044330000000000001 * cse93 - 0.28520000000000001) - 1.6759999999999999) + 11.19) * state.xn[2] * state.xn[6];
		const amrex::Real cse95 = crate * state.xn[7];
		const amrex::Real cse96 = 3.8500000000000001 * cse95;
		const amrex::Real cse97 = 0.15970000000000001 * std::exp(-49550.0 * cse18);
		const amrex::Real cse98 = ((1.6339999999999999e-6 * std::exp(-15230.0 * cse18) + 9.7930000000000004e-9 * std::exp(-73.799999999999997 * cse18) +
					    6.3460000000000002e-9 * std::exp(-12.17 * cse18)) /
					       (state.T * std::sqrt(state.T)) +
					   2.4460185352936309e-10 * std::pow(0.00071740367932898903 * cse2 + 1, -cse97 - 1.7848999999999999) *
					       std::pow(12.244387999457523 * cse2 + 1, cse97 - 0.21509999999999996) / cse2) *
					  state.xn[2] * state.xn[8];
		const amrex::Real cse99 =
		    4.5579999999999993e-11 * cse45 * state.xn[8] /
		    (1.6851501542381549e-14 * std::exp((1.1279999999999999) * ln_cse75) *
			 (433.10000000000002 * std::exp((-0.00013329999999999999 * cse77 - 0.81200000000000006) * ln_cse76) * std::exp((0.04845) * lnT) + 1) +
		     1);
		const amrex::Real cse100 = 1040.0 * cse28 * cse95;
		const amrex::Real cse101 =
		    1.4855039799802863e-17 * interp_8_shielding_H2(state.ncol[ncol_H2]) * interp_8_shielding_self(state.ncol[ncol_C]) * state.xn[7];
		const amrex::Real cse102 = cse101 * cse14;
		const amrex::Real cse103 = -cse100 - cse102 - cse96 + cse98 + cse99;
		const amrex::Real cse104 = 3.2999999999999996e-13 * cse49;
		const amrex::Real cse105 = cse8 + cse9;
		const amrex::Real cse106 = cse104 + cse105;
		const amrex::Real cse107 = -cse88;
		const amrex::Real cse108 = -cse91;
		const amrex::Real cse109 = 1.6999999999999999e-9 * state.xn[16] * state.xn[9];
		const amrex::Real cse110 = cse42 * state.xn[3];
		const amrex::Real cse111 = 7.2000000000000002e-15 * cse57;
		const amrex::Real cse112 = cse1 + cse111 + cse84;
		const amrex::Real cse113 = -cse61;
		const amrex::Real cse114 = 1.6000000000000001e-9 * state.xn[6] * state.xn[9];
		const amrex::Real cse115 = cse111 + cse114 + cse59 + cse94;
		const amrex::Real cse116 = cse107 + cse60;
		const amrex::Real cse117 =
		    1.6797535700197549e-17 * interp_10_shielding_H2(state.ncol[ncol_H2]) * interp_10_shielding_self(state.ncol[ncol_CO]) * state.xn[9];
		const amrex::Real cse118 = cse117 * cse14;
		const amrex::Real cse119 = 184.0 * cse28 * cse63;
		const amrex::Real cse120 = cse118 + cse119 - cse19;
		const amrex::Real cse121 = 9.1500000000000005e-10 * cse55 * state.xn[8];
		const amrex::Real cse122 = cse109 + cse64 - cse71;
		const amrex::Real cse123 = cse114 - cse53;
		const amrex::Real cse124 = cse108 + cse16 + cse17;
		const amrex::Real cse125 = cse83 - cse85 - cse86 + cse87;
		const amrex::Real cse126 = std::exp((0.14999999999999999) * lnT) / (0.00088240812266906664 * std::exp((0.65000000000000002) * lnT) + 1);
		const amrex::Real cse127 = cse126 * state.xn[2];
		const amrex::Real cse128 = std::exp(-118400.0 * cse18) / (4.4460389837719569e-9 * cse127 + 626500000.0);
		const amrex::Real cse129 = 26.399999999999999 * std::sqrt(cse89 / (cse89 + 0.070000000000000007)) + 6.5;
		const amrex::Real cse130 = cse76 + 50;
		const amrex::Real ln_cse130 = std::log(cse130);
		const amrex::Real cse131 = std::log(5.670589618368584e-11 * cse2 * cse74 / (state.xn[2] + 1.0e-50) + 50);
		const amrex::Real cse132 = std::sqrt(cse18) * state.xn[2];
		const amrex::Real cse133 = (1.0 / 100.0) * state.T;
		const amrex::Real ln_cse133 = std::log(cse133);
		const amrex::Real cse134 = std::log(cse133);
		const amrex::Real cse135 = state.T < 500;
		const amrex::Real cse136 = std::exp((0.070000000000000007) * lnT);
		const amrex::Real cse137 =
		    4.5299999999999998e-6 * cse132 +
		    7.5799999999999997e-10 * std::exp((0.0086999999999999994 * cse134 + 0.12809999999999999) * ln_cse133) * state.xn[0] +
		    (0.75 * ((cse135) ? (1.1000000000000001e-13 * state.T + 5.3300000000000002e-10) : (3.74757785025e-10 * cse136)) +
		     0.25 * ((cse135) ? (3.3000000000000001e-13 * state.T + 4.4299999999999997e-10) : (3.8899728635600003e-10 * cse136))) *
			state.xn[3];
		const amrex::Real cse138 = cse137 * std::exp(-91.209999999999994 * cse18) / (cse137 + 2.3e-6);
		const amrex::Real cse139 = 0.001 * state.T;
		const amrex::Real cse140 = cse139 + 1;
		const amrex::Real cse141 = (1.0 / (cse140 * cse140));
		const amrex::Real cse142 = 1.0 / cse140;
		const amrex::Real cse143 =
		    1.0 / (cse27 * (cse40 + 1.9999999999999999e-7) /
			       (std::exp(2.302585092994046 * (-2.3900000000000001 * cse141 + 0.055500000000000001 * cse142 - 11.06)) * cse29 +
				std::exp(2.302585092994046 * (-2.0230000000000001 * cse141 - 3.6709999999999998 * cse142 - 11.08)) * cse28) +
			   1);
		const amrex::Real cse144 = cse21 * std::log(cse23);
		const amrex::Real cse145 = ((cse144 < 2) ? (10.0)
							 : ((cse144 < 4) ? ((3.0 / 2.0) * cse144 + 7)
									 : ((cse144 < 7) ? ((4.0 / 3.0) * cse144 + 23.0 / 3.0)
											 : ((cse144 < 10) ? ((1.0 / 3.0) * cse144 + 44.0 / 3.0) : (18.0)))));
		const amrex::Real cse146 = std::sqrt(std::sqrt(state.T));
		const amrex::Real cse147 = 1.75 * cse146 * state.xn[0] + state.xn[3] + 680.10000000000002 * state.xn[2] / cse146;
		const amrex::Real cse148 = l0_co_interp1d(state.T);
		const amrex::Real cse149 = 1.0 / cse148;
		const amrex::Real cse150 =
		    ((state.ncol[ncol_H] <= 0)
			 ? (100000.0 * state.xn[9] / std::max(gradv, 5.7445531974402559e-18 * M_SQRT2 * cse2))
			 : (1756701454.4310031 * state.ncol[ncol_CO] / std::sqrt(std::max(1.0000000000000001e-30, cse27 * state.ncol[ncol_H]))));
		const amrex::Real cse151 = 1.0 / llte_co_interp2d(state.T, cse150);
		const amrex::Real cse152 = nhalf_co_interp2d(state.T, cse150);
		const amrex::Real cse153 = 0.042999999999999997 * cse134;
		const amrex::Real cse154 = 4.34e-10 * std::exp((0.755 - 0.16 * cse134) * ln_cse133) * state.xn[0] +
					   1.08e-14 * std::exp((0.92600000000000005) * lnT) * state.xn[2] +
					   (2.2500000000000003e-12 * std::exp((0.52500000000000002 * cse134 + 1.198) * ln_cse133) +
					    5.2499999999999998e-13 * std::exp((cse153 + 0.88900000000000001) * ln_cse133)) *
					       state.xn[3];
		const amrex::Real cse155 = 3.5700000000000001e-10 * std::exp((0.41899999999999998 - 0.0030000000000000001 * cse134) * ln_cse133) * state.xn[0] +
					   5.1199999999999999e-10 * std::exp((-0.074999999999999997) * lnT) * state.xn[2] +
					   (3.7250000000000001e-11 * std::exp((0.025000000000000001 * cse134 + 0.26400000000000001) * ln_cse133) +
					    1.0274999999999999e-10 * std::exp((cse153 + 0.29599999999999999) * ln_cse133)) *
					       state.xn[3];
		const amrex::Real cse156 = (1.0 / 5.0) * std::exp(-326.60000000000002 * cse18);
		const amrex::Real cse157 = 3.1899999999999998e-10 * std::exp((0.36899999999999999 - 0.0060000000000000001 * cse134) * ln_cse133) * state.xn[0] +
					   4.8599999999999998e-10 * std::exp((-0.025999999999999999) * lnT) * state.xn[2] +
					   (4.7499999999999998e-11 * std::exp((0.041000000000000002 * cse134 + 0.20300000000000001) * ln_cse133) +
					    1.6725000000000002e-10 * std::exp((0.058000000000000003 * cse134 + 0.23699999999999999) * ln_cse133)) *
					       state.xn[3];
		const amrex::Real cse158 = cse155 + 8.9099999999999997e-5;
		const amrex::Real cse159 = cse156 * cse157;
		const amrex::Real cse160 = cse154 * cse155 * cse156 + (1.0 / 15.0) * cse154 * cse157 * std::exp(-425.40000000000003 * cse18) + cse158 * cse159;
		const amrex::Real cse161 = cse154 + 1.7499999999999998e-5;
		const amrex::Real cse162 = cse157 + 1.34e-10;
		const amrex::Real cse163 = (3.0 / 5.0) * cse155 * std::exp(-227.80000000000001 * cse18);
		const amrex::Real cse164 = cse159 * cse161 + cse161 * cse163 + cse162 * cse163;
		const amrex::Real cse165 =
		    1.0 / ((1.0 / 3.0) * cse154 * cse162 * std::exp(-98.800000000000011 * cse18) + cse158 * cse161 + cse158 * cse162 + cse160 + cse164);
		const amrex::Real cse166 = state.T < 100.0;
		const amrex::Real cse167 = state.T < 6000.0;
		const amrex::Real cse168 = state.T < 10;
		const amrex::Real cse169 = (1.0 / (cse20 * cse20 * cse20 * cse20 * cse20 * cse20 * cse20 * cse20));
		const amrex::Real cse170 = (1.0 / (cse20 * cse20 * cse20 * cse20 * cse20 * cse20 * cse20));
		const amrex::Real cse171 = (1.0 / (cse20 * cse20 * cse20 * cse20 * cse20 * cse20));
		const amrex::Real cse172 = (1.0 / (cse20 * cse20 * cse20 * cse20 * cse20));
		const amrex::Real cse173 = (1.0 / (cse20 * cse20 * cse20 * cse20));
		const amrex::Real cse174 = (1.0 / (cse20 * cse20 * cse20));
		const amrex::Real cse175 =
		    std::exp(2.302585092994046 * (55793.874041176059 * cse169 - 124963.59998693959 * cse170 + 119499.73396083503 * cse171 -
						  63669.349536898742 * cse172 + 20652.971887880416 * cse173 - 4189.7100510639848 * cse174 -
						  38.719649225812759 * cse21 + 512.9223506962187 * cse25 - 21.928795999999998)) *
		    state.xn[2];
		const amrex::Real cse176 = std::log(cse139);
		const amrex::Real cse177 = cse176 * cse21;
		const amrex::Real cse178 = cse172 * (cse176 * cse176 * cse176 * cse176 * cse176);
		const amrex::Real cse179 = cse173 * (cse176 * cse176 * cse176 * cse176);
		const amrex::Real cse180 = (cse176 * cse176 * cse176);
		const amrex::Real cse181 = cse174 * cse180;
		const amrex::Real cse182 = (cse176 * cse176) * cse25;
		const amrex::Real cse183 = state.xn[5] + state.xn[6];
		const amrex::Real cse184 =
		    std::exp(2.302585092994046 * (1.5714710999999999 * cse177 + 0.32168730000000001 * cse178 - 0.51002221000000003 * cse179 -
						  0.23619984999999999 * cse181 + 0.015391166 * cse182 - 22.089523)) *
			state.xn[1] +
		    std::exp(2.302585092994046 * (2.1892372 * cse177 + 0.19191374999999999 * cse178 - 0.16596184 * cse179 + 0.29036281000000003 * cse181 -
						  0.81520437999999995 * cse182 - 23.689236999999999)) *
			cse183 +
		    std::exp(2.302585092994046 * (0.43693353000000001 * cse174 * cse180 + 2.0943374000000001 * cse176 * cse21 - 0.033638326000000003 * cse178 -
						  0.14913216000000001 * cse179 - 0.77151435999999995 * cse182 - 23.962112000000001)) *
			state.xn[3];
		const amrex::Real cse185 = std::exp(2.302585092994046 * (37.383713 * cse177 + 3.8479610000000002 * cse178 + 20.159831000000001 * cse179 +
									 48.656103000000002 * cse181 + 58.145166000000003 * cse182 - 16.818342000000001)) *
					       state.xn[0] +
					   cse184;
		const amrex::Real cse186 = cse169 * (cse176 * cse176 * cse176 * cse176 * cse176 * cse176 * cse176 * cse176);
		const amrex::Real cse187 = cse170 * (cse176 * cse176 * cse176 * cse176 * cse176 * cse176 * cse176);
		const amrex::Real cse188 = cse171 * (cse176 * cse176 * cse176 * cse176 * cse176 * cse176);
		const amrex::Real cse189 =
		    std::exp(2.302585092994046 * (1.6802758 * cse177 - 8.8077017000000009 * cse178 - 4.7274035999999997 * cse179 + 4.0406627000000004 * cse181 +
						  0.93310621999999999 * cse182 - 6.3701156000000001 * cse186 + 6.4380698000000001 * cse187 +
						  8.9167182999999994 * cse188 - 22.921188999999998)) *
		    state.xn[2];
		const amrex::Real cse190 = state.T < 1000.0;
		const amrex::Real cse191 =
		    std::exp(2.302585092994046 * (3.5692468000000002 * cse176 * cse21 - 4.2519023000000002 * cse178 - 21.328264000000001 * cse179 -
						  27.850082 * cse181 - 11.33286 * cse182 - 24.311209000000002)) *
			state.xn[0] +
		    cse184;
		const amrex::Real cse192 =
		    std::exp(2.302585092994046 * (16.815729999999999 * cse177 + 983.67575999999997 * cse178 + 734.71650999999997 * cse179 + 343.1918 * cse181 +
						  96.743155000000002 * cse182 + 70.609154000000004 * cse186 + 364.14445999999998 * cse187 +
						  801.81246999999996 * cse188 - 21.928795999999998)) *
		    state.xn[2];
		const amrex::Real cse193 = std::exp(2.302585092994046 * (4.6450521 * cse177 + 1.5538288 * cse178 - 5.5108049000000001 * cse179 +
									 5.9369081000000001 * cse181 - 3.7209846 * cse182 - 24.311209000000002)) *
					       state.xn[0] +
					   cse184;
		const amrex::Real cse194 =
		    std::exp(2.302585092994046 * (-0.62120210234156448 * cse172 - 1.5370575959566981 * cse173 + 2.5133588394664677 * cse174 +
						  3.7525488682084647 * cse21 - 2.4768712409536775 * cse25 - 23.962112000000001)) *
			state.xn[3] +
		    std::exp(2.302585092994046 * (3.5440891133599637 * cse172 - 1.7105157385968939 * cse173 + 1.6702447513374001 * cse174 +
						  3.9225864834863131 * cse21 - 2.6171337683481011 * cse25 - 23.689236999999999)) *
			cse183 +
		    std::exp(2.302585092994046 * (5.9406293599919788 * cse172 - 5.2566362076906961 * cse173 - 1.3586848802337363 * cse174 +
						  2.8156982240432273 * cse21 + 0.049411830040524525 * cse25 - 22.089523)) *
			state.xn[1] +
		    std::exp(2.302585092994046 * (28.69470131298657 * cse172 - 56.79810800956944 * cse173 + 34.150687525022562 * cse174 +
						  8.3228160852326614 * cse21 - 11.945856385319289 * cse25 - 24.311209000000002)) *
			state.xn[0];
		const amrex::Real cse195 = ((cse167) ? (cse18) : (0.00016666666666666666));
		const amrex::Real cse196 = 1.6e-18 * std::exp(-11700.0 * cse195) + 6.7e-19 * std::exp(-5860.0 * cse195) + 3.0e-24 * std::exp(-510.0 * cse195) +
					   4.985670872372847e-33 * ((cse167) ? (std::exp((3.7599999999999998) * lnT)) : (160638152785302.53)) *
					       std::exp(-2197000.0 * ((cse167) ? ((1.0 / (state.T * state.T * state.T))) : (4.6296296296296294e-12))) /
					       (6.0142468035272636e-8 * ((cse167) ? (std::exp((2.1000000000000001) * lnT)) : (85924621.118090734)) + 1.0);
		const amrex::Real cse197 = std::log(0.001 * ((cse167) ? (state.T) : (6000.0)));
		const amrex::Real cse198 = cse172 * (cse197 * cse197 * cse197 * cse197 * cse197);
		const amrex::Real cse199 = cse173 * (cse197 * cse197 * cse197 * cse197);
		const amrex::Real cse200 = (cse197 * cse197 * cse197);
		const amrex::Real cse201 = (cse197 * cse197) * cse25;
		const amrex::Real cse202 = cse197 * cse21;
		const amrex::Real cse203 = cse174 * cse200;
		const amrex::Real cse204 = ((cse167) ? (cse166) : (false));
		const amrex::Real cse205 = cse169 * (cse197 * cse197 * cse197 * cse197 * cse197 * cse197 * cse197 * cse197);
		const amrex::Real cse206 = cse170 * (cse197 * cse197 * cse197 * cse197 * cse197 * cse197 * cse197);
		const amrex::Real cse207 = cse171 * (cse197 * cse197 * cse197 * cse197 * cse197 * cse197);
		const amrex::Real cse208 = 0.045999999999999999 * cse134;
		const amrex::Real cse209 = 1.7258000000000001e-6 * cse132;
		const amrex::Real cse210 =
		    8.9000000000000003e-11 * std::exp((cse208 + 0.22800000000000001) * ln_cse133) * state.xn[0] +
		    cse209 * std::exp(((cse190) ? (cse77 * (cse77 * (cse77 * (0.0070527699999999999 * cse77 - 0.11133800000000001) + 0.69763799999999998) -
							    1.3074300000000001) -
						   7.6973500000000001)
						: (cse77 * (cse77 * (cse77 * (0.093813800000000003 * cse77 - 3.0328300000000001) + 36.180300000000003) -
							    187.47399999999999) +
						   350.60899999999998))) +
		    (5.1749999999999996e-11 * std::exp((0.037999999999999999 * cse134 + 0.16900000000000001) * ln_cse133) +
		     2.15e-11 * std::exp((0.048000000000000001 * cse134 - 0.01) * ln_cse133)) *
			state.xn[3];
		const amrex::Real cse211 =
		    2.6400000000000002e-10 * std::exp((cse208 + 0.23100000000000001) * ln_cse133) * state.xn[0] +
		    cse209 *
			std::exp((
			    (cse190)
				? (cse77 * (cse77 * (cse77 * (0.00235272 * cse77 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999)
				: (cse77 * (cse77 * (cse77 * (0.097857299999999994 * cse77 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) +
				   386.18599999999998))) +
		    (1.11e-10 * std::exp((0.031 * cse134 + 0.26300000000000001) * ln_cse133) +
		     4.3749999999999998e-11 * std::exp((0.064000000000000001 * cse134 + 0.071999999999999995) * ln_cse133)) *
			state.xn[3];
		const amrex::Real cse212 =
		    2.8763333333333333e-6 * cse132 *
			std::exp(((cse190) ? (cse77 * (cse77 * (cse77 * (-0.00065632500000000003 * cse77 - 0.015089200000000001) + 0.36118400000000001) -
						       0.77378199999999997) -
					      9.2514099999999999)
					   : (cse77 * (cse77 * (cse77 * (0.10508000000000001 * cse77 - 3.4762) + 42.594999999999999) - 227.91300000000001) +
					      444.60000000000002))) +
		    1.26e-10 * std::exp((0.057000000000000002 * cse134 + 0.115) * ln_cse133) * state.xn[0] +
		    (5.3249999999999997e-11 * std::exp((0.049000000000000002 * cse134 - 0.0040000000000000001) * ln_cse133) +
		     1.675e-11 * std::exp((0.10199999999999999 * cse134 - 0.085000000000000006) * ln_cse133)) *
			state.xn[3];
		const amrex::Real cse213 = 5 * std::exp(-62.460000000000001 * cse18);
		const amrex::Real cse214 = cse212 + 7.8800000000000004e-8;
		const amrex::Real cse215 = cse210 * cse213;
		const amrex::Real cse216 = (25.0 / 3.0) * cse210 * cse211 * std::exp(-101.30000000000001 * cse18) + cse211 * cse212 * cse213 + cse214 * cse215;
		const amrex::Real cse217 = cse211 + 2.65e-7;
		const amrex::Real cse218 = cse210 + 1.81e-14;
		const amrex::Real cse219 = 3 * cse212 * std::exp(-23.619999999999997 * cse18);
		const amrex::Real cse220 = cse215 * cse217 + cse217 * cse219 + cse218 * cse219;
		const amrex::Real cse221 =
		    1.0 / ((5.0 / 3.0) * cse211 * cse218 * std::exp(-38.840000000000003 * cse18) + cse214 * cse217 + cse214 * cse218 + cse216 + cse220);
		const amrex::Real cse222 = 1.8689621674132952e-20 * state.xn[3];
		const amrex::Real cse223 = Zd * cse23;
		const amrex::Real cse224 = state.c_hat * state.rn[2 * 0 + 1];
		const amrex::Real cse225 = 2.6551134042892157e-17 * state.xn[3];
		const amrex::Real cse226 = state.c_hat * state.rn[2 * 1 + 1];
		ydot[0] = cse1 + cse11 + cse16 + cse17 + cse19 + cse22 * cse40 + cse22 * cse42 + 2 * cse35 - cse41 + 2.9935562710997168e-6 * cse44 -
			  6.0000000000000005e-15 * cse45 * state.xn[0] + cse46 + 4.2899999999999994e-13 * cse49 + cse54 + cse59 + cse62 +
			  3.2000000000000001e-9 * cse7 + cse79 + cse9;
		ydot[1] = cse0 + cse58 + cse67 + cse70 - cse72 - cse78 + cse80;
		ydot[2] = -cse103 - cse106 - cse79 + cse83 + cse84 - cse85 - cse86 + cse87 - cse92 - cse94;
		ydot[3] = -cse106 - cse107 - cse108 + cse109 + cse11 - cse110 - cse112 + 3.0000000000000002e-15 * cse23 * d2g * state.xn[0] - cse35 -
			  cse40 * state.xn[3] + 4.5427507615910105e-7 * cse43 * state.xn[16] * state.xn[2] - cse58 - cse62;
		ydot[4] = cse112 + cse113;
		ydot[5] = cse115;
		ydot[6] = -cse115;
		ydot[7] = cse103 + cse116 + cse120 + 9.8999999999999995e-14 * cse49 + cse52;
		ydot[8] = cse100 + cse102 - cse104 + cse114 - cse121 + cse96 - cse98 - cse99;
		ydot[9] = -cse120 - cse122 - cse123;
		ydot[10] = cse121 + cse122;
		ydot[11] = cse118 + cse119 + cse123 + cse124 + cse8 + cse80;
		ydot[12] = -cse125;
		ydot[13] = cse125;
		ydot[14] = -cse116 + 2.3099999999999997e-13 * cse47 * cse48 * state.xn[3] * state.xn[8] - cse54;
		ydot[15] = -cse121 - cse124 - cse19 - cse56 + cse9;
		ydot[16] = -cse109 - cse113 - cse92;
		ydot[17] = -cse105 + cse56 - cse80;
		ydot[18] =
		    5.6705896183685846e-35 * state.c_hat * cse23 * cse73 * d2g * (2.25 * std::exp((0.14699999999999999) * lnT) + 5.2199999999999998) /
			(0.049959999999999997 * std::exp((0.43099999999999999) * ln_cse130) * (0.0043 * std::exp((0.69199999999999995) * ln_cse130) + 1) + 1) +
		    1.602177e-12 * crate * cse129 * state.xn[0] + 3.2043540000000001e-12 * crate * cse145 * state.xn[3] +
		    1.7623947000000002e-12 * crate * (cse129 * cse29 + 2.0 * cse145 * cse28) * state.xn[5] / cse65 - 7.1777529600000012e-12 * cse110 -
		    1.3659996292638161e-10 * cse126 * cse128 * cse69 / (1.3338116951315871e-8 * cse127 * cse128 + 1) -
		    5.7927337833400002e-20 * cse138 * state.xn[8] / (2 * cse138 + 1) -
		    cse147 * std::pow(1 - std::exp(-state.T), 1000) * state.xn[9] /
			(cse147 * cse151 + cse149 * std::pow(cse147 / cse152, alpha_co_interp2d(state.T, cse150)) * (-cse148 * cse151 * cse152 + 1) + cse149) +
		    3.0000000000000002e-15 * cse23 * d2g * (6.7291434000000005e-12 * cse143 + 3.2043540000000002e-13) * state.xn[0] -
		    7.1777529600000012e-12 * cse35 + cse40 * (2.5634832e-11 * cse143 + 6.4087080000000003e-13) * state.xn[3] -
		    1.8998126555754159e-29 * cse45 * std::exp((0.45350000000000001 + 2.234 / cse131) * lnT) *
			std::exp(cse131 * (1.4419999999999999 - 0.050889999999999998 * cse131)) * state.xn[2] -
		    2.1789607199999999e-11 * cse70 - (2.3872025442750958e-19 * cse160 * cse165 + 2.8023005140019997e-18 * cse164 * cse165) * state.xn[11] -
		    ((((cse135 && cse166 && cse167)
			   ? (cse168 || cse175 + cse185 < 1.0e-100)
			   : (((cse166 && cse167)
				   ? (cse168 || cse185 + cse189 < 1.0e-100)
				   : (((cse135 && cse166 && cse167 && cse190)
					   ? (cse168 || cse175 + cse191 < 1.0e-100)
					   : (((cse135 && cse167 && cse190)
						   ? (cse168 || cse191 + cse192 < 1.0e-100)
						   : (((cse167 && cse190)
							   ? (cse168 || cse189 + cse191 < 1.0e-100)
							   : (((cse135 && cse167)
								   ? (cse168 || cse192 + cse193 < 1.0e-100)
								   : (((cse167)
									   ? (cse168 || cse189 + cse193 < 1.0e-100)
									   : (((cse135) ? (cse168 ||
											   std::exp(2.302585092994046 *
												    (7500.646099484944 * cse169 + 21588.957206746247 * cse170 +
												     26530.843406310876 * cse171 +
												     18165.631967965237 * cse172 +
												     7572.4886742758581 * cse173 +
												     1974.131269262874 * cse174 + 30.129743459482278 * cse21 +
												     310.58441786958315 * cse25 - 21.928795999999998)) *
												       state.xn[2] +
												   cse194 <
											       1.0e-100)
											: (cse168 ||
											   std::exp(2.302585092994046 *
												    (-676.68255490510751 * cse169 +
												     381.69251128040054 * cse170 + 295.04162851880631 * cse171 -
												     162.65264843551881 * cse172 - 48.723840736518795 * cse173 +
												     23.242975457496804 * cse174 + 3.0106500755647456 * cse21 +
												     2.9956460707652877 * cse25 - 22.921188999999998)) *
												       state.xn[2] +
												   cse194 <
											       1.0e-100)))))))))))))))))
			 ? (0.0)
			 : (cse196 /
			    (cse196 /
				 (std::exp(2.302585092994046 * (0.19191374999999999 * cse198 - 0.16596184 * cse199 - 0.81520437999999995 * cse201 +
								2.1892372 * cse202 + 0.29036281000000003 * cse203 - 23.689236999999999)) *
				      cse183 +
				  std::exp(2.302585092994046 * (0.32168730000000001 * cse198 - 0.51002221000000003 * cse199 + 0.015391166 * cse201 +
								1.5714710999999999 * cse202 - 0.23619984999999999 * cse203 - 22.089523)) *
				      state.xn[1] +
				  std::exp(2.302585092994046 *
					   (0.43693353000000001 * cse174 * cse200 + 2.0943374000000001 * cse197 * cse21 - 0.033638326000000003 * cse198 -
					    0.14913216000000001 * cse199 - 0.77151435999999995 * cse201 - 23.962112000000001)) *
				      state.xn[3] +
				  ((cse135) ? (std::exp(2.302585092994046 *
							(70.609154000000004 * ((cse204) ? (790.17904733961348 * cse169) : (cse205)) +
							 364.14445999999998 * ((cse204) ? (-343.17039997516258 * cse170) : (cse206)) +
							 801.81246999999996 * ((cse204) ? (149.03701106174495 * cse171) : (cse207)) +
							 983.67575999999997 * ((cse204) ? (-64.725951503469744 * cse172) : (cse198)) +
							 734.71650999999997 * ((cse204) ? (28.110123573894395 * cse173) : (cse199)) +
							 343.1918 * ((cse204) ? (-12.208071553760854 * cse174) : (cse203)) +
							 16.815729999999999 * ((cse204) ? (-2.3025850929940455 * cse21) : (cse202)) +
							 96.743155000000002 * ((cse204) ? (5.3018981104783967 * cse25) : (cse201)) - 21.928795999999998)))
					    : (std::exp(2.302585092994046 *
							(-8.8077017000000009 * cse198 - 4.7274035999999997 * cse199 + 0.93310621999999999 * cse201 +
							 1.6802758 * cse202 + 4.0406627000000004 * cse203 - 6.3701156000000001 * cse205 +
							 6.4380698000000001 * cse206 + 8.9167182999999994 * cse207 - 22.921188999999998)))) *
				      state.xn[2] +
				  ((cse204) ? (std::exp(2.302585092994046 *
							(3.8479610000000002 * cse198 + 20.159831000000001 * cse199 + 58.145166000000003 * cse201 +
							 37.383713 * cse202 + 48.656103000000002 * cse203 - 16.818342000000001)))
					    : ((((cse167) ? (cse190) : (false)))
						   ? (std::exp(2.302585092994046 *
							       (3.5692468000000002 * cse197 * cse21 - 4.2519023000000002 * cse198 -
								21.328264000000001 * cse199 - 11.33286 * cse201 - 27.850082 * cse203 - 24.311209000000002)))
						   : (std::exp(2.302585092994046 * (1.5538288 * cse198 - 5.5108049000000001 * cse199 - 3.7209846 * cse201 +
										    4.6450521 * cse202 + 5.9369081000000001 * cse203 - 24.311209000000002))))) *
				      state.xn[0]) +
			     1))) *
			state.xn[3] -
		    ((state.T < 1000000.0) ? (1.4210469458259591e-21 * cse216 * cse221 + 2.5697412351439998e-22 * cse220 * cse221) : (0.0)) * state.xn[7] -
		    std::min(100.0 * cse45 * psigd_coll_interp2d(cse23, state.T),
			     3.2000000000000002e-32 * cse2 * (cse23 * cse23) * d2g *
				 (state.T - 2.1544346900318838 * std::pow(2.8677558904086741e-6 * cse74 + 4.1397668473788896, 1.0 / 6.0)));
		ydot[19] = -1.252779946046115e-21 * cse12 * cse223 - cse13 * state.xn[15] - cse222 * cse38 - cse50 * state.xn[14] - cse81 * state.xn[12];
		ydot[20] = -cse222 * cse224 * cse37 - 9.1016827996670106e-22 * cse223 * cse224 - 1.9917698086432708e-17 * cse224 * state.xn[12] -
			   4.2188032355736613e-18 * cse224 * state.xn[14] - 1.7141251340095377e-18 * cse224 * state.xn[15];
		ydot[21] = -cse102 - cse118 - 2.0584539190345922e-21 * cse14 * cse223 - cse15 * state.xn[15] - cse225 * cse39 - cse51 * state.xn[14] -
			   cse82 * state.xn[12];
		ydot[22] = -cse101 * cse226 - cse117 * cse226 - 1.6856051574414119e-21 * cse223 * cse226 - cse225 * cse226 * cse37 -
			   1.9088222220940885e-17 * cse226 * state.xn[12] - 1.168047934563505e-18 * cse226 * state.xn[14] -
			   4.1730320353928637e-18 * cse226 * state.xn[15];
		// $$JAFF END
		// $$JAFF SUB nspec
		ydot[18] /= state.rho;
		// $$JAFF END
	}

	/// The Jacobian of actual_rhs.
	template <typename BurnT, class MatrixType> AMREX_GPU_HOST_DEVICE static void actual_jac(const BurnT &state, MatrixType &jac)
	{
		using System = quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>;
		static_assert(BurnT::traits::integrate_rad_flux, "a jaff network needs Microphysics_Traits::integrate_rad_flux");

		// $$JAFF REPEAT idx, expr, cse IN jacobian { USE quokka }
		const amrex::Real lnT = std::log(state.T);
		const amrex::Real cse0 =
		    1.0 / (2.0709734999999996e-16 * state.xn[0] + 2.0709734999999996e-16 * state.xn[10] + 2.0709734999999996e-16 * state.xn[11] +
			   2.0709734999999996e-16 * state.xn[12] + 2.0709734999999996e-16 * state.xn[13] + 2.0709734999999996e-16 * state.xn[14] +
			   2.0709734999999996e-16 * state.xn[15] + 2.0709734999999996e-16 * state.xn[16] + 2.0709734999999996e-16 * state.xn[17] +
			   2.0709734999999996e-16 * state.xn[1] + 2.0709734999999996e-16 * state.xn[2] + 3.4516225000000006e-16 * state.xn[3] +
			   3.4516225000000006e-16 * state.xn[4] + 2.0709734999999996e-16 * state.xn[5] + 2.0709734999999996e-16 * state.xn[6] +
			   2.0709734999999996e-16 * state.xn[7] + 2.0709734999999996e-16 * state.xn[8] + 2.0709734999999996e-16 * state.xn[9]);
		const amrex::Real cse1 = (1.0 / (state.T * state.T));
		const amrex::Real cse2 = 1.0 / state.T;
		const amrex::Real ln_cse2 = std::log(cse2);
		const amrex::Real cse3 = std::exp(-22.5 * cse2);
		const amrex::Real cse4 = 2.8349999999999999e-12 * cse1 * cse3 * state.xn[3] * state.xn[6];
		const amrex::Real cse5 = -cse4;
		const amrex::Real cse6 = 7.3171760954257497e-12 * std::exp((-0.73999999999999999) * lnT) * state.xn[0] * state.xn[14];
		const amrex::Real cse7 = std::sqrt(cse2);
		const amrex::Real cse8 = std::sqrt(3) * cse2 * cse7 * state.xn[15];
		const amrex::Real cse9 = 1.7697487500000004e-8 * cse8 * state.xn[6];
		const amrex::Real cse10 = ((state.xn[3]) * (state.xn[3]));
		const amrex::Real cse11 = std::exp((4.1881000000000004) * lnT);
		const amrex::Real cse12 = 6.7610000000000002e-6 * state.T + 1;
		const amrex::Real ln_cse12 = std::log(cse12);
		const amrex::Real cse13 = std::exp((5.6881000000000004) * ln_cse12);
		const amrex::Real cse14 = 54657.400000000001 * cse2;
		const amrex::Real cse15 = std::exp(-cse14);
		const amrex::Real cse16 = cse15 / cse13;
		const amrex::Real cse17 = M_LN10;
		const amrex::Real cse18 = 1.0 / cse17;
		const amrex::Real cse19 = 2 * state.xn[3];
		const amrex::Real cse20 = cse19 + state.xn[0] + state.xn[1] + state.xn[10] + state.xn[14] + state.xn[15] + 3 * state.xn[16] + 2 * state.xn[4];
		const amrex::Real cse21 = 1.0 / cse20;
		const amrex::Real cse22 = std::log((1.0 / 10000.0) * state.T);
		const amrex::Real cse23 = (1.0 / (cse17 * cse17));
		const amrex::Real cse24 = (cse22 * cse22) * cse23;
		const amrex::Real cse25 = std::exp(2.302585092994046 * (1.3 * cse18 * cse22 - 1.6200000000000001 * cse24 - 4.8449999999999998));
		const amrex::Real cse26 = cse21 * cse25 * state.xn[3];
		const amrex::Real cse27 = std::exp(2.302585092994046 * (0.41599999999999998 * cse18 * cse22 + 0.32700000000000001 * cse24 - 3));
		const amrex::Real cse28 = cse21 * cse27 * state.xn[0];
		const amrex::Real cse29 = cse26 + cse28;
		const amrex::Real cse30 = cse29 < 1.0e-50;
		const amrex::Real cse31 = ((cse30) ? (1.0e-50) : (cse29));
		const amrex::Real cse32 = cse20 * cse31;
		const amrex::Real cse33 = cse32 + 1;
		const amrex::Real cse34 = cse18 / cse33;
		const amrex::Real cse35 = cse32 * cse34;
		const amrex::Real cse36 = cse1 * cse35;
		const amrex::Real cse37 = cse2 * cse23;
		const amrex::Real cse38 = cse22 * cse37;
		const amrex::Real cse39 = cse18 * cse2;
		const amrex::Real cse40 = ((cse30) ? (0)
						   : (cse17 * cse26 * (1.3 * cse18 * cse2 - 3.2400000000000002 * cse38) +
						      cse17 * cse28 * (0.65400000000000003 * cse38 + 0.41599999999999998 * cse39)));
		const amrex::Real cse41 = cse20 * cse40;
		const amrex::Real cse42 = std::log(1.3000000000000001e-9 * std::exp(-53300 * cse2));
		const amrex::Real cse43 = cse34 * cse42;
		const amrex::Real cse44 = std::log(5.9959999999999997e-30 * cse11 * cse16);
		const amrex::Real cse45 = cse18 / (cse33 * cse33);
		const amrex::Real cse46 = cse41 * cse45;
		const amrex::Real cse47 = (cse20 * cse20);
		const amrex::Real cse48 = cse31 * cse40 * cse45 * cse47;
		const amrex::Real cse49 = std::exp(2.302585092994046 * (cse34 * cse44 + cse35 * cse42));
		const amrex::Real cse50 = cse17 * cse49;
		const amrex::Real cse51 = state.T > 700.0;
		const amrex::Real cse52 = ((cse51) ? (cse50 * (53300.0 * cse36 + cse41 * cse43 - cse42 * cse48 - cse44 * cse46 +
							       1.6677785190126752e+29 * cse13 * cse34 *
								   (-2.3058963562359999e-34 * cse11 * std::exp((-6.6881000000000004) * ln_cse12) * cse15 +
								    3.2772577039999999e-25 * cse16 * std::exp((2.1881000000000004) * lnT) +
								    2.51118476e-29 * cse16 * std::exp((3.1881000000000004) * lnT)) *
								   std::exp(cse14) / cse11))
						   : (0));
		const amrex::Real cse53 =
		    state.xn[0] * state.xn[17] * (-3.435e-10 * std::exp((-1.458) * lnT) + 2.0209499999999999e-11 * std::exp((-0.59499999999999997) * lnT));
		const amrex::Real cse54 = std::sqrt(state.T);
		const amrex::Real cse55 = 1.0 / cse54;
		const amrex::Real cse56 = std::exp((-3.0 / 2.0) * lnT);
		const amrex::Real cse57 = 63590 * cse2 + 1;
		const amrex::Real cse58 = std::exp(-cse57);
		const amrex::Real cse59 = std::log(3.5199999999999998e-9 * std::exp(-43900 * cse2));
		const amrex::Real cse60 = cse34 * cse59;
		const amrex::Real cse61 = std::log(6.6699999999999996e-12 * cse54 * cse58);
		const amrex::Real cse62 = std::exp(2.302585092994046 * (cse34 * cse61 + cse35 * cse59));
		const amrex::Real cse63 = cse17 * cse62;
		const amrex::Real cse64 =
		    ((cse51) ? (cse63 * (149925037481.25937 * cse34 * cse55 * (3.3349999999999998e-12 * cse55 * cse58 + 4.2414529999999999e-7 * cse56 * cse58) *
					     std::exp(cse57) +
					 43900.0 * cse36 + cse41 * cse60 - cse46 * cse61 - cse48 * cse59))
			     : (0));
		const amrex::Real cse65 = std::exp((-1.52) * lnT) * state.xn[16] * state.xn[2];
		const amrex::Real cse66 = std::exp(-227 * cse2);
		const amrex::Real cse67 =
		    cse66 * state.xn[1] * state.xn[11] *
		    (2.6759999999999999e-12 * std::exp((-0.99331000000000003) * lnT) + 5.6870000000000003e-12 * std::exp((-0.48299999999999998) * lnT));
		const amrex::Real cse68 = std::exp((-2.2999999999999998) * lnT);
		const amrex::Real cse69 = std::exp(-23 * cse2);
		const amrex::Real cse70 = cse69 * state.xn[3] * state.xn[8];
		const amrex::Real cse71 = cse68 * cse70;
		const amrex::Real cse72 = std::exp((0.042000000000000003) * lnT);
		const amrex::Real cse73 = std::exp(-2.145922746781116e-5 * state.T);
		const amrex::Real cse74 = cse73 * state.xn[4];
		const amrex::Real cse75 = cse74 * state.xn[3];
		const amrex::Real cse76 = 3.7768240343347639e-14 * cse72 * cse75;
		const amrex::Real cse77 = 7.3920000000000002e-11 * cse75 * std::exp((-0.95799999999999996) * lnT);
		const amrex::Real cse78 = std::exp((-3.2999999999999998) * lnT);
		const amrex::Real cse79 =
		    cse66 * (4.0000000000000001e-10 * std::exp((0.0066899999999999998) * lnT) + 1.1000000000000001e-11 * std::exp((0.51700000000000002) * lnT));
		const amrex::Real cse80 = cse79 * state.xn[11];
		const amrex::Real cse81 = 227 * cse1 * cse80 * state.xn[1];
		const amrex::Real cse82 = 1.0 / state.xn[2];
		const amrex::Real cse83 = cse55 * cse82;
		const amrex::Real cse84 = cse54 * cse82;
		const amrex::Real cse85 = 0.00011320754716981131 * cse84;
		const amrex::Real cse86 = cse85 * state.xn[3];
		const amrex::Real cse87 = (((cse86 - 10000000000.0 > 0) ? (0) : ((cse86 - 10000000000.0 == 0) ? (1.0 / 2.0) : (1))));
		const amrex::Real cse88 = cse87 * state.xn[17];
		const amrex::Real cse89 = cse10 * cse88;
		const amrex::Real cse90 = cse83 * cse89;
		const amrex::Real cse91 = std::min(10000000000.0, cse86);
		const amrex::Real cse92 = cse91 + 1;
		const amrex::Real cse93 = (1.0 / (cse92 * cse92));
		const amrex::Real cse94 = 1.811320754716981e-13 * cse93;
		const amrex::Real cse95 = 6.7743911789075847e-6 * std::exp((-1.6400000000000001) * lnT) * state.xn[10] * state.xn[2];
		const amrex::Real cse96 = std::exp(0.11 * cse2);
		const amrex::Real cse97 = cse96 * state.xn[15];
		const amrex::Real cse98 = cse97 * state.xn[7];
		const amrex::Real cse99 = 8.7465066538127767e-11 * cse98 * std::exp((-2.339) * lnT) + 2.6955143233113923e-10 * cse98 * std::exp((-1.339) * lnT);
		const amrex::Real cse100 = cse95 + cse99;
		const amrex::Real cse101 = std::log(8.6173000000000003e-5 * state.T);
		const amrex::Real cse102 = (cse101 * cse101);
		const amrex::Real cse103 = (cse101 * cse101 * cse101);
		const amrex::Real cse104 = (cse101 * cse101 * cse101 * cse101);
		const amrex::Real cse105 = (cse101 * cse101 * cse101 * cse101 * cse101);
		const amrex::Real cse106 = (cse101 * cse101 * cse101 * cse101 * cse101 * cse101);
		const amrex::Real cse107 = (cse101 * cse101 * cse101 * cse101 * cse101 * cse101 * cse101);
		const amrex::Real cse108 = std::exp(-2.0391498499999999e-6 * (cse101 * cse101 * cse101 * cse101 * cse101 * cse101 * cse101 * cse101) -
						    5.7393287500000003 * cse102 + 1.56315498 * cse103 - 0.28770560000000001 * cse104 +
						    0.034825597700000002 * cse105 - 0.00263197617 * cse106 + 0.000111954395 * cse107);
		const amrex::Real cse109 = 5.9078447946382801e-70 * cse108 * std::exp((13.536555999999999) * lnT);
		const amrex::Real cse110 = state.xn[0] * state.xn[2];
		const amrex::Real cse111 =
		    cse110 * ((cse51) ? (7.9971871901929577e-69 * cse108 * std::exp((12.536555999999999) * lnT) +
					 cse109 * (-11.478657500000001 * cse101 * cse2 + 4.6894649399999997 * cse102 * cse2 - 1.1508224 * cse103 * cse2 +
						   0.1741279885 * cse104 * cse2 - 0.015791857020000001 * cse105 * cse2 +
						   0.00078368076500000001 * cse106 * cse2 - 1.6313198799999999e-5 * cse107 * cse2))
				      : (0));
		const amrex::Real cse112 = (cse2 * std::sqrt(cse2));
		const amrex::Real cse113 = 114.81344434488582 * std::exp((0.40699999999999997) * ln_cse2) + 1;
		const amrex::Real ln_cse113 = std::log(cse113);
		const amrex::Real cse114 = std::exp((-2.242) * ln_cse113);
		const amrex::Real cse115 = cse2 * state.xn[1] * state.xn[2];
		const amrex::Real cse116 = 7.3220359950196314e-6 * cse112 * cse114 * cse115;
		const amrex::Real cse117 = 0.0005114031087522854 * std::exp((-3.242) * ln_cse113) * cse115 * std::exp((1.907) * ln_cse2);
		const amrex::Real cse118 = 144.3803103740824 * state.rn[2 * 0] + 220.5700827961663 * state.rn[2 * 1];
		const amrex::Real cse119 = state.c_hat * cse118;
		const amrex::Real cse120 = cse119 * cse84;
		const amrex::Real ln_cse120 = std::log(cse120);
		const amrex::Real cse121 = std::exp((1.3779999999999999) * ln_cse120);
		const amrex::Real cse122 = 5.670589618368584e-11 * cse120;
		const amrex::Real ln_cse122 = std::log(cse122);
		const amrex::Real cse123 = std::log(state.T);
		const amrex::Real cse124 = 1.102e-5 * cse123 + 0.4723;
		const amrex::Real cse125 = std::exp((-cse124) * ln_cse122);
		const amrex::Real cse126 = cse125 * std::exp((0.015859999999999999) * lnT);
		const amrex::Real cse127 = 508.69999999999999 * cse126;
		const amrex::Real cse128 = cse127 + 1;
		const amrex::Real cse129 = cse121 * cse128;
		const amrex::Real cse130 = -cse124;
		const amrex::Real cse131 = 0.5 * cse2;
		const amrex::Real cse132 = cse2 * std::log(cse122);
		const amrex::Real cse133 = 6.1318063394789369e-20 * cse121;
		const amrex::Real cse134 = cse128 * cse133 + 1;
		const amrex::Real cse135 = cse20 * d2g;
		const amrex::Real cse136 = 1.2250000000000001e-11 * cse135 * state.xn[1] / (cse134 * cse134);
		const amrex::Real cse137 =
		    cse136 * (-4.2248145679009874e-20 * cse129 * cse2 -
			      cse133 * (8.0679819999999989 * cse125 * std::exp((-0.98414000000000001) * lnT) + cse127 * (cse130 * cse131 - 1.102e-5 * cse132)));
		const amrex::Real cse138 = 1.0 / cse92;
		const amrex::Real cse139 = 9.0566037735849048e-14 * cse90;
		const amrex::Real cse140 = cse138 * cse139;
		const amrex::Real cse141 = cse139 * cse93;
		const amrex::Real cse142 = cse141 * cse91;
		const amrex::Real cse143 = -cse140 + cse142;
		const amrex::Real cse144 = cse138 * state.xn[16];
		const amrex::Real cse145 = 3.7810000000000002e-10 * std::exp((-1.1899999999999999) * lnT) * state.xn[11];
		const amrex::Real cse146 = cse82 * cse87 * cse93 * state.xn[11] * state.xn[16];
		const amrex::Real cse147 = cse144 * cse145 + 1.1264150943396225e-13 * cse146 * std::exp((-0.68999999999999995) * lnT) * state.xn[3];
		const amrex::Real cse148 = cse111 + cse116 - cse117 - cse137 + cse143 + cse147;
		const amrex::Real cse149 = cse0 * (2 * cse10 * cse52 - cse100 - cse148 - cse5 - cse53 - cse6 + 2 * cse64 * state.xn[0] * state.xn[3] -
						   1.5566492609718528e-6 * cse65 + cse67 + 9.8669999999999991e-12 * cse69 * cse78 * state.xn[3] * state.xn[8] -
						   5.5769999999999994e-13 * cse71 - cse76 + cse77 + cse81 - cse9 - cse90 * cse94);
		const amrex::Real cse150 = cse149 * state.T;
		const amrex::Real cse151 = 2.0709734999999996e-16 * cse150;
		const amrex::Real cse152 = 2.8142984982406731e-11 * std::exp((0.26000000000000001) * lnT);
		const amrex::Real cse153 = cse152 * state.xn[14];
		const amrex::Real cse154 = ((cse51) ? (cse62) : (0.0));
		const amrex::Real cse155 = cse31 * cse34;
		const amrex::Real cse156 = cse155 * cse42;
		const amrex::Real cse157 = 1.0 / cse47;
		const amrex::Real cse158 = cse157 * state.xn[3];
		const amrex::Real cse159 = cse158 * cse25;
		const amrex::Real cse160 = cse157 * state.xn[0];
		const amrex::Real cse161 = cse160 * cse27;
		const amrex::Real cse162 = cse159 + cse161;
		const amrex::Real cse163 = cse20 * ((cse30) ? (0) : (-cse162 + cse21 * cse27));
		const amrex::Real cse164 = cse163 * cse34;
		const amrex::Real cse165 = cse45 * (-cse163 - cse31);
		const amrex::Real cse166 = cse165 * cse32;
		const amrex::Real cse167 = ((cse51) ? (cse50 * (cse156 + cse164 * cse42 + cse165 * cse44 + cse166 * cse42)) : (0));
		const amrex::Real cse168 = 6.0000000000000005e-15 * d2g;
		const amrex::Real cse169 = cse168 * state.xn[0];
		const amrex::Real cse170 = cse155 * cse59;
		const amrex::Real cse171 = ((cse51) ? (cse63 * (cse164 * cse59 + cse165 * cse61 + cse166 * cse59 + cse170)) : (0));
		const amrex::Real cse172 = 7.5e-10 * std::exp((-0.45800000000000002) * lnT) + 4.9899999999999997e-11 * std::exp((0.40500000000000003) * lnT);
		const amrex::Real cse173 = cse172 * state.xn[17];
		const amrex::Real cse174 = 6.3999999999999996e-10 * state.xn[4];
		const amrex::Real cse175 = 1.5 * cse21 * state.xn[0] + 2.2999999999999998 * cse21 * state.xn[3];
		const amrex::Real cse176 = crate * cse175;
		const amrex::Real cse177 = ((cse51) ? (cse109) : (0.0));
		const amrex::Real cse178 = cse177 * state.xn[2];
		const amrex::Real cse179 = 2.2999999999999998 * cse158 + 1.5 * cse160;
		const amrex::Real cse180 = cse179 - 1.5 * cse21;
		const amrex::Real cse181 = -cse180;
		const amrex::Real cse182 = crate * state.xn[0];
		const amrex::Real cse183 = cse181 * cse182;
		const amrex::Real cse184 = 1.0 / cse134;
		const amrex::Real cse185 = 1.2250000000000001e-11 * cse184;
		const amrex::Real cse186 = d2g * state.xn[1];
		const amrex::Real cse187 = cse185 * cse186;
		const amrex::Real cse188 = cse173 + cse174 + cse176 + cse178 + cse183 - cse187;
		const amrex::Real cse189 = 4.8813573300130879e-6 * cse112 * cse114;
		const amrex::Real cse190 = cse135 * cse185 + cse189 * state.xn[2];
		const amrex::Real cse191 = cse190 + cse80;
		const amrex::Real cse192 = -cse179;
		const amrex::Real cse193 = -cse182 * cse192 + cse187;
		const amrex::Real cse194 = -cse151;
		const amrex::Real cse195 = cse20 * ((cse30) ? (0) : (-cse162));
		const amrex::Real cse196 = cse45 * (-cse195 - cse31);
		const amrex::Real cse197 = cse196 * cse32;
		const amrex::Real cse198 = cse10 * ((cse51) ? (cse50 * (cse156 + cse195 * cse43 + cse196 * cse44 + cse197 * cse42)) : (0));
		const amrex::Real cse199 = -cse169;
		const amrex::Real cse200 = ((cse51) ? (cse63 * (cse170 + cse195 * cse60 + cse196 * cse61 + cse197 * cse59)) : (0));
		const amrex::Real cse201 = cse19 * state.xn[0];
		const amrex::Real cse202 = cse194 + 2 * cse198 + cse199 + cse200 * cse201;
		const amrex::Real cse203 = cse193 + cse202;
		const amrex::Real cse204 = ((state.xn[2]) * (state.xn[2]));
		const amrex::Real cse205 = 1.0 / cse204;
		const amrex::Real cse206 = 2.2528301886792451e-13 * std::exp((0.31) * lnT);
		const amrex::Real cse207 = cse205 * cse206 * cse87 * cse93 * state.xn[11] * state.xn[16] * state.xn[3];
		const amrex::Real cse208 = std::exp((-0.52000000000000002) * lnT);
		const amrex::Real cse209 = 2.9935562710997168e-6 * cse208;
		const amrex::Real cse210 = cse205 * cse54;
		const amrex::Real cse211 = cse210 * cse89;
		const amrex::Real cse212 = 3.6226415094339619e-13 * cse93;
		const amrex::Real cse213 = 1.811320754716981e-13 * cse138;
		const amrex::Real cse214 = cse211 * cse213;
		const amrex::Real cse215 = cse211 * cse94;
		const amrex::Real cse216 = cse215 * cse91;
		const amrex::Real cse217 = -cse214 + cse216;
		const amrex::Real cse218 = 1.05849862170431e-5 * std::exp((-0.64000000000000001) * lnT);
		const amrex::Real cse219 = cse218 * state.xn[10];
		const amrex::Real cse220 = cse177 * state.xn[0];
		const amrex::Real cse221 = cse121 * cse126 * cse130;
		const amrex::Real cse222 =
		    cse136 * (8.4496291358019747e-20 * cse129 * cse82 + 3.1192498848929353e-17 * cse221 * cse82) + cse189 * state.xn[1] - cse220;
		const amrex::Real cse223 = cse219 + cse222;
		const amrex::Real cse224 = 1.2599999999999999e-13 * cse3;
		const amrex::Real cse225 = cse224 * state.xn[6];
		const amrex::Real cse226 = 1.7599999999999999e-9 * cse72;
		const amrex::Real cse227 = cse226 * cse74;
		const amrex::Real cse228 = cse184 * cse186;
		const amrex::Real cse229 = 2.4500000000000003e-11 * cse228;
		const amrex::Real cse230 = cse154 * state.xn[0];
		const amrex::Real cse231 = 2 * cse31;
		const amrex::Real cse232 = cse231 * cse43;
		const amrex::Real cse233 = cse157 * cse19;
		const amrex::Real cse234 = 2 * state.xn[0];
		const amrex::Real cse235 = cse157 * cse234;
		const amrex::Real cse236 = cse233 * cse25 + cse235 * cse27;
		const amrex::Real cse237 = cse20 * ((cse30) ? (0) : (cse21 * cse25 - cse236));
		const amrex::Real cse238 = cse45 * (-cse231 - cse237);
		const amrex::Real cse239 = cse238 * cse32;
		const amrex::Real cse240 = cse10 * ((cse51) ? (cse50 * (cse232 + cse237 * cse43 + cse238 * cse44 + cse239 * cse42)) : (0));
		const amrex::Real cse241 = ((cse51) ? (cse49) : (0.0));
		const amrex::Real cse242 = 4 * state.xn[3];
		const amrex::Real cse243 = d2g * state.xn[0];
		const amrex::Real cse244 = 1.2000000000000001e-14 * cse243;
		const amrex::Real cse245 = cse231 * cse60;
		const amrex::Real cse246 = ((cse51) ? (cse63 * (cse237 * cse60 + cse238 * cse61 + cse239 * cse59 + cse245)) : (0));
		const amrex::Real cse247 = state.c_hat * state.rn[2 * 0];
		const amrex::Real cse248 = std::max(1.0e-35, 2.0000000000000002e-15 * state.ncol[ncol_H2]);
		const amrex::Real cse249 = 0.035000000000000003 * (1.0 / std::sqrt(cse248 + 1)) * std::exp(-0.00084999999999999995 * std::sqrt(cse248 + 1.0)) +
					   0.96499999999999997 * (1.0 / ((cse248 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0) *
									 (cse248 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0)));
		const amrex::Real cse250 = 2.4433717132788467e-27 * cse249;
		const amrex::Real cse251 = state.c_hat * state.rn[2 * 1];
		const amrex::Real cse252 = 7.3498720606844327e-18 * cse249;
		const amrex::Real cse253 = cse69 * std::exp((-1.3) * lnT);
		const amrex::Real cse254 = 4.2899999999999994e-13 * cse253;
		const amrex::Real cse255 = 3.4516225000000006e-16 * cse150;
		const amrex::Real cse256 = cse84 * cse88 * state.xn[3];
		const amrex::Real cse257 = 1.6000000000000001e-9 * cse138;
		const amrex::Real cse258 = cse257 * state.xn[17];
		const amrex::Real cse259 = cse256 * cse94;
		const amrex::Real cse260 = cse213 * cse256 + cse258 * cse91 - cse259 * cse91;
		const amrex::Real cse261 = 4.5999999999999996 * cse158 + 3.0 * cse160;
		const amrex::Real cse262 = -2.2999999999999998 * cse21 + cse261;
		const amrex::Real cse263 = -cse262;
		const amrex::Real cse264 = cse182 * cse263;
		const amrex::Real cse265 = cse146 * cse206;
		const amrex::Real cse266 = -cse264 - cse265;
		const amrex::Real cse267 = -cse229;
		const amrex::Real cse268 = cse20 * ((cse30) ? (0) : (-cse236));
		const amrex::Real cse269 = cse45 * (-cse231 - cse268);
		const amrex::Real cse270 = cse269 * cse32;
		const amrex::Real cse271 = ((cse51) ? (cse50 * (cse232 + cse268 * cse43 + cse269 * cse44 + cse270 * cse42)) : (0));
		const amrex::Real cse272 = ((cse51) ? (cse63 * (cse245 + cse268 * cse60 + cse269 * cse61 + cse270 * cse59)) : (0));
		const amrex::Real cse273 = cse226 * cse73 * state.xn[3];
		const amrex::Real cse274 = 6.3999999999999996e-10 * state.xn[0];
		const amrex::Real cse275 = -cse261;
		const amrex::Real cse276 = cse182 * cse275;
		const amrex::Real cse277 = cse274 + cse276;
		const amrex::Real cse278 = 26.218500000000002 * std::sqrt(3) * std::sqrt(cse2) + 0.62;
		const amrex::Real cse279 = 1.3500000000000001e-9 * cse278;
		const amrex::Real cse280 = cse279 * state.xn[15];
		const amrex::Real cse281 = cse224 * state.xn[3];
		const amrex::Real cse282 = cse280 + cse281;
		const amrex::Real cse283 = 7.9513696852843427e-10 * std::exp((-0.33900000000000002) * lnT);
		const amrex::Real cse284 = cse283 * cse97;
		const amrex::Real cse285 = 6.5199999999999996 * crate;
		const amrex::Real cse286 = cse218 * state.xn[2];
		const amrex::Real cse287 = cse193 + cse286;
		const amrex::Real cse288 = 1.99e-9 * std::exp((-0.19) * lnT);
		const amrex::Real cse289 = cse144 * cse288;
		const amrex::Real cse290 = 3.5000000000000002e-11 * state.xn[15];
		const amrex::Real cse291 = cse79 * state.xn[1];
		const amrex::Real cse292 = 7.0000000000000004e-11 * state.xn[14];
		const amrex::Real cse293 = cse291 + cse292;
		const amrex::Real cse294 = 4.2188032355736613e-18 * cse247;
		const amrex::Real cse295 = 1.168047934563505e-18 * cse251;
		const amrex::Real cse296 = 7.0000000000000004e-11 * state.xn[11];
		const amrex::Real cse297 = cse152 * state.xn[0];
		const amrex::Real cse298 = -cse297;
		const amrex::Real cse299 = cse279 * state.xn[6];
		const amrex::Real cse300 = cse283 * cse96 * state.xn[7];
		const amrex::Real cse301 = 1.7141251340095377e-18 * cse247;
		const amrex::Real cse302 = 4.1730320353928637e-18 * cse251;
		const amrex::Real cse303 = cse301 + cse302 + 3.5000000000000002e-11 * state.xn[11];
		const amrex::Real cse304 = cse299 + cse300 + cse303;
		const amrex::Real cse305 = 3 * cse31;
		const amrex::Real cse306 = cse20 * ((cse30) ? (0) : (-3 * cse159 - 3 * cse161));
		const amrex::Real cse307 = cse45 * (-cse305 - cse306);
		const amrex::Real cse308 = cse307 * cse32;
		const amrex::Real cse309 = cse10 * ((cse51) ? (cse50 * (cse305 * cse43 + cse306 * cse43 + cse307 * cse44 + cse308 * cse42)) : (0));
		const amrex::Real cse310 = ((cse51) ? (cse63 * (cse305 * cse60 + cse306 * cse60 + cse307 * cse61 + cse308 * cse59)) : (0));
		const amrex::Real cse311 = cse288 * state.xn[11];
		const amrex::Real cse312 = cse138 * cse311;
		const amrex::Real cse313 = 6.8999999999999995 * cse158 + 4.5 * cse160;
		const amrex::Real cse314 = -cse313;
		const amrex::Real cse315 = -cse182 * cse314 + 3.6750000000000002e-11 * cse228;
		const amrex::Real cse316 = cse312 + cse315;
		const amrex::Real cse317 = cse172 * state.xn[0];
		const amrex::Real cse318 = cse257 * state.xn[3];
		const amrex::Real cse319 = cse318 * cse91;
		const amrex::Real cse320 = -cse319;
		const amrex::Real cse321 = state.c_hat * state.xn[3];
		const amrex::Real cse322 = 1.0 / cse118;
		const amrex::Real cse323 = cse129 * cse322;
		const amrex::Real cse324 = cse221 * cse322;
		const amrex::Real cse325 = cse136 * (-1.2199600771729787e-17 * cse323 - 4.5035826651516278e-15 * cse324);
		const amrex::Real cse326 = state.c_hat * state.xn[14];
		const amrex::Real cse327 = 4.2188032355736613e-18 * cse326;
		const amrex::Real cse328 = state.c_hat * state.xn[15];
		const amrex::Real cse329 = 1.7141251340095377e-18 * cse328;
		const amrex::Real cse330 = cse327 + cse329;
		const amrex::Real cse331 = 1.168047934563505e-18 * cse326;
		const amrex::Real cse332 = 4.1730320353928637e-18 * cse328;
		const amrex::Real cse333 = cse136 * (-1.8637353980807409e-17 * cse323 - 6.8801320537276696e-15 * cse324);
		const amrex::Real cse334 = -cse53 + cse67 + cse81;
		const amrex::Real cse335 = cse0 * (cse111 + cse116 - cse117 - cse137 - cse334 - cse5);
		const amrex::Real cse336 = cse335 * state.T;
		const amrex::Real cse337 = 2.0709734999999996e-16 * cse336;
		const amrex::Real cse338 = -cse337;
		const amrex::Real cse339 = cse193 + cse337;
		const amrex::Real cse340 = cse267 - 3.4516225000000006e-16 * cse336;
		const amrex::Real cse341 = -cse339;
		const amrex::Real cse342 = crate * cse19;
		const amrex::Real cse343 = cse181 * cse342;
		const amrex::Real cse344 = -cse343;
		const amrex::Real cse345 = 5.4939999999999998e-7 * cse123 + 0.49559999999999998;
		const amrex::Real cse346 = std::exp((-cse345) * ln_cse122);
		const amrex::Real cse347 = cse346 * std::exp((3.9029999999999999e-7) * lnT);
		const amrex::Real cse348 = 5115.0 * cse347;
		const amrex::Real cse349 = cse348 + 1;
		const amrex::Real cse350 = std::exp((1.512) * ln_cse120);
		const amrex::Real cse351 = 1.0246966183388124e-22 * cse350;
		const amrex::Real cse352 = cse349 * cse351 + 1;
		const amrex::Real cse353 = 1.0 / cse352;
		const amrex::Real cse354 = 5.5720000000000001e-12 * cse353;
		const amrex::Real cse355 = d2g * state.xn[6];
		const amrex::Real cse356 = cse354 * cse355;
		const amrex::Real cse357 = std::exp((0.00231) * ln_cse2);
		const amrex::Real cse358 = std::exp(-8110.0 * cse2);
		const amrex::Real cse359 = std::exp(-26.600000000000001 * cse2);
		const amrex::Real cse360 = std::exp(-7.6200000000000001 * cse2);
		const amrex::Real cse361 = std::exp(-1.3799999999999999 * cse2);
		const amrex::Real cse362 = 0.00015100000000000001 * cse358 + 1.31e-7 * cse359 + 3.4e-8 * cse360 + 6.9699999999999997e-9 * cse361;
		const amrex::Real cse363 = std::exp((-1.5) * lnT);
		const amrex::Real cse364 = state.xn[16] * state.xn[7] *
					   (-2.4342628390636543e-12 * cse2 * cse357 - 1.5 * cse362 * std::exp((-2.5) * lnT) +
					    cse363 * (1.2246100000000002 * cse1 * cse358 + 3.4846000000000001e-6 * cse1 * cse359 +
						      2.5908000000000001e-7 * cse1 * cse360 + 9.6185999999999982e-9 * cse1 * cse361));
		const amrex::Real cse365 = -cse364;
		const amrex::Real cse366 = 6.7636511339243933e-7 * cse65;
		const amrex::Real cse367 = cse21 * state.xn[2];
		const amrex::Real cse368 = cse367 < 1.0e-50;
		const amrex::Real cse369 = ((cse368) ? (10000000000.0) : (cse86));
		const amrex::Real cse370 = cse369 + 1;
		const amrex::Real cse371 = 1.0 / cse370;
		const amrex::Real cse372 = cse369 * cse371;
		const amrex::Real cse373 = cse372 * state.xn[16];
		const amrex::Real cse374 = cse145 * cse373;
		const amrex::Real cse375 = ((cse368) ? (0) : (5.6603773584905654e-5 * cse83 * state.xn[3]));
		const amrex::Real cse376 = cse311 * state.xn[16];
		const amrex::Real cse377 = cse371 * cse376;
		const amrex::Real cse378 = cse375 * cse377;
		const amrex::Real cse379 = -cse378;
		const amrex::Real cse380 = cse369 * cse376 / (cse370 * cse370);
		const amrex::Real cse381 = cse375 * cse380;
		const amrex::Real cse382 = std::exp(-15230.0 * cse2);
		const amrex::Real cse383 = std::exp(-73.799999999999997 * cse2);
		const amrex::Real cse384 = std::exp(-12.17 * cse2);
		const amrex::Real cse385 = 1.6339999999999999e-6 * cse382 + 9.7930000000000004e-9 * cse383 + 6.3460000000000002e-9 * cse384;
		const amrex::Real cse386 = 0.00071740367932898903 * cse54 + 1;
		const amrex::Real cse387 = std::exp(-49550.0 * cse2);
		const amrex::Real cse388 = 0.15970000000000001 * cse387;
		const amrex::Real cse389 = cse388 + 1.7848999999999999;
		const amrex::Real cse390 = 12.244387999457523 * cse54 + 1;
		const amrex::Real cse391 = 0.21509999999999996 - cse388;
		const amrex::Real cse392 = std::pow(cse386, -cse389) * std::pow(cse390, -cse391);
		const amrex::Real cse393 = 7913.1350000000002 * cse1 * cse387;
		const amrex::Real cse394 = 2.4460185352936309e-10 * cse392 * cse55;
		const amrex::Real cse395 =
		    state.xn[2] * state.xn[8] *
		    (-3.0 / 2.0 * cse385 / std::exp((5.0 / 2.0) * lnT) - 1.2230092676468155e-10 * cse392 * cse56 +
		     cse394 * (-cse393 * std::log(cse386) - 0.00035870183966449452 * cse389 * cse55 / cse386) +
		     cse394 * (cse393 * std::log(cse390) - 6.1221939997287613 * cse391 * cse55 / cse390) +
		     cse56 * (0.024885819999999999 * cse1 * cse382 + 7.2272340000000005e-7 * cse1 * cse383 + 7.7230820000000007e-8 * cse1 * cse384));
		const amrex::Real cse396 = 9.0489001802816564e-11 * std::exp((-1.6200000000000001) * lnT) * state.xn[13] * state.xn[2];
		const amrex::Real cse397 = cse123 * cse18;
		const amrex::Real cse398 = 0.044330000000000001 * cse397 - 0.28520000000000001;
		const amrex::Real cse399 = cse397 * cse398 - 1.6759999999999999;
		const amrex::Real cse400 = cse397 * cse399 + 11.19;
		const amrex::Real cse401 = cse400 * state.xn[6];
		const amrex::Real cse402 = 4.9999999999999997e-12 * cse363 * cse401 * state.xn[2];
		const amrex::Real cse403 = cse70 * cse78;
		const amrex::Real cse404 = 7.5899999999999998e-12 * cse403;
		const amrex::Real cse405 = 9.9999999999999994e-12 * (1.0 / std::sqrt(state.T));
		const amrex::Real cse406 = cse405 * state.xn[2];
		const amrex::Real cse407 = cse406 * state.xn[6] * (cse39 * cse399 + cse397 * (0.044330000000000001 * cse123 * cse37 + cse39 * cse398));
		const amrex::Real cse408 = std::exp((1.8740000000000001) * ln_cse120);
		const amrex::Real cse409 = 7.538e-5 * cse123 + 0.89639999999999997;
		const amrex::Real cse410 = std::exp((-cse409) * ln_cse122);
		const amrex::Real cse411 = cse410 * std::exp((1.635e-6) * lnT);
		const amrex::Real cse412 = 43750.0 * cse411;
		const amrex::Real cse413 = cse412 + 1;
		const amrex::Real cse414 = cse408 * cse413;
		const amrex::Real cse415 = -cse409;
		const amrex::Real cse416 = 3.5685736926834518e-27 * cse408;
		const amrex::Real cse417 = cse413 * cse416 + 1;
		const amrex::Real cse418 = 2.1659999999999999e-12 * cse135 * state.xn[13] / (cse417 * cse417);
		const amrex::Real cse419 =
		    cse418 * (-3.3437535500443945e-27 * cse2 * cse414 - cse416 * (0.071531250000000005 * cse410 * std::exp((-0.99999836499999994) * lnT) +
										  cse412 * (cse131 * cse415 - 7.538e-5 * cse132)));
		const amrex::Real cse420 = cse349 * cse350;
		const amrex::Real cse421 = -cse345;
		const amrex::Real cse422 = 5.5720000000000001e-12 * cse135 * state.xn[6] / (cse352 * cse352);
		const amrex::Real cse423 =
		    cse422 * (-7.7467064346414215e-23 * cse2 * cse420 - cse351 * (0.0019963845000000001 * cse346 * std::exp((-0.99999960970000001) * lnT) +
										  cse348 * (cse131 * cse421 - 5.4939999999999998e-7 * cse132)));
		const amrex::Real cse424 = std::exp((1.1279999999999999) * ln_cse120);
		const amrex::Real cse425 = 0.00013329999999999999 * cse123 + 0.81200000000000006;
		const amrex::Real cse426 = std::exp((-cse425) * ln_cse122);
		const amrex::Real cse427 = cse426 * std::exp((0.04845) * lnT);
		const amrex::Real cse428 = 433.10000000000002 * cse427;
		const amrex::Real cse429 = cse428 + 1;
		const amrex::Real cse430 = cse424 * cse429;
		const amrex::Real cse431 = -cse425;
		const amrex::Real cse432 = 1.6851501542381549e-14 * cse424;
		const amrex::Real cse433 = cse429 * cse432 + 1;
		const amrex::Real cse434 = 4.5579999999999993e-11 * cse135 * state.xn[8] / (cse433 * cse433);
		const amrex::Real cse435 =
		    cse434 * (-9.5042468699031921e-15 * cse2 * cse430 - cse432 * (20.983695000000001 * cse426 * std::exp((-0.95155000000000001) * lnT) +
										  cse428 * (cse131 * cse431 - 0.00013329999999999999 * cse132)));
		const amrex::Real cse436 = cse0 * (cse141 + cse148 + cse365 + cse366 + cse374 + cse379 + cse381 - cse395 + cse396 + cse402 - cse404 - cse407 -
						   cse419 - cse423 - cse435 + 4.2899999999999994e-13 * cse71 + cse95);
		const amrex::Real cse437 = cse436 * state.T;
		const amrex::Real cse438 = 2.0709734999999996e-16 * cse437;
		const amrex::Real cse439 = 1.0 / cse433;
		const amrex::Real cse440 = 4.5579999999999993e-11 * cse439;
		const amrex::Real cse441 = d2g * state.xn[8];
		const amrex::Real cse442 = cse440 * cse441;
		const amrex::Real cse443 = crate * cse158;
		const amrex::Real cse444 = cse443 * state.xn[7];
		const amrex::Real cse445 = 1040.0 * cse444;
		const amrex::Real cse446 = cse442 + cse445;
		const amrex::Real cse447 = 1.0 / cse417;
		const amrex::Real cse448 = 2.1659999999999999e-12 * cse447;
		const amrex::Real cse449 = d2g * state.xn[13];
		const amrex::Real cse450 = cse443 * state.xn[12];
		const amrex::Real cse451 = cse448 * cse449 + 16800.0 * cse450;
		const amrex::Real cse452 = cse356 + cse438 + cse446 + cse451;
		const amrex::Real cse453 = cse192 * cse342;
		const amrex::Real cse454 = cse452 - cse453;
		const amrex::Real cse455 = cse193 + cse454;
		const amrex::Real cse456 = cse347 * cse350 * cse421;
		const amrex::Real cse457 = cse401 * cse405 + cse422 * (1.5493412869282843e-22 * cse420 * cse82 + 5.2413232028030254e-19 * cse456 * cse82);
		const amrex::Real cse458 = cse385 * cse56 + cse394;
		const amrex::Real cse459 = cse424 * cse427 * cse431;
		const amrex::Real cse460 = cse434 * (1.9008493739806384e-14 * cse430 * cse82 + 7.2983853180054489e-12 * cse459 * cse82) + cse458 * state.xn[8];
		const amrex::Real cse461 = 1.4595000290776864e-10 * std::exp((-0.62) * lnT);
		const amrex::Real cse462 = cse408 * cse411 * cse415;
		const amrex::Real cse463 = cse418 * (6.687507100088789e-27 * cse414 * cse82 + 1.5612509905490101e-22 * cse462 * cse82) + cse461 * state.xn[13];
		const amrex::Real cse464 = cse215 + cse217;
		const amrex::Real cse465 = 1.3007021411393064e-6 * cse208;
		const amrex::Real cse466 = ((cse368) ? (0) : (-0.00011320754716981131 * cse210 * state.xn[3]));
		const amrex::Real cse467 = cse377 * cse466;
		const amrex::Real cse468 = cse380 * cse466;
		const amrex::Real cse469 = cse467 - cse468;
		const amrex::Real cse470 = cse207 + cse465 * state.xn[16] + cse469;
		const amrex::Real cse471 = crate * cse21;
		const amrex::Real cse472 = 16800.0 * cse471;
		const amrex::Real cse473 = -cse472 * state.xn[12];
		const amrex::Real cse474 = 2 * cse176;
		const amrex::Real cse475 = cse263 * cse342;
		const amrex::Real cse476 = 3.2999999999999996e-13 * cse253;
		const amrex::Real cse477 = cse476 * state.xn[8];
		const amrex::Real cse478 = 1040.0 * cse471;
		const amrex::Real cse479 = -cse478 * state.xn[7];
		const amrex::Real cse480 = cse477 + cse479;
		const amrex::Real cse481 = ((cse368) ? (0) : (cse85));
		const amrex::Real cse482 = cse377 * cse481;
		const amrex::Real cse483 = cse380 * cse481;
		const amrex::Real cse484 = cse482 - cse483;
		const amrex::Real cse485 = cse260 + cse484;
		const amrex::Real cse486 = cse258 - cse259;
		const amrex::Real cse487 = cse353 * cse355;
		const amrex::Real cse488 = 1.1144e-11 * cse487;
		const amrex::Real cse489 = cse439 * cse441;
		const amrex::Real cse490 = 2080.0 * cse444 + 9.1159999999999986e-11 * cse489;
		const amrex::Real cse491 = cse447 * cse449;
		const amrex::Real cse492 = 33600.0 * cse450 + 4.3319999999999997e-12 * cse491;
		const amrex::Real cse493 = cse229 + 3.4516225000000006e-16 * cse437 + cse488 + cse490 + cse492;
		const amrex::Real cse494 = cse275 * cse342;
		const amrex::Real cse495 = -cse494;
		const amrex::Real cse496 = 1.1000000000000001 * crate;
		const amrex::Real cse497 = -cse438;
		const amrex::Real cse498 = cse135 * cse354 + cse400 * cse406;
		const amrex::Real cse499 = 1.0537934368240927e-9 * cse357 + cse362 * cse363;
		const amrex::Real cse500 = cse499 * state.xn[16];
		const amrex::Real cse501 = -cse500;
		const amrex::Real cse502 = 1.4855039799802863e-17 * interp_8_shielding_H2(state.ncol[ncol_H2]) * interp_8_shielding_self(state.ncol[ncol_C]);
		const amrex::Real cse503 = cse251 * cse502;
		const amrex::Real cse504 = 3.8500000000000001 * crate + cse478 * state.xn[3] + cse503;
		const amrex::Real cse505 = cse476 * state.xn[3];
		const amrex::Real cse506 = cse135 * cse440 + cse458 * state.xn[2];
		const amrex::Real cse507 = cse505 + cse506;
		const amrex::Real cse508 = cse288 * cse373;
		const amrex::Real cse509 = cse289 + cse508;
		const amrex::Real cse510 = 1.9917698086432708e-17 * cse247;
		const amrex::Real cse511 = 1.9088222220940885e-17 * cse251;
		const amrex::Real cse512 = cse472 * state.xn[3] + cse510 + cse511;
		const amrex::Real cse513 = cse135 * cse448 + cse461 * state.xn[2];
		const amrex::Real cse514 = -cse455;
		const amrex::Real cse515 = 1.6715999999999999e-11 * cse487;
		const amrex::Real cse516 = cse311 * cse372;
		const amrex::Real cse517 = cse499 * state.xn[7];
		const amrex::Real cse518 = cse465 * state.xn[2];
		const amrex::Real cse519 = cse314 * cse342;
		const amrex::Real cse520 = -cse519;
		const amrex::Real cse521 = 1.3673999999999998e-10 * cse489;
		const amrex::Real cse522 = 3120.0 * cse444;
		const amrex::Real cse523 = cse521 + cse522;
		const amrex::Real cse524 = 50400.0 * cse450 + 6.4979999999999992e-12 * cse491;
		const amrex::Real cse525 = cse318 + cse319;
		const amrex::Real cse526 = state.c_hat * state.xn[12];
		const amrex::Real cse527 = 1.9917698086432708e-17 * cse526;
		const amrex::Real cse528 = cse322 * cse414;
		const amrex::Real cse529 = cse322 * cse462;
		const amrex::Real cse530 = cse418 * (-9.6554435073969902e-25 * cse528 - 2.2541390258730966e-20 * cse529);
		const amrex::Real cse531 = cse322 * cse420;
		const amrex::Real cse532 = cse322 * cse456;
		const amrex::Real cse533 = cse422 * (-2.2369437588208594e-20 * cse531 - 7.5674387079158044e-17 * cse532);
		const amrex::Real cse534 = cse322 * cse430;
		const amrex::Real cse535 = cse322 * cse459;
		const amrex::Real cse536 = cse434 * (-2.744452225897048e-12 * cse534 - 1.0537431374432729e-9 * cse535);
		const amrex::Real cse537 = 1.9088222220940885e-17 * cse526;
		const amrex::Real cse538 = cse418 * (-1.4750639947665343e-24 * cse528 - 3.4436526025099181e-20 * cse529);
		const amrex::Real cse539 = cse422 * (-3.4173833593729057e-20 * cse531 - 1.1560790928037308e-16 * cse532);
		const amrex::Real cse540 = state.c_hat * cse502 * state.xn[7];
		const amrex::Real cse541 = cse434 * (-4.1927050380195035e-12 * cse534 - 1.6098054538707863e-9 * cse535);
		const amrex::Real cse542 = -cse540 + cse541;
		const amrex::Real cse543 = cse154 * state.xn[3];
		const amrex::Real cse544 = cse10 * cse167;
		const amrex::Real cse545 = state.xn[0] * state.xn[3];
		const amrex::Real cse546 = cse171 * cse545;
		const amrex::Real cse547 = -cse6;
		const amrex::Real cse548 = cse10 * cse52;
		const amrex::Real cse549 = cse545 * cse64;
		const amrex::Real cse550 = cse404 - 4.2899999999999994e-13 * cse68 * cse69 * state.xn[3] * state.xn[8];
		const amrex::Real cse551 = cse140 - cse142;
		const amrex::Real cse552 = -cse76 + cse77;
		const amrex::Real cse553 = cse147 + cse365 + cse374 + cse379 + cse381 + cse552;
		const amrex::Real cse554 = cse0 * (cse141 - cse4 - cse547 - cse548 - cse549 - cse550 - cse551 - cse553 - 2.3622303960273254e-7 * cse65);
		const amrex::Real cse555 = cse554 * state.T;
		const amrex::Real cse556 = 2.0709734999999996e-16 * cse555;
		const amrex::Real cse557 = cse556 - 3.0000000000000002e-15 * d2g * state.xn[0];
		const amrex::Real cse558 = cse200 * cse545;
		const amrex::Real cse559 = cse198 + cse453 + cse557 + cse558;
		const amrex::Real cse560 = -cse559;
		const amrex::Real cse561 = -cse556;
		const amrex::Real cse562 = 4.5427507615910105e-7 * cse208;
		const amrex::Real cse563 = -cse215 + cse469;
		const amrex::Real cse564 = cse214 - cse216;
		const amrex::Real cse565 = cse246 * cse545;
		const amrex::Real cse566 = 1.2216858566394234e-27 * cse249;
		const amrex::Real cse567 = 3.6749360303422164e-18 * cse249;
		const amrex::Real cse568 = cse247 * cse566 + cse251 * cse567;
		const amrex::Real cse569 = 7.2000000000000002e-15 * state.xn[6];
		const amrex::Real cse570 = cse225 + cse569;
		const amrex::Real cse571 = cse474 + cse475;
		const amrex::Real cse572 = cse227 + cse265 - cse482 + cse483;
		const amrex::Real cse573 = cse260 + cse486;
		const amrex::Real cse574 = cse199 + 3.4516225000000006e-16 * cse555;
		const amrex::Real cse575 = cse10 * cse271;
		const amrex::Real cse576 = cse272 * cse545;
		const amrex::Real cse577 = 7.2000000000000002e-15 * state.xn[3];
		const amrex::Real cse578 = 1.6999999999999999e-9 * state.xn[16];
		const amrex::Real cse579 = 9.0000000000000011e-15 * cse243;
		const amrex::Real cse580 = cse310 * cse545;
		const amrex::Real cse581 = 1.6999999999999999e-9 * state.xn[9];
		const amrex::Real cse582 = cse312 + cse516 + cse517 + cse581;
		const amrex::Real cse583 = cse321 * cse566;
		const amrex::Real cse584 = cse321 * cse567;
		const amrex::Real cse585 = -cse0 * cse552;
		const amrex::Real cse586 = cse585 * state.T;
		const amrex::Real cse587 = 2.0709734999999996e-16 * cse586;
		const amrex::Real cse588 = -cse587;
		const amrex::Real cse589 = cse453 + cse588;
		const amrex::Real cse590 = 3.4516225000000006e-16 * cse586;
		const amrex::Real cse591 = -cse9;
		const amrex::Real cse592 = cse4 - cse402 + cse407 + cse423 + cse591;
		const amrex::Real cse593 = cse0 * cse592;
		const amrex::Real cse594 = cse593 * state.T;
		const amrex::Real cse595 = 2.0709734999999996e-16 * cse594;
		const amrex::Real cse596 = -cse595;
		const amrex::Real cse597 = cse356 + cse596;
		const amrex::Real cse598 = cse488 - 3.4516225000000006e-16 * cse594;
		const amrex::Real cse599 = 1.6000000000000001e-9 * state.xn[9];
		const amrex::Real cse600 = cse282 + cse498 + cse577 + cse599;
		const amrex::Real cse601 = 1.6000000000000001e-9 * state.xn[6];
		const amrex::Real cse602 = -cse601;
		const amrex::Real cse603 = -cse0 * cse592;
		const amrex::Real cse604 = cse603 * state.T;
		const amrex::Real cse605 = 2.0709734999999996e-16 * cse604;
		const amrex::Real cse606 = cse356 + cse605;
		const amrex::Real cse607 = -cse606;
		const amrex::Real cse608 = cse488 + 3.4516225000000006e-16 * cse604;
		const amrex::Real cse609 = -cse605;
		const amrex::Real cse610 = cse395 + cse435;
		const amrex::Real cse611 = cse0 * (cse365 + 2.2769999999999997e-12 * cse403 + cse6 + cse610 - 1.287e-13 * cse71 + cse99);
		const amrex::Real cse612 = cse611 * state.T;
		const amrex::Real cse613 = 2.0709734999999996e-16 * cse612;
		const amrex::Real cse614 = -cse613;
		const amrex::Real cse615 = cse443 * state.xn[9];
		const amrex::Real cse616 = 184.0 * cse615;
		const amrex::Real cse617 = -cse616;
		const amrex::Real cse618 = cse446 + cse614 + cse617;
		const amrex::Real cse619 = 9.8999999999999995e-14 * cse253;
		const amrex::Real cse620 = 184.0 * cse471;
		const amrex::Real cse621 = cse620 * state.xn[9];
		const amrex::Real cse622 = 368.0 * cse615;
		const amrex::Real cse623 = -cse622;
		const amrex::Real cse624 = cse621 + cse623;
		const amrex::Real cse625 = cse490 - 3.4516225000000006e-16 * cse612;
		const amrex::Real cse626 = 1.6797535700197549e-17 * interp_10_shielding_H2(state.ncol[ncol_H2]) * interp_10_shielding_self(state.ncol[ncol_CO]);
		const amrex::Real cse627 = cse251 * cse626;
		const amrex::Real cse628 = cse620 * state.xn[3] + cse627;
		const amrex::Real cse629 = cse294 + cse295 + cse297;
		const amrex::Real cse630 = 552.0 * cse615;
		const amrex::Real cse631 = state.c_hat * cse626 * state.xn[9];
		const amrex::Real cse632 = 1.1994963750000002e-8 * cse8 * state.xn[8];
		const amrex::Real cse633 = -cse632;
		const amrex::Real cse634 = cse0 * (-cse550 - cse610 - cse633);
		const amrex::Real cse635 = cse634 * state.T;
		const amrex::Real cse636 = 2.0709734999999996e-16 * cse635;
		const amrex::Real cse637 = cse446 + cse636;
		const amrex::Real cse638 = -cse637;
		const amrex::Real cse639 = cse490 + 3.4516225000000006e-16 * cse635;
		const amrex::Real cse640 = -cse636;
		const amrex::Real cse641 = -cse599;
		const amrex::Real cse642 = 9.1500000000000005e-10 * cse278;
		const amrex::Real cse643 = cse642 * state.xn[15];
		const amrex::Real cse644 = cse642 * state.xn[8];
		const amrex::Real cse645 = -cse0 * cse100;
		const amrex::Real cse646 = cse645 * state.T;
		const amrex::Real cse647 = 2.0709734999999996e-16 * cse646;
		const amrex::Real cse648 = -cse647;
		const amrex::Real cse649 = cse616 + cse648;
		const amrex::Real cse650 = 3.4516225000000006e-16 * cse646;
		const amrex::Real cse651 = cse285 + cse578;
		const amrex::Real cse652 = cse601 + cse628;
		const amrex::Real cse653 = cse0 * (cse633 + cse95);
		const amrex::Real cse654 = cse653 * state.T;
		const amrex::Real cse655 = 2.0709734999999996e-16 * cse654;
		const amrex::Real cse656 = -cse655;
		const amrex::Real cse657 = -3.4516225000000006e-16 * cse654;
		const amrex::Real cse658 = -cse374 + cse378 - cse381;
		const amrex::Real cse659 = cse141 + cse334;
		const amrex::Real cse660 = cse0 * (-cse658 - cse659);
		const amrex::Real cse661 = cse660 * state.T;
		const amrex::Real cse662 = 2.0709734999999996e-16 * cse661;
		const amrex::Real cse663 = cse616 + cse662;
		const amrex::Real cse664 = cse622 + 3.4516225000000006e-16 * cse661;
		const amrex::Real cse665 = -cse662;
		const amrex::Real cse666 = cse332 + cse631;
		const amrex::Real cse667 = -cse396 + cse419;
		const amrex::Real cse668 = cse0 * cse667;
		const amrex::Real cse669 = cse668 * state.T;
		const amrex::Real cse670 = 2.0709734999999996e-16 * cse669;
		const amrex::Real cse671 = -cse670;
		const amrex::Real cse672 = cse451 + cse671;
		const amrex::Real cse673 = cse492 - 3.4516225000000006e-16 * cse669;
		const amrex::Real cse674 = cse527 - cse530;
		const amrex::Real cse675 = cse537 - cse538;
		const amrex::Real cse676 = -cse0 * cse667;
		const amrex::Real cse677 = cse676 * state.T;
		const amrex::Real cse678 = 2.0709734999999996e-16 * cse677;
		const amrex::Real cse679 = -cse451 - cse678;
		const amrex::Real cse680 = cse492 + 3.4516225000000006e-16 * cse677;
		const amrex::Real cse681 = -cse678;
		const amrex::Real cse682 = cse0 * (cse364 + 5.3129999999999997e-12 * cse403 + cse547 - 3.0029999999999999e-13 * cse71);
		const amrex::Real cse683 = cse682 * state.T;
		const amrex::Real cse684 = 2.0709734999999996e-16 * cse683;
		const amrex::Real cse685 = -cse684;
		const amrex::Real cse686 = 2.3099999999999997e-13 * cse253;
		const amrex::Real cse687 = -3.4516225000000006e-16 * cse683;
		const amrex::Real cse688 = cse0 * (cse551 + cse632 + cse658 + cse9 + cse99);
		const amrex::Real cse689 = cse688 * state.T;
		const amrex::Real cse690 = 2.0709734999999996e-16 * cse689;
		const amrex::Real cse691 = -cse690;
		const amrex::Real cse692 = -3.4516225000000006e-16 * cse689;
		const amrex::Real cse693 = cse0 * (cse366 + cse553);
		const amrex::Real cse694 = cse693 * state.T;
		const amrex::Real cse695 = 2.0709734999999996e-16 * cse694;
		const amrex::Real cse696 = -cse695;
		const amrex::Real cse697 = -3.4516225000000006e-16 * cse694;
		const amrex::Real cse698 = cse0 * (cse143 + cse591 + cse659);
		const amrex::Real cse699 = cse698 * state.T;
		const amrex::Real cse700 = 2.0709734999999996e-16 * cse699;
		const amrex::Real cse701 = 3.4516225000000006e-16 * cse699;
		const amrex::Real cse702 = -cse700;
		const amrex::Real cse703 = cse367 + 0.070000000000000007;
		const amrex::Real cse704 = 1.0 / cse703;
		const amrex::Real cse705 = std::sqrt(cse367 * cse704);
		const amrex::Real cse706 = 26.399999999999999 * cse705;
		const amrex::Real cse707 = cse706 + 6.5;
		const amrex::Real cse708 = 2.25 * std::exp((0.14699999999999999) * lnT) + 5.2199999999999998;
		const amrex::Real cse709 = cse122 + 50;
		const amrex::Real ln_cse709 = std::log(cse709);
		const amrex::Real cse710 = 0.0043 * std::exp((0.69199999999999995) * ln_cse709) + 1;
		const amrex::Real cse711 = 0.049959999999999997 * std::exp((0.43099999999999999) * ln_cse709) * cse710 + 1;
		const amrex::Real cse712 = 1.0 / cse711;
		const amrex::Real cse713 = state.xn[2] + 1.0e-50;
		const amrex::Real cse714 = 1.0 / cse713;
		const amrex::Real cse715 = cse54 * cse714;
		const amrex::Real cse716 = 5.670589618368584e-11 * cse119;
		const amrex::Real cse717 = cse715 * cse716 + 50;
		const amrex::Real cse718 = std::log(cse717);
		const amrex::Real cse719 = 0.45350000000000001 + 2.234 / cse718;
		const amrex::Real cse720 = std::exp((cse719)*lnT);
		const amrex::Real cse721 = 1.4419999999999999 - 0.050889999999999998 * cse718;
		const amrex::Real cse722 = std::exp(cse718 * cse721);
		const amrex::Real cse723 = 1.8998126555754159e-29 * cse720 * cse722;
		const amrex::Real cse724 = d2g * state.xn[2];
		const amrex::Real cse725 = cse723 * cse724;
		const amrex::Real cse726 = std::exp((0.14999999999999999) * lnT);
		const amrex::Real cse727 = 0.00088240812266906664 * std::exp((0.65000000000000002) * lnT) + 1;
		const amrex::Real cse728 = 1.0 / cse727;
		const amrex::Real cse729 = cse728 * state.xn[2];
		const amrex::Real cse730 = cse726 * cse729;
		const amrex::Real cse731 = 4.4460389837719569e-9 * cse730 + 626500000.0;
		const amrex::Real cse732 = 1.0 / cse731;
		const amrex::Real cse733 = std::exp(-118400.0 * cse2);
		const amrex::Real cse734 = cse732 * cse733;
		const amrex::Real cse735 = 1.3338116951315871e-8 * cse734;
		const amrex::Real cse736 = cse730 * cse735 + 1;
		const amrex::Real cse737 = 1.0 / cse736;
		const amrex::Real cse738 = cse734 * cse737;
		const amrex::Real cse739 = 1.3659996292638161e-10 * cse738;
		const amrex::Real cse740 = psigd_coll_interp2d(cse20, state.T);
		const amrex::Real cse741 = 100.0 * cse135;
		const amrex::Real cse742 = 2.8677558904086741e-6 * cse119 + 4.1397668473788896;
		const amrex::Real ln_cse742 = std::log(cse742);
		const amrex::Real cse743 = -2.1544346900318838 * std::exp((1.0 / 6.0) * ln_cse742) + state.T;
		const amrex::Real cse744 = cse47 * d2g;
		const amrex::Real cse745 = 3.2000000000000002e-32 * cse54;
		const amrex::Real cse746 = cse744 * cse745;
		const amrex::Real cse747 = -cse740 * cse741 + cse743 * cse746;
		const amrex::Real cse748 = (((cse747 > 0) ? (0) : ((cse747 == 0) ? (1.0 / 2.0) : (1))));
		const amrex::Real cse749 = cse743 * cse745 * cse748 * d2g;
		const amrex::Real cse750 =
		    cse749 * (cse234 + cse242 + 2 * state.xn[1] + 2 * state.xn[10] + 2 * state.xn[14] + 2 * state.xn[15] + 6 * state.xn[16] + 4 * state.xn[4]);
		const amrex::Real cse751 = cse740 * d2g;
		const amrex::Real cse752 = psigd_coll_interp2d_partial_0(cse20, state.T);
		const amrex::Real cse753 = (((cse747 < 0) ? (0) : ((cse747 == 0) ? (1.0 / 2.0) : (1))));
		const amrex::Real cse754 = cse753 * (cse741 * cse752 + 100.0 * cse751);
		const amrex::Real cse755 = cse157 * state.xn[2];
		const amrex::Real cse756 = (1.0 / (cse703 * cse703));
		const amrex::Real cse757 = 0.5 * cse756;
		const amrex::Real cse758 = cse204 / (cse20 * cse20 * cse20);
		const amrex::Real cse759 = -0.5 * cse704 * cse755 + cse757 * cse758;
		const amrex::Real cse760 = cse18 * std::log(cse20);
		const amrex::Real cse761 = cse760 < 2;
		const amrex::Real cse762 = cse18 * cse21;
		const amrex::Real cse763 = cse760 < 4;
		const amrex::Real cse764 = cse760 < 7;
		const amrex::Real cse765 = cse760 < 10;
		const amrex::Real cse766 =
		    ((cse761) ? (0) : ((cse763) ? ((3.0 / 2.0) * cse762) : ((cse764) ? ((4.0 / 3.0) * cse762) : ((cse765) ? ((1.0 / 3.0) * cse762) : (0)))));
		const amrex::Real cse767 = 0.001 * state.T;
		const amrex::Real cse768 = cse767 + 1;
		const amrex::Real cse769 = (1.0 / (cse768 * cse768));
		const amrex::Real cse770 = 1.0 / cse768;
		const amrex::Real cse771 = std::exp(2.302585092994046 * (-2.0230000000000001 * cse769 - 3.6709999999999998 * cse770 - 11.08));
		const amrex::Real cse772 = cse21 * cse771;
		const amrex::Real cse773 = cse772 * state.xn[3];
		const amrex::Real cse774 = std::exp(2.302585092994046 * (-2.3900000000000001 * cse769 + 0.055500000000000001 * cse770 - 11.06));
		const amrex::Real cse775 = cse21 * cse774;
		const amrex::Real cse776 = cse775 * state.xn[0];
		const amrex::Real cse777 = cse773 + cse776;
		const amrex::Real cse778 = 1.0 / cse777;
		const amrex::Real cse779 = cse568 + 1.9999999999999999e-7;
		const amrex::Real cse780 = cse778 * cse779;
		const amrex::Real cse781 = cse21 * cse780 + 1;
		const amrex::Real cse782 = 1.0 / cse781;
		const amrex::Real cse783 = 6.7291434000000005e-12 * cse782 + 3.2043540000000002e-13;
		const amrex::Real cse784 = cse7 * state.xn[2];
		const amrex::Real cse785 = (1.0 / 100.0) * state.T;
		const amrex::Real ln_cse785 = std::log(cse785);
		const amrex::Real cse786 = std::log(cse785);
		const amrex::Real cse787 = 0.0086999999999999994 * cse786;
		const amrex::Real cse788 = std::exp((cse787 + 0.12809999999999999) * ln_cse785);
		const amrex::Real cse789 = 7.5799999999999997e-10 * cse788 * state.xn[0];
		const amrex::Real cse790 = state.T < 500;
		const amrex::Real cse791 = std::exp((0.070000000000000007) * lnT);
		const amrex::Real cse792 = 0.75 * ((cse790) ? (1.1000000000000001e-13 * state.T + 5.3300000000000002e-10) : (3.74757785025e-10 * cse791)) +
					   0.25 * ((cse790) ? (3.3000000000000001e-13 * state.T + 4.4299999999999997e-10) : (3.8899728635600003e-10 * cse791));
		const amrex::Real cse793 = 4.5299999999999998e-6 * cse784 + cse789 + cse792 * state.xn[3];
		const amrex::Real cse794 = cse793 + 2.3e-6;
		const amrex::Real cse795 = std::exp(-91.209999999999994 * cse2);
		const amrex::Real cse796 = cse795 / cse794;
		const amrex::Real cse797 = 2 * cse796;
		const amrex::Real cse798 = cse793 * cse797 + 1;
		const amrex::Real cse799 = 1.0 / cse798;
		const amrex::Real cse800 = cse796 * state.xn[8];
		const amrex::Real cse801 = cse799 * cse800;
		const amrex::Real cse802 = (1.0 / (cse794 * cse794));
		const amrex::Real cse803 = std::sqrt(std::sqrt(state.T));
		const amrex::Real cse804 = 1.75 * cse803;
		const amrex::Real cse805 = std::exp(-state.T);
		const amrex::Real cse806 = 1 - cse805;
		const amrex::Real ln_cse806 = std::log(cse806);
		const amrex::Real cse807 = std::exp((1000) * ln_cse806);
		const amrex::Real cse808 = l0_co_interp1d(state.T);
		const amrex::Real cse809 = 1.0 / cse808;
		const amrex::Real cse810 = 680.10000000000002 / cse803;
		const amrex::Real cse811 = cse804 * state.xn[0] + cse810 * state.xn[2] + state.xn[3];
		const amrex::Real cse812 = M_SQRT2;
		const amrex::Real cse813 = std::max(gradv, 5.7445531974402559e-18 * cse54 * cse812);
		const amrex::Real cse814 = 100000.0 / cse813;
		const amrex::Real cse815 = state.ncol[ncol_H] <= 0;
		const amrex::Real cse816 = cse21 * state.ncol[ncol_H];
		const amrex::Real cse817 = std::max(1.0000000000000001e-30, cse816);
		const amrex::Real cse818 = 1756701454.4310031 * state.ncol[ncol_CO];
		const amrex::Real cse819 = ((cse815) ? (cse814 * state.xn[9]) : (cse818 / std::sqrt(cse817)));
		const amrex::Real cse820 = llte_co_interp2d(state.T, cse819);
		const amrex::Real cse821 = 1.0 / cse820;
		const amrex::Real cse822 = nhalf_co_interp2d(state.T, cse819);
		const amrex::Real cse823 = cse808 * cse821;
		const amrex::Real cse824 = -cse822 * cse823 + 1;
		const amrex::Real cse825 = 1.0 / cse822;
		const amrex::Real cse826 = cse811 * cse825;
		const amrex::Real cse827 = alpha_co_interp2d(state.T, cse819);
		const amrex::Real cse828 = std::pow(cse826, cse827);
		const amrex::Real cse829 = cse809 * cse828;
		const amrex::Real cse830 = cse824 * cse829;
		const amrex::Real cse831 = cse809 + cse811 * cse821 + cse830;
		const amrex::Real cse832 = 1.0 / cse831;
		const amrex::Real cse833 = cse807 * cse832;
		const amrex::Real cse834 = cse833 * state.xn[9];
		const amrex::Real cse835 = (1.0 / (cse175 * cse175));
		const amrex::Real cse836 = cse21 * cse707;
		const amrex::Real cse837 =
		    ((cse761) ? (10.0)
			      : ((cse763) ? ((3.0 / 2.0) * cse760 + 7)
					  : ((cse764) ? ((4.0 / 3.0) * cse760 + 23.0 / 3.0) : ((cse765) ? ((1.0 / 3.0) * cse760 + 44.0 / 3.0) : (18.0)))));
		const amrex::Real cse838 = 2.0 * cse21;
		const amrex::Real cse839 = cse837 * cse838;
		const amrex::Real cse840 = cse836 * state.xn[0] + cse839 * state.xn[3];
		const amrex::Real cse841 = 1.5159999999999999e-9 * cse788;
		const amrex::Real cse842 = cse793 * cse795 * cse802;
		const amrex::Real cse843 = cse793 * cse796;
		const amrex::Real cse844 = 5.7927337833400002e-20 * cse843 * state.xn[8] / (cse798 * cse798);
		const amrex::Real cse845 = (1.0 / (cse781 * cse781));
		const amrex::Real cse846 = cse157 * cse780;
		const amrex::Real cse847 = cse158 * cse771;
		const amrex::Real cse848 = cse160 * cse774;
		const amrex::Real cse849 = cse847 + cse848;
		const amrex::Real cse850 = cse779 / (cse777 * cse777);
		const amrex::Real cse851 = cse21 * cse850;
		const amrex::Real cse852 = cse846 - cse851 * (-cse775 + cse849);
		const amrex::Real cse853 = 1.0 / cse175;
		const amrex::Real cse854 = cse160 * cse707;
		const amrex::Real cse855 = cse703 * cse82;
		const amrex::Real cse856 = cse759 * cse855;
		const amrex::Real cse857 = cse706 * state.xn[0];
		const amrex::Real cse858 = cse766 * state.xn[3];
		const amrex::Real cse859 = cse158 * cse837;
		const amrex::Real cse860 = 2.0 * cse859;
		const amrex::Real cse861 = cse157 * state.ncol[ncol_H] *
					   (((cse816 - 1.0000000000000001e-30 < 0) ? (0) : ((cse816 - 1.0000000000000001e-30 == 0) ? (1.0 / 2.0) : (1)))) /
					   (cse817 * std::sqrt(cse817));
		const amrex::Real cse862 = cse861 * state.ncol[ncol_CO];
		const amrex::Real cse863 = ((cse815) ? (0) : (878350727.21550155 * cse862));
		const amrex::Real cse864 = alpha_co_interp2d_partial_1(state.T, cse819);
		const amrex::Real cse865 = std::log(cse826);
		const amrex::Real cse866 = cse864 * cse865;
		const amrex::Real cse867 = cse811 / (cse822 * cse822);
		const amrex::Real cse868 = nhalf_co_interp2d_partial_1(state.T, cse819);
		const amrex::Real cse869 = cse863 * cse868;
		const amrex::Real cse870 = cse827 / cse811;
		const amrex::Real cse871 = cse822 * cse870;
		const amrex::Real cse872 = (1.0 / (cse820 * cse820));
		const amrex::Real cse873 = llte_co_interp2d_partial_1(state.T, cse819);
		const amrex::Real cse874 = -cse811 * cse863 * cse872 * cse873 + cse829 * (cse808 * cse822 * cse863 * cse872 * cse873 - cse823 * cse869);
		const amrex::Real cse875 = cse811 * state.xn[9];
		const amrex::Real cse876 = cse807 * cse875 / (cse831 * cse831);
		const amrex::Real cse877 = state.T < 100.0;
		const amrex::Real cse878 = state.T < 6000.0;
		const amrex::Real cse879 = state.T < 10;
		const amrex::Real cse880 = (1.0 / (cse17 * cse17 * cse17 * cse17 * cse17 * cse17 * cse17 * cse17));
		const amrex::Real cse881 = (1.0 / (cse17 * cse17 * cse17 * cse17 * cse17 * cse17 * cse17));
		const amrex::Real cse882 = (1.0 / (cse17 * cse17 * cse17 * cse17 * cse17 * cse17));
		const amrex::Real cse883 = (1.0 / (cse17 * cse17 * cse17 * cse17 * cse17));
		const amrex::Real cse884 = (1.0 / (cse17 * cse17 * cse17 * cse17));
		const amrex::Real cse885 = (1.0 / (cse17 * cse17 * cse17));
		const amrex::Real cse886 =
		    std::exp(2.302585092994046 * (-38.719649225812759 * cse18 + 512.9223506962187 * cse23 + 55793.874041176059 * cse880 -
						  124963.59998693959 * cse881 + 119499.73396083503 * cse882 - 63669.349536898742 * cse883 +
						  20652.971887880416 * cse884 - 4189.7100510639848 * cse885 - 21.928795999999998)) *
		    state.xn[2];
		const amrex::Real cse887 = std::log(cse767);
		const amrex::Real cse888 = cse18 * cse887;
		const amrex::Real cse889 = cse883 * (cse887 * cse887 * cse887 * cse887 * cse887);
		const amrex::Real cse890 = cse884 * (cse887 * cse887 * cse887 * cse887);
		const amrex::Real cse891 = (cse887 * cse887 * cse887);
		const amrex::Real cse892 = cse885 * cse891;
		const amrex::Real cse893 = cse23 * (cse887 * cse887);
		const amrex::Real cse894 = state.xn[5] + state.xn[6];
		const amrex::Real cse895 =
		    std::exp(2.302585092994046 * (1.5714710999999999 * cse888 + 0.32168730000000001 * cse889 - 0.51002221000000003 * cse890 -
						  0.23619984999999999 * cse892 + 0.015391166 * cse893 - 22.089523)) *
			state.xn[1] +
		    std::exp(2.302585092994046 * (2.1892372 * cse888 + 0.19191374999999999 * cse889 - 0.16596184 * cse890 + 0.29036281000000003 * cse892 -
						  0.81520437999999995 * cse893 - 23.689236999999999)) *
			cse894 +
		    std::exp(2.302585092994046 * (2.0943374000000001 * cse18 * cse887 + 0.43693353000000001 * cse885 * cse891 - 0.033638326000000003 * cse889 -
						  0.14913216000000001 * cse890 - 0.77151435999999995 * cse893 - 23.962112000000001)) *
			state.xn[3];
		const amrex::Real cse896 = std::exp(2.302585092994046 * (37.383713 * cse888 + 3.8479610000000002 * cse889 + 20.159831000000001 * cse890 +
									 48.656103000000002 * cse892 + 58.145166000000003 * cse893 - 16.818342000000001)) *
					       state.xn[0] +
					   cse895;
		const amrex::Real cse897 = cse880 * (cse887 * cse887 * cse887 * cse887 * cse887 * cse887 * cse887 * cse887);
		const amrex::Real cse898 = cse881 * (cse887 * cse887 * cse887 * cse887 * cse887 * cse887 * cse887);
		const amrex::Real cse899 = cse882 * (cse887 * cse887 * cse887 * cse887 * cse887 * cse887);
		const amrex::Real cse900 =
		    std::exp(2.302585092994046 * (1.6802758 * cse888 - 8.8077017000000009 * cse889 - 4.7274035999999997 * cse890 + 4.0406627000000004 * cse892 +
						  0.93310621999999999 * cse893 - 6.3701156000000001 * cse897 + 6.4380698000000001 * cse898 +
						  8.9167182999999994 * cse899 - 22.921188999999998)) *
		    state.xn[2];
		const amrex::Real cse901 = state.T < 1000.0;
		const amrex::Real cse902 =
		    std::exp(2.302585092994046 * (3.5692468000000002 * cse18 * cse887 - 4.2519023000000002 * cse889 - 21.328264000000001 * cse890 -
						  27.850082 * cse892 - 11.33286 * cse893 - 24.311209000000002)) *
			state.xn[0] +
		    cse895;
		const amrex::Real cse903 =
		    std::exp(2.302585092994046 * (16.815729999999999 * cse888 + 983.67575999999997 * cse889 + 734.71650999999997 * cse890 + 343.1918 * cse892 +
						  96.743155000000002 * cse893 + 70.609154000000004 * cse897 + 364.14445999999998 * cse898 +
						  801.81246999999996 * cse899 - 21.928795999999998)) *
		    state.xn[2];
		const amrex::Real cse904 = std::exp(2.302585092994046 * (4.6450521 * cse888 + 1.5538288 * cse889 - 5.5108049000000001 * cse890 +
									 5.9369081000000001 * cse892 - 3.7209846 * cse893 - 24.311209000000002)) *
					       state.xn[0] +
					   cse895;
		const amrex::Real cse905 =
		    std::exp(2.302585092994046 * (2.8156982240432273 * cse18 + 0.049411830040524525 * cse23 + 5.9406293599919788 * cse883 -
						  5.2566362076906961 * cse884 - 1.3586848802337363 * cse885 - 22.089523)) *
			state.xn[1] +
		    std::exp(2.302585092994046 * (3.7525488682084647 * cse18 - 2.4768712409536775 * cse23 - 0.62120210234156448 * cse883 -
						  1.5370575959566981 * cse884 + 2.5133588394664677 * cse885 - 23.962112000000001)) *
			state.xn[3] +
		    std::exp(2.302585092994046 * (3.9225864834863131 * cse18 - 2.6171337683481011 * cse23 + 3.5440891133599637 * cse883 -
						  1.7105157385968939 * cse884 + 1.6702447513374001 * cse885 - 23.689236999999999)) *
			cse894 +
		    std::exp(2.302585092994046 * (8.3228160852326614 * cse18 - 11.945856385319289 * cse23 + 28.69470131298657 * cse883 -
						  56.79810800956944 * cse884 + 34.150687525022562 * cse885 - 24.311209000000002)) *
			state.xn[0];
		const amrex::Real cse906 =
		    ((cse790 && cse877 && cse878)
			 ? (cse879 || cse886 + cse896 < 1.0e-100)
			 : (((cse877 && cse878)
				 ? (cse879 || cse896 + cse900 < 1.0e-100)
				 : (((cse790 && cse877 && cse878 && cse901)
					 ? (cse879 || cse886 + cse902 < 1.0e-100)
					 : (((cse790 && cse878 && cse901)
						 ? (cse879 || cse902 + cse903 < 1.0e-100)
						 : (((cse878 && cse901)
							 ? (cse879 || cse900 + cse902 < 1.0e-100)
							 : (((cse790 && cse878)
								 ? (cse879 || cse903 + cse904 < 1.0e-100)
								 : (((cse878)
									 ? (cse879 || cse900 + cse904 < 1.0e-100)
									 : (((cse790) ? (cse879 ||
											 std::exp(2.302585092994046 *
												  (30.129743459482278 * cse18 + 310.58441786958315 * cse23 +
												   7500.646099484944 * cse880 + 21588.957206746247 * cse881 +
												   26530.843406310876 * cse882 + 18165.631967965237 * cse883 +
												   7572.4886742758581 * cse884 + 1974.131269262874 * cse885 -
												   21.928795999999998)) *
												     state.xn[2] +
												 cse905 <
											     1.0e-100)
										      : (cse879 ||
											 std::exp(2.302585092994046 *
												  (3.0106500755647456 * cse18 + 2.9956460707652877 * cse23 -
												   676.68255490510751 * cse880 + 381.69251128040054 * cse881 +
												   295.04162851880631 * cse882 - 162.65264843551881 * cse883 -
												   48.723840736518795 * cse884 + 23.242975457496804 * cse885 -
												   22.921188999999998)) *
												     state.xn[2] +
												 cse905 <
											     1.0e-100))))))))))))))));
		const amrex::Real cse907 = std::log(0.001 * ((cse878) ? (state.T) : (6000.0)));
		const amrex::Real cse908 = cse18 * cse907;
		const amrex::Real cse909 = (cse907 * cse907 * cse907 * cse907 * cse907);
		const amrex::Real cse910 = cse883 * cse909;
		const amrex::Real cse911 = (cse907 * cse907 * cse907 * cse907);
		const amrex::Real cse912 = cse884 * cse911;
		const amrex::Real cse913 = (cse907 * cse907 * cse907);
		const amrex::Real cse914 = cse885 * cse913;
		const amrex::Real cse915 = (cse907 * cse907);
		const amrex::Real cse916 = cse23 * cse915;
		const amrex::Real cse917 = std::exp(2.302585092994046 * (37.383713 * cse908 + 3.8479610000000002 * cse910 + 20.159831000000001 * cse912 +
									 48.656103000000002 * cse914 + 58.145166000000003 * cse916 - 16.818342000000001));
		const amrex::Real cse918 = ((cse878) ? (cse877) : (false));
		const amrex::Real cse919 =
		    std::exp(2.302585092994046 * (3.5692468000000002 * cse18 * cse907 - 4.2519023000000002 * cse910 - 21.328264000000001 * cse912 -
						  27.850082 * cse914 - 11.33286 * cse916 - 24.311209000000002));
		const amrex::Real cse920 = ((cse878) ? (cse901) : (false));
		const amrex::Real cse921 = std::exp(2.302585092994046 * (4.6450521 * cse908 + 1.5538288 * cse910 - 5.5108049000000001 * cse912 +
									 5.9369081000000001 * cse914 - 3.7209846 * cse916 - 24.311209000000002));
		const amrex::Real cse922 = ((cse918) ? (cse917) : ((cse920) ? (cse919) : (cse921)));
		const amrex::Real cse923 =
		    std::exp(2.302585092994046 * (2.0943374000000001 * cse18 * cse907 + 0.43693353000000001 * cse885 * cse913 - 0.033638326000000003 * cse910 -
						  0.14913216000000001 * cse912 - 0.77151435999999995 * cse916 - 23.962112000000001));
		const amrex::Real cse924 = cse923 * state.xn[3];
		const amrex::Real cse925 =
		    std::exp(2.302585092994046 * (1.5714710999999999 * cse908 + 0.32168730000000001 * cse910 - 0.51002221000000003 * cse912 -
						  0.23619984999999999 * cse914 + 0.015391166 * cse916 - 22.089523));
		const amrex::Real cse926 = cse925 * state.xn[1];
		const amrex::Real cse927 = std::exp(2.302585092994046 * (2.1892372 * cse908 + 0.19191374999999999 * cse910 - 0.16596184 * cse912 +
									 0.29036281000000003 * cse914 - 0.81520437999999995 * cse916 - 23.689236999999999));
		const amrex::Real cse928 = cse894 * cse927;
		const amrex::Real cse929 = cse880 * (cse907 * cse907 * cse907 * cse907 * cse907 * cse907 * cse907 * cse907);
		const amrex::Real cse930 = (cse907 * cse907 * cse907 * cse907 * cse907 * cse907 * cse907);
		const amrex::Real cse931 = cse881 * cse930;
		const amrex::Real cse932 = (cse907 * cse907 * cse907 * cse907 * cse907 * cse907);
		const amrex::Real cse933 = cse882 * cse932;
		const amrex::Real cse934 =
		    std::exp(2.302585092994046 * (16.815729999999999 * ((cse918) ? (-2.3025850929940455 * cse18) : (cse908)) +
						  96.743155000000002 * ((cse918) ? (5.3018981104783967 * cse23) : (cse916)) +
						  70.609154000000004 * ((cse918) ? (790.17904733961348 * cse880) : (cse929)) +
						  364.14445999999998 * ((cse918) ? (-343.17039997516258 * cse881) : (cse931)) +
						  801.81246999999996 * ((cse918) ? (149.03701106174495 * cse882) : (cse933)) +
						  983.67575999999997 * ((cse918) ? (-64.725951503469744 * cse883) : (cse910)) +
						  734.71650999999997 * ((cse918) ? (28.110123573894395 * cse884) : (cse912)) +
						  343.1918 * ((cse918) ? (-12.208071553760854 * cse885) : (cse914)) - 21.928795999999998));
		const amrex::Real cse935 =
		    std::exp(2.302585092994046 * (1.6802758 * cse908 - 8.8077017000000009 * cse910 - 4.7274035999999997 * cse912 + 4.0406627000000004 * cse914 +
						  0.93310621999999999 * cse916 - 6.3701156000000001 * cse929 + 6.4380698000000001 * cse931 +
						  8.9167182999999994 * cse933 - 22.921188999999998));
		const amrex::Real cse936 = ((cse790) ? (cse934) : (cse935));
		const amrex::Real cse937 = cse922 * state.xn[0] + cse924 + cse926 + cse928 + cse936 * state.xn[2];
		const amrex::Real cse938 = (1.0 / (cse937 * cse937));
		const amrex::Real cse939 = ((cse878) ? (cse2) : (0.00016666666666666666));
		const amrex::Real cse940 = std::exp(-11700.0 * cse939);
		const amrex::Real cse941 = std::exp(-5860.0 * cse939);
		const amrex::Real cse942 = std::exp(-510.0 * cse939);
		const amrex::Real cse943 = std::exp(-2197000.0 * ((cse878) ? ((1.0 / (state.T * state.T * state.T))) : (4.6296296296296294e-12)));
		const amrex::Real cse944 = cse943 * ((cse878) ? (std::exp((3.7599999999999998) * lnT)) : (160638152785302.53));
		const amrex::Real cse945 = 6.0142468035272636e-8 * ((cse878) ? (std::exp((2.1000000000000001) * lnT)) : (85924621.118090734)) + 1.0;
		const amrex::Real cse946 = 1.0 / cse945;
		const amrex::Real cse947 = 1.6e-18 * cse940 + 6.7e-19 * cse941 + 3.0e-24 * cse942 + 4.985670872372847e-33 * cse944 * cse946;
		const amrex::Real cse948 = 1.0 / cse937;
		const amrex::Real cse949 = cse947 * cse948 + 1;
		const amrex::Real cse950 = (1.0 / (cse949 * cse949));
		const amrex::Real cse951 = cse938 * (cse947 * cse947) * cse950;
		const amrex::Real cse952 = 0.0060000000000000001 * cse786;
		const amrex::Real cse953 = std::exp((0.36899999999999999 - cse952) * ln_cse785);
		const amrex::Real cse954 = std::exp((0.92600000000000005) * lnT);
		const amrex::Real cse955 = 1.08e-14 * cse954;
		const amrex::Real cse956 = 0.16 * cse786;
		const amrex::Real cse957 = std::exp((0.755 - cse956) * ln_cse785);
		const amrex::Real cse958 = 4.34e-10 * cse957;
		const amrex::Real cse959 = cse958 * state.xn[0];
		const amrex::Real cse960 = 0.042999999999999997 * cse786;
		const amrex::Real cse961 = 5.2499999999999998e-13 * std::exp((cse960 + 0.88900000000000001) * ln_cse785);
		const amrex::Real cse962 = 0.52500000000000002 * cse786;
		const amrex::Real cse963 = 2.2500000000000003e-12 * std::exp((cse962 + 1.198) * ln_cse785);
		const amrex::Real cse964 = cse961 + cse963;
		const amrex::Real cse965 = cse955 * state.xn[2] + cse959 + cse964 * state.xn[3];
		const amrex::Real cse966 = std::exp(-425.40000000000003 * cse2);
		const amrex::Real cse967 = cse965 * cse966;
		const amrex::Real cse968 = std::exp(-326.60000000000002 * cse2);
		const amrex::Real cse969 = std::exp((-0.025999999999999999) * lnT);
		const amrex::Real cse970 = 4.8599999999999998e-10 * cse969;
		const amrex::Real cse971 = 3.1899999999999998e-10 * cse953;
		const amrex::Real cse972 = cse971 * state.xn[0];
		const amrex::Real cse973 = 0.041000000000000002 * cse786;
		const amrex::Real cse974 = 4.7499999999999998e-11 * std::exp((cse973 + 0.20300000000000001) * ln_cse785);
		const amrex::Real cse975 = 0.058000000000000003 * cse786;
		const amrex::Real cse976 = 1.6725000000000002e-10 * std::exp((cse975 + 0.23699999999999999) * ln_cse785);
		const amrex::Real cse977 = cse974 + cse976;
		const amrex::Real cse978 = cse970 * state.xn[2] + cse972 + cse977 * state.xn[3];
		const amrex::Real cse979 = cse968 * cse978;
		const amrex::Real cse980 = 0.0030000000000000001 * cse786;
		const amrex::Real cse981 = std::exp((0.41899999999999998 - cse980) * ln_cse785);
		const amrex::Real cse982 = 7.1400000000000007e-11 * cse981;
		const amrex::Real cse983 = cse965 * cse968;
		const amrex::Real cse984 = 8.68e-11 * cse957;
		const amrex::Real cse985 = std::exp((-0.074999999999999997) * lnT);
		const amrex::Real cse986 = 5.1199999999999999e-10 * cse985;
		const amrex::Real cse987 = 3.5700000000000001e-10 * cse981;
		const amrex::Real cse988 = cse987 * state.xn[0];
		const amrex::Real cse989 = 0.025000000000000001 * cse786;
		const amrex::Real cse990 = 3.7250000000000001e-11 * std::exp((cse989 + 0.26400000000000001) * ln_cse785);
		const amrex::Real cse991 = 1.0274999999999999e-10 * std::exp((cse960 + 0.29599999999999999) * ln_cse785);
		const amrex::Real cse992 = cse990 + cse991;
		const amrex::Real cse993 = cse986 * state.xn[2] + cse988 + cse992 * state.xn[3];
		const amrex::Real cse994 = cse968 * cse993;
		const amrex::Real cse995 = cse966 * cse978;
		const amrex::Real cse996 = 6.3800000000000002e-11 * cse953;
		const amrex::Real cse997 = cse993 + 8.9099999999999997e-5;
		const amrex::Real cse998 = cse968 * cse997;
		const amrex::Real cse999 = 2.1266666666666666e-11 * cse953 * cse967 + 2.8933333333333332e-11 * cse957 * cse995 + cse979 * cse982 +
					   cse982 * cse983 + cse984 * cse994 + cse996 * cse998;
		const amrex::Real cse1000 = cse965 + 1.7499999999999998e-5;
		const amrex::Real cse1001 = cse978 + 1.34e-10;
		const amrex::Real cse1002 = std::exp(-98.800000000000011 * cse2);
		const amrex::Real cse1003 = cse1002 * cse965;
		const amrex::Real cse1004 = (1.0 / 3.0) * cse1003;
		const amrex::Real cse1005 = std::exp(-227.80000000000001 * cse2);
		const amrex::Real cse1006 = cse1005 * cse993;
		const amrex::Real cse1007 = (3.0 / 5.0) * cse1006;
		const amrex::Real cse1008 = (1.0 / 5.0) * cse968;
		const amrex::Real cse1009 = cse1008 * cse978;
		const amrex::Real cse1010 = cse1000 * cse1007 + cse1000 * cse1009 + cse1001 * cse1007;
		const amrex::Real cse1011 = cse1008 * cse993;
		const amrex::Real cse1012 = (1.0 / 15.0) * cse978;
		const amrex::Real cse1013 = cse1009 * cse997 + cse1011 * cse965 + cse1012 * cse967;
		const amrex::Real cse1014 = cse1000 * cse997 + cse1001 * cse1004 + cse1001 * cse997 + cse1010 + cse1013;
		const amrex::Real cse1015 = 1.0 / cse1014;
		const amrex::Real cse1016 = 2.3872025442750958e-19 * cse1015;
		const amrex::Real cse1017 = cse1000 * cse968;
		const amrex::Real cse1018 = 2.1419999999999999e-10 * cse981;
		const amrex::Real cse1019 = cse1001 * cse1005;
		const amrex::Real cse1020 = cse1000 * cse1005;
		const amrex::Real cse1021 = 1.9139999999999998e-10 * cse1006 * cse953 + 2.604e-10 * cse1006 * cse957 + cse1017 * cse996 + cse1018 * cse1019 +
					    cse1018 * cse1020 + cse979 * cse984;
		const amrex::Real cse1022 = 2.8023005140019997e-18 * cse1015;
		const amrex::Real cse1023 = (1.0 / (cse1014 * cse1014));
		const amrex::Real cse1024 = cse1001 * cse1002;
		const amrex::Real cse1025 = cse1023 * (-cse1000 * cse987 - cse1001 * cse987 - 1.0633333333333333e-10 * cse1003 * cse953 - cse1021 -
						       1.4466666666666666e-10 * cse1024 * cse957 - cse958 * cse997 - cse971 * cse997 - cse999);
		const amrex::Real cse1026 = 2.3872025442750958e-19 * cse1013;
		const amrex::Real cse1027 = 2.8023005140019997e-18 * cse1010;
		const amrex::Real cse1028 = 0.045999999999999999 * cse786;
		const amrex::Real cse1029 = std::exp((cse1028 + 0.22800000000000001) * ln_cse785);
		const amrex::Real cse1030 = std::exp((cse1028 + 0.23100000000000001) * ln_cse785);
		const amrex::Real cse1031 = 2.6400000000000002e-10 * cse1030;
		const amrex::Real cse1032 = cse1031 * state.xn[0];
		const amrex::Real cse1033 = 0.031 * cse786;
		const amrex::Real cse1034 = 1.11e-10 * std::exp((cse1033 + 0.26300000000000001) * ln_cse785);
		const amrex::Real cse1035 = 0.064000000000000001 * cse786;
		const amrex::Real cse1036 = 4.3749999999999998e-11 * std::exp((cse1035 + 0.071999999999999995) * ln_cse785);
		const amrex::Real cse1037 = cse1034 + cse1036;
		const amrex::Real cse1038 = 0.00235272 * cse123;
		const amrex::Real cse1039 = cse1038 - 0.041816600000000002;
		const amrex::Real cse1040 = cse1039 * cse123 + 0.35826400000000003;
		const amrex::Real cse1041 = cse1040 * cse123 - 0.57443;
		const amrex::Real cse1042 = 0.097857299999999994 * cse123;
		const amrex::Real cse1043 = cse1042 - 3.1926800000000002;
		const amrex::Real cse1044 = cse1043 * cse123 + 38.504899999999999;
		const amrex::Real cse1045 = cse1044 * cse123 - 202.19300000000001;
		const amrex::Real cse1046 = std::exp(((cse901) ? (cse1041 * cse123 - 7.4386999999999999) : (cse1045 * cse123 + 386.18599999999998)));
		const amrex::Real cse1047 = 1.7258000000000001e-6 * cse784;
		const amrex::Real cse1048 = cse1046 * cse1047;
		const amrex::Real cse1049 = cse1032 + cse1037 * state.xn[3] + cse1048;
		const amrex::Real cse1050 = std::exp(-101.30000000000001 * cse2);
		const amrex::Real cse1051 = cse1049 * cse1050;
		const amrex::Real cse1052 = 8.9000000000000003e-11 * cse1029;
		const amrex::Real cse1053 = cse1052 * state.xn[0];
		const amrex::Real cse1054 = 0.037999999999999999 * cse786;
		const amrex::Real cse1055 = 5.1749999999999996e-11 * std::exp((cse1054 + 0.16900000000000001) * ln_cse785);
		const amrex::Real cse1056 = 0.048000000000000001 * cse786;
		const amrex::Real cse1057 = 2.15e-11 * std::exp((cse1056 - 0.01) * ln_cse785);
		const amrex::Real cse1058 = cse1055 + cse1057;
		const amrex::Real cse1059 = 0.0070527699999999999 * cse123;
		const amrex::Real cse1060 = cse1059 - 0.11133800000000001;
		const amrex::Real cse1061 = cse1060 * cse123 + 0.69763799999999998;
		const amrex::Real cse1062 = cse1061 * cse123 - 1.3074300000000001;
		const amrex::Real cse1063 = 0.093813800000000003 * cse123;
		const amrex::Real cse1064 = cse1063 - 3.0328300000000001;
		const amrex::Real cse1065 = cse1064 * cse123 + 36.180300000000003;
		const amrex::Real cse1066 = cse1065 * cse123 - 187.47399999999999;
		const amrex::Real cse1067 = std::exp(((cse901) ? (cse1062 * cse123 - 7.6973500000000001) : (cse1066 * cse123 + 350.60899999999998)));
		const amrex::Real cse1068 = cse1047 * cse1067;
		const amrex::Real cse1069 = cse1053 + cse1058 * state.xn[3] + cse1068;
		const amrex::Real cse1070 = cse1050 * cse1069;
		const amrex::Real cse1071 = 0.057000000000000002 * cse786;
		const amrex::Real cse1072 = std::exp((cse1071 + 0.115) * ln_cse785);
		const amrex::Real cse1073 = 1.26e-10 * cse1072;
		const amrex::Real cse1074 = cse1073 * state.xn[0];
		const amrex::Real cse1075 = 0.049000000000000002 * cse786;
		const amrex::Real cse1076 = 5.3249999999999997e-11 * std::exp((cse1075 - 0.0040000000000000001) * ln_cse785);
		const amrex::Real cse1077 = 0.10199999999999999 * cse786;
		const amrex::Real cse1078 = 1.675e-11 * std::exp((cse1077 - 0.085000000000000006) * ln_cse785);
		const amrex::Real cse1079 = cse1076 + cse1078;
		const amrex::Real cse1080 = 0.00065632500000000003 * cse123;
		const amrex::Real cse1081 = -cse1080 - 0.015089200000000001;
		const amrex::Real cse1082 = cse1081 * cse123 + 0.36118400000000001;
		const amrex::Real cse1083 = cse1082 * cse123 - 0.77378199999999997;
		const amrex::Real cse1084 = 0.10508000000000001 * cse123;
		const amrex::Real cse1085 = cse1084 - 3.4762;
		const amrex::Real cse1086 = cse1085 * cse123 + 42.594999999999999;
		const amrex::Real cse1087 = cse1086 * cse123 - 227.91300000000001;
		const amrex::Real cse1088 = std::exp(((cse901) ? (cse1083 * cse123 - 9.2514099999999999) : (cse1087 * cse123 + 444.60000000000002)));
		const amrex::Real cse1089 = 2.8763333333333333e-6 * cse1088;
		const amrex::Real cse1090 = cse1089 * cse784;
		const amrex::Real cse1091 = cse1074 + cse1079 * state.xn[3] + cse1090;
		const amrex::Real cse1092 = std::exp(-62.460000000000001 * cse2);
		const amrex::Real cse1093 = cse1091 * cse1092;
		const amrex::Real cse1094 = 1.32e-9 * cse1030;
		const amrex::Real cse1095 = 6.3e-10 * cse1072;
		const amrex::Real cse1096 = cse1069 * cse1092;
		const amrex::Real cse1097 = cse1049 * cse1092;
		const amrex::Real cse1098 = cse1091 + 7.8800000000000004e-8;
		const amrex::Real cse1099 = cse1092 * cse1098;
		const amrex::Real cse1100 = 4.4500000000000001e-10 * cse1029;
		const amrex::Real cse1101 = 7.4166666666666679e-10 * cse1029 * cse1051 + 2.2000000000000003e-9 * cse1030 * cse1070 + cse1093 * cse1094 +
					    cse1095 * cse1096 + cse1095 * cse1097 + cse1099 * cse1100;
		const amrex::Real cse1102 = cse1049 + 2.65e-7;
		const amrex::Real cse1103 = cse1069 + 1.81e-14;
		const amrex::Real cse1104 = std::exp(-38.840000000000003 * cse2);
		const amrex::Real cse1105 = cse1049 * cse1104;
		const amrex::Real cse1106 = (5.0 / 3.0) * cse1105;
		const amrex::Real cse1107 = (25.0 / 3.0) * cse1049;
		const amrex::Real cse1108 = 5 * cse1092;
		const amrex::Real cse1109 = cse1091 * cse1108;
		const amrex::Real cse1110 = cse1069 * cse1108;
		const amrex::Real cse1111 = cse1049 * cse1109 + cse1070 * cse1107 + cse1098 * cse1110;
		const amrex::Real cse1112 = std::exp(-23.619999999999997 * cse2);
		const amrex::Real cse1113 = cse1091 * cse1112;
		const amrex::Real cse1114 = 3 * cse1113;
		const amrex::Real cse1115 = cse1102 * cse1110 + cse1102 * cse1114 + cse1103 * cse1114;
		const amrex::Real cse1116 = cse1098 * cse1102 + cse1098 * cse1103 + cse1103 * cse1106 + cse1111 + cse1115;
		const amrex::Real cse1117 = 1.0 / cse1116;
		const amrex::Real cse1118 = 1.4210469458259591e-21 * cse1117;
		const amrex::Real cse1119 = cse1092 * cse1102;
		const amrex::Real cse1120 = 3.7800000000000004e-10 * cse1072;
		const amrex::Real cse1121 = cse1103 * cse1112;
		const amrex::Real cse1122 = cse1102 * cse1112;
		const amrex::Real cse1123 = 2.6700000000000001e-10 * cse1029 * cse1113 + 7.9200000000000006e-10 * cse1030 * cse1113 + cse1094 * cse1096 +
					    cse1100 * cse1119 + cse1120 * cse1121 + cse1120 * cse1122;
		const amrex::Real cse1124 = 2.5697412351439998e-22 * cse1117;
		const amrex::Real cse1125 = (1.0 / (cse1116 * cse1116));
		const amrex::Real cse1126 = cse1103 * cse1104;
		const amrex::Real cse1127 = cse1125 * (-1.4833333333333334e-10 * cse1029 * cse1105 - 4.4000000000000003e-10 * cse1030 * cse1126 -
						       cse1031 * cse1098 - cse1052 * cse1098 - cse1073 * cse1102 - cse1073 * cse1103 - cse1101 - cse1123);
		const amrex::Real cse1128 = 1.4210469458259591e-21 * cse1111;
		const amrex::Real cse1129 = 2.5697412351439998e-22 * cse1115;
		const amrex::Real cse1130 = state.T < 1000000.0;
		const amrex::Real cse1131 = cse734 * std::exp((-1.8500000000000001) * lnT);
		const amrex::Real cse1132 = cse110 * cse737;
		const amrex::Real cse1133 = std::exp((-0.84999999999999998) * lnT);
		const amrex::Real cse1134 = std::exp((-0.19999999999999998) * lnT);
		const amrex::Real cse1135 = (1.0 / (cse727 * cse727));
		const amrex::Real cse1136 = cse1133 * cse729;
		const amrex::Real cse1137 = (1.0 / (cse731 * cse731));
		const amrex::Real cse1138 = cse1137 * cse733 * (2.5500935934394033e-12 * cse1134 * cse1135 * state.xn[2] - 6.6690584756579352e-10 * cse1136);
		const amrex::Real cse1139 = cse726 * cse728;
		const amrex::Real cse1140 = 1.3659996292638161e-10 * cse1139;
		const amrex::Real cse1141 = (1.0 / (cse711 * cse711));
		const amrex::Real cse1142 = std::exp((0.12299999999999994) * ln_cse709);
		const amrex::Real cse1143 = cse1142 * cse119;
		const amrex::Real cse1144 = std::exp((-0.56899999999999995) * ln_cse709) * cse710;
		const amrex::Real cse1145 = cse1144 * cse119;
		const amrex::Real cse1146 = cse119 * cse55 * cse714;
		const amrex::Real cse1147 = 1.0 / cse717;
		const amrex::Real cse1148 = cse1147 * cse123 / (cse718 * cse718);
		const amrex::Real cse1149 = cse135 * cse723;
		const amrex::Real cse1150 = cse1149 * state.xn[2];
		const amrex::Real cse1151 = cse1147 * cse718;
		const amrex::Real cse1152 = cse1147 * cse721;
		const amrex::Real cse1153 = cse110 * cse1140 * cse734 / (cse736 * cse736);
		const amrex::Real cse1154 = cse1 * cse843;
		const amrex::Real cse1155 = cse2 * cse784;
		const amrex::Real cse1156 = std::exp((-0.92999999999999994) * lnT);
		const amrex::Real cse1157 = 100 * cse2;
		const amrex::Real cse1158 = -2.2649999999999999e-6 * cse1155 +
					    cse789 * (cse1157 * (8.7000000000000001e-5 * cse786 + 0.001281) + cse2 * cse787) +
					    state.xn[3] * (0.75 * ((cse790) ? (1.1000000000000001e-13) : (2.6233044951750002e-11 * cse1156)) +
							   0.25 * ((cse790) ? (3.3000000000000001e-13) : (2.7229810044920004e-11 * cse1156)));
		const amrex::Real cse1159 = 5.7927337833400002e-20 * cse799;
		const amrex::Real cse1160 = cse1159 * cse800;
		const amrex::Real cse1161 = 0.4375 * state.xn[0] / std::exp((3.0 / 4.0) * lnT) - 170.02500000000001 * state.xn[2] / std::exp((5.0 / 4.0) * lnT);
		const amrex::Real cse1162 = -cse1158;
		const amrex::Real cse1163 = cse1159 * cse842 * state.xn[8];
		const amrex::Real cse1164 = (1.0 / (cse768 * cse768 * cse768));
		const amrex::Real cse1165 = -cse17 * cse773 * (0.0040460000000000001 * cse1164 + 0.0036709999999999998 * cse769) -
					    cse17 * cse776 * (0.0047800000000000004 * cse1164 - 5.5500000000000001e-5 * cse769);
		const amrex::Real cse1166 = 2.0187430200000004e-26 * cse243;
		const amrex::Real cse1167 = 2.5634832e-11 * cse568 * cse845 * state.xn[3];
		const amrex::Real cse1168 = 2 * cse842;
		const amrex::Real cse1169 = l0_co_interp1d_partial_0(state.T);
		const amrex::Real cse1170 = (1.0 / (cse808 * cse808));
		const amrex::Real cse1171 = ((cse815) ? (-2.872276598720128e-13 * cse55 * cse812 * state.xn[9] *
							 (((5.7445531974402559e-18 * cse54 * cse812 - gradv < 0)
							       ? (0)
							       : ((5.7445531974402559e-18 * cse54 * cse812 - gradv == 0) ? (1.0 / 2.0) : (1)))) /
							 (cse813 * cse813))
						      : (0));
		const amrex::Real cse1172 = cse872 * (-cse1171 * cse873 - llte_co_interp2d_partial_0(state.T, cse819));
		const amrex::Real cse1173 = cse1171 * cse868 + nhalf_co_interp2d_partial_0(state.T, cse819);
		const amrex::Real cse1174 = ((cse878) ? (-cse1) : (0));
		const amrex::Real cse1175 =
		    -1.8719999999999998e-14 * cse1174 * cse940 - 3.9261999999999998e-15 * cse1174 * cse941 - 1.53e-21 * cse1174 * cse942 +
		    4.985670872372847e-33 * cse943 * cse946 * ((cse878) ? (3.7599999999999998 * std::exp((2.7599999999999998) * lnT)) : (0)) -
		    1.0953518906603145e-26 * cse944 * cse946 * ((cse878) ? (-3 / (state.T * state.T * state.T * state.T)) : (0)) -
		    2.9985055107607381e-40 * cse944 * ((cse878) ? (2.1000000000000001 * std::exp((1.1000000000000001) * lnT)) : (0)) / (cse945 * cse945);
		const amrex::Real cse1176 = 1.0 / cse949;
		const amrex::Real cse1177 = ((cse878) ? (1) : (0));
		const amrex::Real cse1178 = cse1177 * cse939;
		const amrex::Real cse1179 = cse1178 * cse23 * cse907;
		const amrex::Real cse1180 = cse1178 * cse883 * cse911;
		const amrex::Real cse1181 = cse1178 * cse884 * cse913;
		const amrex::Real cse1182 = cse1178 * cse18;
		const amrex::Real cse1183 = cse1178 * cse885 * cse915;
		const amrex::Real cse1184 = cse1178 * cse880 * cse930;
		const amrex::Real cse1185 = cse1178 * cse881 * cse932;
		const amrex::Real cse1186 = cse1178 * cse882 * cse909;
		const amrex::Real cse1187 = 65.320000000000007 * cse979;
		const amrex::Real cse1188 = cse2 * cse960;
		const amrex::Real cse1189 = 0.00042999999999999999 * cse786;
		const amrex::Real cse1190 = cse988 * (cse1157 * (0.0041900000000000001 - 3.0000000000000001e-5 * cse786) - cse2 * cse980) -
					    3.8399999999999998e-11 * std::exp((-1.075) * lnT) * state.xn[2] +
					    state.xn[3] * (cse990 * (cse1157 * (0.00025000000000000001 * cse786 + 0.00264) + cse2 * cse989) +
							   cse991 * (cse1157 * (cse1189 + 0.00296) + cse1188));
		const amrex::Real cse1191 = cse1008 * cse965;
		const amrex::Real cse1192 = cse972 * (cse1157 * (0.0036900000000000001 - 6.0000000000000002e-5 * cse786) - cse2 * cse952) -
					    1.2635999999999999e-11 * std::exp((-1.026) * lnT) * state.xn[2] +
					    state.xn[3] * (cse974 * (cse1157 * (0.00041000000000000005 * cse786 + 0.0020300000000000001) + cse2 * cse973) +
							   cse976 * (cse1157 * (0.00058 * cse786 + 0.0023700000000000001) + cse2 * cse975));
		const amrex::Real cse1193 = (1.0 / 15.0) * cse967;
		const amrex::Real cse1194 = cse959 * (cse1157 * (0.0075500000000000003 - 0.0016000000000000001 * cse786) - cse2 * cse956) +
					    1.0000800000000001e-14 * std::exp((-0.073999999999999955) * lnT) * state.xn[2] +
					    state.xn[3] * (cse961 * (cse1157 * (cse1189 + 0.0088900000000000003) + cse1188) +
							   cse963 * (cse1157 * (0.0052500000000000003 * cse786 + 0.011979999999999999) + cse2 * cse962));
		const amrex::Real cse1195 = cse1012 * cse966;
		const amrex::Real cse1196 = cse1192 * cse997;
		const amrex::Real cse1197 = cse1 * cse1187 * cse997 + 65.320000000000007 * cse1 * cse965 * cse994 +
					    28.360000000000003 * cse1 * cse967 * cse978 + cse1008 * cse1196 + cse1009 * cse1190 + cse1011 * cse1194 +
					    cse1190 * cse1191 + cse1192 * cse1193 + cse1194 * cse1195;
		const amrex::Real cse1198 = 136.68000000000001 * cse1006;
		const amrex::Real cse1199 = cse1 * cse1001;
		const amrex::Real cse1200 = cse1 * cse1000;
		const amrex::Real cse1201 = cse1001 * cse1190;
		const amrex::Real cse1202 = (3.0 / 5.0) * cse1005;
		const amrex::Real cse1203 = cse1000 * cse1190;
		const amrex::Real cse1204 = cse1000 * cse1008;
		const amrex::Real cse1205 = cse1007 * cse1192 + cse1007 * cse1194 + cse1009 * cse1194 + cse1187 * cse1200 + cse1192 * cse1204 +
					    cse1198 * cse1199 + cse1198 * cse1200 + cse1201 * cse1202 + cse1202 * cse1203;
		const amrex::Real cse1206 = (1.0 / 3.0) * cse1024;
		const amrex::Real cse1207 = cse1023 * (-32.933333333333337 * cse1003 * cse1199 - cse1004 * cse1192 - cse1194 * cse1206 - cse1194 * cse997 -
						       cse1196 - cse1197 - cse1201 - cse1203 - cse1205);
		const amrex::Real cse1208 = 312.30000000000001 * cse1;
		const amrex::Real cse1209 = cse1028 * cse2;
		const amrex::Real cse1210 = 0.00046000000000000001 * cse786;
		const amrex::Real cse1211 = 8.6290000000000003e-7 * cse1155;
		const amrex::Real cse1212 = cse1032 * (cse1157 * (cse1210 + 0.00231) + cse1209) - cse1046 * cse1211 +
					    cse1048 * ((cse901) ? (cse1041 * cse2 + cse123 * (cse1040 * cse2 + cse123 * (cse1038 * cse2 + cse1039 * cse2)))
								: (cse1045 * cse2 + cse123 * (cse1044 * cse2 + cse123 * (cse1042 * cse2 + cse1043 * cse2)))) +
					    state.xn[3] * (cse1034 * (cse1033 * cse2 + cse1157 * (0.00031 * cse786 + 0.00263)) +
							   cse1036 * (cse1035 * cse2 + cse1157 * (0.00064000000000000005 * cse786 + 0.00071999999999999994)));
		const amrex::Real cse1213 = (25.0 / 3.0) * cse1070;
		const amrex::Real cse1214 = cse1074 * (cse1071 * cse2 + cse1157 * (0.00057000000000000009 * cse786 + 0.00115)) -
					    1.4381666666666666e-6 * cse1088 * cse1155 +
					    cse1090 * ((cse901) ? (cse1083 * cse2 + cse123 * (cse1082 * cse2 + cse123 * (-cse1080 * cse2 + cse1081 * cse2)))
								: (cse1087 * cse2 + cse123 * (cse1086 * cse2 + cse123 * (cse1084 * cse2 + cse1085 * cse2)))) +
					    state.xn[3] * (cse1076 * (cse1075 * cse2 + cse1157 * (0.00048999999999999998 * cse786 - 4.0000000000000003e-5)) +
							   cse1078 * (cse1077 * cse2 + cse1157 * (0.0010199999999999999 * cse786 - 0.00085000000000000006)));
		const amrex::Real cse1215 = cse1053 * (cse1157 * (cse1210 + 0.0022800000000000003) + cse1209) - cse1067 * cse1211 +
					    cse1068 * ((cse901) ? (cse1062 * cse2 + cse123 * (cse1061 * cse2 + cse123 * (cse1059 * cse2 + cse1060 * cse2)))
								: (cse1066 * cse2 + cse123 * (cse1065 * cse2 + cse123 * (cse1063 * cse2 + cse1064 * cse2)))) +
					    state.xn[3] * (cse1055 * (cse1054 * cse2 + cse1157 * (0.00038000000000000002 * cse786 + 0.0016900000000000001)) +
							   cse1057 * (cse1056 * cse2 + cse1157 * (0.00048000000000000001 * cse786 - 0.0001)));
		const amrex::Real cse1216 = cse1050 * cse1107;
		const amrex::Real cse1217 = cse1049 * cse1108;
		const amrex::Real cse1218 = cse1098 * cse1215;
		const amrex::Real cse1219 = 844.16666666666686 * cse1 * cse1049 * cse1070 + cse1049 * cse1093 * cse1208 + cse1069 * cse1099 * cse1208 +
					    cse1108 * cse1218 + cse1109 * cse1212 + cse1110 * cse1214 + cse1212 * cse1213 + cse1214 * cse1217 +
					    cse1215 * cse1216;
		const amrex::Real cse1220 = 70.859999999999985 * cse1 * cse1113;
		const amrex::Real cse1221 = cse1103 * cse1214;
		const amrex::Real cse1222 = 3 * cse1112;
		const amrex::Real cse1223 = cse1102 * cse1108;
		const amrex::Real cse1224 = cse1102 * cse1214;
		const amrex::Real cse1225 = cse1096 * cse1102 * cse1208 + cse1102 * cse1220 + cse1103 * cse1220 + cse1110 * cse1212 + cse1114 * cse1212 +
					    cse1114 * cse1215 + cse1215 * cse1223 + cse1221 * cse1222 + cse1222 * cse1224;
		const amrex::Real cse1226 = (5.0 / 3.0) * cse1126;
		const amrex::Real cse1227 = cse1125 * (-64.733333333333348 * cse1 * cse1103 * cse1105 - cse1098 * cse1212 - cse1106 * cse1215 -
						       cse1212 * cse1226 - cse1218 - cse1219 - cse1221 - cse1224 - cse1225);
		const amrex::Real cse1228 =
		    cse0 *
		    (5.6705896183685846e-35 * state.c_hat * cse1141 * cse118 * cse20 * cse708 * d2g *
			 (-4.2149769358107055e-15 * cse1143 * cse83 - 6.1051722655411154e-13 * cse1145 * cse83) +
		     1.8755475162754094e-35 * state.c_hat * cse118 * cse20 * cse712 * d2g * std::exp((-0.85299999999999998) * lnT) -
		     2.0489994438957239e-11 * cse110 * cse1133 * cse728 * cse738 - 2.1789607199999999e-11 * cse111 -
		     1.6173435610483583e-5 * cse1131 * cse1132 * cse728 - cse1132 * cse1138 * cse1140 +
		     7.8348995947646122e-14 * cse1134 * cse1135 * cse732 * cse733 * cse737 * state.xn[0] * state.xn[2] -
		     cse1150 * (-6.3340486037177084e-11 * cse1146 * cse1148 + cse2 * cse719) -
		     cse1150 * (-1.4428815283938862e-12 * cse1146 * cse1151 + 2.835294809184292e-11 * cse1146 * cse1152) -
		     cse1153 * (-0.0015792330470357991 * cse1131 * cse729 + 7.6502807803182096e-12 * cse1134 * cse1135 * cse732 * cse733 * state.xn[2] -
				2.0007175426973804e-9 * cse1136 * cse734 - 1.3338116951315871e-8 * cse1138 * cse730) -
		     5.2835524837844135e-18 * cse1154 * cse799 * state.xn[8] - cse1158 * cse1160 - cse1161 * cse834 - cse1162 * cse1163 -
		     cse1165 * cse1166 * cse845 * cse850 - cse1165 * cse1167 * cse851 - 7.1777529600000012e-12 * cse548 - 7.1777529600000012e-12 * cse549 -
		     cse741 * cse753 * psigd_coll_interp2d_partial_1(cse20, state.T) - cse748 * (1.6000000000000001e-32 * cse55 * cse743 * cse744 + cse746) -
		     1000 * cse805 * std::exp((999) * ln_cse806) * cse832 * cse875 -
		     cse844 * (-182.41999999999999 * cse1154 - cse1158 * cse797 - cse1162 * cse1168) -
		     cse876 * (-cse1161 * cse821 + cse1169 * cse1170 * cse824 * cse828 + cse1169 * cse1170 - cse1172 * cse811 -
			       cse829 * (-cse1169 * cse821 * cse822 - cse1172 * cse808 * cse822 - cse1173 * cse823) -
			       cse830 * (cse865 * (cse1171 * cse864 + alpha_co_interp2d_partial_0(state.T, cse819)) +
					 cse871 * (cse1161 * cse825 - cse1173 * cse867))) -
		     state.xn[11] * (cse1016 * cse1197 + cse1022 * cse1205 + cse1026 * cse1207 + cse1027 * cse1207) -
		     state.xn[3] *
			 ((cse906)
			      ? (0)
			      : (cse1175 * cse1176 +
				 cse947 * cse950 *
				     (-cse1175 * cse948 -
				      cse938 * cse947 *
					  (-cse924 * (4.8223900769399082 * cse1177 * cse18 * cse939 + 3.0182298984217999 * cse1177 * cse885 * cse915 * cse939 -
						      3.5529549287336835 * cse1179 - 0.3872755400043702 * cse1180 - 1.3735579540080118 * cse1181) -
					   cse926 * (0.0708789387907936 * cse1179 + 3.7035619079275177 * cse1180 - 4.6974781513675152 * cse1181 +
						     3.6184459289309552 * cse1182 - 1.6316107607322889 * cse1183) -
					   cse928 * (-3.7541549062629067 * cse1179 + 2.2094886994529301 * cse1180 - 1.528565035159452 * cse1181 +
						     5.0409049417480247 * cse1182 + 2.0057552335975877 * cse1183) -
					   state.xn[0] * ((cse918) ? (cse17 * cse917 *
								      (116.29033200000001 * cse1179 + 19.239805 * cse1180 + 80.639324000000002 * cse1181 +
								       37.383713 * cse1182 + 145.968309 * cse1183))
								   : ((cse920) ? (cse919 * (8.2184944748967013 * cse1177 * cse18 * cse939 -
											    52.189748993977005 * cse1179 - 48.951834264235494 * cse1180 -
											    196.44057098336626 * cse1181 - 192.38155095558542 * cse1183))
									       : (cse921 * (-17.135767342440825 * cse1179 + 17.889115159724135 * cse1180 -
											    50.756388852554174 * cse1181 + 10.695627721640689 * cse1182 +
											    41.010708268606813 * cse1183)))) -
					   state.xn[2] * ((cse790) ? (cse934 * (222.75934655221241 * ((cse918) ? (0) : (2.0 * cse1179)) +
										2264.9971413155886 * ((cse918) ? (0) : (5.0 * cse1180)) +
										1691.7472835026108 * ((cse918) ? (0) : (4.0 * cse1181)) +
										38.719649225812766 * ((cse918) ? (0) : (1.0 * cse1182)) +
										790.22832271779396 * ((cse918) ? (0) : (3.0 * cse1183)) +
										162.58358542932092 * ((cse918) ? (0) : (8.0 * cse1184)) +
										838.47360529236653 * ((cse918) ? (0) : (7.0 * cse1185)) +
										1846.2414407987355 * ((cse918) ? (0) : (6.0 * cse1186))))
								   : (cse935 * (4.297112944704045 * cse1179 - 101.40241318979159 * cse1180 -
										43.540996231705549 * cse1181 + 3.8689780091986448 * cse1182 +
										27.91190909651122 * cse1183 - 117.34186576967058 * cse1184 +
										103.76942484394611 * cse1185 + 123.18901581604325 * cse1186))))))) -
		     state.xn[7] * ((cse1130) ? (cse1118 * cse1219 + cse1124 * cse1225 + cse1128 * cse1227 + cse1129 * cse1227) : (0)));
		const amrex::Real cse1229 = cse1228 * state.T;
		const amrex::Real cse1230 = 2.0709734999999996e-16 * cse1229;
		const amrex::Real cse1231 = 5.6705896183685846e-35 * cse119;
		const amrex::Real cse1232 = cse708 * cse712;
		const amrex::Real cse1233 = cse1232 * d2g;
		const amrex::Real cse1234 = 4.22974728e-11 * cse182 * cse20 * cse705;
		const amrex::Real cse1235 = 3.2043540000000001e-12 * crate;
		const amrex::Real cse1236 = 1.7623947000000002e-12 * crate * cse835 * cse840 * state.xn[5];
		const amrex::Real cse1237 = cse157 * cse778 * cse779 - cse849 * cse851;
		const amrex::Real cse1238 = cse1166 * cse20 * cse845;
		const amrex::Real cse1239 = 1.7623947000000002e-12 * crate * cse853 * state.xn[5];
		const amrex::Real cse1240 = cse825 * cse827;
		const amrex::Real cse1241 = -cse1230;
		const amrex::Real cse1242 =
		    cse1167 * cse1237 + cse1231 * cse1233 + cse1234 * cse856 + cse1235 * cse858 + cse1236 * cse179 + cse1237 * cse1238 +
		    cse1239 * (2.0 * cse21 * cse766 * state.xn[3] + 26.399999999999999 * cse703 * cse705 * cse759 * cse82 * state.xn[0] - cse854 - cse860) +
		    cse1241 - 7.1777529600000012e-12 * cse198 + 3.0000000000000002e-15 * cse243 * cse783 - 7.1777529600000012e-12 * cse558 - cse725 - cse750 -
		    cse754 - cse876 * (-cse830 * (-cse1240 * cse869 + cse863 * cse864 * cse865) - cse874);
		const amrex::Real cse1243 = std::exp((0.29999999999999999) * lnT);
		const amrex::Real cse1244 = (1.0 / (cse713 * cse713));
		const amrex::Real cse1245 = cse1244 * cse54;
		const amrex::Real cse1246 = cse720 * cse722;
		const amrex::Real cse1247 = cse1148 * cse1246 * state.xn[2];
		const amrex::Real cse1248 = 0.5 * cse21 * cse704 - cse755 * cse757;
		const amrex::Real cse1249 = 9.0599999999999997e-6 * cse7;
		const amrex::Real cse1250 = 1.024e-10 * cse985;
		const amrex::Real cse1251 = 2.1600000000000004e-15 * cse954;
		const amrex::Real cse1252 = 9.7199999999999998e-11 * cse969;
		const amrex::Real cse1253 = cse1250 * cse979 + cse1250 * cse983 + cse1251 * cse994 + cse1252 * cse998 +
					    7.1999999999999997e-16 * cse954 * cse995 + 3.2399999999999999e-11 * cse967 * cse969;
		const amrex::Real cse1254 = 3.0719999999999998e-10 * cse985;
		const amrex::Real cse1255 = 6.4799999999999996e-15 * cse1006 * cse954 + 2.9159999999999995e-10 * cse1006 * cse969 + cse1017 * cse1252 +
					    cse1019 * cse1254 + cse1020 * cse1254 + cse1251 * cse979;
		const amrex::Real cse1256 = cse1023 * (-cse1000 * cse986 - cse1001 * cse986 - 1.6199999999999998e-10 * cse1003 * cse969 -
						       3.6000000000000001e-15 * cse1024 * cse954 - cse1253 - cse1255 - cse955 * cse997 - cse970 * cse997);
		const amrex::Real cse1257 = cse1046 * cse7;
		const amrex::Real cse1258 = cse1088 * cse7;
		const amrex::Real cse1259 = 1.4381666666666666e-5 * cse1258;
		const amrex::Real cse1260 = cse1067 * cse7;
		const amrex::Real cse1261 = 8.6289999999999999e-6 * cse1257;
		const amrex::Real cse1262 = 8.6289999999999999e-6 * cse1260;
		const amrex::Real cse1263 = 1.4381666666666668e-5 * cse1051 * cse1260 + 1.4381666666666668e-5 * cse1070 * cse1257 + cse1093 * cse1261 +
					    cse1096 * cse1259 + cse1097 * cse1259 + cse1099 * cse1262;
		const amrex::Real cse1264 = 5.1774000000000006e-6 * cse1113;
		const amrex::Real cse1265 = 8.6289999999999999e-6 * cse1258;
		const amrex::Real cse1266 =
		    cse1096 * cse1261 + cse1119 * cse1262 + cse1121 * cse1265 + cse1122 * cse1265 + cse1257 * cse1264 + cse1260 * cse1264;
		const amrex::Real cse1267 = cse1089 * cse7;
		const amrex::Real cse1268 = 1.7258000000000001e-6 * cse1098;
		const amrex::Real cse1269 = cse1125 * (-cse1102 * cse1267 - cse1103 * cse1267 - 2.8763333333333337e-6 * cse1105 * cse1260 -
						       2.8763333333333337e-6 * cse1126 * cse1257 - cse1257 * cse1268 - cse1260 * cse1268 - cse1263 - cse1266);
		const amrex::Real cse1270 = cse1246 * cse724;
		const amrex::Real cse1271 = 3.7996253111508318e-29 * cse1270;
		const amrex::Real cse1272 = cse749 * (4 * state.xn[0] + 4 * state.xn[1] + 4 * state.xn[10] + 4 * state.xn[14] + 4 * state.xn[15] +
						      12 * state.xn[16] + 8 * state.xn[3] + 8 * state.xn[4]);
		const amrex::Real cse1273 = cse135 * cse752;
		const amrex::Real cse1274 = cse753 * (200.0 * cse1273 + 200.0 * cse751);
		const amrex::Real cse1275 = cse704 * cse755;
		const amrex::Real cse1276 = cse756 * cse758;
		const amrex::Real cse1277 = -1.0 * cse1275 + 1.0 * cse1276;
		const amrex::Real cse1278 =
		    ((cse761) ? (0) : ((cse763) ? (3 * cse762) : ((cse764) ? ((8.0 / 3.0) * cse762) : ((cse765) ? ((2.0 / 3.0) * cse762) : (0)))));
		const amrex::Real cse1279 = 2.5634832e-11 * cse782 + 6.4087080000000003e-13;
		const amrex::Real cse1280 = -cse792;
		const amrex::Real cse1281 = cse233 * cse771 + cse235 * cse774;
		const amrex::Real cse1282 = 2 * cse846 - cse851 * (cse1281 - cse772);
		const amrex::Real cse1283 = cse235 * cse707;
		const amrex::Real cse1284 = cse1278 * state.xn[3];
		const amrex::Real cse1285 = 4.0 * cse859;
		const amrex::Real cse1286 = ((cse815) ? (0) : (cse818 * cse861));
		const amrex::Real cse1287 = cse1286 * cse868;
		const amrex::Real cse1288 = -cse1286 * cse811 * cse872 * cse873 + cse829 * (cse1286 * cse808 * cse822 * cse872 * cse873 - cse1287 * cse823);
		const amrex::Real cse1289 = cse977 * cse997;
		const amrex::Real cse1290 = cse1008 * cse1289 + cse1009 * cse992 + cse1011 * cse964 + cse1191 * cse992 + cse1193 * cse977 + cse1195 * cse964;
		const amrex::Real cse1291 = cse1001 * cse992;
		const amrex::Real cse1292 = cse1000 * cse992;
		const amrex::Real cse1293 = cse1007 * cse964 + cse1007 * cse977 + cse1009 * cse964 + cse1202 * cse1291 + cse1202 * cse1292 + cse1204 * cse977;
		const amrex::Real cse1294 =
		    cse1023 * (-cse1004 * cse977 - cse1206 * cse964 - cse1289 - cse1290 - cse1291 - cse1292 - cse1293 - cse964 * cse997);
		const amrex::Real cse1295 = cse1058 * cse1098;
		const amrex::Real cse1296 =
		    cse1037 * cse1109 + cse1037 * cse1213 + cse1058 * cse1216 + cse1079 * cse1110 + cse1079 * cse1217 + cse1108 * cse1295;
		const amrex::Real cse1297 = cse1079 * cse1103;
		const amrex::Real cse1298 = cse1079 * cse1102;
		const amrex::Real cse1299 =
		    cse1037 * cse1110 + cse1037 * cse1114 + cse1058 * cse1114 + cse1058 * cse1223 + cse1222 * cse1297 + cse1222 * cse1298;
		const amrex::Real cse1300 =
		    cse1125 * (-cse1037 * cse1098 - cse1037 * cse1226 - cse1058 * cse1106 - cse1295 - cse1296 - cse1297 - cse1298 - cse1299);
		const amrex::Real cse1301 = 3.4516225000000006e-16 * cse1229;
		const amrex::Real cse1302 = cse119 * cse1233;
		const amrex::Real cse1303 = cse1234 * cse855;
		const amrex::Real cse1304 = -cse1281 * cse851 + 2 * cse157 * cse778 * cse779;
		const amrex::Real cse1305 = cse1230 + state.xn[3] * ((cse906) ? (0) : (cse927 * cse951));
		const amrex::Real cse1306 = ((cse815) ? (cse814) : (0));
		const amrex::Real cse1307 = cse1306 * cse868;
		const amrex::Real cse1308 = -1.5 * cse1275 + 1.5 * cse1276;
		const amrex::Real cse1309 = ((cse761) ? (0) : ((cse763) ? ((9.0 / 2.0) * cse762) : ((cse764) ? (4 * cse762) : ((cse765) ? (cse762) : (0)))));
		const amrex::Real cse1310 = 3 * cse157 * cse778 * cse779 - cse851 * (3 * cse847 + 3 * cse848);
		const amrex::Real cse1311 = ((cse815) ? (0) : (2635052181.6465044 * cse862));
		const amrex::Real cse1312 = cse1311 * cse868;
		const amrex::Real cse1313 = state.c_hat * cse135;
		const amrex::Real cse1314 = cse1232 * cse1313;
		const amrex::Real cse1315 = state.c_hat * cse54 * cse744 * cse748 / std::exp((5.0 / 6.0) * ln_cse742);
		const amrex::Real cse1316 = cse1247 * cse1313 * cse715;
		const amrex::Real cse1317 = state.c_hat * cse84;
		const amrex::Real cse1318 = cse1142 * cse1317;
		const amrex::Real cse1319 = cse1144 * cse1317;
		const amrex::Real cse1320 = cse1141 * cse1231 * cse135 * cse708;
		const amrex::Real cse1321 = state.c_hat * cse715;
		const amrex::Real cse1322 = cse1151 * cse1321;
		const amrex::Real cse1323 = cse1152 * cse1321;
		const amrex::Real cse1324 = cse249 * cse778 * cse845;
		const amrex::Real cse1325 = state.c_hat * cse1324 * cse243;
		const amrex::Real cse1326 = cse1324 * cse21 * cse321 * cse568;
		const amrex::Real cse1327 = Zd * cse247;
		const amrex::Real cse1328 = 1.252779946046115e-21 * cse1327;
		const amrex::Real cse1329 = -cse1328;
		const amrex::Real cse1330 = 2.50555989209223e-21 * cse1327;
		const amrex::Real cse1331 = 1.8689621674132952e-20 * cse249;
		const amrex::Real cse1332 = Zd * state.c_hat * cse20;
		const amrex::Real cse1333 = cse1331 * cse321 + cse330 + cse527;
		const amrex::Real cse1334 = state.c_hat * state.rn[2 * 0 + 1];
		const amrex::Real cse1335 = Zd * cse1334;
		const amrex::Real cse1336 = 9.1016827996670106e-22 * cse1335;
		const amrex::Real cse1337 = -cse1336;
		const amrex::Real cse1338 = 1.8203365599334021e-21 * cse1335;
		const amrex::Real cse1339 = Zd * cse251;
		const amrex::Real cse1340 = 2.0584539190345922e-21 * cse1339;
		const amrex::Real cse1341 = -cse1340;
		const amrex::Real cse1342 = 4.1169078380691844e-21 * cse1339;
		const amrex::Real cse1343 = 2.6551134042892157e-17 * cse249;
		const amrex::Real cse1344 = cse1343 * cse321 + cse331 + cse537 + cse540 + cse666;
		const amrex::Real cse1345 = state.c_hat * state.rn[2 * 1 + 1];
		const amrex::Real cse1346 = Zd * cse1345;
		const amrex::Real cse1347 = 1.6856051574414119e-21 * cse1346;
		const amrex::Real cse1348 = -cse1347;
		const amrex::Real cse1349 = 3.3712103148828238e-21 * cse1346;
		// structural zeros (95 of 529 entries)
		jac(19, 2) = 0.0;
		jac(20, 2) = 0.0;
		jac(21, 2) = 0.0;
		jac(22, 2) = 0.0;
		jac(19, 5) = 0.0;
		jac(20, 5) = 0.0;
		jac(21, 5) = 0.0;
		jac(22, 5) = 0.0;
		jac(19, 6) = 0.0;
		jac(20, 6) = 0.0;
		jac(21, 6) = 0.0;
		jac(22, 6) = 0.0;
		jac(19, 7) = 0.0;
		jac(20, 7) = 0.0;
		jac(19, 8) = 0.0;
		jac(20, 8) = 0.0;
		jac(21, 8) = 0.0;
		jac(22, 8) = 0.0;
		jac(19, 9) = 0.0;
		jac(20, 9) = 0.0;
		jac(19, 11) = 0.0;
		jac(20, 11) = 0.0;
		jac(21, 11) = 0.0;
		jac(22, 11) = 0.0;
		jac(19, 13) = 0.0;
		jac(20, 13) = 0.0;
		jac(21, 13) = 0.0;
		jac(22, 13) = 0.0;
		jac(19, 17) = 0.0;
		jac(20, 17) = 0.0;
		jac(21, 17) = 0.0;
		jac(22, 17) = 0.0;
		jac(19, 18) = 0.0;
		jac(20, 18) = 0.0;
		jac(21, 18) = 0.0;
		jac(22, 18) = 0.0;
		jac(4, 19) = 0.0;
		jac(9, 19) = 0.0;
		jac(10, 19) = 0.0;
		jac(16, 19) = 0.0;
		jac(17, 19) = 0.0;
		jac(20, 19) = 0.0;
		jac(21, 19) = 0.0;
		jac(22, 19) = 0.0;
		jac(0, 20) = 0.0;
		jac(1, 20) = 0.0;
		jac(2, 20) = 0.0;
		jac(3, 20) = 0.0;
		jac(4, 20) = 0.0;
		jac(5, 20) = 0.0;
		jac(6, 20) = 0.0;
		jac(7, 20) = 0.0;
		jac(8, 20) = 0.0;
		jac(9, 20) = 0.0;
		jac(10, 20) = 0.0;
		jac(11, 20) = 0.0;
		jac(12, 20) = 0.0;
		jac(13, 20) = 0.0;
		jac(14, 20) = 0.0;
		jac(15, 20) = 0.0;
		jac(16, 20) = 0.0;
		jac(17, 20) = 0.0;
		jac(18, 20) = 0.0;
		jac(19, 20) = 0.0;
		jac(21, 20) = 0.0;
		jac(22, 20) = 0.0;
		jac(4, 21) = 0.0;
		jac(10, 21) = 0.0;
		jac(16, 21) = 0.0;
		jac(17, 21) = 0.0;
		jac(19, 21) = 0.0;
		jac(20, 21) = 0.0;
		jac(22, 21) = 0.0;
		jac(0, 22) = 0.0;
		jac(1, 22) = 0.0;
		jac(2, 22) = 0.0;
		jac(3, 22) = 0.0;
		jac(4, 22) = 0.0;
		jac(5, 22) = 0.0;
		jac(6, 22) = 0.0;
		jac(7, 22) = 0.0;
		jac(8, 22) = 0.0;
		jac(9, 22) = 0.0;
		jac(10, 22) = 0.0;
		jac(11, 22) = 0.0;
		jac(12, 22) = 0.0;
		jac(13, 22) = 0.0;
		jac(14, 22) = 0.0;
		jac(15, 22) = 0.0;
		jac(16, 22) = 0.0;
		jac(17, 22) = 0.0;
		jac(18, 22) = 0.0;
		jac(19, 22) = 0.0;
		jac(20, 22) = 0.0;
		jac(21, 22) = 0.0;
		jac(0, 0) =
		    2 * cse10 * cse167 - cse151 - cse153 + 2 * cse154 * state.xn[3] - cse168 * cse20 - cse169 + 2 * cse171 * state.xn[0] * state.xn[3] - cse188;
		jac(0, 1) = cse191 + cse203;
		jac(0, 2) = cse194 + cse207 + cse209 * state.xn[16] + cse211 * cse212 + cse217 + cse223;
		jac(0, 3) = 3.2000000000000001e-9 * cse138 * state.xn[17] + cse201 * cse246 - cse212 * cse256 + cse225 + cse227 + cse229 + 2 * cse230 +
			    2 * cse240 + cse241 * cse242 - cse244 + cse247 * cse250 + cse251 * cse252 + cse254 * state.xn[8] - cse255 + cse260 + cse266;
		jac(0, 4) = 2 * cse10 * cse271 - cse244 - cse255 - cse267 + 2 * cse272 * state.xn[0] * state.xn[3] + cse273 - cse277;
		jac(0, 5) = cse194;
		jac(0, 6) = cse194 + cse282;
		jac(0, 7) = cse194 + cse284;
		jac(0, 8) = cse194 + cse254 * state.xn[3];
		jac(0, 9) = -cse151 - cse285;
		jac(0, 10) = cse202 + cse287;
		jac(0, 11) = cse194 + cse289 + cse290 + cse293;
		jac(0, 12) = cse194;
		jac(0, 13) = cse194;
		jac(0, 14) = cse203 + cse294 + cse295 + cse296 + cse298;
		jac(0, 15) = cse203 + cse304;
		jac(0, 16) = cse194 + cse201 * cse310 + cse209 * state.xn[2] - 1.8000000000000002e-14 * cse243 + 2 * cse309 + cse316;
		jac(0, 17) = 3.2000000000000001e-9 * cse138 * state.xn[3] - cse151 - cse317 - cse320;
		jac(0, 18) = cse149;
		jac(0, 19) = cse250 * cse321 + cse325 + cse330;
		jac(0, 21) = cse252 * cse321 + cse331 + cse332 + cse333;
		jac(1, 0) = cse188 + cse338;
		jac(1, 1) = -cse191 - cse339;
		jac(1, 2) = -cse222 - cse337;
		jac(1, 3) = cse225 + cse264 + cse340;
		jac(1, 4) = cse277 + cse340;
		jac(1, 5) = cse338;
		jac(1, 6) = cse281 - cse337;
		jac(1, 7) = cse338;
		jac(1, 8) = cse338;
		jac(1, 9) = cse338;
		jac(1, 10) = cse341;
		jac(1, 11) = -cse291 - cse337;
		jac(1, 12) = cse338;
		jac(1, 13) = cse338;
		jac(1, 14) = cse341;
		jac(1, 15) = cse341;
		jac(1, 16) = -cse315 - cse337;
		jac(1, 17) = cse317 - cse337;
		jac(1, 18) = cse335;
		jac(1, 19) = -cse325;
		jac(1, 21) = -cse333;
		jac(2, 0) = cse176 + cse178 + cse183 - cse187 - cse344 - cse452;
		jac(2, 1) = -cse190 - cse455;
		jac(2, 2) = -cse223 - cse438 - cse457 - cse460 - cse463 - cse464 - cse470;
		jac(2, 3) = -cse266 - cse473 + cse474 + cse475 - cse480 - cse485 - cse486 - cse493;
		jac(2, 4) = cse276 - cse493 - cse495;
		jac(2, 5) = cse496 + cse497;
		jac(2, 6) = -cse438 - cse498;
		jac(2, 7) = cse497 + cse501 + cse504;
		jac(2, 8) = -cse438 - cse507;
		jac(2, 9) = cse285 + cse497;
		jac(2, 10) = -cse287 - cse454;
		jac(2, 11) = -cse438 - cse509;
		jac(2, 12) = cse497 + cse512;
		jac(2, 13) = -cse438 - cse513;
		jac(2, 14) = cse514;
		jac(2, 15) = cse514;
		jac(2, 16) = -cse316 - cse438 - cse515 - cse516 - cse517 - cse518 - cse520 - cse523 - cse524;
		jac(2, 17) = -cse438 - cse525;
		jac(2, 18) = cse436;
		jac(2, 19) = -cse325 + cse527 - cse530 - cse533 - cse536;
		jac(2, 21) = -cse333 + cse537 - cse538 - cse539 - cse542;
		jac(3, 0) = cse153 + cse174 + 3.0000000000000002e-15 * cse20 * d2g - cse343 - cse543 - cse544 - cse546 - cse557;
		jac(3, 1) = cse560;
		jac(3, 2) = cse207 + cse561 + cse562 * state.xn[16] + cse563 + cse564;
		jac(3, 3) = -cse19 * cse241 - cse230 - cse240 - cse477 - cse565 - cse568 - cse570 - cse571 - cse572 - cse573 - cse574;
		jac(3, 4) = -cse273 + cse274 - cse494 - cse574 - cse575 - cse576;
		jac(3, 5) = cse561;
		jac(3, 6) = -cse281 - cse556 - cse577;
		jac(3, 7) = -cse501 - cse556;
		jac(3, 8) = -cse505 - cse556;
		jac(3, 9) = -cse556 + cse578;
		jac(3, 10) = cse560;
		jac(3, 11) = cse509 + cse561;
		jac(3, 12) = cse561;
		jac(3, 13) = cse561;
		jac(3, 14) = -cse298 - cse559;
		jac(3, 15) = cse560;
		jac(3, 16) = -cse309 + cse520 + cse561 + cse562 * state.xn[2] + cse579 - cse580 + cse582;
		jac(3, 17) = -cse525 - cse556;
		jac(3, 18) = cse554;
		jac(3, 19) = -cse583;
		jac(3, 21) = -cse584;
		jac(4, 0) = -cse174 - cse344 - cse587;
		jac(4, 1) = cse589;
		jac(4, 2) = cse588;
		jac(4, 3) = -cse227 + cse569 + cse571 - cse590;
		jac(4, 4) = -cse273 - cse274 - cse495 - cse590;
		jac(4, 5) = cse588;
		jac(4, 6) = cse577 - cse587;
		jac(4, 7) = cse588;
		jac(4, 8) = cse588;
		jac(4, 9) = cse588;
		jac(4, 10) = cse589;
		jac(4, 11) = cse588;
		jac(4, 12) = cse588;
		jac(4, 13) = cse588;
		jac(4, 14) = cse589;
		jac(4, 15) = cse589;
		jac(4, 16) = cse519 + cse588;
		jac(4, 17) = cse588;
		jac(4, 18) = cse585;
		jac(5, 0) = cse597;
		jac(5, 1) = cse597;
		jac(5, 2) = cse457 + cse596;
		jac(5, 3) = cse570 + cse598;
		jac(5, 4) = cse598;
		jac(5, 5) = -cse496 - cse595;
		jac(5, 6) = cse596 + cse600;
		jac(5, 7) = cse596;
		jac(5, 8) = cse596;
		jac(5, 9) = -cse595 - cse602;
		jac(5, 10) = cse597;
		jac(5, 11) = cse596;
		jac(5, 12) = cse596;
		jac(5, 13) = cse596;
		jac(5, 14) = cse597;
		jac(5, 15) = cse299 + cse597;
		jac(5, 16) = cse515 + cse596;
		jac(5, 17) = cse596;
		jac(5, 18) = cse593;
		jac(5, 19) = cse533;
		jac(5, 21) = cse539;
		jac(6, 0) = cse607;
		jac(6, 1) = cse607;
		jac(6, 2) = -cse457 - cse605;
		jac(6, 3) = -cse570 - cse608;
		jac(6, 4) = -cse608;
		jac(6, 5) = cse496 + cse609;
		jac(6, 6) = -cse600 - cse605;
		jac(6, 7) = cse609;
		jac(6, 8) = cse609;
		jac(6, 9) = -cse601 - cse605;
		jac(6, 10) = cse607;
		jac(6, 11) = cse609;
		jac(6, 12) = cse609;
		jac(6, 13) = cse609;
		jac(6, 14) = cse607;
		jac(6, 15) = -cse299 - cse606;
		jac(6, 16) = -cse515 - cse605;
		jac(6, 17) = cse609;
		jac(6, 18) = cse603;
		jac(6, 19) = -cse533;
		jac(6, 21) = -cse539;
		jac(7, 0) = cse153 + cse618;
		jac(7, 1) = cse618;
		jac(7, 2) = cse460 + cse614;
		jac(7, 3) = cse479 + cse619 * state.xn[8] + cse624 + cse625;
		jac(7, 4) = cse623 + cse625;
		jac(7, 5) = cse614;
		jac(7, 6) = cse614;
		jac(7, 7) = -cse284 - cse500 - cse504 - cse613;
		jac(7, 8) = cse506 + cse614 + cse619 * state.xn[3];
		jac(7, 9) = cse614 + cse628;
		jac(7, 10) = cse618;
		jac(7, 11) = cse614;
		jac(7, 12) = cse614;
		jac(7, 13) = cse614;
		jac(7, 14) = cse618 + cse629;
		jac(7, 15) = -cse300 + cse442 + cse445 - cse613 - cse616;
		jac(7, 16) = -cse517 + cse521 + cse522 - cse613 - cse630;
		jac(7, 17) = cse614;
		jac(7, 18) = cse611;
		jac(7, 19) = cse327 + cse536;
		jac(7, 21) = cse331 + cse542 + cse631;
		jac(8, 0) = cse638;
		jac(8, 1) = cse638;
		jac(8, 2) = -cse460 - cse636;
		jac(8, 3) = -cse480 - cse639;
		jac(8, 4) = -cse639;
		jac(8, 5) = cse640;
		jac(8, 6) = -cse636 - cse641;
		jac(8, 7) = cse504 + cse640;
		jac(8, 8) = -cse507 - cse636 - cse643;
		jac(8, 9) = -cse602 - cse636;
		jac(8, 10) = cse638;
		jac(8, 11) = cse640;
		jac(8, 12) = cse640;
		jac(8, 13) = cse640;
		jac(8, 14) = cse638;
		jac(8, 15) = -cse637 - cse644;
		jac(8, 16) = -cse523 - cse636;
		jac(8, 17) = cse640;
		jac(8, 18) = cse634;
		jac(8, 19) = -cse536;
		jac(8, 21) = cse540 - cse541;
		jac(9, 0) = cse649;
		jac(9, 1) = cse649;
		jac(9, 2) = cse219 + cse648;
		jac(9, 3) = -cse624 - cse650;
		jac(9, 4) = cse622 - cse650;
		jac(9, 5) = cse648;
		jac(9, 6) = -cse599 - cse647;
		jac(9, 7) = cse284 + cse648;
		jac(9, 8) = cse648;
		jac(9, 9) = -cse647 - cse651 - cse652;
		jac(9, 10) = cse286 + cse649;
		jac(9, 11) = cse292 - cse647;
		jac(9, 12) = cse648;
		jac(9, 13) = cse648;
		jac(9, 14) = cse296 + cse649;
		jac(9, 15) = cse300 + cse649;
		jac(9, 16) = -cse581 + cse630 - cse647;
		jac(9, 17) = cse648;
		jac(9, 18) = cse645;
		jac(9, 21) = -cse631;
		jac(10, 0) = cse656;
		jac(10, 1) = cse656;
		jac(10, 2) = -cse219 - cse655;
		jac(10, 3) = cse657;
		jac(10, 4) = cse657;
		jac(10, 5) = cse656;
		jac(10, 6) = cse656;
		jac(10, 7) = cse656;
		jac(10, 8) = cse643 - cse655;
		jac(10, 9) = cse651 + cse656;
		jac(10, 10) = -cse286 - cse655;
		jac(10, 11) = cse656;
		jac(10, 12) = cse656;
		jac(10, 13) = cse656;
		jac(10, 14) = cse656;
		jac(10, 15) = cse644 - cse655;
		jac(10, 16) = cse581 - cse655;
		jac(10, 17) = cse656;
		jac(10, 18) = cse653;
		jac(11, 0) = cse173 - cse663;
		jac(11, 1) = -cse663 - cse80;
		jac(11, 2) = -cse563 - cse662;
		jac(11, 3) = cse258 - cse259 - cse484 + cse621 - cse664;
		jac(11, 4) = -cse664;
		jac(11, 5) = cse665;
		jac(11, 6) = -cse641 - cse662;
		jac(11, 7) = cse665;
		jac(11, 8) = cse665;
		jac(11, 9) = cse652 + cse665;
		jac(11, 10) = -cse663;
		jac(11, 11) = cse290 - cse293 - cse508 - cse662;
		jac(11, 12) = cse665;
		jac(11, 13) = cse665;
		jac(11, 14) = -cse296 - cse663;
		jac(11, 15) = cse303 + cse617 + cse665;
		jac(11, 16) = -cse516 - cse630 - cse662;
		jac(11, 17) = cse317 + cse318 + cse665;
		jac(11, 18) = cse660;
		jac(11, 19) = cse329;
		jac(11, 21) = cse666;
		jac(12, 0) = cse672;
		jac(12, 1) = cse672;
		jac(12, 2) = cse463 + cse671;
		jac(12, 3) = cse473 + cse673;
		jac(12, 4) = cse673;
		jac(12, 5) = cse671;
		jac(12, 6) = cse671;
		jac(12, 7) = cse671;
		jac(12, 8) = cse671;
		jac(12, 9) = cse671;
		jac(12, 10) = cse672;
		jac(12, 11) = cse671;
		jac(12, 12) = -cse512 - cse670;
		jac(12, 13) = cse513 + cse671;
		jac(12, 14) = cse672;
		jac(12, 15) = cse672;
		jac(12, 16) = cse524 + cse671;
		jac(12, 17) = cse671;
		jac(12, 18) = cse668;
		jac(12, 19) = -cse674;
		jac(12, 21) = -cse675;
		jac(13, 0) = cse679;
		jac(13, 1) = cse679;
		jac(13, 2) = -cse463 - cse678;
		jac(13, 3) = -cse473 - cse680;
		jac(13, 4) = -cse680;
		jac(13, 5) = cse681;
		jac(13, 6) = cse681;
		jac(13, 7) = cse681;
		jac(13, 8) = cse681;
		jac(13, 9) = cse681;
		jac(13, 10) = cse679;
		jac(13, 11) = cse681;
		jac(13, 12) = cse512 + cse681;
		jac(13, 13) = -cse513 - cse678;
		jac(13, 14) = cse679;
		jac(13, 15) = cse679;
		jac(13, 16) = -cse524 - cse678;
		jac(13, 17) = cse681;
		jac(13, 18) = cse676;
		jac(13, 19) = cse674;
		jac(13, 21) = cse675;
		jac(14, 0) = -cse153 - cse684;
		jac(14, 1) = cse685;
		jac(14, 2) = cse685;
		jac(14, 3) = cse686 * state.xn[8] + cse687;
		jac(14, 4) = cse687;
		jac(14, 5) = cse685;
		jac(14, 6) = cse685;
		jac(14, 7) = -cse501 - cse684;
		jac(14, 8) = cse685 + cse686 * state.xn[3];
		jac(14, 9) = cse685;
		jac(14, 10) = cse685;
		jac(14, 11) = -cse292 - cse684;
		jac(14, 12) = cse685;
		jac(14, 13) = cse685;
		jac(14, 14) = -cse296 - cse629 - cse684;
		jac(14, 15) = cse685;
		jac(14, 16) = cse517 - cse684;
		jac(14, 17) = cse685;
		jac(14, 18) = cse682;
		jac(14, 19) = -cse327;
		jac(14, 21) = -cse331;
		jac(15, 0) = cse691;
		jac(15, 1) = cse691;
		jac(15, 2) = cse467 - cse468 - cse564 - cse690;
		jac(15, 3) = cse485 + cse692;
		jac(15, 4) = cse692;
		jac(15, 5) = cse691;
		jac(15, 6) = -cse280 - cse690;
		jac(15, 7) = -cse284 - cse690;
		jac(15, 8) = -cse643 - cse690;
		jac(15, 9) = cse691;
		jac(15, 10) = cse691;
		jac(15, 11) = -cse290 + cse508 - cse690;
		jac(15, 12) = cse691;
		jac(15, 13) = cse691;
		jac(15, 14) = cse691;
		jac(15, 15) = -cse304 - cse644 - cse690;
		jac(15, 16) = cse516 + cse691;
		jac(15, 17) = -cse320 - cse690;
		jac(15, 18) = cse688;
		jac(15, 19) = -cse329;
		jac(15, 21) = -cse332;
		jac(16, 0) = cse696;
		jac(16, 1) = cse696;
		jac(16, 2) = -cse470 - cse695;
		jac(16, 3) = cse572 + cse697;
		jac(16, 4) = cse273 + cse697;
		jac(16, 5) = cse696;
		jac(16, 6) = cse696;
		jac(16, 7) = -cse500 - cse695;
		jac(16, 8) = cse696;
		jac(16, 9) = -cse578 - cse695;
		jac(16, 10) = cse696;
		jac(16, 11) = -cse509 - cse695;
		jac(16, 12) = cse696;
		jac(16, 13) = cse696;
		jac(16, 14) = cse696;
		jac(16, 15) = cse696;
		jac(16, 16) = -cse518 - cse582 - cse695;
		jac(16, 17) = cse696;
		jac(16, 18) = cse693;
		jac(17, 0) = -cse173 - cse700;
		jac(17, 1) = -cse700 + cse80;
		jac(17, 2) = -cse464 - cse700;
		jac(17, 3) = -cse573 - cse701;
		jac(17, 4) = -cse701;
		jac(17, 5) = cse702;
		jac(17, 6) = cse280 - cse700;
		jac(17, 7) = cse702;
		jac(17, 8) = cse702;
		jac(17, 9) = cse702;
		jac(17, 10) = cse702;
		jac(17, 11) = cse291 - cse700;
		jac(17, 12) = cse702;
		jac(17, 13) = cse702;
		jac(17, 14) = cse702;
		jac(17, 15) = cse299 - cse700;
		jac(17, 16) = cse702;
		jac(17, 17) = -cse317 - cse525 - cse700;
		jac(17, 18) = cse698;
		jac(18, 0) = 5.6705896183685846e-35 * state.c_hat * cse118 * cse708 * cse712 * d2g +
			     1.7623947000000002e-12 * crate * cse180 * cse835 * cse840 * state.xn[5] +
			     4.22974728e-11 * crate * cse20 * cse703 * cse705 * cse759 * cse82 * state.xn[0] + 1.602177e-12 * crate * cse707 +
			     3.2043540000000001e-12 * crate * cse766 * state.xn[3] +
			     1.7623947000000002e-12 * crate * cse853 * state.xn[5] * (cse836 + cse838 * cse858 - cse854 + cse856 * cse857 - cse860) - cse1230 -
			     2.1789607199999999e-11 * cse178 + 3.0000000000000002e-15 * cse20 * cse783 * d2g +
			     2.0187430200000004e-26 * cse20 * cse845 * cse852 * d2g * state.xn[0] - 7.1777529600000012e-12 * cse543 -
			     7.1777529600000012e-12 * cse544 - 7.1777529600000012e-12 * cse546 + 2.5634832e-11 * cse568 * cse845 * cse852 * state.xn[3] -
			     cse725 - cse730 * cse739 - cse750 - cse754 + 3.0000000000000002e-15 * cse783 * d2g * state.xn[0] +
			     4.39089220777172e-29 * cse788 * cse793 * cse795 * cse799 * cse802 * state.xn[8] - 4.39089220777172e-29 * cse788 * cse801 -
			     cse804 * cse834 - cse844 * (-cse796 * cse841 + cse841 * cse842) -
			     cse876 * (-cse804 * cse821 - cse830 * (cse863 * cse866 + cse871 * (cse804 * cse825 - cse867 * cse869)) - cse874) -
			     state.xn[11] * (cse1016 * cse999 + cse1021 * cse1022 + cse1025 * cse1026 + cse1025 * cse1027) -
			     state.xn[3] * ((cse906) ? (0) : (cse922 * cse951)) -
			     state.xn[7] * ((cse1130) ? (cse1101 * cse1118 + cse1123 * cse1124 + cse1127 * cse1128 + cse1127 * cse1129) : (0));
		jac(18, 1) = cse1242 - state.xn[3] * ((cse906) ? (0) : (cse925 * cse951));
		jac(18, 2) = 5.6705896183685846e-35 * state.c_hat * cse1141 * cse118 * cse20 * cse708 * d2g *
				 (8.429953871621411e-15 * cse1143 * cse210 + 1.2210344531082231e-12 * cse1145 * cse210) +
			     4.22974728e-11 * crate * cse1248 * cse20 * cse703 * cse705 * cse82 * state.xn[0] +
			     4.6527220080000002e-11 * crate * cse1248 * cse703 * cse705 * cse82 * cse853 * state.xn[0] * state.xn[5] +
			     6.0732876035249664e-19 * cse1135 * cse1137 * cse1243 * cse733 * cse737 * state.xn[0] * state.xn[2] -
			     cse1139 * cse739 * state.xn[0] - cse1149 -
			     cse1150 * (2.8857630567877724e-12 * state.c_hat * cse1147 * cse118 * cse1244 * cse54 * cse718 - cse1152 * cse1245 * cse716) -
			     cse1153 * (5.9301787935659931e-17 * cse1135 * cse1137 * cse1243 * cse733 * state.xn[2] - cse1139 * cse735) -
			     2.4067011396745388e-39 * cse119 * cse1245 * cse1247 * cse135 - cse1230 - 2.1789607199999999e-11 * cse220 +
			     2.6241084038530201e-25 * cse7 * cse793 * cse795 * cse799 * cse802 * state.xn[8] - 2.6241084038530201e-25 * cse7 * cse801 -
			     cse810 * cse834 - cse844 * (-cse1249 * cse796 + cse1249 * cse842) - cse876 * (-cse810 * cse821 - cse810 * cse830 * cse870) -
			     state.xn[11] * (cse1016 * cse1253 + cse1022 * cse1255 + cse1026 * cse1256 + cse1027 * cse1256) -
			     state.xn[3] * ((cse906) ? (0) : (cse936 * cse951)) -
			     state.xn[7] * ((cse1130) ? (cse1118 * cse1263 + cse1124 * cse1266 + cse1128 * cse1269 + cse1129 * cse1269) : (0));
		jac(18, 3) =
		    1.1341179236737169e-34 * state.c_hat * cse118 * cse708 * cse712 * d2g +
		    4.22974728e-11 * crate * cse1277 * cse20 * cse703 * cse705 * cse82 * state.xn[0] + 3.2043540000000001e-12 * crate * cse1278 * state.xn[3] +
		    1.7623947000000002e-12 * crate * cse262 * cse835 * cse840 * state.xn[5] + 3.2043540000000001e-12 * crate * cse837 +
		    1.7623947000000002e-12 * crate * cse853 * state.xn[5] * (cse1277 * cse855 * cse857 - cse1283 + cse1284 * cse838 - cse1285 + cse839) -
		    cse1160 * cse792 - cse1163 * cse1280 - cse1271 - cse1272 - cse1274 + cse1279 * cse568 +
		    2.0187430200000004e-26 * cse1282 * cse20 * cse845 * d2g * state.xn[0] + 2.5634832e-11 * cse1282 * cse568 * cse845 * state.xn[3] - cse1301 -
		    7.1777529600000012e-12 * cse230 - 7.1777529600000012e-12 * cse240 - 1.4355505920000002e-11 * cse241 * state.xn[3] -
		    7.1777529600000012e-12 * cse565 + 6.0000000000000005e-15 * cse783 * d2g * state.xn[0] - cse834 -
		    cse844 * (-cse1168 * cse1280 - cse792 * cse797) -
		    cse876 * (-cse1288 - cse821 - cse830 * (cse1286 * cse866 + cse871 * (-cse1287 * cse867 + cse825))) -
		    state.xn[11] * (cse1016 * cse1290 + cse1022 * cse1293 + cse1026 * cse1294 + cse1027 * cse1294) -
		    state.xn[3] * ((cse906) ? (0) : (cse923 * cse951)) -
		    state.xn[7] * ((cse1130) ? (cse1118 * cse1296 + cse1124 * cse1299 + cse1128 * cse1300 + cse1129 * cse1300) : (0)) -
		    ((cse906) ? (0.0) : (cse1176 * cse947));
		jac(18, 4) =
		    cse1167 * cse1304 + cse1235 * cse1284 + cse1236 * cse261 + cse1238 * cse1304 +
		    cse1239 * (26.399999999999999 * cse1277 * cse703 * cse705 * cse82 * state.xn[0] + 2.0 * cse1278 * cse21 * state.xn[3] - cse1283 - cse1285) -
		    cse1271 - cse1272 - cse1274 + cse1277 * cse1303 - cse1301 + 1.1341179236737169e-34 * cse1302 + cse169 * cse783 -
		    7.1777529600000012e-12 * cse575 - 7.1777529600000012e-12 * cse576 -
		    cse876 * (-cse1288 - cse830 * (-cse1240 * cse1287 + cse1286 * cse864 * cse865));
		jac(18, 5) = 1.7623947000000002e-12 * crate * cse840 * cse853 - cse1305;
		jac(18, 6) = -cse1305;
		jac(18, 7) = -cse1230 - ((cse1130) ? (cse1111 * cse1118 + cse1115 * cse1124) : (0.0));
		jac(18, 8) = -cse1159 * cse843 - cse1230;
		jac(18, 9) = -cse1230 - cse811 * cse833 -
			     cse876 * (cse1306 * cse811 * cse872 * cse873 - cse829 * (cse1306 * cse808 * cse822 * cse872 * cse873 - cse1307 * cse823) -
				       cse830 * (-cse1240 * cse1307 + cse1306 * cse864 * cse865));
		jac(18, 10) = cse1242;
		jac(18, 11) = -cse1010 * cse1022 - cse1013 * cse1016 - cse1230;
		jac(18, 12) = cse1241;
		jac(18, 13) = cse1241;
		jac(18, 14) = cse1242;
		jac(18, 15) = cse1242;
		jac(18, 16) = cse1167 * cse1310 + cse1235 * cse1309 * state.xn[3] + cse1236 * cse313 + cse1238 * cse1310 +
			      cse1239 * (26.399999999999999 * cse1308 * cse703 * cse705 * cse82 * state.xn[0] + 2.0 * cse1309 * cse21 * state.xn[3] -
					 3 * cse854 - 6.0 * cse859) +
			      cse1241 - 5.6994379667262474e-29 * cse1270 + 1.7011768855105754e-34 * cse1302 + cse1303 * cse1308 -
			      7.1777529600000012e-12 * cse309 + cse579 * cse783 - 7.1777529600000012e-12 * cse580 -
			      cse749 * (6 * state.xn[0] + 6 * state.xn[1] + 6 * state.xn[10] + 6 * state.xn[14] + 6 * state.xn[15] + 18 * state.xn[16] +
					12 * state.xn[3] + 12 * state.xn[4]) -
			      cse753 * (300.0 * cse1273 + 300.0 * cse751) -
			      cse876 * (cse1311 * cse811 * cse872 * cse873 - cse829 * (cse1311 * cse808 * cse822 * cse872 * cse873 - cse1312 * cse823) -
					cse830 * (-cse1240 * cse1312 + cse1311 * cse864 * cse865));
		jac(18, 17) = cse1241;
		jac(18, 18) = cse1228;
		jac(18, 19) = -cse1150 * (-4.1664736580507933e-10 * cse1322 + 8.1872148910410564e-9 * cse1323) + cse1279 * cse583 +
			      8.1872148910410565e-33 * cse1314 + 4.7575374194964615e-36 * cse1315 + 3.4748025752386779e-37 * cse1316 +
			      cse1320 * (-1.2171193564238968e-12 * cse1318 - 1.7629333331721322e-10 * cse1319) - 2.466269795723557e-53 * cse1325 -
			      3.1317711691727702e-38 * cse1326;
		jac(18, 21) = -cse1150 * (-6.365129963657969e-10 * cse1322 + 1.2507624216266397e-8 * cse1323) + cse1279 * cse584 +
			      1.2507624216266398e-32 * cse1314 + 7.2680992290799608e-36 * cse1315 + 5.3084626964364086e-37 * cse1316 +
			      cse1320 * (-1.8593956234313972e-12 * cse1318 - 2.6932367041905241e-10 * cse1319) - 7.4187514601998589e-44 * cse1325 -
			      9.4206367748569615e-29 * cse1326;
		jac(19, 0) = cse1329;
		jac(19, 1) = cse1329;
		jac(19, 3) = -cse1330 - cse1331 * cse247;
		jac(19, 4) = -cse1330;
		jac(19, 10) = cse1329;
		jac(19, 12) = -cse510;
		jac(19, 14) = -cse1328 - cse294;
		jac(19, 15) = -cse1328 - cse301;
		jac(19, 16) = -3.7583398381383454e-21 * cse1327;
		jac(19, 19) = -1.252779946046115e-21 * cse1332 - cse1333;
		jac(20, 0) = cse1337;
		jac(20, 1) = cse1337;
		jac(20, 3) = -cse1331 * cse1334 - cse1338;
		jac(20, 4) = -cse1338;
		jac(20, 10) = cse1337;
		jac(20, 12) = -1.9917698086432708e-17 * cse1334;
		jac(20, 14) = -4.2188032355736613e-18 * cse1334 - cse1336;
		jac(20, 15) = -1.7141251340095377e-18 * cse1334 - cse1336;
		jac(20, 16) = -2.7305048399001034e-21 * cse1335;
		jac(20, 20) = -9.1016827996670106e-22 * cse1332 - cse1333;
		jac(21, 0) = cse1341;
		jac(21, 1) = cse1341;
		jac(21, 3) = -cse1342 - cse1343 * cse251;
		jac(21, 4) = -cse1342;
		jac(21, 7) = -cse503;
		jac(21, 9) = -cse627;
		jac(21, 10) = cse1341;
		jac(21, 12) = -cse511;
		jac(21, 14) = -cse1340 - cse295;
		jac(21, 15) = -cse1340 - cse302;
		jac(21, 16) = -6.175361757103777e-21 * cse1339;
		jac(21, 21) = -2.0584539190345922e-21 * cse1332 - cse1344;
		jac(22, 0) = cse1348;
		jac(22, 1) = cse1348;
		jac(22, 3) = -cse1343 * cse1345 - cse1349;
		jac(22, 4) = -cse1349;
		jac(22, 7) = -cse1345 * cse502;
		jac(22, 9) = -cse1345 * cse626;
		jac(22, 10) = cse1348;
		jac(22, 12) = -1.9088222220940885e-17 * cse1345;
		jac(22, 14) = -1.168047934563505e-18 * cse1345 - cse1347;
		jac(22, 15) = -4.1730320353928637e-18 * cse1345 - cse1347;
		jac(22, 16) = -5.0568154723242353e-21 * cse1346;
		jac(22, 22) = -1.6856051574414119e-21 * cse1332 - cse1344;
		// $$JAFF END
		// jaff's energy row and column are those of E = rho e: the row is d(dE/dt)
		// and the column d/dE. Change them to e, with rho fixed during the burn:
		// the row divides by rho, the column multiplies by rho, and d(de/dt)/de =
		// d(dE/dt)/dE is unchanged.
		// $$JAFF SUB nspec
		for (int i = 0; i < System::neqs; i++) {
			if (i != 18) {
				jac(18, i) /= state.rho;
				jac(i, 18) *= state.rho;
			}
		}
		// $$JAFF END
	}

	/// The right-hand side and the Jacobian at one state (jaff's rhs_jacobian):
	/// every temperature-only rate coefficient is computed once and read by
	/// both, its temperature derivative is k d(ln k)/dT, and one CSE spans the
	/// rates, the RHS and the Jacobian. The integrator calls it at the start of
	/// each step, where it needs both, and actual_rhs for the other stages. The
	/// matrix comes in zeroed; only the structurally nonzero entries are written.
	template <typename BurnT, class MatrixType>
	AMREX_GPU_HOST_DEVICE static void
	actual_rhs_and_jac(const BurnT &state, amrex::GpuArray<amrex::Real, quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>::neqs> &ydot,
			   MatrixType &jac)
	{
		using System = quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>;
		static_assert(BurnT::traits::integrate_rad_flux, "a jaff network needs Microphysics_Traits::integrate_rad_flux");
		// $$JAFF REPEAT idx, rhs, cse IN rhs_jacobian { USE quokka }
		const amrex::Real lnT = std::log(state.T);
		const amrex::Real cse0 = 1.0 / state.T;
		const amrex::Real ln_cse0 = std::log(cse0);
		const amrex::Real cse1 = 26.218500000000002 * std::sqrt(3) * std::sqrt(cse0) + 0.62;
		const amrex::Real cse2 = std::log(state.T);
		const amrex::Real cse3 = M_LN10;
		const amrex::Real cse4 = 1.0 / cse3;
		const amrex::Real cse5 = cse2 * cse4;
		const amrex::Real cse6 = 0.044330000000000001 * cse5 - 0.28520000000000001;
		const amrex::Real cse7 = cse5 * cse6 - 1.6759999999999999;
		const amrex::Real cse8 = cse5 * cse7 + 11.19;
		const amrex::Real cse9 = (1.0 / std::sqrt(state.T));
		const amrex::Real cse10 = 19.413464793123975 * std::exp((-0.52000000000000002) * lnT);
		const amrex::Real cse11 = std::exp((-3.0 / 2.0) * lnT);
		const amrex::Real cse12 = std::exp(-15230.0 * cse0);
		const amrex::Real cse13 = std::exp(-73.799999999999997 * cse0);
		const amrex::Real cse14 = std::exp(-12.17 * cse0);
		const amrex::Real cse15 = 1.6339999999999999e-6 * cse12 + 9.7930000000000004e-9 * cse13 + 6.3460000000000002e-9 * cse14;
		const amrex::Real cse16 = std::sqrt(state.T);
		const amrex::Real cse17 = 1.0 / cse16;
		const amrex::Real cse18 = 0.00071740367932898903 * cse16 + 1;
		const amrex::Real cse19 = std::exp(-49550.0 * cse0);
		const amrex::Real cse20 = 0.15970000000000001 * cse19;
		const amrex::Real cse21 = cse20 + 1.7848999999999999;
		const amrex::Real cse22 = 12.244387999457523 * cse16 + 1;
		const amrex::Real cse23 = 0.21509999999999996 - cse20;
		const amrex::Real cse24 = std::pow(cse18, -cse21) * std::pow(cse22, -cse23);
		const amrex::Real cse25 = 2.4460185352936309e-10 * cse17 * cse24;
		const amrex::Real cse26 = 114.81344434488582 * std::exp((0.40699999999999997) * ln_cse0) + 1;
		const amrex::Real ln_cse26 = std::log(cse26);
		const amrex::Real cse27 = (cse0 * std::sqrt(cse0)) * std::exp((-2.242) * ln_cse26);
		const amrex::Real cse28 = std::exp(-227 * cse0);
		const amrex::Real cse29 =
		    cse28 * (4.0000000000000001e-10 * std::exp((0.0066899999999999998) * lnT) + 1.1000000000000001e-11 * std::exp((0.51700000000000002) * lnT));
		const amrex::Real cse30 = 1.0 / state.xn[2];
		const amrex::Real cse31 = cse16 * cse30;
		const amrex::Real cse32 = 0.00011320754716981131 * cse31;
		const amrex::Real cse33 = cse32 * state.xn[3];
		const amrex::Real cse34 = std::min(10000000000.0, cse33);
		const amrex::Real cse35 = cse34 + 1;
		const amrex::Real cse36 = 1.0 / cse35;
		const amrex::Real cse37 = 1.6000000000000001e-9 * cse36;
		const amrex::Real cse38 = cse37 * state.xn[17];
		const amrex::Real cse39 = cse38 * state.xn[3];
		const amrex::Real cse40 = cse34 * cse39;
		const amrex::Real cse1425 = 1.2599999999999999e-13 * std::exp(-22.5 * cse0);
		const amrex::Real cse41 = 1.0 * cse1425;
		const amrex::Real cse42 = cse41 * state.xn[6];
		const amrex::Real cse43 = cse42 * state.xn[3];
		const amrex::Real cse44 = state.c_hat * state.rn[2 * 0];
		const amrex::Real cse45 = 1.7141251340095377e-18 * cse44;
		const amrex::Real cse46 = state.c_hat * state.rn[2 * 1];
		const amrex::Real cse47 = 4.1730320353928637e-18 * cse46;
		const amrex::Real cse48 = cse45 + cse47;
		const amrex::Real cse49 = cse48 * state.xn[15];
		const amrex::Real cse50 = 1.0 * state.xn[15];
		const amrex::Real cse1440 = 3.5000000000000002e-11;
		const amrex::Real cse51 = cse1440 * cse50;
		const amrex::Real cse52 = cse51 * state.xn[11];
		const amrex::Real cse53 = std::max(1.0e-35, 2.0000000000000002e-15 * state.ncol[ncol_H2]);
		const amrex::Real cse54 = 0.035000000000000003 * (1.0 / std::sqrt(cse53 + 1)) * std::exp(-0.00084999999999999995 * std::sqrt(cse53 + 1.0)) +
					  0.96499999999999997 * (1.0 / ((cse53 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0) *
									(cse53 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0)));
		const amrex::Real cse55 = cse44 * cse54;
		const amrex::Real cse56 = cse46 * cse54;
		const amrex::Real cse57 = 1.2216858566394234e-27 * cse55 + 3.6749360303422164e-18 * cse56;
		const amrex::Real cse58 = 2 * state.xn[3];
		const amrex::Real cse59 = ((state.xn[3]) * (state.xn[3]));
		const amrex::Real cse60 = std::exp((4.1881000000000004) * lnT);
		const amrex::Real cse61 = 6.7610000000000002e-6 * state.T + 1;
		const amrex::Real ln_cse61 = std::log(cse61);
		const amrex::Real cse62 = std::exp((5.6881000000000004) * ln_cse61);
		const amrex::Real cse63 = 54657.400000000001 * cse0;
		const amrex::Real cse64 = std::exp(-cse63);
		const amrex::Real cse65 = cse64 / cse62;
		const amrex::Real cse66 = std::log(5.9959999999999997e-30 * cse60 * cse65);
		const amrex::Real cse67 = cse58 + state.xn[0] + state.xn[1] + state.xn[10] + state.xn[14] + state.xn[15] + 3 * state.xn[16] + 2 * state.xn[4];
		const amrex::Real cse68 = 1.0 / cse67;
		const amrex::Real cse69 = std::log((1.0 / 10000.0) * state.T);
		const amrex::Real cse70 = (1.0 / (cse3 * cse3));
		const amrex::Real cse71 = (cse69 * cse69) * cse70;
		const amrex::Real cse72 = std::exp(2.302585092994046 * (1.3 * cse4 * cse69 - 1.6200000000000001 * cse71 - 4.8449999999999998));
		const amrex::Real cse73 = cse68 * cse72 * state.xn[3];
		const amrex::Real cse74 = std::exp(2.302585092994046 * (0.41599999999999998 * cse4 * cse69 + 0.32700000000000001 * cse71 - 3));
		const amrex::Real cse75 = cse68 * cse74 * state.xn[0];
		const amrex::Real cse76 = cse73 + cse75;
		const amrex::Real cse77 = cse76 < 1.0e-50;
		const amrex::Real cse78 = ((cse77) ? (1.0e-50) : (cse76));
		const amrex::Real cse79 = cse67 * cse78;
		const amrex::Real cse80 = cse79 + 1;
		const amrex::Real cse81 = cse4 / cse80;
		const amrex::Real cse82 = std::log(1.3000000000000001e-9 * std::exp(-53300 * cse0));
		const amrex::Real cse83 = cse79 * cse81;
		const amrex::Real cse84 = std::exp(2.302585092994046 * (cse66 * cse81 + cse82 * cse83));
		const amrex::Real cse85 = state.T > 700.0;
		const amrex::Real cse86 = ((cse85) ? (cse84) : (0.0));
		const amrex::Real cse87 = cse59 * cse86;
		const amrex::Real cse1443 = cse29;
		const amrex::Real cse88 = cse1443 * state.xn[11];
		const amrex::Real cse89 = cse88 * state.xn[1];
		const amrex::Real cse1444 = 7.5e-10 * std::exp((-0.45800000000000002) * lnT) + 4.9899999999999997e-11 * std::exp((0.40500000000000003) * lnT);
		const amrex::Real cse90 = cse1444 * state.xn[17];
		const amrex::Real cse91 = cse90 * state.xn[0];
		const amrex::Real cse92 = 63590 * cse0 + 1;
		const amrex::Real cse93 = std::exp(-cse92);
		const amrex::Real cse94 = std::log(6.6699999999999996e-12 * cse16 * cse93);
		const amrex::Real cse95 = std::log(3.5199999999999998e-9 * std::exp(-43900 * cse0));
		const amrex::Real cse96 = std::exp(2.302585092994046 * (cse81 * cse94 + cse83 * cse95));
		const amrex::Real cse97 = ((cse85) ? (cse96) : (0.0));
		const amrex::Real cse98 = cse97 * state.xn[0];
		const amrex::Real cse1432 = 4.36e-8 * cse10;
		const amrex::Real cse99 = 3.0 * cse1432;
		const amrex::Real cse100 = cse99 * state.xn[16];
		const amrex::Real cse101 = 3.2000000000000001e-9 * cse36 * state.xn[17];
		const amrex::Real cse102 = 6.0000000000000005e-15 * d2g;
		const amrex::Real cse103 = cse102 * state.xn[0];
		const amrex::Real cse104 = std::exp((-1.3) * lnT);
		const amrex::Real cse105 = std::exp(-23 * cse0);
		const amrex::Real cse106 = cse104 * cse105;
		const amrex::Real cse107 = 4.2899999999999994e-13 * cse106;
		const amrex::Real cse108 = cse107 * state.xn[8];
		const amrex::Real cse109 = 4.2188032355736613e-18 * cse44;
		const amrex::Real cse110 = 1.168047934563505e-18 * cse46;
		const amrex::Real cse111 = cse109 + cse110;
		const amrex::Real cse112 = cse111 * state.xn[14];
		const amrex::Real cse1428 = 7.0000000000000004e-11;
		const amrex::Real cse113 = 1.0 * cse1428;
		const amrex::Real cse114 = cse113 * state.xn[14];
		const amrex::Real cse115 = cse114 * state.xn[11];
		const amrex::Real cse116 = cse112 + cse115;
		const amrex::Real cse1442 = 1.3500000000000001e-9 * cse1;
		const amrex::Real cse117 = cse1442 * cse50;
		const amrex::Real cse118 = cse117 * state.xn[6];
		const amrex::Real cse1429 = 7.9513696852843416e-10 * std::exp((-0.33900000000000002) * lnT) * std::exp(0.11 * cse0);
		const amrex::Real cse119 = 1.0000000000000002 * cse1429;
		const amrex::Real cse120 = cse119 * state.xn[15];
		const amrex::Real cse121 = cse120 * state.xn[7];
		const amrex::Real cse122 = cse118 + cse121;
		const amrex::Real cse1431 = 2.3400000000000001e-8 * cse10;
		const amrex::Real cse123 = 1.0 * cse1431;
		const amrex::Real cse124 = cse123 * state.xn[16];
		const amrex::Real cse125 = cse124 * state.xn[2];
		const amrex::Real cse126 = cse36 * state.xn[16];
		const amrex::Real cse127 = 1.99e-9 * std::exp((-0.19) * lnT);
		const amrex::Real cse128 = cse126 * cse127;
		const amrex::Real cse129 = cse128 * state.xn[11];
		const amrex::Real cse130 = cse125 + cse129;
		const amrex::Real cse1435 = 1.7599999999999999e-9 * std::exp((0.042000000000000003) * lnT) * std::exp(-2.145922746781116e-5 * state.T);
		const amrex::Real cse131 = 1.0 * cse1435;
		const amrex::Real cse132 = cse131 * state.xn[4];
		const amrex::Real cse133 = cse132 * state.xn[3];
		const amrex::Real cse1439 = 2.8142984982406731e-11 * std::exp((0.26000000000000001) * lnT);
		const amrex::Real cse134 = 0.99999999999999978 * cse1439;
		const amrex::Real cse135 = cse134 * state.xn[14];
		const amrex::Real cse136 = cse135 * state.xn[0];
		const amrex::Real cse1436 = 6.3999999999999996e-10;
		const amrex::Real cse137 = 0.99999999999999989 * cse1436;
		const amrex::Real cse138 = cse137 * state.xn[4];
		const amrex::Real cse139 = cse138 * state.xn[0];
		const amrex::Real cse140 = cse133 - cse136 - cse139;
		const amrex::Real cse141 = 6.5199999999999996 * crate;
		const amrex::Real cse142 = cse141 * state.xn[9];
		const amrex::Real cse143 = 1.5 * cse68 * state.xn[0] + 2.2999999999999998 * cse68 * state.xn[3];
		const amrex::Real cse144 = crate * cse143;
		const amrex::Real cse145 = cse144 * state.xn[0];
		const amrex::Real cse146 = std::log(8.6173000000000003e-5 * state.T);
		const amrex::Real cse147 = (cse146 * cse146);
		const amrex::Real cse148 = (cse146 * cse146 * cse146);
		const amrex::Real cse149 = (cse146 * cse146 * cse146 * cse146);
		const amrex::Real cse150 = (cse146 * cse146 * cse146 * cse146 * cse146);
		const amrex::Real cse151 = (cse146 * cse146 * cse146 * cse146 * cse146 * cse146);
		const amrex::Real cse152 = (cse146 * cse146 * cse146 * cse146 * cse146 * cse146 * cse146);
		const amrex::Real cse153 = std::exp(-2.0391498499999999e-6 * (cse146 * cse146 * cse146 * cse146 * cse146 * cse146 * cse146 * cse146) -
						    5.7393287500000003 * cse147 + 1.56315498 * cse148 - 0.28770560000000001 * cse149 +
						    0.034825597700000002 * cse150 - 0.00263197617 * cse151 + 0.000111954395 * cse152);
		const amrex::Real cse154 = 5.9078447946382801e-70 * cse153 * std::exp((13.536555999999999) * lnT);
		const amrex::Real cse155 = ((cse85) ? (cse154) : (0.0));
		const amrex::Real cse156 = cse155 * state.xn[2];
		const amrex::Real cse157 = cse156 * state.xn[0];
		const amrex::Real cse1434 = 1.05849862170431e-5 * std::exp((-0.64000000000000001) * lnT);
		const amrex::Real cse158 = 1.0 * cse1434;
		const amrex::Real cse159 = cse158 * state.xn[10];
		const amrex::Real cse160 = cse159 * state.xn[2];
		const amrex::Real cse1437 = 4.8813573300130879e-6 * cse27;
		const amrex::Real cse161 = 0.99999999999999989 * cse1437;
		const amrex::Real cse162 = cse161 * state.xn[2];
		const amrex::Real cse163 = cse162 * state.xn[1];
		const amrex::Real cse164 = cse67 * d2g;
		const amrex::Real cse165 = 144.3803103740824 * state.rn[2 * 0] + 220.5700827961663 * state.rn[2 * 1];
		const amrex::Real cse166 = state.c_hat * cse165;
		const amrex::Real cse167 = cse166 * cse31;
		const amrex::Real ln_cse167 = std::log(cse167);
		const amrex::Real cse168 = 5.670589618368584e-11 * cse167;
		const amrex::Real ln_cse168 = std::log(cse168);
		const amrex::Real cse169 = 1.102e-5 * cse2 + 0.4723;
		const amrex::Real cse170 = std::exp((-cse169) * ln_cse168);
		const amrex::Real cse171 = cse170 * std::exp((0.015859999999999999) * lnT);
		const amrex::Real cse172 = 508.69999999999999 * cse171;
		const amrex::Real cse173 = cse172 + 1;
		const amrex::Real cse174 = std::exp((1.3779999999999999) * ln_cse167);
		const amrex::Real cse175 = 6.1318063394789369e-20 * cse174;
		const amrex::Real cse176 = cse173 * cse175 + 1;
		const amrex::Real cse177 = 1.0 / cse176;
		const amrex::Real cse178 = 1.2250000000000001e-11 * cse177;
		const amrex::Real cse179 = cse164 * cse178;
		const amrex::Real cse180 = cse179 * state.xn[1];
		const amrex::Real cse181 = -cse142 - cse145 - cse157 + cse160 + cse163 + cse180;
		const amrex::Real cse182 = -cse89 + cse91;
		const amrex::Real cse183 = cse144 * cse58;
		const amrex::Real cse184 = -cse183;
		const amrex::Real cse1441 = 1.4595000290776864e-10 * std::exp((-0.62) * lnT);
		const amrex::Real cse185 = 1.0000000000000002 * cse1441;
		const amrex::Real cse186 = cse185 * state.xn[13];
		const amrex::Real cse187 = 1.9917698086432708e-17 * cse44;
		const amrex::Real cse188 = 1.9088222220940885e-17 * cse46;
		const amrex::Real cse189 = cse187 + cse188;
		const amrex::Real cse190 = crate * cse68;
		const amrex::Real cse191 = 16800.0 * cse190;
		const amrex::Real cse192 = cse191 * state.xn[12];
		const amrex::Real cse193 = 7.538e-5 * cse2 + 0.89639999999999997;
		const amrex::Real cse194 = std::exp((-cse193) * ln_cse168);
		const amrex::Real cse195 = cse194 * std::exp((1.635e-6) * lnT);
		const amrex::Real cse196 = 43750.0 * cse195;
		const amrex::Real cse197 = cse196 + 1;
		const amrex::Real cse198 = std::exp((1.8740000000000001) * ln_cse167);
		const amrex::Real cse199 = 3.5685736926834518e-27 * cse198;
		const amrex::Real cse200 = cse197 * cse199 + 1;
		const amrex::Real cse201 = 1.0 / cse200;
		const amrex::Real cse202 = 2.1659999999999999e-12 * cse201;
		const amrex::Real cse203 = cse164 * cse202;
		const amrex::Real cse204 = cse186 * state.xn[2] - cse189 * state.xn[12] - cse192 * state.xn[3] + cse203 * state.xn[13];
		const amrex::Real cse1422 = 1.1000000000000001 * crate;
		const amrex::Real cse205 = 1.0 * cse1422;
		const amrex::Real cse1430 = 9.9999999999999994e-12 * cse8 * cse9;
		const amrex::Real cse206 = 0.99999999999999989 * cse1430;
		const amrex::Real cse207 = cse206 * state.xn[6];
		const amrex::Real cse208 = 5.4939999999999998e-7 * cse2 + 0.49559999999999998;
		const amrex::Real cse209 = std::exp((-cse208) * ln_cse168);
		const amrex::Real cse210 = cse209 * std::exp((3.9029999999999999e-7) * lnT);
		const amrex::Real cse211 = 5115.0 * cse210;
		const amrex::Real cse212 = cse211 + 1;
		const amrex::Real cse213 = std::exp((1.512) * ln_cse167);
		const amrex::Real cse214 = 1.0246966183388124e-22 * cse213;
		const amrex::Real cse215 = cse212 * cse214 + 1;
		const amrex::Real cse216 = 1.0 / cse215;
		const amrex::Real cse217 = 5.5720000000000001e-12 * cse216;
		const amrex::Real cse218 = cse164 * cse217;
		const amrex::Real cse219 = -cse205 * state.xn[5] + cse207 * state.xn[2] + cse218 * state.xn[6];
		const amrex::Real cse220 = std::exp((0.00231) * ln_cse0);
		const amrex::Real cse221 = std::exp((-1.5) * lnT);
		const amrex::Real cse222 = std::exp(-8110.0 * cse0);
		const amrex::Real cse223 = std::exp(-26.600000000000001 * cse0);
		const amrex::Real cse224 = std::exp(-7.6200000000000001 * cse0);
		const amrex::Real cse225 = std::exp(-1.3799999999999999 * cse0);
		const amrex::Real cse226 = 0.00015100000000000001 * cse222 + 1.31e-7 * cse223 + 3.4e-8 * cse224 + 6.9699999999999997e-9 * cse225;
		const amrex::Real cse227 = 1.0537934368240927e-9 * cse220 + cse221 * cse226;
		const amrex::Real cse228 = cse227 * state.xn[16];
		const amrex::Real cse229 = cse228 * state.xn[7];
		const amrex::Real cse230 = 1.0 * cse1432;
		const amrex::Real cse231 = cse230 * state.xn[16];
		const amrex::Real cse232 = cse127 * state.xn[11];
		const amrex::Real cse233 = cse68 * state.xn[2];
		const amrex::Real cse234 = cse233 < 1.0e-50;
		const amrex::Real cse235 = ((cse234) ? (10000000000.0) : (cse33));
		const amrex::Real cse236 = cse235 + 1;
		const amrex::Real cse237 = 1.0 / cse236;
		const amrex::Real cse238 = cse235 * cse237;
		const amrex::Real cse239 = cse238 * state.xn[16];
		const amrex::Real cse240 = cse232 * cse239;
		const amrex::Real cse241 = cse130 + cse229 + cse231 * state.xn[2] + cse240;
		const amrex::Real cse1423 = 3.8500000000000001 * crate;
		const amrex::Real cse242 = 0.99999999999999989 * cse1423;
		const amrex::Real cse243 = cse242 * state.xn[7];
		const amrex::Real cse1433 = cse11 * cse15 + cse25;
		const amrex::Real cse244 = cse1433 * state.xn[8];
		const amrex::Real cse245 = cse244 * state.xn[2];
		const amrex::Real cse246 = 0.00013329999999999999 * cse2 + 0.81200000000000006;
		const amrex::Real cse247 = std::exp((-cse246) * ln_cse168);
		const amrex::Real cse248 = cse247 * std::exp((0.04845) * lnT);
		const amrex::Real cse249 = 433.10000000000002 * cse248;
		const amrex::Real cse250 = cse249 + 1;
		const amrex::Real cse251 = std::exp((1.1279999999999999) * ln_cse167);
		const amrex::Real cse252 = 1.6851501542381549e-14 * cse251;
		const amrex::Real cse253 = cse250 * cse252 + 1;
		const amrex::Real cse254 = 1.0 / cse253;
		const amrex::Real cse255 = 4.5579999999999993e-11 * cse254;
		const amrex::Real cse256 = cse164 * cse255;
		const amrex::Real cse257 = cse256 * state.xn[8];
		const amrex::Real cse258 = 1040.0 * cse190;
		const amrex::Real cse259 = cse258 * state.xn[7];
		const amrex::Real cse260 = cse259 * state.xn[3];
		const amrex::Real cse261 = 1.4855039799802863e-17 * interp_8_shielding_H2(state.ncol[ncol_H2]) * interp_8_shielding_self(state.ncol[ncol_C]);
		const amrex::Real cse262 = cse261 * cse46;
		const amrex::Real cse263 = cse262 * state.xn[7];
		const amrex::Real cse264 = -cse243 + cse245 + cse257 - cse260 - cse263;
		const amrex::Real cse265 = 3.2999999999999996e-13 * cse106;
		const amrex::Real cse266 = cse265 * state.xn[8];
		const amrex::Real cse267 = cse266 * state.xn[3];
		const amrex::Real cse268 = cse39 + cse40;
		const amrex::Real cse269 = cse267 + cse268;
		const amrex::Real cse270 = -cse229;
		const amrex::Real cse271 = -cse240;
		const amrex::Real cse272 = cse57 * state.xn[3];
		const amrex::Real cse273 = cse97 * state.xn[3];
		const amrex::Real cse274 = cse273 * state.xn[0];
		const amrex::Real cse1424 = 1.6999999999999999e-9;
		const amrex::Real cse275 = 0.99999999999999989 * cse1424;
		const amrex::Real cse276 = cse275 * state.xn[16];
		const amrex::Real cse277 = cse276 * state.xn[9];
		const amrex::Real cse278 = 1.0 * state.xn[6];
		const amrex::Real cse1438 = 7.2000000000000002e-15;
		const amrex::Real cse279 = cse1438 * cse278;
		const amrex::Real cse280 = cse279 * state.xn[3];
		const amrex::Real cse281 = cse280 + cse43;
		const amrex::Real cse1426 = 1.6000000000000001e-9;
		const amrex::Real cse282 = cse1426 * state.xn[9];
		const amrex::Real cse283 = cse278 * cse282;
		const amrex::Real cse284 = cse118 + cse219 + cse281 + cse283;
		const amrex::Real cse285 = 9.8999999999999995e-14 * cse106;
		const amrex::Real cse286 = cse285 * state.xn[8];
		const amrex::Real cse287 = cse136 + cse270;
		const amrex::Real cse288 = 1.6797535700197549e-17 * interp_10_shielding_H2(state.ncol[ncol_H2]) * interp_10_shielding_self(state.ncol[ncol_CO]);
		const amrex::Real cse289 = cse288 * cse46;
		const amrex::Real cse290 = cse289 * state.xn[9];
		const amrex::Real cse291 = 184.0 * cse190;
		const amrex::Real cse292 = cse291 * state.xn[9];
		const amrex::Real cse293 = cse292 * state.xn[3];
		const amrex::Real cse294 = -cse121 + cse290 + cse293;
		const amrex::Real cse1427 = 9.1500000000000005e-10 * cse1;
		const amrex::Real cse295 = 0.99999999999999989 * cse1427;
		const amrex::Real cse296 = cse295 * state.xn[15];
		const amrex::Real cse297 = cse296 * state.xn[8];
		const amrex::Real cse298 = cse142 - cse160 + cse277;
		const amrex::Real cse299 = -cse115 + cse283;
		const amrex::Real cse300 = cse271 + cse49 + cse52;
		const amrex::Real cse301 = cse233 + 0.070000000000000007;
		const amrex::Real cse302 = 1.0 / cse301;
		const amrex::Real cse303 = std::sqrt(cse233 * cse302);
		const amrex::Real cse304 = 26.399999999999999 * cse303;
		const amrex::Real cse305 = cse304 + 6.5;
		const amrex::Real cse306 = psigd_coll_interp2d(cse67, state.T);
		const amrex::Real cse307 = 100.0 * cse164;
		const amrex::Real cse308 = cse306 * cse307;
		const amrex::Real cse309 = 2.8677558904086741e-6 * cse166 + 4.1397668473788896;
		const amrex::Real ln_cse309 = std::log(cse309);
		const amrex::Real cse310 = -2.1544346900318838 * std::exp((1.0 / 6.0) * ln_cse309) + state.T;
		const amrex::Real cse311 = (cse67 * cse67);
		const amrex::Real cse312 = cse311 * d2g;
		const amrex::Real cse313 = 3.2000000000000002e-32 * cse16;
		const amrex::Real cse314 = cse312 * cse313;
		const amrex::Real cse315 = cse310 * cse314;
		const amrex::Real cse316 = 2.25 * std::exp((0.14699999999999999) * lnT) + 5.2199999999999998;
		const amrex::Real cse317 = cse168 + 50;
		const amrex::Real ln_cse317 = std::log(cse317);
		const amrex::Real cse318 = 0.0043 * std::exp((0.69199999999999995) * ln_cse317) + 1;
		const amrex::Real cse319 = 0.049959999999999997 * std::exp((0.43099999999999999) * ln_cse317) * cse318 + 1;
		const amrex::Real cse320 = 1.0 / cse319;
		const amrex::Real cse321 = state.xn[0] * state.xn[2];
		const amrex::Real cse322 = std::exp((0.14999999999999999) * lnT);
		const amrex::Real cse323 = 0.00088240812266906664 * std::exp((0.65000000000000002) * lnT) + 1;
		const amrex::Real cse324 = 1.0 / cse323;
		const amrex::Real cse325 = cse322 * cse324;
		const amrex::Real cse326 = cse324 * state.xn[2];
		const amrex::Real cse327 = cse322 * cse326;
		const amrex::Real cse328 = 4.4460389837719569e-9 * cse327 + 626500000.0;
		const amrex::Real cse329 = 1.0 / cse328;
		const amrex::Real cse330 = std::exp(-118400.0 * cse0);
		const amrex::Real cse331 = cse329 * cse330;
		const amrex::Real cse332 = 1.3338116951315871e-8 * cse331;
		const amrex::Real cse333 = cse327 * cse332 + 1;
		const amrex::Real cse334 = 1.0 / cse333;
		const amrex::Real cse335 = cse331 * cse334;
		const amrex::Real cse336 = 1.3659996292638161e-10 * cse335;
		const amrex::Real cse337 = cse325 * cse336;
		const amrex::Real cse338 = state.xn[2] + 1.0e-50;
		const amrex::Real cse339 = 1.0 / cse338;
		const amrex::Real cse340 = cse16 * cse339;
		const amrex::Real cse341 = 5.670589618368584e-11 * cse166;
		const amrex::Real cse342 = cse340 * cse341 + 50;
		const amrex::Real cse343 = std::log(cse342);
		const amrex::Real cse344 = 0.45350000000000001 + 2.234 / cse343;
		const amrex::Real cse345 = std::exp((cse344)*lnT);
		const amrex::Real cse346 = 1.4419999999999999 - 0.050889999999999998 * cse343;
		const amrex::Real cse347 = std::exp(cse343 * cse346);
		const amrex::Real cse348 = 1.8998126555754159e-29 * cse345 * cse347;
		const amrex::Real cse349 = cse164 * cse348;
		const amrex::Real cse350 = cse349 * state.xn[2];
		const amrex::Real cse351 = cse4 * std::log(cse67);
		const amrex::Real cse352 = cse351 < 2;
		const amrex::Real cse353 = cse351 < 4;
		const amrex::Real cse354 = cse351 < 7;
		const amrex::Real cse355 = cse351 < 10;
		const amrex::Real cse356 =
		    ((cse352) ? (10.0)
			      : ((cse353) ? ((3.0 / 2.0) * cse351 + 7)
					  : ((cse354) ? ((4.0 / 3.0) * cse351 + 23.0 / 3.0) : ((cse355) ? ((1.0 / 3.0) * cse351 + 44.0 / 3.0) : (18.0)))));
		const amrex::Real cse357 = 0.001 * state.T;
		const amrex::Real cse358 = cse357 + 1;
		const amrex::Real cse359 = (1.0 / (cse358 * cse358));
		const amrex::Real cse360 = 1.0 / cse358;
		const amrex::Real cse361 = std::exp(2.302585092994046 * (-2.0230000000000001 * cse359 - 3.6709999999999998 * cse360 - 11.08));
		const amrex::Real cse362 = cse361 * cse68;
		const amrex::Real cse363 = cse362 * state.xn[3];
		const amrex::Real cse364 = std::exp(2.302585092994046 * (-2.3900000000000001 * cse359 + 0.055500000000000001 * cse360 - 11.06));
		const amrex::Real cse365 = cse364 * cse68;
		const amrex::Real cse366 = cse365 * state.xn[0];
		const amrex::Real cse367 = cse363 + cse366;
		const amrex::Real cse368 = 1.0 / cse367;
		const amrex::Real cse369 = cse57 + 1.9999999999999999e-7;
		const amrex::Real cse370 = cse368 * cse369;
		const amrex::Real cse371 = cse370 * cse68 + 1;
		const amrex::Real cse372 = 1.0 / cse371;
		const amrex::Real cse373 = 6.7291434000000005e-12 * cse372 + 3.2043540000000002e-13;
		const amrex::Real cse374 = std::sqrt(cse0);
		const amrex::Real cse375 = cse374 * state.xn[2];
		const amrex::Real cse376 = (1.0 / 100.0) * state.T;
		const amrex::Real ln_cse376 = std::log(cse376);
		const amrex::Real cse377 = std::log(cse376);
		const amrex::Real cse378 = 0.0086999999999999994 * cse377;
		const amrex::Real cse379 = std::exp((cse378 + 0.12809999999999999) * ln_cse376);
		const amrex::Real cse380 = 7.5799999999999997e-10 * cse379 * state.xn[0];
		const amrex::Real cse381 = state.T < 500;
		const amrex::Real cse382 = std::exp((0.070000000000000007) * lnT);
		const amrex::Real cse383 = 0.75 * ((cse381) ? (1.1000000000000001e-13 * state.T + 5.3300000000000002e-10) : (3.74757785025e-10 * cse382)) +
					   0.25 * ((cse381) ? (3.3000000000000001e-13 * state.T + 4.4299999999999997e-10) : (3.8899728635600003e-10 * cse382));
		const amrex::Real cse384 = 4.5299999999999998e-6 * cse375 + cse380 + cse383 * state.xn[3];
		const amrex::Real cse385 = cse384 + 2.3e-6;
		const amrex::Real cse386 = std::exp(-91.209999999999994 * cse0);
		const amrex::Real cse387 = cse386 / cse385;
		const amrex::Real cse388 = cse384 * cse387;
		const amrex::Real cse389 = 2 * cse387;
		const amrex::Real cse390 = cse384 * cse389 + 1;
		const amrex::Real cse391 = 1.0 / cse390;
		const amrex::Real cse392 = 5.7927337833400002e-20 * cse391;
		const amrex::Real cse393 = cse388 * cse392;
		const amrex::Real cse394 = 2.5634832e-11 * cse372 + 6.4087080000000003e-13;
		const amrex::Real cse395 = 1.0 / cse143;
		const amrex::Real cse396 = cse305 * cse68;
		const amrex::Real cse397 = 2.0 * cse68;
		const amrex::Real cse398 = cse356 * cse397;
		const amrex::Real cse399 = cse396 * state.xn[0] + cse398 * state.xn[3];
		const amrex::Real cse400 = std::sqrt(std::sqrt(state.T));
		const amrex::Real cse401 = 680.10000000000002 / cse400;
		const amrex::Real cse402 = 1.75 * cse400;
		const amrex::Real cse403 = cse401 * state.xn[2] + cse402 * state.xn[0] + state.xn[3];
		const amrex::Real cse404 = std::exp(-state.T);
		const amrex::Real cse405 = 1 - cse404;
		const amrex::Real ln_cse405 = std::log(cse405);
		const amrex::Real cse406 = std::exp((1000) * ln_cse405);
		const amrex::Real cse407 = l0_co_interp1d(state.T);
		const amrex::Real cse408 = 1.0 / cse407;
		const amrex::Real cse409 = M_SQRT2;
		const amrex::Real cse410 = std::max(gradv, 5.7445531974402559e-18 * cse16 * cse409);
		const amrex::Real cse411 = 100000.0 / cse410;
		const amrex::Real cse412 = state.ncol[ncol_H] <= 0;
		const amrex::Real cse413 = cse68 * state.ncol[ncol_H];
		const amrex::Real cse414 = std::max(1.0000000000000001e-30, cse413);
		const amrex::Real cse415 = 1756701454.4310031 * state.ncol[ncol_CO];
		const amrex::Real cse416 = ((cse412) ? (cse411 * state.xn[9]) : (cse415 / std::sqrt(cse414)));
		const amrex::Real cse417 = llte_co_interp2d(state.T, cse416);
		const amrex::Real cse418 = 1.0 / cse417;
		const amrex::Real cse419 = nhalf_co_interp2d(state.T, cse416);
		const amrex::Real cse420 = cse407 * cse418;
		const amrex::Real cse421 = -cse419 * cse420 + 1;
		const amrex::Real cse422 = 1.0 / cse419;
		const amrex::Real cse423 = cse403 * cse422;
		const amrex::Real cse424 = alpha_co_interp2d(state.T, cse416);
		const amrex::Real cse425 = std::pow(cse423, cse424);
		const amrex::Real cse426 = cse408 * cse425;
		const amrex::Real cse427 = cse421 * cse426;
		const amrex::Real cse428 = cse403 * cse418 + cse408 + cse427;
		const amrex::Real cse429 = 1.0 / cse428;
		const amrex::Real cse430 = cse406 * cse429;
		const amrex::Real cse431 = cse430 * state.xn[9];
		const amrex::Real cse432 = std::exp((0.92600000000000005) * lnT);
		const amrex::Real cse433 = 1.08e-14 * cse432;
		const amrex::Real cse434 = 0.16 * cse377;
		const amrex::Real cse435 = std::exp((0.755 - cse434) * ln_cse376);
		const amrex::Real cse436 = 4.34e-10 * cse435;
		const amrex::Real cse437 = cse436 * state.xn[0];
		const amrex::Real cse438 = 0.042999999999999997 * cse377;
		const amrex::Real cse439 = 5.2499999999999998e-13 * std::exp((cse438 + 0.88900000000000001) * ln_cse376);
		const amrex::Real cse440 = 0.52500000000000002 * cse377;
		const amrex::Real cse441 = 2.2500000000000003e-12 * std::exp((cse440 + 1.198) * ln_cse376);
		const amrex::Real cse442 = cse439 + cse441;
		const amrex::Real cse443 = cse433 * state.xn[2] + cse437 + cse442 * state.xn[3];
		const amrex::Real cse444 = std::exp((-0.074999999999999997) * lnT);
		const amrex::Real cse445 = 5.1199999999999999e-10 * cse444;
		const amrex::Real cse446 = 0.0030000000000000001 * cse377;
		const amrex::Real cse447 = std::exp((0.41899999999999998 - cse446) * ln_cse376);
		const amrex::Real cse448 = 3.5700000000000001e-10 * cse447;
		const amrex::Real cse449 = cse448 * state.xn[0];
		const amrex::Real cse450 = 0.025000000000000001 * cse377;
		const amrex::Real cse451 = 3.7250000000000001e-11 * std::exp((cse450 + 0.26400000000000001) * ln_cse376);
		const amrex::Real cse452 = 1.0274999999999999e-10 * std::exp((cse438 + 0.29599999999999999) * ln_cse376);
		const amrex::Real cse453 = cse451 + cse452;
		const amrex::Real cse454 = cse445 * state.xn[2] + cse449 + cse453 * state.xn[3];
		const amrex::Real cse455 = std::exp(-326.60000000000002 * cse0);
		const amrex::Real cse456 = (1.0 / 5.0) * cse455;
		const amrex::Real cse457 = cse454 * cse456;
		const amrex::Real cse458 = std::exp(-425.40000000000003 * cse0);
		const amrex::Real cse459 = cse443 * cse458;
		const amrex::Real cse460 = std::exp((-0.025999999999999999) * lnT);
		const amrex::Real cse461 = 4.8599999999999998e-10 * cse460;
		const amrex::Real cse462 = 0.0060000000000000001 * cse377;
		const amrex::Real cse463 = std::exp((0.36899999999999999 - cse462) * ln_cse376);
		const amrex::Real cse464 = 3.1899999999999998e-10 * cse463;
		const amrex::Real cse465 = cse464 * state.xn[0];
		const amrex::Real cse466 = 0.041000000000000002 * cse377;
		const amrex::Real cse467 = 4.7499999999999998e-11 * std::exp((cse466 + 0.20300000000000001) * ln_cse376);
		const amrex::Real cse468 = 0.058000000000000003 * cse377;
		const amrex::Real cse469 = 1.6725000000000002e-10 * std::exp((cse468 + 0.23699999999999999) * ln_cse376);
		const amrex::Real cse470 = cse467 + cse469;
		const amrex::Real cse471 = cse461 * state.xn[2] + cse465 + cse470 * state.xn[3];
		const amrex::Real cse472 = (1.0 / 15.0) * cse471;
		const amrex::Real cse473 = cse454 + 8.9099999999999997e-5;
		const amrex::Real cse474 = cse456 * cse471;
		const amrex::Real cse475 = cse443 * cse457 + cse459 * cse472 + cse473 * cse474;
		const amrex::Real cse476 = cse443 + 1.7499999999999998e-5;
		const amrex::Real cse477 = cse471 + 1.34e-10;
		const amrex::Real cse478 = std::exp(-98.800000000000011 * cse0);
		const amrex::Real cse479 = cse443 * cse478;
		const amrex::Real cse480 = (1.0 / 3.0) * cse479;
		const amrex::Real cse481 = std::exp(-227.80000000000001 * cse0);
		const amrex::Real cse482 = cse454 * cse481;
		const amrex::Real cse483 = (3.0 / 5.0) * cse482;
		const amrex::Real cse484 = cse474 * cse476 + cse476 * cse483 + cse477 * cse483;
		const amrex::Real cse485 = cse473 * cse476 + cse473 * cse477 + cse475 + cse477 * cse480 + cse484;
		const amrex::Real cse486 = 1.0 / cse485;
		const amrex::Real cse487 = 2.3872025442750958e-19 * cse486;
		const amrex::Real cse488 = 2.8023005140019997e-18 * cse486;
		const amrex::Real cse489 = cse475 * cse487 + cse484 * cse488;
		const amrex::Real cse490 = state.T < 100.0;
		const amrex::Real cse491 = state.T < 6000.0;
		const amrex::Real cse492 = state.T < 10;
		const amrex::Real cse493 = (1.0 / (cse3 * cse3 * cse3 * cse3 * cse3 * cse3 * cse3 * cse3));
		const amrex::Real cse494 = (1.0 / (cse3 * cse3 * cse3 * cse3 * cse3 * cse3 * cse3));
		const amrex::Real cse495 = (1.0 / (cse3 * cse3 * cse3 * cse3 * cse3 * cse3));
		const amrex::Real cse496 = (1.0 / (cse3 * cse3 * cse3 * cse3 * cse3));
		const amrex::Real cse497 = (1.0 / (cse3 * cse3 * cse3 * cse3));
		const amrex::Real cse498 = (1.0 / (cse3 * cse3 * cse3));
		const amrex::Real cse499 =
		    std::exp(2.302585092994046 * (-38.719649225812759 * cse4 + 55793.874041176059 * cse493 - 124963.59998693959 * cse494 +
						  119499.73396083503 * cse495 - 63669.349536898742 * cse496 + 20652.971887880416 * cse497 -
						  4189.7100510639848 * cse498 + 512.9223506962187 * cse70 - 21.928795999999998)) *
		    state.xn[2];
		const amrex::Real cse500 = std::log(cse357);
		const amrex::Real cse501 = cse4 * cse500;
		const amrex::Real cse502 = cse496 * (cse500 * cse500 * cse500 * cse500 * cse500);
		const amrex::Real cse503 = cse497 * (cse500 * cse500 * cse500 * cse500);
		const amrex::Real cse504 = (cse500 * cse500 * cse500);
		const amrex::Real cse505 = cse498 * cse504;
		const amrex::Real cse506 = (cse500 * cse500) * cse70;
		const amrex::Real cse507 = state.xn[5] + state.xn[6];
		const amrex::Real cse508 =
		    std::exp(2.302585092994046 * (1.5714710999999999 * cse501 + 0.32168730000000001 * cse502 - 0.51002221000000003 * cse503 -
						  0.23619984999999999 * cse505 + 0.015391166 * cse506 - 22.089523)) *
			state.xn[1] +
		    std::exp(2.302585092994046 * (2.1892372 * cse501 + 0.19191374999999999 * cse502 - 0.16596184 * cse503 + 0.29036281000000003 * cse505 -
						  0.81520437999999995 * cse506 - 23.689236999999999)) *
			cse507 +
		    std::exp(2.302585092994046 * (2.0943374000000001 * cse4 * cse500 + 0.43693353000000001 * cse498 * cse504 - 0.033638326000000003 * cse502 -
						  0.14913216000000001 * cse503 - 0.77151435999999995 * cse506 - 23.962112000000001)) *
			state.xn[3];
		const amrex::Real cse509 = std::exp(2.302585092994046 * (37.383713 * cse501 + 3.8479610000000002 * cse502 + 20.159831000000001 * cse503 +
									 48.656103000000002 * cse505 + 58.145166000000003 * cse506 - 16.818342000000001)) *
					       state.xn[0] +
					   cse508;
		const amrex::Real cse510 = cse493 * (cse500 * cse500 * cse500 * cse500 * cse500 * cse500 * cse500 * cse500);
		const amrex::Real cse511 = cse494 * (cse500 * cse500 * cse500 * cse500 * cse500 * cse500 * cse500);
		const amrex::Real cse512 = cse495 * (cse500 * cse500 * cse500 * cse500 * cse500 * cse500);
		const amrex::Real cse513 =
		    std::exp(2.302585092994046 * (1.6802758 * cse501 - 8.8077017000000009 * cse502 - 4.7274035999999997 * cse503 + 4.0406627000000004 * cse505 +
						  0.93310621999999999 * cse506 - 6.3701156000000001 * cse510 + 6.4380698000000001 * cse511 +
						  8.9167182999999994 * cse512 - 22.921188999999998)) *
		    state.xn[2];
		const amrex::Real cse514 = state.T < 1000.0;
		const amrex::Real cse515 =
		    std::exp(2.302585092994046 * (3.5692468000000002 * cse4 * cse500 - 4.2519023000000002 * cse502 - 21.328264000000001 * cse503 -
						  27.850082 * cse505 - 11.33286 * cse506 - 24.311209000000002)) *
			state.xn[0] +
		    cse508;
		const amrex::Real cse516 =
		    std::exp(2.302585092994046 * (16.815729999999999 * cse501 + 983.67575999999997 * cse502 + 734.71650999999997 * cse503 + 343.1918 * cse505 +
						  96.743155000000002 * cse506 + 70.609154000000004 * cse510 + 364.14445999999998 * cse511 +
						  801.81246999999996 * cse512 - 21.928795999999998)) *
		    state.xn[2];
		const amrex::Real cse517 = std::exp(2.302585092994046 * (4.6450521 * cse501 + 1.5538288 * cse502 - 5.5108049000000001 * cse503 +
									 5.9369081000000001 * cse505 - 3.7209846 * cse506 - 24.311209000000002)) *
					       state.xn[0] +
					   cse508;
		const amrex::Real cse518 =
		    std::exp(2.302585092994046 * (2.8156982240432273 * cse4 + 5.9406293599919788 * cse496 - 5.2566362076906961 * cse497 -
						  1.3586848802337363 * cse498 + 0.049411830040524525 * cse70 - 22.089523)) *
			state.xn[1] +
		    std::exp(2.302585092994046 * (3.7525488682084647 * cse4 - 0.62120210234156448 * cse496 - 1.5370575959566981 * cse497 +
						  2.5133588394664677 * cse498 - 2.4768712409536775 * cse70 - 23.962112000000001)) *
			state.xn[3] +
		    std::exp(2.302585092994046 * (3.9225864834863131 * cse4 + 3.5440891133599637 * cse496 - 1.7105157385968939 * cse497 +
						  1.6702447513374001 * cse498 - 2.6171337683481011 * cse70 - 23.689236999999999)) *
			cse507 +
		    std::exp(2.302585092994046 * (8.3228160852326614 * cse4 + 28.69470131298657 * cse496 - 56.79810800956944 * cse497 +
						  34.150687525022562 * cse498 - 11.945856385319289 * cse70 - 24.311209000000002)) *
			state.xn[0];
		const amrex::Real cse519 =
		    ((cse381 && cse490 && cse491)
			 ? (cse492 || cse499 + cse509 < 1.0e-100)
			 : (((cse490 && cse491)
				 ? (cse492 || cse509 + cse513 < 1.0e-100)
				 : (((cse381 && cse490 && cse491 && cse514)
					 ? (cse492 || cse499 + cse515 < 1.0e-100)
					 : (((cse381 && cse491 && cse514)
						 ? (cse492 || cse515 + cse516 < 1.0e-100)
						 : (((cse491 && cse514)
							 ? (cse492 || cse513 + cse515 < 1.0e-100)
							 : (((cse381 && cse491)
								 ? (cse492 || cse516 + cse517 < 1.0e-100)
								 : (((cse491)
									 ? (cse492 || cse513 + cse517 < 1.0e-100)
									 : (((cse381) ? (cse492 ||
											 std::exp(2.302585092994046 *
												  (30.129743459482278 * cse4 + 7500.646099484944 * cse493 +
												   21588.957206746247 * cse494 +
												   26530.843406310876 * cse495 +
												   18165.631967965237 * cse496 +
												   7572.4886742758581 * cse497 +
												   1974.131269262874 * cse498 + 310.58441786958315 * cse70 -
												   21.928795999999998)) *
												     state.xn[2] +
												 cse518 <
											     1.0e-100)
										      : (cse492 ||
											 std::exp(2.302585092994046 *
												  (3.0106500755647456 * cse4 - 676.68255490510751 * cse493 +
												   381.69251128040054 * cse494 + 295.04162851880631 * cse495 -
												   162.65264843551881 * cse496 - 48.723840736518795 * cse497 +
												   23.242975457496804 * cse498 + 2.9956460707652877 * cse70 -
												   22.921188999999998)) *
												     state.xn[2] +
												 cse518 <
											     1.0e-100))))))))))))))));
		const amrex::Real cse520 = ((cse491) ? (cse0) : (0.00016666666666666666));
		const amrex::Real cse521 = std::exp(-11700.0 * cse520);
		const amrex::Real cse522 = std::exp(-5860.0 * cse520);
		const amrex::Real cse523 = std::exp(-510.0 * cse520);
		const amrex::Real cse524 = std::exp(-2197000.0 * ((cse491) ? ((1.0 / (state.T * state.T * state.T))) : (4.6296296296296294e-12)));
		const amrex::Real cse525 = cse524 * ((cse491) ? (std::exp((3.7599999999999998) * lnT)) : (160638152785302.53));
		const amrex::Real cse526 = 6.0142468035272636e-8 * ((cse491) ? (std::exp((2.1000000000000001) * lnT)) : (85924621.118090734)) + 1.0;
		const amrex::Real cse527 = 1.0 / cse526;
		const amrex::Real cse528 = 1.6e-18 * cse521 + 6.7e-19 * cse522 + 3.0e-24 * cse523 + 4.985670872372847e-33 * cse525 * cse527;
		const amrex::Real cse529 = std::log(0.001 * ((cse491) ? (state.T) : (6000.0)));
		const amrex::Real cse530 = (cse529 * cse529 * cse529 * cse529 * cse529);
		const amrex::Real cse531 = cse496 * cse530;
		const amrex::Real cse532 = (cse529 * cse529 * cse529 * cse529);
		const amrex::Real cse533 = cse497 * cse532;
		const amrex::Real cse534 = (cse529 * cse529 * cse529);
		const amrex::Real cse535 = (cse529 * cse529);
		const amrex::Real cse536 = cse535 * cse70;
		const amrex::Real cse537 =
		    std::exp(2.302585092994046 * (2.0943374000000001 * cse4 * cse529 + 0.43693353000000001 * cse498 * cse534 - 0.033638326000000003 * cse531 -
						  0.14913216000000001 * cse533 - 0.77151435999999995 * cse536 - 23.962112000000001));
		const amrex::Real cse538 = cse537 * state.xn[3];
		const amrex::Real cse539 = cse4 * cse529;
		const amrex::Real cse540 = cse498 * cse534;
		const amrex::Real cse541 = std::exp(2.302585092994046 * (0.32168730000000001 * cse531 - 0.51002221000000003 * cse533 + 0.015391166 * cse536 +
									 1.5714710999999999 * cse539 - 0.23619984999999999 * cse540 - 22.089523));
		const amrex::Real cse542 = cse541 * state.xn[1];
		const amrex::Real cse543 = std::exp(2.302585092994046 * (0.19191374999999999 * cse531 - 0.16596184 * cse533 - 0.81520437999999995 * cse536 +
									 2.1892372 * cse539 + 0.29036281000000003 * cse540 - 23.689236999999999));
		const amrex::Real cse544 = cse507 * cse543;
		const amrex::Real cse545 =
		    std::exp(2.302585092994046 * (3.8479610000000002 * cse531 + 20.159831000000001 * cse533 + 58.145166000000003 * cse536 + 37.383713 * cse539 +
						  48.656103000000002 * cse540 - 16.818342000000001));
		const amrex::Real cse546 = ((cse491) ? (cse490) : (false));
		const amrex::Real cse547 =
		    std::exp(2.302585092994046 * (3.5692468000000002 * cse4 * cse529 - 4.2519023000000002 * cse531 - 21.328264000000001 * cse533 -
						  11.33286 * cse536 - 27.850082 * cse540 - 24.311209000000002));
		const amrex::Real cse548 = ((cse491) ? (cse514) : (false));
		const amrex::Real cse549 = std::exp(2.302585092994046 * (1.5538288 * cse531 - 5.5108049000000001 * cse533 - 3.7209846 * cse536 +
									 4.6450521 * cse539 + 5.9369081000000001 * cse540 - 24.311209000000002));
		const amrex::Real cse550 = ((cse546) ? (cse545) : ((cse548) ? (cse547) : (cse549)));
		const amrex::Real cse551 = cse493 * (cse529 * cse529 * cse529 * cse529 * cse529 * cse529 * cse529 * cse529);
		const amrex::Real cse552 = (cse529 * cse529 * cse529 * cse529 * cse529 * cse529 * cse529);
		const amrex::Real cse553 = cse494 * cse552;
		const amrex::Real cse554 = (cse529 * cse529 * cse529 * cse529 * cse529 * cse529);
		const amrex::Real cse555 = cse495 * cse554;
		const amrex::Real cse556 =
		    std::exp(2.302585092994046 * (16.815729999999999 * ((cse546) ? (-2.3025850929940455 * cse4) : (cse539)) +
						  70.609154000000004 * ((cse546) ? (790.17904733961348 * cse493) : (cse551)) +
						  364.14445999999998 * ((cse546) ? (-343.17039997516258 * cse494) : (cse553)) +
						  801.81246999999996 * ((cse546) ? (149.03701106174495 * cse495) : (cse555)) +
						  983.67575999999997 * ((cse546) ? (-64.725951503469744 * cse496) : (cse531)) +
						  734.71650999999997 * ((cse546) ? (28.110123573894395 * cse497) : (cse533)) +
						  343.1918 * ((cse546) ? (-12.208071553760854 * cse498) : (cse540)) +
						  96.743155000000002 * ((cse546) ? (5.3018981104783967 * cse70) : (cse536)) - 21.928795999999998));
		const amrex::Real cse557 =
		    std::exp(2.302585092994046 * (-8.8077017000000009 * cse531 - 4.7274035999999997 * cse533 + 0.93310621999999999 * cse536 +
						  1.6802758 * cse539 + 4.0406627000000004 * cse540 - 6.3701156000000001 * cse551 + 6.4380698000000001 * cse553 +
						  8.9167182999999994 * cse555 - 22.921188999999998));
		const amrex::Real cse558 = ((cse381) ? (cse556) : (cse557));
		const amrex::Real cse559 = cse538 + cse542 + cse544 + cse550 * state.xn[0] + cse558 * state.xn[2];
		const amrex::Real cse560 = 1.0 / cse559;
		const amrex::Real cse561 = cse528 * cse560 + 1;
		const amrex::Real cse562 = 1.0 / cse561;
		const amrex::Real cse563 = ((cse519) ? (0.0) : (cse528 * cse562));
		const amrex::Real cse564 = 0.045999999999999999 * cse377;
		const amrex::Real cse565 = std::exp((cse564 + 0.22800000000000001) * ln_cse376);
		const amrex::Real cse566 = 8.9000000000000003e-11 * cse565;
		const amrex::Real cse567 = cse566 * state.xn[0];
		const amrex::Real cse568 = 0.037999999999999999 * cse377;
		const amrex::Real cse569 = 5.1749999999999996e-11 * std::exp((cse568 + 0.16900000000000001) * ln_cse376);
		const amrex::Real cse570 = 0.048000000000000001 * cse377;
		const amrex::Real cse571 = 2.15e-11 * std::exp((cse570 - 0.01) * ln_cse376);
		const amrex::Real cse572 = cse569 + cse571;
		const amrex::Real cse573 = 0.0070527699999999999 * cse2;
		const amrex::Real cse574 = cse573 - 0.11133800000000001;
		const amrex::Real cse575 = cse2 * cse574 + 0.69763799999999998;
		const amrex::Real cse576 = cse2 * cse575 - 1.3074300000000001;
		const amrex::Real cse577 = 0.093813800000000003 * cse2;
		const amrex::Real cse578 = cse577 - 3.0328300000000001;
		const amrex::Real cse579 = cse2 * cse578 + 36.180300000000003;
		const amrex::Real cse580 = cse2 * cse579 - 187.47399999999999;
		const amrex::Real cse581 = std::exp(((cse514) ? (cse2 * cse576 - 7.6973500000000001) : (cse2 * cse580 + 350.60899999999998)));
		const amrex::Real cse582 = 1.7258000000000001e-6 * cse375;
		const amrex::Real cse583 = cse581 * cse582;
		const amrex::Real cse584 = cse567 + cse572 * state.xn[3] + cse583;
		const amrex::Real cse585 = std::exp(-101.30000000000001 * cse0);
		const amrex::Real cse586 = cse584 * cse585;
		const amrex::Real cse587 = std::exp((cse564 + 0.23100000000000001) * ln_cse376);
		const amrex::Real cse588 = 2.6400000000000002e-10 * cse587;
		const amrex::Real cse589 = cse588 * state.xn[0];
		const amrex::Real cse590 = 0.031 * cse377;
		const amrex::Real cse591 = 1.11e-10 * std::exp((cse590 + 0.26300000000000001) * ln_cse376);
		const amrex::Real cse592 = 0.064000000000000001 * cse377;
		const amrex::Real cse593 = 4.3749999999999998e-11 * std::exp((cse592 + 0.071999999999999995) * ln_cse376);
		const amrex::Real cse594 = cse591 + cse593;
		const amrex::Real cse595 = 0.00235272 * cse2;
		const amrex::Real cse596 = cse595 - 0.041816600000000002;
		const amrex::Real cse597 = cse2 * cse596 + 0.35826400000000003;
		const amrex::Real cse598 = cse2 * cse597 - 0.57443;
		const amrex::Real cse599 = 0.097857299999999994 * cse2;
		const amrex::Real cse600 = cse599 - 3.1926800000000002;
		const amrex::Real cse601 = cse2 * cse600 + 38.504899999999999;
		const amrex::Real cse602 = cse2 * cse601 - 202.19300000000001;
		const amrex::Real cse603 = std::exp(((cse514) ? (cse2 * cse598 - 7.4386999999999999) : (cse2 * cse602 + 386.18599999999998)));
		const amrex::Real cse604 = cse582 * cse603;
		const amrex::Real cse605 = cse589 + cse594 * state.xn[3] + cse604;
		const amrex::Real cse606 = (25.0 / 3.0) * cse605;
		const amrex::Real cse607 = 0.057000000000000002 * cse377;
		const amrex::Real cse608 = std::exp((cse607 + 0.115) * ln_cse376);
		const amrex::Real cse609 = 1.26e-10 * cse608;
		const amrex::Real cse610 = cse609 * state.xn[0];
		const amrex::Real cse611 = 0.049000000000000002 * cse377;
		const amrex::Real cse612 = 5.3249999999999997e-11 * std::exp((cse611 - 0.0040000000000000001) * ln_cse376);
		const amrex::Real cse613 = 0.10199999999999999 * cse377;
		const amrex::Real cse614 = 1.675e-11 * std::exp((cse613 - 0.085000000000000006) * ln_cse376);
		const amrex::Real cse615 = cse612 + cse614;
		const amrex::Real cse616 = 0.00065632500000000003 * cse2;
		const amrex::Real cse617 = -cse616 - 0.015089200000000001;
		const amrex::Real cse618 = cse2 * cse617 + 0.36118400000000001;
		const amrex::Real cse619 = cse2 * cse618 - 0.77378199999999997;
		const amrex::Real cse620 = 0.10508000000000001 * cse2;
		const amrex::Real cse621 = cse620 - 3.4762;
		const amrex::Real cse622 = cse2 * cse621 + 42.594999999999999;
		const amrex::Real cse623 = cse2 * cse622 - 227.91300000000001;
		const amrex::Real cse624 = std::exp(((cse514) ? (cse2 * cse619 - 9.2514099999999999) : (cse2 * cse623 + 444.60000000000002)));
		const amrex::Real cse625 = 2.8763333333333333e-6 * cse624;
		const amrex::Real cse626 = cse375 * cse625;
		const amrex::Real cse627 = cse610 + cse615 * state.xn[3] + cse626;
		const amrex::Real cse628 = std::exp(-62.460000000000001 * cse0);
		const amrex::Real cse629 = 5 * cse628;
		const amrex::Real cse630 = cse627 * cse629;
		const amrex::Real cse631 = cse627 + 7.8800000000000004e-8;
		const amrex::Real cse632 = cse584 * cse629;
		const amrex::Real cse633 = cse586 * cse606 + cse605 * cse630 + cse631 * cse632;
		const amrex::Real cse634 = cse605 + 2.65e-7;
		const amrex::Real cse635 = cse584 + 1.81e-14;
		const amrex::Real cse636 = std::exp(-38.840000000000003 * cse0);
		const amrex::Real cse637 = cse605 * cse636;
		const amrex::Real cse638 = (5.0 / 3.0) * cse637;
		const amrex::Real cse639 = std::exp(-23.619999999999997 * cse0);
		const amrex::Real cse640 = cse627 * cse639;
		const amrex::Real cse641 = 3 * cse640;
		const amrex::Real cse642 = cse632 * cse634 + cse634 * cse641 + cse635 * cse641;
		const amrex::Real cse643 = cse631 * cse634 + cse631 * cse635 + cse633 + cse635 * cse638 + cse642;
		const amrex::Real cse644 = 1.0 / cse643;
		const amrex::Real cse645 = 1.4210469458259591e-21 * cse644;
		const amrex::Real cse646 = 2.5697412351439998e-22 * cse644;
		const amrex::Real cse647 = state.T < 1000000.0;
		const amrex::Real cse648 = ((cse647) ? (cse633 * cse645 + cse642 * cse646) : (0.0));
		const amrex::Real cse649 = Zd * cse44;
		const amrex::Real cse650 = 1.252779946046115e-21 * cse649;
		const amrex::Real cse651 = 1.8689621674132952e-20 * cse55;
		const amrex::Real cse652 = state.c_hat * state.xn[12];
		const amrex::Real cse653 = 1.9917698086432708e-17 * cse652;
		const amrex::Real cse654 = state.c_hat * state.xn[14];
		const amrex::Real cse655 = 4.2188032355736613e-18 * cse654;
		const amrex::Real cse656 = state.c_hat * state.xn[15];
		const amrex::Real cse657 = 1.7141251340095377e-18 * cse656;
		const amrex::Real cse658 = state.c_hat * state.rn[2 * 0 + 1];
		const amrex::Real cse659 = Zd * cse67;
		const amrex::Real cse660 = cse54 * state.xn[3];
		const amrex::Real cse661 = 1.8689621674132952e-20 * cse658;
		const amrex::Real cse662 = 2.6551134042892157e-17 * cse56;
		const amrex::Real cse663 = 1.9088222220940885e-17 * cse652;
		const amrex::Real cse664 = 1.168047934563505e-18 * cse654;
		const amrex::Real cse665 = 4.1730320353928637e-18 * cse656;
		const amrex::Real cse666 = state.c_hat * state.rn[2 * 1 + 1];
		const amrex::Real cse667 = cse261 * state.xn[7];
		const amrex::Real cse668 = cse288 * state.xn[9];
		const amrex::Real cse669 = 2.6551134042892157e-17 * cse666;
		const amrex::Real cse670 = 3.4516225000000006e-16 * state.xn[3];
		const amrex::Real cse671 =
		    1.0 / (cse670 + 2.0709734999999996e-16 * state.xn[0] + 2.0709734999999996e-16 * state.xn[1] + 2.0709734999999996e-16 * state.xn[10] +
			   2.0709734999999996e-16 * state.xn[11] + 2.0709734999999996e-16 * state.xn[12] + 2.0709734999999996e-16 * state.xn[13] +
			   2.0709734999999996e-16 * state.xn[14] + 2.0709734999999996e-16 * state.xn[15] + 2.0709734999999996e-16 * state.xn[16] +
			   2.0709734999999996e-16 * state.xn[17] + 2.0709734999999996e-16 * state.xn[2] + 3.4516225000000006e-16 * state.xn[4] +
			   2.0709734999999996e-16 * state.xn[5] + 2.0709734999999996e-16 * state.xn[6] + 2.0709734999999996e-16 * state.xn[7] +
			   2.0709734999999996e-16 * state.xn[8] + 2.0709734999999996e-16 * state.xn[9]);
		const amrex::Real cse672 = 6.7743911789075847e-6 * std::exp((-1.6400000000000001) * lnT) * state.xn[10] * state.xn[2];
		const amrex::Real cse673 = std::sqrt(3) * cse0 * cse374 * state.xn[15];
		const amrex::Real cse674 = 1.7697487500000004e-8 * cse673 * state.xn[6];
		const amrex::Real cse675 = (1.0 / (state.T * state.T));
		const amrex::Real cse676 = cse675 * cse83;
		const amrex::Real cse677 = cse0 * cse70;
		const amrex::Real cse678 = cse677 * cse69;
		const amrex::Real cse679 = cse0 * cse4;
		const amrex::Real cse680 = ((cse77) ? (0)
						    : (cse3 * cse73 * (1.3 * cse0 * cse4 - 3.2400000000000002 * cse678) +
						       cse3 * cse75 * (0.65400000000000003 * cse678 + 0.41599999999999998 * cse679)));
		const amrex::Real cse681 = cse67 * cse680;
		const amrex::Real cse682 = cse81 * cse82;
		const amrex::Real cse683 = cse4 / (cse80 * cse80);
		const amrex::Real cse684 = cse681 * cse683;
		const amrex::Real cse685 = cse311 * cse680 * cse683 * cse78;
		const amrex::Real cse686 = cse3 * cse84;
		const amrex::Real cse687 = ((cse85) ? (cse686 * (-cse66 * cse684 + 53300.0 * cse676 + cse681 * cse682 - cse685 * cse82 +
								 1.6677785190126752e+29 * cse62 * cse81 *
								     (-2.3058963562359999e-34 * cse60 * std::exp((-6.6881000000000004) * ln_cse61) * cse64 +
								      3.2772577039999999e-25 * cse65 * std::exp((2.1881000000000004) * lnT) +
								      2.51118476e-29 * cse65 * std::exp((3.1881000000000004) * lnT)) *
								     std::exp(cse63) / cse60))
						    : (0));
		const amrex::Real cse688 = cse81 * cse95;
		const amrex::Real cse689 = cse3 * cse96;
		const amrex::Real cse690 =
		    ((cse85) ? (cse689 * (149925037481.25937 * cse17 * cse81 *
					      (4.2414529999999999e-7 * cse11 * cse93 + 3.3349999999999998e-12 * cse17 * cse93) * std::exp(cse92) +
					  43900.0 * cse676 + cse681 * cse688 - cse684 * cse94 - cse685 * cse95))
			     : (0));
		const amrex::Real cse691 = std::exp((-1.52) * lnT);
		const amrex::Real cse692 = cse691 * state.xn[16] * state.xn[2];
		const amrex::Real cse693 = std::exp((-2.2999999999999998) * lnT);
		const amrex::Real cse694 = cse105 * state.xn[3] * state.xn[8];
		const amrex::Real cse695 = cse693 * cse694;
		const amrex::Real cse696 = 0.042000000000000003 * cse0 - 2.145922746781116e-5;
		const amrex::Real cse697 = cse133 * cse696;
		const amrex::Real cse698 = 22.5 * cse1425 * cse675 * state.xn[3] * state.xn[6];
		const amrex::Real cse699 = std::exp((-3.2999999999999998) * lnT);
		const amrex::Real cse700 = cse17 * cse30;
		const amrex::Real cse701 = (((cse33 - 10000000000.0 > 0) ? (0) : ((cse33 - 10000000000.0 == 0) ? (1.0 / 2.0) : (1))));
		const amrex::Real cse702 = cse701 * state.xn[17];
		const amrex::Real cse703 = cse59 * cse702;
		const amrex::Real cse704 = cse700 * cse703;
		const amrex::Real cse705 = (1.0 / (cse35 * cse35));
		const amrex::Real cse706 = 1.811320754716981e-13 * cse705;
		const amrex::Real cse707 =
		    cse321 * ((cse85) ? (7.9971871901929577e-69 * cse153 * std::exp((12.536555999999999) * lnT) +
					 cse154 * (-11.478657500000001 * cse0 * cse146 + 4.6894649399999997 * cse0 * cse147 - 1.1508224 * cse0 * cse148 +
						   0.1741279885 * cse0 * cse149 - 0.015791857020000001 * cse0 * cse150 +
						   0.00078368076500000001 * cse0 * cse151 - 1.6313198799999999e-5 * cse0 * cse152))
				      : (0));
		const amrex::Real cse708 = 0.99999999999999989 * state.xn[2];
		const amrex::Real cse709 =
		    cse708 * state.xn[1] *
		    (-7.3220359950196314e-6 * cse0 * cse27 + 0.0005114031087522854 * std::exp((2.907) * ln_cse0) * std::exp((-3.242) * ln_cse26));
		const amrex::Real cse710 = cse173 * cse174;
		const amrex::Real cse711 = -cse169;
		const amrex::Real cse712 = 0.5 * cse0;
		const amrex::Real cse713 = cse0 * std::log(cse168);
		const amrex::Real cse714 = 1.2250000000000001e-11 * cse164 * state.xn[1] / (cse176 * cse176);
		const amrex::Real cse715 =
		    cse714 * (-4.2248145679009874e-20 * cse0 * cse710 -
			      cse175 * (8.0679819999999989 * cse170 * std::exp((-0.98414000000000001) * lnT) + cse172 * (cse711 * cse712 - 1.102e-5 * cse713)));
		const amrex::Real cse716 =
		    state.xn[0] * state.xn[17] * (-3.435e-10 * std::exp((-1.458) * lnT) + 2.0209499999999999e-11 * std::exp((-0.59499999999999997) * lnT)) -
		    state.xn[1] * state.xn[11] *
			(cse28 * (2.6759999999999999e-12 * std::exp((-0.99331000000000003) * lnT) +
				  5.6870000000000003e-12 * std::exp((-0.48299999999999998) * lnT)) +
			 227 * cse29 * cse675);
		const amrex::Real cse717 = cse707 - cse709 - cse715 + cse716;
		const amrex::Real cse718 = 7.3171760954257481e-12 * std::exp((-0.73999999999999999) * lnT) * state.xn[0] * state.xn[14];
		const amrex::Real cse719 = cse121 * (-0.33900000000000002 * cse0 - 0.11 * cse675);
		const amrex::Real cse720 = cse718 - cse719;
		const amrex::Real cse721 = 9.0566037735849048e-14 * cse704;
		const amrex::Real cse722 = cse36 * cse721;
		const amrex::Real cse723 = cse705 * cse721;
		const amrex::Real cse724 = cse34 * cse723;
		const amrex::Real cse725 = -cse722 + cse724;
		const amrex::Real cse726 = 3.7810000000000002e-10 * std::exp((-1.1899999999999999) * lnT) * state.xn[11];
		const amrex::Real cse727 = cse126 * cse726;
		const amrex::Real cse728 = cse30 * cse701 * cse705 * state.xn[11] * state.xn[16];
		const amrex::Real cse729 = 1.1264150943396225e-13 * cse728 * std::exp((-0.68999999999999995) * lnT) * state.xn[3];
		const amrex::Real cse730 = cse727 + cse729;
		const amrex::Real cse731 = cse671 * (9.8669999999999991e-12 * cse105 * cse699 * state.xn[3] * state.xn[8] + 2 * cse59 * cse687 - cse672 -
						     cse674 + 2 * cse690 * state.xn[0] * state.xn[3] - 1.5566492609718528e-6 * cse692 -
						     5.5769999999999994e-13 * cse695 + cse697 + cse698 - cse704 * cse706 - cse717 - cse720 - cse725 - cse730);
		const amrex::Real cse732 = cse731 * state.T;
		const amrex::Real cse733 = 2.0709734999999996e-16 * cse732;
		const amrex::Real cse734 = cse78 * cse81;
		const amrex::Real cse735 = cse734 * cse82;
		const amrex::Real cse736 = 1.0 / cse311;
		const amrex::Real cse737 = cse736 * state.xn[3];
		const amrex::Real cse738 = cse72 * cse737;
		const amrex::Real cse739 = cse736 * state.xn[0];
		const amrex::Real cse740 = cse739 * cse74;
		const amrex::Real cse741 = cse738 + cse740;
		const amrex::Real cse742 = cse67 * ((cse77) ? (0) : (cse68 * cse74 - cse741));
		const amrex::Real cse743 = cse742 * cse81;
		const amrex::Real cse744 = cse683 * (-cse742 - cse78);
		const amrex::Real cse745 = cse744 * cse79;
		const amrex::Real cse746 = ((cse85) ? (cse686 * (cse66 * cse744 + cse735 + cse743 * cse82 + cse745 * cse82)) : (0));
		const amrex::Real cse747 = cse734 * cse95;
		const amrex::Real cse748 = ((cse85) ? (cse689 * (cse743 * cse95 + cse744 * cse94 + cse745 * cse95 + cse747)) : (0));
		const amrex::Real cse749 = 2.2999999999999998 * cse737 + 1.5 * cse739;
		const amrex::Real cse750 = -1.5 * cse68 + cse749;
		const amrex::Real cse751 = -cse750;
		const amrex::Real cse752 = crate * state.xn[0];
		const amrex::Real cse753 = cse751 * cse752;
		const amrex::Real cse754 = d2g * state.xn[1];
		const amrex::Real cse755 = cse178 * cse754;
		const amrex::Real cse756 = cse138 + cse144 + cse156 + cse753 - cse755 + cse90;
		const amrex::Real cse757 = cse162 + cse179;
		const amrex::Real cse758 = cse757 + cse88;
		const amrex::Real cse759 = -cse749;
		const amrex::Real cse760 = -cse752 * cse759 + cse755;
		const amrex::Real cse761 = -cse733;
		const amrex::Real cse762 = cse67 * ((cse77) ? (0) : (-cse741));
		const amrex::Real cse763 = cse683 * (-cse762 - cse78);
		const amrex::Real cse764 = cse763 * cse79;
		const amrex::Real cse765 = cse59 * ((cse85) ? (cse686 * (cse66 * cse763 + cse682 * cse762 + cse735 + cse764 * cse82)) : (0));
		const amrex::Real cse766 = -cse103;
		const amrex::Real cse767 = ((cse85) ? (cse689 * (cse688 * cse762 + cse747 + cse763 * cse94 + cse764 * cse95)) : (0));
		const amrex::Real cse768 = cse58 * state.xn[0];
		const amrex::Real cse769 = cse761 + 2 * cse765 + cse766 + cse767 * cse768;
		const amrex::Real cse770 = cse760 + cse769;
		const amrex::Real cse771 = ((state.xn[2]) * (state.xn[2]));
		const amrex::Real cse772 = 1.0 / cse771;
		const amrex::Real cse773 = cse16 * cse772;
		const amrex::Real cse774 = cse703 * cse773;
		const amrex::Real cse775 = 3.6226415094339619e-13 * cse705;
		const amrex::Real cse776 = 1.811320754716981e-13 * cse36;
		const amrex::Real cse777 = cse774 * cse776;
		const amrex::Real cse778 = cse706 * cse774;
		const amrex::Real cse779 = cse34 * cse778;
		const amrex::Real cse780 = -cse777 + cse779;
		const amrex::Real cse781 = 2.2528301886792451e-13 * std::exp((0.31) * lnT);
		const amrex::Real cse782 = cse124 + cse701 * cse705 * cse772 * cse781 * state.xn[11] * state.xn[16] * state.xn[3];
		const amrex::Real cse783 = cse155 * state.xn[0];
		const amrex::Real cse784 = cse171 * cse174 * cse711;
		const amrex::Real cse785 =
		    cse161 * state.xn[1] + cse714 * (8.4496291358019747e-20 * cse30 * cse710 + 3.1192498848929353e-17 * cse30 * cse784) - cse783;
		const amrex::Real cse786 = cse159 + cse785;
		const amrex::Real cse787 = 2 * cse78;
		const amrex::Real cse788 = cse682 * cse787;
		const amrex::Real cse789 = cse58 * cse736;
		const amrex::Real cse790 = 2 * state.xn[0];
		const amrex::Real cse791 = cse736 * cse790;
		const amrex::Real cse792 = cse72 * cse789 + cse74 * cse791;
		const amrex::Real cse793 = cse67 * ((cse77) ? (0) : (cse68 * cse72 - cse792));
		const amrex::Real cse794 = cse683 * (-cse787 - cse793);
		const amrex::Real cse795 = cse79 * cse794;
		const amrex::Real cse796 = cse59 * ((cse85) ? (cse686 * (cse66 * cse794 + cse682 * cse793 + cse788 + cse795 * cse82)) : (0));
		const amrex::Real cse797 = 4 * state.xn[3];
		const amrex::Real cse798 = cse688 * cse787;
		const amrex::Real cse799 = ((cse85) ? (cse689 * (cse688 * cse793 + cse794 * cse94 + cse795 * cse95 + cse798)) : (0));
		const amrex::Real cse800 = cse31 * cse702 * state.xn[3];
		const amrex::Real cse801 = cse706 * cse800;
		const amrex::Real cse802 = cse34 * cse38 - cse34 * cse801 + cse776 * cse800;
		const amrex::Real cse803 = d2g * state.xn[0];
		const amrex::Real cse804 = -3.4516225000000006e-16 * cse732 - 1.2000000000000001e-14 * cse803;
		const amrex::Real cse805 = cse177 * cse754;
		const amrex::Real cse806 = 2.4500000000000003e-11 * cse805;
		const amrex::Real cse807 = 4.5999999999999996 * cse737 + 3.0 * cse739;
		const amrex::Real cse808 = -2.2999999999999998 * cse68 + cse807;
		const amrex::Real cse809 = -cse808;
		const amrex::Real cse810 = cse752 * cse809;
		const amrex::Real cse811 = cse728 * cse781;
		const amrex::Real cse812 = cse806 - cse810 - cse811;
		const amrex::Real cse813 = cse131 * state.xn[3];
		const amrex::Real cse814 = cse137 * state.xn[0];
		const amrex::Real cse815 = -cse814;
		const amrex::Real cse816 = cse67 * ((cse77) ? (0) : (-cse792));
		const amrex::Real cse817 = cse683 * (-cse787 - cse816);
		const amrex::Real cse818 = cse79 * cse817;
		const amrex::Real cse819 = cse59 * ((cse85) ? (cse686 * (cse66 * cse817 + cse682 * cse816 + cse788 + cse818 * cse82)) : (0));
		const amrex::Real cse820 = ((cse85) ? (cse689 * (cse688 * cse816 + cse798 + cse817 * cse94 + cse818 * cse95)) : (0));
		const amrex::Real cse821 = -cse807;
		const amrex::Real cse822 = cse752 * cse821;
		const amrex::Real cse823 = cse806 - cse822;
		const amrex::Real cse824 = cse41 * state.xn[3];
		const amrex::Real cse825 = cse117 + cse824;
		const amrex::Real cse826 = cse158 * state.xn[2];
		const amrex::Real cse827 = cse760 + cse826;
		const amrex::Real cse828 = cse1443 * state.xn[1];
		const amrex::Real cse829 = cse114 + cse828;
		const amrex::Real cse830 = cse113 * state.xn[11];
		const amrex::Real cse831 = cse134 * state.xn[0];
		const amrex::Real cse832 = -cse831;
		const amrex::Real cse833 = cse1442 * cse278;
		const amrex::Real cse834 = cse119 * state.xn[7];
		const amrex::Real cse835 = 1.0 * cse1440 * state.xn[11] + cse48;
		const amrex::Real cse836 = cse833 + cse834 + cse835;
		const amrex::Real cse837 = 3 * cse78;
		const amrex::Real cse838 = cse67 * ((cse77) ? (0) : (-3 * cse738 - 3 * cse740));
		const amrex::Real cse839 = cse683 * (-cse837 - cse838);
		const amrex::Real cse840 = cse79 * cse839;
		const amrex::Real cse841 = cse59 * ((cse85) ? (cse686 * (cse66 * cse839 + cse682 * cse837 + cse682 * cse838 + cse82 * cse840)) : (0));
		const amrex::Real cse842 = ((cse85) ? (cse689 * (cse688 * cse837 + cse688 * cse838 + cse839 * cse94 + cse840 * cse95)) : (0));
		const amrex::Real cse843 = 6.8999999999999995 * cse737 + 4.5 * cse739;
		const amrex::Real cse844 = -cse843;
		const amrex::Real cse845 = -cse752 * cse844 + 3.6750000000000002e-11 * cse805;
		const amrex::Real cse846 = cse123 * state.xn[2] + cse232 * cse36;
		const amrex::Real cse847 = cse845 + cse846;
		const amrex::Real cse848 = cse1444 * state.xn[0];
		const amrex::Real cse849 = cse37 * state.xn[3];
		const amrex::Real cse850 = cse34 * cse849;
		const amrex::Real cse851 = -cse850;
		const amrex::Real cse852 = state.c_hat * cse660;
		const amrex::Real cse853 = 1.0 / cse165;
		const amrex::Real cse854 = cse710 * cse853;
		const amrex::Real cse855 = cse784 * cse853;
		const amrex::Real cse856 = cse714 * (-1.2199600771729787e-17 * cse854 - 4.5035826651516278e-15 * cse855);
		const amrex::Real cse857 = cse655 + cse657;
		const amrex::Real cse858 = cse714 * (-1.8637353980807409e-17 * cse854 - 6.8801320537276696e-15 * cse855);
		const amrex::Real cse859 = cse671 * (cse698 + cse717);
		const amrex::Real cse860 = cse859 * state.T;
		const amrex::Real cse861 = 2.0709734999999996e-16 * cse860;
		const amrex::Real cse862 = -cse861;
		const amrex::Real cse863 = cse760 + cse861;
		const amrex::Real cse864 = -cse806 - 3.4516225000000006e-16 * cse860;
		const amrex::Real cse865 = -cse863;
		const amrex::Real cse866 = crate * cse58;
		const amrex::Real cse867 = cse751 * cse866;
		const amrex::Real cse868 = d2g * state.xn[6];
		const amrex::Real cse869 = cse217 * cse868;
		const amrex::Real cse870 = -cse672;
		const amrex::Real cse871 = state.xn[16] * state.xn[7] *
					   (-2.4342628390636543e-12 * cse0 * cse220 +
					    cse221 * (1.2246100000000002 * cse222 * cse675 + 3.4846000000000001e-6 * cse223 * cse675 +
						      2.5908000000000001e-7 * cse224 * cse675 + 9.6185999999999982e-9 * cse225 * cse675) -
					    1.5 * cse226 * std::exp((-2.5) * lnT));
		const amrex::Real cse872 = ((cse234) ? (0) : (5.6603773584905654e-5 * cse700 * state.xn[3]));
		const amrex::Real cse873 = cse232 * state.xn[16];
		const amrex::Real cse874 = cse237 * cse873;
		const amrex::Real cse875 = cse872 * cse874;
		const amrex::Real cse876 = cse239 * cse726;
		const amrex::Real cse877 = cse235 * cse873 / (cse236 * cse236);
		const amrex::Real cse878 = cse872 * cse877;
		const amrex::Real cse879 = cse197 * cse198;
		const amrex::Real cse880 = -cse193;
		const amrex::Real cse881 = 2.1659999999999999e-12 * cse164 * state.xn[13] / (cse200 * cse200);
		const amrex::Real cse882 =
		    cse881 * (-3.3437535500443945e-27 * cse0 * cse879 - cse199 * (0.071531250000000005 * cse194 * std::exp((-0.99999836499999994) * lnT) +
										  cse196 * (cse712 * cse880 - 7.538e-5 * cse713))) -
		    9.0489001802816577e-11 * std::exp((-1.6200000000000001) * lnT) * state.xn[13] * state.xn[2];
		const amrex::Real cse883 = cse212 * cse213;
		const amrex::Real cse884 = -cse208;
		const amrex::Real cse885 = 5.5720000000000001e-12 * cse164 * state.xn[6] / (cse215 * cse215);
		const amrex::Real cse886 =
		    cse708 * state.xn[6] *
			(-4.9999999999999997e-12 * cse221 * cse8 +
			 9.9999999999999994e-12 * cse9 * (cse5 * (0.044330000000000001 * cse2 * cse677 + cse6 * cse679) + cse679 * cse7)) +
		    cse885 * (-7.7467064346414215e-23 * cse0 * cse883 - cse214 * (0.0019963845000000001 * cse209 * std::exp((-0.99999960970000001) * lnT) +
										  cse211 * (cse712 * cse884 - 5.4939999999999998e-7 * cse713)));
		const amrex::Real cse887 = 7913.1350000000002 * cse19 * cse675;
		const amrex::Real cse888 = cse250 * cse251;
		const amrex::Real cse889 = -cse246;
		const amrex::Real cse890 = 4.5579999999999993e-11 * cse164 * state.xn[8] / (cse253 * cse253);
		const amrex::Real cse891 =
		    cse890 * (-9.5042468699031921e-15 * cse0 * cse888 - cse252 * (20.983695000000001 * cse247 * std::exp((-0.95155000000000001) * lnT) +
										  cse249 * (cse712 * cse889 - 0.00013329999999999999 * cse713))) +
		    state.xn[2] * state.xn[8] *
			(-1.2230092676468155e-10 * cse11 * cse24 +
			 cse11 * (0.024885819999999999 * cse12 * cse675 + 7.2272340000000005e-7 * cse13 * cse675 + 7.7230820000000007e-8 * cse14 * cse675) -
			 3.0 / 2.0 * cse15 / std::exp((5.0 / 2.0) * lnT) +
			 cse25 * (-0.00035870183966449452 * cse17 * cse21 / cse18 - cse887 * std::log(cse18)) +
			 cse25 * (-6.1221939997287613 * cse17 * cse23 / cse22 + cse887 * std::log(cse22)));
		const amrex::Real cse892 = cse694 * cse699;
		const amrex::Real cse893 = -4.2899999999999994e-13 * cse105 * cse693 * state.xn[3] * state.xn[8] + 7.5899999999999998e-12 * cse892;
		const amrex::Real cse894 = cse891 + cse893;
		const amrex::Real cse895 = -cse723;
		const amrex::Real cse896 = cse722 - cse724;
		const amrex::Real cse897 = cse895 + cse896;
		const amrex::Real cse898 = cse671 * (6.7636511339243923e-7 * cse691 * state.xn[16] * state.xn[2] + cse707 - cse709 - cse715 + cse727 + cse729 -
						     cse870 - cse871 - cse875 + cse876 + cse878 - cse882 - cse886 - cse894 - cse897);
		const amrex::Real cse899 = cse898 * state.T;
		const amrex::Real cse900 = 2.0709734999999996e-16 * cse899;
		const amrex::Real cse901 = d2g * state.xn[8];
		const amrex::Real cse902 = cse255 * cse901;
		const amrex::Real cse903 = crate * cse737;
		const amrex::Real cse904 = cse903 * state.xn[7];
		const amrex::Real cse905 = 1040.0 * cse904;
		const amrex::Real cse906 = cse902 + cse905;
		const amrex::Real cse907 = d2g * state.xn[13];
		const amrex::Real cse908 = cse903 * state.xn[12];
		const amrex::Real cse909 = cse202 * cse907 + 16800.0 * cse908;
		const amrex::Real cse910 = cse869 + cse900 + cse906 + cse909;
		const amrex::Real cse911 = cse759 * cse866;
		const amrex::Real cse912 = cse910 - cse911;
		const amrex::Real cse913 = cse760 + cse912;
		const amrex::Real cse914 = cse210 * cse213 * cse884;
		const amrex::Real cse915 = cse207 + cse885 * (1.5493412869282843e-22 * cse30 * cse883 + 5.2413232028030254e-19 * cse30 * cse914);
		const amrex::Real cse916 = cse248 * cse251 * cse889;
		const amrex::Real cse917 = cse244 + cse890 * (1.9008493739806384e-14 * cse30 * cse888 + 7.2983853180054489e-12 * cse30 * cse916);
		const amrex::Real cse918 = cse195 * cse198 * cse880;
		const amrex::Real cse919 = cse186 + cse881 * (6.687507100088789e-27 * cse30 * cse879 + 1.5612509905490101e-22 * cse30 * cse918);
		const amrex::Real cse920 = cse778 + cse780;
		const amrex::Real cse921 = ((cse234) ? (0) : (-0.00011320754716981131 * cse773 * state.xn[3]));
		const amrex::Real cse922 = cse874 * cse921;
		const amrex::Real cse923 = cse877 * cse921;
		const amrex::Real cse924 = cse922 - cse923;
		const amrex::Real cse925 = cse231 + cse782 + cse924;
		const amrex::Real cse926 = -cse192;
		const amrex::Real cse927 = 2 * cse144;
		const amrex::Real cse928 = cse809 * cse866;
		const amrex::Real cse929 = -cse259;
		const amrex::Real cse930 = cse266 + cse929;
		const amrex::Real cse931 = ((cse234) ? (0) : (cse32));
		const amrex::Real cse932 = cse874 * cse931;
		const amrex::Real cse933 = cse877 * cse931;
		const amrex::Real cse934 = cse932 - cse933;
		const amrex::Real cse935 = cse802 + cse934;
		const amrex::Real cse936 = cse38 - cse801;
		const amrex::Real cse937 = cse216 * cse868;
		const amrex::Real cse938 = 1.1144e-11 * cse937;
		const amrex::Real cse939 = cse254 * cse901;
		const amrex::Real cse940 = 2080.0 * cse904 + 9.1159999999999986e-11 * cse939;
		const amrex::Real cse941 = cse201 * cse907;
		const amrex::Real cse942 = 33600.0 * cse908 + 4.3319999999999997e-12 * cse941;
		const amrex::Real cse943 = 3.4516225000000006e-16 * cse899 + cse938 + cse940 + cse942;
		const amrex::Real cse944 = cse821 * cse866;
		const amrex::Real cse945 = -cse900;
		const amrex::Real cse946 = cse206 * state.xn[2] + cse218;
		const amrex::Real cse947 = -cse228;
		const amrex::Real cse948 = cse242 + cse258 * state.xn[3] + cse262;
		const amrex::Real cse949 = cse265 * state.xn[3];
		const amrex::Real cse950 = cse1433 * state.xn[2] + cse256;
		const amrex::Real cse951 = cse949 + cse950;
		const amrex::Real cse952 = cse127 * cse239;
		const amrex::Real cse953 = cse128 + cse952;
		const amrex::Real cse954 = cse189 + cse191 * state.xn[3];
		const amrex::Real cse955 = cse185 * state.xn[2] + cse203;
		const amrex::Real cse956 = -cse913;
		const amrex::Real cse957 = 1.6715999999999999e-11 * cse937;
		const amrex::Real cse958 = cse232 * cse238;
		const amrex::Real cse959 = cse227 * state.xn[7];
		const amrex::Real cse960 = cse230 * state.xn[2];
		const amrex::Real cse961 = cse844 * cse866;
		const amrex::Real cse962 = -cse961;
		const amrex::Real cse963 = 1.3673999999999998e-10 * cse939;
		const amrex::Real cse964 = 3120.0 * cse904;
		const amrex::Real cse965 = cse963 + cse964;
		const amrex::Real cse966 = 50400.0 * cse908 + 6.4979999999999992e-12 * cse941;
		const amrex::Real cse967 = cse849 + cse850;
		const amrex::Real cse968 = cse853 * cse879;
		const amrex::Real cse969 = cse853 * cse918;
		const amrex::Real cse970 = cse881 * (-9.6554435073969902e-25 * cse968 - 2.2541390258730966e-20 * cse969);
		const amrex::Real cse971 = cse853 * cse883;
		const amrex::Real cse972 = cse853 * cse914;
		const amrex::Real cse973 = cse885 * (-2.2369437588208594e-20 * cse971 - 7.5674387079158044e-17 * cse972);
		const amrex::Real cse974 = cse853 * cse888;
		const amrex::Real cse975 = cse853 * cse916;
		const amrex::Real cse976 = cse890 * (-2.744452225897048e-12 * cse974 - 1.0537431374432729e-9 * cse975);
		const amrex::Real cse977 = cse881 * (-1.4750639947665343e-24 * cse968 - 3.4436526025099181e-20 * cse969);
		const amrex::Real cse978 = cse885 * (-3.4173833593729057e-20 * cse971 - 1.1560790928037308e-16 * cse972);
		const amrex::Real cse979 = state.c_hat * cse667;
		const amrex::Real cse980 = cse890 * (-4.1927050380195035e-12 * cse974 - 1.6098054538707863e-9 * cse975);
		const amrex::Real cse981 = -cse979 + cse980;
		const amrex::Real cse982 = cse59 * cse746;
		const amrex::Real cse983 = state.xn[0] * state.xn[3];
		const amrex::Real cse984 = cse748 * cse983;
		const amrex::Real cse985 = -cse138 + cse867;
		const amrex::Real cse986 = -cse718;
		const amrex::Real cse987 = cse59 * cse687;
		const amrex::Real cse988 = cse690 * cse983;
		const amrex::Real cse989 = -cse871;
		const amrex::Real cse990 = -cse875 + cse876 + cse878;
		const amrex::Real cse991 = cse697 + cse730 + cse989 + cse990;
		const amrex::Real cse992 = cse671 * (-2.3622303960273252e-7 * cse692 - cse698 - cse893 - cse897 - cse986 - cse987 - cse988 - cse991);
		const amrex::Real cse993 = cse992 * state.T;
		const amrex::Real cse994 = 2.0709734999999996e-16 * cse993;
		const amrex::Real cse995 = cse994 - 3.0000000000000002e-15 * d2g * state.xn[0];
		const amrex::Real cse996 = cse767 * cse983;
		const amrex::Real cse997 = cse765 + cse911 + cse995 + cse996;
		const amrex::Real cse998 = -cse997;
		const amrex::Real cse999 = -cse994;
		const amrex::Real cse1000 = -cse778 + cse924;
		const amrex::Real cse1001 = cse777 - cse779;
		const amrex::Real cse1002 = cse799 * cse983;
		const amrex::Real cse1003 = cse279 + cse42;
		const amrex::Real cse1004 = cse927 + cse928;
		const amrex::Real cse1005 = cse132 + cse811 - cse932 + cse933;
		const amrex::Real cse1006 = cse802 + cse936;
		const amrex::Real cse1007 = cse766 + 3.4516225000000006e-16 * cse993;
		const amrex::Real cse1008 = cse820 * cse983;
		const amrex::Real cse1009 = cse815 + cse944;
		const amrex::Real cse1010 = 1.0 * cse1438 * state.xn[3];
		const amrex::Real cse1011 = 9.0000000000000011e-15 * cse803;
		const amrex::Real cse1012 = cse842 * cse983;
		const amrex::Real cse1013 = cse275 * state.xn[9];
		const amrex::Real cse1014 = cse1013 + cse846 + cse958 + cse959;
		const amrex::Real cse1015 = 1.2216858566394234e-27 * cse852;
		const amrex::Real cse1016 = 3.6749360303422164e-18 * cse852;
		const amrex::Real cse1017 = cse1435 * cse671 * cse696 * state.T * state.xn[4];
		const amrex::Real cse1018 = 2.0709734999999996e-16 * cse1017 * state.xn[3];
		const amrex::Real cse1019 = cse1018 + cse911;
		const amrex::Real cse1020 = cse1017 * cse670;
		const amrex::Real cse1021 = -cse674;
		const amrex::Real cse1022 = cse1021 + cse698 + cse886;
		const amrex::Real cse1023 = cse1022 * cse671;
		const amrex::Real cse1024 = cse1023 * state.T;
		const amrex::Real cse1025 = 2.0709734999999996e-16 * cse1024;
		const amrex::Real cse1026 = -cse1025;
		const amrex::Real cse1027 = cse1026 + cse869;
		const amrex::Real cse1028 = -3.4516225000000006e-16 * cse1024 + cse938;
		const amrex::Real cse1029 = 1.0 * cse282;
		const amrex::Real cse1030 = cse1010 + cse1029 + cse825 + cse946;
		const amrex::Real cse1031 = cse1426 * cse278;
		const amrex::Real cse1032 = -cse1022 * cse671;
		const amrex::Real cse1033 = cse1032 * state.T;
		const amrex::Real cse1034 = 2.0709734999999996e-16 * cse1033;
		const amrex::Real cse1035 = cse1034 + cse869;
		const amrex::Real cse1036 = -cse1035;
		const amrex::Real cse1037 = 3.4516225000000006e-16 * cse1033 + cse938;
		const amrex::Real cse1038 = -cse1034;
		const amrex::Real cse1039 = cse671 * (-1.287e-13 * cse695 + cse720 + cse891 + 2.2769999999999997e-12 * cse892 + cse989);
		const amrex::Real cse1040 = cse1039 * state.T;
		const amrex::Real cse1041 = 2.0709734999999996e-16 * cse1040;
		const amrex::Real cse1042 = -cse1041;
		const amrex::Real cse1043 = cse903 * state.xn[9];
		const amrex::Real cse1044 = 184.0 * cse1043;
		const amrex::Real cse1045 = -cse1044;
		const amrex::Real cse1046 = cse1042 + cse1045 + cse906;
		const amrex::Real cse1047 = 368.0 * cse1043;
		const amrex::Real cse1048 = -cse1047;
		const amrex::Real cse1049 = cse1048 + cse292;
		const amrex::Real cse1050 = -3.4516225000000006e-16 * cse1040 + cse940;
		const amrex::Real cse1051 = cse289 + cse291 * state.xn[3];
		const amrex::Real cse1052 = cse111 + cse831;
		const amrex::Real cse1053 = 552.0 * cse1043;
		const amrex::Real cse1054 = state.c_hat * cse668;
		const amrex::Real cse1055 = -1.199496375e-8 * cse673 * state.xn[8];
		const amrex::Real cse1056 = cse671 * (-cse1055 - cse894);
		const amrex::Real cse1057 = cse1056 * state.T;
		const amrex::Real cse1058 = 2.0709734999999996e-16 * cse1057;
		const amrex::Real cse1059 = cse1058 + cse906;
		const amrex::Real cse1060 = -cse1059;
		const amrex::Real cse1061 = 3.4516225000000006e-16 * cse1057 + cse940;
		const amrex::Real cse1062 = -cse1058;
		const amrex::Real cse1063 = cse295 * state.xn[8];
		const amrex::Real cse1064 = cse671 * (cse719 + cse870);
		const amrex::Real cse1065 = cse1064 * state.T;
		const amrex::Real cse1066 = 2.0709734999999996e-16 * cse1065;
		const amrex::Real cse1067 = -cse1066;
		const amrex::Real cse1068 = cse1044 + cse1067;
		const amrex::Real cse1069 = 3.4516225000000006e-16 * cse1065;
		const amrex::Real cse1070 = cse141 + cse276;
		const amrex::Real cse1071 = cse1031 + cse1051;
		const amrex::Real cse1072 = cse671 * (cse1055 + cse672);
		const amrex::Real cse1073 = cse1072 * state.T;
		const amrex::Real cse1074 = 2.0709734999999996e-16 * cse1073;
		const amrex::Real cse1075 = -cse1074;
		const amrex::Real cse1076 = -3.4516225000000006e-16 * cse1073;
		const amrex::Real cse1077 = cse716 + cse895;
		const amrex::Real cse1078 = cse671 * (cse1077 + cse990);
		const amrex::Real cse1079 = cse1078 * state.T;
		const amrex::Real cse1080 = 2.0709734999999996e-16 * cse1079;
		const amrex::Real cse1081 = cse1044 + cse1080;
		const amrex::Real cse1082 = cse1047 + 3.4516225000000006e-16 * cse1079;
		const amrex::Real cse1083 = -cse1080;
		const amrex::Real cse1084 = cse1054 + cse665;
		const amrex::Real cse1085 = cse671 * cse882;
		const amrex::Real cse1086 = cse1085 * state.T;
		const amrex::Real cse1087 = 2.0709734999999996e-16 * cse1086;
		const amrex::Real cse1088 = -cse1087;
		const amrex::Real cse1089 = cse1088 + cse909;
		const amrex::Real cse1090 = -3.4516225000000006e-16 * cse1086 + cse942;
		const amrex::Real cse1091 = cse653 - cse970;
		const amrex::Real cse1092 = cse663 - cse977;
		const amrex::Real cse1093 = -cse671 * cse882;
		const amrex::Real cse1094 = cse1093 * state.T;
		const amrex::Real cse1095 = 2.0709734999999996e-16 * cse1094;
		const amrex::Real cse1096 = -cse1095 - cse909;
		const amrex::Real cse1097 = 3.4516225000000006e-16 * cse1094 + cse942;
		const amrex::Real cse1098 = -cse1095;
		const amrex::Real cse1099 = cse671 * (-3.0029999999999999e-13 * cse695 + cse871 + 5.3129999999999997e-12 * cse892 + cse986);
		const amrex::Real cse1100 = cse1099 * state.T;
		const amrex::Real cse1101 = 2.0709734999999996e-16 * cse1100;
		const amrex::Real cse1102 = -cse1101;
		const amrex::Real cse1103 = 2.3099999999999997e-13 * cse106;
		const amrex::Real cse1104 = -3.4516225000000006e-16 * cse1100;
		const amrex::Real cse1105 = cse671 * (-cse1021 - cse1055 - cse719 - cse725 - cse990);
		const amrex::Real cse1106 = cse1105 * state.T;
		const amrex::Real cse1107 = 2.0709734999999996e-16 * cse1106;
		const amrex::Real cse1108 = -cse1107;
		const amrex::Real cse1109 = -3.4516225000000006e-16 * cse1106;
		const amrex::Real cse1110 = cse671 * (6.7636511339243923e-7 * cse692 + cse991);
		const amrex::Real cse1111 = cse1110 * state.T;
		const amrex::Real cse1112 = 2.0709734999999996e-16 * cse1111;
		const amrex::Real cse1113 = -cse1112;
		const amrex::Real cse1114 = -3.4516225000000006e-16 * cse1111;
		const amrex::Real cse1115 = cse671 * (-cse1077 - cse674 - cse896);
		const amrex::Real cse1116 = cse1115 * state.T;
		const amrex::Real cse1117 = 2.0709734999999996e-16 * cse1116;
		const amrex::Real cse1118 = -cse1117;
		const amrex::Real cse1119 = 3.4516225000000006e-16 * cse1116;
		const amrex::Real cse1120 = d2g * state.xn[2];
		const amrex::Real cse1121 = cse1120 * cse348;
		const amrex::Real cse1122 = -cse308 + cse315;
		const amrex::Real cse1123 = (((cse1122 > 0) ? (0) : ((cse1122 == 0) ? (1.0 / 2.0) : (1))));
		const amrex::Real cse1124 = cse1123 * cse310 * cse313 * d2g;
		const amrex::Real cse1125 =
		    cse1124 * (cse790 + cse797 + 2 * state.xn[1] + 2 * state.xn[10] + 2 * state.xn[14] + 2 * state.xn[15] + 6 * state.xn[16] + 4 * state.xn[4]);
		const amrex::Real cse1126 = cse306 * d2g;
		const amrex::Real cse1127 = psigd_coll_interp2d_partial_0(cse67, state.T);
		const amrex::Real cse1128 = (((cse1122 < 0) ? (0) : ((cse1122 == 0) ? (1.0 / 2.0) : (1))));
		const amrex::Real cse1129 = cse1128 * (100.0 * cse1126 + cse1127 * cse307);
		const amrex::Real cse1130 = cse736 * state.xn[2];
		const amrex::Real cse1131 = (1.0 / (cse301 * cse301));
		const amrex::Real cse1132 = 0.5 * cse1131;
		const amrex::Real cse1133 = cse771 / (cse67 * cse67 * cse67);
		const amrex::Real cse1134 = -0.5 * cse1130 * cse302 + cse1132 * cse1133;
		const amrex::Real cse1135 = cse4 * cse68;
		const amrex::Real cse1136 =
		    ((cse352) ? (0) : ((cse353) ? ((3.0 / 2.0) * cse1135) : ((cse354) ? ((4.0 / 3.0) * cse1135) : ((cse355) ? ((1.0 / 3.0) * cse1135) : (0)))));
		const amrex::Real cse1137 = cse387 * state.xn[8];
		const amrex::Real cse1138 = cse1137 * cse391;
		const amrex::Real cse1139 = (1.0 / (cse385 * cse385));
		const amrex::Real cse1140 = (1.0 / (cse143 * cse143));
		const amrex::Real cse1141 = 1.5159999999999999e-9 * cse379;
		const amrex::Real cse1142 = cse1139 * cse384 * cse386;
		const amrex::Real cse1143 = 5.7927337833400002e-20 * cse388 * state.xn[8] / (cse390 * cse390);
		const amrex::Real cse1144 = (1.0 / (cse371 * cse371));
		const amrex::Real cse1145 = cse370 * cse736;
		const amrex::Real cse1146 = cse361 * cse737;
		const amrex::Real cse1147 = cse364 * cse739;
		const amrex::Real cse1148 = cse1146 + cse1147;
		const amrex::Real cse1149 = cse369 / (cse367 * cse367);
		const amrex::Real cse1150 = cse1149 * cse68;
		const amrex::Real cse1151 = cse1145 - cse1150 * (cse1148 - cse365);
		const amrex::Real cse1152 = cse305 * cse739;
		const amrex::Real cse1153 = cse30 * cse301;
		const amrex::Real cse1154 = cse1134 * cse1153;
		const amrex::Real cse1155 = cse304 * state.xn[0];
		const amrex::Real cse1156 = cse1136 * state.xn[3];
		const amrex::Real cse1157 = cse356 * cse737;
		const amrex::Real cse1158 = 2.0 * cse1157;
		const amrex::Real cse1159 = cse736 * state.ncol[ncol_H] *
					    (((cse413 - 1.0000000000000001e-30 < 0) ? (0) : ((cse413 - 1.0000000000000001e-30 == 0) ? (1.0 / 2.0) : (1)))) /
					    (cse414 * std::sqrt(cse414));
		const amrex::Real cse1160 = cse1159 * state.ncol[ncol_CO];
		const amrex::Real cse1161 = ((cse412) ? (0) : (878350727.21550155 * cse1160));
		const amrex::Real cse1162 = alpha_co_interp2d_partial_1(state.T, cse416);
		const amrex::Real cse1163 = std::log(cse423);
		const amrex::Real cse1164 = cse1162 * cse1163;
		const amrex::Real cse1165 = cse403 / (cse419 * cse419);
		const amrex::Real cse1166 = nhalf_co_interp2d_partial_1(state.T, cse416);
		const amrex::Real cse1167 = cse1161 * cse1166;
		const amrex::Real cse1168 = cse424 / cse403;
		const amrex::Real cse1169 = cse1168 * cse419;
		const amrex::Real cse1170 = (1.0 / (cse417 * cse417));
		const amrex::Real cse1171 = llte_co_interp2d_partial_1(state.T, cse416);
		const amrex::Real cse1172 = -cse1161 * cse1170 * cse1171 * cse403 + cse426 * (cse1161 * cse1170 * cse1171 * cse407 * cse419 - cse1167 * cse420);
		const amrex::Real cse1173 = cse403 * state.xn[9];
		const amrex::Real cse1174 = cse1173 * cse406 / (cse428 * cse428);
		const amrex::Real cse1175 = (1.0 / (cse559 * cse559));
		const amrex::Real cse1176 = (1.0 / (cse561 * cse561));
		const amrex::Real cse1177 = cse1175 * cse1176 * (cse528 * cse528);
		const amrex::Real cse1178 = cse455 * cse471;
		const amrex::Real cse1179 = 7.1400000000000007e-11 * cse447;
		const amrex::Real cse1180 = cse443 * cse455;
		const amrex::Real cse1181 = 8.68e-11 * cse435;
		const amrex::Real cse1182 = cse454 * cse455;
		const amrex::Real cse1183 = cse458 * cse471;
		const amrex::Real cse1184 = 6.3800000000000002e-11 * cse463;
		const amrex::Real cse1185 = cse455 * cse473;
		const amrex::Real cse1186 = cse1178 * cse1179 + cse1179 * cse1180 + cse1181 * cse1182 + 2.8933333333333332e-11 * cse1183 * cse435 +
					    cse1184 * cse1185 + 2.1266666666666666e-11 * cse459 * cse463;
		const amrex::Real cse1187 = cse455 * cse476;
		const amrex::Real cse1188 = 2.1419999999999999e-10 * cse447;
		const amrex::Real cse1189 = cse477 * cse481;
		const amrex::Real cse1190 = cse476 * cse481;
		const amrex::Real cse1191 = cse1178 * cse1181 + cse1184 * cse1187 + cse1188 * cse1189 + cse1188 * cse1190 + 2.604e-10 * cse435 * cse482 +
					    1.9139999999999998e-10 * cse463 * cse482;
		const amrex::Real cse1192 = (1.0 / (cse485 * cse485));
		const amrex::Real cse1193 = cse477 * cse478;
		const amrex::Real cse1194 = cse1192 * (-cse1186 - cse1191 - 1.4466666666666666e-10 * cse1193 * cse435 - cse436 * cse473 - cse448 * cse476 -
						       cse448 * cse477 - 1.0633333333333333e-10 * cse463 * cse479 - cse464 * cse473);
		const amrex::Real cse1195 = 2.3872025442750958e-19 * cse475;
		const amrex::Real cse1196 = 2.8023005140019997e-18 * cse484;
		const amrex::Real cse1197 = cse585 * cse605;
		const amrex::Real cse1198 = cse627 * cse628;
		const amrex::Real cse1199 = 1.32e-9 * cse587;
		const amrex::Real cse1200 = 6.3e-10 * cse608;
		const amrex::Real cse1201 = cse584 * cse628;
		const amrex::Real cse1202 = cse605 * cse628;
		const amrex::Real cse1203 = cse628 * cse631;
		const amrex::Real cse1204 = 4.4500000000000001e-10 * cse565;
		const amrex::Real cse1205 = 7.4166666666666679e-10 * cse1197 * cse565 + cse1198 * cse1199 + cse1200 * cse1201 + cse1200 * cse1202 +
					    cse1203 * cse1204 + 2.2000000000000003e-9 * cse586 * cse587;
		const amrex::Real cse1206 = cse628 * cse634;
		const amrex::Real cse1207 = 3.7800000000000004e-10 * cse608;
		const amrex::Real cse1208 = cse635 * cse639;
		const amrex::Real cse1209 = cse634 * cse639;
		const amrex::Real cse1210 = cse1199 * cse1201 + cse1204 * cse1206 + cse1207 * cse1208 + cse1207 * cse1209 +
					    2.6700000000000001e-10 * cse565 * cse640 + 7.9200000000000006e-10 * cse587 * cse640;
		const amrex::Real cse1211 = (1.0 / (cse643 * cse643));
		const amrex::Real cse1212 = cse635 * cse636;
		const amrex::Real cse1213 =
		    cse1211 * (-cse1205 - cse1210 - 4.4000000000000003e-10 * cse1212 * cse587 - 1.4833333333333334e-10 * cse565 * cse637 - cse566 * cse631 -
			       cse588 * cse631 - cse609 * cse634 - cse609 * cse635);
		const amrex::Real cse1214 = 1.4210469458259591e-21 * cse633;
		const amrex::Real cse1215 = 2.5697412351439998e-22 * cse642;
		const amrex::Real cse1216 = cse331 * std::exp((-1.8500000000000001) * lnT);
		const amrex::Real cse1217 = cse321 * cse334;
		const amrex::Real cse1218 = std::exp((-0.84999999999999998) * lnT);
		const amrex::Real cse1219 = std::exp((-0.19999999999999998) * lnT);
		const amrex::Real cse1220 = (1.0 / (cse323 * cse323));
		const amrex::Real cse1221 = cse1218 * cse326;
		const amrex::Real cse1222 = (1.0 / (cse328 * cse328));
		const amrex::Real cse1223 = cse1222 * cse330 * (2.5500935934394033e-12 * cse1219 * cse1220 * state.xn[2] - 6.6690584756579352e-10 * cse1221);
		const amrex::Real cse1224 = 1.3659996292638161e-10 * cse325;
		const amrex::Real cse1225 = (1.0 / (cse319 * cse319));
		const amrex::Real cse1226 = std::exp((0.12299999999999994) * ln_cse317);
		const amrex::Real cse1227 = cse1226 * cse166;
		const amrex::Real cse1228 = std::exp((-0.56899999999999995) * ln_cse317) * cse318;
		const amrex::Real cse1229 = cse1228 * cse166;
		const amrex::Real cse1230 = cse166 * cse17 * cse339;
		const amrex::Real cse1231 = 1.0 / cse342;
		const amrex::Real cse1232 = cse1231 * cse2 / (cse343 * cse343);
		const amrex::Real cse1233 = cse1231 * cse343;
		const amrex::Real cse1234 = cse1231 * cse346;
		const amrex::Real cse1235 = cse1224 * cse321 * cse331 / (cse333 * cse333);
		const amrex::Real cse1236 = cse388 * cse675;
		const amrex::Real cse1237 = cse0 * cse375;
		const amrex::Real cse1238 = std::exp((-0.92999999999999994) * lnT);
		const amrex::Real cse1239 = 100 * cse0;
		const amrex::Real cse1240 = -2.2649999999999999e-6 * cse1237 +
					    cse380 * (cse0 * cse378 + cse1239 * (8.7000000000000001e-5 * cse377 + 0.001281)) +
					    state.xn[3] * (0.75 * ((cse381) ? (1.1000000000000001e-13) : (2.6233044951750002e-11 * cse1238)) +
							   0.25 * ((cse381) ? (3.3000000000000001e-13) : (2.7229810044920004e-11 * cse1238)));
		const amrex::Real cse1241 = cse1137 * cse392;
		const amrex::Real cse1242 = 0.4375 * state.xn[0] / std::exp((3.0 / 4.0) * lnT) - 170.02500000000001 * state.xn[2] / std::exp((5.0 / 4.0) * lnT);
		const amrex::Real cse1243 = -cse1240;
		const amrex::Real cse1244 = cse1142 * cse392 * state.xn[8];
		const amrex::Real cse1245 = (1.0 / (cse358 * cse358 * cse358));
		const amrex::Real cse1246 = -cse3 * cse363 * (0.0040460000000000001 * cse1245 + 0.0036709999999999998 * cse359) -
					    cse3 * cse366 * (0.0047800000000000004 * cse1245 - 5.5500000000000001e-5 * cse359);
		const amrex::Real cse1247 = 2.0187430200000004e-26 * cse803;
		const amrex::Real cse1248 = 2.5634832e-11 * cse1144 * cse272;
		const amrex::Real cse1249 = 2 * cse1142;
		const amrex::Real cse1250 = l0_co_interp1d_partial_0(state.T);
		const amrex::Real cse1251 = (1.0 / (cse407 * cse407));
		const amrex::Real cse1252 = ((cse412) ? (-2.872276598720128e-13 * cse17 * cse409 * state.xn[9] *
							 (((5.7445531974402559e-18 * cse16 * cse409 - gradv < 0)
							       ? (0)
							       : ((5.7445531974402559e-18 * cse16 * cse409 - gradv == 0) ? (1.0 / 2.0) : (1)))) /
							 (cse410 * cse410))
						      : (0));
		const amrex::Real cse1253 = cse1170 * (-cse1171 * cse1252 - llte_co_interp2d_partial_0(state.T, cse416));
		const amrex::Real cse1254 = cse1166 * cse1252 + nhalf_co_interp2d_partial_0(state.T, cse416);
		const amrex::Real cse1255 = ((cse491) ? (-cse675) : (0));
		const amrex::Real cse1256 =
		    -1.8719999999999998e-14 * cse1255 * cse521 - 3.9261999999999998e-15 * cse1255 * cse522 - 1.53e-21 * cse1255 * cse523 +
		    4.985670872372847e-33 * cse524 * cse527 * ((cse491) ? (3.7599999999999998 * std::exp((2.7599999999999998) * lnT)) : (0)) -
		    1.0953518906603145e-26 * cse525 * cse527 * ((cse491) ? (-3 / (state.T * state.T * state.T * state.T)) : (0)) -
		    2.9985055107607381e-40 * cse525 * ((cse491) ? (2.1000000000000001 * std::exp((1.1000000000000001) * lnT)) : (0)) / (cse526 * cse526);
		const amrex::Real cse1257 = ((cse491) ? (1) : (0));
		const amrex::Real cse1258 = cse1257 * cse520;
		const amrex::Real cse1259 = cse1258 * cse529 * cse70;
		const amrex::Real cse1260 = cse1258 * cse496 * cse532;
		const amrex::Real cse1261 = cse1258 * cse497 * cse534;
		const amrex::Real cse1262 = cse1258 * cse4;
		const amrex::Real cse1263 = cse1258 * cse498 * cse535;
		const amrex::Real cse1264 = cse1258 * cse493 * cse552;
		const amrex::Real cse1265 = cse1258 * cse494 * cse554;
		const amrex::Real cse1266 = cse1258 * cse495 * cse530;
		const amrex::Real cse1267 = 65.320000000000007 * cse1178;
		const amrex::Real cse1268 = cse0 * cse438;
		const amrex::Real cse1269 = 0.00042999999999999999 * cse377;
		const amrex::Real cse1270 = cse449 * (-cse0 * cse446 + cse1239 * (0.0041900000000000001 - 3.0000000000000001e-5 * cse377)) -
					    3.8399999999999998e-11 * std::exp((-1.075) * lnT) * state.xn[2] +
					    state.xn[3] * (cse451 * (cse0 * cse450 + cse1239 * (0.00025000000000000001 * cse377 + 0.00264)) +
							   cse452 * (cse1239 * (cse1269 + 0.00296) + cse1268));
		const amrex::Real cse1271 = cse443 * cse456;
		const amrex::Real cse1272 = cse465 * (-cse0 * cse462 + cse1239 * (0.0036900000000000001 - 6.0000000000000002e-5 * cse377)) -
					    1.2635999999999999e-11 * std::exp((-1.026) * lnT) * state.xn[2] +
					    state.xn[3] * (cse467 * (cse0 * cse466 + cse1239 * (0.00041000000000000005 * cse377 + 0.0020300000000000001)) +
							   cse469 * (cse0 * cse468 + cse1239 * (0.00058 * cse377 + 0.0023700000000000001)));
		const amrex::Real cse1273 = (1.0 / 15.0) * cse459;
		const amrex::Real cse1274 = cse437 * (-cse0 * cse434 + cse1239 * (0.0075500000000000003 - 0.0016000000000000001 * cse377)) +
					    1.0000800000000001e-14 * std::exp((-0.073999999999999955) * lnT) * state.xn[2] +
					    state.xn[3] * (cse439 * (cse1239 * (cse1269 + 0.0088900000000000003) + cse1268) +
							   cse441 * (cse0 * cse440 + cse1239 * (0.0052500000000000003 * cse377 + 0.011979999999999999)));
		const amrex::Real cse1275 = cse458 * cse472;
		const amrex::Real cse1276 = cse1272 * cse473;
		const amrex::Real cse1277 = 65.320000000000007 * cse1182 * cse443 * cse675 + cse1267 * cse473 * cse675 + cse1270 * cse1271 + cse1270 * cse474 +
					    cse1272 * cse1273 + cse1274 * cse1275 + cse1274 * cse457 + cse1276 * cse456 +
					    28.360000000000003 * cse459 * cse471 * cse675;
		const amrex::Real cse1278 = 136.68000000000001 * cse482;
		const amrex::Real cse1279 = cse477 * cse675;
		const amrex::Real cse1280 = cse476 * cse675;
		const amrex::Real cse1281 = cse1270 * cse477;
		const amrex::Real cse1282 = (3.0 / 5.0) * cse481;
		const amrex::Real cse1283 = cse1270 * cse476;
		const amrex::Real cse1284 = cse456 * cse476;
		const amrex::Real cse1285 = cse1267 * cse1280 + cse1272 * cse1284 + cse1272 * cse483 + cse1274 * cse474 + cse1274 * cse483 + cse1278 * cse1279 +
					    cse1278 * cse1280 + cse1281 * cse1282 + cse1282 * cse1283;
		const amrex::Real cse1286 = (1.0 / 3.0) * cse1193;
		const amrex::Real cse1287 = cse1192 * (-cse1272 * cse480 - cse1274 * cse1286 - cse1274 * cse473 - cse1276 - cse1277 -
						       32.933333333333337 * cse1279 * cse479 - cse1281 - cse1283 - cse1285);
		const amrex::Real cse1288 = 312.30000000000001 * cse675;
		const amrex::Real cse1289 = cse0 * cse564;
		const amrex::Real cse1290 = 0.00046000000000000001 * cse377;
		const amrex::Real cse1291 = 8.6290000000000003e-7 * cse1237;
		const amrex::Real cse1292 = -cse1291 * cse603 + cse589 * (cse1239 * (cse1290 + 0.00231) + cse1289) +
					    cse604 * ((cse514) ? (cse0 * cse598 + cse2 * (cse0 * cse597 + cse2 * (cse0 * cse595 + cse0 * cse596)))
							       : (cse0 * cse602 + cse2 * (cse0 * cse601 + cse2 * (cse0 * cse599 + cse0 * cse600)))) +
					    state.xn[3] * (cse591 * (cse0 * cse590 + cse1239 * (0.00031 * cse377 + 0.00263)) +
							   cse593 * (cse0 * cse592 + cse1239 * (0.00064000000000000005 * cse377 + 0.00071999999999999994)));
		const amrex::Real cse1293 = (25.0 / 3.0) * cse586;
		const amrex::Real cse1294 = -1.4381666666666666e-6 * cse1237 * cse624 +
					    cse610 * (cse0 * cse607 + cse1239 * (0.00057000000000000009 * cse377 + 0.00115)) +
					    cse626 * ((cse514) ? (cse0 * cse619 + cse2 * (cse0 * cse618 + cse2 * (-cse0 * cse616 + cse0 * cse617)))
							       : (cse0 * cse623 + cse2 * (cse0 * cse622 + cse2 * (cse0 * cse620 + cse0 * cse621)))) +
					    state.xn[3] * (cse612 * (cse0 * cse611 + cse1239 * (0.00048999999999999998 * cse377 - 4.0000000000000003e-5)) +
							   cse614 * (cse0 * cse613 + cse1239 * (0.0010199999999999999 * cse377 - 0.00085000000000000006)));
		const amrex::Real cse1295 = -cse1291 * cse581 + cse567 * (cse1239 * (cse1290 + 0.0022800000000000003) + cse1289) +
					    cse583 * ((cse514) ? (cse0 * cse576 + cse2 * (cse0 * cse575 + cse2 * (cse0 * cse573 + cse0 * cse574)))
							       : (cse0 * cse580 + cse2 * (cse0 * cse579 + cse2 * (cse0 * cse577 + cse0 * cse578)))) +
					    state.xn[3] * (cse569 * (cse0 * cse568 + cse1239 * (0.00038000000000000002 * cse377 + 0.0016900000000000001)) +
							   cse571 * (cse0 * cse570 + cse1239 * (0.00048000000000000001 * cse377 - 0.0001)));
		const amrex::Real cse1296 = cse585 * cse606;
		const amrex::Real cse1297 = cse605 * cse629;
		const amrex::Real cse1298 = cse1295 * cse631;
		const amrex::Real cse1299 = cse1198 * cse1288 * cse605 + cse1203 * cse1288 * cse584 + cse1292 * cse1293 + cse1292 * cse630 + cse1294 * cse1297 +
					    cse1294 * cse632 + cse1295 * cse1296 + cse1298 * cse629 + 844.16666666666686 * cse586 * cse605 * cse675;
		const amrex::Real cse1300 = 70.859999999999985 * cse640 * cse675;
		const amrex::Real cse1301 = cse1294 * cse635;
		const amrex::Real cse1302 = 3 * cse639;
		const amrex::Real cse1303 = cse629 * cse634;
		const amrex::Real cse1304 = cse1294 * cse634;
		const amrex::Real cse1305 = cse1201 * cse1288 * cse634 + cse1292 * cse632 + cse1292 * cse641 + cse1295 * cse1303 + cse1295 * cse641 +
					    cse1300 * cse634 + cse1300 * cse635 + cse1301 * cse1302 + cse1302 * cse1304;
		const amrex::Real cse1306 = (5.0 / 3.0) * cse1212;
		const amrex::Real cse1307 = cse1211 * (-cse1292 * cse1306 - cse1292 * cse631 - cse1295 * cse638 - cse1298 - cse1299 - cse1301 - cse1304 -
						       cse1305 - 64.733333333333348 * cse635 * cse637 * cse675);
		const amrex::Real cse1308 =
		    cse671 *
		    (5.6705896183685846e-35 * state.c_hat * cse1225 * cse165 * cse316 * cse67 * d2g *
			 (-4.2149769358107055e-15 * cse1227 * cse700 - 6.1051722655411154e-13 * cse1229 * cse700) +
		     1.8755475162754094e-35 * state.c_hat * cse165 * cse320 * cse67 * d2g * std::exp((-0.85299999999999998) * lnT) -
		     cse1123 * (1.6000000000000001e-32 * cse17 * cse310 * cse312 + cse314) - cse1128 * cse307 * psigd_coll_interp2d_partial_1(cse67, state.T) -
		     cse1143 * (-182.41999999999999 * cse1236 - cse1240 * cse389 - cse1243 * cse1249) - cse1144 * cse1149 * cse1246 * cse1247 -
		     cse1150 * cse1246 * cse1248 - 1000 * cse1173 * cse404 * std::exp((999) * ln_cse405) * cse429 -
		     cse1174 * (-cse1242 * cse418 + cse1250 * cse1251 * cse421 * cse425 + cse1250 * cse1251 - cse1253 * cse403 -
				cse426 * (-cse1250 * cse418 * cse419 - cse1253 * cse407 * cse419 - cse1254 * cse420) -
				cse427 * (cse1163 * (cse1162 * cse1252 + alpha_co_interp2d_partial_0(state.T, cse416)) +
					  cse1169 * (-cse1165 * cse1254 + cse1242 * cse422))) -
		     1.6173435610483583e-5 * cse1216 * cse1217 * cse324 - cse1217 * cse1223 * cse1224 -
		     2.0489994438957239e-11 * cse1218 * cse321 * cse324 * cse335 +
		     7.8348995947646122e-14 * cse1219 * cse1220 * cse329 * cse330 * cse334 * state.xn[0] * state.xn[2] -
		     cse1235 * (-0.0015792330470357991 * cse1216 * cse326 + 7.6502807803182096e-12 * cse1219 * cse1220 * cse329 * cse330 * state.xn[2] -
				2.0007175426973804e-9 * cse1221 * cse331 - 1.3338116951315871e-8 * cse1223 * cse327) -
		     5.2835524837844135e-18 * cse1236 * cse391 * state.xn[8] - cse1240 * cse1241 - cse1242 * cse431 - cse1243 * cse1244 -
		     cse350 * (cse0 * cse344 - 6.3340486037177084e-11 * cse1230 * cse1232) -
		     cse350 * (-1.4428815283938862e-12 * cse1230 * cse1233 + 2.835294809184292e-11 * cse1230 * cse1234) - 2.1789607199999999e-11 * cse707 -
		     7.1777529600000012e-12 * cse987 - 7.1777529600000012e-12 * cse988 -
		     state.xn[11] * (cse1195 * cse1287 + cse1196 * cse1287 + cse1277 * cse487 + cse1285 * cse488) -
		     state.xn[3] *
			 ((cse519)
			      ? (0)
			      : (cse1176 * cse528 *
				     (-cse1175 * cse528 *
					  (-cse538 * (4.8223900769399082 * cse1257 * cse4 * cse520 + 3.0182298984217999 * cse1257 * cse498 * cse520 * cse535 -
						      3.5529549287336835 * cse1259 - 0.3872755400043702 * cse1260 - 1.3735579540080118 * cse1261) -
					   cse542 * (0.0708789387907936 * cse1259 + 3.7035619079275177 * cse1260 - 4.6974781513675152 * cse1261 +
						     3.6184459289309552 * cse1262 - 1.6316107607322889 * cse1263) -
					   cse544 * (-3.7541549062629067 * cse1259 + 2.2094886994529301 * cse1260 - 1.528565035159452 * cse1261 +
						     5.0409049417480247 * cse1262 + 2.0057552335975877 * cse1263) -
					   state.xn[0] * ((cse546) ? (cse3 * cse545 *
								      (116.29033200000001 * cse1259 + 19.239805 * cse1260 + 80.639324000000002 * cse1261 +
								       37.383713 * cse1262 + 145.968309 * cse1263))
								   : ((cse548) ? (cse547 * (8.2184944748967013 * cse1257 * cse4 * cse520 -
											    52.189748993977005 * cse1259 - 48.951834264235494 * cse1260 -
											    196.44057098336626 * cse1261 - 192.38155095558542 * cse1263))
									       : (cse549 * (-17.135767342440825 * cse1259 + 17.889115159724135 * cse1260 -
											    50.756388852554174 * cse1261 + 10.695627721640689 * cse1262 +
											    41.010708268606813 * cse1263)))) -
					   state.xn[2] * ((cse381) ? (cse556 * (222.75934655221241 * ((cse546) ? (0) : (2.0 * cse1259)) +
										2264.9971413155886 * ((cse546) ? (0) : (5.0 * cse1260)) +
										1691.7472835026108 * ((cse546) ? (0) : (4.0 * cse1261)) +
										38.719649225812766 * ((cse546) ? (0) : (1.0 * cse1262)) +
										790.22832271779396 * ((cse546) ? (0) : (3.0 * cse1263)) +
										162.58358542932092 * ((cse546) ? (0) : (8.0 * cse1264)) +
										838.47360529236653 * ((cse546) ? (0) : (7.0 * cse1265)) +
										1846.2414407987355 * ((cse546) ? (0) : (6.0 * cse1266))))
								   : (cse557 * (4.297112944704045 * cse1259 - 101.40241318979159 * cse1260 -
										43.540996231705549 * cse1261 + 3.8689780091986448 * cse1262 +
										27.91190909651122 * cse1263 - 117.34186576967058 * cse1264 +
										103.76942484394611 * cse1265 + 123.18901581604325 * cse1266)))) -
				      cse1256 * cse560) +
				 cse1256 * cse562)) -
		     state.xn[7] * ((cse647) ? (cse1214 * cse1307 + cse1215 * cse1307 + cse1299 * cse645 + cse1305 * cse646) : (0)));
		const amrex::Real cse1309 = cse1308 * state.T;
		const amrex::Real cse1310 = 2.0709734999999996e-16 * cse1309;
		const amrex::Real cse1311 = 5.6705896183685846e-35 * cse166;
		const amrex::Real cse1312 = cse316 * cse320;
		const amrex::Real cse1313 = cse1312 * d2g;
		const amrex::Real cse1314 = 4.22974728e-11 * cse303 * cse67 * cse752;
		const amrex::Real cse1315 = 3.2043540000000001e-12 * crate;
		const amrex::Real cse1316 = 1.7623947000000002e-12 * crate * cse1140 * cse399 * state.xn[5];
		const amrex::Real cse1317 = -cse1148 * cse1150 + cse368 * cse369 * cse736;
		const amrex::Real cse1318 = cse1144 * cse1247 * cse67;
		const amrex::Real cse1319 = 1.7623947000000002e-12 * crate * cse395 * state.xn[5];
		const amrex::Real cse1320 = cse422 * cse424;
		const amrex::Real cse1321 = -cse1310;
		const amrex::Real cse1322 =
		    -cse1121 - cse1125 - cse1129 + cse1154 * cse1314 + cse1156 * cse1315 -
		    cse1174 * (-cse1172 - cse427 * (cse1161 * cse1162 * cse1163 - cse1167 * cse1320)) + cse1248 * cse1317 + cse1311 * cse1313 +
		    cse1316 * cse749 + cse1317 * cse1318 +
		    cse1319 * (26.399999999999999 * cse1134 * cse30 * cse301 * cse303 * state.xn[0] + 2.0 * cse1136 * cse68 * state.xn[3] - cse1152 - cse1158) +
		    cse1321 + 3.0000000000000002e-15 * cse373 * cse803 - 7.1777529600000012e-12 * cse765 - 7.1777529600000012e-12 * cse996;
		const amrex::Real cse1323 = std::exp((0.29999999999999999) * lnT);
		const amrex::Real cse1324 = (1.0 / (cse338 * cse338));
		const amrex::Real cse1325 = cse1324 * cse16;
		const amrex::Real cse1326 = cse345 * cse347;
		const amrex::Real cse1327 = cse1232 * cse1326 * state.xn[2];
		const amrex::Real cse1328 = -cse1130 * cse1132 + 0.5 * cse302 * cse68;
		const amrex::Real cse1329 = 9.0599999999999997e-6 * cse374;
		const amrex::Real cse1330 = 1.024e-10 * cse444;
		const amrex::Real cse1331 = 2.1600000000000004e-15 * cse432;
		const amrex::Real cse1332 = 9.7199999999999998e-11 * cse460;
		const amrex::Real cse1333 = cse1178 * cse1330 + cse1180 * cse1330 + cse1182 * cse1331 + 7.1999999999999997e-16 * cse1183 * cse432 +
					    cse1185 * cse1332 + 3.2399999999999999e-11 * cse459 * cse460;
		const amrex::Real cse1334 = 3.0719999999999998e-10 * cse444;
		const amrex::Real cse1335 = cse1178 * cse1331 + cse1187 * cse1332 + cse1189 * cse1334 + cse1190 * cse1334 +
					    6.4799999999999996e-15 * cse432 * cse482 + 2.9159999999999995e-10 * cse460 * cse482;
		const amrex::Real cse1336 = cse1192 * (-3.6000000000000001e-15 * cse1193 * cse432 - cse1333 - cse1335 - cse433 * cse473 - cse445 * cse476 -
						       cse445 * cse477 - 1.6199999999999998e-10 * cse460 * cse479 - cse461 * cse473);
		const amrex::Real cse1337 = cse374 * cse603;
		const amrex::Real cse1338 = cse374 * cse624;
		const amrex::Real cse1339 = 1.4381666666666666e-5 * cse1338;
		const amrex::Real cse1340 = cse374 * cse581;
		const amrex::Real cse1341 = 8.6289999999999999e-6 * cse1337;
		const amrex::Real cse1342 = 8.6289999999999999e-6 * cse1340;
		const amrex::Real cse1343 = 1.4381666666666668e-5 * cse1197 * cse1340 + cse1198 * cse1341 + cse1201 * cse1339 + cse1202 * cse1339 +
					    cse1203 * cse1342 + 1.4381666666666668e-5 * cse1337 * cse586;
		const amrex::Real cse1344 = 5.1774000000000006e-6 * cse640;
		const amrex::Real cse1345 = 8.6289999999999999e-6 * cse1338;
		const amrex::Real cse1346 =
		    cse1201 * cse1341 + cse1206 * cse1342 + cse1208 * cse1345 + cse1209 * cse1345 + cse1337 * cse1344 + cse1340 * cse1344;
		const amrex::Real cse1347 = cse374 * cse625;
		const amrex::Real cse1348 = 1.7258000000000001e-6 * cse631;
		const amrex::Real cse1349 = cse1211 * (-2.8763333333333337e-6 * cse1212 * cse1337 - cse1337 * cse1348 - cse1340 * cse1348 -
						       2.8763333333333337e-6 * cse1340 * cse637 - cse1343 - cse1346 - cse1347 * cse634 - cse1347 * cse635);
		const amrex::Real cse1350 = cse1120 * cse1326;
		const amrex::Real cse1351 = 3.7996253111508318e-29 * cse1350;
		const amrex::Real cse1352 = cse1124 * (4 * state.xn[0] + 4 * state.xn[1] + 4 * state.xn[10] + 4 * state.xn[14] + 4 * state.xn[15] +
						       12 * state.xn[16] + 8 * state.xn[3] + 8 * state.xn[4]);
		const amrex::Real cse1353 = cse1127 * cse164;
		const amrex::Real cse1354 = cse1128 * (200.0 * cse1126 + 200.0 * cse1353);
		const amrex::Real cse1355 = cse1130 * cse302;
		const amrex::Real cse1356 = cse1131 * cse1133;
		const amrex::Real cse1357 = -1.0 * cse1355 + 1.0 * cse1356;
		const amrex::Real cse1358 =
		    ((cse352) ? (0) : ((cse353) ? (3 * cse1135) : ((cse354) ? ((8.0 / 3.0) * cse1135) : ((cse355) ? ((2.0 / 3.0) * cse1135) : (0)))));
		const amrex::Real cse1359 = -cse383;
		const amrex::Real cse1360 = cse361 * cse789 + cse364 * cse791;
		const amrex::Real cse1361 = 2 * cse1145 - cse1150 * (cse1360 - cse362);
		const amrex::Real cse1362 = cse305 * cse791;
		const amrex::Real cse1363 = cse1358 * state.xn[3];
		const amrex::Real cse1364 = 4.0 * cse1157;
		const amrex::Real cse1365 = ((cse412) ? (0) : (cse1159 * cse415));
		const amrex::Real cse1366 = cse1166 * cse1365;
		const amrex::Real cse1367 = -cse1170 * cse1171 * cse1365 * cse403 + cse426 * (cse1170 * cse1171 * cse1365 * cse407 * cse419 - cse1366 * cse420);
		const amrex::Real cse1368 = cse470 * cse473;
		const amrex::Real cse1369 = cse1271 * cse453 + cse1273 * cse470 + cse1275 * cse442 + cse1368 * cse456 + cse442 * cse457 + cse453 * cse474;
		const amrex::Real cse1370 = cse453 * cse477;
		const amrex::Real cse1371 = cse453 * cse476;
		const amrex::Real cse1372 = cse1282 * cse1370 + cse1282 * cse1371 + cse1284 * cse470 + cse442 * cse474 + cse442 * cse483 + cse470 * cse483;
		const amrex::Real cse1373 = cse1192 * (-cse1286 * cse442 - cse1368 - cse1369 - cse1370 - cse1371 - cse1372 - cse442 * cse473 - cse470 * cse480);
		const amrex::Real cse1374 = cse572 * cse631;
		const amrex::Real cse1375 = cse1293 * cse594 + cse1296 * cse572 + cse1297 * cse615 + cse1374 * cse629 + cse594 * cse630 + cse615 * cse632;
		const amrex::Real cse1376 = cse615 * cse635;
		const amrex::Real cse1377 = cse615 * cse634;
		const amrex::Real cse1378 = cse1302 * cse1376 + cse1302 * cse1377 + cse1303 * cse572 + cse572 * cse641 + cse594 * cse632 + cse594 * cse641;
		const amrex::Real cse1379 = cse1211 * (-cse1306 * cse594 - cse1374 - cse1375 - cse1376 - cse1377 - cse1378 - cse572 * cse638 - cse594 * cse631);
		const amrex::Real cse1380 = 3.4516225000000006e-16 * cse1309;
		const amrex::Real cse1381 = cse1313 * cse166;
		const amrex::Real cse1382 = cse1153 * cse1314;
		const amrex::Real cse1383 = -cse1150 * cse1360 + 2 * cse368 * cse369 * cse736;
		const amrex::Real cse1384 = cse1310 + state.xn[3] * ((cse519) ? (0) : (cse1177 * cse543));
		const amrex::Real cse1385 = ((cse412) ? (cse411) : (0));
		const amrex::Real cse1386 = cse1166 * cse1385;
		const amrex::Real cse1387 = -1.5 * cse1355 + 1.5 * cse1356;
		const amrex::Real cse1388 = ((cse352) ? (0) : ((cse353) ? ((9.0 / 2.0) * cse1135) : ((cse354) ? (4 * cse1135) : ((cse355) ? (cse1135) : (0)))));
		const amrex::Real cse1389 = -cse1150 * (3 * cse1146 + 3 * cse1147) + 3 * cse368 * cse369 * cse736;
		const amrex::Real cse1390 = ((cse412) ? (0) : (2635052181.6465044 * cse1160));
		const amrex::Real cse1391 = cse1166 * cse1390;
		const amrex::Real cse1392 = state.c_hat * cse1312 * cse164;
		const amrex::Real cse1393 = state.c_hat * cse1123 * cse16 * cse312 / std::exp((5.0 / 6.0) * ln_cse309);
		const amrex::Real cse1394 = state.c_hat * cse340;
		const amrex::Real cse1395 = cse1327 * cse1394 * cse164;
		const amrex::Real cse1396 = state.c_hat * cse31;
		const amrex::Real cse1397 = cse1226 * cse1396;
		const amrex::Real cse1398 = cse1228 * cse1396;
		const amrex::Real cse1399 = cse1225 * cse1311 * cse164 * cse316;
		const amrex::Real cse1400 = cse1233 * cse1394;
		const amrex::Real cse1401 = cse1234 * cse1394;
		const amrex::Real cse1402 = state.c_hat * cse1144 * cse368 * cse54;
		const amrex::Real cse1403 = cse1402 * cse803;
		const amrex::Real cse1404 = cse1402 * cse272 * cse68;
		const amrex::Real cse1405 = -cse650;
		const amrex::Real cse1406 = 2.50555989209223e-21 * cse649;
		const amrex::Real cse1407 = state.c_hat * cse659;
		const amrex::Real cse1408 = cse653 + 1.8689621674132952e-20 * cse852 + cse857;
		const amrex::Real cse1409 = Zd * cse658;
		const amrex::Real cse1410 = 9.1016827996670106e-22 * cse1409;
		const amrex::Real cse1411 = -cse1410;
		const amrex::Real cse1412 = 1.8203365599334021e-21 * cse1409;
		const amrex::Real cse1413 = Zd * cse46;
		const amrex::Real cse1414 = 2.0584539190345922e-21 * cse1413;
		const amrex::Real cse1415 = -cse1414;
		const amrex::Real cse1416 = 4.1169078380691844e-21 * cse1413;
		const amrex::Real cse1417 = cse1084 + cse663 + cse664 + 2.6551134042892157e-17 * cse852 + cse979;
		const amrex::Real cse1418 = Zd * cse666;
		const amrex::Real cse1419 = 1.6856051574414119e-21 * cse1418;
		const amrex::Real cse1420 = -cse1419;
		const amrex::Real cse1421 = 3.3712103148828238e-21 * cse1418;
		ydot[0] = cse100 * state.xn[2] + cse101 * state.xn[3] - cse103 * cse67 + cse108 * state.xn[3] + cse116 + cse122 + cse130 + cse140 + cse181 +
			  cse40 + cse43 + cse49 + cse52 + cse57 * cse58 + cse58 * cse98 + 2 * cse87 + cse89 - cse91;
		ydot[1] = cse139 + cse145 + cse157 - cse163 - cse180 + cse182 + cse43;
		ydot[2] = -cse181 - cse184 - cse204 - cse219 - cse241 - cse264 - cse269;
		ydot[3] = cse125 + cse129 - cse140 - cse183 - cse269 - cse270 - cse271 - cse272 - cse274 + cse277 - cse281 +
			  3.0000000000000002e-15 * cse67 * d2g * state.xn[0] - cse87;
		ydot[4] = -cse133 - cse139 - cse184 + cse280;
		ydot[5] = cse284;
		ydot[6] = -cse284;
		ydot[7] = cse112 + cse264 + cse286 * state.xn[3] + cse287 + cse294;
		ydot[8] = cse243 - cse245 - cse257 + cse260 + cse263 - cse267 + cse283 - cse297;
		ydot[9] = -cse294 - cse298 - cse299;
		ydot[10] = cse297 + cse298;
		ydot[11] = cse182 + cse290 + cse293 + cse299 + cse300 + cse39;
		ydot[12] = cse204;
		ydot[13] = -cse204;
		ydot[14] = 2.3099999999999997e-13 * cse104 * cse105 * state.xn[3] * state.xn[8] - cse116 - cse287;
		ydot[15] = -cse122 - cse297 - cse300 + cse40;
		ydot[16] = cse133 - cse241 - cse277;
		ydot[17] = cse118 - cse182 - cse268;
		ydot[18] = 5.6705896183685846e-35 * state.c_hat * cse165 * cse316 * cse320 * cse67 * d2g + 1.602177e-12 * crate * cse305 * state.xn[0] +
			   3.2043540000000001e-12 * crate * cse356 * state.xn[3] + 1.7623947000000002e-12 * crate * cse395 * cse399 * state.xn[5] -
			   2.1789607199999999e-11 * cse157 - 7.1777529600000012e-12 * cse274 - cse321 * cse337 - cse350 +
			   3.0000000000000002e-15 * cse373 * cse67 * d2g * state.xn[0] - cse393 * state.xn[8] + cse394 * cse57 * state.xn[3] - cse403 * cse431 -
			   cse489 * state.xn[11] - cse563 * state.xn[3] - cse648 * state.xn[7] - 7.1777529600000012e-12 * cse87 - std::min(cse308, cse315);
		ydot[19] = -cse109 * state.xn[14] - cse187 * state.xn[12] - cse45 * state.xn[15] - cse650 * cse67 - cse651 * state.xn[3];
		ydot[20] = -cse653 * state.rn[2 * 0 + 1] - cse655 * state.rn[2 * 0 + 1] - cse657 * state.rn[2 * 0 + 1] -
			   9.1016827996670106e-22 * cse658 * cse659 - cse660 * cse661;
		ydot[21] = -cse110 * state.xn[14] - cse188 * state.xn[12] - cse263 - cse290 - 2.0584539190345922e-21 * cse46 * cse659 - cse47 * state.xn[15] -
			   cse662 * state.xn[3];
		ydot[22] = -1.6856051574414119e-21 * cse659 * cse666 - cse660 * cse669 - cse663 * state.rn[2 * 1 + 1] - cse664 * state.rn[2 * 1 + 1] -
			   cse665 * state.rn[2 * 1 + 1] - cse666 * cse667 - cse666 * cse668;
		// $$JAFF END
		// $$JAFF REPEAT idx, expr IN rhs_jacobian_entries { USE quokka }
		// structural zeros (95 of 529 entries)
		jac(19, 2) = 0.0;
		jac(20, 2) = 0.0;
		jac(21, 2) = 0.0;
		jac(22, 2) = 0.0;
		jac(19, 5) = 0.0;
		jac(20, 5) = 0.0;
		jac(21, 5) = 0.0;
		jac(22, 5) = 0.0;
		jac(19, 6) = 0.0;
		jac(20, 6) = 0.0;
		jac(21, 6) = 0.0;
		jac(22, 6) = 0.0;
		jac(19, 7) = 0.0;
		jac(20, 7) = 0.0;
		jac(19, 8) = 0.0;
		jac(20, 8) = 0.0;
		jac(21, 8) = 0.0;
		jac(22, 8) = 0.0;
		jac(19, 9) = 0.0;
		jac(20, 9) = 0.0;
		jac(19, 11) = 0.0;
		jac(20, 11) = 0.0;
		jac(21, 11) = 0.0;
		jac(22, 11) = 0.0;
		jac(19, 13) = 0.0;
		jac(20, 13) = 0.0;
		jac(21, 13) = 0.0;
		jac(22, 13) = 0.0;
		jac(19, 17) = 0.0;
		jac(20, 17) = 0.0;
		jac(21, 17) = 0.0;
		jac(22, 17) = 0.0;
		jac(19, 18) = 0.0;
		jac(20, 18) = 0.0;
		jac(21, 18) = 0.0;
		jac(22, 18) = 0.0;
		jac(4, 19) = 0.0;
		jac(9, 19) = 0.0;
		jac(10, 19) = 0.0;
		jac(16, 19) = 0.0;
		jac(17, 19) = 0.0;
		jac(20, 19) = 0.0;
		jac(21, 19) = 0.0;
		jac(22, 19) = 0.0;
		jac(0, 20) = 0.0;
		jac(1, 20) = 0.0;
		jac(2, 20) = 0.0;
		jac(3, 20) = 0.0;
		jac(4, 20) = 0.0;
		jac(5, 20) = 0.0;
		jac(6, 20) = 0.0;
		jac(7, 20) = 0.0;
		jac(8, 20) = 0.0;
		jac(9, 20) = 0.0;
		jac(10, 20) = 0.0;
		jac(11, 20) = 0.0;
		jac(12, 20) = 0.0;
		jac(13, 20) = 0.0;
		jac(14, 20) = 0.0;
		jac(15, 20) = 0.0;
		jac(16, 20) = 0.0;
		jac(17, 20) = 0.0;
		jac(18, 20) = 0.0;
		jac(19, 20) = 0.0;
		jac(21, 20) = 0.0;
		jac(22, 20) = 0.0;
		jac(4, 21) = 0.0;
		jac(10, 21) = 0.0;
		jac(16, 21) = 0.0;
		jac(17, 21) = 0.0;
		jac(19, 21) = 0.0;
		jac(20, 21) = 0.0;
		jac(22, 21) = 0.0;
		jac(0, 22) = 0.0;
		jac(1, 22) = 0.0;
		jac(2, 22) = 0.0;
		jac(3, 22) = 0.0;
		jac(4, 22) = 0.0;
		jac(5, 22) = 0.0;
		jac(6, 22) = 0.0;
		jac(7, 22) = 0.0;
		jac(8, 22) = 0.0;
		jac(9, 22) = 0.0;
		jac(10, 22) = 0.0;
		jac(11, 22) = 0.0;
		jac(12, 22) = 0.0;
		jac(13, 22) = 0.0;
		jac(14, 22) = 0.0;
		jac(15, 22) = 0.0;
		jac(16, 22) = 0.0;
		jac(17, 22) = 0.0;
		jac(18, 22) = 0.0;
		jac(19, 22) = 0.0;
		jac(20, 22) = 0.0;
		jac(21, 22) = 0.0;
		jac(0, 0) =
		    -cse102 * cse67 - cse103 - cse135 + 2 * cse59 * cse746 - cse733 + 2 * cse748 * state.xn[0] * state.xn[3] - cse756 + 2 * cse97 * state.xn[3];
		jac(0, 1) = cse758 + cse770;
		jac(0, 2) = cse100 + cse761 + cse774 * cse775 + cse780 + cse782 + cse786;
		jac(0, 3) = cse101 + cse108 + cse132 + cse42 + 2.4433717132788467e-27 * cse55 + 7.3498720606844327e-18 * cse56 + cse768 * cse799 -
			    cse775 * cse800 + 2 * cse796 + cse797 * cse86 + cse802 + cse804 + cse812 + 2 * cse98;
		jac(0, 4) = cse768 * cse820 + cse804 + cse813 + cse815 + 2 * cse819 + cse823;
		jac(0, 5) = cse761;
		jac(0, 6) = cse761 + cse825;
		jac(0, 7) = cse120 + cse761;
		jac(0, 8) = cse107 * state.xn[3] + cse761;
		jac(0, 9) = -cse141 - cse733;
		jac(0, 10) = cse769 + cse827;
		jac(0, 11) = cse128 + cse51 + cse761 + cse829;
		jac(0, 12) = cse761;
		jac(0, 13) = cse761;
		jac(0, 14) = cse111 + cse770 + cse830 + cse832;
		jac(0, 15) = cse770 + cse836;
		jac(0, 16) = cse761 + cse768 * cse842 - 1.8000000000000002e-14 * cse803 + 2 * cse841 + cse847 + cse99 * state.xn[2];
		jac(0, 17) = 3.2000000000000001e-9 * cse36 * state.xn[3] - cse733 - cse848 - cse851;
		jac(0, 18) = cse731;
		jac(0, 19) = 2.4433717132788467e-27 * cse852 + cse856 + cse857;
		jac(0, 21) = cse664 + cse665 + 7.3498720606844327e-18 * cse852 + cse858;
		jac(1, 0) = cse756 + cse862;
		jac(1, 1) = -cse758 - cse863;
		jac(1, 2) = -cse785 - cse861;
		jac(1, 3) = cse42 + cse810 + cse864;
		jac(1, 4) = cse814 + cse822 + cse864;
		jac(1, 5) = cse862;
		jac(1, 6) = cse824 + cse862;
		jac(1, 7) = cse862;
		jac(1, 8) = cse862;
		jac(1, 9) = cse862;
		jac(1, 10) = cse865;
		jac(1, 11) = -cse828 - cse861;
		jac(1, 12) = cse862;
		jac(1, 13) = cse862;
		jac(1, 14) = cse865;
		jac(1, 15) = cse865;
		jac(1, 16) = -cse845 - cse861;
		jac(1, 17) = cse848 + cse862;
		jac(1, 18) = cse859;
		jac(1, 19) = -cse856;
		jac(1, 21) = -cse858;
		jac(2, 0) = cse144 + cse156 + cse753 - cse755 + cse867 - cse910;
		jac(2, 1) = -cse757 - cse913;
		jac(2, 2) = -cse786 - cse900 - cse915 - cse917 - cse919 - cse920 - cse925;
		jac(2, 3) = -cse812 - cse926 + cse927 + cse928 - cse930 - cse935 - cse936 - cse943;
		jac(2, 4) = -cse823 - cse943 + cse944;
		jac(2, 5) = cse205 + cse945;
		jac(2, 6) = -cse900 - cse946;
		jac(2, 7) = cse945 + cse947 + cse948;
		jac(2, 8) = -cse900 - cse951;
		jac(2, 9) = cse141 + cse945;
		jac(2, 10) = -cse827 - cse912;
		jac(2, 11) = -cse900 - cse953;
		jac(2, 12) = cse945 + cse954;
		jac(2, 13) = -cse900 - cse955;
		jac(2, 14) = cse956;
		jac(2, 15) = cse956;
		jac(2, 16) = -cse847 - cse900 - cse957 - cse958 - cse959 - cse960 - cse962 - cse965 - cse966;
		jac(2, 17) = -cse900 - cse967;
		jac(2, 18) = cse898;
		jac(2, 19) = cse653 - cse856 - cse970 - cse973 - cse976;
		jac(2, 21) = cse663 - cse858 - cse977 - cse978 - cse981;
		jac(3, 0) = cse135 - cse273 + 3.0000000000000002e-15 * cse67 * d2g - cse982 - cse984 - cse985 - cse995;
		jac(3, 1) = cse998;
		jac(3, 2) = cse1000 + cse1001 + cse782 + cse999;
		jac(3, 3) = -cse1002 - cse1003 - cse1004 - cse1005 - cse1006 - cse1007 - cse266 - cse57 - cse58 * cse86 - cse796 - cse98;
		jac(3, 4) = -cse1007 - cse1008 - cse1009 - cse813 - cse819;
		jac(3, 5) = cse999;
		jac(3, 6) = -cse1010 - cse824 - cse994;
		jac(3, 7) = -cse947 - cse994;
		jac(3, 8) = -cse949 - cse994;
		jac(3, 9) = cse276 + cse999;
		jac(3, 10) = cse998;
		jac(3, 11) = cse953 + cse999;
		jac(3, 12) = cse999;
		jac(3, 13) = cse999;
		jac(3, 14) = -cse832 - cse997;
		jac(3, 15) = cse998;
		jac(3, 16) = cse1011 - cse1012 + cse1014 - cse841 + cse962 + cse999;
		jac(3, 17) = -cse967 - cse994;
		jac(3, 18) = cse992;
		jac(3, 19) = -cse1015;
		jac(3, 21) = -cse1016;
		jac(4, 0) = cse1018 + cse985;
		jac(4, 1) = cse1019;
		jac(4, 2) = cse1018;
		jac(4, 3) = cse1004 + cse1020 - cse132 + cse279;
		jac(4, 4) = cse1009 + cse1020 - cse813;
		jac(4, 5) = cse1018;
		jac(4, 6) = cse1010 + cse1018;
		jac(4, 7) = cse1018;
		jac(4, 8) = cse1018;
		jac(4, 9) = cse1018;
		jac(4, 10) = cse1019;
		jac(4, 11) = cse1018;
		jac(4, 12) = cse1018;
		jac(4, 13) = cse1018;
		jac(4, 14) = cse1019;
		jac(4, 15) = cse1019;
		jac(4, 16) = cse1018 + cse961;
		jac(4, 17) = cse1018;
		jac(4, 18) = -cse671 * cse697;
		jac(5, 0) = cse1027;
		jac(5, 1) = cse1027;
		jac(5, 2) = cse1026 + cse915;
		jac(5, 3) = cse1003 + cse1028;
		jac(5, 4) = cse1028;
		jac(5, 5) = -cse1025 - cse205;
		jac(5, 6) = cse1026 + cse1030;
		jac(5, 7) = cse1026;
		jac(5, 8) = cse1026;
		jac(5, 9) = cse1026 + cse1031;
		jac(5, 10) = cse1027;
		jac(5, 11) = cse1026;
		jac(5, 12) = cse1026;
		jac(5, 13) = cse1026;
		jac(5, 14) = cse1027;
		jac(5, 15) = cse1027 + cse833;
		jac(5, 16) = cse1026 + cse957;
		jac(5, 17) = cse1026;
		jac(5, 18) = cse1023;
		jac(5, 19) = cse973;
		jac(5, 21) = cse978;
		jac(6, 0) = cse1036;
		jac(6, 1) = cse1036;
		jac(6, 2) = -cse1034 - cse915;
		jac(6, 3) = -cse1003 - cse1037;
		jac(6, 4) = -cse1037;
		jac(6, 5) = cse1038 + cse205;
		jac(6, 6) = -cse1030 - cse1034;
		jac(6, 7) = cse1038;
		jac(6, 8) = cse1038;
		jac(6, 9) = -cse1031 - cse1034;
		jac(6, 10) = cse1036;
		jac(6, 11) = cse1038;
		jac(6, 12) = cse1038;
		jac(6, 13) = cse1038;
		jac(6, 14) = cse1036;
		jac(6, 15) = -cse1035 - cse833;
		jac(6, 16) = -cse1034 - cse957;
		jac(6, 17) = cse1038;
		jac(6, 18) = cse1032;
		jac(6, 19) = -cse973;
		jac(6, 21) = -cse978;
		jac(7, 0) = cse1046 + cse135;
		jac(7, 1) = cse1046;
		jac(7, 2) = cse1042 + cse917;
		jac(7, 3) = cse1049 + cse1050 + cse286 + cse929;
		jac(7, 4) = cse1048 + cse1050;
		jac(7, 5) = cse1042;
		jac(7, 6) = cse1042;
		jac(7, 7) = -cse1041 - cse120 - cse228 - cse948;
		jac(7, 8) = cse1042 + cse285 * state.xn[3] + cse950;
		jac(7, 9) = cse1042 + cse1051;
		jac(7, 10) = cse1046;
		jac(7, 11) = cse1042;
		jac(7, 12) = cse1042;
		jac(7, 13) = cse1042;
		jac(7, 14) = cse1046 + cse1052;
		jac(7, 15) = -cse1041 - cse1044 - cse834 + cse902 + cse905;
		jac(7, 16) = -cse1041 - cse1053 - cse959 + cse963 + cse964;
		jac(7, 17) = cse1042;
		jac(7, 18) = cse1039;
		jac(7, 19) = cse655 + cse976;
		jac(7, 21) = cse1054 + cse664 + cse981;
		jac(8, 0) = cse1060;
		jac(8, 1) = cse1060;
		jac(8, 2) = -cse1058 - cse917;
		jac(8, 3) = -cse1061 - cse930;
		jac(8, 4) = -cse1061;
		jac(8, 5) = cse1062;
		jac(8, 6) = cse1029 + cse1062;
		jac(8, 7) = cse1062 + cse948;
		jac(8, 8) = -cse1058 - cse296 - cse951;
		jac(8, 9) = cse1031 + cse1062;
		jac(8, 10) = cse1060;
		jac(8, 11) = cse1062;
		jac(8, 12) = cse1062;
		jac(8, 13) = cse1062;
		jac(8, 14) = cse1060;
		jac(8, 15) = -cse1059 - cse1063;
		jac(8, 16) = -cse1058 - cse965;
		jac(8, 17) = cse1062;
		jac(8, 18) = cse1056;
		jac(8, 19) = -cse976;
		jac(8, 21) = cse979 - cse980;
		jac(9, 0) = cse1068;
		jac(9, 1) = cse1068;
		jac(9, 2) = cse1067 + cse159;
		jac(9, 3) = -cse1049 - cse1069;
		jac(9, 4) = cse1047 - cse1069;
		jac(9, 5) = cse1067;
		jac(9, 6) = -cse1029 - cse1066;
		jac(9, 7) = cse1067 + cse120;
		jac(9, 8) = cse1067;
		jac(9, 9) = -cse1066 - cse1070 - cse1071;
		jac(9, 10) = cse1068 + cse826;
		jac(9, 11) = cse1067 + cse114;
		jac(9, 12) = cse1067;
		jac(9, 13) = cse1067;
		jac(9, 14) = cse1068 + cse830;
		jac(9, 15) = cse1068 + cse834;
		jac(9, 16) = -cse1013 + cse1053 - cse1066;
		jac(9, 17) = cse1067;
		jac(9, 18) = cse1064;
		jac(9, 21) = -cse1054;
		jac(10, 0) = cse1075;
		jac(10, 1) = cse1075;
		jac(10, 2) = -cse1074 - cse159;
		jac(10, 3) = cse1076;
		jac(10, 4) = cse1076;
		jac(10, 5) = cse1075;
		jac(10, 6) = cse1075;
		jac(10, 7) = cse1075;
		jac(10, 8) = cse1075 + cse296;
		jac(10, 9) = cse1070 + cse1075;
		jac(10, 10) = -cse1074 - cse826;
		jac(10, 11) = cse1075;
		jac(10, 12) = cse1075;
		jac(10, 13) = cse1075;
		jac(10, 14) = cse1075;
		jac(10, 15) = cse1063 + cse1075;
		jac(10, 16) = cse1013 + cse1075;
		jac(10, 17) = cse1075;
		jac(10, 18) = cse1072;
		jac(11, 0) = -cse1081 + cse90;
		jac(11, 1) = -cse1081 - cse88;
		jac(11, 2) = -cse1000 - cse1080;
		jac(11, 3) = -cse1082 + cse292 + cse38 - cse801 - cse934;
		jac(11, 4) = -cse1082;
		jac(11, 5) = cse1083;
		jac(11, 6) = cse1029 + cse1083;
		jac(11, 7) = cse1083;
		jac(11, 8) = cse1083;
		jac(11, 9) = cse1071 + cse1083;
		jac(11, 10) = -cse1081;
		jac(11, 11) = -cse1080 + cse51 - cse829 - cse952;
		jac(11, 12) = cse1083;
		jac(11, 13) = cse1083;
		jac(11, 14) = -cse1081 - cse830;
		jac(11, 15) = cse1045 + cse1083 + cse835;
		jac(11, 16) = -cse1053 - cse1080 - cse958;
		jac(11, 17) = cse1083 + cse848 + cse849;
		jac(11, 18) = cse1078;
		jac(11, 19) = cse657;
		jac(11, 21) = cse1084;
		jac(12, 0) = cse1089;
		jac(12, 1) = cse1089;
		jac(12, 2) = cse1088 + cse919;
		jac(12, 3) = cse1090 + cse926;
		jac(12, 4) = cse1090;
		jac(12, 5) = cse1088;
		jac(12, 6) = cse1088;
		jac(12, 7) = cse1088;
		jac(12, 8) = cse1088;
		jac(12, 9) = cse1088;
		jac(12, 10) = cse1089;
		jac(12, 11) = cse1088;
		jac(12, 12) = -cse1087 - cse954;
		jac(12, 13) = cse1088 + cse955;
		jac(12, 14) = cse1089;
		jac(12, 15) = cse1089;
		jac(12, 16) = cse1088 + cse966;
		jac(12, 17) = cse1088;
		jac(12, 18) = cse1085;
		jac(12, 19) = -cse1091;
		jac(12, 21) = -cse1092;
		jac(13, 0) = cse1096;
		jac(13, 1) = cse1096;
		jac(13, 2) = -cse1095 - cse919;
		jac(13, 3) = -cse1097 - cse926;
		jac(13, 4) = -cse1097;
		jac(13, 5) = cse1098;
		jac(13, 6) = cse1098;
		jac(13, 7) = cse1098;
		jac(13, 8) = cse1098;
		jac(13, 9) = cse1098;
		jac(13, 10) = cse1096;
		jac(13, 11) = cse1098;
		jac(13, 12) = cse1098 + cse954;
		jac(13, 13) = -cse1095 - cse955;
		jac(13, 14) = cse1096;
		jac(13, 15) = cse1096;
		jac(13, 16) = -cse1095 - cse966;
		jac(13, 17) = cse1098;
		jac(13, 18) = cse1093;
		jac(13, 19) = cse1091;
		jac(13, 21) = cse1092;
		jac(14, 0) = -cse1101 - cse135;
		jac(14, 1) = cse1102;
		jac(14, 2) = cse1102;
		jac(14, 3) = cse1103 * state.xn[8] + cse1104;
		jac(14, 4) = cse1104;
		jac(14, 5) = cse1102;
		jac(14, 6) = cse1102;
		jac(14, 7) = -cse1101 - cse947;
		jac(14, 8) = cse1102 + cse1103 * state.xn[3];
		jac(14, 9) = cse1102;
		jac(14, 10) = cse1102;
		jac(14, 11) = -cse1101 - cse114;
		jac(14, 12) = cse1102;
		jac(14, 13) = cse1102;
		jac(14, 14) = -cse1052 - cse1101 - cse830;
		jac(14, 15) = cse1102;
		jac(14, 16) = -cse1101 + cse959;
		jac(14, 17) = cse1102;
		jac(14, 18) = cse1099;
		jac(14, 19) = -cse655;
		jac(14, 21) = -cse664;
		jac(15, 0) = cse1108;
		jac(15, 1) = cse1108;
		jac(15, 2) = -cse1001 - cse1107 + cse922 - cse923;
		jac(15, 3) = cse1109 + cse935;
		jac(15, 4) = cse1109;
		jac(15, 5) = cse1108;
		jac(15, 6) = -cse1107 - cse117;
		jac(15, 7) = -cse1107 - cse120;
		jac(15, 8) = -cse1107 - cse296;
		jac(15, 9) = cse1108;
		jac(15, 10) = cse1108;
		jac(15, 11) = -cse1107 - cse51 + cse952;
		jac(15, 12) = cse1108;
		jac(15, 13) = cse1108;
		jac(15, 14) = cse1108;
		jac(15, 15) = -cse1063 - cse1107 - cse836;
		jac(15, 16) = cse1108 + cse958;
		jac(15, 17) = -cse1107 - cse851;
		jac(15, 18) = cse1105;
		jac(15, 19) = -cse657;
		jac(15, 21) = -cse665;
		jac(16, 0) = cse1113;
		jac(16, 1) = cse1113;
		jac(16, 2) = -cse1112 - cse925;
		jac(16, 3) = cse1005 + cse1114;
		jac(16, 4) = cse1114 + cse813;
		jac(16, 5) = cse1113;
		jac(16, 6) = cse1113;
		jac(16, 7) = -cse1112 - cse228;
		jac(16, 8) = cse1113;
		jac(16, 9) = -cse1112 - cse276;
		jac(16, 10) = cse1113;
		jac(16, 11) = -cse1112 - cse953;
		jac(16, 12) = cse1113;
		jac(16, 13) = cse1113;
		jac(16, 14) = cse1113;
		jac(16, 15) = cse1113;
		jac(16, 16) = -cse1014 - cse1112 - cse960;
		jac(16, 17) = cse1113;
		jac(16, 18) = cse1110;
		jac(17, 0) = -cse1117 - cse90;
		jac(17, 1) = cse1118 + cse88;
		jac(17, 2) = -cse1117 - cse920;
		jac(17, 3) = -cse1006 - cse1119;
		jac(17, 4) = -cse1119;
		jac(17, 5) = cse1118;
		jac(17, 6) = cse1118 + cse117;
		jac(17, 7) = cse1118;
		jac(17, 8) = cse1118;
		jac(17, 9) = cse1118;
		jac(17, 10) = cse1118;
		jac(17, 11) = cse1118 + cse828;
		jac(17, 12) = cse1118;
		jac(17, 13) = cse1118;
		jac(17, 14) = cse1118;
		jac(17, 15) = cse1118 + cse833;
		jac(17, 16) = cse1118;
		jac(17, 17) = -cse1117 - cse848 - cse967;
		jac(17, 18) = cse1115;
		jac(18, 0) =
		    5.6705896183685846e-35 * state.c_hat * cse165 * cse316 * cse320 * d2g +
		    4.22974728e-11 * crate * cse1134 * cse30 * cse301 * cse303 * cse67 * state.xn[0] + 3.2043540000000001e-12 * crate * cse1136 * state.xn[3] +
		    1.7623947000000002e-12 * crate * cse1140 * cse399 * cse750 * state.xn[5] + 1.602177e-12 * crate * cse305 +
		    1.7623947000000002e-12 * crate * cse395 * state.xn[5] * (-cse1152 + cse1154 * cse1155 + cse1156 * cse397 - cse1158 + cse396) - cse1121 -
		    cse1125 - cse1129 - 4.39089220777172e-29 * cse1138 * cse379 +
		    4.39089220777172e-29 * cse1139 * cse379 * cse384 * cse386 * cse391 * state.xn[8] - cse1143 * (cse1141 * cse1142 - cse1141 * cse387) +
		    2.5634832e-11 * cse1144 * cse1151 * cse57 * state.xn[3] + 2.0187430200000004e-26 * cse1144 * cse1151 * cse67 * d2g * state.xn[0] -
		    cse1174 * (-cse1172 - cse402 * cse418 - cse427 * (cse1161 * cse1164 + cse1169 * (-cse1165 * cse1167 + cse402 * cse422))) - cse1310 -
		    2.1789607199999999e-11 * cse156 - 7.1777529600000012e-12 * cse273 - cse327 * cse336 + 3.0000000000000002e-15 * cse373 * cse67 * d2g +
		    3.0000000000000002e-15 * cse373 * d2g * state.xn[0] - cse402 * cse431 - 7.1777529600000012e-12 * cse982 - 7.1777529600000012e-12 * cse984 -
		    state.xn[11] * (cse1186 * cse487 + cse1191 * cse488 + cse1194 * cse1195 + cse1194 * cse1196) -
		    state.xn[3] * ((cse519) ? (0) : (cse1177 * cse550)) -
		    state.xn[7] * ((cse647) ? (cse1205 * cse645 + cse1210 * cse646 + cse1213 * cse1214 + cse1213 * cse1215) : (0));
		jac(18, 1) = cse1322 - state.xn[3] * ((cse519) ? (0) : (cse1177 * cse541));
		jac(18, 2) = 5.6705896183685846e-35 * state.c_hat * cse1225 * cse165 * cse316 * cse67 * d2g *
				 (8.429953871621411e-15 * cse1227 * cse773 + 1.2210344531082231e-12 * cse1229 * cse773) +
			     4.6527220080000002e-11 * crate * cse1328 * cse30 * cse301 * cse303 * cse395 * state.xn[0] * state.xn[5] +
			     4.22974728e-11 * crate * cse1328 * cse30 * cse301 * cse303 * cse67 * state.xn[0] - 2.6241084038530201e-25 * cse1138 * cse374 +
			     2.6241084038530201e-25 * cse1139 * cse374 * cse384 * cse386 * cse391 * state.xn[8] -
			     cse1143 * (cse1142 * cse1329 - cse1329 * cse387) - cse1174 * (-cse1168 * cse401 * cse427 - cse401 * cse418) +
			     6.0732876035249664e-19 * cse1220 * cse1222 * cse1323 * cse330 * cse334 * state.xn[0] * state.xn[2] -
			     cse1235 * (5.9301787935659931e-17 * cse1220 * cse1222 * cse1323 * cse330 * state.xn[2] - cse325 * cse332) - cse1310 -
			     2.4067011396745388e-39 * cse1325 * cse1327 * cse164 * cse166 - cse337 * state.xn[0] - cse349 -
			     cse350 * (2.8857630567877724e-12 * state.c_hat * cse1231 * cse1324 * cse16 * cse165 * cse343 - cse1234 * cse1325 * cse341) -
			     cse401 * cse431 - 2.1789607199999999e-11 * cse783 -
			     state.xn[11] * (cse1195 * cse1336 + cse1196 * cse1336 + cse1333 * cse487 + cse1335 * cse488) -
			     state.xn[3] * ((cse519) ? (0) : (cse1177 * cse558)) -
			     state.xn[7] * ((cse647) ? (cse1214 * cse1349 + cse1215 * cse1349 + cse1343 * cse645 + cse1346 * cse646) : (0));
		jac(18, 3) =
		    1.1341179236737169e-34 * state.c_hat * cse165 * cse316 * cse320 * d2g +
		    1.7623947000000002e-12 * crate * cse1140 * cse399 * cse808 * state.xn[5] +
		    4.22974728e-11 * crate * cse1357 * cse30 * cse301 * cse303 * cse67 * state.xn[0] + 3.2043540000000001e-12 * crate * cse1358 * state.xn[3] +
		    3.2043540000000001e-12 * crate * cse356 +
		    1.7623947000000002e-12 * crate * cse395 * state.xn[5] * (cse1153 * cse1155 * cse1357 - cse1362 + cse1363 * cse397 - cse1364 + cse398) -
		    7.1777529600000012e-12 * cse1002 - cse1143 * (-cse1249 * cse1359 - cse383 * cse389) +
		    2.5634832e-11 * cse1144 * cse1361 * cse57 * state.xn[3] + 2.0187430200000004e-26 * cse1144 * cse1361 * cse67 * d2g * state.xn[0] -
		    cse1174 * (-cse1367 - cse418 - cse427 * (cse1164 * cse1365 + cse1169 * (-cse1165 * cse1366 + cse422))) - cse1241 * cse383 -
		    cse1244 * cse1359 - cse1351 - cse1352 - cse1354 - cse1380 + 6.0000000000000005e-15 * cse373 * d2g * state.xn[0] + cse394 * cse57 - cse431 -
		    cse563 - 7.1777529600000012e-12 * cse796 - 1.4355505920000002e-11 * cse86 * state.xn[3] - 7.1777529600000012e-12 * cse98 -
		    state.xn[11] * (cse1195 * cse1373 + cse1196 * cse1373 + cse1369 * cse487 + cse1372 * cse488) -
		    state.xn[3] * ((cse519) ? (0) : (cse1177 * cse537)) -
		    state.xn[7] * ((cse647) ? (cse1214 * cse1379 + cse1215 * cse1379 + cse1375 * cse645 + cse1378 * cse646) : (0));
		jac(18, 4) =
		    -7.1777529600000012e-12 * cse1008 + cse103 * cse373 - cse1174 * (-cse1367 - cse427 * (cse1162 * cse1163 * cse1365 - cse1320 * cse1366)) +
		    cse1248 * cse1383 + cse1315 * cse1363 + cse1316 * cse807 + cse1318 * cse1383 +
		    cse1319 * (26.399999999999999 * cse1357 * cse30 * cse301 * cse303 * state.xn[0] + 2.0 * cse1358 * cse68 * state.xn[3] - cse1362 - cse1364) -
		    cse1351 - cse1352 - cse1354 + cse1357 * cse1382 - cse1380 + 1.1341179236737169e-34 * cse1381 - 7.1777529600000012e-12 * cse819;
		jac(18, 5) = 1.7623947000000002e-12 * crate * cse395 * cse399 - cse1384;
		jac(18, 6) = -cse1384;
		jac(18, 7) = -cse1310 - cse648;
		jac(18, 8) = -cse1310 - cse393;
		jac(18, 9) = -cse1174 * (cse1170 * cse1171 * cse1385 * cse403 - cse426 * (cse1170 * cse1171 * cse1385 * cse407 * cse419 - cse1386 * cse420) -
					 cse427 * (cse1162 * cse1163 * cse1385 - cse1320 * cse1386)) -
			     cse1310 - cse403 * cse430;
		jac(18, 10) = cse1322;
		jac(18, 11) = -cse1310 - cse489;
		jac(18, 12) = cse1321;
		jac(18, 13) = cse1321;
		jac(18, 14) = cse1322;
		jac(18, 15) = cse1322;
		jac(18, 16) = cse1011 * cse373 - 7.1777529600000012e-12 * cse1012 -
			      cse1124 * (6 * state.xn[0] + 6 * state.xn[1] + 6 * state.xn[10] + 6 * state.xn[14] + 6 * state.xn[15] + 18 * state.xn[16] +
					 12 * state.xn[3] + 12 * state.xn[4]) -
			      cse1128 * (300.0 * cse1126 + 300.0 * cse1353) -
			      cse1174 * (cse1170 * cse1171 * cse1390 * cse403 - cse426 * (cse1170 * cse1171 * cse1390 * cse407 * cse419 - cse1391 * cse420) -
					 cse427 * (cse1162 * cse1163 * cse1390 - cse1320 * cse1391)) +
			      cse1248 * cse1389 + cse1315 * cse1388 * state.xn[3] + cse1316 * cse843 + cse1318 * cse1389 +
			      cse1319 * (-3 * cse1152 - 6.0 * cse1157 + 26.399999999999999 * cse1387 * cse30 * cse301 * cse303 * state.xn[0] +
					 2.0 * cse1388 * cse68 * state.xn[3]) +
			      cse1321 - 5.6994379667262474e-29 * cse1350 + 1.7011768855105754e-34 * cse1381 + cse1382 * cse1387 -
			      7.1777529600000012e-12 * cse841;
		jac(18, 17) = cse1321;
		jac(18, 18) = cse1308;
		jac(18, 19) = cse1015 * cse394 + 8.1872148910410565e-33 * cse1392 + 4.7575374194964615e-36 * cse1393 + 3.4748025752386779e-37 * cse1395 +
			      cse1399 * (-1.2171193564238968e-12 * cse1397 - 1.7629333331721322e-10 * cse1398) - 2.466269795723557e-53 * cse1403 -
			      3.1317711691727702e-38 * cse1404 - cse350 * (-4.1664736580507933e-10 * cse1400 + 8.1872148910410564e-9 * cse1401);
		jac(18, 21) = cse1016 * cse394 + 1.2507624216266398e-32 * cse1392 + 7.2680992290799608e-36 * cse1393 + 5.3084626964364086e-37 * cse1395 +
			      cse1399 * (-1.8593956234313972e-12 * cse1397 - 2.6932367041905241e-10 * cse1398) - 7.4187514601998589e-44 * cse1403 -
			      9.4206367748569615e-29 * cse1404 - cse350 * (-6.365129963657969e-10 * cse1400 + 1.2507624216266397e-8 * cse1401);
		jac(19, 0) = cse1405;
		jac(19, 1) = cse1405;
		jac(19, 3) = -cse1406 - cse651;
		jac(19, 4) = -cse1406;
		jac(19, 10) = cse1405;
		jac(19, 12) = -cse187;
		jac(19, 14) = -cse109 - cse650;
		jac(19, 15) = -cse45 - cse650;
		jac(19, 16) = -3.7583398381383454e-21 * cse649;
		jac(19, 19) = -1.252779946046115e-21 * cse1407 - cse1408;
		jac(20, 0) = cse1411;
		jac(20, 1) = cse1411;
		jac(20, 3) = -cse1412 - cse54 * cse661;
		jac(20, 4) = -cse1412;
		jac(20, 10) = cse1411;
		jac(20, 12) = -1.9917698086432708e-17 * cse658;
		jac(20, 14) = -cse1410 - 4.2188032355736613e-18 * cse658;
		jac(20, 15) = -cse1410 - 1.7141251340095377e-18 * cse658;
		jac(20, 16) = -2.7305048399001034e-21 * cse1409;
		jac(20, 20) = -9.1016827996670106e-22 * cse1407 - cse1408;
		jac(21, 0) = cse1415;
		jac(21, 1) = cse1415;
		jac(21, 3) = -cse1416 - cse662;
		jac(21, 4) = -cse1416;
		jac(21, 7) = -cse262;
		jac(21, 9) = -cse289;
		jac(21, 10) = cse1415;
		jac(21, 12) = -cse188;
		jac(21, 14) = -cse110 - cse1414;
		jac(21, 15) = -cse1414 - cse47;
		jac(21, 16) = -6.175361757103777e-21 * cse1413;
		jac(21, 21) = -2.0584539190345922e-21 * cse1407 - cse1417;
		jac(22, 0) = cse1420;
		jac(22, 1) = cse1420;
		jac(22, 3) = -cse1421 - cse54 * cse669;
		jac(22, 4) = -cse1421;
		jac(22, 7) = -cse261 * cse666;
		jac(22, 9) = -cse288 * cse666;
		jac(22, 10) = cse1420;
		jac(22, 12) = -1.9088222220940885e-17 * cse666;
		jac(22, 14) = -cse1419 - 1.168047934563505e-18 * cse666;
		jac(22, 15) = -cse1419 - 4.1730320353928637e-18 * cse666;
		jac(22, 16) = -5.0568154723242353e-21 * cse1418;
		jac(22, 22) = -1.6856051574414119e-21 * cse1407 - cse1417;
		// $$JAFF END
		// the energy row and column from E = rho e to e, as in actual_rhs and
		// actual_jac
		// $$JAFF SUB nspec
		ydot[18] /= state.rho;
		for (int i = 0; i < System::neqs; i++) {
			if (i != 18) {
				jac(18, i) /= state.rho;
				jac(i, 18) *= state.rho;
			}
		}
		// $$JAFF END
	}
};

// $$JAFF SUB label
} // namespace quokka::networks::GOW
// $$JAFF END

#endif // JAFF_NETWORK_HPP_
