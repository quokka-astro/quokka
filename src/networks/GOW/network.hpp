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
		const amrex::Real cse10 = 1.99e-9 * std::pow(state.T, -0.19) * state.xn[11] * state.xn[16];
		const amrex::Real cse11 = cse10 * cse6;
		const amrex::Real cse12 = state.c_hat * state.rn[2 * 0];
		const amrex::Real cse13 = 1.7141251340095377e-18 * cse12;
		const amrex::Real cse14 = state.c_hat * state.rn[2 * 1];
		const amrex::Real cse15 = 4.1730320353928637e-18 * cse14;
		const amrex::Real cse16 = (cse13 + cse15) * state.xn[15];
		const amrex::Real cse17 = 3.5000000000000002e-11 * state.xn[11] * state.xn[15];
		const amrex::Real cse18 = 1.0 / state.T;
		const amrex::Real cse19 =
		    7.9513696852843427e-10 * std::pow(state.T, -0.33900000000000002) * std::exp(0.11 * cse18) * state.xn[15] * state.xn[7];
		const amrex::Real cse20 = M_LN10;
		const amrex::Real cse21 = 1.0 / cse20;
		const amrex::Real cse22 = 2 * state.xn[3];
		const amrex::Real cse23 = cse22 + state.xn[0] + state.xn[10] + state.xn[14] + state.xn[15] + 3 * state.xn[16] + state.xn[1] + 2 * state.xn[4];
		const amrex::Real cse24 = std::log((1.0 / 10000.0) * state.T);
		const amrex::Real cse25 = std::pow(cse20, -2);
		const amrex::Real cse26 = std::pow(cse24, 2) * cse25;
		const amrex::Real cse27 = 1.0 / cse23;
		const amrex::Real cse28 = cse27 * state.xn[3];
		const amrex::Real cse29 = cse27 * state.xn[0];
		const amrex::Real cse30 = std::pow(10, 0.41599999999999998 * cse21 * cse24 + 0.32700000000000001 * cse26 - 3) * cse29 +
					  std::pow(10, 1.3 * cse21 * cse24 - 1.6200000000000001 * cse26 - 4.8449999999999998) * cse28;
		const amrex::Real cse31 = cse23 * ((cse30 < 1.0e-50) ? (1.0e-50) : (cse30));
		const amrex::Real cse32 = cse21 / (cse31 + 1);
		const amrex::Real cse33 = cse31 * cse32;
		const amrex::Real cse34 = state.T > 700.0;
		const amrex::Real cse35 = ((cse34) ? (std::pow(10, cse32 * std::log(5.9959999999999997e-30 * std::pow(state.T, 4.1881000000000004) *
										    std::pow(6.7610000000000002e-6 * state.T + 1, -5.6881000000000004) *
										    std::exp(-54657.400000000001 * cse18)) +
								       cse33 * std::log(1.3000000000000001e-9 * std::exp(-53300 * cse18))))
						   : (0.0)) *
					  std::pow(state.xn[3], 2);
		const amrex::Real cse36 = std::max(1.0e-35, 2.0000000000000002e-15 * state.ncol[ncol_H2]);
		const amrex::Real cse37 = 0.035000000000000003 * std::pow(cse36 + 1, -0.5) * std::exp(-0.00084999999999999995 * std::sqrt(cse36 + 1.0)) +
					  0.96499999999999997 * std::pow(cse36 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0, -2.0);
		const amrex::Real cse38 = cse12 * cse37;
		const amrex::Real cse39 = cse14 * cse37;
		const amrex::Real cse40 = 1.2216858566394234e-27 * cse38 + 3.6749360303422164e-18 * cse39;
		const amrex::Real cse41 =
		    (7.5e-10 * std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11 * std::pow(state.T, 0.40500000000000003)) * state.xn[0] *
		    state.xn[17];
		const amrex::Real cse42 = ((cse34) ? (std::pow(10, cse32 * std::log(6.6699999999999996e-12 * cse2 * std::exp(-63590 * cse18 - 1)) +
								       cse33 * std::log(3.5199999999999998e-9 * std::exp(-43900 * cse18))))
						   : (0.0)) *
					  state.xn[0];
		const amrex::Real cse43 = std::pow(state.T, -0.52000000000000002);
		const amrex::Real cse44 = cse43 * state.xn[16] * state.xn[2];
		const amrex::Real cse45 = cse23 * d2g;
		const amrex::Real cse46 =
		    (4.0000000000000001e-10 * std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11 * std::pow(state.T, 0.51700000000000002)) *
		    std::exp(-227 * cse18) * state.xn[11] * state.xn[1];
		const amrex::Real cse47 = std::pow(state.T, -1.3);
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
		const amrex::Real cse60 = 2.8142984982406731e-11 * std::pow(state.T, 0.26000000000000001) * state.xn[0] * state.xn[14];
		const amrex::Real cse61 =
		    1.7599999999999999e-9 * std::pow(state.T, 0.042000000000000003) * std::exp(-2.145922746781116e-5 * state.T) * state.xn[3] * state.xn[4];
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
		    ((cse34) ? (5.9078447946382801e-70 * std::pow(state.T, 13.536555999999999) *
				std::exp(-2.0391498499999999e-6 * std::pow(cse68, 8) + 0.000111954395 * std::pow(cse68, 7) -
					 0.00263197617 * std::pow(cse68, 6) + 0.034825597700000002 * std::pow(cse68, 5) -
					 0.28770560000000001 * std::pow(cse68, 4) + 1.56315498 * std::pow(cse68, 3) - 5.7393287500000003 * std::pow(cse68, 2)))
			     : (0.0));
		const amrex::Real cse71 = 1.05849862170431e-5 * std::pow(state.T, -0.64000000000000001) * state.xn[10] * state.xn[2];
		const amrex::Real cse72 = 4.8813573300130879e-6 * std::pow(cse18, 1.5) *
					  std::pow(114.81344434488582 * std::pow(cse18, 0.40699999999999997) + 1, -2.242) * state.xn[1] * state.xn[2];
		const amrex::Real cse73 = 144.3803103740824 * state.rn[2 * 0] + 220.5700827961663 * state.rn[2 * 1];
		const amrex::Real cse74 = state.c_hat * cse73;
		const amrex::Real cse75 = cse3 * cse74;
		const amrex::Real cse76 = 5.670589618368584e-11 * cse75;
		const amrex::Real cse77 = std::log(state.T);
		const amrex::Real cse78 =
		    1.2250000000000001e-11 * cse45 * state.xn[1] /
		    (6.1318063394789369e-20 * std::pow(cse75, 1.3779999999999999) *
			 (508.69999999999999 * std::pow(cse76, -1.102e-5 * cse77 - 0.4723) * std::pow(state.T, 0.015859999999999999) + 1) +
		     1);
		const amrex::Real cse79 = -cse64 - cse67 - cse70 + cse71 + cse72 + cse78;
		const amrex::Real cse80 = cse41 - cse46;
		const amrex::Real cse81 = 1.9917698086432708e-17 * cse12;
		const amrex::Real cse82 = 1.9088222220940885e-17 * cse14;
		const amrex::Real cse83 = (cse81 + cse82) * state.xn[12];
		const amrex::Real cse84 = cse22 * cse66;
		const amrex::Real cse85 = 1.4595000290776864e-10 * std::pow(state.T, -0.62) * state.xn[13] * state.xn[2];
		const amrex::Real cse86 = 2.1659999999999999e-12 * cse45 * state.xn[13] /
					  (3.5685736926834518e-27 * std::pow(cse75, 1.8740000000000001) *
					       (43750.0 * std::pow(cse76, -7.538e-5 * cse77 - 0.89639999999999997) * std::pow(state.T, 1.635e-6) + 1) +
					   1);
		const amrex::Real cse87 = 16800.0 * crate * cse28 * state.xn[12];
		const amrex::Real cse88 =
		    (1.0537934368240927e-9 * std::pow(cse18, 0.00231) +
		     std::pow(state.T, -1.5) *
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
			(1.0246966183388124e-22 * std::pow(cse75, 1.512) *
			     (5115.0 * std::pow(cse76, -5.4939999999999998e-7 * cse77 - 0.49559999999999998) * std::pow(state.T, 3.9029999999999999e-7) + 1) +
			 1) +
		    9.9999999999999994e-12 * std::pow(state.T, -0.5) *
			(cse93 * (cse93 * (0.044330000000000001 * cse93 - 0.28520000000000001) - 1.6759999999999999) + 11.19) * state.xn[2] * state.xn[6];
		const amrex::Real cse95 = crate * state.xn[7];
		const amrex::Real cse96 = 3.8500000000000001 * cse95;
		const amrex::Real cse97 = 0.15970000000000001 * std::exp(-49550.0 * cse18);
		const amrex::Real cse98 = ((1.6339999999999999e-6 * std::exp(-15230.0 * cse18) + 9.7930000000000004e-9 * std::exp(-73.799999999999997 * cse18) +
					    6.3460000000000002e-9 * std::exp(-12.17 * cse18)) /
					       std::pow(state.T, 3.0 / 2.0) +
					   2.4460185352936309e-10 * std::pow(0.00071740367932898903 * cse2 + 1, -cse97 - 1.7848999999999999) *
					       std::pow(12.244387999457523 * cse2 + 1, cse97 - 0.21509999999999996) / cse2) *
					  state.xn[2] * state.xn[8];
		const amrex::Real cse99 =
		    4.5579999999999993e-11 * cse45 * state.xn[8] /
		    (1.6851501542381549e-14 * std::pow(cse75, 1.1279999999999999) *
			 (433.10000000000002 * std::pow(cse76, -0.00013329999999999999 * cse77 - 0.81200000000000006) * std::pow(state.T, 0.04845) + 1) +
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
		const amrex::Real cse126 = std::pow(state.T, 0.14999999999999999) / (0.00088240812266906664 * std::pow(state.T, 0.65000000000000002) + 1);
		const amrex::Real cse127 = cse126 * state.xn[2];
		const amrex::Real cse128 = std::exp(-118400.0 * cse18) / (4.4460389837719569e-9 * cse127 + 626500000.0);
		const amrex::Real cse129 = 26.399999999999999 * std::sqrt(cse89 / (cse89 + 0.070000000000000007)) + 6.5;
		const amrex::Real cse130 = cse76 + 50;
		const amrex::Real cse131 = std::log(5.670589618368584e-11 * cse2 * cse74 / (state.xn[2] + 1.0e-50) + 50);
		const amrex::Real cse132 = std::sqrt(cse18) * state.xn[2];
		const amrex::Real cse133 = (1.0 / 100.0) * state.T;
		const amrex::Real cse134 = std::log(cse133);
		const amrex::Real cse135 = state.T < 500;
		const amrex::Real cse136 = std::pow(state.T, 0.070000000000000007);
		const amrex::Real cse137 =
		    4.5299999999999998e-6 * cse132 +
		    7.5799999999999997e-10 * std::pow(cse133, 0.0086999999999999994 * cse134 + 0.12809999999999999) * state.xn[0] +
		    (0.75 * ((cse135) ? (1.1000000000000001e-13 * state.T + 5.3300000000000002e-10) : (3.74757785025e-10 * cse136)) +
		     0.25 * ((cse135) ? (3.3000000000000001e-13 * state.T + 4.4299999999999997e-10) : (3.8899728635600003e-10 * cse136))) *
			state.xn[3];
		const amrex::Real cse138 = cse137 * std::exp(-91.209999999999994 * cse18) / (cse137 + 2.3e-6);
		const amrex::Real cse139 = 0.001 * state.T;
		const amrex::Real cse140 = cse139 + 1;
		const amrex::Real cse141 = std::pow(cse140, -2);
		const amrex::Real cse142 = 1.0 / cse140;
		const amrex::Real cse143 = 1.0 / (cse27 * (cse40 + 1.9999999999999999e-7) /
						      (std::pow(10, -2.3900000000000001 * cse141 + 0.055500000000000001 * cse142 - 11.06) * cse29 +
						       std::pow(10, -2.0230000000000001 * cse141 - 3.6709999999999998 * cse142 - 11.08) * cse28) +
						  1);
		const amrex::Real cse144 = cse21 * std::log(cse23);
		const amrex::Real cse145 = ((cse144 < 2) ? (10.0)
							 : ((cse144 < 4) ? ((3.0 / 2.0) * cse144 + 7)
									 : ((cse144 < 7) ? ((4.0 / 3.0) * cse144 + 23.0 / 3.0)
											 : ((cse144 < 10) ? ((1.0 / 3.0) * cse144 + 44.0 / 3.0) : (18.0)))));
		const amrex::Real cse146 = std::pow(state.T, 1.0 / 4.0);
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
		const amrex::Real cse154 = 4.34e-10 * std::pow(cse133, 0.755 - 0.16 * cse134) * state.xn[0] +
					   1.08e-14 * std::pow(state.T, 0.92600000000000005) * state.xn[2] +
					   (2.2500000000000003e-12 * std::pow(cse133, 0.52500000000000002 * cse134 + 1.198) +
					    5.2499999999999998e-13 * std::pow(cse133, cse153 + 0.88900000000000001)) *
					       state.xn[3];
		const amrex::Real cse155 = 3.5700000000000001e-10 * std::pow(cse133, 0.41899999999999998 - 0.0030000000000000001 * cse134) * state.xn[0] +
					   5.1199999999999999e-10 * std::pow(state.T, -0.074999999999999997) * state.xn[2] +
					   (3.7250000000000001e-11 * std::pow(cse133, 0.025000000000000001 * cse134 + 0.26400000000000001) +
					    1.0274999999999999e-10 * std::pow(cse133, cse153 + 0.29599999999999999)) *
					       state.xn[3];
		const amrex::Real cse156 = (1.0 / 5.0) * std::exp(-326.60000000000002 * cse18);
		const amrex::Real cse157 = 3.1899999999999998e-10 * std::pow(cse133, 0.36899999999999999 - 0.0060000000000000001 * cse134) * state.xn[0] +
					   4.8599999999999998e-10 * std::pow(state.T, -0.025999999999999999) * state.xn[2] +
					   (4.7499999999999998e-11 * std::pow(cse133, 0.041000000000000002 * cse134 + 0.20300000000000001) +
					    1.6725000000000002e-10 * std::pow(cse133, 0.058000000000000003 * cse134 + 0.23699999999999999)) *
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
		const amrex::Real cse166 = state.T < 50.0;
		const amrex::Real cse167 = state.T < 100.0;
		const amrex::Real cse168 = state.T < 6000.0;
		const amrex::Real cse169 = state.T < 10;
		const amrex::Real cse170 = std::pow(cse20, -8);
		const amrex::Real cse171 = std::pow(cse20, -7);
		const amrex::Real cse172 = std::pow(cse20, -6);
		const amrex::Real cse173 = std::pow(cse20, -5);
		const amrex::Real cse174 = std::pow(cse20, -4);
		const amrex::Real cse175 = std::pow(cse20, -3);
		const amrex::Real cse176 = std::pow(10.0, 458020.58113849716 * cse170 - 788487.2849144974 * cse171 + 579549.84668999433 * cse172 -
							      237337.82625126583 * cse173 + 59174.117778841515 * cse174 - 9226.6893143396956 * cse175 -
							      50.375425064370049 * cse21 + 868.21291710400806 * cse25 - 21.928795999999998) *
					   state.xn[2];
		const amrex::Real cse177 = std::log(cse139);
		const amrex::Real cse178 = cse177 * cse21;
		const amrex::Real cse179 = cse173 * std::pow(cse177, 5);
		const amrex::Real cse180 = cse174 * std::pow(cse177, 4);
		const amrex::Real cse181 = std::pow(cse177, 3);
		const amrex::Real cse182 = cse175 * cse181;
		const amrex::Real cse183 = std::pow(cse177, 2) * cse25;
		const amrex::Real cse184 = state.xn[5] + state.xn[6];
		const amrex::Real cse185 =
		    std::pow(10.0, 1.5714710999999999 * cse178 + 0.32168730000000001 * cse179 - 0.51002221000000003 * cse180 - 0.23619984999999999 * cse182 +
				       0.015391166 * cse183 - 22.089523) *
			state.xn[1] +
		    std::pow(10.0, 2.1892372 * cse178 + 0.19191374999999999 * cse179 - 0.16596184 * cse180 + 0.29036281000000003 * cse182 -
				       0.81520437999999995 * cse183 - 23.689236999999999) *
			cse184 +
		    std::pow(10.0, 0.43693353000000001 * cse175 * cse181 + 2.0943374000000001 * cse177 * cse21 - 0.033638326000000003 * cse179 -
				       0.14913216000000001 * cse180 - 0.77151435999999995 * cse183 - 23.962112000000001) *
			state.xn[3];
		const amrex::Real cse186 = std::pow(10, 37.383713 * cse178 + 3.8479610000000002 * cse179 + 20.159831000000001 * cse180 +
							    48.656103000000002 * cse182 + 58.145166000000003 * cse183 - 16.818342000000001) *
					       state.xn[0] +
					   cse185;
		const amrex::Real cse187 = cse170 * std::pow(cse177, 8);
		const amrex::Real cse188 = cse171 * std::pow(cse177, 7);
		const amrex::Real cse189 = cse172 * std::pow(cse177, 6);
		const amrex::Real cse190 = std::pow(10.0, 16.815729999999999 * cse178 + 983.67575999999997 * cse179 + 734.71650999999997 * cse180 +
							      343.1918 * cse182 + 96.743155000000002 * cse183 + 70.609154000000004 * cse187 +
							      364.14445999999998 * cse188 + 801.81246999999996 * cse189 - 21.928795999999998) *
					   state.xn[2];
		const amrex::Real cse191 = std::pow(10.0, 1.6802758 * cse178 - 8.8077017000000009 * cse179 - 4.7274035999999997 * cse180 +
							      4.0406627000000004 * cse182 + 0.93310621999999999 * cse183 - 6.3701156000000001 * cse187 +
							      6.4380698000000001 * cse188 + 8.9167182999999994 * cse189 - 22.921188999999998) *
					   state.xn[2];
		const amrex::Real cse192 = state.T < 1000.0;
		const amrex::Real cse193 = std::pow(10.0, 3.5692468000000002 * cse177 * cse21 - 4.2519023000000002 * cse179 - 21.328264000000001 * cse180 -
							      27.850082 * cse182 - 11.33286 * cse183 - 24.311209000000002) *
					       state.xn[0] +
					   cse185;
		const amrex::Real cse194 = std::pow(10.0, 4.6450521 * cse178 + 1.5538288 * cse179 - 5.5108049000000001 * cse180 + 5.9369081000000001 * cse182 -
							      3.7209846 * cse183 - 24.311209000000002) *
					       state.xn[0] +
					   cse185;
		const amrex::Real cse195 = std::pow(10.0, -0.62120210234156448 * cse173 - 1.5370575959566981 * cse174 + 2.5133588394664677 * cse175 +
							      3.7525488682084647 * cse21 - 2.4768712409536775 * cse25 - 23.962112000000001) *
					       state.xn[3] +
					   std::pow(10.0, 3.5440891133599637 * cse173 - 1.7105157385968939 * cse174 + 1.6702447513374001 * cse175 +
							      3.9225864834863131 * cse21 - 2.6171337683481011 * cse25 - 23.689236999999999) *
					       cse184 +
					   std::pow(10.0, 5.9406293599919788 * cse173 - 5.2566362076906961 * cse174 - 1.3586848802337363 * cse175 +
							      2.8156982240432273 * cse21 + 0.049411830040524525 * cse25 - 22.089523) *
					       state.xn[1] +
					   std::pow(10.0, 28.69470131298657 * cse173 - 56.79810800956944 * cse174 + 34.150687525022562 * cse175 +
							      8.3228160852326614 * cse21 - 11.945856385319289 * cse25 - 24.311209000000002) *
					       state.xn[0];
		const amrex::Real cse196 = ((cse168) ? (cse18) : (0.00016666666666666666));
		const amrex::Real cse197 = 1.6e-18 * std::exp(-11700.0 * cse196) + 6.7e-19 * std::exp(-5860.0 * cse196) + 3.0e-24 * std::exp(-510.0 * cse196) +
					   4.985670872372847e-33 * ((cse168) ? (std::pow(state.T, 3.7599999999999998)) : (160638152785302.53)) *
					       std::exp(-2197000.0 * ((cse168) ? (std::pow(state.T, -3)) : (4.6296296296296294e-12))) /
					       (6.0142468035272636e-8 * ((cse168) ? (std::pow(state.T, 2.1000000000000001)) : (85924621.118090734)) + 1.0);
		const amrex::Real cse198 = std::log(0.001 * ((cse168) ? (state.T) : (6000.0)));
		const amrex::Real cse199 = cse173 * std::pow(cse198, 5);
		const amrex::Real cse200 = cse174 * std::pow(cse198, 4);
		const amrex::Real cse201 = std::pow(cse198, 3);
		const amrex::Real cse202 = std::pow(cse198, 2) * cse25;
		const amrex::Real cse203 = cse198 * cse21;
		const amrex::Real cse204 = cse175 * cse201;
		const amrex::Real cse205 = ((cse168) ? (cse166) : (false));
		const amrex::Real cse206 = cse170 * std::pow(cse198, 8);
		const amrex::Real cse207 = cse171 * std::pow(cse198, 7);
		const amrex::Real cse208 = cse172 * std::pow(cse198, 6);
		const amrex::Real cse209 = 0.045999999999999999 * cse134;
		const amrex::Real cse210 = 1.7258000000000001e-6 * cse132;
		const amrex::Real cse211 =
		    8.9000000000000003e-11 * std::pow(cse133, cse209 + 0.22800000000000001) * state.xn[0] +
		    cse210 * std::exp(((cse192) ? (cse77 * (cse77 * (cse77 * (0.0070527699999999999 * cse77 - 0.11133800000000001) + 0.69763799999999998) -
							    1.3074300000000001) -
						   7.6973500000000001)
						: (cse77 * (cse77 * (cse77 * (0.093813800000000003 * cse77 - 3.0328300000000001) + 36.180300000000003) -
							    187.47399999999999) +
						   350.60899999999998))) +
		    (5.1749999999999996e-11 * std::pow(cse133, 0.037999999999999999 * cse134 + 0.16900000000000001) +
		     2.15e-11 * std::pow(cse133, 0.048000000000000001 * cse134 - 0.01)) *
			state.xn[3];
		const amrex::Real cse212 =
		    2.6400000000000002e-10 * std::pow(cse133, cse209 + 0.23100000000000001) * state.xn[0] +
		    cse210 *
			std::exp((
			    (cse192)
				? (cse77 * (cse77 * (cse77 * (0.00235272 * cse77 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999)
				: (cse77 * (cse77 * (cse77 * (0.097857299999999994 * cse77 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) +
				   386.18599999999998))) +
		    (1.11e-10 * std::pow(cse133, 0.031 * cse134 + 0.26300000000000001) +
		     4.3749999999999998e-11 * std::pow(cse133, 0.064000000000000001 * cse134 + 0.071999999999999995)) *
			state.xn[3];
		const amrex::Real cse213 =
		    2.8763333333333333e-6 * cse132 *
			std::exp(((cse192) ? (cse77 * (cse77 * (cse77 * (-0.00065632500000000003 * cse77 - 0.015089200000000001) + 0.36118400000000001) -
						       0.77378199999999997) -
					      9.2514099999999999)
					   : (cse77 * (cse77 * (cse77 * (0.10508000000000001 * cse77 - 3.4762) + 42.594999999999999) - 227.91300000000001) +
					      444.60000000000002))) +
		    1.26e-10 * std::pow(cse133, 0.057000000000000002 * cse134 + 0.115) * state.xn[0] +
		    (5.3249999999999997e-11 * std::pow(cse133, 0.049000000000000002 * cse134 - 0.0040000000000000001) +
		     1.675e-11 * std::pow(cse133, 0.10199999999999999 * cse134 - 0.085000000000000006)) *
			state.xn[3];
		const amrex::Real cse214 = 5 * std::exp(-62.460000000000001 * cse18);
		const amrex::Real cse215 = cse213 + 7.8800000000000004e-8;
		const amrex::Real cse216 = cse211 * cse214;
		const amrex::Real cse217 = (25.0 / 3.0) * cse211 * cse212 * std::exp(-101.30000000000001 * cse18) + cse212 * cse213 * cse214 + cse215 * cse216;
		const amrex::Real cse218 = cse212 + 2.65e-7;
		const amrex::Real cse219 = cse211 + 1.81e-14;
		const amrex::Real cse220 = 3 * cse213 * std::exp(-23.619999999999997 * cse18);
		const amrex::Real cse221 = cse216 * cse218 + cse218 * cse220 + cse219 * cse220;
		const amrex::Real cse222 =
		    1.0 / ((5.0 / 3.0) * cse212 * cse219 * std::exp(-38.840000000000003 * cse18) + cse215 * cse218 + cse215 * cse219 + cse217 + cse221);
		const amrex::Real cse223 = 1.8689621674132952e-20 * state.xn[3];
		const amrex::Real cse224 = Zd * cse23;
		const amrex::Real cse225 = state.c_hat * state.rn[2 * 0 + 1];
		const amrex::Real cse226 = 2.6551134042892157e-17 * state.xn[3];
		const amrex::Real cse227 = state.c_hat * state.rn[2 * 1 + 1];
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
		    5.6705896183685846e-35 * state.c_hat * cse23 * cse73 * d2g * (2.25 * std::pow(state.T, 0.14699999999999999) + 5.2199999999999998) /
			(0.049959999999999997 * std::pow(cse130, 0.43099999999999999) * (0.0043 * std::pow(cse130, 0.69199999999999995) + 1) + 1) +
		    1.602177e-12 * crate * cse129 * state.xn[0] + 3.2043540000000001e-12 * crate * cse145 * state.xn[3] +
		    1.7623947000000002e-12 * crate * (cse129 * cse29 + 2.0 * cse145 * cse28) * state.xn[5] / cse65 - 7.1777529600000012e-12 * cse110 -
		    1.3659996292638161e-10 * cse126 * cse128 * cse69 / (1.3338116951315871e-8 * cse127 * cse128 + 1) -
		    5.7927337833400002e-20 * cse138 * state.xn[8] / (2 * cse138 + 1) -
		    cse147 * std::pow(1 - std::exp(-state.T), 1000) * state.xn[9] /
			(cse147 * cse151 + cse149 * std::pow(cse147 / cse152, alpha_co_interp2d(state.T, cse150)) * (-cse148 * cse151 * cse152 + 1) + cse149) +
		    3.0000000000000002e-15 * cse23 * d2g * (6.7291434000000005e-12 * cse143 + 3.2043540000000002e-13) * state.xn[0] -
		    7.1777529600000012e-12 * cse35 + cse40 * (2.5634832e-11 * cse143 + 6.4087080000000003e-13) * state.xn[3] -
		    1.8998126555754159e-29 * cse45 * std::pow(state.T, 0.45350000000000001 + 2.234 / cse131) *
			std::exp(cse131 * (1.4419999999999999 - 0.050889999999999998 * cse131)) * state.xn[2] -
		    2.1789607199999999e-11 * cse70 - (2.3872025442750958e-19 * cse160 * cse165 + 2.8023005140019997e-18 * cse164 * cse165) * state.xn[11] -
		    ((((cse135 && cse166 && cse167 && cse168)
			   ? (cse169 || cse176 + cse186 < 1.0e-100)
			   : (((cse135 && cse167 && cse168)
				   ? (cse169 || cse186 + cse190 < 1.0e-100)
				   : (((cse167 && cse168)
					   ? (cse169 || cse186 + cse191 < 1.0e-100)
					   : (((cse135 && cse166 && cse168 && cse192)
						   ? (cse169 || cse176 + cse193 < 1.0e-100)
						   : (((cse135 && cse168 && cse192)
							   ? (cse169 || cse190 + cse193 < 1.0e-100)
							   : (((cse168 && cse192)
								   ? (cse169 || cse191 + cse193 < 1.0e-100)
								   : (((cse135 && cse166 && cse168)
									   ? (cse169 || cse176 + cse194 < 1.0e-100)
									   : (((cse135 && cse168)
										   ? (cse169 || cse190 + cse194 < 1.0e-100)
										   : (((cse168) ? (cse169 || cse191 + cse194 < 1.0e-100)
												: (((cse135) ? (cse169 ||
														std::pow(10.0, 7500.646099484944 * cse170 +
																   21588.957206746247 * cse171 +
																   26530.843406310876 * cse172 +
																   18165.631967965237 * cse173 +
																   7572.4886742758581 * cse174 +
																   1974.131269262874 * cse175 +
																   30.129743459482278 * cse21 +
																   310.58441786958315 * cse25 -
																   21.928795999999998) *
															    state.xn[2] +
															cse195 <
														    1.0e-100)
													     : (cse169 ||
														std::pow(10.0, -676.68255490510751 * cse170 +
																   381.69251128040054 * cse171 +
																   295.04162851880631 * cse172 -
																   162.65264843551881 * cse173 -
																   48.723840736518795 * cse174 +
																   23.242975457496804 * cse175 +
																   3.0106500755647456 * cse21 +
																   2.9956460707652877 * cse25 -
																   22.921188999999998) *
															    state.xn[2] +
															cse195 <
														    1.0e-100)))))))))))))))))))))
			 ? (0.0)
			 : (cse197 /
			    (cse197 /
				 (std::pow(10.0, 0.19191374999999999 * cse199 - 0.16596184 * cse200 - 0.81520437999999995 * cse202 + 2.1892372 * cse203 +
						     0.29036281000000003 * cse204 - 23.689236999999999) *
				      cse184 +
				  std::pow(10.0, 0.32168730000000001 * cse199 - 0.51002221000000003 * cse200 + 0.015391166 * cse202 +
						     1.5714710999999999 * cse203 - 0.23619984999999999 * cse204 - 22.089523) *
				      state.xn[1] +
				  std::pow(10.0, 0.43693353000000001 * cse175 * cse201 + 2.0943374000000001 * cse198 * cse21 - 0.033638326000000003 * cse199 -
						     0.14913216000000001 * cse200 - 0.77151435999999995 * cse202 - 23.962112000000001) *
				      state.xn[3] +
				  ((cse135)
				       ? (std::pow(10.0, 70.609154000000004 * ((cse205) ? (6486.7025759648268 * cse170) : (cse206)) +
							     364.14445999999998 * ((cse205) ? (-2165.3145153286073 * cse171) : (cse207)) +
							     801.81246999999996 * ((cse205) ? (722.79974230133178 * cse172) : (cse208)) +
							     983.67575999999997 * ((cse205) ? (-241.2764814406587 * cse173) : (cse199)) +
							     734.71650999999997 * ((cse205) ? (80.540068139807445 * cse174) : (cse200)) +
							     343.1918 * ((cse205) ? (-26.884935229628727 * cse175) : (cse204)) +
							     16.815729999999999 * ((cse205) ? (-2.9957322735539909 * cse21) : (cse203)) +
							     96.743155000000002 * ((cse205) ? (8.9744118548129634 * cse25) : (cse202)) - 21.928795999999998))
				       : (std::pow(10.0, -8.8077017000000009 * cse199 - 4.7274035999999997 * cse200 + 0.93310621999999999 * cse202 +
							     1.6802758 * cse203 + 4.0406627000000004 * cse204 - 6.3701156000000001 * cse206 +
							     6.4380698000000001 * cse207 + 8.9167182999999994 * cse208 - 22.921188999999998))) *
				      state.xn[2] +
				  ((((cse168) ? (cse167) : (false)))
				       ? (std::pow(10, 3.8479610000000002 * cse199 + 20.159831000000001 * cse200 + 58.145166000000003 * cse202 +
							   37.383713 * cse203 + 48.656103000000002 * cse204 - 16.818342000000001))
				       : ((((cse168) ? (cse192) : (false)))
					      ? (std::pow(10.0, 3.5692468000000002 * cse198 * cse21 - 4.2519023000000002 * cse199 -
								    21.328264000000001 * cse200 - 11.33286 * cse202 - 27.850082 * cse204 - 24.311209000000002))
					      : (std::pow(10.0, 1.5538288 * cse199 - 5.5108049000000001 * cse200 - 3.7209846 * cse202 + 4.6450521 * cse203 +
								    5.9369081000000001 * cse204 - 24.311209000000002)))) *
				      state.xn[0]) +
			     1))) *
			state.xn[3] -
		    ((state.T < 1000000.0) ? (1.4210469458259591e-21 * cse217 * cse222 + 2.5697412351439998e-22 * cse221 * cse222) : (0.0)) * state.xn[7] -
		    std::min(100.0 * cse45 * psigd_coll_interp2d(cse23, state.T),
			     3.2000000000000002e-32 * cse2 * std::pow(cse23, 2) * d2g *
				 (state.T - 2.1544346900318838 * std::pow(2.8677558904086741e-6 * cse74 + 4.1397668473788896, 1.0 / 6.0)));
		ydot[19] = -1.252779946046115e-21 * cse12 * cse224 - cse13 * state.xn[15] - cse223 * cse38 - cse50 * state.xn[14] - cse81 * state.xn[12];
		ydot[20] = -cse223 * cse225 * cse37 - 9.1016827996670106e-22 * cse224 * cse225 - 1.9917698086432708e-17 * cse225 * state.xn[12] -
			   4.2188032355736613e-18 * cse225 * state.xn[14] - 1.7141251340095377e-18 * cse225 * state.xn[15];
		ydot[21] = -cse102 - cse118 - 2.0584539190345922e-21 * cse14 * cse224 - cse15 * state.xn[15] - cse226 * cse39 - cse51 * state.xn[14] -
			   cse82 * state.xn[12];
		ydot[22] = -cse101 * cse227 - cse117 * cse227 - 1.6856051574414119e-21 * cse224 * cse227 - cse226 * cse227 * cse37 -
			   1.9088222220940885e-17 * cse227 * state.xn[12] - 1.168047934563505e-18 * cse227 * state.xn[14] -
			   4.1730320353928637e-18 * cse227 * state.xn[15];
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
		const amrex::Real cse0 =
		    1.0 / (2.0709734999999996e-16 * state.xn[0] + 2.0709734999999996e-16 * state.xn[10] + 2.0709734999999996e-16 * state.xn[11] +
			   2.0709734999999996e-16 * state.xn[12] + 2.0709734999999996e-16 * state.xn[13] + 2.0709734999999996e-16 * state.xn[14] +
			   2.0709734999999996e-16 * state.xn[15] + 2.0709734999999996e-16 * state.xn[16] + 2.0709734999999996e-16 * state.xn[17] +
			   2.0709734999999996e-16 * state.xn[1] + 2.0709734999999996e-16 * state.xn[2] + 3.4516225000000006e-16 * state.xn[3] +
			   3.4516225000000006e-16 * state.xn[4] + 2.0709734999999996e-16 * state.xn[5] + 2.0709734999999996e-16 * state.xn[6] +
			   2.0709734999999996e-16 * state.xn[7] + 2.0709734999999996e-16 * state.xn[8] + 2.0709734999999996e-16 * state.xn[9]);
		const amrex::Real cse1 = std::pow(state.T, -2);
		const amrex::Real cse2 = 1.0 / state.T;
		const amrex::Real cse3 = std::exp(-22.5 * cse2);
		const amrex::Real cse4 = 2.8349999999999999e-12 * cse1 * cse3 * state.xn[3] * state.xn[6];
		const amrex::Real cse5 = -cse4;
		const amrex::Real cse6 = 7.3171760954257497e-12 * std::pow(state.T, -0.73999999999999999) * state.xn[0] * state.xn[14];
		const amrex::Real cse7 = std::sqrt(cse2);
		const amrex::Real cse8 = std::sqrt(3) * cse2 * cse7 * state.xn[15];
		const amrex::Real cse9 = 1.7697487500000004e-8 * cse8 * state.xn[6];
		const amrex::Real cse10 = std::pow(state.xn[3], 2);
		const amrex::Real cse11 = std::pow(state.T, 4.1881000000000004);
		const amrex::Real cse12 = 6.7610000000000002e-6 * state.T + 1;
		const amrex::Real cse13 = std::pow(cse12, 5.6881000000000004);
		const amrex::Real cse14 = 54657.400000000001 * cse2;
		const amrex::Real cse15 = std::exp(-cse14);
		const amrex::Real cse16 = cse15 / cse13;
		const amrex::Real cse17 = M_LN10;
		const amrex::Real cse18 = 1.0 / cse17;
		const amrex::Real cse19 = 2 * state.xn[3];
		const amrex::Real cse20 = cse19 + state.xn[0] + state.xn[1] + state.xn[10] + state.xn[14] + state.xn[15] + 3 * state.xn[16] + 2 * state.xn[4];
		const amrex::Real cse21 = 1.0 / cse20;
		const amrex::Real cse22 = std::log((1.0 / 10000.0) * state.T);
		const amrex::Real cse23 = std::pow(cse17, -2);
		const amrex::Real cse24 = std::pow(cse22, 2) * cse23;
		const amrex::Real cse25 = std::pow(10, 1.3 * cse18 * cse22 - 1.6200000000000001 * cse24 - 4.8449999999999998);
		const amrex::Real cse26 = cse21 * cse25 * state.xn[3];
		const amrex::Real cse27 = std::pow(10, 0.41599999999999998 * cse18 * cse22 + 0.32700000000000001 * cse24 - 3);
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
		const amrex::Real cse45 = cse18 / std::pow(cse33, 2);
		const amrex::Real cse46 = cse41 * cse45;
		const amrex::Real cse47 = std::pow(cse20, 2);
		const amrex::Real cse48 = cse31 * cse40 * cse45 * cse47;
		const amrex::Real cse49 = std::pow(10, cse34 * cse44 + cse35 * cse42);
		const amrex::Real cse50 = cse17 * cse49;
		const amrex::Real cse51 = state.T > 700.0;
		const amrex::Real cse52 = ((cse51) ? (cse50 * (53300.0 * cse36 + cse41 * cse43 - cse42 * cse48 - cse44 * cse46 +
							       1.6677785190126752e+29 * cse13 * cse34 *
								   (-2.3058963562359999e-34 * cse11 * std::pow(cse12, -6.6881000000000004) * cse15 +
								    3.2772577039999999e-25 * cse16 * std::pow(state.T, 2.1881000000000004) +
								    2.51118476e-29 * cse16 * std::pow(state.T, 3.1881000000000004)) *
								   std::exp(cse14) / cse11))
						   : (0));
		const amrex::Real cse53 =
		    state.xn[0] * state.xn[17] * (-3.435e-10 * std::pow(state.T, -1.458) + 2.0209499999999999e-11 * std::pow(state.T, -0.59499999999999997));
		const amrex::Real cse54 = std::sqrt(state.T);
		const amrex::Real cse55 = 1.0 / cse54;
		const amrex::Real cse56 = std::pow(state.T, -3.0 / 2.0);
		const amrex::Real cse57 = 63590 * cse2 + 1;
		const amrex::Real cse58 = std::exp(-cse57);
		const amrex::Real cse59 = std::log(3.5199999999999998e-9 * std::exp(-43900 * cse2));
		const amrex::Real cse60 = cse34 * cse59;
		const amrex::Real cse61 = std::log(6.6699999999999996e-12 * cse54 * cse58);
		const amrex::Real cse62 = std::pow(10, cse34 * cse61 + cse35 * cse59);
		const amrex::Real cse63 = cse17 * cse62;
		const amrex::Real cse64 =
		    ((cse51) ? (cse63 * (149925037481.25937 * cse34 * cse55 * (3.3349999999999998e-12 * cse55 * cse58 + 4.2414529999999999e-7 * cse56 * cse58) *
					     std::exp(cse57) +
					 43900.0 * cse36 + cse41 * cse60 - cse46 * cse61 - cse48 * cse59))
			     : (0));
		const amrex::Real cse65 = std::pow(state.T, -1.52) * state.xn[16] * state.xn[2];
		const amrex::Real cse66 = std::exp(-227 * cse2);
		const amrex::Real cse67 =
		    cse66 * state.xn[1] * state.xn[11] *
		    (2.6759999999999999e-12 * std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12 * std::pow(state.T, -0.48299999999999998));
		const amrex::Real cse68 = std::pow(state.T, -2.2999999999999998);
		const amrex::Real cse69 = std::exp(-23 * cse2);
		const amrex::Real cse70 = cse69 * state.xn[3] * state.xn[8];
		const amrex::Real cse71 = cse68 * cse70;
		const amrex::Real cse72 = std::pow(state.T, 0.042000000000000003);
		const amrex::Real cse73 = std::exp(-2.145922746781116e-5 * state.T);
		const amrex::Real cse74 = cse73 * state.xn[4];
		const amrex::Real cse75 = cse74 * state.xn[3];
		const amrex::Real cse76 = 3.7768240343347639e-14 * cse72 * cse75;
		const amrex::Real cse77 = 7.3920000000000002e-11 * cse75 * std::pow(state.T, -0.95799999999999996);
		const amrex::Real cse78 = std::pow(state.T, -3.2999999999999998);
		const amrex::Real cse79 = cse66 * (4.0000000000000001e-10 * std::pow(state.T, 0.0066899999999999998) +
						   1.1000000000000001e-11 * std::pow(state.T, 0.51700000000000002));
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
		const amrex::Real cse93 = std::pow(cse92, -2);
		const amrex::Real cse94 = 1.811320754716981e-13 * cse93;
		const amrex::Real cse95 = 6.7743911789075847e-6 * std::pow(state.T, -1.6400000000000001) * state.xn[10] * state.xn[2];
		const amrex::Real cse96 = std::exp(0.11 * cse2);
		const amrex::Real cse97 = cse96 * state.xn[15];
		const amrex::Real cse98 = cse97 * state.xn[7];
		const amrex::Real cse99 =
		    8.7465066538127767e-11 * cse98 * std::pow(state.T, -2.339) + 2.6955143233113923e-10 * cse98 * std::pow(state.T, -1.339);
		const amrex::Real cse100 = cse95 + cse99;
		const amrex::Real cse101 = std::log(8.6173000000000003e-5 * state.T);
		const amrex::Real cse102 = std::pow(cse101, 2);
		const amrex::Real cse103 = std::pow(cse101, 3);
		const amrex::Real cse104 = std::pow(cse101, 4);
		const amrex::Real cse105 = std::pow(cse101, 5);
		const amrex::Real cse106 = std::pow(cse101, 6);
		const amrex::Real cse107 = std::pow(cse101, 7);
		const amrex::Real cse108 =
		    std::exp(-2.0391498499999999e-6 * std::pow(cse101, 8) - 5.7393287500000003 * cse102 + 1.56315498 * cse103 - 0.28770560000000001 * cse104 +
			     0.034825597700000002 * cse105 - 0.00263197617 * cse106 + 0.000111954395 * cse107);
		const amrex::Real cse109 = 5.9078447946382801e-70 * cse108 * std::pow(state.T, 13.536555999999999);
		const amrex::Real cse110 = state.xn[0] * state.xn[2];
		const amrex::Real cse111 =
		    cse110 * ((cse51) ? (7.9971871901929577e-69 * cse108 * std::pow(state.T, 12.536555999999999) +
					 cse109 * (-11.478657500000001 * cse101 * cse2 + 4.6894649399999997 * cse102 * cse2 - 1.1508224 * cse103 * cse2 +
						   0.1741279885 * cse104 * cse2 - 0.015791857020000001 * cse105 * cse2 +
						   0.00078368076500000001 * cse106 * cse2 - 1.6313198799999999e-5 * cse107 * cse2))
				      : (0));
		const amrex::Real cse112 = std::pow(cse2, 1.5);
		const amrex::Real cse113 = 114.81344434488582 * std::pow(cse2, 0.40699999999999997) + 1;
		const amrex::Real cse114 = std::pow(cse113, -2.242);
		const amrex::Real cse115 = cse2 * state.xn[1] * state.xn[2];
		const amrex::Real cse116 = 7.3220359950196314e-6 * cse112 * cse114 * cse115;
		const amrex::Real cse117 = 0.0005114031087522854 * std::pow(cse113, -3.242) * cse115 * std::pow(cse2, 1.907);
		const amrex::Real cse118 = 144.3803103740824 * state.rn[2 * 0] + 220.5700827961663 * state.rn[2 * 1];
		const amrex::Real cse119 = state.c_hat * cse118;
		const amrex::Real cse120 = cse119 * cse84;
		const amrex::Real cse121 = std::pow(cse120, 1.3779999999999999);
		const amrex::Real cse122 = 5.670589618368584e-11 * cse120;
		const amrex::Real cse123 = std::log(state.T);
		const amrex::Real cse124 = 1.102e-5 * cse123 + 0.4723;
		const amrex::Real cse125 = std::pow(cse122, -cse124);
		const amrex::Real cse126 = cse125 * std::pow(state.T, 0.015859999999999999);
		const amrex::Real cse127 = 508.69999999999999 * cse126;
		const amrex::Real cse128 = cse127 + 1;
		const amrex::Real cse129 = cse121 * cse128;
		const amrex::Real cse130 = -cse124;
		const amrex::Real cse131 = 0.5 * cse2;
		const amrex::Real cse132 = cse2 * std::log(cse122);
		const amrex::Real cse133 = 6.1318063394789369e-20 * cse121;
		const amrex::Real cse134 = cse128 * cse133 + 1;
		const amrex::Real cse135 = cse20 * d2g;
		const amrex::Real cse136 = 1.2250000000000001e-11 * cse135 * state.xn[1] / std::pow(cse134, 2);
		const amrex::Real cse137 =
		    cse136 * (-4.2248145679009874e-20 * cse129 * cse2 - cse133 * (8.0679819999999989 * cse125 * std::pow(state.T, -0.98414000000000001) +
										  cse127 * (cse130 * cse131 - 1.102e-5 * cse132)));
		const amrex::Real cse138 = 1.0 / cse92;
		const amrex::Real cse139 = 9.0566037735849048e-14 * cse90;
		const amrex::Real cse140 = cse138 * cse139;
		const amrex::Real cse141 = cse139 * cse93;
		const amrex::Real cse142 = cse141 * cse91;
		const amrex::Real cse143 = -cse140 + cse142;
		const amrex::Real cse144 = cse138 * state.xn[16];
		const amrex::Real cse145 = 3.7810000000000002e-10 * std::pow(state.T, -1.1899999999999999) * state.xn[11];
		const amrex::Real cse146 = cse82 * cse87 * cse93 * state.xn[11] * state.xn[16];
		const amrex::Real cse147 = cse144 * cse145 + 1.1264150943396225e-13 * cse146 * std::pow(state.T, -0.68999999999999995) * state.xn[3];
		const amrex::Real cse148 = cse111 + cse116 - cse117 - cse137 + cse143 + cse147;
		const amrex::Real cse149 = cse0 * (2 * cse10 * cse52 - cse100 - cse148 - cse5 - cse53 - cse6 + 2 * cse64 * state.xn[0] * state.xn[3] -
						   1.5566492609718528e-6 * cse65 + cse67 + 9.8669999999999991e-12 * cse69 * cse78 * state.xn[3] * state.xn[8] -
						   5.5769999999999994e-13 * cse71 - cse76 + cse77 + cse81 - cse9 - cse90 * cse94);
		const amrex::Real cse150 = cse149 * state.T;
		const amrex::Real cse151 = 2.0709734999999996e-16 * cse150;
		const amrex::Real cse152 = 2.8142984982406731e-11 * std::pow(state.T, 0.26000000000000001);
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
		const amrex::Real cse172 = 7.5e-10 * std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11 * std::pow(state.T, 0.40500000000000003);
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
		const amrex::Real cse204 = std::pow(state.xn[2], 2);
		const amrex::Real cse205 = 1.0 / cse204;
		const amrex::Real cse206 = 2.2528301886792451e-13 * std::pow(state.T, 0.31);
		const amrex::Real cse207 = cse205 * cse206 * cse87 * cse93 * state.xn[11] * state.xn[16] * state.xn[3];
		const amrex::Real cse208 = std::pow(state.T, -0.52000000000000002);
		const amrex::Real cse209 = 2.9935562710997168e-6 * cse208;
		const amrex::Real cse210 = cse205 * cse54;
		const amrex::Real cse211 = cse210 * cse89;
		const amrex::Real cse212 = 3.6226415094339619e-13 * cse93;
		const amrex::Real cse213 = 1.811320754716981e-13 * cse138;
		const amrex::Real cse214 = cse211 * cse213;
		const amrex::Real cse215 = cse211 * cse94;
		const amrex::Real cse216 = cse215 * cse91;
		const amrex::Real cse217 = -cse214 + cse216;
		const amrex::Real cse218 = 1.05849862170431e-5 * std::pow(state.T, -0.64000000000000001);
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
		const amrex::Real cse249 = 0.035000000000000003 * std::pow(cse248 + 1, -0.5) * std::exp(-0.00084999999999999995 * std::sqrt(cse248 + 1.0)) +
					   0.96499999999999997 * std::pow(cse248 / std::max(1.0e-50, 1.4142135623730953e-5 * vdisp) + 1.0, -2.0);
		const amrex::Real cse250 = 2.4433717132788467e-27 * cse249;
		const amrex::Real cse251 = state.c_hat * state.rn[2 * 1];
		const amrex::Real cse252 = 7.3498720606844327e-18 * cse249;
		const amrex::Real cse253 = cse69 * std::pow(state.T, -1.3);
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
		const amrex::Real cse283 = 7.9513696852843427e-10 * std::pow(state.T, -0.33900000000000002);
		const amrex::Real cse284 = cse283 * cse97;
		const amrex::Real cse285 = 6.5199999999999996 * crate;
		const amrex::Real cse286 = cse218 * state.xn[2];
		const amrex::Real cse287 = cse193 + cse286;
		const amrex::Real cse288 = 1.99e-9 * std::pow(state.T, -0.19);
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
		const amrex::Real cse346 = std::pow(cse122, -cse345);
		const amrex::Real cse347 = cse346 * std::pow(state.T, 3.9029999999999999e-7);
		const amrex::Real cse348 = 5115.0 * cse347;
		const amrex::Real cse349 = cse348 + 1;
		const amrex::Real cse350 = std::pow(cse120, 1.512);
		const amrex::Real cse351 = 1.0246966183388124e-22 * cse350;
		const amrex::Real cse352 = cse349 * cse351 + 1;
		const amrex::Real cse353 = 1.0 / cse352;
		const amrex::Real cse354 = 5.5720000000000001e-12 * cse353;
		const amrex::Real cse355 = d2g * state.xn[6];
		const amrex::Real cse356 = cse354 * cse355;
		const amrex::Real cse357 = std::pow(cse2, 0.00231);
		const amrex::Real cse358 = std::exp(-8110.0 * cse2);
		const amrex::Real cse359 = std::exp(-26.600000000000001 * cse2);
		const amrex::Real cse360 = std::exp(-7.6200000000000001 * cse2);
		const amrex::Real cse361 = std::exp(-1.3799999999999999 * cse2);
		const amrex::Real cse362 = 0.00015100000000000001 * cse358 + 1.31e-7 * cse359 + 3.4e-8 * cse360 + 6.9699999999999997e-9 * cse361;
		const amrex::Real cse363 = std::pow(state.T, -1.5);
		const amrex::Real cse364 = state.xn[16] * state.xn[7] *
					   (-2.4342628390636543e-12 * cse2 * cse357 - 1.5 * cse362 * std::pow(state.T, -2.5) +
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
		const amrex::Real cse380 = cse369 * cse376 / std::pow(cse370, 2);
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
		    (-3.0 / 2.0 * cse385 / std::pow(state.T, 5.0 / 2.0) - 1.2230092676468155e-10 * cse392 * cse56 +
		     cse394 * (-cse393 * std::log(cse386) - 0.00035870183966449452 * cse389 * cse55 / cse386) +
		     cse394 * (cse393 * std::log(cse390) - 6.1221939997287613 * cse391 * cse55 / cse390) +
		     cse56 * (0.024885819999999999 * cse1 * cse382 + 7.2272340000000005e-7 * cse1 * cse383 + 7.7230820000000007e-8 * cse1 * cse384));
		const amrex::Real cse396 = 9.0489001802816564e-11 * std::pow(state.T, -1.6200000000000001) * state.xn[13] * state.xn[2];
		const amrex::Real cse397 = cse123 * cse18;
		const amrex::Real cse398 = 0.044330000000000001 * cse397 - 0.28520000000000001;
		const amrex::Real cse399 = cse397 * cse398 - 1.6759999999999999;
		const amrex::Real cse400 = cse397 * cse399 + 11.19;
		const amrex::Real cse401 = cse400 * state.xn[6];
		const amrex::Real cse402 = 4.9999999999999997e-12 * cse363 * cse401 * state.xn[2];
		const amrex::Real cse403 = cse70 * cse78;
		const amrex::Real cse404 = 7.5899999999999998e-12 * cse403;
		const amrex::Real cse405 = 9.9999999999999994e-12 * std::pow(state.T, -0.5);
		const amrex::Real cse406 = cse405 * state.xn[2];
		const amrex::Real cse407 = cse406 * state.xn[6] * (cse39 * cse399 + cse397 * (0.044330000000000001 * cse123 * cse37 + cse39 * cse398));
		const amrex::Real cse408 = std::pow(cse120, 1.8740000000000001);
		const amrex::Real cse409 = 7.538e-5 * cse123 + 0.89639999999999997;
		const amrex::Real cse410 = std::pow(cse122, -cse409);
		const amrex::Real cse411 = cse410 * std::pow(state.T, 1.635e-6);
		const amrex::Real cse412 = 43750.0 * cse411;
		const amrex::Real cse413 = cse412 + 1;
		const amrex::Real cse414 = cse408 * cse413;
		const amrex::Real cse415 = -cse409;
		const amrex::Real cse416 = 3.5685736926834518e-27 * cse408;
		const amrex::Real cse417 = cse413 * cse416 + 1;
		const amrex::Real cse418 = 2.1659999999999999e-12 * cse135 * state.xn[13] / std::pow(cse417, 2);
		const amrex::Real cse419 =
		    cse418 * (-3.3437535500443945e-27 * cse2 * cse414 - cse416 * (0.071531250000000005 * cse410 * std::pow(state.T, -0.99999836499999994) +
										  cse412 * (cse131 * cse415 - 7.538e-5 * cse132)));
		const amrex::Real cse420 = cse349 * cse350;
		const amrex::Real cse421 = -cse345;
		const amrex::Real cse422 = 5.5720000000000001e-12 * cse135 * state.xn[6] / std::pow(cse352, 2);
		const amrex::Real cse423 =
		    cse422 * (-7.7467064346414215e-23 * cse2 * cse420 - cse351 * (0.0019963845000000001 * cse346 * std::pow(state.T, -0.99999960970000001) +
										  cse348 * (cse131 * cse421 - 5.4939999999999998e-7 * cse132)));
		const amrex::Real cse424 = std::pow(cse120, 1.1279999999999999);
		const amrex::Real cse425 = 0.00013329999999999999 * cse123 + 0.81200000000000006;
		const amrex::Real cse426 = std::pow(cse122, -cse425);
		const amrex::Real cse427 = cse426 * std::pow(state.T, 0.04845);
		const amrex::Real cse428 = 433.10000000000002 * cse427;
		const amrex::Real cse429 = cse428 + 1;
		const amrex::Real cse430 = cse424 * cse429;
		const amrex::Real cse431 = -cse425;
		const amrex::Real cse432 = 1.6851501542381549e-14 * cse424;
		const amrex::Real cse433 = cse429 * cse432 + 1;
		const amrex::Real cse434 = 4.5579999999999993e-11 * cse135 * state.xn[8] / std::pow(cse433, 2);
		const amrex::Real cse435 =
		    cse434 * (-9.5042468699031921e-15 * cse2 * cse430 - cse432 * (20.983695000000001 * cse426 * std::pow(state.T, -0.95155000000000001) +
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
		const amrex::Real cse461 = 1.4595000290776864e-10 * std::pow(state.T, -0.62);
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
		const amrex::Real cse708 = 2.25 * std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
		const amrex::Real cse709 = cse122 + 50;
		const amrex::Real cse710 = 0.0043 * std::pow(cse709, 0.69199999999999995) + 1;
		const amrex::Real cse711 = 0.049959999999999997 * std::pow(cse709, 0.43099999999999999) * cse710 + 1;
		const amrex::Real cse712 = 1.0 / cse711;
		const amrex::Real cse713 = state.xn[2] + 1.0e-50;
		const amrex::Real cse714 = 1.0 / cse713;
		const amrex::Real cse715 = cse54 * cse714;
		const amrex::Real cse716 = 5.670589618368584e-11 * cse119;
		const amrex::Real cse717 = cse715 * cse716 + 50;
		const amrex::Real cse718 = std::log(cse717);
		const amrex::Real cse719 = 0.45350000000000001 + 2.234 / cse718;
		const amrex::Real cse720 = std::pow(state.T, cse719);
		const amrex::Real cse721 = 1.4419999999999999 - 0.050889999999999998 * cse718;
		const amrex::Real cse722 = std::exp(cse718 * cse721);
		const amrex::Real cse723 = 1.8998126555754159e-29 * cse720 * cse722;
		const amrex::Real cse724 = d2g * state.xn[2];
		const amrex::Real cse725 = cse723 * cse724;
		const amrex::Real cse726 = std::pow(state.T, 0.14999999999999999);
		const amrex::Real cse727 = 0.00088240812266906664 * std::pow(state.T, 0.65000000000000002) + 1;
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
		const amrex::Real cse743 = -2.1544346900318838 * std::pow(cse742, 1.0 / 6.0) + state.T;
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
		const amrex::Real cse756 = std::pow(cse703, -2);
		const amrex::Real cse757 = 0.5 * cse756;
		const amrex::Real cse758 = cse204 / std::pow(cse20, 3);
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
		const amrex::Real cse769 = std::pow(cse768, -2);
		const amrex::Real cse770 = 1.0 / cse768;
		const amrex::Real cse771 = std::pow(10, -2.0230000000000001 * cse769 - 3.6709999999999998 * cse770 - 11.08);
		const amrex::Real cse772 = cse21 * cse771;
		const amrex::Real cse773 = cse772 * state.xn[3];
		const amrex::Real cse774 = std::pow(10, -2.3900000000000001 * cse769 + 0.055500000000000001 * cse770 - 11.06);
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
		const amrex::Real cse786 = std::log(cse785);
		const amrex::Real cse787 = 0.0086999999999999994 * cse786;
		const amrex::Real cse788 = std::pow(cse785, cse787 + 0.12809999999999999);
		const amrex::Real cse789 = 7.5799999999999997e-10 * cse788 * state.xn[0];
		const amrex::Real cse790 = state.T < 500;
		const amrex::Real cse791 = std::pow(state.T, 0.070000000000000007);
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
		const amrex::Real cse802 = std::pow(cse794, -2);
		const amrex::Real cse803 = std::pow(state.T, 1.0 / 4.0);
		const amrex::Real cse804 = 1.75 * cse803;
		const amrex::Real cse805 = std::exp(-state.T);
		const amrex::Real cse806 = 1 - cse805;
		const amrex::Real cse807 = std::pow(cse806, 1000);
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
		const amrex::Real cse835 = std::pow(cse175, -2);
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
		const amrex::Real cse844 = 5.7927337833400002e-20 * cse843 * state.xn[8] / std::pow(cse798, 2);
		const amrex::Real cse845 = std::pow(cse781, -2);
		const amrex::Real cse846 = cse157 * cse780;
		const amrex::Real cse847 = cse158 * cse771;
		const amrex::Real cse848 = cse160 * cse774;
		const amrex::Real cse849 = cse847 + cse848;
		const amrex::Real cse850 = cse779 / std::pow(cse777, 2);
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
					   std::pow(cse817, 3.0 / 2.0);
		const amrex::Real cse862 = cse861 * state.ncol[ncol_CO];
		const amrex::Real cse863 = ((cse815) ? (0) : (878350727.21550155 * cse862));
		const amrex::Real cse864 = alpha_co_interp2d_partial_1(state.T, cse819);
		const amrex::Real cse865 = std::log(cse826);
		const amrex::Real cse866 = cse864 * cse865;
		const amrex::Real cse867 = cse811 / std::pow(cse822, 2);
		const amrex::Real cse868 = nhalf_co_interp2d_partial_1(state.T, cse819);
		const amrex::Real cse869 = cse863 * cse868;
		const amrex::Real cse870 = cse827 / cse811;
		const amrex::Real cse871 = cse822 * cse870;
		const amrex::Real cse872 = std::pow(cse820, -2);
		const amrex::Real cse873 = llte_co_interp2d_partial_1(state.T, cse819);
		const amrex::Real cse874 = -cse811 * cse863 * cse872 * cse873 + cse829 * (cse808 * cse822 * cse863 * cse872 * cse873 - cse823 * cse869);
		const amrex::Real cse875 = cse811 * state.xn[9];
		const amrex::Real cse876 = cse807 * cse875 / std::pow(cse831, 2);
		const amrex::Real cse877 = state.T < 50.0;
		const amrex::Real cse878 = state.T < 100.0;
		const amrex::Real cse879 = state.T < 6000.0;
		const amrex::Real cse880 = state.T < 10;
		const amrex::Real cse881 = std::pow(cse17, -8);
		const amrex::Real cse882 = std::pow(cse17, -7);
		const amrex::Real cse883 = std::pow(cse17, -6);
		const amrex::Real cse884 = std::pow(cse17, -5);
		const amrex::Real cse885 = std::pow(cse17, -4);
		const amrex::Real cse886 = std::pow(cse17, -3);
		const amrex::Real cse887 = std::pow(10.0, -50.375425064370049 * cse18 + 868.21291710400806 * cse23 + 458020.58113849716 * cse881 -
							      788487.2849144974 * cse882 + 579549.84668999433 * cse883 - 237337.82625126583 * cse884 +
							      59174.117778841515 * cse885 - 9226.6893143396956 * cse886 - 21.928795999999998) *
					   state.xn[2];
		const amrex::Real cse888 = std::log(cse767);
		const amrex::Real cse889 = cse18 * cse888;
		const amrex::Real cse890 = cse884 * std::pow(cse888, 5);
		const amrex::Real cse891 = cse885 * std::pow(cse888, 4);
		const amrex::Real cse892 = std::pow(cse888, 3);
		const amrex::Real cse893 = cse886 * cse892;
		const amrex::Real cse894 = cse23 * std::pow(cse888, 2);
		const amrex::Real cse895 = state.xn[5] + state.xn[6];
		const amrex::Real cse896 =
		    std::pow(10.0, 1.5714710999999999 * cse889 + 0.32168730000000001 * cse890 - 0.51002221000000003 * cse891 - 0.23619984999999999 * cse893 +
				       0.015391166 * cse894 - 22.089523) *
			state.xn[1] +
		    std::pow(10.0, 2.1892372 * cse889 + 0.19191374999999999 * cse890 - 0.16596184 * cse891 + 0.29036281000000003 * cse893 -
				       0.81520437999999995 * cse894 - 23.689236999999999) *
			cse895 +
		    std::pow(10.0, 2.0943374000000001 * cse18 * cse888 + 0.43693353000000001 * cse886 * cse892 - 0.033638326000000003 * cse890 -
				       0.14913216000000001 * cse891 - 0.77151435999999995 * cse894 - 23.962112000000001) *
			state.xn[3];
		const amrex::Real cse897 = std::pow(10, 37.383713 * cse889 + 3.8479610000000002 * cse890 + 20.159831000000001 * cse891 +
							    48.656103000000002 * cse893 + 58.145166000000003 * cse894 - 16.818342000000001) *
					       state.xn[0] +
					   cse896;
		const amrex::Real cse898 = cse881 * std::pow(cse888, 8);
		const amrex::Real cse899 = cse882 * std::pow(cse888, 7);
		const amrex::Real cse900 = cse883 * std::pow(cse888, 6);
		const amrex::Real cse901 = std::pow(10.0, 16.815729999999999 * cse889 + 983.67575999999997 * cse890 + 734.71650999999997 * cse891 +
							      343.1918 * cse893 + 96.743155000000002 * cse894 + 70.609154000000004 * cse898 +
							      364.14445999999998 * cse899 + 801.81246999999996 * cse900 - 21.928795999999998) *
					   state.xn[2];
		const amrex::Real cse902 = std::pow(10.0, 1.6802758 * cse889 - 8.8077017000000009 * cse890 - 4.7274035999999997 * cse891 +
							      4.0406627000000004 * cse893 + 0.93310621999999999 * cse894 - 6.3701156000000001 * cse898 +
							      6.4380698000000001 * cse899 + 8.9167182999999994 * cse900 - 22.921188999999998) *
					   state.xn[2];
		const amrex::Real cse903 = state.T < 1000.0;
		const amrex::Real cse904 = std::pow(10.0, 3.5692468000000002 * cse18 * cse888 - 4.2519023000000002 * cse890 - 21.328264000000001 * cse891 -
							      27.850082 * cse893 - 11.33286 * cse894 - 24.311209000000002) *
					       state.xn[0] +
					   cse896;
		const amrex::Real cse905 = std::pow(10.0, 4.6450521 * cse889 + 1.5538288 * cse890 - 5.5108049000000001 * cse891 + 5.9369081000000001 * cse893 -
							      3.7209846 * cse894 - 24.311209000000002) *
					       state.xn[0] +
					   cse896;
		const amrex::Real cse906 = std::pow(10.0, 2.8156982240432273 * cse18 + 0.049411830040524525 * cse23 + 5.9406293599919788 * cse884 -
							      5.2566362076906961 * cse885 - 1.3586848802337363 * cse886 - 22.089523) *
					       state.xn[1] +
					   std::pow(10.0, 3.7525488682084647 * cse18 - 2.4768712409536775 * cse23 - 0.62120210234156448 * cse884 -
							      1.5370575959566981 * cse885 + 2.5133588394664677 * cse886 - 23.962112000000001) *
					       state.xn[3] +
					   std::pow(10.0, 3.9225864834863131 * cse18 - 2.6171337683481011 * cse23 + 3.5440891133599637 * cse884 -
							      1.7105157385968939 * cse885 + 1.6702447513374001 * cse886 - 23.689236999999999) *
					       cse895 +
					   std::pow(10.0, 8.3228160852326614 * cse18 - 11.945856385319289 * cse23 + 28.69470131298657 * cse884 -
							      56.79810800956944 * cse885 + 34.150687525022562 * cse886 - 24.311209000000002) *
					       state.xn[0];
		const amrex::Real cse907 =
		    ((cse790 && cse877 && cse878 && cse879)
			 ? (cse880 || cse887 + cse897 < 1.0e-100)
			 : (((cse790 && cse878 && cse879)
				 ? (cse880 || cse897 + cse901 < 1.0e-100)
				 : (((cse878 && cse879)
					 ? (cse880 || cse897 + cse902 < 1.0e-100)
					 : (((cse790 && cse877 && cse879 && cse903)
						 ? (cse880 || cse887 + cse904 < 1.0e-100)
						 : (((cse790 && cse879 && cse903)
							 ? (cse880 || cse901 + cse904 < 1.0e-100)
							 : (((cse879 && cse903)
								 ? (cse880 || cse902 + cse904 < 1.0e-100)
								 : (((cse790 && cse877 && cse879)
									 ? (cse880 || cse887 + cse905 < 1.0e-100)
									 : (((cse790 && cse879)
										 ? (cse880 || cse901 + cse905 < 1.0e-100)
										 : (((cse879) ? (cse880 || cse902 + cse905 < 1.0e-100)
											      : (((cse790) ? (cse880 ||
													      std::pow(10.0, 30.129743459482278 * cse18 +
																 310.58441786958315 * cse23 +
																 7500.646099484944 * cse881 +
																 21588.957206746247 * cse882 +
																 26530.843406310876 * cse883 +
																 18165.631967965237 * cse884 +
																 7572.4886742758581 * cse885 +
																 1974.131269262874 * cse886 -
																 21.928795999999998) *
															  state.xn[2] +
														      cse906 <
														  1.0e-100)
													   : (cse880 ||
													      std::pow(10.0, 3.0106500755647456 * cse18 +
																 2.9956460707652877 * cse23 -
																 676.68255490510751 * cse881 +
																 381.69251128040054 * cse882 +
																 295.04162851880631 * cse883 -
																 162.65264843551881 * cse884 -
																 48.723840736518795 * cse885 +
																 23.242975457496804 * cse886 -
																 22.921188999999998) *
															  state.xn[2] +
														      cse906 <
														  1.0e-100))))))))))))))))))));
		const amrex::Real cse908 = std::log(0.001 * ((cse879) ? (state.T) : (6000.0)));
		const amrex::Real cse909 = cse18 * cse908;
		const amrex::Real cse910 = std::pow(cse908, 5);
		const amrex::Real cse911 = cse884 * cse910;
		const amrex::Real cse912 = std::pow(cse908, 4);
		const amrex::Real cse913 = cse885 * cse912;
		const amrex::Real cse914 = std::pow(cse908, 3);
		const amrex::Real cse915 = cse886 * cse914;
		const amrex::Real cse916 = std::pow(cse908, 2);
		const amrex::Real cse917 = cse23 * cse916;
		const amrex::Real cse918 = std::pow(10, 37.383713 * cse909 + 3.8479610000000002 * cse911 + 20.159831000000001 * cse913 +
							    48.656103000000002 * cse915 + 58.145166000000003 * cse917 - 16.818342000000001);
		const amrex::Real cse919 = ((cse879) ? (cse878) : (false));
		const amrex::Real cse920 = std::pow(10.0, 3.5692468000000002 * cse18 * cse908 - 4.2519023000000002 * cse911 - 21.328264000000001 * cse913 -
							      27.850082 * cse915 - 11.33286 * cse917 - 24.311209000000002);
		const amrex::Real cse921 = ((cse879) ? (cse903) : (false));
		const amrex::Real cse922 = std::pow(10.0, 4.6450521 * cse909 + 1.5538288 * cse911 - 5.5108049000000001 * cse913 + 5.9369081000000001 * cse915 -
							      3.7209846 * cse917 - 24.311209000000002);
		const amrex::Real cse923 = ((cse919) ? (cse918) : ((cse921) ? (cse920) : (cse922)));
		const amrex::Real cse924 =
		    std::pow(10.0, 2.0943374000000001 * cse18 * cse908 + 0.43693353000000001 * cse886 * cse914 - 0.033638326000000003 * cse911 -
				       0.14913216000000001 * cse913 - 0.77151435999999995 * cse917 - 23.962112000000001);
		const amrex::Real cse925 = cse924 * state.xn[3];
		const amrex::Real cse926 = std::pow(10.0, 1.5714710999999999 * cse909 + 0.32168730000000001 * cse911 - 0.51002221000000003 * cse913 -
							      0.23619984999999999 * cse915 + 0.015391166 * cse917 - 22.089523);
		const amrex::Real cse927 = cse926 * state.xn[1];
		const amrex::Real cse928 = std::pow(10.0, 2.1892372 * cse909 + 0.19191374999999999 * cse911 - 0.16596184 * cse913 +
							      0.29036281000000003 * cse915 - 0.81520437999999995 * cse917 - 23.689236999999999);
		const amrex::Real cse929 = cse895 * cse928;
		const amrex::Real cse930 = ((cse879) ? (cse877) : (false));
		const amrex::Real cse931 = cse881 * std::pow(cse908, 8);
		const amrex::Real cse932 = std::pow(cse908, 7);
		const amrex::Real cse933 = cse882 * cse932;
		const amrex::Real cse934 = std::pow(cse908, 6);
		const amrex::Real cse935 = cse883 * cse934;
		const amrex::Real cse936 = std::pow(10.0, 16.815729999999999 * ((cse930) ? (-2.9957322735539909 * cse18) : (cse909)) +
							      96.743155000000002 * ((cse930) ? (8.9744118548129634 * cse23) : (cse917)) +
							      70.609154000000004 * ((cse930) ? (6486.7025759648268 * cse881) : (cse931)) +
							      364.14445999999998 * ((cse930) ? (-2165.3145153286073 * cse882) : (cse933)) +
							      801.81246999999996 * ((cse930) ? (722.79974230133178 * cse883) : (cse935)) +
							      983.67575999999997 * ((cse930) ? (-241.2764814406587 * cse884) : (cse911)) +
							      734.71650999999997 * ((cse930) ? (80.540068139807445 * cse885) : (cse913)) +
							      343.1918 * ((cse930) ? (-26.884935229628727 * cse886) : (cse915)) - 21.928795999999998);
		const amrex::Real cse937 = std::pow(10.0, 1.6802758 * cse909 - 8.8077017000000009 * cse911 - 4.7274035999999997 * cse913 +
							      4.0406627000000004 * cse915 + 0.93310621999999999 * cse917 - 6.3701156000000001 * cse931 +
							      6.4380698000000001 * cse933 + 8.9167182999999994 * cse935 - 22.921188999999998);
		const amrex::Real cse938 = ((cse790) ? (cse936) : (cse937));
		const amrex::Real cse939 = cse923 * state.xn[0] + cse925 + cse927 + cse929 + cse938 * state.xn[2];
		const amrex::Real cse940 = std::pow(cse939, -2);
		const amrex::Real cse941 = ((cse879) ? (cse2) : (0.00016666666666666666));
		const amrex::Real cse942 = std::exp(-11700.0 * cse941);
		const amrex::Real cse943 = std::exp(-5860.0 * cse941);
		const amrex::Real cse944 = std::exp(-510.0 * cse941);
		const amrex::Real cse945 = std::exp(-2197000.0 * ((cse879) ? (std::pow(state.T, -3)) : (4.6296296296296294e-12)));
		const amrex::Real cse946 = cse945 * ((cse879) ? (std::pow(state.T, 3.7599999999999998)) : (160638152785302.53));
		const amrex::Real cse947 = 6.0142468035272636e-8 * ((cse879) ? (std::pow(state.T, 2.1000000000000001)) : (85924621.118090734)) + 1.0;
		const amrex::Real cse948 = 1.0 / cse947;
		const amrex::Real cse949 = 1.6e-18 * cse942 + 6.7e-19 * cse943 + 3.0e-24 * cse944 + 4.985670872372847e-33 * cse946 * cse948;
		const amrex::Real cse950 = 1.0 / cse939;
		const amrex::Real cse951 = cse949 * cse950 + 1;
		const amrex::Real cse952 = std::pow(cse951, -2);
		const amrex::Real cse953 = cse940 * std::pow(cse949, 2) * cse952;
		const amrex::Real cse954 = 0.0060000000000000001 * cse786;
		const amrex::Real cse955 = std::pow(cse785, 0.36899999999999999 - cse954);
		const amrex::Real cse956 = std::pow(state.T, 0.92600000000000005);
		const amrex::Real cse957 = 1.08e-14 * cse956;
		const amrex::Real cse958 = 0.16 * cse786;
		const amrex::Real cse959 = std::pow(cse785, 0.755 - cse958);
		const amrex::Real cse960 = 4.34e-10 * cse959;
		const amrex::Real cse961 = cse960 * state.xn[0];
		const amrex::Real cse962 = 0.042999999999999997 * cse786;
		const amrex::Real cse963 = 5.2499999999999998e-13 * std::pow(cse785, cse962 + 0.88900000000000001);
		const amrex::Real cse964 = 0.52500000000000002 * cse786;
		const amrex::Real cse965 = 2.2500000000000003e-12 * std::pow(cse785, cse964 + 1.198);
		const amrex::Real cse966 = cse963 + cse965;
		const amrex::Real cse967 = cse957 * state.xn[2] + cse961 + cse966 * state.xn[3];
		const amrex::Real cse968 = std::exp(-425.40000000000003 * cse2);
		const amrex::Real cse969 = cse967 * cse968;
		const amrex::Real cse970 = std::exp(-326.60000000000002 * cse2);
		const amrex::Real cse971 = std::pow(state.T, -0.025999999999999999);
		const amrex::Real cse972 = 4.8599999999999998e-10 * cse971;
		const amrex::Real cse973 = 3.1899999999999998e-10 * cse955;
		const amrex::Real cse974 = cse973 * state.xn[0];
		const amrex::Real cse975 = 0.041000000000000002 * cse786;
		const amrex::Real cse976 = 4.7499999999999998e-11 * std::pow(cse785, cse975 + 0.20300000000000001);
		const amrex::Real cse977 = 0.058000000000000003 * cse786;
		const amrex::Real cse978 = 1.6725000000000002e-10 * std::pow(cse785, cse977 + 0.23699999999999999);
		const amrex::Real cse979 = cse976 + cse978;
		const amrex::Real cse980 = cse972 * state.xn[2] + cse974 + cse979 * state.xn[3];
		const amrex::Real cse981 = cse970 * cse980;
		const amrex::Real cse982 = 0.0030000000000000001 * cse786;
		const amrex::Real cse983 = std::pow(cse785, 0.41899999999999998 - cse982);
		const amrex::Real cse984 = 7.1400000000000007e-11 * cse983;
		const amrex::Real cse985 = cse967 * cse970;
		const amrex::Real cse986 = 8.68e-11 * cse959;
		const amrex::Real cse987 = std::pow(state.T, -0.074999999999999997);
		const amrex::Real cse988 = 5.1199999999999999e-10 * cse987;
		const amrex::Real cse989 = 3.5700000000000001e-10 * cse983;
		const amrex::Real cse990 = cse989 * state.xn[0];
		const amrex::Real cse991 = 0.025000000000000001 * cse786;
		const amrex::Real cse992 = 3.7250000000000001e-11 * std::pow(cse785, cse991 + 0.26400000000000001);
		const amrex::Real cse993 = 1.0274999999999999e-10 * std::pow(cse785, cse962 + 0.29599999999999999);
		const amrex::Real cse994 = cse992 + cse993;
		const amrex::Real cse995 = cse988 * state.xn[2] + cse990 + cse994 * state.xn[3];
		const amrex::Real cse996 = cse970 * cse995;
		const amrex::Real cse997 = cse968 * cse980;
		const amrex::Real cse998 = 6.3800000000000002e-11 * cse955;
		const amrex::Real cse999 = cse995 + 8.9099999999999997e-5;
		const amrex::Real cse1000 = cse970 * cse999;
		const amrex::Real cse1001 = cse1000 * cse998 + 2.1266666666666666e-11 * cse955 * cse969 + 2.8933333333333332e-11 * cse959 * cse997 +
					    cse981 * cse984 + cse984 * cse985 + cse986 * cse996;
		const amrex::Real cse1002 = cse967 + 1.7499999999999998e-5;
		const amrex::Real cse1003 = cse980 + 1.34e-10;
		const amrex::Real cse1004 = std::exp(-98.800000000000011 * cse2);
		const amrex::Real cse1005 = cse1004 * cse967;
		const amrex::Real cse1006 = (1.0 / 3.0) * cse1005;
		const amrex::Real cse1007 = std::exp(-227.80000000000001 * cse2);
		const amrex::Real cse1008 = cse1007 * cse995;
		const amrex::Real cse1009 = (3.0 / 5.0) * cse1008;
		const amrex::Real cse1010 = (1.0 / 5.0) * cse970;
		const amrex::Real cse1011 = cse1010 * cse980;
		const amrex::Real cse1012 = cse1002 * cse1009 + cse1002 * cse1011 + cse1003 * cse1009;
		const amrex::Real cse1013 = cse1010 * cse995;
		const amrex::Real cse1014 = (1.0 / 15.0) * cse980;
		const amrex::Real cse1015 = cse1011 * cse999 + cse1013 * cse967 + cse1014 * cse969;
		const amrex::Real cse1016 = cse1002 * cse999 + cse1003 * cse1006 + cse1003 * cse999 + cse1012 + cse1015;
		const amrex::Real cse1017 = 1.0 / cse1016;
		const amrex::Real cse1018 = 2.3872025442750958e-19 * cse1017;
		const amrex::Real cse1019 = cse1002 * cse970;
		const amrex::Real cse1020 = 2.1419999999999999e-10 * cse983;
		const amrex::Real cse1021 = cse1003 * cse1007;
		const amrex::Real cse1022 = cse1002 * cse1007;
		const amrex::Real cse1023 = 1.9139999999999998e-10 * cse1008 * cse955 + 2.604e-10 * cse1008 * cse959 + cse1019 * cse998 + cse1020 * cse1021 +
					    cse1020 * cse1022 + cse981 * cse986;
		const amrex::Real cse1024 = 2.8023005140019997e-18 * cse1017;
		const amrex::Real cse1025 = std::pow(cse1016, -2);
		const amrex::Real cse1026 = cse1003 * cse1004;
		const amrex::Real cse1027 = cse1025 * (-cse1001 - cse1002 * cse989 - cse1003 * cse989 - 1.0633333333333333e-10 * cse1005 * cse955 - cse1023 -
						       1.4466666666666666e-10 * cse1026 * cse959 - cse960 * cse999 - cse973 * cse999);
		const amrex::Real cse1028 = 2.3872025442750958e-19 * cse1015;
		const amrex::Real cse1029 = 2.8023005140019997e-18 * cse1012;
		const amrex::Real cse1030 = 0.045999999999999999 * cse786;
		const amrex::Real cse1031 = std::pow(cse785, cse1030 + 0.22800000000000001);
		const amrex::Real cse1032 = std::pow(cse785, cse1030 + 0.23100000000000001);
		const amrex::Real cse1033 = 2.6400000000000002e-10 * cse1032;
		const amrex::Real cse1034 = cse1033 * state.xn[0];
		const amrex::Real cse1035 = 0.031 * cse786;
		const amrex::Real cse1036 = 1.11e-10 * std::pow(cse785, cse1035 + 0.26300000000000001);
		const amrex::Real cse1037 = 0.064000000000000001 * cse786;
		const amrex::Real cse1038 = 4.3749999999999998e-11 * std::pow(cse785, cse1037 + 0.071999999999999995);
		const amrex::Real cse1039 = cse1036 + cse1038;
		const amrex::Real cse1040 = 0.00235272 * cse123;
		const amrex::Real cse1041 = cse1040 - 0.041816600000000002;
		const amrex::Real cse1042 = cse1041 * cse123 + 0.35826400000000003;
		const amrex::Real cse1043 = cse1042 * cse123 - 0.57443;
		const amrex::Real cse1044 = 0.097857299999999994 * cse123;
		const amrex::Real cse1045 = cse1044 - 3.1926800000000002;
		const amrex::Real cse1046 = cse1045 * cse123 + 38.504899999999999;
		const amrex::Real cse1047 = cse1046 * cse123 - 202.19300000000001;
		const amrex::Real cse1048 = std::exp(((cse903) ? (cse1043 * cse123 - 7.4386999999999999) : (cse1047 * cse123 + 386.18599999999998)));
		const amrex::Real cse1049 = 1.7258000000000001e-6 * cse784;
		const amrex::Real cse1050 = cse1048 * cse1049;
		const amrex::Real cse1051 = cse1034 + cse1039 * state.xn[3] + cse1050;
		const amrex::Real cse1052 = std::exp(-101.30000000000001 * cse2);
		const amrex::Real cse1053 = cse1051 * cse1052;
		const amrex::Real cse1054 = 8.9000000000000003e-11 * cse1031;
		const amrex::Real cse1055 = cse1054 * state.xn[0];
		const amrex::Real cse1056 = 0.037999999999999999 * cse786;
		const amrex::Real cse1057 = 5.1749999999999996e-11 * std::pow(cse785, cse1056 + 0.16900000000000001);
		const amrex::Real cse1058 = 0.048000000000000001 * cse786;
		const amrex::Real cse1059 = 2.15e-11 * std::pow(cse785, cse1058 - 0.01);
		const amrex::Real cse1060 = cse1057 + cse1059;
		const amrex::Real cse1061 = 0.0070527699999999999 * cse123;
		const amrex::Real cse1062 = cse1061 - 0.11133800000000001;
		const amrex::Real cse1063 = cse1062 * cse123 + 0.69763799999999998;
		const amrex::Real cse1064 = cse1063 * cse123 - 1.3074300000000001;
		const amrex::Real cse1065 = 0.093813800000000003 * cse123;
		const amrex::Real cse1066 = cse1065 - 3.0328300000000001;
		const amrex::Real cse1067 = cse1066 * cse123 + 36.180300000000003;
		const amrex::Real cse1068 = cse1067 * cse123 - 187.47399999999999;
		const amrex::Real cse1069 = std::exp(((cse903) ? (cse1064 * cse123 - 7.6973500000000001) : (cse1068 * cse123 + 350.60899999999998)));
		const amrex::Real cse1070 = cse1049 * cse1069;
		const amrex::Real cse1071 = cse1055 + cse1060 * state.xn[3] + cse1070;
		const amrex::Real cse1072 = cse1052 * cse1071;
		const amrex::Real cse1073 = 0.057000000000000002 * cse786;
		const amrex::Real cse1074 = std::pow(cse785, cse1073 + 0.115);
		const amrex::Real cse1075 = 1.26e-10 * cse1074;
		const amrex::Real cse1076 = cse1075 * state.xn[0];
		const amrex::Real cse1077 = 0.049000000000000002 * cse786;
		const amrex::Real cse1078 = 5.3249999999999997e-11 * std::pow(cse785, cse1077 - 0.0040000000000000001);
		const amrex::Real cse1079 = 0.10199999999999999 * cse786;
		const amrex::Real cse1080 = 1.675e-11 * std::pow(cse785, cse1079 - 0.085000000000000006);
		const amrex::Real cse1081 = cse1078 + cse1080;
		const amrex::Real cse1082 = 0.00065632500000000003 * cse123;
		const amrex::Real cse1083 = -cse1082 - 0.015089200000000001;
		const amrex::Real cse1084 = cse1083 * cse123 + 0.36118400000000001;
		const amrex::Real cse1085 = cse1084 * cse123 - 0.77378199999999997;
		const amrex::Real cse1086 = 0.10508000000000001 * cse123;
		const amrex::Real cse1087 = cse1086 - 3.4762;
		const amrex::Real cse1088 = cse1087 * cse123 + 42.594999999999999;
		const amrex::Real cse1089 = cse1088 * cse123 - 227.91300000000001;
		const amrex::Real cse1090 = std::exp(((cse903) ? (cse1085 * cse123 - 9.2514099999999999) : (cse1089 * cse123 + 444.60000000000002)));
		const amrex::Real cse1091 = 2.8763333333333333e-6 * cse1090;
		const amrex::Real cse1092 = cse1091 * cse784;
		const amrex::Real cse1093 = cse1076 + cse1081 * state.xn[3] + cse1092;
		const amrex::Real cse1094 = std::exp(-62.460000000000001 * cse2);
		const amrex::Real cse1095 = cse1093 * cse1094;
		const amrex::Real cse1096 = 1.32e-9 * cse1032;
		const amrex::Real cse1097 = 6.3e-10 * cse1074;
		const amrex::Real cse1098 = cse1071 * cse1094;
		const amrex::Real cse1099 = cse1051 * cse1094;
		const amrex::Real cse1100 = cse1093 + 7.8800000000000004e-8;
		const amrex::Real cse1101 = cse1094 * cse1100;
		const amrex::Real cse1102 = 4.4500000000000001e-10 * cse1031;
		const amrex::Real cse1103 = 7.4166666666666679e-10 * cse1031 * cse1053 + 2.2000000000000003e-9 * cse1032 * cse1072 + cse1095 * cse1096 +
					    cse1097 * cse1098 + cse1097 * cse1099 + cse1101 * cse1102;
		const amrex::Real cse1104 = cse1051 + 2.65e-7;
		const amrex::Real cse1105 = cse1071 + 1.81e-14;
		const amrex::Real cse1106 = std::exp(-38.840000000000003 * cse2);
		const amrex::Real cse1107 = cse1051 * cse1106;
		const amrex::Real cse1108 = (5.0 / 3.0) * cse1107;
		const amrex::Real cse1109 = (25.0 / 3.0) * cse1051;
		const amrex::Real cse1110 = 5 * cse1094;
		const amrex::Real cse1111 = cse1093 * cse1110;
		const amrex::Real cse1112 = cse1071 * cse1110;
		const amrex::Real cse1113 = cse1051 * cse1111 + cse1072 * cse1109 + cse1100 * cse1112;
		const amrex::Real cse1114 = std::exp(-23.619999999999997 * cse2);
		const amrex::Real cse1115 = cse1093 * cse1114;
		const amrex::Real cse1116 = 3 * cse1115;
		const amrex::Real cse1117 = cse1104 * cse1112 + cse1104 * cse1116 + cse1105 * cse1116;
		const amrex::Real cse1118 = cse1100 * cse1104 + cse1100 * cse1105 + cse1105 * cse1108 + cse1113 + cse1117;
		const amrex::Real cse1119 = 1.0 / cse1118;
		const amrex::Real cse1120 = 1.4210469458259591e-21 * cse1119;
		const amrex::Real cse1121 = cse1094 * cse1104;
		const amrex::Real cse1122 = 3.7800000000000004e-10 * cse1074;
		const amrex::Real cse1123 = cse1105 * cse1114;
		const amrex::Real cse1124 = cse1104 * cse1114;
		const amrex::Real cse1125 = 2.6700000000000001e-10 * cse1031 * cse1115 + 7.9200000000000006e-10 * cse1032 * cse1115 + cse1096 * cse1098 +
					    cse1102 * cse1121 + cse1122 * cse1123 + cse1122 * cse1124;
		const amrex::Real cse1126 = 2.5697412351439998e-22 * cse1119;
		const amrex::Real cse1127 = std::pow(cse1118, -2);
		const amrex::Real cse1128 = cse1105 * cse1106;
		const amrex::Real cse1129 = cse1127 * (-1.4833333333333334e-10 * cse1031 * cse1107 - 4.4000000000000003e-10 * cse1032 * cse1128 -
						       cse1033 * cse1100 - cse1054 * cse1100 - cse1075 * cse1104 - cse1075 * cse1105 - cse1103 - cse1125);
		const amrex::Real cse1130 = 1.4210469458259591e-21 * cse1113;
		const amrex::Real cse1131 = 2.5697412351439998e-22 * cse1117;
		const amrex::Real cse1132 = state.T < 1000000.0;
		const amrex::Real cse1133 = cse734 * std::pow(state.T, -1.8500000000000001);
		const amrex::Real cse1134 = cse110 * cse737;
		const amrex::Real cse1135 = std::pow(state.T, -0.84999999999999998);
		const amrex::Real cse1136 = std::pow(state.T, -0.19999999999999998);
		const amrex::Real cse1137 = std::pow(cse727, -2);
		const amrex::Real cse1138 = cse1135 * cse729;
		const amrex::Real cse1139 = std::pow(cse731, -2);
		const amrex::Real cse1140 = cse1139 * cse733 * (2.5500935934394033e-12 * cse1136 * cse1137 * state.xn[2] - 6.6690584756579352e-10 * cse1138);
		const amrex::Real cse1141 = cse726 * cse728;
		const amrex::Real cse1142 = 1.3659996292638161e-10 * cse1141;
		const amrex::Real cse1143 = std::pow(cse711, -2);
		const amrex::Real cse1144 = std::pow(cse709, 0.12299999999999994);
		const amrex::Real cse1145 = cse1144 * cse119;
		const amrex::Real cse1146 = std::pow(cse709, -0.56899999999999995) * cse710;
		const amrex::Real cse1147 = cse1146 * cse119;
		const amrex::Real cse1148 = cse119 * cse55 * cse714;
		const amrex::Real cse1149 = 1.0 / cse717;
		const amrex::Real cse1150 = cse1149 * cse123 / std::pow(cse718, 2);
		const amrex::Real cse1151 = cse135 * cse723;
		const amrex::Real cse1152 = cse1151 * state.xn[2];
		const amrex::Real cse1153 = cse1149 * cse718;
		const amrex::Real cse1154 = cse1149 * cse721;
		const amrex::Real cse1155 = cse110 * cse1142 * cse734 / std::pow(cse736, 2);
		const amrex::Real cse1156 = cse1 * cse843;
		const amrex::Real cse1157 = cse2 * cse784;
		const amrex::Real cse1158 = std::pow(state.T, -0.92999999999999994);
		const amrex::Real cse1159 = 100 * cse2;
		const amrex::Real cse1160 = -2.2649999999999999e-6 * cse1157 +
					    cse789 * (cse1159 * (8.7000000000000001e-5 * cse786 + 0.001281) + cse2 * cse787) +
					    state.xn[3] * (0.75 * ((cse790) ? (1.1000000000000001e-13) : (2.6233044951750002e-11 * cse1158)) +
							   0.25 * ((cse790) ? (3.3000000000000001e-13) : (2.7229810044920004e-11 * cse1158)));
		const amrex::Real cse1161 = 5.7927337833400002e-20 * cse799;
		const amrex::Real cse1162 = cse1161 * cse800;
		const amrex::Real cse1163 =
		    0.4375 * state.xn[0] / std::pow(state.T, 3.0 / 4.0) - 170.02500000000001 * state.xn[2] / std::pow(state.T, 5.0 / 4.0);
		const amrex::Real cse1164 = -cse1160;
		const amrex::Real cse1165 = cse1161 * cse842 * state.xn[8];
		const amrex::Real cse1166 = std::pow(cse768, -3);
		const amrex::Real cse1167 = -cse17 * cse773 * (0.0040460000000000001 * cse1166 + 0.0036709999999999998 * cse769) -
					    cse17 * cse776 * (0.0047800000000000004 * cse1166 - 5.5500000000000001e-5 * cse769);
		const amrex::Real cse1168 = 2.0187430200000004e-26 * cse243;
		const amrex::Real cse1169 = 2.5634832e-11 * cse568 * cse845 * state.xn[3];
		const amrex::Real cse1170 = 2 * cse842;
		const amrex::Real cse1171 = l0_co_interp1d_partial_0(state.T);
		const amrex::Real cse1172 = std::pow(cse808, -2);
		const amrex::Real cse1173 = ((cse815) ? (-2.872276598720128e-13 * cse55 * cse812 * state.xn[9] *
							 (((5.7445531974402559e-18 * cse54 * cse812 - gradv < 0)
							       ? (0)
							       : ((5.7445531974402559e-18 * cse54 * cse812 - gradv == 0) ? (1.0 / 2.0) : (1)))) /
							 std::pow(cse813, 2))
						      : (0));
		const amrex::Real cse1174 = cse872 * (-cse1173 * cse873 - llte_co_interp2d_partial_0(state.T, cse819));
		const amrex::Real cse1175 = cse1173 * cse868 + nhalf_co_interp2d_partial_0(state.T, cse819);
		const amrex::Real cse1176 = ((cse879) ? (-cse1) : (0));
		const amrex::Real cse1177 =
		    -1.8719999999999998e-14 * cse1176 * cse942 - 3.9261999999999998e-15 * cse1176 * cse943 - 1.53e-21 * cse1176 * cse944 +
		    4.985670872372847e-33 * cse945 * cse948 * ((cse879) ? (3.7599999999999998 * std::pow(state.T, 2.7599999999999998)) : (0)) -
		    1.0953518906603145e-26 * cse946 * cse948 * ((cse879) ? (-3 / std::pow(state.T, 4)) : (0)) -
		    2.9985055107607381e-40 * cse946 * ((cse879) ? (2.1000000000000001 * std::pow(state.T, 1.1000000000000001)) : (0)) / std::pow(cse947, 2);
		const amrex::Real cse1178 = 1.0 / cse951;
		const amrex::Real cse1179 = ((cse879) ? (1) : (0));
		const amrex::Real cse1180 = cse1179 * cse941;
		const amrex::Real cse1181 = cse1180 * cse23 * cse908;
		const amrex::Real cse1182 = cse1180 * cse884 * cse912;
		const amrex::Real cse1183 = cse1180 * cse885 * cse914;
		const amrex::Real cse1184 = cse1180 * cse18;
		const amrex::Real cse1185 = cse1180 * cse886 * cse916;
		const amrex::Real cse1186 = cse1180 * cse881 * cse932;
		const amrex::Real cse1187 = cse1180 * cse882 * cse934;
		const amrex::Real cse1188 = cse1180 * cse883 * cse910;
		const amrex::Real cse1189 = 65.320000000000007 * cse981;
		const amrex::Real cse1190 = cse2 * cse962;
		const amrex::Real cse1191 = 0.00042999999999999999 * cse786;
		const amrex::Real cse1192 = cse990 * (cse1159 * (0.0041900000000000001 - 3.0000000000000001e-5 * cse786) - cse2 * cse982) -
					    3.8399999999999998e-11 * std::pow(state.T, -1.075) * state.xn[2] +
					    state.xn[3] * (cse992 * (cse1159 * (0.00025000000000000001 * cse786 + 0.00264) + cse2 * cse991) +
							   cse993 * (cse1159 * (cse1191 + 0.00296) + cse1190));
		const amrex::Real cse1193 = cse1010 * cse967;
		const amrex::Real cse1194 = cse974 * (cse1159 * (0.0036900000000000001 - 6.0000000000000002e-5 * cse786) - cse2 * cse954) -
					    1.2635999999999999e-11 * std::pow(state.T, -1.026) * state.xn[2] +
					    state.xn[3] * (cse976 * (cse1159 * (0.00041000000000000005 * cse786 + 0.0020300000000000001) + cse2 * cse975) +
							   cse978 * (cse1159 * (0.00058 * cse786 + 0.0023700000000000001) + cse2 * cse977));
		const amrex::Real cse1195 = (1.0 / 15.0) * cse969;
		const amrex::Real cse1196 = cse961 * (cse1159 * (0.0075500000000000003 - 0.0016000000000000001 * cse786) - cse2 * cse958) +
					    1.0000800000000001e-14 * std::pow(state.T, -0.073999999999999955) * state.xn[2] +
					    state.xn[3] * (cse963 * (cse1159 * (cse1191 + 0.0088900000000000003) + cse1190) +
							   cse965 * (cse1159 * (0.0052500000000000003 * cse786 + 0.011979999999999999) + cse2 * cse964));
		const amrex::Real cse1197 = cse1014 * cse968;
		const amrex::Real cse1198 = cse1194 * cse999;
		const amrex::Real cse1199 = cse1 * cse1189 * cse999 + 65.320000000000007 * cse1 * cse967 * cse996 +
					    28.360000000000003 * cse1 * cse969 * cse980 + cse1010 * cse1198 + cse1011 * cse1192 + cse1013 * cse1196 +
					    cse1192 * cse1193 + cse1194 * cse1195 + cse1196 * cse1197;
		const amrex::Real cse1200 = 136.68000000000001 * cse1008;
		const amrex::Real cse1201 = cse1 * cse1003;
		const amrex::Real cse1202 = cse1 * cse1002;
		const amrex::Real cse1203 = cse1003 * cse1192;
		const amrex::Real cse1204 = (3.0 / 5.0) * cse1007;
		const amrex::Real cse1205 = cse1002 * cse1192;
		const amrex::Real cse1206 = cse1002 * cse1010;
		const amrex::Real cse1207 = cse1009 * cse1194 + cse1009 * cse1196 + cse1011 * cse1196 + cse1189 * cse1202 + cse1194 * cse1206 +
					    cse1200 * cse1201 + cse1200 * cse1202 + cse1203 * cse1204 + cse1204 * cse1205;
		const amrex::Real cse1208 = (1.0 / 3.0) * cse1026;
		const amrex::Real cse1209 = cse1025 * (-32.933333333333337 * cse1005 * cse1201 - cse1006 * cse1194 - cse1196 * cse1208 - cse1196 * cse999 -
						       cse1198 - cse1199 - cse1203 - cse1205 - cse1207);
		const amrex::Real cse1210 = 312.30000000000001 * cse1;
		const amrex::Real cse1211 = cse1030 * cse2;
		const amrex::Real cse1212 = 0.00046000000000000001 * cse786;
		const amrex::Real cse1213 = 8.6290000000000003e-7 * cse1157;
		const amrex::Real cse1214 = cse1034 * (cse1159 * (cse1212 + 0.00231) + cse1211) - cse1048 * cse1213 +
					    cse1050 * ((cse903) ? (cse1043 * cse2 + cse123 * (cse1042 * cse2 + cse123 * (cse1040 * cse2 + cse1041 * cse2)))
								: (cse1047 * cse2 + cse123 * (cse1046 * cse2 + cse123 * (cse1044 * cse2 + cse1045 * cse2)))) +
					    state.xn[3] * (cse1036 * (cse1035 * cse2 + cse1159 * (0.00031 * cse786 + 0.00263)) +
							   cse1038 * (cse1037 * cse2 + cse1159 * (0.00064000000000000005 * cse786 + 0.00071999999999999994)));
		const amrex::Real cse1215 = (25.0 / 3.0) * cse1072;
		const amrex::Real cse1216 = cse1076 * (cse1073 * cse2 + cse1159 * (0.00057000000000000009 * cse786 + 0.00115)) -
					    1.4381666666666666e-6 * cse1090 * cse1157 +
					    cse1092 * ((cse903) ? (cse1085 * cse2 + cse123 * (cse1084 * cse2 + cse123 * (-cse1082 * cse2 + cse1083 * cse2)))
								: (cse1089 * cse2 + cse123 * (cse1088 * cse2 + cse123 * (cse1086 * cse2 + cse1087 * cse2)))) +
					    state.xn[3] * (cse1078 * (cse1077 * cse2 + cse1159 * (0.00048999999999999998 * cse786 - 4.0000000000000003e-5)) +
							   cse1080 * (cse1079 * cse2 + cse1159 * (0.0010199999999999999 * cse786 - 0.00085000000000000006)));
		const amrex::Real cse1217 = cse1055 * (cse1159 * (cse1212 + 0.0022800000000000003) + cse1211) - cse1069 * cse1213 +
					    cse1070 * ((cse903) ? (cse1064 * cse2 + cse123 * (cse1063 * cse2 + cse123 * (cse1061 * cse2 + cse1062 * cse2)))
								: (cse1068 * cse2 + cse123 * (cse1067 * cse2 + cse123 * (cse1065 * cse2 + cse1066 * cse2)))) +
					    state.xn[3] * (cse1057 * (cse1056 * cse2 + cse1159 * (0.00038000000000000002 * cse786 + 0.0016900000000000001)) +
							   cse1059 * (cse1058 * cse2 + cse1159 * (0.00048000000000000001 * cse786 - 0.0001)));
		const amrex::Real cse1218 = cse1052 * cse1109;
		const amrex::Real cse1219 = cse1051 * cse1110;
		const amrex::Real cse1220 = cse1100 * cse1217;
		const amrex::Real cse1221 = 844.16666666666686 * cse1 * cse1051 * cse1072 + cse1051 * cse1095 * cse1210 + cse1071 * cse1101 * cse1210 +
					    cse1110 * cse1220 + cse1111 * cse1214 + cse1112 * cse1216 + cse1214 * cse1215 + cse1216 * cse1219 +
					    cse1217 * cse1218;
		const amrex::Real cse1222 = 70.859999999999985 * cse1 * cse1115;
		const amrex::Real cse1223 = cse1105 * cse1216;
		const amrex::Real cse1224 = 3 * cse1114;
		const amrex::Real cse1225 = cse1104 * cse1110;
		const amrex::Real cse1226 = cse1104 * cse1216;
		const amrex::Real cse1227 = cse1098 * cse1104 * cse1210 + cse1104 * cse1222 + cse1105 * cse1222 + cse1112 * cse1214 + cse1116 * cse1214 +
					    cse1116 * cse1217 + cse1217 * cse1225 + cse1223 * cse1224 + cse1224 * cse1226;
		const amrex::Real cse1228 = (5.0 / 3.0) * cse1128;
		const amrex::Real cse1229 = cse1127 * (-64.733333333333348 * cse1 * cse1105 * cse1107 - cse1100 * cse1214 - cse1108 * cse1217 -
						       cse1214 * cse1228 - cse1220 - cse1221 - cse1223 - cse1226 - cse1227);
		const amrex::Real cse1230 =
		    cse0 *
		    (5.6705896183685846e-35 * state.c_hat * cse1143 * cse118 * cse20 * cse708 * d2g *
			 (-4.2149769358107055e-15 * cse1145 * cse83 - 6.1051722655411154e-13 * cse1147 * cse83) +
		     1.8755475162754094e-35 * state.c_hat * cse118 * cse20 * cse712 * d2g * std::pow(state.T, -0.85299999999999998) -
		     2.0489994438957239e-11 * cse110 * cse1135 * cse728 * cse738 - 2.1789607199999999e-11 * cse111 -
		     1.6173435610483583e-5 * cse1133 * cse1134 * cse728 - cse1134 * cse1140 * cse1142 +
		     7.8348995947646122e-14 * cse1136 * cse1137 * cse732 * cse733 * cse737 * state.xn[0] * state.xn[2] -
		     cse1152 * (-6.3340486037177084e-11 * cse1148 * cse1150 + cse2 * cse719) -
		     cse1152 * (-1.4428815283938862e-12 * cse1148 * cse1153 + 2.835294809184292e-11 * cse1148 * cse1154) -
		     cse1155 * (-0.0015792330470357991 * cse1133 * cse729 + 7.6502807803182096e-12 * cse1136 * cse1137 * cse732 * cse733 * state.xn[2] -
				2.0007175426973804e-9 * cse1138 * cse734 - 1.3338116951315871e-8 * cse1140 * cse730) -
		     5.2835524837844135e-18 * cse1156 * cse799 * state.xn[8] - cse1160 * cse1162 - cse1163 * cse834 - cse1164 * cse1165 -
		     cse1167 * cse1168 * cse845 * cse850 - cse1167 * cse1169 * cse851 - 7.1777529600000012e-12 * cse548 - 7.1777529600000012e-12 * cse549 -
		     cse741 * cse753 * psigd_coll_interp2d_partial_1(cse20, state.T) - cse748 * (1.6000000000000001e-32 * cse55 * cse743 * cse744 + cse746) -
		     1000 * cse805 * std::pow(cse806, 999) * cse832 * cse875 - cse844 * (-182.41999999999999 * cse1156 - cse1160 * cse797 - cse1164 * cse1170) -
		     cse876 * (-cse1163 * cse821 + cse1171 * cse1172 * cse824 * cse828 + cse1171 * cse1172 - cse1174 * cse811 -
			       cse829 * (-cse1171 * cse821 * cse822 - cse1174 * cse808 * cse822 - cse1175 * cse823) -
			       cse830 * (cse865 * (cse1173 * cse864 + alpha_co_interp2d_partial_0(state.T, cse819)) +
					 cse871 * (cse1163 * cse825 - cse1175 * cse867))) -
		     state.xn[11] * (cse1018 * cse1199 + cse1024 * cse1207 + cse1028 * cse1209 + cse1029 * cse1209) -
		     state.xn[3] *
			 ((cse907)
			      ? (0)
			      : (cse1177 * cse1178 +
				 cse949 * cse952 *
				     (-cse1177 * cse950 -
				      cse940 * cse949 *
					  (-cse925 * (4.8223900769399082 * cse1179 * cse18 * cse941 + 3.0182298984217999 * cse1179 * cse886 * cse916 * cse941 -
						      3.5529549287336835 * cse1181 - 0.3872755400043702 * cse1182 - 1.3735579540080118 * cse1183) -
					   cse927 * (0.0708789387907936 * cse1181 + 3.7035619079275177 * cse1182 - 4.6974781513675152 * cse1183 +
						     3.6184459289309552 * cse1184 - 1.6316107607322889 * cse1185) -
					   cse929 * (-3.7541549062629067 * cse1181 + 2.2094886994529301 * cse1182 - 1.528565035159452 * cse1183 +
						     5.0409049417480247 * cse1184 + 2.0057552335975877 * cse1185) -
					   state.xn[0] * ((cse919) ? (cse17 * cse918 *
								      (116.29033200000001 * cse1181 + 19.239805 * cse1182 + 80.639324000000002 * cse1183 +
								       37.383713 * cse1184 + 145.968309 * cse1185))
								   : ((cse921) ? (cse920 * (8.2184944748967013 * cse1179 * cse18 * cse941 -
											    52.189748993977005 * cse1181 - 48.951834264235494 * cse1182 -
											    196.44057098336626 * cse1183 - 192.38155095558542 * cse1185))
									       : (cse922 * (-17.135767342440825 * cse1181 + 17.889115159724135 * cse1182 -
											    50.756388852554174 * cse1183 + 10.695627721640689 * cse1184 +
											    41.010708268606813 * cse1185)))) -
					   state.xn[2] * ((cse790) ? (cse936 * (222.75934655221241 * ((cse930) ? (0) : (2.0 * cse1181)) +
										2264.9971413155886 * ((cse930) ? (0) : (5.0 * cse1182)) +
										1691.7472835026108 * ((cse930) ? (0) : (4.0 * cse1183)) +
										38.719649225812766 * ((cse930) ? (0) : (1.0 * cse1184)) +
										790.22832271779396 * ((cse930) ? (0) : (3.0 * cse1185)) +
										162.58358542932092 * ((cse930) ? (0) : (8.0 * cse1186)) +
										838.47360529236653 * ((cse930) ? (0) : (7.0 * cse1187)) +
										1846.2414407987355 * ((cse930) ? (0) : (6.0 * cse1188))))
								   : (cse937 * (4.297112944704045 * cse1181 - 101.40241318979159 * cse1182 -
										43.540996231705549 * cse1183 + 3.8689780091986448 * cse1184 +
										27.91190909651122 * cse1185 - 117.34186576967058 * cse1186 +
										103.76942484394611 * cse1187 + 123.18901581604325 * cse1188))))))) -
		     state.xn[7] * ((cse1132) ? (cse1120 * cse1221 + cse1126 * cse1227 + cse1130 * cse1229 + cse1131 * cse1229) : (0)));
		const amrex::Real cse1231 = cse1230 * state.T;
		const amrex::Real cse1232 = 2.0709734999999996e-16 * cse1231;
		const amrex::Real cse1233 = 5.6705896183685846e-35 * cse119;
		const amrex::Real cse1234 = cse708 * cse712;
		const amrex::Real cse1235 = cse1234 * d2g;
		const amrex::Real cse1236 = 4.22974728e-11 * cse182 * cse20 * cse705;
		const amrex::Real cse1237 = 3.2043540000000001e-12 * crate;
		const amrex::Real cse1238 = 1.7623947000000002e-12 * crate * cse835 * cse840 * state.xn[5];
		const amrex::Real cse1239 = cse157 * cse778 * cse779 - cse849 * cse851;
		const amrex::Real cse1240 = cse1168 * cse20 * cse845;
		const amrex::Real cse1241 = 1.7623947000000002e-12 * crate * cse853 * state.xn[5];
		const amrex::Real cse1242 = cse825 * cse827;
		const amrex::Real cse1243 = -cse1232;
		const amrex::Real cse1244 =
		    cse1169 * cse1239 + cse1233 * cse1235 + cse1236 * cse856 + cse1237 * cse858 + cse1238 * cse179 + cse1239 * cse1240 +
		    cse1241 * (2.0 * cse21 * cse766 * state.xn[3] + 26.399999999999999 * cse703 * cse705 * cse759 * cse82 * state.xn[0] - cse854 - cse860) +
		    cse1243 - 7.1777529600000012e-12 * cse198 + 3.0000000000000002e-15 * cse243 * cse783 - 7.1777529600000012e-12 * cse558 - cse725 - cse750 -
		    cse754 - cse876 * (-cse830 * (-cse1242 * cse869 + cse863 * cse864 * cse865) - cse874);
		const amrex::Real cse1245 = std::pow(state.T, 0.29999999999999999);
		const amrex::Real cse1246 = std::pow(cse713, -2);
		const amrex::Real cse1247 = cse1246 * cse54;
		const amrex::Real cse1248 = cse720 * cse722;
		const amrex::Real cse1249 = cse1150 * cse1248 * state.xn[2];
		const amrex::Real cse1250 = 0.5 * cse21 * cse704 - cse755 * cse757;
		const amrex::Real cse1251 = 9.0599999999999997e-6 * cse7;
		const amrex::Real cse1252 = 1.024e-10 * cse987;
		const amrex::Real cse1253 = 2.1600000000000004e-15 * cse956;
		const amrex::Real cse1254 = 9.7199999999999998e-11 * cse971;
		const amrex::Real cse1255 = cse1000 * cse1254 + cse1252 * cse981 + cse1252 * cse985 + cse1253 * cse996 +
					    7.1999999999999997e-16 * cse956 * cse997 + 3.2399999999999999e-11 * cse969 * cse971;
		const amrex::Real cse1256 = 3.0719999999999998e-10 * cse987;
		const amrex::Real cse1257 = 6.4799999999999996e-15 * cse1008 * cse956 + 2.9159999999999995e-10 * cse1008 * cse971 + cse1019 * cse1254 +
					    cse1021 * cse1256 + cse1022 * cse1256 + cse1253 * cse981;
		const amrex::Real cse1258 = cse1025 * (-cse1002 * cse988 - cse1003 * cse988 - 1.6199999999999998e-10 * cse1005 * cse971 -
						       3.6000000000000001e-15 * cse1026 * cse956 - cse1255 - cse1257 - cse957 * cse999 - cse972 * cse999);
		const amrex::Real cse1259 = cse1048 * cse7;
		const amrex::Real cse1260 = cse1090 * cse7;
		const amrex::Real cse1261 = 1.4381666666666666e-5 * cse1260;
		const amrex::Real cse1262 = cse1069 * cse7;
		const amrex::Real cse1263 = 8.6289999999999999e-6 * cse1259;
		const amrex::Real cse1264 = 8.6289999999999999e-6 * cse1262;
		const amrex::Real cse1265 = 1.4381666666666668e-5 * cse1053 * cse1262 + 1.4381666666666668e-5 * cse1072 * cse1259 + cse1095 * cse1263 +
					    cse1098 * cse1261 + cse1099 * cse1261 + cse1101 * cse1264;
		const amrex::Real cse1266 = 5.1774000000000006e-6 * cse1115;
		const amrex::Real cse1267 = 8.6289999999999999e-6 * cse1260;
		const amrex::Real cse1268 =
		    cse1098 * cse1263 + cse1121 * cse1264 + cse1123 * cse1267 + cse1124 * cse1267 + cse1259 * cse1266 + cse1262 * cse1266;
		const amrex::Real cse1269 = cse1091 * cse7;
		const amrex::Real cse1270 = 1.7258000000000001e-6 * cse1100;
		const amrex::Real cse1271 = cse1127 * (-cse1104 * cse1269 - cse1105 * cse1269 - 2.8763333333333337e-6 * cse1107 * cse1262 -
						       2.8763333333333337e-6 * cse1128 * cse1259 - cse1259 * cse1270 - cse1262 * cse1270 - cse1265 - cse1268);
		const amrex::Real cse1272 = cse1248 * cse724;
		const amrex::Real cse1273 = 3.7996253111508318e-29 * cse1272;
		const amrex::Real cse1274 = cse749 * (4 * state.xn[0] + 4 * state.xn[1] + 4 * state.xn[10] + 4 * state.xn[14] + 4 * state.xn[15] +
						      12 * state.xn[16] + 8 * state.xn[3] + 8 * state.xn[4]);
		const amrex::Real cse1275 = cse135 * cse752;
		const amrex::Real cse1276 = cse753 * (200.0 * cse1275 + 200.0 * cse751);
		const amrex::Real cse1277 = cse704 * cse755;
		const amrex::Real cse1278 = cse756 * cse758;
		const amrex::Real cse1279 = -1.0 * cse1277 + 1.0 * cse1278;
		const amrex::Real cse1280 =
		    ((cse761) ? (0) : ((cse763) ? (3 * cse762) : ((cse764) ? ((8.0 / 3.0) * cse762) : ((cse765) ? ((2.0 / 3.0) * cse762) : (0)))));
		const amrex::Real cse1281 = 2.5634832e-11 * cse782 + 6.4087080000000003e-13;
		const amrex::Real cse1282 = -cse792;
		const amrex::Real cse1283 = cse233 * cse771 + cse235 * cse774;
		const amrex::Real cse1284 = 2 * cse846 - cse851 * (cse1283 - cse772);
		const amrex::Real cse1285 = cse235 * cse707;
		const amrex::Real cse1286 = cse1280 * state.xn[3];
		const amrex::Real cse1287 = 4.0 * cse859;
		const amrex::Real cse1288 = ((cse815) ? (0) : (cse818 * cse861));
		const amrex::Real cse1289 = cse1288 * cse868;
		const amrex::Real cse1290 = -cse1288 * cse811 * cse872 * cse873 + cse829 * (cse1288 * cse808 * cse822 * cse872 * cse873 - cse1289 * cse823);
		const amrex::Real cse1291 = cse979 * cse999;
		const amrex::Real cse1292 = cse1010 * cse1291 + cse1011 * cse994 + cse1013 * cse966 + cse1193 * cse994 + cse1195 * cse979 + cse1197 * cse966;
		const amrex::Real cse1293 = cse1003 * cse994;
		const amrex::Real cse1294 = cse1002 * cse994;
		const amrex::Real cse1295 = cse1009 * cse966 + cse1009 * cse979 + cse1011 * cse966 + cse1204 * cse1293 + cse1204 * cse1294 + cse1206 * cse979;
		const amrex::Real cse1296 =
		    cse1025 * (-cse1006 * cse979 - cse1208 * cse966 - cse1291 - cse1292 - cse1293 - cse1294 - cse1295 - cse966 * cse999);
		const amrex::Real cse1297 = cse1060 * cse1100;
		const amrex::Real cse1298 =
		    cse1039 * cse1111 + cse1039 * cse1215 + cse1060 * cse1218 + cse1081 * cse1112 + cse1081 * cse1219 + cse1110 * cse1297;
		const amrex::Real cse1299 = cse1081 * cse1105;
		const amrex::Real cse1300 = cse1081 * cse1104;
		const amrex::Real cse1301 =
		    cse1039 * cse1112 + cse1039 * cse1116 + cse1060 * cse1116 + cse1060 * cse1225 + cse1224 * cse1299 + cse1224 * cse1300;
		const amrex::Real cse1302 =
		    cse1127 * (-cse1039 * cse1100 - cse1039 * cse1228 - cse1060 * cse1108 - cse1297 - cse1298 - cse1299 - cse1300 - cse1301);
		const amrex::Real cse1303 = 3.4516225000000006e-16 * cse1231;
		const amrex::Real cse1304 = cse119 * cse1235;
		const amrex::Real cse1305 = cse1236 * cse855;
		const amrex::Real cse1306 = -cse1283 * cse851 + 2 * cse157 * cse778 * cse779;
		const amrex::Real cse1307 = cse1232 + state.xn[3] * ((cse907) ? (0) : (cse928 * cse953));
		const amrex::Real cse1308 = ((cse815) ? (cse814) : (0));
		const amrex::Real cse1309 = cse1308 * cse868;
		const amrex::Real cse1310 = -1.5 * cse1277 + 1.5 * cse1278;
		const amrex::Real cse1311 = ((cse761) ? (0) : ((cse763) ? ((9.0 / 2.0) * cse762) : ((cse764) ? (4 * cse762) : ((cse765) ? (cse762) : (0)))));
		const amrex::Real cse1312 = 3 * cse157 * cse778 * cse779 - cse851 * (3 * cse847 + 3 * cse848);
		const amrex::Real cse1313 = ((cse815) ? (0) : (2635052181.6465044 * cse862));
		const amrex::Real cse1314 = cse1313 * cse868;
		const amrex::Real cse1315 = state.c_hat * cse135;
		const amrex::Real cse1316 = cse1234 * cse1315;
		const amrex::Real cse1317 = state.c_hat * cse54 * cse744 * cse748 / std::pow(cse742, 5.0 / 6.0);
		const amrex::Real cse1318 = cse1249 * cse1315 * cse715;
		const amrex::Real cse1319 = state.c_hat * cse84;
		const amrex::Real cse1320 = cse1144 * cse1319;
		const amrex::Real cse1321 = cse1146 * cse1319;
		const amrex::Real cse1322 = cse1143 * cse1233 * cse135 * cse708;
		const amrex::Real cse1323 = state.c_hat * cse715;
		const amrex::Real cse1324 = cse1153 * cse1323;
		const amrex::Real cse1325 = cse1154 * cse1323;
		const amrex::Real cse1326 = cse249 * cse778 * cse845;
		const amrex::Real cse1327 = state.c_hat * cse1326 * cse243;
		const amrex::Real cse1328 = cse1326 * cse21 * cse321 * cse568;
		const amrex::Real cse1329 = Zd * cse247;
		const amrex::Real cse1330 = 1.252779946046115e-21 * cse1329;
		const amrex::Real cse1331 = -cse1330;
		const amrex::Real cse1332 = 2.50555989209223e-21 * cse1329;
		const amrex::Real cse1333 = 1.8689621674132952e-20 * cse249;
		const amrex::Real cse1334 = Zd * state.c_hat * cse20;
		const amrex::Real cse1335 = cse1333 * cse321 + cse330 + cse527;
		const amrex::Real cse1336 = state.c_hat * state.rn[2 * 0 + 1];
		const amrex::Real cse1337 = Zd * cse1336;
		const amrex::Real cse1338 = 9.1016827996670106e-22 * cse1337;
		const amrex::Real cse1339 = -cse1338;
		const amrex::Real cse1340 = 1.8203365599334021e-21 * cse1337;
		const amrex::Real cse1341 = Zd * cse251;
		const amrex::Real cse1342 = 2.0584539190345922e-21 * cse1341;
		const amrex::Real cse1343 = -cse1342;
		const amrex::Real cse1344 = 4.1169078380691844e-21 * cse1341;
		const amrex::Real cse1345 = 2.6551134042892157e-17 * cse249;
		const amrex::Real cse1346 = cse1345 * cse321 + cse331 + cse537 + cse540 + cse666;
		const amrex::Real cse1347 = state.c_hat * state.rn[2 * 1 + 1];
		const amrex::Real cse1348 = Zd * cse1347;
		const amrex::Real cse1349 = 1.6856051574414119e-21 * cse1348;
		const amrex::Real cse1350 = -cse1349;
		const amrex::Real cse1351 = 3.3712103148828238e-21 * cse1348;
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
			     1.7623947000000002e-12 * crate * cse853 * state.xn[5] * (cse836 + cse838 * cse858 - cse854 + cse856 * cse857 - cse860) - cse1232 -
			     2.1789607199999999e-11 * cse178 + 3.0000000000000002e-15 * cse20 * cse783 * d2g +
			     2.0187430200000004e-26 * cse20 * cse845 * cse852 * d2g * state.xn[0] - 7.1777529600000012e-12 * cse543 -
			     7.1777529600000012e-12 * cse544 - 7.1777529600000012e-12 * cse546 + 2.5634832e-11 * cse568 * cse845 * cse852 * state.xn[3] -
			     cse725 - cse730 * cse739 - cse750 - cse754 + 3.0000000000000002e-15 * cse783 * d2g * state.xn[0] +
			     4.39089220777172e-29 * cse788 * cse793 * cse795 * cse799 * cse802 * state.xn[8] - 4.39089220777172e-29 * cse788 * cse801 -
			     cse804 * cse834 - cse844 * (-cse796 * cse841 + cse841 * cse842) -
			     cse876 * (-cse804 * cse821 - cse830 * (cse863 * cse866 + cse871 * (cse804 * cse825 - cse867 * cse869)) - cse874) -
			     state.xn[11] * (cse1001 * cse1018 + cse1023 * cse1024 + cse1027 * cse1028 + cse1027 * cse1029) -
			     state.xn[3] * ((cse907) ? (0) : (cse923 * cse953)) -
			     state.xn[7] * ((cse1132) ? (cse1103 * cse1120 + cse1125 * cse1126 + cse1129 * cse1130 + cse1129 * cse1131) : (0));
		jac(18, 1) = cse1244 - state.xn[3] * ((cse907) ? (0) : (cse926 * cse953));
		jac(18, 2) = 5.6705896183685846e-35 * state.c_hat * cse1143 * cse118 * cse20 * cse708 * d2g *
				 (8.429953871621411e-15 * cse1145 * cse210 + 1.2210344531082231e-12 * cse1147 * cse210) +
			     4.22974728e-11 * crate * cse1250 * cse20 * cse703 * cse705 * cse82 * state.xn[0] +
			     4.6527220080000002e-11 * crate * cse1250 * cse703 * cse705 * cse82 * cse853 * state.xn[0] * state.xn[5] +
			     6.0732876035249664e-19 * cse1137 * cse1139 * cse1245 * cse733 * cse737 * state.xn[0] * state.xn[2] -
			     cse1141 * cse739 * state.xn[0] - cse1151 -
			     cse1152 * (2.8857630567877724e-12 * state.c_hat * cse1149 * cse118 * cse1246 * cse54 * cse718 - cse1154 * cse1247 * cse716) -
			     cse1155 * (5.9301787935659931e-17 * cse1137 * cse1139 * cse1245 * cse733 * state.xn[2] - cse1141 * cse735) -
			     2.4067011396745388e-39 * cse119 * cse1247 * cse1249 * cse135 - cse1232 - 2.1789607199999999e-11 * cse220 +
			     2.6241084038530201e-25 * cse7 * cse793 * cse795 * cse799 * cse802 * state.xn[8] - 2.6241084038530201e-25 * cse7 * cse801 -
			     cse810 * cse834 - cse844 * (-cse1251 * cse796 + cse1251 * cse842) - cse876 * (-cse810 * cse821 - cse810 * cse830 * cse870) -
			     state.xn[11] * (cse1018 * cse1255 + cse1024 * cse1257 + cse1028 * cse1258 + cse1029 * cse1258) -
			     state.xn[3] * ((cse907) ? (0) : (cse938 * cse953)) -
			     state.xn[7] * ((cse1132) ? (cse1120 * cse1265 + cse1126 * cse1268 + cse1130 * cse1271 + cse1131 * cse1271) : (0));
		jac(18, 3) =
		    1.1341179236737169e-34 * state.c_hat * cse118 * cse708 * cse712 * d2g +
		    4.22974728e-11 * crate * cse1279 * cse20 * cse703 * cse705 * cse82 * state.xn[0] + 3.2043540000000001e-12 * crate * cse1280 * state.xn[3] +
		    1.7623947000000002e-12 * crate * cse262 * cse835 * cse840 * state.xn[5] + 3.2043540000000001e-12 * crate * cse837 +
		    1.7623947000000002e-12 * crate * cse853 * state.xn[5] * (cse1279 * cse855 * cse857 - cse1285 + cse1286 * cse838 - cse1287 + cse839) -
		    cse1162 * cse792 - cse1165 * cse1282 - cse1273 - cse1274 - cse1276 + cse1281 * cse568 +
		    2.0187430200000004e-26 * cse1284 * cse20 * cse845 * d2g * state.xn[0] + 2.5634832e-11 * cse1284 * cse568 * cse845 * state.xn[3] - cse1303 -
		    7.1777529600000012e-12 * cse230 - 7.1777529600000012e-12 * cse240 - 1.4355505920000002e-11 * cse241 * state.xn[3] -
		    7.1777529600000012e-12 * cse565 + 6.0000000000000005e-15 * cse783 * d2g * state.xn[0] - cse834 -
		    cse844 * (-cse1170 * cse1282 - cse792 * cse797) -
		    cse876 * (-cse1290 - cse821 - cse830 * (cse1288 * cse866 + cse871 * (-cse1289 * cse867 + cse825))) -
		    state.xn[11] * (cse1018 * cse1292 + cse1024 * cse1295 + cse1028 * cse1296 + cse1029 * cse1296) -
		    state.xn[3] * ((cse907) ? (0) : (cse924 * cse953)) -
		    state.xn[7] * ((cse1132) ? (cse1120 * cse1298 + cse1126 * cse1301 + cse1130 * cse1302 + cse1131 * cse1302) : (0)) -
		    ((cse907) ? (0.0) : (cse1178 * cse949));
		jac(18, 4) =
		    cse1169 * cse1306 + cse1237 * cse1286 + cse1238 * cse261 + cse1240 * cse1306 +
		    cse1241 * (26.399999999999999 * cse1279 * cse703 * cse705 * cse82 * state.xn[0] + 2.0 * cse1280 * cse21 * state.xn[3] - cse1285 - cse1287) -
		    cse1273 - cse1274 - cse1276 + cse1279 * cse1305 - cse1303 + 1.1341179236737169e-34 * cse1304 + cse169 * cse783 -
		    7.1777529600000012e-12 * cse575 - 7.1777529600000012e-12 * cse576 -
		    cse876 * (-cse1290 - cse830 * (-cse1242 * cse1289 + cse1288 * cse864 * cse865));
		jac(18, 5) = 1.7623947000000002e-12 * crate * cse840 * cse853 - cse1307;
		jac(18, 6) = -cse1307;
		jac(18, 7) = -cse1232 - ((cse1132) ? (cse1113 * cse1120 + cse1117 * cse1126) : (0.0));
		jac(18, 8) = -cse1161 * cse843 - cse1232;
		jac(18, 9) = -cse1232 - cse811 * cse833 -
			     cse876 * (cse1308 * cse811 * cse872 * cse873 - cse829 * (cse1308 * cse808 * cse822 * cse872 * cse873 - cse1309 * cse823) -
				       cse830 * (-cse1242 * cse1309 + cse1308 * cse864 * cse865));
		jac(18, 10) = cse1244;
		jac(18, 11) = -cse1012 * cse1024 - cse1015 * cse1018 - cse1232;
		jac(18, 12) = cse1243;
		jac(18, 13) = cse1243;
		jac(18, 14) = cse1244;
		jac(18, 15) = cse1244;
		jac(18, 16) = cse1169 * cse1312 + cse1237 * cse1311 * state.xn[3] + cse1238 * cse313 + cse1240 * cse1312 +
			      cse1241 * (26.399999999999999 * cse1310 * cse703 * cse705 * cse82 * state.xn[0] + 2.0 * cse1311 * cse21 * state.xn[3] -
					 3 * cse854 - 6.0 * cse859) +
			      cse1243 - 5.6994379667262474e-29 * cse1272 + 1.7011768855105754e-34 * cse1304 + cse1305 * cse1310 -
			      7.1777529600000012e-12 * cse309 + cse579 * cse783 - 7.1777529600000012e-12 * cse580 -
			      cse749 * (6 * state.xn[0] + 6 * state.xn[1] + 6 * state.xn[10] + 6 * state.xn[14] + 6 * state.xn[15] + 18 * state.xn[16] +
					12 * state.xn[3] + 12 * state.xn[4]) -
			      cse753 * (300.0 * cse1275 + 300.0 * cse751) -
			      cse876 * (cse1313 * cse811 * cse872 * cse873 - cse829 * (cse1313 * cse808 * cse822 * cse872 * cse873 - cse1314 * cse823) -
					cse830 * (-cse1242 * cse1314 + cse1313 * cse864 * cse865));
		jac(18, 17) = cse1243;
		jac(18, 18) = cse1230;
		jac(18, 19) = -cse1152 * (-4.1664736580507933e-10 * cse1324 + 8.1872148910410564e-9 * cse1325) + cse1281 * cse583 +
			      8.1872148910410565e-33 * cse1316 + 4.7575374194964615e-36 * cse1317 + 3.4748025752386779e-37 * cse1318 +
			      cse1322 * (-1.2171193564238968e-12 * cse1320 - 1.7629333331721322e-10 * cse1321) - 2.466269795723557e-53 * cse1327 -
			      3.1317711691727702e-38 * cse1328;
		jac(18, 21) = -cse1152 * (-6.365129963657969e-10 * cse1324 + 1.2507624216266397e-8 * cse1325) + cse1281 * cse584 +
			      1.2507624216266398e-32 * cse1316 + 7.2680992290799608e-36 * cse1317 + 5.3084626964364086e-37 * cse1318 +
			      cse1322 * (-1.8593956234313972e-12 * cse1320 - 2.6932367041905241e-10 * cse1321) - 7.4187514601998589e-44 * cse1327 -
			      9.4206367748569615e-29 * cse1328;
		jac(19, 0) = cse1331;
		jac(19, 1) = cse1331;
		jac(19, 3) = -cse1332 - cse1333 * cse247;
		jac(19, 4) = -cse1332;
		jac(19, 10) = cse1331;
		jac(19, 12) = -cse510;
		jac(19, 14) = -cse1330 - cse294;
		jac(19, 15) = -cse1330 - cse301;
		jac(19, 16) = -3.7583398381383454e-21 * cse1329;
		jac(19, 19) = -1.252779946046115e-21 * cse1334 - cse1335;
		jac(20, 0) = cse1339;
		jac(20, 1) = cse1339;
		jac(20, 3) = -cse1333 * cse1336 - cse1340;
		jac(20, 4) = -cse1340;
		jac(20, 10) = cse1339;
		jac(20, 12) = -1.9917698086432708e-17 * cse1336;
		jac(20, 14) = -4.2188032355736613e-18 * cse1336 - cse1338;
		jac(20, 15) = -1.7141251340095377e-18 * cse1336 - cse1338;
		jac(20, 16) = -2.7305048399001034e-21 * cse1337;
		jac(20, 20) = -9.1016827996670106e-22 * cse1334 - cse1335;
		jac(21, 0) = cse1343;
		jac(21, 1) = cse1343;
		jac(21, 3) = -cse1344 - cse1345 * cse251;
		jac(21, 4) = -cse1344;
		jac(21, 7) = -cse503;
		jac(21, 9) = -cse627;
		jac(21, 10) = cse1343;
		jac(21, 12) = -cse511;
		jac(21, 14) = -cse1342 - cse295;
		jac(21, 15) = -cse1342 - cse302;
		jac(21, 16) = -6.175361757103777e-21 * cse1341;
		jac(21, 21) = -2.0584539190345922e-21 * cse1334 - cse1346;
		jac(22, 0) = cse1350;
		jac(22, 1) = cse1350;
		jac(22, 3) = -cse1345 * cse1347 - cse1351;
		jac(22, 4) = -cse1351;
		jac(22, 7) = -cse1347 * cse502;
		jac(22, 9) = -cse1347 * cse626;
		jac(22, 10) = cse1350;
		jac(22, 12) = -1.9088222220940885e-17 * cse1347;
		jac(22, 14) = -1.168047934563505e-18 * cse1347 - cse1349;
		jac(22, 15) = -4.1730320353928637e-18 * cse1347 - cse1349;
		jac(22, 16) = -5.0568154723242353e-21 * cse1348;
		jac(22, 22) = -1.6856051574414119e-21 * cse1334 - cse1346;
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
};

// $$JAFF SUB label
} // namespace quokka::networks::GOW
// $$JAFF END

#endif // JAFF_NETWORK_HPP_
