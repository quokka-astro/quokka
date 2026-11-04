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

/// The jaff network's own namespace, named after the network, so that several
/// jaff networks can be linked into one executable.
// $$JAFF SUB label
namespace quokka::networks::photoionization_H_caseB
{
// $$JAFF END

// The runtime parameters of Network, read from the `network` prefix. They are
// defined in network.cpp, not inline here: HIP cannot register inline managed
// variables.

/// floor of the species and radiation after a burn, and of the mass scalars
extern AMREX_GPU_MANAGED amrex::Real small_x; // NOLINT

// Network symbols mapped onto the burn state, shared by the RHS and the
// Jacobian. jaff's energy row is the volumetric dE/dt; Quokka integrates the
// specific internal energy e = E / rho, so the functions below divide that row
// by state.rho. jaff emits a density and a flux moment per band, interleaved in
// state.rn, so the problem must integrate the photon flux.
// $$JAFF DEFINE quokka {
// REPLACE {
// "radeden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
// "photden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
// "rflux\[\s*(\d+)\s*\]" -> "state.rn[2*\1+1]"
// "nden\[\s*(\d+)\s*\]" -> "state.xn[\1]"
// "tgas" -> "state.T"
// "c_hat" -> "state.c_hat"
// }
// THERMAL { MODE -> dedt  ENERGY -> volumetric }
// RADIATION { ENABLED -> True }
// }

// $$JAFF SUB label
/// The jaff network photoionization_H_caseB.
// $$JAFF END
struct Network {
	// $$JAFF SUB nspec
	static constexpr int NumSpec = 3;
	// $$JAFF END
	/// indices of the species in xn
	enum Species : int { // NOLINT(cppcoreguidelines-use-enum-class)
		// $$JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign
		h = 0,
		hj = 1,
		ek = 2,
		// $$JAFF END
	};

	/// mass of each species [g]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesMass() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		// $$JAFF REPEAT specie_mass IN specie_masses
		return {{1.673773e-24, 1.6728620616289998e-24, 9.10938371e-28}};
		// $$JAFF END
	}
	/// adiabatic index of each species, from the network's [network.eos]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesGamma() -> amrex::GpuArray<amrex::Real, NumSpec>
	{
		// $$JAFF REPEAT specie_gamma IN specie_gammas
		return {{1.6666666666666667, 1.6666666666666667, 1.6666666666666667}};
		// $$JAFF END
	}
	/// the equation of state the network assumes, as jaff names it (the type of
	/// [network.eos]); Quokka maps it to one of its EOSs
	// $$JAFF SUB eos_type
	static constexpr std::string_view EOSType = "ideal";
	// $$JAFF END

	// $$JAFF SUB nbands
	static constexpr int NumChemBands = 1;
	// $$JAFF END
	/// edges of the chemical bands [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1>
	{
		// $$JAFF REPEAT radiation_boundary IN radiation_boundaries
		return {{13.6, 26.0}};
		// $$JAFF END
	}
	/// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a
	/// chemical band; Quokka takes one alpha for all bands
	// $$JAFF GET profile_index IN 0
	static constexpr double ChemBandsPowerLawIndex = 0;
	// $$JAFF END
	// $$JAFF REPEAT idx, profile_index IN profile_indices
	static_assert(0 == ChemBandsPowerLawIndex, "band 0: Quokka takes one profile index for all chemical bands");
	// $$JAFF END

	static constexpr int NumThermalBands = 0;

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
		const amrex::Real cse0 = 3.2385396673401957e-18 * state.c_hat * state.xn[0] * state.rn[2 * 0];
		const amrex::Real cse1 = std::sqrt(state.T);
		const amrex::Real cse2 = 1.0 / state.T;
		const amrex::Real cse3 = cse1 * std::exp(-157896.75724967025 * cse2) * state.xn[0] * state.xn[2];
		const amrex::Real cse4 = cse0 + 5.84e-11 * cse3 - 1.6594178159829075e-10 * std::pow(state.T, -0.69999999999999996) * state.xn[1] * state.xn[2];
		const amrex::Real cse5 = state.T < 100.0;
		const amrex::Real cse6 = state.xn[1] * state.xn[2];
		ydot[0] = -cse4;
		ydot[1] = cse4;
		ydot[2] = cse4;
		ydot[3] =
		    2.5316840555840673e-29 * state.c_hat * state.xn[0] * state.rn[2 * 0] - 1.27251276978816e-21 * cse3 -
		    cse6 *
			(1.8551e-27 * cse1 +
			 ((cse5)
			      ? (3.47e-29 * std::pow(state.T, 1.915))
			      : ((state.T < 630.957344480193)
				     ? (2.34e-26 * std::sqrt(state.T))
				     : ((state.T < 3981.0717055349733)
					    ? (1.11e-24 * std::pow(state.T, -0.099000000000000005))
					    : ((state.T < 10000.0)
						   ? (1.08e-32 * std::pow(state.T, 2.1269999999999998))
						   : ((state.T < 31622.776601683792)
							  ? (2.6700000000000001e-30 * std::pow(state.T, 1.5289999999999999))
							  : ((state.T < 100000.0) ? (1.7399999999999998e-24 * std::pow(state.T, 0.23699999999999999))
										  : ((state.T < 1000000.0) ? (1.1e-21 * std::pow(state.T, -0.32300000000000001))
													   : (7.4900000000000001e-21 *
													      std::pow(state.T, -0.46200000000000002)))))))))) -
		    cse6 * ((cse5) ? (0.0) : (8.4219589e-26 * std::pow(state.T, 0.10999999999999999))) -
		    (2.8000000000000002e-28 * cse1 * std::exp(-92.0 * cse2) + 2.0000000000000002e-19 * std::exp(-118400.0 / (state.T + 1000.0))) *
			std::pow(state.xn[0], 2) +
		    2.0000000000000001e-26 * state.xn[0];
		ydot[4] = -cse0;
		ydot[5] = -3.2385396673401957e-18 * state.c_hat * state.xn[0] * state.rn[2 * 0 + 1];
		// $$JAFF END
		// $$JAFF SUB nspec
		ydot[3] /= state.rho;
		// $$JAFF END
	}

	/// The Jacobian of actual_rhs.
	template <typename BurnT, class MatrixType> AMREX_GPU_HOST_DEVICE static void actual_jac(const BurnT &state, MatrixType &jac)
	{
		using System = quokka::microphysics::ODESystem<typename BurnT::problem_type, BurnT>;
		static_assert(BurnT::traits::integrate_rad_flux, "a jaff network needs Microphysics_Traits::integrate_rad_flux");

		// $$JAFF REPEAT idx, expr, cse IN jacobian { USE quokka }
		const amrex::Real cse0 =
		    1.0 / (2.0709734999999996e-16 * state.xn[0] + 2.0709734999999996e-16 * state.xn[1] + 2.0709734999999996e-16 * state.xn[2]);
		const amrex::Real cse1 = state.xn[1] * state.xn[2];
		const amrex::Real cse2 = std::pow(state.T, -3.0 / 2.0);
		const amrex::Real cse3 = 1.0 / state.T;
		const amrex::Real cse4 = std::exp(-157896.75724967025 * cse3);
		const amrex::Real cse5 = cse4 * state.xn[0] * state.xn[2];
		const amrex::Real cse6 = cse2 * cse5;
		const amrex::Real cse7 = std::sqrt(state.T);
		const amrex::Real cse8 = 1.0 / cse7;
		const amrex::Real cse9 = cse5 * cse8;
		const amrex::Real cse10 = 1.1615924711880352e-10 * cse1 * std::pow(state.T, -1.7) + 9.2211706233807419e-6 * cse6 + 2.92e-11 * cse9;
		const amrex::Real cse11 = -cse0 * cse10;
		const amrex::Real cse12 = 2.0709734999999996e-16 * state.T;
		const amrex::Real cse13 = cse11 * cse12;
		const amrex::Real cse14 = 3.2385396673401957e-18 * state.c_hat;
		const amrex::Real cse15 = cse14 * state.rn[2 * 0];
		const amrex::Real cse16 = cse4 * cse7;
		const amrex::Real cse17 = 5.84e-11 * cse16;
		const amrex::Real cse18 = cse15 + cse17 * state.xn[2];
		const amrex::Real cse19 = 1.6594178159829075e-10 * std::pow(state.T, -0.69999999999999996);
		const amrex::Real cse20 = cse19 * state.xn[2];
		const amrex::Real cse21 = cse19 * state.xn[1];
		const amrex::Real cse22 = cse17 * state.xn[0];
		const amrex::Real cse23 = cse14 * state.xn[0];
		const amrex::Real cse24 = -cse23;
		const amrex::Real cse25 = cse0 * cse10;
		const amrex::Real cse26 = cse12 * cse25;
		const amrex::Real cse27 = cse18 - cse26;
		const amrex::Real cse28 = -cse20 - cse26;
		const amrex::Real cse29 = -cse21 + cse22 - cse26;
		const amrex::Real cse30 = 1.27251276978816e-21 * cse16;
		const amrex::Real cse31 = state.T + 1000.0;
		const amrex::Real cse32 = std::exp(-118400.0 / cse31);
		const amrex::Real cse33 = std::exp(-92.0 * cse3);
		const amrex::Real cse34 = state.T < 100.0;
		const amrex::Real cse35 = std::sqrt(state.T);
		const amrex::Real cse36 = state.T < 630.957344480193;
		const amrex::Real cse37 = state.T < 3981.0717055349733;
		const amrex::Real cse38 = state.T < 10000.0;
		const amrex::Real cse39 = state.T < 31622.776601683792;
		const amrex::Real cse40 = state.T < 100000.0;
		const amrex::Real cse41 = state.T < 1000000.0;
		const amrex::Real cse42 =
		    cse0 *
		    (-cse1 *
			 (9.2755000000000002e-28 * cse8 +
			  ((cse34)
			       ? (6.6450500000000003e-29 * std::pow(state.T, 0.91500000000000004))
			       : ((cse36)
				      ? (1.17e-26 / cse35)
				      : ((cse37) ? (-1.0989000000000001e-25 * std::pow(state.T, -1.099))
						 : ((cse38) ? (2.2971599999999999e-32 * std::pow(state.T, 1.1269999999999998))
							    : ((cse39) ? (4.0824300000000001e-30 * std::pow(state.T, 0.52899999999999991))
								       : ((cse40) ? (4.1237999999999993e-25 * std::pow(state.T, -0.76300000000000001))
										  : ((cse41) ? (-3.5530000000000001e-22 * std::pow(state.T, -1.323))
											     : (-3.4603799999999999e-21 * std::pow(state.T, -1.462)))))))))) -
		     cse1 * ((cse34) ? (0) : (9.2641547899999994e-27 * std::pow(state.T, -0.89000000000000001))) - 2.0092563990834663e-16 * cse6 -
		     6.3625638489408001e-22 * cse9 -
		     std::pow(state.xn[0], 2) *
			 (2.5760000000000003e-26 * cse2 * cse33 + 1.4000000000000001e-28 * cse33 * cse8 + 2.3680000000000003e-14 * cse32 / std::pow(cse31, 2)));
		const amrex::Real cse43 = cse12 * cse42;
		const amrex::Real cse44 = ((cse34) ? (0.0) : (8.4219589e-26 * std::pow(state.T, 0.10999999999999999)));
		const amrex::Real cse45 =
		    1.8551e-27 * cse7 +
		    ((cse34)
			 ? (3.47e-29 * std::pow(state.T, 1.915))
			 : ((cse36)
				? (2.34e-26 * cse35)
				: ((cse37)
				       ? (1.11e-24 * std::pow(state.T, -0.099000000000000005))
				       : ((cse38) ? (1.08e-32 * std::pow(state.T, 2.1269999999999998))
						  : ((cse39) ? (2.6700000000000001e-30 * std::pow(state.T, 1.5289999999999999))
							     : ((cse40) ? (1.7399999999999998e-24 * std::pow(state.T, 0.23699999999999999))
									: ((cse41) ? (1.1e-21 * std::pow(state.T, -0.32300000000000001))
										   : (7.4900000000000001e-21 * std::pow(state.T, -0.46200000000000002)))))))));
		jac(0, 0) = -cse13 - cse18;
		jac(0, 1) = -cse13 + cse20;
		jac(0, 2) = -cse13 + cse21 - cse22;
		jac(0, 3) = cse11;
		jac(0, 4) = cse24;
		jac(1, 0) = cse27;
		jac(1, 1) = cse28;
		jac(1, 2) = cse29;
		jac(1, 3) = cse25;
		jac(1, 4) = cse23;
		jac(2, 0) = cse27;
		jac(2, 1) = cse28;
		jac(2, 2) = cse29;
		jac(2, 3) = cse25;
		jac(2, 4) = cse23;
		jac(3, 0) = 2.5316840555840673e-29 * state.c_hat * state.rn[2 * 0] - cse30 * state.xn[2] - cse43 -
			    2 * state.xn[0] * (2.0000000000000002e-19 * cse32 + 2.8000000000000002e-28 * cse33 * cse7) + 2.0000000000000001e-26;
		jac(3, 1) = -cse43 - cse44 * state.xn[2] - cse45 * state.xn[2];
		jac(3, 2) = -cse30 * state.xn[0] - cse43 - cse44 * state.xn[1] - cse45 * state.xn[1];
		jac(3, 3) = cse42;
		jac(3, 4) = 2.5316840555840673e-29 * state.c_hat * state.xn[0];
		jac(4, 0) = -cse15;
		jac(4, 4) = cse24;
		jac(5, 0) = -cse14 * state.rn[2 * 0 + 1];
		jac(5, 5) = cse24;
		// $$JAFF END
		// jaff's energy row and column are those of E = rho e: the row is d(dE/dt)
		// and the column d/dE. Change them to e, with rho fixed during the burn:
		// the row divides by rho, the column multiplies by rho, and d(de/dt)/de =
		// d(dE/dt)/dE is unchanged.
		// $$JAFF SUB nspec
		for (int i = 0; i < System::neqs; i++) {
			if (i != 3) {
				jac(3, i) /= state.rho;
				jac(i, 3) *= state.rho;
			}
		}
		// $$JAFF END
	}
};

// $$JAFF SUB label
} // namespace quokka::networks::photoionization_H_caseB
// $$JAFF END

#endif // JAFF_NETWORK_HPP_
