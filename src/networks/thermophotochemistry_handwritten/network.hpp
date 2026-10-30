#ifndef NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
#define NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
/// \file network.hpp
/// \brief The constants and runtime parameters of the thermophotochemistry_handwritten network.

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::networks
{

/// The runtime parameters of Thermophotochemistry, read from the `network` prefix with Microphysics' names and
/// defaults. They are defined in network.cpp, not inline here: HIP cannot register inline managed variables.
namespace thermophotochemistry
{
extern AMREX_GPU_MANAGED int recombination_switch;		  // NOLINT
extern AMREX_GPU_MANAGED int recombination_temperature_dependent; // NOLINT
extern AMREX_GPU_MANAGED int collisional_ionization_switch;	  // NOLINT
extern AMREX_GPU_MANAGED int energy_switch;			  // NOLINT
extern AMREX_GPU_MANAGED int photoheating_switch;		  // NOLINT
extern AMREX_GPU_MANAGED int KI_heating_switch;			  // NOLINT
extern AMREX_GPU_MANAGED int recombination_cooling_switch;	  // NOLINT
extern AMREX_GPU_MANAGED int KI_cooling_switch;			  // NOLINT
extern AMREX_GPU_MANAGED int ion_ff_cooling_switch;		  // NOLINT
/// cooling in mixed cells; with 0, cells with mixed_cell_lb < x_HII < mixed_cell_ub do not cool
extern AMREX_GPU_MANAGED int allow_mixed_cell_cooling; // NOLINT
extern AMREX_GPU_MANAGED amrex::Real mixed_cell_lb;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real mixed_cell_ub;    // NOLINT
extern AMREX_GPU_MANAGED amrex::Real dust_kappa;       // NOLINT
extern AMREX_GPU_MANAGED amrex::Real small_x;	       // NOLINT
} // namespace thermophotochemistry

/// Hydrogen photoionization in one chemical band, with dust in two thermal bands.
struct Thermophotochemistry {
	static constexpr int NumSpec = 3;
	/// indices of the species in xn
	enum Species : int { H = 0, H_p, e };

	static constexpr int NumChemBands = 1;
	/// edges of the chemical bands [eV]
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands() -> amrex::GpuArray<double, NumChemBands + 1> { return {{13.6, 26.0}}; }
	/// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a chemical band
	static constexpr double ChemBandsPowerLawIndex = 0;

	/// the IR and the optical band, in that order
	static constexpr int NumThermalBands = 2;

	/// gray dust opacity to ionizing radiation [cm^2/g] (thermophotochemistry::dust_kappa)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto DustKappa() -> amrex::Real { return thermophotochemistry::dust_kappa; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SmallX() -> amrex::Real { return thermophotochemistry::small_x; }

	/// Read the runtime parameters (namespace thermophotochemistry).
	static void ReadParameters();
};

} // namespace quokka::networks

#endif // NETWORKS_THERMOPHOTOCHEMISTRY_HANDWRITTEN_NETWORK_HPP_
