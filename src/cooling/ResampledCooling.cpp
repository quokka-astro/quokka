// ABOUTME: Implementation for resampled cooling tables that interpolate on (rho, e_int) grid
// ABOUTME: Reads HDF5-format tables produced by extern/cooling/resample_cooling_tables.py
//==============================================================================
//  TwoMomentRad - a radiation transport library for patch-based AMR codes
//  Copyright 2020 Benjamin Wibking.
//  Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ResampledCooling.cpp
/// \brief Implements methods for interpolating cooling rates from resampled
/// tables.
///

#include "cooling/ResampledCooling.hpp"

#include "AMReX_Print.H"

namespace quokka::ResampledCooling
{

constexpr double cloudy_H_mass_fraction = 1. / (1. + 0.1 * 3.971);

void readResampledData(std::string const &hdf5_file, resampled_tables &resampledTables)
{
	amrex::Print() << "Initializing resampled cooling.\n";
	amrex::Print() << fmt::format("resampled_table_file: {}.\n", hdf5_file);

	// Read all 2D datasets using new generic DataTable H5Reader (file path + group name)
	resampledTables.cooling_rates = quokka::DataTable<2, 1>::H5Reader(hdf5_file, "/cooling_rates");
	resampledTables.temperatures = quokka::DataTable<2, 1>::H5Reader(hdf5_file, "/temperatures");
	resampledTables.sound_speeds = quokka::DataTable<2, 1>::H5Reader(hdf5_file, "/sound_speeds");
	resampledTables.pressures = quokka::DataTable<2, 1>::H5Reader(hdf5_file, "/pressures");
	resampledTables.entropies = quokka::DataTable<2, 1>::H5Reader(hdf5_file, "/entropies");

	// Set coordinate bounds from the table metadata
	// Dimension 0: rho, Dimension 1: eint
	resampledTables.rho_min = resampledTables.cooling_rates.coord_min(0);
	resampledTables.rho_max = resampledTables.cooling_rates.coord_max(0);
	resampledTables.eint_min = resampledTables.cooling_rates.coord_min(1);
	resampledTables.eint_max = resampledTables.cooling_rates.coord_max(1);

	// Get grid dimensions from the DataTable objects for logging
	const int n_rho = resampledTables.cooling_rates.size(0);
	const int n_eint = resampledTables.cooling_rates.size(1);

	resampledTables.cloudy_H_mass_fraction = cloudy_H_mass_fraction;

	amrex::Print() << fmt::format("\tDensity range: {} to {} g/cm^3 ({} steps).\n", FastMath::pow2(resampledTables.rho_min),
				      FastMath::pow2(resampledTables.rho_max), n_rho);
	amrex::Print() << fmt::format("\tSpecific energy range: {} to {} erg/g ({} steps).\n", FastMath::pow2(resampledTables.eint_min),
				      FastMath::pow2(resampledTables.eint_max), n_eint);
}

auto resampled_tables::const_tables() const -> resampledGpuConstTables
{
	resampledGpuConstTables tables{cooling_rates.const_tables(),
				       temperatures.const_tables(),
				       sound_speeds.const_tables(),
				       pressures.const_tables(),
				       entropies.const_tables(),
				       rho_min,
				       rho_max,
				       eint_min,
				       eint_max,
				       cloudy_H_mass_fraction};
	return tables;
}

} // namespace quokka::ResampledCooling