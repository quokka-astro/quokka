// ABOUTME: Implementation for resampled cooling tables that interpolate on (rho, e_int) grid
// ABOUTME: Reads HDF5-format tables produced by extern/cooling/resample_grackle_cooling_tables.py
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

#include "AMReX_FileSystem.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include <H5Apublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>
#include <H5Tpublic.h>
#include <format>

namespace quokka::ResampledCooling
{

// return: is_include_pe
auto readResampledData(std::string const &hdf5_file, resampled_tables &resampledTables) -> bool
{
	amrex::Print() << "Initializing resampled cooling.\n";
	amrex::Print() << std::format("resampled_table_file: {}.\n", hdf5_file);

	if (!amrex::FileSystem::Exists(hdf5_file)) {
		amrex::Abort("Resampled cooling table file does not exist!");
	}

	// Read all 5 cooling outputs: cooling rate is linear, T/cs/P/S are fast_log
	resampledTables.all_tables =
	    quokka::DataTable<2, 5>::H5Reader(hdf5_file, "tab1",
					      {quokka::SpacingType::linear, quokka::SpacingType::fast_log, quokka::SpacingType::fast_log,
					       quokka::SpacingType::fast_log, quokka::SpacingType::fast_log});

	// Read domain-specific attributes (include_pe, cloudy_H_mass_fraction) from tab1
	amrex::Real cloudy_H_mass_fraction_val = 0.0;
	int include_pe_val = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		hid_t const file_id = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id >= 0, "Failed to reopen HDF5 file for extra attributes");
		hid_t const group_id = H5Gopen2(file_id, "tab1", H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(group_id >= 0, "Failed to open tab1 group for extra attributes");

		hid_t attr_id = H5Aopen(group_id, "cloudy_H_mass_fraction", H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open cloudy_H_mass_fraction attribute");
		herr_t status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &cloudy_H_mass_fraction_val);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status >= 0, "Failed to read cloudy_H_mass_fraction attribute");
		H5Aclose(attr_id);

		attr_id = H5Aopen(group_id, "include_pe", H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0, "Failed to open include_pe attribute");
		status = H5Aread(attr_id, H5T_NATIVE_INT, &include_pe_val);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status >= 0, "Failed to read include_pe attribute");
		H5Aclose(attr_id);

		H5Gclose(group_id);
		H5Fclose(file_id);
	}

	amrex::ParallelDescriptor::Bcast(&cloudy_H_mass_fraction_val, 1, amrex::ParallelDescriptor::IOProcessorNumber());
	amrex::ParallelDescriptor::Bcast(&include_pe_val, 1, amrex::ParallelDescriptor::IOProcessorNumber());

	resampledTables.cloudy_H_mass_fraction = cloudy_H_mass_fraction_val;
	resampledTables.include_pe = (include_pe_val != 0);

	// Log info using physical bounds from the DataTable
	const int n_rho = resampledTables.all_tables.size(0);
	const int n_eint = resampledTables.all_tables.size(1);
	const amrex::Real rho_min = resampledTables.all_tables.coord_xlo()[0];
	const amrex::Real rho_max = resampledTables.all_tables.coord_xhi()[0];
	const amrex::Real eint_min = resampledTables.all_tables.coord_xlo()[1];
	const amrex::Real eint_max = resampledTables.all_tables.coord_xhi()[1];

	amrex::Print() << std::format("\tDensity range: {} to {} g/cm^3 ({} steps).\n", rho_min, rho_max, n_rho);
	amrex::Print() << std::format("\tSpecific energy range: {} to {} erg/g ({} steps).\n", eint_min, eint_max, n_eint);
	amrex::Print() << std::format("\tPhotoelectric heating: {}.\n", resampledTables.include_pe ? "included in table" : "NOT included in table");

	return resampledTables.include_pe;
}

auto resampled_tables::const_tables() const -> resampledGpuConstTables
{
	return resampledGpuConstTables{
	    .all_tables = all_tables.const_tables(),
	    .cloudy_H_mass_fraction = cloudy_H_mass_fraction,
	    .eint_min = all_tables.coord_xlo()[1],
	    .eint_max = all_tables.coord_xhi()[1],
	};
}

auto resampled_tables::const_tables_host() const -> resampledGpuConstTables
{
	return resampledGpuConstTables{
	    .all_tables = all_tables.const_tables_host(),
	    .cloudy_H_mass_fraction = cloudy_H_mass_fraction,
	    .eint_min = all_tables.coord_xlo()[1],
	    .eint_max = all_tables.coord_xhi()[1],
	};
}

} // namespace quokka::ResampledCooling
