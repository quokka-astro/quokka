//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file TabulatedCooling.cpp
/// \brief Implements methods for interpolating cooling rates from Cloudy
/// tables.
///

#include "TabulatedCooling.hpp"
#include "CloudyDataReader.hpp"

namespace quokka::TabulatedCooling
{

void readCloudyData(std::string &hdf5_file, cloudy_tables &cloudyTables)
{
	cloudy_cooling_tools_data cloudy;
	code_units my_units; // cgs
	my_units.density_units = 1.0;
	my_units.length_units = 1.0;
	my_units.time_units = 1.0;
	my_units.velocity_units = 1.0;

	initialize_cloudy_data(cloudy, hdf5_file, my_units);
	cloudyTables.T_min = cloudy.T_min;
	cloudyTables.T_max = cloudy.T_max;
	cloudyTables.mmw_min = cloudy.mmw_min;
	cloudyTables.mmw_max = cloudy.mmw_max;
	cloudyTables.log_nH = std::make_unique<amrex::TableData<double, 1>>(copy_1d_table(cloudy.grid_parameters[0]));
	cloudyTables.log_Tgas = std::make_unique<amrex::TableData<double, 1>>(copy_1d_table(cloudy.grid_parameters[1]));
	cloudyTables.cooling = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy.cooling_data));
	cloudyTables.heating = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy.heating_data));
	cloudyTables.mean_mol_weight = std::make_unique<amrex::TableData<double, 2>>(extract_2d_table(cloudy.mmw_data));
}

auto cloudy_tables::const_tables() const -> cloudyGpuConstTables
{
	cloudyGpuConstTables tables{log_nH->const_table(),
				    log_Tgas->const_table(),
				    cooling->const_table(),
				    heating->const_table(),
				    mean_mol_weight->const_table(),
				    T_min,
				    T_max,
				    mmw_min,
				    mmw_max};

	return tables;
}

} // namespace quokka::TabulatedCooling