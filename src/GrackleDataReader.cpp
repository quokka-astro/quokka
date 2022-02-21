//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file GrackleDataReader.cpp
/// \brief Implements methods for reading the cooling rate tables used by
/// Grackle. Significantly modified from the original version in Grackle to use
/// modern C++ constructs.
///

/***********************************************************************
/ Initialize Cloudy cooling data
/ Copyright (c) 2013, Enzo/Grackle Development Team.
/
/ Distributed under the terms of the Enzo Public Licence.
/ The full license is in the file ENZO_LICENSE, distributed with this
/ software.
************************************************************************/

#include "GrackleDataReader.hpp"
#include "AMReX_Arena.H"
#include "AMReX_TableData.H"

static const bool grackle_verbose = true;

void initialize_cloudy_data(cloudy_data &my_cloudy, char const *group_name,
                            std::string &grackle_data_file,
                            code_units &my_units) {
  // Initialize vectors
  my_cloudy.grid_parameters.resize(CLOUDY_MAX_DIMENSION);
  my_cloudy.grid_dimension.resize(CLOUDY_MAX_DIMENSION);
  for (int64_t q = 0; q < CLOUDY_MAX_DIMENSION; q++) {
    my_cloudy.grid_dimension[q] = 0;
  }

  if (grackle_verbose) {
    fprintf(stdout, "Initializing Cloudy cooling: %s.\n", group_name);
    fprintf(stdout, "cloudy_table_file: %s.\n", grackle_data_file.c_str());
  }

  // Get unit conversion factors (assuming z=0)
  double co_length_units = NAN;
  double co_density_units = NAN;
  co_length_units = my_units.length_units;
  co_density_units = my_units.density_units;
  double tbase1 = my_units.time_units;
  double xbase1 = co_length_units;
  double dbase1 = co_density_units;
  double mh = 1.67e-24;
  double CoolUnit =
      (xbase1 * xbase1 * mh * mh) / (tbase1 * tbase1 * tbase1 * dbase1);

  // Read cooling data from hdf5 file
  hid_t file_id = 0;
  hid_t dset_id = 0;
  hid_t attr_id = 0;
  herr_t status = 0;
  herr_t h5_error = -1;

  file_id = H5Fopen(grackle_data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error,
                                   "Failed to open Grackle data file!");

  if (H5Aexists(file_id, "old_style") != 0) {
    amrex::Abort("Old-style Grackle data tables are not supported!");
  }

  // Open cooling dataset and get grid dimensions
  std::string parameter_name;
  parameter_name = fmt::format("/CoolingRates/{}/Cooling", group_name);

  dset_id = H5Dopen2(file_id, parameter_name.c_str(),
                     H5P_DEFAULT); // new API in HDF5 1.8.0+

  if (dset_id == h5_error) {
    fprintf(stderr, "Can't open Cooling in %s.\n", grackle_data_file.c_str());
    amrex::Abort();
  }

  {
    // Grid rank
    attr_id = H5Aopen_name(dset_id, "Rank");

    if (attr_id == h5_error) {
      fprintf(stderr, "Failed to open Rank attribute in Cooling dataset.\n");
      amrex::Abort();
    }

    int64_t temp_int = 0;
    status = H5Aread(attr_id, HDF5_I8, &temp_int);

    if (attr_id == h5_error) {
      fprintf(stderr, "Failed to read Rank attribute in Cooling dataset.\n");
      amrex::Abort();
    }

    my_cloudy.grid_rank = temp_int;

    if (grackle_verbose) {
      fprintf(stdout, "Cloudy cooling grid rank: %ld.\n", my_cloudy.grid_rank);
    }

    status = H5Aclose(attr_id);

    if (attr_id == h5_error) {
      fprintf(stderr, "Failed to close Rank attribute in Cooling dataset.\n");
      amrex::Abort();
    }
  }

  {
    // Grid dimension
    std::vector<int64_t> temp_int_arr(my_cloudy.grid_rank);
    attr_id = H5Aopen_name(dset_id, "Dimension");

    if (attr_id == h5_error) {
      fprintf(stderr,
              "Failed to open Dimension attribute in Cooling dataset.\n");
      amrex::Abort();
    }

    status = H5Aread(attr_id, HDF5_I8, temp_int_arr.data());

    if (attr_id == h5_error) {
      fprintf(stderr,
              "Failed to read Dimension attribute in Cooling dataset.\n");
      amrex::Abort();
    }

    if (grackle_verbose) {
      fprintf(stdout, "Cloudy cooling grid dimensions:");
    }

    for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {
      my_cloudy.grid_dimension[q] = temp_int_arr[q];
      if (grackle_verbose) {
        fprintf(stdout, " %ld", my_cloudy.grid_dimension[q]);
      }
    }
    if (grackle_verbose) {
      std::cout << std::endl;
    }

    status = H5Aclose(attr_id);

    if (attr_id == h5_error) {
      fprintf(stderr,
              "Failed to close Dimension attribute in Cooling dataset.\n");
      amrex::Abort();
    }
  }

  // Grid parameters
  for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {

    if (q < my_cloudy.grid_rank - 1) {
      parameter_name = fmt::format("Parameter{}", q + 1);
    } else {
      parameter_name = "Temperature";
    }

    std::vector<double> temp_data(my_cloudy.grid_dimension[q]);

    attr_id = H5Aopen_name(dset_id, parameter_name.c_str());

    if (attr_id == h5_error) {
      fprintf(stderr, "Failed to open %s attribute in Cooling dataset.\n",
              parameter_name.c_str());
      amrex::Abort();
    }

    status = H5Aread(attr_id, HDF5_R8, temp_data.data());

    if (attr_id == h5_error) {
      fprintf(stderr, "Failed to read %s attribute in Cooling dataset.\n",
              parameter_name.c_str());
      amrex::Abort();
    }

    my_cloudy.grid_parameters[q].resize(my_cloudy.grid_dimension[q]);

    for (int64_t w = 0; w < my_cloudy.grid_dimension[q]; w++) {
      if (q < my_cloudy.grid_rank - 1) {
        my_cloudy.grid_parameters[q][w] = temp_data[w];
      } else {
        // convert temperature to log
        my_cloudy.grid_parameters[q][w] = log10(temp_data[w]);
      }
    }

    if (grackle_verbose) {
      std::cout << fmt::format(
          "{}: {} to {} ({} steps).\n", parameter_name,
          my_cloudy.grid_parameters[q][0],
          my_cloudy.grid_parameters[q][my_cloudy.grid_dimension[q] - 1],
          my_cloudy.grid_dimension[q]);
    }

    status = H5Aclose(attr_id);

    if (attr_id == h5_error) {
      std::cerr << fmt::format(
          "Failed to close {} attribute in Cooling dataset.\n",
          parameter_name.c_str());
      amrex::Abort();
    }
  }

  my_cloudy.data_size = 1;

  for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {
    my_cloudy.data_size *= my_cloudy.grid_dimension[q];
  }

  {
    // Read Cooling data
    double *temp_data = new double[my_cloudy.data_size];

    status =
        H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

    if (status == h5_error) {
      fprintf(stderr, "Failed to read Cooling dataset.\n");
      amrex::Abort();
    }

    amrex::GpuArray<int, 3> lo{0, 0, 0};
    amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[0]),
                               static_cast<int>(my_cloudy.grid_dimension[1]),
                               static_cast<int>(my_cloudy.grid_dimension[2])};

    my_cloudy.cooling_data = amrex::Table3D<double>(temp_data, lo, hi);

    for (int64_t q = 0; q < my_cloudy.data_size; q++) {
      temp_data[q] = temp_data[q] > 0 ? log10(temp_data[q]) : SMALL_LOG_VALUE;
      // Convert to code units
      temp_data[q] -= log10(CoolUnit);
    }

    status = H5Dclose(dset_id);

    if (status == h5_error) {
      fprintf(stderr, "Failed to close Cooling dataset.\n");
      amrex::Abort();
    }
  }

  {
    // Read Heating data
    double *temp_data = new double[my_cloudy.data_size];

    parameter_name = fmt::format("/CoolingRates/{}/Heating", group_name);

    dset_id = H5Dopen2(file_id, parameter_name.c_str(),
                       H5P_DEFAULT); // new API in HDF5 1.8.0+

    if (dset_id == h5_error) {
      fprintf(stderr, "Can't open Heating in %s.\n", grackle_data_file.c_str());
      amrex::Abort();
    }

    status =
        H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

    if (status == h5_error) {
      fprintf(stderr, "Failed to read Heating dataset.\n");
      amrex::Abort();
    }

    amrex::GpuArray<int, 3> lo{0, 0, 0};
    amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[0]),
                               static_cast<int>(my_cloudy.grid_dimension[1]),
                               static_cast<int>(my_cloudy.grid_dimension[2])};

    my_cloudy.heating_data = amrex::Table3D<double>(temp_data, lo, hi);

    for (int64_t q = 0; q < my_cloudy.data_size; q++) {
      temp_data[q] = temp_data[q] > 0 ? log10(temp_data[q]) : SMALL_LOG_VALUE;
      // Convert to code units
      temp_data[q] -= log10(CoolUnit);
    }

    status = H5Dclose(dset_id);

    if (status == h5_error) {
      fprintf(stderr, "Failed to close Heating dataset.\n");
      amrex::Abort();
    }
  }

  if (std::string(group_name) == "Primordial") {
    // Read mean molecular weight table
    double *temp_data = new double[my_cloudy.data_size];

    amrex::GpuArray<int, 3> lo{0, 0, 0};
    amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[0]),
                               static_cast<int>(my_cloudy.grid_dimension[1]),
                               static_cast<int>(my_cloudy.grid_dimension[2])};

    my_cloudy.mmw_data = amrex::Table3D<double>(temp_data, lo, hi);

    parameter_name = fmt::format("/CoolingRates/{}/MMW", group_name);

    dset_id = H5Dopen2(file_id, parameter_name.c_str(),
                       H5P_DEFAULT); // new API in HDF5 1.8.0+

    if (dset_id == h5_error) {
      fprintf(stderr, "Can't open MMW in %s.\n", grackle_data_file.c_str());
      amrex::Abort();
    }

    status =
        H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

    if (status == h5_error) {
      fprintf(stderr, "Failed to read MMW dataset.\n");
      amrex::Abort();
    }

    status = H5Dclose(dset_id);

    if (status == h5_error) {
      fprintf(stderr, "Failed to close MMW dataset.\n");
      amrex::Abort();
    }
  }

  status = H5Fclose(file_id);

  if (my_cloudy.grid_rank > CLOUDY_MAX_DIMENSION) {
    fprintf(stderr,
            "Error: rank of Cloudy cooling data must be less than or equal to "
            "%d.\n",
            CLOUDY_MAX_DIMENSION);
    amrex::Abort();
  }
}

auto extract_2d_table(amrex::Table3D<double> const &table3D, int redshift_index)
    -> amrex::TableData<double, 2> {
  // array dimensions are: density, redshift, temperature
  auto lo = table3D.begin;
  auto hi = table3D.end;
  std::array<int, 2> newlo{lo[0], lo[2]};
  std::array<int, 2> newhi{hi[0], hi[2]};

  amrex::TableData<double, 2> tableData(newlo, newhi, amrex::The_Pinned_Arena());
  auto table = tableData.table();

  for (int i = lo[0]; i < hi[0]; ++i) {
    for (int j = lo[2]; j < hi[2]; ++j) {
      table(i, j) = table3D(i, redshift_index, j);
    }
  }
  return tableData;
}
