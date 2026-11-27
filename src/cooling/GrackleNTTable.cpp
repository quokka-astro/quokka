// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file GrackleNTTable.cpp
/// \brief Reading and registering the Grackle (nH, T) table.

#include "cooling/GrackleNTTable.hpp"

#include "AMReX_Arena.H"
#include "AMReX_FileSystem.H"
#include "AMReX_ParallelDescriptor.H"

#include <H5Apublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>
#include <H5Tpublic.h>

#include <array>
#include <cstddef>
#include <format>
#include <memory>
#include <new>
#include <string>

namespace quokka::ISMCooling
{

AMREX_GPU_MANAGED GrackleNTRegistry *g_grackle_nt_registry = nullptr; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

auto ReadGrackleNTTable(std::string const &path) -> std::unique_ptr<quokka::DataTable<2, 2>>
{
	if (!amrex::FileSystem::Exists(path)) {
		amrex::Abort(std::format("The Grackle (nH, T) table file '{}' does not exist (ism_cooling.grackle_nT_table).", path));
	}
	auto table = std::make_unique<quokka::DataTable<2, 2>>(
	    quokka::DataTable<2, 2>::H5Reader(path, "tab1", {quokka::TransformType::linear, quokka::TransformType::linear}));
	const auto handles = table->const_tables_host();
	for (int dim = 0; dim < 2; ++dim) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(handles.spacing_types[dim] == quokka::TransformType::log,
						 std::format("The Grackle (nH, T) table '{}' must have log spacing in both inputs.", path).c_str());
	}
	return table;
}

auto ReadGrackleNTAttributes(std::string const &path) -> GrackleNTAttributes
{
	std::array<double, 2> values{};
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id >= 0, std::format("Failed to open the Grackle (nH, T) table '{}'", path).c_str());
		const hid_t group_id = H5Gopen2(file_id, "tab1", H5P_DEFAULT);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(group_id >= 0, std::format("The Grackle (nH, T) table '{}' has no group tab1", path).c_str());
		constexpr std::array<const char *, 2> names = {"zmet", "H_mass_fraction"};
		for (std::size_t i = 0; i < names.size(); ++i) {
			const hid_t attr_id = H5Aopen(group_id, names.at(i), H5P_DEFAULT);
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(attr_id >= 0,
							 std::format("The Grackle (nH, T) table '{}' has no attribute {}", path, names.at(i)).c_str());
			const herr_t status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, &values.at(i));
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status >= 0, std::format("Failed to read the attribute {} of '{}'", names.at(i), path).c_str());
			H5Aclose(attr_id);
		}
		H5Gclose(group_id);
		H5Fclose(file_id);
	}
	amrex::ParallelDescriptor::Bcast(values.data(), 2, amrex::ParallelDescriptor::IOProcessorNumber());
	return {static_cast<Real>(values[0]), static_cast<Real>(values[1])};
}

void RegisterGrackleNTTable(GrackleNTGpuConst host, GrackleNTGpuConst device)
{
	if (g_grackle_nt_registry == nullptr) {
		auto *mem = amrex::The_Managed_Arena()->alloc(sizeof(GrackleNTRegistry));
		g_grackle_nt_registry = new (mem) GrackleNTRegistry{}; // NOLINT(cppcoreguidelines-owning-memory)
	}
	g_grackle_nt_registry->host = host;
	g_grackle_nt_registry->device = device;
}

} // namespace quokka::ISMCooling
