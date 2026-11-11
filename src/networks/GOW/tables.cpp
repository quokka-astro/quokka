// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file tables.cpp
/// \brief Reads the GOW network's tables and keeps them for the run. Written by hand; regenerate.sh keeps this file.

#include "tables.hpp"

#include <new>
#include <string>

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_FileSystem.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "util/DataTable.hpp"

namespace quokka::networks::GOW
{

AMREX_GPU_MANAGED TableRegistry *table_registry = nullptr; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

namespace
{
// the owners of the table memory, which must outlive every kernel
quokka::DataTable<1, 1> l0_table;	 // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<2, 3> co_table;	 // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<2, 1> dust_table;	 // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<1, 2> c_shield_table;	 // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<1, 2> co_shield_table; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

auto MakeTables(const bool device) -> Tables
{
	if (device) {
		return Tables{.l0 = l0_table.const_tables(),
			      .co = co_table.const_tables(),
			      .dust = dust_table.const_tables(),
			      .c_shield = c_shield_table.const_tables(),
			      .co_shield = co_shield_table.const_tables()};
	}
	return Tables{.l0 = l0_table.const_tables_host(),
		      .co = co_table.const_tables_host(),
		      .dust = dust_table.const_tables_host(),
		      .c_shield = c_shield_table.const_tables_host(),
		      .co_shield = co_shield_table.const_tables_host()};
}

// Frees the tables inside amrex::Finalize(), while the arenas they were allocated from still exist.
void ReleaseTables()
{
	l0_table = {};
	co_table = {};
	dust_table = {};
	c_shield_table = {};
	co_shield_table = {};
	if (table_registry != nullptr) {
		table_registry->~TableRegistry();
		amrex::The_Managed_Arena()->free(table_registry);
		table_registry = nullptr;
	}
}
} // namespace

void LoadTables()
{
	if (table_registry != nullptr) {
		return;
	}

	std::string filename = QUOKKA_NETWORKS_DIR "/GOW/gow_tables.hdf5";
	const amrex::ParmParse pp("network");
	pp.query("tables_file", filename);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::FileSystem::Exists(filename), ("GOW table file '" + filename + "' does not exist").c_str());
	amrex::Print() << "Reading the GOW network's tables from " << filename << "\n";

	using quokka::TransformType;
	l0_table = quokka::DataTable<1, 1>::H5Reader(filename, "l0", {TransformType::log});
	co_table = quokka::DataTable<2, 3>::H5Reader(filename, "co", {TransformType::log, TransformType::log, TransformType::linear});
	dust_table = quokka::DataTable<2, 1>::H5Reader(filename, "dust", {TransformType::log});
	// the shielding factors span hundreds of decades
	c_shield_table = quokka::DataTable<1, 2>::H5Reader(filename, "c_shield", {TransformType::log, TransformType::log});
	co_shield_table = quokka::DataTable<1, 2>::H5Reader(filename, "co_shield", {TransformType::log, TransformType::log});

	void *mem = amrex::The_Managed_Arena()->alloc(sizeof(TableRegistry));
	table_registry = new (mem) TableRegistry{.host = MakeTables(false), .device = MakeTables(true)}; // NOLINT(cppcoreguidelines-owning-memory)

	amrex::ExecOnFinalize(ReleaseTables);
}

} // namespace quokka::networks::GOW
