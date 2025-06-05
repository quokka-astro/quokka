#include "util/DataTable.hpp"
#include "AMReX_Algorithm.H"
#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include <algorithm>

namespace quokka
{

DataTable::DataTable(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
		     const amrex::Vector<amrex::Vector<amrex::Real>> &data)
{
	initialize(x_coords, y_coords, data);
}

void DataTable::initialize(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
			   const amrex::Vector<amrex::Vector<amrex::Real>> &data)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!x_coords.empty(), "X coordinates cannot be empty!");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!y_coords.empty(), "Y coordinates cannot be empty!");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data.empty(), "Data cannot be empty!");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data.size() == x_coords.size(), "Data first dimension must match x_coords size!");

	x_size_ = static_cast<int>(x_coords.size());
	y_size_ = static_cast<int>(y_coords.size());

	// Verify data dimensions
	for (const auto &row : data) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(row.size() == y_coords.size(), "All data rows must match y_coords size!");
	}

	// Store coordinate bounds (assuming ascending order)
	x_min_ = x_coords.front();
	x_max_ = x_coords.back();
	y_min_ = y_coords.front();
	y_max_ = y_coords.back();

	// Create x coordinates table
	x_coords_ = std::make_unique<amrex::TableData<amrex::Real, 1>>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{x_size_ - 1}, amrex::The_Pinned_Arena());
	auto x_table = x_coords_->table();
	for (int i = 0; i < x_size_; ++i) {
		x_table(i) = x_coords[i];
	}

	// Create y coordinates table
	y_coords_ = std::make_unique<amrex::TableData<amrex::Real, 1>>(amrex::Array<int, 1>{0}, amrex::Array<int, 1>{y_size_ - 1}, amrex::The_Pinned_Arena());
	auto y_table = y_coords_->table();
	for (int j = 0; j < y_size_; ++j) {
		y_table(j) = y_coords[j];
	}

	// Create 2D data table
	data_ = std::make_unique<amrex::TableData<amrex::Real, 2>>(amrex::Array<int, 2>{0, 0}, amrex::Array<int, 2>{x_size_ - 1, y_size_ - 1},
								   amrex::The_Pinned_Arena());
	auto data_table = data_->table();

	// Copy data (input is data[i][j], table is accessed as table(i,j))
	for (int i = 0; i < x_size_; ++i) {
		for (int j = 0; j < y_size_; ++j) {
			data_table(i, j) = data[i][j];
		}
	}
}

auto DataTable::const_tables() const -> DataTableGpuConst
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_initialized(), "DataTable must be initialized before getting const tables!");

	DataTableGpuConst tables{x_coords_->const_table(), y_coords_->const_table(), data_->const_table(), x_min_, x_max_, y_min_, y_max_, x_size_, y_size_};
	return tables;
}

bool DataTable::is_initialized() const { return (x_coords_ != nullptr && y_coords_ != nullptr && data_ != nullptr); }

int DataTable::x_size() const { return x_size_; }

int DataTable::y_size() const { return y_size_; }

} // namespace quokka