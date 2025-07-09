#include "util/DataTable.hpp"
#include "AMReX_Algorithm.H"
#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>

namespace quokka
{

DataTable::DataTable(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
		     const amrex::Vector<amrex::Vector<amrex::Real>> &data, bool is_log)
{
	initialize(x_coords, y_coords, data, is_log);
}

void DataTable::initialize(const amrex::Vector<amrex::Real> &x_coords, const amrex::Vector<amrex::Real> &y_coords,
			   const amrex::Vector<amrex::Vector<amrex::Real>> &data, bool is_log)
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!x_coords.empty(), "X coordinates cannot be empty!");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!y_coords.empty(), "Y coordinates cannot be empty!");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!data.empty(), "Data cannot be empty!");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(data.size() == x_coords.size(), "Data first dimension must match x_coords size!");

	x_size_ = static_cast<int>(x_coords.size());
	y_size_ = static_cast<int>(y_coords.size());
	is_log_ = is_log;

	// Verify data dimensions
	for (const auto &row : data) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(row.size() == y_coords.size(), "All data rows must match y_coords size!");
	}

	// Store coordinate bounds (assuming ascending order)
	x_min_ = x_coords.front();
	x_max_ = x_coords.back();
	y_min_ = y_coords.front();
	y_max_ = y_coords.back();

	// Calculate uniform grid spacing once during initialization for optimization
	dx_ = (x_max_ - x_min_) / static_cast<amrex::Real>(x_size_ - 1);
	dy_ = (y_max_ - y_min_) / static_cast<amrex::Real>(y_size_ - 1);

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

void DataTable::read_from_ascii_2d(const std::string &filename)
{
	std::ifstream file(filename);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file.is_open(), "Failed to open ASCII data file: " + filename);

	// Read header
	int is_log_int, n_dim, nx, ny;
	amrex::Real x_min, x_max, y_min, y_max;

	file >> is_log_int >> n_dim >> nx >> ny;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_dim == 2, "Only 2D tables are supported!");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nx > 0 && ny > 0, "Grid dimensions must be positive!");

	// Read ranges
	char comma;
	file >> x_min >> comma >> x_max;
	file >> y_min >> comma >> y_max;

	bool is_log = (is_log_int == 1);

	// Create coordinate arrays
	amrex::Vector<amrex::Real> x_coords(nx);
	amrex::Vector<amrex::Real> y_coords(ny);

	if (is_log) {
		// Generate uniform grid in log space
		amrex::Real log_x_min = std::log10(x_min);
		amrex::Real log_x_max = std::log10(x_max);
		amrex::Real log_y_min = std::log10(y_min);
		amrex::Real log_y_max = std::log10(y_max);

		for (int i = 0; i < nx; ++i) {
			x_coords[i] = log_x_min + (log_x_max - log_x_min) * static_cast<amrex::Real>(i) / static_cast<amrex::Real>(nx - 1);
		}
		for (int j = 0; j < ny; ++j) {
			y_coords[j] = log_y_min + (log_y_max - log_y_min) * static_cast<amrex::Real>(j) / static_cast<amrex::Real>(ny - 1);
		}
	} else {
		// Generate uniform grid in linear space
		for (int i = 0; i < nx; ++i) {
			x_coords[i] = x_min + (x_max - x_min) * static_cast<amrex::Real>(i) / static_cast<amrex::Real>(nx - 1);
		}
		for (int j = 0; j < ny; ++j) {
			y_coords[j] = y_min + (y_max - y_min) * static_cast<amrex::Real>(j) / static_cast<amrex::Real>(ny - 1);
		}
	}

	// Read data values
	amrex::Vector<amrex::Vector<amrex::Real>> data(nx);
	for (int i = 0; i < nx; ++i) {
		data[i].resize(ny);
		for (int j = 0; j < ny; ++j) {
			file >> data[i][j];
		}
	}

	file.close();

	// Initialize the table
	initialize(x_coords, y_coords, data, is_log);
}

auto DataTable::const_tables() const -> DataTableGpuConst
{
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_initialized(), "DataTable must be initialized before getting const tables!");

	DataTableGpuConst tables{
	    x_coords_->const_table(), y_coords_->const_table(), data_->const_table(), x_min_, x_max_, y_min_, y_max_, dx_, dy_, x_size_, y_size_, is_log_};
	return tables;
}

auto DataTable::is_initialized() const -> bool { return (x_coords_ != nullptr && y_coords_ != nullptr && data_ != nullptr); }

auto DataTable::x_size() const -> int { return x_size_; }

auto DataTable::y_size() const -> int { return y_size_; }

} // namespace quokka