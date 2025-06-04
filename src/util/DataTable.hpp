#ifndef DATATABLE_HPP_
#define DATATABLE_HPP_

#include "AMReX.H"
#include "AMReX_TableData.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"
#include "math/Interpolate2D.hpp"
#include <memory>

namespace quokka
{

// GPU-friendly struct containing const table references
struct DataTableGpuConst {
	amrex::Table1D<const amrex::Real> x_coords;
	amrex::Table1D<const amrex::Real> y_coords;
	amrex::Table2D<const amrex::Real> data;
	
	amrex::Real x_min;
	amrex::Real x_max;
	amrex::Real y_min;
	amrex::Real y_max;
	
	int x_size;
	int y_size;
	
	// Member function for interpolation - cleaner API
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
	auto interpolate(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		// Clamp x and y to valid bounds
		x = amrex::max(x_min, amrex::min(x, x_max));
		y = amrex::max(y_min, amrex::min(y, y_max));
		
		return interpolate2d(x, y, x_coords, y_coords, data);
	}
};

// Generic 2D data table class
class DataTable
{
public:
	// Default constructor
	DataTable() = default;
	
	// Constructor with data
	DataTable(const amrex::Vector<amrex::Real>& x_coords,
	          const amrex::Vector<amrex::Real>& y_coords,
	          const amrex::Vector<amrex::Vector<amrex::Real>>& data);
	
	// Move constructor and assignment
	DataTable(DataTable&&) = default;
	DataTable& operator=(DataTable&&) = default;
	
	// Delete copy constructor and assignment (expensive operations)
	DataTable(const DataTable&) = delete;
	DataTable& operator=(const DataTable&) = delete;
	
	// Initialize from vectors
	void initialize(const amrex::Vector<amrex::Real>& x_coords,
	                const amrex::Vector<amrex::Real>& y_coords,
	                const amrex::Vector<amrex::Vector<amrex::Real>>& data);
	
	// Get GPU-friendly const tables
	[[nodiscard]] auto const_tables() const -> DataTableGpuConst;
	
	// Check if table is initialized
	[[nodiscard]] bool is_initialized() const;
	
	// Get dimensions
	[[nodiscard]] int x_size() const;
	[[nodiscard]] int y_size() const;
	
private:
	std::unique_ptr<amrex::TableData<amrex::Real, 1>> x_coords_;
	std::unique_ptr<amrex::TableData<amrex::Real, 1>> y_coords_;
	std::unique_ptr<amrex::TableData<amrex::Real, 2>> data_;
	
	amrex::Real x_min_ = 0.0;
	amrex::Real x_max_ = 0.0;
	amrex::Real y_min_ = 0.0;
	amrex::Real y_max_ = 0.0;
	
	int x_size_ = 0;
	int y_size_ = 0;
};

} // namespace quokka

#endif // DATATABLE_HPP_ 