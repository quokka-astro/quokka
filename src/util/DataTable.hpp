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

// Structure to hold interpolation indices and weights
struct InterpData {
	int ix, iy, iix, iiy;  // grid indices
	amrex::Real w11, w12, w21, w22;  // bilinear weights
	amrex::Real x1, x2, y1, y2;     // actual coordinate values at grid points
	
	// Default constructor
	AMREX_GPU_HOST_DEVICE InterpData() 
		: ix(0), iy(0), iix(0), iiy(0), 
		  w11(0.0), w12(0.0), w21(0.0), w22(0.0),
		  x1(0.0), x2(0.0), y1(0.0), y2(0.0) {}
};

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
	
	// Original interpolation method (for backward compatibility)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
	auto interpolate0(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		// Clamp x and y to valid bounds
		x = amrex::max(x_min, amrex::min(x, x_max));
		y = amrex::max(y_min, amrex::min(y, y_max));
		
		return interpolate2d(x, y, x_coords, y_coords, data);
	}
	
	// Part 1: Find interpolation indices and weights
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
	auto find_interpolation_data(amrex::Real x, amrex::Real y) const -> InterpData
	{
		InterpData interp;
		
		// Get table bounds
		amrex::Real xi = x_coords(x_coords.begin);
		amrex::Real xf = x_coords(x_coords.end - 1);
		amrex::Real yi = y_coords(y_coords.begin);
		amrex::Real yf = y_coords(y_coords.end - 1);
		
		amrex::Real dx = (xf - xi) / static_cast<amrex::Real>(x_coords.end - x_coords.begin - 1);
		amrex::Real dy = (yf - yi) / static_cast<amrex::Real>(y_coords.end - y_coords.begin - 1);
		
		// Clamp coordinates to valid bounds
		x = amrex::max(xi, amrex::min(x, xf));
		y = amrex::max(yi, amrex::min(y, yf));
		
		// Compute indices
		interp.ix = amrex::max(x_coords.begin, 
			amrex::min(static_cast<int>(std::floor((x - xi) / dx)), x_coords.end - 1));
		interp.iy = amrex::max(y_coords.begin, 
			amrex::min(static_cast<int>(std::floor((y - yi) / dy)), y_coords.end - 1));
		interp.iix = (interp.ix == x_coords.end - 1) ? interp.ix : interp.ix + 1;
		interp.iiy = (interp.iy == y_coords.end - 1) ? interp.iy : interp.iy + 1;
		
		// Get coordinate values at grid points
		interp.x1 = x_coords(interp.ix);
		interp.x2 = x_coords(interp.iix);
		interp.y1 = y_coords(interp.iy);
		interp.y2 = y_coords(interp.iiy);
		
		// Compute weights
		if (interp.ix != interp.iix && interp.iy != interp.iiy) {
			const amrex::Real vol = (interp.x2 - interp.x1) * (interp.y2 - interp.y1);
			AMREX_ASSERT(vol > 0.0);
			interp.w11 = (interp.x2 - x) * (interp.y2 - y) / vol;
			interp.w12 = (interp.x2 - x) * (y - interp.y1) / vol;
			interp.w21 = (x - interp.x1) * (interp.y2 - y) / vol;
			interp.w22 = (x - interp.x1) * (y - interp.y1) / vol;
		} else if (interp.ix == interp.iix && interp.iy != interp.iiy) {
			const amrex::Real vol = (interp.y2 - interp.y1);
			AMREX_ASSERT(vol > 0.0);
			interp.w11 = (interp.y2 - y) / vol;
			interp.w12 = (y - interp.y1) / vol;
		} else if (interp.ix != interp.iix && interp.iy == interp.iiy) {
			const amrex::Real vol = (interp.x2 - interp.x1);
			AMREX_ASSERT(vol > 0.0);
			interp.w11 = (interp.x2 - x) / vol;
			interp.w21 = (x - interp.x1) / vol;
		} else { // interp.ix == interp.iix && interp.iy == interp.iiy
			interp.w11 = 1.0;
		}
		
		return interp;
	}
	
	// Part 2: Compute interpolated value using precomputed indices and weights
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
	auto interpolate_with_data(const InterpData& interp) const -> amrex::Real
	{
		amrex::Real A = data(interp.ix, interp.iy);
		amrex::Real B = data(interp.ix, interp.iiy);
		amrex::Real C = data(interp.iix, interp.iy);
		amrex::Real D = data(interp.iix, interp.iiy);
		
		amrex::Real value = interp.w11 * A + interp.w12 * B + 
		                    interp.w21 * C + interp.w22 * D;
		AMREX_ASSERT(!std::isnan(value));
		
		return value;
	}
	
	// Convenience method: find interpolation data and compute value in one call
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
	auto interpolate(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		InterpData interp = find_interpolation_data(x, y);
		return interpolate_with_data(interp);
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