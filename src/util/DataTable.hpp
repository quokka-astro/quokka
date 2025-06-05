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

// Structure to hold interpolation indices and normalized coordinates
struct InterpData {
	int ix, iy, iix, iiy;  // grid indices
	amrex::Real x1, x2, y1, y2;     // actual coordinate values at grid points
	amrex::Real h, v;               // normalized coordinates: h = (x-x1)/(x2-x1), v = (y-y1)/(y2-y1)
	
	// Default constructor
	AMREX_GPU_HOST_DEVICE InterpData() 
		: ix(0), iy(0), iix(0), iiy(0), 
		  x1(0.0), x2(0.0), y1(0.0), y2(0.0),
		  h(0.0), v(0.0) {}
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
		
		// Compute normalized coordinates
		if (interp.ix != interp.iix) {
			interp.h = (x - interp.x1) / (interp.x2 - interp.x1);
		} else {
			interp.h = 0.0;  // No variation in x direction
		}
		
		if (interp.iy != interp.iiy) {
			interp.v = (y - interp.y1) / (interp.y2 - interp.y1);
		} else {
			interp.v = 0.0;  // No variation in y direction
		}
		
		return interp;
	}
	
	// Convenience method: find interpolation data and compute value in one call
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
	auto interpolate(amrex::Real x, amrex::Real y) const -> amrex::Real
	{
		// Part 1: Find interpolation indices and normalized coordinates
		InterpData interp = find_interpolation_data(x, y);

		// Part 2: Compute interpolated value using precomputed indices and normalized coordinates
		// Get the four corner values: (z1, z2, z3, z4) = (A, C, B, D)
		// z1 = f(0,0) -> (x1, y1), z2 = f(1,0) -> (x2, y1)
		// z3 = f(0,1) -> (x1, y2), z4 = f(1,1) -> (x2, y2)
		amrex::Real z1 = data(interp.ix, interp.iy);    // A = data(ix, iy) - bottom left
		amrex::Real z2 = data(interp.iix, interp.iy);   // C = data(iix, iy) - bottom right
		amrex::Real z3 = data(interp.ix, interp.iiy);   // B = data(ix, iiy) - top left
		amrex::Real z4 = data(interp.iix, interp.iiy);  // D = data(iix, iiy) - top right
		
		// f(h, v) = (1 - v)((1 - h) z1 + h z2) + v((1 - h) z3 + h z4)
		amrex::Real value = (1.0 - interp.v) * ((1.0 - interp.h) * z1 + interp.h * z2) + 
		                    interp.v * ((1.0 - interp.h) * z3 + interp.h * z4);
		AMREX_ASSERT(!std::isnan(value));
		
		return value;
	}
	
	// Compute numeric derivatives (∂f/∂x, ∂f/∂y) using normalized coordinate algorithm
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
	auto numeric_derivative(amrex::Real x, amrex::Real y) const -> amrex::Array<amrex::Real, 2>
	{
		// Get interpolation data (includes precomputed h and v)
		InterpData interp = find_interpolation_data(x, y);
		
		// Get the four corner values: (z1, z2, z3, z4) = (A, C, B, D)
		// z1 = f(0,0) -> (x1, y1), z2 = f(1,0) -> (x2, y1)
		// z3 = f(0,1) -> (x1, y2), z4 = f(1,1) -> (x2, y2)
		amrex::Real z1 = data(interp.ix, interp.iy);    // A = data(ix, iy) - bottom left
		amrex::Real z2 = data(interp.iix, interp.iy);   // C = data(iix, iy) - bottom right
		amrex::Real z3 = data(interp.ix, interp.iiy);   // B = data(ix, iiy) - top left
		amrex::Real z4 = data(interp.iix, interp.iiy);  // D = data(iix, iiy) - top right
		
		amrex::Real f_h = 0.0;
		amrex::Real f_v = 0.0;
		
		// Compute derivatives in normalized coordinates
		if (interp.ix != interp.iix && interp.iy != interp.iiy) {
			// Full bilinear case
			// f_h = v (z4 - z3) + (1 - v) (z2 - z1)
			f_h = interp.v * (z4 - z3) + (1.0 - interp.v) * (z2 - z1);
			
			// f_v = h (z4 - z2) + (1 - h) (z3 - z1)
			f_v = interp.h * (z4 - z2) + (1.0 - interp.h) * (z3 - z1);
			
		} else if (interp.ix == interp.iix && interp.iy != interp.iiy) {
			// Linear interpolation in y direction only
			f_h = 0.0;  // No variation in x direction
			f_v = z3 - z1;  // Linear derivative in normalized v coordinate
			
		} else if (interp.ix != interp.iix && interp.iy == interp.iiy) {
			// Linear interpolation in x direction only
			f_h = z2 - z1;  // Linear derivative in normalized h coordinate
			f_v = 0.0;  // No variation in y direction
			
		} else {
			// Point interpolation - no derivatives
			f_h = 0.0;
			f_v = 0.0;
		}
		
		// Convert to physical coordinates: f_x = f_h / (x2 - x1), f_y = f_v / (y2 - y1)
		amrex::Real dfdx = (interp.ix != interp.iix) ? f_h / (interp.x2 - interp.x1) : 0.0;
		amrex::Real dfdy = (interp.iy != interp.iiy) ? f_v / (interp.y2 - interp.y1) : 0.0;
		
		return {dfdx, dfdy};
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