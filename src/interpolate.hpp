#ifndef INTERPOLATE_H_ // NOLINT
#define INTERPOLATE_H_

#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "AMReX.H"

AMREX_GPU_HOST_DEVICE int64_t binary_search_with_guess(double key, const double *arr, int64_t len, int64_t guess);

AMREX_GPU_HOST_DEVICE void interpolate_arrays(double *x, double *y, int len, double *arr_x, double *arr_y, int arr_len);

AMREX_GPU_HOST_DEVICE double interpolate_value(double x, double const *arr_x, double const *arr_y, int arr_len);

#endif // INTERPOLATE_H_
