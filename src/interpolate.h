#ifndef INTERPOLATE_H_ // NOLINT
#define INTERPOLATE_H_

#include <stdint.h>
#include <math.h>
#include <assert.h>

int64_t binary_search_with_guess(const double key, const double *arr,
					int64_t len, int64_t guess);

void interpolate_arrays(double *x, double *y, int len,
					double *arr_x, double *arr_y, int arr_len);

#endif // INTERPOLATE_H_