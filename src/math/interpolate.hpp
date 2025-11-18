#ifndef INTERPOLATE_H_ // NOLINT
#define INTERPOLATE_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"

constexpr int LIKELY_IN_CACHE_SIZE = 8;

// Boundary policies for interpolation
enum class BoundaryPolicy {
	ErrorOnBounds, // Return NAN and assert on out-of-bounds (default behavior)
	Clamp,	       // Return first/last element on out-of-bounds
	Extrapolation  // Linear extrapolation beyond bounds
};

/** @brief find index of a sorted array such that arr[i] <= key < arr[i + 1].
 *
 * If an starting index guess is in-range, the array values around this
 * index are first checked.  This allows for repeated calls for well-ordered
 * keys (a very common case) to use the previous index as a very good guess.
 *
 * If the guess value is not useful, bisection of the array is used to
 * find the index.  If there is no such index, the return values are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @param guess initial guess of index
 * @return index
 */
AMREX_GPU_HOST_DEVICE inline auto binary_search_with_guess(const double key, const double *arr, int64_t len, int64_t guess) -> int64_t
{
	int64_t imin = 0;
	int64_t imax = len;

	/* Handle keys outside of the arr range first */
	if (key > arr[len - 1]) {
		return len;
	}

	if (key < arr[0]) {
		return -1;
	}

	/*
	 * If len <= 4 use linear search.
	 * From above we know key >= arr[0] when we start.
	 */
	if (len <= 4) {
		int64_t i = 0;

		for (i = 1; i < len && key >= arr[i]; ++i) {
			;
		}
		return i - 1;
	}

	guess = std::min(guess, len - 3);
	guess = std::max<int64_t>(guess, 1);

	/* check most likely values: guess - 1, guess, guess + 1 */
	if (key < arr[guess]) {
		if (key < arr[guess - 1]) {
			imax = guess - 1;
			/* last attempt to restrict search to items in cache */
			if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
				imin = guess - LIKELY_IN_CACHE_SIZE;
			}
		} else {
			/* key >= arr[guess - 1] */
			return guess - 1;
		}
	} else {
		/* key >= arr[guess] */
		if (key < arr[guess + 1]) {
			return guess;
		} /* key >= arr[guess + 1] */
		if (key < arr[guess + 2]) {
			return guess + 1;
		}

		/* key >= arr[guess + 2] */
		imin = guess + 2;
		/* last attempt to restrict search to items in
		 * cache */
		if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
			imax = guess + LIKELY_IN_CACHE_SIZE;
		}
	}

	/* finally, find index by bisection */
	while (imin < imax) {
		const int64_t imid = imin + ((imax - imin) / 2); // equivalent to imin + ((imax - imin) >> 1) as long as imax - imin is non-negative
		if (key >= arr[imid]) {
			imin = imid + 1;
		} else {
			imax = imid;
		}
	}
	return imin - 1;
}

AMREX_GPU_HOST_DEVICE inline void interpolate_arrays(double *x, double *y, int len, double *arr_x, const double *arr_y, int arr_len)
{
	/* Note: arr_x must be sorted in ascending order,
		and arr_len must be >= 3. */

	// check if x is within the range of arr_x
	AMREX_ASSERT(x[0] >= arr_x[0] && x[len - 1] <= arr_x[arr_len - 1]);

	int64_t j = 0;
	for (int i = 0; i < len; i++) {
		j = binary_search_with_guess(x[i], arr_x, arr_len, j);

		if (j == -1 || j == arr_len) {
			y[i] = NAN;
		} else if (j == arr_len - 1) { // NOLINT
			y[i] = arr_y[j];
		} else if (x[i] == arr_x[j]) { // avoid roundoff error
			y[i] = arr_y[j];
		} else {
			const double slope = (arr_y[j + 1] - arr_y[j]) / (arr_x[j + 1] - arr_x[j]);
			y[i] = slope * (x[i] - arr_x[j]) + arr_y[j];
		}
		AMREX_ASSERT(!std::isnan(y[i]));
	}
}

template <BoundaryPolicy Policy = BoundaryPolicy::ErrorOnBounds>
AMREX_GPU_HOST_DEVICE auto interpolate_value(double x, double const *arr_x, double const *arr_y, int arr_len) -> double
{
	/* Note: arr_x must be sorted in ascending order,
		and arr_len must be >= 3. */

	AMREX_ASSERT(!std::isnan(x));

	int64_t j = 0;
	j = binary_search_with_guess(x, arr_x, arr_len, j);

	double y = NAN;
	if (j == -1) {
		if constexpr (Policy == BoundaryPolicy::ErrorOnBounds) {
			y = NAN;
			AMREX_ASSERT(false);
			return y;
		} else if constexpr (Policy == BoundaryPolicy::Clamp) {
			y = arr_y[0]; // return first element when x is below range
			return y;
		} else if constexpr (Policy == BoundaryPolicy::Extrapolation) {
			// Linear extrapolation using the first two points
			j = 0;
		}
	} else if (j == arr_len) {
		if constexpr (Policy == BoundaryPolicy::ErrorOnBounds) {
			y = NAN;
			AMREX_ASSERT(false);
			return y;
		} else if constexpr (Policy == BoundaryPolicy::Clamp) {
			y = arr_y[arr_len - 1]; // return last element when x is above range
			return y;
		} else if constexpr (Policy == BoundaryPolicy::Extrapolation) {
			// Linear extrapolation using the last two points
			j = arr_len - 2;
		}
	}

	if (j == arr_len - 1) { // NOLINT
		y = arr_y[j];
	} else if (x == arr_x[j]) { // avoid roundoff error
		y = arr_y[j];
	} else {
		const double slope = (arr_y[j + 1] - arr_y[j]) / (arr_x[j + 1] - arr_x[j]);
		y = slope * (x - arr_x[j]) + arr_y[j];
	}

	return y;
}

#endif // INTERPOLATE_H_
