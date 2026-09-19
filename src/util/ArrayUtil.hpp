#ifndef ARRAYUTIL_HPP_ // NOLINT
#define ARRAYUTIL_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ArrayUtil.hpp
/// \brief Implements functions to manipulate arrays (CPU only).

#include <stdexcept>
#include <vector>

// The stride must be positive, including for an empty input vector.
template <typename T> auto strided_vector_from(const std::vector<T> &v, int stride) -> std::vector<T>
{
	if (stride <= 0) {
		throw std::invalid_argument("strided_vector_from: stride must be positive");
	}
	std::vector<T> strided_v;
	for (std::size_t i = 0; i < v.size(); i += static_cast<std::size_t>(stride)) {
		strided_v.push_back(v[i]);
	}
	return strided_v; // move semantics implied
}

#endif // ARRAYUTIL_HPP_
