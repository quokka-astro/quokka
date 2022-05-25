#ifndef ARRAYUTIL_HPP_ // NOLINT
#define ARRAYUTIL_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file ArrayUtil.hpp
/// \brief Implements functions to manipulate arrays (CPU only).

#include "AMReX_Array4.H"
#include "AMReX_REAL.H"
#include <vector>

template <typename T>
auto strided_vector_from(std::vector<T> &v, int stride) -> std::vector<T> {
  std::vector<T> strided_v;
  for (std::size_t i = 0; i < v.size(); i += stride) {
    strided_v.push_back(v[i]);
  }
  return strided_v; // move semantics implied
}

#endif // ARRAYUTIL_HPP_
