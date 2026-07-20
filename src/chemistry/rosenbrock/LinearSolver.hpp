#ifndef QUOKKA_CHEMISTRY_ROSENBROCK_LINEAR_SOLVER_HPP_
#define QUOKKA_CHEMISTRY_ROSENBROCK_LINEAR_SOLVER_HPP_

#include <cmath>

#include "chemistry/ChemistryNetwork.hpp"

namespace quokka::chemistry::rosenbrock
{

template <int N> AMREX_GPU_HOST_DEVICE auto factor(DenseMatrix<N> &matrix, amrex::GpuArray<short, N> &pivots, bool const pivoting) noexcept -> bool
{
	for (int column = 0; column < N - 1; ++column) {
		int pivot = column;
		if (pivoting) {
			amrex::Real largest = std::abs(matrix(column, column));
			for (int row = column + 1; row < N; ++row) {
				if (std::abs(matrix(row, column)) > largest) {
					largest = std::abs(matrix(row, column));
					pivot = row;
				}
			}
		}
		pivots[column] = static_cast<short>(pivot);
		if (matrix(pivot, column) == 0.0) {
			return false;
		}
		if (pivot != column) {
			const amrex::Real temporary = matrix(pivot, column);
			matrix(pivot, column) = matrix(column, column);
			matrix(column, column) = temporary;
		}
		for (int row = column + 1; row < N; ++row) {
			matrix(row, column) *= -1.0 / matrix(column, column);
		}
		for (int j = column + 1; j < N; ++j) {
			amrex::Real value = matrix(pivot, j);
			if (pivot != column) {
				matrix(pivot, j) = matrix(column, j);
				matrix(column, j) = value;
			}
			for (int row = column + 1; row < N; ++row) {
				matrix(row, j) += value * matrix(row, column);
			}
		}
	}
	pivots[N - 1] = static_cast<short>(N - 1);
	return matrix(N - 1, N - 1) != 0.0;
}

template <int N>
AMREX_GPU_HOST_DEVICE void solve(DenseMatrix<N> const &matrix, amrex::GpuArray<short, N> const &pivots,
				 amrex::GpuArray<amrex::Real, N> &right_hand_side) noexcept
{
	for (int row = 0; row < N - 1; ++row) {
		const int pivot = pivots[row];
		const amrex::Real value = right_hand_side[pivot];
		if (pivot != row) {
			right_hand_side[pivot] = right_hand_side[row];
			right_hand_side[row] = value;
		}
		for (int i = row + 1; i < N; ++i) {
			right_hand_side[i] += value * matrix(i, row);
		}
	}
	for (int row = N - 1; row >= 0; --row) {
		right_hand_side[row] /= matrix(row, row);
		const amrex::Real value = -right_hand_side[row];
		for (int i = 0; i < row; ++i) {
			right_hand_side[i] += value * matrix(i, row);
		}
	}
}

} // namespace quokka::chemistry::rosenbrock

#endif
