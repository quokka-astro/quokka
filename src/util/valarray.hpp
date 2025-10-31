#ifndef VALARRAY_HPP_
#define VALARRAY_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file valarray.hpp
/// \brief A container for a vector with addition, multiplication with expression templates
/// (This is necessary because std::valarray is not defined in CUDA C++!)

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

// library headers
#include "AMReX_Extension.H"
#include <AMReX_GpuQualifiers.H>

namespace quokka
{
namespace detail
{
template <typename T> using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename Expr> concept Expression = requires(Expr const &expr, size_t idx)
{
	typename remove_cvref_t<Expr>::value_type;
	{remove_cvref_t<Expr>::extent}->std::convertible_to<int>;
	{expr[idx]}->std::convertible_to<typename remove_cvref_t<Expr>::value_type>;
};

template <typename Expr> constexpr int expr_extent_v = static_cast<int>(remove_cvref_t<Expr>::extent);

template <typename Expr> using expr_value_t = typename remove_cvref_t<Expr>::value_type;

template <typename Expr, typename T, int d>
concept CompatibleExpr = Expression<Expr> && std::convertible_to<expr_value_t<Expr>, T> && (expr_extent_v<Expr> == d);

template <typename LHS, typename RHS> concept BinaryCompatible = Expression<LHS> && Expression<RHS> && (expr_extent_v<LHS> == expr_extent_v<RHS>);

struct AddOp {
	template <typename L, typename R> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto apply(L const &lhs, R const &rhs) -> decltype(lhs + rhs)
	{
		return lhs + rhs;
	}
};
struct SubOp {
	template <typename L, typename R> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto apply(L const &lhs, R const &rhs) -> decltype(lhs - rhs)
	{
		return lhs - rhs;
	}
};
struct MulOp {
	template <typename L, typename R> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto apply(L const &lhs, R const &rhs) -> decltype(lhs * rhs)
	{
		return lhs * rhs;
	}
};
struct DivOp {
	template <typename L, typename R> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto apply(L const &lhs, R const &rhs) -> decltype(lhs / rhs)
	{
		return lhs / rhs;
	}
};

template <typename T, int d> struct ScalarExpr {
	using value_type = T;
	static constexpr int extent = d;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit ScalarExpr(T scalar) : value(scalar) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[]([[maybe_unused]] size_t index) const -> T { return value; }

      private:
	T value;
};

template <typename Op, Expression LHS, Expression RHS> requires BinaryCompatible<LHS, RHS> struct BinaryExpr {
	using lhs_type = remove_cvref_t<LHS>;
	using rhs_type = remove_cvref_t<RHS>;

	using value_type = std::decay_t<decltype(Op::apply(std::declval<typename lhs_type::value_type>(), std::declval<typename rhs_type::value_type>()))>;
	static constexpr int extent = expr_extent_v<LHS>;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE BinaryExpr(LHS lhs_in, RHS rhs_in) : lhs(lhs_in), rhs(rhs_in) {}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) const -> value_type { return Op::apply(lhs[i], rhs[i]); }

	LHS lhs;
	RHS rhs;
};

template <typename Op, Expression LHS, Expression RHS>
requires BinaryCompatible<LHS, RHS> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto make_binary_expr(LHS lhs, RHS rhs) -> BinaryExpr<Op, LHS, RHS>
{
	return BinaryExpr<Op, LHS, RHS>(lhs, rhs);
}

template <Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto make_scalar_expr(Scalar const &scalar)
{
	using value_type = expr_value_t<Expr>;
	return ScalarExpr<value_type, expr_extent_v<Expr>>(static_cast<value_type>(scalar));
}

} // namespace detail

template <typename T, int d> class valarray
{
      public:
	using value_type = T;
	static constexpr int extent = d;
	static_assert(d >= 0, "valarray extent must be non-negative");
	static constexpr size_t extent_size = static_cast<size_t>(extent);

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray() = default;
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(valarray const &) = default;
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(valarray &&) noexcept = default;
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator=(valarray const &) -> valarray & = default;
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator=(valarray &&) noexcept -> valarray & = default;
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ~valarray() = default;

	// we *want* implicit construction from initializer lists for valarrays,
	// (although not cppcore-compliant)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(std::initializer_list<T> list) // NOLINT
	{
		const size_t max_count = std::min(list.size(), extent_size);

		T const *input = std::data(list); // requires nvcc to be in C++17 mode (or newer)! (if it fails, the
						  // compiler flags are wrong, probably due to a CMake issue.)

		for (size_t i = 0; i < max_count; ++i) {
			values[i] = input[i]; // NOLINT
		}

		// it is undefined behavior to not fully initialize an object!
		// (this does happen in practice with gcc 10+, which optimizes out ctor
		//  calls if an object is unused before a subsequent assignment.)
		for (size_t i = max_count; i < extent_size; ++i) {
			values[i] = default_value;
		}
	}

template <detail::CompatibleExpr<T, d> Expr>
// NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
requires(!std::same_as<detail::remove_cvref_t<Expr>, valarray>) AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(Expr const &expr)
	{
		assign_from(expr);
	}

	template <detail::CompatibleExpr<T, d> Expr> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator=(Expr const &expr) -> valarray &
	{
		assign_from(expr);
		return *this;
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) -> T & { return values[i]; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) const -> T { return values[i]; }

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto size() const -> size_t { return extent_size; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void fillin(T const &scalar)
	{
		for (size_t i = 0; i < extent_size; ++i) {
			values[i] = scalar;
		}
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto hasnan() const -> bool
	{
		for (size_t i = 0; i < extent_size; ++i) {
			if (std::isnan(values[i])) {
				return true;
			}
		}
		return false;
	}

	template <detail::CompatibleExpr<T, d> Expr> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+=(Expr const &expr) -> valarray &
	{
		for (size_t i = 0; i < extent_size; ++i) {
			values[i] += static_cast<T>(expr[i]);
		}
		return *this;
	}

	template <detail::CompatibleExpr<T, d> Expr> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-=(Expr const &expr) -> valarray &
	{
		for (size_t i = 0; i < extent_size; ++i) {
			values[i] -= static_cast<T>(expr[i]);
		}
		return *this;
	}

	template <typename Scalar>
	requires std::convertible_to<Scalar, T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*=(Scalar const &scalar) -> valarray &
	{
		for (size_t i = 0; i < extent_size; ++i) {
			values[i] *= static_cast<T>(scalar);
		}
		return *this;
	}

	template <typename Scalar>
	requires std::convertible_to<Scalar, T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/=(Scalar const &scalar) -> valarray &
	{
		for (size_t i = 0; i < extent_size; ++i) {
			values[i] /= static_cast<T>(scalar);
		}
		return *this;
	}

      private:
	template <typename Expr> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void assign_from(Expr const &expr)
	{
		for (size_t i = 0; i < extent_size; ++i) {
			values[i] = static_cast<T>(expr[i]);
		}
	}

	T values[d]; // NOLINT
	static constexpr T default_value = 0;
};

template <detail::Expression Expr1, detail::Expression Expr2>
requires detail::BinaryCompatible<Expr1, Expr2> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(Expr1 const &lhs, Expr2 const &rhs)
{
	return detail::make_binary_expr<detail::AddOp>(lhs, rhs);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(Expr const &expr, Scalar const &scalar)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::AddOp>(expr, scalar_expr);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(Scalar const &scalar, Expr const &expr)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::AddOp>(scalar_expr, expr);
}

template <detail::Expression Expr1, detail::Expression Expr2>
requires detail::BinaryCompatible<Expr1, Expr2> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(Expr1 const &lhs, Expr2 const &rhs)
{
	return detail::make_binary_expr<detail::SubOp>(lhs, rhs);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(Expr const &expr, Scalar const &scalar)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::SubOp>(expr, scalar_expr);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(Scalar const &scalar, Expr const &expr)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::SubOp>(scalar_expr, expr);
}

template <detail::Expression Expr1, detail::Expression Expr2>
requires detail::BinaryCompatible<Expr1, Expr2> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(Expr1 const &lhs, Expr2 const &rhs)
{
	return detail::make_binary_expr<detail::MulOp>(lhs, rhs);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(Expr const &expr, Scalar const &scalar)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::MulOp>(expr, scalar_expr);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(Scalar const &scalar, Expr const &expr)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::MulOp>(scalar_expr, expr);
}

template <detail::Expression Expr1, detail::Expression Expr2>
requires detail::BinaryCompatible<Expr1, Expr2> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(Expr1 const &lhs, Expr2 const &rhs)
{
	return detail::make_binary_expr<detail::DivOp>(lhs, rhs);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(Expr const &expr, Scalar const &scalar)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::DivOp>(expr, scalar_expr);
}

template <detail::Expression Expr, typename Scalar>
requires std::convertible_to<Scalar, detail::expr_value_t<Expr>> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(Scalar const &scalar, Expr const &expr)
{
	auto scalar_expr = detail::make_scalar_expr<Expr>(scalar);
	return detail::make_binary_expr<detail::DivOp>(scalar_expr, expr);
}

template <detail::Expression Expr>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto abs(Expr const &expr) -> valarray<typename detail::expr_value_t<Expr>, detail::expr_extent_v<Expr>>
{
	using value_type = typename detail::expr_value_t<Expr>;
	constexpr int extent = detail::expr_extent_v<Expr>;
	valarray<value_type, extent> abs_v;
	for (size_t i = 0; i < abs_v.size(); ++i) {
		abs_v[i] = std::abs(static_cast<value_type>(expr[i]));
	}
	return abs_v;
}

template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto min(valarray<T, d> const &v) -> T
{
	static_assert(d >= 1);
	T min_val = v[0]; // v must have at least 1 element

	for (size_t i = 0; i < v.size(); ++i) {
		min_val = std::min(min_val, v[i]);
	}
	return min_val;
}

template <detail::Expression Expr> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto min(Expr const &expr) -> typename detail::expr_value_t<Expr>
{
	using value_type = typename detail::expr_value_t<Expr>;
	constexpr int extent = detail::expr_extent_v<Expr>;
	static_assert(extent >= 1);
	auto min_val = static_cast<value_type>(expr[0]);
	for (int i = 1; i < extent; ++i) {
		min_val = std::min(min_val, static_cast<value_type>(expr[static_cast<size_t>(i)]));
	}
	return min_val;
}

template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto max(valarray<T, d> const &v) -> T
{
	static_assert(d >= 1);
	T max_val = v[0]; // v must have at least 1 element

	for (size_t i = 0; i < v.size(); ++i) {
		max_val = std::max(max_val, v[i]);
	}
	return max_val;
}

template <detail::Expression Expr> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto max(Expr const &expr) -> typename detail::expr_value_t<Expr>
{
	using value_type = typename detail::expr_value_t<Expr>;
	constexpr int extent = detail::expr_extent_v<Expr>;
	static_assert(extent >= 1);
	auto max_val = static_cast<value_type>(expr[0]);
	for (int i = 1; i < extent; ++i) {
		max_val = std::max(max_val, static_cast<value_type>(expr[static_cast<size_t>(i)]));
	}
	return max_val;
}

template <detail::Expression Expr> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sum(Expr const &expr) -> typename detail::expr_value_t<Expr>
{
	using value_type = typename detail::expr_value_t<Expr>;
	constexpr int extent = detail::expr_extent_v<Expr>;
	auto sum_val = static_cast<value_type>(0);
	for (int i = 0; i < extent; ++i) {
		sum_val += static_cast<value_type>(expr[i]);
	}
	return sum_val;
}

template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(valarray<T, d> const &a, valarray<T, d> const &b) -> valarray<bool, d>
{
	valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] > b[i];
	}
	return comp;
}

template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(valarray<T, d> const &a, T const &scalar) -> valarray<bool, d>
{
	valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] > scalar;
	}
	return comp;
}

template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(valarray<T, d> const &a, valarray<T, d> const &b) -> valarray<bool, d>
{
	valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] < b[i];
	}
	return comp;
}

template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(valarray<T, d> const &a, T const &scalar) -> valarray<bool, d>
{
	valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] < scalar;
	}
	return comp;
}

} // namespace quokka

#endif // VALARRAY_HPP_
