#ifndef TYPELIST_HPP_
#define TYPELIST_HPP_

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace quokka
{

// Convenience struct for holding a variadic pack of types
// and providing compile time indexing into that pack as
// well as the ability to get the index of a given type within
// the pack. Functions are available below for compile time
// concatenation of TypeLists
template <class... Args> struct TypeList {
	using types = std::tuple<Args...>;

	static constexpr std::size_t n_types{sizeof...(Args)};

	template <std::size_t I> using type = typename std::tuple_element<I, types>::type;

	template <std::size_t... Idxs> using sublist = TypeList<type<Idxs>...>;

	template <class T, std::size_t I = 0> static constexpr std::size_t GetIdx()
	{
		static_assert(I < n_types, "Type is not present in TypeList.");
		if constexpr (std::is_same_v<T, type<I>>) {
			return I;
		} else {
			return GetIdx<T, I + 1>();
		}
	}

	template <class F> static void IterateTypes(F func) { (func(Args()), ...); }

      private:
	template <std::size_t Start, std::size_t End> static auto ContinuousSublist()
	{
		return ContinuousSublistImpl<Start>(std::make_index_sequence<End - Start + 1>());
	}
	template <std::size_t Start, std::size_t... Is> static auto ContinuousSublistImpl(std::index_sequence<Is...>) { return sublist<(Start + Is)...>(); }

      public:
	template <std::size_t Start, std::size_t End> using continuous_sublist = decltype(ContinuousSublist<Start, End>());
};

// Helper to check if a type is in a TypeList
template <class T, class TL> struct TypeListContains;

template <class T, class... Args> struct TypeListContains<T, TypeList<Args...>> {
	static constexpr bool value = (std::is_same_v<T, Args> || ...);
};

template <class T, class TL> inline constexpr bool TypeListContains_v = TypeListContains<T, TL>::value;

// Concatenate two TypeLists
template <class TL1, class TL2> struct TypeListCat;

template <class... Args1, class... Args2> struct TypeListCat<TypeList<Args1...>, TypeList<Args2...>> {
	using type = TypeList<Args1..., Args2...>;
};

template <class TL1, class TL2> using TypeListCat_t = typename TypeListCat<TL1, TL2>::type;

// Extract types from TL1 that are in TL2
template <class TL1, class TL2> struct TypeListIntersection;

template <class TL2> struct TypeListIntersection<TypeList<>, TL2> {
	using type = TypeList<>;
};

template <class Head, class... Tail, class TL2> struct TypeListIntersection<TypeList<Head, Tail...>, TL2> {
	using tail_intersection = typename TypeListIntersection<TypeList<Tail...>, TL2>::type;
	using type = std::conditional_t<TypeListContains_v<Head, TL2>, TypeListCat_t<TypeList<Head>, tail_intersection>, tail_intersection>;
};

template <class TL1, class TL2> using TypeListIntersection_t = typename TypeListIntersection<TL1, TL2>::type;

} // namespace quokka

#endif // TYPELIST_HPP_