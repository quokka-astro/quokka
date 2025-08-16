#ifndef TYPELIST_HPP_ // NOLINT
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

	template <class F> static void IterateTypes(F func)
	{
		(func(Args()), ...);
	}

      private:
	template <std::size_t Start, std::size_t End> static auto ContinuousSublist()
	{
		return ContinuousSublistImpl<Start>(std::make_index_sequence<End - Start + 1>());
	}
	template <std::size_t Start, std::size_t... Is> static auto ContinuousSublistImpl(std::index_sequence<Is...>)
	{
		return sublist<(Start + Is)...>();
	}

      public:
	template <std::size_t Start, std::size_t End> using continuous_sublist = decltype(ContinuousSublist<Start, End>());
};

// Concatenate two TypeLists
template <class... Args1, class... Args2> constexpr auto operator+(TypeList<Args1...> /*unused*/, TypeList<Args2...> /*unused*/)
{
	return TypeList<Args1..., Args2...>{};
}

// Helper type alias for concatenating TypeLists
template <class TL1, class TL2> using ConcatTypeLists = decltype(std::declval<TL1>() + std::declval<TL2>());

} // namespace quokka

#endif // TYPELIST_HPP_