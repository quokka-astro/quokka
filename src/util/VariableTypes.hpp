#ifndef VARIABLE_TYPES_HPP_
#define VARIABLE_TYPES_HPP_

#include <string>

namespace quokka::variable_names
{

template <bool dummy> struct base_t {
	// Empty base class for variable types
	template <class... Ts> explicit base_t(Ts &&... /*unused*/) {}
};

} // namespace quokka::variable_names

// Macro to define strongly-typed variable names
#define VARIABLE(ns, varname)                                                                                                                                  \
	struct varname : public quokka::variable_names::base_t<false> {                                                                                        \
		template <class... Ts> explicit varname(Ts &&...args) : quokka::variable_names::base_t<false>(std::forward<Ts>(args)...) {}                    \
		static auto name() -> std::string { return #ns "." #varname; }                                                                                 \
		static constexpr bool is_multi_component = false;                                                                                              \
	}

// Macro to define multi-component variables with N components
#define MULTI_VARIABLE(ns, varname, N)                                                                                                                         \
	template <int Component = -1> struct varname : public quokka::variable_names::base_t<false> {                                                          \
		static constexpr int num_components = N;                                                                                                       \
		static constexpr int component_index = Component;                                                                                              \
		template <class... Ts> explicit varname(Ts &&...args) : quokka::variable_names::base_t<false>(std::forward<Ts>(args)...) {}                    \
		static auto name() -> std::string                                                                                                              \
		{                                                                                                                                              \
			if constexpr (Component >= 0) {                                                                                                        \
				return #ns "." #varname "[" + std::to_string(Component) + "]";                                                                 \
			} else {                                                                                                                               \
				return #ns "." #varname;                                                                                                       \
			}                                                                                                                                      \
		}                                                                                                                                              \
		static constexpr bool is_multi_component = true;                                                                                              \
	};                                                                                                                                                     \
	using varname##_all = varname<-1>

// Macro to create strong type alias for specific component
#define COMPONENT_ALIAS(base_type, component_idx, alias_name) using alias_name = base_type<component_idx>

#endif // VARIABLE_TYPES_HPP_