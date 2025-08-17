#ifndef VARIABLE_TYPES_HPP_
#define VARIABLE_TYPES_HPP_

#include <string>

namespace quokka
{
namespace variable_names
{

template <bool dummy> struct base_t {
	// Empty base class for variable types
	template <class... Ts> explicit base_t(Ts &&...) {}
};

} // namespace variable_names
} // namespace quokka

// Macro to define strongly-typed variable names
#define VARIABLE(ns, varname)                                                                                                                                  \
	struct varname : public quokka::variable_names::base_t<false> {                                                                                        \
		template <class... Ts> explicit varname(Ts &&...args) : quokka::variable_names::base_t<false>(std::forward<Ts>(args)...) {}                    \
		static std::string name() { return #ns "." #varname; }                                                                                         \
	}

#endif // VARIABLE_TYPES_HPP_