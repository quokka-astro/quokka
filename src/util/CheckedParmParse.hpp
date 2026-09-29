#ifndef QUOKKA_CHECKED_PARM_PARSE_HPP_
#define QUOKKA_CHECKED_PARM_PARSE_HPP_

#include "AMReX_ParmParse.H"
#include "util/ParmParseOptionRegistry.hpp"
#include <string_view>
#include <utility>

namespace quokka
{

template <FixedString Prefix, FixedString Name, typename... Args> auto query(amrex::ParmParse const &pp, Args &&...args) -> decltype(auto)
{
#if !defined(QUOKKA_VALIDATE_PARM_PARSE_OPTIONS) || QUOKKA_VALIDATE_PARM_PARSE_OPTIONS
	static_assert(isRegisteredOption<Prefix, Name>(), "ParmParse option is missing from QUOKKA_PARM_PARSE_OPTIONS");
#endif
	if constexpr (Prefix.view() != "*") {
		AMREX_ASSERT(pp.getPrefix() == Prefix.view());
	}
	return pp.query(Name.view(), std::forward<Args>(args)...); // NOLINT(custom-no-direct-parmparse-access)
}

template <FixedString Prefix, FixedString Name, typename... Args> auto queryarr(amrex::ParmParse const &pp, Args &&...args) -> decltype(auto)
{
#if !defined(QUOKKA_VALIDATE_PARM_PARSE_OPTIONS) || QUOKKA_VALIDATE_PARM_PARSE_OPTIONS
	static_assert(isRegisteredOption<Prefix, Name>(), "ParmParse option is missing from QUOKKA_PARM_PARSE_OPTIONS");
#endif
	if constexpr (Prefix.view() != "*") {
		AMREX_ASSERT(pp.getPrefix() == Prefix.view());
	}
	return pp.queryarr(Name.view(), std::forward<Args>(args)...); // NOLINT(custom-no-direct-parmparse-access)
}

template <FixedString Prefix, FixedString Name, typename... Args> auto queryWithParser(amrex::ParmParse const &pp, Args &&...args) -> decltype(auto)
{
#if !defined(QUOKKA_VALIDATE_PARM_PARSE_OPTIONS) || QUOKKA_VALIDATE_PARM_PARSE_OPTIONS
	static_assert(isRegisteredOption<Prefix, Name>(), "ParmParse option is missing from QUOKKA_PARM_PARSE_OPTIONS");
#endif
	if constexpr (Prefix.view() != "*") {
		AMREX_ASSERT(pp.getPrefix() == Prefix.view());
	}
	return pp.queryWithParser(Name.view(), std::forward<Args>(args)...); // NOLINT(custom-no-direct-parmparse-access)
}

template <FixedString Prefix, FixedString Name, typename... Args> auto getarr(amrex::ParmParse const &pp, Args &&...args) -> decltype(auto)
{
#if !defined(QUOKKA_VALIDATE_PARM_PARSE_OPTIONS) || QUOKKA_VALIDATE_PARM_PARSE_OPTIONS
	static_assert(isRegisteredOption<Prefix, Name>(), "ParmParse option is missing from QUOKKA_PARM_PARSE_OPTIONS");
#endif
	if constexpr (Prefix.view() != "*") {
		AMREX_ASSERT(pp.getPrefix() == Prefix.view());
	}
	return pp.getarr(Name.view(), std::forward<Args>(args)...); // NOLINT(custom-no-direct-parmparse-access)
}

template <FixedString Prefix, FixedString Name, typename... Args> auto get(amrex::ParmParse const &pp, Args &&...args) -> decltype(auto)
{
#if !defined(QUOKKA_VALIDATE_PARM_PARSE_OPTIONS) || QUOKKA_VALIDATE_PARM_PARSE_OPTIONS
	static_assert(isRegisteredOption<Prefix, Name>(), "ParmParse option is missing from QUOKKA_PARM_PARSE_OPTIONS");
#endif
	if constexpr (Prefix.view() != "*") {
		AMREX_ASSERT(pp.getPrefix() == Prefix.view());
	}
	return pp.get(Name.view(), std::forward<Args>(args)...); // NOLINT(custom-no-direct-parmparse-access)
}

} // namespace quokka

#endif // QUOKKA_CHECKED_PARM_PARSE_HPP_
