// from:
// https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <cstdlib>
#include <memory>
#include <string>

namespace quokka {
template <class T> std::string type_name() {
  using TR = typename std::remove_reference<T>::type;
  std::unique_ptr<char, void (*)(void *)> own(
#ifndef _MSC_VER
      abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
#else
      nullptr,
#endif
      std::free);
  std::string my_str = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const<TR>::value) {
    my_str += " const";
  }
  if (std::is_volatile<TR>::value) {
    my_str += " volatile";
  }
  if (std::is_lvalue_reference<T>::value) {
    my_str += "&";
  } else if (std::is_rvalue_reference<T>::value) {
    my_str += "&&";
  }
  return my_str;
}
} // namespace quokka

// EXAMPLE:
// std::cout << "\n"
//           << "check_type(phase) : " << quokka::type_name<decltype(phase)>() << "\n";
