#include "ArrayUtil.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

// Bound the zero-stride reproducer even before input validation exists.
struct CopyLimited {
	static inline int copies = 0;
	CopyLimited() = default;
	CopyLimited(const CopyLimited &)
	{
		if (++copies > 20) {
			throw std::runtime_error("Zero stride repeatedly copies the first element");
		}
	}
};

auto main() -> int
{
	int failures = 0;
	const auto check = [&failures](bool passed, const char *message) {
		if (!passed) {
			std::cerr << message << '\n';
			++failures;
		}
	};
	const std::vector<int> values{0, 1, 2, 3, 4};
	check(strided_vector_from(values, 1) == values, "Stride one must preserve all values");
	check(strided_vector_from(values, 2) == std::vector<int>({0, 2, 4}), "Stride two must preserve selected values and order");
	check(strided_vector_from(values, 10) == std::vector<int>({0}), "Large stride must select first value");
	check(strided_vector_from(values, std::numeric_limits<int>::max()) == std::vector<int>({0}), "Maximum stride must select first value");
	std::vector<int> empty;
	check(strided_vector_from(empty, 1).empty(), "Empty input must return empty output");
	for (const int stride : {0, -1, std::numeric_limits<int>::min()}) {
		std::vector<CopyLimited> bounded(1);
		CopyLimited::copies = 0;
		bool rejected = false;
		try {
			(void)strided_vector_from(bounded, stride);
		} catch (const std::invalid_argument &) {
			rejected = true;
		} catch (const std::runtime_error &error) {
			std::cerr << error.what() << '\n';
		}
		check(rejected, "Nonpositive stride must throw invalid_argument before copying");
		check(CopyLimited::copies == 0, "Invalid stride must not copy input values");
		rejected = false;
		try {
			(void)strided_vector_from(empty, stride);
		} catch (const std::invalid_argument &) {
			rejected = true;
		}
		check(rejected, "Invalid stride must also be rejected for empty input");
	}
	return failures == 0 ? 0 : 1;
}
