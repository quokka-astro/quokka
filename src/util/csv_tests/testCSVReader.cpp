#include "util/DataTable.hpp"
#include <iostream>

template <int N> auto readAndCheck(const std::string &path) -> int
{
	const auto table = quokka::DataTable<N, 2>::CSVReader(path, quokka::TransformType::linear);
	const auto view = table.const_tables_host();
	for (int out = 0; out < 2; ++out) {
		for (int flat = 0; flat < static_cast<int>(1U << N); ++flat) {
			amrex::Real value = 0.;
			if constexpr (N == 1) {
				value = view.dataViewArrays[out](flat);
			}
			if constexpr (N == 2) {
				value = view.dataViewArrays[out](flat % 2, flat / 2);
			}
			if constexpr (N == 3) {
				value = view.dataViewArrays[out](flat % 2, (flat / 2) % 2, flat / 4);
			}
			if constexpr (N == 4) {
				value = view.dataViewArrays[out](flat % 2, (flat / 2) % 2, (flat / 4) % 2, flat / 8);
			}
			if (value != out * 100 + flat + 1) {
				std::cerr << "Incorrect data ordering or value\n";
				return 1;
			}
		}
	}
	return 0;
}

auto main(int argc, char **argv) -> int
{
	if (argc != 3) {
		return 2;
	}
	const std::string path = argv[1];
	const int dim = std::stoi(argv[2]);
	amrex::Initialize(argc, argv, false);
	int result = 1;
	switch (dim) {
		case 1:
			result = readAndCheck<1>(path);
			break;
		case 2:
			result = readAndCheck<2>(path);
			break;
		case 3:
			result = readAndCheck<3>(path);
			break;
		case 4:
			result = readAndCheck<4>(path);
			break;
		default:
			break;
	}
	amrex::Finalize();
	return result;
}
