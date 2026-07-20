#ifndef QUOKKA_CHEMISTRY_ROSENBROCK_TABLEAU_HPP_
#define QUOKKA_CHEMISTRY_ROSENBROCK_TABLEAU_HPP_

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

namespace quokka::chemistry::rosenbrock
{

// ROS2S from Hamkar et al. (2012). This is the production tableau used by
// Quokka's photoionization problems and the smallest useful parity target.
struct Ros2s {
	static constexpr int stages = 3;
	static constexpr amrex::Real gamma = 0.292893218813452;

	AMREX_GPU_HOST_DEVICE static constexpr auto ctime(int i) noexcept -> amrex::Real { return i == 1 ? 0.585786437626905 : 1.0; }
	AMREX_GPU_HOST_DEVICE static constexpr auto a(int i, int j) noexcept -> amrex::Real
	{
		if (i == 1 && j == 0) {
			return 2.0000000000000036;
		}
		if (i == 2 && j == 0) {
			return 6.828427124746214;
		}
		return (i == 2 && j == 1) ? 3.4142135623731007 : 0.0;
	}
	AMREX_GPU_HOST_DEVICE static constexpr auto c(int i, int j) noexcept -> amrex::Real
	{
		if (i == 1 && j == 0) {
			return -6.828427124746214;
		}
		if (i == 2 && j == 0) {
			return -10.949747468305889;
		}
		return (i == 2 && j == 1) ? -7.535533905932761 : 0.0;
	}
	AMREX_GPU_HOST_DEVICE static constexpr auto b(int i) noexcept -> amrex::Real { return i == 0 ? 6.828427124746214 : (i == 1 ? 3.414213562373101 : 1.0); }
	AMREX_GPU_HOST_DEVICE static constexpr auto error(int i) noexcept -> amrex::Real
	{
		return i == 0 ? -0.23570226039551292 : (i == 1 ? -0.23570226039551567 : -0.13807118745769906);
	}
};

// RODAS5P, the Microphysics default and the tableau used by primordial
// chemistry. Coefficients retain the upstream values; indices are zero-based.
struct Rodas5p {
	static constexpr int stages = 8;
	static constexpr amrex::Real gamma = 0.21193756319429014;

	AMREX_GPU_HOST_DEVICE static constexpr auto ctime(int i) noexcept -> amrex::Real
	{
		if (i == 1) {
			return 0.6358126895828704;
		}
		if (i == 2) {
			return 0.4095798393397535;
		}
		if (i == 3) {
			return 0.9769306725060716;
		}
		if (i == 4) {
			return 0.4288403609558664;
		}
		return i >= 5 ? 1.0 : 0.0;
	}

	AMREX_GPU_HOST_DEVICE static constexpr auto a(int i, int j) noexcept -> amrex::Real
	{
		if (i == 1 && j == 0) {
			return 3.0;
		}
		if (i == 2 && j == 0) {
			return 2.849394379747939;
		}
		if (i == 2 && j == 1) {
			return 0.45842242204463923;
		}
		if (i == 3 && j == 0) {
			return -6.954028509809101;
		}
		if (i == 3 && j == 1) {
			return 2.489845061869568;
		}
		if (i == 3 && j == 2) {
			return -10.358996098473584;
		}
		if (i == 4 && j == 0) {
			return 2.8029986275628964;
		}
		if (i == 4 && j == 1) {
			return 0.5072464736228206;
		}
		if (i == 4 && j == 2) {
			return -0.3988312541770524;
		}
		if (i == 4 && j == 3) {
			return -0.04721187230404641;
		}
		if (i >= 5 && j == 0) {
			return -7.502846399306121;
		}
		if (i >= 5 && j == 1) {
			return 2.561846144803919;
		}
		if (i >= 5 && j == 2) {
			return -11.627539656261098;
		}
		if (i >= 5 && j == 3) {
			return -0.18268767659942256;
		}
		if (i >= 5 && j == 4) {
			return 0.030198172008377946;
		}
		if (i >= 6 && j == 5) {
			return 1.0;
		}
		return (i == 7 && j == 6) ? 1.0 : 0.0;
	}

	AMREX_GPU_HOST_DEVICE static constexpr auto c(int i, int j) noexcept -> amrex::Real
	{
		if (i == 1 && j == 0) {
			return -14.155112264123755;
		}
		if (i == 2 && j == 0) {
			return -17.97296035885952;
		}
		if (i == 2 && j == 1) {
			return -2.859693295451294;
		}
		if (i == 3 && j == 0) {
			return 147.12150275711716;
		}
		if (i == 3 && j == 1) {
			return -1.41221402718213;
		}
		if (i == 3 && j == 2) {
			return 71.68940251302358;
		}
		if (i == 4 && j == 0) {
			return 165.43517024871676;
		}
		if (i == 4 && j == 1) {
			return -0.4592823456491126;
		}
		if (i == 4 && j == 2) {
			return 42.90938336958603;
		}
		if (i == 4 && j == 3) {
			return -5.961986721573306;
		}
		if (i == 5 && j == 0) {
			return 24.854864614690072;
		}
		if (i == 5 && j == 1) {
			return -3.0009227002832186;
		}
		if (i == 5 && j == 2) {
			return 47.4931110020768;
		}
		if (i == 5 && j == 3) {
			return 5.5814197821558125;
		}
		if (i == 5 && j == 4) {
			return -0.6610691825249471;
		}
		if (i == 6 && j == 0) {
			return 30.91273214028599;
		}
		if (i == 6 && j == 1) {
			return -3.1208243349937974;
		}
		if (i == 6 && j == 2) {
			return 77.79954646070892;
		}
		if (i == 6 && j == 3) {
			return 34.28646028294783;
		}
		if (i == 6 && j == 4) {
			return -19.097331116725623;
		}
		if (i == 6 && j == 5) {
			return -28.087943162872662;
		}
		if (i == 7 && j == 0) {
			return 37.80277123390563;
		}
		if (i == 7 && j == 1) {
			return -3.2571969029072276;
		}
		if (i == 7 && j == 2) {
			return 112.26918849496327;
		}
		if (i == 7 && j == 3) {
			return 66.9347231244047;
		}
		if (i == 7 && j == 4) {
			return -40.06618937091002;
		}
		if (i == 7 && j == 5) {
			return -54.66780262877968;
		}
		return (i == 7 && j == 6) ? -9.48861652309627 : 0.0;
	}

	AMREX_GPU_HOST_DEVICE static constexpr auto b(int i) noexcept -> amrex::Real
	{
		if (i == 0) {
			return -7.502846399306121;
		}
		if (i == 1) {
			return 2.561846144803919;
		}
		if (i == 2) {
			return -11.627539656261098;
		}
		if (i == 3) {
			return -0.18268767659942256;
		}
		if (i == 4) {
			return 0.030198172008377946;
		}
		return 1.0;
	}

	AMREX_GPU_HOST_DEVICE static constexpr auto error(int i) noexcept -> amrex::Real { return i == 7 ? 1.0 : 0.0; }
};

} // namespace quokka::chemistry::rosenbrock

#endif
