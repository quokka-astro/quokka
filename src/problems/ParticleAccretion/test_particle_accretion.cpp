/// \file particle_sink_accretion.cpp
/// \brief Defines a test problem for Bondi-Hoyle accretion.

#include "test_particle_accretion.hpp"
#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <gcem.hpp>

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

using amrex::Real;

struct AccretionProblem {
};

// In this test, r_B = 0.1214 pc. Ball radius is R = 32 r_B, box half-size is 2 R = 64 r_B = 7.7696 pc = 2.397448054e+19 cm

constexpr double rho0 = C::m_p;
constexpr double T0 = 10.0;
constexpr double mu = 2.33 * C::m_p;
constexpr double k_B = C::k_B;
constexpr double cs0 = gcem::sqrt(k_B * T0 / mu);

AMREX_GPU_MANAGED double M_star_in_Msun = 1.0; // NOLINT

// constexpr double r_B = C::Gconst * C::M_solar / (cs0 * cs0);

template <> struct Particle_Traits<AccretionProblem> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Sink;
};

template <> struct quokka::EOS_Traits<AccretionProblem> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = cs0;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<AccretionProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<AccretionProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<AccretionProblem> {
	std::vector<Real> time;
	std::vector<Real> Mstar;
};

template <> void QuokkaSimulation<AccretionProblem>::createInitialSinkParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	SinkParticles->SetVerbose(1);
	SinkParticles->InitFromAsciiFile("sink.txt", nreal_extra, nullptr);

	// manually set particle mass to M_star_in_Msun * C::M_solar
	for (auto &kv : SinkParticles->GetParticles()) {
		for (auto &ikv : kv) {
			auto &particle_array = ikv.second.GetArrayOfStructs();
			const int np = particle_array.numParticles();

			if (np == 0) {
				continue;
			}

			auto *pdata = particle_array().data();

			// Launch GPU kernel to set integer components
			amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
				auto &p = pdata[i]; // NOLINT
				p.rdata(0) = M_star_in_Msun * C::M_solar;
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Gpu::DeviceVector<double> x_array = {
	    0.1,  0.2,	0.3,  0.4,  0.5,  0.6,	0.7,  0.8,  0.9,  1.0,	1.1,  1.2,  1.3,  1.4,	1.5,  1.6,  1.7,  1.8,	1.9,  2.0,  2.1,  2.2,	2.3,  2.4,
	    2.5,  2.6,	2.7,  2.8,  2.9,  3.0,	3.1,  3.2,  3.3,  3.4,	3.5,  3.6,  3.7,  3.8,	3.9,  4.0,  4.1,  4.2,	4.3,  4.4,  4.5,  4.6,	4.7,  4.8,
	    4.9,  5.0,	5.1,  5.2,  5.3,  5.4,	5.5,  5.6,  5.7,  5.8,	5.9,  6.0,  6.1,  6.2,	6.3,  6.4,  6.5,  6.6,	6.7,  6.8,  6.9,  7.0,	7.1,  7.2,
	    7.3,  7.4,	7.5,  7.6,  7.7,  7.8,	7.9,  8.0,  8.1,  8.2,	8.3,  8.4,  8.5,  8.6,	8.7,  8.8,  8.9,  9.0,	9.1,  9.2,  9.3,  9.4,	9.5,  9.6,
	    9.7,  9.8,	9.9,  10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0,
	    12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4,
	    14.5, 14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8,
	    16.9, 17.0, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18.0, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19.0, 19.1, 19.2,
	    19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 21.2, 21.3, 21.4, 21.5, 21.6,
	    21.7, 21.8, 21.9, 22.0, 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9, 23.0, 23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.7, 23.8, 23.9, 24.0,
	    24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 25.0, 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7, 25.8, 25.9, 26.0, 26.1, 26.2, 26.3, 26.4,
	    26.5, 26.6, 26.7, 26.8, 26.9, 27.0, 27.1, 27.2, 27.3, 27.4, 27.5, 27.6, 27.7, 27.8, 27.9, 28.0, 28.1, 28.2, 28.3, 28.4, 28.5, 28.6, 28.7, 28.8,
	    28.9, 29.0, 29.1, 29.2, 29.3, 29.4, 29.5, 29.6, 29.7, 29.8, 29.9, 30.0, 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7, 30.8, 30.9, 31.0, 31.1, 31.2,
	    31.3, 31.4, 31.5, 31.6, 31.7, 31.8, 31.9, 32.0, 32.1, 32.2, 32.3, 32.4, 32.5, 32.6, 32.7, 32.8, 32.9, 33.0, 33.1, 33.2, 33.3, 33.4, 33.5, 33.6,
	    33.7, 33.8, 33.9, 34.0, 34.1, 34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 34.9, 35.0, 35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9, 36.0,
	    36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7, 36.8, 36.9, 37.0, 37.1, 37.2, 37.3, 37.4, 37.5, 37.6, 37.7, 37.8, 37.9, 38.0, 38.1, 38.2, 38.3, 38.4,
	    38.5, 38.6, 38.7, 38.8, 38.9, 39.0, 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 39.7, 39.8, 39.9, 40.0};

	const amrex::Gpu::DeviceVector<double> v_array = {
	    3.6246016521, 2.2204100654, 1.6017268124, 1.2400559901, 1.0000000000, 0.8285919344, 0.7002183926, 0.6007682620, 0.5217455719, 0.4576951453,
	    0.4049394880, 0.3609033048, 0.3237272303, 0.2920347742, 0.2647854007, 0.2411784669, 0.2205884513, 0.2025201184, 0.1865767723, 0.1724373312,
	    0.1598394833, 0.1485671192, 0.1384408273, 0.1293106171, 0.1210502853, 0.1135530111, 0.1067278804, 0.1004971191, 0.0947938750, 0.0895604254,
	    0.0847467208, 0.0803091917, 0.0762097679, 0.0724150665, 0.0688957164, 0.0656257941, 0.0625823496, 0.0597450070, 0.0570956260, 0.0546180142,
	    0.0522976823, 0.0501216333, 0.0480781825, 0.0461568018, 0.0443479850, 0.0426431318, 0.0410344458, 0.0395148464, 0.0380778915, 0.0367177096,
	    0.0354289404, 0.0342066826, 0.0330464473, 0.0319441172, 0.0308959105, 0.0298983486, 0.0289482274, 0.0280425916, 0.0271787123, 0.0263540663,
	    0.0255663176, 0.0248133015, 0.0240930093, 0.0234035753, 0.0227432646, 0.0221104623, 0.0215036640, 0.0209214662, 0.0203625591, 0.0198257185,
	    0.0193097995, 0.0188137306, 0.0183365078, 0.0178771897, 0.0174348928, 0.0170087876, 0.0165980942, 0.0162020791, 0.0158200521, 0.0154513628,
	    0.0150953982, 0.0147515800, 0.0144193625, 0.0140982301, 0.0137876958, 0.0134872986, 0.0131966025, 0.0129151948, 0.0126426843, 0.0123787002,
	    0.0121228908, 0.0118749223, 0.0116344778, 0.0114012563, 0.0111749716, 0.0109553516, 0.0107421372, 0.0105350819, 0.0103339511, 0.0101385210,
	    0.0099485783, 0.0097639197, 0.0095843511, 0.0094096874, 0.0092397515, 0.0090743746, 0.0089133951, 0.0087566586, 0.0086040174, 0.0084553301,
	    0.0083104615, 0.0081692819, 0.0080316673, 0.0078974987, 0.0077666621, 0.0076390481, 0.0075145518, 0.0073930726, 0.0072745137, 0.0071587824,
	    0.0070457895, 0.0069354494, 0.0068276797, 0.0067224012, 0.0066195377, 0.0065190161, 0.0064207658, 0.0063247188, 0.0062308099, 0.0061389761,
	    0.0060491567, 0.0059612933, 0.0058753295, 0.0057912109, 0.0057088852, 0.0056283018, 0.0055494119, 0.0054721683, 0.0053965257, 0.0053224400,
	    0.0052498690, 0.0051787715, 0.0051091080, 0.0050408402, 0.0049739310, 0.0049083447, 0.0048440466, 0.0047810033, 0.0047191823, 0.0046585521,
	    0.0045990826, 0.0045407441, 0.0044835083, 0.0044273474, 0.0043722349, 0.0043181447, 0.0042650518, 0.0042129317, 0.0041617609, 0.0041115164,
	    0.0040621761, 0.0040137184, 0.0039661223, 0.0039193676, 0.0038734346, 0.0038283040, 0.0037839574, 0.0037403767, 0.0036975444, 0.0036554433,
	    0.0036140571, 0.0035733695, 0.0035333650, 0.0034940284, 0.0034553449, 0.0034173000, 0.0033798799, 0.0033430709, 0.0033068598, 0.0032712337,
	    0.0032361801, 0.0032016868, 0.0031677419, 0.0031343339, 0.0031014514, 0.0030690836, 0.0030372198, 0.0030058494, 0.0029749625, 0.0029445492,
	    0.0029145997, 0.0028851047, 0.0028560551, 0.0028274419, 0.0027992565, 0.0027714904, 0.0027441352, 0.0027171830, 0.0026906258, 0.0026644560,
	    0.0026386661, 0.0026132487, 0.0025881968, 0.0025635033, 0.0025391614, 0.0025151646, 0.0024915062, 0.0024681801, 0.0024451799, 0.0024224997,
	    0.0024001335, 0.0023780756, 0.0023563203, 0.0023348621, 0.0023136957, 0.0022928157, 0.0022722170, 0.0022518947, 0.0022318437, 0.0022120593,
	    0.0021925368, 0.0021732715, 0.0021542590, 0.0021354949, 0.0021169748, 0.0020986946, 0.0020806501, 0.0020628372, 0.0020452521, 0.0020278909,
	    0.0020107497, 0.0019938250, 0.0019771130, 0.0019606102, 0.0019443131, 0.0019282184, 0.0019123227, 0.0018966226, 0.0018811151, 0.0018657970,
	    0.0018506653, 0.0018357168, 0.0018209487, 0.0018063580, 0.0017919420, 0.0017776979, 0.0017636229, 0.0017497144, 0.0017359697, 0.0017223864,
	    0.0017089618, 0.0016956935, 0.0016825792, 0.0016696164, 0.0016568028, 0.0016441361, 0.0016316141, 0.0016192346, 0.0016069954, 0.0015948945,
	    0.0015829297, 0.0015710990, 0.0015594005, 0.0015478321, 0.0015363920, 0.0015250782, 0.0015138889, 0.0015028222, 0.0014918765, 0.0014810499,
	    0.0014703407, 0.0014597473, 0.0014492679, 0.0014389009, 0.0014286448, 0.0014184979, 0.0014084587, 0.0013985257, 0.0013886974, 0.0013789724,
	    0.0013693491, 0.0013598262, 0.0013504023, 0.0013410760, 0.0013318459, 0.0013227109, 0.0013136695, 0.0013047204, 0.0012958625, 0.0012870945,
	    0.0012784151, 0.0012698233, 0.0012613177, 0.0012528974, 0.0012445610, 0.0012363076, 0.0012281360, 0.0012200451, 0.0012120339, 0.0012041014,
	    0.0011962465, 0.0011884681, 0.0011807654, 0.0011731373, 0.0011655829, 0.0011581013, 0.0011506914, 0.0011433524, 0.0011360834, 0.0011288835,
	    0.0011217518, 0.0011146875, 0.0011076897, 0.0011007575, 0.0010938903, 0.0010870871, 0.0010803471, 0.0010736697, 0.0010670539, 0.0010604991,
	    0.0010540045, 0.0010475694, 0.0010411931, 0.0010348748, 0.0010286138, 0.0010224094, 0.0010162610, 0.0010101679, 0.0010041295, 0.0009981450,
	    0.0009922138, 0.0009863354, 0.0009805090, 0.0009747341, 0.0009690101, 0.0009633363, 0.0009577123, 0.0009521373, 0.0009466109, 0.0009411324,
	    0.0009357013, 0.0009303172, 0.0009249793, 0.0009196873, 0.0009144406, 0.0009092386, 0.0009040808, 0.0008989669, 0.0008938962, 0.0008888683,
	    0.0008838826, 0.0008789389, 0.0008740364, 0.0008691749, 0.0008643538, 0.0008595727, 0.0008548312, 0.0008501288, 0.0008454650, 0.0008408396,
	    0.0008362520, 0.0008317018, 0.0008271887, 0.0008227122, 0.0008182720, 0.0008138676, 0.0008094986, 0.0008051648, 0.0008008656, 0.0007966008,
	    0.0007923700, 0.0007881727, 0.0007840088, 0.0007798777, 0.0007757792, 0.0007717129, 0.0007676786, 0.0007636757, 0.0007597041, 0.0007557634,
	    0.0007518533, 0.0007479734, 0.0007441235, 0.0007403032, 0.0007365123, 0.0007327504, 0.0007290173, 0.0007253126, 0.0007216361, 0.0007179875,
	    0.0007143665, 0.0007107727, 0.0007072061, 0.0007036662, 0.0007001528, 0.0006966657, 0.0006932046, 0.0006897692, 0.0006863592, 0.0006829745};

	// set initial conditions
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double r_B = C::Gconst * M_star_in_Msun * C::M_solar / std::pow(cs0, 2);

	// assert that the box size is bigger than sphere_radius_over_r_B * r_B
	// AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(prob_lo[0]) > sphere_radius_over_r_B * r_B, "Box size is not big enough to cover 16 * r_B");

	auto const &x_array_ptr = x_array.dataPtr();
	auto const &v_array_ptr = v_array.dataPtr();
	const int array_size = static_cast<int>(x_array.size());

	const Real Lx = prob_hi[0] - prob_lo[0];
	const Real R_ball = Lx / 4.0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		Real r = std::sqrt(x * x + y * y + z * z);
		if (r == 0.0) {
			r = 1.0;
		}
		const Real xx = r / r_B;

		Real rho = NAN;
		Real vv = NAN;

		if (r > R_ball) {
			rho = rho0;
			vv = 0.0;
		} else {
			// interpolate for v
			if (xx <= x_array_ptr[0]) {  // NOLINT
				vv = v_array_ptr[0]; // NOLINT
			} else {
				AMREX_ASSERT(xx <= x_array_ptr[array_size - 1]); // NOLINT
				vv = interpolate_value(xx, x_array_ptr, v_array_ptr, array_size);
			}
			const Real lam = std::exp(1.5) / 4.0;
			const Real aa = lam / (xx * xx * vv);

			rho = aa * rho0;
		}

		const Real v = vv * cs0;
		const Real vx = v * x / r;
		const Real vy = v * y / r;
		const Real vz = v * z / r;
		const Real Eint = rho / mu * k_B * T0; // arbitrary choice, since Eint is not used in isothermal gas EOS
		const Real Ekin = 0.5 * rho * v * v;
		const Real Etot = Eint + Ekin;

		state_cc(i, j, k, HydroSystem<AccretionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::energy_index) = Etot;
	});
}

template <> void QuokkaSimulation<AccretionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// refine on Jeans length
	const int N_cells = 4; // inverse of the 'Jeans number' [Truelove et al. (1997)]
	const amrex::Real cs = quokka::EOS_Traits<AccretionProblem>::cs_isothermal;
	const amrex::Real dx = geom[lev].CellSizeArray()[0];
	const amrex::Real G = Gconst_;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		Real const rho = state[bx](i, j, k, HydroSystem<AccretionProblem>::density_index);
		const amrex::Real l_Jeans = cs * std::sqrt(M_PI / (G * rho));

		if (l_Jeans < (N_cells * dx)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

template <> void QuokkaSimulation<AccretionProblem>::computeAfterTimestep()
{
	// every step, save particle mass to userData_
	userData_.time.push_back(tNew_[0]);
	// userData_.Mstar.push_back(1.0);

	// Get particle data using the physics particle descriptor
	const int finest_level = finestLevel();
	const auto &real_data = particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(finest_level).first;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		Real Mstar = 0.0;
		const int mass_index = 3;
		for (const auto &p : real_data) {
			Mstar += p[mass_index];
		}

		userData_.Mstar.push_back(Mstar);
	}
}

auto problem_main() -> int
{
	// read problem parameters
	amrex::ParmParse const pp("problem");

	// particle mass
	pp.query("star_mass", M_star_in_Msun);

	const double M_star_in_g = M_star_in_Msun * C::M_solar;

	// boundary conditions
	const int ncomp_cc = Physics_Indices<AccretionProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<AccretionProblem> sim(BCs_cc);
	sim.doPoissonSolve_ = 1;      // enable self-gravity
	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.initDt_ = 3.0e10;	      // ~1 kyr
	sim.tempFloor_ = 10.0;	      // K

	// initialize
	sim.setInitialConditions();

	// get total gas mass of the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	amrex::Real const m_gas_init = sim.state_new_cc_[0].sum(HydroSystem<AccretionProblem>::density_index) * vol;
	amrex::Print() << "Initial gas mass = " << m_gas_init << "\n";

	// get total particle mass of the initial state
	const auto &real_data_init = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevelZero().first;
	const int n_stars_init = static_cast<int>(real_data_init.size());
	amrex::Real const m_stars_init = std::accumulate(real_data_init.begin(), real_data_init.end(), 0.0, [](Real acc, const auto &p) { return acc + p[3]; });
	amrex::Print() << "Initial particle mass = " << m_stars_init << "\n";

	const double m_tot_init = m_gas_init + m_stars_init;

	// evolve
	sim.evolve();

	// get total gas mass of the final state
	amrex::Real const m_gas_final = sim.state_new_cc_[0].sum(HydroSystem<AccretionProblem>::density_index) * vol;
	amrex::Print() << "Final gas mass = " << m_gas_final << "\n";

	// get total particle mass of the final state
	const auto &real_data_final = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevelZero().first;
	const int n_stars_final = static_cast<int>(real_data_final.size());
	amrex::Real const m_stars_final =
	    std::accumulate(real_data_final.begin(), real_data_final.end(), 0.0, [](Real acc, const auto &p) { return acc + p[3]; });
	amrex::Print() << "Final particle mass = " << m_stars_final << "\n";

	const double m_tot_final = m_gas_final + m_stars_final;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// check mass conservation
		const double rel_error_total_mass = std::abs(m_tot_final - m_tot_init) / m_tot_init;
		amrex::Print() << "rel_error_total_mass = " << rel_error_total_mass << "\n";

		// plot particle mass vs time
		std::vector<Real> &time = sim.userData_.time;
		std::vector<Real> &Mstar_ = sim.userData_.Mstar;

		// print mass vs time
		for (int i = 0; i < static_cast<int>(time.size()); ++i) {
			amrex::Print() << "time = " << time[i] << ", Mstar = " << Mstar_[i] << "\n";
		}

		// compute exact accretion rate
		const Real r_BH = C::Gconst * M_star_in_g / (cs0 * cs0);
		const Real lam = std::exp(1.5) / 4.0;
		const Real Mdot_exact = 4.0 * M_PI * rho0 * r_BH * r_BH * (lam * cs0);
		amrex::Print() << "Mdot_exact = " << Mdot_exact << "\n";

		// Estimate the accretion rate from the particle data
		const int last_step = static_cast<int>(time.size()) - 1;
		int first_step = 0;
		if (sim.istep[0] >= 22) {
			first_step = last_step - 20;
		}
		const Real Mdot_sim = (Mstar_[last_step] - Mstar_[first_step]) / (time[last_step] - time[first_step]);
		if (sim.istep[0] >= 22) {
			amrex::Print() << "Steady state Mdot_sim = " << Mdot_sim << "\n";
		} else {
			amrex::Print() << "From first to last step, Mdot_sim = " << Mdot_sim << "\n";
		}

		// compute relative difference
		const Real rel_diff = std::abs(Mdot_sim - Mdot_exact) / Mdot_exact;
		amrex::Print() << "rel_diff = " << rel_diff << "\n";

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		matplotlibcpp::plot(time, Mstar_);
		matplotlibcpp::xlabel("Time");
		matplotlibcpp::ylabel("Particle Mass");
		const std::string title = fmt::format("Exact Bondi accretion rate = {:.2e} g/s", Mdot_exact);
		matplotlibcpp::title(title);
		matplotlibcpp::save("particle_mass.png");
#endif
	}

	return 0;
}
