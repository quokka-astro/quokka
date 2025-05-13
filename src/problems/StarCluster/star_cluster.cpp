//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file star_cluster.cpp
/// \brief Defines a test problem for pressureless spherical collapse of a star cluster.

#include "star_cluster.hpp"
#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <gcem.hpp>

using amrex::Real;

struct StarCluster {
};

constexpr double rho0 = C::m_p;
constexpr double T0 = 10.0;
constexpr double mu = 2.33 * C::m_p;
constexpr double k_B = C::k_B;
constexpr double cs0 = gcem::sqrt(k_B * T0 / mu);

AMREX_GPU_MANAGED double M_star_in_Msun = 1.0; // NOLINT

template <> struct quokka::EOS_Traits<StarCluster> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = cs0;
	static constexpr double mean_molecular_weight = mu;
};

template <> struct HydroSystem_Traits<StarCluster> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<StarCluster> {
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

template <> struct Particle_Traits<StarCluster> {
	// static constexpr ParticleSwitch particle_switch = ParticleSwitch::None;
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::Sink;
};

template <> void QuokkaSimulation<StarCluster>::createInitialSinkParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 4; // mass vx vy vz
	SinkParticles->SetVerbose(1);
	SinkParticles->InitFromAsciiFile("sink.txt", nreal_extra, nullptr);

	// // Loop over all particle at all levels and set first integer component to SNProgenitor
	// for (int lev = 0; lev <= SinkParticles->finestLevel(); ++lev) {
	// 	auto &particles = SinkParticles->GetParticles(lev);

	// 	for (auto &kv : particles) {
	// 		auto &particle_array = kv.second.GetArrayOfStructs();
	// 		const int np = particle_array.numParticles();
	// 		auto *pdata = particle_array().data();

	// 		// Launch GPU kernel to set integer components
	// 		amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
	// 			auto &p = pdata[i]; // NOLINT
	// 			p.rdata(0) = M_star_in_Msun * C::M_solar;
	// 		});
	// 	}
	// }

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<StarCluster>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	std::vector<Real> x_array = {
	    0.0001, 0.04,  0.08,  0.12,	 0.16,	0.20,  0.24,  0.28,  0.32,  0.36,  0.40,  0.44,	 0.48,	0.52,  0.56,  0.60,  0.64,  0.68,  0.72,  0.76,	 0.80,
	    0.84,   0.88,  0.92,  0.96,	 1.00,	1.04,  1.08,  1.12,  1.16,  1.20,  1.24,  1.28,	 1.32,	1.36,  1.40,  1.44,  1.48,  1.52,  1.56,  1.60,	 1.64,
	    1.68,   1.72,  1.76,  1.80,	 1.84,	1.88,  1.92,  1.96,  2.00,  2.04,  2.08,  2.12,	 2.16,	2.20,  2.24,  2.28,  2.32,  2.36,  2.40,  2.44,	 2.48,
	    2.52,   2.56,  2.60,  2.64,	 2.68,	2.72,  2.76,  2.80,  2.84,  2.88,  2.92,  2.96,	 3.00,	3.04,  3.08,  3.12,  3.16,  3.20,  3.24,  3.28,	 3.32,
	    3.36,   3.40,  3.44,  3.48,	 3.52,	3.56,  3.60,  3.64,  3.68,  3.72,  3.76,  3.80,	 3.84,	3.88,  3.92,  3.96,  4.00,  4.04,  4.08,  4.12,	 4.16,
	    4.20,   4.24,  4.28,  4.32,	 4.36,	4.40,  4.44,  4.48,  4.52,  4.56,  4.60,  4.64,	 4.68,	4.72,  4.76,  4.80,  4.84,  4.88,  4.92,  4.96,	 5.00,
	    5.04,   5.08,  5.12,  5.16,	 5.20,	5.24,  5.28,  5.32,  5.36,  5.40,  5.44,  5.48,	 5.52,	5.56,  5.60,  5.64,  5.68,  5.72,  5.76,  5.80,	 5.84,
	    5.88,   5.92,  5.96,  6.00,	 6.04,	6.08,  6.12,  6.16,  6.20,  6.24,  6.28,  6.32,	 6.36,	6.40,  6.44,  6.48,  6.52,  6.56,  6.60,  6.64,	 6.68,
	    6.72,   6.76,  6.80,  6.84,	 6.88,	6.92,  6.96,  7.00,  7.04,  7.08,  7.12,  7.16,	 7.20,	7.24,  7.28,  7.32,  7.36,  7.40,  7.44,  7.48,	 7.52,
	    7.56,   7.60,  7.64,  7.68,	 7.72,	7.76,  7.80,  7.84,  7.88,  7.92,  7.96,  8.00,	 8.04,	8.08,  8.12,  8.16,  8.20,  8.24,  8.28,  8.32,	 8.36,
	    8.40,   8.44,  8.48,  8.52,	 8.56,	8.60,  8.64,  8.68,  8.72,  8.76,  8.80,  8.84,	 8.88,	8.92,  8.96,  9.00,  9.04,  9.08,  9.12,  9.16,	 9.20,
	    9.24,   9.28,  9.32,  9.36,	 9.40,	9.44,  9.48,  9.52,  9.56,  9.60,  9.64,  9.68,	 9.72,	9.76,  9.80,  9.84,  9.88,  9.92,  9.96,  10.00, 10.04,
	    10.08,  10.12, 10.16, 10.20, 10.24, 10.28, 10.32, 10.36, 10.40, 10.44, 10.48, 10.52, 10.56, 10.60, 10.64, 10.68, 10.72, 10.76, 10.80, 10.84, 10.88,
	    10.92,  10.96, 11.00, 11.04, 11.08, 11.12, 11.16, 11.20, 11.24, 11.28, 11.32, 11.36, 11.40, 11.44, 11.48, 11.52, 11.56, 11.60, 11.64, 11.68, 11.72,
	    11.76,  11.80, 11.84, 11.88, 11.92, 11.96, 12.00, 12.04, 12.08, 12.12, 12.16, 12.20, 12.24, 12.28, 12.32, 12.36, 12.40, 12.44, 12.48, 12.52, 12.56,
	    12.60,  12.64, 12.68, 12.72, 12.76, 12.80, 12.84, 12.88, 12.92, 12.96, 13.00, 13.04, 13.08, 13.12, 13.16, 13.20, 13.24, 13.28, 13.32, 13.36, 13.40,
	    13.44,  13.48, 13.52, 13.56, 13.60, 13.64, 13.68, 13.72, 13.76, 13.80, 13.84, 13.88, 13.92, 13.96, 14.00, 14.04, 14.08, 14.12, 14.16, 14.20, 14.24,
	    14.28,  14.32, 14.36, 14.40, 14.44, 14.48, 14.52, 14.56, 14.60, 14.64, 14.68, 14.72, 14.76, 14.80, 14.84, 14.88, 14.92, 14.96, 15.00, 15.04, 15.08,
	    15.12,  15.16, 15.20, 15.24, 15.28, 15.32, 15.36, 15.40, 15.44, 15.48, 15.52, 15.56, 15.60, 15.64, 15.68, 15.72, 15.76, 15.80, 15.84, 15.88, 15.92,
	    15.96,  16.00};
	std::vector<Real> v_array = {
	    6.371881593525, 6.371881593525, 4.187338638014, 3.207694870928, 2.620940546521, 2.220410065376, 1.925598751310, 1.697675943531, 1.515267485382,
	    1.365492388620, 1.240055990063, 1.133339112310, 1.041379368041, 0.961290369858, 0.890912265037, 0.828591934444, 0.773039520182, 0.723231774385,
	    0.678345156254, 0.637708412603, 0.600768261984, 0.567064099956, 0.536209045091, 0.507875525290, 0.481784169865, 0.457695145301, 0.435401322287,
	    0.414722832475, 0.395502692149, 0.377603253561, 0.360903304775, 0.345295682188, 0.330685291825, 0.316987459031, 0.304126544049, 0.292034774232,
	    0.280651253953, 0.269921121158, 0.259794825551, 0.250227508310, 0.241178466855, 0.232610691334, 0.224490461716, 0.216786996449, 0.209472145055,
	    0.202520118389, 0.195907251244, 0.189611792869, 0.183613721608, 0.177894580532, 0.172437331246, 0.167226223701, 0.162246679856, 0.157485189631,
	    0.152929217604, 0.148567119166, 0.144388065100, 0.140381973568, 0.136539448690, 0.132851724995, 0.129310617073, 0.125908473926, 0.122638137464,
	    0.119492904752, 0.116466493588, 0.113553011138, 0.110746925240, 0.108043038191, 0.105436462705, 0.102922599917, 0.100497119140, 0.098155939331,
	    0.095895211955, 0.093711305307, 0.091600789965, 0.089560425443, 0.087587147825, 0.085678058335, 0.083830412790, 0.082041611797, 0.080309191698,
	    0.078630816150, 0.077004268312, 0.075427443608, 0.073898342951, 0.072415066494, 0.070975807768, 0.069578848238, 0.068222552219, 0.066905362138,
	    0.065625794111, 0.064382433795, 0.063173932521, 0.061999003680, 0.060856419326, 0.059745007008, 0.058663646781, 0.057611268426, 0.056586848816,
	    0.055589409474, 0.054618014236, 0.053671767089, 0.052749810138, 0.051851321654, 0.050975514288, 0.050121633349, 0.049288955191, 0.048476785713,
	    0.047684458906, 0.046911335515, 0.046156801763, 0.045420268134, 0.044701168245, 0.043998957754, 0.043313113359, 0.042643131810, 0.041988529014,
	    0.041348839162, 0.040723613897, 0.040112421561, 0.039514846430, 0.038930488030, 0.038358960467, 0.037799891800, 0.037252923438, 0.036717709564,
	    0.036193916603, 0.035681222708, 0.035179317255, 0.034687900400, 0.034206682609, 0.033735384264, 0.033273735223, 0.032821474481, 0.032378349766,
	    0.031944117204, 0.031518541002, 0.031101393103, 0.030692452896, 0.030291506931, 0.029898348641, 0.029512778067, 0.029134601618, 0.028763631830,
	    0.028399687128, 0.028042591604, 0.027692174817, 0.027348271587, 0.027010721795, 0.026679370203, 0.026354066281, 0.026034664029, 0.025721021820,
	    0.025413002241, 0.025110471949, 0.024813301524, 0.024521365324, 0.024234541363, 0.023952711193, 0.023675759755, 0.023403575289, 0.023136049210,
	    0.022873075999, 0.022614553109, 0.022360380853, 0.022110462319, 0.021864703273, 0.021623012071, 0.021385299580, 0.021151479085, 0.020921466224,
	    0.020695178892, 0.020472537199, 0.020253463367, 0.020037881692, 0.019825718451, 0.019616901860, 0.019411362007, 0.019209030792, 0.019009841877,
	    0.018813730616, 0.018620634026, 0.018430490720, 0.018243240868, 0.018058826133, 0.017877189656, 0.017698275977, 0.017522031031, 0.017348402064,
	    0.017177337635, 0.017008787554, 0.016842702845, 0.016679035731, 0.016517739571, 0.016358768851, 0.016202079141, 0.016047627059, 0.015895370259,
	    0.015745267386, 0.015597278047, 0.015451362804, 0.015307483119, 0.015165601350, 0.015025680729, 0.014887685308, 0.014751579981, 0.014617330420,
	    0.014484903086, 0.014354265178, 0.014225384642, 0.014098230136, 0.013972771006, 0.013848977278, 0.013726819636, 0.013606269417, 0.013487298559,
	    0.013369879635, 0.013253985788, 0.013139590751, 0.013026668821, 0.012915194833, 0.012805144171, 0.012696492721, 0.012589216894, 0.012483293586,
	    0.012378700180, 0.012275414525, 0.012173414931, 0.012072680157, 0.011973189400, 0.011874922272, 0.011777858816, 0.011681979465, 0.011587265061,
	    0.011493696821, 0.011401256345, 0.011309925595, 0.011219686898, 0.011130522926, 0.011042416697, 0.010955351564, 0.010869311198, 0.010784279594,
	    0.010700241058, 0.010617180201, 0.010535081929, 0.010453931438, 0.010373714206, 0.010294415989, 0.010216022814, 0.010138520970, 0.010061897011,
	    0.009986137735, 0.009911230191, 0.009837161675, 0.009763919704, 0.009691492039, 0.009619866661, 0.009549031781, 0.009478975808, 0.009409687376,
	    0.009341155324, 0.009273368688, 0.009206316709, 0.009139988812, 0.009074374618, 0.009009463932, 0.008945246739, 0.008881713206, 0.008818853665,
	    0.008756658623, 0.008695118758, 0.008634224901, 0.008573968052, 0.008514339361, 0.008455330134, 0.008396931826, 0.008339136035, 0.008281934513,
	    0.008225319139, 0.008169281941, 0.008113815082, 0.008058910845, 0.008004561658, 0.007950760062, 0.007897498740, 0.007844770477, 0.007792568196,
	    0.007740884925, 0.007689713809, 0.007639048111, 0.007588881198, 0.007539206550, 0.007490017752, 0.007441308490, 0.007393072558, 0.007345303845,
	    0.007297996338, 0.007251144123, 0.007204741385, 0.007158782389, 0.007113261501, 0.007068173175, 0.007023511950, 0.006979272451, 0.006935449386,
	    0.006892037554, 0.006849031821, 0.006806427147, 0.006764218558, 0.006722401165, 0.006680970148, 0.006639920769, 0.006599248355, 0.006558948310,
	    0.006519016097, 0.006479447263, 0.006440237414, 0.006401382219, 0.006362877416, 0.006324718813, 0.006286902266, 0.006249423707, 0.006212279118,
	    0.006175464548, 0.006138976101, 0.006102809936, 0.006066962273, 0.006031429381, 0.005996207591, 0.005961293282, 0.005926682889, 0.005892372897,
	    0.005858359836, 0.005824640294, 0.005791210907, 0.005758068356, 0.005725209369, 0.005692630722, 0.005660329239, 0.005628301785, 0.005596545267,
	    0.005565056642, 0.005533832908, 0.005502871104, 0.005472168306, 0.005441721639, 0.005411528256, 0.005381585366, 0.005351890198, 0.005322440034,
	    0.005293232187, 0.005264264003, 0.005235532872, 0.005207036214, 0.005178771487, 0.005150736178, 0.005122927819, 0.005095343962, 0.005067982197,
	    0.005040840154, 0.005013915481, 0.004987205867, 0.004960709028, 0.004934422713, 0.004908344695, 0.004882472783, 0.004856804808, 0.004831338637,
	    0.004806072156, 0.004781003292, 0.004756129979, 0.004731450198, 0.004706961944, 0.004682663240, 0.004658552138, 0.004634626711, 0.004610885060,
	    0.004587325305, 0.004563945596, 0.004540744102, 0.004517719018, 0.004494868560, 0.004472190970, 0.004449684504, 0.004427347449, 0.004405178110,
	    0.004383174810, 0.004361335900, 0.004339659742, 0.004318144726, 0.004296789259, 0.004275591767, 0.004254550700, 0.004233664518, 0.004212931706,
	    0.004192350772, 0.004171920231, 0.004151638621, 0.004131504502, 0.004111516447};

	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double r_B = C::Gconst * M_star_in_Msun * C::M_solar / std::pow(cs0, 2);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		Real r = std::sqrt(x * x + y * y + z * z);
		if (r == 0.0) {
			r = 1.0;
		}
		const Real xx = r / r_B;

		AMREX_ASSERT(xx >= x_array[0]);
		AMREX_ASSERT(xx <= x_array.back());

		// interpolate for v
		const Real vv = interpolate_value(xx, x_array.data(), v_array.data(), static_cast<int>(x_array.size()));
		const Real lam = std::exp(1.5) / 4.0;
		const Real aa = lam / (xx * xx * vv);

		const Real rho = aa * rho0;
		const Real v = vv * cs0;
		const Real vx = v * x / r;
		const Real vy = v * y / r;
		const Real vz = v * z / r;
		const Real Eint = rho / mu * k_B * T0; // arbitrary choice, since Eint is not used in isothermal gas EOS
		const Real Ekin = 0.5 * rho * v * v;
		const Real Etot = Eint + Ekin;

		state_cc(i, j, k, HydroSystem<StarCluster>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StarCluster>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<StarCluster>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<StarCluster>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<StarCluster>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<StarCluster>::energy_index) = Etot;
	});
}

template <> void QuokkaSimulation<StarCluster>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// refine on Jeans length
	const int N_cells = 4; // inverse of the 'Jeans number' [Truelove et al. (1997)]
	const amrex::Real cs = quokka::EOS_Traits<StarCluster>::cs_isothermal;
	const amrex::Real dx = geom[lev].CellSizeArray()[0];
	const amrex::Real G = Gconst_;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		Real const rho = state[bx](i, j, k, HydroSystem<StarCluster>::density_index);
		const amrex::Real l_Jeans = cs * std::sqrt(M_PI / (G * rho));

		if (l_Jeans < (N_cells * dx)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

auto problem_main() -> int
{
	// read problem parameters
	amrex::ParmParse const pp("problem");

	// particle mass
	pp.query("star_mass", M_star_in_Msun);

	// boundary conditions
	const int ncomp_cc = Physics_Indices<StarCluster>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<StarCluster> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	sim.stopTime_ = 1.0e6 * 3.0e7; // ~1 Myr

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int const status = 0;
	return status;
}
