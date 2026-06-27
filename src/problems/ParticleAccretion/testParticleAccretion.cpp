/// \file testParticleAccretion.cpp
/// \brief Defines a test problem for Bondi-Hoyle accretion.

#include "AMReX.H"
#include "AMReX_Array.H"
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
#include "util/BC.hpp"
#include "util/fextract.hpp"
#include <format>
#include <gcem.hpp>

#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

using amrex::Real;

bool turnon_fextract = false;		       // NOLINT
constexpr bool particle_in_cell_center = true; // NOLINT
bool return_1_at_fail = false;		       // NOLINT
std::string sink_file = "../inputs/sink.txt";  // NOLINT

struct AccretionProblem {
};

// In this test, r_B = 0.1214 pc. Ball radius is R = 32 r_B, box half-size is 2 R = 64 r_B = 7.7696 pc = 2.397448054e+19 cm

constexpr double T0 = 10.0;
constexpr double mu = 2.33 * C::m_p;
constexpr double k_B = C::k_B;
constexpr double cs0 = gcem::sqrt(k_B * T0 / mu); // = 18821.95750 cm / s for T = 10 K
constexpr double B0 = 1.0e-7;			  // constant background field [Gauss-equivalent units]

double rho0 = C::m_p;				 // NOLINT
double t_end_over_t_b = 10.0;			 // NOLINT
AMREX_GPU_MANAGED double M_star_in_Msun = 1.0;	 // NOLINT
AMREX_GPU_MANAGED double uniform_density = -1.0; // NOLINT. Default is not using uniform density. If set to a positive value, the density will be set to this
						 // value instead of an exact solution to the Bondi problem.
bool refine_center = true;			 // NOLINT

// constexpr double r_B = C::Gconst * C::M_solar / (cs0 * cs0);

template <> struct Particle_Traits<AccretionProblem> : DefaultParticleTraits {
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

template <> struct Physics_Traits<AccretionProblem> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
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
	SinkParticles->InitFromAsciiFile(sink_file, nreal_extra, nullptr);

	const int max_lev = max_level;

	// For the test problem in the Sink Particle paper, we want to set max_lev to 2.
	// AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_lev == 2, "amx_lev is not 2");

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom[max_lev].CellSizeArray();

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
				if (particle_in_cell_center) {
					p.pos(0) = 0.5 * dx[0];
					p.pos(1) = 0.5 * dx[1];
					p.pos(2) = 0.5 * dx[2];
				} else {
					p.pos(0) = 0.0;
					p.pos(1) = 0.0;
					p.pos(2) = 0.0;
				}
			});
		}
	}

	// Ensure GPU operations are complete
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Gpu::DeviceVector<double> x_array = {
	    1.0000e-02, 1.0293e-02, 1.0594e-02, 1.0904e-02, 1.1223e-02, 1.1552e-02, 1.1890e-02, 1.2238e-02, 1.2597e-02, 1.2965e-02, 1.3345e-02, 1.3736e-02,
	    1.4138e-02, 1.4551e-02, 1.4977e-02, 1.5416e-02, 1.5867e-02, 1.6332e-02, 1.6810e-02, 1.7302e-02, 1.7808e-02, 1.8330e-02, 1.8866e-02, 1.9419e-02,
	    1.9987e-02, 2.0572e-02, 2.1175e-02, 2.1794e-02, 2.2432e-02, 2.3089e-02, 2.3765e-02, 2.4461e-02, 2.5177e-02, 2.5914e-02, 2.6673e-02, 2.7453e-02,
	    2.8257e-02, 2.9084e-02, 2.9936e-02, 3.0812e-02, 3.1714e-02, 3.2643e-02, 3.3598e-02, 3.4582e-02, 3.5594e-02, 3.6636e-02, 3.7709e-02, 3.8813e-02,
	    3.9949e-02, 4.1118e-02, 4.2322e-02, 4.3561e-02, 4.4836e-02, 4.6149e-02, 4.7500e-02, 4.8890e-02, 5.0322e-02, 5.1795e-02, 5.3311e-02, 5.4872e-02,
	    5.6478e-02, 5.8131e-02, 5.9833e-02, 6.1585e-02, 6.3388e-02, 6.5243e-02, 6.7153e-02, 6.9119e-02, 7.1143e-02, 7.3225e-02, 7.5369e-02, 7.7575e-02,
	    7.9846e-02, 8.2184e-02, 8.4590e-02, 8.7066e-02, 8.9615e-02, 9.2239e-02, 9.4939e-02, 9.7718e-02, 1.0058e-01, 1.0352e-01, 1.0655e-01, 1.0967e-01,
	    1.1288e-01, 1.1619e-01, 1.1959e-01, 1.2309e-01, 1.2669e-01, 1.3040e-01, 1.3422e-01, 1.3815e-01, 1.4219e-01, 1.4636e-01, 1.5064e-01, 1.5505e-01,
	    1.5959e-01, 1.6426e-01, 1.6907e-01, 1.7402e-01, 1.7912e-01, 1.8436e-01, 1.8976e-01, 1.9531e-01, 2.0103e-01, 2.0691e-01, 2.1297e-01, 2.1921e-01,
	    2.2562e-01, 2.3223e-01, 2.3903e-01, 2.4602e-01, 2.5323e-01, 2.6064e-01, 2.6827e-01, 2.7612e-01, 2.8421e-01, 2.9253e-01, 3.0109e-01, 3.0990e-01,
	    3.1898e-01, 3.2832e-01, 3.3793e-01, 3.4782e-01, 3.5800e-01, 3.6848e-01, 3.7927e-01, 3.9037e-01, 4.0180e-01, 4.1356e-01, 4.2567e-01, 4.3813e-01,
	    4.5096e-01, 4.6416e-01, 4.7775e-01, 4.9173e-01, 5.0613e-01, 5.2095e-01, 5.3620e-01, 5.5189e-01, 5.6805e-01, 5.8468e-01, 6.0180e-01, 6.1941e-01,
	    6.3755e-01, 6.5621e-01, 6.7542e-01, 6.9519e-01, 7.1554e-01, 7.3649e-01, 7.5805e-01, 7.8024e-01, 8.0309e-01, 8.2660e-01, 8.5079e-01, 8.7570e-01,
	    9.0134e-01, 9.2772e-01, 9.5488e-01, 9.8284e-01, 1.0116e+00, 1.0412e+00, 1.0717e+00, 1.1031e+00, 1.1354e+00, 1.1686e+00, 1.2028e+00, 1.2380e+00,
	    1.2743e+00, 1.3116e+00, 1.3500e+00, 1.3895e+00, 1.4302e+00, 1.4720e+00, 1.5151e+00, 1.5595e+00, 1.6051e+00, 1.6521e+00, 1.7005e+00, 1.7503e+00,
	    1.8015e+00, 1.8543e+00, 1.9085e+00, 1.9644e+00, 2.0219e+00, 2.0811e+00, 2.1420e+00, 2.2047e+00, 2.2693e+00, 2.3357e+00, 2.4041e+00, 2.4745e+00,
	    2.5469e+00, 2.6215e+00, 2.6982e+00, 2.7772e+00, 2.8585e+00, 2.9422e+00, 3.0283e+00, 3.1170e+00, 3.2082e+00, 3.3022e+00, 3.3988e+00, 3.4983e+00,
	    3.6007e+00, 3.7061e+00, 3.8146e+00, 3.9263e+00, 4.0413e+00, 4.1596e+00, 4.2813e+00, 4.4067e+00, 4.5357e+00, 4.6685e+00, 4.8051e+00, 4.9458e+00,
	    5.0906e+00, 5.2396e+00, 5.3930e+00, 5.5509e+00, 5.7134e+00, 5.8806e+00, 6.0528e+00, 6.2300e+00, 6.4124e+00, 6.6001e+00, 6.7933e+00, 6.9922e+00,
	    7.1969e+00, 7.4075e+00, 7.6244e+00, 7.8476e+00, 8.0773e+00, 8.3138e+00, 8.5572e+00, 8.8077e+00, 9.0655e+00, 9.3309e+00, 9.6041e+00, 9.8852e+00,
	    1.0175e+01, 1.0472e+01, 1.0779e+01, 1.1095e+01, 1.1419e+01, 1.1754e+01, 1.2098e+01, 1.2452e+01, 1.2816e+01, 1.3192e+01, 1.3578e+01, 1.3975e+01,
	    1.4384e+01, 1.4806e+01, 1.5239e+01, 1.5685e+01, 1.6144e+01, 1.6617e+01, 1.7103e+01, 1.7604e+01, 1.8119e+01, 1.8650e+01, 1.9196e+01, 1.9758e+01,
	    2.0336e+01, 2.0932e+01, 2.1544e+01, 2.2175e+01, 2.2824e+01, 2.3492e+01, 2.4180e+01, 2.4888e+01, 2.5617e+01, 2.6367e+01, 2.7138e+01, 2.7933e+01,
	    2.8751e+01, 2.9592e+01, 3.0459e+01, 3.1350e+01, 3.2268e+01, 3.3213e+01, 3.4185e+01, 3.5186e+01, 3.6216e+01, 3.7276e+01, 3.8367e+01, 3.9490e+01,
	    4.0646e+01, 4.1836e+01, 4.3061e+01, 4.4322e+01, 4.5619e+01, 4.6955e+01, 4.8329e+01, 4.9744e+01, 5.1200e+01, 5.2699e+01, 5.4242e+01, 5.5830e+01,
	    5.7464e+01, 5.9147e+01, 6.0878e+01, 6.2660e+01, 6.4495e+01, 6.6383e+01, 6.8326e+01, 7.0326e+01, 7.2385e+01, 7.4504e+01, 7.6685e+01, 7.8930e+01,
	    8.1241e+01, 8.3619e+01, 8.6067e+01, 8.8587e+01, 9.1180e+01, 9.3849e+01, 9.6597e+01, 9.9425e+01, 1.0234e+02, 1.0533e+02, 1.0841e+02, 1.1159e+02,
	    1.1486e+02, 1.1822e+02, 1.2168e+02, 1.2524e+02, 1.2891e+02, 1.3268e+02, 1.3656e+02, 1.4056e+02, 1.4468e+02, 1.4891e+02, 1.5327e+02, 1.5776e+02,
	    1.6238e+02, 1.6713e+02, 1.7202e+02, 1.7706e+02, 1.8224e+02, 1.8758e+02, 1.9307e+02, 1.9872e+02, 2.0454e+02, 2.1053e+02, 2.1669e+02, 2.2303e+02,
	    2.2956e+02, 2.3628e+02, 2.4320e+02, 2.5032e+02, 2.5765e+02, 2.6519e+02, 2.7295e+02, 2.8095e+02, 2.8917e+02, 2.9764e+02, 3.0635e+02, 3.1532e+02,
	    3.2455e+02, 3.3405e+02, 3.4383e+02, 3.5389e+02, 3.6425e+02, 3.7492e+02, 3.8589e+02, 3.9719e+02, 4.0882e+02, 4.2078e+02, 4.3310e+02, 4.4578e+02,
	    4.5883e+02, 4.7226e+02, 4.8609e+02, 5.0032e+02, 5.1497e+02, 5.3004e+02, 5.4556e+02, 5.6153e+02, 5.7797e+02, 5.9489e+02, 6.1230e+02, 6.3023e+02,
	    6.4868e+02, 6.6767e+02, 6.8722e+02, 7.0733e+02, 7.2804e+02, 7.4935e+02, 7.7129e+02, 7.9387e+02, 8.1711e+02, 8.4103e+02, 8.6565e+02, 8.9099e+02,
	    9.1708e+02, 9.4392e+02, 9.7156e+02, 1.0000e+03};

	const amrex::Gpu::DeviceVector<double> v_array = {
	    1.3659457245e+01, 1.3452779272e+01, 1.3248981918e+01, 1.3048023792e+01, 1.2849864108e+01, 1.2654462670e+01, 1.2461779872e+01, 1.2271776684e+01,
	    1.2084414645e+01, 1.1899655856e+01, 1.1717462972e+01, 1.1537799196e+01, 1.1360628266e+01, 1.1185914454e+01, 1.1013622555e+01, 1.0843717881e+01,
	    1.0676166253e+01, 1.0510933995e+01, 1.0347987926e+01, 1.0187295354e+01, 1.0028824069e+01, 9.8725423366e+00, 9.7184188918e+00, 9.5664229315e+00,
	    9.4165241089e+00, 9.2686925276e+00, 9.1228987349e+00, 8.9791137158e+00, 8.8373088876e+00, 8.6974560934e+00, 8.5595275969e+00, 8.4234960762e+00,
	    8.2893346188e+00, 8.1570167156e+00, 8.0265162557e+00, 7.8978075212e+00, 7.7708651817e+00, 7.6456642893e+00, 7.5221802736e+00, 7.4003889363e+00,
	    7.2802664469e+00, 7.1617893373e+00, 7.0449344972e+00, 6.9296791697e+00, 6.8160009461e+00, 6.7038777619e+00, 6.5932878921e+00, 6.4842099468e+00,
	    6.3766228667e+00, 6.2705059192e+00, 6.1658386940e+00, 6.0626010989e+00, 5.9607733559e+00, 5.8603359972e+00, 5.7612698611e+00, 5.6635560883e+00,
	    5.5671761181e+00, 5.4721116845e+00, 5.3783448128e+00, 5.2858578158e+00, 5.1946332900e+00, 5.1046541128e+00, 5.0159034384e+00, 4.9283646948e+00,
	    4.8420215801e+00, 4.7568580596e+00, 4.6728583627e+00, 4.5900069792e+00, 4.5082886565e+00, 4.4276883967e+00, 4.3481914533e+00, 4.2697833285e+00,
	    4.1924497701e+00, 4.1161767688e+00, 4.0409505551e+00, 3.9667575969e+00, 3.8935845966e+00, 3.8214184883e+00, 3.7502464355e+00, 3.6800558282e+00,
	    3.6108342802e+00, 3.5425696272e+00, 3.4752499237e+00, 3.4088634406e+00, 3.3433986633e+00, 3.2788442886e+00, 3.2151892229e+00, 3.1524225796e+00,
	    3.0905336766e+00, 3.0295120345e+00, 2.9693473740e+00, 2.9100296137e+00, 2.8515488677e+00, 2.7938954437e+00, 2.7370598406e+00, 2.6810327462e+00,
	    2.6258050352e+00, 2.5713677668e+00, 2.5177121825e+00, 2.4648297042e+00, 2.4127119312e+00, 2.3613506389e+00, 2.3107377758e+00, 2.2608654616e+00,
	    2.2117259846e+00, 2.1633117995e+00, 2.1156155251e+00, 2.0686299415e+00, 2.0223479880e+00, 1.9767627601e+00, 1.9318675074e+00, 1.8876556304e+00,
	    1.8441206780e+00, 1.8012563448e+00, 1.7590564677e+00, 1.7175150232e+00, 1.6766261242e+00, 1.6363840168e+00, 1.5967830766e+00, 1.5578178057e+00,
	    1.5194828286e+00, 1.4817728888e+00, 1.4446828445e+00, 1.4082076650e+00, 1.3723424259e+00, 1.3370823052e+00, 1.3024225782e+00, 1.2683586135e+00,
	    1.2348858670e+00, 1.2019998779e+00, 1.1696962623e+00, 1.1379707086e+00, 1.1068189712e+00, 1.0762368648e+00, 1.0462202582e+00, 1.0167650681e+00,
	    9.8786725262e-01, 9.5952280454e-01, 9.3172774451e-01, 9.0447811409e-01, 8.7776996854e-01, 8.5159936955e-01, 8.2596237787e-01, 8.0085504579e-01,
	    7.7627340956e-01, 7.5221348174e-01, 7.2867124361e-01, 7.0564263746e-01, 6.8312355900e-01, 6.6110984985e-01, 6.3959729006e-01, 6.1858159089e-01,
	    5.9805838766e-01, 5.7802323295e-01, 5.5847158996e-01, 5.3939882629e-01, 5.2080020803e-01, 5.0267089428e-01, 4.8500593208e-01, 4.6780025194e-01,
	    4.5104866380e-01, 4.3474585364e-01, 4.1888638067e-01, 4.0346467523e-01, 3.8847503728e-01, 3.7391163570e-01, 3.5976850826e-01, 3.4603956232e-01,
	    3.3271857635e-01, 3.1979920219e-01, 3.0727496801e-01, 2.9513928210e-01, 2.8338543738e-01, 2.7200661656e-01, 2.6099589811e-01, 2.5034626273e-01,
	    2.4005060062e-01, 2.3010171913e-01, 2.2049235107e-01, 2.1121516346e-01, 2.0226276662e-01, 1.9362772373e-01, 1.8530256055e-01, 1.7727977547e-01,
	    1.6955184967e-01, 1.6211125738e-01, 1.5495047614e-01, 1.4806199710e-01, 1.4143833517e-01, 1.3507203901e-01, 1.2895570082e-01, 1.2308196593e-01,
	    1.1744354197e-01, 1.1203320779e-01, 1.0684382191e-01, 1.0186833059e-01, 9.7099775476e-02, 9.2531300660e-02, 8.8156159353e-02, 8.3967719987e-02,
	    7.9959471803e-02, 7.6125029918e-02, 7.2458139853e-02, 6.8952681538e-02, 6.5602672791e-02, 6.2402272285e-02, 5.9345782017e-02, 5.6427649287e-02,
	    5.3642468209e-02, 5.0984980773e-02, 4.8450077477e-02, 4.6032797553e-02, 4.3728328807e-02, 4.1532007100e-02, 3.9439315490e-02, 3.7445883063e-02,
	    3.5547483481e-02, 3.3740033259e-02, 3.2019589802e-02, 3.0382349240e-02, 2.8824644045e-02, 2.7342940503e-02, 2.5933836010e-02, 2.4594056259e-02,
	    2.3320452300e-02, 2.2109997515e-02, 2.0959784516e-02, 1.9867021977e-02, 1.8829031431e-02, 1.7843244025e-02, 1.6907197265e-02, 1.6018531750e-02,
	    1.5174987913e-02, 1.4374402772e-02, 1.3614706709e-02, 1.2893920280e-02, 1.2210151059e-02, 1.1561590532e-02, 1.0946511037e-02, 1.0363262765e-02,
	    9.8102708179e-03, 9.2860323268e-03, 8.7891136433e-03, 8.3181475958e-03, 7.8718308173e-03, 7.4489211475e-03, 7.0482351074e-03, 6.6686454498e-03,
	    6.3090787839e-03, 5.9685132758e-03, 5.6459764247e-03, 5.3405429125e-03, 5.0513325290e-03, 4.7775081694e-03, 4.5182739048e-03, 4.2728731231e-03,
	    4.0405867409e-03, 3.8207314828e-03, 3.6126582282e-03, 3.4157504238e-03, 3.2294225588e-03, 3.0531187042e-03, 2.8863111098e-03, 2.7284988626e-03,
	    2.5792065995e-03, 2.4379832771e-03, 2.3044009935e-03, 2.1780538620e-03, 2.0585569346e-03, 1.9455451732e-03, 1.8386724674e-03, 1.7376106961e-03,
	    1.6420488331e-03, 1.5516920927e-03, 1.4662611153e-03, 1.3854911919e-03, 1.3091315231e-03, 1.2369445153e-03, 1.1687051086e-03, 1.1042001384e-03,
	    1.0432277263e-03, 9.8559670127e-04, 9.3112604901e-04, 8.7964438797e-04, 8.3098947065e-04, 7.8500771080e-04, 7.4155373263e-04, 7.0048994289e-04,
	    6.6168612514e-04, 6.2501905284e-04, 5.9037212344e-04, 5.5763500954e-04, 5.2670332924e-04, 4.9747833164e-04, 4.6986659960e-04, 4.4377976706e-04,
	    4.1913425145e-04, 3.9585099883e-04, 3.7385524371e-04, 3.5307627938e-04, 3.3344724217e-04, 3.1490490536e-04, 2.9738948470e-04, 2.8084445350e-04,
	    2.6521636802e-04, 2.5045470163e-04, 2.3651168767e-04, 2.2334217087e-04, 2.1090346625e-04, 1.9915522556e-04, 1.8805931141e-04, 1.7757967701e-04,
	    1.6768225290e-04, 1.5833484014e-04, 1.4950700828e-04, 1.4116999944e-04, 1.3329663713e-04, 1.2586124028e-04, 1.1883954164e-04, 1.1220861081e-04,
	    1.0594678100e-04, 1.0003358041e-04, 9.4449666122e-05, 8.9176763434e-05, 8.4197606159e-05, 7.9495882282e-05, 7.5056181212e-05, 7.0863944552e-05,
	    6.6905419038e-05, 6.3167612311e-05, 5.9638251535e-05, 5.6305743269e-05, 5.3159136286e-05, 5.0188086450e-05, 4.7382822830e-05, 4.4734116443e-05,
	    4.2233250237e-05, 3.9871990785e-05, 3.7642561679e-05, 3.5537618408e-05, 3.3550224143e-05, 3.1673827567e-05, 2.9902241386e-05, 2.8229622097e-05,
	    2.6650451228e-05, 2.5159517033e-05, 2.3751897895e-05, 2.2422945615e-05, 2.1168271197e-05, 1.9983729507e-05, 1.8865406306e-05, 1.7809605409e-05,
	    1.6812836288e-05, 1.5871803275e-05, 1.4983393739e-05, 1.4144668988e-05, 1.3352854041e-05, 1.2605328302e-05, 1.1899617605e-05, 1.1233385685e-05,
	    1.0604426450e-05, 1.0010657155e-05, 9.4501110197e-06, 8.9209309688e-06, 8.4213637240e-06, 7.9497538913e-06, 7.5045381631e-06, 7.0842409978e-06,
	    6.6874688222e-06, 6.3129060522e-06, 5.9593106592e-06, 5.6255098498e-06, 5.3103964278e-06, 5.0129249292e-06, 4.7321085520e-06, 4.4670155192e-06,
	    4.2167658944e-06, 3.9805293092e-06, 3.7575214379e-06, 3.5470021793e-06, 3.3482724730e-06, 3.1606725944e-06, 2.9835795395e-06, 2.8164052064e-06,
	    2.6585943484e-06, 2.5096228695e-06, 2.3689958906e-06, 2.2362462726e-06, 2.1109329106e-06, 1.9926394834e-06, 1.8809729756e-06, 1.7755623133e-06,
	    1.6760571138e-06, 1.5821267757e-06, 1.4934590012e-06, 1.4097592279e-06, 1.3307489228e-06, 1.2561658105e-06, 1.1857615990e-06, 1.1193024344e-06};

	const amrex::Gpu::DeviceVector<double> a_array = {
	    8.20254e+02, 7.86153e+02, 7.53484e+02, 7.22186e+02, 6.92202e+02, 6.63475e+02, 6.35954e+02, 6.09587e+02, 5.84326e+02, 5.60123e+02, 5.36935e+02,
	    5.14718e+02, 4.93432e+02, 4.73038e+02, 4.53497e+02, 4.34774e+02, 4.16835e+02, 3.99646e+02, 3.83176e+02, 3.67394e+02, 3.52272e+02, 3.37782e+02,
	    3.23898e+02, 3.10593e+02, 2.97843e+02, 2.85625e+02, 2.73918e+02, 2.62698e+02, 2.51946e+02, 2.41642e+02, 2.31767e+02, 2.22304e+02, 2.13234e+02,
	    2.04542e+02, 1.96211e+02, 1.88227e+02, 1.80575e+02, 1.73240e+02, 1.66210e+02, 1.59472e+02, 1.53013e+02, 1.46822e+02, 1.40888e+02, 1.35199e+02,
	    1.29746e+02, 1.24519e+02, 1.19508e+02, 1.14704e+02, 1.10099e+02, 1.05684e+02, 1.01451e+02, 9.73929e+01, 9.35020e+01, 8.97715e+01, 8.61946e+01,
	    8.27649e+01, 7.94764e+01, 7.63230e+01, 7.32992e+01, 7.03995e+01, 6.76188e+01, 6.49521e+01, 6.23947e+01, 5.99420e+01, 5.75897e+01, 5.53336e+01,
	    5.31696e+01, 5.10941e+01, 4.91032e+01, 4.71934e+01, 4.53615e+01, 4.36041e+01, 4.19182e+01, 4.03008e+01, 3.87491e+01, 3.72603e+01, 3.58319e+01,
	    3.44614e+01, 3.31463e+01, 3.18843e+01, 3.06734e+01, 2.95113e+01, 2.83960e+01, 2.73257e+01, 2.62985e+01, 2.53125e+01, 2.43661e+01, 2.34577e+01,
	    2.25857e+01, 2.17487e+01, 2.09451e+01, 2.01736e+01, 1.94329e+01, 1.87217e+01, 1.80388e+01, 1.73831e+01, 1.67534e+01, 1.61488e+01, 1.55681e+01,
	    1.50104e+01, 1.44747e+01, 1.39602e+01, 1.34660e+01, 1.29913e+01, 1.25353e+01, 1.20972e+01, 1.16762e+01, 1.12718e+01, 1.08833e+01, 1.05099e+01,
	    1.01511e+01, 9.80628e+00, 9.47490e+00, 9.15643e+00, 8.85032e+00, 8.55610e+00, 8.27327e+00, 8.00139e+00, 7.74002e+00, 7.48874e+00, 7.24715e+00,
	    7.01485e+00, 6.79149e+00, 6.57670e+00, 6.37015e+00, 6.17151e+00, 5.98047e+00, 5.79672e+00, 5.61998e+00, 5.44998e+00, 5.28644e+00, 5.12912e+00,
	    4.97777e+00, 4.83215e+00, 4.69205e+00, 4.55725e+00, 4.42753e+00, 4.30271e+00, 4.18260e+00, 4.06700e+00, 3.95575e+00, 3.84868e+00, 3.74562e+00,
	    3.64643e+00, 3.55095e+00, 3.45904e+00, 3.37056e+00, 3.28539e+00, 3.20339e+00, 3.12444e+00, 3.04843e+00, 2.97525e+00, 2.90479e+00, 2.83694e+00,
	    2.77161e+00, 2.70869e+00, 2.64811e+00, 2.58977e+00, 2.53358e+00, 2.47947e+00, 2.42735e+00, 2.37716e+00, 2.32882e+00, 2.28225e+00, 2.23740e+00,
	    2.19419e+00, 2.15257e+00, 2.11248e+00, 2.07385e+00, 2.03665e+00, 2.00080e+00, 1.96626e+00, 1.93298e+00, 1.90091e+00, 1.87002e+00, 1.84025e+00,
	    1.81155e+00, 1.78390e+00, 1.75726e+00, 1.73157e+00, 1.70682e+00, 1.68296e+00, 1.65996e+00, 1.63778e+00, 1.61641e+00, 1.59579e+00, 1.57592e+00,
	    1.55676e+00, 1.53828e+00, 1.52046e+00, 1.50327e+00, 1.48669e+00, 1.47069e+00, 1.45527e+00, 1.44038e+00, 1.42602e+00, 1.41216e+00, 1.39878e+00,
	    1.38587e+00, 1.37341e+00, 1.36138e+00, 1.34977e+00, 1.33856e+00, 1.32773e+00, 1.31728e+00, 1.30718e+00, 1.29743e+00, 1.28801e+00, 1.27891e+00,
	    1.27012e+00, 1.26162e+00, 1.25341e+00, 1.24548e+00, 1.23781e+00, 1.23039e+00, 1.22323e+00, 1.21630e+00, 1.20959e+00, 1.20311e+00, 1.19685e+00,
	    1.19078e+00, 1.18492e+00, 1.17925e+00, 1.17376e+00, 1.16845e+00, 1.16331e+00, 1.15834e+00, 1.15352e+00, 1.14886e+00, 1.14435e+00, 1.13998e+00,
	    1.13576e+00, 1.13166e+00, 1.12770e+00, 1.12386e+00, 1.12014e+00, 1.11654e+00, 1.11305e+00, 1.10967e+00, 1.10640e+00, 1.10322e+00, 1.10015e+00,
	    1.09717e+00, 1.09428e+00, 1.09148e+00, 1.08877e+00, 1.08615e+00, 1.08360e+00, 1.08113e+00, 1.07873e+00, 1.07641e+00, 1.07416e+00, 1.07198e+00,
	    1.06986e+00, 1.06781e+00, 1.06582e+00, 1.06389e+00, 1.06202e+00, 1.06020e+00, 1.05844e+00, 1.05674e+00, 1.05508e+00, 1.05347e+00, 1.05191e+00,
	    1.05040e+00, 1.04893e+00, 1.04751e+00, 1.04613e+00, 1.04478e+00, 1.04348e+00, 1.04222e+00, 1.04100e+00, 1.03981e+00, 1.03865e+00, 1.03753e+00,
	    1.03645e+00, 1.03539e+00, 1.03437e+00, 1.03338e+00, 1.03241e+00, 1.03148e+00, 1.03057e+00, 1.02968e+00, 1.02883e+00, 1.02800e+00, 1.02719e+00,
	    1.02641e+00, 1.02565e+00, 1.02491e+00, 1.02419e+00, 1.02349e+00, 1.02282e+00, 1.02216e+00, 1.02153e+00, 1.02091e+00, 1.02031e+00, 1.01972e+00,
	    1.01916e+00, 1.01861e+00, 1.01807e+00, 1.01755e+00, 1.01705e+00, 1.01656e+00, 1.01609e+00, 1.01563e+00, 1.01518e+00, 1.01474e+00, 1.01432e+00,
	    1.01391e+00, 1.01351e+00, 1.01313e+00, 1.01275e+00, 1.01239e+00, 1.01203e+00, 1.01169e+00, 1.01135e+00, 1.01103e+00, 1.01071e+00, 1.01041e+00,
	    1.01011e+00, 1.00982e+00, 1.00954e+00, 1.00927e+00, 1.00900e+00, 1.00874e+00, 1.00849e+00, 1.00825e+00, 1.00802e+00, 1.00779e+00, 1.00757e+00,
	    1.00735e+00, 1.00714e+00, 1.00694e+00, 1.00674e+00, 1.00655e+00, 1.00636e+00, 1.00618e+00, 1.00600e+00, 1.00583e+00, 1.00566e+00, 1.00550e+00,
	    1.00535e+00, 1.00519e+00, 1.00504e+00, 1.00490e+00, 1.00476e+00, 1.00463e+00, 1.00449e+00, 1.00437e+00, 1.00424e+00, 1.00412e+00, 1.00400e+00,
	    1.00389e+00, 1.00378e+00, 1.00367e+00, 1.00357e+00, 1.00346e+00, 1.00337e+00, 1.00327e+00, 1.00318e+00, 1.00309e+00, 1.00300e+00, 1.00291e+00,
	    1.00283e+00, 1.00275e+00, 1.00267e+00, 1.00259e+00, 1.00252e+00, 1.00245e+00, 1.00238e+00, 1.00231e+00, 1.00225e+00, 1.00218e+00, 1.00212e+00,
	    1.00206e+00, 1.00200e+00, 1.00194e+00, 1.00189e+00, 1.00183e+00, 1.00178e+00, 1.00173e+00, 1.00168e+00, 1.00163e+00, 1.00159e+00, 1.00154e+00,
	    1.00150e+00, 1.00146e+00, 1.00141e+00, 1.00137e+00, 1.00134e+00, 1.00130e+00, 1.00126e+00, 1.00122e+00, 1.00119e+00, 1.00116e+00, 1.00112e+00,
	    1.00109e+00, 1.00106e+00, 1.00103e+00, 1.00100e+00};

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
	auto const &a_array_ptr = a_array.dataPtr();
	const int array_size = static_cast<int>(x_array.size());
	const int array_size_a = static_cast<int>(a_array.size());
	const int array_size_v = static_cast<int>(v_array.size());
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(array_size == array_size_a, "x_array and a_array must have the same size");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(array_size == array_size_v, "x_array and v_array must have the same size");

	const Real Lx = prob_hi[0] - prob_lo[0];
	const Real R_ball = Lx / 4.0;
	Real par_center = NAN;

	if (particle_in_cell_center) {
		par_center = dx[0] / 2.0;
	} else {
		par_center = 0.0;
	}

	const auto rho0_g = rho0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// compute x,y,z relative to the particle position
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0] - par_center;
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1] - par_center;
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2] - par_center;
		Real r = std::sqrt(x * x + y * y + z * z);
		Real xx = r / r_B;

		// if xx is very small, it means the particle is at the cell center. Set xx to 0.5 dx / r_B
		if (xx < 1.0e-10) {
			// smaller radius, higher density in the cell residing the particle
			r = 0.25 * dx[0]; // avoid division by zero in `vx = v * x / r`. `vx = 0` anyway.
			// larger radius, lower density in the cell residing the particle
			// r = 0.5 * dx[0]; // avoid division by zero in `vx = v * x / r`. `vx = 0` anyway.
			xx = r / r_B;
		}

		Real vv = NAN;
		Real aa = NAN;
		if (r > R_ball) {
			vv = 0.0;
			aa = 1.0;
		} else {
			// interpolate for v
			vv = interpolate_value<BoundaryPolicy::Clamp>(xx, x_array_ptr, v_array_ptr, array_size);
			const Real lam = std::exp(1.5) / 4.0;
			if (xx <= x_array_ptr[0]) {
				xx = x_array_ptr[0];
				// aa = lam / (xx * xx * vv);
				aa = a_array_ptr[0];
				vv = std::sqrt(lam / (xx * xx * aa));
			} else if (xx >= x_array_ptr[array_size - 1]) {
				aa = 1.0;
				vv = 0.0;
			} else {
				// vv = interpolate_value(xx, x_array_ptr, v_array_ptr, array_size);
				// aa = lam / (xx * xx * vv);
				aa = interpolate_value<BoundaryPolicy::Clamp>(xx, x_array_ptr, a_array_ptr, array_size);
				vv = std::sqrt(lam / (xx * xx * aa));
			}
		}

		Real rho = aa * rho0_g;
		Real v = vv * cs0;
		Real vx = v * x / r;
		Real vy = v * y / r;
		Real vz = v * z / r;

		if (uniform_density > 0.0) {
			rho = uniform_density;
			v = 0.0;
			vx = 0.0;
			vy = 0.0;
			vz = 0.0;
		}

		const Real Eint = rho / mu * k_B * T0; // arbitrary choice, since Eint is not used in isothermal gas EOS
		const Real Ekin = 0.5 * rho * v * v;
		const Real Emag = 0.5 * B0 * B0;
		const Real Etot = Eint + Ekin + Emag;

		state_cc(i, j, k, HydroSystem<AccretionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<AccretionProblem>::energy_index) = Etot;
	});
}

template <> void QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	const double B_val = (dir == quokka::direction::x) ? B0 : 0.0;

	amrex::ParallelFor(indexRange,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k) { state_fc(i, j, k, Physics_Indices<AccretionProblem>::mhdFirstIndex) = B_val; });
}

template <> void QuokkaSimulation<AccretionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	if (!refine_center) {
		return;
	}

	const amrex::Real dx = geom[lev].CellSizeArray()[0];

	const auto &prob_lo = geom[lev].ProbLoArray();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx;
		const Real y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx;
		const Real z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx;
		const Real r = std::sqrt(x * x + y * y + z * z);
		if (r < 3.0 * dx) {
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
	pp.query("sink_file", sink_file);
	pp.query("turnon_fextract", turnon_fextract);
	pp.query("rho0", rho0);
	pp.query("uniform_density", uniform_density);
	pp.query("return_1_at_fail", return_1_at_fail);
	pp.query("t_end_over_t_b", t_end_over_t_b);
	pp.query("refine_center", refine_center);

	const double M_star_in_g = M_star_in_Msun * C::M_solar;
	const Real r_BH = C::Gconst * M_star_in_g / (cs0 * cs0);
	const Real t_BH = r_BH / cs0;
	const Real t_end = t_end_over_t_b * t_BH;

	// Problem initialization
	QuokkaSimulation<AccretionProblem> sim;
	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!
	// sim.initDt_ = 3.0e10;	      // ~1 kyr
	sim.tempFloor_ = 10.0; // K
	sim.stopTime_ = t_end;

	// initialize
	sim.setInitialConditions();

	if (amrex::ParallelDescriptor::IOProcessor()) {
		sim.userData_.time.push_back(0.0);
		sim.userData_.Mstar.push_back(M_star_in_g);
	}

	// set particle to live in refined region
	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->setForceFinestLevel(true);

	// get cell density as a function of x
	// Note: fextract does not work with MPI
	amrex::Vector<amrex::Real> position;
	amrex::Vector<amrex::Gpu::HostVector<amrex::Real>> values0;
	if (turnon_fextract) {
		std::tie(position, values0) = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true);
	}

	// get total gas mass of the initial state
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = sim.geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	// get total particle mass of the initial state
	amrex::Real const m_gas_init = sim.state_new_cc_[0].sum(HydroSystem<AccretionProblem>::density_index) * vol;

	// evolve
	sim.evolve();

	amrex::Vector<amrex::Gpu::HostVector<amrex::Real>> values1;
	if (turnon_fextract) {
		// get cell density as a function of x
		values1 = std::get<1>(fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0, true));
	}

	// get total gas mass of the final state
	amrex::Real const m_gas_final = sim.state_new_cc_[0].sum(HydroSystem<AccretionProblem>::density_index) * vol;

	// get total particle mass of the final state
	const int finest_level = sim.finestLevel();
	const auto &real_data_final = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::Sink)->getParticleDataAtLevel(finest_level).first;

	int status = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Real const m_stars_init = M_star_in_g;

		amrex::Real const m_stars_final =
		    std::accumulate(real_data_final.begin(), real_data_final.end(), 0.0, [](Real acc, const auto &p) { return acc + p[3]; });

		amrex::Print() << "Initial gas mass = " << m_gas_init << "\n";
		amrex::Print() << "Initial particle mass = " << m_stars_init << "\n";
		amrex::Print() << "Final gas mass = " << m_gas_final << "\n";
		amrex::Print() << "Final particle mass = " << m_stars_final << "\n";

		if (turnon_fextract) {
			const int nx = static_cast<int>(position.size());
			// const double m_tot_init = m_gas_init + m_stars_init;
			// const double m_tot_final = m_gas_final + m_stars_final;
			std::vector<double> x(nx);
			std::vector<double> rho(nx);
			std::vector<double> rho1(nx);

			for (int i = 0; i < nx; ++i) {
				x[i] = position[i];
				rho[i] = values0.at(HydroSystem<AccretionProblem>::density_index)[i] / rho0;
				rho1[i] = values1.at(HydroSystem<AccretionProblem>::density_index)[i] / rho0;
			}

			// Mass will not be conserved because of the open boundary conditions
			// // check mass conservation
			// const double rel_error_total_mass = std::abs(m_tot_final - m_tot_init) / m_tot_init;
			// amrex::Print() << "rel_error_total_mass = " << rel_error_total_mass << "\n";

#ifdef HAVE_PYTHON
			// plot density profile at beginning and end
			matplotlibcpp::clf();
			std::map<std::string, std::string> rho_args;
			rho_args["label"] = "Initial";
			rho_args["color"] = "red";
			matplotlibcpp::plot(x, rho, rho_args);
			std::map<std::string, std::string> rho1_args;
			rho1_args["label"] = "Final";
			rho1_args["color"] = "blue";
			matplotlibcpp::plot(x, rho1, rho1_args);
			matplotlibcpp::xlabel("x");
			matplotlibcpp::ylabel("Density / rho0");
			matplotlibcpp::legend();
			matplotlibcpp::save("sink_accretion_density_profile.png");
#endif
		}

		// plot particle mass vs time
		std::vector<Real> &time = sim.userData_.time;
		std::vector<Real> &Mstar_ = sim.userData_.Mstar;

		// print mass vs time
		for (int i = 0; i < static_cast<int>(time.size()); ++i) {
			amrex::Print() << "time = " << time[i] << ", Mstar = " << Mstar_[i] << "\n";
		}

		// compute exact accretion rate (MHD-aware Bondi formula)
		Real Mdot_exact = NAN;
		{
			const Real magnetic_pressure = 0.5 * B0 * B0;
			const Real rho_bg = uniform_density > 0.0 ? uniform_density : rho0;
			const Real beta = (rho_bg / mu) * C::k_B * T0 / magnetic_pressure;
			// MHD-aware fast magnetosonic speed: cf^2 = cs^2 * (1 + 2/beta) (isothermal)
			const Real cf_sqr = cs0 * cs0 * (1.0 + 2.0 / beta);
			const Real v_infty_sqr = 0.0;
			const Real r_BH_mhd = C::Gconst * M_star_in_g / (v_infty_sqr + cf_sqr);
			const Real lambda = gcem::exp(1.5) / 4.0;
			// M_dot = 4 pi rho_infty r_BH^2 * sqrt(v_infty^2 + lambda^2 cf^2), where lambda = exp(3/2) / 4
			Mdot_exact = 4.0 * M_PI * rho_bg * r_BH_mhd * r_BH_mhd * std::sqrt(v_infty_sqr + lambda * lambda * cf_sqr);
			amrex::Print() << "Mdot_exact = " << Mdot_exact << "\n";
		}

		// Estimate the accretion rate from the particle data
		const int last_step = static_cast<int>(time.size()) - 1;
		const int n_steps_to_average = last_step > 4 ? 4 : last_step;
		if (last_step >= 1) {
			const int first_step = last_step - n_steps_to_average;
			const Real Mdot_sim = (Mstar_[last_step] - Mstar_[first_step]) / (time[last_step] - time[first_step]);
			amrex::Print() << "Steady state Mdot_sim = " << Mdot_sim << "\n";

			// compute relative difference
			const Real rel_diff = std::abs(Mdot_sim - Mdot_exact) / Mdot_exact;
			amrex::Print() << "rel_diff = " << rel_diff << "\n";

			// Check if accretion rate is within tolerance when star mass is small (i.e. when the accretion rate is exactly Bondi accretion rate)
			if (return_1_at_fail) {
				const Real rel_diff_tol = 0.01;
				if (!(rel_diff < rel_diff_tol)) {
					status = 1;
				}
			}
		}

#ifdef HAVE_PYTHON
		matplotlibcpp::clf();
		matplotlibcpp::plot(time, Mstar_);
		// plot scatter plot of Mstar vs time
		matplotlibcpp::scatter(time, Mstar_, 10.0);
		matplotlibcpp::xlabel("Time");
		matplotlibcpp::ylabel("Particle Mass");
		const std::string title = std::format("Exact Bondi accretion rate = {:.2e} g/s", Mdot_exact);
		matplotlibcpp::title(title);
		matplotlibcpp::save("sink_accretion_particle_mass.png");
#endif
	}

	return status;
}
