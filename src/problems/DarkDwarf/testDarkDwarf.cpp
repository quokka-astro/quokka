#include <algorithm>
#include <cmath>

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "particles/particle_utils.hpp"
#include "physics_info.hpp"

#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"

#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Vector.H"

namespace
{
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real surfaceDensity(amrex::Real R, amrex::Real Sigma0, amrex::Real Rd, amrex::Real alpha, amrex::Real beta)
{
	return Sigma0 * std::exp(-R / Rd - beta * std::exp(-alpha * R / Rd));
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real hernquistVertical(amrex::Real R, amrex::Real z, amrex::Real Mhalo, amrex::Real a)
{
	amrex::Real const r = std::sqrt(R * R + z * z);

	if (r <= 0.0) {
		return 0.0;
	}

	return C::Gconst * Mhalo * z / (r * (r + a) * (r + a));
}

// Vertical ODE
struct VerticalState {
	amrex::Real lnRho;
	amrex::Real gGas;
	amrex::Real columnHalf;
};

struct VerticalDeriv {
	amrex::Real dlnRhoDz;
	amrex::Real dgGasDz;
	amrex::Real dColumnDz;
};

VerticalDeriv verticalRHS(amrex::Real R, amrex::Real z, VerticalState const &s, amrex::Real cs, amrex::Real Mhalo, amrex::Real a)
{
	amrex::Real const rho = std::exp(s.lnRho);

	amrex::Real const gDM = hernquistVertical(R, z, Mhalo, a);

	VerticalDeriv rhs{};

	rhs.dlnRhoDz = -(s.gGas + gDM) / (cs * cs);

	rhs.dgGasDz = 4.0 * M_PI * C::Gconst * rho;

	rhs.dColumnDz = rho;

	return rhs;
}

VerticalState rk4Step(amrex::Real R, amrex::Real z, amrex::Real dz, VerticalState const &y, amrex::Real cs, amrex::Real Mhalo, amrex::Real a)
{
	auto const k1 = verticalRHS(R, z, y, cs, Mhalo, a);

	VerticalState y2{y.lnRho + 0.5 * dz * k1.dlnRhoDz, y.gGas + 0.5 * dz * k1.dgGasDz, y.columnHalf + 0.5 * dz * k1.dColumnDz};

	auto const k2 = verticalRHS(R, z + 0.5 * dz, y2, cs, Mhalo, a);

	VerticalState y3{y.lnRho + 0.5 * dz * k2.dlnRhoDz, y.gGas + 0.5 * dz * k2.dgGasDz, y.columnHalf + 0.5 * dz * k2.dColumnDz};

	auto const k3 = verticalRHS(R, z + 0.5 * dz, y3, cs, Mhalo, a);

	VerticalState y4{y.lnRho + dz * k3.dlnRhoDz, y.gGas + dz * k3.dgGasDz, y.columnHalf + dz * k3.dColumnDz};

	auto const k4 = verticalRHS(R, z + dz, y4, cs, Mhalo, a);

	VerticalState out{};

	out.lnRho = y.lnRho + dz / 6.0 * (k1.dlnRhoDz + 2.0 * k2.dlnRhoDz + 2.0 * k3.dlnRhoDz + k4.dlnRhoDz);

	out.gGas = y.gGas + dz / 6.0 * (k1.dgGasDz + 2.0 * k2.dgGasDz + 2.0 * k3.dgGasDz + k4.dgGasDz);

	out.columnHalf = y.columnHalf + dz / 6.0 * (k1.dColumnDz + 2.0 * k2.dColumnDz + 2.0 * k3.dColumnDz + k4.dColumnDz);

	return out;
}

amrex::Real integratedSurfaceDensity(amrex::Real R, amrex::Real rhoMid, amrex::Real cs, amrex::Real Mhalo, amrex::Real a, amrex::Real zMax, int nSteps)
{
	amrex::Real const dz = zMax / nSteps;

	VerticalState s{std::log(rhoMid),
			0.0, // symmetry: g_gas(z=0)=0
			0.0};

	amrex::Real z = 0.0;

	for (int n = 0; n < nSteps; ++n) {
		s = rk4Step(R, z, dz, s, cs, Mhalo, a);

		z += dz;
	}

	return 2.0 * s.columnHalf;
}

amrex::Real solveMidplaneDensity(amrex::Real R, amrex::Real SigmaTarget, amrex::Real cs, amrex::Real Mhalo, amrex::Real a, amrex::Real zMax, int nSteps)
{
	amrex::Real const rhoSG = M_PI * C::Gconst * SigmaTarget * SigmaTarget / (2.0 * cs * cs);

	amrex::Real rhoLo = rhoSG * 1.0e-4;
	amrex::Real rhoHi = rhoSG * 1.0e4;

	while (integratedSurfaceDensity(R, rhoLo, cs, Mhalo, a, zMax, nSteps) > SigmaTarget) {
		rhoLo *= 0.1;
	}

	while (integratedSurfaceDensity(R, rhoHi, cs, Mhalo, a, zMax, nSteps) < SigmaTarget) {
		rhoHi *= 10.0;
	}

	amrex::Real logLo = std::log(rhoLo);
	amrex::Real logHi = std::log(rhoHi);

	for (int iter = 0; iter < 50; ++iter) {
		amrex::Real const logMid = 0.5 * (logLo + logHi);

		amrex::Real const rhoMid = std::exp(logMid);

		amrex::Real const SigmaMid = integratedSurfaceDensity(R, rhoMid, cs, Mhalo, a, zMax, nSteps);

		if (SigmaMid < SigmaTarget) {
			logLo = logMid;
		} else {
			logHi = logMid;
		}
	}

	return std::exp(0.5 * (logLo + logHi));
}

// bilinear interpolation
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real interpolateDensity(amrex::Real R, amrex::Real z, amrex::Real const *table, int nR, int nZ, amrex::Real Rmax, amrex::Real Zmax)
{
	z = std::abs(z);

	amrex::Real const fr = amrex::min(static_cast<amrex::Real>(nR - 1), R / Rmax * (nR - 1));

	amrex::Real const fz = amrex::min(static_cast<amrex::Real>(nZ - 1), z / Zmax * (nZ - 1));

	int const i0 = amrex::min(static_cast<int>(fr), nR - 2);

	int const k0 = amrex::min(static_cast<int>(fz), nZ - 2);

	amrex::Real const wr = fr - i0;
	amrex::Real const wz = fz - k0;

	auto const at = [=] AMREX_GPU_DEVICE(int ir, int iz) { return table[ir * nZ + iz]; };

	amrex::Real const rho00 = at(i0, k0);
	amrex::Real const rho10 = at(i0 + 1, k0);
	amrex::Real const rho01 = at(i0, k0 + 1);
	amrex::Real const rho11 = at(i0 + 1, k0 + 1);

	return (1.0 - wr) * (1.0 - wz) * rho00 + wr * (1.0 - wz) * rho10 + (1.0 - wr) * wz * rho01 + wr * wz * rho11;
}

} // namespace

struct DarkDwarf {};

template <> struct quokka::EOS_Traits<DarkDwarf> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<DarkDwarf> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
};

template <> struct Particle_Traits<DarkDwarf> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct SimulationData<DarkDwarf> {
	bool tables_ready = false;

	int nRTable = 256;
	int nZTable = 512;

	amrex::Real RTableMax{};
	amrex::Real ZTableMax{};

	amrex::Real cs{};
	amrex::Real rho_cut{};
	amrex::Real rho_bg{};
	amrex::Real T_bg = 1.0e6;

	amrex::Gpu::DeviceVector<amrex::Real> rhoTableDevice;
	amrex::Gpu::DeviceVector<amrex::Real> vphiTableDevice;
};

template <> void QuokkaSimulation<DarkDwarf>::preCalculateInitialConditions()
{
	auto &data = userData_;

	if (data.tables_ready) {
		return;
	}

	// Inputs
	amrex::ParmParse const pp("dark_dwarf");

	amrex::Real alpha;
	amrex::Real beta;

	amrex::Real Sigma0_Msun_pc2;
	amrex::Real Rd_kpc;
	amrex::Real cs_kms;

	amrex::Real M200_Msun;
	amrex::Real concentration;
	amrex::Real H0_kms_Mpc = 70.0;

	amrex::Real MD;

	pp.get("alpha", alpha);
	pp.get("beta", beta);

	pp.get("Sigma0_Msun_pc2", Sigma0_Msun_pc2);
	pp.get("Rd_kpc", Rd_kpc);
	pp.get("cs_kms", cs_kms);

	pp.get("M200_Msun", M200_Msun);
	pp.get("concentration", concentration);
	pp.query("H0_kms_Mpc", H0_kms_Mpc);

	pp.get("MD", MD);

	// input conversion
	amrex::Real const Sigma0 = Sigma0_Msun_pc2 * C::M_solar / (C::parsec * C::parsec);

	amrex::Real const Rd = Rd_kpc * 1.0e3 * C::parsec;

	amrex::Real const cs = cs_kms * 1.0e5;

	amrex::Real const M200 = M200_Msun * C::M_solar;

	amrex::Real const H0 = H0_kms_Mpc * 1.0e5 / (1.0e6 * C::parsec);

	amrex::Real const rhoCrit = 3.0 * H0 * H0 / (8.0 * M_PI * C::Gconst);

	amrex::Real const R200 = std::cbrt(3.0 * M200 / (4.0 * M_PI * 200.0 * rhoCrit));

	amrex::Real const Mhalo = (1.0 - MD) * M200;

	amrex::Real const fc = std::log(1.0 + concentration) - concentration / (1.0 + concentration);

	amrex::Real const a = R200 / concentration * std::sqrt(2.0 * fc);

	// amrex::Print() << "M200 = " << M200_Msun << " Msun\n"
	//	       << "R200 = " << R200 / (1.0e3 * C::parsec) << " kpc\n"
	//	       << "rs = " << rs / (1.0e3 * C::parsec) << " kpc\n";

	data.cs = cs;

	int const nRTable = data.nRTable;
	int const nZTable = data.nZTable;
	int const nIntegrate = 2048;

	data.RTableMax = 10.0e3 * C::parsec;
	data.ZTableMax = 10.0e3 * C::parsec;

	amrex::Real const RTableMax = data.RTableMax;
	amrex::Real const ZTableMax = data.ZTableMax;

	amrex::Real const ZNormMax = 20.0 * 1.0e3 * C::parsec;

	amrex::Vector<amrex::Real> rhoTableHost(nRTable * nZTable);

	for (int ir = 0; ir < nRTable; ++ir) {

		amrex::Real const R = RTableMax * static_cast<amrex::Real>(ir) / static_cast<amrex::Real>(nRTable - 1);

		amrex::Real const Sigma = surfaceDensity(R, Sigma0, Rd, alpha, beta);

		amrex::Real const rhoMid = solveMidplaneDensity(R, Sigma, cs, Mhalo, a, ZNormMax, nIntegrate);

		VerticalState s{std::log(rhoMid), 0.0, 0.0};

		rhoTableHost[ir * nZTable] = rhoMid;

		amrex::Real const dzTable = ZTableMax / static_cast<amrex::Real>(nZTable - 1);

		amrex::Real z = 0.0;

		for (int iz = 1; iz < nZTable; ++iz) {

			s = rk4Step(R, z, dzTable, s, cs, Mhalo, a);

			z += dzTable;

			rhoTableHost[ir * nZTable + iz] = std::exp(s.lnRho);
		}
	}
	// vphi table
	amrex::Vector<amrex::Real> vphiTableHost(nRTable, 0.0);

	amrex::Real const dR = RTableMax / (nRTable - 1);

	for (int ir = 1; ir < nRTable - 1; ++ir) {
		amrex::Real const R = ir * dR;

		amrex::Real const rhoLeft = rhoTableHost[(ir - 1) * nZTable];
		amrex::Real const rhoRight = rhoTableHost[(ir + 1) * nZTable];

		amrex::Real const dlnrho_dR = (std::log(rhoRight) - std::log(rhoLeft)) / (2.0 * dR);

		amrex::Real const gR = C::Gconst * Mhalo / ((R + a) * (R + a));

		amrex::Real const vphi2 = R * (gR + cs * cs * dlnrho_dR);

		vphiTableHost[ir] = std::sqrt(std::max(0.0, vphi2));
	}

	// cutoff
	amrex::Real const rho_max = *std::max_element(rhoTableHost.begin(), rhoTableHost.end());
	data.rho_cut = rho_max * 1.0e-3;
	data.rho_bg = rho_max * 1.0e-6;

	vphiTableHost[nRTable - 1] = vphiTableHost[nRTable - 2];
	// copy rhoTable to Device
	data.rhoTableDevice.resize(rhoTableHost.size());

	amrex::Gpu::copy(amrex::Gpu::hostToDevice, rhoTableHost.begin(), rhoTableHost.end(), data.rhoTableDevice.begin());

	// copy  vphiTable to Device
	data.vphiTableDevice.resize(vphiTableHost.size());

	amrex::Gpu::copy(amrex::Gpu::hostToDevice, vphiTableHost.begin(), vphiTableHost.end(), data.vphiTableDevice.begin());

	amrex::Gpu::streamSynchronize();
	data.tables_ready = true;
}

template <> void QuokkaSimulation<DarkDwarf>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	auto const &data = userData_;

	AMREX_ALWAYS_ASSERT(data.tables_ready);

	auto const dx = grid_elem.dx_;
	auto const prob_lo = grid_elem.prob_lo_;
	auto const prob_hi = grid_elem.prob_hi_;

	auto const &indexRange = grid_elem.indexRange_;
	auto const &state_cc = grid_elem.array_;

	amrex::Real const x_center = 0.5 * (prob_lo[0] + prob_hi[0]);
	amrex::Real const y_center = 0.5 * (prob_lo[1] + prob_hi[1]);
	amrex::Real const z_center = 0.5 * (prob_lo[2] + prob_hi[2]);

	int const ncomp_cc = Physics_Indices<DarkDwarf>::nvarTotal_cc;

	int const nRTable = data.nRTable;
	int const nZTable = data.nZTable;

	amrex::Real const RTableMax = data.RTableMax;
	amrex::Real const ZTableMax = data.ZTableMax;

	amrex::Real const cs = data.cs;
	amrex::Real const rho_cut = data.rho_cut;
	amrex::Real const rho_bg = data.rho_bg;
	amrex::Real const T_bg = data.T_bg;

	amrex::Real const *rhoTablePtr = data.rhoTableDevice.data();
	amrex::Real const *vphiTablePtr = data.vphiTableDevice.data();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.0;
		}

		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];

		amrex::Real const xp = x - x_center;
		amrex::Real const yp = y - y_center;
		amrex::Real const zp = z - z_center;

		amrex::Real const R = std::sqrt(xp * xp + yp * yp);
		// Density

		amrex::Real rho_disk = 0.0;

		if (R <= RTableMax && std::abs(zp) <= ZTableMax) {
			rho_disk = interpolateDensity(R, zp, rhoTablePtr, nRTable, nZTable, RTableMax, ZTableMax);
		}

		bool const is_disk = (rho_disk >= rho_cut);

		amrex::Real const rho = is_disk ? rho_disk : rho_bg;
		amrex::Real const P = rho * cs * cs;

		amrex::Real vphi = 0.0;
		// rotational vel
		if (is_disk && R > 0.0) {
			amrex::Real const fr = amrex::min(static_cast<amrex::Real>(nRTable - 1), R / RTableMax * (nRTable - 1));
			int const i0 = amrex::min(static_cast<int>(fr), nRTable - 2);

			amrex::Real const w = fr - i0;

			vphi = (1.0 - w) * vphiTablePtr[i0] + w * vphiTablePtr[i0 + 1];
		}

		amrex::Real vx = 0.0;
		amrex::Real vy = 0.0;

		if (is_disk && R > 0.0) {
			vx = -vphi * yp / R;
			vy = vphi * xp / R;
		}

		// apply density and momentum
		state_cc(i, j, k, HydroSystem<DarkDwarf>::density_index) = rho;

		state_cc(i, j, k, HydroSystem<DarkDwarf>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<DarkDwarf>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<DarkDwarf>::x3Momentum_index) = 0.0;
		// internal energy
		amrex::Real const eint = is_disk ? quokka::EOS<DarkDwarf>::ComputeEintFromPres(rho, P) : quokka::EOS<DarkDwarf>::ComputeEintFromTgas(rho, T_bg);

		// rotational kinetic energy
		amrex::Real const ekin = 0.5 * rho * (vx * vx + vy * vy);

		// total energy
		state_cc(i, j, k, HydroSystem<DarkDwarf>::energy_index) = eint + ekin;

		state_cc(i, j, k, HydroSystem<DarkDwarf>::internalEnergy_index) = eint;
	});

	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<DarkDwarf>::createInitialCICParticles()
{
	// read particles from ASCII file
	amrex::ParmParse const pp("dark_dwarf");
	std::string filename;
	pp.query("particle_file", filename);

	amrex::Print() << "\nReading particles from ASCII file " << filename << "...\n";
	CICParticles->SetVerbose(1);
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->InitFromAsciiFile(filename, nreal_extra, nullptr);
	amrex::Print() << "\n";
}

template <> void QuokkaSimulation<DarkDwarf>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	amrex::ParmParse const pp("dark_dwarf");

	int cyl_max;
	pp.get("cylinder_max_level", cyl_max);

	if (lev < cyl_max) {
		// Refinement cylinder parameters, in kpc
		amrex::Real Rcyl_kpc;
		amrex::Real Hcyl_kpc;
		amrex::Real shrinkR_kpc = 0.0;
		amrex::Real shrinkH_kpc = 0.0;
		amrex::Real Rinner_kpc = 0.0;

		pp.get("refine_Rcyl_kpc", Rcyl_kpc);
		pp.get("refine_Hcyl_kpc", Hcyl_kpc);
		pp.query("refine_Rcyl_shrink_per_level_kpc", shrinkR_kpc);
		pp.query("refine_Hcyl_shrink_per_level_kpc", shrinkH_kpc);
		pp.query("refine_Rcyl_inner_kpc", Rinner_kpc);

		// Convert to CGS
		constexpr amrex::Real kpc = 1.0e3 * C::parsec;

		amrex::Real const Rcyl = Rcyl_kpc * kpc;
		amrex::Real const Hcyl = Hcyl_kpc * kpc;
		amrex::Real const shrinkR = shrinkR_kpc * kpc;
		amrex::Real const shrinkH = shrinkH_kpc * kpc;
		amrex::Real const Rinner = Rinner_kpc * kpc;

		// Progressively shrink the cylinder on finer levels
		amrex::Real const Rlevel = std::max(Rcyl - lev * shrinkR, std::max(0.3 * Rcyl, Rinner));

		amrex::Real const Hlevel = std::max(Hcyl - lev * shrinkH, 0.3 * Hcyl);

		auto const prob_lo = geom[lev].ProbLoArray();
		auto const prob_hi = geom[lev].ProbHiArray();
		auto const dx = geom[lev].CellSizeArray();
		auto const tag = tags.arrays();

		// Same center convention as the gas disk IC
		amrex::Real const xc = 0.5 * (prob_lo[0] + prob_hi[0]);
		amrex::Real const yc = 0.5 * (prob_lo[1] + prob_hi[1]);
		amrex::Real const zc = 0.5 * (prob_lo[2] + prob_hi[2]);

		constexpr int subsamples = 4;

		amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			amrex::Real const x0 = prob_lo[0] + i * dx[0];
			amrex::Real const y0 = prob_lo[1] + j * dx[1];
			amrex::Real const z0 = prob_lo[2] + k * dx[2];

			for (int ia = 0; ia < subsamples; ++ia) {
				amrex::Real const wx = static_cast<amrex::Real>(ia) / (subsamples - 1);

				amrex::Real const x = x0 + wx * dx[0];

				for (int ib = 0; ib < subsamples; ++ib) {
					amrex::Real const wy = static_cast<amrex::Real>(ib) / (subsamples - 1);

					amrex::Real const y = y0 + wy * dx[1];

					for (int ic = 0; ic < subsamples; ++ic) {
						amrex::Real const wz = static_cast<amrex::Real>(ic) / (subsamples - 1);

						amrex::Real const z = z0 + wz * dx[2];

						amrex::Real const xp = x - xc;
						amrex::Real const yp = y - yc;
						amrex::Real const zp = z - zc;

						amrex::Real const R = std::sqrt(xp * xp + yp * yp);

						if (R >= Rinner && R < Rlevel && std::abs(zp) < Hlevel) {
							tag[bx](i, j, k) = amrex::TagBox::SET;
							return;
						}
					}
				}
			}
		});
	}

	// Jeans refinement
	auto const &state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();

	amrex::Real const dx = geom[lev].CellSizeArray()[0];

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		amrex::Real const rho = state[bx](i, j, k, HydroSystem<DarkDwarf>::density_index);

		if (rho <= 0.0) {
			return;
		}

		amrex::Real const P = HydroSystem<DarkDwarf>::ComputePressure(state[bx], i, j, k);
		amrex::Real const cs = quokka::EOS<DarkDwarf>::ComputeSoundSpeed(rho, P);
		amrex::Real const rhoJeans = quokka::ParticleUtils::computeJeansDensity(cs, dx);

		if (rho > rhoJeans) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});

	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	QuokkaSimulation<DarkDwarf> sim;
	sim.setInitialConditions();
	sim.evolve();

	return 0;
}
