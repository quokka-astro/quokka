#include "BuiltinFields.H"
#include "AMReX_Print.H"

// Temperature field implementation
void TemperatureField::init(const std::string &fieldName, const amrex::ParmParse &pp)
{
	fieldName_ = fieldName;
	pp.query("gamma", gamma_);
	pp.query("mean_molecular_weight", mean_molecular_weight_);
	pp.query("boltzmann_constant", boltzmann_constant_);
}

void TemperatureField::compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const
{
	auto const &stateArrays = state.const_arrays();
	auto const &outputArrays = mf.arrays();

	amrex::Real gamma = gamma_;
	amrex::Real mmw = mean_molecular_weight_;
	amrex::Real kb = boltzmann_constant_;

	amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = stateArrays[bx](i, j, k, 0);  // density
		const amrex::Real momx = stateArrays[bx](i, j, k, 1); // x1Momentum
		const amrex::Real momy = stateArrays[bx](i, j, k, 2); // x2Momentum
		const amrex::Real momz = stateArrays[bx](i, j, k, 3); // x3Momentum
		const amrex::Real etot = stateArrays[bx](i, j, k, 4); // energy

		// Compute kinetic energy
		const amrex::Real Ekin = (momx * momx + momy * momy + momz * momz) / (2.0 * rho);

		// Compute internal energy
		const amrex::Real Eint = etot - Ekin;

		// Compute temperature from ideal gas law
		const amrex::Real temperature = (gamma - 1.0) * Eint * mmw / (rho * kb);

		outputArrays[bx](i, j, k, ncomp) = temperature;
	});
}

// Vorticity field implementation
void VorticityField::init(const std::string &fieldName, const amrex::ParmParse &pp) { fieldName_ = fieldName; }

auto VorticityField::getComponentNames() const -> std::vector<std::string>
{
#if AMREX_SPACEDIM == 1
	return {"vorticity_x"};
#elif AMREX_SPACEDIM == 2
	return {"vorticity_z"};
#else
	return {"vorticity_x", "vorticity_y", "vorticity_z"};
#endif
}

void VorticityField::compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const
{
	auto const &stateArrays = state.const_arrays();
	auto const &outputArrays = mf.arrays();
	auto const &dx = geom.CellSizeArray();

	amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = stateArrays[bx](i, j, k, 0);  // density
		const amrex::Real momx = stateArrays[bx](i, j, k, 1); // x1Momentum
		const amrex::Real momy = stateArrays[bx](i, j, k, 2); // x2Momentum
		const amrex::Real momz = stateArrays[bx](i, j, k, 3); // x3Momentum

		// Compute velocities
		const amrex::Real vx = momx / rho;
		const amrex::Real vy = momy / rho;
		const amrex::Real vz = momz / rho;

		// Compute vorticity components (curl of velocity)
		// Note: This is a simplified calculation - proper vorticity requires ghost cells
		// and more sophisticated finite difference schemes

#if AMREX_SPACEDIM == 1
		// In 1D, vorticity is essentially zero
		outputArrays[bx](i, j, k, ncomp) = 0.0;
#elif AMREX_SPACEDIM == 2
		// In 2D, vorticity_z = dvy/dx - dvx/dy
		const amrex::Real dvydx = (i < mf.nGrowVect()[0]-1) ? 
			(stateArrays[bx](i+1, j, k, 2)/stateArrays[bx](i+1, j, k, 0) - 
			 stateArrays[bx](i-1, j, k, 2)/stateArrays[bx](i-1, j, k, 0)) / (2.0 * dx[0]) : 0.0;
		const amrex::Real dvxdy = (j < mf.nGrowVect()[1]-1) ? 
			(stateArrays[bx](i, j+1, k, 1)/stateArrays[bx](i, j+1, k, 0) - 
			 stateArrays[bx](i, j-1, k, 1)/stateArrays[bx](i, j-1, k, 0)) / (2.0 * dx[1]) : 0.0;
		outputArrays[bx](i, j, k, ncomp) = dvydx - dvxdy;
#else
		// In 3D, compute all three components
		// This is a simplified implementation
		outputArrays[bx](i, j, k, ncomp) = 0.0;     // vorticity_x
		outputArrays[bx](i, j, k, ncomp+1) = 0.0;   // vorticity_y
		outputArrays[bx](i, j, k, ncomp+2) = 0.0;   // vorticity_z
#endif
	});
}

// B-field divergence implementation
void BFieldDivergenceField::init(const std::string &fieldName, const amrex::ParmParse &pp) { fieldName_ = fieldName; }

void BFieldDivergenceField::compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const
{
	auto const &stateArrays = state.const_arrays();
	auto const &outputArrays = mf.arrays();
	auto const &dx = geom.CellSizeArray();

	amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// Compute divergence of B-field: div(B) = dBx/dx + dBy/dy + dBz/dz
		// This is a simplified finite difference calculation

		amrex::Real divB = 0.0;

		// Assume B-field components are at specific indices (these would need to be adjusted)
		// For MHD problems, B-field components are typically after momentum and energy
		const int bx_idx = 5; // x1BField index
		const int by_idx = 6; // x2BField index
		const int bz_idx = 7; // x3BField index

		// Compute dBx/dx
		if (i > 0 && i < mf.nGrowVect()[0] - 1) {
			const amrex::Real dBxdx = (stateArrays[bx](i + 1, j, k, bx_idx) - stateArrays[bx](i - 1, j, k, bx_idx)) / (2.0 * dx[0]);
			divB += dBxdx;
		}

#if AMREX_SPACEDIM >= 2
		// Compute dBy/dy
		if (j > 0 && j < mf.nGrowVect()[1] - 1) {
			const amrex::Real dBydy = (stateArrays[bx](i, j + 1, k, by_idx) - stateArrays[bx](i, j - 1, k, by_idx)) / (2.0 * dx[1]);
			divB += dBydy;
		}
#endif

#if AMREX_SPACEDIM == 3
		// Compute dBz/dz
		if (k > 0 && k < mf.nGrowVect()[2] - 1) {
			const amrex::Real dBzdz = (stateArrays[bx](i, j, k + 1, bz_idx) - stateArrays[bx](i, j, k - 1, bz_idx)) / (2.0 * dx[2]);
			divB += dBzdz;
		}
#endif

		outputArrays[bx](i, j, k, ncomp) = divB;
	});
}

// Sound speed implementation
void SoundSpeedField::init(const std::string &fieldName, const amrex::ParmParse &pp)
{
	fieldName_ = fieldName;
	pp.query("gamma", gamma_);
}

void SoundSpeedField::compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const
{
	auto const &stateArrays = state.const_arrays();
	auto const &outputArrays = mf.arrays();

	amrex::Real gamma = gamma_;

	amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = stateArrays[bx](i, j, k, 0);  // density
		const amrex::Real momx = stateArrays[bx](i, j, k, 1); // x1Momentum
		const amrex::Real momy = stateArrays[bx](i, j, k, 2); // x2Momentum
		const amrex::Real momz = stateArrays[bx](i, j, k, 3); // x3Momentum
		const amrex::Real etot = stateArrays[bx](i, j, k, 4); // energy

		// Compute kinetic energy
		const amrex::Real Ekin = (momx * momx + momy * momy + momz * momz) / (2.0 * rho);

		// Compute internal energy
		const amrex::Real Eint = etot - Ekin;

		// Compute pressure
		const amrex::Real pressure = (gamma - 1.0) * Eint;

		// Compute sound speed
		const amrex::Real cs = std::sqrt(gamma * pressure / rho);

		outputArrays[bx](i, j, k, ncomp) = cs;
	});
}

// Number density implementation
void NumberDensityField::init(const std::string &fieldName, const amrex::ParmParse &pp)
{
	fieldName_ = fieldName;
	pp.query("particle_mass", particle_mass_);
}

void NumberDensityField::compute(amrex::MultiFab &mf, const amrex::MultiFab &state, const amrex::Geometry &geom, amrex::Real time, int ncomp) const
{
	auto const &stateArrays = state.const_arrays();
	auto const &outputArrays = mf.arrays();

	amrex::Real pmass = particle_mass_;

	amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const amrex::Real rho = stateArrays[bx](i, j, k, 0); // density
		const amrex::Real ndens = rho / pmass;
		outputArrays[bx](i, j, k, ncomp) = ndens;
	});
}