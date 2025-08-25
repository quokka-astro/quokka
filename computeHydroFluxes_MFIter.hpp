// New implementation of computeHydroFluxes using MFIter loops to avoid temporary MultiFab allocations
template <typename problem_t>
auto QuokkaSimulation<problem_t>::computeHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc,
						     const int nvars, const int lev)
    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	const BL_PROFILE("QuokkaSimulation::computeHydroFluxes()");

	const auto ba = grids[lev];
	const auto dm = dmap[lev];
	const int reconstructGhost = 3; // reconstruct *two* additional cells outside valid region
	// we need two additional ghost cells in order to compute two ghost face velocities
	const int flatteningGhost = reconstructGhost + 1;

	// allocate output MultiFabs that we need to return
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fast_mhd_wavespeeds;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost - 1);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, reconstructGhost - 1);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			fast_mhd_wavespeeds[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructGhost - 1);
		}
	}

	// Use MFIter loop to process each box
	for (amrex::MFIter mfi(consVar_cc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.tilebox();
		
		// Grow box for ghost cells
		const amrex::Box bxg_cc = amrex::grow(bx, nghost_cc_);
		const amrex::Box bxg_flat = amrex::grow(bx, flatteningGhost);
		
		// Allocate temporary FArrayBoxes for this box only
		amrex::FArrayBox primVar_fab(bxg_cc, nvars);
		std::array<amrex::FArrayBox, 3> flatCoefs_fab;
		std::array<amrex::FArrayBox, AMREX_SPACEDIM> leftState_fab;
		std::array<amrex::FArrayBox, AMREX_SPACEDIM> rightState_fab;
		
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			flatCoefs_fab[idim].resize(bxg_flat, 1);
		}
		
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			const amrex::Box bx_face = amrex::surroundingNodes(bx, idim);
			const amrex::Box bxg_face = amrex::grow(bx_face, reconstructGhost);
			leftState_fab[idim].resize(bxg_face, nvars);
			rightState_fab[idim].resize(bxg_face, nvars);
		}
		
		// Get arrays for this box
		auto const &consVar = consVar_cc.const_array(mfi);
		auto primVar = primVar_fab.array();
		
		// Convert conserved to primitive variables for this box
		const amrex::Box primBox = primVar_fab.box();
		amrex::ParallelFor(primBox, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
			// Extract conserved variables
			const amrex::Real rho = consVar(i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real px1 = consVar(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const amrex::Real px2 = consVar(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const amrex::Real px3 = consVar(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const amrex::Real E = consVar(i, j, k, HydroSystem<problem_t>::energy_index);
			
			// Compute primitive variables
			if (n == HydroSystem<problem_t>::density_index) {
				primVar(i, j, k, n) = rho;
			} else if (n == HydroSystem<problem_t>::x1Velocity_index) {
				primVar(i, j, k, n) = px1 / rho;
			} else if (n == HydroSystem<problem_t>::x2Velocity_index) {
				primVar(i, j, k, n) = px2 / rho;
			} else if (n == HydroSystem<problem_t>::x3Velocity_index) {
				primVar(i, j, k, n) = px3 / rho;
			} else if (n == HydroSystem<problem_t>::pressure_index) {
				const amrex::Real vx1 = px1 / rho;
				const amrex::Real vx2 = px2 / rho;
				const amrex::Real vx3 = px3 / rho;
				const amrex::Real vsq = vx1 * vx1 + vx2 * vx2 + vx3 * vx3;
				const amrex::Real Eint = E - 0.5 * rho * vsq;
				primVar(i, j, k, n) = quokka::EOS<problem_t>::ComputePressure(rho, Eint);
			} else if (n >= HydroSystem<problem_t>::scalar0_index) {
				// passive scalars
				primVar(i, j, k, n) = consVar(i, j, k, n);
			}
		});
		
		// Compute flattening coefficients for each direction
		AMREX_D_TERM(
			{
				auto flatCoef = flatCoefs_fab[0].array();
				HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X1>(
					primVar_fab, flatCoef, flatCoefs_fab[0].box());
			},
			{
				auto flatCoef = flatCoefs_fab[1].array();
				HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X2>(
					primVar_fab, flatCoef, flatCoefs_fab[1].box());
			},
			{
				auto flatCoef = flatCoefs_fab[2].array();
				HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X3>(
					primVar_fab, flatCoef, flatCoefs_fab[2].box());
			}
		)
		
		// Compute fluxes for each direction
		AMREX_D_TERM(
			{
				// X1 direction
				auto leftState = leftState_fab[0].array();
				auto rightState = rightState_fab[0].array();
				auto fluxArray = flux[0].array(mfi);
				auto faceVelArray = facevel[0].array(mfi);
				auto fSpdsArray = (Physics_Traits<problem_t>::is_mhd_enabled) ? 
					fast_mhd_wavespeeds[0].array(mfi) : amrex::Array4<amrex::Real>{};
				
				hydroFluxFunctionMFIter<FluxDir::X1>(
					primVar_fab, leftState_fab[0], rightState_fab[0], 
					flux[0][mfi], facevel[0][mfi], 
					(Physics_Traits<problem_t>::is_mhd_enabled) ? &fast_mhd_wavespeeds[0][mfi] : nullptr,
					consVar_fc, flatCoefs_fab[0], flatCoefs_fab[1], flatCoefs_fab[2], 
					mfi, reconstructGhost, nvars);
			},
			{
				// X2 direction
				auto leftState = leftState_fab[1].array();
				auto rightState = rightState_fab[1].array();
				auto fluxArray = flux[1].array(mfi);
				auto faceVelArray = facevel[1].array(mfi);
				auto fSpdsArray = (Physics_Traits<problem_t>::is_mhd_enabled) ? 
					fast_mhd_wavespeeds[1].array(mfi) : amrex::Array4<amrex::Real>{};
				
				hydroFluxFunctionMFIter<FluxDir::X2>(
					primVar_fab, leftState_fab[1], rightState_fab[1], 
					flux[1][mfi], facevel[1][mfi], 
					(Physics_Traits<problem_t>::is_mhd_enabled) ? &fast_mhd_wavespeeds[1][mfi] : nullptr,
					consVar_fc, flatCoefs_fab[0], flatCoefs_fab[1], flatCoefs_fab[2], 
					mfi, reconstructGhost, nvars);
			},
			{
				// X3 direction
				auto leftState = leftState_fab[2].array();
				auto rightState = rightState_fab[2].array();
				auto fluxArray = flux[2].array(mfi);
				auto faceVelArray = facevel[2].array(mfi);
				auto fSpdsArray = (Physics_Traits<problem_t>::is_mhd_enabled) ? 
					fast_mhd_wavespeeds[2].array(mfi) : amrex::Array4<amrex::Real>{};
				
				hydroFluxFunctionMFIter<FluxDir::X3>(
					primVar_fab, leftState_fab[2], rightState_fab[2], 
					flux[2][mfi], facevel[2][mfi], 
					(Physics_Traits<problem_t>::is_mhd_enabled) ? &fast_mhd_wavespeeds[2][mfi] : nullptr,
					consVar_fc, flatCoefs_fab[0], flatCoefs_fab[1], flatCoefs_fab[2], 
					mfi, reconstructGhost, nvars);
			}
		)
	}

	// synchronization point to ensure all MFIter work is done
	amrex::Gpu::streamSynchronizeAll();

	// LOW LEVEL DEBUGGING: output all of the temporary MultiFabs
	if (lowLevelDebuggingOutput_ == 1) {
		// Note: We can't output primVar and flatCoefs here since they're not MultiFabs anymore
		// We'd need to restructure the debugging output if needed
		amrex::Print() << "WARNING: Low-level debugging output not fully supported with MFIter implementation\n";
	}

	// return flux and face-centered velocities
	return std::make_tuple(std::move(flux), std::move(facevel), std::move(fast_mhd_wavespeeds));
}

// New version of hydroFluxFunction for MFIter processing
template <typename problem_t>
template <FluxDir DIR>
void QuokkaSimulation<problem_t>::hydroFluxFunctionMFIter(
	amrex::FArrayBox &primVar_fab, amrex::FArrayBox &leftState_fab, amrex::FArrayBox &rightState_fab,
	amrex::FArrayBox &flux_fab, amrex::FArrayBox &faceVel_fab, amrex::FArrayBox *x1FSpds_fab,
	std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, 
	amrex::FArrayBox const &x1Flat_fab, amrex::FArrayBox const &x2Flat_fab, amrex::FArrayBox const &x3Flat_fab,
	amrex::MFIter const &mfi, const int ng_reconstruct, const int nvars)
{
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		// Handle MHD magnetic field averaging
		std::array<int, 3> delta_x2 = {0, 0, 0};
		std::array<int, 3> delta_x3 = {0, 0, 0};
		const amrex::Array4<const amrex::Real> *x2State_fc_arr = nullptr;
		const amrex::Array4<const amrex::Real> *x3State_fc_arr = nullptr;
		
		if constexpr (DIR == FluxDir::X1) {
			delta_x2[1] = 1;
			delta_x3[2] = 1;
			x2State_fc_arr = &consVar_fc[1].const_array(mfi);
			x3State_fc_arr = &consVar_fc[2].const_array(mfi);
		} else if constexpr (DIR == FluxDir::X2) {
			delta_x2[2] = 1;
			delta_x3[0] = 1;
			x2State_fc_arr = &consVar_fc[2].const_array(mfi);
			x3State_fc_arr = &consVar_fc[0].const_array(mfi);
		} else if constexpr (DIR == FluxDir::X3) {
			delta_x2[0] = 1;
			delta_x3[1] = 1;
			x2State_fc_arr = &consVar_fc[0].const_array(mfi);
			x3State_fc_arr = &consVar_fc[1].const_array(mfi);
		}
		
		auto primVar_arr = primVar_fab.array();
		const amrex::Box &primBox = primVar_fab.box();
		const auto &x2State_fc = *x2State_fc_arr;
		const auto &x3State_fc = *x3State_fc_arr;
		
		amrex::ParallelFor(primBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const double bx2_m = x2State_fc(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const double bx2_p = x2State_fc(i + delta_x2[0], j + delta_x2[1], k + delta_x2[2], Physics_Indices<problem_t>::mhdFirstIndex);
			primVar_arr(i, j, k, HydroSystem<problem_t>::x2Magnetic_index) = 0.5 * (bx2_m + bx2_p);

			const double bx3_m = x3State_fc(i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
			const double bx3_p = x3State_fc(i + delta_x3[0], j + delta_x3[1], k + delta_x3[2], Physics_Indices<problem_t>::mhdFirstIndex);
			primVar_arr(i, j, k, HydroSystem<problem_t>::x3Magnetic_index) = 0.5 * (bx3_m + bx3_p);
		});
	}

	// Reconstruct states
	if (reconstructionOrder_ == 5) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(
			primVar_fab, leftState_fab, rightState_fab, ng_reconstruct, nvars);
	} else if (reconstructionOrder_ == 3) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(
			primVar_fab, leftState_fab, rightState_fab, ng_reconstruct, nvars);
	} else if (reconstructionOrder_ == 2) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, SlopeLimiter::minmod>(
			primVar_fab, leftState_fab, rightState_fab, ng_reconstruct, nvars);
	} else if (reconstructionOrder_ == 1) {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(
			primVar_fab, leftState_fab, rightState_fab, ng_reconstruct, nvars);
	} else {
		amrex::Abort("Invalid reconstruction order specified!");
	}

	// Flatten shocks
	HydroSystem<problem_t>::template FlattenShocks<DIR>(
		primVar_fab, x1Flat_fab, x2Flat_fab, x3Flat_fab, 
		leftState_fab, rightState_fab, ng_reconstruct, nvars);

	// Compute fluxes
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		auto const &fc_arr = consVar_fc[static_cast<int>(DIR)].const_array(mfi);
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLD, DIR>(
			flux_fab, faceVel_fab, leftState_fab, rightState_fab, primVar_fab,
			artificialViscosityK_, x1FSpds_fab, &fc_arr, nghost_vel_);
	} else {
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLC, DIR>(
			flux_fab, faceVel_fab, leftState_fab, rightState_fab, primVar_fab,
			artificialViscosityK_, nullptr, nullptr, nghost_vel_);
	}
}