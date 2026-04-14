#ifndef HYDRO_RK_POLICY_HPP_ // NOLINT
#define HYDRO_RK_POLICY_HPP_
//==============================================================================
// Quokka -- a radiation-hydrodynamics code built on AMReX
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================

template <typename problem_t> struct HydroRKPolicy
{
	QuokkaSimulation<problem_t> &sim_;
	int lev_ = -1;
	int nghost_Riemann_ = 0;
	amrex::FluxRegister *step_fr_as_crse_ = nullptr;
	amrex::FluxRegister *step_fr_as_fine_ = nullptr;
	amrex::EdgeFluxRegister *step_emf_as_crse_ = nullptr;
	amrex::EdgeFluxRegister *step_emf_as_fine_ = nullptr;

	[[nodiscard]] auto nghost_cc() const -> int { return sim_.nghost_cc_; }
	[[nodiscard]] auto nghost_fc() const -> int { return sim_.nghost_fc_; }

	template <FluxDir DIR>
	static AMREX_FORCE_INLINE void computeCCPerpBfieldComps(QuokkaSimulation<problem_t> const &sim, amrex::MultiFab &cc_bfield_perp_comps_mf,
								std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc)
	{
		std::array<int, 3> delta_x2{0, 0, 0};
		std::array<int, 3> delta_x3{0, 0, 0};

		amrex::MultiArray4<const amrex::Real> x2State_fc_bfield_in;
		amrex::MultiArray4<const amrex::Real> x3State_fc_bfield_in;
		if constexpr (DIR == FluxDir::X1) {
			x2State_fc_bfield_in = consVar_fc[1].const_arrays();
			x3State_fc_bfield_in = consVar_fc[2].const_arrays();
			delta_x2[1] = 1;
			delta_x3[2] = 1;
		} else if constexpr (DIR == FluxDir::X2) {
			x2State_fc_bfield_in = consVar_fc[2].const_arrays();
			x3State_fc_bfield_in = consVar_fc[0].const_arrays();
			delta_x2[2] = 1;
			delta_x3[0] = 1;
		} else if constexpr (DIR == FluxDir::X3) {
			x2State_fc_bfield_in = consVar_fc[0].const_arrays();
			x3State_fc_bfield_in = consVar_fc[1].const_arrays();
			delta_x2[0] = 1;
			delta_x3[1] = 1;
		}

		auto cc_bfield_perp_comps_out = cc_bfield_perp_comps_mf.arrays();
		constexpr int b_comp = Physics_Indices<problem_t>::mhdFirstIndex;

		amrex::IntVect ng{AMREX_D_DECL(sim.nghost_fc_, sim.nghost_fc_, sim.nghost_fc_)};
		amrex::ParallelFor(cc_bfield_perp_comps_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
			const amrex::Real bx2_m = x2State_fc_bfield_in[bx](i, j, k, b_comp);
			const amrex::Real bx2_p = x2State_fc_bfield_in[bx](i + delta_x2[0], j + delta_x2[1], k + delta_x2[2], b_comp);
			cc_bfield_perp_comps_out[bx](i, j, k, 0) = 0.5 * (bx2_m + bx2_p);

			const amrex::Real bx3_m = x3State_fc_bfield_in[bx](i, j, k, b_comp);
			const amrex::Real bx3_p = x3State_fc_bfield_in[bx](i + delta_x3[0], j + delta_x3[1], k + delta_x3[2], b_comp);
			cc_bfield_perp_comps_out[bx](i, j, k, 1) = 0.5 * (bx3_m + bx3_p);
		});
	}

	template <FluxDir DIR>
	static void hydroFluxFunction(QuokkaSimulation<problem_t> const &sim, amrex::MultiFab &primVar_mf, amrex::MultiFab &cc_bfield_perp_comps_mf,
				      amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield,
				      amrex::MultiFab &rightState_bfield, amrex::MultiFab &flux, amrex::MultiFab &faceVel, amrex::MultiFab &x1FSpds,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, amrex::MultiFab const &x1Flat,
				      amrex::MultiFab const &x2Flat, amrex::MultiFab const &x3Flat, int ng_reconstruct, int nvars, int nghost_Riemann)
	{
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			computeCCPerpBfieldComps<DIR>(sim, cc_bfield_perp_comps_mf, consVar_fc);
		}

		if (sim.reconstructionOrder_ == 5) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM_EP<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield,
												   ng_reconstruct, 2);
			}
		} else if (sim.reconstructionOrder_ == 3) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield,
												ng_reconstruct, 2);
			}
		} else if (sim.reconstructionOrder_ == 2) {
			HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars,
												 sim.plmLimiter_);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield,
												ng_reconstruct, 2, sim.plmLimiter_);
			}
		} else if (sim.reconstructionOrder_ == 1) {
			HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield,
												     ng_reconstruct, 2);
			}
		} else {
			amrex::Abort("Invalid reconstruction order specified!");
		}

		HydroSystem<problem_t>::template FlattenShocks<DIR>(primVar_mf, x1Flat, x2Flat, x3Flat, leftState, rightState, ng_reconstruct, nvars);

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLD, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
												 rightState_bfield, primVar_mf, sim.artificialViscosityK_, &x1FSpds,
												 &consVar_fc[static_cast<int>(DIR)], nghost_Riemann);
		} else {
			HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLC, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
												 rightState_bfield, primVar_mf, sim.artificialViscosityK_, nullptr,
												 nullptr, nghost_Riemann);
		}
	}

	static auto computeHydroFluxes(QuokkaSimulation<problem_t> &sim, amrex::MultiFab const &consVar_cc,
				       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, int nvars, int nghost_Riemann, int lev)
	    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>,
			  std::array<amrex::MultiFab, AMREX_SPACEDIM>>
	{
		const BL_PROFILE("HydroRKPolicy::computeHydroFluxes()");

		const auto ba = sim.grids[lev];
		const auto dm = sim.dmap[lev];

		const int reconstructGhost = nghost_Riemann + 1;
		const int flatteningGhost = reconstructGhost + 1;

		amrex::MultiFab primVar(ba, dm, nvars, sim.nghost_cc_);
		amrex::MultiFab cc_bfield_perp_comps(ba, dm, 2, sim.nghost_cc_);
		std::array<amrex::MultiFab, 3> flatCoefs;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState_bfield;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState_bfield;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> fast_mhd_wavespeeds;

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			flatCoefs[idim] = amrex::MultiFab(ba, dm, 1, flatteningGhost);
		}

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
			leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost);
			rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost);
			leftState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructGhost);
			rightState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructGhost);
			flux[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost - 1);
			facevel[idim] = amrex::MultiFab(ba_face, dm, 1, reconstructGhost - 1);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				fast_mhd_wavespeeds[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructGhost - 1);
			}
		}

		HydroSystem<problem_t>::ConservedToPrimitive(consVar_cc, consVar_fc, primVar, sim.nghost_cc_);

		AMREX_D_TERM(HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X1>(primVar, flatCoefs[0], flatteningGhost);
			     , HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X2>(primVar, flatCoefs[1], flatteningGhost);
			     , HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X3>(primVar, flatCoefs[2], flatteningGhost);)

		AMREX_D_TERM(hydroFluxFunction<FluxDir::X1>(sim, primVar, cc_bfield_perp_comps, leftState[0], rightState[0], leftState_bfield[0],
							    rightState_bfield[0], flux[0], facevel[0], fast_mhd_wavespeeds[0], consVar_fc, flatCoefs[0],
							    flatCoefs[1], flatCoefs[2], reconstructGhost, nvars, nghost_Riemann);
			     , hydroFluxFunction<FluxDir::X2>(sim, primVar, cc_bfield_perp_comps, leftState[1], rightState[1], leftState_bfield[1],
							      rightState_bfield[1], flux[1], facevel[1], fast_mhd_wavespeeds[1], consVar_fc, flatCoefs[0],
							      flatCoefs[1], flatCoefs[2], reconstructGhost, nvars, nghost_Riemann);
			     , hydroFluxFunction<FluxDir::X3>(sim, primVar, cc_bfield_perp_comps, leftState[2], rightState[2], leftState_bfield[2],
							      rightState_bfield[2], flux[2], facevel[2], fast_mhd_wavespeeds[2], consVar_fc, flatCoefs[0],
							      flatCoefs[1], flatCoefs[2], reconstructGhost, nvars, nghost_Riemann);)

		amrex::Gpu::streamSynchronizeAll();

		if (sim.lowLevelDebuggingOutput_ == 1) {
			std::string plotfile_name = sim.CustomPlotFileName("debug_reconstruction", sim.istep[lev] + 1);
			sim.WriteSingleLevelPlotfileSimplified("debug_reconstruction", primVar, sim.componentNames_cc_, lev, 1);

			amrex::Vector<std::string> flatCompNames{"chi"};
			sim.WriteSingleLevelPlotfileSimplified("debug_flattening_x", flatCoefs[0], flatCompNames, lev, 1);
			sim.WriteSingleLevelPlotfileSimplified("debug_flattening_y", flatCoefs[1], flatCompNames, lev, 1);
			sim.WriteSingleLevelPlotfileSimplified("debug_flattening_z", flatCoefs[2], flatCompNames, lev, 1);

			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (amrex::ParallelDescriptor::IOProcessor()) {
					std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
				}
				std::string const fullprefix = amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_",
												 std::string("StateL_") + quokka::face_dir_str[idim]);
				amrex::VisMF::Write(leftState[idim], fullprefix);
			}
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (amrex::ParallelDescriptor::IOProcessor()) {
					std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
				}
				std::string const fullprefix = amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_",
												 std::string("StateR_") + quokka::face_dir_str[idim]);
				amrex::VisMF::Write(rightState[idim], fullprefix);
			}
		}

		return std::make_tuple(std::move(flux), std::move(facevel), std::move(fast_mhd_wavespeeds));
	}

	template <FluxDir DIR>
	static void hydroFOFluxFunction(QuokkaSimulation<problem_t> const &sim, amrex::MultiFab &primVar_mf, amrex::MultiFab &cc_bfield_perp_comps_mf,
					amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield,
					amrex::MultiFab &rightState_bfield, amrex::MultiFab &flux, amrex::MultiFab &faceVel, amrex::MultiFab &x1FSpds,
					std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, int ng_reconstruct, int nvars, int nghost_Riemann)
	{
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			computeCCPerpBfieldComps<DIR>(sim, cc_bfield_perp_comps_mf, consVar_fc);
		}

		HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar_mf, leftState, rightState, ng_reconstruct, nvars);
		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(cc_bfield_perp_comps_mf, leftState_bfield, rightState_bfield, ng_reconstruct,
											2);
		}

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::LLF_MHD, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
												    rightState_bfield, primVar_mf, sim.artificialViscosityK_, &x1FSpds,
												    &consVar_fc[static_cast<int>(DIR)], nghost_Riemann);
		} else {
			HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::LLF, DIR>(flux, faceVel, leftState, rightState, leftState_bfield,
												rightState_bfield, primVar_mf, sim.artificialViscosityK_, nullptr,
												nullptr, nghost_Riemann);
		}
	}

	static auto computeFOHydroFluxes(QuokkaSimulation<problem_t> const &sim, amrex::MultiFab const &consVar_cc,
					 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, int nvars, int nghost_Riemann, int lev)
	    -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>,
			  std::array<amrex::MultiFab, AMREX_SPACEDIM>>
	{
		const BL_PROFILE("HydroRKPolicy::computeFOHydroFluxes()");

		const auto ba = sim.grids[lev];
		const auto dm = sim.dmap[lev];
		const int reconstructRange = nghost_Riemann + 1;

		amrex::MultiFab primVar(ba, dm, nvars, sim.nghost_cc_);
		amrex::MultiFab cc_bfield_perp_comps(ba, dm, 2, sim.nghost_cc_);
		std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState_bfield;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState_bfield;
		std::array<amrex::MultiFab, AMREX_SPACEDIM> fast_mhd_wavespeeds;

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
			leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
			rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
			leftState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructRange);
			rightState_bfield[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructRange);
			flux[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange - 1);
			facevel[idim] = amrex::MultiFab(ba_face, dm, 1, reconstructRange - 1);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				fast_mhd_wavespeeds[idim] = amrex::MultiFab(ba_face, dm, 2, reconstructRange - 1);
			}
		}

		HydroSystem<problem_t>::ConservedToPrimitive(consVar_cc, consVar_fc, primVar, sim.nghost_cc_);

		AMREX_D_TERM(hydroFOFluxFunction<FluxDir::X1>(sim, primVar, cc_bfield_perp_comps, leftState[0], rightState[0], leftState_bfield[0],
							      rightState_bfield[0], flux[0], facevel[0], fast_mhd_wavespeeds[0], consVar_fc,
							      reconstructRange, nvars, nghost_Riemann);
			     , hydroFOFluxFunction<FluxDir::X2>(sim, primVar, cc_bfield_perp_comps, leftState[1], rightState[1], leftState_bfield[1],
								rightState_bfield[1], flux[1], facevel[1], fast_mhd_wavespeeds[1], consVar_fc,
								reconstructRange, nvars, nghost_Riemann);
			     , hydroFOFluxFunction<FluxDir::X3>(sim, primVar, cc_bfield_perp_comps, leftState[2], rightState[2], leftState_bfield[2],
								rightState_bfield[2], flux[2], facevel[2], fast_mhd_wavespeeds[2], consVar_fc,
								reconstructRange, nvars, nghost_Riemann);)

		amrex::Gpu::streamSynchronizeAll();

		return std::make_tuple(std::move(flux), std::move(facevel), std::move(fast_mhd_wavespeeds));
	}

	static void replaceFluxes(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes,
				  amrex::iMultiFab &redoFlag)
	{
		const BL_PROFILE("HydroRKPolicy::replaceFluxes()");

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			AMREX_ASSERT(fluxes[idim].nComp() == FOfluxes[idim].nComp());
			int ncomp = fluxes[idim].nComp();

			auto const &FOflux_arrs = FOfluxes[idim].const_arrays();
			auto const &redoFlag_arrs = redoFlag.const_arrays();
			auto flux_arrs = fluxes[idim].arrays();
			amrex::IntVect ng{AMREX_D_DECL(1, 1, 1)};

			amrex::ParallelFor(redoFlag, ng, ncomp, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
				if (redoFlag_arrs[bx](i, j, k) == quokka::redoFlag::redo) {
					if (flux_arrs[bx].contains(i, j, k)) {
						flux_arrs[bx](i, j, k, n) = FOflux_arrs[bx](i, j, k, n);
					}
					if (idim == 0) {
						if (flux_arrs[bx].contains(i + 1, j, k)) {
							flux_arrs[bx](i + 1, j, k, n) = FOflux_arrs[bx](i + 1, j, k, n);
						}
					} else if (idim == 1) {
						if (flux_arrs[bx].contains(i, j + 1, k)) {
							flux_arrs[bx](i, j + 1, k, n) = FOflux_arrs[bx](i, j + 1, k, n);
						}
					} else if (idim == 2) {
						if (flux_arrs[bx].contains(i, j, k + 1)) {
							flux_arrs[bx](i, j, k + 1, n) = FOflux_arrs[bx](i, j, k + 1, n);
						}
					}
				}
			});
		}
	}

	static void replaceEMFs(std::array<amrex::MultiFab, AMREX_SPACEDIM> &emf_components,
				std::array<amrex::MultiFab, AMREX_SPACEDIM> &FO_emf_components, amrex::iMultiFab &redoFlag)
	{
		const BL_PROFILE("HydroRKPolicy::replaceEMFs()");

		for (int iedge = 0; iedge < 3; ++iedge) {
			AMREX_ASSERT(emf_components[iedge].nComp() == FO_emf_components[iedge].nComp());
			int ncomp = emf_components[iedge].nComp();

			auto const &FO_emf_components_arrs = FO_emf_components[iedge].const_arrays();
			auto const &redoFlag_arrs = redoFlag.const_arrays();
			auto emf_components_arrs = emf_components[iedge].arrays();
			amrex::IntVect ng{AMREX_D_DECL(1, 1, 1)};

			amrex::ParallelFor(redoFlag, ng, ncomp, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
				if (redoFlag_arrs[bx](i, j, k) == quokka::redoFlag::redo) {
					if (emf_components_arrs[bx].contains(i, j, k)) {
						emf_components_arrs[bx](i, j, k, n) = FO_emf_components_arrs[bx](i, j, k, n);
					}
					if (iedge == 0) {
						if (emf_components_arrs[bx].contains(i, j + 1, k)) {
							emf_components_arrs[bx](i, j + 1, k, n) = FO_emf_components_arrs[bx](i, j + 1, k, n);
						}
						if (emf_components_arrs[bx].contains(i, j, k + 1)) {
							emf_components_arrs[bx](i, j, k + 1, n) = FO_emf_components_arrs[bx](i, j, k + 1, n);
						}
						if (emf_components_arrs[bx].contains(i, j + 1, k + 1)) {
							emf_components_arrs[bx](i, j + 1, k + 1, n) = FO_emf_components_arrs[bx](i, j + 1, k + 1, n);
						}
					} else if (iedge == 1) {
						if (emf_components_arrs[bx].contains(i + 1, j, k)) {
							emf_components_arrs[bx](i + 1, j, k, n) = FO_emf_components_arrs[bx](i + 1, j, k, n);
						}
						if (emf_components_arrs[bx].contains(i, j, k + 1)) {
							emf_components_arrs[bx](i, j, k + 1, n) = FO_emf_components_arrs[bx](i, j, k + 1, n);
						}
						if (emf_components_arrs[bx].contains(i + 1, j, k + 1)) {
							emf_components_arrs[bx](i + 1, j, k + 1, n) = FO_emf_components_arrs[bx](i + 1, j, k + 1, n);
						}
					} else if (iedge == 2) {
						if (emf_components_arrs[bx].contains(i + 1, j, k)) {
							emf_components_arrs[bx](i + 1, j, k, n) = FO_emf_components_arrs[bx](i + 1, j, k, n);
						}
						if (emf_components_arrs[bx].contains(i, j + 1, k)) {
							emf_components_arrs[bx](i, j + 1, k, n) = FO_emf_components_arrs[bx](i, j + 1, k, n);
						}
						if (emf_components_arrs[bx].contains(i + 1, j + 1, k)) {
							emf_components_arrs[bx](i + 1, j + 1, k, n) = FO_emf_components_arrs[bx](i + 1, j + 1, k, n);
						}
					}
				}
			});
		}
	}

	void define(int /*lev*/, quokka::CompositeStateView const & /*reference*/) {}

	void define_stage_scratch(quokka::StageScratch &scratch, quokka::CompositeStateView const &reference, int /*lev*/) const
	{
		scratch.rhs_cc.define(reference.cc->boxArray(), reference.cc->DistributionMap(), reference.cc->nComp(), 0);
		scratch.redo_flag.define(reference.cc->boxArray(), reference.cc->DistributionMap(), 1, 1);
		scratch.has_redo_flag = true;

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				auto ba_ec = amrex::convert(reference.cc->boxArray(),
							    amrex::IntVect(AMREX_D_DECL(1, 1, 1)) - amrex::IntVect::TheDimensionVector(idim));
				scratch.emf[idim].define(ba_ec, reference.cc->DistributionMap(), 1, 0);
			}
		}
	}

	void define_step_accumulators(quokka::StepAccumulators &accum, quokka::CompositeStateView const &reference, int /*lev*/) const
	{
		if (sim_.do_tracers != 0) {
			const int nghost_vel = 2;
			accum.defineFaceVelocityLike(reference, nghost_vel);
		}
	}

	void reset_accumulators(quokka::StepAccumulators &accum) const { accum.reset(); }

	void fill_boundary(int stage, quokka::CompositeStateView state, amrex::Real stage_time) const
	{
		sim_.fillBoundaryConditions(*state.cc, *state.cc, lev_, stage_time, quokka::centering::cc, quokka::direction::na,
					    QuokkaSimulation<problem_t>::PreInterpState, QuokkaSimulation<problem_t>::PostInterpState);

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				sim_.fillBoundaryConditions((*state.fc_data)[idim], (*state.fc_data)[idim], lev_, stage_time, quokka::centering::fc,
							    quokka::direction{idim}, AMRSimulation<problem_t>::InterpHookNone,
							    AMRSimulation<problem_t>::InterpHookNone, FillPatchType::fillpatch_function);
			}
		} else {
			amrex::ignore_unused(stage);
		}
	}

	void compute_stage(int /*stage*/, quokka::StageScratch &scratch, quokka::CompositeStateView const &input, amrex::Real /*stage_time*/,
			   amrex::Real /*dt_stage*/) const
	{
		scratch.clearFlags();
		scratch.redo_flag.setVal(quokka::redoFlag::none);
		scratch.has_redo_flag = true;

		auto [fluxes, face_vel, fast_mhd_wavespeeds] =
		    computeHydroFluxes(sim_, *input.cc, *input.fc_data, QuokkaSimulation<problem_t>::nvars_, nghost_Riemann_, lev_);
		scratch.fluxes_hi = std::move(fluxes);
		scratch.face_vel = std::move(face_vel);
		scratch.has_fluxes_hi = true;
		scratch.has_face_vel = true;

		auto const dx = sim_.geom[lev_].CellSizeArray();
		HydroSystem<problem_t>::ComputeRhsFromFluxes(scratch.rhs_cc, scratch.fluxes_hi, dx, QuokkaSimulation<problem_t>::nvars_);
		HydroSystem<problem_t>::AddInternalEnergyPdV(scratch.rhs_cc, *input.cc, *input.fc_data, dx, scratch.face_vel, scratch.redo_flag);

		if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
			for (auto &mf : scratch.emf) {
				mf.setVal(0.0);
			}
			MHDSystem<problem_t>::ComputeEMF(scratch.emf, *input.cc, scratch.face_vel, *input.fc_data, fast_mhd_wavespeeds,
							 sim_.emfReconstructionOrder_, sim_.emfAveragingScheme_, sim_.mhdPlmLimiter_,
							 sim_.emfComputingScheme_);
			scratch.has_emf = true;
		}
	}

	void update_stage(int stage, quokka::CompositeStateView output, quokka::CompositeStateView const &old_state,
			  quokka::CompositeStateView const &stage_input, quokka::StageScratch const &scratch, amrex::Real dt) const
	{
		if (stage == 1) {
			HydroSystem<problem_t>::PredictStep(*old_state.cc, *output.cc, scratch.rhs_cc, dt, QuokkaSimulation<problem_t>::nvars_,
							    const_cast<amrex::iMultiFab &>(scratch.redo_flag));

			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				MHDSystem<problem_t>::SolveInductionEqn(*old_state.fc_data, *output.fc_data, scratch.emf, dt, sim_.geom[lev_].CellSizeArray());
			}
		} else {
			HydroSystem<problem_t>::AddFluxesRK2(*output.cc, *old_state.cc, *stage_input.cc, scratch.rhs_cc, dt, QuokkaSimulation<problem_t>::nvars_,
							     const_cast<amrex::iMultiFab &>(scratch.redo_flag));

			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				MHDSystem<problem_t>::SolveInductionEqn(*old_state.fc_data, *output.fc_data, scratch.emf, 0.5 * dt,
									sim_.geom[lev_].CellSizeArray());
				for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
					amrex::MultiFab::Saxpy((*output.fc_data)[idim], 0.5, (*stage_input.fc_data)[idim], 0, 0,
							       (*output.fc_data)[idim].nComp(), 0);
					amrex::MultiFab::Saxpy((*output.fc_data)[idim], -0.5, (*old_state.fc_data)[idim], 0, 0,
							       (*output.fc_data)[idim].nComp(), 0);
				}
			}
		}
	}

	auto validate_stage(int stage, quokka::CompositeStateView output, quokka::CompositeStateView const & /*old_state*/,
			    quokka::CompositeStateView const & /*stage_input*/, quokka::StageScratch &scratch, amrex::Real /*dt*/) const -> bool
	{
		amrex::Gpu::streamSynchronizeAll();
		amrex::Long const ncells_bad = scratch.redo_flag.sum(0);
		if (ncells_bad > 0) {
			if (sim_.Verbose()) {
				auto const cell_idx = scratch.redo_flag.maxIndex(0);
				amrex::Print() << "[RK-" << stage << "] invalid hydro state in " << ncells_bad << " cells on level " << lev_ << "\n";
				sim_.printCoordinates(lev_, cell_idx);
				amrex::print_state(*output.cc, cell_idx);
			}
			return false;
		}
		return true;
	}

	void post_stage(int /*stage*/, quokka::CompositeStateView output, quokka::StageScratch const & /*scratch*/, amrex::Real /*stage_time*/,
			amrex::Real /*dt*/) const
	{
		if (sim_.useDensityFloorParser_) {
			auto const density_floor_parser = sim_.densityFloorParserExe_.value();
			auto const density_floor_func = [=] AMREX_GPU_HOST_DEVICE(amrex::Real x, amrex::Real y, amrex::Real z,
										  amrex::Real base_density_floor) -> amrex::Real {
				return density_floor_parser(x, y, z, base_density_floor);
			};
			HydroSystem<problem_t>::EnforceLimits(sim_.densityFloor_, sim_.dustDensityFloor_, sim_.tempFloor_, *output.cc, sim_.geom[lev_],
							      density_floor_func);
		} else {
			auto const density_floor_func = [this] AMREX_GPU_HOST_DEVICE(amrex::Real x, amrex::Real y, amrex::Real z,
										     amrex::Real base_density_floor) -> amrex::Real {
				return sim_.densityFloor(x, y, z, base_density_floor);
			};
			HydroSystem<problem_t>::EnforceLimits(sim_.densityFloor_, sim_.dustDensityFloor_, sim_.tempFloor_, *output.cc, sim_.geom[lev_],
							      density_floor_func);
		}

		if (sim_.useDualEnergy_ == 1) {
			HydroSystem<problem_t>::SyncDualEnergy(*output.cc, *output.fc_data);
		}
	}

	void accumulate_stage(int stage, quokka::StageScratch const &scratch, amrex::Real dt, quokka::StepAccumulators &accum) const
	{
		const amrex::Real weight = quokka::SSPRK2Scheme::stage_integral_weights[stage - 1];

		if ((step_fr_as_crse_ != nullptr) || (step_fr_as_fine_ != nullptr)) {
			sim_.incrementFluxRegisters(step_fr_as_crse_, step_fr_as_fine_, const_cast<std::array<amrex::MultiFab, AMREX_SPACEDIM> &>(scratch.fluxes_hi),
						    lev_, weight * dt);
			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
				if ((step_emf_as_crse_ != nullptr) || (step_emf_as_fine_ != nullptr)) {
					sim_.incrementEMFRegisters(step_emf_as_crse_, step_emf_as_fine_,
								   const_cast<std::array<amrex::MultiFab, AMREX_SPACEDIM> &>(scratch.emf), lev_,
								   -weight * dt);
				}
			}
		}

		if (accum.has_avg_face_vel) {
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				amrex::MultiFab::Saxpy(accum.avg_face_vel[idim], weight, scratch.face_vel[idim], 0, 0, 1, 0);
			}
		}
	}

	void finalize_step(quokka::CompositeStateView /*new_state*/, amrex::Real /*time*/, amrex::Real dt,
			   quokka::StepAccumulators const & /*accum*/) const
	{
		amrex::ignore_unused(dt);
	}
};

#endif
