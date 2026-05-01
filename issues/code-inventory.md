# Code Inventory

Generated from declarations and definitions in `src/**/*.hpp`, `src/**/*.cpp`, and `src/**/*.H`; vendored and submodule trees are excluded. This regex-based inventory is intended for audit coverage, not ABI documentation.

## `src/Factory.H`

- L16: struct `Factory`: `template <class Base, class... Args> struct Factory {`
- L18: function `create`: `static std::unique_ptr<Base> create(const std::string &key, Args... args) {`
- L31: function `print`: `static void print(std::ostream &os) {`
- L41: struct `Register`: `template <class T> struct Register : public Base {`
- L45: function `add_sub_type`: `static bool add_sub_type() {`
- L54: function `~Register`: `~Register() override {`
- L64: function `Register`: `Register() {`
- L67: function `~Factory`: `virtual ~Factory() = default;`
- L76: function `key_exists_or_error`: `static void key_exists_or_error(const std::string &key) {`
- L90: function `table`: `static LookupTable &table() {`
- L96: function `Factory`: `Factory() = default;`

## `src/QuokkaSimulation.hpp`

- L85: class `QuokkaSimulation`: `template <typename problem_t> class QuokkaSimulation : public AMRSimulation<problem_t>`
- L230: enum class `SourceOrder`: `enum class SourceOrder { forward, reverse };`
- L233: function `QuokkaSimulation`: `explicit QuokkaSimulation(amrex::Vector<amrex::BCRec> &BCs_cc, amrex::Vector<amrex::BCRec> &BCs_fc) : AMRSimulation<problem_t>(BCs_cc, BCs_fc) {`
- L238: function `QuokkaSimulation`: `explicit QuokkaSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : AMRSimulation<problem_t>(BCs_cc) {`
- L240: function `QuokkaSimulation`: `explicit QuokkaSimulation() : AMRSimulation<problem_t>() {`
- L242: function `initialize`: `void initialize() {`
- L270: function `defineComponentNames`: `void defineComponentNames();`
- L271: function `defineDefaultPlotfileVariables`: `void defineDefaultPlotfileVariables();`
- L272: function `readParmParse`: `void readParmParse();`
- L273: function `rereadRuntimeParameters`: `void rereadRuntimeParameters();`
- L275: function `checkHydroStates`: `void checkHydroStates(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &mf_fc, char const *file, int line);`
- L276: function `CheckHydroStates`: `void CheckHydroStates(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &mf_fc, std::source_location const &location = std::source_location::current());`
- L278: function `computeMaxSignalLocal`: `void computeMaxSignalLocal(int level) override;`
- L279: function `printCellProperties`: `void printCellProperties(int lev, amrex::IntVect const &index) override;`
- L280: function `preCalculateInitialConditions`: `void preCalculateInitialConditions() override;`
- L281: function `setInitialConditionsOnGrid`: `void setInitialConditionsOnGrid(quokka::grid const &grid_elem) override;`
- L282: function `setInitialConditionsOnGridFaceVars`: `void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) override;`
- L283: function `postInitialization`: `void postInitialization() override;`
- L286: function `projectFaceCenteredMagneticField`: `void projectFaceCenteredMagneticField();`
- L287: function `updateInitialMagneticEnergyFromFaceField`: `void updateInitialMagneticEnergyFromFaceField();`
- L288: function `refineGrid`: `void refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;`
- L290: function `createInitialRadParticles`: `void createInitialRadParticles() override;`
- L291: function `createInitialCICParticles`: `void createInitialCICParticles() override;`
- L292: function `createInitialCICRadParticles`: `void createInitialCICRadParticles() override;`
- L293: function `createInitialStochasticStellarPopParticles`: `void createInitialStochasticStellarPopParticles() override;`
- L294: function `createInitialSinkParticles`: `void createInitialSinkParticles() override;`
- L295: function `createInitialTestParticles`: `void createInitialTestParticles() override;`
- L297: function `advanceSingleTimestepAtLevel`: `void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle) override;`
- L298: function `computeBeforeTimestep`: `void computeBeforeTimestep() override;`
- L299: function `computeAfterTimestep`: `void computeAfterTimestep() override;`
- L300: function `computeAfterLevelAdvance`: `void computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int );`
- L301: function `computeAfterEvolve`: `void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;`
- L302: function `computeReferenceSolution`: `void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo);`
- L304: function `computeReferenceSolution_fc`: `void computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction dir);`
- L306: function `computeErrorNorm`: `auto computeErrorNorm(bool use_rel_err = true) -> amrex::Real;`
- L307: function `computeComponentErrors`: `auto computeComponentErrors() -> std::vector<std::tuple<std::string, amrex::Real, amrex::Real>>;`
- L308: function `WriteSingleLevelPlotfileSimplified`: `void WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf, const amrex::Vector<std::string> &compNames, int lev, int interval) override;`
- L312: function `ComputeDerivedVar`: `void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;`
- L313: function `ComputeDensityFloorDebug`: `void ComputeDensityFloorDebug(int lev, amrex::MultiFab &mf, int ncomp) const override;`
- L318: function `ComputeStatistics`: `auto ComputeStatistics() -> std::map<std::string, amrex::Real> override;`
- L321: function `FixupState`: `void FixupState(int level) override;`
- L322: function `ApplyHydroStateFixup`: `void ApplyHydroStateFixup(amrex::MultiFab &state_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> &state_fc, int lev);`
- L325: function `FillPatch`: `void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir, FillPatchType fptype) override;`
- L329: function `PreInterpState`: `static void PreInterpState(amrex::MultiFab &mf, int scomp, int ncomp);`
- L330: function `PostInterpState`: `static void PostInterpState(amrex::MultiFab &mf, int scomp, int ncomp);`
- L333: function `computeAxisAlignedProfile`: `template <typename F> auto computeAxisAlignedProfile(int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>;`
- L336: function `ErrorEst`: `void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;`
- L339: function `fillPoissonRhsAtLevel`: `void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) override;`
- L341: function `print_multifab_fc`: `void print_multifab_fc(amrex::MultiFab &mf, std::string const &name, int lev, int idim);`
- L344: function `applyPoissonGravityAtLevel`: `void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) override;`
- L346: function `addFluxArrays`: `void addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes, int srccomp, int dstcomp);`
- L349: function `expandFluxArrays`: `auto expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, int nstartNew, int ncompNew) -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;`
- L352: function `printCoordinates`: `void printCoordinates(int lev, const amrex::IntVect &cell_idx);`
- L354: function `advanceHydroAtLevelWithRetries`: `void advanceHydroAtLevelWithRetries(int lev, amrex::Real time, amrex::Real dt_lev, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::EdgeFluxRegister *emf_as_crse = nullptr, amrex::EdgeFluxRegister *emf_as_fine = nullptr);`
- L357: function `advanceHydroAtLevel`: `auto advanceHydroAtLevel(amrex::MultiFab &state_old_cc_tmp, std::array<amrex::MultiFab, AMREX_SPACEDIM> &state_old_fc_tmp, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::EdgeFluxRegister *emf_as_crse, amrex::EdgeFluxRegister *emf_as_fine, int lev, amrex::Real time, amrex::Real dt_lev) -> bool;`
- L361: function `addStrangSplitSources`: `void addStrangSplitSources(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt_lev);`
- L363: function `addStrangSplitSourcesWithBuiltin`: `auto addStrangSplitSourcesWithBuiltin(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int lev, amrex::Real time, amrex::Real dt_lev) -> bool;`
- L365: function `callInOrder`: `template <SourceOrder Order, typename... Fs> static auto callInOrder(Fs &&...fs) -> void {`
- L377: function `computePhotoelectricHeatingRate`: `auto computePhotoelectricHeatingRate(Real current_time) -> amrex::Real;`
- L378: function `computeExternalHeatingRate`: `auto computeExternalHeatingRate(Real current_time, Real dt) -> amrex::Real;`
- L380: function `isCflViolated`: `auto isCflViolated(int lev, amrex::Real time, amrex::Real dt_actual) -> bool;`
- L383: function `swapRadiationState`: `void swapRadiationState(amrex::MultiFab &stateOld_cc, amrex::MultiFab const &stateNew_cc);`
- L384: function `computeNumberOfRadiationSubsteps`: `auto computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int;`
- L385: function `advanceRadiationForwardEuler`: `void advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::MultiFab &state_out);`
- L387: function `advanceRadiationMidpointRK2`: `void advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::MultiFab &state_inter);`
- L390: function `subcycleRadiationAtLevel`: `void subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine);`
- L392: function `computeRadiationFluxes`: `auto computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>, std::array<amrex::FArrayBox, AMREX_SPACEDIM>>;`
- L396: function `computeHydroFluxes`: `auto computeHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, int nvars, int nghost_Riemann, int lev) -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>;`
- L400: function `computeFOHydroFluxes`: `auto computeFOHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, int nvars, int nghost_Riemann, int lev) -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>;`
- L406: function `computeCCPerpBfieldComps`: `void computeCCPerpBfieldComps(amrex::MultiFab &cc_bfield_perp_comps_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc) const;`
- L409: function `fluxFunction`: `void fluxFunction(amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive, const amrex::Box &indexRange, int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);`
- L413: function `hydroFluxFunction`: `void hydroFluxFunction(amrex::MultiFab &primVar, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield, amrex::MultiFab &x1Flux, amrex::MultiFab &x1FaceVel, amrex::MultiFab &x1FSpds, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, amrex::MultiFab const &x1Flat, amrex::MultiFab const &x2Flat, amrex::MultiFab const &x3Flat, int ng_reconstruct_total, int nvars, int nghost_Riemann);`
- L419: function `hydroFOFluxFunction`: `void hydroFOFluxFunction(amrex::MultiFab &primVar, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield, amrex::MultiFab &x1Flux, amrex::MultiFab &x1FaceVel, amrex::MultiFab &x1FSpds, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &x1ConsVar_fc_mf, int ng_reconstruct_total, int nvars, int nghost_Riemann);`
- L424: function `replaceFluxes`: `void replaceFluxes(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes, amrex::iMultiFab &redoFlag);`
- L427: function `replaceEMFs`: `void replaceEMFs(std::array<amrex::MultiFab, AMREX_SPACEDIM> &emf_components, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FO_emf_components, amrex::iMultiFab &redoFlag);`
- L433: function `defineComponentNames`: `template <typename problem_t> void QuokkaSimulation<problem_t>::defineComponentNames() {`
- L481: function `defineDefaultPlotfileVariables`: `template <typename problem_t> void QuokkaSimulation<problem_t>::defineDefaultPlotfileVariables() {`
- L509: function `initializeSimulationMetadata`: `template <typename problem_t> void AMRSimulation<problem_t>::initializeSimulationMetadata() {`
- L555: function `getScalarVariableNames`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::getScalarVariableNames() -> std::vector<std::string> {`
- L571: function `readParmParse`: `template <typename problem_t> void QuokkaSimulation<problem_t>::readParmParse() {`
- L683: function `amrex::Print`: `amrex::Print() << std::format("\tTable dimension: {`
- L684: function `amrex::Print`: `amrex::Print() << std::format("\tNumber of outputs: {`
- L731: function `rereadRuntimeParameters`: `template <typename problem_t> void QuokkaSimulation<problem_t>::rereadRuntimeParameters() {`
- L746: function `computeNumberOfRadiationSubsteps`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int {`
- L757: function `computeMaxSignalLocal`: `template <typename problem_t> void QuokkaSimulation<problem_t>::computeMaxSignalLocal(int const level) {`
- L798: function `printCellProperties`: `template <typename problem_t> void QuokkaSimulation<problem_t>::printCellProperties(int lev, amrex::IntVect const &index) {`
- L821: function `amrex::AllPrint`: `amrex::AllPrint() << std::format("...[level {`
- L826: function `CheckHydroStates`: `void QuokkaSimulation<problem_t>::CheckHydroStates(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &mf_fc, std::source_location const &location) {`
- L839: function `checkHydroStates`: `void QuokkaSimulation<problem_t>::checkHydroStates(amrex::MultiFab &mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &mf_fc, char const *file, int line) {`
- L855: function `preCalculateInitialConditions`: `template <typename problem_t> void QuokkaSimulation<problem_t>::preCalculateInitialConditions() {`
- L861: function `setInitialConditionsOnGrid`: `template <typename problem_t> void QuokkaSimulation<problem_t>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L867: function `setInitialConditionsOnGridFaceVars`: `template <typename problem_t> void QuokkaSimulation<problem_t>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L876: function `createInitialRadParticles`: `template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialRadParticles() {`
- L884: function `createInitialCICParticles`: `template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialCICParticles() {`
- L892: function `createInitialCICRadParticles`: `template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialCICRadParticles() {`
- L900: function `createInitialStochasticStellarPopParticles`: `template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialStochasticStellarPopParticles() {`
- L909: function `createInitialSinkParticles`: `template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialSinkParticles() {`
- L918: function `createInitialTestParticles`: `template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialTestParticles() {`
- L928: function `computeBeforeTimestep`: `template <typename problem_t> void QuokkaSimulation<problem_t>::computeBeforeTimestep() {`
- L933: function `computeAfterTimestep`: `template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterTimestep() {`
- L938: function `computeAfterLevelAdvance`: `template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle) {`
- L943: function `addStrangSplitSources`: `template <typename problem_t> void QuokkaSimulation<problem_t>::addStrangSplitSources(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt) {`
- L949: function `computePhotoelectricHeatingRate`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::computePhotoelectricHeatingRate(amrex::Real current_time) -> amrex::Real {`
- L977: function `computeExternalHeatingRate`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::computeExternalHeatingRate(amrex::Real current_time, amrex::Real dt) -> amrex::Real {`
- L989: function `addStrangSplitSourcesWithBuiltin`: `auto QuokkaSimulation<problem_t>::addStrangSplitSourcesWithBuiltin(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &state_fc, int lev, amrex::Real time, amrex::Real dt) -> bool {`
- L1045: function `ComputeDerivedVar`: `template <typename problem_t> void QuokkaSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp) const {`
- L1054: function `ComputeDensityFloorDebug`: `template <typename problem_t> void QuokkaSimulation<problem_t>::ComputeDensityFloorDebug(int lev, amrex::MultiFab &mf, int ncomp) const {`
- L1109: function `ComputeStatistics`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::ComputeStatistics() -> std::map<std::string, amrex::Real> {`
- L1116: function `refineGrid`: `template <typename problem_t> void QuokkaSimulation<problem_t>::refineGrid(int , amrex::TagBoxArray & , amrex::Real , int ) {`
- L1122: function `ErrorEst`: `template <typename problem_t> void QuokkaSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) {`
- L1136: function `computeReferenceSolution`: `void QuokkaSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L1143: function `computeReferenceSolution_fc`: `void QuokkaSimulation<problem_t>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L1149: function `print_multifab_fc`: `template <typename problem_t> void QuokkaSimulation<problem_t>::print_multifab_fc(amrex::MultiFab &mf, std::string const & , int , int idim) {`
- L1158: function `densityFloor`: `AMREX_GPU_HOST_DEVICE auto QuokkaSimulation<problem_t>::densityFloor(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real base_density_floor) const -> amrex::Real {`
- L1165: function `computeComponentErrors`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::computeComponentErrors() -> std::vector<std::tuple<std::string, amrex::Real, amrex::Real>> {`
- L1280: function `computeErrorNorm`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::computeErrorNorm(bool use_rel_err) -> amrex::Real {`
- L1336: function `computeAfterEvolve`: `template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) {`
- L1392: function `advanceSingleTimestepAtLevel`: `template <typename problem_t> void QuokkaSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle) {`
- L1461: function `fillPoissonRhsAtLevel`: `template <typename problem_t> void QuokkaSimulation<problem_t>::fillPoissonRhsAtLevel(amrex::MultiFab &rhs_mf, const int lev) {`
- L1476: function `applyPoissonGravityAtLevel`: `template <typename problem_t> void QuokkaSimulation<problem_t>::applyPoissonGravityAtLevel(amrex::MultiFab const &phi_mf, const int lev, const amrex::Real dt) {`
- L1513: function `projectFaceCenteredMagneticField`: `template <typename problem_t> void QuokkaSimulation<problem_t>::projectFaceCenteredMagneticField() {`
- L1738: function `updateInitialMagneticEnergyFromFaceField`: `template <typename problem_t> void QuokkaSimulation<problem_t>::updateInitialMagneticEnergyFromFaceField() {`
- L1800: function `postInitialization`: `template <typename problem_t> void QuokkaSimulation<problem_t>::postInitialization() {`
- L1820: function `ApplyHydroStateFixup`: `void QuokkaSimulation<problem_t>::ApplyHydroStateFixup(amrex::MultiFab &state_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> &state_fc, int lev) {`
- L1846: function `FixupState`: `template <typename problem_t> void QuokkaSimulation<problem_t>::FixupState(int lev) {`
- L1858: function `FillPatch`: `void QuokkaSimulation<problem_t>::FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir, FillPatchType fptype) {`
- L1885: function `PreInterpState`: `template <typename problem_t> void QuokkaSimulation<problem_t>::PreInterpState(amrex::MultiFab &mf, int , int ) {`
- L1904: function `PostInterpState`: `template <typename problem_t> void QuokkaSimulation<problem_t>::PostInterpState(amrex::MultiFab &mf, int , int ) {`
- L1926: function `computeAxisAlignedProfile`: `auto QuokkaSimulation<problem_t>::computeAxisAlignedProfile(const int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real> {`
- L1967: function `advanceHydroAtLevelWithRetries`: `void QuokkaSimulation<problem_t>::advanceHydroAtLevelWithRetries(int lev, amrex::Real time, amrex::Real dt_lev, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::EdgeFluxRegister *emf_as_crse, amrex::EdgeFluxRegister *emf_as_fine) {`
- L2089: function `isCflViolated`: `template <typename problem_t> auto QuokkaSimulation<problem_t>::isCflViolated(int lev, amrex::Real , amrex::Real dt_actual) -> bool {`
- L2112: function `printCoordinates`: `template <typename problem_t> void QuokkaSimulation<problem_t>::printCoordinates(int lev, const amrex::IntVect &cell_idx) {`
- L2130: function `advanceHydroAtLevel`: `auto QuokkaSimulation<problem_t>::advanceHydroAtLevel(amrex::MultiFab &state_old_cc_tmp, std::array<amrex::MultiFab, AMREX_SPACEDIM> &state_old_fc_tmp, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::EdgeFluxRegister *emf_as_crse, amrex::EdgeFluxRegister *emf_as_fine, int lev, amrex::Real time, amrex::Real dt_lev) -> bool {`
- L2483: function `replaceFluxes`: `void QuokkaSimulation<problem_t>::replaceFluxes(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes, amrex::iMultiFab &redoFlag) {`
- L2529: function `replaceEMFs`: `void QuokkaSimulation<problem_t>::replaceEMFs(std::array<amrex::MultiFab, AMREX_SPACEDIM> &emf_components, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FO_emf_components, amrex::iMultiFab &redoFlag) {`
- L2588: function `addFluxArrays`: `void QuokkaSimulation<problem_t>::addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes, const int srccomp, const int dstcomp) {`
- L2601: function `expandFluxArrays`: `auto QuokkaSimulation<problem_t>::expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, const int nstartNew, const int ncompNew) -> std::array<amrex::FArrayBox, AMREX_SPACEDIM> {`
- L2621: function `computeHydroFluxes`: `auto QuokkaSimulation<problem_t>::computeHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, const int nvars, const int nghost_Riemann, const int lev) -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>> {`
- L2733: function `computeCCPerpBfieldComps`: `AMREX_FORCE_INLINE void QuokkaSimulation<problem_t>::computeCCPerpBfieldComps(amrex::MultiFab &cc_bfield_perp_comps_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc) const {`
- L2776: function `hydroFluxFunction`: `void QuokkaSimulation<problem_t>::hydroFluxFunction(amrex::MultiFab &primVar_mf, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield, amrex::MultiFab &flux, amrex::MultiFab &faceVel, amrex::MultiFab &x1FSpds, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, amrex::MultiFab const &x1Flat, amrex::MultiFab const &x2Flat, amrex::MultiFab const &x3Flat, const int ng_reconstruct, const int nvars, const int nghost_Riemann) {`
- L2831: function `computeFOHydroFluxes`: `auto QuokkaSimulation<problem_t>::computeFOHydroFluxes(amrex::MultiFab const &consVar_cc, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc, const int nvars, const int nghost_Riemann, const int lev) -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>> {`
- L2888: function `hydroFOFluxFunction`: `void QuokkaSimulation<problem_t>::hydroFOFluxFunction(amrex::MultiFab &primVar_mf, amrex::MultiFab &cc_bfield_perp_comps_mf, amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &leftState_bfield, amrex::MultiFab &rightState_bfield, amrex::MultiFab &flux, amrex::MultiFab &faceVel, amrex::MultiFab &x1FSpds, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &x1ConsVar_fc_mf, const int ng_reconstruct, const int nvars, const int nghost_Riemann) {`
- L2917: function `swapRadiationState`: `template <typename problem_t> void QuokkaSimulation<problem_t>::swapRadiationState(amrex::MultiFab &stateOld, amrex::MultiFab const &stateNew) {`
- L2924: function `subcycleRadiationAtLevel`: `void QuokkaSimulation<problem_t>::subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro, amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine) {`
- L3164: function `advanceRadiationForwardEuler`: `void QuokkaSimulation<problem_t>::advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation, int const , int const , amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::MultiFab &state_out) {`
- L3216: function `advanceRadiationMidpointRK2`: `void QuokkaSimulation<problem_t>::advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation, int const , int const , amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, amrex::MultiFab &state_inter) {`
- L3271: function `computeRadiationFluxes`: `auto QuokkaSimulation<problem_t>::computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, const int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>, std::array<amrex::FArrayBox, AMREX_SPACEDIM>> {`
- L3302: function `fluxFunction`: `void QuokkaSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive, const amrex::Box &indexRange, const int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) {`
- L3361: function `WriteSingleLevelPlotfileSimplified`: `void QuokkaSimulation<problem_t>::WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf, const amrex::Vector<std::string> &compNames, int lev, int interval) {`

## `src/SimulationData.hpp`

- L11: struct `SimulationData`: `template <typename problem_t> struct SimulationData {`

## `src/chemistry/Chemistry.cpp`

- L17: function `chemburner`: `AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, const Real dt) {`

## `src/chemistry/Chemistry.hpp`

- L29: function `chemburner`: `AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, Real dt);`
- L31: function `computeChemistry`: `template <typename problem_t> auto computeChemistry(amrex::MultiFab &mf, const Real dt, const Real max_density_allowed, const Real min_density_allowed) -> bool {`

## `src/cooling/PhotoelectricHeating.hpp`

- L22: struct `PeHeatingGpuConstTables`: `template <quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> struct PeHeatingGpuConstTables {`
- L27: class `PeHeatingTables`: `template <quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> class PeHeatingTables`
- L49: function `PeHeatingFromSfh`: `auto PeHeatingFromSfh(const std::vector<std::tuple<int, amrex::Real, amrex::Real>> &sfh_data, amrex::Real current_time, PeHeatingGpuConstTables<oob_policy> const &gpu_tables, amrex::Real sf_area_kpc2) -> amrex::Real {`
- L87: function `PeHeatingFromConstSfr`: `auto PeHeatingFromConstSfr(amrex::Real const_sfr_Msun_per_year_per_kpc2, PeHeatingGpuConstTables<oob_policy> const &gpu_tables) -> amrex::Real {`

## `src/cooling/ResampledCooling.cpp`

- L24: function `readResampledData`: `auto readResampledData(std::string const &hdf5_file, resampled_tables &resampledTables) -> bool {`
- L27: function `amrex::Print`: `amrex::Print() << std::format("resampled_table_file: {`
- L62: function `amrex::Print`: `amrex::Print() << std::format("\tDensity range: {`
- L63: function `amrex::Print`: `amrex::Print() << std::format("\tSpecific energy range: {`
- L64: function `amrex::Print`: `amrex::Print() << std::format("\tPhotoelectric heating: {`
- L69: function `resampled_tables::const_tables`: `auto resampled_tables::const_tables() const -> resampledGpuConstTables {`

## `src/cooling/ResampledCooling.hpp`

- L29: struct `resampledGpuConstTables`: `struct resampledGpuConstTables {`
- L49: class `resampled_tables`: `class resampled_tables`
- L67: function `resampled_cooling_function`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto resampled_cooling_function(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real {`
- L81: function `ComputeTgasFromEgas`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real {`
- L93: function `ComputeEgasFromTgas`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(Real const rho, Real const Tgas, resampledGpuConstTables const &tables) -> Real {`
- L118: function `ComputeCoolingLength`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeCoolingLength(Real const rho, Real const Eint, resampledGpuConstTables const &tables, Real const_heating_rate = 0.0) -> Real {`
- L135: function `ComputePressureFromRhoEint`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputePressureFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real {`
- L147: function `ComputeEntropyFromRhoEint`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEntropyFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real {`
- L159: function `ComputeSoundSpeedFromRhoEint`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeSoundSpeedFromRhoEint(Real const rho, Real const Eint, resampledGpuConstTables const &tables) -> Real {`
- L171: struct `ResampledCoolingFunctor`: `struct ResampledCoolingFunctor {`
- L176: function `ResampledCoolingFunctor`: `AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(Real rho_in, resampledGpuConstTables const &tables_in, Real const_heating_rate_in) : rho(rho_in), tables(tables_in), const_heating_rate(const_heating_rate_in) {`
- L181: function `~ResampledCoolingFunctor`: `AMREX_GPU_HOST_DEVICE ~ResampledCoolingFunctor() = default;`
- L182: function `ResampledCoolingFunctor`: `AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(ResampledCoolingFunctor const &) = default;`
- L183: function `ResampledCoolingFunctor`: `AMREX_GPU_HOST_DEVICE ResampledCoolingFunctor(ResampledCoolingFunctor &&) = default;`
- L184: function `operator=`: `AMREX_GPU_HOST_DEVICE auto operator=(ResampledCoolingFunctor const &) -> ResampledCoolingFunctor & = default;`
- L185: function `operator=`: `AMREX_GPU_HOST_DEVICE auto operator=(ResampledCoolingFunctor &&) -> ResampledCoolingFunctor & = default;`
- L187: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(Real , quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs) const -> int {`
- L198: function `computeCooling`: `auto computeCooling(amrex::MultiFab &mf, const Real dt_in, resampled_tables &resampledTables, const Real temp_floor, const Real const_heating_rate_per_H) -> bool {`
- L262: function `amrex::Print`: `amrex::Print() << std::format("\tcooling substeps (per cell): avg {`
- L272: function `readResampledData`: `auto readResampledData(std::string const &hdf5_file, resampled_tables &resampledTables) -> bool;`

## `src/dust/DustDrag.hpp`

- L16: class `DustDrag`: `template <typename problem_t> class DustDrag`
- L24: enum `consVarIndex`: `enum consVarIndex {`
- L34: enum `primVarIndex`: `enum primVarIndex {`
- L44: enum `dustVarIndex`: `enum dustVarIndex {`
- L52: enum `primDustVarIndex`: `enum primDustVarIndex { primDustDensity_index = primDustFirstIndex, x1DustVelocity_index, x2DustVelocity_index, x3DustVelocity_index };`
- L55: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE static auto ComputeReciprocalStoppingTime(amrex::Real , amrex::GpuArray<amrex::Real, nDustGroups_> , amrex::GpuArray<amrex::Real, nDustGroups_> , double ) -> amrex::GpuArray<amrex::Real, nDustGroups_>;`
- L59: function `ComputeReciprocalStoppingTimeKwok`: `static AMREX_GPU_HOST_DEVICE auto ComputeReciprocalStoppingTimeKwok(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d, amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs, amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_radius, amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_density, bool enable_supersonic_correction) -> amrex::GpuArray<amrex::Real, nDustGroups_>;`
- L65: function `computeDustDrag`: `static void computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt, amrex::Real dust_omega_, int enableIterDustStoptime_, bool print_dust_counter_);`
- L70: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeReciprocalStoppingTime(amrex::Real , amrex::GpuArray<amrex::Real, nDustGroups_> , amrex::GpuArray<amrex::Real, nDustGroups_> , double ) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L81: function `ComputeReciprocalStoppingTimeKwok`: `AMREX_GPU_HOST_DEVICE auto DustDrag<problem_t>::ComputeReciprocalStoppingTimeKwok(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d, amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs, amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_radius, amrex::GpuArray<amrex::Real, nDustGroups_> dust_grain_density, bool enable_supersonic_correction) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L110: function `computeDustDrag`: `void DustDrag<problem_t>::computeDustDrag(amrex::MultiFab &consVar_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &consVar_fc_mf, amrex::Real dt, amrex::Real dust_omega_, int enableIterDustStoptime_, bool print_dust_counter_) {`

## `src/dust/DustState.hpp`

- L9: struct `DustState`: `struct DustState {`

## `src/dust/dustRiemannSolver.hpp`

- L14: function `dustRiemannSolver`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto dustRiemannSolver(quokka::DustState const &sL, quokka::DustState const &sR) -> quokka::valarray<double, fluxdim> {`

## `src/dust/dust_system.hpp`

- L18: class `DustSystem`: `template <typename problem_t> class DustSystem`
- L25: enum `consVarIndex`: `enum consVarIndex {`
- L35: enum `primVarIndex`: `enum primVarIndex {`
- L45: enum `dustVarIndex`: `enum dustVarIndex {`
- L53: enum `primDustVarIndex`: `enum primDustVarIndex { primDustDensity_index = primDustFirstIndex, x1DustVelocity_index, x2DustVelocity_index, x3DustVelocity_index };`
- L57: function `ComputeDustFluxes`: `AMREX_GPU_DEVICE static void ComputeDustFluxes(quokka::Array4View<amrex::Real, DIR> &x1Flux, quokka::Array4View<const amrex::Real, DIR> &x1LeftState, quokka::Array4View<const amrex::Real, DIR> &x1RightState, int i, int j, int k);`
- L63: function `ComputeDustFluxes`: `AMREX_GPU_DEVICE void DustSystem<problem_t>::ComputeDustFluxes(quokka::Array4View<amrex::Real, DIR> &x1Flux, quokka::Array4View<const amrex::Real, DIR> &x1LeftState, quokka::Array4View<const amrex::Real, DIR> &x1RightState, int i, int j, int k) {`

## `src/grid.hpp`

- L12: enum class `centering`: `enum class centering { cc = 0, fc, ec };`
- L13: enum class `direction`: `enum class direction { na = -1, x, y, z };`
- L16: struct `grid`: `struct grid {`
- L22: enum `centering`: `enum centering cen_;`
- L23: enum `direction`: `enum direction dir_;`
- L25: function `grid`: `grid(amrex::Array4<double> const &array, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi, centering cen, direction dir) : array_(array), indexRange_(indexRange), dx_(dx), prob_lo_(prob_lo), prob_hi_(prob_hi), cen_(cen), dir_(dir) {`

## `src/hydro/EOS.hpp`

- L34: struct `EOS_Traits`: `template <typename problem_t> struct EOS_Traits {`
- L41: class `EOS`: `template <typename problem_t> class EOS`
- L93: function `ComputeTgasFromEint`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::Real {`
- L135: function `ComputeEintFromTgas`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::Real {`
- L178: function `ComputeEintFromPres`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::Real {`
- L320: function `ComputePressure`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputePressure(amrex::Real rho, amrex::Real Eint, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::Real {`
- L371: function `ComputeSoundSpeed`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, quokka::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::Real {`
- L420: function `ComputeIsothermalSoundSpeed`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeIsothermalSoundSpeed(amrex::Real rho, amrex::Real Pressure) -> amrex::Real {`

## `src/hydro/HLLC.hpp`

- L22: function `HLLC`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLC(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR, const double gamma, const double du, const double dw) -> quokka::valarray<double, fluxdim> {`

## `src/hydro/HLLD.hpp`

- L21: function `HLLD`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLD(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR, const double gamma, const double bx, const double perp_v_jump) -> std::tuple<quokka::valarray<double, fluxdim>, double, double> {`

## `src/hydro/HydroState.hpp`

- L9: struct `HydroState`: `template <int Nall, int Nmass> struct HydroState {`
- L26: struct `ConsHydro1D`: `template <int N_passiveScalars> struct ConsHydro1D {`
- L38: function `SQUARE`: `template <class T> constexpr auto SQUARE(const T x) -> T {`
- L41: function `FastMagnetoSonicSpeed`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto FastMagnetoSonicSpeed(double gamma, quokka::HydroState<N_scalars, N_mscalars> const state, const double bx) -> double {`

## `src/hydro/LLF.hpp`

- L16: function `LLF`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto LLF(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR) -> quokka::valarray<double, fluxdim> {`

## `src/hydro/LLF_mhd.hpp`

- L16: function `LLF_MHD`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto LLF_MHD(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR, const double gamma, const double bx) -> std::tuple<quokka::valarray<double, fluxdim>, double, double> {`

## `src/hydro/NSCBC_inflow.hpp`

- L25: function `dQ_dx_inflow_x1_lower`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_inflow_x1_lower(quokka::valarray<Real, HydroSystem<problem_t>::nvar_> const &Q, quokka::valarray<Real, HydroSystem<problem_t>::nvar_> const &dQ_dx_data, const Real T_t, const Real u_t, const Real v_t, const Real w_t, amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t, const Real L_x) -> quokka::valarray<Real, HydroSystem<problem_t>::nvar_> {`
- L107: function `setInflowX1Lower`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setInflowX1Lower(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, const amrex::Real T_t, const amrex::Real u_t, const amrex::Real v_t, const amrex::Real w_t, amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t) {`
- L168: function `setInflowX1LowerLowOrder`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setInflowX1LowerLowOrder(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, const amrex::Real T_t, const amrex::Real u_t, const amrex::Real v_t, const amrex::Real w_t, amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t) {`

## `src/hydro/NSCBC_outflow.hpp`

- L21: enum class `BoundarySide`: `enum class BoundarySide { Lower, Upper };`
- L25: function `dQ_dx_outflow`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_outflow(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dx_data, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dy_data, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dz_data, const amrex::Real P_t, const amrex::Real L_x) -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> {`
- L110: function `transverse_xdir_dQ_data`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_xdir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom) -> std::tuple<quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>> {`
- L144: function `transverse_ydir_dQ_data`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_ydir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom) -> std::tuple<quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>> {`
- L177: function `transverse_zdir_dQ_data`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_zdir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom) -> std::tuple<quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>> {`
- L210: function `permute_vel`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto permute_vel(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q) -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> {`
- L236: function `unpermute_vel`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto unpermute_vel(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q) -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> {`
- L263: function `setOutflowBoundary`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setOutflowBoundary(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, const amrex::Real P_outflow) {`
- L368: function `setOutflowBoundaryLowOrder`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setOutflowBoundaryLowOrder(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, const amrex::Real P_outflow) {`

## `src/hydro/hydro_system.hpp`

- L45: struct `HydroSystem_Traits`: `template <typename problem_t> struct HydroSystem_Traits {`
- L52: struct `dependent_false`: `template <typename T> struct dependent_false : std::false_type {`
- L56: enum class `RiemannSolver`: `enum class RiemannSolver { HLLC, LLF, LLF_MHD, HLLD };`
- L60: class `HydroSystem`: `template <typename problem_t> class HydroSystem : public HyperbolicSystem<problem_t>`
- L71: enum `consVarIndex`: `enum consVarIndex {`
- L81: enum `primVarIndex`: `enum primVarIndex {`
- L91: enum `dustVarIndex`: `enum dustVarIndex {`
- L99: enum `primDustVarIndex`: `enum primDustVarIndex { primDustDensity_index = primDustFirstIndex, x1DustVelocity_index, x2DustVelocity_index, x3DustVelocity_index };`
- L101: function `ConservedToPrimitive`: `static void ConservedToPrimitive(amrex::MultiFab const &cons_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf, amrex::MultiFab &primVar_mf, int nghost);`
- L104: function `maxSignalSpeedLocal`: `static auto maxSignalSpeedLocal(amrex::MultiFab const &cons, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> amrex::Real;`
- L106: function `ComputeMaxSignalSpeed`: `static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &cons_fc, array_t &maxSignal, amrex::Box const &indexRange);`
- L110: function `CheckStatesValid`: `static auto CheckStatesValid(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> bool;`
- L112: function `ComputePrimVars`: `AMREX_GPU_DEVICE static auto ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> quokka::valarray<amrex::Real, nHydroScalars_>;`
- L117: function `ComputeConsVars`: `AMREX_GPU_DEVICE static auto ComputeConsVars(quokka::valarray<amrex::Real, nHydroScalars_> const &prim) -> quokka::valarray<amrex::Real, nHydroScalars_>;`
- L120: function `ComputePressure`: `AMREX_GPU_DEVICE static auto ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;`
- L123: function `ComputeInternalEnergy`: `AMREX_GPU_DEVICE static auto ComputeInternalEnergy(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;`
- L127: function `ComputeSoundSpeed`: `AMREX_GPU_DEVICE static auto ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;`
- L130: function `ComputeIsothermalSoundSpeed`: `AMREX_GPU_DEVICE static auto ComputeIsothermalSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;`
- L134: function `ComputeMagneticEnergy`: `AMREX_GPU_DEVICE static auto ComputeMagneticEnergy(int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;`
- L138: function `ComputePlasmaBeta`: `AMREX_GPU_DEVICE static auto ComputePlasmaBeta(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc = nullptr) -> amrex::Real;`
- L141: function `ComputeVelocityX1`: `AMREX_GPU_DEVICE static auto ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;`
- L143: function `ComputeVelocityX2`: `AMREX_GPU_DEVICE static auto ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;`
- L145: function `ComputeVelocityX3`: `AMREX_GPU_DEVICE static auto ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;`
- L147: function `isStateValid`: `AMREX_GPU_DEVICE static auto isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool;`
- L149: function `ComputeRhsFromFluxes`: `static void ComputeRhsFromFluxes(amrex::MultiFab &rhs_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int nvars);`
- L152: function `PredictStep`: `static void PredictStep(amrex::MultiFab const &consVarOld, amrex::MultiFab &consVarNew, amrex::MultiFab const &rhs, double dt, int nvars, amrex::iMultiFab &redoFlag_mf);`
- L155: function `AddFluxesRK2`: `static void AddFluxesRK2(amrex::MultiFab &Unew_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, amrex::MultiFab const &rhs_mf, double dt, int nvars, amrex::iMultiFab &redoFlag_mf);`
- L158: function `GetGradFixedPotential`: `AMREX_GPU_DEVICE static auto GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;`
- L161: function `EnforceLimits`: `static void EnforceLimits(amrex::Real densityFloor, amrex::Real dustDensityFloor, amrex::Real tempFloor, amrex::MultiFab &state_mf, amrex::Geometry const &geom, DensityFloorFunc const &density_floor_func);`
- L164: function `AddInternalEnergyPdV`: `static void AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf);`
- L168: function `SyncDualEnergy`: `static void SyncDualEnergy(amrex::MultiFab &consVar_mf, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &faceVar_mf);`
- L171: function `ComputeFluxes`: `static void ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf, amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &leftState_bfield_mf, amrex::MultiFab const &rightState_bfield_mf, amrex::MultiFab const &primVar_mf, amrex::Real K_visc, amrex::MultiFab *x1FSpds_mf = nullptr, amrex::MultiFab const *x1ConsVar_fc_mf = nullptr, int nghost_vel = 2);`
- L177: function `ComputeFirstOrderFluxes`: `static void ComputeFirstOrderFluxes(amrex::Array4<const amrex::Real> const &consVar, array_t &x1FluxDiffusive, amrex::Box const &indexRange);`
- L179: function `ComputeFlatteningCoefficients`: `template <FluxDir DIR> static void ComputeFlatteningCoefficients(amrex::MultiFab const &primVar_mf, amrex::MultiFab &x1Chi_mf, int nghost);`
- L182: function `FlattenShocks`: `static void FlattenShocks(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf, amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf, int nghost, int nvars);`
- L189: function `is_eos_isothermal`: `static constexpr auto is_eos_isothermal() -> bool {`
- L195: function `ConservedToPrimitive`: `void HydroSystem<problem_t>::ConservedToPrimitive(amrex::MultiFab const &cons_cc_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf, amrex::MultiFab &primVar_mf, const int nghost) {`
- L337: function `maxSignalSpeedLocal`: `auto HydroSystem<problem_t>::maxSignalSpeedLocal(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> amrex::Real {`
- L380: function `ComputeMaxSignalSpeed`: `void HydroSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons_cc, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &cons_fc, array_t &maxSignal, amrex::Box const &indexRange) {`
- L459: function `CheckStatesValid`: `auto HydroSystem<problem_t>::CheckStatesValid(amrex::MultiFab const &cons_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf) -> bool {`
- L517: function `ComputePrimVars`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> quokka::valarray<amrex::Real, nHydroScalars_> {`
- L552: function `ComputeConsVars`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeConsVars(quokka::valarray<amrex::Real, nHydroScalars_> const &prim) -> quokka::valarray<amrex::Real, nHydroScalars_> {`
- L575: function `ComputePressure`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> amrex::Real {`
- L619: function `ComputeSoundSpeed`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> amrex::Real {`
- L681: function `ComputePlasmaBeta`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePlasmaBeta(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *cons_fc) -> amrex::Real {`
- L698: function `ComputeVelocityX1`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real {`
- L707: function `ComputeVelocityX2`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real {`
- L716: function `ComputeVelocityX3`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real {`
- L725: function `isStateValid`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool {`
- L757: function `ComputeRhsFromFluxes`: `void HydroSystem<problem_t>::ComputeRhsFromFluxes(amrex::MultiFab &rhs_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const int nvars) {`
- L784: function `PredictStep`: `void HydroSystem<problem_t>::PredictStep(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, amrex::MultiFab const &rhs_mf, const double dt, const int nvars, amrex::iMultiFab &redoFlag_mf) {`
- L808: function `AddFluxesRK2`: `void HydroSystem<problem_t>::AddFluxesRK2(amrex::MultiFab &Unew_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, amrex::MultiFab const &rhs_mf, const double dt, const int nvars, amrex::iMultiFab &redoFlag_mf) {`
- L841: function `ComputeFlatteningCoefficients`: `void HydroSystem<problem_t>::ComputeFlatteningCoefficients(amrex::MultiFab const &primVar_mf, amrex::MultiFab &x1Chi_mf, const int nghost) {`
- L938: function `FlattenShocks`: `void HydroSystem<problem_t>::FlattenShocks(amrex::MultiFab const &q_mf, amrex::MultiFab const &x1Chi_mf, amrex::MultiFab const &x2Chi_mf, amrex::MultiFab const &x3Chi_mf, amrex::MultiFab &x1LeftState_mf, amrex::MultiFab &x1RightState_mf, const int nghost, const int nvars) {`
- L1008: function `EnforceLimits`: `void HydroSystem<problem_t>::EnforceLimits(amrex::Real const densityFloor, amrex::Real const dustDensityFloor, amrex::Real const tempFloor, amrex::MultiFab &state_mf, amrex::Geometry const &geom, DensityFloorFunc const &density_floor_func) {`
- L1114: function `AddInternalEnergyPdV`: `void HydroSystem<problem_t>::AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &cons_fc_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArray, amrex::iMultiFab const &redoFlag_mf) {`
- L1170: function `SyncDualEnergy`: `void HydroSystem<problem_t>::SyncDualEnergy(amrex::MultiFab &consVar_mf, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &faceVar_mf) {`
- L1232: function `ComputeFluxes`: `void HydroSystem<problem_t>::ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab &x1FaceVel_mf, amrex::MultiFab const &x1LeftState_mf, amrex::MultiFab const &x1RightState_mf, amrex::MultiFab const &x1LeftState_bfield_mf, amrex::MultiFab const &x1RightState_bfield_mf, amrex::MultiFab const &primVar_mf, const amrex::Real K_visc, amrex::MultiFab *x1FSpds_mf, amrex::MultiFab const *x1ConsVar_fc_mf, const int nghost_vel) {`

## `src/hydro/mhd_system.hpp`

- L36: function `MinimumHydroRiemannGhost`: `AMREX_FORCE_INLINE constexpr auto MinimumHydroRiemannGhost(bool is_mhd_enabled, EMFComputeScheme emf_compute_scheme, EMFAvgScheme emf_avg_scheme, bool require_tracer_ghosts = false) -> int {`
- L58: class `MHDSystem`: `template <typename problem_t> class MHDSystem : public HyperbolicSystem<problem_t>`
- L66: function `ComputeEMF`: `static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgScheme emf_avg_scheme, SlopeLimiter plmLimiter, EMFComputeScheme emf_compute_scheme);`
- L71: function `AverageEMF`: `static void AverageEMF(amrex::Array4<amrex::Real> const &E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_E_q, amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds, std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_Bi_ieside, EMFAvgScheme emf_avg_scheme);`
- L75: function `ComputeEMF_FelkerStone2017`: `static void ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme);`
- L80: function `ComputeEMF_Balsara2025`: `static void ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme);`
- L85: function `ComputeEMF_Quokka2026`: `static void ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme);`
- L91: function `EMFAverage_LondrilloDelZanna2004`: `static void EMFAverage_LondrilloDelZanna2004(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds, std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_Bi_ieside);`
- L96: function `EMFAverage_Balsara2025`: `static void EMFAverage_Balsara2025(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds, std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_Bi_ieside);`
- L100: function `ReconstructTo`: `static void ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_iValid, int reconstructionOrder, SlopeLimiter plmLimiter);`
- L103: function `SolveInductionEqn`: `static void SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);`
- L109: function `ComputeEMF`: `void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, EMFAvgScheme emf_avg_scheme, SlopeLimiter plmLimiter, EMFComputeScheme emf_compute_scheme) {`
- L130: function `AverageEMF`: `void MHDSystem<problem_t>::AverageEMF(amrex::Array4<amrex::Real> const &E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_E_q, amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds, std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_Bi_ieside, EMFAvgScheme emf_avg_scheme) {`
- L147: function `ComputeEMF_FelkerStone2017`: `void MHDSystem<problem_t>::ComputeEMF_FelkerStone2017(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme) {`
- L371: function `ComputeEMF_Quokka2026`: `void MHDSystem<problem_t>::ComputeEMF_Quokka2026(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_vel, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme) {`
- L507: function `ComputeEMF_Balsara2025`: `void MHDSystem<problem_t>::ComputeEMF_Balsara2025(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_components, amrex::MultiFab const &cc_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, int reconstructionOrder, SlopeLimiter plmLimiter, EMFAvgScheme emf_avg_scheme) {`
- L708: function `EMFAverage_LondrilloDelZanna2004`: `void MHDSystem<problem_t>::EMFAverage_LondrilloDelZanna2004(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds, std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_Bi_ieside) {`
- L777: function `EMFAverage_Balsara2025`: `void MHDSystem<problem_t>::EMFAverage_Balsara2025(amrex::Array4<amrex::Real> E2_ave, std::array<amrex::FArrayBox, 4> const &ec_fabs_EMF_q, amrex::Box const &box_ec, std::array<int, 2> const &extrap_dirs, std::array<amrex::Array4<const amrex::Real>, 3> const &fspds, std::array<std::array<amrex::FArrayBox, 2>, 2> const &ec_fabs_Bi_ieside) {`
- L876: function `ReconstructTo`: `void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_iValid, int reconstructionOrder, SlopeLimiter plmLimiter) {`
- L950: function `SolveInductionEqn`: `void MHDSystem<problem_t>::SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx) {`

## `src/hyperbolic_system.hpp`

- L37: enum `redoFlag`: `enum redoFlag { none = 0, redo = 1 };`
- L47: class `HyperbolicSystem`: `template <typename problem_t> class HyperbolicSystem`
- L50: function `SlopeFunc`: `template <SlopeLimiter limiter> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SlopeFunc(amrex::Real x, amrex::Real y) -> amrex::Real {`
- L93: function `AssertReconstructionRanges`: `template <FluxDir DIR> static void AssertReconstructionRanges(amrex::Box const &cellRange, amrex::Box const &interfaceRange) {`
- L100: function `ReconstructStatesConstant`: `static void ReconstructStatesConstant(amrex::MultiFab const &q, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int nghost, int nvars);`
- L103: function `ReconstructStatesConstant`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesConstant(arrayconst_t &q, array_t &leftState, array_t &rightState, amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars);`
- L113: function `ReconstructStatesPLM`: `static void ReconstructStatesPLM(amrex::MultiFab const &q, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int nghost, int nvars);`
- L116: function `ReconstructStatesPLM`: `static void ReconstructStatesPLM(amrex::MultiFab const &q, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int nghost, int nvars, SlopeLimiter limiter);`
- L120: function `ReconstructStatesPLM`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPLM(arrayconst_t &q, array_t &leftState, array_t &rightState, amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars);`
- L124: function `ReconstructStatesPLM`: `static void ReconstructStatesPLM(arrayconst_t &q, array_t &leftState, array_t &rightState, amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars, SlopeLimiter limiter);`
- L133: function `ReconstructStatesPPM`: `static void ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, int nghost, int nvars, int iReadFrom = 0, int iWriteFrom = 0);`
- L137: function `ReconstructStatesPPM`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in, amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars, int iReadFrom = 0, int iWriteFrom = 0);`
- L142: function `ReconstructStatesPPM`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState, quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in, int iReadFrom = 0, int iWriteFrom = 0);`
- L147: function `MonotonizeEdges`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto MonotonizeEdges(double qL_in, double qR_in, double q, double qminus, double qplus) -> std::pair<double, double>;`
- L151: function `ComputeSteepPPM`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto ComputeSteepPPM(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n) -> amrex::Real;`
- L155: function `ComputeWENOMoments`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto ComputeWENOMoments(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n) -> std::pair<amrex::Real, amrex::Real>;`
- L159: function `ComputeWENO`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto ComputeWENO(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n) -> std::pair<amrex::Real, amrex::Real>;`
- L163: function `ReconstructStatesPPM_EP`: `static void ReconstructStatesPPM_EP(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, int nghost, int nvars, int iReadFrom = 0, int iWriteFrom = 0);`
- L167: function `ReconstructStatesPPM_EP`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM_EP(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in, amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars, int iReadFrom = 0, int iWriteFrom = 0);`
- L172: function `ReconstructStatesPPM_EP`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM_EP(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState, quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in, int iReadFrom = 0, int iWriteFrom = 0);`
- L196: function `ReconstructStatesConstant`: `void HyperbolicSystem<problem_t>::ReconstructStatesConstant(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, const int nghost, const int nvars) {`
- L216: function `ReconstructStatesConstant`: `AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in, amrex::Box const &cellRange, amrex::Box const &interfaceRange, const int nvars) {`
- L234: function `ReconstructStatesConstant`: `AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState, quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in) {`
- L253: function `ReconstructStatesPLM`: `void HyperbolicSystem<problem_t>::ReconstructStatesPLM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, const int nghost, const int nvars) {`
- L273: function `ReconstructStatesPLM`: `void HyperbolicSystem<problem_t>::ReconstructStatesPLM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, const int nghost, const int nvars, SlopeLimiter limiter) {`
- L298: function `ReconstructStatesPLM`: `AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPLM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in, amrex::Box const &cellRange, amrex::Box const &interfaceRange, const int nvars) {`
- L315: function `ReconstructStatesPLM`: `void HyperbolicSystem<problem_t>::ReconstructStatesPLM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in, amrex::Box const &cellRange, amrex::Box const &interfaceRange, const int nvars, SlopeLimiter limiter) {`
- L373: function `ReconstructStatesPPM`: `void HyperbolicSystem<problem_t>::ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, const int nghost, const int nvars, const int iReadFrom, const int iWriteFrom) {`
- L396: function `ReconstructStatesPPM`: `AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in, amrex::Box const &cellRange, amrex::Box const &interfaceRange, const int nvars, const int iReadFrom, const int iWriteFrom) {`
- L416: function `ReconstructStatesPPM`: `AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState, quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in, int iReadFrom, int iWriteFrom) {`
- L506: function `MonotonizeEdges`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::MonotonizeEdges(double qL_in, double qR_in, double q, double qminus, double qplus) -> std::pair<double, double> {`
- L523: function `ComputeSteepPPM`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::ComputeSteepPPM(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n) -> amrex::Real {`
- L539: function `ComputeWENOMoments`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::ComputeWENOMoments(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n) -> std::pair<amrex::Real, amrex::Real> {`
- L586: function `ComputeWENO`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HyperbolicSystem<problem_t>::ComputeWENO(quokka::Array4View<const amrex::Real, DIR> const &q, int i, int j, int k, int n) -> std::pair<amrex::Real, amrex::Real> {`
- L601: function `ReconstructStatesPPM_EP`: `void HyperbolicSystem<problem_t>::ReconstructStatesPPM_EP(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, const int nghost, const int nvars, const int iReadFrom, const int iWriteFrom) {`
- L624: function `ReconstructStatesPPM_EP`: `AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM_EP(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in, amrex::Box const &cellRange, amrex::Box const &interfaceRange, const int nvars, const int iReadFrom, const int iWriteFrom) {`
- L713: function `PredictStep`: `void HyperbolicSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int nvars, F &&isStateValid, amrex::Array4<int> const &redoFlag) {`
- L754: function `AddFluxesRK2`: `void HyperbolicSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int nvars, F &&isStateValid, amrex::Array4<int> const &redoFlag) {`
- L797: function `redoFlag`: `redoFlag(i, j, k) = quokka::redoFlag::redo;`
- L799: function `redoFlag`: `redoFlag(i, j, k) = quokka::redoFlag::none;`

## `src/io/DerivedFieldBase.H`

- L18: class `DerivedFieldBase`: `class DerivedFieldBase : public Factory<DerivedFieldBase>`
- L21: function `~DerivedFieldBase`: `~DerivedFieldBase() override = default;`
- L23: function `base_identifier`: `static auto base_identifier() -> std::string {`
- L25: function `init`: `virtual void init(const std::string &a_prefix, std::string_view a_fieldName);`
- L26: function `prepare`: `virtual void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_availableVars);`
- L29: function `addVars`: `virtual void addVars(amrex::Vector<std::string> &a_varList);`
- L30: struct `ComputeContext`: `struct ComputeContext {`
- L34: function `computeField`: `virtual auto computeField(int lev, const std::string &fieldName, amrex::MultiFab &mf, int ncomp, ComputeContext const &ctx) const -> bool;`

## `src/io/DerivedFieldBase.cpp`

- L8: function `DerivedFieldBase::init`: `void DerivedFieldBase::init(const std::string & , std::string_view a_fieldName) {`
- L10: function `DerivedFieldBase::prepare`: `void DerivedFieldBase::prepare(int , const amrex::Vector<amrex::Geometry> & , const amrex::Vector<amrex::BoxArray> & , const amrex::Vector<amrex::DistributionMapping> & , const amrex::Vector<std::string> & ) {`
- L15: function `DerivedFieldBase::addVars`: `void DerivedFieldBase::addVars(amrex::Vector<std::string> &a_varList) {`
- L22: function `DerivedFieldBase::computeField`: `auto DerivedFieldBase::computeField(int , const std::string & , amrex::MultiFab & , int , ComputeContext const & ) const -> bool {`
- L28: function `DerivedFieldBase::hasField`: `auto DerivedFieldBase::hasField(std::string_view field) const -> bool {`

## `src/io/DerivedParticleDeposition.H`

- L14: class `DerivedParticleDeposition`: `class DerivedParticleDeposition : public DerivedFieldBase::Register<DerivedParticleDeposition>`
- L17: function `identifier`: `static auto identifier() -> std::string {`
- L19: function `init`: `void init(const std::string &a_prefix, std::string_view a_fieldName) override;`
- L20: function `computeField`: `auto computeField(int lev, const std::string &fieldName, amrex::MultiFab &mf, int ncomp, ComputeContext const &ctx) const -> bool override;`
- L23: struct `OutputSpec`: `struct OutputSpec {`

## `src/io/DerivedParticleDeposition.cpp`

- L15: function `DerivedParticleDeposition::isSupportedParticleType`: `auto DerivedParticleDeposition::isSupportedParticleType(std::string_view particleType) -> bool {`
- L21: function `DerivedParticleDeposition::init`: `void DerivedParticleDeposition::init(const std::string &a_prefix, std::string_view a_fieldName) {`
- L93: function `DerivedParticleDeposition::computeField`: `auto DerivedParticleDeposition::computeField(int lev, const std::string &fieldName, amrex::MultiFab &mf, int ncomp, ComputeContext const &ctx) const -> bool {`
- L114: function `DerivedParticleDeposition::getFieldName`: `auto DerivedParticleDeposition::getFieldName(const std::string &particleType, const std::string &field) const -> std::string {`

## `src/io/DiagBase.H`

- L13: class `AMRSimulation`: `template <typename problem_t> class AMRSimulation;`
- L15: class `DiagBase`: `class DiagBase : public quokka::Factory<DiagBase>`
- L18: function `~DiagBase`: `~DiagBase() override = default;`
- L20: function `base_identifier`: `static auto base_identifier() -> std::string {`
- L22: function `init`: `virtual void init(const std::string &a_prefix, std::string_view a_diagName);`
- L24: function `close`: `virtual void close() = 0;`
- L28: function `doDiag`: `virtual auto doDiag(const amrex::Real &a_time, int a_nstep) -> bool;`
- L30: function `prepare`: `virtual void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames);`
- L40: function `addVars`: `virtual void addVars(amrex::Vector<std::string> &a_varList);`
- L42: function `getFieldIndex`: `static auto getFieldIndex(const std::string &a_field, const amrex::Vector<std::string> &a_varList) -> int;`
- L44: function `getFieldIndexVec`: `static auto getFieldIndexVec(const std::vector<std::string> &a_field, const amrex::Vector<std::string> &a_varList) -> amrex::Vector<int>;`
- L48: function `setDiagData`: `void setDiagData(AMRSimulation<problem_t> *sim, const amrex::Vector<const amrex::MultiFab *> *diagMF, const amrex::Vector<std::string> *diagVars, const amrex::Vector<amrex::Geometry> *geoms, const amrex::Vector<amrex::IntVect> *refRatio, const YAML::Node *metadata) {`
- L84: function `getSim`: `template <typename problem_t> auto getSim() const -> AMRSimulation<problem_t> * {`

## `src/io/DiagBase.cpp`

- L4: function `DiagBase::init`: `void DiagBase::init(const std::string &a_prefix, std::string_view a_diagName) {`
- L35: function `DiagBase::prepare`: `void DiagBase::prepare(int , const amrex::Vector<amrex::Geometry> & , const amrex::Vector<amrex::BoxArray> & , const amrex::Vector<amrex::DistributionMapping> & , const amrex::Vector<std::string> &a_varNames) {`
- L53: function `DiagBase::doDiag`: `auto DiagBase::doDiag(const amrex::Real &a_time, int a_nstep) -> bool {`
- L98: function `DiagBase::addVars`: `void DiagBase::addVars(amrex::Vector<std::string> &a_varList) {`
- L106: function `DiagBase::getFieldIndex`: `auto DiagBase::getFieldIndex(const std::string &a_field, const amrex::Vector<std::string> &a_varList) -> int {`
- L121: function `DiagBase::getFieldIndexVec`: `auto DiagBase::getFieldIndexVec(const std::vector<std::string> &a_field, const amrex::Vector<std::string> &a_varList) -> amrex::Vector<int> {`

## `src/io/DiagFilter.H`

- L7: struct `DiagFilterData`: `struct DiagFilterData {`
- L13: struct `DiagFilter`: `struct DiagFilter {`
- L16: function `init`: `void init(const std::string &a_prefix);`
- L17: function `setup`: `void setup(const amrex::Vector<std::string> &a_varNames);`

## `src/io/DiagFilter.cpp`

- L4: function `DiagFilter::init`: `void DiagFilter::init(const std::string &a_prefix) {`
- L34: function `DiagFilter::setup`: `void DiagFilter::setup(const amrex::Vector<std::string> &a_varNames) {`

## `src/io/DiagFramePlane.H`

- L14: class `PhysicsParticleRegister`: `template <typename problem_t> class PhysicsParticleRegister;`
- L17: class `DiagFramePlane`: `class DiagFramePlane : public DiagBase::Register<DiagFramePlane>`
- L20: function `identifier`: `static auto identifier() -> std::string {`
- L22: function `init`: `void init(const std::string &a_prefix, std::string_view a_diagName) override;`
- L24: function `prepare`: `void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;`
- L28: function `processDiag`: `template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);`
- L30: function `addVars`: `void addVars(amrex::Vector<std::string> &a_varList) override;`
- L35: function `Write2DMultiLevelPlotfile`: `void Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice, const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time, const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref, const YAML::Node &simulationMetadata);`
- L39: function `VisMF2D`: `static void VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name);`
- L41: function `Write2DMFHeader`: `static void Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm);`
- L43: function `Find2FOffsets`: `static void Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr, amrex::VisMF::Header::Version , amrex::NFilesIter &nfi, int nOutFile, MPI_Comm comm);`
- L46: function `write_2D_header`: `static void write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar);`
- L48: function `Write2DPlotfileHeader`: `static void Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray, const amrex::Vector<std::string> &varnames, const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time, const amrex::Vector<int> &level_steps, const amrex::Vector<amrex::IntVect> &ref_ratio, const std::string &versionName = "HyperCLaw-V1.1", const std::string &levelPrefix = "Level_", const std::string &mfPrefix = "Cell");`
- L54: function `close`: `void close() override {`
- L80: function `DiagFramePlane::processDiag`: `template <typename problem_t> void DiagFramePlane::processDiag(int a_nstep, const amrex::Real &a_time) {`

## `src/io/DiagFramePlane.cpp`

- L16: function `printLowerDimIntVect`: `void printLowerDimIntVect(std::ostream &a_File, const amrex::IntVect &a_IntVect, int skipDim) {`
- L32: function `printLowerDimBox`: `void printLowerDimBox(std::ostream &a_File, const amrex::Box &a_box, int skipDim) {`
- L43: function `DiagFramePlane::init`: `void DiagFramePlane::init(const std::string &a_prefix, std::string_view a_diagName) {`
- L96: function `DiagFramePlane::addVars`: `void DiagFramePlane::addVars(amrex::Vector<std::string> &a_varList) {`
- L104: function `DiagFramePlane::prepare`: `void DiagFramePlane::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) {`
- L213: function `DiagFramePlane::Write2DMultiLevelPlotfile`: `void DiagFramePlane::Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice, const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time, const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref, const YAML::Node &simulationMetadata) {`
- L222: function `callBarrier`: `bool const callBarrier(false);`
- L284: function `DiagFramePlane::Write2DPlotfileHeader`: `void DiagFramePlane::Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray, const amrex::Vector<std::string> &varnames, const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time, const amrex::Vector<int> &level_steps, const amrex::Vector<amrex::IntVect> &ref_ratio, const std::string &versionName, const std::string &levelPrefix, const std::string &mfPrefix) {`
- L289: function `finest_level`: `int const finest_level(nlevels - 1);`
- L351: function `DiagFramePlane::VisMF2D`: `void DiagFramePlane::VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name) {`
- L354: function `doConvert`: `bool const doConvert(*whichRD != amrex::FPC::NativeRealDescriptor());`
- L392: function `whichRDBytes`: `int const whichRDBytes(whichRD->numBytes());`
- L393: function `nFABs`: `int nFABs(0);`
- L414: function `hLength`: `int hLength(0);`
- L469: function `coordinatorProc`: `int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber());`
- L480: function `DiagFramePlane::Write2DMFHeader`: `void DiagFramePlane::Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm) {`
- L482: function `myProc`: `const int myProc(amrex::ParallelDescriptor::MyProc(comm));`
- L543: function `DiagFramePlane::Find2FOffsets`: `void DiagFramePlane::Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr, amrex::VisMF::Header::Version , amrex::NFilesIter &nfi, int nOutFiles, MPI_Comm comm) {`
- L548: function `myProc`: `const int myProc(amrex::ParallelDescriptor::MyProc(comm));`
- L549: function `nProcs`: `const int nProcs(amrex::ParallelDescriptor::NProcs(comm));`
- L550: function `coordinatorProc`: `int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber(comm));`
- L556: function `whichRDBytes`: `int const whichRDBytes(whichRD->numBytes());`
- L557: function `nComps`: `int const nComps(mf.nComp());`
- L563: function `nFiles`: `int const nFiles(amrex::NFilesIter::ActualNFiles(nOutFiles));`
- L564: function `whichFileNumber`: `int whichFileNumber(-1);`
- L617: function `DiagFramePlane::write_2D_header`: `void DiagFramePlane::write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar) {`

## `src/io/DiagPDF.H`

- L8: class `DiagPDF`: `class DiagPDF : public DiagBase::Register<DiagPDF>`
- L11: function `identifier`: `static auto identifier() -> std::string {`
- L13: function `init`: `void init(const std::string &a_prefix, std::string_view a_diagName) override;`
- L15: function `prepare`: `void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;`
- L19: function `processDiag`: `template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);`
- L21: function `addVars`: `void addVars(amrex::Vector<std::string> &a_varList) override;`
- L23: function `MFVecMin`: `static auto MFVecMin(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real;`
- L24: function `MFVecMax`: `static auto MFVecMax(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real;`
- L25: function `writePDFToFile`: `void writePDFToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_pdf);`
- L27: function `close`: `void close() override {`
- L45: function `getIdxVec`: `static auto getIdxVec(int linidx, std::vector<int> const &nBins) -> std::vector<int>;`
- L47: function `getBinIndex1D`: `AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto getBinIndex1D(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd, const amrex::Real &transformedBinWidth, bool doLog) -> int;`
- L50: function `getTotalBinCount`: `AMREX_GPU_HOST_DEVICE AMREX_INLINE auto getTotalBinCount() -> amrex::Long;`
- L54: function `DiagPDF::getBinIndex1D`: `AMREX_GPU_HOST_DEVICE AMREX_INLINE auto DiagPDF::getBinIndex1D(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd, const amrex::Real &transformedBinWidth, const bool doLog) -> int {`
- L62: function `DiagPDF::getTotalBinCount`: `AMREX_GPU_HOST_DEVICE AMREX_INLINE auto DiagPDF::getTotalBinCount() -> amrex::Long {`
- L72: function `DiagPDF::processDiag`: `template <typename problem_t> void DiagPDF::processDiag(int a_nstep, const amrex::Real &a_time) {`
- L124: function `amrex::IntVect`: `amrex::ParallelFor( *a_state[lev], amrex::IntVect(0), [=, nFilters = m_filters.size()] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {`

## `src/io/DiagPDF.cpp`

- L14: function `DiagPDF::init`: `void DiagPDF::init(const std::string &a_prefix, std::string_view a_diagName) {`
- L56: function `DiagPDF::addVars`: `void DiagPDF::addVars(amrex::Vector<std::string> &a_varList) {`
- L68: function `DiagPDF::prepare`: `void DiagPDF::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) {`
- L86: function `DiagPDF::getIdxVec`: `auto DiagPDF::getIdxVec(const int linidx, std::vector<int> const &nBins) -> std::vector<int> {`
- L103: function `DiagPDF::MFVecMin`: `auto DiagPDF::MFVecMin(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real {`
- L115: function `DiagPDF::MFVecMax`: `auto DiagPDF::MFVecMax(const amrex::Vector<const amrex::MultiFab *> &a_state, int comp) -> amrex::Real {`
- L127: function `DiagPDF::writePDFToFile`: `void DiagPDF::writePDFToFile(int a_nstep, const amrex::Real &a_time, const amrex::Vector<amrex::Real> &a_pdf) {`

## `src/io/DiagParticleTxt.H`

- L12: class `DiagParticleTxt`: `class DiagParticleTxt : public DiagBase::Register<DiagParticleTxt>`
- L15: function `identifier`: `static auto identifier() -> std::string {`
- L17: function `init`: `void init(const std::string &a_prefix, std::string_view a_diagName) override;`
- L19: function `prepare`: `void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;`
- L22: function `addVars`: `void addVars(amrex::Vector<std::string> &a_varList) override;`
- L24: function `close`: `void close() override {`
- L27: function `processDiag`: `template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);`
- L34: function `DiagParticleTxt::processDiag`: `template <typename problem_t> void DiagParticleTxt::processDiag(int a_nstep, const amrex::Real & ) {`

## `src/io/DiagParticleTxt.cpp`

- L5: function `DiagParticleTxt::init`: `void DiagParticleTxt::init(const std::string &a_prefix, std::string_view a_diagName) {`
- L39: function `DiagParticleTxt::prepare`: `void DiagParticleTxt::prepare(int , const amrex::Vector<amrex::Geometry> & , const amrex::Vector<amrex::BoxArray> & , const amrex::Vector<amrex::DistributionMapping> & , const amrex::Vector<std::string> & ) {`
- L47: function `DiagParticleTxt::addVars`: `void DiagParticleTxt::addVars(amrex::Vector<std::string> & ) {`

## `src/io/DiagPlotfile.H`

- L24: class `PhysicsParticleRegister`: `template <typename problem_t> class PhysicsParticleRegister;`
- L25: enum class `ParticleType`: `enum class ParticleType;`
- L28: class `DiagPlotfile`: `class DiagPlotfile : public DiagBase::Register<DiagPlotfile>`
- L31: function `identifier`: `static auto identifier() -> std::string {`
- L33: function `init`: `void init(const std::string &a_prefix, std::string_view a_diagName) override;`
- L35: function `prepare`: `void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;`
- L38: function `addVars`: `void addVars(amrex::Vector<std::string> &a_varList) override;`
- L40: function `close`: `void close() override {`
- L43: function `processDiag`: `template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);`
- L58: function `WriteMetadataFile`: `static void WriteMetadataFile(const std::string &metadataFilename, const YAML::Node &simulationMetadata);`
- L62: function `DiagPlotfile::processDiag`: `template <typename problem_t> void DiagPlotfile::processDiag(int a_nstep, const amrex::Real &a_time) {`
- L181: function `DiagPlotfile::WriteMetadataFile`: `inline void DiagPlotfile::WriteMetadataFile(const std::string &metadataFilename, const YAML::Node &simulationMetadata) {`

## `src/io/DiagPlotfile.cpp`

- L5: function `DiagPlotfile::init`: `void DiagPlotfile::init(const std::string &a_prefix, std::string_view a_diagName) {`
- L67: function `DiagPlotfile::prepare`: `void DiagPlotfile::prepare(int , const amrex::Vector<amrex::Geometry> & , const amrex::Vector<amrex::BoxArray> & , const amrex::Vector<amrex::DistributionMapping> & , const amrex::Vector<std::string> & ) {`
- L75: function `DiagPlotfile::addVars`: `void DiagPlotfile::addVars(amrex::Vector<std::string> & ) {`

## `src/io/DiagProjectionPlot.H`

- L19: class `PhysicsParticleRegister`: `template <typename problem_t> class PhysicsParticleRegister;`
- L22: class `DiagProjectionPlot`: `class DiagProjectionPlot : public DiagBase::Register<DiagProjectionPlot>`
- L25: function `identifier`: `static auto identifier() -> std::string {`
- L27: function `init`: `void init(const std::string &a_prefix, std::string_view a_diagName) override;`
- L29: function `prepare`: `void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids, const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;`
- L32: function `addVars`: `void addVars(amrex::Vector<std::string> &a_varList) override;`
- L34: function `close`: `void close() override {`
- L37: function `processDiag`: `template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);`
- L49: function `DiagProjectionPlot::processDiag`: `template <typename problem_t> void DiagProjectionPlot::processDiag(int a_nstep, const amrex::Real &a_time) {`

## `src/io/DiagProjectionPlot.cpp`

- L5: function `DiagProjectionPlot::init`: `void DiagProjectionPlot::init(const std::string &a_prefix, std::string_view a_diagName) {`
- L95: function `DiagProjectionPlot::prepare`: `void DiagProjectionPlot::prepare(int , const amrex::Vector<amrex::Geometry> & , const amrex::Vector<amrex::BoxArray> & , const amrex::Vector<amrex::DistributionMapping> & , const amrex::Vector<std::string> &a_varNames) {`
- L117: function `DiagProjectionPlot::addVars`: `void DiagProjectionPlot::addVars(amrex::Vector<std::string> &a_varList) {`

## `src/io/io_utils.hpp`

- L12: class `ScopedVisMFNOutFiles`: `class ScopedVisMFNOutFiles`
- L18: function `ScopedVisMFNOutFiles`: `explicit ScopedVisMFNOutFiles(int nfiles) : originalNOutFiles_(amrex::VisMF::GetNOutFiles()) {`
- L24: function `~ScopedVisMFNOutFiles`: `~ScopedVisMFNOutFiles() {`
- L31: function `ScopedVisMFNOutFiles`: `ScopedVisMFNOutFiles(const ScopedVisMFNOutFiles &) = delete;`
- L32: function `operator=`: `auto operator=(const ScopedVisMFNOutFiles &) -> ScopedVisMFNOutFiles & = delete;`
- L33: function `ScopedVisMFNOutFiles`: `ScopedVisMFNOutFiles(ScopedVisMFNOutFiles &&) = delete;`
- L34: function `operator=`: `auto operator=(ScopedVisMFNOutFiles &&) -> ScopedVisMFNOutFiles & = delete;`

## `src/io/openPMD.cpp`

- L28: function `getReversedVec`: `auto getReversedVec(const amrex::IntVect &v) -> std::vector<std::uint64_t> {`
- L39: function `getReversedVec`: `auto getReversedVec(const amrex::Real *v) -> std::vector<double> {`
- L49: function `SetupMeshComponent`: `void SetupMeshComponent(openPMD::Mesh &mesh, amrex::Geometry &full_geom) {`
- L72: function `GetMeshComponentName`: `auto GetMeshComponentName(int meshLevel, std::string const &field_name) -> std::string {`
- L86: function `WriteFile`: `void WriteFile(const std::vector<std::string> &varnames, int const output_levels, const amrex::Vector<const amrex::MultiFab *> &mf, const amrex::Vector<amrex::Geometry> &geom, const std::string &output_basename, amrex::Real const time, int const file_number) {`

## `src/io/openPMD.hpp`

- L28: function `getReversedVec`: `auto getReversedVec(const amrex::IntVect &v) -> std::vector<std::uint64_t>;`
- L29: function `getReversedVec`: `auto getReversedVec(const amrex::Real *v) -> std::vector<double>;`
- L30: function `SetupMeshComponent`: `void SetupMeshComponent(openPMD::Mesh &mesh, int , const std::string &comp_name, amrex::Geometry &full_geom);`
- L31: function `GetMeshComponentName`: `auto GetMeshComponentName(int meshLevel, std::string const &field_name) -> std::string;`
- L35: function `WriteFile`: `void WriteFile(const std::vector<std::string> &varnames, int output_levels, const amrex::Vector<const amrex::MultiFab *> &mf, const amrex::Vector<amrex::Geometry> &geom, const std::string &output_basename, amrex::Real time, int file_number);`

## `src/io/projection.cpp`

- L30: function `direction_to_string`: `auto direction_to_string(const amrex::Direction dir) -> std::string {`
- L50: function `printLowerDimIntVect`: `void printLowerDimIntVect(std::ostream &a_File, const amrex::IntVect &a_IntVect, int skipDim) {`
- L66: function `printLowerDimBox`: `void printLowerDimBox(std::ostream &a_File, const amrex::Box &a_box, int skipDim) {`
- L77: function `Write2DMultiLevelPlotfile`: `void Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice, const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time, const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref) {`
- L87: function `callBarrier`: `bool const callBarrier(false);`
- L118: function `Write2DPlotfileHeader`: `void Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray, const amrex::Vector<std::string> &varnames, const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time, const amrex::Vector<int> &level_steps, const amrex::Vector<amrex::IntVect> &ref_ratio, const std::string &versionName, const std::string &levelPrefix, const std::string &mfPrefix) {`
- L123: function `finest_level`: `int const finest_level(nlevels - 1);`
- L185: function `VisMF2D`: `void VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name) {`
- L188: function `doConvert`: `bool const doConvert(*whichRD != amrex::FPC::NativeRealDescriptor());`
- L226: function `whichRDBytes`: `int const whichRDBytes(whichRD->numBytes());`
- L227: function `nFABs`: `int nFABs(0);`
- L248: function `hLength`: `int hLength(0);`
- L303: function `coordinatorProc`: `int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber());`
- L314: function `Write2DMFHeader`: `void Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm) {`
- L316: function `myProc`: `const int myProc(amrex::ParallelDescriptor::MyProc(comm));`
- L377: function `Find2FOffsets`: `void Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr, amrex::VisMF::Header::Version , amrex::NFilesIter &nfi, int nOutFiles, MPI_Comm comm) {`
- L382: function `myProc`: `const int myProc(amrex::ParallelDescriptor::MyProc(comm));`
- L383: function `nProcs`: `const int nProcs(amrex::ParallelDescriptor::NProcs(comm));`
- L384: function `coordinatorProc`: `int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber(comm));`
- L390: function `whichRDBytes`: `int const whichRDBytes(whichRD->numBytes());`
- L391: function `nComps`: `int const nComps(mf.nComp());`
- L397: function `nFiles`: `int const nFiles(amrex::NFilesIter::ActualNFiles(nOutFiles));`
- L398: function `whichFileNumber`: `int whichFileNumber(-1);`
- L451: function `write_2D_header`: `void write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar) {`
- L461: function `transform_box_to_2D`: `auto transform_box_to_2D(amrex::Direction const &dir, amrex::Box const &box) -> amrex::Box {`
- L490: function `transform_realbox_to_2D`: `auto transform_realbox_to_2D(amrex::Direction const &dir, amrex::RealBox const &box) -> amrex::RealBox {`
- L519: function `transform_ref_ratio_to_2D`: `auto transform_ref_ratio_to_2D(amrex::Direction const &dir, amrex::IntVect const &ref_ratio) -> amrex::IntVect {`
- L541: function `WriteProjection`: `void WriteProjection(amrex::Direction dir, std::unordered_map<std::string, amrex::Vector<amrex::MultiFab>> const &proj, amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Real time, int istep, const std::string &basename, const YAML::Node &simulationMetadata) {`

## `src/io/projection.hpp`

- L32: class `PhysicsParticleRegister`: `template <typename problem_t> class PhysicsParticleRegister;`
- L43: function `direction_to_string`: `auto direction_to_string(amrex::Direction dir) -> std::string;`
- L44: function `transform_box_to_2D`: `auto transform_box_to_2D(amrex::Direction const &dir, amrex::Box const &box) -> amrex::Box;`
- L45: function `transform_realbox_to_2D`: `auto transform_realbox_to_2D(amrex::Direction const &dir, amrex::RealBox const &box) -> amrex::RealBox;`
- L47: function `printLowerDimIntVect`: `void printLowerDimIntVect(std::ostream &a_File, const amrex::IntVect &a_IntVect, int skipDim);`
- L48: function `printLowerDimBox`: `void printLowerDimBox(std::ostream &a_File, const amrex::Box &a_box, int skipDim);`
- L50: function `Write2DMultiLevelPlotfile`: `void Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice, const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time, const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref);`
- L54: function `Write2DPlotfileHeader`: `void Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray, const amrex::Vector<std::string> &varnames, const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time, const amrex::Vector<int> &level_steps, const amrex::Vector<amrex::IntVect> &ref_ratio, const std::string &versionName, const std::string &levelPrefix, const std::string &mfPrefix);`
- L59: function `VisMF2D`: `void VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name);`
- L61: function `Write2DMFHeader`: `void Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm);`
- L63: function `Find2FOffsets`: `void Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr, amrex::VisMF::Header::Version , amrex::NFilesIter &nfi, int nOutFiles, MPI_Comm comm);`
- L66: function `write_2D_header`: `void write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar);`
- L70: function `ComputePlaneProjectionFromMultiFab`: `inline auto ComputePlaneProjectionFromMultiFab(const amrex::Vector<const amrex::MultiFab *> &mfs, const int finest_level, amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Vector<amrex::IntVect> const &max_grid_size, const amrex::Direction dir, const int comp) -> amrex::Vector<amrex::MultiFab> {`
- L181: function `WriteProjection`: `void WriteProjection(amrex::Direction dir, std::unordered_map<std::string, amrex::Vector<amrex::MultiFab>> const &proj, amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Real time, int istep, const std::string &basename, const YAML::Node &simulationMetadata);`
- L187: function `WriteProjection`: `void WriteProjection(amrex::Direction dir, std::unordered_map<std::string, amrex::Vector<amrex::MultiFab>> const &proj, amrex::Vector<amrex::Geometry> const &geom, amrex::Vector<amrex::IntVect> const &ref_ratio, amrex::Real time, int istep, const std::string &basename, quokka::PhysicsParticleRegister<problem_t> &particleRegister, const std::vector<std::string> &particleTypes, const YAML::Node &simulationMetadata) {`

## `src/linear_advection/AdvectionSimulation.hpp`

- L33: class `AdvectionSimulation`: `template <typename problem_t> class AdvectionSimulation : public AMRSimulation<problem_t>`
- L73: function `AdvectionSimulation`: `explicit AdvectionSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : AMRSimulation<problem_t>(BCs_cc) {`
- L74: function `AdvectionSimulation`: `explicit AdvectionSimulation() : AMRSimulation<problem_t>() {`
- L76: function `initialize`: `void initialize() {`
- L82: function `setCustomGhostCells`: `void setCustomGhostCells() override {`
- L91: function `computeMaxSignalLocal`: `void computeMaxSignalLocal(int level) override;`
- L92: function `printCellProperties`: `void printCellProperties(int lev, amrex::IntVect const &index) override;`
- L93: function `preCalculateInitialConditions`: `void preCalculateInitialConditions() override;`
- L94: function `setInitialConditionsOnGrid`: `void setInitialConditionsOnGrid(quokka::grid const &grid_elem) override;`
- L95: function `setInitialConditionsOnGridFaceVars`: `void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) override;`
- L97: function `createInitialRadParticles`: `void createInitialRadParticles() override;`
- L98: function `createInitialCICParticles`: `void createInitialCICParticles() override;`
- L99: function `createInitialCICRadParticles`: `void createInitialCICRadParticles() override;`
- L100: function `createInitialStochasticStellarPopParticles`: `void createInitialStochasticStellarPopParticles() override;`
- L101: function `createInitialSinkParticles`: `void createInitialSinkParticles() override;`
- L102: function `createInitialTestParticles`: `void createInitialTestParticles() override;`
- L104: function `advanceSingleTimestepAtLevel`: `void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ) override;`
- L105: function `computeBeforeTimestep`: `void computeBeforeTimestep() override;`
- L106: function `computeAfterTimestep`: `void computeAfterTimestep() override;`
- L107: function `computeAfterEvolve`: `void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;`
- L108: function `computeReferenceSolution`: `void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi);`
- L110: function `WriteSingleLevelPlotfileSimplified`: `void WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf, const amrex::Vector<std::string> &compNames, int lev, int interval) override;`
- L112: function `fillPoissonRhsAtLevel`: `void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) override;`
- L113: function `applyPoissonGravityAtLevel`: `void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) override;`
- L116: function `ComputeDerivedVar`: `void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;`
- L120: function `ComputeStatistics`: `auto ComputeStatistics() -> std::map<std::string, amrex::Real> override;`
- L122: function `FixupState`: `void FixupState(int lev) override;`
- L125: function `refineGrid`: `void refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;`
- L127: function `ErrorEst`: `void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;`
- L129: function `computeFluxes`: `auto computeFluxes(amrex::MultiFab const &consVar, int nvars, int lev) -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>;`
- L134: function `fluxFunction`: `void fluxFunction(amrex::MultiFab const &consState, amrex::MultiFab &primVar, amrex::MultiFab &x1Flux, amrex::MultiFab &x1FaceVel, amrex::MultiFab &x1LeftState, amrex::MultiFab &x1RightState, int ng_reconstruct, int nvars);`
- L146: function `computeMaxSignalLocal`: `template <typename problem_t> void AdvectionSimulation<problem_t>::computeMaxSignalLocal(int const level) {`
- L157: function `printCellProperties`: `template <typename problem_t> void AdvectionSimulation<problem_t>::printCellProperties(int lev, amrex::IntVect const &index) {`
- L162: function `fillPoissonRhsAtLevel`: `template <typename problem_t> void AdvectionSimulation<problem_t>::fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) {`
- L167: function `applyPoissonGravityAtLevel`: `template <typename problem_t> void AdvectionSimulation<problem_t>::applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) {`
- L172: function `preCalculateInitialConditions`: `template <typename problem_t> void AdvectionSimulation<problem_t>::preCalculateInitialConditions() {`
- L178: function `setInitialConditionsOnGrid`: `template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L184: function `setInitialConditionsOnGridFaceVars`: `template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L193: function `createInitialRadParticles`: `template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialRadParticles() {`
- L200: function `createInitialCICParticles`: `template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialCICParticles() {`
- L207: function `createInitialCICRadParticles`: `template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialCICRadParticles() {`
- L214: function `createInitialStochasticStellarPopParticles`: `template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialStochasticStellarPopParticles() {`
- L220: function `createInitialSinkParticles`: `template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialSinkParticles() {`
- L226: function `createInitialTestParticles`: `template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialTestParticles() {`
- L233: function `computeBeforeTimestep`: `template <typename problem_t> void AdvectionSimulation<problem_t>::computeBeforeTimestep() {`
- L238: function `computeAfterTimestep`: `template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterTimestep() {`
- L243: function `ComputeDerivedVar`: `template <typename problem_t> void AdvectionSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const {`
- L248: function `ComputeStatistics`: `template <typename problem_t> auto AdvectionSimulation<problem_t>::ComputeStatistics() -> std::map<std::string, amrex::Real> {`
- L254: function `refineGrid`: `template <typename problem_t> void AdvectionSimulation<problem_t>::refineGrid(int , amrex::TagBoxArray & , amrex::Real , int ) {`
- L260: function `ErrorEst`: `template <typename problem_t> void AdvectionSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) {`
- L266: function `FixupState`: `template <typename problem_t> void AdvectionSimulation<problem_t>::FixupState(int lev) {`
- L272: function `computeReferenceSolution`: `void AdvectionSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi) {`
- L279: function `computeAfterEvolve`: `template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> & ) {`
- L306: function `advanceSingleTimestepAtLevel`: `template <typename problem_t> void AdvectionSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ) {`
- L415: function `computeFluxes`: `auto AdvectionSimulation<problem_t>::computeFluxes(amrex::MultiFab const &consVar, const int nvars, const int lev) -> std::tuple<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>> {`
- L452: function `fluxFunction`: `void AdvectionSimulation<problem_t>::fluxFunction(amrex::MultiFab const &consState, amrex::MultiFab &primVar, amrex::MultiFab &x1Flux, amrex::MultiFab &x1FaceVel, amrex::MultiFab &x1LeftState, amrex::MultiFab &x1RightState, const int ng_reconstruct, const int nvars) {`
- L480: function `WriteSingleLevelPlotfileSimplified`: `void AdvectionSimulation<problem_t>::WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf, const amrex::Vector<std::string> &compNames, int lev, int interval) {`

## `src/linear_advection/linear_advection.hpp`

- L22: class `LinearAdvectionSystem`: `template <typename problem_t> class LinearAdvectionSystem : public HyperbolicSystem<problem_t>`
- L25: enum `varIndex`: `enum varIndex { density_index = 0 };`
- L29: function `ConservedToPrimitive`: `static void ConservedToPrimitive(amrex::MultiFab const &cons_mf, amrex::MultiFab &primVar_mf, int nghost, int nvars);`
- L31: function `ComputeMaxSignalSpeed`: `static void ComputeMaxSignalSpeed(amrex::Array4<amrex::Real const> const & , amrex::Array4<amrex::Real> const &maxSignal, double advectionVx, double advectionVy, double advectionVz, amrex::Box const &indexRange);`
- L35: function `isStateValid`: `static auto isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool;`
- L37: function `PredictStep`: `static void PredictStep(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, int nvars);`
- L41: function `AddFluxesRK2`: `static void AddFluxesRK2(amrex::MultiFab &U_new_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, int nvars);`
- L46: function `ComputeFluxes`: `static void ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab const &x1LeftState_mf, amrex::MultiFab const &x1RightState_mf, amrex::MultiFab &x1FaceVel_mf, double advectionVx, int nvars);`
- L51: function `ComputeMaxSignalSpeed`: `void LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<amrex::Real const> const & , amrex::Array4<amrex::Real> const &maxSignal, const double advectionVx, const double advectionVy, const double advectionVz, amrex::Box const &indexRange) {`
- L65: function `ConservedToPrimitive`: `void LinearAdvectionSystem<problem_t>::ConservedToPrimitive(amrex::MultiFab const &cons_mf, amrex::MultiFab &primVar_mf, const int nghost, const int nvars) {`
- L75: function `isStateValid`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto LinearAdvectionSystem<problem_t>::isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool {`
- L85: function `PredictStep`: `void LinearAdvectionSystem<problem_t>::PredictStep(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, const double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, const int nvars) {`
- L120: function `AddFluxesRK2`: `void LinearAdvectionSystem<problem_t>::AddFluxesRK2(amrex::MultiFab &U_new_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, const double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, const int nvars) {`
- L167: function `ComputeFluxes`: `void LinearAdvectionSystem<problem_t>::ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab const &x1LeftState_mf, amrex::MultiFab const &x1RightState_mf, amrex::MultiFab &x1FaceVel_mf, const double vx, const int nvars) {`

## `src/main.cpp`

- L18: function `main`: `auto main(int argc, char **argv) -> int {`

## `src/main.hpp`

- L18: function `problem_main`: `auto problem_main() -> int;`

## `src/math/FastMath.hpp`

- L34: function `fastlg`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto fastlg(const double x) -> double {`
- L42: function `fastpow2`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto fastpow2(const double x) -> double {`
- L51: function `lg`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto lg(const double x) -> double {`
- L57: function `pow2`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow2(const double x) -> double {`
- L59: function `log10`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto log10(const double x) -> double {`
- L65: function `pow10`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow10(const double x) -> double {`
- L74: function `inverse_pow2`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto inverse_pow2(const double x) -> double {`
- L109: function `log10`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto log10(const double x) -> double {`
- L111: function `pow10`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow10(const double x) -> double {`

## `src/math/Interpolate2D.hpp`

- L17: function `interpolate2d`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate2d(double x, double y, amrex::Table1D<const double> const &xv, amrex::Table1D<const double> const &yv, amrex::Table2D<const double> const &table) -> double {`

## `src/math/ODEIntegrate.hpp`

- L27: function `rk12_single_step`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rk12_single_step(F const &rhs, Real t0, quokka::valarray<Real, N> const &y, Real dt, quokka::valarray<Real, N> &ynew, quokka::valarray<Real, N> &yerr) -> int {`
- L60: function `rk23_single_step`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rk23_single_step(F const &rhs, Real t0, quokka::valarray<Real, N> const &y, Real dt, quokka::valarray<Real, N> &ynew, quokka::valarray<Real, N> &yerr) -> int {`
- L110: function `error_norm`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto error_norm(quokka::valarray<Real, N> const &y0, quokka::valarray<Real, N> const &yerr, Real reltol, quokka::valarray<Real, N> const &abstol) -> Real {`
- L128: function `rk_adaptive_integrate`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rk_adaptive_integrate(F const &rhs, Real t0, quokka::valarray<Real, N> &y0, Real t1, Real reltol, quokka::valarray<Real, N> const &abstol, int &steps_taken) {`

## `src/math/gauss.hpp`

- L26: struct `gauss_constant_category`: `template <class T> struct gauss_constant_category {`
- L28: function `get_value`: `static constexpr auto get_value() -> unsigned {`
- L59: class `gauss_detail`: `template <class Real, unsigned N, unsigned Category> class gauss_detail;`
- L61: class `gauss_detail`: `template <class T> class gauss_detail<T, 7, 0>`
- L64: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const & {`
- L74: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const & {`
- L86: class `gauss_detail`: `template <class T> class gauss_detail<T, 7, 1>`
- L89: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const & {`
- L99: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const & {`
- L111: class `gauss_detail`: `template <class T> class gauss_detail<T, 7, 2>`
- L114: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const & {`
- L124: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const & {`
- L136: class `gauss_detail`: `template <class T> class gauss_detail<T, 7, 3>`
- L139: function `abscissa`: `static std::array<T, 4> const &abscissa() {`
- L149: function `weights`: `static std::array<T, 4> const &weights() {`
- L162: class `gauss_detail`: `template <class T> class gauss_detail<T, 10, 0>`
- L165: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const & {`
- L172: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const & {`
- L181: class `gauss_detail`: `template <class T> class gauss_detail<T, 10, 1>`
- L184: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const & {`
- L191: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const & {`
- L200: class `gauss_detail`: `template <class T> class gauss_detail<T, 10, 2>`
- L203: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const & {`
- L211: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const & {`
- L221: class `gauss_detail`: `template <class T> class gauss_detail<T, 10, 3>`
- L224: function `abscissa`: `static std::array<T, 5> const &abscissa() {`
- L232: function `weights`: `static std::array<T, 5> const &weights() {`
- L243: class `gauss_detail`: `template <class T> class gauss_detail<T, 15, 0>`
- L246: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const & {`
- L254: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const & {`
- L264: class `gauss_detail`: `template <class T> class gauss_detail<T, 15, 1>`
- L267: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const & {`
- L275: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const & {`
- L285: class `gauss_detail`: `template <class T> class gauss_detail<T, 15, 2>`
- L288: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const & {`
- L297: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const & {`
- L308: class `gauss_detail`: `template <class T> class gauss_detail<T, 15, 3>`
- L311: function `abscissa`: `static std::array<T, 8> const &abscissa() {`
- L320: function `weights`: `static std::array<T, 8> const &weights() {`
- L332: class `gauss_detail`: `template <class T> class gauss_detail<T, 20, 0>`
- L335: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const & {`
- L343: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const & {`
- L353: class `gauss_detail`: `template <class T> class gauss_detail<T, 20, 1>`
- L356: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const & {`
- L364: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const & {`
- L374: class `gauss_detail`: `template <class T> class gauss_detail<T, 20, 2>`
- L377: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const & {`
- L387: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const & {`
- L399: class `gauss_detail`: `template <class T> class gauss_detail<T, 20, 3>`
- L402: function `abscissa`: `static std::array<T, 10> const &abscissa() {`
- L412: function `weights`: `static std::array<T, 10> const &weights() {`
- L425: class `gauss_detail`: `template <class T> class gauss_detail<T, 25, 0>`
- L428: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const & {`
- L436: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const & {`
- L446: class `gauss_detail`: `template <class T> class gauss_detail<T, 25, 1>`
- L449: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const & {`
- L458: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const & {`
- L469: class `gauss_detail`: `template <class T> class gauss_detail<T, 25, 2>`
- L472: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const & {`
- L483: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const & {`
- L496: class `gauss_detail`: `template <class T> class gauss_detail<T, 25, 3>`
- L499: function `abscissa`: `static std::array<T, 13> const &abscissa() {`
- L510: function `weights`: `static std::array<T, 13> const &weights() {`
- L524: class `gauss_detail`: `template <class T> class gauss_detail<T, 30, 0>`
- L527: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const & {`
- L536: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const & {`
- L547: class `gauss_detail`: `template <class T> class gauss_detail<T, 30, 1>`
- L550: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const & {`
- L559: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const & {`
- L570: class `gauss_detail`: `template <class T> class gauss_detail<T, 30, 2>`
- L573: function `abscissa`: `AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const & {`
- L584: function `weights`: `AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const & {`
- L597: class `gauss_detail`: `template <class T> class gauss_detail<T, 30, 3>`
- L600: function `abscissa`: `static std::array<T, 15> const &abscissa() {`
- L611: function `weights`: `static std::array<T, 15> const &weights() {`
- L629: class `gauss`: `template <class Real, unsigned N> class gauss : public detail::gauss_detail<Real, N, detail::gauss_constant_category<Real>::value>`
- L638: function `integrate`: `template <class F> AMREX_GPU_DEVICE static auto integrate(F f, Real *pL1 = nullptr) -> decltype(f(Real(0.0))) {`
- L666: function `integrate`: `template <class F> AMREX_GPU_DEVICE static auto integrate(F f, Real a, Real b, Real *pL1 = nullptr) -> decltype(f(Real(0.0))) {`

## `src/math/interpolate.hpp`

- L15: enum class `BoundaryPolicy`: `enum class BoundaryPolicy {`
- L22: function `binary_search_with_guess`: `AMREX_GPU_HOST_DEVICE inline auto binary_search_with_guess(const double key, const double *arr, int64_t len, int64_t guess) -> int64_t {`
- L90: function `interpolate_arrays`: `AMREX_GPU_HOST_DEVICE inline void interpolate_arrays(double *x, double *y, int len, double *arr_x, const double *arr_y, int arr_len) {`
- L116: function `interpolate_value`: `AMREX_GPU_HOST_DEVICE auto interpolate_value(double x, double const *arr_x, double const *arr_y, int arr_len) -> double {`

## `src/math/math_impl.hpp`

- L15: function `clamp`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto clamp(double v, double lo, double hi) -> double {`
- L18: function `sgn`: `template <typename T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sgn(T val) -> int {`

## `src/math/quadrature.hpp`

- L14: function `kernel_wendland_c2`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto kernel_wendland_c2(const amrex::Real r) -> amrex::Real {`
- L26: function `quad_3d`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_3d(F f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1, amrex::Real z0, amrex::Real z1) -> amrex::Real {`
- L72: function `quad_2d`: `template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_2d(F f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1) -> amrex::Real {`
- L79: function `quad_1d`: `template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_1d(F f, amrex::Real x0, amrex::Real x1) -> amrex::Real {`

## `src/math/root_finding.hpp`

- L71: class `eps_tolerance`: `template <class T> class eps_tolerance`
- L74: function `eps_tolerance`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE eps_tolerance() {`
- L76: function `eps_tolerance`: `AMREX_FORCE_INLINE explicit eps_tolerance(amrex::Real eps_) {`
- L78: function `eps_tolerance`: `AMREX_FORCE_INLINE explicit eps_tolerance(unsigned bits) {`
- L83: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool operator()(const T &a, const T &b) {`
- L96: function `bracket`: `template <class F, class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void bracket(F f, T &a, T &b, T c, T &fa, T &fb, T &d, T &fd) {`
- L150: function `safe_div`: `template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T safe_div(T num, T denom, T r) {`
- L167: function `secant_interpolate`: `template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T secant_interpolate(const T &a, const T &b, const T &fa, const T &fb) {`
- L189: function `quadratic_interpolate`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T quadratic_interpolate(const T &a, const T &b, T const &d, const T &fa, const T &fb, T const &fd, unsigned count) {`
- L237: function `cubic_interpolate`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T cubic_interpolate(const T &a, const T &b, const T &d, const T &e, const T &fa, const T &fb, const T &fd, const T &fe) {`
- L279: function `toms748_solve`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE std::pair<T, T> toms748_solve(F f, const T &ax, const T &bx, const T &fax, const T &fbx, Tol tol, int &max_iter) {`
- L441: function `toms748_solve`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE std::pair<T, T> toms748_solve(F f, const T &ax, const T &bx, Tol tol, int &max_iter) {`

## `src/math/spherical_geometry.hpp`

- L19: function `minDistSqToInterval`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto minDistSqToInterval(amrex::Real const a0, amrex::Real const a1) -> amrex::Real {`
- L30: function `maxDistSqToInterval`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto maxDistSqToInterval(amrex::Real const a0, amrex::Real const a1) -> amrex::Real {`
- L39: function `addPointUnique`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto addPointUnique(amrex::GpuArray<Point, MaxPts> &pts, int &npts, amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real tol) -> void {`
- L57: function `planeBoxSectionArea`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto planeBoxSectionArea(amrex::Real const x0, amrex::Real const x1, amrex::Real const y0, amrex::Real const y1, amrex::Real const z0, amrex::Real const z1, amrex::Real const nx, amrex::Real const ny, amrex::Real const nz, amrex::Real const d) -> amrex::Real {`
- L192: function `sphericalSectionAreaInCell`: `AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto sphericalSectionAreaInCell(amrex::Real const R, amrex::Real const x0, amrex::Real const x1, amrex::Real const y0, amrex::Real const y1, amrex::Real const z0, amrex::Real const z1) -> amrex::Real {`

## `src/particles/PhysicsParticles.hpp`

- L108: class `PhysicsParticleRegister`: `template <typename problem_t> class PhysicsParticleRegister;`
- L111: class `PhysicsParticleDescriptorBase`: `class PhysicsParticleDescriptorBase`
- L129: function `PhysicsParticleDescriptorBase`: `PhysicsParticleDescriptorBase(int mass_idx, int lum_idx, int birth_time_idx, int death_time_idx, bool allows_creation, bool allows_destruction = false, int evolution_stage_idx = -1, bool allows_accretion = false, int mass_at_birth_idx = -1, int mdot_idx = -1, int ang_mom_idx = -1) : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), deathTimeIndex_(death_time_idx), allowsCreation_(allows_creation), allowsDestruction_(allows_destruction), evolutionStageIndex_(evolution_stage_idx), allowsAccretion_(allows_accretion), massAtBirthIndex_(mass_at_birth_idx), mdotIndex_(mdot_idx), angMomIndex_(ang_mom_idx) {`
- L138: function `~PhysicsParticleDescriptorBase`: `virtual ~PhysicsParticleDescriptorBase() = default;`
- L141: function `PhysicsParticleDescriptorBase`: `PhysicsParticleDescriptorBase(const PhysicsParticleDescriptorBase &) = default;`
- L142: function `operator=`: `auto operator=(const PhysicsParticleDescriptorBase &) -> PhysicsParticleDescriptorBase & = default;`
- L143: function `PhysicsParticleDescriptorBase`: `PhysicsParticleDescriptorBase(PhysicsParticleDescriptorBase &&) = default;`
- L144: function `operator=`: `auto operator=(PhysicsParticleDescriptorBase &&) -> PhysicsParticleDescriptorBase & = default;`
- L161: function `setForceFinestLevel`: `AMREX_FORCE_INLINE void setForceFinestLevel(bool force) {`
- L171: function `depositRadiation`: `virtual void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) = 0;`
- L174: function `redistribute`: `virtual void redistribute(int lev) const = 0;`
- L177: function `redistribute`: `virtual void redistribute(int lev, int ngrow) const = 0;`
- L180: function `writePlotFile`: `virtual void writePlotFile(const std::string &plotfilename, const std::string &name) = 0;`
- L183: function `writeCheckpoint`: `virtual void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) = 0;`
- L186: function `writeUnitsFile`: `virtual void writeUnitsFile(const std::string &snapshot_name, const std::string &name) = 0;`
- L189: function `printParticleStatistics`: `virtual void printParticleStatistics() const = 0;`
- L192: function `saveParticleDataToTxtFile`: `virtual void saveParticleDataToTxtFile(const std::string &plotfilename, const std::string &name) = 0;`
- L202: function `depositMass`: `virtual void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) = 0;`
- L203: function `depositParticleMassDensity`: `virtual void depositParticleMassDensity(amrex::MultiFab &deposition_field, int lev, int start_mesh_comp, amrex::Real mass_min, amrex::Real mass_max, bool use_age_filter, amrex::Real current_time, amrex::Real age_max, bool deposit_birth_mass) const = 0;`
- L207: function `driftParticles`: `virtual void driftParticles(int lev_min, int lev_max, amrex::Real dt) const = 0;`
- L210: function `kickParticles`: `virtual void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const &accel) = 0;`
- L213: function `destroyParticles`: `virtual void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt) = 0;`
- L215: function `splitParticles`: `virtual void splitParticles(int lev, int splitFactor) = 0;`
- L219: function `tagCellsAroundParticles`: `virtual void tagCellsAroundParticles(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) const = 0;`
- L223: function `depositSN`: `virtual auto depositSN(amrex::MultiFab & , std::array<amrex::MultiFab, AMREX_SPACEDIM> const * , int , amrex::Real , amrex::Real ) -> std::pair<int, amrex::Real> {`
- L229: function `computeSinkAccretion`: `virtual void computeSinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time, amrex::Real dt) {`
- L237: function `createParticlesFromState`: `virtual void createParticlesFromState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int verbose) {`
- L242: function `applySinkAccretion`: `virtual void applySinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, const amrex::Geometry &geom, int lev, amrex::Real time, amrex::Real dt) {`
- L249: function `updateParticleProperties`: `virtual void updateParticleProperties(amrex::Real current_time, Real dt) {`
- L253: class `PhysicsParticleDescriptor`: `template <typename ContainerType, typename problem_t, ParticleType particleType> class PhysicsParticleDescriptor : public PhysicsParticleDescriptorBase`
- L266: function `PhysicsParticleDescriptor`: `PhysicsParticleDescriptor(ContainerType *container, int mass_idx, int lum_idx, int birth_time_idx, int death_time_idx, bool allows_creation, bool allows_destruction = false, int evolution_stage_idx = -1, bool allows_accretion = false, int mass_at_birth_idx = -1, int mdot_idx = -1, int ang_mom_idx = -1) : PhysicsParticleDescriptorBase(mass_idx, lum_idx, birth_time_idx, death_time_idx, allows_creation, allows_destruction, evolution_stage_idx, allows_accretion, mass_at_birth_idx, mdot_idx, ang_mom_idx), container_(container) {`
- L391: function `depositMass`: `void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) override {`
- L404: function `depositParticleMassDensity`: `void depositParticleMassDensity(amrex::MultiFab &deposition_field, int lev, int start_mesh_comp, amrex::Real mass_min, amrex::Real mass_max, bool use_age_filter, amrex::Real current_time, amrex::Real age_max, bool deposit_birth_mass) const override {`
- L434: function `driftParticles`: `void driftParticles(int lev_min, int lev_max, amrex::Real dt) const override {`
- L462: function `kickParticles`: `void kickParticles(int lev, amrex::Real dt, amrex::MultiFab const &accel) override {`
- L500: function `destroyParticles`: `void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt) override {`
- L508: function `splitParticles`: `void splitParticles(int const lev, int const splitFactor) override {`
- L659: function `depositRadiation`: `void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int nGroups) override {`
- L668: function `redistribute`: `void redistribute(int lev) const override {`
- L676: function `redistribute`: `void redistribute(int lev, int ngrow) const override {`
- L690: function `writePlotFile`: `void writePlotFile(const std::string &plotfilename, const std::string &name) override {`
- L700: function `writeCheckpoint`: `void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) override {`
- L710: function `writeUnitsFile`: `void writeUnitsFile(const std::string &snapshot_name, const std::string &name) override {`
- L717: function `printParticleStatistics`: `void printParticleStatistics() const override {`
- L724: function `saveParticleDataToTxtFile`: `void saveParticleDataToTxtFile(const std::string &filename, const std::string &name) override {`
- L732: function `tagCellsAroundParticles`: `void tagCellsAroundParticles(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) const override {`
- L763: function `updateParticleProperties`: `void updateParticleProperties(amrex::Real current_time, amrex::Real dt) override {`
- L769: function `depositSN`: `auto depositSN(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time, amrex::Real dt) -> std::pair<int, amrex::Real> override {`
- L798: function `computeSinkAccretion`: `void computeSinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time, amrex::Real dt) override {`
- L806: function `applySinkAccretion`: `void applySinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, const amrex::Geometry &geom, int lev, amrex::Real time, amrex::Real dt) override {`
- L813: function `createParticlesFromState`: `void createParticlesFromState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int verbose) override {`
- L824: class `PhysicsParticleRegister`: `template <typename problem_t> class PhysicsParticleRegister`
- L835: function `PhysicsParticleRegister`: `PhysicsParticleRegister() = default;`
- L837: function `~PhysicsParticleRegister`: `~PhysicsParticleRegister() = default;`
- L897: function `registerParticleType`: `template <ParticleType particleType, typename ContainerType> void registerParticleType(ContainerType *container) {`
- L944: function `depositRadiation`: `void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time) {`
- L955: function `depositMass`: `void depositMass(const amrex::Vector<amrex::MultiFab *> &rhs, int finest_lev, amrex::Real Gconst) {`
- L965: function `depositParticleMassDensity`: `void depositParticleMassDensity(std::string_view particle_type_name, bool deposit_birth_mass, amrex::MultiFab &deposition_field, int lev, int start_mesh_comp, amrex::Real mass_min, amrex::Real mass_max, bool use_age_filter, amrex::Real current_time, amrex::Real age_max) const {`
- L985: function `depositSN`: `auto depositSN(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time, amrex::Real dt) -> std::pair<int, amrex::Real> {`
- L1001: function `computeSinkAccretion`: `void computeSinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time, amrex::Real dt) {`
- L1013: function `applySinkAccretion`: `void applySinkAccretion(amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, const amrex::Geometry &geom, int lev, amrex::Real time, amrex::Real dt) {`
- L1025: function `redistribute`: `void redistribute(int lev) const {`
- L1034: function `redistribute`: `void redistribute(int lev, int ngrow) const {`
- L1043: function `writePlotFile`: `void writePlotFile(const std::string &plotfilename) {`
- L1053: function `writePlotFileFiltered`: `void writePlotFileFiltered(const std::string &plotfilename, const std::vector<std::string> &particleTypeNames) {`
- L1084: function `saveParticleDataToTxtFileFiltered`: `void saveParticleDataToTxtFileFiltered(const std::string &plotfilename, const std::vector<std::string> &particleTypeNames) {`
- L1096: function `writeCheckpoint`: `void writeCheckpoint(const std::string &checkpointname, bool include_header) const {`
- L1106: function `driftParticlesAllLevels`: `void driftParticlesAllLevels(amrex::Real dt, int lev_max) {`
- L1119: function `kickParticlesAtLevel`: `void kickParticlesAtLevel(int lev, amrex::Real dt, amrex::MultiFab &accel) {`
- L1130: function `createParticlesFromState`: `void createParticlesFromState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0) {`
- L1147: function `destroyParticles`: `void destroyParticles(int lev_min, amrex::Real current_time, amrex::Real dt) {`
- L1174: function `refineGridsAroundParticles`: `void refineGridsAroundParticles(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow, const amrex::IntVect &n_error_buf) {`
- L1186: function `updateParticleProperties`: `void updateParticleProperties(amrex::Real current_time, amrex::Real dt) {`
- L1195: function `printParticleStatistics`: `void printParticleStatistics() const {`
- L1199: function `amrex::Print`: `amrex::Print() << std::format(" {`
- L1207: function `PhysicsParticleRegister`: `PhysicsParticleRegister(const PhysicsParticleRegister &) = delete;`
- L1208: function `operator=`: `auto operator=(const PhysicsParticleRegister &) -> PhysicsParticleRegister & = delete;`
- L1209: function `PhysicsParticleRegister`: `PhysicsParticleRegister(PhysicsParticleRegister &&) = delete;`
- L1210: function `operator=`: `auto operator=(PhysicsParticleRegister &&) -> PhysicsParticleRegister & = delete;`
- L1213: function `updateSFH`: `void updateSFH(int nstep, amrex::Real time) {`
- L1259: function `writeSFHToMetadata`: `void writeSFHToMetadata(YAML::Node &metadata, int sn_count_cumulative) const {`
- L1284: function `readSFH`: `auto readSFH(const YAML::Node &metadata, int &sn_count_cumulative) -> Real {`
- L1326: function `computePhotoelectricHeatingRate`: `auto computePhotoelectricHeatingRate(amrex::Real current_time, quokka::PeHeatingGpuConstTables<quokka::OutOfBounds::clamp> const &gpu_tables, amrex::Real sfh_area_kpc2) -> Real {`

## `src/particles/particle_IO.hpp`

- L19: class `PhysicsParticleRegister`: `template <typename problem_t> class PhysicsParticleRegister;`
- L279: function `writeUnitsFile`: `void writeUnitsFile(ContainerType *container, const std::string &snapshot_name, const std::string &name) {`
- L323: function `printParticleStatistics`: `void printParticleStatistics(ContainerType *container, int massIndex, int evolutionStageIndex) {`
- L328: function `amrex::Print`: `amrex::Print() << std::format("number of {`
- L340: function `amrex::Print`: `amrex::Print() << std::format("\t {`
- L342: function `amrex::Print`: `amrex::Print() << std::format("\t {`
- L350: function `amrex::Print`: `amrex::Print() << std::format("\t {`
- L353: function `amrex::Print`: `amrex::Print() << std::format("\t {`
- L373: function `saveParticleDataToTxtFile`: `template <typename ContainerType> auto saveParticleDataToTxtFile(ContainerType *container, const std::string &filename, const std::string &name) -> bool {`

## `src/particles/particle_accretion.hpp`

- L17: enum class `AccretionScheme`: `enum class AccretionScheme { Threshold = 0, BondiHoyle = 1 };`
- L31: function `get_delta_rho`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_delta_rho(double rho, double rho_sink) -> double {`
- L34: function `compute_Mdot_and_r_K`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto compute_Mdot_and_r_K(const amrex::Array4<const amrex::Real> &local_state, int ix, int iy, int iz, double par_mass, double par_x, double par_y, double par_z, double par_vx, double par_vy, double par_vz, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc = nullptr) -> std::tuple<double, double> {`
- L146: function `compute_accretion_kernel`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto compute_accretion_kernel(const double r_sqr, const double r_K) -> double {`
- L153: function `ComputeAccretionRateInBox`: `void ComputeAccretionRateInBox(const typename ContainerType::ParIterType &pti, const amrex::Array4<const amrex::Real> &local_state, const amrex::Array4<amrex::Real> &local_accretion_rate, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc, amrex::Real , amrex::Real dt, int ) {`
- L253: function `ComputeScaleDown`: `void ComputeScaleDown(amrex::MultiFab &state, amrex::MultiFab &accretion_rate, amrex::MultiFab &scale_down, const amrex::Geometry &geom, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc) {`
- L327: function `UpdateParticleMassAndMomentumInBox`: `void UpdateParticleMassAndMomentumInBox(const typename ContainerType::ParIterType &pti, const amrex::Array4<const amrex::Real> &local_state, const amrex::Array4<const amrex::Real> &local_scale_down, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &plo, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> fab_fc, int mass_index, amrex::Real , amrex::Real dt, amrex::Real , int mdot_index = -1, int ang_mom_index = -1) {`
- L479: function `UpdateParticleMassAndMomentum`: `void UpdateParticleMassAndMomentum(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &scale_down, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, int mass_index, amrex::Real time, amrex::Real dt, int mdot_index = -1, int ang_mom_index = -1) {`
- L509: function `UpdateHydroState`: `template <typename problem_t> void UpdateHydroState(amrex::MultiFab &state, amrex::MultiFab &accretion_rate) {`
- L537: function `computeAccretion`: `void computeAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time, amrex::Real dt, int mass_index) {`
- L567: function `applyAccretion`: `void applyAccretion(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, const amrex::Geometry &geom, int lev, amrex::Real time, amrex::Real dt, int mass_index, int mdot_index = -1, int ang_mom_index = -1) {`

## `src/particles/particle_creation.hpp`

- L23: class `CreatorType`: `template <typename problem_t, typename ContainerType, template <typename> class CheckerType, template <typename> class CreatorType>`
- L24: function `createParticlesImpl`: `static void createParticlesImpl(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1, int death_time_index = -1, int mass_at_birth_index = -1, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0) {`
- L128: struct `ParticleCreationTraits`: `template <ParticleType particleType> struct ParticleCreationTraits {`
- L130: struct `ParticleChecker`: `template <typename problem_t> struct ParticleChecker {`
- L134: function `ParticleChecker`: `AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {`
- L136: function `operator`: `AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, amrex::Array4<const amrex::Real> const &accretion_rate_arr, int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, amrex::RandomEngine const &engine) const -> int {`
- L148: struct `ParticleCreator`: `template <typename problem_t> struct ParticleCreator {`
- L160: function `ParticleCreator`: `ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index, int mass_at_birth_index, amrex::Real current_time, amrex::Real dt) : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index), evolution_stage_index(evolution_stage_index), mass_at_birth_idx(mass_at_birth_index), cpu_id(processor_id), pid_start(particle_id_start), current_time(current_time), dt(dt) {`
- L169: function `operator`: `AMREX_GPU_DEVICE void operator()(ParticleType *particles, int num_particles, StateArray const &state_arr, StateArray const &accretion_rate_arr, int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, amrex::Long base_offset, amrex::RandomEngine const &engine) const {`
- L182: function `createParticles`: `static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1, int death_time_index = -1, int mass_at_birth_index = -1, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0) {`
- L205: function `checkSinkCreation`: `AMREX_GPU_DEVICE inline auto checkSinkCreation(amrex::Array4<const amrex::Real> const &state_arr, amrex::Array4<const amrex::Real> const &accretion_rate_arr, int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc) -> int {`
- L264: function `initializeSinkLikeParticles`: `AMREX_GPU_DEVICE inline void initializeSinkLikeParticles(PType *particles, int num_particles, StateArray const &state_arr, int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, int mass_idx, int cpu_id, amrex::Long pid_start, amrex::Long base_offset) {`
- L327: struct `ParticleCreationTraits`: `template <> struct ParticleCreationTraits<ParticleType::Sink> {`
- L330: struct `ParticleChecker`: `template <typename problem_t> struct ParticleChecker {`
- L334: function `ParticleChecker`: `AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {`
- L336: function `operator`: `AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, amrex::Array4<const amrex::Real> const &accretion_rate_arr, int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, amrex::RandomEngine const & ) const -> int {`
- L347: struct `ParticleCreator`: `template <typename problem_t> struct ParticleCreator {`
- L359: function `ParticleCreator`: `ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index, int mass_at_birth_index, amrex::Real current_time, amrex::Real dt) : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index), evolution_stage_index(evolution_stage_index), mass_at_birth_idx(mass_at_birth_index), cpu_id(processor_id), pid_start(particle_id_start), current_time(current_time), dt(dt) {`
- L381: function `createParticles`: `static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1, int death_time_index = -1, int mass_at_birth_index = -1, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0) {`
- L395: struct `ParticleCreationTraits`: `template <> struct ParticleCreationTraits<ParticleType::StochasticStellarPop> {`
- L421: function `ParticleCreationTraits`: `ParticleCreationTraits() = default;`
- L423: struct `ParticleChecker`: `template <typename problem_t> struct ParticleChecker {`
- L431: function `ParticleChecker`: `AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {`
- L433: function `operator`: `AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const &state_arr, amrex::Array4<const amrex::Real> const & , int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const *fab_fc, amrex::RandomEngine const &engine) const -> int {`
- L474: struct `ParticleCreator`: `template <typename problem_t> struct ParticleCreator {`
- L491: function `ParticleCreator`: `ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index, int mass_at_birth_index, amrex::Real current_time, amrex::Real dt) : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index), evolution_stage_index(evolution_stage_index), mass_at_birth_idx(mass_at_birth_index), cpu_id(processor_id), pid_start(particle_id_start), current_time(current_time), dt(dt) {`
- L737: function `createParticles`: `static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, int evolution_stage_index = -1, int birth_time_index = -1, int death_time_index = -1, int mass_at_birth_index = -1, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0) {`

## `src/particles/particle_deposition.hpp`

- L24: struct `NearestEight`: `struct NearestEight : public Base<NearestEight, amrex::Real> {`
- L34: function `NearestEight`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE NearestEight(const P &p, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) {`
- L65: struct `RadDeposition`: `struct RadDeposition {`
- L74: function `operator`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &radEnergySource, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept {`
- L95: struct `ParticleMassDensityDeposition`: `struct ParticleMassDensityDeposition {`
- L107: function `operator`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &deposition_array, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept {`
- L139: function `depositParticleMassDensity`: `void depositParticleMassDensity(ContainerType *container, amrex::MultiFab &deposition_field, int lev, int mass_comp, int start_mesh_comp = 0, amrex::Real mass_min = std::numeric_limits<amrex::Real>::lowest(), amrex::Real mass_max = std::numeric_limits<amrex::Real>::max(), bool use_age_filter = false, int birth_time_comp = -1, amrex::Real current_time = 0.0, amrex::Real age_max = std::numeric_limits<amrex::Real>::max()) {`
- L189: struct `MassDeposition`: `struct MassDeposition {`
- L197: function `operator`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &rho, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept {`
- L211: struct `DepositionCount`: `struct DepositionCount {`
- L218: function `operator`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ContainerType &p, amrex::Array4<amrex::Real> const &rho_count, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept {`
- L439: function `depositToBuffer`: `void depositToBuffer(ContainerType *container, amrex::MultiFab &state, amrex::MultiFab &state_buffer, int lev, amrex::Real time, amrex::Real dt, int mass_index, int evolutionStageIndex, int birthTimeIndex, const SNScheme SN_scheme_d, int *p_sn_count = nullptr) {`
- L766: function `addBufferToState`: `void addBufferToState(amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, amrex::MultiFab &state_buffer, const SNScheme SN_scheme_d, amrex::Real *p_max_velocity) {`
- L799: function `updateEvolutionStageAndDeathDensity`: `void updateEvolutionStageAndDeathDensity(ContainerType *container, amrex::MultiFab &state, int lev, amrex::Real step_end_time, int birthTimeIndex, int evolutionStageIndex) {`
- L850: function `updateEvolutionStage`: `void updateEvolutionStage(ContainerType *container, int lev_min, amrex::Real step_end_time, int birthTimeIndex, int evolutionStageIndex) {`
- L886: function `SNDeposition`: `auto SNDeposition(ContainerType *container, amrex::MultiFab &state, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc, int lev, amrex::Real time, amrex::Real dt, int mass_index, int evolutionStageIndex, int birthTimeIndex) -> std::pair<int, Real> {`

## `src/particles/particle_destruction.hpp`

- L15: class `CheckerType`: `template <typename problem_t, typename ContainerType, template <typename> class CheckerType>`
- L16: function `destroyParticlesImpl`: `static void destroyParticlesImpl(ContainerType *container, int mass_idx, int lev_min, amrex::Real current_time, amrex::Real dt, int birth_time_index, int evolution_stage_index) {`
- L82: function `amrex::Print`: `amrex::Print() << std::format("[PARTICLES] Particle destruction: Time: {`
- L93: struct `ParticleDestructionTraits`: `template <ParticleType particleType> struct ParticleDestructionTraits {`
- L95: struct `ParticleChecker`: `template <typename problem_t> struct ParticleChecker {`
- L99: function `ParticleChecker`: `AMREX_GPU_HOST_DEVICE explicit ParticleChecker(int birth_time_index, int evolution_stage_index) : birth_time_index(birth_time_index), evolution_stage_index(evolution_stage_index) {`
- L105: function `operator`: `AMREX_GPU_DEVICE auto operator()(ParticleType &p, int mass_idx, amrex::Real current_time, amrex::Real dt) const -> bool {`
- L116: function `destroyParticles`: `static void destroyParticles(ContainerType *container, int mass_idx, int lev_min, amrex::Real current_time, amrex::Real dt, int birth_time_index, int evolution_stage_index) {`

## `src/particles/particle_radiation.hpp`

- L16: struct `LuminosityGpuConstTables`: `template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> struct LuminosityGpuConstTables {`
- L21: class `LuminosityTables`: `template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> class LuminosityTables`
- L41: class `LuminosityUpdate`: `class LuminosityUpdate`
- L45: function `updateLuminosity`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateLuminosity(ParticleType &p, amrex::Real current_time, LuminosityGpuConstTables<Nout, oob_policy> const &gpu_tables) noexcept {`

## `src/particles/particle_types.hpp`

- L11: function `bitflag`: `template <unsigned int position> constexpr auto bitflag() -> unsigned int {`
- L18: enum class `ParticleSwitch`: `enum class ParticleSwitch : unsigned int {`
- L29: function `operator|`: `constexpr auto operator|(ParticleSwitch a, ParticleSwitch b) -> ParticleSwitch {`
- L34: function `operator&`: `constexpr auto operator&(ParticleSwitch flags, ParticleSwitch flag) -> bool {`
- L52: struct `Particle_Traits`: `template <typename problem_t> struct Particle_Traits {`
- L59: function `verify_particle_switch_type`: `template <typename problem_t> constexpr void verify_particle_switch_type() {`
- L72: enum class `ParticleType`: `enum class ParticleType {`
- L83: struct `ParticleTypeTraits`: `template <ParticleType particleType> struct ParticleTypeTraits {`
- L87: struct `ParticleTypeTraits`: `template <> struct ParticleTypeTraits<ParticleType::CIC> {`
- L91: struct `ParticleTypeTraits`: `template <> struct ParticleTypeTraits<ParticleType::CICRad> {`
- L198: enum class `StellarEvolutionStage`: `enum class StellarEvolutionStage { HighMassNonExploding, SNProgenitor, SNRemnant, LowMassComposite, Removed };`
- L347: function `expandEnumNames`: `template <typename EnumType, int nComps, bool expandLast> auto expandEnumNames() -> amrex::Vector<std::string> {`
- L379: function `getParticleRealCompNames`: `template <ParticleType particleType, typename problem_t> auto getParticleRealCompNames() -> amrex::Vector<std::string> {`
- L400: function `getParticleIntCompNames`: `template <ParticleType particleType, typename problem_t> auto getParticleIntCompNames() -> amrex::Vector<std::string> {`
- L427: function `get_units_data`: `inline auto get_units_data() -> const auto & {`
- L533: function `particleParmParse`: `inline void particleParmParse() {`

## `src/particles/particle_update.hpp`

- L15: struct `ParticlePropertyUpdateTraits`: `template <ParticleType particleType> struct ParticlePropertyUpdateTraits;`
- L22: struct `ParticlePropertyUpdateBase`: `template <ParticleType particleType> struct ParticlePropertyUpdateBase {`
- L24: function `updateParticleProperties`: `static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept {`
- L39: function `applyUpdate`: `static void applyUpdate(ContainerType *container, amrex::Real current_time, amrex::Real dt, LuminosityGpuConstTables<Physics_Traits<problem_t>::nGroups> const &gpu_tables) noexcept {`
- L62: struct `ParticlePropertyUpdateTraits`: `template <ParticleType particleType> struct ParticlePropertyUpdateTraits : ParticlePropertyUpdateBase<particleType> {`
- L65: function `updateProperties`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType & , amrex::Real , amrex::Real , LuminosityGpuConstTables<Nout> const & ) noexcept {`
- L73: function `updateParticleProperties`: `static void updateParticleProperties(ContainerType * , amrex::Real , amrex::Real ) noexcept {`
- L81: struct `ParticlePropertyUpdateTraits`: `template <> struct ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop> : ParticlePropertyUpdateBase<ParticleType::StochasticStellarPop> {`
- L83: function `updateParticleProperties`: `static void updateParticleProperties(ContainerType *container, amrex::Real current_time, amrex::Real dt) noexcept {`
- L101: function `updateProperties`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProperties(ParticleType &p, amrex::Real current_time, amrex::Real , LuminosityGpuConstTables<Nout> const &gpu_tables) noexcept {`

## `src/particles/particle_utils.hpp`

- L101: function `computeJeansDensity`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto computeJeansDensity(double cs_cell, double dx, double plasma_beta = std::numeric_limits<double>::max()) -> double {`
- L124: function `computePlasmaBeta`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto computePlasmaBeta(double pressure_thermal, double magnetic_energy) -> double {`
- L132: function `roundoffMultiFab`: `inline void roundoffMultiFab(amrex::MultiFab &mf) {`

## `src/particles/stellarpop_data.hpp`

- L33: function `interpolate_whether_SN_explosion`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_whether_SN_explosion(Real mass_star) -> bool {`
- L68: function `interpolate_death_time`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_death_time(Real mass_star) -> Real {`

## `src/physics_info.hpp`

- L12: enum class `UnitSystem`: `enum class UnitSystem { CGS, CONSTANTS, CUSTOM };`
- L15: struct `Physics_Traits`: `template <typename problem_t> struct Physics_Traits {`
- L37: struct `Physics_Indices`: `template <typename problem_t> struct Physics_Indices {`

## `src/physics_numVars.hpp`

- L6: struct `Physics_NumVars`: `struct Physics_NumVars {`

## `src/problems/Advection/testAdvection.cpp`

- L30: struct `SawtoothProblem`: `struct SawtoothProblem {`
- L33: struct `Physics_Traits`: `template <> struct Physics_Traits<SawtoothProblem> {`
- L48: function `ComputeExactSolution`: `AMREX_GPU_DEVICE void ComputeExactSolution(int i, int j, int k, int n, amrex::Array4<amrex::Real> const &exact_arr, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi) {`
- L59: function `setInitialConditionsOnGrid`: `template <> void AdvectionSimulation<SawtoothProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L74: function `computeReferenceSolution`: `void AdvectionSimulation<SawtoothProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi) {`
- L127: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/AdvectionSemiellipse/testAdvectionSemiellipse.cpp`

- L26: struct `SemiellipseProblem`: `struct SemiellipseProblem {`
- L29: struct `Physics_Traits`: `template <> struct Physics_Traits<SemiellipseProblem> {`
- L44: function `ComputeExactSolution`: `AMREX_GPU_DEVICE void ComputeExactSolution(int i, int j, int k, int n, amrex::Array4<amrex::Real> const &exact_arr, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L55: function `setInitialConditionsOnGrid`: `template <> void AdvectionSimulation<SemiellipseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L68: function `computeReferenceSolution`: `void AdvectionSimulation<SemiellipseProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & ) {`
- L120: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/AlfvenWaveCircular/testAlfvenWaveCircular.cpp`

- L22: struct `AlfvenWaveCircular`: `struct AlfvenWaveCircular {`
- L25: struct `quokka`: `template <> struct quokka::EOS_Traits<AlfvenWaveCircular> {`
- L31: struct `Physics_Traits`: `template <> struct Physics_Traits<AlfvenWaveCircular> {`
- L67: function `computeMagneticVectorPotential_x`: `AMREX_GPU_DEVICE auto computeMagneticVectorPotential_x(double x1, double , double x3, double time) -> double {`
- L72: function `computeMagneticVectorPotential_y`: `AMREX_GPU_DEVICE auto computeMagneticVectorPotential_y(double x1, double , double , double time) -> double {`
- L77: function `computeMagneticVectorPotential_z`: `AMREX_GPU_DEVICE auto computeMagneticVectorPotential_z(double , double x2, double , double ) -> double {`
- L79: function `computeWaveSolution`: `AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, double time) {`
- L146: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<AlfvenWaveCircular>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L166: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<AlfvenWaveCircular>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L187: function `computeReferenceSolution`: `void QuokkaSimulation<AlfvenWaveCircular>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L205: function `computeReferenceSolution_fc`: `void QuokkaSimulation<AlfvenWaveCircular>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L222: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp`

- L30: struct `AlfvenWaveLinear`: `struct AlfvenWaveLinear {`
- L33: struct `quokka`: `template <> struct quokka::EOS_Traits<AlfvenWaveLinear> {`
- L39: struct `Physics_Traits`: `template <> struct Physics_Traits<AlfvenWaveLinear> {`
- L60: function `computeMagnitude`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeMagnitude(const std::array<amrex::Real, 3> &vfield) -> double {`
- L65: function `computeDotProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeDotProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> double {`
- L70: function `computeCrossProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeCrossProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> std::array<amrex::Real, 3> {`
- L77: function `normalizeVector`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void normalizeVector(std::array<amrex::Real, 3> &vfield) {`
- L152: function `rotatePRF2MRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotatePRF2MRF(const std::array<amrex::Real, 3> &vec_prf) -> std::array<amrex::Real, 3> {`
- L164: function `rotateMRF2PRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotateMRF2PRF(const std::array<amrex::Real, 3> &vec_mrf) -> std::array<amrex::Real, 3> {`
- L172: function `computeVectorPotentialComponent_prf`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time, const int icomp) -> double {`
- L200: function `Ax_prf`: `AMREX_GPU_DEVICE inline auto Ax_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L205: function `Ay_prf`: `AMREX_GPU_DEVICE inline auto Ay_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L210: function `Az_prf`: `AMREX_GPU_DEVICE inline auto Az_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L216: function `computeWaveSolution`: `void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, amrex::Real time) {`
- L294: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L312: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L333: function `computeReferenceSolution`: `void QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L352: function `computeReferenceSolution_fc`: `void QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L370: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp`

- L34: struct `AlfvenWaveLinear`: `struct AlfvenWaveLinear {`
- L37: struct `quokka`: `template <> struct quokka::EOS_Traits<AlfvenWaveLinear> {`
- L43: struct `Physics_Traits`: `template <> struct Physics_Traits<AlfvenWaveLinear> {`
- L64: function `computeMagnitude`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeMagnitude(const std::array<amrex::Real, 3> &vfield) -> double {`
- L69: function `computeDotProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeDotProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> double {`
- L74: function `computeCrossProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeCrossProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> std::array<amrex::Real, 3> {`
- L81: function `normalizeVector`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void normalizeVector(std::array<amrex::Real, 3> &vfield) {`
- L156: function `rotatePRF2MRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotatePRF2MRF(const std::array<amrex::Real, 3> &vec_prf) -> std::array<amrex::Real, 3> {`
- L168: function `rotateMRF2PRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotateMRF2PRF(const std::array<amrex::Real, 3> &vec_mrf) -> std::array<amrex::Real, 3> {`
- L176: function `computeVectorPotentialComponent_prf`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time, const int icomp) -> double {`
- L204: function `Ax_prf`: `AMREX_GPU_DEVICE inline auto Ax_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L209: function `Ay_prf`: `AMREX_GPU_DEVICE inline auto Ay_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L214: function `Az_prf`: `AMREX_GPU_DEVICE inline auto Az_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L220: function `computeWaveSolution`: `void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, amrex::Real time) {`
- L298: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L316: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L337: function `computeReferenceSolution`: `void QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L355: function `computeReferenceSolution_fc`: `void QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L372: function `runWaveTest`: `auto runWaveTest(int nx) -> double {`
- L479: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp`

- L32: struct `BinaryOrbit`: `struct BinaryOrbit {`
- L38: struct `quokka`: `template <> struct quokka::EOS_Traits<BinaryOrbit> {`
- L44: struct `Particle_Traits`: `template <> struct Particle_Traits<BinaryOrbit> {`
- L48: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<BinaryOrbit> {`
- L52: struct `Physics_Traits`: `template <> struct Physics_Traits<BinaryOrbit> {`
- L65: struct `SimulationData`: `template <> struct SimulationData<BinaryOrbit> {`
- L70: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<BinaryOrbit>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L86: function `createInitialCICParticles`: `template <> void QuokkaSimulation<BinaryOrbit>::createInitialCICParticles() {`
- L102: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<BinaryOrbit>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const {`
- L113: function `computeAfterTimestep`: `template <> void QuokkaSimulation<BinaryOrbit>::computeAfterTimestep() {`
- L157: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/BrioWuShockTube/testBrioWuShockTube.cpp`

- L31: struct `MHDShocktubeProblem`: `struct MHDShocktubeProblem {`
- L34: struct `quokka`: `template <> struct quokka::EOS_Traits<MHDShocktubeProblem> {`
- L39: struct `Physics_Traits`: `template <> struct Physics_Traits<MHDShocktubeProblem> {`
- L66: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L115: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L209: function `refineGrid`: `template <> void QuokkaSimulation<MHDShocktubeProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, Real , int ) {`
- L235: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/CurrentSheet/testCurrentSheet.cpp`

- L24: struct `CurrentSheet`: `struct CurrentSheet {`
- L27: struct `quokka`: `template <> struct quokka::EOS_Traits<CurrentSheet> {`
- L33: struct `Physics_Traits`: `template <> struct Physics_Traits<CurrentSheet> {`
- L55: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<CurrentSheet>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L80: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<CurrentSheet>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L109: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DiskGalaxy/testDiskGalaxy.cpp`

- L47: struct `DiskGalaxy`: `struct DiskGalaxy {`
- L52: struct `quokka`: `template <> struct quokka::EOS_Traits<DiskGalaxy> {`
- L58: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<DiskGalaxy> {`
- L62: struct `Physics_Traits`: `template <> struct Physics_Traits<DiskGalaxy> {`
- L75: struct `Particle_Traits`: `template <> struct Particle_Traits<DiskGalaxy> {`
- L79: struct `SimulationData`: `template <> struct SimulationData<DiskGalaxy> {`
- L104: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<DiskGalaxy>::preCalculateInitialConditions() {`
- L170: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L455: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L508: function `createInitialCICParticles`: `template <> void QuokkaSimulation<DiskGalaxy>::createInitialCICParticles() {`
- L522: function `refineGrid`: `template <> void QuokkaSimulation<DiskGalaxy>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L567: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<DiskGalaxy>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const {`
- L714: function `ComputeStatistics`: `template <> auto QuokkaSimulation<DiskGalaxy>::ComputeStatistics() -> std::map<std::string, amrex::Real> {`
- L899: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DustAdvection/testDustAdvection.cpp`

- L12: struct `DustAdvection`: `struct DustAdvection {`
- L27: struct `quokka`: `template <> struct quokka::EOS_Traits<DustAdvection> {`
- L32: struct `Physics_Traits`: `template <> struct Physics_Traits<DustAdvection> {`
- L49: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustAdvection>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L91: function `computeReferenceSolution`: `void QuokkaSimulation<DustAdvection>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L231: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DustAdvection3D/testDustAdvection3D.cpp`

- L12: struct `DustAdvection3D`: `struct DustAdvection3D {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<DustAdvection3D> {`
- L35: struct `Physics_Traits`: `template <> struct Physics_Traits<DustAdvection3D> {`
- L52: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustAdvection3D>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L95: function `computeReferenceSolution`: `void QuokkaSimulation<DustAdvection3D>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L344: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DustDamping/testDustDamping.cpp`

- L67: function `v_gas_analytic`: `auto v_gas_analytic(double t) -> double;`
- L68: function `v_dust1_analytic`: `auto v_dust1_analytic(double t) -> double;`
- L69: function `v_dust2_analytic`: `auto v_dust2_analytic(double t) -> double;`
- L70: function `E_gas_analytic`: `auto E_gas_analytic(double t) -> double;`
- L72: struct `DustDamping`: `struct DustDamping {`
- L75: struct `SimulationData`: `template <> struct SimulationData<DustDamping> {`
- L83: struct `quokka`: `template <> struct quokka::EOS_Traits<DustDamping> {`
- L95: struct `Physics_Traits`: `template <> struct Physics_Traits<DustDamping> {`
- L113: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE auto DustDrag<DustDamping>::ComputeReciprocalStoppingTime(amrex::Real , amrex::GpuArray<amrex::Real, nDustGroups_> , amrex::GpuArray<amrex::Real, nDustGroups_> , double ) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L123: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustDamping>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L160: function `computeAfterTimestep`: `template <> void QuokkaSimulation<DustDamping>::computeAfterTimestep() {`
- L196: function `analytic_velocity`: `auto analytic_velocity(double t, double c1, double c2) -> double {`
- L198: function `v_gas_analytic`: `auto v_gas_analytic(double t) -> double {`
- L200: function `v_dust1_analytic`: `auto v_dust1_analytic(double t) -> double {`
- L202: function `v_dust2_analytic`: `auto v_dust2_analytic(double t) -> double {`
- L205: function `E_gas_analytic`: `auto E_gas_analytic(double t) -> double {`
- L236: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DustDampingIteration/testDustDampingIteration.cpp`

- L18: struct `DustDampingWithCorrection`: `struct DustDampingWithCorrection {`
- L21: struct `DustDampingWithoutCorrection`: `struct DustDampingWithoutCorrection {`
- L24: struct `SimulationData`: `template <> struct SimulationData<DustDampingWithCorrection> {`
- L32: struct `SimulationData`: `template <> struct SimulationData<DustDampingWithoutCorrection> {`
- L40: struct `quokka`: `template <> struct quokka::EOS_Traits<DustDampingWithCorrection> {`
- L46: struct `quokka`: `template <> struct quokka::EOS_Traits<DustDampingWithoutCorrection> {`
- L67: struct `Physics_Traits`: `template <> struct Physics_Traits<DustDampingWithCorrection> {`
- L84: struct `Physics_Traits`: `template <> struct Physics_Traits<DustDampingWithoutCorrection> {`
- L102: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE auto DustDrag<DustDampingWithCorrection>::ComputeReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d, amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L111: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE auto DustDrag<DustDampingWithoutCorrection>::ComputeReciprocalStoppingTime(amrex::Real rho_g, amrex::GpuArray<amrex::Real, nDustGroups_> rho_d, amrex::GpuArray<amrex::Real, nDustGroups_> rel_vel_mag, double cs) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L119: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustDampingWithCorrection>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L156: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustDampingWithoutCorrection>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L193: function `computeAfterTimestep`: `template <> void QuokkaSimulation<DustDampingWithCorrection>::computeAfterTimestep() {`
- L228: function `computeAfterTimestep`: `template <> void QuokkaSimulation<DustDampingWithoutCorrection>::computeAfterTimestep() {`
- L263: function `run_reference_simulation`: `auto run_reference_simulation() -> SimulationData<DustDampingWithCorrection> {`
- L306: function `run_iterative_with_correction`: `auto run_iterative_with_correction() -> SimulationData<DustDampingWithCorrection> {`
- L348: function `run_iterative_without_correction`: `auto run_iterative_without_correction() -> SimulationData<DustDampingWithoutCorrection> {`
- L390: function `compute_relative_error`: `auto compute_relative_error(const std::vector<double> &t_test, const std::vector<double> &v_test, const std::vector<double> &t_ref, const std::vector<double> &v_ref) -> double {`
- L433: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DustDampingWithExternalForce/testDustDampingWithExternalForce.cpp`

- L48: function `v_gas_analytic`: `auto v_gas_analytic(double t) -> double;`
- L49: function `v_dust1_analytic`: `auto v_dust1_analytic(double t) -> double;`
- L50: function `v_dust2_analytic`: `auto v_dust2_analytic(double t) -> double;`
- L51: function `E_gas_analytic`: `auto E_gas_analytic(double t) -> double;`
- L53: struct `DustDampingWithExternalForce`: `struct DustDampingWithExternalForce {`
- L56: struct `SimulationData`: `template <> struct SimulationData<DustDampingWithExternalForce> {`
- L64: struct `quokka`: `template <> struct quokka::EOS_Traits<DustDampingWithExternalForce> {`
- L74: struct `Physics_Traits`: `template <> struct Physics_Traits<DustDampingWithExternalForce> {`
- L92: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE auto DustDrag<DustDampingWithExternalForce>::ComputeReciprocalStoppingTime(amrex::Real , amrex::GpuArray<amrex::Real, nDustGroups_> , amrex::GpuArray<amrex::Real, nDustGroups_> , double ) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L103: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustDampingWithExternalForce>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L140: function `computeAfterTimestep`: `template <> void QuokkaSimulation<DustDampingWithExternalForce>::computeAfterTimestep() {`
- L175: function `v_gas_analytic`: `auto v_gas_analytic(double t) -> double {`
- L177: function `v_dust1_analytic`: `auto v_dust1_analytic(double t) -> double {`
- L182: function `v_dust2_analytic`: `auto v_dust2_analytic(double t) -> double {`
- L187: function `E_gas_analytic`: `auto E_gas_analytic(double t) -> double {`
- L223: function `addStrangSplitSources`: `void QuokkaSimulation<DustDampingWithExternalForce>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) {`
- L257: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DustSoundwave/testDustSoundwave.cpp`

- L15: struct `DustSoundwave`: `struct DustSoundwave {`
- L33: function `real_part_analytic`: `auto real_part_analytic(double t, double re, double im) -> double {`
- L43: function `v_gas_analytic`: `auto v_gas_analytic(double t) -> double {`
- L45: function `rho_gas_analytic`: `auto rho_gas_analytic(double t) -> double {`
- L47: function `v_dust_analytic`: `auto v_dust_analytic(double t) -> double {`
- L49: function `rho_dust_analytic`: `auto rho_dust_analytic(double t) -> double {`
- L51: struct `SimulationData`: `template <> struct SimulationData<DustSoundwave> {`
- L60: struct `quokka`: `template <> struct quokka::EOS_Traits<DustSoundwave> {`
- L68: struct `Physics_Traits`: `template <> struct Physics_Traits<DustSoundwave> {`
- L86: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE auto DustDrag<DustSoundwave>::ComputeReciprocalStoppingTime(amrex::Real , amrex::GpuArray<amrex::Real, nDustGroups_> , amrex::GpuArray<amrex::Real, nDustGroups_> , double ) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L95: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustSoundwave>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L158: function `computeAfterTimestep`: `template <> void QuokkaSimulation<DustSoundwave>::computeAfterTimestep() {`
- L178: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/DustyShock/testDustyShock.cpp`

- L28: struct `DustyShock`: `struct DustyShock {`
- L31: struct `quokka`: `template <> struct quokka::EOS_Traits<DustyShock> {`
- L37: struct `Physics_Traits`: `template <> struct Physics_Traits<DustyShock> {`
- L55: function `ComputeReciprocalStoppingTime`: `AMREX_GPU_HOST_DEVICE auto DustDrag<DustyShock>::ComputeReciprocalStoppingTime(amrex::Real , amrex::GpuArray<amrex::Real, nDustGroups_> rho_d, amrex::GpuArray<amrex::Real, nDustGroups_> , double ) -> amrex::GpuArray<amrex::Real, nDustGroups_> {`
- L66: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustyShock>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L107: function `solve_quadratic_root_in_0_1`: `auto solve_quadratic_root_in_0_1(double a, double b, double c) -> double {`
- L135: function `linear_interpolate`: `auto linear_interpolate(const std::vector<double> &x, const std::vector<double> &y, double xi) -> double {`
- L156: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp`

- L37: struct `EntropyWaveLinear`: `struct EntropyWaveLinear {`
- L40: struct `quokka`: `template <> struct quokka::EOS_Traits<EntropyWaveLinear> {`
- L46: struct `Physics_Traits`: `template <> struct Physics_Traits<EntropyWaveLinear> {`
- L68: function `computeMagnitude`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeMagnitude(const std::array<amrex::Real, 3> &vfield) -> double {`
- L73: function `computeDotProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeDotProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> double {`
- L78: function `computeCrossProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeCrossProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> std::array<amrex::Real, 3> {`
- L85: function `normalizeVector`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void normalizeVector(std::array<amrex::Real, 3> &vfield) {`
- L112: function `rotatePRF2MRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotatePRF2MRF(const std::array<amrex::Real, 3> &vec_prf) -> std::array<amrex::Real, 3> {`
- L119: function `rotateMRF2PRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotateMRF2PRF(const std::array<amrex::Real, 3> &vec_mrf) -> std::array<amrex::Real, 3> {`
- L127: function `computeVectorPotentialComponent_prf`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double , const int icomp) -> double {`
- L144: function `Ax_prf`: `AMREX_GPU_DEVICE inline auto Ax_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L150: function `Ay_prf`: `AMREX_GPU_DEVICE inline auto Ay_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L156: function `Az_prf`: `AMREX_GPU_DEVICE inline auto Az_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L163: function `computeWaveSolution`: `void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, amrex::Real time) {`
- L234: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<EntropyWaveLinear>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L252: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<EntropyWaveLinear>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L273: function `computeReferenceSolution`: `void QuokkaSimulation<EntropyWaveLinear>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L291: function `computeReferenceSolution_fc`: `void QuokkaSimulation<EntropyWaveLinear>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L308: function `runWaveTest`: `auto runWaveTest(int nx) -> double {`
- L414: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/FCQuantities/testFCQuantities.cpp`

- L31: struct `FCQuantities`: `struct FCQuantities {`
- L34: struct `quokka`: `template <> struct quokka::EOS_Traits<FCQuantities> {`
- L39: struct `Physics_Traits`: `template <> struct Physics_Traits<FCQuantities> {`
- L59: function `computeWaveSolution`: `AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L83: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<FCQuantities>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L101: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<FCQuantities>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L137: function `setAmrNCell`: `void setAmrNCell(amrex::Vector<int> const &n_cell) {`
- L143: function `setPlotfileParams`: `void setPlotfileParams(std::string const &prefix) {`
- L151: function `checkDivFreeRestart`: `void checkDivFreeRestart(QuokkaSimulation<FCQuantities> const &sim) {`
- L184: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/FastWave/testFastWave.cpp`

- L27: struct `FastWave`: `struct FastWave {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<FastWave> {`
- L36: struct `Physics_Traits`: `template <> struct Physics_Traits<FastWave> {`
- L78: function `computeMagneticVectorPotential_x`: `AMREX_GPU_DEVICE auto computeMagneticVectorPotential_x(double x1, double x2, double , double time) {`
- L82: function `computeMagneticVectorPotential_y`: `AMREX_GPU_DEVICE auto computeMagneticVectorPotential_y(double x1, double , double , double time) -> double {`
- L86: function `computeMagneticVectorPotential_z`: `AMREX_GPU_DEVICE auto computeMagneticVectorPotential_z(double , double , double , double ) -> double {`
- L89: function `computeWaveSolution`: `AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, double time) {`
- L158: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<FastWave>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L178: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<FastWave>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L199: function `computeReferenceSolution`: `void QuokkaSimulation<FastWave>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L218: function `computeReferenceSolution_fc`: `void QuokkaSimulation<FastWave>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L236: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/FastWaveConvergence/testFastWaveConvergence.cpp`

- L28: struct `FastWaveConvergence`: `struct FastWaveConvergence {`
- L31: struct `quokka`: `template <> struct quokka::EOS_Traits<FastWaveConvergence> {`
- L37: struct `Physics_Traits`: `template <> struct Physics_Traits<FastWaveConvergence> {`
- L58: function `computeMagnitude`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeMagnitude(const std::array<amrex::Real, 3> &vfield) -> double {`
- L63: function `computeDotProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeDotProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> double {`
- L68: function `computeCrossProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeCrossProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> std::array<amrex::Real, 3> {`
- L75: function `normalizeVector`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void normalizeVector(std::array<amrex::Real, 3> &vfield) {`
- L150: function `rotatePRF2MRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotatePRF2MRF(const std::array<amrex::Real, 3> &vec_prf) -> std::array<amrex::Real, 3> {`
- L162: function `rotateMRF2PRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotateMRF2PRF(const std::array<amrex::Real, 3> &vec_mrf) -> std::array<amrex::Real, 3> {`
- L170: function `computeVectorPotentialComponent_prf`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time, const int icomp) -> double {`
- L219: function `Ax_prf`: `AMREX_GPU_DEVICE inline auto Ax_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L224: function `Ay_prf`: `AMREX_GPU_DEVICE inline auto Ay_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L229: function `Az_prf`: `AMREX_GPU_DEVICE inline auto Az_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L235: function `computeWaveSolution`: `void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, amrex::Real time) {`
- L350: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<FastWaveConvergence>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L368: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<FastWaveConvergence>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L387: function `computeReferenceSolution`: `void QuokkaSimulation<FastWaveConvergence>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L405: function `computeReferenceSolution_fc`: `void QuokkaSimulation<FastWaveConvergence>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L423: function `runWaveTest`: `auto runWaveTest(int nx) -> double {`
- L532: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/FieldLoop/testFieldLoop.cpp`

- L25: struct `FieldLoop`: `struct FieldLoop {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<FieldLoop> {`
- L36: struct `Physics_Traits`: `template <> struct Physics_Traits<FieldLoop> {`
- L52: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<FieldLoop>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L96: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<FieldLoop>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L128: function `refineGrid`: `template <> void QuokkaSimulation<FieldLoop>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L172: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<FieldLoop>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp) const {`
- L204: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/GravRadParticle3D/testGravRadParticle3D.cpp`

- L24: struct `ParticleProblem`: `struct ParticleProblem {`
- L40: struct `quokka`: `template <> struct quokka::EOS_Traits<ParticleProblem> {`
- L45: struct `Particle_Traits`: `template <> struct Particle_Traits<ParticleProblem> {`
- L49: struct `Physics_Traits`: `template <> struct Physics_Traits<ParticleProblem> {`
- L68: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<ParticleProblem> {`
- L74: function `createInitialCICRadParticles`: `template <> void QuokkaSimulation<ParticleProblem>::createInitialCICRadParticles() {`
- L82: function `createInitialCICParticles`: `template <> void QuokkaSimulation<ParticleProblem>::createInitialCICParticles() {`
- L90: function `createInitialRadParticles`: `template <> void QuokkaSimulation<ParticleProblem>::createInitialRadParticles() {`
- L98: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L103: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L108: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ParticleProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L133: function `checkGasDensityProjection`: `auto checkGasDensityProjection(const QuokkaSimulation<ParticleProblem> &sim) -> int {`
- L173: function `amrex::Print`: `amrex::Print() << std::format( "Projection check FAILED along {`
- L185: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroBlast3D/testHydroBlast3D.cpp`

- L23: struct `SedovProblem`: `struct SedovProblem {`
- L31: struct `quokka`: `template <> struct quokka::EOS_Traits<SedovProblem> {`
- L36: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<SedovProblem> {`
- L40: struct `Physics_Traits`: `template <> struct Physics_Traits<SedovProblem> {`
- L59: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<SedovProblem>::preCalculateInitialConditions() {`
- L66: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SedovProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L107: function `refineGrid`: `template <> void QuokkaSimulation<SedovProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L142: function `computeAfterEvolve`: `template <> void QuokkaSimulation<SedovProblem>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) {`
- L220: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroContact/testHydroContact.cpp`

- L27: struct `ContactProblem`: `struct ContactProblem {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<ContactProblem> {`
- L35: struct `Physics_Traits`: `template <> struct Physics_Traits<ContactProblem> {`
- L52: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ContactProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L95: function `computeReferenceSolution`: `void QuokkaSimulation<ContactProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L192: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroHighMach/testHydroHighMach.cpp`

- L34: struct `HighMachProblem`: `struct HighMachProblem {`
- L37: struct `quokka`: `template <> struct quokka::EOS_Traits<HighMachProblem> {`
- L42: struct `Physics_Traits`: `template <> struct Physics_Traits<HighMachProblem> {`
- L57: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<HighMachProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L93: function `computeReferenceSolution`: `void QuokkaSimulation<HighMachProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<Real, AMREX_SPACEDIM> const & , amrex::GpuArray<Real, AMREX_SPACEDIM> const & ) {`
- L255: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroLeblanc/testHydroLeblanc.cpp`

- L36: struct `ShocktubeProblem`: `struct ShocktubeProblem {`
- L39: struct `quokka`: `template <> struct quokka::EOS_Traits<ShocktubeProblem> {`
- L44: struct `Physics_Traits`: `template <> struct Physics_Traits<ShocktubeProblem> {`
- L59: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L154: function `computeReferenceSolution`: `void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L344: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroQuirk/testHydroQuirk.cpp`

- L36: struct `QuirkProblem`: `struct QuirkProblem {`
- L39: struct `quokka`: `template <> struct quokka::EOS_Traits<QuirkProblem> {`
- L44: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<QuirkProblem> {`
- L48: struct `Physics_Traits`: `template <> struct Physics_Traits<QuirkProblem> {`
- L71: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<QuirkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L130: function `getDeltaEntropyVector`: `auto getDeltaEntropyVector() -> std::vector<Real> & {`
- L136: function `computeAfterTimestep`: `template <> void QuokkaSimulation<QuirkProblem>::computeAfterTimestep() {`
- L189: function `computeAfterEvolve`: `template <> void QuokkaSimulation<QuirkProblem>::computeAfterEvolve(amrex::Vector<amrex::Real> & ) {`
- L239: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroSMS/testHydroSMS.cpp`

- L29: struct `ShocktubeProblem`: `struct ShocktubeProblem {`
- L32: struct `quokka`: `template <> struct quokka::EOS_Traits<ShocktubeProblem> {`
- L37: struct `Physics_Traits`: `template <> struct Physics_Traits<ShocktubeProblem> {`
- L52: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L141: function `computeReferenceSolution`: `void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L263: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroShocktube/testHydroShocktube.cpp`

- L34: struct `ShocktubeProblem`: `struct ShocktubeProblem {`
- L37: struct `quokka`: `template <> struct quokka::EOS_Traits<ShocktubeProblem> {`
- L42: struct `Physics_Traits`: `template <> struct Physics_Traits<ShocktubeProblem> {`
- L63: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L146: function `refineGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, Real , int ) {`
- L173: function `computeReferenceSolution`: `void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L340: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp`

- L32: struct `ShocktubeProblem`: `struct ShocktubeProblem {`
- L37: struct `SimulationData`: `template <> struct SimulationData<ShocktubeProblem> {`
- L42: struct `quokka`: `template <> struct quokka::EOS_Traits<ShocktubeProblem> {`
- L47: struct `Physics_Traits`: `template <> struct Physics_Traits<ShocktubeProblem> {`
- L68: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L179: function `refineGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, Real , int ) {`
- L205: function `computeAfterTimestep`: `template <> void QuokkaSimulation<ShocktubeProblem>::computeAfterTimestep() {`
- L238: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroShuOsher/testHydroShuOsher.cpp`

- L24: struct `ShocktubeProblem`: `struct ShocktubeProblem {`
- L27: struct `quokka`: `template <> struct quokka::EOS_Traits<ShocktubeProblem> {`
- L32: struct `Physics_Traits`: `template <> struct Physics_Traits<ShocktubeProblem> {`
- L47: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L133: function `computeReferenceSolution`: `void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L274: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroVacuum/testHydroVacuum.cpp`

- L31: struct `ShocktubeProblem`: `struct ShocktubeProblem {`
- L34: struct `quokka`: `template <> struct quokka::EOS_Traits<ShocktubeProblem> {`
- L39: struct `Physics_Traits`: `template <> struct Physics_Traits<ShocktubeProblem> {`
- L54: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L143: function `computeReferenceSolution`: `void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L306: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroWave/testHydroWave.cpp`

- L26: struct `WaveProblem`: `struct WaveProblem {`
- L29: struct `quokka`: `template <> struct quokka::EOS_Traits<WaveProblem> {`
- L34: struct `Physics_Traits`: `template <> struct Physics_Traits<WaveProblem> {`
- L54: function `computeWaveSolution`: `AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L78: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<WaveProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L95: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp`

- L24: struct `WaveProblem`: `struct WaveProblem {`
- L27: struct `quokka`: `template <> struct quokka::EOS_Traits<WaveProblem> {`
- L32: struct `Physics_Traits`: `template <> struct Physics_Traits<WaveProblem> {`
- L52: function `computeWaveSolution`: `AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L76: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<WaveProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L93: function `runWaveTest`: `auto runWaveTest(int nx) -> double {`
- L166: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp`

- L20: struct `HydrostaticAtmosphereProblem`: `struct HydrostaticAtmosphereProblem {`
- L23: struct `SimulationData`: `template <> struct SimulationData<HydrostaticAtmosphereProblem> {`
- L27: struct `quokka`: `template <> struct quokka::EOS_Traits<HydrostaticAtmosphereProblem> {`
- L32: struct `Physics_Traits`: `template <> struct Physics_Traits<HydrostaticAtmosphereProblem> {`
- L75: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<HydrostaticAtmosphereProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L105: function `computeReferenceSolution`: `void QuokkaSimulation<HydrostaticAtmosphereProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L182: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp`

- L26: struct `MHDBalsaraVortex`: `struct MHDBalsaraVortex {`
- L29: struct `quokka`: `template <> struct quokka::EOS_Traits<MHDBalsaraVortex> {`
- L35: struct `Physics_Traits`: `template <> struct Physics_Traits<MHDBalsaraVortex> {`
- L64: function `computeRadiusSq`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeRadiusSq(const double x1, const double x2) -> double {`
- L74: function `computeRadialProfile`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeRadialProfile(const double radius_sq) -> double {`
- L76: function `Az`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto Az(const double x1, const double x2) -> double {`
- L89: function `computeVortexSolution`: `void computeVortexSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir) {`
- L151: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L170: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L190: function `computeReferenceSolution`: `void QuokkaSimulation<MHDBalsaraVortex>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L208: function `computeReferenceSolution_fc`: `void QuokkaSimulation<MHDBalsaraVortex>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L225: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp`

- L30: struct `MHDBitwiseICs`: `struct MHDBitwiseICs {`
- L33: struct `quokka`: `template <> struct quokka::EOS_Traits<MHDBitwiseICs> {`
- L39: struct `Physics_Traits`: `template <> struct Physics_Traits<MHDBitwiseICs> {`
- L53: function `computeWaveSolution`: `void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir) {`
- L82: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MHDBitwiseICs>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L100: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<MHDBitwiseICs>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L121: function `computeReferenceSolution`: `void QuokkaSimulation<MHDBitwiseICs>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L139: function `computeReferenceSolution_fc`: `void QuokkaSimulation<MHDBitwiseICs>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L156: function `verifyPeriodicBCs`: `auto verifyPeriodicBCs(const amrex::MultiFab &mf, const std::string &label) -> int {`
- L246: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/MHDBlast/testMHDBlast.cpp`

- L19: struct `MHDBlast`: `struct MHDBlast {`
- L22: struct `quokka`: `template <> struct quokka::EOS_Traits<MHDBlast> {`
- L27: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<MHDBlast> {`
- L31: struct `Physics_Traits`: `template <> struct Physics_Traits<MHDBlast> {`
- L44: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MHDBlast>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L78: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<MHDBlast>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L99: function `refineGrid`: `template <> void QuokkaSimulation<MHDBlast>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L134: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<MHDBlast>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp) const {`
- L166: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/MHDQuirk/testMHDQuirk.cpp`

- L36: struct `MHDQuirk`: `struct MHDQuirk {`
- L39: struct `quokka`: `template <> struct quokka::EOS_Traits<MHDQuirk> {`
- L44: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<MHDQuirk> {`
- L48: struct `Physics_Traits`: `template <> struct Physics_Traits<MHDQuirk> {`
- L71: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<MHDQuirk>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L86: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MHDQuirk>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L145: function `getDeltaEntropyVector`: `auto getDeltaEntropyVector() -> std::vector<Real> & {`
- L151: function `computeAfterTimestep`: `template <> void QuokkaSimulation<MHDQuirk>::computeAfterTimestep() {`
- L206: function `computeAfterEvolve`: `template <> void QuokkaSimulation<MHDQuirk>::computeAfterEvolve(amrex::Vector<amrex::Real> & ) {`
- L256: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/NscbcChannel/testNscbcChannel.cpp`

- L49: struct `Channel`: `struct Channel {`
- L52: struct `quokka`: `template <> struct quokka::EOS_Traits<Channel> {`
- L57: struct `Physics_Traits`: `template <> struct Physics_Traits<Channel> {`
- L81: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<Channel>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L107: function `setCustomBoundaryConditions`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Channel>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int , int , amrex::GeometryData const &geom, const Real , const amrex::BCRec * , int , int ) {`
- L126: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/NscbcVortex/testNscbcVortex.cpp`

- L46: struct `Vortex`: `struct Vortex {`
- L49: struct `quokka`: `template <> struct quokka::EOS_Traits<Vortex> {`
- L54: struct `Physics_Traits`: `template <> struct Physics_Traits<Vortex> {`
- L80: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<Vortex>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L126: function `setCustomBoundaryConditions`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Vortex>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int , int , amrex::GeometryData const &geom, const Real , const amrex::BCRec * , int , int ) {`
- L155: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/ODEIntegration/testODEIntegration.cpp`

- L18: struct `ODETest`: `struct ODETest {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<ODETest> {`
- L35: struct `ODEUserData`: `struct ODEUserData {`
- L39: function `cooling_function`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cooling_function(Real const rho, Real const T) -> Real {`
- L49: struct `ODECoolingFunctor`: `struct ODECoolingFunctor {`
- L52: function `ODECoolingFunctor`: `AMREX_GPU_HOST_DEVICE explicit ODECoolingFunctor(Real rho_in) : rho(rho_in) {`
- L54: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(Real , quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs) const -> int {`
- L66: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/OrszagTang/testOrszagTang.cpp`

- L27: struct `OrszagTang`: `struct OrszagTang {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<OrszagTang> {`
- L36: struct `Physics_Traits`: `template <> struct Physics_Traits<OrszagTang> {`
- L51: function `A_z`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto A_z(double x, double y) -> double {`
- L56: function `B_x`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_x(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double {`
- L61: function `B_y`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_y(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double {`
- L66: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L101: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L123: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/ParticleAccretion/testParticleAccretion.cpp`

- L33: struct `AccretionProblem`: `struct AccretionProblem {`
- L53: struct `Particle_Traits`: `template <> struct Particle_Traits<AccretionProblem> {`
- L58: struct `quokka`: `template <> struct quokka::EOS_Traits<AccretionProblem> {`
- L64: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<AccretionProblem> {`
- L68: struct `Physics_Traits`: `template <> struct Physics_Traits<AccretionProblem> {`
- L83: struct `SimulationData`: `template <> struct SimulationData<AccretionProblem> {`
- L88: function `createInitialSinkParticles`: `template <> void QuokkaSimulation<AccretionProblem>::createInitialSinkParticles() {`
- L135: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L367: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L379: function `refineGrid`: `template <> void QuokkaSimulation<AccretionProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L401: function `computeAfterTimestep`: `template <> void QuokkaSimulation<AccretionProblem>::computeAfterTimestep() {`
- L422: function `problem_main`: `auto problem_main() -> int {`
- L498: function `std::accumulate`: `std::accumulate(real_data_final.begin(), real_data_final.end(), 0.0, [](Real acc, const auto &p) {`

## `src/problems/ParticleCreation/testParticleCreation.cpp`

- L14: struct `TestParticle`: `struct TestParticle {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<TestParticle> {`
- L37: enum class `TestEnum`: `enum class TestEnum : unsigned int {`
- L41: struct `Particle_Traits`: `template <> struct Particle_Traits<TestParticle> {`
- L50: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<TestParticle> {`
- L54: struct `Physics_Traits`: `template <> struct Physics_Traits<TestParticle> {`
- L71: function `createInitialTestParticles`: `template <> void QuokkaSimulation<TestParticle>::createInitialTestParticles() {`
- L113: struct `ParticleCreationTraits`: `template <> struct ParticleCreationTraits<ParticleType::Test> {`
- L115: struct `ParticleChecker`: `template <typename problem_t> struct ParticleChecker {`
- L123: function `ParticleChecker`: `AMREX_GPU_HOST_DEVICE ParticleChecker(amrex::Real current_time, amrex::Real dt) : current_time(current_time), dt(dt) {`
- L125: function `operator`: `AMREX_GPU_DEVICE auto operator()(amrex::Array4<const amrex::Real> const & , amrex::Array4<const amrex::Real> const & , int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const * , amrex::RandomEngine const & ) const -> int {`
- L152: struct `ParticleCreator`: `template <typename problem_t> struct ParticleCreator {`
- L163: function `ParticleCreator`: `ParticleCreator(int mass_index, int birth_time_index, int death_time_index, int processor_id, amrex::Long particle_id_start, int evolution_stage_index, int , amrex::Real current_time, amrex::Real dt) : mass_idx(mass_index), birth_time_index(birth_time_index), death_time_index(death_time_index), evolution_stage_index(evolution_stage_index), cpu_id(processor_id), pid_start(particle_id_start), current_time(current_time), dt(dt) {`
- L221: function `createParticles`: `static void createParticles(ContainerType *container, int mass_idx, amrex::MultiFab &state, amrex::MultiFab &state_accretion_rate, int lev, amrex::Real current_time, amrex::Real dt, int evolution_stage_index, int birth_time_index, int death_time_index, int mass_at_birth_index, std::array<amrex::MultiFab, AMREX_SPACEDIM> const *state_fc = nullptr, int verbose = 0) {`
- L234: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<TestParticle>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L250: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<TestParticle>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L260: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/ParticleDeposition/testParticleDeposition.cpp`

- L19: struct `ParticleDepositionProblem`: `struct ParticleDepositionProblem {`
- L22: struct `Particle_Traits`: `template <> struct Particle_Traits<ParticleDepositionProblem> {`
- L30: struct `quokka`: `template <> struct quokka::EOS_Traits<ParticleDepositionProblem> {`
- L36: struct `Physics_Traits`: `template <> struct Physics_Traits<ParticleDepositionProblem> {`
- L49: struct `SimulationData`: `template <> struct SimulationData<ParticleDepositionProblem> {`
- L54: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ParticleDepositionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L75: function `createInitialCICParticles`: `template <> void QuokkaSimulation<ParticleDepositionProblem>::createInitialCICParticles() {`
- L104: function `createInitialTestParticles`: `template <> void QuokkaSimulation<ParticleDepositionProblem>::createInitialTestParticles() {`
- L127: function `computeAfterTimestep`: `template <> void QuokkaSimulation<ParticleDepositionProblem>::computeAfterTimestep() {`
- L156: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<ParticleDepositionProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ) const {`
- L166: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/ParticleRadiation/testParticleRadiation.cpp`

- L17: struct `ParticleRadiationProblem`: `struct ParticleRadiationProblem {`
- L31: struct `SimulationData`: `template <> struct SimulationData<ParticleRadiationProblem> {`
- L35: struct `quokka`: `template <> struct quokka::EOS_Traits<ParticleRadiationProblem> {`
- L41: enum class `TestEnum`: `enum class TestEnum : unsigned int {`
- L45: struct `Particle_Traits`: `template <> struct Particle_Traits<ParticleRadiationProblem> {`
- L50: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<ParticleRadiationProblem> {`
- L54: struct `Physics_Traits`: `template <> struct Physics_Traits<ParticleRadiationProblem> {`
- L67: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<ParticleRadiationProblem> {`
- L91: function `createInitialStochasticStellarPopParticles`: `template <> void QuokkaSimulation<ParticleRadiationProblem>::createInitialStochasticStellarPopParticles() {`
- L126: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ParticleRadiationProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L153: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/ParticleSF/testParticleSF.cpp`

- L20: struct `ParticleSFProblem`: `struct ParticleSFProblem {`
- L30: struct `Particle_Traits`: `template <> struct Particle_Traits<ParticleSFProblem> {`
- L35: struct `quokka`: `template <> struct quokka::EOS_Traits<ParticleSFProblem> {`
- L40: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<ParticleSFProblem> {`
- L44: struct `Physics_Traits`: `template <> struct Physics_Traits<ParticleSFProblem> {`
- L59: struct `SimulationData`: `template <> struct SimulationData<ParticleSFProblem> {`
- L63: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ParticleSFProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L83: function `refineGrid`: `template <> void QuokkaSimulation<ParticleSFProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L95: function `computeAfterTimestep`: `template <> void QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep() {`
- L231: function `problem_main`: `auto problem_main() -> int {`
- L322: function `amrex::Print`: `amrex::Print() << std::format("Mass of all stars [expected] = {`

## `src/problems/ParticleSink/testParticleSink.cpp`

- L28: struct `SinkProblem`: `struct SinkProblem {`
- L45: struct `Particle_Traits`: `template <> struct Particle_Traits<SinkProblem> {`
- L50: struct `quokka`: `template <> struct quokka::EOS_Traits<SinkProblem> {`
- L55: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<SinkProblem> {`
- L59: struct `Physics_Traits`: `template <> struct Physics_Traits<SinkProblem> {`
- L74: struct `SimulationData`: `template <> struct SimulationData<SinkProblem> {`
- L78: function `createInitialSinkParticles`: `template <> void QuokkaSimulation<SinkProblem>::createInitialSinkParticles() {`
- L109: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L130: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L140: function `refineGrid`: `template <> void QuokkaSimulation<SinkProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L164: function `problem_main`: `auto problem_main() -> int {`
- L254: function `amrex::Print`: `amrex::Print() << std::format("Particle {`
- L307: function `std::abs`: `} else if (std::abs(xs[i]) <= outer_radius) {`
- L395: function `std::sort`: `std::sort(idx1.begin(), idx1.end(), [&](int a, int b) {`
- L401: function `std::sort`: `std::sort(idx2.begin(), idx2.end(), [&](int a, int b) {`
- L412: function `amrex::Print`: `amrex::Print() << std::format("Particle {`
- L420: function `amrex::Print`: `amrex::Print() << std::format("Particle {`
- L423: function `amrex::Print`: `amrex::Print() << std::format("Test failed: angular momentum not Galilean invariant for particle {`
- L445: function `std::abs`: `} else if (std::abs(x) <= outer_radius) {`
- L525: function `amrex::Print`: `amrex::Print() << std::format("Particle {`

## `src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp`

- L25: struct `SinkProblem`: `struct SinkProblem {`
- L38: struct `Particle_Traits`: `template <> struct Particle_Traits<SinkProblem> {`
- L43: struct `quokka`: `template <> struct quokka::EOS_Traits<SinkProblem> {`
- L48: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<SinkProblem> {`
- L52: struct `Physics_Traits`: `template <> struct Physics_Traits<SinkProblem> {`
- L67: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L102: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<SinkProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L112: function `refineGrid`: `template <> void QuokkaSimulation<SinkProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L128: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/PassiveScalar/testPassiveScalar.cpp`

- L29: struct `ScalarProblem`: `struct ScalarProblem {`
- L32: struct `quokka`: `template <> struct quokka::EOS_Traits<ScalarProblem> {`
- L37: struct `Physics_Traits`: `template <> struct Physics_Traits<ScalarProblem> {`
- L54: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ScalarProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L96: function `computeReferenceSolution`: `void QuokkaSimulation<ScalarProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo) {`
- L222: function `refineGrid`: `template <> void QuokkaSimulation<ScalarProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L249: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/PopIII/testPopIII.cpp`

- L38: struct `PopIII`: `struct PopIII {`
- L41: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<PopIII> {`
- L45: struct `Physics_Traits`: `template <> struct Physics_Traits<PopIII> {`
- L60: struct `SimulationData`: `template <> struct SimulationData<PopIII> {`
- L93: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<PopIII>::preCalculateInitialConditions() {`
- L180: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L332: function `refineGrid`: `template <> void QuokkaSimulation<PopIII>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L368: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<PopIII>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const {`
- L428: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/PrimordialChem/testPrimordialChem.cpp`

- L44: struct `PrimordialChemTest`: `struct PrimordialChemTest {`
- L47: struct `Physics_Traits`: `template <> struct Physics_Traits<PrimordialChemTest> {`
- L62: struct `SimulationData`: `template <> struct SimulationData<PrimordialChemTest> {`
- L82: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<PrimordialChemTest>::preCalculateInitialConditions() {`
- L132: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<PrimordialChemTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L249: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadDust/testRadDust.cpp`

- L20: struct `DustProblem`: `struct DustProblem {`
- L42: struct `SimulationData`: `template <> struct SimulationData<DustProblem> {`
- L48: struct `quokka`: `template <> struct quokka::EOS_Traits<DustProblem> {`
- L53: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<DustProblem> {`
- L59: struct `ISM_Traits`: `template <> struct ISM_Traits<DustProblem> {`
- L65: struct `Physics_Traits`: `template <> struct Physics_Traits<DustProblem> {`
- L84: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputePlanckOpacity(const double rho, const double ) -> amrex::Real {`
- L89: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L94: function `ComputeThermalRadiationSingleGroup`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> amrex::Real {`
- L100: function `ComputeThermalRadiationTempDerivativeSingleGroup`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real ) -> amrex::Real {`
- L106: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L129: function `computeAfterTimestep`: `template <> void QuokkaSimulation<DustProblem>::computeAfterTimestep() {`
- L149: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadDustMG/testRadDustMG.cpp`

- L21: struct `DustProblem`: `struct DustProblem {`
- L43: struct `SimulationData`: `template <> struct SimulationData<DustProblem> {`
- L49: struct `quokka`: `template <> struct quokka::EOS_Traits<DustProblem> {`
- L54: struct `Physics_Traits`: `template <> struct Physics_Traits<DustProblem> {`
- L73: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<DustProblem> {`
- L83: struct `ISM_Traits`: `template <> struct ISM_Traits<DustProblem> {`
- L103: function `ComputeThermalRadiationMultiGroup`: `AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const & ) -> quokka::valarray<amrex::Real, nGroups_> {`
- L115: function `ComputeThermalRadiationTempDerivativeMultiGroup`: `AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real , amrex::GpuArray<double, nGroups_ + 1> const & ) -> quokka::valarray<amrex::Real, nGroups_> {`
- L125: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<DustProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L150: function `computeAfterTimestep`: `template <> void QuokkaSimulation<DustProblem>::computeAfterTimestep() {`
- L173: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadForce/testRadForce.cpp`

- L36: struct `TubeProblem`: `struct TubeProblem {`
- L53: struct `quokka`: `template <> struct quokka::EOS_Traits<TubeProblem> {`
- L59: struct `Physics_Traits`: `template <> struct Physics_Traits<TubeProblem> {`
- L75: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<TubeProblem> {`
- L82: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L84: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L89: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<TubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L164: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadLineCooling/testRadLineCooling.cpp`

- L25: struct `CoolingProblem`: `struct CoolingProblem {`
- L47: struct `SimulationData`: `template <> struct SimulationData<CoolingProblem> {`
- L53: struct `quokka`: `template <> struct quokka::EOS_Traits<CoolingProblem> {`
- L58: struct `Physics_Traits`: `template <> struct Physics_Traits<CoolingProblem> {`
- L84: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<CoolingProblem> {`
- L91: struct `ISM_Traits`: `template <> struct ISM_Traits<CoolingProblem> {`
- L99: function `DefineNetCoolingRate`: `AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const ) -> quokka::valarray<double, nGroups_> {`
- L109: function `DefineNetCoolingRateTempDerivative`: `AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRateTempDerivative(amrex::Real const , amrex::Real const ) -> quokka::valarray<double, nGroups_> {`
- L118: function `DefineCosmicRayHeatingRate`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineCosmicRayHeatingRate(amrex::Real const ) -> double {`
- L124: function `ComputePlanckOpacity`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<CoolingProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L129: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L149: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<CoolingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L172: function `computeAfterTimestep`: `template <> void QuokkaSimulation<CoolingProblem>::computeAfterTimestep() {`
- L191: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadLineCoolingMG/testRadLineCoolingMG.cpp`

- L25: struct `CoolingProblemMG`: `struct CoolingProblemMG {`
- L52: struct `SimulationData`: `template <> struct SimulationData<CoolingProblemMG> {`
- L58: struct `quokka`: `template <> struct quokka::EOS_Traits<CoolingProblemMG> {`
- L63: struct `Physics_Traits`: `template <> struct Physics_Traits<CoolingProblemMG> {`
- L82: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<CoolingProblemMG> {`
- L92: struct `ISM_Traits`: `template <> struct ISM_Traits<CoolingProblemMG> {`
- L99: function `DefinePhotoelectricHeatingE1Derivative`: `AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const , amrex::Real const ) -> amrex::Real {`
- L106: function `DefineNetCoolingRate`: `AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const ) -> quokka::valarray<double, nGroups_> {`
- L116: function `DefineNetCoolingRateTempDerivative`: `AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRateTempDerivative(amrex::Real const , amrex::Real const ) -> quokka::valarray<double, nGroups_> {`
- L125: function `DefineCosmicRayHeatingRate`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineCosmicRayHeatingRate(amrex::Real const ) -> double {`
- L145: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<CoolingProblemMG>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L171: function `computeAfterTimestep`: `template <> void QuokkaSimulation<CoolingProblemMG>::computeAfterTimestep() {`
- L190: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMarshak/testRadMarshak.cpp`

- L26: struct `SuOlsonProblem`: `struct SuOlsonProblem {`
- L40: struct `quokka`: `template <> struct quokka::EOS_Traits<SuOlsonProblem> {`
- L45: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<SuOlsonProblem> {`
- L52: struct `Physics_Traits`: `template <> struct Physics_Traits<SuOlsonProblem> {`
- L71: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L76: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L164: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SuOlsonProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L187: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMarshakAsymptotic/testRadMarshakAsymptotic.cpp`

- L23: struct `SuOlsonProblemCgs`: `struct SuOlsonProblemCgs {`
- L36: struct `quokka`: `template <> struct quokka::EOS_Traits<SuOlsonProblemCgs> {`
- L41: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<SuOlsonProblemCgs> {`
- L47: struct `Physics_Traits`: `template <> struct Physics_Traits<SuOlsonProblemCgs> {`
- L62: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L69: function `ComputeFluxMeanOpacity`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L74: function `ComputeEddingtonFactor`: `template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeEddingtonFactor(double ) -> double {`
- L144: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L167: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMarshakCGS/testRadMarshakCGS.cpp`

- L30: struct `SuOlsonProblemCgs`: `struct SuOlsonProblemCgs {`
- L43: struct `quokka`: `template <> struct quokka::EOS_Traits<SuOlsonProblemCgs> {`
- L48: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<SuOlsonProblemCgs> {`
- L54: struct `Physics_Traits`: `template <> struct Physics_Traits<SuOlsonProblemCgs> {`
- L69: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L74: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblemCgs>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L173: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L197: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMarshakDust/testRadMarshakDust.cpp`

- L22: struct `MarshakProblem`: `struct MarshakProblem {`
- L50: struct `quokka`: `template <> struct quokka::EOS_Traits<MarshakProblem> {`
- L55: struct `Physics_Traits`: `template <> struct Physics_Traits<MarshakProblem> {`
- L74: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<MarshakProblem> {`
- L83: struct `ISM_Traits`: `template <> struct ISM_Traits<MarshakProblem> {`
- L90: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L95: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L117: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MarshakProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L178: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp`

- L22: struct `MarshakProblem`: `struct MarshakProblem {`
- L49: struct `quokka`: `template <> struct quokka::EOS_Traits<MarshakProblem> {`
- L54: struct `Physics_Traits`: `template <> struct Physics_Traits<MarshakProblem> {`
- L73: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<MarshakProblem> {`
- L82: struct `ISM_Traits`: `template <> struct ISM_Traits<MarshakProblem> {`
- L89: function `DefinePhotoelectricHeatingE1Derivative`: `AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const , amrex::Real const ) -> amrex::Real {`
- L119: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MarshakProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L182: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMarshakVaytet/testRadMarshakVaytet.cpp`

- L89: struct `SuOlsonProblemCgs`: `struct SuOlsonProblemCgs {`
- L104: struct `quokka`: `template <> struct quokka::EOS_Traits<SuOlsonProblemCgs> {`
- L109: struct `Physics_Traits`: `template <> struct Physics_Traits<SuOlsonProblemCgs> {`
- L124: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<SuOlsonProblemCgs> {`
- L221: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L249: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMatterCoupling/testRadMatterCoupling.cpp`

- L23: struct `CouplingProblem`: `struct CouplingProblem {`
- L31: struct `SimulationData`: `template <> struct SimulationData<CouplingProblem> {`
- L37: struct `quokka`: `template <> struct quokka::EOS_Traits<CouplingProblem> {`
- L42: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<CouplingProblem> {`
- L48: struct `Physics_Traits`: `template <> struct Physics_Traits<CouplingProblem> {`
- L63: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L68: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L110: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<CouplingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L130: function `computeAfterTimestep`: `template <> void QuokkaSimulation<CouplingProblem>::computeAfterTimestep() {`
- L151: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadMatterCouplingRSLA/testRadMatterCouplingRSLA.cpp`

- L23: struct `CouplingProblem`: `struct CouplingProblem {`
- L34: struct `SimulationData`: `template <> struct SimulationData<CouplingProblem> {`
- L40: struct `quokka`: `template <> struct quokka::EOS_Traits<CouplingProblem> {`
- L45: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<CouplingProblem> {`
- L51: struct `Physics_Traits`: `template <> struct Physics_Traits<CouplingProblem> {`
- L66: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L71: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L113: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<CouplingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L133: function `computeAfterTimestep`: `template <> void QuokkaSimulation<CouplingProblem>::computeAfterTimestep() {`
- L154: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadStreaming/testRadStreaming.cpp`

- L22: struct `StreamingProblem`: `struct StreamingProblem {`
- L32: struct `quokka`: `template <> struct quokka::EOS_Traits<StreamingProblem> {`
- L37: struct `Physics_Traits`: `template <> struct Physics_Traits<StreamingProblem> {`
- L56: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<StreamingProblem> {`
- L62: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L67: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L72: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L157: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadStreamingY/testRadStreamingY.cpp`

- L22: struct `StreamingProblem`: `struct StreamingProblem {`
- L32: struct `quokka`: `template <> struct quokka::EOS_Traits<StreamingProblem> {`
- L37: struct `Physics_Traits`: `template <> struct Physics_Traits<StreamingProblem> {`
- L56: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<StreamingProblem> {`
- L62: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L67: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L72: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L143: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadSuOlson/testRadSuOlson.cpp`

- L34: struct `MarshakProblem`: `struct MarshakProblem {`
- L52: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<MarshakProblem> {`
- L60: struct `quokka`: `template <> struct quokka::EOS_Traits<MarshakProblem> {`
- L65: struct `Physics_Traits`: `template <> struct Physics_Traits<MarshakProblem> {`
- L85: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputePlanckOpacity(const double rho, const double ) -> amrex::Real {`
- L90: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeFluxMeanOpacity(const double rho, const double ) -> amrex::Real {`
- L132: function `SetRadEnergySource`: `void RadSystem<MarshakProblem>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & , amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & , amrex::Real time) {`
- L158: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MarshakProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L181: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadTube/testRadTube.cpp`

- L33: struct `TubeProblem`: `struct TubeProblem {`
- L48: struct `quokka`: `template <> struct quokka::EOS_Traits<TubeProblem> {`
- L53: struct `Physics_Traits`: `template <> struct Physics_Traits<TubeProblem> {`
- L69: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<TubeProblem> {`
- L94: struct `SimulationData`: `template <> struct SimulationData<TubeProblem> {`
- L101: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<TubeProblem>::preCalculateInitialConditions() {`
- L145: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<TubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L251: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroBB/testRadhydroBB.cpp`

- L30: struct `PulseProblem`: `struct PulseProblem {`
- L109: struct `quokka`: `template <> struct quokka::EOS_Traits<PulseProblem> {`
- L114: struct `Physics_Traits`: `template <> struct Physics_Traits<PulseProblem> {`
- L133: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<PulseProblem> {`
- L158: function `compute_exact_bb`: `auto compute_exact_bb(const double nu, const double T) -> double {`
- L166: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L191: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroPulse/testRadhydroPulse.cpp`

- L19: struct `PulseProblem`: `struct PulseProblem {`
- L21: struct `AdvPulseProblem`: `struct AdvPulseProblem {`
- L42: struct `quokka`: `template <> struct quokka::EOS_Traits<PulseProblem> {`
- L46: struct `quokka`: `template <> struct quokka::EOS_Traits<AdvPulseProblem> {`
- L51: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<PulseProblem> {`
- L56: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<AdvPulseProblem> {`
- L62: struct `Physics_Traits`: `template <> struct Physics_Traits<PulseProblem> {`
- L76: struct `Physics_Traits`: `template <> struct Physics_Traits<AdvPulseProblem> {`
- L92: function `compute_initial_Tgas`: `auto compute_initial_Tgas(const double x) -> double {`
- L100: function `compute_exact_rho`: `auto compute_exact_rho(const double x) -> double {`
- L107: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L111: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L116: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L120: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L125: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L156: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<AdvPulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L197: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroPulseDyn/testRadhydroPulseDyn.cpp`

- L19: struct `PulseProblem`: `struct PulseProblem {`
- L21: struct `AdvPulseProblem`: `struct AdvPulseProblem {`
- L42: struct `quokka`: `template <> struct quokka::EOS_Traits<PulseProblem> {`
- L46: struct `quokka`: `template <> struct quokka::EOS_Traits<AdvPulseProblem> {`
- L51: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<PulseProblem> {`
- L56: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<AdvPulseProblem> {`
- L62: struct `Physics_Traits`: `template <> struct Physics_Traits<PulseProblem> {`
- L76: struct `Physics_Traits`: `template <> struct Physics_Traits<AdvPulseProblem> {`
- L92: function `compute_initial_Tgas`: `auto compute_initial_Tgas(const double x) -> double {`
- L100: function `compute_exact_rho`: `auto compute_exact_rho(const double x) -> double {`
- L107: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L111: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L116: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L120: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L125: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L156: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<AdvPulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L197: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp`

- L19: struct `PulseProblem`: `struct PulseProblem {`
- L21: struct `AdvPulseProblem`: `struct AdvPulseProblem {`
- L43: struct `quokka`: `template <> struct quokka::EOS_Traits<PulseProblem> {`
- L47: struct `quokka`: `template <> struct quokka::EOS_Traits<AdvPulseProblem> {`
- L52: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<PulseProblem> {`
- L57: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<AdvPulseProblem> {`
- L63: struct `Physics_Traits`: `template <> struct Physics_Traits<PulseProblem> {`
- L77: struct `Physics_Traits`: `template <> struct Physics_Traits<AdvPulseProblem> {`
- L93: function `compute_initial_Tgas`: `auto compute_initial_Tgas(const double x) -> double {`
- L101: function `compute_exact_rho`: `auto compute_exact_rho(const double x) -> double {`
- L108: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L113: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L119: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L124: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L130: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L161: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<AdvPulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L194: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroPulseMGconst/testRadhydroPulseMGconst.cpp`

- L21: struct `SGProblem`: `struct SGProblem {`
- L24: struct `MGproblem`: `struct MGproblem {`
- L76: function `compute_initial_Tgas`: `auto compute_initial_Tgas(const double x) -> double {`
- L84: function `compute_exact_rho`: `auto compute_exact_rho(const double x) -> double {`
- L91: struct `quokka`: `template <> struct quokka::EOS_Traits<SGProblem> {`
- L96: struct `Physics_Traits`: `template <> struct Physics_Traits<SGProblem> {`
- L111: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<SGProblem> {`
- L117: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SGProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L119: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SGProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L124: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SGProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L157: struct `quokka`: `template <> struct quokka::EOS_Traits<MGproblem> {`
- L162: struct `Physics_Traits`: `template <> struct Physics_Traits<MGproblem> {`
- L177: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<MGproblem> {`
- L203: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MGproblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L241: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp`

- L22: struct `MGProblem`: `struct MGProblem {`
- L24: struct `ExactProblem`: `struct ExactProblem {`
- L97: struct `quokka`: `template <> struct quokka::EOS_Traits<MGProblem> {`
- L101: struct `quokka`: `template <> struct quokka::EOS_Traits<ExactProblem> {`
- L106: struct `Physics_Traits`: `template <> struct Physics_Traits<MGProblem> {`
- L120: struct `Physics_Traits`: `template <> struct Physics_Traits<ExactProblem> {`
- L135: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<MGProblem> {`
- L144: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<ExactProblem> {`
- L153: function `compute_initial_Tgas`: `auto compute_initial_Tgas(const double x) -> double {`
- L161: function `compute_exact_rho`: `auto compute_exact_rho(const double x) -> double {`
- L169: function `compute_kappa`: `auto compute_kappa(const double nu, const double Tgas) -> double {`
- L212: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ExactProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L218: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ExactProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L224: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<MGProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L265: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ExactProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L299: function `problem_main`: `auto problem_main() -> int {`
- L511: function `matplotlibcpp::grid`: `matplotlibcpp::grid(true);`

## `src/problems/RadhydroShell/testRadhydroShell.cpp`

- L29: struct `ShellProblem`: `struct ShellProblem {`
- L42: struct `quokka`: `template <> struct quokka::EOS_Traits<ShellProblem> {`
- L47: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<ShellProblem> {`
- L53: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<ShellProblem> {`
- L57: struct `Physics_Traits`: `template <> struct Physics_Traits<ShellProblem> {`
- L90: function `SetRadEnergySource`: `void RadSystem<ShellProblem>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real ) {`
- L121: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShellProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L127: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShellProblem>::ComputeFluxMeanOpacity(const double , const double ) -> amrex::Real {`
- L132: struct `SimulationData`: `template <> struct SimulationData<ShellProblem> {`
- L143: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<ShellProblem>::preCalculateInitialConditions() {`
- L181: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShellProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L255: function `refineGrid`: `template <> void QuokkaSimulation<ShellProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L293: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroShock/testRadhydroShock.cpp`

- L28: struct `ShockProblem`: `struct ShockProblem {`
- L67: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<ShockProblem> {`
- L73: struct `quokka`: `template <> struct quokka::EOS_Traits<ShockProblem> {`
- L78: struct `Physics_Traits`: `template <> struct Physics_Traits<ShockProblem> {`
- L97: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputePlanckOpacity(const double rho, const double ) -> amrex::Real {`
- L102: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShockProblem>::ComputeFluxMeanOpacity(const double rho, const double ) -> amrex::Real {`
- L107: function `ComputeEddingtonFactor`: `template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double ) -> double {`
- L164: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShockProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L215: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroShockCGS/testRadhydroShockCGS.cpp`

- L29: struct `ShockProblem`: `struct ShockProblem {`
- L67: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<ShockProblem> {`
- L73: struct `quokka`: `template <> struct quokka::EOS_Traits<ShockProblem> {`
- L78: struct `Physics_Traits`: `template <> struct Physics_Traits<ShockProblem> {`
- L98: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputePlanckOpacity(const double rho, const double ) -> amrex::Real {`
- L103: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShockProblem>::ComputeFluxMeanOpacity(const double rho, const double ) -> amrex::Real {`
- L108: function `ComputeEddingtonFactor`: `template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double ) -> double {`
- L172: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShockProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L232: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroShockMultigroup/testRadhydroShockMultigroup.cpp`

- L25: struct `ShockProblem`: `struct ShockProblem {`
- L56: struct `Physics_Traits`: `template <> struct Physics_Traits<ShockProblem> {`
- L71: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<ShockProblem> {`
- L83: struct `quokka`: `template <> struct quokka::EOS_Traits<ShockProblem> {`
- L101: function `ComputeEddingtonFactor`: `template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double ) -> double {`
- L170: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShockProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L222: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RadhydroUniformAdvecting/testRadhydroUniformAdvecting.cpp`

- L19: struct `PulseProblem`: `struct PulseProblem {`
- L60: struct `quokka`: `template <> struct quokka::EOS_Traits<PulseProblem> {`
- L65: struct `RadSystem_Traits`: `template <> struct RadSystem_Traits<PulseProblem> {`
- L71: struct `Physics_Traits`: `template <> struct Physics_Traits<PulseProblem> {`
- L90: function `ComputePlanckOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double , const double ) -> amrex::Real {`
- L95: function `ComputeFluxMeanOpacity`: `template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real {`
- L100: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L139: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RandomBlast/testRandomBlast.cpp`

- L24: struct `RandomBlast`: `struct RandomBlast {`
- L30: struct `Physics_Traits`: `template <> struct Physics_Traits<RandomBlast> {`
- L43: struct `quokka`: `template <> struct quokka::EOS_Traits<RandomBlast> {`
- L48: struct `Particle_Traits`: `template <> struct Particle_Traits<RandomBlast> {`
- L54: struct `SimulationData`: `template <> struct SimulationData<RandomBlast> {`
- L64: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<RandomBlast>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L96: function `createInitialStochasticStellarPopParticles`: `template <> void QuokkaSimulation<RandomBlast>::createInitialStochasticStellarPopParticles() {`
- L132: function `computeAfterTimestep`: `template <> void QuokkaSimulation<RandomBlast>::computeAfterTimestep() {`
- L138: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<RandomBlast>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const {`
- L166: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/RayleighTaylor3D/testRayleighTaylor3D.cpp`

- L26: struct `RTProblem`: `struct RTProblem {`
- L29: struct `quokka`: `template <> struct quokka::EOS_Traits<RTProblem> {`
- L34: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<RTProblem> {`
- L38: struct `Physics_Traits`: `template <> struct Physics_Traits<RTProblem> {`
- L57: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<RTProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L109: function `addStrangSplitSources`: `void QuokkaSimulation<RTProblem>::addStrangSplitSources(amrex::MultiFab &state_mf, const int , const amrex::Real , const amrex::Real dt) {`
- L140: function `refineGrid`: `template <> void QuokkaSimulation<RTProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L173: function `computeAfterTimestep`: `template <> void QuokkaSimulation<RTProblem>::computeAfterTimestep() {`
- L200: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp`

- L31: struct `ResampledCoolingTest`: `struct ResampledCoolingTest {`
- L35: function `readReferenceCSV`: `auto readReferenceCSV(const std::string &filename) -> std::pair<std::vector<double>, std::vector<double>> {`
- L82: struct `SimulationData`: `template <> struct SimulationData<ResampledCoolingTest> {`
- L87: struct `quokka`: `template <> struct quokka::EOS_Traits<ResampledCoolingTest> {`
- L92: struct `Physics_Traits`: `template <> struct Physics_Traits<ResampledCoolingTest> {`
- L109: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ResampledCoolingTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L136: function `computeAfterTimestep`: `template <> void QuokkaSimulation<ResampledCoolingTest>::computeAfterTimestep() {`
- L164: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/SN/testSN.cpp`

- L29: struct `SNProblem`: `struct SNProblem {`
- L51: struct `Particle_Traits`: `template <> struct Particle_Traits<SNProblem> {`
- L56: struct `quokka`: `template <> struct quokka::EOS_Traits<SNProblem> {`
- L61: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<SNProblem> {`
- L65: struct `Physics_Traits`: `template <> struct Physics_Traits<SNProblem> {`
- L80: struct `SimulationData`: `template <> struct SimulationData<SNProblem> {`
- L84: function `createInitialTestParticles`: `template <> void QuokkaSimulation<SNProblem>::createInitialTestParticles() {`
- L116: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SNProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L174: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<SNProblem>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L184: function `refineGrid`: `template <> void QuokkaSimulation<SNProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L208: function `computeAfterTimestep`: `template <> void QuokkaSimulation<SNProblem>::computeAfterTimestep() {`
- L217: function `problem_main`: `auto problem_main() -> int {`
- L380: function `amrex::Print`: `amrex::Print() << std::format("Relative L1 norm for vx = {`
- L381: function `amrex::Print`: `amrex::Print() << std::format("Relative L1 norm for T = {`

## `src/problems/ShockCloud/testShockCloud.cpp`

- L40: struct `ShockCloud`: `struct ShockCloud {`
- L49: struct `Physics_Traits`: `template <> struct Physics_Traits<ShockCloud> {`
- L62: struct `quokka`: `template <> struct quokka::EOS_Traits<ShockCloud> {`
- L86: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<ShockCloud>::setInitialConditionsOnGrid(quokka::grid const &grid) {`
- L148: function `setCustomBoundaryConditions`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<ShockCloud>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int , int , amrex::GeometryData const &geom, const Real time, const amrex::BCRec * , int , int ) {`
- L207: function `computeAfterTimestep`: `template <> void QuokkaSimulation<ShockCloud>::computeAfterTimestep() {`
- L275: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<ShockCloud>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in) const {`
- L483: function `ComputeCellTempResampled`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto ComputeCellTempResampled(int i, int j, int k, amrex::Array4<const Real> const &state, amrex::Real , quokka::ResampledCooling::resampledGpuConstTables const &tables) {`
- L495: function `ComputeStatistics`: `template <> auto QuokkaSimulation<ShockCloud>::ComputeStatistics() -> std::map<std::string, amrex::Real> {`
- L658: function `refineGrid`: `template <> void QuokkaSimulation<ShockCloud>::refineGrid(int lev, amrex::TagBoxArray &tags, Real , int ) {`
- L688: function `problem_main`: `auto problem_main() -> int {`
- L736: function `amrex::Print`: `amrex::Print() << std::format("Pressure = {`
- L755: function `amrex::Print`: `amrex::Print() << std::format("T_bg = {`
- L756: function `amrex::Print`: `amrex::Print() << std::format("T_cl = {`
- L771: function `amrex::Print`: `amrex::Print() << std::format("T_wind = {`
- L776: function `amrex::Print`: `amrex::Print() << std::format("v_wind = {`
- L777: function `amrex::Print`: `amrex::Print() << std::format("P_wind = {`
- L778: function `amrex::Print`: `amrex::Print() << std::format("v_shock = {`
- L782: function `amrex::Print`: `amrex::Print() << std::format("shock crossing time = {`
- L787: function `amrex::Print`: `amrex::Print() << std::format("t_cc = {`

## `src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp`

- L28: struct `SlowWaveConvergence`: `struct SlowWaveConvergence {`
- L31: struct `quokka`: `template <> struct quokka::EOS_Traits<SlowWaveConvergence> {`
- L37: struct `Physics_Traits`: `template <> struct Physics_Traits<SlowWaveConvergence> {`
- L58: function `computeMagnitude`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeMagnitude(const std::array<amrex::Real, 3> &vfield) -> double {`
- L63: function `computeDotProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeDotProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> double {`
- L68: function `computeCrossProduct`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto computeCrossProduct(const std::array<amrex::Real, 3> &vfield1, const std::array<amrex::Real, 3> &vfield2) -> std::array<amrex::Real, 3> {`
- L75: function `normalizeVector`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void normalizeVector(std::array<amrex::Real, 3> &vfield) {`
- L150: function `rotatePRF2MRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotatePRF2MRF(const std::array<amrex::Real, 3> &vec_prf) -> std::array<amrex::Real, 3> {`
- L162: function `rotateMRF2PRF`: `AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto rotateMRF2PRF(const std::array<amrex::Real, 3> &vec_mrf) -> std::array<amrex::Real, 3> {`
- L170: function `computeVectorPotentialComponent_prf`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeVectorPotentialComponent_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time, const int icomp) -> double {`
- L220: function `Ax_prf`: `AMREX_GPU_DEVICE inline auto Ax_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L225: function `Ay_prf`: `AMREX_GPU_DEVICE inline auto Ay_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L230: function `Az_prf`: `AMREX_GPU_DEVICE inline auto Az_prf(const double x1_prf, const double x2_prf, const double x3_prf, const double time) -> double {`
- L236: function `computeWaveSolution`: `void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::centering cen, quokka::direction dir, amrex::Real time) {`
- L287: function `std::abs`: `} else if (std::abs(cosθ) < tiny) {`
- L359: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<SlowWaveConvergence>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L377: function `setInitialConditionsOnGridFaceVars`: `template <> void QuokkaSimulation<SlowWaveConvergence>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) {`
- L396: function `computeReferenceSolution`: `void QuokkaSimulation<SlowWaveConvergence>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo) {`
- L414: function `computeReferenceSolution_fc`: `void QuokkaSimulation<SlowWaveConvergence>::computeReferenceSolution_fc(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, quokka::direction const dir) {`
- L432: function `runWaveTest`: `auto runWaveTest(int nx) -> double {`
- L541: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/SphericalCollapse/testSphericalCollapse.cpp`

- L17: struct `GlobalConfig`: `struct GlobalConfig {`
- L26: struct `CollapseProblem`: `struct CollapseProblem {`
- L29: struct `quokka`: `template <> struct quokka::EOS_Traits<CollapseProblem> {`
- L34: struct `Particle_Traits`: `template <> struct Particle_Traits<CollapseProblem> {`
- L38: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<CollapseProblem> {`
- L42: struct `Physics_Traits`: `template <> struct Physics_Traits<CollapseProblem> {`
- L59: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<CollapseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L97: function `createInitialCICParticles`: `template <> void QuokkaSimulation<CollapseProblem>::createInitialCICParticles() {`
- L110: function `refineGrid`: `template <> void QuokkaSimulation<CollapseProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L130: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<CollapseProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const {`
- L141: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/StarCluster/testStarCluster.cpp`

- L36: struct `StarCluster`: `struct StarCluster {`
- L39: struct `quokka`: `template <> struct quokka::EOS_Traits<StarCluster> {`
- L45: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<StarCluster> {`
- L49: struct `Physics_Traits`: `template <> struct Physics_Traits<StarCluster> {`
- L66: struct `SimulationData`: `template <> struct SimulationData<StarCluster> {`
- L81: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<StarCluster>::preCalculateInitialConditions() {`
- L125: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<StarCluster>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L175: function `refineGrid`: `template <> void QuokkaSimulation<StarCluster>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L196: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<StarCluster>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const {`
- L211: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/TallBoxSf/testTallBoxSf.cpp`

- L30: struct `TheProblem`: `struct TheProblem {`
- L33: struct `SimulationData`: `template <> struct SimulationData<TheProblem> {`
- L59: struct `Particle_Traits`: `template <> struct Particle_Traits<TheProblem> {`
- L63: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<TheProblem> {`
- L67: struct `quokka`: `template <> struct quokka::EOS_Traits<TheProblem> {`
- L72: struct `Physics_Traits`: `template <> struct Physics_Traits<TheProblem> {`
- L86: function `createInitialStochasticStellarPopParticles`: `template <> void QuokkaSimulation<TheProblem>::createInitialStochasticStellarPopParticles() {`
- L125: function `refineGrid`: `template <> void QuokkaSimulation<TheProblem>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L154: function `preCalculateInitialConditions`: `template <> void QuokkaSimulation<TheProblem>::preCalculateInitialConditions() {`
- L209: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L300: function `ComputeDerivedVar`: `template <> void QuokkaSimulation<TheProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in) const {`
- L375: function `addStrangSplitSources`: `template <> void QuokkaSimulation<TheProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev) {`
- L449: function `setCustomBoundaryConditions`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<TheProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar, int , int , amrex::GeometryData const &geom, const Real , const amrex::BCRec * , int , int ) {`
- L459: function `problem_main`: `auto problem_main() -> int {`

## `src/problems/Turbulence/testTurbulence.cpp`

- L20: struct `TurbulentBox`: `struct TurbulentBox {`
- L23: struct `Physics_Traits`: `template <> struct Physics_Traits<TurbulentBox> {`
- L38: struct `quokka`: `template <> struct quokka::EOS_Traits<TurbulentBox> {`
- L44: struct `HydroSystem_Traits`: `template <> struct HydroSystem_Traits<TurbulentBox> {`
- L48: struct `SimulationData`: `template <> struct SimulationData<TurbulentBox> {`
- L53: function `setInitialConditionsOnGrid`: `template <> void QuokkaSimulation<TurbulentBox>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {`
- L70: function `refineGrid`: `template <> void QuokkaSimulation<TurbulentBox>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real , int ) {`
- L103: function `computeAfterTimestep`: `template <> void QuokkaSimulation<TurbulentBox>::computeAfterTimestep() {`
- L114: function `problem_main`: `auto problem_main() -> int {`

## `src/radiation/planck_integral.hpp`

- L29: function `interpolate_planck_integral`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_planck_integral(Real logx) -> Real {`
- L233: function `integrate_planck_from_0_to_x`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto integrate_planck_from_0_to_x(const Real x) -> Real {`

## `src/radiation/radiation_dust_system.hpp`

- L8: function `DefinePhotoelectricHeatingE1Derivative`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const , amrex::Real const ) -> amrex::Real {`
- L23: function `ComputeJacobianForGasAndDust`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDust( double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double , quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, const double num_den, const double dt) -> JacobianResult<problem_t> {`
- L86: function `ComputeJacobianForGasAndDustDecoupled`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustDecoupled( double , double , double , quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double , quokka::valarray<double, nGroups_> const &tau, double , double lambda_gd_time_dt, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t) -> JacobianResult<problem_t> {`
- L131: function `ComputeJacobianForGasAndDustWithPE`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustWithPE( double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad, quokka::valarray<double, nGroups_> const &Erad0, double PE_heating_energy_derivative, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double , quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double const num_den, double const dt) -> JacobianResult<problem_t> {`
- L199: function `SolveLinearEqsWithLastColumn`: `AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqsWithLastColumn(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi) {`
- L599: function `SolveGasDustRadiationEnergyExchangeWithPE`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasDustRadiationEnergyExchangeWithPE( double const Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double const rho, double const coeff_n, double const dt, amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work, quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, double resid_tol, double rel_change_tol, double , int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t> {`

## `src/radiation/radiation_system.hpp`

- L56: enum class `OpacityModel`: `enum class OpacityModel {`
- L66: struct `RadSystem_Traits`: `template <typename problem_t> struct RadSystem_Traits {`
- L77: struct `ISM_Traits`: `template <typename problem_t> struct ISM_Traits {`
- L84: struct `RadPressureResult`: `struct RadPressureResult {`
- L91: struct `OpacityTerms`: `template <typename problem_t> struct OpacityTerms {`
- L103: struct `NewtonIterationResult`: `template <typename problem_t> struct NewtonIterationResult {`
- L114: struct `JacobianResult`: `template <typename problem_t> struct JacobianResult {`
- L127: struct `FluxUpdateResult`: `template <typename problem_t> struct FluxUpdateResult {`
- L139: struct `RadSystem_Has_Opacity_Model`: `template <typename problem_t, typename = void> struct RadSystem_Has_Opacity_Model : std::false_type {`
- L143: struct `RadSystem_Has_Opacity_Model`: `struct RadSystem_Has_Opacity_Model<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::opacity_model)>> : std::true_type {`
- L148: class `RadSystem`: `template <typename problem_t> class RadSystem : public HyperbolicSystem<problem_t>`
- L162: enum `gasVarIndex`: `enum gasVarIndex {`
- L172: enum `radVarIndex`: `enum radVarIndex { radEnergy_index = nstartHyperbolic_, x1RadFlux_index, x2RadFlux_index, x3RadFlux_index };`
- L174: enum `primVarIndex`: `enum primVarIndex {`
- L261: function `ComputeMaxSignalSpeed`: `static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange);`
- L262: function `ConservedToPrimitive`: `static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange);`
- L264: function `PredictStep`: `static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);`
- L268: function `AddFluxesRK2`: `static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrayOld, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, double alpha, double Aex_s1_coeff, double Aex_s2_coeff);`
- L275: function `ComputeFluxes`: `static void ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in, amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool use_wavespeed_correction);`
- L279: function `SetRadEnergySource`: `static void SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time);`
- L283: function `UpdateFlux`: `AMREX_GPU_DEVICE static auto UpdateFlux(int i, int j, int k, arrayconst_t const &consPrev, NewtonIterationResult<problem_t> &energy, double dt, double gas_update_factor, double Ekin0) -> FluxUpdateResult<problem_t>;`
- L286: function `AddSourceTermsMultiGroup`: `static void AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_implicit, double gas_update_factor, double dustGasCoeff, double tol_h, double tol_rel_h, double tempFloor, int *p_iteration_counter, int *p_iteration_failure_counter);`
- L290: function `AddSourceTermsSingleGroup`: `static void AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_implicit, double gas_update_factor, double dustGasCoeff, double tol_h, double tol_rel_h, double tempFloor, int *p_iteration_counter, int *p_iteration_failure_counter);`
- L294: function `balanceMatterRadiation`: `static void balanceMatterRadiation(arrayconst_t &consPrev, array_t &consNew, amrex::Box const &indexRange);`
- L298: function `ComputeMassScalars`: `AMREX_GPU_DEVICE static auto ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>;`
- L300: function `ComputeEddingtonFactor`: `AMREX_GPU_HOST_DEVICE static auto ComputeEddingtonFactor(double f) -> double;`
- L302: function `ComputeNumberDensityH`: `AMREX_GPU_HOST_DEVICE static auto ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const &massScalars) -> double;`
- L305: function `ComputePlanckOpacity`: `AMREX_GPU_HOST_DEVICE static auto ComputePlanckOpacity(double rho, double Tgas) -> Real;`
- L306: function `ComputeFluxMeanOpacity`: `AMREX_GPU_HOST_DEVICE static auto ComputeFluxMeanOpacity(double rho, double Tgas) -> Real;`
- L307: function `ComputeEnergyMeanOpacity`: `AMREX_GPU_HOST_DEVICE static auto ComputeEnergyMeanOpacity(double rho, double Tgas) -> Real;`
- L310: function `DefineOpacityExponentsAndLowerValues`: `AMREX_GPU_HOST_DEVICE static auto DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double rho, double Tgas) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>;`
- L313: function `ComputeGroupMeanOpacity`: `AMREX_GPU_HOST_DEVICE static auto ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value, amrex::GpuArray<double, nGroups_> const &radBoundaryRatios, amrex::GpuArray<double, nGroups_> const &alpha_quant) -> quokka::valarray<double, nGroups_>;`
- L316: function `ComputeBinCenterOpacity`: `AMREX_GPU_HOST_DEVICE static auto ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value) -> quokka::valarray<double, nGroups_>;`
- L322: function `ComputeEintFromEgas`: `AMREX_GPU_HOST_DEVICE static auto ComputeEintFromEgas(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Etot) -> double;`
- L323: function `ComputeEgasFromEint`: `AMREX_GPU_HOST_DEVICE static auto ComputeEgasFromEint(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Eint) -> double;`
- L324: function `PlanckFunction`: `AMREX_GPU_HOST_DEVICE static auto PlanckFunction(double nu, double T) -> double;`
- L330: function `ComputeFluxInDiffusionLimit`: `AMREX_GPU_HOST_DEVICE static auto ComputeFluxInDiffusionLimit(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double T, double vel) -> amrex::GpuArray<double, nGroups_>;`
- L334: function `ComputeRadQuantityExponents`: `AMREX_GPU_HOST_DEVICE static auto ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> amrex::GpuArray<double, nGroups_>;`
- L337: function `SolveLinearEqs`: `AMREX_GPU_HOST_DEVICE static void SolveLinearEqs(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi);`
- L339: function `SolveLinearEqsWithLastColumn`: `AMREX_GPU_HOST_DEVICE static void SolveLinearEqsWithLastColumn(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi);`
- L342: function `Solve3x3matrix`: `AMREX_GPU_HOST_DEVICE static auto Solve3x3matrix(double C00, double C01, double C02, double C10, double C11, double C12, double C20, double C21, double C22, double Y0, double Y1, double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>;`
- L345: function `ComputePlanckEnergyFractions`: `AMREX_GPU_HOST_DEVICE static auto ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature) -> quokka::valarray<amrex::Real, nGroups_>;`
- L348: function `ComputeThermalRadiationSingleGroup`: `AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> double;`
- L350: function `ComputeThermalRadiationMultiGroup`: `AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>;`
- L353: function `ComputeThermalRadiationTempDerivativeSingleGroup`: `AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real;`
- L355: function `ComputeThermalRadiationTempDerivativeMultiGroup`: `AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>;`
- L360: function `BackwardEulerOneVariable`: `AMREX_GPU_DEVICE static auto BackwardEulerOneVariable(RHSFunction const &rhs, JacFunction const &jac, double x0, double compare) -> double;`
- L373: function `DefinePhotoelectricHeatingE1Derivative`: `AMREX_GPU_HOST_DEVICE static auto DefinePhotoelectricHeatingE1Derivative(amrex::Real temperature, amrex::Real num_density) -> amrex::Real;`
- L375: function `DefineBackgroundHeatingRate`: `AMREX_GPU_HOST_DEVICE static auto DefineBackgroundHeatingRate(amrex::Real num_density) -> amrex::Real;`
- L377: function `DefineNetCoolingRate`: `AMREX_GPU_HOST_DEVICE static auto DefineNetCoolingRate(amrex::Real temperature, amrex::Real num_density) -> quokka::valarray<double, nGroups_>;`
- L379: function `DefineNetCoolingRateTempDerivative`: `AMREX_GPU_HOST_DEVICE static auto DefineNetCoolingRateTempDerivative(amrex::Real temperature, amrex::Real num_density) -> quokka::valarray<double, nGroups_>;`
- L382: function `DefineCosmicRayHeatingRate`: `AMREX_GPU_HOST_DEVICE static auto DefineCosmicRayHeatingRate(amrex::Real num_density) -> double;`
- L384: function `ComputeModelDependentKappaFAndDeltaTerms`: `AMREX_GPU_DEVICE static void ComputeModelDependentKappaFAndDeltaTerms(double T, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, quokka::valarray<double, nGroups_> const &fourPiBoverC, OpacityTerms<problem_t> &opacity_terms);`
- L395: function `ComputeJacobianForGas`: `AMREX_GPU_DEVICE static auto ComputeJacobianForGas(double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, quokka::valarray<double, nGroups_> const &tau, double c_v, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double num_den, double dt) -> JacobianResult<problem_t>;`
- L402: function `ComputeJacobianForGasAndDust`: `AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDust(double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double lambda_gd_time_dt, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double num_den, double dt) -> JacobianResult<problem_t>;`
- L410: function `ComputeJacobianForGasAndDustDecoupled`: `AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDustDecoupled( double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double lambda_gd_time_dt, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t) -> JacobianResult<problem_t>;`
- L415: function `ComputeJacobianForGasAndDustWithPE`: `AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDustWithPE( double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad, quokka::valarray<double, nGroups_> const &Erad0, double PE_heating_energy_derivative, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double lambda_gd_time_dt, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double num_den, double dt) -> JacobianResult<problem_t>;`
- L428: function `SolveGasDustRadiationEnergyExchange`: `AMREX_GPU_DEVICE static auto SolveGasDustRadiationEnergyExchange(double Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double rho, double coeff_n, double dt, amrex::GpuArray<Real, nmscalars_> const &massScalars, int n_outer_iter, quokka::valarray<double, nGroups_> const &work, quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, double resid_tol, double rel_change_tol, double tempFloor, int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>;`
- L437: function `SolveGasDustRadiationEnergyExchangeWithPE`: `AMREX_GPU_DEVICE static auto SolveGasDustRadiationEnergyExchangeWithPE(double Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double rho, double coeff_n, double dt, amrex::GpuArray<Real, nmscalars_> const &massScalars, int n_outer_iter, quokka::valarray<double, nGroups_> const &work, quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, double resid_tol, double rel_change_tol, double tempFloor, int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>;`
- L447: function `ComputeCellOpticalDepth`: `AMREX_GPU_DEVICE static auto ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k, const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries) -> quokka::valarray<double, nGroups_>;`
- L452: function `isStateValid`: `AMREX_GPU_DEVICE static auto isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool;`
- L454: function `amendRadState`: `AMREX_GPU_DEVICE static void amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons);`
- L457: function `ComputeRadPressure`: `AMREX_GPU_DEVICE static auto ComputeRadPressure(double erad_L, double Fx_L, double Fy_L, double Fz_L, double fx_L, double fy_L, double fz_L) -> RadPressureResult;`
- L460: function `ComputeEddingtonTensor`: `AMREX_GPU_DEVICE static auto ComputeEddingtonTensor(double fx_L, double fy_L, double fz_L) -> std::array<std::array<double, 3>, 3>;`
- L466: function `ComputePlanckEnergyFractions`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature) -> quokka::valarray<amrex::Real, nGroups_> {`
- L497: function `ComputeNumberDensityH`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const & ) -> double {`
- L503: function `ComputeThermalRadiationSingleGroup`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> Real {`
- L515: function `ComputeThermalRadiationMultiGroup`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_> {`
- L531: function `ComputeThermalRadiationTempDerivativeSingleGroup`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real {`
- L538: function `ComputeThermalRadiationTempDerivativeMultiGroup`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_> {`
- L549: function `DefineBackgroundHeatingRate`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineBackgroundHeatingRate(amrex::Real const ) -> amrex::Real {`
- L556: function `DefineNetCoolingRate`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineNetCoolingRate(amrex::Real const , amrex::Real const ) -> quokka::valarray<double, nGroups_> {`
- L566: function `DefineNetCoolingRateTempDerivative`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineNetCoolingRateTempDerivative(amrex::Real const , amrex::Real const ) -> quokka::valarray<double, nGroups_> {`
- L574: function `DefineCosmicRayHeatingRate`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineCosmicRayHeatingRate(amrex::Real const ) -> double {`
- L585: function `SolveLinearEqs`: `AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqs(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi) {`
- L593: function `Solve3x3matrix`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::Solve3x3matrix(const double C00, const double C01, const double C02, const double C10, const double C11, const double C12, const double C20, const double C21, const double C22, const double Y0, const double Y1, const double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real> {`
- L614: function `SetRadEnergySource`: `void RadSystem<problem_t>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time) {`
- L624: function `ConservedToPrimitive`: `void RadSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange) {`
- L651: function `ComputeMaxSignalSpeed`: `void RadSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const & , array_t &maxSignal, amrex::Box const &indexRange) {`
- L660: function `isStateValid`: `template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool {`
- L680: function `amendRadState`: `template <typename problem_t> AMREX_GPU_DEVICE void RadSystem<problem_t>::amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons) {`
- L716: function `PredictStep`: `void RadSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> , const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int ) {`
- L761: function `AddFluxesRK2`: `void RadSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> , amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> , const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int , const double alpha, const double Aex_s1_coeff, const double Aex_s2_coeff) {`
- L823: function `ComputeEddingtonFactor`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEddingtonFactor(double f_in) -> double {`
- L844: function `ComputeMassScalars`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_> {`
- L855: function `ComputeCellOpticalDepth`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k, const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries) -> quokka::valarray<double, nGroups_> {`
- L924: function `ComputeEddingtonTensor`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeEddingtonTensor(const double fx, const double fy, const double fz) -> std::array<std::array<double, 3>, 3> {`
- L970: function `ComputeRadPressure`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeRadPressure(const double erad, const double Fx, const double Fy, const double Fz, const double fx, const double fy, const double fz) -> RadPressureResult {`
- L1037: function `ComputeFluxes`: `void RadSystem<problem_t>::ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in, amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool const use_wavespeed_correction) {`
- L1191: function `ComputePlanckOpacity`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckOpacity(const double , const double ) -> Real {`
- L1196: function `ComputeFluxMeanOpacity`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> Real {`
- L1201: function `ComputeEnergyMeanOpacity`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEnergyMeanOpacity(const double rho, const double Tgas) -> Real {`
- L1207: function `DefineOpacityExponentsAndLowerValues`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> , const double , const double ) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> {`
- L1221: function `ComputeRadQuantityExponents`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> amrex::GpuArray<double, nGroups_> {`
- L1340: function `ComputeEintFromEgas`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEintFromEgas(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom, const double Etot) -> double {`
- L1351: function `ComputeEgasFromEint`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEgasFromEint(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom, const double Eint) -> double {`
- L1360: function `PlanckFunction`: `template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::PlanckFunction(const double nu, const double T) -> double {`
- L1379: function `ComputeDiffusionFluxMeanOpacity`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeDiffusionFluxMeanOpacity(const quokka::valarray<double, nGroups_> kappaPVec, const quokka::valarray<double, nGroups_> kappaEVec, const quokka::valarray<double, nGroups_> fourPiBoverC, const amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge, const amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge, const amrex::GpuArray<double, nGroups_ + 1> kappa_slope) -> quokka::valarray<double, nGroups_> {`
- L1405: function `ComputeBinCenterOpacity`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value) -> quokka::valarray<double, nGroups_> {`
- L1418: function `ComputeFluxInDiffusionLimit`: `AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxInDiffusionLimit(const amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, const double T, const double vel) -> amrex::GpuArray<double, nGroups_> {`
- L1438: function `BackwardEulerOneVariable`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::BackwardEulerOneVariable(RHSFunction const &rhs, JacFunction const &jac, const double x0, const double compare) -> double {`
- L1471: function `ComputeDustTemperatureBateKeto`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeDustTemperatureBateKeto(double const T_gas, double const T_d_init, double const rho, quokka::valarray<double, nGroups_> const &Erad, double N_d, double dt, double R_sum, int n_step, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries) -> double {`

## `src/radiation/source_terms_multi_group.hpp`

- L9: function `ComputeModelDependentKappaEAndKappaP`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeModelDependentKappaEAndKappaP( double const T, double const rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios, quokka::valarray<double, nGroups_> const &fourPiBoverC, quokka::valarray<double, nGroups_> const &Erad, int const n_iter, amrex::GpuArray<double, nGroups_> const &alpha_E, amrex::GpuArray<double, nGroups_> const &alpha_P) -> OpacityTerms<problem_t> {`
- L106: function `ComputeJacobianForGas`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGas(double , double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, quokka::valarray<double, nGroups_> const &tau, double c_v, quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double const num_den, double const dt) -> JacobianResult<problem_t> {`
- L150: function `SolveGasRadiationEnergyExchange`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasRadiationEnergyExchange( double const Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double const rho, double const dt, amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work, quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, double const resid_tol, double const rel_change_tol, double const , int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t> {`
- L428: function `UpdateFlux`: `AMREX_GPU_DEVICE auto RadSystem<problem_t>::UpdateFlux(int const i, int const j, int const k, arrayconst_t &consPrev, NewtonIterationResult<problem_t> &energy, double const dt, double const gas_update_factor, double const Ekin0) -> FluxUpdateResult<problem_t> {`
- L589: function `AddSourceTermsMultiGroup`: `void RadSystem<problem_t>::AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_implicit, double gas_update_factor_in, double dustGasCoeff, double const tol_h, double const tol_rel_h, double const tempFloor_local, int *p_iteration_counter, int *p_iteration_failure_counter) {`

## `src/radiation/source_terms_single_group.hpp`

- L10: function `AddSourceTermsSingleGroup`: `void RadSystem<problem_t>::AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, Real dt_implicit, double gas_update_factor_in, double dustGasCoeff, double tol_h, double , double , int *p_iteration_counter, int *p_iteration_failure_counter) {`

## `src/simulation.hpp`

- L117: function `formatIntVect`: `inline auto formatIntVect(amrex::IntVect const &iv) -> std::string {`
- L128: function `formatRealVect`: `inline auto formatRealVect(amrex::RealVect const &rv) -> std::string {`
- L143: struct `as_if`: `template <typename T> struct as_if<T, std::optional<T>> {`
- L144: function `as_if`: `explicit as_if(const Node &node_) : node(node_) {`
- L146: function `operator`: `auto operator()() const -> std::optional<T> {`
- L159: struct `as_if`: `template <> struct as_if<std::string, std::optional<std::string>> {`
- L160: function `as_if`: `explicit as_if(const Node &node_) : node(node_) {`
- L162: function `operator`: `auto operator()() const -> std::optional<std::string> {`
- L174: enum class `FillPatchType`: `enum class FillPatchType { fillpatch_class, fillpatch_function };`
- L177: class `AMRSimulation`: `template <typename problem_t> class AMRSimulation : public amrex::AmrCore`
- L245: function `AMRSimulation`: `explicit AMRSimulation(amrex::Vector<amrex::BCRec> &BCs_cc, amrex::Vector<amrex::BCRec> &BCs_fc) : BCs_cc_(BCs_cc), BCs_fc_(BCs_fc) {`
- L247: function `AMRSimulation`: `explicit AMRSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : BCs_cc_(BCs_cc), BCs_fc_(builtin_BCs_fc(BCs_cc)) {`
- L249: function `AMRSimulation`: `explicit AMRSimulation() {`
- L251: function `builtin_BCs_fc`: `auto builtin_BCs_fc(amrex::Vector<amrex::BCRec> & ) -> amrex::Vector<amrex::BCRec> {`
- L258: function `readBCs`: `void readBCs() {`
- L273: function `initialize`: `void initialize();`
- L274: function `PerformanceHints`: `void PerformanceHints();`
- L275: function `readParameters`: `void readParameters();`
- L276: function `rereadRuntimeParameters`: `void rereadRuntimeParameters();`
- L277: function `setInitialConditions`: `void setInitialConditions();`
- L278: function `setInitialConditionsAtLevel_cc`: `void setInitialConditionsAtLevel_cc(int level, amrex::Real time);`
- L279: function `setInitialConditionsAtLevel_fc`: `void setInitialConditionsAtLevel_fc(int level, amrex::Real time);`
- L280: function `evolve`: `void evolve();`
- L281: function `computeTimestep`: `void computeTimestep();`
- L282: function `computeTimestepAtLevel`: `auto computeTimestepAtLevel(int lev) -> amrex::ValLocPair<amrex::Real, amrex::IntVect>;`
- L284: function `AverageFCToCC`: `void AverageFCToCC(amrex::MultiFab &mf_cc, const amrex::MultiFab &mf_fc, int idim, int dstcomp_start, int srccomp_start, int srccomp_total) const;`
- L285: function `setCustomGhostCells`: `virtual void setCustomGhostCells() {`
- L287: function `computeMaxSignalLocal`: `virtual void computeMaxSignalLocal(int level) = 0;`
- L288: function `printCellProperties`: `virtual void printCellProperties(int lev, amrex::IntVect const &index) = 0;`
- L289: function `advanceSingleTimestepAtLevel`: `virtual void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle) = 0;`
- L290: function `preCalculateInitialConditions`: `virtual void preCalculateInitialConditions() = 0;`
- L291: function `setInitialConditionsOnGrid`: `virtual void setInitialConditionsOnGrid(quokka::grid const &grid_elem) = 0;`
- L292: function `setInitialConditionsOnGridFaceVars`: `virtual void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) = 0;`
- L293: function `postInitialization`: `virtual void postInitialization() {`
- L294: function `refineGrid`: `virtual void refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) = 0;`
- L296: function `createInitialRadParticles`: `virtual void createInitialRadParticles() = 0;`
- L297: function `createInitialCICParticles`: `virtual void createInitialCICParticles() = 0;`
- L298: function `createInitialCICRadParticles`: `virtual void createInitialCICRadParticles() = 0;`
- L299: function `createInitialStochasticStellarPopParticles`: `virtual void createInitialStochasticStellarPopParticles() = 0;`
- L300: function `createInitialSinkParticles`: `virtual void createInitialSinkParticles() = 0;`
- L301: function `createInitialTestParticles`: `virtual void createInitialTestParticles() = 0;`
- L302: function `particleMeshInteraction`: `void particleMeshInteraction(amrex::Real time, amrex::Real dt);`
- L306: function `computeBeforeTimestep`: `virtual void computeBeforeTimestep() = 0;`
- L307: function `computeAfterTimestep`: `virtual void computeAfterTimestep() = 0;`
- L308: function `computeAfterEvolve`: `virtual void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) = 0;`
- L309: function `fillPoissonRhsAtLevel`: `virtual void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) = 0;`
- L310: function `applyPoissonGravityAtLevel`: `virtual void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) = 0;`
- L311: function `WriteSingleLevelPlotfileSimplified`: `virtual void WriteSingleLevelPlotfileSimplified(const std::string &plotfile_prefix, const amrex::MultiFab &mf, const amrex::Vector<std::string> &compNames, int lev, int interval) = 0;`
- L315: function `ComputeDerivedVar`: `virtual void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const = 0;`
- L316: function `ComputeDensityFloorDebug`: `virtual void ComputeDensityFloorDebug(int lev, amrex::MultiFab &mf, int ncomp) const;`
- L319: function `ComputeStatistics`: `virtual auto ComputeStatistics() -> std::map<std::string, amrex::Real> = 0;`
- L323: function `FixupState`: `virtual void FixupState(int level) = 0;`
- L327: function `ErrorEst`: `void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override = 0;`
- L330: function `MakeNewLevelFromCoarse`: `void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;`
- L333: function `RemakeLevel`: `void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;`
- L336: function `ClearLevel`: `void ClearLevel(int lev) override;`
- L340: function `MakeNewLevelFromScratch`: `void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;`
- L345: function `fillBoundaryConditions`: `void fillBoundaryConditions(amrex::MultiFab &S_filled, amrex::MultiFab &state, int lev, amrex::Real time, quokka::centering cen, quokka::direction dir, PreInterpHook const &pre_interp, PostInterpHook const &post_interp, FillPatchType fptype = FillPatchType::fillpatch_class);`
- L349: function `FillPatchWithData`: `void FillPatchWithData(int lev, amrex::Real time, amrex::MultiFab &mf, amrex::Vector<amrex::MultiFab *> &coarseData, amrex::Vector<amrex::Real> &coarseTime, amrex::Vector<amrex::MultiFab *> &fineData, amrex::Vector<amrex::Real> &fineTime, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering &cen, quokka::direction dir, FillPatchType fptype, PreInterpHook const &pre_interp, PostInterpHook const &post_interp);`
- L354: function `InterpHookNone`: `static void InterpHookNone(amrex::MultiFab &mf, int scomp, int ncomp);`
- L355: function `FillPatch`: `virtual void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir, FillPatchType fptype);`
- L358: function `getAmrInterpolaterCellCentered`: `auto getAmrInterpolaterCellCentered() -> amrex::MFInterpolater *;`
- L359: function `getAmrInterpolaterFaceCentered`: `auto getAmrInterpolaterFaceCentered() -> amrex::Interpolater *;`
- L360: function `FillCoarsePatch`: `void FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering cen, quokka::direction dir);`
- L362: function `FillCoarsePatchFaceArray`: `void FillCoarsePatchFaceArray(int lev, amrex::Real time, amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> &mf_array, int icomp, int ncomp, amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> &BCs_array);`
- L364: function `GetData`: `void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab *> &data, amrex::Vector<amrex::Real> &datatime, quokka::centering cen, quokka::direction dir);`
- L366: function `GetDataFaceArray`: `void GetDataFaceArray(int lev, amrex::Real time, amrex::Array<amrex::Vector<amrex::MultiFab *>, AMREX_SPACEDIM> &data_array, amrex::Vector<amrex::Real> &datatime);`
- L368: function `AverageDown`: `void AverageDown();`
- L369: function `AverageDownTo`: `void AverageDownTo(int crse_lev);`
- L370: function `timeStepWithSubcycling`: `void timeStepWithSubcycling(int lev, amrex::Real time, int iteration);`
- L371: function `calculateGpotAllLevels`: `void calculateGpotAllLevels();`
- L372: function `gravAccelAllLevels`: `void gravAccelAllLevels(amrex::Real dt);`
- L373: function `ellipticSolveAllLevels`: `void ellipticSolveAllLevels(amrex::Real dt);`
- L375: function `incrementFluxRegisters`: `void incrementFluxRegisters(amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays, int lev, amrex::Real dt_lev);`
- L377: function `incrementEMFRegisters`: `void incrementEMFRegisters(amrex::EdgeFluxRegister *emf_as_crse, amrex::EdgeFluxRegister *emf_as_fine, std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_emf_components, int lev, amrex::Real dt_lev);`
- L381: function `setCustomBoundaryConditions`: `AMREX_GPU_DEVICE static void setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp, amrex::GeometryData const &geom, amrex::Real time, const amrex::BCRec *bcr, int bcomp, int orig_comp);`
- L387: function `setCustomBoundaryConditionsFaceVar`: `AMREX_GPU_DEVICE static void setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp, amrex::GeometryData const &geom, amrex::Real time, const amrex::BCRec *bcr, int bcomp, int orig_comp);`
- L398: function `setConstantDirichletBCLo`: `AMREX_GPU_DEVICE static void setConstantDirichletBCLo(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, amrex::GpuArray<amrex::Real, N> const &values);`
- L408: function `setConstantDirichletBCHi`: `AMREX_GPU_DEVICE static void setConstantDirichletBCHi(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, amrex::GpuArray<amrex::Real, N> const &values);`
- L418: function `setDiodeBCLo`: `AMREX_GPU_DEVICE static void setDiodeBCLo(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom);`
- L427: function `setDiodeBCHi`: `AMREX_GPU_DEVICE static void setDiodeBCHi(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom);`
- L438: function `setConstantDirichletBCFaceVarLo`: `AMREX_GPU_DEVICE static void setConstantDirichletBCFaceVarLo(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar_fc, amrex::GeometryData const &geom, amrex::GpuArray<amrex::Real, ncomp> const &values);`
- L450: function `setConstantDirichletBCFaceVarHi`: `AMREX_GPU_DEVICE static void setConstantDirichletBCFaceVarHi(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar_fc, amrex::GeometryData const &geom, amrex::GpuArray<amrex::Real, ncomp> const &values);`
- L454: function `computeVolumeIntegral`: `template <typename F> auto computeVolumeIntegral(F const &user_f) -> amrex::Real;`
- L465: function `AverageDownDerived`: `void AverageDownDerived(const amrex::Vector<amrex::MultiFab *> &mfs, const amrex::Vector<std::string> &varnames) const;`
- L466: function `createDiagnostics`: `void createDiagnostics();`
- L467: function `createRuntimeDerivedFields`: `void createRuntimeDerivedFields();`
- L468: function `updateRuntimeDerivedFields`: `void updateRuntimeDerivedFields();`
- L470: function `updateDiagnostics`: `void updateDiagnostics();`
- L471: function `doDiagnostics`: `void doDiagnostics();`
- L472: function `WriteMetadataFile`: `void WriteMetadataFile(std::string const &MetadataFileName) const;`
- L473: function `ReadMetadataFile`: `void ReadMetadataFile(std::string const &chkfilename);`
- L474: function `WriteStatisticsFile`: `void WriteStatisticsFile();`
- L475: function `WritePlotFile`: `void WritePlotFile();`
- L476: function `WriteCheckpointFile`: `void WriteCheckpointFile() const;`
- L477: function `SetLastCheckpointSymlink`: `void SetLastCheckpointSymlink(std::string const &checkpointname) const;`
- L478: function `writeFaceVelocitiesToDisk`: `void writeFaceVelocitiesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVel, int lev, int step);`
- L479: function `writeReconstructedStatesToDisk`: `void writeReconstructedStatesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &leftState, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &rightState, int lev, int step);`
- L483: struct `RefinementContext`: `struct RefinementContext {`
- L489: function `ReadCheckpointFile`: `void ReadCheckpointFile();`
- L492: function `detectRefinementContext`: `auto detectRefinementContext(const amrex::BoxArray &restart_ba, const amrex::Geometry &current_geom) -> RefinementContext;`
- L493: function `readCheckpointHeader`: `auto readCheckpointHeader(const std::string &restart_file) -> amrex::Vector<amrex::BoxArray>;`
- L494: function `interpolateMultiFabFromRestart`: `void interpolateMultiFabFromRestart(amrex::MultiFab &target, const amrex::MultiFab &source, const RefinementContext &context, const amrex::Geometry &coarse_geom, const amrex::Geometry &fine_geom, const amrex::Vector<amrex::BCRec> &bcs);`
- L496: function `interpolateFaceMultiFabFromRestart`: `void interpolateFaceMultiFabFromRestart(int lev, const RefinementContext &context, const amrex::Vector<amrex::Geometry> &restart_geom, amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> &restart_fc);`
- L498: function `loadMultiFabData`: `void loadMultiFabData(const RefinementContext &context);`
- L499: function `loadBalanceOnRestart`: `auto loadBalanceOnRestart(const amrex::BoxArray &input_ba, int lev) -> amrex::BoxArray;`
- L502: function `restartParticleContainerWithRefinement`: `void restartParticleContainerWithRefinement(std::unique_ptr<ParticleContainer> &particles, std::string const &restart_chkfile, std::string const &particle_type_name, amrex::Vector<amrex::BoxArray> const &header_box_arrays);`
- L506: function `initializeParticleContainerFromCheckpoint`: `void initializeParticleContainerFromCheckpoint(std::unique_ptr<ContainerType> &container, amrex::Vector<amrex::BoxArray> const &header_box_arrays);`
- L508: function `getGitHashForQuokka`: `auto getGitHashForQuokka() const -> std::string;`
- L509: function `getGitHashForAmrex`: `auto getGitHashForAmrex() const -> std::string;`
- L510: function `getWalltime`: `auto getWalltime() -> amrex::Real;`
- L511: function `getCycleWalltime`: `auto getCycleWalltime() -> amrex::Real;`
- L512: function `setChkFile`: `void setChkFile(std::string const &chkfile_number);`
- L520: function `kickParticlesAllLevels`: `void kickParticlesAllLevels(amrex::Real dt);`
- L524: function `initializeSimulationMetadata`: `void initializeSimulationMetadata();`
- L659: function `InitParticles`: `void InitParticles();`
- L663: function `InitPhyParticles`: `void InitPhyParticles(amrex::Vector<amrex::BoxArray> const *header_box_arrays = nullptr);`
- L678: function `GetParticleRegister`: `auto GetParticleRegister() -> quokka::PhysicsParticleRegister<problem_t> & {`
- L682: function `getGitHashForQuokka`: `template <typename problem_t> auto AMRSimulation<problem_t>::getGitHashForQuokka() const -> std::string {`
- L688: function `getGitHashForAmrex`: `template <typename problem_t> auto AMRSimulation<problem_t>::getGitHashForAmrex() const -> std::string {`
- L694: function `setChkFile`: `template <typename problem_t> void AMRSimulation<problem_t>::setChkFile(std::string const &chkfile_number) {`
- L696: function `getOldMF_cc`: `template <typename problem_t> auto AMRSimulation<problem_t>::getOldMF_cc() const -> const amrex::Vector<amrex::MultiFab> & {`
- L698: function `getNewMF_cc`: `template <typename problem_t> auto AMRSimulation<problem_t>::getNewMF_cc() const -> const amrex::Vector<amrex::MultiFab> & {`
- L700: function `getOldMF_fc`: `template <typename problem_t> auto AMRSimulation<problem_t>::getOldMF_fc() const -> const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> & {`
- L705: function `getNewMF_fc`: `template <typename problem_t> auto AMRSimulation<problem_t>::getNewMF_fc() const -> const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> & {`
- L710: function `initialize`: `template <typename problem_t> void AMRSimulation<problem_t>::initialize() {`
- L792: function `PerformanceHints`: `template <typename problem_t> void AMRSimulation<problem_t>::PerformanceHints() {`
- L845: function `readParameters`: `template <typename problem_t> void AMRSimulation<problem_t>::readParameters() {`
- L1050: function `amrex::Print`: `amrex::Print() << std::format("Setting walltime limit to {`
- L1089: function `amrex::Print`: `amrex::Print() << std::format("\tTable dimensions: {`
- L1091: function `amrex::Print`: `amrex::Print() << std::format("\tNumber of outputs: {`
- L1122: function `rereadRuntimeParameters`: `template <typename problem_t> void AMRSimulation<problem_t>::rereadRuntimeParameters() {`
- L1129: function `setInitialConditions`: `template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditions() {`
- L1197: function `computeTimestepAtLevel`: `template <typename problem_t> auto AMRSimulation<problem_t>::computeTimestepAtLevel(int lev) -> amrex::ValLocPair<amrex::Real, amrex::IntVect> {`
- L1220: function `amrex::Print`: `amrex::Print() << std::format("...[level {`
- L1221: function `amrex::Print`: `amrex::Print() << std::format("...[level {`
- L1245: function `amrex::Print`: `amrex::Print() << std::format("...[level {`
- L1246: function `amrex::Print`: `amrex::Print() << std::format("...[level {`
- L1247: function `amrex::Print`: `amrex::Print() << std::format("...[level {`
- L1260: function `amrex::Print`: `amrex::Print() << std::format("...[level {`
- L1262: function `amrex::Print`: `amrex::Print() << std::format("...[level {`
- L1269: function `computeTimestep`: `template <typename problem_t> void AMRSimulation<problem_t>::computeTimestep() {`
- L1340: function `getWalltime`: `template <typename problem_t> auto AMRSimulation<problem_t>::getWalltime() -> amrex::Real {`
- L1347: function `getCycleWalltime`: `template <typename problem_t> auto AMRSimulation<problem_t>::getCycleWalltime() -> amrex::Real {`
- L1356: function `evolve`: `template <typename problem_t> void AMRSimulation<problem_t>::evolve() {`
- L1691: function `calculateGpotAllLevels`: `template <typename problem_t> void AMRSimulation<problem_t>::calculateGpotAllLevels() {`
- L1890: function `gravAccelAllLevels`: `template <typename problem_t> void AMRSimulation<problem_t>::gravAccelAllLevels(const amrex::Real dt) {`
- L1905: function `ellipticSolveAllLevels`: `template <typename problem_t> void AMRSimulation<problem_t>::ellipticSolveAllLevels(const amrex::Real dt) {`
- L1945: struct `setFunctorParticleAccel`: `struct setFunctorParticleAccel {`
- L1946: function `operator`: `AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp, amrex::GeometryData const &geom, const amrex::Real &time, const amrex::BCRec *bcr, int bcomp, const int &orig_comp) const {`
- L1957: function `kickParticlesAllLevels`: `template <typename problem_t> void AMRSimulation<problem_t>::kickParticlesAllLevels(const amrex::Real dt) {`
- L2051: function `particleMeshInteraction`: `template <typename problem_t> void AMRSimulation<problem_t>::particleMeshInteraction(amrex::Real time, amrex::Real dt) {`
- L2107: function `amrex::Print`: `amrex::Print() << std::format("[PARTICLES] SN explosions: Time: {`
- L2121: function `timeStepWithSubcycling`: `template <typename problem_t> void AMRSimulation<problem_t>::timeStepWithSubcycling(int lev, amrex::Real time, int iteration) {`
- L2129: function `last_regrid_step`: `static amrex::Vector<int> last_regrid_step(max_level + 1, 0);`
- L2254: function `incrementFluxRegisters`: `void AMRSimulation<problem_t>::incrementFluxRegisters(amrex::FluxRegister *fr_as_crse, amrex::FluxRegister *fr_as_fine, std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxArrays, int const lev, amrex::Real const dt_lev) {`
- L2303: function `incrementEMFRegisters`: `void AMRSimulation<problem_t>::incrementEMFRegisters(amrex::EdgeFluxRegister *emf_as_crse, amrex::EdgeFluxRegister *emf_as_fine, std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_emf_components, int const lev, amrex::Real const dt_lev) {`
- L2329: function `getAmrInterpolaterCellCentered`: `template <typename problem_t> auto AMRSimulation<problem_t>::getAmrInterpolaterCellCentered() -> amrex::MFInterpolater * {`
- L2347: function `getAmrInterpolaterFaceCentered`: `template <typename problem_t> auto AMRSimulation<problem_t>::getAmrInterpolaterFaceCentered() -> amrex::Interpolater * {`
- L2357: function `MakeNewLevelFromCoarse`: `void AMRSimulation<problem_t>::MakeNewLevelFromCoarse(int level, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) {`
- L2409: function `RemakeLevel`: `void AMRSimulation<problem_t>::RemakeLevel(int level, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) {`
- L2465: function `ClearLevel`: `template <typename problem_t> void AMRSimulation<problem_t>::ClearLevel(int level) {`
- L2485: function `InterpHookNone`: `template <typename problem_t> void AMRSimulation<problem_t>::InterpHookNone(amrex::MultiFab &mf, int scomp, int ncomp) {`
- L2490: struct `setBoundaryFunctor`: `template <typename problem_t> struct setBoundaryFunctor {`
- L2491: function `operator`: `AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp, amrex::GeometryData const &geom, const amrex::Real &time, const amrex::BCRec *bcr, int bcomp, const int &orig_comp) const {`
- L2499: struct `setBoundaryFunctorFaceVar`: `template <typename problem_t> struct setBoundaryFunctorFaceVar {`
- L2503: function `setBoundaryFunctorFaceVar`: `AMREX_GPU_HOST_DEVICE explicit setBoundaryFunctorFaceVar(quokka::direction dir) : dir_(dir) {`
- L2506: function `setBoundaryFunctorFaceVar`: `AMREX_GPU_HOST_DEVICE setBoundaryFunctorFaceVar() : dir_(quokka::direction::na) {`
- L2508: function `operator`: `AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp, amrex::GeometryData const &geom, const amrex::Real &time, const amrex::BCRec *bcr, int bcomp, const int &orig_comp) const {`
- L2527: function `setCustomBoundaryConditions`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp, amrex::GeometryData const &geom, const amrex::Real time, const amrex::BCRec *bcr, int bcomp, int orig_comp) {`
- L2555: function `setConstantDirichletBCLo`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setConstantDirichletBCLo(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, amrex::GpuArray<amrex::Real, N> const &values) {`
- L2588: function `setConstantDirichletBCHi`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setConstantDirichletBCHi(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom, amrex::GpuArray<amrex::Real, N> const &values) {`
- L2622: function `setDiodeBCLo`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setDiodeBCLo(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom) {`
- L2742: function `setDiodeBCHi`: `AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setDiodeBCHi(amrex::IntVect const &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom) {`
- L2983: function `FillPatch`: `void AMRSimulation<problem_t>::FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir, FillPatchType fptype) {`
- L3010: function `setInitialConditionsAtLevel_cc`: `template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditionsAtLevel_cc(int level, amrex::Real time) {`
- L3030: function `setInitialConditionsAtLevel_fc`: `template <typename problem_t> void AMRSimulation<problem_t>::setInitialConditionsAtLevel_fc(int level, amrex::Real time) {`
- L3065: function `MakeNewLevelFromScratch`: `void AMRSimulation<problem_t>::MakeNewLevelFromScratch(int level, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) {`
- L3118: function `fillBoundaryConditions`: `void AMRSimulation<problem_t>::fillBoundaryConditions(amrex::MultiFab &S_filled, amrex::MultiFab &state, int const lev, amrex::Real const time, quokka::centering cen, quokka::direction dir, PreInterpHook const &pre_interp, PostInterpHook const &post_interp, FillPatchType fptype) {`
- L3204: function `FillPatchWithData`: `void AMRSimulation<problem_t>::FillPatchWithData(int lev, amrex::Real time, amrex::MultiFab &mf, amrex::Vector<amrex::MultiFab *> &coarseData, amrex::Vector<amrex::Real> &coarseTime, amrex::Vector<amrex::MultiFab *> &fineData, amrex::Vector<amrex::Real> &fineTime, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering &cen, quokka::direction dir, FillPatchType fptype, PreInterpHook const &pre_interp, PostInterpHook const &post_interp) {`
- L3276: function `FillCoarsePatch`: `void AMRSimulation<problem_t>::FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, amrex::Vector<amrex::BCRec> &BCs, quokka::centering cen, quokka::direction dir) {`
- L3309: function `FillCoarsePatchFaceArray`: `void AMRSimulation<problem_t>::FillCoarsePatchFaceArray(int lev, amrex::Real time, amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> &mf_array, int icomp, int ncomp, amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> &BCs_array) {`
- L3350: function `GetData`: `void AMRSimulation<problem_t>::GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab *> &data, amrex::Vector<amrex::Real> &datatime, quokka::centering cen, quokka::direction dir) {`
- L3372: function `amrex::almostEqual`: `} else if (amrex::almostEqual(time, tOld_[lev], 5)) {`
- L3394: function `GetDataFaceArray`: `void AMRSimulation<problem_t>::GetDataFaceArray(int lev, amrex::Real time, amrex::Array<amrex::Vector<amrex::MultiFab *>, AMREX_SPACEDIM> &data_array, amrex::Vector<amrex::Real> &datatime) {`
- L3409: function `amrex::almostEqual`: `} else if (amrex::almostEqual(time, tOld_[lev], 5)) {`
- L3425: function `AverageDown`: `template <typename problem_t> void AMRSimulation<problem_t>::AverageDown() {`
- L3435: function `AverageDownTo`: `template <typename problem_t> void AMRSimulation<problem_t>::AverageDownTo(int crse_lev) {`
- L3452: function `computeVolumeIntegral`: `template <typename problem_t> template <typename F> auto AMRSimulation<problem_t>::computeVolumeIntegral(F const &user_f) -> amrex::Real {`
- L3478: function `InitParticles`: `template <typename problem_t> void AMRSimulation<problem_t>::InitParticles() {`
- L3499: function `InitPhyParticles`: `template <typename problem_t> void AMRSimulation<problem_t>::InitPhyParticles(amrex::Vector<amrex::BoxArray> const *header_box_arrays) {`
- L3625: function `PlotFileName`: `template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileName(int lev) const -> std::string {`
- L3628: function `CustomPlotFileName`: `template <typename problem_t> auto AMRSimulation<problem_t>::CustomPlotFileName(const char *base, int lev) const -> std::string {`
- L3635: function `AverageFCToCC`: `void AMRSimulation<problem_t>::AverageFCToCC(amrex::MultiFab &mf_cc, const amrex::MultiFab &mf_fc, int idim, int dstcomp_start, int srccomp_start, int srccomp_total) const {`
- L3654: function `amrex::IntVect`: `amrex::ParallelFor(mf_cc, amrex::IntVect(AMREX_D_DECL(ng_cc, ng_cc, ng_cc)), [=] AMREX_GPU_DEVICE(int boxidx, int i, int j, int k) {`
- L3663: function `PlotFileMFAtLevel_cc`: `template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMFAtLevel_cc(const int lev, const int included_ghosts) -> amrex::MultiFab {`
- L3736: function `ComputeDensityFloorDebug`: `template <typename problem_t> void AMRSimulation<problem_t>::ComputeDensityFloorDebug(int lev, amrex::MultiFab &mf, int ncomp) const {`
- L3774: function `PlotFileMFAtLevel_fc`: `template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMFAtLevel_fc(const int lev, int idim, const int nghost_fc_) -> amrex::MultiFab {`
- L3799: function `AverageDownDerived`: `void AMRSimulation<problem_t>::AverageDownDerived(const amrex::Vector<amrex::MultiFab *> &mfs, const amrex::Vector<std::string> &varnames) const {`
- L3824: function `PlotFileMF_cc`: `template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMF_cc(const int included_ghosts) -> amrex::Vector<amrex::MultiFab> {`
- L3844: function `PlotFileMF_fc`: `template <typename problem_t> auto AMRSimulation<problem_t>::PlotFileMF_fc(const int nghost_fc_) -> std::array<amrex::Vector<amrex::MultiFab>, AMREX_SPACEDIM> {`
- L3856: function `createRuntimeDerivedFields`: `template <typename problem_t> void AMRSimulation<problem_t>::createRuntimeDerivedFields() {`
- L3956: function `updateRuntimeDerivedFields`: `template <typename problem_t> void AMRSimulation<problem_t>::updateRuntimeDerivedFields() {`
- L3969: function `computeRuntimeDerivedVar`: `auto AMRSimulation<problem_t>::computeRuntimeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const -> bool {`
- L3980: function `createDiagnostics`: `template <typename problem_t> void AMRSimulation<problem_t>::createDiagnostics() {`
- L4053: function `updateDiagnostics`: `template <typename problem_t> void AMRSimulation<problem_t>::updateDiagnostics() {`
- L4065: function `doDiagnostics`: `template <typename problem_t> void AMRSimulation<problem_t>::doDiagnostics() {`
- L4167: function `GetPlotfileVarNames`: `template <typename problem_t> auto AMRSimulation<problem_t>::GetPlotfileVarNames() const -> amrex::Vector<std::string> {`
- L4169: function `GetPlotfileVarNames_fc`: `template <typename problem_t> auto AMRSimulation<problem_t>::GetPlotfileVarNames_fc() const -> std::array<amrex::Vector<std::string>, AMREX_SPACEDIM> {`
- L4183: function `WritePlotFile`: `template <typename problem_t> void AMRSimulation<problem_t>::WritePlotFile() {`
- L4256: function `WriteMetadataFile`: `template <typename problem_t> void AMRSimulation<problem_t>::WriteMetadataFile(std::string const &MetadataFileName) const {`
- L4275: function `ReadMetadataFile`: `template <typename problem_t> void AMRSimulation<problem_t>::ReadMetadataFile(std::string const &chkfilename) {`
- L4294: function `amrex::Print`: `amrex::Print() << std::format("\t {`
- L4297: function `amrex::Print`: `amrex::Print() << std::format("\t {`
- L4301: function `amrex::Print`: `amrex::Print() << std::format("\t {`
- L4308: function `WriteStatisticsFile`: `template <typename problem_t> void AMRSimulation<problem_t>::WriteStatisticsFile() {`
- L4352: function `SetLastCheckpointSymlink`: `template <typename problem_t> void AMRSimulation<problem_t>::SetLastCheckpointSymlink(std::string const &checkpointname) const {`
- L4368: function `WriteCheckpointFile`: `template <typename problem_t> void AMRSimulation<problem_t>::WriteCheckpointFile() const {`
- L4485: function `GotoNextLine`: `inline void GotoNextLine(std::istream &is) {`
- L4492: function `detectRefinementContext`: `auto AMRSimulation<problem_t>::detectRefinementContext(const amrex::BoxArray &restart_ba, const amrex::Geometry &current_geom) -> AMRSimulation<problem_t>::RefinementContext {`
- L4528: function `readCheckpointHeader`: `template <typename problem_t> auto AMRSimulation<problem_t>::readCheckpointHeader(const std::string &restart_file) -> amrex::Vector<amrex::BoxArray> {`
- L4590: function `interpolateMultiFabFromRestart`: `void AMRSimulation<problem_t>::interpolateMultiFabFromRestart(amrex::MultiFab &target, const amrex::MultiFab &source, const RefinementContext &context, const amrex::Geometry &coarse_geom, const amrex::Geometry &fine_geom, const amrex::Vector<amrex::BCRec> &bcs) {`
- L4614: function `interpolateFaceMultiFabFromRestart`: `void AMRSimulation<problem_t>::interpolateFaceMultiFabFromRestart(int lev, const RefinementContext &context, const amrex::Vector<amrex::Geometry> &restart_geom, amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> &restart_fc) {`
- L4711: function `loadMultiFabData`: `template <typename problem_t> void AMRSimulation<problem_t>::loadMultiFabData(const RefinementContext &context) {`
- L4767: function `loadBalanceOnRestart`: `template <typename problem_t> auto AMRSimulation<problem_t>::loadBalanceOnRestart(const amrex::BoxArray &input_ba, int lev) -> amrex::BoxArray {`
- L4794: function `ReadCheckpointFile`: `template <typename problem_t> void AMRSimulation<problem_t>::ReadCheckpointFile() {`
- L4880: function `restartParticleContainerWithRefinement`: `void AMRSimulation<problem_t>::restartParticleContainerWithRefinement(std::unique_ptr<ParticleContainer> &particles, std::string const &restart_chkfile, std::string const &particle_type_name, amrex::Vector<amrex::BoxArray> const &header_box_arrays) {`
- L4976: function `initializeParticleContainerFromCheckpoint`: `void AMRSimulation<problem_t>::initializeParticleContainerFromCheckpoint(std::unique_ptr<ContainerType> &container, amrex::Vector<amrex::BoxArray> const &header_box_arrays) {`
- L4992: function `amrex::Print`: `amrex::Print() << std::format("Splitting {`
- L5004: function `writeFaceVelocitiesToDisk`: `void AMRSimulation<problem_t>::writeFaceVelocitiesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &faceVelArrays, int lev, int timestep) {`
- L5050: function `amrex::Loop`: `amrex::Loop(bx, [&](int i, int j, int k) {`
- L5066: function `writeReconstructedStatesToDisk`: `void AMRSimulation<problem_t>::writeReconstructedStatesToDisk(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &leftState, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &rightState, int lev, int timestep) {`

## `src/turbulence/TurbDataReader.cpp`

- L18: function `read_dataset`: `auto read_dataset(hid_t &file_id, char const *dataset_name) -> amrex::Table3D<double> {`
- L54: function `initialize_turbdata`: `void initialize_turbdata(turb_data &data, std::string &data_file) {`
- L57: function `amrex::Print`: `amrex::Print() << std::format("data_file: {`
- L74: function `get_tabledata`: `auto get_tabledata(amrex::Table3D<double> &in_t) -> amrex::TableData<double, 3> {`
- L95: function `computeRms`: `auto computeRms(amrex::TableData<amrex::Real, 3> &dvx, amrex::TableData<amrex::Real, 3> &dvy, amrex::TableData<amrex::Real, 3> &dvz) -> amrex::Real {`

## `src/turbulence/TurbDataReader.hpp`

- L51: function `initialize_turbdata`: `void initialize_turbdata(turb_data &data, std::string &data_file);`
- L53: function `read_dataset`: `auto read_dataset(hid_t &file_id, char const *dataset_name) -> amrex::Table3D<double>;`
- L55: function `get_tabledata`: `auto get_tabledata(amrex::Table3D<double> &in_t) -> amrex::TableData<double, 3>;`
- L57: function `computeRms`: `auto computeRms(amrex::TableData<amrex::Real, 3> &dvx, amrex::TableData<amrex::Real, 3> &dvy, amrex::TableData<amrex::Real, 3> &dvz) -> amrex::Real;`

## `src/turbulence/TurbulentDriving.hpp`

- L46: function `calculate_dispersion`: `template <typename problem_t> auto calculate_dispersion(amrex::MultiFab &state) -> amrex::GpuArray<amrex::Real, 3>;`
- L48: class `turbulentDriving`: `template <typename problem_t> class turbulentDriving`
- L55: function `update`: `void update(const amrex::Real &time, amrex::MultiFab &state) {`
- L66: function `turbulentDriving`: `turbulentDriving() = default;`
- L67: function `turbulentDriving`: `explicit turbulentDriving(const std::map<std::string, std::string> &turb_params) {`
- L69: function `applyDriving`: `auto applyDriving(amrex::MultiFab &state, const amrex::Real time, const amrex::Real dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes) -> bool {`
- L107: function `calculate_dispersion`: `template <typename problem_t> auto calculate_dispersion(amrex::MultiFab &state) -> amrex::GpuArray<amrex::Real, 3> {`

## `src/util/ArrayUtil.hpp`

- L13: function `strided_vector_from`: `template <typename T> auto strided_vector_from(std::vector<T> &v, int stride) -> std::vector<T> {`

## `src/util/ArrayView_2d.hpp`

- L17: enum class `FluxDir`: `enum class FluxDir { X1 = 0, X2 = 1, X3 = 2 };`
- L21: function `reorderMultiIndex`: `template <FluxDir N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex(int, int, int);`
- L27: struct `Array4View`: `template <class T, FluxDir N, class Enable = void> struct Array4View {`
- L31: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L37: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X1, std::enable_if_t<!std::is_const_v<T>>> {`
- L41: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L43: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & {`
- L45: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & {`
- L49: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X1, std::enable_if_t<std::is_const_v<T>>> {`
- L53: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L55: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T {`
- L57: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T {`
- L63: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X2, std::enable_if_t<!std::is_const_v<T>>> {`
- L67: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L69: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & {`
- L71: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & {`
- L75: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X2, std::enable_if_t<std::is_const_v<T>>> {`
- L79: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L81: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T {`
- L83: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T {`

## `src/util/ArrayView_3d.hpp`

- L18: enum class `FluxDir`: `enum class FluxDir { X1 = 0, X2 = 1, X3 = 2 };`
- L22: function `reorderMultiIndex`: `template <FluxDir N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex(int, int, int);`
- L30: struct `Array4View`: `template <class T, FluxDir N, class Enable = void> struct Array4View {`
- L34: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L40: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X1, std::enable_if_t<!std::is_const_v<T>>> {`
- L44: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L46: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & {`
- L48: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & {`
- L52: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X1, std::enable_if_t<std::is_const_v<T>>> {`
- L56: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L58: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T {`
- L60: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T {`
- L66: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X2, std::enable_if_t<!std::is_const_v<T>>> {`
- L70: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L72: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & {`
- L74: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & {`
- L78: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X2, std::enable_if_t<std::is_const_v<T>>> {`
- L82: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L84: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T {`
- L86: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T {`
- L92: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X3, std::enable_if_t<!std::is_const_v<T>>> {`
- L96: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L98: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & {`
- L100: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & {`
- L104: struct `Array4View`: `template <class T> struct Array4View<T, FluxDir::X3, std::enable_if_t<std::is_const_v<T>>> {`
- L108: function `Array4View`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {`
- L110: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T {`
- L112: function `operator`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T {`

## `src/util/BC.hpp`

- L19: enum `mathematicalBndryTypesInteger`: `enum mathematicalBndryTypesInteger : int {`
- L85: function `isNormalComponent`: `template <typename problem_t> constexpr auto isNormalComponent(int n, int dim) -> bool {`
- L131: function `BC`: `template <typename problem_t> auto BC(int bc_x, int bc_y, int bc_z) -> amrex::Vector<amrex::BCRec> {`
- L163: function `BC`: `template <typename problem_t> auto BC(int bc) -> amrex::Vector<amrex::BCRec> {`
- L167: function `BC_cc`: `auto BC_cc(BCType::mathematicalBndryTypes bc_x, BCType::mathematicalBndryTypes bc_y, BCType::mathematicalBndryTypes bc_z) -> amrex::Vector<amrex::BCRec> {`
- L199: function `BC_fc`: `auto BC_fc(BCType::mathematicalBndryTypes bc_x, BCType::mathematicalBndryTypes bc_y, BCType::mathematicalBndryTypes bc_z) -> amrex::Vector<amrex::BCRec> {`

## `src/util/CheckNaN.hpp`

- L18: function `CheckSymmetryArray`: `AMREX_GPU_HOST_DEVICE auto CheckSymmetryArray(amrex::Array4<const amrex::Real> const & , amrex::Box const & , const int , amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ) -> bool {`
- L25: function `CheckSymmetryFluxes`: `AMREX_GPU_HOST_DEVICE auto CheckSymmetryFluxes(amrex::Array4<const amrex::Real> const & , amrex::Array4<const amrex::Real> const & , amrex::Box const & , const int , amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ) -> bool {`
- L33: function `CheckNaN`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CheckNaN(amrex::FArrayBox const &arr, amrex::Box const & , amrex::Box const &nanRange, const int ncomp, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ) {`

## `src/util/DataTable.hpp`

- L41: struct `InterpData`: `template <int Ndim> struct InterpData {`
- L46: function `InterpData`: `AMREX_GPU_HOST_DEVICE InterpData() = default;`
- L50: struct `DataTableGpuConst`: `template <int Ndim, int Nout = 1, OutOfBounds oob_policy = OutOfBounds::clamp> struct DataTableGpuConst {`
- L420: class `DataTable`: `template <int Ndim, int Nout = 1, OutOfBounds oob_policy = OutOfBounds::clamp> class DataTable`
- L452: function `bcastScalar`: `template <typename T> static void bcastScalar(T &value) {`
- L454: function `bcastArray`: `template <typename T, std::size_t N> static void bcastArray(std::array<T, N> &values) {`
- L459: function `bcastString`: `static void bcastString(std::string &value) {`
- L471: function `bcastStringArray`: `template <std::size_t N> static void bcastStringArray(std::array<std::string, N> &values) {`
- L478: function `bcastVector`: `template <typename T> static void bcastVector(amrex::Vector<T> &values) {`
- L490: function `bcastSpacingType`: `static void bcastSpacingType(SpacingType &value) {`
- L499: function `bcastSpacingTypes`: `static void bcastSpacingTypes(std::array<SpacingType, Ndim> &values) {`
- L533: function `setMetadata`: `void setMetadata(const std::array<std::string, Ndim> &input_names, const std::array<std::string, Nout> &output_names, const std::array<std::string, Ndim> &input_units, const std::array<std::string, Nout> &output_units, SpacingType output_spacing) {`
- L545: function `DataTable`: `DataTable() = default;`
- L549: function `DataTable`: `DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const std::array<amrex::Vector<amrex::Vector<amrex::Real>>, Nout> &data) {`
- L556: function `DataTable`: `DataTable(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const amrex::Vector<amrex::Vector<amrex::Real>> &data) requires(N == 1) {`
- L563: function `~DataTable`: `~DataTable() = default;`
- L566: function `DataTable`: `DataTable(DataTable &&) = default;`
- L567: function `operator=`: `auto operator=(DataTable &&) -> DataTable & = default;`
- L570: function `DataTable`: `DataTable(const DataTable &) = delete;`
- L571: function `operator=`: `auto operator=(const DataTable &) -> DataTable & = delete;`
- L574: function `initialize`: `void initialize(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const amrex::Vector<amrex::Vector<amrex::Real>> &data) {`
- L798: function `initializeStorage`: `void initializeStorage(const std::array<amrex::Real, Ndim> &x_mins, const std::array<amrex::Real, Ndim> &x_maxs, const std::array<int, Ndim> &n_xs, const std::array<SpacingType, Ndim> &spacing_types) {`
- L849: function `fillDataTables`: `template <typename DataType> void fillDataTables(const DataType &data) {`
- L886: function `fillDataTablesFlat`: `void fillDataTablesFlat(const amrex::Vector<amrex::Real> &flat_data) {`
- L927: function `initializeCommonFlat`: `void initializeCommonFlat(const std::array<amrex::Real, Ndim> &x_mins, const std::array<amrex::Real, Ndim> &x_maxs, const std::array<int, Ndim> &n_xs, const std::array<SpacingType, Ndim> &spacing_types, const amrex::Vector<amrex::Real> &flat_data) {`
- L937: function `initialize_common`: `void initialize_common(const std::array<amrex::Real, Ndim> &x_mins, const std::array<amrex::Real, Ndim> &x_maxs, const std::array<int, Ndim> &n_xs, const std::array<SpacingType, Ndim> &spacing_types, const std::array<amrex::Vector<amrex::Real>, Ndim> & , const DataType &data) {`
- L946: function `initialize_common`: `template <typename DataType> void initialize_common(const std::array<amrex::Vector<amrex::Real>, Ndim> &coords, const DataType &data) {`
- L998: function `CSVReader`: `static auto CSVReader(const std::string &file_path, SpacingType output_spacing) -> DataTable {`
- L1282: function `H5Reader`: `static auto H5Reader(const std::string &file_path, const std::string &dataset_path, const std::vector<std::string> &coord_names, int is_fast_log = 0, std::array<std::pair<amrex::Real, amrex::Real>, Ndim> *coord_bounds = nullptr, bool *include_pe = nullptr) -> DataTable {`

## `src/util/Optional.hpp`

- L17: class `optional`: `template <typename T> class optional`
- L26: function `optional`: `AMREX_GPU_HOST_DEVICE constexpr optional() noexcept = default;`
- L30: function `optional`: `AMREX_GPU_HOST_DEVICE constexpr optional(const T &value) : has_value_(true) {`
- L33: function `optional`: `AMREX_GPU_HOST_DEVICE optional(const optional &other) : has_value_(other.has_value_) {`
- L41: function `optional`: `AMREX_GPU_HOST_DEVICE optional(optional &&other) noexcept : has_value_(other.has_value_) {`
- L51: function `operator=`: `AMREX_GPU_HOST_DEVICE auto operator=(const optional &other) -> optional & {`
- L66: function `operator=`: `AMREX_GPU_HOST_DEVICE auto operator=(optional &&other) noexcept -> optional & {`
- L83: function `~optional`: `AMREX_GPU_HOST_DEVICE ~optional() {`
- L91: function `bool`: `AMREX_GPU_HOST_DEVICE constexpr explicit operator bool() const noexcept {`
- L94: function `operator*`: `AMREX_GPU_HOST_DEVICE constexpr auto operator*() const & noexcept -> const T & {`

## `src/util/fextract.cpp`

- L17: function `fextract`: `auto fextract(MultiFab &mf, Geometry &geom, const int idir, const Real slice_coord, const bool center = false) -> std::tuple<Vector<Real>, Vector<Gpu::HostVector<Real>>> {`
- L235: function `std::sort`: `std::sort(p.begin(), p.end(), [&](size_t i, size_t j) {`

## `src/util/fextract.hpp`

- L12: function `fextract`: `auto fextract(amrex::MultiFab &mf, amrex::Geometry &geom, int idir, amrex::Real slice_coord, bool center = false) -> std::tuple<amrex::Vector<amrex::Real>, amrex::Vector<amrex::Gpu::HostVector<amrex::Real>>>;`

## `src/util/richardson.hpp`

- L25: struct `Parameters`: `struct Parameters {`
- L35: function `applyQuietDefaults`: `inline void applyQuietDefaults() {`
- L52: function `run`: `template <typename Callable> auto run(const Parameters &params, Callable &&runTest) -> int {`
- L60: function `amrex::Print`: `amrex::Print() << std::format("Running Richardson convergence test for {`
- L72: function `amrex::Print`: `amrex::Print() << std::format(" {`
- L80: function `amrex::Print`: `amrex::Print() << std::format("\nReached maximum resolution (nx = {`
- L94: function `amrex::Print`: `amrex::Print() << std::format(" {`
- L102: function `amrex::Print`: `amrex::Print() << std::format(" {`
- L114: function `amrex::Print`: `amrex::Print() << std::format(" nx= {`
- L121: function `amrex::Print`: `amrex::Print() << std::format("\nOverall convergence rate: {`
- L122: function `amrex::Print`: `amrex::Print() << std::format("Expected rate: {`
- L137: function `amrex::Print`: `amrex::Print() << std::format("\nConvergence data written to {`
- L142: function `amrex::Print`: `amrex::Print() << std::format("\n✓ Richardson convergence test PASSED (target error {`

## `src/util/time_units.hpp`

- L26: function `registerTimeUnitConstants`: `inline void registerTimeUnitConstants() {`

## `src/util/valarray.hpp`

- L23: class `valarray`: `template <typename T, int d> class valarray`
- L26: function `valarray`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray() = default;`
- L30: function `valarray`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(std::initializer_list<T> list) {`
- L49: function `operator[]`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) -> T & {`
- L51: function `operator[]`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) const -> T {`
- L55: function `fillin`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void fillin(T const &scalar) {`
- L80: function `operator+`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d> {`
- L90: function `operator+`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d> {`
- L100: function `operator+`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d> {`
- L111: function `operator-`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d> {`
- L122: function `operator*`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d> {`
- L133: function `operator/`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d> {`
- L143: function `operator*`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d> {`
- L153: function `operator*`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d> {`
- L163: function `operator*=`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator*=(quokka::valarray<T, d> &v, T const &scalar) {`
- L171: function `operator+=`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator+=(quokka::valarray<T, d> &a, quokka::valarray<T, d> const &b) {`
- L179: function `operator/`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d> {`
- L189: function `operator/`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d> {`
- L199: function `operator/=`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator/=(quokka::valarray<T, d> &v, T const &scalar) {`
- L207: function `abs`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto abs(quokka::valarray<T, d> const &v) -> quokka::valarray<T, d> {`
- L217: function `min`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto min(quokka::valarray<T, d> const &v) -> T {`
- L229: function `max`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto max(quokka::valarray<T, d> const &v) -> T {`
- L241: function `sum`: `template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sum(quokka::valarray<T, d> const &v) -> T {`
- L252: function `operator>`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<bool, d> {`
- L263: function `operator>`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(quokka::valarray<T, d> const &a, T const &scalar) -> quokka::valarray<bool, d> {`
- L274: function `operator<`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<bool, d> {`
- L285: function `operator<`: `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(quokka::valarray<T, d> const &a, T const &scalar) -> quokka::valarray<bool, d> {`
