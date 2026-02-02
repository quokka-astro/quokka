#ifndef DIAGVOLUMERENDER_H
#define DIAGVOLUMERENDER_H

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "AMReX_RealVect.H"
#include "AMReX_Vector.H"
#include "DiagBase.H"
#include <VolumeRenderer/VolumeRendererApi.hpp>

class DiagVolumeRender : public DiagBase::Register<DiagVolumeRender>
{
      public:
	static auto identifier() -> std::string { return "VolumeRender"; }

	void init(const std::string &a_prefix, std::string_view a_diagName) override;

	void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
		     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames) override;

	void addVars(amrex::Vector<std::string> &a_varList) override;

	void close() override {}

	// Template implementation - called from AMRSimulation::doDiagnostics()
	template <typename problem_t> void processDiag(int a_nstep, const amrex::Real &a_time);

      private:
	[[nodiscard]] auto outputFilename(int a_nstep) const -> std::string;

	std::string m_fieldName;
	int m_width{512};
	int m_height{512};
	float m_boxTransparency{0.0F};
	int m_antialiasing{1};
	bool m_visibilityGraph{true};
	bool m_writeVisibilityGraph{false};
	int m_minLevel{0};
	int m_maxLevel{-1};
	bool m_logScaleInput{false};
	std::string m_outputExt{"png"};
	std::optional<std::pair<float, float>> m_scalarRange;
	std::optional<amrex::RealVect> m_upVector;
	std::optional<VolumeRenderer::ColorMap> m_colorMap;
	bool m_hasCamera{false};
	amrex::RealVect m_cameraEye{AMREX_D_DECL(0.0, 0.0, 0.0)};
	amrex::RealVect m_cameraLookAt{AMREX_D_DECL(0.0, 0.0, 0.0)};
	amrex::RealVect m_cameraUp{AMREX_D_DECL(0.0, 1.0, 0.0)};
	float m_cameraFovYDegrees{45.0F};
	float m_cameraNear{0.01F};
	float m_cameraFar{1000.0F};
};

template <typename problem_t> void DiagVolumeRender::processDiag(int a_nstep, const amrex::Real & /*a_time*/)
{
	const BL_PROFILE("DiagVolumeRender::processDiag()");

	if (AMREX_SPACEDIM != 3) {
		amrex::Abort("DiagVolumeRender requires AMREX_SPACEDIM=3.");
	}

	amrVolumeRenderer::api::AmrData data;
	data.levelData = *m_diagMF;
	data.levelGeometry = *m_geoms;
	data.refinementRatios = *m_refRatio;

	amrVolumeRenderer::api::RenderOptions options;
	options.width = m_width;
	options.height = m_height;
	options.boxTransparency = m_boxTransparency;
	options.antialiasing = m_antialiasing;
	options.visibilityGraph = m_visibilityGraph;
	options.writeVisibilityGraph = m_writeVisibilityGraph;
	options.minLevel = m_minLevel;
	options.maxLevel = m_maxLevel;
	options.logScaleInput = m_logScaleInput;
	options.component = getFieldIndex(m_fieldName, *m_diagVars);
	options.outputFilename = outputFilename(a_nstep);
	if (m_scalarRange.has_value()) {
		options.scalarRange = *m_scalarRange;
	}
	if (m_colorMap.has_value()) {
		options.colorMap = *m_colorMap;
	}
	if (m_upVector.has_value()) {
		options.upVector = *m_upVector;
	}
	if (m_hasCamera) {
		VolumeRenderer::CameraParameters camera{};
		camera.eye = m_cameraEye;
		camera.lookAt = m_cameraLookAt;
		camera.up = m_cameraUp;
		camera.fovYDegrees = m_cameraFovYDegrees;
		camera.nearPlane = m_cameraNear;
		camera.farPlane = m_cameraFar;
		options.camera = camera;
	}

	int const exitCode = amrVolumeRenderer::api::Render(data, options);
	if (exitCode != 0) {
		amrex::Abort("DiagVolumeRender failed with exit code " + std::to_string(exitCode));
	}
}

#endif // DIAGVOLUMERENDER_H
