#include "DiagVolumeRender.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include <algorithm>

void DiagVolumeRender::init(const std::string &a_prefix, std::string_view a_diagName)
{
	DiagBase::init(a_prefix, a_diagName);

	if (!m_filters.empty()) {
		amrex::Print() << "DiagVolumeRender: filters are not supported and will be ignored.\n";
	}

	amrex::ParmParse const pp(a_prefix);

	if (pp.query("field", m_fieldName) == 0) {
		amrex::Abort("DiagVolumeRender requires a 'field' parameter.");
	}

	pp.query("width", m_width);
	pp.query("height", m_height);
	pp.query("box_transparency", m_boxTransparency);
	pp.query("antialiasing", m_antialiasing);
	pp.query("visibility_graph", m_visibilityGraph);
	pp.query("write_visibility_graph", m_writeVisibilityGraph);
	pp.query("min_level", m_minLevel);
	pp.query("max_level", m_maxLevel);
	pp.query("log_scale_input", m_logScaleInput);
	pp.query("output_ext", m_outputExt);

	if (!m_outputExt.empty() && m_outputExt.front() == '.') {
		m_outputExt.erase(0, 1);
	}
	if (m_outputExt.empty()) {
		m_outputExt = "png";
	}

	if (pp.countval("scalar_range") > 0) {
		amrex::Vector<amrex::Real> range;
		pp.getarr("scalar_range", range, 0, 2);
		if (range.size() != 2) {
			amrex::Abort("DiagVolumeRender scalar_range must have exactly two values.");
		}
		auto const minVal = static_cast<float>(std::min(range[0], range[1]));
		auto const maxVal = static_cast<float>(std::max(range[0], range[1]));
		m_scalarRange = std::make_pair(minVal, maxVal);
	}

	bool const hasColorMap = (pp.countval("color_map_values") > 0) || (pp.countval("color_map_r") > 0) || (pp.countval("color_map_g") > 0) ||
				 (pp.countval("color_map_b") > 0) || (pp.countval("color_map_a") > 0);
	if (hasColorMap) {
		if ((pp.countval("color_map_values") == 0) || (pp.countval("color_map_r") == 0) || (pp.countval("color_map_g") == 0) ||
		    (pp.countval("color_map_b") == 0) || (pp.countval("color_map_a") == 0)) {
			amrex::Abort("DiagVolumeRender color_map_* requires color_map_values, color_map_r, color_map_g, color_map_b, and color_map_a.");
		}

		amrex::Vector<amrex::Real> values;
		amrex::Vector<amrex::Real> reds;
		amrex::Vector<amrex::Real> greens;
		amrex::Vector<amrex::Real> blues;
		amrex::Vector<amrex::Real> alphas;
		pp.getarr("color_map_values", values);
		pp.getarr("color_map_r", reds);
		pp.getarr("color_map_g", greens);
		pp.getarr("color_map_b", blues);
		pp.getarr("color_map_a", alphas);

		auto const count = values.size();
		if (count < 2) {
			amrex::Abort("DiagVolumeRender color_map_* must provide at least two control points.");
		}
		if ((reds.size() != count) || (greens.size() != count) || (blues.size() != count) || (alphas.size() != count)) {
			amrex::Abort("DiagVolumeRender color_map_* arrays must have matching lengths.");
		}

		VolumeRenderer::ColorMap controlPoints;
		controlPoints.reserve(count);
		for (std::size_t i = 0; i < count; ++i) {
			VolumeRenderer::ColorMapControlPoint point{};
			point.value = static_cast<float>(values[i]);
			point.red = static_cast<float>(reds[i]);
			point.green = static_cast<float>(greens[i]);
			point.blue = static_cast<float>(blues[i]);
			point.alpha = static_cast<float>(alphas[i]);
			controlPoints.push_back(point);
		}
		m_colorMap = std::move(controlPoints);
	}

	if (pp.countval("up_vector") > 0) {
		amrex::Vector<amrex::Real> upVec;
		pp.getarr("up_vector", upVec, 0, AMREX_SPACEDIM);
		if (upVec.size() != AMREX_SPACEDIM) {
			amrex::Abort("DiagVolumeRender up_vector must match AMREX_SPACEDIM.");
		}
		m_upVector = amrex::RealVect(AMREX_D_DECL(upVec[0], upVec[1], upVec[2]));
	}

	bool const hasCameraParams = (pp.countval("camera_eye") > 0) || (pp.countval("camera_look_at") > 0) || (pp.countval("camera_up") > 0) ||
				     (pp.countval("camera_fov_y_degrees") > 0) || (pp.countval("camera_near") > 0) || (pp.countval("camera_far") > 0);
	if (hasCameraParams) {
		bool const hasEye = (pp.countval("camera_eye") > 0);
		bool const hasLookAt = (pp.countval("camera_look_at") > 0);
		if (hasEye && !hasLookAt) {
			amrex::Abort("DiagVolumeRender camera_eye requires camera_look_at.");
		}
		if (hasLookAt && !hasEye) {
			amrex::Abort("DiagVolumeRender camera_look_at requires camera_eye.");
		}

		if (hasEye) {
			amrex::Vector<amrex::Real> eye;
			pp.getarr("camera_eye", eye, 0, AMREX_SPACEDIM);
			if (eye.size() != AMREX_SPACEDIM) {
				amrex::Abort("DiagVolumeRender camera_eye must match AMREX_SPACEDIM.");
			}
			m_cameraEye = amrex::RealVect(AMREX_D_DECL(eye[0], eye[1], eye[2]));
		}
		if (hasLookAt) {
			amrex::Vector<amrex::Real> lookAt;
			pp.getarr("camera_look_at", lookAt, 0, AMREX_SPACEDIM);
			if (lookAt.size() != AMREX_SPACEDIM) {
				amrex::Abort("DiagVolumeRender camera_look_at must match AMREX_SPACEDIM.");
			}
			m_cameraLookAt = amrex::RealVect(AMREX_D_DECL(lookAt[0], lookAt[1], lookAt[2]));
		}
		if (pp.countval("camera_up") > 0) {
			amrex::Vector<amrex::Real> up;
			pp.getarr("camera_up", up, 0, AMREX_SPACEDIM);
			if (up.size() != AMREX_SPACEDIM) {
				amrex::Abort("DiagVolumeRender camera_up must match AMREX_SPACEDIM.");
			}
			m_cameraUp = amrex::RealVect(AMREX_D_DECL(up[0], up[1], up[2]));
		}

		pp.query("camera_fov_y_degrees", m_cameraFovYDegrees);
		pp.query("camera_near", m_cameraNear);
		pp.query("camera_far", m_cameraFar);
		m_hasCamera = hasEye && hasLookAt;
		if (!m_hasCamera && pp.countval("camera_up") > 0) {
			m_upVector = m_cameraUp;
		}
	}

	amrex::Print() << "DiagVolumeRender initialized: field=" << m_fieldName << ", file=" << m_diagfile << ", interval=" << m_interval << "\n";
}

void DiagVolumeRender::prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
			       const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames)
{
	if (first_time) {
		DiagBase::prepare(a_nlevels, a_geoms, a_grids, a_dmap, a_varNames);
		first_time = false;
	}
}

void DiagVolumeRender::addVars(amrex::Vector<std::string> &a_varList)
{
	DiagBase::addVars(a_varList);
	a_varList.push_back(m_fieldName);
}

auto DiagVolumeRender::outputFilename(int a_nstep) const -> std::string
{
	std::string base = amrex::Concatenate(m_diagfile, a_nstep, 7);
	if (m_outputExt.empty()) {
		return base;
	}
	return base + "." + m_outputExt;
}
