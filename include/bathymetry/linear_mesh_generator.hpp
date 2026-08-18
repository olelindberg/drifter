#pragma once

/// @file linear_mesh_generator.hpp
/// @brief Adaptive 2D mesh generation with error-driven refinement

#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/linear_mesh_error_estimator.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/lowrider_config.hpp"
#include "mesh/geotiff_reader.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace drifter {

/// @brief Adaptive mesh generator with linear (bilinear) elements
///
/// Generates 2D quadrilateral meshes for seabed surfaces using
/// error-driven adaptive refinement. The mesh is refined where
/// the bilinear surface approximation error exceeds a threshold.
class LinearMeshGenerator {
public:
    /// @brief Construct with domain bounds and configuration
    LinearMeshGenerator(Real xmin, Real xmax, Real ymin, Real ymax,
                        int nx, int ny, const LowriderRefinementConfig& config);

    /// @brief Load bathymetry from GeoTIFF file
    /// @param geotiff_path Path to GeoTIFF file
    void load_bathymetry(const std::string& geotiff_path);

    /// @brief Load multi-source bathymetry (drifter-style)
    /// @param data_config Data configuration with primary file and tiles
    void load_bathymetry(const LowriderDataConfig& data_config);

    /// @brief Set bathymetry evaluation function directly
    /// @param depth_func Function returning depth at (x, y)
    /// @param land_mask Function returning true if point is land
    void set_bathymetry_functions(
        std::function<Real(Real, Real)> depth_func,
        std::function<bool(Real, Real)> land_mask);

    /// @brief Run adaptive refinement loop
    /// @return Final result with statistics
    LowriderAdaptiveResult solve_adaptive();

    /// @brief Perform single adaptation iteration
    /// @return True if refinement occurred
    bool adapt_once();

    /// @brief Get the current mesh
    const QuadtreeAdapter& mesh() const { return mesh_; }

    /// @brief Get the current surface
    const LinearBezierSurface& surface() const;

    /// @brief Get bathymetry data
    const BathymetryData& bathymetry() const;

    /// @brief Write VTK output
    /// @param filename Output filename (without extension)
    void write_vtk(const std::string& filename) const;

    /// @brief Get current error estimates
    std::vector<LinearMeshElementError> get_errors() const;

    /// @brief Get maximum error
    Real max_error() const;

    /// @brief Get mean error
    Real mean_error() const;

private:
    QuadtreeAdapter mesh_;
    std::unique_ptr<LinearBezierSurface> surface_;
    std::shared_ptr<BathymetryData> bathymetry_;
    std::function<Real(Real, Real)> depth_func_;      ///< Depth evaluation function
    std::function<bool(Real, Real)> land_mask_func_;  ///< Land mask function
    LowriderRefinementConfig config_;
    int iteration_ = 0;

    /// @brief Rebuild surface after mesh changes
    void rebuild_surface();

    /// @brief Perform single adaptation iteration with pre-computed errors
    /// @param errors Pre-computed error estimates
    /// @return True if refinement occurred
    bool adapt_once(const std::vector<LinearMeshElementError>& errors);

    /// @brief Compute maximum error from pre-computed errors
    Real max_error_from(const std::vector<LinearMeshElementError>& errors) const;

    /// @brief Compute mean error from pre-computed errors
    Real mean_error_from(const std::vector<LinearMeshElementError>& errors) const;

    /// @brief Select elements for refinement using Dorfler marking
    /// @param errors Per-element error estimates
    /// @return Element indices to refine
    std::vector<Index> select_for_refinement(const std::vector<LinearMeshElementError>& errors) const;

    /// @brief Refine selected elements
    /// @param elements_to_refine Element indices
    void refine_elements(const std::vector<Index>& elements_to_refine);

    /// @brief Check stopping criteria
    /// @param max_err Current maximum error
    /// @return Convergence reason, or empty string if not converged
    std::string check_convergence(Real max_err) const;
};

} // namespace drifter
