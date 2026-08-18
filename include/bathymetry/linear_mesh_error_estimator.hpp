#pragma once

/// @file linear_mesh_error_estimator.hpp
/// @brief Error estimation for adaptive refinement of linear meshes

#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/lowrider_config.hpp"
#include "mesh/geotiff_reader.hpp"
#include <vector>

namespace drifter {

/// @brief Per-element error estimate for linear mesh
struct LinearMeshElementError {
    Index element;            ///< Element index
    Real l2_error;            ///< ||z_data - z_surface||_L2
    Real normalized_error;    ///< L2 / sqrt(area) = RMS
    Real mean_difference;     ///< integral |z_data - z_surface| dA / area
    Real volume_error;        ///< integral |z_data - z_surface| dA
    Real area;                ///< Element area
};

/// @brief Error estimator using Gauss quadrature for linear meshes
class LinearMeshErrorEstimator {
public:
    /// @brief Construct error estimator
    /// @param surface Linear Bezier surface to evaluate
    /// @param data Bathymetry data from GeoTIFF
    /// @param ngauss Number of Gauss points per direction (default 4)
    LinearMeshErrorEstimator(const LinearBezierSurface& surface,
                             const BathymetryData& data,
                             int ngauss = 4);

    /// @brief Estimate error for all elements
    /// @return Vector of per-element error estimates
    std::vector<LinearMeshElementError> estimate_all() const;

    /// @brief Estimate error for single element
    /// @param elem Element index
    /// @return Error estimate for this element
    LinearMeshElementError estimate_element(Index elem) const;

    /// @brief Get maximum error across all elements
    /// @param metric Which error metric to use
    /// @return Maximum error value
    Real max_error(ErrorMetricType metric = ErrorMetricType::NormalizedError) const;

    /// @brief Get mean error across all elements
    /// @param metric Which error metric to use
    /// @return Mean error value
    Real mean_error(ErrorMetricType metric = ErrorMetricType::NormalizedError) const;

    /// @brief Get error value based on metric type
    /// @param err Element error
    /// @param metric Which error metric
    /// @return The selected metric value
    static Real get_metric(const LinearMeshElementError& err, ErrorMetricType metric);

private:
    const LinearBezierSurface& surface_;
    const BathymetryData& data_;
    int ngauss_;
    std::vector<Real> gauss_nodes_;
    std::vector<Real> gauss_weights_;

    /// @brief Initialize Gauss quadrature nodes and weights
    void init_gauss_quadrature();

    /// @brief Map reference coordinate [0,1] to world coordinate
    static Real ref_to_world(Real ref, Real wmin, Real wmax);
};

} // namespace drifter
