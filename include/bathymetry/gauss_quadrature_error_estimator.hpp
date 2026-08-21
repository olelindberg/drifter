#pragma once

/// @file gauss_quadrature_error_estimator.hpp
/// @brief Gauss quadrature-based error estimation for adaptive refinement

#include "bathymetry/element_error_estimator.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/geotiff_reader.hpp"
#include <vector>

namespace drifter {

/// @brief Error estimator using Gauss quadrature integration
///
/// Computes error integrals over elements using Gauss-Legendre quadrature.
/// Supports multiple error metrics:
/// - NormalizedError: ||z_data - z_surface||_L2 / sqrt(area) = RMS [meters]
/// - MeanDifference: integral |z_data - z_surface| dA / area [meters]
/// - VolumeChange: integral |z_data - z_surface| dA [m³]
class GaussQuadratureErrorEstimator : public ElementErrorEstimator {
public:
    /// @brief Construct Gauss quadrature error estimator
    /// @param surface Linear Bezier surface to evaluate
    /// @param data Bathymetry data from GeoTIFF
    /// @param mesh Quadtree mesh
    /// @param metric Error metric type (NormalizedError, MeanDifference, VolumeChange)
    /// @param ngauss Number of Gauss points per direction (1-4, default 4)
    GaussQuadratureErrorEstimator(const LinearBezierSurface& surface,
                                   const BathymetryData& data,
                                   const QuadtreeAdapter& mesh,
                                   ErrorMetricType metric,
                                   int ngauss = 4);

    /// @brief Estimate error for a single element
    ElementError estimate_element(Index elem) const override;

    /// @brief Get number of elements
    Index num_elements() const override;

private:
    const LinearBezierSurface& surface_;
    const BathymetryData& data_;
    const QuadtreeAdapter& mesh_;
    ErrorMetricType metric_;
    int ngauss_;
    std::vector<Real> gauss_nodes_;
    std::vector<Real> gauss_weights_;

    /// @brief Initialize Gauss quadrature nodes and weights on [0,1]
    void init_gauss_quadrature();

    /// @brief Map reference coordinate [0,1] to world coordinate
    static Real ref_to_world(Real ref, Real wmin, Real wmax);
};

} // namespace drifter
