#include "bathymetry/gauss_quadrature_error_estimator.hpp"
#include <cmath>

namespace drifter {

GaussQuadratureErrorEstimator::GaussQuadratureErrorEstimator(
    const LinearBezierSurface& surface,
    const BathymetryData& data,
    const QuadtreeAdapter& mesh,
    ErrorMetricType metric,
    int ngauss)
    : surface_(surface), data_(data), mesh_(mesh), metric_(metric), ngauss_(ngauss) {
    init_gauss_quadrature();
}

void GaussQuadratureErrorEstimator::init_gauss_quadrature() {
    // Gauss-Legendre quadrature nodes and weights on [0, 1]
    // (transformed from standard [-1, 1] interval)

    gauss_nodes_.clear();
    gauss_weights_.clear();

    if (ngauss_ == 1) {
        gauss_nodes_ = {0.5};
        gauss_weights_ = {1.0};
    } else if (ngauss_ == 2) {
        Real x = 0.5 / std::sqrt(3.0);
        gauss_nodes_ = {0.5 - x, 0.5 + x};
        gauss_weights_ = {0.5, 0.5};
    } else if (ngauss_ == 3) {
        Real x = 0.5 * std::sqrt(0.6);
        gauss_nodes_ = {0.5 - x, 0.5, 0.5 + x};
        gauss_weights_ = {5.0/18.0, 8.0/18.0, 5.0/18.0};
    } else {
        // ngauss_ == 4 (default)
        Real a = 0.5 * std::sqrt((3.0 - 2.0 * std::sqrt(6.0/5.0)) / 7.0);
        Real b = 0.5 * std::sqrt((3.0 + 2.0 * std::sqrt(6.0/5.0)) / 7.0);
        Real wa = (18.0 + std::sqrt(30.0)) / 72.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 72.0;
        gauss_nodes_ = {0.5 - b, 0.5 - a, 0.5 + a, 0.5 + b};
        gauss_weights_ = {wb/2.0, wa/2.0, wa/2.0, wb/2.0};
    }
}

ElementError GaussQuadratureErrorEstimator::estimate_element(Index elem) const {
    ElementError result;
    result.element = elem;
    result.sample_count = ngauss_ * ngauss_;

    const auto& bounds = mesh_.element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    result.area = dx * dy;

    Real l2_sq = 0.0;  // integral (z_data - z_surf)^2 dA
    Real l1 = 0.0;     // integral |z_data - z_surf| dA

    for (int i = 0; i < ngauss_; ++i) {
        Real x = ref_to_world(gauss_nodes_[i], bounds.xmin, bounds.xmax);
        for (int j = 0; j < ngauss_; ++j) {
            Real y = ref_to_world(gauss_nodes_[j], bounds.ymin, bounds.ymax);
            Real w = gauss_weights_[i] * gauss_weights_[j] * dx * dy;

            // Get surface value
            Real z_surf = surface_.evaluate(x, y);

            // Get data value (depth is positive, surface is negative z)
            Real z_data = -data_.get_depth(x, y);

            Real diff = z_data - z_surf;
            l2_sq += diff * diff * w;
            l1 += std::abs(diff) * w;
        }
    }

    // Compute metric-specific error value
    switch (metric_) {
        case ErrorMetricType::NormalizedError:
            result.error = std::sqrt(l2_sq) / std::sqrt(result.area);  // RMS
            break;
        case ErrorMetricType::MeanDifference:
            result.error = l1 / result.area;
            break;
        case ErrorMetricType::VolumeChange:
            result.error = l1;
            break;
        default:
            result.error = std::sqrt(l2_sq) / std::sqrt(result.area);
    }

    return result;
}

Index GaussQuadratureErrorEstimator::num_elements() const {
    return mesh_.num_elements();
}

Real GaussQuadratureErrorEstimator::ref_to_world(Real ref, Real wmin, Real wmax) {
    return wmin + ref * (wmax - wmin);
}

} // namespace drifter
