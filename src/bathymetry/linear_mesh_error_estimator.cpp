#include "bathymetry/linear_mesh_error_estimator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace drifter {

LinearMeshErrorEstimator::LinearMeshErrorEstimator(const LinearBezierSurface& surface,
                                                   const BathymetryData& data,
                                                   int ngauss)
    : surface_(surface), data_(data), ngauss_(ngauss) {
    init_gauss_quadrature();
}

void LinearMeshErrorEstimator::init_gauss_quadrature() {
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

std::vector<LinearMeshElementError> LinearMeshErrorEstimator::estimate_all() const {
    std::vector<LinearMeshElementError> errors;
    errors.reserve(surface_.mesh().num_elements());

    for (Index elem = 0; elem < surface_.mesh().num_elements(); ++elem) {
        errors.push_back(estimate_element(elem));
    }

    return errors;
}

LinearMeshElementError LinearMeshErrorEstimator::estimate_element(Index elem) const {
    LinearMeshElementError result;
    result.element = elem;

    const auto& bounds = surface_.mesh().element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    result.area = dx * dy;

    // Integrate error using Gauss quadrature
    Real l2_sq = 0.0;      // integral (z_data - z_surf)^2 dA
    Real l1 = 0.0;         // integral |z_data - z_surf| dA

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

    result.l2_error = std::sqrt(l2_sq);
    result.normalized_error = result.l2_error / std::sqrt(result.area);
    result.mean_difference = l1 / result.area;
    result.volume_error = l1;

    return result;
}

Real LinearMeshErrorEstimator::max_error(ErrorMetricType metric) const {
    Real max_err = 0.0;
    for (Index elem = 0; elem < surface_.mesh().num_elements(); ++elem) {
        auto err = estimate_element(elem);
        max_err = std::max(max_err, get_metric(err, metric));
    }
    return max_err;
}

Real LinearMeshErrorEstimator::mean_error(ErrorMetricType metric) const {
    Real sum = 0.0;
    for (Index elem = 0; elem < surface_.mesh().num_elements(); ++elem) {
        auto err = estimate_element(elem);
        sum += get_metric(err, metric);
    }
    return sum / surface_.mesh().num_elements();
}

Real LinearMeshErrorEstimator::get_metric(const LinearMeshElementError& err, ErrorMetricType metric) {
    switch (metric) {
        case ErrorMetricType::NormalizedError:
            return err.normalized_error;
        case ErrorMetricType::MeanDifference:
            return err.mean_difference;
        case ErrorMetricType::VolumeError:
            return err.volume_error;
        default:
            return err.normalized_error;
    }
}

Real LinearMeshErrorEstimator::ref_to_world(Real ref, Real wmin, Real wmax) {
    return wmin + ref * (wmax - wmin);
}

} // namespace drifter
