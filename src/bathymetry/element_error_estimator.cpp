#include "bathymetry/element_error_estimator.hpp"
#include "bathymetry/gauss_quadrature_error_estimator.hpp"
#include "bathymetry/pixel_max_error_estimator.hpp"
#include "bathymetry/pixel_rmse_estimator.hpp"
#include <algorithm>
#include <cmath>

namespace drifter {

std::vector<ElementError> ElementErrorEstimator::estimate_all() const {
    std::vector<ElementError> errors;
    errors.reserve(num_elements());

    for (Index elem = 0; elem < num_elements(); ++elem) {
        errors.push_back(estimate_element(elem));
    }

    return errors;
}

Real ElementErrorEstimator::max_error() const {
    Real max_err = 0.0;
    for (Index elem = 0; elem < num_elements(); ++elem) {
        auto err = estimate_element(elem);
        if (!std::isnan(err.error)) {
            max_err = std::max(max_err, err.error);
        }
    }
    return max_err;
}

Real ElementErrorEstimator::mean_error() const {
    Real sum = 0.0;
    Index valid_count = 0;

    for (Index elem = 0; elem < num_elements(); ++elem) {
        auto err = estimate_element(elem);
        if (!std::isnan(err.error)) {
            sum += err.error;
            ++valid_count;
        }
    }

    return valid_count > 0 ? sum / static_cast<Real>(valid_count) : 0.0;
}

std::unique_ptr<ElementErrorEstimator> create_error_estimator(
    ErrorMetricType metric,
    const LinearBezierSurface& surface,
    const BathymetryData& data,
    const QuadtreeAdapter& mesh,
    int ngauss) {

    switch (metric) {
        case ErrorMetricType::PixelRMSE:
            return std::make_unique<PixelRMSEEstimator>(surface, data, mesh);

        case ErrorMetricType::PixelMaxError:
            return std::make_unique<PixelMaxErrorEstimator>(surface, data, mesh);

        case ErrorMetricType::NormalizedError:
        case ErrorMetricType::MeanDifference:
        case ErrorMetricType::VolumeChange:
        default:
            return std::make_unique<GaussQuadratureErrorEstimator>(
                surface, data, mesh, metric, ngauss);
    }
}

} // namespace drifter
