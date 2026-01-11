#include "bathymetry/weno_sampler.hpp"
#include <algorithm>
#include <cmath>

namespace drifter {

// =============================================================================
// WENO5Sampler - Construction
// =============================================================================

WENO5Sampler::WENO5Sampler(std::shared_ptr<BathymetryData> raw_data)
    : raw_data_(std::move(raw_data)) {
}

// =============================================================================
// WENO5Sampler - Sampling
// =============================================================================

Real WENO5Sampler::sample(Real x, Real y) const {
    // Convert world coordinates to pixel coordinates
    double px, py;
    raw_data_->world_to_pixel(x, y, px, py);

    // Get 5x5 stencil
    std::array<std::array<Real, 5>, 5> stencil;
    if (!get_stencil(px, py, stencil)) {
        // Fall back to bilinear interpolation for boundary cases
        return raw_data_->get_depth(x, y);
    }

    // Fractional position within central cell
    Real tx = px - std::floor(px);
    Real ty = py - std::floor(py);

    // WENO5 interpolation: first in x (rows), then in y (columns)
    std::array<Real, 5> row_results;
    for (int j = 0; j < 5; ++j) {
        row_results[j] = weno5_interp_1d(stencil[j], tx);
    }

    Real elevation = weno5_interp_1d(row_results, ty);

    // Convert elevation to depth (same logic as BathymetryData::get_depth)
    if (raw_data_->is_depth_positive) {
        // Values are already depth (positive = water)
        return elevation > 0.0 ? elevation : 0.0;
    } else {
        // Values are elevation (negative = water)
        return elevation < 0.0 ? -elevation : 0.0;
    }
}

void WENO5Sampler::sample_gradient(Real x, Real y, Real& dh_dx, Real& dh_dy) const {
    // Use central differences with appropriate step size
    Real pixel_size = std::abs(raw_data_->geotransform[1]);
    Real eps = std::max(pixel_size * 0.5, 1e-6);

    Real h_xp = sample(x + eps, y);
    Real h_xm = sample(x - eps, y);
    Real h_yp = sample(x, y + eps);
    Real h_ym = sample(x, y - eps);

    dh_dx = (h_xp - h_xm) / (2.0 * eps);
    dh_dy = (h_yp - h_ym) / (2.0 * eps);
}

void WENO5Sampler::sample_batch(const std::vector<Vec2>& points,
                               std::vector<Real>& values) const {
    values.resize(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        values[i] = sample(points[i](0), points[i](1));
    }
}

// =============================================================================
// WENO5Sampler - Stencil Extraction
// =============================================================================

bool WENO5Sampler::get_stencil(Real px, Real py,
                               std::array<std::array<Real, 5>, 5>& stencil) const {
    int ix = static_cast<int>(std::floor(px));
    int iy = static_cast<int>(std::floor(py));

    int rows = raw_data_->sizey;
    int cols = raw_data_->sizex;

    // Check if we have enough room for 5x5 stencil (2 pixels on each side)
    if (ix < 2 || ix >= cols - 2 || iy < 2 || iy >= rows - 2) {
        return false;
    }

    // Extract 5x5 stencil
    for (int dj = -2; dj <= 2; ++dj) {
        int j = iy + dj;
        for (int di = -2; di <= 2; ++di) {
            int i = ix + di;
            stencil[dj + 2][di + 2] = raw_data_->elevation[j * cols + i];
        }
    }

    return true;
}

// =============================================================================
// WENO5Sampler - WENO5 Interpolation
// =============================================================================

Real WENO5Sampler::weno5_interp_1d(const std::array<Real, 5>& v, Real t) const {
    // v[0..4] = values at positions -2, -1, 0, 1, 2 relative to interpolation point
    // t in [0, 1] is fractional position within cell [0, 1]

    // Compute smoothness indicators
    std::array<Real, 3> beta = compute_smoothness_indicators(v);

    // Compute nonlinear weights
    std::array<Real, 3> alpha;
    for (int k = 0; k < 3; ++k) {
        alpha[k] = optimal_weights_[k] / std::pow(epsilon_ + beta[k], power_);
    }

    Real alpha_sum = alpha[0] + alpha[1] + alpha[2];
    std::array<Real, 3> omega;
    for (int k = 0; k < 3; ++k) {
        omega[k] = alpha[k] / alpha_sum;
    }

    // Evaluate candidate polynomials
    std::array<Real, 3> p = evaluate_candidates(v, t);

    // Weighted combination
    return omega[0] * p[0] + omega[1] * p[1] + omega[2] * p[2];
}

std::array<Real, 3> WENO5Sampler::compute_smoothness_indicators(
    const std::array<Real, 5>& v) const {
    // Smoothness indicators for WENO5 (Jiang-Shu formulation)
    // These measure the smoothness of each candidate stencil

    std::array<Real, 3> beta;

    // Stencil 0: v[0], v[1], v[2] (left-biased)
    Real d2_0 = v[2] - 2.0 * v[1] + v[0];
    beta[0] = (13.0 / 12.0) * d2_0 * d2_0 +
              0.25 * (v[0] - 4.0 * v[1] + 3.0 * v[2]) *
                     (v[0] - 4.0 * v[1] + 3.0 * v[2]);

    // Stencil 1: v[1], v[2], v[3] (central)
    Real d2_1 = v[3] - 2.0 * v[2] + v[1];
    beta[1] = (13.0 / 12.0) * d2_1 * d2_1 +
              0.25 * (v[1] - v[3]) * (v[1] - v[3]);

    // Stencil 2: v[2], v[3], v[4] (right-biased)
    Real d2_2 = v[4] - 2.0 * v[3] + v[2];
    beta[2] = (13.0 / 12.0) * d2_2 * d2_2 +
              0.25 * (3.0 * v[2] - 4.0 * v[3] + v[4]) *
                     (3.0 * v[2] - 4.0 * v[3] + v[4]);

    return beta;
}

std::array<Real, 3> WENO5Sampler::evaluate_candidates(
    const std::array<Real, 5>& v, Real t) const {
    // Evaluate the three quadratic polynomials at position t in [0, 1]
    // Each polynomial interpolates 3 consecutive points from the stencil

    std::array<Real, 3> p;

    // Polynomial 0: interpolates v[0], v[1], v[2]
    // p0(t) passes through (-2, v[0]), (-1, v[1]), (0, v[2])
    // At t=0, we're at position 0 in the stencil
    Real c0_0 = v[2];
    Real c0_1 = -0.5 * v[0] + 2.0 * v[1] - 1.5 * v[2];
    Real c0_2 = 0.5 * v[0] - v[1] + 0.5 * v[2];
    p[0] = c0_0 + t * (c0_1 + t * c0_2);

    // Polynomial 1: interpolates v[1], v[2], v[3]
    // p1(t) passes through (-1, v[1]), (0, v[2]), (1, v[3])
    Real c1_0 = v[2];
    Real c1_1 = -0.5 * v[1] + 0.5 * v[3];
    Real c1_2 = 0.5 * v[1] - v[2] + 0.5 * v[3];
    p[1] = c1_0 + t * (c1_1 + t * c1_2);

    // Polynomial 2: interpolates v[2], v[3], v[4]
    // p2(t) passes through (0, v[2]), (1, v[3]), (2, v[4])
    Real c2_0 = v[2];
    Real c2_1 = -1.5 * v[2] + 2.0 * v[3] - 0.5 * v[4];
    Real c2_2 = 0.5 * v[2] - v[3] + 0.5 * v[4];
    p[2] = c2_0 + t * (c2_1 + t * c2_2);

    return p;
}

}  // namespace drifter
