#pragma once

// WENO5Sampler - Weighted Essentially Non-Oscillatory sampling for bathymetry
//
// Implements 5th-order WENO interpolation for sampling bathymetry data at
// arbitrary points. WENO provides high-order accuracy in smooth regions while
// remaining non-oscillatory near discontinuities (cliffs, seamounts, etc.).
//
// Usage:
//   WENO5Sampler sampler(bathy_data);
//   Real depth = sampler.sample(x, y);

#include "mesh/geotiff_reader.hpp"
#include <array>
#include <memory>

namespace drifter {

/// @brief WENO5 stencil-based bathymetry sampler
///
/// Provides high-order essentially non-oscillatory sampling of bathymetry
/// data using 5-point WENO reconstruction. Samples directly from raw
/// bathymetry data without requiring a wavelet pyramid.
class WENO5Sampler {
public:
    /// @brief Construct sampler with bathymetry data
    /// @param raw_data Bathymetry data to sample from
    explicit WENO5Sampler(std::shared_ptr<BathymetryData> raw_data);

    /// @brief Sample bathymetry depth at a world coordinate
    /// @param x World x-coordinate
    /// @param y World y-coordinate
    /// @return Interpolated depth value
    Real sample(Real x, Real y) const;

    /// @brief Sample bathymetry gradient at a world coordinate
    /// @param x World x-coordinate
    /// @param y World y-coordinate
    /// @param dh_dx Output: depth gradient in x
    /// @param dh_dy Output: depth gradient in y
    void sample_gradient(Real x, Real y, Real& dh_dx, Real& dh_dy) const;

    /// @brief Batch sample at multiple points (more efficient)
    /// @param points Vector of (x, y) world coordinates
    /// @param values Output: interpolated values
    void sample_batch(const std::vector<Vec2>& points,
                     std::vector<Real>& values) const;

    /// @brief Set WENO regularization parameter
    /// @param eps Small value to prevent division by zero (default: 1e-6)
    void set_epsilon(Real eps) { epsilon_ = eps; }

    /// @brief Set WENO smoothness indicator power
    /// @param p Power for nonlinear weights (default: 2.0)
    void set_power(Real p) { power_ = p; }

    /// @brief Get current epsilon value
    Real epsilon() const { return epsilon_; }

    /// @brief Get current power value
    Real power() const { return power_; }

    /// @brief Access the raw bathymetry data
    const BathymetryData& raw_data() const { return *raw_data_; }

private:
    std::shared_ptr<BathymetryData> raw_data_;

    Real epsilon_ = 1e-6;
    Real power_ = 2.0;

    // WENO optimal weights for 5th-order accuracy
    static constexpr std::array<Real, 3> optimal_weights_ = {0.1, 0.6, 0.3};

    /// @brief Get 5x5 stencil from raw data centered at pixel position
    /// @param px Pixel x-coordinate (fractional)
    /// @param py Pixel y-coordinate (fractional)
    /// @param stencil Output: 5x5 stencil values [row][col]
    /// @return true if stencil is valid, false if out of bounds
    bool get_stencil(Real px, Real py,
                     std::array<std::array<Real, 5>, 5>& stencil) const;

    /// @brief 1D WENO5 interpolation
    /// @param values 5 stencil values centered at interpolation point
    /// @param t Fractional position within central cell [0, 1]
    /// @return WENO5 interpolated value
    Real weno5_interp_1d(const std::array<Real, 5>& values, Real t) const;

    /// @brief Compute smoothness indicators for WENO
    /// @param values 5 stencil values
    /// @return 3 smoothness indicators for the 3 candidate stencils
    std::array<Real, 3> compute_smoothness_indicators(
        const std::array<Real, 5>& values) const;

    /// @brief Evaluate candidate polynomials at position t
    /// @param values 5 stencil values
    /// @param t Fractional position [0, 1]
    /// @return 3 polynomial values from the 3 candidate stencils
    std::array<Real, 3> evaluate_candidates(
        const std::array<Real, 5>& values, Real t) const;
};

}  // namespace drifter
