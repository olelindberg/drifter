#pragma once

// AdaptiveBathymetry - Main interface for adaptive bathymetry filtering
//
// Uses WENO5 or bilinear sampling for interpolation directly from raw
// bathymetry data, followed by L2 projection onto Bernstein basis for
// bounded polynomial representation on DG elements.
//
// Usage:
//   AdaptiveBathymetry adaptive(bathy_ptr);
//   adaptive.set_sampling_method(SamplingMethod::WENO5);  // or Bilinear
//   VecX coeffs = adaptive.project_element(bounds, order);

#include "bathymetry/weno_sampler.hpp"
#include "dg/bernstein_basis.hpp"
#include "dg/quadrature_3d.hpp"
#include "mesh/octree_adapter.hpp"
#include <map>
#include <memory>

namespace drifter {

/// @brief Sampling method for bathymetry interpolation
enum class SamplingMethod {
    Bilinear,  ///< Simple bilinear interpolation (fast)
    WENO5      ///< WENO5 interpolation (smooth, non-oscillatory)
};

// Forward declaration
class SeabedSurface;

/// @brief Adaptive bathymetry filtering with WENO5 sampling and Bernstein projection
///
/// This class provides the main interface for projecting bathymetry data onto
/// DG mesh elements using:
/// 1. WENO5 sampling for smooth, non-oscillatory interpolation from raw data
/// 2. L2 projection onto Bernstein basis for bounded polynomial representation
class AdaptiveBathymetry {
public:
    /// @brief Construct adaptive bathymetry from raw data
    /// @param raw_data Bathymetry data to sample from
    explicit AdaptiveBathymetry(std::shared_ptr<BathymetryData> raw_data);

    /// @brief Project bathymetry onto a single element
    /// @param bounds Element bounds in world coordinates
    /// @param order Polynomial order for Bernstein representation
    /// @return Bernstein coefficients (order+1)^2 values for 2D bottom face
    VecX project_element(const ElementBounds& bounds, int order) const;

    /// @brief Project bathymetry onto all bottom elements of a mesh
    /// @param mesh The octree mesh
    /// @param order Polynomial order
    /// @param seabed Output: SeabedSurface with projected coefficients
    void project_to_seabed(const OctreeAdapter& mesh, int order,
                          SeabedSurface& seabed) const;

    /// @brief Estimate L2 projection error for an element
    /// @param bounds Element bounds
    /// @param coeffs Bernstein coefficients
    /// @param order Polynomial order
    /// @return Estimated error norm
    Real estimate_projection_error(const ElementBounds& bounds,
                                   const VecX& coeffs, int order) const;

    /// @brief Access the WENO sampler
    const WENO5Sampler& sampler() const { return *sampler_; }

    /// @brief Access the raw bathymetry data
    const BathymetryData& raw_data() const { return *raw_data_; }

    /// @brief Set overintegration factor for L2 projection
    /// @param factor Multiplier for quadrature order (default: 2)
    void set_overintegration_factor(int factor) { overintegration_factor_ = factor; }

    /// @brief Set the sampling method
    /// @param method SamplingMethod::Bilinear or SamplingMethod::WENO5
    void set_sampling_method(SamplingMethod method) { sampling_method_ = method; }

    /// @brief Get the current sampling method
    SamplingMethod sampling_method() const { return sampling_method_; }

private:
    std::shared_ptr<BathymetryData> raw_data_;
    std::unique_ptr<WENO5Sampler> sampler_;

    int overintegration_factor_ = 2;
    SamplingMethod sampling_method_ = SamplingMethod::WENO5;

    // Cached projection matrices and quadrature rules per order
    mutable std::map<int, GaussQuadrature2D> quad_cache_;
    mutable std::map<int, MatX> projection_matrix_cache_;
    mutable std::map<int, BernsteinBasis1D> basis_cache_;

    /// @brief Get or create 2D quadrature rule for given order
    const GaussQuadrature2D& get_quadrature(int order) const;

    /// @brief Get or create Bernstein basis for given order
    const BernsteinBasis1D& get_basis(int order) const;

    /// @brief Get or create L2 projection matrix for given order
    /// The projection matrix P satisfies: coeffs = P * f_quad
    /// where f_quad are function values at quadrature points
    const MatX& get_projection_matrix(int order) const;

    /// @brief Build L2 projection matrix
    /// @param order Polynomial order
    /// @return Matrix mapping quadrature values to Bernstein coefficients
    MatX build_projection_matrix(int order) const;

    /// @brief Sample bathymetry at quadrature points for an element
    /// @param bounds Element bounds
    /// @param quad Quadrature rule
    /// @return Values at quadrature points
    VecX sample_at_quadrature(const ElementBounds& bounds,
                             const GaussQuadrature2D& quad) const;

    /// @brief Evaluate Bernstein polynomial at a point
    /// @param coeffs Bernstein coefficients
    /// @param xi, eta Reference coordinates [-1, 1]^2
    /// @param order Polynomial order
    /// @return Evaluated value
    Real evaluate_bernstein(const VecX& coeffs, Real xi, Real eta, int order) const;
};

}  // namespace drifter
