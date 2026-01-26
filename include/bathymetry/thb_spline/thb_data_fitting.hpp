#pragma once

#include "bathymetry/biharmonic_assembler.hpp"  // For BathymetrySource
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/thb_spline/thb_hierarchy.hpp"
#include "bathymetry/thb_spline/thb_refinement_mask.hpp"
#include "bathymetry/thb_spline/thb_truncation.hpp"
#include "core/types.hpp"
#include <functional>
#include <vector>

namespace drifter {

/**
 * @brief Data fitting assembler for THB-spline surfaces
 *
 * Builds the weighted least-squares system for fitting THB-spline surfaces
 * to bathymetry data. Uses truncated basis functions for proper partition
 * of unity across refinement levels.
 *
 * Builds: minimize ||B * c - d||²_W
 * where:
 *   B = evaluation matrix of truncated basis functions
 *   c = control point coefficients (active DOFs only)
 *   d = bathymetry data values
 *   W = weight matrix
 */
class THBDataFitting {
  public:
    /**
     * @brief Construct data fitting assembler
     * @param quadtree Adaptive 2D mesh
     * @param hierarchy THB hierarchy with knot vectors
     * @param mask Active function mask
     * @param truncation Truncation coefficients
     */
    THBDataFitting(const QuadtreeAdapter& quadtree, const THBHierarchy& hierarchy,
                   const THBRefinementMask& mask, const THBTruncation& truncation);

    /**
     * @brief Sample from BathymetrySource at Gauss quadrature points
     * @param source Bathymetry data source (e.g., GeoTIFF)
     * @param ngauss Number of Gauss points per direction per element
     */
    void set_from_bathymetry_source(const BathymetrySource& source, int ngauss = 4);

    /**
     * @brief Sample from a function interface
     * @param bathy_func Function taking (x, y) -> depth
     * @param ngauss Number of Gauss points per direction per element
     */
    void set_from_function(std::function<Real(Real, Real)> bathy_func, int ngauss = 4);

    /**
     * @brief Build sparse normal equations: (B^T W B) and (B^T W d)
     *
     * Matrix size is num_active x num_active (only active DOFs).
     *
     * @param AtWA Output: B^T * diag(w) * B (sparse, num_active x num_active)
     * @param AtWb Output: B^T * diag(w) * d (num_active)
     */
    void assemble_normal_equations(SpMat& AtWA, VecX& AtWb) const;

    /// Number of data points
    Index num_points() const { return static_cast<Index>(data_x_.size()); }

    /// Number of active DOFs
    Index num_active_dofs() const { return mask_.num_active(); }

    /// Access data points
    const std::vector<Real>& data_x() const { return data_x_; }
    const std::vector<Real>& data_y() const { return data_y_; }
    const std::vector<Real>& data_z() const { return data_z_; }
    const std::vector<Real>& data_w() const { return data_w_; }

  private:
    const QuadtreeAdapter& quadtree_;
    const THBHierarchy& hierarchy_;
    const THBRefinementMask& mask_;
    const THBTruncation& truncation_;

    // Sampled data points
    std::vector<Real> data_x_;
    std::vector<Real> data_y_;
    std::vector<Real> data_z_;
    std::vector<Real> data_w_;

    /**
     * @brief Evaluate all active truncated basis functions at (x, y)
     * @param x Physical x-coordinate
     * @param y Physical y-coordinate
     * @return Vector of values for each active function
     */
    VecX evaluate_active_basis(Real x, Real y) const;
};

}  // namespace drifter
