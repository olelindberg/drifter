#pragma once

/// @file bezier_data_fitting.hpp
/// @brief Data fitting assembler for Bezier bathymetry surface fitting
///
/// Builds the weighted least-squares matrices for fitting Bezier patches
/// to bathymetry data. Supports both scattered XYZ point clouds and
/// gridded data from BathymetrySource (e.g., GeoTIFF).

#include "core/types.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/bezier_basis_2d.hpp"
#include "bathymetry/biharmonic_assembler.hpp"  // For BathymetrySource
#include <memory>
#include <vector>

namespace drifter {

/// @brief Data point for scattered bathymetry fitting
struct BathymetryPoint {
    Real x, y;     ///< Horizontal position
    Real z;        ///< Depth/elevation
    Real weight;   ///< Optional weight (default 1.0)

    BathymetryPoint() : x(0), y(0), z(0), weight(1.0) {}
    BathymetryPoint(Real x_, Real y_, Real z_, Real w = 1.0)
        : x(x_), y(y_), z(z_), weight(w) {}
};

/// @brief Assembles data fitting matrices for Bezier surface least squares
///
/// Creates evaluation matrix B, RHS vector b, and weight vector w for:
///   minimize ||B * coeffs - b||²_W
/// where W = diag(w).
class BezierDataFittingAssembler {
public:
    /// @brief Construct assembler for a mesh
    /// @param mesh 2D quadtree mesh
    explicit BezierDataFittingAssembler(const QuadtreeAdapter& mesh);

    // =========================================================================
    // Data input methods
    // =========================================================================

    /// @brief Set scattered data points
    /// @param points Vector of (x, y, z, weight) points
    void set_scattered_points(const std::vector<BathymetryPoint>& points);

    /// @brief Set scattered data from Vec3 (x, y, z), uniform weights
    /// @param points Vector of (x, y, z) points
    void set_scattered_points(const std::vector<Vec3>& points);

    /// @brief Sample from BathymetrySource at Gauss quadrature points
    /// @param source Bathymetry data source (e.g., GeoTIFF)
    /// @param ngauss Number of Gauss points per direction per element
    void set_from_bathymetry_source(const BathymetrySource& source, int ngauss = 4);

    /// @brief Sample from BathymetrySource using a function interface
    /// @param bathy_func Function taking (x, y) -> depth
    /// @param ngauss Number of Gauss points per direction per element
    void set_from_function(std::function<Real(Real, Real)> bathy_func, int ngauss = 4);

    // =========================================================================
    // Assembly
    // =========================================================================

    /// @brief Assemble the data fitting matrices
    ///
    /// Builds:
    ///   - B: Evaluation matrix (num_points x total_dofs)
    ///         B[p, k] = basis function k evaluated at point p
    ///   - b: RHS vector (num_points) with z-values
    ///   - w: Weight vector (num_points) for weighted least squares
    ///
    /// @param B Output sparse evaluation matrix
    /// @param b Output RHS vector
    /// @param w Output weight vector
    void assemble(SpMat& B, VecX& b, VecX& w) const;

    /// @brief Build normal equations directly: (B^T W B) and (B^T W b)
    ///
    /// For small to medium problems, this is more efficient than
    /// storing the full B matrix.
    ///
    /// @param AtWA Output: B^T * diag(w) * B (total_dofs x total_dofs)
    /// @param AtWb Output: B^T * diag(w) * b (total_dofs)
    void assemble_normal_equations(MatX& AtWA, VecX& AtWb) const;

    /// @brief Build normal equations with sparse output (memory-efficient)
    ///
    /// Same as assemble_normal_equations() but returns sparse matrix directly
    /// without allocating dense intermediate storage. This avoids the 26.4 GB
    /// dense allocation for large meshes (e.g., 40×40 = 1,600 elements).
    ///
    /// The normal matrix B^T W B is block-diagonal sparse with only 0.06%
    /// non-zeros (36×36 blocks per element), so sparse storage uses ~16 MB
    /// instead of 26.4 GB for a 57,600 DOF system.
    ///
    /// @param AtWA Output: B^T * diag(w) * B (sparse, total_dofs x total_dofs)
    /// @param AtWb Output: B^T * diag(w) * b (total_dofs)
    void assemble_normal_equations_sparse(SpMat& AtWA, VecX& AtWb) const;

    // =========================================================================
    // Queries
    // =========================================================================

    /// @brief Number of data points
    Index num_points() const { return static_cast<Index>(points_.size()); }

    /// @brief Number of DOFs per element (36)
    int dofs_per_element() const { return BezierBasis2D::NDOF; }

    /// @brief Total number of DOFs
    Index total_dofs() const {
        return dofs_per_element() * mesh_.num_elements();
    }

    /// @brief Get global DOF index
    Index global_dof(Index elem, int local_dof) const {
        return elem * dofs_per_element() + local_dof;
    }

    /// @brief Get the data points
    const std::vector<BathymetryPoint>& points() const { return points_; }

    /// @brief Check if data has been set
    bool has_data() const { return !points_.empty(); }

    /// @brief Evaluate bathymetry at a point
    ///
    /// Uses the stored bathymetry function if available, otherwise
    /// returns 0 (no interpolation from scattered points).
    /// @param x, y Physical coordinates
    /// @return Bathymetry depth value
    Real evaluate_bathymetry(Real x, Real y) const {
        if (bathy_func_) {
            return bathy_func_(x, y);
        }
        return 0.0;  // No function stored, fallback
    }

    /// @brief Check if bathymetry function is available for point evaluation
    bool has_bathymetry_function() const { return bathy_func_ != nullptr; }

private:
    const QuadtreeAdapter& mesh_;
    std::unique_ptr<BezierBasis2D> basis_;

    /// Stored bathymetry function for point evaluation (optional)
    std::function<Real(Real, Real)> bathy_func_;

    /// Data points
    std::vector<BathymetryPoint> points_;

    /// Mapping from point index to containing element
    std::vector<Index> point_elements_;

    /// Gauss-Legendre nodes and weights for sampling
    VecX gauss_nodes_;
    VecX gauss_weights_;

    /// Assign points to elements
    void assign_points_to_elements();

    /// Find element containing a point
    Index find_element(Real x, Real y) const;

    /// Map physical position to element parameter space [0,1]^2
    Vec2 physical_to_param(Index elem, Real x, Real y) const;

    /// Compute Gauss-Legendre quadrature on [0,1]
    void compute_gauss_quadrature(int ngauss);
};

}  // namespace drifter
