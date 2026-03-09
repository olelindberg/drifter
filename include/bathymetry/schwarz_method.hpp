#pragma once

/// @file schwarz_method.hpp
/// @brief Block Schwarz iterative methods (multiplicative, additive, colored)

#include "bathymetry/iterative_method.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>
#include <vector>

namespace drifter {

/// @brief Base class for Schwarz methods with common data
///
/// All Schwarz methods operate on element blocks, requiring:
/// - Element-to-DOF mappings
/// - Pre-factorized element blocks
class SchwarzMethodBase : public IIterativeMethod {
public:
    /// @brief Construct Schwarz method base
    /// @param Q System matrix (reference, must outlive this object)
    /// @param element_free_dofs Element DOF lists (indices into Q)
    /// @param element_block_lu Pre-factorized element blocks
    SchwarzMethodBase(
        const SpMat &Q,
        const std::vector<std::vector<Index>> &element_free_dofs,
        const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu);

    /// @brief Get the system matrix
    const SpMat &matrix() const override { return Q_; }

    /// @brief Get number of elements
    size_t num_elements() const { return element_free_dofs_.size(); }

protected:
    const SpMat &Q_;
    const std::vector<std::vector<Index>> &element_free_dofs_;
    const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu_;
};

/// @brief Multiplicative Schwarz (Block Gauss-Seidel) method
///
/// Applies element corrections sequentially, immediately updating the
/// global residual after each element. This provides Gauss-Seidel style
/// information exchange but is expensive due to incremental Qx updates.
class MultiplicativeSchwarzMethod : public SchwarzMethodBase {
public:
    using SchwarzMethodBase::SchwarzMethodBase;

    /// @brief Apply multiplicative Schwarz iterations
    void apply(VecX &x, const VecX &b, int iters) const override;
};

/// @brief Additive Schwarz (Jacobi-style block) method
///
/// Accumulates all element corrections before applying them in a single
/// damped update. This is parallelizable and cheaper than multiplicative,
/// but provides weaker smoothing.
class AdditiveSchwarzMethod : public SchwarzMethodBase {
public:
    /// @brief Construct additive Schwarz method
    /// @param Q System matrix
    /// @param element_free_dofs Element DOF lists
    /// @param element_block_lu Pre-factorized element blocks
    /// @param omega Damping parameter (typically 0.8)
    AdditiveSchwarzMethod(
        const SpMat &Q,
        const std::vector<std::vector<Index>> &element_free_dofs,
        const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu,
        Real omega = 0.8);

    /// @brief Apply additive Schwarz iterations
    void apply(VecX &x, const VecX &b, int iters) const override;

    /// @brief Get the damping parameter
    Real omega() const { return omega_; }

private:
    Real omega_;
};

/// @brief Colored Multiplicative Schwarz method
///
/// Uses graph coloring to batch elements that don't share DOFs.
/// Within each color: Jacobi-style (corrections use same residual).
/// Between colors: Gauss-Seidel style (updated x used).
/// Provides good convergence while being partially parallelizable.
class ColoredSchwarzMethod : public SchwarzMethodBase {
public:
    /// @brief Construct colored Schwarz method
    /// @param Q System matrix
    /// @param element_free_dofs Element DOF lists
    /// @param element_block_lu Pre-factorized element blocks
    /// @param elements_by_color Element indices grouped by color
    ColoredSchwarzMethod(
        const SpMat &Q,
        const std::vector<std::vector<Index>> &element_free_dofs,
        const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu,
        const std::vector<std::vector<Index>> &elements_by_color);

    /// @brief Apply colored Schwarz iterations
    void apply(VecX &x, const VecX &b, int iters) const override;

    /// @brief Get number of colors
    int num_colors() const {
        return static_cast<int>(elements_by_color_.size());
    }

private:
    const std::vector<std::vector<Index>> &elements_by_color_;
};

} // namespace drifter
