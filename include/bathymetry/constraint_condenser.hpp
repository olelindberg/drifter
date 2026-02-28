#pragma once

/// @file constraint_condenser.hpp
/// @brief Shared utilities for constraint condensation in CG Bezier smoothers
///
/// Provides functions to condense sparse matrices and RHS vectors by eliminating
/// slave DOFs (hanging node constraints). Used by both linear and cubic Bezier
/// bathymetry smoothers.

#include "core/types.hpp"
#include <functional>
#include <utility>
#include <vector>

namespace drifter {

/// @brief Assemble KKT system [[Q, A^T], [A, -εI]]
///
/// Builds the saddle-point system for constrained optimization:
/// [[Q,   A^T ]   [x]     [b]
///  [A,  -εI ]] * [λ]  =  [0]
///
/// @param Q System matrix (n × n)
/// @param A Constraint matrix (m × n)
/// @param b RHS vector (n)
/// @param constraint_reg Small regularization for constraint block (default 1e-10)
/// @return Pair of (KKT matrix, rhs vector)
std::pair<SpMat, VecX> assemble_kkt(const SpMat &Q, const SpMat &A, const VecX &b,
                                    Real constraint_reg = 1e-10);

/// @brief Condense sparse matrix and RHS by eliminating slave DOFs
///
/// Given an expand_dof function that maps global DOF indices to (free_index, weight) pairs,
/// transforms Q and c to a reduced system on free DOFs only.
///
/// For a free DOF g, expand_dof returns {(free_index, 1.0)}.
/// For a slave DOF g = sum(w_i * master_i), expand_dof returns {(free_i, w_i), ...}
/// where free_i is the free index of master_i.
///
/// The condensed system satisfies: Q_reduced * x_free = c_reduced
/// where the original system was: Q * x = c with constraints x_slave = sum(w_i * x_master_i)
///
/// @param Q Input sparse matrix (num_global x num_global)
/// @param c Input RHS vector (num_global)
/// @param expand_dof Callable: Index -> vector<pair<Index, Real>> mapping global to free DOFs
/// @param num_free Number of free (non-slave) DOFs
/// @param Q_reduced Output condensed matrix (num_free x num_free)
/// @param c_reduced Output condensed RHS (num_free)
void condense_matrix_and_rhs(
    const SpMat &Q, const VecX &c,
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof, Index num_free,
    SpMat &Q_reduced, VecX &c_reduced);

/// @brief Back-substitute to recover slave DOF values from masters
///
/// Template to work with both LinearHangingNodeConstraint and CubicHangingNodeConstraint.
/// Both constraint types have the same interface: slave_dof, master_dofs, weights.
///
/// @tparam ConstraintT Constraint type with slave_dof, master_dofs, weights members
/// @param solution Solution vector to update (must be sized for all global DOFs)
/// @param constraints Vector of hanging node constraints
template <typename ConstraintT>
void back_substitute_slaves(VecX &solution, const std::vector<ConstraintT> &constraints) {
    for (const auto &hc : constraints) {
        Real val = 0.0;
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            val += hc.weights[i] * solution(hc.master_dofs[i]);
        }
        solution(hc.slave_dof) = val;
    }
}

} // namespace drifter
