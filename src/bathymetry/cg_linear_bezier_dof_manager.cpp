#include "bathymetry/cg_linear_bezier_dof_manager.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace drifter {

CGLinearBezierDofManager::CGLinearBezierDofManager(const QuadtreeAdapter &mesh)
    : CGBezierDofManagerBase(mesh) {

    Index num_elements = mesh_.num_elements();
    if (num_elements == 0) {
        num_global_dofs_ = 0;
        num_free_dofs_ = 0;
        return;
    }

    // Compute domain bounds and quantization tolerance for mesh-relative DOF
    // sharing
    Real xmin = std::numeric_limits<Real>::max();
    Real xmax = std::numeric_limits<Real>::lowest();
    Real ymin = std::numeric_limits<Real>::max();
    Real ymax = std::numeric_limits<Real>::lowest();
    Real min_element_size = std::numeric_limits<Real>::max();

    for (Index e = 0; e < num_elements; ++e) {
        const auto &b = mesh_.element_bounds(e);
        xmin = std::min(xmin, b.xmin);
        xmax = std::max(xmax, b.xmax);
        ymin = std::min(ymin, b.ymin);
        ymax = std::max(ymax, b.ymax);
        min_element_size = std::min(min_element_size, std::min(b.xmax - b.xmin, b.ymax - b.ymin));
    }

    xmin_domain_ = xmin;
    ymin_domain_ = ymin;
    // Scale: resolve positions to 1e-8 of minimum element size
    inv_quantization_tol_ = 1.0 / (min_element_size * 1e-8);

    initialize_elem_to_global(num_elements, LinearBezierBasis2D::NDOF);

    // For linear elements, all DOFs are corners - single pass suffices
    assign_vertex_dofs();
    identify_boundary_dofs_impl([this](int edge) { return basis_.edge_dofs(edge); });
    build_hanging_node_constraints();

    // Reorder DOFs by Morton Z-curve for better spatial locality
    auto perm = reorder_dofs_by_morton();
    if (!perm.empty()) {
        for (auto &c : constraints_) {
            c.slave_dof = perm[c.slave_dof];
            for (auto &m : c.master_dofs) {
                m = perm[m];
            }
        }
    }

    build_dof_mappings();
}

// =============================================================================
// Position handling
// =============================================================================

std::pair<int64_t, int64_t> CGLinearBezierDofManager::quantize_position(const Vec2 &pos) const {
    // Normalize to domain origin to reduce floating-point error for large
    // coords
    Real x_rel = pos(0) - xmin_domain_;
    Real y_rel = pos(1) - ymin_domain_;
    return std::make_pair(static_cast<int64_t>(std::round(x_rel * inv_quantization_tol_)),
                          static_cast<int64_t>(std::round(y_rel * inv_quantization_tol_)));
}

Vec2 CGLinearBezierDofManager::get_dof_position(Index elem, int local_dof) const {
    const auto &bounds = mesh_.element_bounds(elem);
    Vec2 param = basis_.control_point_position(local_dof);
    Real x = bounds.xmin + param(0) * (bounds.xmax - bounds.xmin);
    Real y = bounds.ymin + param(1) * (bounds.ymax - bounds.ymin);
    return Vec2(x, y);
}

// =============================================================================
// DOF assignment (all DOFs are corners for linear elements)
// =============================================================================

void CGLinearBezierDofManager::assign_vertex_dofs() {
    // For linear elements, all 4 DOFs are corners
    // DOF layout:
    //   [1]──[3]   (v=1)
    //    │    │
    //   [0]──[2]   (v=0)
    //   u=0  u=1

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int local_dof = 0; local_dof < LinearBezierBasis2D::NDOF; ++local_dof) {
            Vec2 pos = get_dof_position(e, local_dof);

            Index dof = find_dof_at_position(pos);
            if (dof < 0) {
                dof = register_dof_at_position(pos);
            }

            elem_to_global_[e][local_dof] = dof;
        }
    }
}

// =============================================================================
// Hanging node constraints
// =============================================================================

void CGLinearBezierDofManager::build_hanging_node_constraints() {
    constraints_.clear();
    constrained_dofs_.clear();

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int edge = 0; edge < 4; ++edge) {
            EdgeNeighborInfo info = mesh_.get_neighbor(elem, edge);

            if (info.type != EdgeNeighborInfo::Type::FineToCoarse) {
                continue;
            }

            Index coarse_elem = info.neighbor_elements[0];
            int coarse_edge = info.neighbor_edges[0];

            std::vector<int> fine_dofs = basis_.edge_dofs(edge);
            std::vector<int> coarse_dofs = basis_.edge_dofs(coarse_edge);

            // Determine parameter range on coarse edge
            Real t_start, t_end;
            if (info.subedge_index == 0) {
                t_start = 0.0;
                t_end = 0.5;
            } else {
                t_start = 0.5;
                t_end = 1.0;
            }

            // Compute extraction matrix for subdivision
            MatX S = basis_.compute_1d_extraction_matrix(t_start, t_end);

            // For linear: fine_dofs has 2 DOFs (k=0, k=1)
            // One is shared with coarse, one is at T-junction (midpoint)
            for (int k = 0; k < static_cast<int>(fine_dofs.size()); ++k) {
                int fine_local = fine_dofs[k];
                Index fine_global = elem_to_global_[elem][fine_local];

                // Check if this is a shared corner (not a T-junction)
                bool is_shared = false;
                if (info.subedge_index == 0) {
                    is_shared = (k == 0); // First DOF is shared
                } else {
                    is_shared = (k == 1); // Last DOF is shared
                }

                if (is_shared) {
                    // Ensure the DOF is properly shared
                    int coarse_local = coarse_dofs[info.subedge_index == 0 ? 0 : 1];
                    elem_to_global_[elem][fine_local] = elem_to_global_[coarse_elem][coarse_local];
                    continue;
                }

                // T-junction point (midpoint of coarse edge)
                if (constrained_dofs_.count(fine_global) > 0) {
                    continue;
                }

                LinearHangingNodeConstraint constraint;
                constraint.slave_dof = fine_global;

                // Weights from extraction matrix row k
                VecX weights = S.row(k);

                for (int m = 0; m < static_cast<int>(coarse_dofs.size()); ++m) {
                    int coarse_local = coarse_dofs[m];
                    Index coarse_global = elem_to_global_[coarse_elem][coarse_local];

                    if (std::abs(weights(m)) > 1e-14) {
                        constraint.master_dofs.push_back(coarse_global);
                        constraint.weights.push_back(weights(m));
                    }
                }

                if (!constraint.master_dofs.empty()) {
                    constraints_.push_back(constraint);
                    constrained_dofs_.insert(fine_global);
                }
            }
        }
    }
}

// =============================================================================
// Base class virtual method implementation
// =============================================================================

void CGLinearBezierDofManager::get_constraint_triplets(
    std::vector<Eigen::Triplet<Real>> &triplets) const {
    triplets.reserve(num_constraints() * 3); // Each constraint: 1 slave + ~2 masters

    for (Index row = 0; row < num_constraints(); ++row) {
        const auto &c = constraints_[row];
        triplets.emplace_back(row, c.slave_dof, 1.0);
        for (size_t i = 0; i < c.master_dofs.size(); ++i) {
            triplets.emplace_back(row, c.master_dofs[i], -c.weights[i]);
        }
    }
}

} // namespace drifter
