#include "bathymetry/cg_cubic_bezier_dof_manager.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace drifter {

static constexpr Real POSITION_SCALE = 1e8;

CGCubicBezierDofManager::CGCubicBezierDofManager(const QuadtreeAdapter &mesh)
    : CGBezierDofManagerBase(mesh) {

    Index num_elements = mesh_.num_elements();
    if (num_elements == 0) {
        num_global_dofs_ = 0;
        num_free_dofs_ = 0;
        return;
    }

    initialize_elem_to_global(num_elements, CubicBezierBasis2D::NDOF);

    assign_vertex_dofs();
    assign_edge_dofs();
    assign_interior_dofs();
    assign_edge_dofs_nonconforming();
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

std::pair<int64_t, int64_t> CGCubicBezierDofManager::quantize_position(const Vec2 &pos) const {
    return std::make_pair(static_cast<int64_t>(std::round(pos(0) * POSITION_SCALE)),
                          static_cast<int64_t>(std::round(pos(1) * POSITION_SCALE)));
}

Vec2 CGCubicBezierDofManager::get_dof_position(Index elem, int local_dof) const {
    const auto &bounds = mesh_.element_bounds(elem);
    Vec2 param = basis_.control_point_position(local_dof);
    Real x = bounds.xmin + param(0) * (bounds.xmax - bounds.xmin);
    Real y = bounds.ymin + param(1) * (bounds.ymax - bounds.ymin);
    return Vec2(x, y);
}

// =============================================================================
// DOF classification (cubic: 4×4 grid, corners at 0, 3, 12, 15)
// =============================================================================

bool CGCubicBezierDofManager::is_corner_dof(int local_dof) const {
    // Corners at (i,j) = (0,0), (3,0), (0,3), (3,3)
    // DOF indices: 0, 12, 3, 15
    return local_dof == 0 || local_dof == 3 || local_dof == 12 || local_dof == 15;
}

bool CGCubicBezierDofManager::is_edge_dof(int local_dof) const {
    int i, j;
    CubicBezierBasis2D::dof_ij(local_dof, i, j);
    bool on_boundary = (i == 0 || i == 3 || j == 0 || j == 3);
    return on_boundary && !is_corner_dof(local_dof);
}

bool CGCubicBezierDofManager::is_interior_dof(int local_dof) const {
    int i, j;
    CubicBezierBasis2D::dof_ij(local_dof, i, j);
    return (i > 0 && i < 3 && j > 0 && j < 3);
}

int CGCubicBezierDofManager::get_edge_for_dof(int local_dof) const {
    int i, j;
    CubicBezierBasis2D::dof_ij(local_dof, i, j);

    if (i == 0 && j > 0 && j < 3)
        return 0; // Left edge
    if (i == 3 && j > 0 && j < 3)
        return 1; // Right edge
    if (j == 0 && i > 0 && i < 3)
        return 2; // Bottom edge
    if (j == 3 && i > 0 && i < 3)
        return 3; // Top edge

    return -1;
}

// =============================================================================
// DOF assignment (4-pass algorithm)
// =============================================================================

void CGCubicBezierDofManager::assign_vertex_dofs() {
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int corner = 0; corner < 4; ++corner) {
            int local_dof = basis_.corner_dof(corner);
            Vec2 pos = get_dof_position(e, local_dof);

            Index dof = find_dof_at_position(pos);
            if (dof < 0) {
                dof = register_dof_at_position(pos);
            }

            elem_to_global_[e][local_dof] = dof;
        }
    }
}

void CGCubicBezierDofManager::assign_edge_dofs() {
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int edge = 0; edge < 4; ++edge) {
            std::vector<int> edge_dof_list = basis_.edge_dofs(edge);

            // Skip first and last (corners)
            for (size_t k = 1; k < edge_dof_list.size() - 1; ++k) {
                int local_dof = edge_dof_list[k];
                Vec2 pos = get_dof_position(e, local_dof);

                Index dof = find_dof_at_position(pos);
                if (dof < 0) {
                    dof = register_dof_at_position(pos);
                }

                elem_to_global_[e][local_dof] = dof;
            }
        }
    }
}

void CGCubicBezierDofManager::assign_interior_dofs() {
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int local_dof = 0; local_dof < CubicBezierBasis2D::NDOF; ++local_dof) {
            if (elem_to_global_[e][local_dof] < 0) {
                Index dof = num_global_dofs_++;
                elem_to_global_[e][local_dof] = dof;
                register_dof_position(dof, get_dof_position(e, local_dof));
            }
        }
    }
}

void CGCubicBezierDofManager::assign_edge_dofs_nonconforming() {
    // For cubic: edges have 4 DOFs (k=0,1,2,3)
    // - Corner DOFs (k=0, k=3) may be shared
    // - Interior edge DOFs (k=1, k=2) should NOT be shared at non-conforming
    // edges

    std::set<Index> coarse_interior_dofs;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int edge = 0; edge < 4; ++edge) {
            EdgeNeighborInfo info = mesh_.get_neighbor(elem, edge);
            if (info.type != EdgeNeighborInfo::Type::FineToCoarse) {
                continue;
            }

            Index coarse_elem = info.neighbor_elements[0];
            int coarse_edge = info.neighbor_edges[0];
            std::vector<int> coarse_dofs = basis_.edge_dofs(coarse_edge);

            // Mark coarse interior DOFs (m=1,2) as not to be shared
            for (int m = 1; m <= 2; ++m) {
                int coarse_local = coarse_dofs[m];
                Index coarse_global = elem_to_global_[coarse_elem][coarse_local];
                coarse_interior_dofs.insert(coarse_global);
            }
        }
    }

    // Fix fine element DOF assignments
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

            // For cubic: k=0..3
            int shared_k = (info.subedge_index == 0) ? 0 : 3;
            int shared_m = (info.subedge_index == 0) ? 0 : 3;
            int tjunction_k = (info.subedge_index == 0) ? 3 : 0;

            for (int k = 0; k < static_cast<int>(fine_dofs.size()); ++k) {
                int fine_local = fine_dofs[k];
                Index current_global = elem_to_global_[elem][fine_local];

                if (k == shared_k) {
                    int coarse_local = coarse_dofs[shared_m];
                    elem_to_global_[elem][fine_local] = elem_to_global_[coarse_elem][coarse_local];
                } else if (k == tjunction_k) {
                    // T-junction: keep position-based sharing
                } else {
                    if (coarse_interior_dofs.count(current_global) > 0) {
                        Index new_dof = num_global_dofs_++;
                        elem_to_global_[elem][fine_local] = new_dof;
                        register_dof_position(new_dof, get_dof_position(elem, fine_local));
                    }
                }
            }
        }
    }
}

// =============================================================================
// Hanging node constraints
// =============================================================================

void CGCubicBezierDofManager::build_hanging_node_constraints() {
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

            Real t_start, t_end;
            if (info.subedge_index == 0) {
                t_start = 0.0;
                t_end = 0.5;
            } else {
                t_start = 0.5;
                t_end = 1.0;
            }

            MatX S = basis_.compute_1d_extraction_matrix(t_start, t_end);

            for (int k = 0; k < static_cast<int>(fine_dofs.size()); ++k) {
                int fine_local = fine_dofs[k];
                Index fine_global = elem_to_global_[elem][fine_local];

                bool is_shared = false;
                if (info.subedge_index == 0) {
                    is_shared = (k == 0);
                } else {
                    is_shared = (k == 3); // For cubic, last DOF is k=3
                }

                if (is_shared) {
                    continue;
                }

                if (constrained_dofs_.count(fine_global) > 0) {
                    continue;
                }

                CubicHangingNodeConstraint constraint;
                constraint.slave_dof = fine_global;

                VecX weights = S.row(k);

                for (size_t m = 0; m < coarse_dofs.size(); ++m) {
                    int coarse_local = coarse_dofs[m];
                    Index coarse_global = elem_to_global_[coarse_elem][coarse_local];

                    if (std::abs(weights(static_cast<int>(m))) > 1e-14) {
                        constraint.master_dofs.push_back(coarse_global);
                        constraint.weights.push_back(weights(static_cast<int>(m)));
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

void CGCubicBezierDofManager::get_constraint_triplets(
    std::vector<Eigen::Triplet<Real>> &triplets) const {
    triplets.reserve(num_constraints() * 5); // Each constraint: 1 slave + ~4 masters

    for (Index row = 0; row < num_constraints(); ++row) {
        const auto &c = constraints_[row];
        triplets.emplace_back(row, c.slave_dof, 1.0);
        for (size_t i = 0; i < c.master_dofs.size(); ++i) {
            triplets.emplace_back(row, c.master_dofs[i], -c.weights[i]);
        }
    }
}

// =============================================================================
// C¹ edge derivative constraints
// =============================================================================

namespace {
Vec2 get_edge_param_cubic(int edge, Real t) {
    switch (edge) {
    case 0:
        return Vec2(0.0, t);
    case 1:
        return Vec2(1.0, t);
    case 2:
        return Vec2(t, 0.0);
    case 3:
        return Vec2(t, 1.0);
    default:
        return Vec2(0.5, 0.5);
    }
}
} // anonymous namespace

void CGCubicBezierDofManager::build_edge_derivative_constraints(int ngauss) {
    edge_derivative_constraints_.clear();

    // Build edge midpoint map
    std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<Index, int>>> edge_map;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int edge = 0; edge < 4; ++edge) {
            const auto &bounds = mesh_.element_bounds(elem);
            Vec2 midpoint;
            switch (edge) {
            case 0:
                midpoint = Vec2(bounds.xmin, 0.5 * (bounds.ymin + bounds.ymax));
                break;
            case 1:
                midpoint = Vec2(bounds.xmax, 0.5 * (bounds.ymin + bounds.ymax));
                break;
            case 2:
                midpoint = Vec2(0.5 * (bounds.xmin + bounds.xmax), bounds.ymin);
                break;
            case 3:
                midpoint = Vec2(0.5 * (bounds.xmin + bounds.xmax), bounds.ymax);
                break;
            }
            auto key = quantize_position(midpoint);
            edge_map[key].push_back({elem, edge});
        }
    }

    // Gauss points on [0, 1]
    std::vector<Real> gauss_pts(ngauss);
    if (ngauss == 2) {
        gauss_pts[0] = 0.5 - 0.5 / std::sqrt(3.0);
        gauss_pts[1] = 0.5 + 0.5 / std::sqrt(3.0);
    } else if (ngauss == 3) {
        gauss_pts[0] = 0.5 - 0.5 * std::sqrt(0.6);
        gauss_pts[1] = 0.5;
        gauss_pts[2] = 0.5 + 0.5 * std::sqrt(0.6);
    } else if (ngauss >= 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        gauss_pts.resize(4);
        gauss_pts[0] = 0.5 * (1.0 - b);
        gauss_pts[1] = 0.5 * (1.0 - a);
        gauss_pts[2] = 0.5 * (1.0 + a);
        gauss_pts[3] = 0.5 * (1.0 + b);
    }

    // Create constraints at shared conforming edges
    for (const auto &[key, elem_edges] : edge_map) {
        if (elem_edges.size() != 2) {
            continue;
        }

        Index elem1 = elem_edges[0].first;
        int edge1 = elem_edges[0].second;
        Index elem2 = elem_edges[1].first;
        int edge2 = elem_edges[1].second;

        const auto &bounds1 = mesh_.element_bounds(elem1);
        const auto &bounds2 = mesh_.element_bounds(elem2);
        Real dx1 = bounds1.xmax - bounds1.xmin;
        Real dy1 = bounds1.ymax - bounds1.ymin;
        Real dx2 = bounds2.xmax - bounds2.xmin;
        Real dy2 = bounds2.ymax - bounds2.ymin;

        // Skip non-conforming edges
        bool is_horizontal = (edge1 == 2 || edge1 == 3);
        if (is_horizontal) {
            if (std::abs(dx1 - dx2) > 1e-10 * std::max(dx1, dx2)) {
                continue;
            }
        } else {
            if (std::abs(dy1 - dy2) > 1e-10 * std::max(dy1, dy2)) {
                continue;
            }
        }

        // For C¹: only first normal derivative
        std::pair<int, int> deriv_order;
        if (is_horizontal) {
            deriv_order = {0, 1}; // z_v
        } else {
            deriv_order = {1, 0}; // z_u
        }

        for (int gp = 0; gp < static_cast<int>(gauss_pts.size()); ++gp) {
            Real t = gauss_pts[gp];
            Vec2 param1 = get_edge_param_cubic(edge1, t);
            Vec2 param2 = get_edge_param_cubic(edge2, t);

            CubicEdgeDerivativeConstraint c;
            c.elem1 = elem1;
            c.elem2 = elem2;
            c.edge1 = edge1;
            c.edge2 = edge2;
            c.t = t;
            c.deriv_order = 1;

            int nu = deriv_order.first;
            int nv = deriv_order.second;

            c.coeffs1 = basis_.evaluate_derivative(param1(0), param1(1), nu, nv);
            c.coeffs2 = basis_.evaluate_derivative(param2(0), param2(1), nu, nv);

            c.scale1 = std::pow(dx1, nu) * std::pow(dy1, nv);
            c.scale2 = std::pow(dx2, nu) * std::pow(dy2, nv);

            edge_derivative_constraints_.push_back(c);
        }
    }

    // Part 2: Non-conforming edge constraints
    build_nonconforming_edge_derivative_constraints(gauss_pts);
}

void CGCubicBezierDofManager::build_nonconforming_edge_derivative_constraints(
    const std::vector<Real> &gauss_pts) {

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int edge = 0; edge < 4; ++edge) {
            EdgeNeighborInfo info = mesh_.get_neighbor(elem, edge);

            if (info.type != EdgeNeighborInfo::Type::FineToCoarse) {
                continue;
            }

            Index fine_elem = elem;
            int fine_edge = edge;
            Index coarse_elem = info.neighbor_elements[0];
            int coarse_edge = info.neighbor_edges[0];

            // Get bounds for scaling
            const auto &fine_bounds = mesh_.element_bounds(fine_elem);
            const auto &coarse_bounds = mesh_.element_bounds(coarse_elem);

            Real dx_fine = fine_bounds.xmax - fine_bounds.xmin;
            Real dy_fine = fine_bounds.ymax - fine_bounds.ymin;
            Real dx_coarse = coarse_bounds.xmax - coarse_bounds.xmin;
            Real dy_coarse = coarse_bounds.ymax - coarse_bounds.ymin;

            // Parameter mapping: fine edge maps to [t_start, t_start + 0.5] of coarse
            Real t_coarse_start = (info.subedge_index == 0) ? 0.0 : 0.5;
            Real t_coarse_scale = 0.5;

            // Derivative direction based on edge orientation
            bool is_horizontal = (fine_edge == 2 || fine_edge == 3);
            int nu = is_horizontal ? 0 : 1;
            int nv = is_horizontal ? 1 : 0;

            for (const Real &t_fine : gauss_pts) {
                Real t_coarse = t_coarse_start + t_fine * t_coarse_scale;

                Vec2 param_fine = get_edge_param_cubic(fine_edge, t_fine);
                Vec2 param_coarse = get_edge_param_cubic(coarse_edge, t_coarse);

                CubicEdgeDerivativeConstraint c;
                c.elem1 = fine_elem;
                c.elem2 = coarse_elem;
                c.edge1 = fine_edge;
                c.edge2 = coarse_edge;
                c.t = t_fine;
                c.deriv_order = 1;

                c.coeffs1 = basis_.evaluate_derivative(param_fine(0), param_fine(1), nu, nv);
                c.coeffs2 = basis_.evaluate_derivative(param_coarse(0), param_coarse(1), nu, nv);

                c.scale1 = std::pow(dx_fine, nu) * std::pow(dy_fine, nv);
                c.scale2 = std::pow(dx_coarse, nu) * std::pow(dy_coarse, nv);

                edge_derivative_constraints_.push_back(c);
            }
        }
    }
}

void CGCubicBezierDofManager::build_boundary_curvature_constraints(int ngauss) {
    boundary_curvature_constraints_.clear();

    // Gauss points on [0, 1] (reuse same quadrature as edge constraints)
    std::vector<Real> gauss_pts(ngauss);
    if (ngauss == 2) {
        gauss_pts[0] = 0.5 - 0.5 / std::sqrt(3.0);
        gauss_pts[1] = 0.5 + 0.5 / std::sqrt(3.0);
    } else if (ngauss == 3) {
        gauss_pts[0] = 0.5 - 0.5 * std::sqrt(0.6);
        gauss_pts[1] = 0.5;
        gauss_pts[2] = 0.5 + 0.5 * std::sqrt(0.6);
    } else if (ngauss >= 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        gauss_pts.resize(4);
        gauss_pts[0] = 0.5 * (1.0 - b);
        gauss_pts[1] = 0.5 * (1.0 - a);
        gauss_pts[2] = 0.5 * (1.0 + a);
        gauss_pts[3] = 0.5 * (1.0 + b);
    }

    // Iterate over domain boundary edges
    mesh_.for_each_boundary_edge([this, &gauss_pts](Index elem, int edge) {
        const auto &bounds = mesh_.element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        // Natural BC: ∂²z/∂n² = 0 where n is normal to boundary
        // Edge 0,1 (left/right): normal is u-direction → ∂²z/∂u² = 0
        // Edge 2,3 (bottom/top): normal is v-direction → ∂²z/∂v² = 0
        bool is_u_normal = (edge == 0 || edge == 1);

        for (const Real &t : gauss_pts) {
            Vec2 param = get_edge_param_cubic(edge, t);

            BoundaryCurvatureConstraint c;
            c.elem = elem;
            c.edge = edge;
            c.t = t;

            if (is_u_normal) {
                // ∂²z/∂u² at (u,v) where u=0 or u=1
                c.coeffs = basis_.evaluate_d2u(param(0), param(1));
                c.scale = 1.0 / (dx * dx);  // Convert from parametric to physical
            } else {
                // ∂²z/∂v² at (u,v) where v=0 or v=1
                c.coeffs = basis_.evaluate_d2v(param(0), param(1));
                c.scale = 1.0 / (dy * dy);  // Convert from parametric to physical
            }

            boundary_curvature_constraints_.push_back(c);
        }
    });
}

void CGCubicBezierDofManager::build_boundary_gradient_constraints(int ngauss) {
    boundary_gradient_constraints_.clear();

    // Gauss points on [0, 1] (reuse same quadrature as other constraints)
    std::vector<Real> gauss_pts(ngauss);
    if (ngauss == 2) {
        gauss_pts[0] = 0.5 - 0.5 / std::sqrt(3.0);
        gauss_pts[1] = 0.5 + 0.5 / std::sqrt(3.0);
    } else if (ngauss == 3) {
        gauss_pts[0] = 0.5 - 0.5 * std::sqrt(0.6);
        gauss_pts[1] = 0.5;
        gauss_pts[2] = 0.5 + 0.5 * std::sqrt(0.6);
    } else if (ngauss >= 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        gauss_pts.resize(4);
        gauss_pts[0] = 0.5 * (1.0 - b);
        gauss_pts[1] = 0.5 * (1.0 - a);
        gauss_pts[2] = 0.5 * (1.0 + a);
        gauss_pts[3] = 0.5 * (1.0 + b);
    }

    // Iterate over domain boundary edges
    mesh_.for_each_boundary_edge([this, &gauss_pts](Index elem, int edge) {
        const auto &bounds = mesh_.element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        // Zero gradient BC: ∂z/∂n = 0 where n is normal to boundary
        // Edge 0,1 (left/right): normal is u-direction → ∂z/∂u = 0
        // Edge 2,3 (bottom/top): normal is v-direction → ∂z/∂v = 0
        bool is_u_normal = (edge == 0 || edge == 1);

        for (const Real &t : gauss_pts) {
            Vec2 param = get_edge_param_cubic(edge, t);

            BoundaryGradientConstraint c;
            c.elem = elem;
            c.edge = edge;
            c.t = t;

            if (is_u_normal) {
                // ∂z/∂u at (u,v) where u=0 or u=1
                c.coeffs = basis_.evaluate_du(param(0), param(1));
                c.scale = 1.0 / dx;  // Convert from parametric to physical
            } else {
                // ∂z/∂v at (u,v) where v=0 or v=1
                c.coeffs = basis_.evaluate_dv(param(0), param(1));
                c.scale = 1.0 / dy;  // Convert from parametric to physical
            }

            boundary_gradient_constraints_.push_back(c);
        }
    });
}

} // namespace drifter
