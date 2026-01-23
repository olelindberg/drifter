#include "bathymetry/bezier_c2_constraints.hpp"
#include <cmath>
#include <map>
#include <set>

namespace drifter {

BezierC2ConstraintBuilder::BezierC2ConstraintBuilder(const QuadtreeAdapter& mesh)
    : mesh_(mesh)
    , basis_(std::make_unique<BezierBasis2D>())
{
}

void BezierC2ConstraintBuilder::find_all_constraints() const {
    if (constraints_built_) return;

    vertex_constraints_.clear();
    hanging_node_constraints_.clear();
    edge_constraints_.clear();

    // Tolerance for matching positions
    const Real tol = 1e-10;

    // Use a map from quantized position to element/corner pairs
    // Quantization key: round(x * scale), round(y * scale)
    const Real scale = 1e8;
    std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<Index, int>>> vertex_map;

    // Collect all corners
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int corner = 0; corner < 4; ++corner) {
            Vec2 pos = get_corner_position(e, corner);
            int64_t qx = static_cast<int64_t>(std::round(pos(0) * scale));
            int64_t qy = static_cast<int64_t>(std::round(pos(1) * scale));
            vertex_map[{qx, qy}].push_back({e, corner});
        }
    }

    // Create vertex constraints for shared vertices
    // IMPORTANT: For vertices shared by >2 elements, only constrain each element
    // to a single reference element (the one with smallest index) to avoid
    // redundant (linearly dependent) constraints that make the KKT matrix singular.

    for (const auto& [pos_key, elem_corners] : vertex_map) {
        if (elem_corners.size() < 2) continue;

        // Use the first element (smallest index) as the reference
        Index ref_elem = elem_corners[0].first;
        int ref_corner = elem_corners[0].second;

        // Constrain all other elements at this vertex to the reference
        for (size_t i = 1; i < elem_corners.size(); ++i) {
            Index other_elem = elem_corners[i].first;
            int other_corner = elem_corners[i].second;

            VertexConstraintInfo info;
            info.elem1 = ref_elem;
            info.elem2 = other_elem;
            info.corner1 = ref_corner;
            info.corner2 = other_corner;
            info.position = get_corner_position(ref_elem, ref_corner);

            vertex_constraints_.push_back(info);
        }
    }

    // Find conforming shared edges AND hanging nodes at non-conforming edges
    // Track edges we've already processed to avoid duplicates
    std::set<std::pair<Index, Index>> processed_edges;  // (min_elem, max_elem) pairs

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int edge = 0; edge < 4; ++edge) {
            EdgeNeighborInfo neighbor = mesh_.get_neighbor(e, edge);

            if (neighbor.type == EdgeNeighborInfo::Type::CoarseToFine) {
                // This element is coarse, neighbors are fine
                // Fine element edges need constraints at their midpoint

                for (size_t n = 0; n < neighbor.neighbor_elements.size(); ++n) {
                    HangingNodeConstraintInfo info;
                    info.coarse_elem = e;
                    info.fine_elem = neighbor.neighbor_elements[n];
                    info.coarse_edge = edge;
                    info.fine_edge = neighbor.neighbor_edges[n];

                    // The fine edge midpoint is at t = 0.25 or 0.75 on coarse edge
                    // depending on which half
                    info.t_coarse = (n == 0) ? 0.25 : 0.75;

                    // All control points on the fine edge are constrained
                    // to match the coarse element's Bezier surface
                    info.fine_dofs = basis_->edge_dofs(info.fine_edge);

                    hanging_node_constraints_.push_back(info);
                }
            } else if (neighbor.is_conforming()) {
                // Conforming edge: elements at the same refinement level share an edge
                // Add edge constraints for interior DOFs to ensure C² along entire edge
                Index neighbor_elem = neighbor.neighbor_elements[0];
                int neighbor_edge = neighbor.neighbor_edges[0];

                // Avoid processing the same edge twice (from both elements)
                Index min_elem = std::min(e, neighbor_elem);
                Index max_elem = std::max(e, neighbor_elem);
                std::pair<Index, Index> edge_key = {min_elem, max_elem};

                if (processed_edges.count(edge_key) == 0) {
                    processed_edges.insert(edge_key);

                    EdgeConstraintInfo info;
                    info.elem1 = e;
                    info.elem2 = neighbor_elem;
                    info.edge1 = edge;
                    info.edge2 = neighbor_edge;

                    edge_constraints_.push_back(info);
                }
            }
        }
    }

    constraints_built_ = true;
}

Index BezierC2ConstraintBuilder::num_constraints() const {
    find_all_constraints();

    // Each shared vertex contributes 9 constraints (for C² continuity):
    // z, z_u, z_v, z_uu, z_vv, z_uv, z_uuv, z_uvv, z_uuvv
    Index num = 9 * static_cast<Index>(vertex_constraints_.size());

    // Each hanging node constraint adds:
    // - 9 C² derivative constraints at the T-junction vertex
    // - (edge_dofs.size() - 2) C⁰ interpolation constraints for interior DOFs
    for (const auto& info : hanging_node_constraints_) {
        num += 9;  // C² constraints at T-junction
        num += static_cast<Index>(info.fine_dofs.size()) - 2;  // Interior edge DOFs
    }

    // Each conforming edge constraint adds (edge_dofs.size() - 2) constraints
    // for interior DOFs (corners are handled by vertex constraints)
    // For quintic Bezier: 6 DOFs per edge - 2 corners = 4 interior DOFs
    for (const auto& info : edge_constraints_) {
        (void)info;  // Suppress unused warning
        num += BezierBasis2D::N1D - 2;  // 6 - 2 = 4 interior DOFs
    }

    return num;
}

SpMat BezierC2ConstraintBuilder::build_constraint_matrix() const {
    find_all_constraints();

    Index nrows = num_constraints();
    Index ncols = total_dofs();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(nrows * 2 * BezierBasis2D::NDOF);  // Estimate

    Index constraint_idx = 0;

    // Add vertex constraints
    for (const auto& info : vertex_constraints_) {
        add_vertex_constraints(info, triplets, constraint_idx);
    }

    // Add hanging node constraints
    for (const auto& info : hanging_node_constraints_) {
        add_hanging_node_constraints(info, triplets, constraint_idx);
    }

    // Add conforming edge constraints
    for (const auto& info : edge_constraints_) {
        add_conforming_edge_constraints(info, triplets, constraint_idx);
    }

    SpMat A(nrows, ncols);
    A.setFromTriplets(triplets.begin(), triplets.end());

    return A;
}

void BezierC2ConstraintBuilder::add_vertex_constraints(
    const VertexConstraintInfo& info,
    std::vector<Eigen::Triplet<Real>>& triplets,
    Index& constraint_idx) const
{
    // Get parameters for each element at the shared vertex
    Vec2 uv1 = get_corner_param(info.corner1);
    Vec2 uv2 = get_corner_param(info.corner2);

    // Get element sizes for scaling derivatives
    Vec2 size1 = mesh_.element_size(info.elem1);
    Vec2 size2 = mesh_.element_size(info.elem2);

    Real dx1 = size1(0), dy1 = size1(1);
    Real dx2 = size2(0), dy2 = size2(1);

    // For C² continuity at shared vertex, we enforce:
    //   z1(vertex) - z2(vertex) = 0
    //   dz1/dx - dz2/dx = 0  (scaled by element size)
    //   dz1/dy - dz2/dy = 0
    //   ... up to 4th derivatives for full C² (9 constraints total)
    //
    // Derivative constraint list (nu, nv) orders:
    // (0,0): z           -> constraint 0
    // (1,0): z_u         -> constraint 1
    // (0,1): z_v         -> constraint 2
    // (2,0): z_uu        -> constraint 3
    // (1,1): z_uv        -> constraint 4
    // (0,2): z_vv        -> constraint 5
    // (2,1): z_uuv       -> constraint 6
    // (1,2): z_uvv       -> constraint 7
    // (2,2): z_uuvv      -> constraint 8

    // Constraint equations: deriv1 * scale1 - deriv2 * scale2 = 0
    // where scale converts from parameter to physical derivatives

    const std::vector<std::pair<int, int>> deriv_orders = {
        {0, 0},  // z
        {1, 0},  // z_u
        {0, 1},  // z_v
        {2, 0},  // z_uu
        {1, 1},  // z_uv
        {0, 2},  // z_vv
        {2, 1},  // z_uuv
        {1, 2},  // z_uvv
        {2, 2}   // z_uuvv
    };

    for (const auto& [nu, nv] : deriv_orders) {
        // Scaling factors: physical derivative = param_derivative / (dx^nu * dy^nv)
        // For matching: param1/scale1 = param2/scale2
        // Rearrange: param1 - (scale1/scale2) * param2 = 0
        //
        // But for numerical stability, normalize:
        // param1/scale1 - param2/scale2 = 0 with scale1 = dx1^nu * dy1^nv

        Real scale1 = std::pow(dx1, nu) * std::pow(dy1, nv);
        Real scale2 = std::pow(dx2, nu) * std::pow(dy2, nv);

        // Evaluate basis derivatives at vertices
        VecX phi1 = basis_->evaluate_derivative(uv1(0), uv1(1), nu, nv);
        VecX phi2 = basis_->evaluate_derivative(uv2(0), uv2(1), nu, nv);

        // Add constraint: (1/scale1) * sum_k c1_k * phi1_k - (1/scale2) * sum_k c2_k * phi2_k = 0
        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            Index col1 = global_dof(info.elem1, k);
            Index col2 = global_dof(info.elem2, k);

            if (std::abs(phi1(k) / scale1) > 1e-14) {
                triplets.emplace_back(constraint_idx, col1, phi1(k) / scale1);
            }
            if (std::abs(phi2(k) / scale2) > 1e-14) {
                triplets.emplace_back(constraint_idx, col2, -phi2(k) / scale2);
            }
        }

        ++constraint_idx;
    }
}

void BezierC2ConstraintBuilder::add_hanging_node_constraints(
    const HangingNodeConstraintInfo& info,
    std::vector<Eigen::Triplet<Real>>& triplets,
    Index& constraint_idx) const
{
    // Hanging node constraints for C² continuity at non-conforming interfaces.
    //
    // At a T-junction (hanging node), we need:
    // 1. Edge DOF interpolation: interior control points on the fine edge must
    //    lie on the coarse element's Bezier curve (C⁰ for interior DOFs)
    // 2. Corner derivative matching: at the fine element's corner that lies on
    //    the coarse edge (the T-junction vertex), match all 9 C² derivatives
    //
    // This ensures the fine element's surface has C² continuity with the
    // coarse element's surface at the T-junction point.

    const QuadBounds& coarse_bounds = mesh_.element_bounds(info.coarse_elem);
    const QuadBounds& fine_bounds = mesh_.element_bounds(info.fine_elem);

    // Element sizes for derivative scaling
    Real dx_coarse = coarse_bounds.xmax - coarse_bounds.xmin;
    Real dy_coarse = coarse_bounds.ymax - coarse_bounds.ymin;
    Real dx_fine = fine_bounds.xmax - fine_bounds.xmin;
    Real dy_fine = fine_bounds.ymax - fine_bounds.ymin;

    auto edge_dofs = basis_->edge_dofs(info.fine_edge);

    // Find which corner of the fine element is at the T-junction (hanging node)
    // The T-junction corner is at the start or end of the edge, depending on
    // which half of the coarse edge this fine element covers.
    // For fine_edge:
    //   0 (left, u=0):   corners are at v=0 (dof 0) and v=1 (dof 5)
    //   1 (right, u=1):  corners are at v=0 (dof 30) and v=1 (dof 35)
    //   2 (bottom, v=0): corners are at u=0 (dof 0) and u=1 (dof 30)
    //   3 (top, v=1):    corners are at u=0 (dof 5) and u=1 (dof 35)

    // The T-junction is at the corner that lies INTERIOR to the coarse edge,
    // not at the corner that coincides with the coarse element's corner.
    // For the first fine element (t_coarse=0.25): T-junction at midpoint
    // For the second fine element (t_coarse=0.75): T-junction at midpoint

    // Determine which fine corner DOF is the T-junction (interior to coarse edge)
    int t_junction_dof = -1;
    Vec2 t_junction_fine_param;

    // The T-junction is at the corner that corresponds to t_coarse = 0.5 on the
    // coarse edge (midpoint). For the fine element, this is either the first
    // or last DOF on its edge, depending on orientation.
    //
    // t_coarse = 0.25 means this fine element covers [0, 0.5] of coarse edge
    //   -> T-junction at fine param = 1.0 (end of fine edge in that direction)
    // t_coarse = 0.75 means this fine element covers [0.5, 1.0] of coarse edge
    //   -> T-junction at fine param = 0.0 (start of fine edge in that direction)

    if (info.t_coarse < 0.5) {
        // First half: T-junction at the "upper" end of the fine edge
        t_junction_dof = edge_dofs.back();  // Last DOF on edge
    } else {
        // Second half: T-junction at the "lower" end of the fine edge
        t_junction_dof = edge_dofs.front(); // First DOF on edge
    }

    t_junction_fine_param = basis_->control_point_position(t_junction_dof);

    // Map T-junction to physical position
    Real x_tj = fine_bounds.xmin + t_junction_fine_param(0) * dx_fine;
    Real y_tj = fine_bounds.ymin + t_junction_fine_param(1) * dy_fine;

    // Map to coarse element parameters
    Real u_tj_coarse = (x_tj - coarse_bounds.xmin) / dx_coarse;
    Real v_tj_coarse = (y_tj - coarse_bounds.ymin) / dy_coarse;

    // Add C² derivative constraints at the T-junction (9 constraints)
    // These ensure the fine element's derivatives match the coarse element's
    // derivatives at the hanging node point.
    const std::vector<std::pair<int, int>> deriv_orders = {
        {0, 0},  // z
        {1, 0},  // z_u
        {0, 1},  // z_v
        {2, 0},  // z_uu
        {1, 1},  // z_uv
        {0, 2},  // z_vv
        {2, 1},  // z_uuv
        {1, 2},  // z_uvv
        {2, 2}   // z_uuvv
    };

    for (const auto& [nu, nv] : deriv_orders) {
        // Scaling factors for physical derivatives
        Real scale_coarse = std::pow(dx_coarse, nu) * std::pow(dy_coarse, nv);
        Real scale_fine = std::pow(dx_fine, nu) * std::pow(dy_fine, nv);

        // Evaluate basis derivatives at T-junction position
        VecX phi_coarse = basis_->evaluate_derivative(u_tj_coarse, v_tj_coarse, nu, nv);
        VecX phi_fine = basis_->evaluate_derivative(t_junction_fine_param(0),
                                                     t_junction_fine_param(1), nu, nv);

        // Constraint: deriv_fine/scale_fine - deriv_coarse/scale_coarse = 0
        // => (1/scale_fine) * sum_k c_fine[k] * phi_fine[k]
        //  - (1/scale_coarse) * sum_k c_coarse[k] * phi_coarse[k] = 0

        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            if (std::abs(phi_fine(k) / scale_fine) > 1e-14) {
                Index col_fine = global_dof(info.fine_elem, k);
                triplets.emplace_back(constraint_idx, col_fine, phi_fine(k) / scale_fine);
            }
            if (std::abs(phi_coarse(k) / scale_coarse) > 1e-14) {
                Index col_coarse = global_dof(info.coarse_elem, k);
                triplets.emplace_back(constraint_idx, col_coarse, -phi_coarse(k) / scale_coarse);
            }
        }

        ++constraint_idx;
    }

    // Add C⁰ interpolation constraints for interior edge DOFs (not corners)
    // These ensure interior control points lie on the coarse Bezier curve.
    for (int local_idx = 0; local_idx < static_cast<int>(edge_dofs.size()); ++local_idx) {
        int fine_dof = edge_dofs[local_idx];

        // Skip corner DOFs - they're handled by the C² constraints above
        // or by conforming vertex constraints at domain corners
        if (fine_dof == edge_dofs.front() || fine_dof == edge_dofs.back()) {
            continue;
        }

        // Get position of this DOF on fine element in physical coords
        Vec2 fine_param = basis_->control_point_position(fine_dof);

        // Map to physical position using fine element bounds
        Real x_phys = fine_bounds.xmin + fine_param(0) * dx_fine;
        Real y_phys = fine_bounds.ymin + fine_param(1) * dy_fine;

        // Map physical position to coarse element parameters
        Real u_coarse = (x_phys - coarse_bounds.xmin) / dx_coarse;
        Real v_coarse = (y_phys - coarse_bounds.ymin) / dy_coarse;

        // Clamp to [0,1] for safety
        u_coarse = std::clamp(u_coarse, 0.0, 1.0);
        v_coarse = std::clamp(v_coarse, 0.0, 1.0);

        // Evaluate coarse element's basis at this position
        VecX phi_coarse = basis_->evaluate(u_coarse, v_coarse);

        // Constraint: c_fine[dof] - sum_k c_coarse[k] * phi_coarse[k] = 0
        Index col_fine = global_dof(info.fine_elem, fine_dof);
        triplets.emplace_back(constraint_idx, col_fine, 1.0);

        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            if (std::abs(phi_coarse(k)) > 1e-14) {
                Index col_coarse = global_dof(info.coarse_elem, k);
                triplets.emplace_back(constraint_idx, col_coarse, -phi_coarse(k));
            }
        }

        ++constraint_idx;
    }
}

void BezierC2ConstraintBuilder::add_conforming_edge_constraints(
    const EdgeConstraintInfo& info,
    std::vector<Eigen::Triplet<Real>>& triplets,
    Index& constraint_idx) const
{
    // Conforming edge constraints ensure interior DOFs along a shared edge match.
    // This, combined with vertex C² constraints at the corners, guarantees
    // C² continuity along the entire edge.
    //
    // For each interior DOF on the edge (not corners), add constraint:
    //   c_elem1[dof1] - c_elem2[dof2] = 0
    //
    // where dof1 and dof2 are the corresponding control point indices on each
    // element's edge. The correspondence depends on the edge orientation.

    auto edge_dofs1 = basis_->edge_dofs(info.edge1);
    auto edge_dofs2 = basis_->edge_dofs(info.edge2);

    // Determine orientation: edges may be traversed in opposite directions
    // For opposite edges (0 vs 1, 2 vs 3), the DOF ordering is the same
    // For adjacent edges meeting at a corner, the ordering depends on the corner
    //
    // Edge orientation mapping:
    // - Edge 0 (u=0): DOFs [0,1,2,3,4,5] for v=0,1/5,2/5,3/5,4/5,1
    // - Edge 1 (u=1): DOFs [30,31,32,33,34,35] for v=0,1/5,2/5,3/5,4/5,1
    // - Edge 2 (v=0): DOFs [0,6,12,18,24,30] for u=0,1/5,2/5,3/5,4/5,1
    // - Edge 3 (v=1): DOFs [5,11,17,23,29,35] for u=0,1/5,2/5,3/5,4/5,1
    //
    // When two elements share an edge, their edge DOFs correspond in the same
    // parameter order (both increasing) if the physical positions match.

    // Get physical positions of first and last DOFs on each edge
    Vec2 size1 = mesh_.element_size(info.elem1);
    Vec2 size2 = mesh_.element_size(info.elem2);
    const QuadBounds& bounds1 = mesh_.element_bounds(info.elem1);
    const QuadBounds& bounds2 = mesh_.element_bounds(info.elem2);

    auto param_to_physical = [](const Vec2& param, const QuadBounds& bounds) {
        return Vec2(
            bounds.xmin + param(0) * (bounds.xmax - bounds.xmin),
            bounds.ymin + param(1) * (bounds.ymax - bounds.ymin)
        );
    };

    Vec2 start1_param = basis_->control_point_position(edge_dofs1.front());
    Vec2 end1_param = basis_->control_point_position(edge_dofs1.back());
    Vec2 start2_param = basis_->control_point_position(edge_dofs2.front());
    Vec2 end2_param = basis_->control_point_position(edge_dofs2.back());

    Vec2 start1_phys = param_to_physical(start1_param, bounds1);
    Vec2 end1_phys = param_to_physical(end1_param, bounds1);
    Vec2 start2_phys = param_to_physical(start2_param, bounds2);
    Vec2 end2_phys = param_to_physical(end2_param, bounds2);

    // Check if edges are traversed in the same or opposite directions
    const Real tol = 1e-8;
    bool same_direction = (start1_phys - start2_phys).norm() < tol;
    bool opposite_direction = (start1_phys - end2_phys).norm() < tol;

    if (!same_direction && !opposite_direction) {
        // This shouldn't happen for conforming edges
        return;
    }

    // Add constraints for interior DOFs (skip first and last = corners)
    for (size_t i = 1; i < edge_dofs1.size() - 1; ++i) {
        int dof1 = edge_dofs1[i];
        int dof2;

        if (same_direction) {
            dof2 = edge_dofs2[i];
        } else {
            // Opposite direction: map index i to (size - 1 - i)
            dof2 = edge_dofs2[edge_dofs2.size() - 1 - i];
        }

        Index col1 = global_dof(info.elem1, dof1);
        Index col2 = global_dof(info.elem2, dof2);

        // Constraint: c1[dof1] - c2[dof2] = 0
        triplets.emplace_back(constraint_idx, col1, 1.0);
        triplets.emplace_back(constraint_idx, col2, -1.0);

        ++constraint_idx;
    }
}

Vec2 BezierC2ConstraintBuilder::get_corner_param(int corner_id) const {
    return basis_->corner_param(corner_id);
}

Vec2 BezierC2ConstraintBuilder::get_corner_position(Index elem, int corner_id) const {
    const QuadBounds& bounds = mesh_.element_bounds(elem);
    Vec2 param = get_corner_param(corner_id);

    Real x = bounds.xmin + param(0) * (bounds.xmax - bounds.xmin);
    Real y = bounds.ymin + param(1) * (bounds.ymax - bounds.ymin);

    return Vec2(x, y);
}

int BezierC2ConstraintBuilder::derivative_to_constraint_index(int nu, int nv) const {
    // Map derivative orders to sequential constraint index
    // (0,0)->0, (1,0)->1, (0,1)->2, (2,0)->3, (1,1)->4, (0,2)->5,
    // (2,1)->6, (1,2)->7, (2,2)->8
    if (nu == 0 && nv == 0) return 0;
    if (nu == 1 && nv == 0) return 1;
    if (nu == 0 && nv == 1) return 2;
    if (nu == 2 && nv == 0) return 3;
    if (nu == 1 && nv == 1) return 4;
    if (nu == 0 && nv == 2) return 5;
    if (nu == 2 && nv == 1) return 6;
    if (nu == 1 && nv == 2) return 7;
    if (nu == 2 && nv == 2) return 8;
    return -1;  // Invalid
}

bool BezierC2ConstraintBuilder::positions_match(const Vec2& p1, const Vec2& p2) const {
    const Real tol = 1e-10;
    return std::abs(p1(0) - p2(0)) < tol && std::abs(p1(1) - p2(1)) < tol;
}

std::pair<int, int> BezierC2ConstraintBuilder::corner_to_edges(int corner_id) const {
    // Map corner ID to the two edges meeting at that corner
    // Corner 0 (xmin, ymin): edges 0 (left) and 2 (bottom)
    // Corner 1 (xmax, ymin): edges 1 (right) and 2 (bottom)
    // Corner 2 (xmin, ymax): edges 0 (left) and 3 (top)
    // Corner 3 (xmax, ymax): edges 1 (right) and 3 (top)
    static const std::pair<int, int> mapping[4] = {
        {0, 2},  // corner 0
        {1, 2},  // corner 1
        {0, 3},  // corner 2
        {1, 3}   // corner 3
    };
    return mapping[corner_id];
}

bool BezierC2ConstraintBuilder::is_boundary_vertex(
    const std::vector<std::pair<Index, int>>& elem_corners) const {
    // A vertex is on the domain boundary if ANY edge meeting at that vertex
    // (from any element's perspective) is a boundary edge
    for (const auto& [elem, corner] : elem_corners) {
        auto [edge1, edge2] = corner_to_edges(corner);
        EdgeNeighborInfo info1 = mesh_.get_neighbor(elem, edge1);
        EdgeNeighborInfo info2 = mesh_.get_neighbor(elem, edge2);

        if (info1.is_boundary() || info2.is_boundary()) {
            return true;  // This vertex is on the domain boundary
        }
    }
    return false;  // All edges meeting here are interior
}

bool BezierC2ConstraintBuilder::share_interior_edge_at_vertex(
    Index elem1, int corner1, Index elem2, int corner2) const {
    // Two elements share an interior edge at a vertex if:
    // 1. They are neighbors along one of the edges meeting at the vertex
    // 2. That edge is interior (not on the domain boundary)
    //
    // For elem1's corner, get the two edges meeting there
    auto [edge1a, edge1b] = corner_to_edges(corner1);

    // Check if elem2 is a neighbor along edge1a
    EdgeNeighborInfo info1a = mesh_.get_neighbor(elem1, edge1a);
    if (!info1a.is_boundary()) {
        for (Index neighbor : info1a.neighbor_elements) {
            if (neighbor == elem2) {
                return true;  // They share interior edge edge1a
            }
        }
    }

    // Check if elem2 is a neighbor along edge1b
    EdgeNeighborInfo info1b = mesh_.get_neighbor(elem1, edge1b);
    if (!info1b.is_boundary()) {
        for (Index neighbor : info1b.neighbor_elements) {
            if (neighbor == elem2) {
                return true;  // They share interior edge edge1b
            }
        }
    }

    // They don't share an interior edge at this vertex
    // (they might only touch at the vertex, or share a boundary edge)
    return false;
}

// =============================================================================
// Dirichlet boundary constraint implementation
// =============================================================================

void BezierC2ConstraintBuilder::find_boundary_dofs() const {
    if (dirichlet_built_) return;

    dirichlet_constraints_.clear();
    find_all_constraints();  // Ensure C² constraints are built

    // First, collect all DOFs that lie on domain boundary edges.
    // These should get Dirichlet constraints even if they're C²-constrained.
    std::set<Index> boundary_dofs;
    mesh_.for_each_boundary_edge([&](Index elem, int edge) {
        for (int local_dof : basis_->edge_dofs(edge)) {
            boundary_dofs.insert(global_dof(elem, local_dof));
        }
    });

    // Find C²-constrained DOFs that are INTERIOR (not on domain boundary).
    // Only interior C²-constrained DOFs should be excluded from Dirichlet constraints.
    // Boundary corners should still get Dirichlet constraints; the KKT solver
    // handles this by computing b_c2 = -A_c2 * x_dir and zeroing Dirichlet columns.
    std::set<Index> interior_c2_dofs;
    for (const auto& vc : vertex_constraints_) {
        int corner_dof1 = basis_->corner_dof(vc.corner1);
        int corner_dof2 = basis_->corner_dof(vc.corner2);

        Index gdof1 = global_dof(vc.elem1, corner_dof1);
        Index gdof2 = global_dof(vc.elem2, corner_dof2);

        // Only exclude from Dirichlet if NOT on domain boundary
        if (boundary_dofs.count(gdof1) == 0) {
            interior_c2_dofs.insert(gdof1);
        }
        if (boundary_dofs.count(gdof2) == 0) {
            interior_c2_dofs.insert(gdof2);
        }
    }

    // Track which global DOFs have already been added to avoid duplicates
    // (corner DOFs may be on two boundary edges)
    std::set<Index> added_dofs;

    // Iterate over all boundary edges
    // IMPORTANT: Only apply Dirichlet BCs at corner DOFs, not interior edge DOFs.
    // This preserves C² constraints at shared boundary vertices, which involve
    // derivative terms that depend on interior edge DOFs (e.g., z_u at corner (0,0)
    // involves DOFs at (0,0), (1,0), (2,0)).
    mesh_.for_each_boundary_edge([&](Index elem, int edge) {
        std::vector<int> edge_dofs_list = basis_->edge_dofs(edge);
        // Only corner DOFs (first and last in the edge list)
        std::vector<int> corner_dofs = {edge_dofs_list.front(), edge_dofs_list.back()};
        const QuadBounds& bounds = mesh_.element_bounds(elem);

        for (int local_dof : corner_dofs) {
            Index gdof = global_dof(elem, local_dof);

            // Skip if already added (corner shared by two boundary edges)
            if (added_dofs.count(gdof) > 0) {
                continue;
            }

            // Skip only INTERIOR C²-constrained DOFs
            if (interior_c2_dofs.count(gdof) > 0) {
                continue;
            }

            added_dofs.insert(gdof);

            // Get the parameter position of this control point
            Vec2 param = basis_->control_point_position(local_dof);

            // Map to physical coordinates
            Real x = bounds.xmin + param(0) * (bounds.xmax - bounds.xmin);
            Real y = bounds.ymin + param(1) * (bounds.ymax - bounds.ymin);

            DirichletConstraintInfo info;
            info.elem = elem;
            info.edge = edge;
            info.local_dof = local_dof;
            info.global_dof = gdof;
            info.position = Vec2(x, y);

            dirichlet_constraints_.push_back(info);
        }
    });

    dirichlet_built_ = true;
}

SpMat BezierC2ConstraintBuilder::build_dirichlet_constraint_matrix() const {
    find_boundary_dofs();

    Index nrows = num_dirichlet_constraints();
    Index ncols = total_dofs();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(nrows);  // Exactly one entry per row

    for (Index i = 0; i < nrows; ++i) {
        const auto& info = dirichlet_constraints_[i];
        triplets.emplace_back(i, info.global_dof, 1.0);
    }

    SpMat A(nrows, ncols);
    A.setFromTriplets(triplets.begin(), triplets.end());

    return A;
}

SpMat BezierC2ConstraintBuilder::build_combined_constraint_matrix() const {
    // First build C² continuity constraints
    SpMat A_c2 = build_constraint_matrix();

    // Then build Dirichlet constraints
    SpMat A_dir = build_dirichlet_constraint_matrix();

    Index n_c2 = A_c2.rows();
    Index n_dir = A_dir.rows();
    Index ncols = total_dofs();

    // Stack them vertically: [A_c2; A_dir]
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(A_c2.nonZeros() + A_dir.nonZeros());

    // Copy C² constraint entries
    for (int k = 0; k < A_c2.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_c2, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Copy Dirichlet constraint entries, offset by n_c2 rows
    for (int k = 0; k < A_dir.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_dir, k); it; ++it) {
            triplets.emplace_back(n_c2 + it.row(), it.col(), it.value());
        }
    }

    SpMat A_combined(n_c2 + n_dir, ncols);
    A_combined.setFromTriplets(triplets.begin(), triplets.end());

    return A_combined;
}

// =============================================================================
// Natural boundary condition implementation (z_nn = 0)
// =============================================================================

Vec2 BezierC2ConstraintBuilder::edge_outward_normal(int edge) const {
    // Outward normals for each edge
    // Edge 0: left (u=0)   -> normal points in -x direction
    // Edge 1: right (u=1)  -> normal points in +x direction
    // Edge 2: bottom (v=0) -> normal points in -y direction
    // Edge 3: top (v=1)    -> normal points in +y direction
    static const Vec2 normals[4] = {
        Vec2(-1.0, 0.0),   // Edge 0: left
        Vec2(1.0, 0.0),    // Edge 1: right
        Vec2(0.0, -1.0),   // Edge 2: bottom
        Vec2(0.0, 1.0)     // Edge 3: top
    };
    return normals[edge];
}

Vec2 BezierC2ConstraintBuilder::get_dof_position(Index elem, int local_dof) const {
    const QuadBounds& bounds = mesh_.element_bounds(elem);
    Vec2 param = basis_->control_point_position(local_dof);
    Real x = bounds.xmin + param(0) * (bounds.xmax - bounds.xmin);
    Real y = bounds.ymin + param(1) * (bounds.ymax - bounds.ymin);
    return Vec2(x, y);
}

std::vector<int> BezierC2ConstraintBuilder::edge_dofs(int edge) const {
    return basis_->edge_dofs(edge);
}

void BezierC2ConstraintBuilder::find_natural_bc_constraints() const {
    if (natural_bc_built_) return;

    natural_bc_constraints_.clear();

    // Track (global_dof, edge) pairs to avoid duplicates
    // A DOF at a domain corner will appear on two edges but we need
    // TWO constraints (one for each edge normal) to enforce both z_xx=0 and z_yy=0
    std::set<std::pair<Index, int>> processed_constraints;

    // Iterate over all boundary edges
    mesh_.for_each_boundary_edge([&](Index elem, int edge) {
        Vec2 normal = edge_outward_normal(edge);
        std::vector<int> dofs_on_edge = basis_->edge_dofs(edge);

        for (int local_dof : dofs_on_edge) {
            Index gdof = global_dof(elem, local_dof);

            // Key is (global_dof, edge) to allow multiple constraints at corner DOFs
            auto key = std::make_pair(gdof, edge);
            if (processed_constraints.count(key) > 0) {
                continue;
            }
            processed_constraints.insert(key);

            NaturalBCConstraintInfo info;
            info.elem = elem;
            info.local_dof = local_dof;
            info.global_dof = gdof;
            info.edge = edge;
            info.normal = normal;
            info.position = get_dof_position(elem, local_dof);

            natural_bc_constraints_.push_back(info);
        }
    });

    natural_bc_built_ = true;
}

SpMat BezierC2ConstraintBuilder::build_natural_bc_matrix() const {
    find_natural_bc_constraints();

    Index nrows = num_natural_bc_constraints();
    Index ncols = total_dofs();

    std::vector<Eigen::Triplet<Real>> triplets;

    for (Index row = 0; row < nrows; ++row) {
        const auto& info = natural_bc_constraints_[row];

        // Get element size for physical scaling
        Vec2 size = mesh_.element_size(info.elem);
        Real dx = size(0);
        Real dy = size(1);
        Real nx = info.normal(0);
        Real ny = info.normal(1);

        // Get parameter position of this DOF
        Vec2 uv = basis_->control_point_position(info.local_dof);

        // Evaluate second derivative basis functions at this point
        VecX phi_uu, phi_vv, phi_uv;
        basis_->evaluate_second_derivatives(uv(0), uv(1), phi_uu, phi_vv, phi_uv);

        // z_nn = nx² * z_xx + 2*nx*ny * z_xy + ny² * z_yy = 0
        // In parameter coords with physical scaling:
        //   z_xx = z_uu / dx²
        //   z_yy = z_vv / dy²
        //   z_xy = z_uv / (dx * dy)
        //
        // So: z_nn = (nx²/dx²) * z_uu + (2*nx*ny/(dx*dy)) * z_uv + (ny²/dy²) * z_vv = 0
        //
        // z_uu = sum_k c_k * phi_uu(k)
        // z_vv = sum_k c_k * phi_vv(k)
        // z_uv = sum_k c_k * phi_uv(k)
        //
        // Constraint: sum_k c_k * [(nx²/dx²)*phi_uu(k) + (2*nx*ny/(dx*dy))*phi_uv(k) + (ny²/dy²)*phi_vv(k)] = 0

        Real c_uu = nx * nx / (dx * dx);
        Real c_vv = ny * ny / (dy * dy);
        Real c_uv = 2.0 * nx * ny / (dx * dy);

        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            Real val = c_uu * phi_uu(k) + c_vv * phi_vv(k) + c_uv * phi_uv(k);
            if (std::abs(val) > 1e-14) {
                triplets.emplace_back(row, global_dof(info.elem, k), val);
            }
        }
    }

    SpMat A(nrows, ncols);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

SpMat BezierC2ConstraintBuilder::build_c2_and_natural_bc_matrix() const {
    // Build C² continuity constraints
    SpMat A_c2 = build_constraint_matrix();

    // Build natural BC constraints
    SpMat A_nat = build_natural_bc_matrix();

    Index n_c2 = A_c2.rows();
    Index n_nat = A_nat.rows();
    Index ncols = total_dofs();

    // Stack them vertically: [A_c2; A_nat]
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(A_c2.nonZeros() + A_nat.nonZeros());

    // Copy C² constraint entries
    for (int k = 0; k < A_c2.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_c2, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Copy natural BC constraint entries, offset by n_c2 rows
    for (int k = 0; k < A_nat.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_nat, k); it; ++it) {
            triplets.emplace_back(n_c2 + it.row(), it.col(), it.value());
        }
    }

    SpMat A_combined(n_c2 + n_nat, ncols);
    A_combined.setFromTriplets(triplets.begin(), triplets.end());

    return A_combined;
}

}  // namespace drifter
