#include "bathymetry/cg_bezier_dof_manager.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drifter {

// Scale factor for position quantization (inverse of tolerance)
// Using 1e8 with int64_t to support coordinates up to ~92,000,000
static constexpr Real POSITION_SCALE = 1e8;

CGBezierDofManager::CGBezierDofManager(const QuadtreeAdapter& mesh)
    : mesh_(mesh) {

    Index num_elements = mesh_.num_elements();
    if (num_elements == 0) {
        num_global_dofs_ = 0;
        num_free_dofs_ = 0;
        return;
    }

    // Initialize element DOF vectors
    elem_to_global_.resize(num_elements);
    for (Index e = 0; e < num_elements; ++e) {
        elem_to_global_[e].resize(BezierBasis2D::NDOF, -1);
    }

    // Three-pass DOF assignment
    assign_vertex_dofs();
    assign_edge_dofs();
    assign_interior_dofs();

    // Identify boundary DOFs
    identify_boundary_dofs();

    // Handle non-conforming interfaces
    build_hanging_node_constraints();

    // Build DOF mappings for constraint elimination
    build_dof_mappings();
}

Index CGBezierDofManager::global_dof(Index elem, int local_dof) const {
    if (elem < 0 || elem >= static_cast<Index>(elem_to_global_.size())) {
        throw std::out_of_range("CGBezierDofManager: element index out of range");
    }
    if (local_dof < 0 || local_dof >= BezierBasis2D::NDOF) {
        throw std::out_of_range("CGBezierDofManager: local DOF index out of range");
    }
    return elem_to_global_[elem][local_dof];
}

const std::vector<Index>& CGBezierDofManager::element_dofs(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(elem_to_global_.size())) {
        throw std::out_of_range("CGBezierDofManager: element index out of range");
    }
    return elem_to_global_[elem];
}

bool CGBezierDofManager::is_boundary_dof(Index dof) const {
    return boundary_dof_set_.count(dof) > 0;
}

bool CGBezierDofManager::is_constrained(Index dof) const {
    return constrained_dofs_.count(dof) > 0;
}

Index CGBezierDofManager::global_to_free(Index global_dof) const {
    if (global_dof < 0 || global_dof >= num_global_dofs_) {
        return -1;
    }
    return global_to_free_[global_dof];
}

Index CGBezierDofManager::free_to_global(Index free_dof) const {
    if (free_dof < 0 || free_dof >= num_free_dofs_) {
        return -1;
    }
    return free_to_global_[free_dof];
}

SpMat CGBezierDofManager::build_constraint_matrix() const {
    Index nrows = num_constraints();
    Index ncols = num_global_dofs_;

    if (nrows == 0) {
        return SpMat(0, ncols);
    }

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(nrows * 7);  // Estimate: ~7 master DOFs per constraint

    for (Index row = 0; row < nrows; ++row) {
        const auto& c = constraints_[row];

        // Slave DOF has coefficient 1
        triplets.emplace_back(row, c.slave_dof, 1.0);

        // Master DOFs have negative interpolation weights
        for (size_t i = 0; i < c.master_dofs.size(); ++i) {
            triplets.emplace_back(row, c.master_dofs[i], -c.weights[i]);
        }
    }

    SpMat A(nrows, ncols);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

// =============================================================================
// Position handling
// =============================================================================

std::pair<int64_t, int64_t> CGBezierDofManager::quantize_position(const Vec2& pos) const {
    return std::make_pair(
        static_cast<int64_t>(std::round(pos(0) * POSITION_SCALE)),
        static_cast<int64_t>(std::round(pos(1) * POSITION_SCALE))
    );
}

Vec2 CGBezierDofManager::get_dof_position(Index elem, int local_dof) const {
    const auto& bounds = mesh_.element_bounds(elem);

    // Get parameter position (u, v) in [0, 1]^2
    Vec2 param = basis_.control_point_position(local_dof);

    // Map to physical coordinates
    Real x = bounds.xmin + param(0) * (bounds.xmax - bounds.xmin);
    Real y = bounds.ymin + param(1) * (bounds.ymax - bounds.ymin);

    return Vec2(x, y);
}

Index CGBezierDofManager::find_dof_at_position(const Vec2& pos) const {
    auto key = quantize_position(pos);
    auto it = position_to_dof_.find(key);
    if (it != position_to_dof_.end()) {
        return it->second;
    }
    return -1;
}

// =============================================================================
// DOF classification helpers
// =============================================================================

bool CGBezierDofManager::is_corner_dof(int local_dof) const {
    // Corners are at (i,j) = (0,0), (5,0), (0,5), (5,5)
    // DOF indices: 0, 5, 30, 35
    return local_dof == 0 || local_dof == 5 || local_dof == 30 || local_dof == 35;
}

bool CGBezierDofManager::is_edge_dof(int local_dof) const {
    int i, j;
    BezierBasis2D::dof_ij(local_dof, i, j);

    // Edge DOFs are on the boundary but not corners
    bool on_boundary = (i == 0 || i == 5 || j == 0 || j == 5);
    return on_boundary && !is_corner_dof(local_dof);
}

bool CGBezierDofManager::is_interior_dof(int local_dof) const {
    int i, j;
    BezierBasis2D::dof_ij(local_dof, i, j);

    // Interior if not on any edge
    return (i > 0 && i < 5 && j > 0 && j < 5);
}

int CGBezierDofManager::get_edge_for_dof(int local_dof) const {
    int i, j;
    BezierBasis2D::dof_ij(local_dof, i, j);

    if (i == 0 && j > 0 && j < 5) return 0;  // Left edge
    if (i == 5 && j > 0 && j < 5) return 1;  // Right edge
    if (j == 0 && i > 0 && i < 5) return 2;  // Bottom edge
    if (j == 5 && i > 0 && i < 5) return 3;  // Top edge

    return -1;  // Not an edge DOF (excluding corners)
}

// =============================================================================
// DOF assignment (3-pass algorithm)
// =============================================================================

void CGBezierDofManager::assign_vertex_dofs() {
    // Assign DOFs at corners (shared by multiple elements)
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int corner = 0; corner < 4; ++corner) {
            int local_dof = basis_.corner_dof(corner);
            Vec2 pos = get_dof_position(e, local_dof);
            auto key = quantize_position(pos);

            // Check if DOF already exists at this position
            Index dof = find_dof_at_position(pos);
            if (dof < 0) {
                // Create new DOF
                dof = num_global_dofs_++;
                position_to_dof_[key] = dof;
            }

            elem_to_global_[e][local_dof] = dof;
        }
    }
}

void CGBezierDofManager::assign_edge_dofs() {
    // Assign DOFs along edges (excluding corners)
    // Edge DOFs are shared between adjacent conforming elements

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int edge = 0; edge < 4; ++edge) {
            std::vector<int> edge_dof_list = basis_.edge_dofs(edge);

            // Skip first and last (corners, already assigned)
            for (size_t k = 1; k < edge_dof_list.size() - 1; ++k) {
                int local_dof = edge_dof_list[k];
                Vec2 pos = get_dof_position(e, local_dof);
                auto key = quantize_position(pos);

                // Check if DOF already exists
                Index dof = find_dof_at_position(pos);
                if (dof < 0) {
                    // Create new DOF
                    dof = num_global_dofs_++;
                    position_to_dof_[key] = dof;
                }

                elem_to_global_[e][local_dof] = dof;
            }
        }
    }
}

void CGBezierDofManager::assign_interior_dofs() {
    // Interior DOFs are unique per element (not shared)
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int local_dof = 0; local_dof < BezierBasis2D::NDOF; ++local_dof) {
            if (elem_to_global_[e][local_dof] < 0) {
                // This is an interior DOF (not yet assigned)
                elem_to_global_[e][local_dof] = num_global_dofs_++;
            }
        }
    }
}

void CGBezierDofManager::identify_boundary_dofs() {
    boundary_dofs_.clear();
    boundary_dof_set_.clear();

    // Iterate over boundary edges
    mesh_.for_each_boundary_edge([this](Index elem, int edge) {
        // Get all DOFs on this edge (including corners)
        std::vector<int> edge_dof_list = basis_.edge_dofs(edge);

        for (int local_dof : edge_dof_list) {
            Index global = elem_to_global_[elem][local_dof];
            if (boundary_dof_set_.insert(global).second) {
                boundary_dofs_.push_back(global);
            }
        }
    });

    std::sort(boundary_dofs_.begin(), boundary_dofs_.end());
}

// =============================================================================
// Hanging node constraints
// =============================================================================

void CGBezierDofManager::build_hanging_node_constraints() {
    constraints_.clear();
    constrained_dofs_.clear();

    // Iterate over interior edges to find non-conforming interfaces
    mesh_.for_each_interior_edge([this](Index elem, int edge,
                                        const EdgeNeighborInfo& info) {
        if (info.type != EdgeNeighborInfo::Type::FineToCoarse) {
            return;  // Only process fine-to-coarse edges
        }

        // This element is the finer one - its edge DOFs may be hanging
        Index coarse_elem = info.neighbor_elements[0];
        int coarse_edge = info.neighbor_edges[0];

        // Get fine and coarse edge DOFs
        std::vector<int> fine_dofs = basis_.edge_dofs(edge);
        std::vector<int> coarse_dofs = basis_.edge_dofs(coarse_edge);

        // Get physical bounds
        const auto& fine_bounds = mesh_.element_bounds(elem);
        const auto& coarse_bounds = mesh_.element_bounds(coarse_elem);

        // Determine the parameter range on the coarse edge that corresponds to this fine edge
        // For a 2:1 refinement, the fine edge spans half of the coarse edge
        Real t_start, t_end;
        if (info.subedge_index == 0) {
            t_start = 0.0;
            t_end = 0.5;
        } else {
            t_start = 0.5;
            t_end = 1.0;
        }

        // For each fine edge DOF, create a constraint if it's a hanging node
        for (size_t k = 0; k < fine_dofs.size(); ++k) {
            int fine_local = fine_dofs[k];
            Index fine_global = elem_to_global_[elem][fine_local];

            // Get the parameter position of this DOF on the fine edge
            // fine edge DOFs are at t = k/5 along the edge
            Real t_fine = static_cast<Real>(k) / (BezierBasis2D::N1D - 1);

            // Map to coarse edge parameter
            Real t_coarse = t_start + t_fine * (t_end - t_start);

            // Check if this DOF position matches a coarse DOF position
            // Coarse DOFs are at t = 0, 0.2, 0.4, 0.6, 0.8, 1.0
            bool is_shared = false;
            for (size_t m = 0; m < coarse_dofs.size(); ++m) {
                Real t_coarse_dof = static_cast<Real>(m) / (BezierBasis2D::N1D - 1);
                if (std::abs(t_coarse - t_coarse_dof) < 1e-10) {
                    is_shared = true;
                    break;
                }
            }

            if (is_shared) {
                // This DOF is shared (already handled by position-based sharing)
                continue;
            }

            // Skip if this DOF is already constrained (avoid duplicate constraints
            // at T-junctions where multiple fine elements meet the same coarse edge)
            if (constrained_dofs_.count(fine_global) > 0) {
                continue;
            }

            // This is a hanging node - constrain it to interpolate from coarse
            HangingNodeConstraint constraint;
            constraint.slave_dof = fine_global;

            // Evaluate 1D Bezier basis at t_coarse to get interpolation weights
            VecX B = basis_.evaluate_bernstein_1d(BezierBasis2D::DEGREE, t_coarse);

            for (size_t m = 0; m < coarse_dofs.size(); ++m) {
                int coarse_local = coarse_dofs[m];
                Index coarse_global = elem_to_global_[coarse_elem][coarse_local];

                if (std::abs(B(m)) > 1e-14) {
                    constraint.master_dofs.push_back(coarse_global);
                    constraint.weights.push_back(B(m));
                }
            }

            if (!constraint.master_dofs.empty()) {
                constraints_.push_back(constraint);
                constrained_dofs_.insert(fine_global);
            }
        }
    });
}

void CGBezierDofManager::build_dof_mappings() {
    // Build mapping from global DOFs to free DOFs (excluding constrained)
    global_to_free_.resize(num_global_dofs_, -1);
    free_to_global_.clear();
    free_to_global_.reserve(num_global_dofs_);

    num_free_dofs_ = 0;
    for (Index g = 0; g < num_global_dofs_; ++g) {
        if (!is_constrained(g)) {
            global_to_free_[g] = num_free_dofs_++;
            free_to_global_.push_back(g);
        }
    }
}

// =============================================================================
// Vertex derivative constraints for C² continuity
// =============================================================================

void CGBezierDofManager::build_vertex_derivative_constraints() {
    vertex_derivative_constraints_.clear();

    // Derivative orders for C² continuity at vertices
    // Skip (0,0) - z value is already shared via CG DOFs
    // This matches the DG Bezier smoother approach with 9 derivatives per vertex,
    // but for CG we only need 8 since the value is shared.
    static const std::vector<std::pair<int, int>> deriv_orders = {
        {1, 0}, {0, 1},           // C¹: z_u, z_v
        {2, 0}, {1, 1}, {0, 2},   // C²: z_uu, z_uv, z_vv
        {2, 1}, {1, 2}, {2, 2}    // Higher: z_uuv, z_uvv, z_uuvv
    };

    // Build a map from corner positions to (element, corner_id) pairs
    // This identifies which elements share each vertex
    std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<Index, int>>> vertex_map;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int corner = 0; corner < 4; ++corner) {
            int local_dof = basis_.corner_dof(corner);
            Vec2 pos = get_dof_position(elem, local_dof);
            auto key = quantize_position(pos);
            vertex_map[key].push_back({elem, corner});
        }
    }

    // For each shared vertex (2+ elements meeting), create constraints
    // Use star pattern: first element is reference, constrain others to it
    for (const auto& [key, elem_corners] : vertex_map) {
        if (elem_corners.size() < 2) {
            continue;  // Boundary vertex - no constraints needed
        }

        // Reference element (first in the list)
        Index ref_elem = elem_corners[0].first;
        int ref_corner = elem_corners[0].second;

        // Get reference element's size
        const auto& ref_bounds = mesh_.element_bounds(ref_elem);
        Real ref_dx = ref_bounds.xmax - ref_bounds.xmin;
        Real ref_dy = ref_bounds.ymax - ref_bounds.ymin;

        // Get parameter coordinates for reference corner
        Vec2 ref_param = basis_.corner_param(ref_corner);

        // Constrain all other elements to the reference
        for (size_t i = 1; i < elem_corners.size(); ++i) {
            Index other_elem = elem_corners[i].first;
            int other_corner = elem_corners[i].second;

            // Get other element's size
            const auto& other_bounds = mesh_.element_bounds(other_elem);
            Real other_dx = other_bounds.xmax - other_bounds.xmin;
            Real other_dy = other_bounds.ymax - other_bounds.ymin;

            // Get parameter coordinates for other corner
            Vec2 other_param = basis_.corner_param(other_corner);

            // Add 8 derivative constraints (skip z value - already shared via DOFs)
            for (const auto& [nu, nv] : deriv_orders) {
                VertexDerivativeConstraint c;
                c.elem1 = ref_elem;
                c.elem2 = other_elem;
                c.corner1 = ref_corner;
                c.corner2 = other_corner;
                c.nu = nu;
                c.nv = nv;

                // Evaluate basis derivatives at corners
                c.coeffs1 = basis_.evaluate_derivative(ref_param(0), ref_param(1), nu, nv);
                c.coeffs2 = basis_.evaluate_derivative(other_param(0), other_param(1), nu, nv);

                // Scale factors: dx^nu * dy^nv to convert parameter derivatives to physical
                c.scale1 = std::pow(ref_dx, nu) * std::pow(ref_dy, nv);
                c.scale2 = std::pow(other_dx, nu) * std::pow(other_dy, nv);

                vertex_derivative_constraints_.push_back(c);
            }
        }
    }
}

// =============================================================================
// Edge derivative constraints for C² continuity along edges
// =============================================================================

namespace {
// Helper to get parameter coordinates on an edge at position t in [0,1]
Vec2 get_edge_param(int edge, Real t) {
    switch (edge) {
        case 0: return Vec2(0.0, t);    // Left edge: u=0, v varies
        case 1: return Vec2(1.0, t);    // Right edge: u=1, v varies
        case 2: return Vec2(t, 0.0);    // Bottom edge: v=0, u varies
        case 3: return Vec2(t, 1.0);    // Top edge: v=1, u varies
        default: return Vec2(0.5, 0.5);
    }
}
}  // anonymous namespace

void CGBezierDofManager::build_edge_derivative_constraints(int ngauss) {
    edge_derivative_constraints_.clear();

    // Build a map from edge midpoints to (element, edge_id) pairs
    // This identifies which elements share each edge (conforming edges only)
    std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<Index, int>>> edge_map;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int edge = 0; edge < 4; ++edge) {
            // Get the midpoint of the edge
            const auto& bounds = mesh_.element_bounds(elem);
            Vec2 midpoint;
            switch (edge) {
                case 0:  // Left edge (u=0)
                    midpoint = Vec2(bounds.xmin, 0.5 * (bounds.ymin + bounds.ymax));
                    break;
                case 1:  // Right edge (u=1)
                    midpoint = Vec2(bounds.xmax, 0.5 * (bounds.ymin + bounds.ymax));
                    break;
                case 2:  // Bottom edge (v=0)
                    midpoint = Vec2(0.5 * (bounds.xmin + bounds.xmax), bounds.ymin);
                    break;
                case 3:  // Top edge (v=1)
                    midpoint = Vec2(0.5 * (bounds.xmin + bounds.xmax), bounds.ymax);
                    break;
            }
            auto key = quantize_position(midpoint);
            edge_map[key].push_back({elem, edge});
        }
    }

    // Gauss-Legendre points on [0, 1] for edge integration
    std::vector<Real> gauss_pts(ngauss);
    if (ngauss == 2) {
        gauss_pts[0] = 0.5 - 0.5 / std::sqrt(3.0);
        gauss_pts[1] = 0.5 + 0.5 / std::sqrt(3.0);
    } else if (ngauss == 3) {
        gauss_pts[0] = 0.5 - 0.5 * std::sqrt(0.6);
        gauss_pts[1] = 0.5;
        gauss_pts[2] = 0.5 + 0.5 * std::sqrt(0.6);
    } else if (ngauss == 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        gauss_pts[0] = 0.5 * (1.0 - b);
        gauss_pts[1] = 0.5 * (1.0 - a);
        gauss_pts[2] = 0.5 * (1.0 + a);
        gauss_pts[3] = 0.5 * (1.0 + b);
    } else {
        // Default to 4 points
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        gauss_pts.resize(4);
        gauss_pts[0] = 0.5 * (1.0 - b);
        gauss_pts[1] = 0.5 * (1.0 - a);
        gauss_pts[2] = 0.5 * (1.0 + a);
        gauss_pts[3] = 0.5 * (1.0 + b);
    }

    // For each shared edge (exactly 2 elements), create derivative constraints
    for (const auto& [key, elem_edges] : edge_map) {
        if (elem_edges.size() != 2) {
            continue;  // Skip boundary edges (1 element) and non-conforming (>2)
        }

        Index elem1 = elem_edges[0].first;
        int edge1 = elem_edges[0].second;
        Index elem2 = elem_edges[1].first;
        int edge2 = elem_edges[1].second;

        // Get element sizes
        const auto& bounds1 = mesh_.element_bounds(elem1);
        const auto& bounds2 = mesh_.element_bounds(elem2);
        Real dx1 = bounds1.xmax - bounds1.xmin;
        Real dy1 = bounds1.ymax - bounds1.ymin;
        Real dx2 = bounds2.xmax - bounds2.xmin;
        Real dy2 = bounds2.ymax - bounds2.ymin;

        // Skip non-conforming edges (different sizes)
        bool is_horizontal = (edge1 == 2 || edge1 == 3);
        if (is_horizontal) {
            // For horizontal edges, check x-extents match
            if (std::abs(dx1 - dx2) > 1e-10 * std::max(dx1, dx2)) {
                continue;
            }
        } else {
            // For vertical edges, check y-extents match
            if (std::abs(dy1 - dy2) > 1e-10 * std::max(dy1, dy2)) {
                continue;
            }
        }

        // Derivative orders to constrain for C² continuity
        // For horizontal edges: z_v, z_vv (normal and second normal derivative)
        // For vertical edges: z_u, z_uu (normal and second normal derivative)
        std::vector<std::pair<int, int>> deriv_orders;
        if (is_horizontal) {
            // Constrain v-derivatives (normal to horizontal edge)
            deriv_orders = {{0, 1}, {0, 2}};  // z_v, z_vv
        } else {
            // Constrain u-derivatives (normal to vertical edge)
            deriv_orders = {{1, 0}, {2, 0}};  // z_u, z_uu
        }

        // Add constraints at each Gauss point
        for (int gp = 0; gp < static_cast<int>(gauss_pts.size()); ++gp) {
            Real t = gauss_pts[gp];

            // Get (u, v) coordinates on each element's edge
            Vec2 param1 = get_edge_param(edge1, t);
            Vec2 param2 = get_edge_param(edge2, t);

            for (const auto& [nu, nv] : deriv_orders) {
                EdgeDerivativeConstraint c;
                c.elem1 = elem1;
                c.elem2 = elem2;
                c.edge1 = edge1;
                c.edge2 = edge2;
                c.t = t;
                c.deriv_order = std::max(nu, nv);

                // Evaluate basis derivatives at the edge points
                c.coeffs1 = basis_.evaluate_derivative(param1(0), param1(1), nu, nv);
                c.coeffs2 = basis_.evaluate_derivative(param2(0), param2(1), nu, nv);

                // Scale factors: dx^nu * dy^nv
                c.scale1 = std::pow(dx1, nu) * std::pow(dy1, nv);
                c.scale2 = std::pow(dx2, nu) * std::pow(dy2, nv);

                edge_derivative_constraints_.push_back(c);
            }
        }
    }
}

}  // namespace drifter
