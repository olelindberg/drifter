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

    // Fix DOF sharing at non-conforming edges using index-based logic
    assign_edge_dofs_nonconforming();

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

void CGBezierDofManager::assign_edge_dofs_nonconforming() {
    // For non-conforming (FineToCoarse) edges, only CORNER DOFs should be shared
    // with the coarse element.
    //
    // IMPORTANT: Unlike conforming edges where DOFs at the same physical position
    // can be shared, at non-conforming edges only the curve ENDPOINTS are guaranteed
    // to match. Interior edge DOFs (Bezier control points) do NOT lie on the curve
    // and have values that depend on the curve shape, not just physical position.
    //
    // assign_edge_dofs() may have incorrectly shared interior DOFs based on physical
    // position matching. We must UNDO this by assigning NEW global DOF indices to
    // interior fine edge DOFs that were incorrectly shared with the coarse element.
    //
    // For 2:1 refinement with quintic Bezier (6 DOFs per edge):
    // - subedge_index=0 (left/lower half): k=0 shares with coarse m=0
    // - subedge_index=1 (right/upper half): k=5 shares with coarse m=5
    // - T-junction (k=5 for subedge=0, k=0 for subedge=1): shares with adjacent fine element
    //
    // All other fine edge DOFs get NEW indices and are then constrained via
    // build_hanging_node_constraints() using the Bezier extraction matrix.

    // First pass: identify which coarse DOFs should NOT be shared with fine elements
    // (all interior DOFs on the coarse side of non-conforming edges)
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

            // Mark coarse interior DOFs (m=1,2,3,4) as not to be shared
            for (int m = 1; m <= 4; ++m) {
                int coarse_local = coarse_dofs[m];
                Index coarse_global = elem_to_global_[coarse_elem][coarse_local];
                coarse_interior_dofs.insert(coarse_global);
            }
        }
    }

    // Second pass: fix fine element DOF assignments
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

            // Determine which corner is shared with coarse
            int shared_k = (info.subedge_index == 0) ? 0 : 5;
            int shared_m = (info.subedge_index == 0) ? 0 : 5;

            // Determine which DOF is the T-junction (shared with adjacent fine element)
            int tjunction_k = (info.subedge_index == 0) ? 5 : 0;

            // Process all fine edge DOFs
            for (int k = 0; k < static_cast<int>(fine_dofs.size()); ++k) {
                int fine_local = fine_dofs[k];
                Index current_global = elem_to_global_[elem][fine_local];

                if (k == shared_k) {
                    // This corner DOF should share with the coarse corner
                    int coarse_local = coarse_dofs[shared_m];
                    elem_to_global_[elem][fine_local] =
                        elem_to_global_[coarse_elem][coarse_local];
                } else if (k == tjunction_k) {
                    // T-junction: keep the position-based sharing with adjacent fine element
                    // (assign_edge_dofs already handled this via position matching)
                    // Don't change this DOF
                } else {
                    // Interior DOF: check if it was incorrectly shared with a coarse DOF
                    if (coarse_interior_dofs.count(current_global) > 0) {
                        // This DOF was incorrectly shared with a coarse interior DOF
                        // Assign a NEW global DOF index
                        elem_to_global_[elem][fine_local] = num_global_dofs_++;
                    }
                    // If it wasn't shared with coarse, it might be shared with another
                    // fine element (conforming edge), which is correct - leave it alone
                }
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

    // Iterate over ALL elements and edges directly.
    // NOTE: We do NOT use for_each_interior_edge() because its "lower index owns"
    // filtering causes FineToCoarse edges to be skipped when the coarse element
    // has a lower index than the fine element (common after AMR refinement).
    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int edge = 0; edge < 4; ++edge) {
            EdgeNeighborInfo info = mesh_.get_neighbor(elem, edge);

            if (info.type != EdgeNeighborInfo::Type::FineToCoarse) {
                continue;  // Only process fine-to-coarse edges
            }

            // This element is the finer one - its edge DOFs may be hanging
            Index coarse_elem = info.neighbor_elements[0];
            int coarse_edge = info.neighbor_edges[0];

            // Get fine and coarse edge DOFs
            std::vector<int> fine_dofs = basis_.edge_dofs(edge);
            std::vector<int> coarse_dofs = basis_.edge_dofs(coarse_edge);

            // Determine the parameter range on the coarse edge
            Real t_start, t_end;
            if (info.subedge_index == 0) {
                t_start = 0.0;
                t_end = 0.5;
            } else {
                t_start = 0.5;
                t_end = 1.0;
            }

            // Compute the Bezier extraction matrix for this sub-interval.
            // Row k of S gives the weights for fine control point k.
            MatX S = basis_.compute_1d_extraction_matrix(t_start, t_end);

            // For each fine edge DOF, create a constraint if it's a hanging node
            for (int k = 0; k < static_cast<int>(fine_dofs.size()); ++k) {
                int fine_local = fine_dofs[k];
                Index fine_global = elem_to_global_[elem][fine_local];

                // Only CORNER DOFs are shared (curve endpoints where values must match).
                // All other fine edge DOFs must be CONSTRAINED using the extraction matrix.
                // This is because Bezier control points don't lie on the curve (except at
                // endpoints), so sharing control points at the same physical position
                // does NOT ensure C⁰ continuity of the curve.
                //
                // For 2:1 refinement with quintic (6 DOFs per edge):
                // - subedge_index=0: only k=0 is shared (corner at start)
                // - subedge_index=1: only k=5 is shared (corner at end)
                bool is_shared = false;
                if (info.subedge_index == 0) {
                    // Lower half: only k=0 is the shared corner
                    is_shared = (k == 0);
                } else {
                    // Upper half: only k=5 is the shared corner
                    is_shared = (k == 5);
                }

                if (is_shared) {
                    // This DOF is shared (handled by assign_edge_dofs_nonconforming)
                    continue;
                }

                // Skip if this DOF is already constrained (avoid duplicate constraints
                // at T-junctions where multiple fine elements meet the same coarse edge)
                if (constrained_dofs_.count(fine_global) > 0) {
                    continue;
                }

                // This is a hanging node - constrain it using de Casteljau subdivision weights
                HangingNodeConstraint constraint;
                constraint.slave_dof = fine_global;

                // Use row k of the extraction matrix as the constraint weights.
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

    // Note: Non-conforming edges (FineToCoarse) are handled by hanging node
    // constraints via de Casteljau subdivision, which properly relates fine
    // edge control points to coarse control points. Edge derivative constraints
    // for non-conforming interfaces are not used as they would rely on
    // interior Gauss point extrapolation.
}

}  // namespace drifter
