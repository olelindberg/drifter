#include "bathymetry/cg_dof_manager.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drifter {

// Scale factor for position quantization (inverse of tolerance)
// Using 1e8 with int64_t to support coordinates up to ~92,000,000
static constexpr Real POSITION_SCALE = 1e8;

CGDofManager::CGDofManager(const QuadtreeAdapter &mesh, const LagrangeBasis2D &basis)
    : mesh_(mesh), basis_(basis) {

    Index num_elements = mesh_.num_elements();
    if (num_elements == 0) {
        num_global_dofs_ = 0;
        num_free_dofs_ = 0;
        return;
    }

    int ndof = basis_.num_dofs();

    // Initialize element DOF vectors
    elem_to_global_.resize(num_elements);
    for (Index e = 0; e < num_elements; ++e) {
        elem_to_global_[e].resize(ndof, -1);
    }

    // Three-pass DOF assignment (following wobbler CG2DMesh pattern)
    assign_vertex_dofs();
    assign_edge_dofs();
    assign_interior_dofs();

    // Identify boundary DOFs
    identify_boundary_dofs();

    // Handle non-conforming interfaces
    detect_hanging_nodes();
    build_c2_constraints();

    // Build DOF mappings for constraint elimination
    build_dof_mappings();
}

const std::vector<Index> &CGDofManager::element_dofs(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(elem_to_global_.size())) {
        throw std::out_of_range("CGDofManager: element index out of range");
    }
    return elem_to_global_[elem];
}

bool CGDofManager::is_boundary_dof(Index dof) const { return boundary_dof_set_.count(dof) > 0; }

bool CGDofManager::is_constrained(Index dof) const { return constrained_dofs_.count(dof) > 0; }

Index CGDofManager::global_to_free(Index global_dof) const {
    if (global_dof < 0 || global_dof >= num_global_dofs_) {
        return -1;
    }
    return global_to_free_[global_dof];
}

Index CGDofManager::free_to_global(Index free_dof) const {
    if (free_dof < 0 || free_dof >= num_free_dofs_) {
        return -1;
    }
    return free_to_global_[free_dof];
}

std::pair<int64_t, int64_t> CGDofManager::quantize_position(const Vec2 &pos) const {
    return std::make_pair(static_cast<int64_t>(std::round(pos(0) * POSITION_SCALE)),
                          static_cast<int64_t>(std::round(pos(1) * POSITION_SCALE)));
}

Vec2 CGDofManager::get_vertex_position(Index elem, int corner_id) const {
    const auto &bounds = mesh_.element_bounds(elem);

    // Corner ordering: 0=(-1,-1), 1=(+1,-1), 2=(-1,+1), 3=(+1,+1)
    switch (corner_id) {
    case 0:
        return Vec2(bounds.xmin, bounds.ymin);
    case 1:
        return Vec2(bounds.xmax, bounds.ymin);
    case 2:
        return Vec2(bounds.xmin, bounds.ymax);
    case 3:
        return Vec2(bounds.xmax, bounds.ymax);
    default:
        throw std::invalid_argument("Invalid corner ID");
    }
}

std::vector<Vec2> CGDofManager::get_edge_dof_positions(Index elem, int edge_id) const {
    const auto &bounds = mesh_.element_bounds(elem);
    std::vector<Vec2> positions;

    // LGL nodes on [-1,1] - ALL nodes including endpoints
    const VecX &nodes = basis_.nodes_1d();
    int n1d = basis_.num_nodes_1d();

    // For each edge, compute the physical positions of the DOFs
    // The indexing must match edge_dofs() in LagrangeBasis2D:
    // - Edge 0 (left, xi=-1):  dofs ordered by j (eta varies)
    // - Edge 1 (right, xi=+1): dofs ordered by j (eta varies)
    // - Edge 2 (bottom, eta=-1): dofs ordered by i (xi varies)
    // - Edge 3 (top, eta=+1):    dofs ordered by i (xi varies)

    for (int k = 0; k < n1d; ++k) {
        Vec2 pos;
        switch (edge_id) {
        case 0: // Left edge (x = xmin), j varies
        {
            Real eta = nodes(k); // j index corresponds to eta
            Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);
            pos = Vec2(bounds.xmin, y);
            break;
        }
        case 1: // Right edge (x = xmax), j varies
        {
            Real eta = nodes(k); // j index corresponds to eta
            Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);
            pos = Vec2(bounds.xmax, y);
            break;
        }
        case 2: // Bottom edge (y = ymin), i varies
        {
            Real xi = nodes(k); // i index corresponds to xi
            Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
            pos = Vec2(x, bounds.ymin);
            break;
        }
        case 3: // Top edge (y = ymax), i varies
        {
            Real xi = nodes(k); // i index corresponds to xi
            Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
            pos = Vec2(x, bounds.ymax);
            break;
        }
        }
        positions.push_back(pos);
    }

    return positions;
}

Index CGDofManager::find_dof_at_position(const Vec2 &pos, Real) const {
    auto key = quantize_position(pos);
    auto it = position_to_dof_.find(key);
    if (it != position_to_dof_.end()) {
        return it->second;
    }
    return -1;
}

void CGDofManager::assign_vertex_dofs() {
    // For each element, assign DOFs at corners
    // Corners are shared between adjacent elements

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int corner = 0; corner < 4; ++corner) {
            Vec2 pos = get_vertex_position(e, corner);
            auto key = quantize_position(pos);

            // Check if DOF already exists at this position
            Index dof = find_dof_at_position(pos);
            if (dof < 0) {
                // Create new DOF
                dof = num_global_dofs_++;
                position_to_dof_[key] = dof;
            }

            // Map corner to local DOF index
            int local_dof = basis_.corner_dof(corner);
            elem_to_global_[e][local_dof] = dof;
        }
    }
}

void CGDofManager::assign_edge_dofs() {
    // For each element, assign DOFs along edges (excluding corners)
    // Edge DOFs are shared between adjacent conforming elements

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int edge = 0; edge < 4; ++edge) {
            std::vector<int> local_dofs = basis_.edge_dofs(edge);
            std::vector<Vec2> positions = get_edge_dof_positions(e, edge);

            // edge_dofs returns all n1d DOFs on edge (including corners at
            // index 0 and n1d-1) Skip corners (indices 0 and n1d-1) - they were
            // assigned in assign_vertex_dofs
            for (size_t i = 1; i < local_dofs.size() - 1; ++i) {
                int local_dof = local_dofs[i];
                const Vec2 &pos = positions[i];

                // Check if DOF already exists
                Index dof = find_dof_at_position(pos);
                if (dof < 0) {
                    // Create new DOF
                    dof = num_global_dofs_++;
                    auto key = quantize_position(pos);
                    position_to_dof_[key] = dof;
                }

                elem_to_global_[e][local_dof] = dof;
            }
        }
    }
}

void CGDofManager::assign_interior_dofs() {
    // Interior DOFs are unique per element (not shared)
    int ndof = basis_.num_dofs();

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        for (int local_dof = 0; local_dof < ndof; ++local_dof) {
            if (elem_to_global_[e][local_dof] < 0) {
                // This is an interior DOF
                elem_to_global_[e][local_dof] = num_global_dofs_++;
            }
        }
    }
}

void CGDofManager::identify_boundary_dofs() {
    boundary_dofs_.clear();
    boundary_dof_set_.clear();

    // Iterate over boundary edges
    mesh_.for_each_boundary_edge([this](Index elem, int edge) {
        // Get all DOFs on this edge (including corners)
        std::vector<int> local_dofs = basis_.edge_dofs(edge);

        // Also add corner DOFs
        int c0, c1; // Corners at edge endpoints
        switch (edge) {
        case 0:
            c0 = 0;
            c1 = 2;
            break; // Left edge: corners 0, 2
        case 1:
            c0 = 1;
            c1 = 3;
            break; // Right edge: corners 1, 3
        case 2:
            c0 = 0;
            c1 = 1;
            break; // Bottom edge: corners 0, 1
        case 3:
            c0 = 2;
            c1 = 3;
            break; // Top edge: corners 2, 3
        default:
            return;
        }

        local_dofs.push_back(basis_.corner_dof(c0));
        local_dofs.push_back(basis_.corner_dof(c1));

        for (int local_dof : local_dofs) {
            Index global_dof = elem_to_global_[elem][local_dof];
            if (boundary_dof_set_.insert(global_dof).second) {
                boundary_dofs_.push_back(global_dof);
            }
        }
    });

    std::sort(boundary_dofs_.begin(), boundary_dofs_.end());
}

void CGDofManager::detect_hanging_nodes() {
    // Detect hanging nodes at non-conforming interfaces
    // These are DOFs on fine edges that don't exist at coarse edge positions

    hanging_edges_.clear();

    mesh_.for_each_interior_edge([this](Index elem, int edge, const EdgeNeighborInfo &info) {
        if (info.type == EdgeNeighborInfo::Type::FineToCoarse) {
            // This element is the finer one - its edge DOFs may be hanging
            HangingEdgeInfo hanging;
            hanging.fine_elem = elem;
            hanging.fine_edge = edge;
            hanging.coarse_elem = info.neighbor_elements[0];
            hanging.coarse_edge = info.neighbor_edges[0];
            hanging.subedge_index = info.subedge_index;

            hanging_edges_.push_back(hanging);
        }
    });
}

/// @brief Compute Lagrange basis function values at a point
/// @param nodes 1D nodal positions
/// @param t Evaluation point in [-1, 1]
/// @return Vector of basis function values
static VecX lagrange_basis(const VecX &nodes, Real t) {
    int n = static_cast<int>(nodes.size());
    VecX L(n);

    for (int i = 0; i < n; ++i) {
        Real Li = 1.0;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                Li *= (t - nodes(j)) / (nodes(i) - nodes(j));
            }
        }
        L(i) = Li;
    }

    return L;
}

void CGDofManager::build_c2_constraints() {
    // Build C⁰ constraints at hanging nodes for non-conforming interfaces
    // C¹ continuity is enforced via IPDG penalty in the assembler

    constraints_.clear();
    constrained_dofs_.clear();

    if (hanging_edges_.empty()) {
        // Conforming mesh - no constraints needed
        return;
    }

    const VecX &nodes = basis_.nodes_1d();
    int n1d = basis_.num_nodes_1d();

    for (const auto &hanging : hanging_edges_) {
        // Get DOFs on the fine edge (these are potential slave DOFs)
        std::vector<int> fine_local_dofs = basis_.edge_dofs(hanging.fine_edge);

        // Get DOFs on the coarse edge (these are master DOFs)
        std::vector<int> coarse_local_dofs = basis_.edge_dofs(hanging.coarse_edge);

        // Get physical positions of fine edge DOFs
        std::vector<Vec2> fine_positions =
            get_edge_dof_positions(hanging.fine_elem, hanging.fine_edge);

        // Get physical positions of coarse edge DOFs
        std::vector<Vec2> coarse_positions =
            get_edge_dof_positions(hanging.coarse_elem, hanging.coarse_edge);

        // Determine which direction the edge runs in physical space
        // Edge 0/1 (left/right) run in y-direction
        // Edge 2/3 (bottom/top) run in x-direction
        bool coarse_edge_runs_in_y = (hanging.coarse_edge == 0 || hanging.coarse_edge == 1);

        // Get the physical extent of the coarse edge
        Real coarse_start, coarse_end;
        if (coarse_edge_runs_in_y) {
            coarse_start = coarse_positions.front()(1); // y of first DOF
            coarse_end = coarse_positions.back()(1); // y of last DOF
        } else {
            coarse_start = coarse_positions.front()(0); // x of first DOF
            coarse_end = coarse_positions.back()(0); // x of last DOF
        }
        Real coarse_length = coarse_end - coarse_start;

        // For each fine edge DOF, compute its position along the coarse edge
        for (size_t i = 0; i < fine_local_dofs.size(); ++i) {
            int fine_local = fine_local_dofs[i];
            Index fine_global = elem_to_global_[hanging.fine_elem][fine_local];

            // Get physical position of this fine DOF
            const Vec2 &fine_pos = fine_positions[i];
            Real fine_coord = coarse_edge_runs_in_y ? fine_pos(1) : fine_pos(0);

            // Map to parameter on coarse edge [-1, 1]
            // The fine element spans only HALF of the coarse edge
            // subedge_index=0 means fine element is on the first half: coarse t
            // in
            // [-1, 0] subedge_index=1 means fine element is on the second half:
            // coarse t in [0, 1]
            Real t_coarse = -1.0 + 2.0 * (fine_coord - coarse_start) / coarse_length;

            // Clamp to valid range (handle numerical errors at endpoints)
            t_coarse = std::max(-1.0, std::min(1.0, t_coarse));

            // Check if this DOF position coincides with a coarse DOF position
            // (corners and endpoints may already be shared)
            bool is_shared = false;
            for (int j = 0; j < n1d; ++j) {
                Real t_coarse_node = nodes(j);
                if (std::abs(t_coarse - t_coarse_node) < 1e-10) {
                    is_shared = true;
                    break;
                }
            }

            if (is_shared) {
                // This DOF should already be shared via position matching
                // No constraint needed
                continue;
            }

            // This is a hanging DOF - create value constraint
            // u_fine = sum_j L_j(t_coarse) * u_coarse[j]

            VecX weights = lagrange_basis(nodes, t_coarse);

            C2Constraint constraint;
            constraint.slave_dof = fine_global;
            constraint.type = C2Constraint::Type::Value;

            for (size_t j = 0; j < coarse_local_dofs.size(); ++j) {
                if (std::abs(weights(static_cast<int>(j))) > 1e-14) {
                    int coarse_local = coarse_local_dofs[j];
                    Index coarse_global = elem_to_global_[hanging.coarse_elem][coarse_local];

                    constraint.master_dofs.push_back(coarse_global);
                    constraint.weights.push_back(weights(static_cast<int>(j)));
                }
            }

            if (!constraint.master_dofs.empty()) {
                // Only add constraint if DOF not already constrained
                // (can happen when multiple fine elements share the same DOF
                // at the interface between two fine elements)
                if (constrained_dofs_.find(fine_global) == constrained_dofs_.end()) {
                    constraints_.push_back(constraint);
                    constrained_dofs_.insert(fine_global);
                }
            }
        }
    }
}

void CGDofManager::build_dof_mappings() {
    // Build mappings between global and free DOF indices

    global_to_free_.resize(num_global_dofs_, -1);
    free_to_global_.clear();
    free_to_global_.reserve(num_global_dofs_);

    for (Index g = 0; g < num_global_dofs_; ++g) {
        if (!is_constrained(g)) {
            Index f = static_cast<Index>(free_to_global_.size());
            global_to_free_[g] = f;
            free_to_global_.push_back(g);
        }
    }

    num_free_dofs_ = static_cast<Index>(free_to_global_.size());
}

SpMat CGDofManager::build_transformation_matrix() const {
    if (transformation_matrix_built_) {
        return transformation_matrix_;
    }

    // Build T such that u_global = T * u_free
    // For unconstrained DOFs: T(g, global_to_free[g]) = 1
    // For constrained DOFs: T(g, :) encodes the constraint weights

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_global_dofs_ + constraints_.size() * 10);

    // Identity entries for unconstrained DOFs
    for (Index g = 0; g < num_global_dofs_; ++g) {
        Index f = global_to_free_[g];
        if (f >= 0) {
            triplets.emplace_back(g, f, 1.0);
        }
    }

    // Constraint entries
    for (const auto &constraint : constraints_) {
        for (size_t i = 0; i < constraint.master_dofs.size(); ++i) {
            Index master = constraint.master_dofs[i];
            Index f = global_to_free_[master];
            if (f >= 0) {
                triplets.emplace_back(constraint.slave_dof, f, constraint.weights[i]);
            }
        }
    }

    transformation_matrix_.resize(num_global_dofs_, num_free_dofs_);
    transformation_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    transformation_matrix_built_ = true;

    return transformation_matrix_;
}

SpMat CGDofManager::transform_matrix(const SpMat &K) const {
    SpMat T = build_transformation_matrix();
    return T.transpose() * K * T;
}

VecX CGDofManager::transform_rhs(const VecX &f, const SpMat &K) const {
    SpMat T = build_transformation_matrix();
    VecX f_modified = f;

    // For Dirichlet constraints, modify RHS: f -= K * u_dirichlet
    for (const auto &constraint : constraints_) {
        if (constraint.master_dofs.empty()) {
            // This is a Dirichlet BC
            Index slave = constraint.slave_dof;
            Real value = constraint.rhs_value;

            // Subtract contribution from RHS: f_i -= K_i,slave * value
            for (SpMat::InnerIterator it(K, slave); it; ++it) {
                f_modified(it.row()) -= it.value() * value;
            }
        }
    }

    return T.transpose() * f_modified;
}

VecX CGDofManager::expand_solution(const VecX &u_free) const {
    SpMat T = build_transformation_matrix();
    VecX u_global = T * u_free;

    // Set Dirichlet values directly
    for (const auto &constraint : constraints_) {
        if (constraint.master_dofs.empty()) {
            u_global(constraint.slave_dof) = constraint.rhs_value;
        }
    }

    return u_global;
}

std::vector<Index> CGDofManager::domain_corner_dofs() const {
    const auto &domain = mesh_.domain_bounds();
    std::vector<Vec2> corner_positions = {{domain.xmin, domain.ymin},
                                          {domain.xmax, domain.ymin},
                                          {domain.xmin, domain.ymax},
                                          {domain.xmax, domain.ymax}};

    std::vector<Index> dofs;
    for (const auto &pos : corner_positions) {
        Index dof = find_dof_at_position(pos);
        if (dof >= 0) {
            dofs.push_back(dof);
        }
    }
    return dofs;
}

void CGDofManager::apply_corner_dirichlet(const std::vector<Real> &corner_values) {
    auto corner_dofs = domain_corner_dofs();

    for (size_t i = 0; i < corner_dofs.size() && i < corner_values.size(); ++i) {
        apply_single_dirichlet(corner_dofs[i], corner_values[i]);
    }
}

void CGDofManager::apply_single_dirichlet(Index dof, Real value) {
    if (constrained_dofs_.find(dof) == constrained_dofs_.end()) {
        C2Constraint constraint;
        constraint.slave_dof = dof;
        constraint.type = C2Constraint::Type::Value;
        constraint.rhs_value = value;
        // Empty master_dofs indicates Dirichlet BC
        constraints_.push_back(constraint);
        constrained_dofs_.insert(dof);

        // Rebuild DOF mappings after adding constraint
        build_dof_mappings();
        transformation_matrix_built_ = false;
    }
}

} // namespace drifter
