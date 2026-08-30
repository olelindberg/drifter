#include "bathymetry/cg_hermite_dof_manager.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace drifter {

namespace {

/// Weights below this are dropped from constraint rows
constexpr Real WEIGHT_TOLERANCE = 1e-14;

/// Guard against a pathological constraint graph; 2:1 balance bounds the true
/// chain length by the number of refinement levels.
constexpr int MAX_CLOSURE_PASSES = 64;

} // namespace

CGHermiteDofManager::CGHermiteDofManager(const QuadtreeAdapter &mesh, int r,
                                         bool enable_zero_gradient_bc,
                                         const ElementDataMask *mask)
    : CGSurfaceDofManagerBase(mesh), r_(r), dofs_per_node_((r + 1) * (r + 1)), basis_(r),
      enable_zero_gradient_bc_(enable_zero_gradient_bc) {

    if (r < 0 || r > 2) {
        throw std::invalid_argument("CGHermiteDofManager: continuity order r must be 0, 1 or 2");
    }

    const Index num_elements = mesh_.num_elements();
    if (num_elements == 0) {
        num_global_dofs_ = 0;
        num_free_dofs_ = 0;
        return;
    }

    // Domain bounds and quantization tolerance for position-based node sharing.
    // Mesh-relative rather than a fixed absolute scale, so that large projected
    // coordinates (UTM and similar) do not lose resolution.
    Real xmin = std::numeric_limits<Real>::max();
    Real ymin = std::numeric_limits<Real>::max();
    Real min_element_size = std::numeric_limits<Real>::max();
    for (Index e = 0; e < num_elements; ++e) {
        const auto &b = mesh_.element_bounds(e);
        xmin = std::min(xmin, b.xmin);
        ymin = std::min(ymin, b.ymin);
        min_element_size = std::min(min_element_size, std::min(b.xmax - b.xmin, b.ymax - b.ymin));
    }
    xmin_domain_ = xmin;
    ymin_domain_ = ymin;
    inv_quantization_tol_ = 1.0 / (min_element_size * 1e-8);

    initialize_elem_to_global(num_elements, basis_.num_dofs());

    assign_node_dofs();
    identify_boundary_dofs_impl([this](int edge) { return basis_.edge_dofs(edge); });
    build_hanging_node_constraints();
    if (enable_zero_gradient_bc_) {
        build_zero_gradient_constraints();
    }
    if (mask && mask->has_pinned_elements()) {
        build_dirichlet_pin_constraints(*mask);
    }
    close_constraints_transitively();
    compute_length_scales();

    // Reorder by Morton Z-curve for spatial locality. All DOFs of a node share a
    // position, and the sort is stable, so a node's DOFs stay contiguous and in
    // derivative order.
    auto perm = reorder_dofs_by_morton();
    if (!perm.empty()) {
        for (auto &c : constraints_) {
            c.slave_dof = perm[c.slave_dof];
            for (auto &m : c.master_dofs) {
                m = perm[m];
            }
        }

        std::vector<std::pair<int, int>> new_order(dof_deriv_order_.size());
        std::vector<Real> new_scale(dof_length_scale_.size());
        for (size_t i = 0; i < dof_deriv_order_.size(); ++i) {
            new_order[static_cast<size_t>(perm[static_cast<Index>(i)])] = dof_deriv_order_[i];
            new_scale[static_cast<size_t>(perm[static_cast<Index>(i)])] = dof_length_scale_[i];
        }
        dof_deriv_order_ = std::move(new_order);
        dof_length_scale_ = std::move(new_scale);
    }

    build_dof_mappings();
}

// =============================================================================
// Position handling
// =============================================================================

std::pair<int64_t, int64_t> CGHermiteDofManager::quantize_position(const Vec2 &pos) const {
    const Real x_rel = pos(0) - xmin_domain_;
    const Real y_rel = pos(1) - ymin_domain_;
    return std::make_pair(static_cast<int64_t>(std::round(x_rel * inv_quantization_tol_)),
                          static_cast<int64_t>(std::round(y_rel * inv_quantization_tol_)));
}

Index CGHermiteDofManager::find_node(const Vec2 &pos) const {
    auto it = position_to_node_.find(quantize_position(pos));
    return (it != position_to_node_.end()) ? it->second : -1;
}

Index CGHermiteDofManager::find_or_register_node(const Vec2 &pos) {
    auto key = quantize_position(pos);
    auto it = position_to_node_.find(key);
    if (it != position_to_node_.end()) {
        return it->second;
    }

    const Index node = num_nodes_++;
    position_to_node_[key] = node;

    // Every DOF of the node lives at the node's position
    for (int k = 0; k < dofs_per_node_; ++k) {
        register_dof_position(node * dofs_per_node_ + k, pos);
    }
    num_global_dofs_ = num_nodes_ * dofs_per_node_;
    return node;
}

// =============================================================================
// DOF assignment
// =============================================================================

void CGHermiteDofManager::assign_node_dofs() {
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        const auto &bounds = mesh_.element_bounds(e);
        const Real dx = bounds.xmax - bounds.xmin;
        const Real dy = bounds.ymax - bounds.ymin;

        // Register the four corner nodes once each
        Index corner_node[4];
        for (int c = 0; c < 4; ++c) {
            const Vec2 param = basis_.corner_param(c);
            corner_node[c] =
                find_or_register_node(Vec2(bounds.xmin + param(0) * dx, bounds.ymin + param(1) * dy));
        }

        for (int local_dof = 0; local_dof < basis_.num_dofs(); ++local_dof) {
            const auto [a, b] = basis_.deriv_order(local_dof);
            const int corner = basis_.dof_to_corner(local_dof);
            elem_to_global_[e][local_dof] = node_dof(corner_node[corner], a, b);
        }
    }

    dof_deriv_order_.assign(static_cast<size_t>(num_global_dofs_), {0, 0});
    for (Index node = 0; node < num_nodes_; ++node) {
        for (int b = 0; b <= r_; ++b) {
            for (int a = 0; a <= r_; ++a) {
                dof_deriv_order_[static_cast<size_t>(node_dof(node, a, b))] = {a, b};
            }
        }
    }
}

// =============================================================================
// Hanging node constraints
// =============================================================================

void CGHermiteDofManager::build_hanging_node_constraints() {
    // Nodes shared between elements are already identified by position, so the
    // only work here is the T-junction midpoint node, whose DOFs are fully
    // determined by the coarse edge's trace.
    std::unordered_set<Index> already_slaved;

    for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
        for (int edge = 0; edge < 4; ++edge) {
            const EdgeNeighborInfo info = mesh_.get_neighbor(elem, edge);
            if (info.type != EdgeNeighborInfo::Type::FineToCoarse) {
                continue;
            }
            if (info.neighbor_elements.empty()) {
                continue;
            }

            const Index coarse_elem = info.neighbor_elements[0];
            const int coarse_edge = info.neighbor_edges[0];
            const auto &cb = mesh_.element_bounds(coarse_elem);

            // Edges 0/1 are lines of constant u, so they run along y: the
            // tangential direction is y and the normal direction is x.
            const bool tangential_is_y = (coarse_edge == 0 || coarse_edge == 1);

            Vec2 p0, p1;
            Real h_t;
            if (tangential_is_y) {
                const Real x_e = (coarse_edge == 0) ? cb.xmin : cb.xmax;
                p0 = Vec2(x_e, cb.ymin);
                p1 = Vec2(x_e, cb.ymax);
                h_t = cb.ymax - cb.ymin;
            } else {
                const Real y_e = (coarse_edge == 2) ? cb.ymin : cb.ymax;
                p0 = Vec2(cb.xmin, y_e);
                p1 = Vec2(cb.xmax, y_e);
                h_t = cb.xmax - cb.xmin;
            }

            const Vec2 pm = 0.5 * (p0 + p1);

            const Index node_mid = find_node(pm);
            const Index node_0 = find_node(p0);
            const Index node_1 = find_node(p1);
            if (node_mid < 0 || node_0 < 0 || node_1 < 0) {
                continue;
            }
            if (already_slaved.count(node_mid) > 0) {
                continue; // the sibling fine element already handled this node
            }
            already_slaved.insert(node_mid);

            // G_r^phys(h_t): row = slave tangential derivative order,
            //                col = master 1D DOF on the coarse edge
            const MatX G = basis_.midpoint_matrix(h_t);

            // The constraint acts only on the tangential index; the normal index
            // passes through untouched, because the normal derivative is the same
            // physical quantity on both sides.
            for (int a_n = 0; a_n <= r_; ++a_n) {
                for (int b_t = 0; b_t <= r_; ++b_t) {
                    HermiteConstraint c;
                    c.slave_dof = tangential_is_y ? node_dof(node_mid, a_n, b_t)
                                                  : node_dof(node_mid, b_t, a_n);

                    for (int j = 0; j < G.cols(); ++j) {
                        const Real w = G(b_t, j);
                        if (std::abs(w) <= WEIGHT_TOLERANCE) {
                            continue;
                        }
                        const int s_j = j / (r_ + 1);   // coarse edge endpoint
                        const int m_j = j % (r_ + 1);   // tangential derivative order
                        const Index master_node = (s_j == 0) ? node_0 : node_1;
                        const Index master = tangential_is_y ? node_dof(master_node, a_n, m_j)
                                                             : node_dof(master_node, m_j, a_n);
                        c.master_dofs.push_back(master);
                        c.weights.push_back(w);
                    }

                    if (!c.master_dofs.empty()) {
                        constraints_.push_back(std::move(c));
                        constrained_dofs_.insert(constraints_.back().slave_dof);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Boundary conditions
// =============================================================================

void CGHermiteDofManager::build_zero_gradient_constraints() {
    if (r_ == 0) {
        // At r = 0 there are no derivative DOFs, so a zero normal gradient is not
        // expressible. See docs/hermite_bathymetry_system.md S9.
        return;
    }

    // Collect, per node, which normal directions are constrained by a boundary
    // edge. A domain corner sits on two boundary edges and pins both families.
    std::unordered_map<Index, std::array<bool, 2>> node_normals;

    mesh_.for_each_boundary_edge([&](Index elem, int edge) {
        const auto &b = mesh_.element_bounds(elem);
        const bool tangential_is_y = (edge == 0 || edge == 1);
        const int normal_dir = tangential_is_y ? 0 : 1; // 0 = x, 1 = y

        Vec2 p0, p1;
        if (tangential_is_y) {
            const Real x_e = (edge == 0) ? b.xmin : b.xmax;
            p0 = Vec2(x_e, b.ymin);
            p1 = Vec2(x_e, b.ymax);
        } else {
            const Real y_e = (edge == 2) ? b.ymin : b.ymax;
            p0 = Vec2(b.xmin, y_e);
            p1 = Vec2(b.xmax, y_e);
        }

        for (const Vec2 &p : {p0, p1}) {
            const Index node = find_node(p);
            if (node >= 0) {
                node_normals[node][static_cast<size_t>(normal_dir)] = true;
            }
        }
    });

    for (const auto &[node, normals] : node_normals) {
        for (int dir = 0; dir < 2; ++dir) {
            if (!normals[static_cast<size_t>(dir)]) {
                continue;
            }
            // Pin every DOF with derivative order >= 1 in the normal direction:
            // at r = 1 that is z_n and z_nt.
            for (int b = 0; b <= r_; ++b) {
                for (int a = 0; a <= r_; ++a) {
                    const int normal_order = (dir == 0) ? a : b;
                    if (normal_order == 0) {
                        continue;
                    }
                    const Index dof = node_dof(node, a, b);
                    if (constrained_dofs_.count(dof) > 0) {
                        continue; // already a hanging slave; do not double-constrain
                    }
                    HermiteConstraint c;
                    c.slave_dof = dof; // empty master list => pinned to zero
                    constraints_.push_back(std::move(c));
                    constrained_dofs_.insert(dof);
                }
            }
        }
    }
}

void CGHermiteDofManager::build_dirichlet_pin_constraints(const ElementDataMask &mask) {
    auto pin_dof = [this](Index dof) {
        if (constrained_dofs_.count(dof) > 0) {
            return; // already a hanging slave or a BC pin; do not double-constrain
        }
        HermiteConstraint c;
        c.slave_dof = dof; // empty master list => pinned to zero
        constraints_.push_back(std::move(c));
        constrained_dofs_.insert(dof);
        ++num_pinned_dofs_;
    };

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        if (!mask.is_pinned(e)) {
            continue;
        }

        const auto &b = mesh_.element_bounds(e);
        const std::array<Vec2, 4> corners = {Vec2(b.xmin, b.ymin), Vec2(b.xmax, b.ymin),
                                             Vec2(b.xmin, b.ymax), Vec2(b.xmax, b.ymax)};

        for (const Vec2 &corner : corners) {
            const Index node = find_node(corner);
            if (node < 0) {
                continue;
            }
            for (int bd = 0; bd <= r_; ++bd) {
                for (int ad = 0; ad <= r_; ++ad) {
                    pin_dof(node_dof(node, ad, bd));
                }
            }
        }
    }
}

// =============================================================================
// Transitive closure
// =============================================================================

void CGHermiteDofManager::close_constraints_transitively() {
    if (constraints_.empty()) {
        return;
    }

    std::unordered_map<Index, size_t> slave_to_constraint;
    for (size_t i = 0; i < constraints_.size(); ++i) {
        slave_to_constraint[constraints_[i].slave_dof] = i;
    }

    for (int pass = 0; pass < MAX_CLOSURE_PASSES; ++pass) {
        bool changed = false;

        for (auto &c : constraints_) {
            bool has_slave_master = false;
            for (Index m : c.master_dofs) {
                if (slave_to_constraint.count(m) > 0) {
                    has_slave_master = true;
                    break;
                }
            }
            if (!has_slave_master) {
                continue;
            }

            // Substitute each slave master by its own expansion, accumulating
            // weights so repeated masters collapse into one entry.
            std::unordered_map<Index, Real> expanded;
            for (size_t i = 0; i < c.master_dofs.size(); ++i) {
                const Index m = c.master_dofs[i];
                const Real w = c.weights[i];

                auto it = slave_to_constraint.find(m);
                if (it == slave_to_constraint.end()) {
                    expanded[m] += w;
                    continue;
                }
                const auto &inner = constraints_[it->second];
                for (size_t k = 0; k < inner.master_dofs.size(); ++k) {
                    expanded[inner.master_dofs[k]] += w * inner.weights[k];
                }
            }

            c.master_dofs.clear();
            c.weights.clear();
            for (const auto &[dof, w] : expanded) {
                if (std::abs(w) > WEIGHT_TOLERANCE) {
                    c.master_dofs.push_back(dof);
                    c.weights.push_back(w);
                }
            }
            changed = true;
        }

        if (!changed) {
            return;
        }
    }

    throw std::runtime_error("CGHermiteDofManager: hanging node constraints did not close after " +
                             std::to_string(MAX_CLOSURE_PASSES) +
                             " passes (cyclic constraint graph?)");
}

// =============================================================================
// Length scales for equilibration
// =============================================================================

void CGHermiteDofManager::compute_length_scales() {
    dof_length_scale_.assign(static_cast<size_t>(num_global_dofs_),
                             std::numeric_limits<Real>::max());

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        const auto &b = mesh_.element_bounds(e);
        const Real h = std::min(b.xmax - b.xmin, b.ymax - b.ymin);
        for (Index dof : elem_to_global_[e]) {
            dof_length_scale_[static_cast<size_t>(dof)] =
                std::min(dof_length_scale_[static_cast<size_t>(dof)], h);
        }
    }

    // A DOF touched by no element would leave the sentinel in place
    for (auto &s : dof_length_scale_) {
        if (s == std::numeric_limits<Real>::max()) {
            s = 1.0;
        }
    }
}

// =============================================================================
// Queries
// =============================================================================

std::pair<int, int> CGHermiteDofManager::deriv_order(Index global_dof) const {
    if (global_dof < 0 || global_dof >= num_global_dofs_) {
        throw std::out_of_range("CGHermiteDofManager::deriv_order: DOF out of range");
    }
    return dof_deriv_order_[static_cast<size_t>(global_dof)];
}

int CGHermiteDofManager::total_deriv_order(Index global_dof) const {
    const auto [a, b] = deriv_order(global_dof);
    return a + b;
}

Real CGHermiteDofManager::length_scale(Index global_dof) const {
    if (global_dof < 0 || global_dof >= num_global_dofs_) {
        throw std::out_of_range("CGHermiteDofManager::length_scale: DOF out of range");
    }
    return dof_length_scale_[static_cast<size_t>(global_dof)];
}

VecX CGHermiteDofManager::equilibration_scaling() const {
    VecX S(num_global_dofs_);
    for (Index g = 0; g < num_global_dofs_; ++g) {
        S(g) = std::pow(dof_length_scale_[static_cast<size_t>(g)], total_deriv_order(g));
    }
    return S;
}

void CGHermiteDofManager::get_constraint_triplets(
    std::vector<Eigen::Triplet<Real>> &triplets) const {
    triplets.reserve(constraints_.size() * (1 + 2 * (r_ + 1)));

    for (Index row = 0; row < num_constraints(); ++row) {
        const auto &c = constraints_[static_cast<size_t>(row)];
        triplets.emplace_back(row, c.slave_dof, 1.0);
        for (size_t i = 0; i < c.master_dofs.size(); ++i) {
            triplets.emplace_back(row, c.master_dofs[i], -c.weights[i]);
        }
    }
}

} // namespace drifter
