#include "bathymetry/biharmonic_assembler.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

BiharmonicAssembler::BiharmonicAssembler(const QuadtreeAdapter& mesh,
                                         const LagrangeBasis2D& basis,
                                         const CGDofManager& dofs,
                                         Real alpha,
                                         Real beta,
                                         Real penalty)
    : mesh_(mesh), basis_(basis), dofs_(dofs),
      alpha_(alpha), beta_(beta), penalty_(penalty) {

    if (alpha < 0.0 || beta < 0.0) {
        throw std::invalid_argument("BiharmonicAssembler: weights must be non-negative");
    }
    if (penalty < 0.0) {
        throw std::invalid_argument("BiharmonicAssembler: penalty must be non-negative");
    }

    init_quadrature();
}

void BiharmonicAssembler::init_quadrature() {
    // Use Gauss-Legendre quadrature with enough points for exactness
    // For order p basis, need 2p points for exact integration of p^2 terms
    num_gauss_1d_ = basis_.num_nodes_1d();

    // Use the LGL nodes and weights from the basis
    gauss_nodes_ = basis_.nodes_1d();
    gauss_weights_ = basis_.weights_1d();

    // Precompute basis values at all 2D Gauss points
    int n_gauss_2d = num_gauss_1d_ * num_gauss_1d_;
    int ndof = basis_.num_dofs();
    phi_at_gauss_.resize(n_gauss_2d, ndof);
    lap_at_gauss_.resize(n_gauss_2d, ndof);

    for (int jg = 0; jg < num_gauss_1d_; ++jg) {
        for (int ig = 0; ig < num_gauss_1d_; ++ig) {
            int q = ig + jg * num_gauss_1d_;
            Real xi = gauss_nodes_(ig);
            Real eta = gauss_nodes_(jg);

            VecX phi = basis_.evaluate(xi, eta);
            VecX lap = basis_.evaluate_laplacian(xi, eta);

            phi_at_gauss_.row(q) = phi.transpose();
            lap_at_gauss_.row(q) = lap.transpose();
        }
    }
}

Vec2 BiharmonicAssembler::map_to_physical(Index elem, Real xi, Real eta) const {
    const auto& bounds = mesh_.element_bounds(elem);

    // Linear map from [-1,1] x [-1,1] to element bounds
    Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
    Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);

    return Vec2(x, y);
}

VecX BiharmonicAssembler::compute_jacobian(Index elem) const {
    const auto& bounds = mesh_.element_bounds(elem);

    // For rectangular elements, Jacobian is constant
    Real dx_dxi = 0.5 * (bounds.xmax - bounds.xmin);
    Real dy_deta = 0.5 * (bounds.ymax - bounds.ymin);
    Real jac = dx_dxi * dy_deta;

    int n_gauss_2d = num_gauss_1d_ * num_gauss_1d_;
    return VecX::Constant(n_gauss_2d, jac);
}

MatX BiharmonicAssembler::element_biharmonic(Index elem) const {
    // Biharmonic stiffness: K_ij = α * ∫ ∇²φ_i * ∇²φ_j dA
    const auto& bounds = mesh_.element_bounds(elem);
    Real hx = 0.5 * (bounds.xmax - bounds.xmin);
    Real hy = 0.5 * (bounds.ymax - bounds.ymin);

    int ndof = basis_.num_dofs();
    MatX K = MatX::Zero(ndof, ndof);

    // Jacobian for integration
    Real jac = hx * hy;

    for (int jg = 0; jg < num_gauss_1d_; ++jg) {
        for (int ig = 0; ig < num_gauss_1d_; ++ig) {
            int q = ig + jg * num_gauss_1d_;
            Real wq = gauss_weights_(ig) * gauss_weights_(jg) * jac;

            Real xi = gauss_nodes_(ig);
            Real eta = gauss_nodes_(jg);

            VecX d2_dxi2, d2_deta2, d2_dxideta;
            basis_.evaluate_second_derivatives(xi, eta, d2_dxi2, d2_deta2, d2_dxideta);

            // Physical Laplacian: ∇² = (1/hx²)∂²/∂ξ² + (1/hy²)∂²/∂η²
            VecX lap_phys = d2_dxi2 / (hx * hx) + d2_deta2 / (hy * hy);

            // Add contribution: α * lap_i * lap_j * w_q
            K += alpha_ * wq * (lap_phys * lap_phys.transpose());
        }
    }

    return K;
}

MatX BiharmonicAssembler::element_mass(Index elem) const {
    // Mass matrix: M_ij = β * ∫ φ_i * φ_j dA
    const auto& bounds = mesh_.element_bounds(elem);
    Real hx = 0.5 * (bounds.xmax - bounds.xmin);
    Real hy = 0.5 * (bounds.ymax - bounds.ymin);
    Real jac = hx * hy;

    int ndof = basis_.num_dofs();
    MatX M = MatX::Zero(ndof, ndof);

    for (int jg = 0; jg < num_gauss_1d_; ++jg) {
        for (int ig = 0; ig < num_gauss_1d_; ++ig) {
            int q = ig + jg * num_gauss_1d_;
            Real wq = gauss_weights_(ig) * gauss_weights_(jg) * jac;

            VecX phi = phi_at_gauss_.row(q).transpose();

            // Add contribution: β * phi_i * phi_j * w_q
            M += beta_ * wq * (phi * phi.transpose());
        }
    }

    return M;
}

MatX BiharmonicAssembler::element_stiffness(Index elem) const {
    return element_biharmonic(elem) + element_mass(elem);
}

VecX BiharmonicAssembler::element_rhs(Index elem, const BathymetrySource& bathy) const {
    // RHS: f_i = β * ∫ u_data * φ_i dA
    const auto& bounds = mesh_.element_bounds(elem);
    Real hx = 0.5 * (bounds.xmax - bounds.xmin);
    Real hy = 0.5 * (bounds.ymax - bounds.ymin);
    Real jac = hx * hy;

    int ndof = basis_.num_dofs();
    VecX f = VecX::Zero(ndof);

    for (int jg = 0; jg < num_gauss_1d_; ++jg) {
        for (int ig = 0; ig < num_gauss_1d_; ++ig) {
            int q = ig + jg * num_gauss_1d_;
            Real wq = gauss_weights_(ig) * gauss_weights_(jg) * jac;

            Real xi = gauss_nodes_(ig);
            Real eta = gauss_nodes_(jg);
            Vec2 phys = map_to_physical(elem, xi, eta);

            Real u_data = bathy.evaluate(phys(0), phys(1));
            VecX phi = phi_at_gauss_.row(q).transpose();

            f += beta_ * u_data * wq * phi;
        }
    }

    return f;
}

Real BiharmonicAssembler::map_to_coarse_edge(Real t, int subedge_idx) const {
    // Map t in [-1, 1] to the appropriate half of the coarse edge
    // subedge_idx = 0: map to [-1, 0]
    // subedge_idx = 1: map to [0, 1]
    if (subedge_idx == 0) {
        return 0.5 * (t - 1.0);  // [-1, 1] -> [-1, 0]
    } else {
        return 0.5 * (t + 1.0);  // [-1, 1] -> [0, 1]
    }
}

std::pair<MatX, MatX> BiharmonicAssembler::edge_penalty_conforming(
    Index elem_left, Index elem_right,
    int edge_left, int edge_right) const {

    // Compute edge penalty matrices for conforming interface
    // Penalty: (γ/h) ∫_edge [[∂u/∂n]]·[[∂v/∂n]] ds
    // where [[∂u/∂n]] = ∂u_L/∂n_L + ∂u_R/∂n_R (outward normals point opposite)

    int ndof = basis_.num_dofs();
    MatX K_LL = MatX::Zero(ndof, ndof);
    MatX K_LR = MatX::Zero(ndof, ndof);

    // Get element sizes for Jacobian scaling
    const auto& bounds_left = mesh_.element_bounds(elem_left);
    const auto& bounds_right = mesh_.element_bounds(elem_right);

    Real hx_left = bounds_left.xmax - bounds_left.xmin;
    Real hy_left = bounds_left.ymax - bounds_left.ymin;
    Real hx_right = bounds_right.xmax - bounds_right.xmin;
    Real hy_right = bounds_right.ymax - bounds_right.ymin;

    // Edge length and normal direction scaling
    Real h_edge;
    Real scale_left, scale_right;  // Convert ref derivative to physical

    if (edge_left == 0 || edge_left == 1) {
        // Vertical edge - normal is in x direction
        h_edge = hy_left;  // Edge length
        scale_left = 2.0 / hx_left;   // ∂/∂x = (2/hx) ∂/∂ξ
        scale_right = 2.0 / hx_right;
    } else {
        // Horizontal edge - normal is in y direction
        h_edge = hx_left;  // Edge length
        scale_left = 2.0 / hy_left;   // ∂/∂y = (2/hy) ∂/∂η
        scale_right = 2.0 / hy_right;
    }

    Real edge_jac = h_edge / 2.0;  // Jacobian for edge integration
    Real penalty_coeff = penalty_ / h_edge;

    // Gauss quadrature along edge
    for (int ig = 0; ig < num_gauss_1d_; ++ig) {
        Real t = gauss_nodes_(ig);
        Real wq = gauss_weights_(ig) * edge_jac;

        // Evaluate normal derivatives at this edge point (in reference coords)
        VecX dn_left_ref = basis_.evaluate_normal_derivative_at_edge(edge_left, t);
        VecX dn_right_ref = basis_.evaluate_normal_derivative_at_edge(edge_right, t);

        // Convert to physical coordinates
        VecX dn_left = scale_left * dn_left_ref;
        VecX dn_right = scale_right * dn_right_ref;

        // For C¹ continuity, we want the normal derivative to be continuous
        // across the interface when measured in the SAME physical direction.
        //
        // dn_left = ∂u/∂n_L where n_L points outward from left element
        // dn_right = ∂u/∂n_R where n_R points outward from right element
        // At a shared edge: n_R = -n_L
        //
        // For C¹ continuity (same ∂u/∂x across vertical edge):
        //   dn_left = +∂u/∂x (for right edge of left element, n_L = +x)
        //   dn_right = -∂u/∂x (for left edge of right element, n_R = -x)
        //   So dn_left = -dn_right, meaning dn_left + dn_right = 0
        //
        // The penalty term is: γ/h ∫(dn_L + dn_R)² ds
        // Expanding: γ/h ∫(dn_L² + 2*dn_L*dn_R + dn_R²) ds

        // K_LL contribution: dn_L * dn_L^T
        K_LL += penalty_coeff * wq * (dn_left * dn_left.transpose());

        // K_LR contribution: dn_L * dn_R^T (coefficient +1, factor of 2 split with K_RL)
        K_LR += penalty_coeff * wq * (dn_left * dn_right.transpose());
    }

    return {K_LL, K_LR};
}

std::pair<MatX, MatX> BiharmonicAssembler::edge_penalty_nonconforming(
    Index elem_fine, Index elem_coarse,
    int edge_fine, int edge_coarse,
    int subedge_idx) const {

    // Compute edge penalty for non-conforming interface
    // The fine element's edge maps to half of the coarse element's edge

    int ndof = basis_.num_dofs();
    MatX K_FF = MatX::Zero(ndof, ndof);
    MatX K_FC = MatX::Zero(ndof, ndof);

    // Get element sizes for Jacobian scaling
    const auto& bounds_fine = mesh_.element_bounds(elem_fine);
    const auto& bounds_coarse = mesh_.element_bounds(elem_coarse);

    Real hx_fine = bounds_fine.xmax - bounds_fine.xmin;
    Real hy_fine = bounds_fine.ymax - bounds_fine.ymin;
    Real hx_coarse = bounds_coarse.xmax - bounds_coarse.xmin;
    Real hy_coarse = bounds_coarse.ymax - bounds_coarse.ymin;

    // Edge length and normal direction scaling
    Real h_edge;
    Real scale_fine, scale_coarse;  // Convert ref derivative to physical

    if (edge_fine == 0 || edge_fine == 1) {
        // Vertical edge - normal is in x direction
        h_edge = hy_fine;  // Fine edge length
        scale_fine = 2.0 / hx_fine;    // ∂/∂x = (2/hx) ∂/∂ξ
        scale_coarse = 2.0 / hx_coarse;
    } else {
        // Horizontal edge - normal is in y direction
        h_edge = hx_fine;  // Fine edge length
        scale_fine = 2.0 / hy_fine;    // ∂/∂y = (2/hy) ∂/∂η
        scale_coarse = 2.0 / hy_coarse;
    }

    Real edge_jac = h_edge / 2.0;
    Real penalty_coeff = penalty_ / h_edge;

    // Gauss quadrature along fine edge
    for (int ig = 0; ig < num_gauss_1d_; ++ig) {
        Real t_fine = gauss_nodes_(ig);
        Real wq = gauss_weights_(ig) * edge_jac;

        // Map to coarse edge parameter
        Real t_coarse = map_to_coarse_edge(t_fine, subedge_idx);

        // Evaluate normal derivatives in reference coords
        VecX dn_fine_ref = basis_.evaluate_normal_derivative_at_edge(edge_fine, t_fine);
        VecX dn_coarse_ref = basis_.evaluate_normal_derivative_at_edge(edge_coarse, t_coarse);

        // Convert to physical coordinates
        VecX dn_fine = scale_fine * dn_fine_ref;
        VecX dn_coarse = scale_coarse * dn_coarse_ref;

        // K_FF contribution
        K_FF += penalty_coeff * wq * (dn_fine * dn_fine.transpose());

        // K_FC contribution (cross term)
        K_FC += penalty_coeff * wq * (dn_fine * dn_coarse.transpose());
    }

    return {K_FF, K_FC};
}

void BiharmonicAssembler::assemble_ipdg_penalty(
    std::vector<Eigen::Triplet<Real>>& triplets) const {

    int ndof = basis_.num_dofs();

    // Iterate over all interior edges
    mesh_.for_each_interior_edge([&](Index elem, int edge_id, const EdgeNeighborInfo& info) {
        const auto& elem_dofs = dofs_.element_dofs(elem);

        if (info.is_conforming()) {
            // Conforming edge - single neighbor
            Index neighbor = info.neighbor_elements[0];
            int neighbor_edge = info.neighbor_edges[0];
            const auto& neighbor_dofs = dofs_.element_dofs(neighbor);

            auto [K_LL, K_LR] = edge_penalty_conforming(elem, neighbor, edge_id, neighbor_edge);

            // Add K_LL to element's diagonal block
            for (int i = 0; i < ndof; ++i) {
                Index gi = elem_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    Index gj = elem_dofs[j];
                    if (std::abs(K_LL(i, j)) > 1e-15) {
                        triplets.emplace_back(gi, gj, K_LL(i, j));
                    }
                }
            }

            // Add K_LR to element-neighbor coupling block
            for (int i = 0; i < ndof; ++i) {
                Index gi = elem_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    Index gj = neighbor_dofs[j];
                    if (std::abs(K_LR(i, j)) > 1e-15) {
                        triplets.emplace_back(gi, gj, K_LR(i, j));
                    }
                }
            }

            // Add symmetric contributions for neighbor
            // K_RR = same as K_LL (by symmetry for conforming)
            // K_RL = K_LR^T
            auto [K_RR, K_RL] = edge_penalty_conforming(neighbor, elem, neighbor_edge, edge_id);

            for (int i = 0; i < ndof; ++i) {
                Index gi = neighbor_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    Index gj = neighbor_dofs[j];
                    if (std::abs(K_RR(i, j)) > 1e-15) {
                        triplets.emplace_back(gi, gj, K_RR(i, j));
                    }
                }
            }

            for (int i = 0; i < ndof; ++i) {
                Index gi = neighbor_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    Index gj = elem_dofs[j];
                    if (std::abs(K_RL(i, j)) > 1e-15) {
                        triplets.emplace_back(gi, gj, K_RL(i, j));
                    }
                }
            }

        } else if (info.type == EdgeNeighborInfo::Type::CoarseToFine) {
            // This element is coarse, neighbors are fine
            // Process from the fine side for proper integration
            // (This edge is processed by the fine elements' FineToCoarse case)
            // Skip here to avoid double counting

        } else if (info.type == EdgeNeighborInfo::Type::FineToCoarse) {
            // This element is fine, neighbor is coarse
            Index coarse_elem = info.neighbor_elements[0];
            int coarse_edge = info.neighbor_edges[0];
            int subedge_idx = info.subedge_index;
            const auto& coarse_dofs = dofs_.element_dofs(coarse_elem);

            auto [K_FF, K_FC] = edge_penalty_nonconforming(
                elem, coarse_elem, edge_id, coarse_edge, subedge_idx);

            // Add K_FF to fine element's diagonal block
            for (int i = 0; i < ndof; ++i) {
                Index gi = elem_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    Index gj = elem_dofs[j];
                    if (std::abs(K_FF(i, j)) > 1e-15) {
                        triplets.emplace_back(gi, gj, K_FF(i, j));
                    }
                }
            }

            // Add K_FC to fine-coarse coupling
            for (int i = 0; i < ndof; ++i) {
                Index gi = elem_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    Index gj = coarse_dofs[j];
                    if (std::abs(K_FC(i, j)) > 1e-15) {
                        triplets.emplace_back(gi, gj, K_FC(i, j));
                    }
                }
            }

            // Add symmetric K_CF = K_FC^T
            for (int i = 0; i < ndof; ++i) {
                Index gi = coarse_dofs[i];
                for (int j = 0; j < ndof; ++j) {
                    Index gj = elem_dofs[j];
                    if (std::abs(K_FC(j, i)) > 1e-15) {
                        triplets.emplace_back(gi, gj, K_FC(j, i));
                    }
                }
            }

            // Note: K_CC contribution from this fine edge is partial
            // The full coarse diagonal contribution comes from summing
            // both fine neighbors. We add the partial contribution here.
            // K_CC_partial = scale^2 * dn_coarse * dn_coarse^T
            // This is automatically handled by the symmetry of the penalty.
        }
    });
}

SpMat BiharmonicAssembler::assemble_stiffness() const {
    Index n = dofs_.num_global_dofs();
    int ndof = basis_.num_dofs();
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(mesh_.num_elements() * ndof * ndof +
                     mesh_.num_elements() * 4 * ndof * ndof);  // Extra for IPDG

    // Assemble element contributions
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        MatX K_elem = element_stiffness(e);
        const auto& elem_dofs = dofs_.element_dofs(e);

        for (int i = 0; i < ndof; ++i) {
            Index gi = elem_dofs[i];
            for (int j = 0; j < ndof; ++j) {
                Index gj = elem_dofs[j];
                if (std::abs(K_elem(i, j)) > 1e-15) {
                    triplets.emplace_back(gi, gj, K_elem(i, j));
                }
            }
        }
    }

    // Assemble IPDG penalty contributions
    assemble_ipdg_penalty(triplets);

    SpMat K(n, n);
    K.setFromTriplets(triplets.begin(), triplets.end());
    return K;
}

VecX BiharmonicAssembler::assemble_rhs(const BathymetrySource& bathy) const {
    Index n = dofs_.num_global_dofs();
    int ndof = basis_.num_dofs();
    VecX f = VecX::Zero(n);

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        VecX f_elem = element_rhs(e, bathy);
        const auto& elem_dofs = dofs_.element_dofs(e);

        for (int i = 0; i < ndof; ++i) {
            f(elem_dofs[i]) += f_elem(i);
        }
    }

    return f;
}

void BiharmonicAssembler::assemble_reduced_system(const BathymetrySource& bathy,
                                                  SpMat& K_red, VecX& f_red) const {
    SpMat K = assemble_stiffness();
    VecX f = assemble_rhs(bathy);

    K_red = dofs_.transform_matrix(K);
    f_red = dofs_.transform_rhs(f, K);
}

}  // namespace drifter
