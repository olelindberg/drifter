#include "bathymetry/quintic_basis_2d.hpp"
#include "dg/basis_hexahedron.hpp" // For compute_gauss_lobatto_nodes, etc.
#include <cmath>
#include <stdexcept>

namespace drifter {

QuinticBasis2D::QuinticBasis2D() {
    compute_nodes_and_weights();
    compute_derivative_matrices();
}

void QuinticBasis2D::compute_nodes_and_weights() {
    // Use the existing LGL node computation from basis_hexahedron
    compute_gauss_lobatto_nodes(N1D, nodes_, weights_);

    // Compute barycentric weights
    bary_ = compute_barycentric_weights(nodes_);
}

void QuinticBasis2D::compute_derivative_matrices() {
    // First derivative matrix using existing function
    D_ = compute_derivative_matrix_1d(nodes_);

    // Second derivative matrix: D2 = D * D
    D2_ = D_ * D_;
}

Vec2 QuinticBasis2D::node_position(int dof) const {
    int i, j;
    dof_ij(dof, i, j);
    return Vec2(nodes_(i), nodes_(j));
}

VecX QuinticBasis2D::evaluate_1d(Real xi) const {
    VecX phi(N1D);

    // Use barycentric formula for stable evaluation
    VecX temp(N1D);
    Real sum = 0.0;
    bool at_node = false;
    int node_idx = -1;

    for (int j = 0; j < N1D; ++j) {
        Real diff = xi - nodes_(j);
        if (std::abs(diff) < 1e-14) {
            at_node = true;
            node_idx = j;
            break;
        }
        temp(j) = bary_(j) / diff;
        sum += temp(j);
    }

    if (at_node) {
        phi.setZero();
        phi(node_idx) = 1.0;
    } else {
        for (int j = 0; j < N1D; ++j) {
            phi(j) = temp(j) / sum;
        }
    }

    return phi;
}

VecX QuinticBasis2D::evaluate_derivative_1d(Real xi) const {
    VecX dphi(N1D);

    // Check if at a node
    bool at_node = false;
    int node_idx = -1;
    for (int j = 0; j < N1D; ++j) {
        if (std::abs(xi - nodes_(j)) < 1e-14) {
            at_node = true;
            node_idx = j;
            break;
        }
    }

    if (at_node) {
        // Use the derivative matrix row
        dphi = D_.row(node_idx).transpose();
    } else {
        // General formula: l'_j(x) = l_j(x) * sum_{k!=j} 1/(x - x_k)
        VecX phi = evaluate_1d(xi);
        for (int j = 0; j < N1D; ++j) {
            Real inner_sum = 0.0;
            for (int k = 0; k < N1D; ++k) {
                if (k != j) {
                    inner_sum += 1.0 / (xi - nodes_(k));
                }
            }
            dphi(j) = phi(j) * inner_sum;
        }
    }

    return dphi;
}

VecX QuinticBasis2D::evaluate_second_derivative_1d(Real xi) const {
    VecX d2phi(N1D);

    // Check if at a node
    bool at_node = false;
    int node_idx = -1;
    for (int j = 0; j < N1D; ++j) {
        if (std::abs(xi - nodes_(j)) < 1e-14) {
            at_node = true;
            node_idx = j;
            break;
        }
    }

    if (at_node) {
        // Use the second derivative matrix row
        d2phi = D2_.row(node_idx).transpose();
    } else {
        // Second derivative formula:
        // l''_j(x) = l_j(x) * [(sum_{k!=j} 1/(x-x_k))^2 - sum_{k!=j}
        // 1/(x-x_k)^2]
        VecX phi = evaluate_1d(xi);

        for (int j = 0; j < N1D; ++j) {
            Real sum1 = 0.0; // sum_{k!=j} 1/(x - x_k)
            Real sum2 = 0.0; // sum_{k!=j} 1/(x - x_k)^2

            for (int k = 0; k < N1D; ++k) {
                if (k != j) {
                    Real diff_inv = 1.0 / (xi - nodes_(k));
                    sum1 += diff_inv;
                    sum2 += diff_inv * diff_inv;
                }
            }

            d2phi(j) = phi(j) * (sum1 * sum1 - sum2);
        }
    }

    return d2phi;
}

VecX QuinticBasis2D::evaluate(Real xi, Real eta) const {
    VecX phi_xi = evaluate_1d(xi);
    VecX phi_eta = evaluate_1d(eta);

    VecX phi(NDOF);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            int dof = dof_index(i, j);
            phi(dof) = phi_xi(i) * phi_eta(j);
        }
    }

    return phi;
}

MatX QuinticBasis2D::evaluate_gradient(Real xi, Real eta) const {
    VecX phi_xi = evaluate_1d(xi);
    VecX phi_eta = evaluate_1d(eta);
    VecX dphi_xi = evaluate_derivative_1d(xi);
    VecX dphi_eta = evaluate_derivative_1d(eta);

    MatX grad(NDOF, 2);
    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            int dof = dof_index(i, j);
            grad(dof, 0) = dphi_xi(i) * phi_eta(j); // d/dxi
            grad(dof, 1) = phi_xi(i) * dphi_eta(j); // d/deta
        }
    }

    return grad;
}

void QuinticBasis2D::evaluate_second_derivatives(Real xi, Real eta, VecX &d2_dxi2, VecX &d2_deta2,
                                                 VecX &d2_dxideta) const {
    VecX phi_xi = evaluate_1d(xi);
    VecX phi_eta = evaluate_1d(eta);
    VecX dphi_xi = evaluate_derivative_1d(xi);
    VecX dphi_eta = evaluate_derivative_1d(eta);
    VecX d2phi_xi = evaluate_second_derivative_1d(xi);
    VecX d2phi_eta = evaluate_second_derivative_1d(eta);

    d2_dxi2.resize(NDOF);
    d2_deta2.resize(NDOF);
    d2_dxideta.resize(NDOF);

    for (int j = 0; j < N1D; ++j) {
        for (int i = 0; i < N1D; ++i) {
            int dof = dof_index(i, j);
            d2_dxi2(dof) = d2phi_xi(i) * phi_eta(j);
            d2_deta2(dof) = phi_xi(i) * d2phi_eta(j);
            d2_dxideta(dof) = dphi_xi(i) * dphi_eta(j);
        }
    }
}

VecX QuinticBasis2D::evaluate_laplacian(Real xi, Real eta) const {
    VecX d2_dxi2, d2_deta2, d2_dxideta;
    evaluate_second_derivatives(xi, eta, d2_dxi2, d2_deta2, d2_dxideta);

    // Laplacian in reference coordinates = d^2/dxi^2 + d^2/deta^2
    return d2_dxi2 + d2_deta2;
}

std::vector<Mat2> QuinticBasis2D::evaluate_hessian(Real xi, Real eta) const {
    VecX d2_dxi2, d2_deta2, d2_dxideta;
    evaluate_second_derivatives(xi, eta, d2_dxi2, d2_deta2, d2_dxideta);

    std::vector<Mat2> hessians(NDOF);
    for (int dof = 0; dof < NDOF; ++dof) {
        hessians[dof](0, 0) = d2_dxi2(dof);
        hessians[dof](0, 1) = d2_dxideta(dof);
        hessians[dof](1, 0) = d2_dxideta(dof);
        hessians[dof](1, 1) = d2_deta2(dof);
    }

    return hessians;
}

std::vector<int> QuinticBasis2D::edge_dofs(int edge_id) const {
    std::vector<int> dofs(N1D);

    switch (edge_id) {
    case 0: // xi = -1 (left edge)
        for (int j = 0; j < N1D; ++j) {
            dofs[j] = dof_index(0, j);
        }
        break;
    case 1: // xi = +1 (right edge)
        for (int j = 0; j < N1D; ++j) {
            dofs[j] = dof_index(N1D - 1, j);
        }
        break;
    case 2: // eta = -1 (bottom edge)
        for (int i = 0; i < N1D; ++i) {
            dofs[i] = dof_index(i, 0);
        }
        break;
    case 3: // eta = +1 (top edge)
        for (int i = 0; i < N1D; ++i) {
            dofs[i] = dof_index(i, N1D - 1);
        }
        break;
    default:
        throw std::out_of_range("edge_id must be 0-3");
    }

    return dofs;
}

int QuinticBasis2D::corner_dof(int corner_id) const {
    switch (corner_id) {
    case 0:
        return dof_index(0, 0); // (-1, -1)
    case 1:
        return dof_index(N1D - 1, 0); // (+1, -1)
    case 2:
        return dof_index(0, N1D - 1); // (-1, +1)
    case 3:
        return dof_index(N1D - 1, N1D - 1); // (+1, +1)
    default:
        throw std::out_of_range("corner_id must be 0-3");
    }
}

bool QuinticBasis2D::is_boundary_dof(int dof) const {
    int i, j;
    dof_ij(dof, i, j);
    return (i == 0 || i == N1D - 1 || j == 0 || j == N1D - 1);
}

std::vector<int> QuinticBasis2D::interior_dofs() const {
    std::vector<int> interior;
    interior.reserve((N1D - 2) * (N1D - 2)); // 4x4 = 16 for quintic

    for (int j = 1; j < N1D - 1; ++j) {
        for (int i = 1; i < N1D - 1; ++i) {
            interior.push_back(dof_index(i, j));
        }
    }

    return interior;
}

} // namespace drifter
