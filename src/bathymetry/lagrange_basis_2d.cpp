#include "bathymetry/lagrange_basis_2d.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

LagrangeBasis2D::LagrangeBasis2D(int order)
    : order_(order), n1d_(order + 1), ndof_((order + 1) * (order + 1)) {

    if (order < 1) {
        throw std::invalid_argument("LagrangeBasis2D: order must be >= 1");
    }

    // Create 1D LGL basis
    basis_1d_ = LagrangeBasis1D::create_lgl(order);

    // Compute second derivative matrix: D2 = D * D
    D2_ = basis_1d_.D * basis_1d_.D;
}

Vec2 LagrangeBasis2D::node_position(int dof) const {
    int i, j;
    dof_ij(dof, i, j);
    return Vec2(basis_1d_.nodes(i), basis_1d_.nodes(j));
}

VecX LagrangeBasis2D::evaluate(Real xi, Real eta) const {
    VecX phi_xi = basis_1d_.evaluate(xi);
    VecX phi_eta = basis_1d_.evaluate(eta);

    VecX phi(ndof_);
    for (int j = 0; j < n1d_; ++j) {
        for (int i = 0; i < n1d_; ++i) {
            int dof = dof_index(i, j);
            phi(dof) = phi_xi(i) * phi_eta(j);
        }
    }

    return phi;
}

MatX LagrangeBasis2D::evaluate_gradient(Real xi, Real eta) const {
    VecX phi_xi = basis_1d_.evaluate(xi);
    VecX phi_eta = basis_1d_.evaluate(eta);
    VecX dphi_xi = basis_1d_.evaluate_derivative(xi);
    VecX dphi_eta = basis_1d_.evaluate_derivative(eta);

    MatX grad(ndof_, 2);
    for (int j = 0; j < n1d_; ++j) {
        for (int i = 0; i < n1d_; ++i) {
            int dof = dof_index(i, j);
            grad(dof, 0) = dphi_xi(i) * phi_eta(j);   // d/dxi
            grad(dof, 1) = phi_xi(i) * dphi_eta(j);   // d/deta
        }
    }

    return grad;
}

VecX LagrangeBasis2D::evaluate_second_derivative_1d(Real xi) const {
    VecX d2phi(n1d_);

    // Check if at a node
    bool at_node = false;
    int node_idx = -1;
    for (int j = 0; j < n1d_; ++j) {
        if (std::abs(xi - basis_1d_.nodes(j)) < 1e-14) {
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
        // l''_j(x) = l_j(x) * [(sum_{k!=j} 1/(x-x_k))^2 - sum_{k!=j} 1/(x-x_k)^2]
        VecX phi = basis_1d_.evaluate(xi);

        for (int j = 0; j < n1d_; ++j) {
            Real sum1 = 0.0;  // sum_{k!=j} 1/(x - x_k)
            Real sum2 = 0.0;  // sum_{k!=j} 1/(x - x_k)^2

            for (int k = 0; k < n1d_; ++k) {
                if (k != j) {
                    Real diff_inv = 1.0 / (xi - basis_1d_.nodes(k));
                    sum1 += diff_inv;
                    sum2 += diff_inv * diff_inv;
                }
            }

            d2phi(j) = phi(j) * (sum1 * sum1 - sum2);
        }
    }

    return d2phi;
}

void LagrangeBasis2D::evaluate_second_derivatives(Real xi, Real eta,
                                                   VecX& d2_dxi2, VecX& d2_deta2,
                                                   VecX& d2_dxideta) const {
    VecX phi_xi = basis_1d_.evaluate(xi);
    VecX phi_eta = basis_1d_.evaluate(eta);
    VecX dphi_xi = basis_1d_.evaluate_derivative(xi);
    VecX dphi_eta = basis_1d_.evaluate_derivative(eta);
    VecX d2phi_xi = evaluate_second_derivative_1d(xi);
    VecX d2phi_eta = evaluate_second_derivative_1d(eta);

    d2_dxi2.resize(ndof_);
    d2_deta2.resize(ndof_);
    d2_dxideta.resize(ndof_);

    for (int j = 0; j < n1d_; ++j) {
        for (int i = 0; i < n1d_; ++i) {
            int dof = dof_index(i, j);
            d2_dxi2(dof) = d2phi_xi(i) * phi_eta(j);
            d2_deta2(dof) = phi_xi(i) * d2phi_eta(j);
            d2_dxideta(dof) = dphi_xi(i) * dphi_eta(j);
        }
    }
}

VecX LagrangeBasis2D::evaluate_laplacian(Real xi, Real eta) const {
    VecX d2_dxi2, d2_deta2, d2_dxideta;
    evaluate_second_derivatives(xi, eta, d2_dxi2, d2_deta2, d2_dxideta);

    // Laplacian in reference coordinates = d^2/dxi^2 + d^2/deta^2
    return d2_dxi2 + d2_deta2;
}

std::vector<Mat2> LagrangeBasis2D::evaluate_hessian(Real xi, Real eta) const {
    VecX d2_dxi2, d2_deta2, d2_dxideta;
    evaluate_second_derivatives(xi, eta, d2_dxi2, d2_deta2, d2_dxideta);

    std::vector<Mat2> hessians(ndof_);
    for (int dof = 0; dof < ndof_; ++dof) {
        hessians[dof](0, 0) = d2_dxi2(dof);
        hessians[dof](0, 1) = d2_dxideta(dof);
        hessians[dof](1, 0) = d2_dxideta(dof);
        hessians[dof](1, 1) = d2_deta2(dof);
    }

    return hessians;
}

std::vector<int> LagrangeBasis2D::edge_dofs(int edge_id) const {
    std::vector<int> dofs(n1d_);

    switch (edge_id) {
        case 0:  // xi = -1 (left edge)
            for (int j = 0; j < n1d_; ++j) {
                dofs[j] = dof_index(0, j);
            }
            break;
        case 1:  // xi = +1 (right edge)
            for (int j = 0; j < n1d_; ++j) {
                dofs[j] = dof_index(n1d_ - 1, j);
            }
            break;
        case 2:  // eta = -1 (bottom edge)
            for (int i = 0; i < n1d_; ++i) {
                dofs[i] = dof_index(i, 0);
            }
            break;
        case 3:  // eta = +1 (top edge)
            for (int i = 0; i < n1d_; ++i) {
                dofs[i] = dof_index(i, n1d_ - 1);
            }
            break;
        default:
            throw std::out_of_range("edge_id must be 0-3");
    }

    return dofs;
}

int LagrangeBasis2D::corner_dof(int corner_id) const {
    switch (corner_id) {
        case 0: return dof_index(0, 0);             // (-1, -1)
        case 1: return dof_index(n1d_ - 1, 0);      // (+1, -1)
        case 2: return dof_index(0, n1d_ - 1);      // (-1, +1)
        case 3: return dof_index(n1d_ - 1, n1d_ - 1); // (+1, +1)
        default:
            throw std::out_of_range("corner_id must be 0-3");
    }
}

bool LagrangeBasis2D::is_boundary_dof(int dof) const {
    int i, j;
    dof_ij(dof, i, j);
    return (i == 0 || i == n1d_ - 1 || j == 0 || j == n1d_ - 1);
}

std::vector<int> LagrangeBasis2D::interior_dofs() const {
    std::vector<int> interior;
    interior.reserve((n1d_ - 2) * (n1d_ - 2));

    for (int j = 1; j < n1d_ - 1; ++j) {
        for (int i = 1; i < n1d_ - 1; ++i) {
            interior.push_back(dof_index(i, j));
        }
    }

    return interior;
}

VecX LagrangeBasis2D::evaluate_normal_derivative_at_edge(int edge_id, Real t) const {
    // Evaluate gradient at the edge point, then extract normal component
    // Edge parameterization:
    //   edge 0 (xi=-1): point = (-1, t), outward normal = (-1, 0)
    //   edge 1 (xi=+1): point = (+1, t), outward normal = (+1, 0)
    //   edge 2 (eta=-1): point = (t, -1), outward normal = (0, -1)
    //   edge 3 (eta=+1): point = (t, +1), outward normal = (0, +1)

    Real xi, eta;
    int normal_dir;  // 0 for x-normal, 1 for y-normal
    Real normal_sign;

    switch (edge_id) {
        case 0:  // left edge
            xi = -1.0;
            eta = t;
            normal_dir = 0;
            normal_sign = -1.0;
            break;
        case 1:  // right edge
            xi = 1.0;
            eta = t;
            normal_dir = 0;
            normal_sign = 1.0;
            break;
        case 2:  // bottom edge
            xi = t;
            eta = -1.0;
            normal_dir = 1;
            normal_sign = -1.0;
            break;
        case 3:  // top edge
            xi = t;
            eta = 1.0;
            normal_dir = 1;
            normal_sign = 1.0;
            break;
        default:
            throw std::out_of_range("edge_id must be 0-3");
    }

    MatX grad = evaluate_gradient(xi, eta);

    // Extract normal derivative: grad dot n
    VecX dn(ndof_);
    for (int dof = 0; dof < ndof_; ++dof) {
        dn(dof) = normal_sign * grad(dof, normal_dir);
    }

    return dn;
}

}  // namespace drifter
