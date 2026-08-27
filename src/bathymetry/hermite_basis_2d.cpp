#include "bathymetry/hermite_basis_2d.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

namespace {

/// Binomial coefficient C(n, k) for the small n used here (n <= 5)
Real binomial(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    Real result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * static_cast<Real>(n - i) / static_cast<Real>(i + 1);
    }
    return result;
}

Real factorial(int n) {
    Real result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= static_cast<Real>(i);
    }
    return result;
}

} // namespace

HermiteBasis2D::HermiteBasis2D(int r) : r_(r), n1d_(2 * (r + 1)), basis_1d_(r) {}

void HermiteBasis2D::check_dof(int dof) const {
    if (dof < 0 || dof >= num_dofs()) {
        throw std::invalid_argument("HermiteBasis2D: DOF index out of range: " +
                                    std::to_string(dof));
    }
}

// =============================================================================
// DOF indexing
// =============================================================================

int HermiteBasis2D::dof_index(int sx, int a, int sy, int b) const {
    const int i = basis_1d_.dof_index(sx, a);
    const int j = basis_1d_.dof_index(sy, b);
    return i + n1d_ * j;
}

std::pair<int, int> HermiteBasis2D::deriv_order(int dof) const {
    check_dof(dof);
    return {basis_1d_.deriv_of(index_u(dof)), basis_1d_.deriv_of(index_v(dof))};
}

int HermiteBasis2D::total_deriv_order(int dof) const {
    const auto [a, b] = deriv_order(dof);
    return a + b;
}

Vec2 HermiteBasis2D::control_point_position(int dof) const {
    check_dof(dof);
    return Vec2(static_cast<Real>(basis_1d_.node_of(index_u(dof))),
                static_cast<Real>(basis_1d_.node_of(index_v(dof))));
}

int HermiteBasis2D::corner_dof(int corner_id) const {
    if (corner_id < 0 || corner_id > 3) {
        throw std::invalid_argument("HermiteBasis2D::corner_dof: corner_id must be in [0, 3]");
    }
    // Corner IDs: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    const int sx = corner_id % 2;
    const int sy = corner_id / 2;
    return dof_index(sx, 0, sy, 0);
}

int HermiteBasis2D::dof_to_corner(int dof) const {
    check_dof(dof);
    const int sx = basis_1d_.node_of(index_u(dof));
    const int sy = basis_1d_.node_of(index_v(dof));
    return sx + 2 * sy;
}

Vec2 HermiteBasis2D::corner_param(int corner_id) const {
    if (corner_id < 0 || corner_id > 3) {
        throw std::invalid_argument("HermiteBasis2D::corner_param: corner_id must be in [0, 3]");
    }
    return Vec2(static_cast<Real>(corner_id % 2), static_cast<Real>(corner_id / 2));
}

std::vector<int> HermiteBasis2D::edge_dofs(int edge_id) const {
    if (edge_id < 0 || edge_id > 3) {
        throw std::invalid_argument("HermiteBasis2D::edge_dofs: edge_id must be in [0, 3]");
    }

    std::vector<int> dofs;
    dofs.reserve(static_cast<size_t>(2 * (r_ + 1) * (r_ + 1)));

    for (int dof = 0; dof < num_dofs(); ++dof) {
        const int sx = basis_1d_.node_of(index_u(dof));
        const int sy = basis_1d_.node_of(index_v(dof));

        bool on_edge = false;
        switch (edge_id) {
        case 0:
            on_edge = (sx == 0);
            break; // u = 0 (left)
        case 1:
            on_edge = (sx == 1);
            break; // u = 1 (right)
        case 2:
            on_edge = (sy == 0);
            break; // v = 0 (bottom)
        default:
            on_edge = (sy == 1);
            break; // v = 1 (top)
        }

        if (on_edge) {
            dofs.push_back(dof);
        }
    }
    return dofs;
}

// =============================================================================
// Evaluation (parametric basis Nhat)
// =============================================================================

VecX HermiteBasis2D::evaluate(Real u, Real v) const {
    VecX hu(n1d_), hv(n1d_);
    for (int i = 0; i < n1d_; ++i) {
        hu(i) = basis_1d_.eval(i, 0, u);
        hv(i) = basis_1d_.eval(i, 0, v);
    }

    VecX N(num_dofs());
    for (int j = 0; j < n1d_; ++j) {
        for (int i = 0; i < n1d_; ++i) {
            N(i + n1d_ * j) = hu(i) * hv(j);
        }
    }
    return N;
}

VecX HermiteBasis2D::evaluate_du(Real u, Real v) const {
    VecX dhu(n1d_), hv(n1d_);
    for (int i = 0; i < n1d_; ++i) {
        dhu(i) = basis_1d_.eval(i, 1, u);
        hv(i) = basis_1d_.eval(i, 0, v);
    }

    VecX dN(num_dofs());
    for (int j = 0; j < n1d_; ++j) {
        for (int i = 0; i < n1d_; ++i) {
            dN(i + n1d_ * j) = dhu(i) * hv(j);
        }
    }
    return dN;
}

VecX HermiteBasis2D::evaluate_dv(Real u, Real v) const {
    VecX hu(n1d_), dhv(n1d_);
    for (int i = 0; i < n1d_; ++i) {
        hu(i) = basis_1d_.eval(i, 0, u);
        dhv(i) = basis_1d_.eval(i, 1, v);
    }

    VecX dN(num_dofs());
    for (int j = 0; j < n1d_; ++j) {
        for (int i = 0; i < n1d_; ++i) {
            dN(i + n1d_ * j) = hu(i) * dhv(j);
        }
    }
    return dN;
}

MatX HermiteBasis2D::evaluate_gradient(Real u, Real v) const {
    MatX grad(num_dofs(), 2);
    grad.col(0) = evaluate_du(u, v);
    grad.col(1) = evaluate_dv(u, v);
    return grad;
}

Real HermiteBasis2D::evaluate_scalar(const VecX &coeffs, Real u, Real v) const {
    if (coeffs.size() != num_dofs()) {
        throw std::invalid_argument("HermiteBasis2D::evaluate_scalar: expected " +
                                    std::to_string(num_dofs()) + " coefficients, got " +
                                    std::to_string(coeffs.size()));
    }
    return coeffs.dot(evaluate(u, v));
}

// =============================================================================
// Physical scaling and change of basis
// =============================================================================

VecX HermiteBasis2D::dof_scaling(Real dx, Real dy) const {
    VecX scal(num_dofs());
    for (int dof = 0; dof < num_dofs(); ++dof) {
        const auto [a, b] = deriv_order(dof);
        scal(dof) = std::pow(dx, a) * std::pow(dy, b);
    }
    return scal;
}

MatX HermiteBasis2D::bernstein_change_of_basis_1d(Real h) const {
    const int p = degree();
    MatX M = MatX::Zero(p + 1, p + 1);

    // c_k        = sum_{m=0}^{k} [C(k,m)/C(p,m)] * ( h)^m/m! * z_0^(m)
    // c_{p-k}    = sum_{m=0}^{k} [C(k,m)/C(p,m)] * (-h)^m/m! * z_1^(m)
    // with q ordered node-major: (z_0, z_0', ..., z_1, z_1', ...)
    for (int k = 0; k <= r_; ++k) {
        for (int m = 0; m <= k; ++m) {
            const Real base = binomial(k, m) / binomial(p, m) / factorial(m);
            M(k, basis_1d_.dof_index(0, m)) = base * std::pow(h, m);
            M(p - k, basis_1d_.dof_index(1, m)) = base * std::pow(-h, m);
        }
    }
    return M;
}

MatX HermiteBasis2D::bernstein_change_of_basis(Real dx, Real dy) const {
    const MatX Mx = bernstein_change_of_basis_1d(dx);
    const MatX My = bernstein_change_of_basis_1d(dy);

    // With the u index varying fastest, the tensor product is M(h_y) kron M(h_x)
    MatX Me(num_dofs(), num_dofs());
    for (int jr = 0; jr < n1d_; ++jr) {
        for (int jc = 0; jc < n1d_; ++jc) {
            Me.block(jr * n1d_, jc * n1d_, n1d_, n1d_) = My(jr, jc) * Mx;
        }
    }
    return Me;
}

MatX HermiteBasis2D::midpoint_matrix(Real h_t) const {
    if (h_t <= 0.0) {
        throw std::invalid_argument("HermiteBasis2D::midpoint_matrix: h_t must be positive");
    }

    const int p = degree();
    MatX G(r_ + 1, p + 1);

    for (int i = 0; i <= r_; ++i) {
        for (int j = 0; j <= p; ++j) {
            const int mj = basis_1d_.deriv_of(j);
            G(i, j) = std::pow(h_t, mj - i) * basis_1d_.eval(j, i, 0.5);
        }
    }
    return G;
}

} // namespace drifter
