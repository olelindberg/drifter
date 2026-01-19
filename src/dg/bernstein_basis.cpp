#include "dg/bernstein_basis.hpp"
#include "dg/basis_hexahedron.hpp"
#include "bathymetry/quintic_basis_2d.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

// =============================================================================
// BernsteinBasis1D implementation
// =============================================================================

BernsteinBasis1D::BernsteinBasis1D(int order)
    : order_(order)
{
    if (order < 0) {
        throw std::invalid_argument("BernsteinBasis1D: order must be non-negative");
    }
    build_conversion_matrices();
}

void BernsteinBasis1D::build_conversion_matrices() {
    int n = order_;
    int np = n + 1;

    // Get LGL nodes for conversion
    VecX lgl_nodes(np), lgl_weights(np);
    compute_gauss_lobatto_nodes(np, lgl_nodes, lgl_weights);

    // Build Lagrange-to-Bernstein conversion matrix
    // We need to express each Lagrange basis function in terms of Bernstein basis
    //
    // The Bernstein coefficients b_i of a polynomial p(t) = sum_j c_j L_j(t) are:
    // b_i = sum_j c_j * L_j(t_i^B)
    // where t_i^B are the Bernstein control points (uniformly spaced on [0,1])
    //
    // Actually, for a proper conversion, we use the fact that:
    // p(t) = sum_j c_j L_j(t) = sum_i b_i B_i(t)
    //
    // The conversion matrix M satisfies: b = M * c
    // where M_ij gives the i-th Bernstein coefficient for the j-th Lagrange basis

    // Method: Evaluate each Lagrange basis at Bernstein control points,
    // then solve for Bernstein coefficients

    // Bernstein control points in reference coordinates [-1, 1]
    VecX bernstein_nodes(np);
    for (int i = 0; i < np; ++i) {
        // Uniform spacing on [0, 1], mapped to [-1, 1]
        Real t = static_cast<Real>(i) / static_cast<Real>(n);
        bernstein_nodes(i) = 2.0 * t - 1.0;  // Map to [-1, 1]
    }

    // Build Vandermonde-like matrices
    // V_B(i,j) = B_j(node_i) where node_i are evaluation points
    // V_L(i,j) = L_j(node_i) where L_j is j-th Lagrange basis at LGL nodes

    // For conversion, we use the property:
    // If p(x) has Lagrange coefficients c and Bernstein coefficients b, then
    // The Bernstein basis evaluated at any point gives: p(x) = sum_i b_i B_i(x)
    // The Lagrange basis evaluated at any point gives: p(x) = sum_j c_j L_j(x)

    // Build transformation matrix using collocation at Bernstein nodes
    // p(bernstein_node_k) = sum_j c_j L_j(bernstein_node_k)
    // This equals b_k only for Bezier curves with Bernstein nodes as control points
    // evaluated at the corresponding parameter

    // More rigorous approach: Use the elevation/degree reduction formula
    // For Lagrange at LGL nodes to Bernstein, we use:
    // 1. Build matrix of Lagrange basis values at many points
    // 2. Build matrix of Bernstein basis values at same points
    // 3. Find least squares fit (or exact if points = degree + 1)

    // Actually, the cleanest way is:
    // Bernstein control points b_i are found by evaluating the polynomial at
    // special points using de Casteljau construction, but that's complex.

    // Simpler approach: Use the fact that both bases span the same polynomial space
    // So we can convert via a common representation (e.g., power basis)

    // L2B matrix: Bernstein coefficients from Lagrange coefficients
    // b = L2B * c

    // Build by expressing each Lagrange basis function in Bernstein form
    L2B_.resize(np, np);

    // For each Lagrange basis function L_j (which is 1 at lgl_nodes[j], 0 elsewhere)
    // Find its Bernstein coefficients

    // The j-th column of L2B gives the Bernstein coefficients of L_j
    // L_j(t) = sum_i L2B(i,j) * B_i(t)

    // Using collocation: Evaluate at Bernstein control points
    // Let V_B be the Bernstein Vandermonde: V_B(k,i) = B_i(u_k) where u_k = k/n
    // Let V_L be the Lagrange Vandermonde: V_L(k,j) = L_j(u_k)
    // Then: V_L = V_B * L2B, so L2B = V_B^{-1} * V_L

    // Build V_B: Bernstein basis at uniform parameter values
    MatX V_B(np, np);
    for (int k = 0; k < np; ++k) {
        Real t = static_cast<Real>(k) / static_cast<Real>(n);  // Parameter in [0,1]
        Real xi = 2.0 * t - 1.0;  // Reference coord in [-1,1]
        VecX B = evaluate(xi);
        for (int i = 0; i < np; ++i) {
            V_B(k, i) = B(i);
        }
    }

    // Build V_L: Lagrange basis at same parameter values
    MatX V_L(np, np);
    for (int k = 0; k < np; ++k) {
        Real t = static_cast<Real>(k) / static_cast<Real>(n);
        Real xi = 2.0 * t - 1.0;

        // Evaluate Lagrange basis at xi
        for (int j = 0; j < np; ++j) {
            Real prod = 1.0;
            for (int m = 0; m < np; ++m) {
                if (m != j) {
                    prod *= (xi - lgl_nodes(m)) / (lgl_nodes(j) - lgl_nodes(m));
                }
            }
            V_L(k, j) = prod;
        }
    }

    // L2B = V_B^{-1} * V_L
    L2B_ = V_B.inverse() * V_L;

    // B2L = L2B^{-1}
    B2L_ = L2B_.inverse();
}

// =============================================================================
// BernsteinBasis3D implementation
// =============================================================================

BernsteinBasis3D::BernsteinBasis3D(int order)
    : order_(order)
    , num_dofs_((order + 1) * (order + 1) * (order + 1))
    , basis_1d_(order)
{
    build_3d_conversion_matrix();
}

void BernsteinBasis3D::build_3d_conversion_matrix() {
    // The 3D conversion matrix is the Kronecker product of 1D matrices
    // L2B_3d = L2B_1d ⊗ L2B_1d ⊗ L2B_1d
    //
    // For tensor-product indexing i + n*(j + n*k), this becomes a direct
    // tensor product computation

    int np = order_ + 1;
    const MatX& L2B_1d = basis_1d_.lagrange_to_bernstein_matrix();

    L2B_3d_.resize(num_dofs_, num_dofs_);

    // Build tensor product explicitly
    for (int k2 = 0; k2 < np; ++k2) {
        for (int j2 = 0; j2 < np; ++j2) {
            for (int i2 = 0; i2 < np; ++i2) {
                int row = i2 + np * (j2 + np * k2);  // Bernstein index

                for (int k1 = 0; k1 < np; ++k1) {
                    for (int j1 = 0; j1 < np; ++j1) {
                        for (int i1 = 0; i1 < np; ++i1) {
                            int col = i1 + np * (j1 + np * k1);  // Lagrange index

                            // Tensor product of 1D conversion matrices
                            L2B_3d_(row, col) = L2B_1d(i2, i1) * L2B_1d(j2, j1) * L2B_1d(k2, k1);
                        }
                    }
                }
            }
        }
    }
}

VecX BernsteinBasis3D::evaluate(const Vec3& xi) const {
    VecX phi_xi = basis_1d_.evaluate(xi(0));
    VecX phi_eta = basis_1d_.evaluate(xi(1));
    VecX phi_zeta = basis_1d_.evaluate(xi(2));

    int np = order_ + 1;
    VecX phi(num_dofs_);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = i + np * (j + np * k);
                phi(idx) = phi_xi(i) * phi_eta(j) * phi_zeta(k);
            }
        }
    }

    return phi;
}

VecX BernsteinBasis3D::evaluate_bottom_face(Real xi, Real eta) const {
    // Evaluate at zeta = -1 (bottom face)
    VecX phi_xi = basis_1d_.evaluate(xi);
    VecX phi_eta = basis_1d_.evaluate(eta);
    VecX phi_zeta = basis_1d_.evaluate(-1.0);  // zeta = -1

    int np = order_ + 1;
    VecX phi(num_dofs_);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = i + np * (j + np * k);
                phi(idx) = phi_xi(i) * phi_eta(j) * phi_zeta(k);
            }
        }
    }

    return phi;
}

// =============================================================================
// SeabedInterpolator implementation
// =============================================================================

SeabedInterpolator::SeabedInterpolator(int order, SeabedInterpolation method)
    : order_(order)
    , method_(method)
{
    int np = order + 1;

    // Always need LGL nodes for Lagrange evaluation
    lgl_nodes_.resize(np);
    lgl_weights_.resize(np);
    compute_gauss_lobatto_nodes(np, lgl_nodes_, lgl_weights_);

    // Build Bernstein basis if using that method
    if (method == SeabedInterpolation::Bernstein) {
        bernstein_basis_ = std::make_unique<BernsteinBasis3D>(order);
        L2B_3d_ = bernstein_basis_->lagrange_to_bernstein_matrix();
    }

    // Build Quintic basis if using that method
    if (method == SeabedInterpolation::Quintic) {
        // Quintic is always order 5, override the passed order for interpolation
        quintic_basis_ = std::make_unique<QuinticBasis2D>();
    }
}

VecX SeabedInterpolator::evaluate_lagrange_1d(Real xi) const {
    int np = order_ + 1;
    VecX phi(np);

    for (int i = 0; i < np; ++i) {
        Real prod = 1.0;
        for (int j = 0; j < np; ++j) {
            if (j != i) {
                prod *= (xi - lgl_nodes_(j)) / (lgl_nodes_(i) - lgl_nodes_(j));
            }
        }
        phi(i) = prod;
    }

    return phi;
}

VecX SeabedInterpolator::evaluate_lagrange_bottom_face(Real xi, Real eta) const {
    // Evaluate 3D Lagrange basis at (xi, eta, zeta=-1)
    VecX phi_xi = evaluate_lagrange_1d(xi);
    VecX phi_eta = evaluate_lagrange_1d(eta);
    VecX phi_zeta = evaluate_lagrange_1d(-1.0);  // zeta = -1

    int np = order_ + 1;
    int ndof = np * np * np;
    VecX phi(ndof);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = i + np * (j + np * k);
                phi(idx) = phi_xi(i) * phi_eta(j) * phi_zeta(k);
            }
        }
    }

    return phi;
}

Vec3 SeabedInterpolator::evaluate_point(const VecX& coords, Real xi, Real eta) const {
    // coords contains interleaved [x0,y0,z0, x1,y1,z1, ...] for each DOF

    Vec3 point(0, 0, 0);

    if (method_ == SeabedInterpolation::Quintic) {
        // Quintic Lagrange interpolation in 2D (36 DOFs)
        // coords for quintic should be 3*36 = 108 values
        VecX phi = quintic_basis_->evaluate(xi, eta);
        int ndof = QuinticBasis2D::NDOF;
        for (int dof = 0; dof < ndof; ++dof) {
            point(0) += phi(dof) * coords(3 * dof + 0);
            point(1) += phi(dof) * coords(3 * dof + 1);
            point(2) += phi(dof) * coords(3 * dof + 2);
        }
    } else if (method_ == SeabedInterpolation::Lagrange) {
        // Standard Lagrange interpolation
        VecX phi = evaluate_lagrange_bottom_face(xi, eta);

        int ndof = (order_ + 1) * (order_ + 1) * (order_ + 1);
        for (int dof = 0; dof < ndof; ++dof) {
            point(0) += phi(dof) * coords(3 * dof + 0);
            point(1) += phi(dof) * coords(3 * dof + 1);
            point(2) += phi(dof) * coords(3 * dof + 2);
        }
    } else {
        // Bernstein interpolation (bounded)
        // Treat the input data DIRECTLY as Bernstein control points.
        // This gives a DIFFERENT curve than Lagrange, but one that is
        // guaranteed to stay within the convex hull of the control points.
        //
        // Key property: The interpolated values are bounded by min/max of input data.
        // This prevents spurious overshoots that can make seabed appear above water.
        int ndof = (order_ + 1) * (order_ + 1) * (order_ + 1);

        // Evaluate Bernstein basis directly with input data as control points
        VecX phi = bernstein_basis_->evaluate_bottom_face(xi, eta);

        for (int dof = 0; dof < ndof; ++dof) {
            point(0) += phi(dof) * coords(3 * dof + 0);
            point(1) += phi(dof) * coords(3 * dof + 1);
            point(2) += phi(dof) * coords(3 * dof + 2);
        }
    }

    return point;
}

Real SeabedInterpolator::evaluate_scalar(const VecX& data, Real xi, Real eta) const {
    Real value = 0.0;

    if (method_ == SeabedInterpolation::Quintic) {
        // Quintic Lagrange interpolation in 2D (36 DOFs)
        VecX phi = quintic_basis_->evaluate(xi, eta);
        value = phi.dot(data);
    } else if (method_ == SeabedInterpolation::Lagrange) {
        // Standard Lagrange interpolation
        VecX phi = evaluate_lagrange_bottom_face(xi, eta);

        int ndof = (order_ + 1) * (order_ + 1) * (order_ + 1);
        for (int dof = 0; dof < ndof; ++dof) {
            value += phi(dof) * data(dof);
        }
    } else {
        // Bernstein interpolation (bounded)
        // Treat input data DIRECTLY as Bernstein control points.
        // This gives bounded output: min(data) <= value <= max(data)
        VecX phi = bernstein_basis_->evaluate_bottom_face(xi, eta);

        int ndof = (order_ + 1) * (order_ + 1) * (order_ + 1);
        for (int dof = 0; dof < ndof; ++dof) {
            value += phi(dof) * data(dof);
        }
    }

    return value;
}

VecX SeabedInterpolator::to_bernstein(const VecX& lagrange_data) const {
    if (method_ == SeabedInterpolation::Lagrange) {
        // No conversion needed
        return lagrange_data;
    }
    return L2B_3d_ * lagrange_data;
}

Real SeabedInterpolator::evaluate_scalar_2d(const VecX& data_2d, Real xi, Real eta) const {
    // Evaluate 2D scalar field (for bottom face only, no z-dependence)
    // data_2d has n1d * n1d values indexed as i + n1d * j
    int np = order_ + 1;

    Real value = 0.0;

    if (method_ == SeabedInterpolation::Quintic) {
        // Quintic Lagrange interpolation (order 5, 36 DOFs)
        VecX phi = quintic_basis_->evaluate(xi, eta);
        value = phi.dot(data_2d);
    } else if (method_ == SeabedInterpolation::Lagrange) {
        // Lagrange interpolation in 2D
        VecX phi_xi = evaluate_lagrange_1d(xi);
        VecX phi_eta = evaluate_lagrange_1d(eta);

        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = i + np * j;
                value += phi_xi(i) * phi_eta(j) * data_2d(idx);
            }
        }
    } else {
        // Bernstein interpolation in 2D
        VecX phi_xi = bernstein_basis_->basis_1d().evaluate(xi);
        VecX phi_eta = bernstein_basis_->basis_1d().evaluate(eta);

        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = i + np * j;
                value += phi_xi(i) * phi_eta(j) * data_2d(idx);
            }
        }
    }

    return value;
}

}  // namespace drifter
