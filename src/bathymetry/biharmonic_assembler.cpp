#include "bathymetry/biharmonic_assembler.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

BiharmonicAssembler::BiharmonicAssembler(const QuadtreeAdapter& mesh,
                                         const QuinticBasis2D& basis,
                                         const CGDofManager& dofs,
                                         Real alpha,
                                         Real beta)
    : mesh_(mesh), basis_(basis), dofs_(dofs), alpha_(alpha), beta_(beta) {

    if (alpha < 0.0 || beta < 0.0) {
        throw std::invalid_argument("BiharmonicAssembler: weights must be non-negative");
    }

    init_quadrature();
}

void BiharmonicAssembler::init_quadrature() {
    // Use Gauss-Legendre quadrature with enough points for exactness
    // For quintic basis (order 5), the Laplacian squared term needs 2*(5-2) = 6 points
    // We use the same number as the basis for simplicity (6 points)
    num_gauss_1d_ = QuinticBasis2D::N1D;

    // Use the LGL nodes and weights from the basis
    gauss_nodes_ = basis_.nodes_1d();
    gauss_weights_ = basis_.weights_1d();

    // Precompute basis values at all 2D Gauss points
    int n_gauss_2d = num_gauss_1d_ * num_gauss_1d_;
    phi_at_gauss_.resize(n_gauss_2d, QuinticBasis2D::NDOF);
    lap_at_gauss_.resize(n_gauss_2d, QuinticBasis2D::NDOF);

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
    // dx/dxi = (xmax - xmin) / 2, dy/deta = (ymax - ymin) / 2
    Real dx_dxi = 0.5 * (bounds.xmax - bounds.xmin);
    Real dy_deta = 0.5 * (bounds.ymax - bounds.ymin);
    Real jac = dx_dxi * dy_deta;  // Jacobian determinant

    int n_gauss_2d = num_gauss_1d_ * num_gauss_1d_;
    return VecX::Constant(n_gauss_2d, jac);
}

MatX BiharmonicAssembler::element_biharmonic(Index elem) const {
    // Biharmonic stiffness: K_ij = α * ∫ ∇²φ_i * ∇²φ_j dA
    //
    // In reference coordinates:
    // ∇² in physical = (1/h_x²)∂²/∂ξ² + (1/h_y²)∂²/∂η²
    // where h_x = (xmax-xmin)/2, h_y = (ymax-ymin)/2
    //
    // The Laplacian in reference coordinates needs scaling by 1/h²

    const auto& bounds = mesh_.element_bounds(elem);
    Real hx = 0.5 * (bounds.xmax - bounds.xmin);
    Real hy = 0.5 * (bounds.ymax - bounds.ymin);

    // Scale factors for Laplacian: ∇²_phys = (1/hx²)∂²ξ + (1/hy²)∂²η
    // Our basis_.evaluate_laplacian returns reference Laplacian
    // Need to scale: lap_phys = lap_ref / h² for isotropic case
    // For anisotropic: more complex (handled below)

    int n_gauss_2d = num_gauss_1d_ * num_gauss_1d_;
    MatX K = MatX::Zero(QuinticBasis2D::NDOF, QuinticBasis2D::NDOF);

    // Jacobian for integration
    Real jac = hx * hy;

    for (int jg = 0; jg < num_gauss_1d_; ++jg) {
        for (int ig = 0; ig < num_gauss_1d_; ++ig) {
            int q = ig + jg * num_gauss_1d_;
            Real wq = gauss_weights_(ig) * gauss_weights_(jg) * jac;

            // Get reference Laplacian values
            VecX lap_ref = lap_at_gauss_.row(q).transpose();

            // Compute physical Laplacian
            // For tensor-product basis on rectangle:
            // d²φ/dx² = (1/hx²) d²φ/dξ²
            // d²φ/dy² = (1/hy²) d²φ/dη²
            // We need to compute these separately

            Real xi = gauss_nodes_(ig);
            Real eta = gauss_nodes_(jg);

            VecX d2_dxi2, d2_deta2, d2_dxideta;
            basis_.evaluate_second_derivatives(xi, eta, d2_dxi2, d2_deta2, d2_dxideta);

            // Physical Laplacian
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

    int n_gauss_2d = num_gauss_1d_ * num_gauss_1d_;
    MatX M = MatX::Zero(QuinticBasis2D::NDOF, QuinticBasis2D::NDOF);

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

    VecX f = VecX::Zero(QuinticBasis2D::NDOF);

    for (int jg = 0; jg < num_gauss_1d_; ++jg) {
        for (int ig = 0; ig < num_gauss_1d_; ++ig) {
            int q = ig + jg * num_gauss_1d_;
            Real wq = gauss_weights_(ig) * gauss_weights_(jg) * jac;

            // Get physical coordinates
            Real xi = gauss_nodes_(ig);
            Real eta = gauss_nodes_(jg);
            Vec2 phys = map_to_physical(elem, xi, eta);

            // Evaluate bathymetry at physical point
            Real u_data = bathy.evaluate(phys(0), phys(1));

            // Get basis functions
            VecX phi = phi_at_gauss_.row(q).transpose();

            // Add contribution: β * u_data * phi_i * w_q
            f += beta_ * u_data * wq * phi;
        }
    }

    return f;
}

SpMat BiharmonicAssembler::assemble_stiffness() const {
    Index n = dofs_.num_global_dofs();
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(mesh_.num_elements() * QuinticBasis2D::NDOF * QuinticBasis2D::NDOF);

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        MatX K_elem = element_stiffness(e);
        const auto& elem_dofs = dofs_.element_dofs(e);

        // Scatter to global matrix
        for (int i = 0; i < QuinticBasis2D::NDOF; ++i) {
            Index gi = elem_dofs[i];
            for (int j = 0; j < QuinticBasis2D::NDOF; ++j) {
                Index gj = elem_dofs[j];
                if (std::abs(K_elem(i, j)) > 1e-15) {
                    triplets.emplace_back(gi, gj, K_elem(i, j));
                }
            }
        }
    }

    SpMat K(n, n);
    K.setFromTriplets(triplets.begin(), triplets.end());
    return K;
}

VecX BiharmonicAssembler::assemble_rhs(const BathymetrySource& bathy) const {
    Index n = dofs_.num_global_dofs();
    VecX f = VecX::Zero(n);

    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        VecX f_elem = element_rhs(e, bathy);
        const auto& elem_dofs = dofs_.element_dofs(e);

        // Scatter to global vector
        for (int i = 0; i < QuinticBasis2D::NDOF; ++i) {
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
    f_red = dofs_.transform_rhs(f);
}

}  // namespace drifter
