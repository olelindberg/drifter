#include "bathymetry/adaptive_bathymetry.hpp"
#include "mesh/seabed_surface.hpp"
#include <cmath>

namespace drifter {

// =============================================================================
// AdaptiveBathymetry - Construction
// =============================================================================

AdaptiveBathymetry::AdaptiveBathymetry(std::shared_ptr<BathymetryData> raw_data)
    : raw_data_(std::move(raw_data)) {

    sampler_ = std::make_unique<WENO5Sampler>(raw_data_);
}

// =============================================================================
// AdaptiveBathymetry - Projection
// =============================================================================

VecX AdaptiveBathymetry::project_element(const ElementBounds& bounds, int order) const {
    // Get quadrature and projection matrix
    const auto& quad = get_quadrature(order);
    const auto& P = get_projection_matrix(order);

    // Sample bathymetry at quadrature points using WENO5
    VecX f_quad = sample_at_quadrature(bounds, quad);

    // Apply projection: coeffs = P * f_quad
    VecX coeffs = P * f_quad;

    return coeffs;
}

void AdaptiveBathymetry::project_to_seabed(const OctreeAdapter& mesh, int order,
                                          SeabedSurface& seabed) const {
    const auto& elements = mesh.elements();

    for (size_t s = 0; s < seabed.num_elements(); ++s) {
        Index mesh_idx = seabed.mesh_element_index(s);
        const auto& bounds = elements[mesh_idx]->bounds;

        VecX coeffs = project_element(bounds, order);
        seabed.set_element_coefficients(s, coeffs);
    }

    // Apply non-conforming projection to ensure interface continuity
    seabed.apply_nonconforming_projection();
}

// =============================================================================
// AdaptiveBathymetry - Error Estimation
// =============================================================================

Real AdaptiveBathymetry::estimate_projection_error(const ElementBounds& bounds,
                                                   const VecX& coeffs, int order) const {
    // Compute L2 error by comparing projected polynomial to WENO samples
    // at a finer set of points

    const auto& quad = get_quadrature(order);

    Real error_sq = 0.0;
    Real norm_sq = 0.0;

    for (int q = 0; q < quad.size(); ++q) {
        Real xi = quad.nodes()[q](0);
        Real eta = quad.nodes()[q](1);
        Real w = quad.weights()(q);

        // Map to world coordinates
        Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
        Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);

        // Get WENO sample
        Real f_exact = sampler_->sample(x, y);

        // Evaluate projection
        Real f_proj = evaluate_bernstein(coeffs, xi, eta, order);

        // Accumulate error
        Real diff = f_exact - f_proj;
        error_sq += w * diff * diff;
        norm_sq += w * f_exact * f_exact;
    }

    // Return relative error if norm is significant, else absolute error
    if (norm_sq > 1e-12) {
        return std::sqrt(error_sq / norm_sq);
    }
    return std::sqrt(error_sq);
}

// =============================================================================
// AdaptiveBathymetry - Caching
// =============================================================================

const GaussQuadrature2D& AdaptiveBathymetry::get_quadrature(int order) const {
    auto it = quad_cache_.find(order);
    if (it == quad_cache_.end()) {
        // Over-integrated quadrature for L2 projection
        int quad_order = overintegration_factor_ * order + 1;
        auto [iter, inserted] = quad_cache_.emplace(
            order, GaussQuadrature2D(quad_order, QuadratureType::GaussLegendre));
        return iter->second;
    }
    return it->second;
}

const BernsteinBasis1D& AdaptiveBathymetry::get_basis(int order) const {
    auto it = basis_cache_.find(order);
    if (it == basis_cache_.end()) {
        auto [iter, inserted] = basis_cache_.emplace(order, BernsteinBasis1D(order));
        return iter->second;
    }
    return it->second;
}

const MatX& AdaptiveBathymetry::get_projection_matrix(int order) const {
    auto it = projection_matrix_cache_.find(order);
    if (it == projection_matrix_cache_.end()) {
        MatX P = build_projection_matrix(order);
        auto [iter, inserted] = projection_matrix_cache_.emplace(order, std::move(P));
        return iter->second;
    }
    return it->second;
}

// =============================================================================
// AdaptiveBathymetry - L2 Projection
// =============================================================================

MatX AdaptiveBathymetry::build_projection_matrix(int order) const {
    const auto& quad = get_quadrature(order);
    const auto& basis = get_basis(order);

    int nq = quad.size();
    int n1d = order + 1;
    int n2d = n1d * n1d;

    // Build Bernstein basis matrix at quadrature points
    // B[q, i + n1d*j] = phi_xi(i) * phi_eta(j) at quad point q
    MatX B(nq, n2d);

    for (int q = 0; q < nq; ++q) {
        Real xi = quad.nodes()[q](0);
        Real eta = quad.nodes()[q](1);

        VecX phi_xi = basis.evaluate(xi);
        VecX phi_eta = basis.evaluate(eta);

        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int idx = i + n1d * j;
                B(q, idx) = phi_xi(i) * phi_eta(j);
            }
        }
    }

    // Weight matrix (diagonal quadrature weights)
    VecX w = quad.weights();

    // Build normal equation system: (B^T W B) c = B^T W f
    // We want P such that c = P * f, so P = (B^T W B)^{-1} B^T W

    // B^T W (nq x n2d)^T * diag(w) = (n2d x nq) * diag(w)
    MatX BtW(n2d, nq);
    for (int i = 0; i < n2d; ++i) {
        for (int q = 0; q < nq; ++q) {
            BtW(i, q) = B(q, i) * w(q);
        }
    }

    // B^T W B (n2d x n2d)
    MatX BtWB = BtW * B;

    // Solve for projection matrix: P = (B^T W B)^{-1} B^T W
    // Use LU decomposition for the solve
    Eigen::FullPivLU<MatX> lu(BtWB);
    MatX P = lu.solve(BtW);

    return P;
}

VecX AdaptiveBathymetry::sample_at_quadrature(const ElementBounds& bounds,
                                             const GaussQuadrature2D& quad) const {
    int nq = quad.size();
    VecX f_quad(nq);

    for (int q = 0; q < nq; ++q) {
        Real xi = quad.nodes()[q](0);
        Real eta = quad.nodes()[q](1);

        // Map reference coordinates to world coordinates
        Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
        Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);

        // Sample using selected method
        if (sampling_method_ == SamplingMethod::WENO5) {
            f_quad(q) = sampler_->sample(x, y);
        } else {
            // Bilinear interpolation from raw data
            f_quad(q) = raw_data_->get_depth(x, y);
        }
    }

    return f_quad;
}

Real AdaptiveBathymetry::evaluate_bernstein(const VecX& coeffs, Real xi, Real eta,
                                           int order) const {
    const auto& basis = get_basis(order);

    VecX phi_xi = basis.evaluate(xi);
    VecX phi_eta = basis.evaluate(eta);

    int n1d = order + 1;
    Real value = 0.0;

    for (int j = 0; j < n1d; ++j) {
        for (int i = 0; i < n1d; ++i) {
            int idx = i + n1d * j;
            value += coeffs(idx) * phi_xi(i) * phi_eta(j);
        }
    }

    return value;
}

}  // namespace drifter
