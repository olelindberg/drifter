#include "bathymetry/linear_bezier_surface.hpp"
#include <cmath>
#include <map>
#include <stdexcept>

namespace drifter {

LinearBezierSurface::LinearBezierSurface(const QuadtreeAdapter& mesh)
    : mesh_(&mesh) {
    build_dof_map();
}

Index LinearBezierSurface::update_mesh(const QuadtreeAdapter& mesh) {
    mesh_ = &mesh;

    // IMPORTANT: Element indices change after Morton sorting, so we must
    // rebuild the entire DOF map. However, we reuse corner_to_dof_ to
    // preserve existing DOF assignments (and their coefficient values).

    Index first_new_dof = num_global_dofs_;  // Track where new DOFs start
    Index new_num_elements = mesh_->num_elements();

    const Real tol = 1e-10;
    auto round_coord = [tol](Real x) -> int64_t {
        return static_cast<int64_t>(std::round(x / tol));
    };

    // Rebuild entire DOF map for new element ordering
    dof_map_.resize(new_num_elements);

    for (Index elem = 0; elem < new_num_elements; ++elem) {
        const auto& bounds = mesh_->element_bounds(elem);

        std::array<std::pair<Real, Real>, 4> corners = {{
            {bounds.xmin, bounds.ymin},
            {bounds.xmax, bounds.ymin},
            {bounds.xmin, bounds.ymax},
            {bounds.xmax, bounds.ymax}
        }};

        for (int c = 0; c < 4; ++c) {
            auto key = std::make_pair(round_coord(corners[c].first),
                                       round_coord(corners[c].second));
            auto it = corner_to_dof_.find(key);
            if (it == corner_to_dof_.end()) {
                // New corner - assign new DOF index
                corner_to_dof_[key] = num_global_dofs_;
                dof_map_[elem][c] = num_global_dofs_;
                ++num_global_dofs_;
            } else {
                // Existing corner - reuse existing DOF index
                dof_map_[elem][c] = it->second;
            }
        }
    }

    // Extend coefficient vector (preserve existing values)
    if (num_global_dofs_ > coefficients_.size()) {
        VecX new_coeffs = VecX::Zero(num_global_dofs_);
        new_coeffs.head(coefficients_.size()) = coefficients_;
        coefficients_ = std::move(new_coeffs);
    }

    return first_new_dof;
}

void LinearBezierSurface::build_dof_map() {
    // Build mapping from element corners to global DOFs
    // DOFs are shared at coincident corners for C0 continuity

    // Use a map from corner position (with tolerance) to global DOF index
    const Real tol = 1e-10;
    auto round_coord = [tol](Real x) -> int64_t {
        return static_cast<int64_t>(std::round(x / tol));
    };

    corner_to_dof_.clear();
    dof_map_.resize(mesh_->num_elements());
    num_global_dofs_ = 0;

    for (Index elem = 0; elem < mesh_->num_elements(); ++elem) {
        const auto& bounds = mesh_->element_bounds(elem);

        // Corner positions: [0,0], [1,0], [0,1], [1,1] in local coords
        // Physical: (xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)
        std::array<std::pair<Real, Real>, 4> corners = {{
            {bounds.xmin, bounds.ymin},
            {bounds.xmax, bounds.ymin},
            {bounds.xmin, bounds.ymax},
            {bounds.xmax, bounds.ymax}
        }};

        for (int c = 0; c < 4; ++c) {
            auto key = std::make_pair(round_coord(corners[c].first),
                                       round_coord(corners[c].second));
            auto it = corner_to_dof_.find(key);
            if (it == corner_to_dof_.end()) {
                corner_to_dof_[key] = num_global_dofs_;
                dof_map_[elem][c] = num_global_dofs_;
                ++num_global_dofs_;
            } else {
                dof_map_[elem][c] = it->second;
            }
        }
    }

    // Initialize coefficient vector
    coefficients_.setZero(num_global_dofs_);
}

void LinearBezierSurface::fit(const BathymetryData& data) {
    // Simple fitting: sample bathymetry at each global DOF position
    // This is a direct interpolation (no smoothing)

    // Build position array for global DOFs
    std::vector<Vec2> dof_positions(num_global_dofs_);
    std::vector<bool> dof_set(num_global_dofs_, false);

    for (Index elem = 0; elem < mesh_->num_elements(); ++elem) {
        const auto& bounds = mesh_->element_bounds(elem);
        std::array<Vec2, 4> corners = {{
            {bounds.xmin, bounds.ymin},
            {bounds.xmax, bounds.ymin},
            {bounds.xmin, bounds.ymax},
            {bounds.xmax, bounds.ymax}
        }};

        for (int c = 0; c < 4; ++c) {
            Index dof = dof_map_[elem][c];
            if (!dof_set[dof]) {
                dof_positions[dof] = corners[c];
                dof_set[dof] = true;
            }
        }
    }

    // Sample bathymetry at each DOF position
    for (Index dof = 0; dof < num_global_dofs_; ++dof) {
        Real x = dof_positions[dof](0);
        Real y = dof_positions[dof](1);
        Real depth = data.get_depth(x, y);
        // Store as negative z (bathymetry is depth positive downward)
        coefficients_(dof) = -depth;
    }

    is_fitted_ = true;
}

void LinearBezierSurface::fit_incremental(const BathymetryData& data,
                                           const std::vector<Index>& new_elements,
                                           Index first_new_dof) {
    // Only fit DOFs associated with new elements
    // Existing DOF values are preserved (already sampled in previous fit)

    for (Index elem : new_elements) {
        if (elem >= static_cast<Index>(dof_map_.size())) continue;

        const auto& bounds = mesh_->element_bounds(elem);
        std::array<Vec2, 4> corners = {{
            {bounds.xmin, bounds.ymin},
            {bounds.xmax, bounds.ymin},
            {bounds.xmin, bounds.ymax},
            {bounds.xmax, bounds.ymax}
        }};

        for (int c = 0; c < 4; ++c) {
            Index dof = dof_map_[elem][c];
            // Only sample if this is a new DOF (created after update_mesh was called)
            if (dof >= first_new_dof) {
                Real x = corners[c](0);
                Real y = corners[c](1);
                Real depth = data.get_depth(x, y);
                coefficients_(dof) = -depth;
            }
        }
    }

    is_fitted_ = true;
}

void LinearBezierSurface::fit(std::function<Real(Real, Real)> depth_func) {
    // Simple fitting: sample depth function at each global DOF position
    // This is a direct interpolation (no smoothing)

    // Build position array for global DOFs
    std::vector<Vec2> dof_positions(num_global_dofs_);
    std::vector<bool> dof_set(num_global_dofs_, false);

    for (Index elem = 0; elem < mesh_->num_elements(); ++elem) {
        const auto& bounds = mesh_->element_bounds(elem);
        std::array<Vec2, 4> corners = {
            {Vec2{bounds.xmin, bounds.ymin}, Vec2{bounds.xmax, bounds.ymin},
             Vec2{bounds.xmin, bounds.ymax}, Vec2{bounds.xmax, bounds.ymax}}};

        for (int c = 0; c < 4; ++c) {
            Index dof = dof_map_[elem][c];
            if (!dof_set[dof]) {
                dof_positions[dof] = corners[c];
                dof_set[dof] = true;
            }
        }
    }

    // Sample depth function at each DOF position
    for (Index dof = 0; dof < num_global_dofs_; ++dof) {
        Real x = dof_positions[dof](0);
        Real y = dof_positions[dof](1);
        Real depth = depth_func(x, y);
        // Store as negative z (bathymetry is depth positive downward)
        coefficients_(dof) = -depth;
    }

    is_fitted_ = true;
}

void LinearBezierSurface::fit_incremental(std::function<Real(Real, Real)> depth_func,
                                          const std::vector<Index>& new_elements,
                                          Index first_new_dof) {
    // Only fit DOFs associated with new elements
    // Existing DOF values are preserved (already sampled in previous fit)

    for (Index elem : new_elements) {
        if (elem >= static_cast<Index>(dof_map_.size()))
            continue;

        const auto& bounds = mesh_->element_bounds(elem);
        std::array<Vec2, 4> corners = {
            {Vec2{bounds.xmin, bounds.ymin}, Vec2{bounds.xmax, bounds.ymin},
             Vec2{bounds.xmin, bounds.ymax}, Vec2{bounds.xmax, bounds.ymax}}};

        for (int c = 0; c < 4; ++c) {
            Index dof = dof_map_[elem][c];
            // Only sample if this is a new DOF (created after update_mesh was called)
            if (dof >= first_new_dof) {
                Real x = corners[c](0);
                Real y = corners[c](1);
                Real depth = depth_func(x, y);
                coefficients_(dof) = -depth;
            }
        }
    }

    is_fitted_ = true;
}

Real LinearBezierSurface::evaluate(Real x, Real y) const {
    if (!is_fitted_) {
        throw std::runtime_error("Surface not fitted - call fit() first");
    }

    // Find containing element
    Index elem = mesh_->find_element(Vec2(x, y));
    if (elem < 0) {
        return 0.0;  // Outside domain
    }

    // Map to reference coordinates
    Real xi, eta;
    world_to_reference(x, y, elem, xi, eta);

    // Evaluate bilinear interpolation
    Eigen::Vector4d N = basis(xi, eta);
    Eigen::Vector4d coeffs = element_coefficients(elem);

    return N.dot(coeffs);
}

Real LinearBezierSurface::evaluate_in_element(Index elem, Real x, Real y) const {
    if (!is_fitted_) {
        throw std::runtime_error("Surface not fitted - call fit() first");
    }

    // Map to reference coordinates (no element lookup needed)
    Real xi, eta;
    world_to_reference(x, y, elem, xi, eta);

    // Evaluate bilinear interpolation
    Eigen::Vector4d N = basis(xi, eta);
    Eigen::Vector4d coeffs = element_coefficients(elem);

    return N.dot(coeffs);
}

Eigen::Vector4d LinearBezierSurface::element_coefficients(Index elem) const {
    Eigen::Vector4d coeffs;
    for (int c = 0; c < 4; ++c) {
        coeffs(c) = coefficients_(dof_map_[elem][c]);
    }
    return coeffs;
}

void LinearBezierSurface::world_to_reference(Real x, Real y, Index elem,
                                              Real& xi, Real& eta) const {
    const auto& bounds = mesh_->element_bounds(elem);
    xi = (x - bounds.xmin) / (bounds.xmax - bounds.xmin);
    eta = (y - bounds.ymin) / (bounds.ymax - bounds.ymin);
}

Eigen::Vector4d LinearBezierSurface::basis(Real xi, Real eta) {
    // Bilinear basis functions on [0,1]^2
    // N00 = (1-xi)(1-eta), N10 = xi(1-eta), N01 = (1-xi)eta, N11 = xi*eta
    return Eigen::Vector4d(
        (1 - xi) * (1 - eta),  // N00
        xi * (1 - eta),         // N10
        (1 - xi) * eta,         // N01
        xi * eta                // N11
    );
}

} // namespace drifter
