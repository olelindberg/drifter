#include "bathymetry/adaptive_cg_bezier_smoother.hpp"
#include "bathymetry/biharmonic_assembler.hpp"
#include "mesh/refine_mask.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace drifter {

// =============================================================================
// Constructors
// =============================================================================

AdaptiveCGBezierSmoother::AdaptiveCGBezierSmoother(Real xmin, Real xmax, Real ymin, Real ymax,
                                                   int nx, int ny,
                                                   const AdaptiveCGBezierConfig &config)
    : config_(config) {
    // Create owned octree with domain bounds
    // Use z = [-1, 0] as a dummy vertical extent (only bottom face matters)
    octree_owned_ = std::make_unique<OctreeAdapter>(xmin, xmax, ymin, ymax, -1.0, 0.0);
    octree_owned_->build_uniform(nx, ny, 1); // 1 layer in z
    octree_ = octree_owned_.get();

    // Create quadtree from bottom face
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);

    // Initialize Gauss quadrature
    init_gauss_quadrature();
}

AdaptiveCGBezierSmoother::AdaptiveCGBezierSmoother(OctreeAdapter &octree,
                                                   const AdaptiveCGBezierConfig &config)
    : config_(config), octree_(&octree) {
    // Create quadtree from bottom face of provided octree
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);

    // Initialize Gauss quadrature
    init_gauss_quadrature();
}

// =============================================================================
// Gauss quadrature initialization
// =============================================================================

void AdaptiveCGBezierSmoother::init_gauss_quadrature() {
    int ngauss = config_.ngauss_error;
    gauss_nodes_.resize(ngauss);
    gauss_weights_.resize(ngauss);

    // Gauss-Legendre nodes and weights on [0, 1]
    // Precomputed values for common orders
    if (ngauss == 1) {
        gauss_nodes_ << 0.5;
        gauss_weights_ << 1.0;
    } else if (ngauss == 2) {
        Real a = 0.5 / std::sqrt(3.0);
        gauss_nodes_ << 0.5 - a, 0.5 + a;
        gauss_weights_ << 0.5, 0.5;
    } else if (ngauss == 3) {
        Real a = 0.5 * std::sqrt(0.6);
        gauss_nodes_ << 0.5 - a, 0.5, 0.5 + a;
        gauss_weights_ << 5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0;
    } else if (ngauss == 4) {
        // Nodes and weights for 4-point GL on [0,1]
        gauss_nodes_ << 0.0694318442029737, 0.3300094782075719, 0.6699905217924281,
            0.9305681557970262;
        gauss_weights_ << 0.1739274225687269, 0.3260725774312731, 0.3260725774312731,
            0.1739274225687269;
    } else if (ngauss == 5) {
        gauss_nodes_ << 0.0469100770306680, 0.2307653449471585, 0.5, 0.7692346550528415,
            0.9530899229693319;
        gauss_weights_ << 0.1184634425280945, 0.2393143352496832, 0.2844444444444444,
            0.2393143352496832, 0.1184634425280945;
    } else if (ngauss == 6) {
        gauss_nodes_ << 0.0337652428984240, 0.1693953067668677, 0.3806904069584015,
            0.6193095930415985, 0.8306046932331323, 0.9662347571015760;
        gauss_weights_ << 0.0856622461895852, 0.1803807865240693, 0.2339569672863455,
            0.2339569672863455, 0.1803807865240693, 0.0856622461895852;
    } else {
        throw std::invalid_argument(
            "AdaptiveCGBezierSmoother: ngauss_error must be between 1 and 6");
    }
}

// =============================================================================
// Data input
// =============================================================================

void AdaptiveCGBezierSmoother::set_bathymetry_data(const BathymetrySource &source) {
    // Wrap BathymetrySource in a lambda that captures by reference
    // Note: The source must outlive this smoother
    bathy_func_ = [&source](Real x, Real y) -> Real { return source.evaluate(x, y); };
}

void AdaptiveCGBezierSmoother::set_bathymetry_data(std::function<Real(Real, Real)> bathy_func) {
    bathy_func_ = std::move(bathy_func);
}

// =============================================================================
// Internal: Smoother management
// =============================================================================

void AdaptiveCGBezierSmoother::rebuild_smoother() {
    // Sync quadtree with refined octree
    quadtree_ = std::make_unique<QuadtreeAdapter>(*octree_);

    // Create new CG smoother for updated mesh
    smoother_ = std::make_unique<CGBezierBathymetrySmoother>(*quadtree_, config_.smoother_config);

    // Apply bathymetry data
    apply_bathymetry_to_smoother();
}

void AdaptiveCGBezierSmoother::apply_bathymetry_to_smoother() {
    if (!smoother_) {
        throw std::runtime_error("AdaptiveCGBezierSmoother: smoother not initialized");
    }
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGBezierSmoother: bathymetry data not set");
    }
    smoother_->set_bathymetry_data(bathy_func_);
}

// =============================================================================
// Error estimation
// =============================================================================

Real AdaptiveCGBezierSmoother::compute_element_l2_error(Index elem) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGBezierSmoother: must solve before computing errors");
    }

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;

    Real error_sq = 0.0;
    int ngauss = config_.ngauss_error;

    // 2D Gauss-Legendre quadrature
    for (int j = 0; j < ngauss; ++j) {
        for (int i = 0; i < ngauss; ++i) {
            Real u = gauss_nodes_(i);
            Real v = gauss_nodes_(j);
            Real w = gauss_weights_(i) * gauss_weights_(j);

            // Map to physical coordinates
            Real x = bounds.xmin + u * dx;
            Real y = bounds.ymin + v * dy;

            // Raw bathymetry value
            Real z_data = bathy_func_(x, y);

            // CG Bezier approximation value
            Real z_bezier = smoother_->evaluate(x, y);

            // Accumulate weighted squared difference
            Real diff = z_data - z_bezier;
            error_sq += w * diff * diff;
        }
    }

    // L2 error = sqrt(integral of diff^2 over element)
    // The quadrature weights sum to 1 on [0,1]^2, so multiply by area
    return std::sqrt(error_sq * dx * dy);
}

CGElementErrorEstimate AdaptiveCGBezierSmoother::estimate_element_error(Index elem) const {
    CGElementErrorEstimate result;
    result.element = elem;

    const QuadBounds &bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real area = dx * dy;

    result.l2_error = compute_element_l2_error(elem);
    // Normalize by sqrt(area) to get average error magnitude
    result.normalized_error = result.l2_error / std::sqrt(area);
    result.should_refine = (result.normalized_error > config_.error_threshold);

    return result;
}

std::vector<CGElementErrorEstimate> AdaptiveCGBezierSmoother::estimate_errors() const {
    std::vector<CGElementErrorEstimate> errors;
    errors.reserve(quadtree_->num_elements());

    for (Index e = 0; e < quadtree_->num_elements(); ++e) {
        errors.push_back(estimate_element_error(e));
    }

    return errors;
}

Real AdaptiveCGBezierSmoother::max_error() const {
    auto errors = estimate_errors();
    Real max_err = 0.0;
    for (const auto &err : errors) {
        max_err = std::max(max_err, err.normalized_error);
    }
    return max_err;
}

Real AdaptiveCGBezierSmoother::mean_error() const {
    auto errors = estimate_errors();
    Real sum_err = 0.0;
    for (const auto &err : errors) {
        sum_err += err.normalized_error;
    }
    return errors.empty() ? 0.0 : sum_err / static_cast<Real>(errors.size());
}

// =============================================================================
// Mesh refinement
// =============================================================================

void AdaptiveCGBezierSmoother::refine_elements(const std::vector<Index> &elements_to_refine) {
    if (elements_to_refine.empty())
        return;

    // Create refinement masks (XY only for 2D bathymetry)
    std::vector<RefineMask> masks(elements_to_refine.size(), RefineMask::XY);

    // Refine octree (auto-balances to maintain 2:1 constraint)
    octree_->refine(elements_to_refine, masks);

    // Rebuild smoother for new mesh
    rebuild_smoother();
}

// =============================================================================
// Adaptive solve
// =============================================================================

CGAdaptationResult AdaptiveCGBezierSmoother::adapt_once() {
    CGAdaptationResult result;
    result.iteration = static_cast<int>(history_.size());

    // Create smoother if not exists
    if (!smoother_) {
        rebuild_smoother();
    }

    // Solve on current mesh
    smoother_->solve();

    // Estimate errors
    auto errors = estimate_errors();

    // Compute statistics
    Real max_err = 0.0;
    Real sum_err = 0.0;
    for (const auto &err : errors) {
        max_err = std::max(max_err, err.normalized_error);
        sum_err += err.normalized_error;
    }

    result.num_elements = quadtree_->num_elements();
    result.max_error = max_err;
    result.mean_error = errors.empty() ? 0.0 : sum_err / static_cast<Real>(errors.size());

    // Check stopping criteria
    bool error_converged = (max_err <= config_.error_threshold);
    bool max_elements_reached = (static_cast<int>(result.num_elements) >= config_.max_elements);

    if (error_converged || max_elements_reached) {
        result.converged = true;
        result.elements_refined = 0;
        return result;
    }

    // Sort errors by normalized_error (descending) to find worst elements
    std::sort(errors.begin(), errors.end(),
              [](const CGElementErrorEstimate &a, const CGElementErrorEstimate &b) {
                  return a.normalized_error > b.normalized_error;
              });

    // Select top refine_fraction elements for refinement
    Index num_to_refine =
        static_cast<Index>(std::ceil(config_.refine_fraction * static_cast<Real>(errors.size())));
    num_to_refine = std::max(Index(1), num_to_refine); // At least 1 element

    // Collect elements to refine, filtering by refinement level limit
    std::vector<Index> valid_refine;
    for (Index i = 0; i < num_to_refine && i < static_cast<Index>(errors.size()); ++i) {
        Index elem = errors[i].element;
        QuadLevel level = quadtree_->element_level(elem);
        if (level.max_level() < config_.max_refinement_level) {
            valid_refine.push_back(elem);
        }
    }

    result.elements_refined = static_cast<Index>(valid_refine.size());

    if (valid_refine.empty()) {
        // Can't refine further due to level limit
        result.converged = true;
    } else {
        refine_elements(valid_refine);
        result.converged = false;
    }

    return result;
}

CGAdaptationResult AdaptiveCGBezierSmoother::solve_adaptive() {
    if (!bathy_func_) {
        throw std::runtime_error("AdaptiveCGBezierSmoother: bathymetry data not set");
    }

    CGAdaptationResult result;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        result = adapt_once();
        history_.push_back(result);

        if (config_.verbose) {
            std::cout << "Iteration " << iter << ": " << result.num_elements << " elements, "
                      << "max_error=" << result.max_error << " m, "
                      << "mean_error=" << result.mean_error << " m, "
                      << "refined=" << result.elements_refined << "\n";
        }

        if (result.converged) {
            if (config_.verbose) {
                std::cout << "Converged after " << (iter + 1) << " iterations\n";
            }
            break;
        }
    }

    // Ensure final smoother is solved (in case we stopped after refinement)
    if (smoother_ && !smoother_->is_solved()) {
        smoother_->solve();
    }

    return result;
}

// =============================================================================
// Accessors
// =============================================================================

const CGBezierBathymetrySmoother &AdaptiveCGBezierSmoother::smoother() const {
    if (!smoother_) {
        throw std::runtime_error("AdaptiveCGBezierSmoother: smoother not "
                                 "initialized (call solve first)");
    }
    return *smoother_;
}

Real AdaptiveCGBezierSmoother::evaluate(Real x, Real y) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGBezierSmoother: must solve before evaluating");
    }
    return smoother_->evaluate(x, y);
}

void AdaptiveCGBezierSmoother::write_vtk(const std::string &filename, int resolution) const {
    if (!smoother_ || !smoother_->is_solved()) {
        throw std::runtime_error("AdaptiveCGBezierSmoother: must solve before writing VTK");
    }
    smoother_->write_vtk(filename, resolution);
}

} // namespace drifter
