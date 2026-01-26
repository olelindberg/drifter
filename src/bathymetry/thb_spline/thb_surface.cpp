#include "bathymetry/thb_spline/thb_surface.hpp"

#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace drifter {

THBSurface::THBSurface(const OctreeAdapter& octree, const THBSurfaceConfig& config)
    : config_(config) {
    // Create quadtree from octree bottom face
    owned_quadtree_ = std::make_unique<QuadtreeAdapter>();
    owned_quadtree_->sync_with_octree(octree);
    quadtree_ = owned_quadtree_.get();

    initialize_from_quadtree(*quadtree_);
}

THBSurface::THBSurface(const QuadtreeAdapter& quadtree, const THBSurfaceConfig& config)
    : config_(config), quadtree_(&quadtree) {
    initialize_from_quadtree(quadtree);
}

THBSurface::~THBSurface() = default;

void THBSurface::initialize_from_quadtree(const QuadtreeAdapter& quadtree) {
    // Determine domain bounds from quadtree
    Real xmin = std::numeric_limits<Real>::max();
    Real xmax = std::numeric_limits<Real>::lowest();
    Real ymin = std::numeric_limits<Real>::max();
    Real ymax = std::numeric_limits<Real>::lowest();

    Real max_elem_size = 0.0;
    Real min_elem_size = std::numeric_limits<Real>::max();

    for (Index e = 0; e < quadtree.num_elements(); ++e) {
        const QuadBounds& bounds = quadtree.element_bounds(e);
        xmin = std::min(xmin, bounds.xmin);
        xmax = std::max(xmax, bounds.xmax);
        ymin = std::min(ymin, bounds.ymin);
        ymax = std::max(ymax, bounds.ymax);

        Vec2 size = quadtree.element_size(e);
        Real avg_size = 0.5 * (size.x() + size.y());
        max_elem_size = std::max(max_elem_size, avg_size);
        min_elem_size = std::min(min_elem_size, avg_size);
    }

    // Create knot vectors matching octree sizes
    BSplineKnotVector knots_u =
        BSplineKnotVector::from_octree_sizes(xmin, xmax, max_elem_size, min_elem_size);
    BSplineKnotVector knots_v =
        BSplineKnotVector::from_octree_sizes(ymin, ymax, max_elem_size, min_elem_size);

    // Create hierarchy
    hierarchy_ = std::make_unique<THBHierarchy>(knots_u, knots_v);

    // Create refinement mask from quadtree
    mask_ = std::make_unique<THBRefinementMask>(quadtree, *hierarchy_);

    // Create truncation data
    truncation_ = std::make_unique<THBTruncation>(*hierarchy_, *mask_);

    // Create data fitting assembler
    data_fitting_ =
        std::make_unique<THBDataFitting>(quadtree, *hierarchy_, *mask_, *truncation_);

    if (config_.verbose) {
        std::cout << "THBSurface initialized:\n"
                  << "  Domain: [" << xmin << ", " << xmax << "] x [" << ymin << ", "
                  << ymax << "]\n"
                  << "  Max level: " << hierarchy_->max_level() << "\n"
                  << "  Total hierarchy DOFs: " << hierarchy_->total_dofs() << "\n"
                  << "  Active DOFs: " << mask_->num_active() << "\n";
    }
}

void THBSurface::set_bathymetry_data(const BathymetrySource& source) {
    data_fitting_->set_from_bathymetry_source(source, config_.ngauss);
    data_set_ = true;
    solved_ = false;

    if (config_.verbose) {
        std::cout << "THBSurface: sampled " << data_fitting_->num_points()
                  << " data points\n";
    }
}

void THBSurface::set_bathymetry_data(std::function<Real(Real, Real)> bathy_func) {
    data_fitting_->set_from_function(bathy_func, config_.ngauss);
    data_set_ = true;
    solved_ = false;

    if (config_.verbose) {
        std::cout << "THBSurface: sampled " << data_fitting_->num_points()
                  << " data points\n";
    }
}

void THBSurface::solve() {
    if (!data_set_) {
        throw std::runtime_error("THBSurface::solve: bathymetry data not set");
    }

    const Index num_active = mask_->num_active();

    // Assemble normal equations
    SpMat AtWA;
    VecX AtWb;
    data_fitting_->assemble_normal_equations(AtWA, AtWb);

    if (config_.verbose) {
        std::cout << "THBSurface: assembled " << num_active << " x " << num_active
                  << " normal matrix with " << AtWA.nonZeros() << " non-zeros\n";
    }

    // Add smoothing regularization if requested
    if (config_.smoothing_weight > 0.0) {
        // Simple Tikhonov regularization: add λI to diagonal
        for (Index i = 0; i < num_active; ++i) {
            AtWA.coeffRef(i, i) += config_.smoothing_weight;
        }
    }

    // Solve using SparseLU
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(AtWA);
    solver.factorize(AtWA);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("THBSurface::solve: factorization failed");
    }

    active_coeffs_ = solver.solve(AtWb);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("THBSurface::solve: solve failed");
    }

    solved_ = true;

    if (config_.verbose) {
        std::cout << "THBSurface: solved successfully\n";
    }
}

Real THBSurface::evaluate(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("THBSurface::evaluate: not solved");
    }

    // Find the finest refinement level at this point
    int point_level = mask_->level_at_point(x, y);

    Real value = 0.0;
    const auto& active_funcs = mask_->active_functions();

    for (Index idx = 0; idx < static_cast<Index>(active_funcs.size()); ++idx) {
        const auto& [level, i, j] = active_funcs[idx];

        // Only evaluate functions at the SAME level as the point
        // This ensures partition of unity at each point
        if (level != point_level) {
            continue;
        }

        // Convert physical to parameter coordinates at this level
        auto [u, v] = hierarchy_->physical_to_parameter(level, x, y);

        // Evaluate basis function (no truncation needed with single-level evaluation)
        const auto& basis = hierarchy_->basis(level);
        Real basis_val = basis.evaluate(i, j, u, v);

        value += active_coeffs_(idx) * basis_val;
    }

    return value;
}

Vec2 THBSurface::evaluate_gradient(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("THBSurface::evaluate_gradient: not solved");
    }

    // Find the finest refinement level at this point
    int point_level = mask_->level_at_point(x, y);

    Real dx = 0.0;
    Real dy = 0.0;
    const auto& active_funcs = mask_->active_functions();

    for (Index idx = 0; idx < static_cast<Index>(active_funcs.size()); ++idx) {
        const auto& [level, i, j] = active_funcs[idx];

        // Only evaluate functions at the SAME level as the point
        if (level != point_level) {
            continue;
        }

        // Convert physical to parameter coordinates
        auto [u, v] = hierarchy_->physical_to_parameter(level, x, y);

        // Get scaling factors for derivative conversion
        // du/dx = num_spans / domain_width
        Real du_dx = static_cast<Real>(hierarchy_->knots_u().num_spans(level)) /
                     (hierarchy_->domain_max_u() - hierarchy_->domain_min_u());
        Real dv_dy = static_cast<Real>(hierarchy_->knots_v().num_spans(level)) /
                     (hierarchy_->domain_max_v() - hierarchy_->domain_min_v());

        // Evaluate basis function derivatives (in parameter space)
        const auto& basis = hierarchy_->basis(level);
        Real dN_du = basis.evaluate_du(i, j, u, v);
        Real dN_dv = basis.evaluate_dv(i, j, u, v);

        dx += active_coeffs_(idx) * dN_du * du_dx;
        dy += active_coeffs_(idx) * dN_dv * dv_dy;
    }

    return Vec2(dx, dy);
}

void THBSurface::write_vtk(const std::string& filename, int resolution) const {
    if (!solved_) {
        throw std::runtime_error("THBSurface::write_vtk: not solved");
    }

    const std::string vtk_filename = filename + ".vtk";
    std::ofstream file(vtk_filename);

    if (!file.is_open()) {
        throw std::runtime_error("THBSurface::write_vtk: cannot open file " +
                                 vtk_filename);
    }

    file << std::setprecision(12) << std::scientific;

    // Sample on a regular grid for VTK output
    const int nx = resolution;
    const int ny = resolution;
    const int num_points = nx * ny;
    const int num_cells = (nx - 1) * (ny - 1);

    const Real dx = (domain_max_x() - domain_min_x()) / (nx - 1);
    const Real dy = (domain_max_y() - domain_min_y()) / (ny - 1);

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "THB-spline surface\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << nx << " " << ny << " 1\n";
    file << "POINTS " << num_points << " double\n";

    // Write points
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Real x = domain_min_x() + i * dx;
            Real y = domain_min_y() + j * dy;
            Real z = evaluate(x, y);
            file << x << " " << y << " " << z << "\n";
        }
    }

    // Write depth as point data
    file << "POINT_DATA " << num_points << "\n";
    file << "SCALARS depth double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Real x = domain_min_x() + i * dx;
            Real y = domain_min_y() + j * dy;
            file << evaluate(x, y) << "\n";
        }
    }

    file.close();

    if (config_.verbose) {
        std::cout << "THBSurface: wrote VTK to " << vtk_filename << "\n";
    }
}

}  // namespace drifter
