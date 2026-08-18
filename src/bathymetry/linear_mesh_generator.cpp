#include "bathymetry/linear_mesh_generator.hpp"
#include "io/quadtree_vtk_writer.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace drifter {

LinearMeshGenerator::LinearMeshGenerator(Real xmin, Real xmax, Real ymin, Real ymax,
                                         int nx, int ny, const LowriderRefinementConfig& config)
    : config_(config) {
    // Build initial uniform mesh
    mesh_.build_uniform(xmin, xmax, ymin, ymax, nx, ny);
}

void LinearMeshGenerator::load_bathymetry(const std::string& geotiff_path) {
    GeoTiffReader reader;
    auto data = std::make_shared<BathymetryData>(reader.load(geotiff_path));

    // Create depth function from bathymetry data
    depth_func_ = [data](Real x, Real y) -> Real {
        return data->get_depth(x, y);
    };
    land_mask_func_ = [data](Real x, Real y) -> bool {
        return data->is_land(x, y);
    };

    bathymetry_ = data;
    rebuild_surface();
}

void LinearMeshGenerator::load_bathymetry(const LowriderDataConfig& data_config) {
    if (data_config.primary_file.empty()) {
        std::cerr << "Warning: No primary bathymetry file specified" << std::endl;
        return;
    }

    // Build full paths
    std::string primary_path = data_config.data_dir + data_config.primary_file;
    std::vector<std::string> tile_paths;
    for (const auto& tile : data_config.tile_files) {
        tile_paths.push_back(data_config.data_dir + tile);
    }

    std::cout << "Loading bathymetry from: " << primary_path << std::endl;
    std::cout << "  with " << tile_paths.size() << " tile files" << std::endl;

    // Use MultiSourceBathymetry for blended data
    auto multi_bathy = std::make_shared<MultiSourceBathymetry>(primary_path, tile_paths);

    // Create depth and land mask functions
    depth_func_ = [multi_bathy](Real x, Real y) -> Real {
        try {
            return multi_bathy->evaluate(x, y);
        } catch (const std::out_of_range&) {
            return 0.0;
        }
    };

    land_mask_func_ = [multi_bathy](Real x, Real y) -> bool {
        try {
            return multi_bathy->is_land(x, y);
        } catch (const std::out_of_range&) {
            return true;
        }
    };

    // Also load primary as BathymetryData for compatibility
    GeoTiffReader reader;
    bathymetry_ = std::make_shared<BathymetryData>(reader.load(primary_path));

    rebuild_surface();
}

void LinearMeshGenerator::set_bathymetry_functions(
    std::function<Real(Real, Real)> depth_func,
    std::function<bool(Real, Real)> land_mask) {
    depth_func_ = std::move(depth_func);
    land_mask_func_ = std::move(land_mask);
    rebuild_surface();
}

void LinearMeshGenerator::rebuild_surface() {
    surface_ = std::make_unique<LinearBezierSurface>(mesh_);
    if (bathymetry_) {
        surface_->fit(*bathymetry_);
    }
}

LowriderAdaptiveResult LinearMeshGenerator::solve_adaptive() {
    LowriderAdaptiveResult result;
    result.iterations = 0;
    result.converged = false;

    if (!bathymetry_) {
        std::cerr << "Warning: No bathymetry data loaded, cannot compute error" << std::endl;
        result.num_elements = mesh_.num_elements();
        result.max_error = 0.0;
        result.mean_error = 0.0;
        result.converged = true;
        result.convergence_reason = "No bathymetry data";
        return result;
    }

    std::cout << "\nStarting adaptive refinement..." << std::endl;

    for (iteration_ = 0; iteration_ < config_.max_iterations; ++iteration_) {
        // Compute errors once per iteration (was previously computed 3x)
        auto errors = get_errors();
        Real max_err = max_error_from(errors);
        Real mean_err = mean_error_from(errors);

        std::cout << "Iteration " << iteration_ << ": elements=" << mesh_.num_elements()
                  << ", max_error=" << max_err << " m, mean_error=" << mean_err << " m" << std::endl;

        // Check stopping criteria
        std::string reason = check_convergence(max_err);
        if (!reason.empty()) {
            result.converged = true;
            result.convergence_reason = reason;
            break;
        }

        // Perform one adaptation step with pre-computed errors
        if (!adapt_once(errors)) {
            result.converged = true;
            result.convergence_reason = "No elements marked for refinement";
            break;
        }
    }

    result.iterations = iteration_;
    result.num_elements = mesh_.num_elements();

    // Final error computation
    auto final_errors = get_errors();
    result.max_error = max_error_from(final_errors);
    result.mean_error = mean_error_from(final_errors);

    if (!result.converged) {
        result.convergence_reason = "Max iterations reached";
    }

    return result;
}

bool LinearMeshGenerator::adapt_once() {
    if (!bathymetry_ || !surface_) {
        return false;
    }

    // Estimate errors
    auto errors = get_errors();

    return adapt_once(errors);
}

bool LinearMeshGenerator::adapt_once(const std::vector<LinearMeshElementError>& errors) {
    if (!bathymetry_ || !surface_) {
        return false;
    }

    // Select elements for refinement
    auto to_refine = select_for_refinement(errors);

    if (to_refine.empty()) {
        return false;
    }

    // Refine selected elements
    refine_elements(to_refine);

    // Rebuild surface for new mesh
    rebuild_surface();

    return true;
}

const LinearBezierSurface& LinearMeshGenerator::surface() const {
    if (!surface_) {
        throw std::runtime_error("Surface not built");
    }
    return *surface_;
}

const BathymetryData& LinearMeshGenerator::bathymetry() const {
    if (!bathymetry_) {
        throw std::runtime_error("Bathymetry not loaded");
    }
    return *bathymetry_;
}

std::vector<LinearMeshElementError> LinearMeshGenerator::get_errors() const {
    if (!surface_ || !bathymetry_) {
        return {};
    }
    LinearMeshErrorEstimator estimator(*surface_, *bathymetry_, config_.ngauss);
    return estimator.estimate_all();
}

Real LinearMeshGenerator::max_error() const {
    if (!surface_ || !bathymetry_) {
        return 0.0;
    }
    LinearMeshErrorEstimator estimator(*surface_, *bathymetry_, config_.ngauss);
    return estimator.max_error(config_.error_metric);
}

Real LinearMeshGenerator::mean_error() const {
    if (!surface_ || !bathymetry_) {
        return 0.0;
    }
    LinearMeshErrorEstimator estimator(*surface_, *bathymetry_, config_.ngauss);
    return estimator.mean_error(config_.error_metric);
}

Real LinearMeshGenerator::max_error_from(const std::vector<LinearMeshElementError>& errors) const {
    Real max_err = 0.0;
    for (const auto& err : errors) {
        max_err = std::max(max_err, LinearMeshErrorEstimator::get_metric(err, config_.error_metric));
    }
    return max_err;
}

Real LinearMeshGenerator::mean_error_from(const std::vector<LinearMeshElementError>& errors) const {
    if (errors.empty()) {
        return 0.0;
    }
    Real sum = 0.0;
    for (const auto& err : errors) {
        sum += LinearMeshErrorEstimator::get_metric(err, config_.error_metric);
    }
    return sum / static_cast<Real>(errors.size());
}

std::vector<Index> LinearMeshGenerator::select_for_refinement(
    const std::vector<LinearMeshElementError>& errors) const {
    if (errors.empty()) {
        return {};
    }

    // Dorfler marking: select elements capturing theta fraction of total squared error
    std::vector<std::pair<Real, Index>> error_list;
    error_list.reserve(errors.size());

    Real total_sq = 0.0;
    for (const auto& err : errors) {
        Real metric = LinearMeshErrorEstimator::get_metric(err, config_.error_metric);
        error_list.push_back({metric, err.element});
        total_sq += metric * metric;
    }

    // Sort by error (descending)
    std::sort(error_list.begin(), error_list.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Select until we capture theta * total_sq
    Real target = config_.dorfler_theta * total_sq;
    Real accumulated = 0.0;
    std::vector<Index> selected;

    for (const auto& [err, elem] : error_list) {
        // Check level constraint
        auto level = mesh_.element_level(elem);
        if (level.max_level() >= config_.max_level) {
            continue;  // Already at max level
        }

        selected.push_back(elem);
        accumulated += err * err;

        if (accumulated >= target) {
            break;
        }
    }

    // Also check max elements constraint
    Index projected = mesh_.num_elements() + 3 * static_cast<Index>(selected.size());
    if (projected > config_.max_elements) {
        // Reduce selection to stay within limit
        Index available = (config_.max_elements - mesh_.num_elements()) / 3;
        if (available > 0 && static_cast<Index>(selected.size()) > available) {
            selected.resize(available);
        } else if (available <= 0) {
            selected.clear();
        }
    }

    return selected;
}

void LinearMeshGenerator::refine_elements(const std::vector<Index>& elements_to_refine) {
    if (elements_to_refine.empty()) {
        return;
    }

    std::cout << "  Refining " << elements_to_refine.size() << " elements" << std::endl;

    // Use QuadtreeAdapter's refine() method which handles:
    // - Splitting each element into 4 children
    // - Balancing for 2:1 constraint
    // - Rebuilding lookup structures
    Index refined = mesh_.refine(elements_to_refine);

    std::cout << "  Refined " << refined << " elements, new total: " << mesh_.num_elements() << std::endl;
}

std::string LinearMeshGenerator::check_convergence(Real max_err) const {
    if (max_err < config_.error_threshold) {
        return "Error threshold reached";
    }
    if (mesh_.num_elements() >= config_.max_elements) {
        return "Max elements reached";
    }
    return "";  // Not converged
}

void LinearMeshGenerator::write_vtk(const std::string& filename) const {
    QuadtreeVTKWriter writer;
    if (surface_ && surface_->is_fitted()) {
        writer.write(filename, mesh_, *surface_);
    } else {
        writer.write_mesh_only(filename, mesh_);
    }
}

} // namespace drifter
