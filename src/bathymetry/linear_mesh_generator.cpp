#include "bathymetry/linear_mesh_generator.hpp"
#include "bathymetry/multi_source_pixel_error_estimator.hpp"
#include "core/logger.hpp"
#include "io/quadtree_vtk_writer.hpp"
#include "mesh/multi_source_bathymetry.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

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
        LOG_WARNING("No primary bathymetry file specified");
        return;
    }

    // Build full paths
    std::string primary_path = data_config.data_dir + data_config.primary_file;
    std::vector<std::string> tile_paths;
    for (const auto& tile : data_config.tile_files) {
        tile_paths.push_back(data_config.data_dir + tile);
    }

    LOG_INFO("Loading bathymetry from: " << primary_path);
    LOG_INFO("  with " << tile_paths.size() << " tile files");

    // Use MultiSourceBathymetry for blended data
    multi_bathy_ = std::make_shared<MultiSourceBathymetry>(primary_path, tile_paths);

    // Create depth and land mask functions
    depth_func_ = [this](Real x, Real y) -> Real {
        try {
            return multi_bathy_->evaluate(x, y);
        } catch (const std::out_of_range&) {
            return 0.0;
        }
    };

    land_mask_func_ = [this](Real x, Real y) -> bool {
        try {
            return multi_bathy_->is_land(x, y);
        } catch (const std::out_of_range&) {
            return true;
        }
    };

    // Store primary as BathymetryData for compatibility with single-source error estimation
    bathymetry_ = std::make_shared<BathymetryData>(multi_bathy_->get_primary());

    rebuild_surface();
}

void LinearMeshGenerator::set_bathymetry_functions(
    std::function<Real(Real, Real)> depth_func,
    std::function<bool(Real, Real)> land_mask) {
    depth_func_ = std::move(depth_func);
    land_mask_func_ = std::move(land_mask);
    rebuild_surface();
}

void LinearMeshGenerator::load_coastline(const LowriderCoastlineConfig& config) {
    if (!config.enabled()) {
        return;
    }

    // Get domain bounds for spatial filtering (critical for global datasets)
    const auto& domain = mesh_.domain_bounds();

    CoastlineReader reader;
    // Use load overload with domain bounds for GDAL spatial filter
    if (!reader.load(config.file, config.layer, config.srs,
                     domain.xmin, domain.ymin, domain.xmax, domain.ymax)) {
        LOG_WARNING("Failed to load coastline: " << reader.last_error());
        return;
    }

    if (config.min_polygon_area > 0.0) {
        reader.remove_small_polygons(config.min_polygon_area);
    }

    LOG_INFO("Loaded coastline: " << reader.num_polygons() << " segments");

    // Write coastline to VTK for debugging
    reader.write_vtk("/tmp/coastline_debug");

    // Write circumradius comb visualization
    reader.write_circumradius_comb_vtk("/tmp/coastline_circumradius_comb");

    // Build R-tree with domain filter (segments already filtered during load)
    coastline_index_ = reader.build_index(domain.xmin, domain.ymin,
                                           domain.xmax, domain.ymax);
    coastline_max_level_ = config.max_level;
    LOG_INFO("Built coastline index: " << coastline_index_->num_segments() << " segments, "
                                       << coastline_index_->num_circumradius_points()
                                       << " circumradius samples (filtered to domain)");
}

int LinearMeshGenerator::refine_coastline() {
    if (!coastline_index_ || coastline_index_->num_segments() == 0) {
        LOG_INFO("Coastline refinement stopped: no coastline data (0 iterations, "
                 << mesh_.num_elements() << " elements)");
        return 0;
    }

    // Circumradius-based refinement: refine while the element holds a coastline
    // feature tighter than the element itself. The threshold is the element's own
    // shorter side, so it halves with every pass and an element stops as soon as it
    // is smaller than the tightest feature inside it. Kept separate from the scan
    // so the stopping reason can re-test the elements a limit skipped.
    auto wants_refinement = [this](Index elem) {
        const auto& bounds = mesh_.element_bounds(elem);
        Real element_side = std::min(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin);
        return coastline_index_->has_circumradius_below(bounds.xmin, bounds.ymin, bounds.xmax,
                                                        bounds.ymax, element_side);
    };

    int iterations = 0;
    const char* reason = "coastline resolved to the circumradius of its features";

    while (true) {
        std::vector<Index> to_refine;
        std::vector<Index> at_max_level;
        std::vector<Index> at_pixel_limit;

        for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
            auto level = mesh_.element_level(elem);
            if (level.max_level() >= coastline_max_level_) {
                at_max_level.push_back(elem);
                continue;
            }

            const auto& bounds = mesh_.element_bounds(elem);

            // Check pixel resolution limit for coastline refinement
            if (config_.enforce_pixel_limit && multi_bathy_) {
                Real cx = (bounds.xmin + bounds.xmax) / 2.0;
                Real cy = (bounds.ymin + bounds.ymax) / 2.0;

                Real min_size = multi_bathy_->get_min_element_size_meters(cx, cy);
                if (min_size > 0.0) {
                    auto size = mesh_.element_size(elem);
                    Real elem_min_size = std::min(size(0), size(1));
                    // Check if CHILDREN would be below pixel resolution
                    if (elem_min_size < 2.0 * min_size) {
                        at_pixel_limit.push_back(elem);
                        continue;  // Children would be below pixel resolution
                    }
                }
            }

            if (wants_refinement(elem)) {
                to_refine.push_back(elem);
            }
        }

        if (to_refine.empty()) {
            // Nothing left to refine, so this pass is the last one: the reason is
            // whichever limit held back an element that still wanted refining.
            bool level_bound =
                std::any_of(at_max_level.begin(), at_max_level.end(), wants_refinement);
            bool pixel_bound =
                std::any_of(at_pixel_limit.begin(), at_pixel_limit.end(), wants_refinement);
            if (level_bound && pixel_bound) {
                reason = "max refinement level and pixel resolution reached";
            } else if (level_bound) {
                reason = "max refinement level reached";
            } else if (pixel_bound) {
                reason = "pixel resolution reached";
            }
            break;
        }

        refine_elements(to_refine);
        ++iterations;

        LOG_INFO("Coastline iteration " << iterations << ": refined " << to_refine.size()
                                          << " elements, " << mesh_.num_elements() << " total");
    }

    LOG_INFO("Coastline refinement stopped: " << reason << " (" << iterations
                                              << " iterations, " << mesh_.num_elements()
                                              << " elements)");

    // Rebuild surface after coastline refinement
    if (iterations > 0) {
        rebuild_surface();
        // Reset error cache
        error_valid_.assign(mesh_.num_elements(), false);
    }

    return iterations;
}

void LinearMeshGenerator::rebuild_surface() {
    surface_ = std::make_unique<LinearBezierSurface>(mesh_);
    if (!depth_func_) {
        throw std::runtime_error(
            "LinearMeshGenerator: depth_func_ not set. Call load_bathymetry() first.");
    }
    surface_->fit(depth_func_);
}

void LinearMeshGenerator::rebuild_surface_incremental(const std::vector<Index>& new_elements) {
    if (!surface_) {
        // Full rebuild if no existing surface
        rebuild_surface();
        return;
    }

    if (!depth_func_) {
        throw std::runtime_error(
            "LinearMeshGenerator: depth_func_ not set for incremental fit.");
    }

    // Update mesh reference and extend DOF map
    // Returns the index of first new DOF (for incremental fitting)
    Index first_new_dof = surface_->update_mesh(mesh_);

    // Only fit new DOFs (those with index >= first_new_dof)
    surface_->fit_incremental(depth_func_, new_elements, first_new_dof);
}

std::vector<ElementError> LinearMeshGenerator::get_errors_cached() {
    if (!surface_ || !bathymetry_) {
        return {};
    }

    Index num_elements = mesh_.num_elements();

    // Initialize cache on first call or if mesh changed dramatically
    if (cached_errors_.size() != static_cast<size_t>(num_elements)) {
        cached_errors_.resize(num_elements);
        error_valid_.assign(num_elements, false);
    }

    // Create estimator for computing individual element errors
    std::unique_ptr<ElementErrorEstimator> estimator;
    if (multi_bathy_ &&
        (config_.error_metric == ErrorMetricType::PixelMaxError ||
         config_.error_metric == ErrorMetricType::PixelRMSE)) {
        estimator = std::make_unique<MultiSourcePixelMaxErrorEstimator>(
            *surface_, *bathymetry_, *multi_bathy_, mesh_);
    } else {
        estimator = create_error_estimator(config_.error_metric, *surface_,
                                            *bathymetry_, mesh_, config_.ngauss);
    }

    // Compute errors only for invalidated elements
    Index recomputed = 0;
    for (Index elem = 0; elem < num_elements; ++elem) {
        if (!error_valid_[elem]) {
            cached_errors_[elem] = estimator->estimate_element(elem);
            error_valid_[elem] = true;
            ++recomputed;
        }
    }

    return cached_errors_;
}

void LinearMeshGenerator::invalidate_affected_errors(const std::vector<Index>& new_elements) {
    // Invalidate errors for new elements
    for (Index elem : new_elements) {
        if (elem < static_cast<Index>(error_valid_.size())) {
            error_valid_[elem] = false;
        }
    }

    // Also invalidate neighbors of new elements (their boundaries changed)
    for (Index elem : new_elements) {
        if (elem >= mesh_.num_elements()) continue;

        auto neighbors = mesh_.get_edge_neighbors(elem);
        for (const auto& info : neighbors) {
            for (Index nb : info.neighbor_elements) {
                if (nb < static_cast<Index>(error_valid_.size())) {
                    error_valid_[nb] = false;
                }
            }
        }
    }
}

LowriderAdaptiveResult LinearMeshGenerator::solve_adaptive() {
    LowriderAdaptiveResult result;
    result.iterations = 0;
    result.converged = false;

    if (!bathymetry_) {
        LOG_WARNING("No bathymetry data loaded, cannot compute error");
        result.num_elements = mesh_.num_elements();
        result.max_error = 0.0;
        result.mean_error = 0.0;
        result.converged = true;
        result.convergence_reason = "No bathymetry data";
        return result;
    }

    LOG_INFO("Starting adaptive refinement...");

    // Initialize error cache
    cached_errors_.clear();
    error_valid_.clear();

    for (iteration_ = 0; iteration_ < config_.max_iterations; ++iteration_) {
        // Use incremental error computation after first iteration
        std::vector<ElementError> errors;
        if (iteration_ == 0) {
            // First iteration: compute all errors (populates cache)
            errors = get_errors();
            cached_errors_ = errors;
            error_valid_.assign(errors.size(), true);
        } else {
            // Subsequent iterations: use cached errors with incremental updates
            errors = get_errors_cached();
        }

        Real max_err = max_error_from(errors);
        Real mean_err = mean_error_from(errors);

        LOG_INFO("Iteration " << iteration_ << ": elements=" << mesh_.num_elements() << ", max_error=" << max_err
                              << " m, mean_error=" << mean_err << " m");

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

    // Final error computation (use cached if available)
    auto final_errors = get_errors_cached();
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

bool LinearMeshGenerator::adapt_once(const std::vector<ElementError>& errors) {
    if (!bathymetry_ || !surface_) {
        return false;
    }

    // Select elements for refinement
    auto to_refine = select_for_refinement(errors);

    if (to_refine.empty()) {
        return false;
    }

    // Refine selected elements (stores last_new_elements_)
    refine_elements(to_refine);

    // Use incremental surface rebuild if we have new elements tracked
    if (!last_new_elements_.empty()) {
        rebuild_surface_incremental(last_new_elements_);
        invalidate_affected_errors(last_new_elements_);
    } else {
        rebuild_surface();
        // Invalidate all errors on full rebuild
        error_valid_.assign(mesh_.num_elements(), false);
    }

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

std::vector<ElementError> LinearMeshGenerator::get_errors() const {
    if (!surface_ || !bathymetry_) {
        return {};
    }

    // Use multi-source estimator for pixel-based metrics when multi_bathy_ is available
    if (multi_bathy_ &&
        (config_.error_metric == ErrorMetricType::PixelMaxError ||
         config_.error_metric == ErrorMetricType::PixelRMSE)) {
        auto estimator = std::make_unique<MultiSourcePixelMaxErrorEstimator>(
            *surface_, *bathymetry_, *multi_bathy_, mesh_);
        return estimator->estimate_all();
    }

    auto estimator = create_error_estimator(config_.error_metric, *surface_,
                                             *bathymetry_, mesh_, config_.ngauss);
    return estimator->estimate_all();
}

Real LinearMeshGenerator::max_error() const {
    if (!surface_ || !bathymetry_) {
        return 0.0;
    }
    auto estimator = create_error_estimator(config_.error_metric, *surface_,
                                             *bathymetry_, mesh_, config_.ngauss);
    return estimator->max_error();
}

Real LinearMeshGenerator::mean_error() const {
    if (!surface_ || !bathymetry_) {
        return 0.0;
    }
    auto estimator = create_error_estimator(config_.error_metric, *surface_,
                                             *bathymetry_, mesh_, config_.ngauss);
    return estimator->mean_error();
}

Real LinearMeshGenerator::max_error_from(const std::vector<ElementError>& errors) const {
    Real max_err = 0.0;
    for (const auto& err : errors) {
        if (!std::isnan(err.error)) {
            max_err = std::max(max_err, err.error);
        }
    }
    return max_err;
}

Real LinearMeshGenerator::mean_error_from(const std::vector<ElementError>& errors) const {
    if (errors.empty()) {
        return 0.0;
    }
    Real sum = 0.0;
    Index valid_count = 0;
    for (const auto& err : errors) {
        if (!std::isnan(err.error)) {
            sum += err.error;
            ++valid_count;
        }
    }
    return valid_count > 0 ? sum / static_cast<Real>(valid_count) : 0.0;
}

std::vector<Index> LinearMeshGenerator::select_for_refinement(
    const std::vector<ElementError>& errors) const {
    if (errors.empty()) {
        return {};
    }

    // Default minimum element size (from config)
    Real default_min_size = config_.min_element_size;
    if (config_.enforce_pixel_limit && bathymetry_ && default_min_size <= 0.0) {
        default_min_size = bathymetry_->min_element_size();
    }

    // Dorfler marking: select elements capturing theta fraction of total squared error
    std::vector<std::pair<Real, Index>> error_list;
    error_list.reserve(errors.size());

    Real total_sq = 0.0;
    for (const auto& err : errors) {
        if (!std::isnan(err.error)) {
            error_list.push_back({err.error, err.element});
            total_sq += err.error * err.error;
        }
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

        // Check minimum element size constraint (pixel resolution limit)
        if (config_.enforce_pixel_limit) {
            Real min_size = default_min_size;

            // For multi-source bathymetry, use per-element pixel limits with CRS conversion
            if (multi_bathy_) {
                const auto& bounds = mesh_.element_bounds(elem);
                Real cx = (bounds.xmin + bounds.xmax) / 2.0;
                Real cy = (bounds.ymin + bounds.ymax) / 2.0;

                Real source_min_size = multi_bathy_->get_min_element_size_meters(cx, cy);
                if (source_min_size > 0.0) {
                    min_size = source_min_size;
                }
            }

            if (min_size > 0.0) {
                auto size = mesh_.element_size(elem);
                Real elem_min_size = std::min(size(0), size(1));
                // Check if CHILDREN would be below pixel resolution
                // Refinement splits element in half, so children are elem_size/2
                if (elem_min_size < 2.0 * min_size) {
                    continue;  // Children would be below pixel resolution
                }
            }
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

    LOG_INFO("Refining " << elements_to_refine.size() << " elements");

    // Use QuadtreeAdapter's refine() method which handles:
    // - Splitting each element into 4 children
    // - Balancing for 2:1 constraint
    // - Rebuilding lookup structures
    auto result = mesh_.refine(elements_to_refine);

    // Store new element indices for incremental error/surface updates
    last_new_elements_ = std::move(result.new_elements);

    LOG_INFO("Refined " << result.num_refined << " elements, new total: " << mesh_.num_elements());
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

void LinearMeshGenerator::write_vtk(const std::string& filename,
                                     VTKWriterType writer_type) const {
    QuadtreeVTKWriter writer;
    if (surface_ && surface_->is_fitted()) {
        if (writer_type == VTKWriterType::Water) {
            if (!depth_func_) {
                throw std::runtime_error(
                    "LinearMeshGenerator::write_vtk: cannot use VTKWriterType::Water "
                    "without a depth function. Load bathymetry first.");
            }
            writer.write_water_only(filename, mesh_, *surface_, depth_func_);
        } else {
            writer.write(filename, mesh_, *surface_);
        }
    } else {
        writer.write_mesh_only(filename, mesh_);
    }
}

void LinearMeshGenerator::write_vtk_with_errors(const std::string& filename) const {
    QuadtreeVTKWriter writer;
    if (!surface_ || !surface_->is_fitted()) {
        throw std::runtime_error(
            "LinearMeshGenerator::write_vtk_with_errors: surface not fitted");
    }

    auto errors = get_errors();

    // Create source ID function if multi-source bathymetry is available
    std::function<int(Real, Real)> source_id_func;
    if (multi_bathy_) {
        source_id_func = [this](Real x, Real y) {
            return multi_bathy_->get_source_index(x, y);
        };
    }

    writer.write_with_errors(filename, mesh_, *surface_, errors, depth_func_, source_id_func);
}

} // namespace drifter
