#include "core/lowrider.hpp"
#include "bathymetry/linear_mesh_generator.hpp"
#include "core/logger.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>

namespace drifter {

Lowrider::Lowrider(const LowriderConfig& config) : config_(config) {}

bool Lowrider::data_files_exist() const {
    // Check primary bathymetry file
    if (!config_.data.primary_file.empty()) {
        std::string primary_path = config_.data.data_dir + config_.data.primary_file;
        if (!std::filesystem::exists(primary_path)) {
            LOG_ERROR("Primary bathymetry file not found: " << primary_path);
            return false;
        }

        // Check tile files
        for (const auto& tile : config_.data.tile_files) {
            std::string tile_path = config_.data.data_dir + tile;
            if (!std::filesystem::exists(tile_path)) {
                LOG_ERROR("Tile file not found: " << tile_path);
                return false;
            }
        }
    }

    // Check coastline file if configured
    if (config_.coastline.enabled()) {
        if (!std::filesystem::exists(config_.coastline.file)) {
            LOG_ERROR("Coastline file not found: " << config_.coastline.file);
            return false;
        }
    }

    return true;
}

int Lowrider::run() {
    // Check data files
    if (!data_files_exist()) {
        LOG_ERROR("Required data files not found. Exiting.");
        return 1;
    }

    LOG_INFO("=== Adaptive Linear Mesh Generation ===");
    LOG_INFO("Domain: [" << config_.domain.xmin << ", " << config_.domain.xmax << "] x [" << config_.domain.ymin
                         << ", " << config_.domain.ymax << "]");

    // Create mesh generator
    LinearMeshGenerator generator(
        config_.domain.xmin, config_.domain.xmax,
        config_.domain.ymin, config_.domain.ymax,
        config_.domain.initial_nx, config_.domain.initial_ny,
        config_.refinement
    );

    // Load bathymetry if provided
    if (!config_.data.primary_file.empty()) {
        generator.load_bathymetry(config_.data);
    }

    // Load coastline data (if configured)
    if (config_.coastline.enabled()) {
        generator.load_coastline(config_.coastline);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Stage 1: Coastline refinement (if configured)
    if (config_.coastline.enabled()) {
        LOG_INFO("Stage 1: Coastline refinement...");
        int coast_iters = generator.refine_coastline();
        LOG_INFO("Coastline refinement: " << coast_iters << " iterations, " << generator.mesh().num_elements()
                                          << " elements");
    }

    // Stage 2: Error-driven seabed refinement
    LOG_INFO("Stage 2: Seabed refinement...");
    auto result = generator.solve_adaptive();

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Print results
    LOG_INFO("Final result:");
    LOG_INFO("  Elements: " << result.num_elements);
    LOG_INFO("  Max error: " << result.max_error << " m");
    LOG_INFO("  Mean error: " << result.mean_error << " m");
    LOG_INFO("  Iterations: " << result.iterations);
    LOG_INFO("  Converged: " << (result.converged ? "yes" : "no"));
    if (!result.convergence_reason.empty()) {
        LOG_INFO("  Reason: " << result.convergence_reason);
    }
    LOG_INFO("  Time: " << time_ms << " ms");

    // Compute refinement statistics
    int max_level = 0;
    for (Index i = 0; i < generator.mesh().num_elements(); ++i) {
        max_level = std::max(max_level, generator.mesh().element_level(i).max_level());
    }
    Real domain_size = std::max(config_.domain.xmax - config_.domain.xmin,
                                 config_.domain.ymax - config_.domain.ymin);
    Real element_size_min = domain_size / std::pow(2.0, max_level);

    LOG_INFO("Number of levels         : " << max_level);
    LOG_INFO("Size of smallest element : " << element_size_min);

    // Write VTK output
    generator.write_vtk(config_.output.vtk_file, config_.output.vtk_writer_type);
    LOG_INFO("Output written to        : " << config_.output.vtk_file << ".vtu");

    // Write VTK with per-element error and depth for visualization
    std::string error_vtk_file = config_.output.vtk_file + "_errors";
    generator.write_vtk_with_errors(error_vtk_file);

    LOG_INFO("Mesh generation complete.");
    return 0;
}

} // namespace drifter
