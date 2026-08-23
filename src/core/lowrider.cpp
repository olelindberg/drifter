#include "core/lowrider.hpp"
#include "bathymetry/linear_mesh_generator.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace drifter {

Lowrider::Lowrider(const LowriderConfig& config) : config_(config) {}

bool Lowrider::data_files_exist() const {
    // Check primary bathymetry file
    if (!config_.data.primary_file.empty()) {
        std::string primary_path = config_.data.data_dir + config_.data.primary_file;
        if (!std::filesystem::exists(primary_path)) {
            std::cerr << "Primary bathymetry file not found: " << primary_path << std::endl;
            return false;
        }

        // Check tile files
        for (const auto& tile : config_.data.tile_files) {
            std::string tile_path = config_.data.data_dir + tile;
            if (!std::filesystem::exists(tile_path)) {
                std::cerr << "Tile file not found: " << tile_path << std::endl;
                return false;
            }
        }
    }
    return true;
}

int Lowrider::run() {
    // Check data files
    if (!data_files_exist()) {
        std::cerr << "\nRequired data files not found. Exiting.\n";
        return 1;
    }

    std::cout << "\n=== Adaptive Linear Mesh Generation ===" << std::endl;
    std::cout << "Domain: [" << config_.domain.xmin << ", " << config_.domain.xmax << "] x ["
              << config_.domain.ymin << ", " << config_.domain.ymax << "]" << std::endl;

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

    // Run adaptive refinement
    auto start = std::chrono::high_resolution_clock::now();
    auto result = generator.solve_adaptive();
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Print results
    std::cout << "\nFinal result:" << std::endl;
    std::cout << "  Elements: " << result.num_elements << std::endl;
    std::cout << "  Max error: " << result.max_error << " m" << std::endl;
    std::cout << "  Mean error: " << result.mean_error << " m" << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Converged: " << (result.converged ? "yes" : "no") << std::endl;
    if (!result.convergence_reason.empty()) {
        std::cout << "  Reason: " << result.convergence_reason << std::endl;
    }
    std::cout << "  Time: " << time_ms << " ms" << std::endl;

    // Compute refinement statistics
    int max_level = 0;
    for (Index i = 0; i < generator.mesh().num_elements(); ++i) {
        max_level = std::max(max_level, generator.mesh().element_level(i).max_level());
    }
    Real domain_size = std::max(config_.domain.xmax - config_.domain.xmin,
                                 config_.domain.ymax - config_.domain.ymin);
    Real element_size_min = domain_size / std::pow(2.0, max_level);

    std::cout << "Number of levels         : " << max_level << std::endl;
    std::cout << "Size of smallest element : " << element_size_min << std::endl;

    // Write VTK output
    generator.write_vtk(config_.output.vtk_file, config_.output.vtk_writer_type);
    std::cout << "Output written to        : " << config_.output.vtk_file << ".vtu" << std::endl;

    std::cout << "\nMesh generation complete.\n";
    return 0;
}

} // namespace drifter
