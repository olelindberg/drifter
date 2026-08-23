#include "core/lowrider_config_reader.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace drifter {

LowriderConfig LowriderConfigReader::load(const std::string &path) {
  namespace pt = boost::property_tree;

  pt::ptree tree;
  try {
    pt::read_json(path, tree);
  } catch (const pt::json_parser_error &e) {
    throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
  }

  LowriderConfig config;

    // Data configuration (drifter-style: data_dir + primary_file + tile_files)
  if (tree.count("data")) {
    const auto &data         = tree.get_child("data");
    config.data.data_dir     = data.get<std::string>("data_dir", "");
    config.data.primary_file = data.get<std::string>("primary_file", "");

        // Parse tile_files array
    if (data.count("tile_files")) {
      for (const auto &tile : data.get_child("tile_files")) {
        config.data.tile_files.push_back(tile.second.get_value<std::string>());
      }
    }
  }

    // Domain configuration
  if (tree.count("domain")) {
    const auto &domain       = tree.get_child("domain");
    config.domain.xmin       = domain.get<Real>("xmin", 0.0);
    config.domain.xmax       = domain.get<Real>("xmax", 1000.0);
    config.domain.ymin       = domain.get<Real>("ymin", 0.0);
    config.domain.ymax       = domain.get<Real>("ymax", 1000.0);
    config.domain.initial_nx = domain.get<int>("initial_nx", 4);
    config.domain.initial_ny = domain.get<int>("initial_ny", 4);
  }

    // Refinement configuration
  if (tree.count("refinement")) {
    const auto &ref                   = tree.get_child("refinement");
    config.refinement.error_threshold = ref.get<Real>("error_threshold", 0.5);
    config.refinement.max_iterations  = ref.get<int>("max_iterations", 10);
    config.refinement.max_elements    = ref.get<int>("max_elements", 10000);
    config.refinement.max_level       = ref.get<int>("max_level", 10);
    config.refinement.dorfler_theta   = ref.get<Real>("dorfler_theta", 0.5);
    config.refinement.ngauss          = ref.get<int>("ngauss", 4);

        // Parse error metric
    std::string metric_str = ref.get<std::string>("error_metric", "normalized_error");
    if (metric_str == "normalized_error") {
      config.refinement.error_metric = ErrorMetricType::NormalizedError;
    } else if (metric_str == "mean_difference") {
      config.refinement.error_metric = ErrorMetricType::MeanDifference;
    } else if (metric_str == "volume_error") {
      config.refinement.error_metric = ErrorMetricType::VolumeChange;
    } else if (metric_str == "pixel_rmse") {
      config.refinement.error_metric = ErrorMetricType::PixelRMSE;
    } else if (metric_str == "pixel_max_error") {
      config.refinement.error_metric = ErrorMetricType::PixelMaxError;
    } else {
      throw std::runtime_error("Unknown error_metric: '" + metric_str + "'. Valid values: normalized_error, mean_difference, volume_change, pixel_rmse, pixel_max_error");
    }
  }

    // Output configuration
  if (tree.count("output")) {
    const auto &out                   = tree.get_child("output");
    config.output.vtk_file            = out.get<std::string>("vtk_file", "lowrider_mesh");
    config.output.write_per_iteration = out.get<bool>("write_per_iteration", false);

    // Parse VTK writer type
    if (out.count("vtk_writer_type")) {
      std::string type_str = out.get<std::string>("vtk_writer_type");
      if (type_str == "all") {
        config.output.vtk_writer_type = VTKWriterType::All;
      } else if (type_str == "water") {
        config.output.vtk_writer_type = VTKWriterType::Water;
      } else {
        throw std::runtime_error(
            "Invalid vtk_writer_type '" + type_str + "'. "
            "Valid options: 'all' (all elements), 'water' (water elements only)");
      }
    }
  }

  config.verbose = tree.get<bool>("verbose", true);

  return config;
}

void print_lowrider_config(const LowriderConfig &config) {
  std::cout << "\nConfiguration:\n";

    // Data sources
  if (!config.data.primary_file.empty()) {
    std::cout << "  Data directory: " << config.data.data_dir << "\n";
    std::cout << "  Primary file: " << config.data.primary_file << "\n";
    std::cout << "  Tile files: " << config.data.tile_files.size() << " tiles\n";
  } else {
    std::cout << "  Data: (none)\n";
  }

  std::cout << "  Domain: [" << config.domain.xmin << ", " << config.domain.xmax << "] x [" << config.domain.ymin << ", " << config.domain.ymax << "]\n";
  std::cout << "  Initial mesh: " << config.domain.initial_nx << " x " << config.domain.initial_ny << "\n";
  std::cout << "  Error threshold: " << config.refinement.error_threshold << " m\n";
  std::cout << "  Max iterations: " << config.refinement.max_iterations << "\n";
  std::cout << "  Max elements: " << config.refinement.max_elements << "\n";
  std::cout << "  Max level: " << config.refinement.max_level << "\n";
  std::cout << "  Dorfler theta: " << config.refinement.dorfler_theta << "\n";
  std::cout << "  Output: " << config.output.vtk_file << "\n";
  std::cout << "  VTK writer type: "
            << (config.output.vtk_writer_type == VTKWriterType::Water ? "water" : "all")
            << "\n";
  std::cout << std::endl;
}

} // namespace drifter
