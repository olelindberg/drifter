#include "core/config_reader.hpp"
#include "core/drifter.hpp"
#include "core/logger.hpp"
#include <iostream>
#include <string>

using namespace drifter;

int main(int argc, char *argv[]) {
  LOG_INFO("====================================");
  LOG_INFO("  HIGHRIDER - Coastal Ocean Model");
  LOG_INFO("  Adaptive Bathymetry Smoother");
  LOG_INFO("====================================");

  // Parse command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config.json>\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  " << argv[0] << " config/highrider_example.json\n";
    return 1;
  }

  std::string config_path = argv[1];

  // Load configuration
  DrifterConfig config;
  try {
    LOG_INFO("Loading configuration from: " << config_path);
    config = ConfigReader::load(config_path);
  } catch (const std::exception &e) {
    LOG_ERROR("Error loading config: " << e.what());
    return 1;
  }

  // Print configuration
  print_config(config);

  // Run application
  Drifter app(config);
  return app.run();
}
