#include "core/config_reader.hpp"
#include "core/drifter.hpp"
#include <iostream>
#include <string>

using namespace drifter;

int main(int argc, char *argv[]) {
  std::cout << "====================================\n";
  std::cout << "  DRIFTER - Coastal Ocean Model\n";
  std::cout << "  Adaptive Bathymetry Smoother\n";
  std::cout << "====================================\n\n";

  // Parse command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config.json>\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  " << argv[0] << " config/example.json\n";
    return 1;
  }

  std::string config_path = argv[1];

  // Load configuration
  DrifterConfig config;
  try {
    std::cout << "Loading configuration from: " << config_path << std::endl;
    config = ConfigReader::load(config_path);
  } catch (const std::exception &e) {
    std::cerr << "Error loading config: " << e.what() << "\n";
    return 1;
  }

  // Print configuration
  print_config(config);

  // Run application
  Drifter app(config);
  return app.run();
}
