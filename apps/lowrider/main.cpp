#include "core/lowrider_config_reader.hpp"
#include "core/lowrider.hpp"
#include <iostream>
#include <string>

using namespace drifter;

int main(int argc, char* argv[]) {
    std::cout << "====================================\n";
    std::cout << "  LOWRIDER - Seabed Mesh Generator\n";
    std::cout << "  Adaptive Linear Elements\n";
    std::cout << "====================================\n\n";

    // Parse command line
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " config/lowrider_example.json\n";
        return 1;
    }

    std::string config_path = argv[1];

    // Load configuration
    LowriderConfig config;
    try {
        std::cout << "Loading configuration from: " << config_path << std::endl;
        config = LowriderConfigReader::load(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    // Print configuration
    print_lowrider_config(config);

    // Run application
    Lowrider app(config);
    return app.run();
}
