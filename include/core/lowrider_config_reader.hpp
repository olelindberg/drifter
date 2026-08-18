#pragma once

/// @file lowrider_config_reader.hpp
/// @brief JSON configuration file reader for Lowrider

#include "core/lowrider_config.hpp"
#include <string>

namespace drifter {

/// @brief Configuration file reader for Lowrider
class LowriderConfigReader {
public:
    /// @brief Load configuration from JSON file
    /// @param path Path to JSON configuration file
    /// @return Loaded configuration
    /// @throws std::runtime_error if file cannot be read or parsed
    static LowriderConfig load(const std::string& path);
};

} // namespace drifter
