#pragma once

/// @file lowrider.hpp
/// @brief Main Lowrider application class

#include "core/lowrider_config.hpp"

namespace drifter {

/// @brief Main Lowrider application
class Lowrider {
public:
    /// @brief Construct with configuration
    explicit Lowrider(const LowriderConfig& config);

    /// @brief Run the adaptive mesh generation
    /// @return 0 on success, non-zero on failure
    int run();

private:
    LowriderConfig config_;

    /// @brief Check if required data files exist
    bool data_files_exist() const;
};

} // namespace drifter
