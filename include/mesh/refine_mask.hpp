#pragma once

// Refinement mask utilities for directional (anisotropic) AMR
// Adapted from SeaMesh: /home/ole/Projects/SeaMesh/src/refine_mask_util.h

#include <cstdint>
#include <string>

namespace drifter {

/// @brief Bitmask for directional refinement
/// @details Allows independent refinement in X, Y, and Z directions
enum class RefineMask : uint8_t {
    NONE = 0,
    X = 1 << 0, // Refine in X direction
    Y = 1 << 1, // Refine in Y direction
    Z = 1 << 2, // Refine in Z direction
    XY = X | Y, // Refine in X and Y
    XZ = X | Z, // Refine in X and Z
    YZ = Y | Z, // Refine in Y and Z
    XYZ = X | Y | Z // Refine in all directions (isotropic)
};

// Bitwise operators for RefineMask
inline RefineMask operator|(RefineMask a, RefineMask b) {
    return static_cast<RefineMask>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline RefineMask operator&(RefineMask a, RefineMask b) {
    return static_cast<RefineMask>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

inline RefineMask &operator|=(RefineMask &a, RefineMask b) {
    a = a | b;
    return a;
}

inline RefineMask &operator&=(RefineMask &a, RefineMask b) {
    a = a & b;
    return a;
}

inline RefineMask operator~(RefineMask a) {
    return static_cast<RefineMask>(~static_cast<uint8_t>(a) & 0x07);
}

/// Check if a specific refinement flag is set
inline bool has_flag(RefineMask value, RefineMask flag) {
    return static_cast<uint8_t>(value & flag) != 0;
}

/// Check if mask refines in X direction
inline bool refines_x(RefineMask mask) { return has_flag(mask, RefineMask::X); }

/// Check if mask refines in Y direction
inline bool refines_y(RefineMask mask) { return has_flag(mask, RefineMask::Y); }

/// Check if mask refines in Z direction
inline bool refines_z(RefineMask mask) { return has_flag(mask, RefineMask::Z); }

/// Count the number of refinement directions
inline int refinement_count(RefineMask mask) {
    int count = 0;
    if (refines_x(mask))
        ++count;
    if (refines_y(mask))
        ++count;
    if (refines_z(mask))
        ++count;
    return count;
}

/// Number of children when refining with this mask
inline int num_children(RefineMask mask) { return 1 << refinement_count(mask); }

/// Convert mask to string representation
inline std::string to_string(RefineMask mask) {
    if (mask == RefineMask::NONE)
        return "NONE";
    std::string s;
    if (refines_x(mask))
        s += "X";
    if (refines_y(mask))
        s += "Y";
    if (refines_z(mask))
        s += "Z";
    return s;
}

/// @brief Per-axis refinement level (for directional AMR)
/// @details Each axis can have a different refinement level
struct DirectionalLevel {
    int level_x = 0;
    int level_y = 0;
    int level_z = 0;

    DirectionalLevel() = default;
    DirectionalLevel(int lx, int ly, int lz) : level_x(lx), level_y(ly), level_z(lz) {}

    /// Maximum level across all axes
    int max_level() const { return std::max({level_x, level_y, level_z}); }

    /// Minimum level across all axes
    int min_level() const { return std::min({level_x, level_y, level_z}); }

    /// Check if 2:1 balanced with another level (per-axis)
    bool is_balanced_with(const DirectionalLevel &other) const {
        return std::abs(level_x - other.level_x) <= 1 && std::abs(level_y - other.level_y) <= 1 &&
               std::abs(level_z - other.level_z) <= 1;
    }

    /// Get refinement mask needed to reach target level
    RefineMask mask_to_reach(const DirectionalLevel &target) const {
        RefineMask mask = RefineMask::NONE;
        if (level_x < target.level_x)
            mask |= RefineMask::X;
        if (level_y < target.level_y)
            mask |= RefineMask::Y;
        if (level_z < target.level_z)
            mask |= RefineMask::Z;
        return mask;
    }

    bool operator==(const DirectionalLevel &other) const {
        return level_x == other.level_x && level_y == other.level_y && level_z == other.level_z;
    }

    bool operator!=(const DirectionalLevel &other) const { return !(*this == other); }
};

} // namespace drifter
