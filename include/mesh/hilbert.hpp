#pragma once

// 2D Hilbert curve utilities for spatial indexing
// Better cache locality than Morton Z-curve for iterative solvers

#include <cstdint>

namespace drifter {

/// @brief Hilbert2D provides static methods for encoding and decoding 2D Hilbert
/// curve indices.
/// @details Hilbert curves have better spatial locality than Morton (Z-order)
/// curves, which improves cache performance for sparse matrix operations.
/// The implementation uses the standard rotation-based algorithm.
class Hilbert2D {
public:
    /// @brief Encode 2D grid coordinates into a Hilbert curve index
    /// @param x X coordinate (must be < 2^order)
    /// @param y Y coordinate (must be < 2^order)
    /// @param order Hilbert curve order (grid size = 2^order x 2^order)
    /// @return Hilbert index in range [0, 4^order - 1]
    static inline uint64_t encode(uint32_t x, uint32_t y, int order = 10) {
        uint64_t d = 0;
        for (int s = (1 << (order - 1)); s > 0; s >>= 1) {
            int rx = (x & s) > 0 ? 1 : 0;
            int ry = (y & s) > 0 ? 1 : 0;
            d += static_cast<uint64_t>(s) * s * ((3 * rx) ^ ry);
            rotate(s, x, y, rx, ry);
        }
        return d;
    }

    /// @brief Decode a Hilbert index back to 2D grid coordinates
    /// @param d Hilbert curve index
    /// @param order Hilbert curve order
    /// @param x Output X coordinate
    /// @param y Output Y coordinate
    static inline void decode(uint64_t d, int order, uint32_t &x, uint32_t &y) {
        x = y = 0;
        for (int s = 1; s < (1 << order); s <<= 1) {
            int rx = 1 & (d / 2);
            int ry = 1 & (d ^ rx);
            rotate(s, x, y, rx, ry);
            x += s * rx;
            y += s * ry;
            d /= 4;
        }
    }

    /// @brief Maximum supported order (21 bits per coordinate, like Morton3D)
    static constexpr int MAX_ORDER = 21;

private:
    /// @brief Rotate/flip a quadrant appropriately for Hilbert curve
    static inline void rotate(int n, uint32_t &x, uint32_t &y, int rx, int ry) {
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            // Swap x and y
            uint32_t t = x;
            x = y;
            y = t;
        }
    }
};

} // namespace drifter
