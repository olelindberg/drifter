#pragma once

// Morton code utilities for 3D spatial indexing
// Adapted from SeaMesh: /home/ole/Projects/SeaMesh/src/Morton3D.h and
// MortonUtil.h

#include <cstdint>

namespace drifter {

/// @brief Morton3D provides static methods for encoding and decoding 3D Morton
/// codes (Z-order curve).
/// @details Morton codes interleave the bits of the x, y, and z coordinates to
/// produce a single linear index. This is useful for spatial indexing and
/// hierarchical data structures like octrees. The implementation supports up to
/// 21 bits per coordinate, allowing for grids of size up to 2^21 in each
/// dimension.
class Morton3D {
public:
    /// Encode 3D grid indices (x, y, z) into a 64-bit Morton code
    static inline uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
        return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
    }

    /// Decode a Morton code back to 3D grid indices
    static inline void
    decode(uint64_t code, uint32_t &x, uint32_t &y, uint32_t &z) {
        x = compact1by2(code);
        y = compact1by2(code >> 1);
        z = compact1by2(code >> 2);
    }

    // Bit masks for x, y, z components in Morton code
    static constexpr uint64_t X_MASK = 0x1249249249249249ULL; // 0b001001001...
    static constexpr uint64_t Y_MASK = 0x2492492492492492ULL; // 0b010010010...
    static constexpr uint64_t Z_MASK = 0x4924924924924924ULL; // 0b100100100...

private:
    /// Spread bits of a 21-bit integer into every third bit (for 3D
    /// interleaving)
    static inline uint64_t part1by2(uint64_t n) {
        n &= 0x1fffff; // Keep only 21 bits
        n = (n | (n << 32)) & 0x1f00000000ffffULL;
        n = (n | (n << 16)) & 0x1f0000ff0000ffULL;
        n = (n | (n << 8)) & 0x100f00f00f00f00fULL;
        n = (n | (n << 4)) & 0x10c30c30c30c30c3ULL;
        n = (n | (n << 2)) & 0x1249249249249249ULL;
        return n;
    }

    /// Compact every third bit back into a contiguous integer
    static inline uint32_t compact1by2(uint64_t n) {
        n &= 0x1249249249249249ULL;
        n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3ULL;
        n = (n ^ (n >> 4)) & 0x100f00f00f00f00fULL;
        n = (n ^ (n >> 8)) & 0x1f0000ff0000ffULL;
        n = (n ^ (n >> 16)) & 0x1f00000000ffffULL;
        n = (n ^ (n >> 32)) & 0x1fffffULL;
        return static_cast<uint32_t>(n);
    }
};

/// @brief Morton utilities for directional (anisotropic) octree operations
/// @details Supports per-axis parent/child navigation for directional AMR
class MortonUtil {
public:
    /// Face directions for neighbor queries
    enum class Face : uint8_t {
        X_PLUS = 0,
        X_MINUS = 1,
        Y_PLUS = 2,
        Y_MINUS = 3,
        Z_PLUS = 4,
        Z_MINUS = 5
    };

    /// Get parent code in X direction only (for anisotropic refinement)
    static inline uint64_t parent_x(uint64_t code) {
        uint64_t xbits = code & Morton3D::X_MASK;
        xbits >>= 3; // Remove one x bit (spaced by 3 bits)
        code &= ~Morton3D::X_MASK;
        code |= (xbits & Morton3D::X_MASK);
        return code;
    }

    /// Get parent code in Y direction only
    static inline uint64_t parent_y(uint64_t code) {
        uint64_t ybits = code & Morton3D::Y_MASK;
        ybits >>= 3;
        code &= ~Morton3D::Y_MASK;
        code |= (ybits & Morton3D::Y_MASK);
        return code;
    }

    /// Get parent code in Z direction only
    static inline uint64_t parent_z(uint64_t code) {
        uint64_t zbits = code & Morton3D::Z_MASK;
        zbits >>= 3;
        code &= ~Morton3D::Z_MASK;
        code |= (zbits & Morton3D::Z_MASK);
        return code;
    }

    /// Get child code in X direction (bit: 0 = left, 1 = right)
    static inline uint64_t child_x(uint64_t code, int bit) {
        uint64_t xbits = code & Morton3D::X_MASK;
        xbits = (xbits << 3) | (bit ? 0x1ULL : 0x0ULL);
        code &= ~Morton3D::X_MASK;
        code |= (xbits & Morton3D::X_MASK);
        return code;
    }

    /// Get child code in Y direction
    static inline uint64_t child_y(uint64_t code, int bit) {
        uint64_t ybits = code & Morton3D::Y_MASK;
        ybits = (ybits << 3) | (bit ? 0x2ULL : 0x0ULL);
        code &= ~Morton3D::Y_MASK;
        code |= (ybits & Morton3D::Y_MASK);
        return code;
    }

    /// Get child code in Z direction
    static inline uint64_t child_z(uint64_t code, int bit) {
        uint64_t zbits = code & Morton3D::Z_MASK;
        zbits = (zbits << 3) | (bit ? 0x4ULL : 0x0ULL);
        code &= ~Morton3D::Z_MASK;
        code |= (zbits & Morton3D::Z_MASK);
        return code;
    }

    /// Refine code in specified directions (for directional AMR)
    /// @param code Current Morton code
    /// @param ix, iy, iz Child index in each direction (0 or 1)
    /// @param refine_x, refine_y, refine_z Whether to refine in each direction
    static inline uint64_t refine(
        uint64_t code, int ix, int iy, int iz, bool refine_x, bool refine_y,
        bool refine_z) {
        if (refine_x)
            code = child_x(code, ix);
        if (refine_y)
            code = child_y(code, iy);
        if (refine_z)
            code = child_z(code, iz);
        return code;
    }

    /// Get standard (isotropic) parent code
    static inline uint64_t parent(uint64_t code) { return code >> 3; }

    /// Get standard (isotropic) child code
    static inline uint64_t child(uint64_t parent_code, int child_id) {
        return (parent_code << 3) | (child_id & 0x7);
    }
};

} // namespace drifter
