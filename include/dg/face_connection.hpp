#pragma once

// Face connection types for non-conforming interfaces in directional AMR
// Handles 1:1, 1:2, 1:3, and 1:4 face connections

#include "core/types.hpp"
#include "mesh/refine_mask.hpp"
#include <vector>

namespace drifter {

/// @brief Type of face connection between neighboring elements
/// @details With directional (anisotropic) AMR, a coarse face can connect to 1-4 fine faces
enum class FaceConnectionType : uint8_t {
    /// Conforming 1:1 connection (same level)
    SameLevel,

    /// 1:2 connection: refined in first tangent direction only
    /// For face normal to X: refined in Y, not Z (2 fine faces stacked in Y)
    /// For face normal to Y: refined in X, not Z (2 fine faces stacked in X)
    /// For face normal to Z: refined in X, not Y (2 fine faces stacked in X)
    Fine2x1,

    /// 1:2 connection: refined in second tangent direction only
    /// For face normal to X: refined in Z, not Y (2 fine faces stacked in Z)
    /// For face normal to Y: refined in Z, not X (2 fine faces stacked in Z)
    /// For face normal to Z: refined in Y, not X (2 fine faces stacked in Y)
    Fine1x2,

    /// 1:3 connection: asymmetric (2+1 pattern)
    /// One tangent direction has 2 fine elements, the other has 1
    /// Total 3 fine faces in an L-shaped arrangement
    Fine3_2plus1,

    /// 1:3 connection: asymmetric (1+2 pattern)
    /// Transpose of Fine3_2plus1
    Fine3_1plus2,

    /// 1:4 connection: refined in both tangent directions
    /// 4 fine faces in a 2x2 arrangement
    Fine2x2,

    /// Boundary face (no neighbor)
    Boundary
};

/// @brief Describes how a coarse face connects to fine face(s)
/// @details Used for mortar element flux computation at non-conforming interfaces
struct FaceConnection {
    /// Type of connection
    FaceConnectionType type = FaceConnectionType::Boundary;

    /// Coarse element index
    Index coarse_elem = -1;

    /// Face ID on coarse element (0-5 for hex)
    int coarse_face_id = -1;

    /// Fine element indices (1, 2, 3, or 4 elements)
    std::vector<Index> fine_elems;

    /// Face IDs on each fine element
    std::vector<int> fine_face_ids;

    /// Sub-face index for each fine element within the coarse face
    /// For 1:4 (2x2): indices 0-3 in row-major order
    /// For 1:2: indices 0-1
    /// For 1:3: indices 0-2
    std::vector<int> subface_indices;

    /// Boundary marker ID (only valid if type == Boundary)
    int boundary_id = -1;

    /// Number of fine faces
    int num_fine_faces() const {
        switch (type) {
            case FaceConnectionType::SameLevel: return 1;
            case FaceConnectionType::Fine2x1:
            case FaceConnectionType::Fine1x2: return 2;
            case FaceConnectionType::Fine3_2plus1:
            case FaceConnectionType::Fine3_1plus2: return 3;
            case FaceConnectionType::Fine2x2: return 4;
            case FaceConnectionType::Boundary: return 0;
            default: return 0;
        }
    }

    /// Check if this is a conforming (1:1) connection
    bool is_conforming() const {
        return type == FaceConnectionType::SameLevel;
    }

    /// Check if this is a boundary face
    bool is_boundary() const {
        return type == FaceConnectionType::Boundary;
    }

    /// Check if this requires mortar element treatment
    bool needs_mortar() const {
        return !is_conforming() && !is_boundary();
    }
};

/// @brief Get the face connection type from refinement levels
/// @param coarse_level Directional level of coarse element
/// @param fine_levels Directional levels of fine elements at face
/// @param face_id Face ID (0-5) to determine tangent directions
inline FaceConnectionType get_connection_type(
    const DirectionalLevel& coarse_level,
    const DirectionalLevel& fine_level,
    int face_id)
{
    // Determine tangent directions for this face
    // Face 0,1 (X normal): tangent dirs are Y, Z
    // Face 2,3 (Y normal): tangent dirs are X, Z
    // Face 4,5 (Z normal): tangent dirs are X, Y

    int diff_t1 = 0, diff_t2 = 0;

    switch (face_id) {
        case 0: case 1:  // X-normal face
            diff_t1 = fine_level.level_y - coarse_level.level_y;
            diff_t2 = fine_level.level_z - coarse_level.level_z;
            break;
        case 2: case 3:  // Y-normal face
            diff_t1 = fine_level.level_x - coarse_level.level_x;
            diff_t2 = fine_level.level_z - coarse_level.level_z;
            break;
        case 4: case 5:  // Z-normal face
            diff_t1 = fine_level.level_x - coarse_level.level_x;
            diff_t2 = fine_level.level_y - coarse_level.level_y;
            break;
    }

    // Determine connection type based on level differences
    if (diff_t1 == 0 && diff_t2 == 0) {
        return FaceConnectionType::SameLevel;
    } else if (diff_t1 == 1 && diff_t2 == 0) {
        return FaceConnectionType::Fine2x1;
    } else if (diff_t1 == 0 && diff_t2 == 1) {
        return FaceConnectionType::Fine1x2;
    } else if (diff_t1 == 1 && diff_t2 == 1) {
        return FaceConnectionType::Fine2x2;
    } else {
        // For 1:3 cases, we need additional logic based on which neighbors exist
        // This requires examining the actual neighbor structure
        // For now, return 2x2 as a conservative fallback
        return FaceConnectionType::Fine2x2;
    }
}

/// @brief Get tangent directions for a face
/// @param face_id Face ID (0-5)
/// @return Pair of axis indices (0=X, 1=Y, 2=Z) for tangent directions
inline std::pair<int, int> get_face_tangent_axes(int face_id) {
    switch (face_id) {
        case 0: case 1: return {1, 2};  // Y, Z for X-normal face
        case 2: case 3: return {0, 2};  // X, Z for Y-normal face
        case 4: case 5: return {0, 1};  // X, Y for Z-normal face
        default: return {0, 1};
    }
}

/// @brief Get normal axis for a face
/// @param face_id Face ID (0-5)
/// @return Axis index (0=X, 1=Y, 2=Z)
inline int get_face_normal_axis(int face_id) {
    switch (face_id) {
        case 0: case 1: return 0;  // X
        case 2: case 3: return 1;  // Y
        case 4: case 5: return 2;  // Z
        default: return 0;
    }
}

/// @brief Check if face is on positive side
/// @param face_id Face ID (0-5)
/// @return true if positive direction (1, 3, 5), false if negative (0, 2, 4)
inline bool is_positive_face(int face_id) {
    return (face_id % 2) == 1;
}

}  // namespace drifter
