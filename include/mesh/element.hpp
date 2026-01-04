#pragma once

#include "core/types.hpp"
#include <array>
#include <vector>

namespace drifter {

// Forward declarations
class Mesh;

// Element refinement level for AMR
using Level = std::uint8_t;
constexpr Level MAX_REFINEMENT_LEVEL = 16;

// Element structure for DG discretization
struct Element {
    Index id;                          // Global element ID
    ElementType type;                  // Element geometry type
    Level level;                       // AMR refinement level
    std::uint8_t polynomial_order;     // DG polynomial order (p-adaptivity)

    std::vector<Index> node_ids;       // Vertex node indices
    std::vector<Index> neighbor_ids;   // Neighboring element indices
    std::vector<Index> face_ids;       // Face/edge indices

    // Parent/child relationships for AMR
    Index parent_id = -1;
    std::vector<Index> child_ids;

    // Geometric data (cached for performance)
    Vec3 centroid;
    Real volume;
    Real characteristic_length;        // For CFL computation
    Mat3 jacobian;                     // Reference to physical mapping
    Real jacobian_det;

    // Bathymetry at element (can be high-order representation)
    VecX bathymetry_coeffs;

    // Solution DOFs stored per element (modal coefficients)
    MatX solution;                     // (num_modes x num_vars)

    bool is_leaf() const { return child_ids.empty(); }
    bool is_boundary() const;
    int num_faces() const;
};

// Face structure for flux computation
struct Face {
    Index id;
    Index left_element;                // Element on left side
    Index right_element;               // Element on right side (-1 if boundary)

    std::vector<Index> node_ids;       // Vertices defining the face
    Vec3 normal;                       // Outward normal (from left element)
    Real area;
    Vec3 centroid;

    // Boundary condition info
    bool is_boundary = false;
    int boundary_tag = 0;              // For different BC types

    // Mortar data for non-conforming interfaces (AMR)
    bool is_mortar = false;
    std::vector<Index> mortar_faces;   // Sub-faces for hanging nodes
};

} // namespace drifter
