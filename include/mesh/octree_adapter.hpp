#pragma once

// Octree adapter for SeaMesh integration
// Adapts SeaMesh's DirectionalAdaptiveOctree for DG elements
//
// SeaMesh provides:
// - Directional (anisotropic) AMR with per-axis refinement levels
// - Morton code based spatial indexing
// - 2:1 balanced octree
// - Neighbor finding via lookup tables

#include "core/types.hpp"
#include "dg/face_connection.hpp"
#include "mesh/morton.hpp"
#include "mesh/refine_mask.hpp"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace drifter {

/// @brief Physical bounds of a hexahedral element
struct ElementBounds {
    Real xmin, xmax;
    Real ymin, ymax;
    Real zmin, zmax;

    /// Element center
    Vec3 center() const {
        return Vec3(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax));
    }

    /// Element size in each direction
    Vec3 size() const { return Vec3(xmax - xmin, ymax - ymin, zmax - zmin); }

    /// Check if point is inside bounds
    bool contains(const Vec3 &p) const {
        return p(0) >= xmin && p(0) <= xmax && p(1) >= ymin && p(1) <= ymax && p(2) >= zmin &&
               p(2) <= zmax;
    }
};

/// @brief Octree node adapted from SeaMesh's DirectionalAdaptiveOctree
/// @details Represents a single cell in the adaptive octree with directional
/// refinement
struct OctreeNode {
    /// Morton code for spatial ordering
    uint64_t morton = 0;

    /// Per-axis refinement levels
    DirectionalLevel level;

    /// Physical bounds
    ElementBounds bounds;

    /// Refinement mask (which axes to refine)
    RefineMask mask = RefineMask::NONE;

    /// Children (1-8 depending on refinement mask)
    std::vector<std::unique_ptr<OctreeNode>> children;

    /// Parent pointer (nullptr for root)
    OctreeNode* parent = nullptr;

    /// Local index within octree (leaf numbering)
    Index leaf_index = -1;

    /// Check if this is a leaf node
    bool is_leaf() const { return children.empty(); }

    /// Number of children (1, 2, 4, or 8)
    int num_children() const { return static_cast<int>(children.size()); }

    /// Get the global element index (valid only for leaves)
    Index element_index() const { return leaf_index; }
};

/// @brief Neighbor information for a face
struct NeighborInfo {
    /// Type of face connection
    FaceConnectionType type = FaceConnectionType::Boundary;

    /// Neighbor element indices (1, 2, 3, or 4 for non-conforming)
    std::vector<Index> neighbor_elements;

    /// Face IDs on neighbors (opposite face for conforming)
    std::vector<int> neighbor_faces;

    /// Sub-face indices for each neighbor
    std::vector<int> subface_indices;

    /// Is this a boundary face?
    bool is_boundary() const { return type == FaceConnectionType::Boundary; }

    /// Is this a conforming interface?
    bool is_conforming() const { return type == FaceConnectionType::SameLevel; }
};

/// @brief Adapter for SeaMesh directional adaptive octree
/// @details Provides DG-friendly interface to the octree mesh
class OctreeAdapter {
public:
    /// @brief Construct adapter with domain bounds
    /// @param xmin, xmax X bounds
    /// @param ymin, ymax Y bounds
    /// @param zmin, zmax Z bounds (sigma coordinate: -1 to 0 typically)
    OctreeAdapter(Real xmin, Real xmax, Real ymin, Real ymax, Real zmin, Real zmax);

    /// @brief Build initial uniform grid
    /// @param nx, ny, nz Number of elements in each direction
    void build_uniform(int nx, int ny, int nz);

    /// @brief Build from refinement function
    /// @param refine_func Function returning true if element should be refined
    /// @param max_level Maximum refinement level per axis
    void build_adaptive(const std::function<bool(const ElementBounds &)> &refine_func,
                        int max_level_x, int max_level_y, int max_level_z);

    /// @brief Balance the octree for 2:1 constraint
    /// @details Ensures adjacent elements differ by at most 1 level per axis
    void balance();

    /// @brief Refine specific elements
    /// @param elements Elements to refine
    /// @param masks Refinement mask for each element
    void refine(const std::vector<Index> &elements, const std::vector<RefineMask> &masks);

    /// @brief Coarsen specific elements (group siblings back to parent)
    /// @param parent_elements Parent elements whose children should merge
    void coarsen(const std::vector<Index> &parent_elements);

    // =========================================================================
    // Mesh queries
    // =========================================================================

    /// Number of leaf elements
    Index num_elements() const { return static_cast<Index>(leaves_.size()); }

    /// Get element bounds
    const ElementBounds &element_bounds(Index elem) const;

    /// Get element refinement level
    DirectionalLevel element_level(Index elem) const;

    /// Get element Morton code
    uint64_t element_morton(Index elem) const;

    /// Get element center
    Vec3 element_center(Index elem) const;

    /// Get element size
    Vec3 element_size(Index elem) const;

    /// Get all leaf elements
    const std::vector<OctreeNode*> &elements() const { return leaves_; }

    /// Get root node of the octree (for tree traversal)
    const OctreeNode* root() const { return root_.get(); }

    // =========================================================================
    // Connectivity
    // =========================================================================

    /// @brief Get neighbor information for a face
    /// @param elem Element index
    /// @param face_id Face ID (0-5)
    /// @return Neighbor information
    NeighborInfo get_neighbor(Index elem, int face_id) const;

    /// @brief Get all face connections for an element
    /// @return Array of neighbor info for each face
    std::array<NeighborInfo, 6> get_face_neighbors(Index elem) const;

    /// @brief Build face connection structures for DG
    /// @return Vector of face connections per element
    std::vector<std::vector<FaceConnection>> build_face_connections() const;

    /// @brief Find element containing a point
    /// @param p Point in physical coordinates
    /// @return Element index, or -1 if not found
    Index find_element(const Vec3 &p) const;

    // =========================================================================
    // Morton code operations
    // =========================================================================

    /// @brief Get elements in Morton order (for space-filling curve
    /// partitioning)
    std::vector<Index> morton_order() const;

    /// @brief Partition elements by Morton code for MPI
    /// @param num_partitions Number of partitions
    /// @return Vector of (start, end) indices for each partition
    std::vector<std::pair<Index, Index>> morton_partition(int num_partitions) const;

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Iterate over all leaf elements
    template <typename Func>
    void for_each_element(Func &&f) const {
        for (Index i = 0; i < num_elements(); ++i) {
            f(i, *leaves_[i]);
        }
    }

    /// Iterate over interior faces (between elements)
    template <typename Func>
    void for_each_interior_face(Func &&f) const {
        for (Index e = 0; e < num_elements(); ++e) {
            for (int face = 0; face < 6; ++face) {
                NeighborInfo info = get_neighbor(e, face);
                if (!info.is_boundary()) {
                    // Only process each face once (lower Morton code owns it)
                    if (!info.neighbor_elements.empty() &&
                        leaves_[e]->morton < leaves_[info.neighbor_elements[0]]->morton) {
                        f(e, face, info);
                    }
                }
            }
        }
    }

    /// Iterate over boundary faces
    template <typename Func>
    void for_each_boundary_face(Func &&f) const {
        for (Index e = 0; e < num_elements(); ++e) {
            for (int face = 0; face < 6; ++face) {
                NeighborInfo info = get_neighbor(e, face);
                if (info.is_boundary()) {
                    f(e, face);
                }
            }
        }
    }

private:
    /// Domain bounds
    ElementBounds domain_;

    /// Root of the octree
    std::unique_ptr<OctreeNode> root_;

    /// Leaf nodes (elements) for fast access
    std::vector<OctreeNode*> leaves_;

    /// Lookup table for neighbor finding (Morton code -> node)
    std::map<uint64_t, OctreeNode*> morton_lookup_;

    /// Rebuild leaf list and lookup table
    void rebuild_leaf_list();

    /// Find node by Morton code and level
    OctreeNode* find_node(uint64_t morton, const DirectionalLevel &level) const;

    /// Create children for a node based on refinement mask
    void create_children(OctreeNode* node);

    /// Collect all leaves recursively
    void collect_leaves(OctreeNode* node, std::vector<OctreeNode*> &leaves);

    /// Build neighbor lookup recursively
    void build_lookup(OctreeNode* node);

    /// Find neighbor at same or coarser level
    OctreeNode* find_neighbor_same_or_coarser(OctreeNode* node, int face_id) const;

    /// Find all fine neighbors at a face
    std::vector<OctreeNode*> find_fine_neighbors(OctreeNode* node, int face_id) const;
};

/// @brief Refinement criteria for adaptive mesh
namespace refinement_criteria {

/// Refine based on solution gradient
struct GradientCriterion {
    Real threshold;

    bool should_refine(const VecX &solution, const ElementBounds &bounds,
                       const DirectionalLevel &level) const;

    RefineMask refinement_mask(const VecX &solution, const ElementBounds &bounds) const;
};

/// Refine based on bathymetry gradient (for sigma-coordinate)
struct BathymetryCriterion {
    std::function<Real(Real, Real)> bathymetry; // h(x, y)
    Real threshold;

    bool should_refine(const ElementBounds &bounds, const DirectionalLevel &level) const;
};

/// Refine near coastline
struct CoastlineCriterion {
    std::function<bool(Real, Real)> is_land; // Returns true if land
    int max_level;

    bool should_refine(const ElementBounds &bounds, const DirectionalLevel &level) const;
};

/// Combined criterion
struct CombinedCriterion {
    std::vector<std::function<bool(const ElementBounds &, const DirectionalLevel &)>> criteria;

    bool should_refine(const ElementBounds &bounds, const DirectionalLevel &level) const;
};

} // namespace refinement_criteria

} // namespace drifter
