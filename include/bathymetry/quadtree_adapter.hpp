#pragma once

// QuadtreeAdapter - 2D quadtree synced to bottom face of OctreeAdapter
//
// Provides a 2D mesh for CG bathymetry smoothing that mirrors the
// XY structure of the 3D octree's bottom (zeta=-1) face.
//
// Key features:
// - Automatically syncs with OctreeAdapter bottom elements
// - Inherits 2:1 balance from parent octree
// - Provides edge neighbor queries for CG DOF management
// - Maps quadtree elements to corresponding octree bottom elements
//
// Usage:
//   OctreeAdapter octree(...);
//   QuadtreeAdapter quadtree;
//   quadtree.sync_with_octree(octree);
//   for (Index e = 0; e < quadtree.num_elements(); ++e) {
//       const auto& bounds = quadtree.element_bounds(e);
//       // ... process 2D element
//   }

#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include <array>
#include <functional>
#include <map>
#include <vector>

namespace drifter {

/// @brief Physical bounds of a 2D quadrilateral element
struct QuadBounds {
    Real xmin, xmax;
    Real ymin, ymax;

    /// Element center
    Vec2 center() const {
        return Vec2(0.5 * (xmin + xmax), 0.5 * (ymin + ymax));
    }

    /// Element size in each direction
    Vec2 size() const {
        return Vec2(xmax - xmin, ymax - ymin);
    }

    /// Check if point is inside bounds
    bool contains(const Vec2& p) const {
        return p(0) >= xmin && p(0) <= xmax &&
               p(1) >= ymin && p(1) <= ymax;
    }

    /// Check if point is inside bounds (with tolerance)
    bool contains(const Vec2& p, Real tol) const {
        return p(0) >= xmin - tol && p(0) <= xmax + tol &&
               p(1) >= ymin - tol && p(1) <= ymax + tol;
    }
};

/// @brief Per-axis refinement level for 2D
struct QuadLevel {
    int x = 0;
    int y = 0;

    bool operator==(const QuadLevel& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const QuadLevel& other) const {
        return !(*this == other);
    }

    /// Maximum level across both axes
    int max_level() const { return std::max(x, y); }
};

/// @brief Edge neighbor information
/// Edge IDs: 0=left (x=xmin), 1=right (x=xmax), 2=bottom (y=ymin), 3=top (y=ymax)
struct EdgeNeighborInfo {
    /// Type of edge connection
    enum class Type {
        Boundary,       // No neighbor (domain boundary)
        Conforming,     // Same size neighbor
        CoarseToFine,   // This element is coarser (1 neighbor, 2 sub-edges on our side)
        FineToCoarse    // This element is finer (we are a sub-edge of coarser neighbor)
    };

    Type type = Type::Boundary;

    /// Neighbor element indices (1 for conforming/fine-to-coarse, 2 for coarse-to-fine)
    std::vector<Index> neighbor_elements;

    /// Edge IDs on neighbors (opposite edge for conforming)
    std::vector<int> neighbor_edges;

    /// Sub-edge index within coarser edge (for FineToCoarse)
    int subedge_index = 0;

    /// Is this a boundary edge?
    bool is_boundary() const { return type == Type::Boundary; }

    /// Is this a conforming interface?
    bool is_conforming() const { return type == Type::Conforming; }

    /// Is this a non-conforming interface?
    bool is_nonconforming() const {
        return type == Type::CoarseToFine || type == Type::FineToCoarse;
    }
};

/// @brief Quadtree node for 2D mesh
struct QuadtreeNode {
    /// Per-axis refinement levels
    QuadLevel level;

    /// Physical bounds
    QuadBounds bounds;

    /// Children (1, 2, or 4 depending on refinement)
    std::vector<std::unique_ptr<QuadtreeNode>> children;

    /// Parent pointer (nullptr for root)
    QuadtreeNode* parent = nullptr;

    /// Local index within quadtree (leaf numbering)
    Index leaf_index = -1;

    /// Corresponding octree bottom element index
    Index octree_element = -1;

    /// Check if this is a leaf node
    bool is_leaf() const { return children.empty(); }

    /// Number of children
    int num_children() const { return static_cast<int>(children.size()); }
};

/// @brief 2D quadtree adapter synced to OctreeAdapter bottom face
///
/// Creates a 2D mesh that mirrors the XY structure of the 3D octree.
/// Used for CG bathymetry smoothing before transfer to DG mesh.
class QuadtreeAdapter {
public:
    QuadtreeAdapter() = default;

    /// @brief Construct and sync with octree
    explicit QuadtreeAdapter(const OctreeAdapter& octree);

    /// @brief Sync with an OctreeAdapter
    ///
    /// Extracts the bottom face (zeta = -1) elements and creates
    /// a 2D quadtree matching their XY structure.
    ///
    /// @param octree 3D mesh to sync with
    void sync_with_octree(const OctreeAdapter& octree);

    /// @brief Build a standalone uniform quadtree
    /// @param xmin, xmax X bounds
    /// @param ymin, ymax Y bounds
    /// @param nx, ny Number of elements in each direction
    void build_uniform(Real xmin, Real xmax, Real ymin, Real ymax, int nx, int ny);

    /// @brief Add a single element manually (for testing non-conforming meshes)
    /// @param bounds Element bounds
    /// @param level Refinement level
    /// @return Index of the new element
    Index add_element(const QuadBounds& bounds, QuadLevel level);

    // =========================================================================
    // Mesh queries
    // =========================================================================

    /// Number of leaf elements
    Index num_elements() const { return static_cast<Index>(leaves_.size()); }

    /// Get element bounds
    const QuadBounds& element_bounds(Index elem) const;

    /// Get element refinement level
    QuadLevel element_level(Index elem) const;

    /// Get element center
    Vec2 element_center(Index elem) const;

    /// Get element size
    Vec2 element_size(Index elem) const;

    /// Get domain bounds
    const QuadBounds& domain_bounds() const { return domain_; }

    /// Get all leaf elements
    const std::vector<QuadtreeNode*>& elements() const { return leaves_; }

    // =========================================================================
    // Connectivity
    // =========================================================================

    /// @brief Get neighbor information for an edge
    /// @param elem Element index
    /// @param edge_id Edge ID: 0=left (x-), 1=right (x+), 2=bottom (y-), 3=top (y+)
    /// @return Edge neighbor information
    EdgeNeighborInfo get_neighbor(Index elem, int edge_id) const;

    /// @brief Get all edge connections for an element
    /// @return Array of neighbor info for each edge
    std::array<EdgeNeighborInfo, 4> get_edge_neighbors(Index elem) const;

    /// @brief Find element containing a point
    /// @param p Point in physical coordinates
    /// @return Element index, or -1 if not found
    Index find_element(const Vec2& p) const;

    /// @brief Get corresponding octree element index
    /// @param elem Quadtree element index
    /// @return Octree bottom element index (or -1 if standalone quadtree)
    Index octree_element(Index elem) const;

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Iterate over all leaf elements
    template <typename Func>
    void for_each_element(Func&& f) const {
        for (Index i = 0; i < num_elements(); ++i) {
            f(i, *leaves_[i]);
        }
    }

    /// Iterate over interior edges (between elements)
    template <typename Func>
    void for_each_interior_edge(Func&& f) const {
        for (Index e = 0; e < num_elements(); ++e) {
            for (int edge = 0; edge < 4; ++edge) {
                EdgeNeighborInfo info = get_neighbor(e, edge);
                if (!info.is_boundary() && !info.neighbor_elements.empty()) {
                    // Only process each edge once (lower index owns it)
                    if (e < info.neighbor_elements[0]) {
                        f(e, edge, info);
                    }
                }
            }
        }
    }

    /// Iterate over boundary edges
    template <typename Func>
    void for_each_boundary_edge(Func&& f) const {
        for (Index e = 0; e < num_elements(); ++e) {
            for (int edge = 0; edge < 4; ++edge) {
                EdgeNeighborInfo info = get_neighbor(e, edge);
                if (info.is_boundary()) {
                    f(e, edge);
                }
            }
        }
    }

private:
    /// Domain bounds
    QuadBounds domain_;

    /// Root of the quadtree (for hierarchical structure)
    std::unique_ptr<QuadtreeNode> root_;

    /// Storage for leaf nodes (owning pointers)
    std::vector<std::unique_ptr<QuadtreeNode>> leaf_storage_;

    /// Leaf nodes (elements) for fast access (non-owning pointers)
    std::vector<QuadtreeNode*> leaves_;

    /// Map from bounds to node for neighbor finding
    std::map<std::pair<Real, Real>, std::vector<QuadtreeNode*>> xy_lookup_;

    /// Rebuild leaf list and lookup
    void rebuild_leaf_list();

    /// Collect all leaves recursively
    void collect_leaves(QuadtreeNode* node, std::vector<QuadtreeNode*>& leaves);

    /// Build spatial lookup
    void build_lookup();

    /// Find neighbor along an edge
    QuadtreeNode* find_neighbor_at_edge(QuadtreeNode* node, int edge_id) const;

    /// Find fine neighbors at an edge (for coarse element)
    std::vector<QuadtreeNode*> find_fine_neighbors_at_edge(
        QuadtreeNode* node, int edge_id) const;

    /// Create a node from octree element bounds
    std::unique_ptr<QuadtreeNode> create_node_from_octree(
        const ElementBounds& bounds3d, Index octree_idx);
};

}  // namespace drifter
