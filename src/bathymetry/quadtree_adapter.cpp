#include "bathymetry/quadtree_adapter.hpp"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace drifter {

// Boost.Geometry types for spatial indexing (hidden from header)
namespace bg = boost::geometry;
namespace bgi = bg::index;

using BGPoint2D = bg::model::point<double, 2, bg::cs::cartesian>;
using BGBox2D = bg::model::box<BGPoint2D>;
using ElementRTreeValue = std::pair<BGBox2D, Index>;
using ElementRTree = bgi::rtree<ElementRTreeValue, bgi::rstar<16>>;

// Implementation struct holding Boost-dependent members
struct QuadtreeAdapter::Impl {
    std::unique_ptr<ElementRTree> element_rtree;
};

QuadtreeAdapter::QuadtreeAdapter() : impl_(std::make_unique<Impl>()) {}

QuadtreeAdapter::~QuadtreeAdapter() = default;

QuadtreeAdapter::QuadtreeAdapter(QuadtreeAdapter &&) noexcept = default;
QuadtreeAdapter &QuadtreeAdapter::operator=(QuadtreeAdapter &&) noexcept = default;

QuadtreeAdapter::QuadtreeAdapter(const OctreeAdapter &octree) : impl_(std::make_unique<Impl>()) {
    sync_with_octree(octree);
}

void QuadtreeAdapter::sync_with_octree(const OctreeAdapter &octree) {
    // Clear existing data
    root_.reset();
    leaf_storage_.clear();
    leaves_.clear();
    xy_lookup_.clear();

    // Get octree root - it contains the full tree structure
    const OctreeNode* octree_root = octree.root();
    if (!octree_root) {
        throw std::runtime_error("QuadtreeAdapter: octree has no root node");
    }

    // Copy tree structure from octree, projecting to 2D (bottom face)
    root_ = copy_octree_node_to_2d(octree_root, nullptr);

    if (!root_) {
        throw std::runtime_error("QuadtreeAdapter: failed to copy octree structure");
    }

    // Set domain from root bounds
    domain_ = root_->bounds;

    // Collect leaves and build lookups
    rebuild_leaf_list();
}

void QuadtreeAdapter::build_uniform(Real xmin, Real xmax, Real ymin, Real ymax, int nx, int ny) {
    // Clear existing data
    root_.reset();
    leaf_storage_.clear();
    leaves_.clear();
    xy_lookup_.clear();

    // Set domain bounds
    domain_.xmin = xmin;
    domain_.xmax = xmax;
    domain_.ymin = ymin;
    domain_.ymax = ymax;

    // Create root node
    root_ = std::make_unique<QuadtreeNode>();
    root_->bounds = domain_;
    root_->level = {0, 0};
    root_->morton = Morton3D::encode(0, 0, 0);
    root_->octree_element = -1;

    // Compute target levels from grid size (assumes power-of-2)
    int target_level_x = static_cast<int>(std::round(std::log2(nx)));
    int target_level_y = static_cast<int>(std::round(std::log2(ny)));

    // Recursively subdivide from root to target level
    subdivide_to_level(root_.get(), target_level_x, target_level_y);

    // Collect leaves and build lookups
    rebuild_leaf_list();
}

void QuadtreeAdapter::build_center_graded(int num_levels) {
    // Clear existing data
    root_.reset();
    leaf_storage_.clear();
    leaves_.clear();
    xy_lookup_.clear();

    // Fixed 1000m x 1000m domain
    domain_.xmin = 0.0;
    domain_.xmax = 1000.0;
    domain_.ymin = 0.0;
    domain_.ymax = 1000.0;

    // Create root node
    root_ = std::make_unique<QuadtreeNode>();
    root_->bounds = domain_;
    root_->level = {0, 0};
    root_->morton = Morton3D::encode(0, 0, 0);
    root_->octree_element = -1;

    if (num_levels <= 1) {
        // Single element covering entire domain
        rebuild_leaf_list();
        return;
    }

    // Recursively refine toward center
    Vec2 center(500.0, 500.0);
    subdivide_toward_center(root_.get(), num_levels - 1, center);

    // Collect leaves and build lookups
    rebuild_leaf_list();

    // Apply 2:1 balancing
    balance();
}

const QuadBounds &QuadtreeAdapter::element_bounds(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        throw std::out_of_range("QuadtreeAdapter: element index out of range");
    }
    return leaves_[elem]->bounds;
}

QuadLevel QuadtreeAdapter::element_level(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        throw std::out_of_range("QuadtreeAdapter: element index out of range");
    }
    return leaves_[elem]->level;
}

Vec2 QuadtreeAdapter::element_center(Index elem) const { return element_bounds(elem).center(); }

Vec2 QuadtreeAdapter::element_size(Index elem) const { return element_bounds(elem).size(); }

Index QuadtreeAdapter::octree_element(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        throw std::out_of_range("QuadtreeAdapter: element index out of range");
    }
    return leaves_[elem]->octree_element;
}

Index QuadtreeAdapter::find_element(const Vec2 &p) const {
    if (!impl_->element_rtree || impl_->element_rtree->empty()) {
        // Fallback to linear search if R-tree not built
        const Real tol = 1e-10;
        for (Index i = 0; i < num_elements(); ++i) {
            if (leaves_[i]->bounds.contains(p, tol)) {
                return i;
            }
        }
        return -1;
    }

    // Use R-tree for O(log n) query
    // Use 'intersects' rather than 'contains' to handle boundary points correctly
    BGPoint2D query_point(p(0), p(1));
    std::vector<ElementRTreeValue> result;
    impl_->element_rtree->query(bgi::intersects(query_point), std::back_inserter(result));
    return result.empty() ? -1 : result[0].second;
}

EdgeNeighborInfo QuadtreeAdapter::get_neighbor(Index elem, int edge_id) const {
    if (elem < 0 || elem >= static_cast<Index>(cached_neighbors_.size()) || edge_id < 0 ||
        edge_id >= 4) {
        return EdgeNeighborInfo{};
    }
    return cached_neighbors_[elem][edge_id];
}

void QuadtreeAdapter::precompute_neighbors() {
    Index N = num_elements();
    cached_neighbors_.clear();
    cached_neighbors_.resize(N);

    if (N == 0)
        return;

    const Real tol = 1e-10;

    // Compute quantization parameters
    Real min_size = std::numeric_limits<Real>::max();
    for (const auto* leaf : leaves_) {
        min_size = std::min(min_size, std::min(leaf->bounds.xmax - leaf->bounds.xmin,
                                               leaf->bounds.ymax - leaf->bounds.ymin));
    }
    Real inv_tol = 1.0 / (min_size * 1e-8);
    Real x0 = domain_.xmin;
    Real y0 = domain_.ymin;

    auto quantize_x = [&](Real x) -> int64_t {
        return static_cast<int64_t>(std::round((x - x0) * inv_tol));
    };
    auto quantize_y = [&](Real y) -> int64_t {
        return static_cast<int64_t>(std::round((y - y0) * inv_tol));
    };

    // Edge entry for spatial index
    struct EdgeEntry {
        Index elem;
        Real perp_min, perp_max;
        int edge_id; // which edge of this element (0-3)
    };

    // Build edge index: quantized coordinate -> elements with that edge
    // x_edges: elements whose left (xmin) or right (xmax) edge is at this x
    // y_edges: elements whose bottom (ymin) or top (ymax) edge is at this y
    std::unordered_map<int64_t, std::vector<EdgeEntry>> x_edges;
    std::unordered_map<int64_t, std::vector<EdgeEntry>> y_edges;

    for (Index i = 0; i < N; ++i) {
        const auto &b = leaves_[i]->bounds;
        // Left edge (edge 0): x = xmin, perpendicular range is y
        x_edges[quantize_x(b.xmin)].push_back({i, b.ymin, b.ymax, 0});
        // Right edge (edge 1): x = xmax, perpendicular range is y
        x_edges[quantize_x(b.xmax)].push_back({i, b.ymin, b.ymax, 1});
        // Bottom edge (edge 2): y = ymin, perpendicular range is x
        y_edges[quantize_y(b.ymin)].push_back({i, b.xmin, b.xmax, 2});
        // Top edge (edge 3): y = ymax, perpendicular range is x
        y_edges[quantize_y(b.ymax)].push_back({i, b.xmin, b.xmax, 3});
    }

    // For each element, compute neighbors for all 4 edges
    for (Index elem = 0; elem < N; ++elem) {
        const auto &bounds = leaves_[elem]->bounds;

        for (int edge_id = 0; edge_id < 4; ++edge_id) {
            EdgeNeighborInfo info;

            Real edge_coord;
            Real perp_min, perp_max;
            bool search_x = (edge_id <= 1);

            switch (edge_id) {
            case 0:
                edge_coord = bounds.xmin;
                perp_min = bounds.ymin;
                perp_max = bounds.ymax;
                break;
            case 1:
                edge_coord = bounds.xmax;
                perp_min = bounds.ymin;
                perp_max = bounds.ymax;
                break;
            case 2:
                edge_coord = bounds.ymin;
                perp_min = bounds.xmin;
                perp_max = bounds.xmax;
                break;
            default: // case 3
                edge_coord = bounds.ymax;
                perp_min = bounds.xmin;
                perp_max = bounds.xmax;
                break;
            }

            // Check if at domain boundary
            bool at_boundary = false;
            if (search_x) {
                at_boundary = (edge_id == 0 && std::abs(edge_coord - domain_.xmin) < tol) ||
                              (edge_id == 1 && std::abs(edge_coord - domain_.xmax) < tol);
            } else {
                at_boundary = (edge_id == 2 && std::abs(edge_coord - domain_.ymin) < tol) ||
                              (edge_id == 3 && std::abs(edge_coord - domain_.ymax) < tol);
            }

            if (at_boundary) {
                info.type = EdgeNeighborInfo::Type::Boundary;
                cached_neighbors_[elem][edge_id] = std::move(info);
                continue;
            }

            // Look up candidates from edge index
            // For edge 0 (left, x=xmin): find elements with right edge (edge 1) at
            // same x For edge 1 (right, x=xmax): find elements with left edge (edge
            // 0) at same x For edge 2 (bottom, y=ymin): find elements with top edge
            // (edge 3) at same y For edge 3 (top, y=ymax): find elements with bottom
            // edge (edge 2) at same y
            int opposite_edge = (edge_id % 2 == 0) ? edge_id + 1 : edge_id - 1;

            int64_t key = search_x ? quantize_x(edge_coord) : quantize_y(edge_coord);
            const auto &edge_map = search_x ? x_edges : y_edges;
            auto it = edge_map.find(key);

            std::vector<Index> neighbors;
            if (it != edge_map.end()) {
                for (const auto &entry : it->second) {
                    if (entry.elem == elem)
                        continue;
                    if (entry.edge_id != opposite_edge)
                        continue;

                    // Check overlap in perpendicular direction
                    if (entry.perp_max <= perp_min + tol || entry.perp_min >= perp_max - tol) {
                        continue;
                    }

                    neighbors.push_back(entry.elem);
                }
            }

            if (neighbors.empty()) {
                info.type = EdgeNeighborInfo::Type::Boundary;
                cached_neighbors_[elem][edge_id] = std::move(info);
                continue;
            }

            // Determine connection type based on number and size of neighbors
            if (neighbors.size() == 1) {
                Index nb_idx = neighbors[0];
                const auto &nb = leaves_[nb_idx]->bounds;

                Real my_size = search_x ? (bounds.ymax - bounds.ymin) : (bounds.xmax - bounds.xmin);
                Real nb_size = search_x ? (nb.ymax - nb.ymin) : (nb.xmax - nb.xmin);

                if (std::abs(my_size - nb_size) < tol) {
                    info.type = EdgeNeighborInfo::Type::Conforming;
                } else if (nb_size > my_size * 1.5) {
                    info.type = EdgeNeighborInfo::Type::FineToCoarse;

                    Real my_center = search_x ? 0.5 * (bounds.ymin + bounds.ymax)
                                              : 0.5 * (bounds.xmin + bounds.xmax);
                    Real nb_center =
                        search_x ? 0.5 * (nb.ymin + nb.ymax) : 0.5 * (nb.xmin + nb.xmax);

                    info.subedge_index = (my_center < nb_center) ? 0 : 1;
                } else {
                    info.type = EdgeNeighborInfo::Type::Conforming;
                }

                info.neighbor_elements.push_back(nb_idx);
                info.neighbor_edges.push_back(opposite_edge);
            } else {
                info.type = EdgeNeighborInfo::Type::CoarseToFine;

                std::sort(neighbors.begin(), neighbors.end(), [this, search_x](Index a, Index b) {
                    const auto &ba = leaves_[a]->bounds;
                    const auto &bb = leaves_[b]->bounds;
                    if (search_x) {
                        return ba.ymin < bb.ymin;
                    } else {
                        return ba.xmin < bb.xmin;
                    }
                });

                for (Index nb_idx : neighbors) {
                    info.neighbor_elements.push_back(nb_idx);
                    info.neighbor_edges.push_back(opposite_edge);
                }
            }

            cached_neighbors_[elem][edge_id] = std::move(info);
        }
    }
}

std::array<EdgeNeighborInfo, 4> QuadtreeAdapter::get_edge_neighbors(Index elem) const {
    return {get_neighbor(elem, 0), get_neighbor(elem, 1), get_neighbor(elem, 2),
            get_neighbor(elem, 3)};
}

void QuadtreeAdapter::rebuild_leaf_list() {
    leaves_.clear();
    if (root_) {
        collect_leaves(root_.get(), leaves_);
        // Update leaf indices
        for (Index i = 0; i < static_cast<Index>(leaves_.size()); ++i) {
            leaves_[i]->leaf_index = i;
        }
    }
    build_lookup();
}

void QuadtreeAdapter::collect_leaves(QuadtreeNode* node, std::vector<QuadtreeNode*> &leaves) {
    if (!node)
        return;

    if (node->is_leaf()) {
        leaves.push_back(node);
    } else {
        for (auto &child : node->children) {
            collect_leaves(child.get(), leaves);
        }
    }
}

void QuadtreeAdapter::build_lookup() {
    xy_lookup_.clear();

    for (auto* leaf : leaves_) {
        // Use element center as key
        auto key = std::make_pair(0.5 * (leaf->bounds.xmin + leaf->bounds.xmax),
                                  0.5 * (leaf->bounds.ymin + leaf->bounds.ymax));
        xy_lookup_[key].push_back(leaf);
    }

    // Also build R-tree for fast point location
    build_rtree();

    // Precompute all edge neighbors for O(1) lookup
    precompute_neighbors();
}

void QuadtreeAdapter::build_rtree() {
    // Build R-tree spatial index for O(log n) point location
    std::vector<ElementRTreeValue> values;
    values.reserve(leaves_.size());

    for (Index i = 0; i < num_elements(); ++i) {
        const auto &b = leaves_[i]->bounds;
        BGBox2D box(BGPoint2D(b.xmin, b.ymin), BGPoint2D(b.xmax, b.ymax));
        values.emplace_back(box, i);
    }

    impl_->element_rtree = std::make_unique<ElementRTree>(values.begin(), values.end());
}

Index QuadtreeAdapter::add_element(const QuadBounds &bounds, QuadLevel level) {
    // Create a new node
    auto node = std::make_unique<QuadtreeNode>();
    node->bounds = bounds;
    node->level = level;
    node->octree_element = -1; // Standalone
    node->leaf_index = static_cast<Index>(leaves_.size());

    // Update domain bounds first so we can compute Morton correctly
    if (leaves_.empty()) {
        domain_ = bounds;
    } else {
        domain_.xmin = std::min(domain_.xmin, bounds.xmin);
        domain_.xmax = std::max(domain_.xmax, bounds.xmax);
        domain_.ymin = std::min(domain_.ymin, bounds.ymin);
        domain_.ymax = std::max(domain_.ymax, bounds.ymax);
    }

    // Compute Morton code from grid indices
    Real elem_dx = bounds.xmax - bounds.xmin;
    Real elem_dy = bounds.ymax - bounds.ymin;
    Real cx = 0.5 * (bounds.xmin + bounds.xmax);
    Real cy = 0.5 * (bounds.ymin + bounds.ymax);
    uint32_t ix = static_cast<uint32_t>((cx - domain_.xmin) / elem_dx);
    uint32_t iy = static_cast<uint32_t>((cy - domain_.ymin) / elem_dy);
    node->morton = Morton3D::encode(ix, iy, 0);

    // Store raw pointer for fast access
    leaves_.push_back(node.get());

    // Transfer ownership to storage
    Index idx = node->leaf_index;
    leaf_storage_.push_back(std::move(node));

    // Rebuild lookup
    build_lookup();

    return idx;
}

// =============================================================================
// Tree construction helpers
// =============================================================================

void QuadtreeAdapter::refine_leaf(QuadtreeNode* node) {
    if (!node || !node->is_leaf()) {
        return;
    }

    Real xmid = 0.5 * (node->bounds.xmin + node->bounds.xmax);
    Real ymid = 0.5 * (node->bounds.ymin + node->bounds.ymax);

    uint32_t px, py, pz;
    Morton3D::decode(node->morton, px, py, pz);

    for (int cy = 0; cy < 2; ++cy) {
        for (int cx = 0; cx < 2; ++cx) {
            auto child = std::make_unique<QuadtreeNode>();
            child->parent = node;
            child->level.x = node->level.x + 1;
            child->level.y = node->level.y + 1;
            child->morton = Morton3D::encode(2 * px + cx, 2 * py + cy, 0);

            child->bounds.xmin = (cx == 0) ? node->bounds.xmin : xmid;
            child->bounds.xmax = (cx == 0) ? xmid : node->bounds.xmax;
            child->bounds.ymin = (cy == 0) ? node->bounds.ymin : ymid;
            child->bounds.ymax = (cy == 0) ? ymid : node->bounds.ymax;

            child->octree_element = -1;

            node->children.push_back(std::move(child));
        }
    }
}

void QuadtreeAdapter::balance() {
    // Iterate until no more refinement is needed
    bool changed = true;

    while (changed) {
        changed = false;

        // Collect current leaves (copy since we may modify during iteration)
        std::vector<QuadtreeNode*> current_leaves = leaves_;

        for (QuadtreeNode* node : current_leaves) {
            // Check all 4 edge neighbors
            for (int edge = 0; edge < 4; ++edge) {
                EdgeNeighborInfo info = get_neighbor(node->leaf_index, edge);

                if (info.is_boundary()) {
                    continue;
                }

                for (Index nb_idx : info.neighbor_elements) {
                    QuadtreeNode* neighbor = leaves_[nb_idx];

                    // Check 2:1 balance constraint per axis
                    int diff_x = node->level.x - neighbor->level.x;
                    int diff_y = node->level.y - neighbor->level.y;

                    // If neighbor is more than 1 level coarser, refine it
                    if ((diff_x > 1 || diff_y > 1) && neighbor->is_leaf()) {
                        refine_leaf(neighbor);
                        changed = true;
                    }
                }
            }
        }

        if (changed) {
            rebuild_leaf_list();
        }
    }
}

void QuadtreeAdapter::subdivide_toward_center(QuadtreeNode* node, int remaining_levels,
                                               const Vec2& center) {
    if (remaining_levels <= 0) {
        return;  // This is a leaf
    }

    // Subdivide into 4 children
    Real xmid = 0.5 * (node->bounds.xmin + node->bounds.xmax);
    Real ymid = 0.5 * (node->bounds.ymin + node->bounds.ymax);

    // Determine which child should be further refined (the one containing center)
    // When center is exactly at midpoint, pick upper-right (cx=1, cy=1)
    int refine_cx = (center(0) >= xmid) ? 1 : 0;
    int refine_cy = (center(1) >= ymid) ? 1 : 0;

    uint32_t px, py, pz;
    Morton3D::decode(node->morton, px, py, pz);

    for (int cy = 0; cy < 2; ++cy) {
        for (int cx = 0; cx < 2; ++cx) {
            auto child = std::make_unique<QuadtreeNode>();
            child->parent = node;
            child->level.x = node->level.x + 1;
            child->level.y = node->level.y + 1;
            child->morton = Morton3D::encode(2 * px + cx, 2 * py + cy, 0);

            child->bounds.xmin = (cx == 0) ? node->bounds.xmin : xmid;
            child->bounds.xmax = (cx == 0) ? xmid : node->bounds.xmax;
            child->bounds.ymin = (cy == 0) ? node->bounds.ymin : ymid;
            child->bounds.ymax = (cy == 0) ? ymid : node->bounds.ymax;

            child->octree_element = -1;

            // Only refine the child containing the center
            if (cx == refine_cx && cy == refine_cy) {
                subdivide_toward_center(child.get(), remaining_levels - 1, center);
            }

            node->children.push_back(std::move(child));
        }
    }
}

void QuadtreeAdapter::subdivide_to_level(QuadtreeNode* node, int target_x, int target_y) {
    bool need_x = node->level.x < target_x;
    bool need_y = node->level.y < target_y;

    if (!need_x && !need_y) {
        return;  // Reached target level, this is a leaf
    }

    Real xmid = 0.5 * (node->bounds.xmin + node->bounds.xmax);
    Real ymid = 0.5 * (node->bounds.ymin + node->bounds.ymax);

    uint32_t px, py, pz;
    Morton3D::decode(node->morton, px, py, pz);

    // Determine subdivision pattern based on which dimensions need refinement
    int num_x = need_x ? 2 : 1;
    int num_y = need_y ? 2 : 1;

    for (int cy = 0; cy < num_y; ++cy) {
        for (int cx = 0; cx < num_x; ++cx) {
            auto child = std::make_unique<QuadtreeNode>();
            child->parent = node;

            // Only increment level in dimensions being subdivided
            child->level.x = node->level.x + (need_x ? 1 : 0);
            child->level.y = node->level.y + (need_y ? 1 : 0);

            // Update Morton code based on actual subdivision
            uint32_t new_px = need_x ? (2 * px + cx) : px;
            uint32_t new_py = need_y ? (2 * py + cy) : py;
            child->morton = Morton3D::encode(new_px, new_py, 0);

            // Set bounds based on subdivision pattern
            if (need_x) {
                child->bounds.xmin = (cx == 0) ? node->bounds.xmin : xmid;
                child->bounds.xmax = (cx == 0) ? xmid : node->bounds.xmax;
            } else {
                child->bounds.xmin = node->bounds.xmin;
                child->bounds.xmax = node->bounds.xmax;
            }

            if (need_y) {
                child->bounds.ymin = (cy == 0) ? node->bounds.ymin : ymid;
                child->bounds.ymax = (cy == 0) ? ymid : node->bounds.ymax;
            } else {
                child->bounds.ymin = node->bounds.ymin;
                child->bounds.ymax = node->bounds.ymax;
            }

            child->octree_element = -1;

            // Recurse
            subdivide_to_level(child.get(), target_x, target_y);

            node->children.push_back(std::move(child));
        }
    }
}

std::unique_ptr<QuadtreeNode> QuadtreeAdapter::copy_octree_node_to_2d(
    const OctreeNode* octree_node,
    QuadtreeNode* parent
) {
    if (!octree_node) return nullptr;

    auto node = std::make_unique<QuadtreeNode>();
    node->parent = parent;

    // Project 3D bounds to 2D
    node->bounds.xmin = octree_node->bounds.xmin;
    node->bounds.xmax = octree_node->bounds.xmax;
    node->bounds.ymin = octree_node->bounds.ymin;
    node->bounds.ymax = octree_node->bounds.ymax;

    // Copy level (just x and y from DirectionalLevel)
    node->level.x = octree_node->level.level_x;
    node->level.y = octree_node->level.level_y;

    // Copy Morton code (works for 2D since we use Morton3D with z=0)
    node->morton = octree_node->morton;

    // If this is a leaf in octree, mark octree element index
    if (octree_node->is_leaf()) {
        node->octree_element = octree_node->leaf_index;
    } else {
        // Copy children (project 8 octree children to quadtree)
        // For bottom face: only keep children at zmin
        Real zmin = octree_node->bounds.zmin;
        const Real tol = 1e-10;

        for (const auto& oct_child : octree_node->children) {
            // Only copy children that touch the bottom face
            if (std::abs(oct_child->bounds.zmin - zmin) < tol) {
                auto quad_child = copy_octree_node_to_2d(oct_child.get(), node.get());
                if (quad_child) {
                    node->children.push_back(std::move(quad_child));
                }
            }
        }
    }

    return node;
}

// =============================================================================
// Tree traversal API
// =============================================================================

std::vector<const QuadtreeNode*> QuadtreeAdapter::nodes_at_level(int level) const {
    std::vector<const QuadtreeNode*> result;
    if (root_) {
        collect_nodes_at_level(root_.get(), level, result);
    }
    return result;
}

void QuadtreeAdapter::collect_nodes_at_level(const QuadtreeNode* node, int target_level,
                                              std::vector<const QuadtreeNode*>& result) const {
    if (!node) return;

    int node_level = node->level.max_level();
    if (node_level == target_level) {
        result.push_back(node);
        return;
    }

    // If we haven't reached target level yet, recurse into children
    if (node_level < target_level) {
        for (const auto& child : node->children) {
            collect_nodes_at_level(child.get(), target_level, result);
        }
    }
    // If node_level > target_level, this node is deeper than we want - don't include
}

int QuadtreeAdapter::max_depth() const {
    if (!root_) return 0;
    return max_depth_recursive(root_.get());
}

int QuadtreeAdapter::max_depth_recursive(const QuadtreeNode* node) const {
    if (!node) return 0;

    int depth = node->level.max_level();

    for (const auto& child : node->children) {
        depth = std::max(depth, max_depth_recursive(child.get()));
    }

    return depth;
}

} // namespace drifter
