#include "bathymetry/quadtree_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace drifter {

namespace {

/// Compute Morton code for a point at maximum tree depth
uint64_t compute_morton_at_max_depth(Real x, Real y, const QuadBounds& domain, int max_depth) {
    Real dx = domain.xmax - domain.xmin;
    Real dy = domain.ymax - domain.ymin;
    Real fx = (dx > 0) ? (x - domain.xmin) / dx : 0.0;
    Real fy = (dy > 0) ? (y - domain.ymin) / dy : 0.0;
    fx = std::clamp(fx, 0.0, 1.0);
    fy = std::clamp(fy, 0.0, 1.0);

    uint32_t grid_size = 1u << max_depth;
    uint32_t ix = std::min(static_cast<uint32_t>(fx * grid_size), grid_size - 1);
    uint32_t iy = std::min(static_cast<uint32_t>(fy * grid_size), grid_size - 1);

    return Morton3D::encode(ix, iy, 0);
}

} // namespace

QuadtreeAdapter::QuadtreeAdapter() = default;

QuadtreeAdapter::~QuadtreeAdapter() = default;

QuadtreeAdapter::QuadtreeAdapter(QuadtreeAdapter &&) noexcept = default;
QuadtreeAdapter &QuadtreeAdapter::operator=(QuadtreeAdapter &&) noexcept = default;

QuadtreeAdapter::QuadtreeAdapter(const OctreeAdapter &octree) {
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
    if (leaves_.empty()) {
        return -1;
    }

    // Quick bounds check
    if (p(0) < domain_.xmin || p(0) > domain_.xmax ||
        p(1) < domain_.ymin || p(1) > domain_.ymax) {
        return -1;
    }

    // Compute Morton code of query point at max depth
    uint64_t query_morton = compute_morton_at_max_depth(p(0), p(1), domain_, cached_max_depth_);

    // Binary search: find first leaf with Morton > query_morton
    auto it = std::upper_bound(leaf_morton_codes_.begin(),
                               leaf_morton_codes_.end(),
                               query_morton);

    // The containing element is the one before (or at) the found position
    if (it != leaf_morton_codes_.begin()) {
        --it;
    }
    size_t idx = static_cast<size_t>(std::distance(leaf_morton_codes_.begin(), it));

    // Verify the point is actually in this element's bounds
    if (leaves_[idx]->bounds.contains(p)) {
        return static_cast<Index>(idx);
    }

    // Check next element for boundary cases
    if (idx + 1 < leaves_.size() && leaves_[idx + 1]->bounds.contains(p)) {
        return static_cast<Index>(idx + 1);
    }

    return -1;
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

    // Ensure leaf_morton_codes_ is up-to-date for find_element()
    // This handles cases where add_element() was used without rebuild_leaf_list()
    if (leaf_morton_codes_.size() != static_cast<size_t>(N)) {
        cached_max_depth_ = max_depth();
        leaf_morton_codes_.resize(N);
        for (Index i = 0; i < N; ++i) {
            leaf_morton_codes_[i] = compute_morton_at_max_depth(
                leaves_[i]->bounds.xmin, leaves_[i]->bounds.ymin, domain_, cached_max_depth_);
        }
        // Sort leaves by Morton code for binary search
        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [this](size_t a, size_t b) { return leaf_morton_codes_[a] < leaf_morton_codes_[b]; });

        std::vector<QuadtreeNode *> sorted_leaves(N);
        std::vector<uint64_t> sorted_mortons(N);
        for (Index i = 0; i < N; ++i) {
            sorted_leaves[i] = leaves_[indices[i]];
            sorted_mortons[i] = leaf_morton_codes_[indices[i]];
        }
        leaves_ = std::move(sorted_leaves);
        leaf_morton_codes_ = std::move(sorted_mortons);

        // Update leaf indices after sorting
        for (Index i = 0; i < N; ++i) {
            leaves_[i]->leaf_index = i;
        }
    }

    const Real tol = 1e-10;
    constexpr int opposite_edge[4] = {1, 0, 3, 2};

    for (Index elem = 0; elem < N; ++elem) {
        const auto &bounds = leaves_[elem]->bounds;
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real probe_offset = std::min(dx, dy) * 0.01;

        for (int edge_id = 0; edge_id < 4; ++edge_id) {
            EdgeNeighborInfo info;

            // Check domain boundary
            bool at_boundary =
                (edge_id == 0 && std::abs(bounds.xmin - domain_.xmin) < tol) ||
                (edge_id == 1 && std::abs(bounds.xmax - domain_.xmax) < tol) ||
                (edge_id == 2 && std::abs(bounds.ymin - domain_.ymin) < tol) ||
                (edge_id == 3 && std::abs(bounds.ymax - domain_.ymax) < tol);

            if (at_boundary) {
                info.type = EdgeNeighborInfo::Type::Boundary;
                cached_neighbors_[elem][edge_id] = std::move(info);
                continue;
            }

            // Compute probe points along the edge using Morton-based spatial lookup
            // 3 probes detect both conforming and coarse-to-fine cases
            bool is_x_edge = (edge_id <= 1);
            std::array<Vec2, 3> probes;

            if (is_x_edge) {
                Real x_probe =
                    (edge_id == 0) ? bounds.xmin - probe_offset : bounds.xmax + probe_offset;
                probes[0] = Vec2(x_probe, 0.5 * (bounds.ymin + bounds.ymax));
                probes[1] = Vec2(x_probe, bounds.ymin + 0.25 * dy);
                probes[2] = Vec2(x_probe, bounds.ymax - 0.25 * dy);
            } else {
                Real y_probe =
                    (edge_id == 2) ? bounds.ymin - probe_offset : bounds.ymax + probe_offset;
                probes[0] = Vec2(0.5 * (bounds.xmin + bounds.xmax), y_probe);
                probes[1] = Vec2(bounds.xmin + 0.25 * dx, y_probe);
                probes[2] = Vec2(bounds.xmax - 0.25 * dx, y_probe);
            }

            // Find unique neighbors from probe points via O(log N) binary search
            std::vector<Index> neighbors;
            neighbors.reserve(2);
            for (const auto &probe : probes) {
                Index nb = find_element(probe);
                if (nb >= 0 && nb != elem) {
                    if (std::find(neighbors.begin(), neighbors.end(), nb) == neighbors.end()) {
                        neighbors.push_back(nb);
                    }
                }
            }

            if (neighbors.empty()) {
                info.type = EdgeNeighborInfo::Type::Boundary;
                cached_neighbors_[elem][edge_id] = std::move(info);
                continue;
            }

            // Classify connection type based on number and size of neighbors
            if (neighbors.size() == 1) {
                Index nb_idx = neighbors[0];
                const auto &nb = leaves_[nb_idx]->bounds;

                Real my_size = is_x_edge ? dy : dx;
                Real nb_size = is_x_edge ? (nb.ymax - nb.ymin) : (nb.xmax - nb.xmin);

                if (std::abs(my_size - nb_size) < tol) {
                    info.type = EdgeNeighborInfo::Type::Conforming;
                } else if (nb_size > my_size * 1.5) {
                    info.type = EdgeNeighborInfo::Type::FineToCoarse;
                    Real my_center = is_x_edge ? 0.5 * (bounds.ymin + bounds.ymax)
                                               : 0.5 * (bounds.xmin + bounds.xmax);
                    Real nb_center =
                        is_x_edge ? 0.5 * (nb.ymin + nb.ymax) : 0.5 * (nb.xmin + nb.xmax);
                    info.subedge_index = (my_center < nb_center) ? 0 : 1;
                } else {
                    info.type = EdgeNeighborInfo::Type::Conforming;
                }

                info.neighbor_elements.push_back(nb_idx);
                info.neighbor_edges.push_back(opposite_edge[edge_id]);
            } else {
                info.type = EdgeNeighborInfo::Type::CoarseToFine;

                // Sort by perpendicular coordinate
                std::sort(neighbors.begin(), neighbors.end(), [this, is_x_edge](Index a, Index b) {
                    const auto &ba = leaves_[a]->bounds;
                    const auto &bb = leaves_[b]->bounds;
                    return is_x_edge ? (ba.ymin < bb.ymin) : (ba.xmin < bb.xmin);
                });

                for (Index nb_idx : neighbors) {
                    info.neighbor_elements.push_back(nb_idx);
                    info.neighbor_edges.push_back(opposite_edge[edge_id]);
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

void QuadtreeAdapter::precompute_neighbors_fast() {
    // Use tree traversal for neighbor finding instead of probe-based lookups
    // This is O(N × 4 × log(N)) vs O(N × 12 × log(N)) for probe-based
    Index N = num_elements();
    cached_neighbors_.clear();
    cached_neighbors_.resize(N);

    if (N == 0) return;

    constexpr int opposite_edge[4] = {1, 0, 3, 2};
    const Real tol = 1e-10;

    for (Index elem = 0; elem < N; ++elem) {
        QuadtreeNode* node = leaves_[elem];
        const auto& bounds = node->bounds;
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        for (int edge_id = 0; edge_id < 4; ++edge_id) {
            EdgeNeighborInfo info;

            // Check domain boundary
            bool at_boundary =
                (edge_id == 0 && std::abs(bounds.xmin - domain_.xmin) < tol) ||
                (edge_id == 1 && std::abs(bounds.xmax - domain_.xmax) < tol) ||
                (edge_id == 2 && std::abs(bounds.ymin - domain_.ymin) < tol) ||
                (edge_id == 3 && std::abs(bounds.ymax - domain_.ymax) < tol);

            if (at_boundary) {
                info.type = EdgeNeighborInfo::Type::Boundary;
                cached_neighbors_[elem][edge_id] = std::move(info);
                continue;
            }

            // Find neighbor via tree traversal
            QuadtreeNode* neighbor = find_neighbor_via_tree(node, edge_id);

            if (!neighbor) {
                info.type = EdgeNeighborInfo::Type::Boundary;
                cached_neighbors_[elem][edge_id] = std::move(info);
                continue;
            }

            Index nb_idx = neighbor->leaf_index;
            const auto& nb = neighbor->bounds;
            bool is_x_edge = (edge_id <= 1);

            Real my_size = is_x_edge ? dy : dx;
            Real nb_size = is_x_edge ? (nb.ymax - nb.ymin) : (nb.xmax - nb.xmin);

            if (std::abs(my_size - nb_size) < tol) {
                info.type = EdgeNeighborInfo::Type::Conforming;
            } else if (nb_size > my_size * 1.5) {
                info.type = EdgeNeighborInfo::Type::FineToCoarse;
                Real my_center = is_x_edge ? 0.5 * (bounds.ymin + bounds.ymax)
                                           : 0.5 * (bounds.xmin + bounds.xmax);
                Real nb_center = is_x_edge ? 0.5 * (nb.ymin + nb.ymax)
                                           : 0.5 * (nb.xmin + nb.xmax);
                info.subedge_index = (my_center < nb_center) ? 0 : 1;
            } else if (my_size > nb_size * 1.5) {
                // Coarse-to-fine: need to find both fine neighbors
                info.type = EdgeNeighborInfo::Type::CoarseToFine;
                // First neighbor already found
                info.neighbor_elements.push_back(nb_idx);
                info.neighbor_edges.push_back(opposite_edge[edge_id]);

                // Find second fine neighbor using different probe point
                Real probe_offset = std::min(dx, dy) * 0.01;
                Vec2 probe2;
                if (is_x_edge) {
                    Real x_probe = (edge_id == 0) ? bounds.xmin - probe_offset : bounds.xmax + probe_offset;
                    Real y_other = (nb.ymin < 0.5 * (bounds.ymin + bounds.ymax))
                                   ? bounds.ymax - 0.25 * dy
                                   : bounds.ymin + 0.25 * dy;
                    probe2 = Vec2(x_probe, y_other);
                } else {
                    Real y_probe = (edge_id == 2) ? bounds.ymin - probe_offset : bounds.ymax + probe_offset;
                    Real x_other = (nb.xmin < 0.5 * (bounds.xmin + bounds.xmax))
                                   ? bounds.xmax - 0.25 * dx
                                   : bounds.xmin + 0.25 * dx;
                    probe2 = Vec2(x_other, y_probe);
                }
                Index nb2_idx = find_element(probe2);
                if (nb2_idx >= 0 && nb2_idx != nb_idx) {
                    info.neighbor_elements.push_back(nb2_idx);
                    info.neighbor_edges.push_back(opposite_edge[edge_id]);
                }
                cached_neighbors_[elem][edge_id] = std::move(info);
                continue;
            } else {
                info.type = EdgeNeighborInfo::Type::Conforming;
            }

            info.neighbor_elements.push_back(nb_idx);
            info.neighbor_edges.push_back(opposite_edge[edge_id]);
            cached_neighbors_[elem][edge_id] = std::move(info);
        }
    }
}

void QuadtreeAdapter::rebuild_leaf_list() {
    leaves_.clear();
    leaf_morton_codes_.clear();

    if (root_) {
        collect_leaves(root_.get(), leaves_);

        // Cache max depth for Morton calculations
        cached_max_depth_ = max_depth();

        // Compute Morton codes at max depth for each leaf's lower-left corner
        leaf_morton_codes_.resize(leaves_.size());
        for (size_t i = 0; i < leaves_.size(); ++i) {
            leaf_morton_codes_[i] = compute_morton_at_max_depth(
                leaves_[i]->bounds.xmin, leaves_[i]->bounds.ymin,
                domain_, cached_max_depth_);
        }

        // Check if already sorted (O(N) check vs O(N log N) sort)
        // DFS traversal of tree with Morton-ordered children produces sorted leaves
        bool already_sorted = true;
        for (size_t i = 1; i < leaf_morton_codes_.size(); ++i) {
            if (leaf_morton_codes_[i] < leaf_morton_codes_[i - 1]) {
                already_sorted = false;
                break;
            }
        }

        if (!already_sorted) {
            // Create index array and sort by Morton code
            std::vector<size_t> indices(leaves_.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [this](size_t a, size_t b) {
                    return leaf_morton_codes_[a] < leaf_morton_codes_[b];
                });

            // Reorder leaves and Morton codes by sorted indices
            std::vector<QuadtreeNode*> sorted_leaves(leaves_.size());
            std::vector<uint64_t> sorted_mortons(leaves_.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                sorted_leaves[i] = leaves_[indices[i]];
                sorted_mortons[i] = leaf_morton_codes_[indices[i]];
            }
            leaves_ = std::move(sorted_leaves);
            leaf_morton_codes_ = std::move(sorted_mortons);
        }

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

    // Precompute all edge neighbors for O(1) lookup
    precompute_neighbors();
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

        // Use tree traversal instead of iterating over leaves_
        // This avoids needing to rebuild_leaf_list() inside the loop
        if (root_) {
            balance_subtree(root_.get(), changed);
        }
    }

    // Single rebuild at the end (not inside the loop)
    rebuild_leaf_list();
}

void QuadtreeAdapter::balance_subtree(QuadtreeNode* node, bool& changed) {
    if (!node) return;

    if (node->is_leaf()) {
        // Check all 4 edge neighbors via tree traversal
        for (int edge = 0; edge < 4; ++edge) {
            QuadtreeNode* neighbor = find_neighbor_via_tree(node, edge);

            if (!neighbor) continue;  // Boundary

            // Check 2:1 balance constraint per axis
            int diff_x = node->level.x - neighbor->level.x;
            int diff_y = node->level.y - neighbor->level.y;

            // If neighbor is more than 1 level coarser, refine it
            if ((diff_x > 1 || diff_y > 1) && neighbor->is_leaf()) {
                refine_leaf(neighbor);
                changed = true;
            }
        }
    } else {
        // Recurse into children
        for (auto& child : node->children) {
            balance_subtree(child.get(), changed);
        }
    }
}

QuadtreeNode* QuadtreeAdapter::find_neighbor_via_tree(QuadtreeNode* node, int edge_id) const {
    if (!node) return nullptr;

    // Direction offsets: 0=left(-x), 1=right(+x), 2=bottom(-y), 3=top(+y)
    const Real tol = 1e-10;
    const auto& bounds = node->bounds;

    // Check domain boundary
    bool at_boundary =
        (edge_id == 0 && std::abs(bounds.xmin - domain_.xmin) < tol) ||
        (edge_id == 1 && std::abs(bounds.xmax - domain_.xmax) < tol) ||
        (edge_id == 2 && std::abs(bounds.ymin - domain_.ymin) < tol) ||
        (edge_id == 3 && std::abs(bounds.ymax - domain_.ymax) < tol);

    if (at_boundary) return nullptr;

    // Compute probe point just outside the edge
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real probe_offset = std::min(dx, dy) * 0.01;

    Vec2 probe;
    switch (edge_id) {
        case 0: probe = Vec2(bounds.xmin - probe_offset, 0.5 * (bounds.ymin + bounds.ymax)); break;
        case 1: probe = Vec2(bounds.xmax + probe_offset, 0.5 * (bounds.ymin + bounds.ymax)); break;
        case 2: probe = Vec2(0.5 * (bounds.xmin + bounds.xmax), bounds.ymin - probe_offset); break;
        case 3: probe = Vec2(0.5 * (bounds.xmin + bounds.xmax), bounds.ymax + probe_offset); break;
        default: return nullptr;
    }

    // Traverse tree to find leaf containing probe point
    QuadtreeNode* current = root_.get();
    while (current && !current->is_leaf()) {
        bool found = false;
        for (auto& child : current->children) {
            if (child->bounds.contains(probe, tol)) {
                current = child.get();
                found = true;
                break;
            }
        }
        if (!found) break;
    }

    return (current && current != node && current->is_leaf()) ? current : nullptr;
}

RefinementResult QuadtreeAdapter::refine(const std::vector<Index> &elements_to_refine) {
    RefinementResult result;
    result.num_refined = 0;

    if (elements_to_refine.empty()) {
        return result;
    }

    // Record original leaf pointers to identify new elements later
    std::unordered_set<QuadtreeNode*> original_leaves(leaves_.begin(), leaves_.end());

    // Convert indices to node pointers before refinement (indices will change)
    std::vector<QuadtreeNode*> nodes_to_refine;
    nodes_to_refine.reserve(elements_to_refine.size());

    for (Index idx : elements_to_refine) {
        if (idx >= 0 && idx < static_cast<Index>(leaves_.size())) {
            QuadtreeNode* node = leaves_[idx];
            if (node && node->is_leaf()) {
                nodes_to_refine.push_back(node);
            }
        }
    }

    if (nodes_to_refine.empty()) {
        return result;
    }

    // Refine all selected nodes
    for (QuadtreeNode* node : nodes_to_refine) {
        if (node->is_leaf()) {  // Double-check it's still a leaf
            refine_leaf(node);
            ++result.num_refined;
        }
    }

    // Balance the tree to maintain 2:1 constraint
    // Note: balance() calls rebuild_leaf_list() at the end, so we don't
    // need to call it here. balance_subtree() uses tree traversal, not leaves_.
    balance();

    // Collect indices of newly created elements
    result.new_elements.reserve(leaves_.size() - original_leaves.size() + nodes_to_refine.size());
    for (Index i = 0; i < static_cast<Index>(leaves_.size()); ++i) {
        if (original_leaves.find(leaves_[i]) == original_leaves.end()) {
            result.new_elements.push_back(i);
        }
    }

    return result;
}

void QuadtreeAdapter::subdivide_toward_center(QuadtreeNode* node, int remaining_levels,
                                              const Vec2 &center) {
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
    QuadtreeNode* parent) {
    if (!octree_node)
        return nullptr;

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

        for (const auto &oct_child : octree_node->children) {
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
                                             std::vector<const QuadtreeNode*> &result) const {
    if (!node)
        return;

    int node_level = node->level.max_level();
    if (node_level == target_level) {
        result.push_back(node);
        return;
    }

    // If we haven't reached target level yet, recurse into children
    if (node_level < target_level) {
        for (const auto &child : node->children) {
            collect_nodes_at_level(child.get(), target_level, result);
        }
    }
    // If node_level > target_level, this node is deeper than we want - don't include
}

int QuadtreeAdapter::max_depth() const {
    if (!root_)
        return 0;
    return max_depth_recursive(root_.get());
}

int QuadtreeAdapter::max_depth_recursive(const QuadtreeNode* node) const {
    if (!node)
        return 0;

    int depth = node->level.max_level();

    for (const auto &child : node->children) {
        depth = std::max(depth, max_depth_recursive(child.get()));
    }

    return depth;
}

} // namespace drifter
