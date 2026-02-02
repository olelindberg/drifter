#include "bathymetry/quadtree_adapter.hpp"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

QuadtreeAdapter::QuadtreeAdapter(QuadtreeAdapter&&) noexcept = default;
QuadtreeAdapter& QuadtreeAdapter::operator=(QuadtreeAdapter&&) noexcept = default;

QuadtreeAdapter::QuadtreeAdapter(const OctreeAdapter& octree)
    : impl_(std::make_unique<Impl>()) {
    sync_with_octree(octree);
}

void QuadtreeAdapter::sync_with_octree(const OctreeAdapter& octree) {
    // Find the minimum z value (bottom of domain)
    Real zmin = std::numeric_limits<Real>::max();
    for (Index i = 0; i < octree.num_elements(); ++i) {
        const auto& bounds = octree.element_bounds(i);
        zmin = std::min(zmin, bounds.zmin);
    }

    // Clear existing data
    root_.reset();
    leaf_storage_.clear();
    leaves_.clear();
    xy_lookup_.clear();

    // Find domain XY bounds
    Real xmin = std::numeric_limits<Real>::max();
    Real xmax = std::numeric_limits<Real>::lowest();
    Real ymin = std::numeric_limits<Real>::max();
    Real ymax = std::numeric_limits<Real>::lowest();

    // Collect bottom elements (those with z = zmin)
    std::vector<std::pair<Index, ElementBounds>> bottom_elements;
    const Real tol = 1e-10;

    for (Index i = 0; i < octree.num_elements(); ++i) {
        const auto& bounds = octree.element_bounds(i);
        if (std::abs(bounds.zmin - zmin) < tol) {
            bottom_elements.emplace_back(i, bounds);
            xmin = std::min(xmin, bounds.xmin);
            xmax = std::max(xmax, bounds.xmax);
            ymin = std::min(ymin, bounds.ymin);
            ymax = std::max(ymax, bounds.ymax);
        }
    }

    if (bottom_elements.empty()) {
        throw std::runtime_error("QuadtreeAdapter: no bottom elements found in octree");
    }

    // Set domain bounds
    domain_.xmin = xmin;
    domain_.xmax = xmax;
    domain_.ymin = ymin;
    domain_.ymax = ymax;

    // Create leaf nodes directly from bottom elements
    // Store in leaf_storage_ for ownership, leaves_ for fast access
    leaf_storage_.reserve(bottom_elements.size());
    leaves_.reserve(bottom_elements.size());

    for (const auto& [octree_idx, bounds3d] : bottom_elements) {
        auto node = std::make_unique<QuadtreeNode>();
        node->bounds.xmin = bounds3d.xmin;
        node->bounds.xmax = bounds3d.xmax;
        node->bounds.ymin = bounds3d.ymin;
        node->bounds.ymax = bounds3d.ymax;
        node->octree_element = octree_idx;

        // Compute level from element size relative to domain
        Real dx = bounds3d.xmax - bounds3d.xmin;
        Real dy = bounds3d.ymax - bounds3d.ymin;
        Real domain_dx = xmax - xmin;
        Real domain_dy = ymax - ymin;

        // Level = log2(domain_size / element_size)
        node->level.x = static_cast<int>(std::round(std::log2(domain_dx / dx)));
        node->level.y = static_cast<int>(std::round(std::log2(domain_dy / dy)));

        node->leaf_index = static_cast<Index>(leaves_.size());

        // Store raw pointer for fast access
        leaves_.push_back(node.get());

        // Transfer ownership to storage
        leaf_storage_.push_back(std::move(node));
    }

    // Build spatial lookup for neighbor finding
    build_lookup();
}

void QuadtreeAdapter::build_uniform(Real xmin, Real xmax, Real ymin, Real ymax,
                                    int nx, int ny) {
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

    Real dx = (xmax - xmin) / nx;
    Real dy = (ymax - ymin) / ny;

    // Compute uniform level
    int level_x = static_cast<int>(std::round(std::log2(nx)));
    int level_y = static_cast<int>(std::round(std::log2(ny)));

    leaf_storage_.reserve(nx * ny);
    leaves_.reserve(nx * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            auto node = std::make_unique<QuadtreeNode>();
            node->bounds.xmin = xmin + i * dx;
            node->bounds.xmax = xmin + (i + 1) * dx;
            node->bounds.ymin = ymin + j * dy;
            node->bounds.ymax = ymin + (j + 1) * dy;
            node->level.x = level_x;
            node->level.y = level_y;
            node->octree_element = -1;  // Standalone quadtree
            node->leaf_index = static_cast<Index>(leaves_.size());

            // Store raw pointer for fast access
            leaves_.push_back(node.get());

            // Transfer ownership to storage
            leaf_storage_.push_back(std::move(node));
        }
    }

    build_lookup();
}

const QuadBounds& QuadtreeAdapter::element_bounds(Index elem) const {
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

Vec2 QuadtreeAdapter::element_center(Index elem) const {
    return element_bounds(elem).center();
}

Vec2 QuadtreeAdapter::element_size(Index elem) const {
    return element_bounds(elem).size();
}

Index QuadtreeAdapter::octree_element(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        throw std::out_of_range("QuadtreeAdapter: element index out of range");
    }
    return leaves_[elem]->octree_element;
}

Index QuadtreeAdapter::find_element(const Vec2& p) const {
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
    EdgeNeighborInfo info;

    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        return info;  // Boundary
    }

    const auto& bounds = leaves_[elem]->bounds;
    const Real tol = 1e-10;

    // Edge coordinates to search for
    Real edge_coord;
    Real perp_min, perp_max;  // Perpendicular range
    bool search_x;  // true = search along x, false = search along y

    switch (edge_id) {
        case 0:  // Left edge (x = xmin)
            edge_coord = bounds.xmin;
            perp_min = bounds.ymin;
            perp_max = bounds.ymax;
            search_x = true;
            break;
        case 1:  // Right edge (x = xmax)
            edge_coord = bounds.xmax;
            perp_min = bounds.ymin;
            perp_max = bounds.ymax;
            search_x = true;
            break;
        case 2:  // Bottom edge (y = ymin)
            edge_coord = bounds.ymin;
            perp_min = bounds.xmin;
            perp_max = bounds.xmax;
            search_x = false;
            break;
        case 3:  // Top edge (y = ymax)
            edge_coord = bounds.ymax;
            perp_min = bounds.xmin;
            perp_max = bounds.xmax;
            search_x = false;
            break;
        default:
            return info;  // Invalid edge
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
        return info;
    }

    // Find neighbors by searching through all elements
    // (Could be optimized with spatial indexing)
    std::vector<Index> neighbors;
    int opposite_edge = (edge_id % 2 == 0) ? edge_id + 1 : edge_id - 1;

    for (Index i = 0; i < num_elements(); ++i) {
        if (i == elem) continue;

        const auto& nb = leaves_[i]->bounds;

        // Check if this element shares the edge
        Real nb_edge_coord;
        Real nb_perp_min, nb_perp_max;

        if (search_x) {
            nb_edge_coord = (edge_id == 0) ? nb.xmax : nb.xmin;
            nb_perp_min = nb.ymin;
            nb_perp_max = nb.ymax;
        } else {
            nb_edge_coord = (edge_id == 2) ? nb.ymax : nb.ymin;
            nb_perp_min = nb.xmin;
            nb_perp_max = nb.xmax;
        }

        // Check coordinate alignment
        if (std::abs(nb_edge_coord - edge_coord) > tol) {
            continue;
        }

        // Check overlap in perpendicular direction
        if (nb_perp_max <= perp_min + tol || nb_perp_min >= perp_max - tol) {
            continue;
        }

        neighbors.push_back(i);
    }

    if (neighbors.empty()) {
        info.type = EdgeNeighborInfo::Type::Boundary;
        return info;
    }

    // Determine connection type based on number and size of neighbors
    if (neighbors.size() == 1) {
        Index nb_idx = neighbors[0];
        const auto& nb = leaves_[nb_idx]->bounds;

        // Check if same size
        Real my_size = search_x ? (bounds.ymax - bounds.ymin) : (bounds.xmax - bounds.xmin);
        Real nb_size = search_x ? (nb.ymax - nb.ymin) : (nb.xmax - nb.xmin);

        if (std::abs(my_size - nb_size) < tol) {
            info.type = EdgeNeighborInfo::Type::Conforming;
        } else if (nb_size > my_size * 1.5) {
            // We are smaller, neighbor is larger
            info.type = EdgeNeighborInfo::Type::FineToCoarse;

            // Determine sub-edge index
            Real my_center = search_x ?
                0.5 * (bounds.ymin + bounds.ymax) :
                0.5 * (bounds.xmin + bounds.xmax);
            Real nb_center = search_x ?
                0.5 * (nb.ymin + nb.ymax) :
                0.5 * (nb.xmin + nb.xmax);

            info.subedge_index = (my_center < nb_center) ? 0 : 1;
        } else {
            // This shouldn't happen with 2:1 balance
            info.type = EdgeNeighborInfo::Type::Conforming;
        }

        info.neighbor_elements.push_back(nb_idx);
        info.neighbor_edges.push_back(opposite_edge);
    } else {
        // Multiple neighbors (we are coarser)
        info.type = EdgeNeighborInfo::Type::CoarseToFine;

        // Sort neighbors by position
        std::sort(neighbors.begin(), neighbors.end(),
            [this, search_x](Index a, Index b) {
                const auto& ba = leaves_[a]->bounds;
                const auto& bb = leaves_[b]->bounds;
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

    return info;
}

std::array<EdgeNeighborInfo, 4> QuadtreeAdapter::get_edge_neighbors(Index elem) const {
    return {
        get_neighbor(elem, 0),
        get_neighbor(elem, 1),
        get_neighbor(elem, 2),
        get_neighbor(elem, 3)
    };
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

void QuadtreeAdapter::collect_leaves(QuadtreeNode* node,
                                     std::vector<QuadtreeNode*>& leaves) {
    if (!node) return;

    if (node->is_leaf()) {
        leaves.push_back(node);
    } else {
        for (auto& child : node->children) {
            collect_leaves(child.get(), leaves);
        }
    }
}

void QuadtreeAdapter::build_lookup() {
    xy_lookup_.clear();

    for (auto* leaf : leaves_) {
        // Use element center as key
        auto key = std::make_pair(
            0.5 * (leaf->bounds.xmin + leaf->bounds.xmax),
            0.5 * (leaf->bounds.ymin + leaf->bounds.ymax)
        );
        xy_lookup_[key].push_back(leaf);
    }

    // Also build R-tree for fast point location
    build_rtree();
}

void QuadtreeAdapter::build_rtree() {
    // Build R-tree spatial index for O(log n) point location
    std::vector<ElementRTreeValue> values;
    values.reserve(leaves_.size());

    for (Index i = 0; i < num_elements(); ++i) {
        const auto& b = leaves_[i]->bounds;
        BGBox2D box(BGPoint2D(b.xmin, b.ymin), BGPoint2D(b.xmax, b.ymax));
        values.emplace_back(box, i);
    }

    impl_->element_rtree = std::make_unique<ElementRTree>(values.begin(), values.end());
}

Index QuadtreeAdapter::add_element(const QuadBounds& bounds, QuadLevel level) {
    // Create a new node
    auto node = std::make_unique<QuadtreeNode>();
    node->bounds = bounds;
    node->level = level;
    node->octree_element = -1;  // Standalone
    node->leaf_index = static_cast<Index>(leaves_.size());

    // Update domain bounds
    if (leaves_.empty()) {
        domain_ = bounds;
    } else {
        domain_.xmin = std::min(domain_.xmin, bounds.xmin);
        domain_.xmax = std::max(domain_.xmax, bounds.xmax);
        domain_.ymin = std::min(domain_.ymin, bounds.ymin);
        domain_.ymax = std::max(domain_.ymax, bounds.ymax);
    }

    // Store raw pointer for fast access
    leaves_.push_back(node.get());

    // Transfer ownership to storage
    Index idx = node->leaf_index;
    leaf_storage_.push_back(std::move(node));

    // Rebuild lookup
    build_lookup();

    return idx;
}

}  // namespace drifter
