#include "mesh/area_of_interest.hpp"
#include "mesh/octree_adapter.hpp"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/wkt/wkt.hpp>

#include <stdexcept>

namespace drifter {

namespace bg = boost::geometry;

using Point2D = bg::model::point<double, 2, bg::cs::cartesian>;
using Box2D = bg::model::box<Point2D>;
using Ring2D = bg::model::ring<Point2D>;
using Polygon2D = bg::model::polygon<Point2D, true>; // Clockwise orientation

/// @brief Region definition with polygon geometry
struct AOIRegion {
    std::string name;
    int bit_index;
    Polygon2D polygon;
    Box2D bounding_box; // Cached for fast rejection
};

struct AreaOfInterest::Impl {
    std::vector<AOIRegion> regions;
};

// =============================================================================
// Constructor/Destructor
// =============================================================================

AreaOfInterest::AreaOfInterest() : impl_(std::make_unique<Impl>()) {}

AreaOfInterest::~AreaOfInterest() = default;

AreaOfInterest::AreaOfInterest(AreaOfInterest &&) noexcept = default;
AreaOfInterest &AreaOfInterest::operator=(AreaOfInterest &&) noexcept = default;

// =============================================================================
// Region definition
// =============================================================================

int AreaOfInterest::define_region(const std::string &name, const std::string &wkt) {
    if (next_bit_ >= 8) {
        throw std::runtime_error("AreaOfInterest: Maximum 8 regions supported");
    }
    if (region_bits_.count(name)) {
        throw std::runtime_error("AreaOfInterest: Region '" + name + "' already defined");
    }

    AOIRegion region;
    region.name = name;
    region.bit_index = next_bit_;

    // Parse WKT
    try {
        bg::read_wkt(wkt, region.polygon);
        bg::correct(region.polygon); // Fix orientation and closure
    } catch (const std::exception &e) {
        throw std::runtime_error("AreaOfInterest: Invalid WKT for region '" + name +
                                 "': " + e.what());
    }

    // Cache bounding box for fast rejection
    bg::envelope(region.polygon, region.bounding_box);

    region_bits_[name] = next_bit_;
    impl_->regions.push_back(std::move(region));

    return next_bit_++;
}

int AreaOfInterest::define_region(const std::string &name,
                                  const std::vector<std::pair<Real, Real>> &polygon_points) {
    if (next_bit_ >= 8) {
        throw std::runtime_error("AreaOfInterest: Maximum 8 regions supported");
    }
    if (polygon_points.size() < 3) {
        throw std::runtime_error("AreaOfInterest: Polygon must have at least 3 points");
    }
    if (region_bits_.count(name)) {
        throw std::runtime_error("AreaOfInterest: Region '" + name + "' already defined");
    }

    AOIRegion region;
    region.name = name;
    region.bit_index = next_bit_;

    // Build polygon from points
    Ring2D &outer = region.polygon.outer();
    outer.reserve(polygon_points.size() + 1);
    for (const auto &pt : polygon_points) {
        outer.emplace_back(pt.first, pt.second);
    }
    // Close the ring if not already closed
    if (outer.front().get<0>() != outer.back().get<0>() ||
        outer.front().get<1>() != outer.back().get<1>()) {
        outer.push_back(outer.front());
    }
    bg::correct(region.polygon);

    // Cache bounding box
    bg::envelope(region.polygon, region.bounding_box);

    region_bits_[name] = next_bit_;
    impl_->regions.push_back(std::move(region));

    return next_bit_++;
}

void AreaOfInterest::remove_region(const std::string &name) {
    auto it = region_bits_.find(name);
    if (it == region_bits_.end())
        return;

    int bit_to_remove = it->second;
    region_bits_.erase(it);

    // Remove from regions vector
    auto &regions = impl_->regions;
    regions.erase(std::remove_if(
                      regions.begin(), regions.end(),
                      [bit_to_remove](const AOIRegion &r) { return r.bit_index == bit_to_remove; }),
                  regions.end());

    // Clear the bit from all flags
    AOIFlags clear_mask = ~static_cast<AOIFlags>(1 << bit_to_remove);
    for (auto &f : flags_) {
        f &= clear_mask;
    }
}

void AreaOfInterest::clear_regions() {
    region_bits_.clear();
    impl_->regions.clear();
    next_bit_ = 0;
    std::fill(flags_.begin(), flags_.end(), AOI_NONE);
}

// =============================================================================
// Membership computation
// =============================================================================

AOIFlags AreaOfInterest::compute_element_flags(const ElementBounds &bounds) const {
    AOIFlags flags = AOI_NONE;

    // Element center (2D)
    Point2D center(0.5 * (bounds.xmin + bounds.xmax), 0.5 * (bounds.ymin + bounds.ymax));

    // Element bounding box (for intersection test)
    Box2D elem_box(Point2D(bounds.xmin, bounds.ymin), Point2D(bounds.xmax, bounds.ymax));

    for (const auto &region : impl_->regions) {
        // Fast bounding box rejection
        if (!bg::intersects(elem_box, region.bounding_box)) {
            continue;
        }

        // Check if element center is within polygon
        if (bg::within(center, region.polygon)) {
            flags |= static_cast<AOIFlags>(1 << region.bit_index);
        }
    }

    return flags;
}

void AreaOfInterest::rebuild(const OctreeAdapter &mesh) {
    Index n = mesh.num_elements();
    flags_.resize(n);

    for (Index e = 0; e < n; ++e) {
        flags_[e] = compute_element_flags(mesh.element_bounds(e));
    }
}

void AreaOfInterest::on_refine(const OctreeAdapter &mesh, const std::vector<Index> &children) {
    // Ensure storage is large enough
    Index max_child = 0;
    for (Index c : children) {
        max_child = std::max(max_child, c);
    }
    if (max_child >= static_cast<Index>(flags_.size())) {
        flags_.resize(max_child + 1, AOI_NONE);
    }

    // Recompute membership from geometry for each child
    for (Index c : children) {
        flags_[c] = compute_element_flags(mesh.element_bounds(c));
    }
}

void AreaOfInterest::on_coarsen(const std::vector<Index> &children, Index parent_elem) {
    // Parent gets union of children's flags
    AOIFlags combined = AOI_NONE;
    for (Index c : children) {
        if (c < static_cast<Index>(flags_.size())) {
            combined |= flags_[c];
        }
    }
    if (parent_elem >= static_cast<Index>(flags_.size())) {
        flags_.resize(parent_elem + 1, AOI_NONE);
    }
    flags_[parent_elem] = combined;
}

void AreaOfInterest::resize(Index num_elements) { flags_.resize(num_elements, AOI_NONE); }

// =============================================================================
// Membership queries
// =============================================================================

bool AreaOfInterest::is_in_region(Index elem, const std::string &region_name) const {
    auto it = region_bits_.find(region_name);
    if (it == region_bits_.end())
        return false;
    return is_in_region(elem, it->second);
}

bool AreaOfInterest::is_in_region(Index elem, int bit_index) const {
    if (elem < 0 || elem >= static_cast<Index>(flags_.size()))
        return false;
    if (bit_index < 0 || bit_index >= 8)
        return false;
    return (flags_[elem] & (1 << bit_index)) != 0;
}

bool AreaOfInterest::is_in_any_region(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(flags_.size()))
        return false;
    return flags_[elem] != AOI_NONE;
}

AOIFlags AreaOfInterest::get_flags(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(flags_.size()))
        return AOI_NONE;
    return flags_[elem];
}

int AreaOfInterest::get_region_index(const std::string &name) const {
    auto it = region_bits_.find(name);
    return (it != region_bits_.end()) ? it->second : -1;
}

std::vector<std::string> AreaOfInterest::get_region_names(Index elem) const {
    std::vector<std::string> names;
    AOIFlags flags = get_flags(elem);
    if (flags == AOI_NONE)
        return names;

    for (const auto &region : impl_->regions) {
        if (flags & (1 << region.bit_index)) {
            names.push_back(region.name);
        }
    }
    return names;
}

// =============================================================================
// Bulk queries
// =============================================================================

std::vector<Index> AreaOfInterest::elements_in_region(const std::string &region_name) const {
    std::vector<Index> elements;
    int bit = get_region_index(region_name);
    if (bit < 0)
        return elements;

    AOIFlags mask = static_cast<AOIFlags>(1 << bit);
    for (Index e = 0; e < static_cast<Index>(flags_.size()); ++e) {
        if (flags_[e] & mask) {
            elements.push_back(e);
        }
    }
    return elements;
}

Index AreaOfInterest::count_in_region(const std::string &region_name) const {
    int bit = get_region_index(region_name);
    if (bit < 0)
        return 0;

    Index count = 0;
    AOIFlags mask = static_cast<AOIFlags>(1 << bit);
    for (AOIFlags f : flags_) {
        if (f & mask)
            ++count;
    }
    return count;
}

std::vector<Real> AreaOfInterest::as_cell_data(const std::string &region_name) const {
    std::vector<Real> data(flags_.size(), 0.0);
    int bit = get_region_index(region_name);
    if (bit < 0)
        return data;

    AOIFlags mask = static_cast<AOIFlags>(1 << bit);
    for (Index e = 0; e < static_cast<Index>(flags_.size()); ++e) {
        data[e] = (flags_[e] & mask) ? 1.0 : 0.0;
    }
    return data;
}

std::vector<Real> AreaOfInterest::combined_region_ids() const {
    std::vector<Real> data(flags_.size());
    for (Index e = 0; e < static_cast<Index>(flags_.size()); ++e) {
        data[e] = static_cast<Real>(flags_[e]);
    }
    return data;
}

// =============================================================================
// Configuration
// =============================================================================

int AreaOfInterest::num_regions() const { return static_cast<int>(impl_->regions.size()); }

std::vector<std::string> AreaOfInterest::region_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->regions.size());
    for (const auto &region : impl_->regions) {
        names.push_back(region.name);
    }
    return names;
}

bool AreaOfInterest::has_region(const std::string &name) const {
    return region_bits_.count(name) > 0;
}

} // namespace drifter
