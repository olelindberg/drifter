#pragma once

// Coastline-based adaptive mesh refinement
// Uses land polygon boundaries to refine elements near the coastline
// Adapted from SeaMesh's land_polygon_refinement

#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"

#ifdef DRIFTER_HAS_GDAL
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#endif

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <memory>
#include <string>
#include <vector>

namespace drifter {

namespace bg = boost::geometry;
namespace bgi = bg::index;

// Boost Geometry types for coastline representation
using Point2D = bg::model::point<double, 2, bg::cs::cartesian>;
using Segment2D = bg::model::segment<Point2D>;
using Box2D = bg::model::box<Point2D>;
using Ring2D = bg::model::ring<Point2D>;
using Polygon2D = bg::model::polygon<Point2D, true>;  // Clockwise
using MultiPolygon2D = bg::model::multi_polygon<Polygon2D>;

/// Segment metadata for R-tree queries
struct SegmentInfo {
    size_t polygon_index;
    size_t ring_index;    // 0 = outer ring, 1..N = inner rings (holes)
    size_t segment_index;
};

using SegmentValue = std::pair<Segment2D, SegmentInfo>;
using SegmentRTree = bgi::rtree<SegmentValue, bgi::rstar<16>>;

/// @brief Loads land polygons from GeoPackage files
/// Supports coordinate transformation to target SRS
class CoastlineReader {
public:
    CoastlineReader() = default;

    /// @brief Load land polygons from a GeoPackage file
    /// @param filename Path to the .gpkg file
    /// @param layer_name Layer name to read (e.g., "landpolygon_2500")
    /// @param target_srs Target spatial reference system (e.g., "EPSG:3034")
    /// @return true if successful
    bool load(const std::string& filename,
              const std::string& layer_name = "",
              const std::string& target_srs = "");

    /// @brief Get the loaded multi-polygon
    const MultiPolygon2D& polygons() const { return polygons_; }

    /// @brief Get the number of polygons
    size_t num_polygons() const { return polygons_.size(); }

    /// @brief Swap X and Y coordinates (needed for some coordinate systems)
    void swap_xy();

    /// @brief Remove polygons smaller than a given area
    void remove_small_polygons(double min_area);

    /// @brief Get bounding box of all polygons
    Box2D bounding_box() const;

    /// @brief Check if GDAL/OGR is available
    static bool is_available();

    /// @brief Get last error message
    const std::string& last_error() const { return error_; }

private:
    MultiPolygon2D polygons_;
    std::string error_;
};

/// @brief R-tree based coastline segment index for fast intersection queries
class CoastlineIndex {
public:
    /// @brief Build index from multi-polygon
    void build(const MultiPolygon2D& polygons);

    /// @brief Check if a box intersects any coastline segment
    bool intersects(const Box2D& box) const;

    /// @brief Check if a box intersects any coastline segment
    bool intersects(Real xmin, Real ymin, Real xmax, Real ymax) const;

    /// @brief Get all segments intersecting a box
    std::vector<SegmentValue> query(const Box2D& box) const;

    /// @brief Get number of segments in the index
    size_t num_segments() const { return num_segments_; }

private:
    std::shared_ptr<SegmentRTree> rtree_;
    size_t num_segments_ = 0;
};

/// @brief Coastline-based refinement criterion for adaptive meshing
/// Refines elements that intersect with the coastline (land polygon boundary)
class CoastlineRefinement {
public:
    /// @brief Construct from coastline index
    /// @param index Coastline segment R-tree index
    /// @param max_level Maximum refinement level
    CoastlineRefinement(std::shared_ptr<CoastlineIndex> index, int max_level);

    /// @brief Check if an element should be refined
    /// @param bounds Element bounds
    /// @param level Current refinement level
    /// @return true if element intersects coastline and level < max_level
    bool should_refine(const ElementBounds& bounds, int level) const;

    /// @brief Get refinement mask (which axes to refine)
    /// Refines in X and Y directions for coastline
    RefineMask get_mask(const ElementBounds& bounds) const;

private:
    std::shared_ptr<CoastlineIndex> index_;
    int max_level_;
};

/// @brief Utility functions for coastline geometry
namespace coastline_util {

/// @brief Convert OGR polygon to Boost Geometry polygon
bool ogr_to_boost_polygon(const void* ogr_polygon, Polygon2D& out);

/// @brief Swap X and Y coordinates in a multi-polygon
void swap_xy(MultiPolygon2D& mp);

/// @brief Remove small polygons from a multi-polygon
void remove_small_polygons(MultiPolygon2D& mp, double min_area);

}  // namespace coastline_util

}  // namespace drifter
