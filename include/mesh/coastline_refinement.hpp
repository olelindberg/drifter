#pragma once

// Coastline-based adaptive mesh refinement
// Uses land polygon boundaries to refine elements near the coastline
// Adapted from SeaMesh's land_polygon_refinement

#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"

#include <memory>
#include <string>

namespace drifter {

/// @brief Configuration for curvature comb visualization
struct CurvatureCombConfig {
    Real scale = 0.01;  // Scale factor for comb line lengths
};

// Forward declarations for PIMPL
class CoastlineIndex;

/// @brief Loads land polygons from GeoPackage files
/// Supports coordinate transformation to target SRS
class CoastlineReader {
public:
    CoastlineReader();
    ~CoastlineReader();

    // Move-only (PIMPL)
    CoastlineReader(const CoastlineReader &) = delete;
    CoastlineReader &operator=(const CoastlineReader &) = delete;
    CoastlineReader(CoastlineReader &&) noexcept;
    CoastlineReader &operator=(CoastlineReader &&) noexcept;

    /// @brief Load land polygons from a GeoPackage file
    /// @param filename Path to the .gpkg file
    /// @param layer_name Layer name to read (e.g., "landpolygon_2500")
    /// @param target_srs Target spatial reference system (e.g., "EPSG:3034")
    /// @return true if successful
    bool load(const std::string &filename, const std::string &layer_name = "",
              const std::string &target_srs = "");

    /// @brief Load land polygons with spatial filter (for large global datasets)
    /// Only features intersecting the domain bounds are loaded.
    /// @param filename Path to the vector file (shapefile, GeoPackage, etc.)
    /// @param layer_name Layer name to read (empty = first layer)
    /// @param target_srs Target spatial reference system (e.g., "EPSG:3034")
    /// @param domain_xmin, domain_ymin, domain_xmax, domain_ymax Domain bounds in target SRS
    /// @return true if successful
    bool load(const std::string &filename, const std::string &layer_name,
              const std::string &target_srs,
              Real domain_xmin, Real domain_ymin, Real domain_xmax, Real domain_ymax);

    /// @brief Get the number of polygons
    size_t num_polygons() const;

    /// @brief Swap X and Y coordinates (needed for some coordinate systems)
    void swap_xy();

    /// @brief Remove polygons smaller than a given area
    void remove_small_polygons(double min_area);

    /// @brief Get bounding box of all polygons
    /// @param xmin, ymin, xmax, ymax Output parameters for bounds
    void bounding_box(Real &xmin, Real &ymin, Real &xmax, Real &ymax) const;

    /// @brief Build coastline index from loaded polygons
    /// @return Shared pointer to the built index
    std::shared_ptr<CoastlineIndex> build_index() const;

    /// @brief Build coastline index filtered to domain bounds
    /// Only segments intersecting the domain bounding box are indexed.
    /// This is critical for performance with global datasets.
    /// @param xmin, ymin, xmax, ymax Domain bounding box
    /// @return Shared pointer to the built index
    std::shared_ptr<CoastlineIndex> build_index(Real xmin, Real ymin,
                                                 Real xmax, Real ymax) const;

    /// @brief Write loaded polygons to VTK file (for debugging)
    /// @param filename Output filename (without extension)
    void write_vtk(const std::string &filename) const;

    /// @brief Write curvature comb visualization to VTK file
    /// Shows curvature radius at each coastline vertex as line segments
    /// extending in the normal direction (toward center of curvature)
    /// @param filename Output filename (without extension, .vtp added)
    /// @param config Visualization parameters (scale, radius limits)
    void write_curvature_comb_vtk(const std::string &filename,
                                   const CurvatureCombConfig &config = {}) const;

    /// @brief Check if GDAL/OGR is available
    static bool is_available();

    /// @brief Get last error message
    const std::string &last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief R-tree based coastline segment index for fast intersection queries
class CoastlineIndex {
public:
    CoastlineIndex();
    ~CoastlineIndex();

    // Move-only (PIMPL)
    CoastlineIndex(const CoastlineIndex &) = delete;
    CoastlineIndex &operator=(const CoastlineIndex &) = delete;
    CoastlineIndex(CoastlineIndex &&) noexcept;
    CoastlineIndex &operator=(CoastlineIndex &&) noexcept;

    /// @brief Check if a box intersects any coastline segment
    bool intersects(Real xmin, Real ymin, Real xmax, Real ymax) const;

    /// @brief Query minimum curvature radius within a bounding box
    /// @param xmin, ymin, xmax, ymax Query bounding box
    /// @param min_radius Minimum radius threshold (ignore smaller values as noise)
    /// @return Minimum curvature radius >= min_radius in the box,
    ///         or infinity if no qualifying curvature points are present
    Real min_curvature_radius(Real xmin, Real ymin, Real xmax, Real ymax,
                               Real min_radius = 0.0) const;

    /// @brief Get number of segments in the index
    size_t num_segments() const;

    /// @brief Get number of curvature points in the index
    size_t num_curvature_points() const;

private:
    friend class CoastlineReader;

    struct Impl;
    std::unique_ptr<Impl> impl_;
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
    bool should_refine(const ElementBounds &bounds, int level) const;

    /// @brief Get refinement mask (which axes to refine)
    /// Refines in X and Y directions for coastline
    RefineMask get_mask(const ElementBounds &bounds) const;

private:
    std::shared_ptr<CoastlineIndex> index_;
    int max_level_;
};

} // namespace drifter
