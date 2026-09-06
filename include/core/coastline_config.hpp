#pragma once

/// @file coastline_config.hpp
/// @brief Coastline refinement configuration, shared by the highrider and lowrider apps
///
/// Both apps drive the same circumradius-based coastline pre-pass over their own mesh
/// generator, so they read the same JSON section rather than each carrying a copy of
/// the schema. The parse/serialize helpers live here for the same reason.

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include "core/types.hpp"
#include <boost/property_tree/ptree.hpp>
#include <string>

namespace drifter {

/// @brief Coastline refinement configuration
struct CoastlineConfig {
    std::string file;  ///< Path to shapefile/GeoPackage (empty = disabled)
    std::string layer; ///< Layer name (optional, defaults to first)
    std::string srs;   ///< Target SRS (e.g., "EPSG:3034")
    int max_level = 10; ///< Max refinement level near coastline

    /// Filter small polygons (0 = no filter).
    ///
    /// Currently inert: CoastlineReader stores open polylines rather than
    /// polygons, so CoastlineReader::remove_small_polygons() has nothing to
    /// measure an area over and does nothing. Setting it is warned about at load.
    Real min_polygon_area = 0.0;

    /// The refinement criterion itself carries no parameter: an element is refined
    /// while it holds a coastline vertex whose discrete circumradius is smaller
    /// than the element's own shorter side. How far that runs is bounded by
    /// max_level and by the data-resolution limits of the mesh generator.

    bool enabled() const { return !file.empty(); }
};

/// @brief Parse a CoastlineConfig from the contents of a JSON "coastline" section
CoastlineConfig parse_coastline_config(const boost::property_tree::ptree &tree);

/// @brief Serialize a CoastlineConfig back to a JSON "coastline" section
boost::property_tree::ptree serialize_coastline_config(const CoastlineConfig &config);

} // namespace drifter
