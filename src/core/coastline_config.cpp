/// @file coastline_config.cpp
/// @brief Shared coastline configuration parsing

#include "core/coastline_config.hpp"
#include "core/logger.hpp"

namespace pt = boost::property_tree;

namespace drifter {

CoastlineConfig parse_coastline_config(const pt::ptree &tree) {
    CoastlineConfig config;

    config.file                 = tree.get<std::string>("file", config.file);
    config.layer                = tree.get<std::string>("layer", config.layer);
    config.srs                  = tree.get<std::string>("srs", config.srs);
    config.max_level            = tree.get<int>("max_level", config.max_level);
    config.min_polygon_area     = tree.get<Real>("min_polygon_area", config.min_polygon_area);
    config.min_curvature_radius =
        tree.get<Real>("min_curvature_radius", config.min_curvature_radius);

    if (config.min_polygon_area > 0.0) {
        LOG_WARNING("coastline.min_polygon_area = "
                    << config.min_polygon_area
                    << " has no effect: the coastline is indexed as line segments, not polygons");
    }

    return config;
}

pt::ptree serialize_coastline_config(const CoastlineConfig &config) {
    pt::ptree tree;

    tree.put("file", config.file);
    tree.put("layer", config.layer);
    tree.put("srs", config.srs);
    tree.put("max_level", config.max_level);
    tree.put("min_polygon_area", config.min_polygon_area);
    tree.put("min_curvature_radius", config.min_curvature_radius);

    return tree;
}

} // namespace drifter
