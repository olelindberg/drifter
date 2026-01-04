#include "mesh/coastline_refinement.hpp"
#include <iostream>

namespace drifter {

// =============================================================================
// CoastlineReader implementation
// =============================================================================

#ifdef DRIFTER_HAS_GDAL

namespace {

void ogr_linear_ring_to_boost_ring(const OGRLinearRing* lr, Ring2D& ring) {
    if (!lr) return;
    int n = lr->getNumPoints();
    // OGR rings are closed (last == first). Skip the last duplicate point.
    int limit = (n >= 2) ? n - 1 : n;
    ring.clear();
    ring.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        ring.emplace_back(lr->getX(i), lr->getY(i));
    }
}

bool ogr_polygon_to_boost_polygon(const OGRPolygon* opoly, Polygon2D& bp) {
    if (!opoly) return false;

    // Exterior ring
    const OGRLinearRing* ext = opoly->getExteriorRing();
    if (!ext || ext->getNumPoints() < 4) return false;

    Ring2D outer;
    ogr_linear_ring_to_boost_ring(ext, outer);
    bp.outer().assign(outer.begin(), outer.end());

    // Interior rings (holes)
    bp.inners().clear();
    const int nh = opoly->getNumInteriorRings();
    bp.inners().reserve(nh);
    for (int i = 0; i < nh; ++i) {
        const OGRLinearRing* in = opoly->getInteriorRing(i);
        if (!in || in->getNumPoints() < 4) continue;
        Ring2D inner;
        ogr_linear_ring_to_boost_ring(in, inner);
        bp.inners().emplace_back();
        bp.inners().back().assign(inner.begin(), inner.end());
    }

    // Fix orientation/closure to Boost expectations
    bg::correct(bp);
    return true;
}

void collect_polygons(const OGRGeometry* g, std::vector<const OGRPolygon*>& out) {
    if (!g) return;
    OGRwkbGeometryType t = wkbFlatten(g->getGeometryType());

    if (t == wkbPolygon) {
        out.push_back(g->toPolygon());
        return;
    }
    if (t == wkbMultiPolygon) {
        const OGRMultiPolygon* mp = g->toMultiPolygon();
        for (int i = 0; i < mp->getNumGeometries(); ++i) {
            out.push_back(mp->getGeometryRef(i));
        }
        return;
    }
    if (t == wkbGeometryCollection) {
        const OGRGeometryCollection* gc = g->toGeometryCollection();
        for (int i = 0; i < gc->getNumGeometries(); ++i) {
            collect_polygons(gc->getGeometryRef(i), out);
        }
        return;
    }
    // Ignore lines/points
}

}  // anonymous namespace

bool CoastlineReader::load(const std::string& filename,
                           const std::string& layer_name,
                           const std::string& target_srs) {
    GDALAllRegister();

    std::unique_ptr<GDALDataset> ds(
        static_cast<GDALDataset*>(GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    if (!ds) {
        error_ = "Failed to open: " + filename;
        return false;
    }

    OGRLayer* layer = nullptr;
    if (!layer_name.empty()) {
        layer = ds->GetLayerByName(layer_name.c_str());
        if (!layer) {
            error_ = "Layer not found: " + layer_name;
            return false;
        }
    } else {
        layer = ds->GetLayer(0);
        if (!layer) {
            error_ = "No layers in dataset";
            return false;
        }
    }

    // Optional coordinate transformation
    std::unique_ptr<OGRCoordinateTransformation> coord_tx;
    if (!target_srs.empty()) {
        OGRSpatialReference srcSRS = *layer->GetSpatialRef();
        OGRSpatialReference dstSRS;
        if (dstSRS.SetFromUserInput(target_srs.c_str()) != OGRERR_NONE) {
            error_ = "Invalid target SRS: " + target_srs;
            return false;
        }
        if (layer->GetSpatialRef()) {
            coord_tx.reset(OGRCreateCoordinateTransformation(&srcSRS, &dstSRS));
            if (!coord_tx) {
                error_ = "Failed to create coordinate transformation";
                return false;
            }
        }
    }

    polygons_.clear();

    layer->ResetReading();
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        std::unique_ptr<OGRFeature> feat_guard(feat);
        OGRGeometry* geom = feat->GetGeometryRef();
        if (!geom) continue;

        // Work on a 2D clone
        std::unique_ptr<OGRGeometry> g2d(geom->clone());
        g2d->flattenTo2D();

        if (coord_tx) {
            if (g2d->transform(coord_tx.get()) != OGRERR_NONE) {
                continue;  // Skip failed transforms
            }
        }

        std::vector<const OGRPolygon*> polys;
        collect_polygons(g2d.get(), polys);
        for (const OGRPolygon* op : polys) {
            Polygon2D bp;
            if (ogr_polygon_to_boost_polygon(op, bp)) {
                polygons_.push_back(std::move(bp));
            }
        }
    }

    bg::correct(polygons_);
    return true;
}

bool CoastlineReader::is_available() {
    return true;
}

#else  // No GDAL

bool CoastlineReader::load(const std::string& filename,
                           const std::string& layer_name,
                           const std::string& target_srs) {
    error_ = "GDAL not available";
    return false;
}

bool CoastlineReader::is_available() {
    return false;
}

#endif  // DRIFTER_HAS_GDAL

void CoastlineReader::swap_xy() {
    coastline_util::swap_xy(polygons_);
}

void CoastlineReader::remove_small_polygons(double min_area) {
    coastline_util::remove_small_polygons(polygons_, min_area);
}

Box2D CoastlineReader::bounding_box() const {
    Box2D bbox;
    bg::envelope(polygons_, bbox);
    return bbox;
}

// =============================================================================
// CoastlineIndex implementation
// =============================================================================

void CoastlineIndex::build(const MultiPolygon2D& polygons) {
    rtree_ = std::make_shared<SegmentRTree>();
    num_segments_ = 0;

    for (size_t p = 0; p < polygons.size(); ++p) {
        const auto& poly = polygons[p];

        // Outer ring
        const auto& outer = poly.outer();
        for (size_t i = 0; i + 1 < outer.size(); ++i) {
            rtree_->insert({Segment2D(outer[i], outer[i + 1]), {p, 0, i}});
            ++num_segments_;
        }

        // Inner rings (holes)
        for (size_t r = 0; r < poly.inners().size(); ++r) {
            const auto& inner = poly.inners()[r];
            for (size_t i = 0; i + 1 < inner.size(); ++i) {
                rtree_->insert({Segment2D(inner[i], inner[i + 1]), {p, r + 1, i}});
                ++num_segments_;
            }
        }
    }
}

bool CoastlineIndex::intersects(const Box2D& box) const {
    if (!rtree_) return false;
    std::vector<SegmentValue> candidates;
    rtree_->query(bgi::intersects(box), std::back_inserter(candidates));
    return !candidates.empty();
}

bool CoastlineIndex::intersects(Real xmin, Real ymin, Real xmax, Real ymax) const {
    Box2D box(Point2D(xmin, ymin), Point2D(xmax, ymax));
    return intersects(box);
}

std::vector<SegmentValue> CoastlineIndex::query(const Box2D& box) const {
    std::vector<SegmentValue> result;
    if (rtree_) {
        rtree_->query(bgi::intersects(box), std::back_inserter(result));
    }
    return result;
}

// =============================================================================
// CoastlineRefinement implementation
// =============================================================================

CoastlineRefinement::CoastlineRefinement(std::shared_ptr<CoastlineIndex> index, int max_level)
    : index_(std::move(index)), max_level_(max_level) {}

bool CoastlineRefinement::should_refine(const ElementBounds& bounds, int level) const {
    if (level >= max_level_) return false;
    return index_->intersects(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax);
}

RefineMask CoastlineRefinement::get_mask(const ElementBounds& bounds) const {
    if (index_->intersects(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)) {
        return RefineMask::X | RefineMask::Y;  // Refine horizontally
    }
    return RefineMask::NONE;
}

// =============================================================================
// Utility functions
// =============================================================================

namespace coastline_util {

void swap_xy(MultiPolygon2D& mp) {
    for (auto& poly : mp) {
        for (auto& pt : poly.outer()) {
            double x = bg::get<0>(pt);
            double y = bg::get<1>(pt);
            bg::set<0>(pt, y);
            bg::set<1>(pt, x);
        }
        for (auto& inner : poly.inners()) {
            for (auto& pt : inner) {
                double x = bg::get<0>(pt);
                double y = bg::get<1>(pt);
                bg::set<0>(pt, y);
                bg::set<1>(pt, x);
            }
        }
    }
    bg::correct(mp);
}

void remove_small_polygons(MultiPolygon2D& mp, double min_area) {
    for (auto it = mp.begin(); it != mp.end();) {
        if (bg::area(it->outer()) < min_area) {
            it = mp.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace coastline_util

}  // namespace drifter
