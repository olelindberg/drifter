#include "mesh/coastline_refinement.hpp"

// GDAL includes must be outside the drifter namespace
#include <gdal_priv.h>
#include <ogrsf_frmts.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <iostream>
#include <iterator>
#include <vector>

namespace drifter {

// Boost.Geometry types (hidden from header)
namespace bg = boost::geometry;
namespace bgi = bg::index;

using Point2D = bg::model::point<double, 2, bg::cs::cartesian>;
using Segment2D = bg::model::segment<Point2D>;
using Box2D = bg::model::box<Point2D>;
using Ring2D = bg::model::ring<Point2D>;
using Polygon2D = bg::model::polygon<Point2D, true>; // Clockwise
using MultiPolygon2D = bg::model::multi_polygon<Polygon2D>;

struct SegmentInfo {
    size_t polygon_index;
    size_t ring_index;
    size_t segment_index;
};

using SegmentValue = std::pair<Segment2D, SegmentInfo>;
using SegmentRTree = bgi::rtree<SegmentValue, bgi::rstar<16>>;

// PIMPL implementation structs
struct CoastlineReader::Impl {
    MultiPolygon2D polygons;
    std::string error;
};

struct CoastlineIndex::Impl {
    std::shared_ptr<SegmentRTree> rtree;
    size_t num_segments = 0;
};

// Internal utility functions (moved from public header)
namespace {

void swap_xy_impl(MultiPolygon2D &mp) {
    for (auto &poly : mp) {
        for (auto &pt : poly.outer()) {
            double x = bg::get<0>(pt);
            double y = bg::get<1>(pt);
            bg::set<0>(pt, y);
            bg::set<1>(pt, x);
        }
        for (auto &inner : poly.inners()) {
            for (auto &pt : inner) {
                double x = bg::get<0>(pt);
                double y = bg::get<1>(pt);
                bg::set<0>(pt, y);
                bg::set<1>(pt, x);
            }
        }
    }
    bg::correct(mp);
}

void remove_small_polygons_impl(MultiPolygon2D &mp, double min_area) {
    for (auto it = mp.begin(); it != mp.end();) {
        if (bg::area(it->outer()) < min_area) {
            it = mp.erase(it);
        } else {
            ++it;
        }
    }
}

} // anonymous namespace

// CoastlineReader constructor/destructor
CoastlineReader::CoastlineReader() : impl_(std::make_unique<Impl>()) {}
CoastlineReader::~CoastlineReader() = default;
CoastlineReader::CoastlineReader(CoastlineReader &&) noexcept = default;
CoastlineReader &
CoastlineReader::operator=(CoastlineReader &&) noexcept = default;

// CoastlineIndex constructor/destructor
CoastlineIndex::CoastlineIndex() : impl_(std::make_unique<Impl>()) {}
CoastlineIndex::~CoastlineIndex() = default;
CoastlineIndex::CoastlineIndex(CoastlineIndex &&) noexcept = default;
CoastlineIndex &CoastlineIndex::operator=(CoastlineIndex &&) noexcept = default;

// =============================================================================
// CoastlineReader implementation
// =============================================================================

namespace {

void ogr_linear_ring_to_boost_ring(const OGRLinearRing *lr, Ring2D &ring) {
    if (!lr)
        return;
    int n = lr->getNumPoints();
    // OGR rings are closed (last == first). Skip the last duplicate point.
    int limit = (n >= 2) ? n - 1 : n;
    ring.clear();
    ring.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        ring.emplace_back(lr->getX(i), lr->getY(i));
    }
}

bool ogr_polygon_to_boost_polygon(const OGRPolygon *opoly, Polygon2D &bp) {
    if (!opoly)
        return false;

    // Exterior ring
    const OGRLinearRing *ext = opoly->getExteriorRing();
    if (!ext || ext->getNumPoints() < 4)
        return false;

    Ring2D outer;
    ogr_linear_ring_to_boost_ring(ext, outer);
    bp.outer().assign(outer.begin(), outer.end());

    // Interior rings (holes)
    bp.inners().clear();
    const int nh = opoly->getNumInteriorRings();
    bp.inners().reserve(nh);
    for (int i = 0; i < nh; ++i) {
        const OGRLinearRing *in = opoly->getInteriorRing(i);
        if (!in || in->getNumPoints() < 4)
            continue;
        Ring2D inner;
        ogr_linear_ring_to_boost_ring(in, inner);
        bp.inners().emplace_back();
        bp.inners().back().assign(inner.begin(), inner.end());
    }

    // Fix orientation/closure to Boost expectations
    bg::correct(bp);
    return true;
}

void collect_polygons(
    const OGRGeometry *g, std::vector<const OGRPolygon *> &out) {
    if (!g)
        return;
    OGRwkbGeometryType t = wkbFlatten(g->getGeometryType());

    if (t == wkbPolygon) {
        out.push_back(g->toPolygon());
        return;
    }
    if (t == wkbMultiPolygon) {
        const OGRMultiPolygon *mp = g->toMultiPolygon();
        for (int i = 0; i < mp->getNumGeometries(); ++i) {
            out.push_back(mp->getGeometryRef(i));
        }
        return;
    }
    if (t == wkbGeometryCollection) {
        const OGRGeometryCollection *gc = g->toGeometryCollection();
        for (int i = 0; i < gc->getNumGeometries(); ++i) {
            collect_polygons(gc->getGeometryRef(i), out);
        }
        return;
    }
    // Ignore lines/points
}

} // anonymous namespace

bool CoastlineReader::load(
    const std::string &filename, const std::string &layer_name,
    const std::string &target_srs) {
    GDALAllRegister();

    std::unique_ptr<GDALDataset> ds(static_cast<GDALDataset *>(GDALOpenEx(
        filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    if (!ds) {
        impl_->error = "Failed to open: " + filename;
        return false;
    }

    OGRLayer *layer = nullptr;
    if (!layer_name.empty()) {
        layer = ds->GetLayerByName(layer_name.c_str());
        if (!layer) {
            impl_->error = "Layer not found: " + layer_name;
            return false;
        }
    } else {
        layer = ds->GetLayer(0);
        if (!layer) {
            impl_->error = "No layers in dataset";
            return false;
        }
    }

    // Optional coordinate transformation
    std::unique_ptr<OGRCoordinateTransformation> coord_tx;
    if (!target_srs.empty()) {
        OGRSpatialReference srcSRS = *layer->GetSpatialRef();
        OGRSpatialReference dstSRS;
        if (dstSRS.SetFromUserInput(target_srs.c_str()) != OGRERR_NONE) {
            impl_->error = "Invalid target SRS: " + target_srs;
            return false;
        }
        if (layer->GetSpatialRef()) {
            coord_tx.reset(OGRCreateCoordinateTransformation(&srcSRS, &dstSRS));
            if (!coord_tx) {
                impl_->error = "Failed to create coordinate transformation";
                return false;
            }
        }
    }

    impl_->polygons.clear();

    layer->ResetReading();
    OGRFeature *feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        std::unique_ptr<OGRFeature> feat_guard(feat);
        OGRGeometry *geom = feat->GetGeometryRef();
        if (!geom)
            continue;

        // Work on a 2D clone
        std::unique_ptr<OGRGeometry> g2d(geom->clone());
        g2d->flattenTo2D();

        if (coord_tx) {
            if (g2d->transform(coord_tx.get()) != OGRERR_NONE) {
                continue; // Skip failed transforms
            }
        }

        std::vector<const OGRPolygon *> polys;
        collect_polygons(g2d.get(), polys);
        for (const OGRPolygon *op : polys) {
            Polygon2D bp;
            if (ogr_polygon_to_boost_polygon(op, bp)) {
                impl_->polygons.push_back(std::move(bp));
            }
        }
    }

    bg::correct(impl_->polygons);
    return true;
}

bool CoastlineReader::is_available() { return true; }

void CoastlineReader::swap_xy() { swap_xy_impl(impl_->polygons); }

void CoastlineReader::remove_small_polygons(double min_area) {
    remove_small_polygons_impl(impl_->polygons, min_area);
}

size_t CoastlineReader::num_polygons() const { return impl_->polygons.size(); }

const std::string &CoastlineReader::last_error() const { return impl_->error; }

void CoastlineReader::bounding_box(
    Real &xmin, Real &ymin, Real &xmax, Real &ymax) const {
    Box2D bbox;
    bg::envelope(impl_->polygons, bbox);
    xmin = bg::get<0>(bbox.min_corner());
    ymin = bg::get<1>(bbox.min_corner());
    xmax = bg::get<0>(bbox.max_corner());
    ymax = bg::get<1>(bbox.max_corner());
}

std::shared_ptr<CoastlineIndex> CoastlineReader::build_index() const {
    auto index = std::make_shared<CoastlineIndex>();

    // Build the R-tree index directly (CoastlineReader is a friend of
    // CoastlineIndex)
    index->impl_->rtree = std::make_shared<SegmentRTree>();
    index->impl_->num_segments = 0;

    for (size_t p = 0; p < impl_->polygons.size(); ++p) {
        const auto &poly = impl_->polygons[p];

        // Outer ring
        const auto &outer = poly.outer();
        for (size_t i = 0; i + 1 < outer.size(); ++i) {
            index->impl_->rtree->insert(
                {Segment2D(outer[i], outer[i + 1]), {p, 0, i}});
            ++index->impl_->num_segments;
        }

        // Inner rings (holes)
        for (size_t r = 0; r < poly.inners().size(); ++r) {
            const auto &inner = poly.inners()[r];
            for (size_t i = 0; i + 1 < inner.size(); ++i) {
                index->impl_->rtree->insert(
                    {Segment2D(inner[i], inner[i + 1]), {p, r + 1, i}});
                ++index->impl_->num_segments;
            }
        }
    }

    return index;
}

// =============================================================================
// CoastlineIndex implementation
// =============================================================================

size_t CoastlineIndex::num_segments() const { return impl_->num_segments; }

bool CoastlineIndex::intersects(
    Real xmin, Real ymin, Real xmax, Real ymax) const {
    if (!impl_->rtree)
        return false;
    Box2D box(Point2D(xmin, ymin), Point2D(xmax, ymax));
    std::vector<SegmentValue> candidates;
    impl_->rtree->query(bgi::intersects(box), std::back_inserter(candidates));
    return !candidates.empty();
}

// =============================================================================
// CoastlineRefinement implementation
// =============================================================================

CoastlineRefinement::CoastlineRefinement(
    std::shared_ptr<CoastlineIndex> index, int max_level)
    : index_(std::move(index)), max_level_(max_level) {}

bool CoastlineRefinement::should_refine(
    const ElementBounds &bounds, int level) const {
    if (level >= max_level_)
        return false;
    return index_->intersects(
        bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax);
}

RefineMask CoastlineRefinement::get_mask(const ElementBounds &bounds) const {
    if (index_->intersects(
            bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)) {
        return RefineMask::X | RefineMask::Y; // Refine horizontally
    }
    return RefineMask::NONE;
}

} // namespace drifter
