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

#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
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
    std::vector<Segment2D> segments;  // Store segments directly, no polygon overhead
    std::string error;
};

struct CoastlineIndex::Impl {
    std::shared_ptr<SegmentRTree> rtree;
    size_t num_segments = 0;
};

// Internal utility functions (moved from public header)
namespace {

void swap_xy_segments(std::vector<Segment2D> &segments) {
    for (auto &seg : segments) {
        Point2D p1 = seg.first;
        Point2D p2 = seg.second;
        seg.first = Point2D(bg::get<1>(p1), bg::get<0>(p1));
        seg.second = Point2D(bg::get<1>(p2), bg::get<0>(p2));
    }
}

// Helper to add segments from OGR geometry
void add_segments_from_linestring(const OGRLineString* ls,
                                   OGRCoordinateTransformation* coord_tx,
                                   std::vector<Segment2D>& out) {
    if (!ls || ls->getNumPoints() < 2) return;

    int n = ls->getNumPoints();
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = ls->getX(i);
        y[i] = ls->getY(i);
    }

    // Transform if needed
    if (coord_tx) {
        if (!coord_tx->Transform(n, x.data(), y.data())) {
            return;  // Skip failed transform
        }
    }

    // Add segments
    for (int i = 0; i + 1 < n; ++i) {
        out.emplace_back(Point2D(x[i], y[i]), Point2D(x[i+1], y[i+1]));
    }
}

void add_segments_from_polygon(const OGRPolygon* poly,
                                OGRCoordinateTransformation* coord_tx,
                                std::vector<Segment2D>& out) {
    if (!poly) return;

    // Exterior ring
    const OGRLinearRing* ext = poly->getExteriorRing();
    if (ext) {
        add_segments_from_linestring(ext, coord_tx, out);
    }

    // Interior rings
    for (int i = 0; i < poly->getNumInteriorRings(); ++i) {
        const OGRLinearRing* inner = poly->getInteriorRing(i);
        if (inner) {
            add_segments_from_linestring(inner, coord_tx, out);
        }
    }
}

void collect_segments_recursive(const OGRGeometry* g,
                                 OGRCoordinateTransformation* coord_tx,
                                 std::vector<Segment2D>& out) {
    if (!g) return;
    OGRwkbGeometryType t = wkbFlatten(g->getGeometryType());

    if (t == wkbLineString) {
        add_segments_from_linestring(g->toLineString(), coord_tx, out);
    } else if (t == wkbPolygon) {
        add_segments_from_polygon(g->toPolygon(), coord_tx, out);
    } else if (t == wkbMultiLineString) {
        const OGRMultiLineString* mls = g->toMultiLineString();
        for (int i = 0; i < mls->getNumGeometries(); ++i) {
            add_segments_from_linestring(mls->getGeometryRef(i)->toLineString(), coord_tx, out);
        }
    } else if (t == wkbMultiPolygon) {
        const OGRMultiPolygon* mp = g->toMultiPolygon();
        for (int i = 0; i < mp->getNumGeometries(); ++i) {
            add_segments_from_polygon(mp->getGeometryRef(i), coord_tx, out);
        }
    } else if (t == wkbGeometryCollection) {
        const OGRGeometryCollection* gc = g->toGeometryCollection();
        for (int i = 0; i < gc->getNumGeometries(); ++i) {
            collect_segments_recursive(gc->getGeometryRef(i), coord_tx, out);
        }
    }
}

} // anonymous namespace

// CoastlineReader constructor/destructor
CoastlineReader::CoastlineReader() : impl_(std::make_unique<Impl>()) {}
CoastlineReader::~CoastlineReader() = default;
CoastlineReader::CoastlineReader(CoastlineReader &&) noexcept = default;
CoastlineReader &CoastlineReader::operator=(CoastlineReader &&) noexcept = default;

// CoastlineIndex constructor/destructor
CoastlineIndex::CoastlineIndex() : impl_(std::make_unique<Impl>()) {}
CoastlineIndex::~CoastlineIndex() = default;
CoastlineIndex::CoastlineIndex(CoastlineIndex &&) noexcept = default;
CoastlineIndex &CoastlineIndex::operator=(CoastlineIndex &&) noexcept = default;

// =============================================================================
// CoastlineReader implementation
// =============================================================================

bool CoastlineReader::load(const std::string &filename, const std::string &layer_name,
                           const std::string &target_srs) {
    GDALAllRegister();

    std::unique_ptr<GDALDataset> ds(static_cast<GDALDataset*>(
        GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    if (!ds) {
        impl_->error = "Failed to open: " + filename;
        return false;
    }

    OGRLayer* layer = nullptr;
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
    if (!target_srs.empty() && layer->GetSpatialRef()) {
        OGRSpatialReference srcSRS = *layer->GetSpatialRef();
        OGRSpatialReference dstSRS;
        if (dstSRS.SetFromUserInput(target_srs.c_str()) != OGRERR_NONE) {
            impl_->error = "Invalid target SRS: " + target_srs;
            return false;
        }
        // Use traditional GIS axis order (x=lon/easting, y=lat/northing)
        srcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        dstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        coord_tx.reset(OGRCreateCoordinateTransformation(&srcSRS, &dstSRS));
        if (!coord_tx) {
            impl_->error = "Failed to create coordinate transformation";
            return false;
        }
    }

    impl_->segments.clear();

    layer->ResetReading();
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        std::unique_ptr<OGRFeature> feat_guard(feat);
        OGRGeometry* geom = feat->GetGeometryRef();
        if (!geom) continue;

        // Work on a 2D clone
        std::unique_ptr<OGRGeometry> g2d(geom->clone());
        g2d->flattenTo2D();

        // Collect all segments directly (handles LineStrings, Polygons, etc.)
        collect_segments_recursive(g2d.get(), coord_tx.get(), impl_->segments);
    }

    return true;
}

bool CoastlineReader::load(const std::string &filename, const std::string &layer_name,
                           const std::string &target_srs,
                           Real domain_xmin, Real domain_ymin,
                           Real domain_xmax, Real domain_ymax) {
    GDALAllRegister();

    std::unique_ptr<GDALDataset> ds(static_cast<GDALDataset*>(
        GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
    if (!ds) {
        impl_->error = "Failed to open: " + filename;
        return false;
    }

    OGRLayer* layer = nullptr;
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

    // Set up coordinate transformations
    std::unique_ptr<OGRCoordinateTransformation> coord_tx;        // source -> target
    std::unique_ptr<OGRCoordinateTransformation> coord_tx_inv;    // target -> source (for filter)

    if (!target_srs.empty() && layer->GetSpatialRef()) {
        OGRSpatialReference srcSRS = *layer->GetSpatialRef();
        OGRSpatialReference dstSRS;
        if (dstSRS.SetFromUserInput(target_srs.c_str()) != OGRERR_NONE) {
            impl_->error = "Invalid target SRS: " + target_srs;
            return false;
        }

        // Use traditional GIS axis order (x=lon/easting, y=lat/northing)
        // This is critical for EPSG:4326 which officially uses lat/lon order
        srcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        dstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

        coord_tx.reset(OGRCreateCoordinateTransformation(&srcSRS, &dstSRS));
        if (!coord_tx) {
            impl_->error = "Failed to create coordinate transformation";
            return false;
        }

        // Inverse transformation for domain bounds
        coord_tx_inv.reset(OGRCreateCoordinateTransformation(&dstSRS, &srcSRS));
        if (!coord_tx_inv) {
            impl_->error = "Failed to create inverse coordinate transformation";
            return false;
        }

        // Transform domain bounds to source SRS for spatial filter
        double filter_xmin = domain_xmin, filter_ymin = domain_ymin;
        double filter_xmax = domain_xmax, filter_ymax = domain_ymax;

        // Transform all four corners and take envelope
        double corners_x[4] = {domain_xmin, domain_xmax, domain_xmax, domain_xmin};
        double corners_y[4] = {domain_ymin, domain_ymin, domain_ymax, domain_ymax};
        if (coord_tx_inv->Transform(4, corners_x, corners_y)) {
            filter_xmin = std::min({corners_x[0], corners_x[1], corners_x[2], corners_x[3]});
            filter_xmax = std::max({corners_x[0], corners_x[1], corners_x[2], corners_x[3]});
            filter_ymin = std::min({corners_y[0], corners_y[1], corners_y[2], corners_y[3]});
            filter_ymax = std::max({corners_y[0], corners_y[1], corners_y[2], corners_y[3]});
        }

        // Apply spatial filter to only read features within domain
        layer->SetSpatialFilterRect(filter_xmin, filter_ymin, filter_xmax, filter_ymax);
        std::cout << "Applied spatial filter: [" << filter_xmin << ", " << filter_xmax
                  << "] x [" << filter_ymin << ", " << filter_ymax << "]\n";
    }

    impl_->segments.clear();

    layer->ResetReading();
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        std::unique_ptr<OGRFeature> feat_guard(feat);
        OGRGeometry* geom = feat->GetGeometryRef();
        if (!geom)
            continue;

        // Work on a 2D clone
        std::unique_ptr<OGRGeometry> g2d(geom->clone());
        g2d->flattenTo2D();

        // Collect all segments directly (handles LineStrings, Polygons, etc.)
        collect_segments_recursive(g2d.get(), coord_tx.get(), impl_->segments);
    }

    return true;
}

bool CoastlineReader::is_available() { return true; }

void CoastlineReader::swap_xy() { swap_xy_segments(impl_->segments); }

void CoastlineReader::remove_small_polygons(double /*min_area*/) {
    // No-op for segment-based storage (min_polygon_area doesn't apply to line segments)
}

size_t CoastlineReader::num_polygons() const { return impl_->segments.size(); }

const std::string &CoastlineReader::last_error() const { return impl_->error; }

void CoastlineReader::bounding_box(Real &xmin, Real &ymin, Real &xmax, Real &ymax) const {
    if (impl_->segments.empty()) {
        xmin = ymin = xmax = ymax = 0;
        return;
    }
    xmin = ymin = std::numeric_limits<double>::max();
    xmax = ymax = std::numeric_limits<double>::lowest();
    for (const auto &seg : impl_->segments) {
        xmin = std::min({xmin, bg::get<0>(seg.first), bg::get<0>(seg.second)});
        ymin = std::min({ymin, bg::get<1>(seg.first), bg::get<1>(seg.second)});
        xmax = std::max({xmax, bg::get<0>(seg.first), bg::get<0>(seg.second)});
        ymax = std::max({ymax, bg::get<1>(seg.first), bg::get<1>(seg.second)});
    }
}

void CoastlineReader::write_vtk(const std::string &filename) const {
    std::string vtk_filename = filename + ".vtp";
    std::ofstream out(vtk_filename);
    if (!out) {
        std::cerr << "Failed to open " << vtk_filename << " for writing\n";
        return;
    }

    // Each segment has 2 points
    size_t total_segments = impl_->segments.size();
    size_t total_points = total_segments * 2;

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << total_points
        << "\" NumberOfVerts=\"0\" NumberOfLines=\"" << total_segments
        << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

    // Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto &seg : impl_->segments) {
        out << "          " << bg::get<0>(seg.first) << " " << bg::get<1>(seg.first) << " 0\n";
        out << "          " << bg::get<0>(seg.second) << " " << bg::get<1>(seg.second) << " 0\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Lines (connectivity)
    out << "      <Lines>\n";
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t i = 0; i < total_segments; ++i) {
        out << "          " << (i * 2) << " " << (i * 2 + 1) << "\n";
    }
    out << "        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 0; i < total_segments; ++i) {
        out << "          " << ((i + 1) * 2) << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Lines>\n";

    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";

    out.close();
    std::cout << "Wrote coastline to " << vtk_filename << " (" << total_segments << " segments)\n";
}

std::shared_ptr<CoastlineIndex> CoastlineReader::build_index() const {
    auto index = std::make_shared<CoastlineIndex>();
    index->impl_->rtree = std::make_shared<SegmentRTree>();
    index->impl_->num_segments = 0;

    // Insert all segments into R-tree
    for (size_t i = 0; i < impl_->segments.size(); ++i) {
        index->impl_->rtree->insert({impl_->segments[i], {i, 0, 0}});
        ++index->impl_->num_segments;
    }

    return index;
}

std::shared_ptr<CoastlineIndex> CoastlineReader::build_index(
    Real domain_xmin, Real domain_ymin, Real domain_xmax, Real domain_ymax) const {
    auto index = std::make_shared<CoastlineIndex>();
    index->impl_->rtree = std::make_shared<SegmentRTree>();
    index->impl_->num_segments = 0;

    // Create domain bounding box for filtering
    Box2D domain_box(Point2D(domain_xmin, domain_ymin),
                     Point2D(domain_xmax, domain_ymax));

    // Insert only segments that intersect the domain
    for (size_t i = 0; i < impl_->segments.size(); ++i) {
        if (bg::intersects(impl_->segments[i], domain_box)) {
            index->impl_->rtree->insert({impl_->segments[i], {i, 0, 0}});
            ++index->impl_->num_segments;
        }
    }

    return index;
}

// =============================================================================
// CoastlineIndex implementation
// =============================================================================

size_t CoastlineIndex::num_segments() const { return impl_->num_segments; }

bool CoastlineIndex::intersects(Real xmin, Real ymin, Real xmax, Real ymax) const {
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

CoastlineRefinement::CoastlineRefinement(std::shared_ptr<CoastlineIndex> index, int max_level)
    : index_(std::move(index)), max_level_(max_level) {}

bool CoastlineRefinement::should_refine(const ElementBounds &bounds, int level) const {
    if (level >= max_level_)
        return false;
    return index_->intersects(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax);
}

RefineMask CoastlineRefinement::get_mask(const ElementBounds &bounds) const {
    if (index_->intersects(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)) {
        return RefineMask::X | RefineMask::Y; // Refine horizontally
    }
    return RefineMask::NONE;
}

} // namespace drifter
