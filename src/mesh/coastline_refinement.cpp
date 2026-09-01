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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <unordered_map>
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

// Circumradius sample: location + discrete circumradius
using CircumradiusValue = std::pair<Point2D, double>;
using CircumradiusRTree = bgi::rtree<CircumradiusValue, bgi::rstar<16>>;

// PIMPL implementation structs
struct CoastlineReader::Impl {
    std::vector<Segment2D> segments;  // Store segments directly, no polygon overhead
    std::string error;
};

struct CoastlineIndex::Impl {
    std::shared_ptr<SegmentRTree> rtree;
    std::shared_ptr<CircumradiusRTree> circumradius_rtree;
    size_t num_segments = 0;
    size_t num_circumradius_points = 0;
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

namespace {

// Hash function for vertex positions (with tolerance)
struct VertexHash {
    double tolerance = 1e-8;
    size_t operator()(const Point2D &p) const {
        // Round to tolerance grid for hashing
        int64_t ix = static_cast<int64_t>(std::round(bg::get<0>(p) / tolerance));
        int64_t iy = static_cast<int64_t>(std::round(bg::get<1>(p) / tolerance));
        return std::hash<int64_t>()(ix) ^ (std::hash<int64_t>()(iy) << 1);
    }
};

struct VertexEqual {
    double tolerance = 1e-8;
    bool operator()(const Point2D &a, const Point2D &b) const {
        return std::abs(bg::get<0>(a) - bg::get<0>(b)) < tolerance &&
               std::abs(bg::get<1>(a) - bg::get<1>(b)) < tolerance;
    }
};

// Build ordered vertex chains from segment soup
std::vector<std::vector<Point2D>> build_vertex_chains(const std::vector<Segment2D> &segments) {
    if (segments.empty())
        return {};

    // Build adjacency: vertex -> list of segment indices
    std::unordered_map<Point2D, std::vector<size_t>, VertexHash, VertexEqual> adjacency;
    for (size_t i = 0; i < segments.size(); ++i) {
        adjacency[segments[i].first].push_back(i);
        adjacency[segments[i].second].push_back(i);
    }

    std::vector<bool> used(segments.size(), false);
    std::vector<std::vector<Point2D>> chains;

    for (size_t start_seg = 0; start_seg < segments.size(); ++start_seg) {
        if (used[start_seg])
            continue;

        // Start a new chain
        std::vector<Point2D> chain;
        size_t current_seg = start_seg;
        Point2D current_vertex = segments[current_seg].first;
        chain.push_back(current_vertex);

        while (true) {
            used[current_seg] = true;

            // Get the other endpoint of current segment
            Point2D next_vertex;
            if (VertexEqual()(segments[current_seg].first, current_vertex)) {
                next_vertex = segments[current_seg].second;
            } else {
                next_vertex = segments[current_seg].first;
            }
            chain.push_back(next_vertex);

            // Find next segment from next_vertex
            auto it = adjacency.find(next_vertex);
            if (it == adjacency.end())
                break;

            size_t next_seg = SIZE_MAX;
            for (size_t seg_idx : it->second) {
                if (!used[seg_idx]) {
                    next_seg = seg_idx;
                    break;
                }
            }

            if (next_seg == SIZE_MAX)
                break;

            current_seg = next_seg;
            current_vertex = next_vertex;
        }

        if (chain.size() >= 3) {
            chains.push_back(std::move(chain));
        }
    }

    return chains;
}

// Compute the circumradius of the triangle through 3 consecutive points
double compute_circumradius(const Point2D &p0, const Point2D &p1, const Point2D &p2) {
    double x0 = bg::get<0>(p0), y0 = bg::get<1>(p0);
    double x1 = bg::get<0>(p1), y1 = bg::get<1>(p1);
    double x2 = bg::get<0>(p2), y2 = bg::get<1>(p2);

    // Vectors
    double v1x = x1 - x0, v1y = y1 - y0;
    double v2x = x2 - x1, v2y = y2 - y1;

    // Side lengths
    double a = std::sqrt(v1x * v1x + v1y * v1y);
    double b = std::sqrt(v2x * v2x + v2y * v2y);
    double cx = x2 - x0, cy = y2 - y0;
    double c = std::sqrt(cx * cx + cy * cy);

    // 2D cross product (twice the signed area)
    double cross = v1x * v2y - v1y * v2x;

    // Guard against collinear points
    if (std::abs(cross) < 1e-12 * a * b) {
        return std::numeric_limits<double>::infinity();
    }

    // Circumradius formula: R = abc / (4 * Area)
    return (a * b * c) / (2.0 * std::abs(cross));
}

// Compute normal direction pointing toward the circumcenter
Point2D compute_normal(const Point2D &p0, const Point2D &p1, const Point2D &p2) {
    double x0 = bg::get<0>(p0), y0 = bg::get<1>(p0);
    double x1 = bg::get<0>(p1), y1 = bg::get<1>(p1);
    double x2 = bg::get<0>(p2), y2 = bg::get<1>(p2);

    // Normalized direction vectors
    double v1x = x1 - x0, v1y = y1 - y0;
    double v2x = x2 - x1, v2y = y2 - y1;

    double len1 = std::sqrt(v1x * v1x + v1y * v1y);
    double len2 = std::sqrt(v2x * v2x + v2y * v2y);

    if (len1 < 1e-12 || len2 < 1e-12) {
        return Point2D(0, 0);
    }

    v1x /= len1;
    v1y /= len1;
    v2x /= len2;
    v2y /= len2;

    // Average tangent direction
    double tx = v1x + v2x, ty = v1y + v2y;
    double tlen = std::sqrt(tx * tx + ty * ty);
    if (tlen < 1e-12) {
        // Vectors point in opposite directions (hairpin)
        tx = -v1y;
        ty = v1x;
        tlen = 1.0;
    }
    tx /= tlen;
    ty /= tlen;

    // Normal is perpendicular to tangent
    double nx = -ty, ny = tx;

    // Check which side the center is on using cross product
    double cross = v1x * v2y - v1y * v2x;
    if (cross < 0) {
        nx = -nx;
        ny = -ny;
    }

    return Point2D(nx, ny);
}

} // anonymous namespace

void CoastlineReader::write_circumradius_comb_vtk(const std::string &filename) const {
    std::string vtk_filename = filename + ".vtp";
    std::ofstream out(vtk_filename);
    if (!out) {
        std::cerr << "Failed to open " << vtk_filename << " for writing\n";
        return;
    }

    // Build vertex chains from segments
    auto chains = build_vertex_chains(impl_->segments);

    // Collect circumradius data for all interior vertices
    struct CombLine {
        double x0, y0;  // Start point (on coastline)
        double x1, y1;  // End point (comb tip)
        double radius;  // Discrete circumradius
    };
    std::vector<CombLine> comb_lines;

    for (const auto &chain : chains) {
        // Compute the circumradius at interior vertices (skip endpoints)
        for (size_t i = 1; i + 1 < chain.size(); ++i) {
            double radius = compute_circumradius(chain[i - 1], chain[i], chain[i + 1]);
            Point2D normal = compute_normal(chain[i - 1], chain[i], chain[i + 1]);

            // Skip infinite radii (straight sections)
            if (std::isinf(radius)) {
                continue;
            }

            // Drawn at the same length scale as the mesh, so a tooth is directly
            // comparable to the element it sits in. Clamped only so a nearly
            // straight stretch cannot draw a tooth the size of the domain - the
            // untruncated radius still goes out as cell data below.
            constexpr double MAX_COMB_LENGTH_M = 10000.0;
            double length = std::min(radius, MAX_COMB_LENGTH_M);
            double x0 = bg::get<0>(chain[i]);
            double y0 = bg::get<1>(chain[i]);
            double x1 = x0 + bg::get<0>(normal) * length;
            double y1 = y0 + bg::get<1>(normal) * length;

            comb_lines.push_back({x0, y0, x1, y1, radius});
        }
    }

    size_t num_lines = comb_lines.size();
    size_t num_points = num_lines * 2;

    // VTK XML header
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << num_points
        << "\" NumberOfVerts=\"0\" NumberOfLines=\"" << num_lines
        << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

    // Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto &line : comb_lines) {
        out << "          " << line.x0 << " " << line.y0 << " 0\n";
        out << "          " << line.x1 << " " << line.y1 << " 0\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Lines
    out << "      <Lines>\n";
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t i = 0; i < num_lines; ++i) {
        out << "          " << (i * 2) << " " << (i * 2 + 1) << "\n";
    }
    out << "        </DataArray>\n";
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 0; i < num_lines; ++i) {
        out << "          " << ((i + 1) * 2) << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Lines>\n";

    // Cell data: discrete circumradius
    out << "      <CellData>\n";
    out << "        <DataArray type=\"Float64\" Name=\"circumradius\" format=\"ascii\">\n";
    for (const auto &line : comb_lines) {
        out << "          " << line.radius << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";

    out.close();
    std::cout << "Wrote circumradius comb to " << vtk_filename << " (" << num_lines
              << " comb lines from " << chains.size() << " chains)\n";
}

std::shared_ptr<CoastlineIndex> CoastlineReader::build_index() const {
    auto index = std::make_shared<CoastlineIndex>();
    index->impl_->rtree = std::make_shared<SegmentRTree>();
    index->impl_->circumradius_rtree = std::make_shared<CircumradiusRTree>();
    index->impl_->num_segments = 0;
    index->impl_->num_circumradius_points = 0;

    // Insert all segments into R-tree
    for (size_t i = 0; i < impl_->segments.size(); ++i) {
        index->impl_->rtree->insert({impl_->segments[i], {i, 0, 0}});
        ++index->impl_->num_segments;
    }

    // Build circumradius index from vertex chains
    auto chains = build_vertex_chains(impl_->segments);
    for (const auto& chain : chains) {
        for (size_t i = 1; i + 1 < chain.size(); ++i) {
            double radius = compute_circumradius(chain[i - 1], chain[i], chain[i + 1]);
            if (!std::isinf(radius)) {
                index->impl_->circumradius_rtree->insert({chain[i], radius});
                ++index->impl_->num_circumradius_points;
            }
        }
    }

    return index;
}

std::shared_ptr<CoastlineIndex> CoastlineReader::build_index(
    Real domain_xmin, Real domain_ymin, Real domain_xmax, Real domain_ymax) const {
    auto index = std::make_shared<CoastlineIndex>();
    index->impl_->rtree = std::make_shared<SegmentRTree>();
    index->impl_->circumradius_rtree = std::make_shared<CircumradiusRTree>();
    index->impl_->num_segments = 0;
    index->impl_->num_circumradius_points = 0;

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

    // Build circumradius index from vertex chains
    auto chains = build_vertex_chains(impl_->segments);
    for (const auto& chain : chains) {
        for (size_t i = 1; i + 1 < chain.size(); ++i) {
            double radius = compute_circumradius(chain[i - 1], chain[i], chain[i + 1]);
            if (!std::isinf(radius)) {
                index->impl_->circumradius_rtree->insert({chain[i], radius});
                ++index->impl_->num_circumradius_points;
            }
        }
    }

    return index;
}

// =============================================================================
// CoastlineIndex implementation
// =============================================================================

size_t CoastlineIndex::num_segments() const { return impl_->num_segments; }

size_t CoastlineIndex::num_circumradius_points() const {
    return impl_->num_circumradius_points;
}

bool CoastlineIndex::has_circumradius_below(Real xmin, Real ymin, Real xmax, Real ymax,
                                            Real threshold) const {
    if (!impl_->circumradius_rtree) {
        return false;
    }

    // Stops at the first sample under the ceiling rather than gathering every
    // sample in the box: at coarse levels a single box can cover the whole coast.
    Box2D box(Point2D(xmin, ymin), Point2D(xmax, ymax));
    auto below = [threshold](const CircumradiusValue &value) {
        return value.second < threshold;
    };
    return impl_->circumradius_rtree->qbegin(bgi::intersects(box) && bgi::satisfies(below)) !=
           impl_->circumradius_rtree->qend();
}

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
