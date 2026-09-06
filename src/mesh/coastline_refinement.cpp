#include "mesh/coastline_refinement.hpp"
#include "core/logger.hpp"

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

#include "core/scoped_timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drifter {

// Boost.Geometry types (hidden from header)
namespace bg  = boost::geometry;
namespace bgi = bg::index;

using Point2D        = bg::model::point<double, 2, bg::cs::cartesian>;
using Segment2D      = bg::model::segment<Point2D>;
using Box2D          = bg::model::box<Point2D>;
using Ring2D         = bg::model::ring<Point2D>;
using Polygon2D      = bg::model::polygon<Point2D, true>; // Clockwise
using MultiPolygon2D = bg::model::multi_polygon<Polygon2D>;

// The trees are only ever asked for box overlap, so a segment carries no payload
// and a sample carries only its radius.
using SegmentRTree = bgi::rtree<Segment2D, bgi::rstar<16>>;

// Circumradius sample: location + discrete circumradius
using CircumradiusValue = std::pair<Point2D, double>;
using CircumradiusRTree = bgi::rtree<CircumradiusValue, bgi::rstar<16>>;

// Vertices are stored once, in CSR form: the input is a set of ordered polylines
// (an OSM split line, a polygon ring), and keeping that order is what lets the
// circumradius pass skip rebuilding it from unordered segments. Polyline k spans
// points[offsets[k] .. offsets[k + 1]), so segment j of that polyline runs from
// point offsets[k] + j to offsets[k] + j + 1 and needs no storage of its own.
using PointStore  = std::vector<Point2D>;
using OffsetStore = std::vector<std::uint32_t>;

// PIMPL implementation structs
struct CoastlineReader::Impl {
  // Shared so an index built from this reader borrows the geometry rather than
  // copying it; a continental load is hundreds of MB.
  std::shared_ptr<PointStore> points   = std::make_shared<PointStore>();
  std::shared_ptr<OffsetStore> offsets = std::make_shared<OffsetStore>(1, 0);
  std::string error;

  size_t num_polylines() const { return offsets->size() - 1; }
  size_t num_segments() const { return points->size() - num_polylines(); }
};

struct CoastlineIndex::Impl {
  std::shared_ptr<const PointStore> points;
  std::shared_ptr<const OffsetStore> offsets;

  // First point of each segment that survived the domain filter. The segment
  // tree itself is built from these on the first intersects() call and never at
  // all in the app path, which asks only for num_segments().
  std::vector<std::uint32_t> segment_first_point;
  mutable std::shared_ptr<SegmentRTree> rtree;
  mutable std::once_flag rtree_once;

  std::shared_ptr<CircumradiusRTree> circumradius_rtree;
  size_t num_circumradius_points = 0;

  const SegmentRTree &segment_tree() const;
};

// Internal utility functions (moved from public header)
namespace {

// Accumulates polylines in CSR form while reading a layer
struct PolylineBuilder {
  PointStore &points;
  OffsetStore &offsets;

  void begin_polyline() {}
  void end_polyline() { offsets.push_back(static_cast<std::uint32_t>(points.size())); }
};

// Walks every segment of every polyline, calling fn(a, b, first_point_index).
// The segment-shaped consumers - VTK output, the bounding box, the domain filter
// - go through here so the CSR layout is unpacked in exactly one place.
template <typename Fn> void for_each_segment(const PointStore &points, const OffsetStore &offsets, Fn &&fn) {
  for (size_t k = 0; k + 1 < offsets.size(); ++k) {
    const std::uint32_t begin = offsets[k];
    const std::uint32_t end   = offsets[k + 1];
    for (std::uint32_t i = begin; i + 1 < end; ++i) {
      fn(points[i], points[i + 1], i);
    }
  }
}

void swap_xy_points(PointStore &points) {
  for (auto &p : points) {
    p = Point2D(bg::get<1>(p), bg::get<0>(p));
  }
}

// Helper to add one polyline from OGR geometry
void add_polyline_from_linestring(const OGRLineString* ls, OGRCoordinateTransformation* coord_tx, PolylineBuilder &out) {
  if (!ls || ls->getNumPoints() < 2)
    return;

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

    // Append the vertices in order and close the polyline
  out.begin_polyline();
  for (int i = 0; i < n; ++i) {
    out.points.emplace_back(x[i], y[i]);
  }
  out.end_polyline();
}

void add_polylines_from_polygon(const OGRPolygon* poly, OGRCoordinateTransformation* coord_tx, PolylineBuilder &out) {
  if (!poly)
    return;

    // Exterior ring
  const OGRLinearRing* ext = poly->getExteriorRing();
  if (ext) {
    add_polyline_from_linestring(ext, coord_tx, out);
  }

    // Interior rings
  for (int i = 0; i < poly->getNumInteriorRings(); ++i) {
    const OGRLinearRing* inner = poly->getInteriorRing(i);
    if (inner) {
      add_polyline_from_linestring(inner, coord_tx, out);
    }
  }
}

void collect_polylines_recursive(const OGRGeometry* g, OGRCoordinateTransformation* coord_tx, PolylineBuilder &out) {
  if (!g)
    return;
  OGRwkbGeometryType t = wkbFlatten(g->getGeometryType());

  if (t == wkbLineString) {
    add_polyline_from_linestring(g->toLineString(), coord_tx, out);
  } else if (t == wkbPolygon) {
    add_polylines_from_polygon(g->toPolygon(), coord_tx, out);
  } else if (t == wkbMultiLineString) {
    const OGRMultiLineString* mls = g->toMultiLineString();
    for (int i = 0; i < mls->getNumGeometries(); ++i) {
      add_polyline_from_linestring(mls->getGeometryRef(i)->toLineString(), coord_tx, out);
    }
  } else if (t == wkbMultiPolygon) {
    const OGRMultiPolygon* mp = g->toMultiPolygon();
    for (int i = 0; i < mp->getNumGeometries(); ++i) {
      add_polylines_from_polygon(mp->getGeometryRef(i), coord_tx, out);
    }
  } else if (t == wkbGeometryCollection) {
    const OGRGeometryCollection* gc = g->toGeometryCollection();
    for (int i = 0; i < gc->getNumGeometries(); ++i) {
      collect_polylines_recursive(gc->getGeometryRef(i), coord_tx, out);
    }
  }
}

} // anonymous namespace

// CoastlineReader constructor/destructor
CoastlineReader::CoastlineReader() : impl_(std::make_unique<Impl>()) {}
CoastlineReader::~CoastlineReader()                                      = default;
CoastlineReader::CoastlineReader(CoastlineReader &&) noexcept            = default;
CoastlineReader &CoastlineReader::operator=(CoastlineReader &&) noexcept = default;

// CoastlineIndex constructor/destructor
CoastlineIndex::CoastlineIndex() : impl_(std::make_unique<Impl>()) {}
CoastlineIndex::~CoastlineIndex()                                     = default;
CoastlineIndex::CoastlineIndex(CoastlineIndex &&) noexcept            = default;
CoastlineIndex &CoastlineIndex::operator=(CoastlineIndex &&) noexcept = default;

// =============================================================================
// CoastlineReader implementation
// =============================================================================

bool CoastlineReader::load(const std::string &filename, const std::string &layer_name, const std::string &target_srs) {
  GDALAllRegister();

  std::unique_ptr<GDALDataset> ds(static_cast<GDALDataset*>(GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
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

  impl_->points->clear();
  impl_->offsets->assign(1, 0);
  PolylineBuilder builder{*impl_->points, *impl_->offsets};

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

        // Collect the vertices in order (handles LineStrings, Polygons, etc.)
    collect_polylines_recursive(g2d.get(), coord_tx.get(), builder);
  }

  return true;
}

bool CoastlineReader::load(const std::string &filename, const std::string &layer_name, const std::string &target_srs, Real domain_xmin, Real domain_ymin, Real domain_xmax, Real domain_ymax) {
  GDALAllRegister();

  std::unique_ptr<GDALDataset> ds(static_cast<GDALDataset*>(GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr)));
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
    LOG_INFO("Applied spatial filter: [" << filter_xmin << ", " << filter_xmax << "] x [" << filter_ymin << ", " << filter_ymax << "]");
  }

  impl_->points->clear();
  impl_->offsets->assign(1, 0);
  PolylineBuilder builder{*impl_->points, *impl_->offsets};

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

        // Collect the vertices in order (handles LineStrings, Polygons, etc.)
    collect_polylines_recursive(g2d.get(), coord_tx.get(), builder);
  }

  LOG_INFO("Coastline loaded: " << impl_->num_polylines() << " polylines, " << impl_->points->size() << " vertices, " << impl_->num_segments() << " segments");

  return true;
}

bool CoastlineReader::is_available() { return true; }

void CoastlineReader::swap_xy() { swap_xy_points(*impl_->points); }

void CoastlineReader::remove_small_polygons(double /*min_area*/) {
    // No-op for polyline storage (min_polygon_area doesn't apply to line segments)
}

size_t CoastlineReader::num_polygons() const { return impl_->num_segments(); }

const std::string &CoastlineReader::last_error() const { return impl_->error; }

void CoastlineReader::bounding_box(Real &xmin, Real &ymin, Real &xmax, Real &ymax) const {
  if (impl_->points->empty()) {
    xmin = ymin = xmax = ymax = 0;
    return;
  }
  xmin = ymin = std::numeric_limits<double>::max();
  xmax = ymax = std::numeric_limits<double>::lowest();
  for (const auto &p : *impl_->points) {
    xmin = std::min(xmin, bg::get<0>(p));
    ymin = std::min(ymin, bg::get<1>(p));
    xmax = std::max(xmax, bg::get<0>(p));
    ymax = std::max(ymax, bg::get<1>(p));
  }
}

void CoastlineReader::write_vtk(const std::string &filename) const {
  std::string vtk_filename = filename + ".vtp";
  std::ofstream out(vtk_filename);
  if (!out) {
    LOG_ERROR("Failed to open " << vtk_filename << " for writing");
    return;
  }

    // Each segment has 2 points
  size_t total_segments = impl_->num_segments();
  size_t total_points   = total_segments * 2;

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  out << "  <PolyData>\n";
  out << "    <Piece NumberOfPoints=\"" << total_points << "\" NumberOfVerts=\"0\" NumberOfLines=\"" << total_segments << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

    // Points
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for_each_segment(*impl_->points, *impl_->offsets, [&out](const Point2D &a, const Point2D &b, std::uint32_t) {
    out << "          " << bg::get<0>(a) << " " << bg::get<1>(a) << " 0\n";
    out << "          " << bg::get<0>(b) << " " << bg::get<1>(b) << " 0\n";
  });
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
  LOG_INFO("Wrote coastline to " << vtk_filename << " (" << total_segments << " segments)");
}

namespace {

// Vertex key on a fixed snap grid. Hashing and equality both act on the snapped
// integer pair, so equal keys always hash equally - the previous pair rounded to
// a grid for the hash but compared raw coordinates within a tolerance band, which
// let two "equal" points land in different buckets. A micrometre is far below any
// real coastline detail and keeps the grid index well inside int64 for projected
// metres (5e6 / 1e-6 = 5e12).
constexpr double VERTEX_SNAP = 1e-6;

struct VertexKey {
  int64_t ix;
  int64_t iy;
  bool operator==(const VertexKey &o) const { return ix == o.ix && iy == o.iy; }
};

struct VertexKeyHash {
  size_t operator()(const VertexKey &k) const {
        // std::hash<int64_t> is the identity in libstdc++, so mix explicitly
    size_t h = static_cast<size_t>(k.ix) * 0x9E3779B97F4A7C15ULL;
    h ^= static_cast<size_t>(k.iy) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

VertexKey snap(const Point2D &p) {
  return {static_cast<int64_t>(std::llround(bg::get<0>(p) / VERTEX_SNAP)), static_cast<int64_t>(std::llround(bg::get<1>(p) / VERTEX_SNAP))};
}

// A chain is a run of polylines walked end to end. `reversed` says the polyline's
// vertices are visited back to front, which is how two features that meet at a
// shared *start* vertex are joined into one continuous chain.
struct ChainLink {
  std::uint32_t polyline;
  bool reversed;
};
using Chain = std::vector<ChainLink>;

// Ordered vertex access over a chain without materializing it. Consecutive links
// share a vertex - the tail of one polyline is the head of the next - so the
// duplicate is dropped, exactly as the old segment walk did.
class ChainVertices {
public:
  ChainVertices(const Chain &chain, const PointStore &points, const OffsetStore &offsets) : points_(points) {
    for (const ChainLink &link : chain) {
      const std::uint32_t begin = offsets[link.polyline];
      const std::uint32_t end   = offsets[link.polyline + 1];
      const size_t first        = indices_.empty() ? 0 : 1; // skip the shared vertex
      if (!link.reversed) {
        for (std::uint32_t i = begin + first; i < end; ++i)
          indices_.push_back(i);
      } else {
        for (std::uint32_t i = end - 1 - first; i + 1 > begin; --i)
          indices_.push_back(i);
      }
    }
  }

  size_t size() const { return indices_.size(); }
  const Point2D &operator[](size_t i) const { return points_[indices_[i]]; }

private:
  const PointStore &points_;
  std::vector<std::uint32_t> indices_;
};

// Join polylines that share an endpoint into maximal chains.
//
// The input already carries the vertex ordering - an OSM split line, a polygon
// ring - so only the 2 endpoints of each polyline need to enter the hash map
// (~520k entries on a continental load, against ~32M if every vertex did). The
// greedy walk is the same one that used to run over individual segments, lifted
// one level up, so the chain set it produces is unchanged.
std::vector<Chain> build_chains(const PointStore &points, const OffsetStore &offsets) {
  const size_t num_polylines = offsets.size() - 1;
  if (num_polylines == 0)
    return {};

    // endpoint key -> polylines touching it
  std::unordered_map<VertexKey, std::vector<std::uint32_t>, VertexKeyHash> endpoints;
  endpoints.reserve(num_polylines * 2);
  for (std::uint32_t k = 0; k < num_polylines; ++k) {
    endpoints[snap(points[offsets[k]])].push_back(k);
    endpoints[snap(points[offsets[k + 1] - 1])].push_back(k);
  }

    // Probe the 3x3 neighbourhood so a pair straddling a snap-cell boundary still
    // joins. Returns the first unused polyline other than `exclude` that has an
    // endpoint at `p`, along with which of its ends that is.
  std::vector<bool> used(num_polylines, false);
  auto find_neighbour = [&](const Point2D &p, std::uint32_t exclude, std::uint32_t &out_polyline, bool &out_at_start) {
    const VertexKey k = snap(p);
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        auto it = endpoints.find(VertexKey{k.ix + dx, k.iy + dy});
        if (it == endpoints.end())
          continue;
        for (std::uint32_t cand : it->second) {
          if (cand == exclude || used[cand])
            continue;
          const VertexKey head = snap(points[offsets[cand]]);
          const VertexKey tail = snap(points[offsets[cand + 1] - 1]);
          if (head == k) {
            out_polyline = cand;
            out_at_start = true;
            return true;
          }
          if (tail == k) {
            out_polyline = cand;
            out_at_start = false;
            return true;
          }
        }
      }
    }
    return false;
  };

  std::vector<Chain> chains;
  for (std::uint32_t start = 0; start < num_polylines; ++start) {
    if (used[start])
      continue;

    Chain chain{{start, false}};
    used[start] = true;

        // Extend forward from the chain's tail vertex
    std::uint32_t current = start;
    while (true) {
      const ChainLink &back = chain.back();
      const Point2D &tail   = back.reversed ? points[offsets[back.polyline]] : points[offsets[back.polyline + 1] - 1];
      std::uint32_t next    = 0;
      bool at_start         = false;
      if (!find_neighbour(tail, current, next, at_start))
        break;
            // Joined at the neighbour's start means it continues forward; joined at
            // its end means it must be walked backwards.
      chain.push_back({next, !at_start});
      used[next] = true;
      current    = next;
    }

        // A chain shorter than 3 vertices has no interior vertex and so carries no
        // circumradius sample; dropping it here keeps the reported chain count
        // meaningful, as the old segment walk did.
    size_t vertices = 0;
    for (const ChainLink &link : chain)
      vertices += offsets[link.polyline + 1] - offsets[link.polyline];
    vertices -= chain.size() - 1; // shared vertices counted once
    if (vertices >= 3) {
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
  double a  = std::sqrt(v1x * v1x + v1y * v1y);
  double b  = std::sqrt(v2x * v2x + v2y * v2y);
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
    tx   = -v1y;
    ty   = v1x;
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
    LOG_ERROR("Failed to open " << vtk_filename << " for writing");
    return;
  }

    // Stitch the loaded polylines into ordered chains
  auto chains = build_chains(*impl_->points, *impl_->offsets);

    // Collect circumradius data for all interior vertices
  struct CombLine {
    double x0, y0;  // Start point (on coastline)
    double x1, y1;  // End point (comb tip)
    double radius;  // Discrete circumradius
  };
  std::vector<CombLine> comb_lines;

  for (const auto &chain_links : chains) {
    const ChainVertices chain(chain_links, *impl_->points, *impl_->offsets);
        // Compute the circumradius at interior vertices (skip endpoints)
    for (size_t i = 1; i + 1 < chain.size(); ++i) {
      double radius  = compute_circumradius(chain[i - 1], chain[i], chain[i + 1]);
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
      double length                      = std::min(radius, MAX_COMB_LENGTH_M);
      double x0                          = bg::get<0>(chain[i]);
      double y0                          = bg::get<1>(chain[i]);
      double x1                          = x0 + bg::get<0>(normal) * length;
      double y1                          = y0 + bg::get<1>(normal) * length;

      comb_lines.push_back({x0, y0, x1, y1, radius});
    }
  }

  size_t num_lines  = comb_lines.size();
  size_t num_points = num_lines * 2;

    // VTK XML header
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  out << "  <PolyData>\n";
  out << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfVerts=\"0\" NumberOfLines=\"" << num_lines << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

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
  LOG_INFO("Wrote circumradius comb to " << vtk_filename << " (" << num_lines << " comb lines from " << chains.size() << " chains)");
}

std::shared_ptr<CoastlineIndex> CoastlineReader::build_index() const {
    // One code path: an unbounded box keeps everything the domain-filtered
    // overload would keep.
  constexpr Real inf = std::numeric_limits<Real>::infinity();
  return build_index(-inf, -inf, inf, inf);
}

std::shared_ptr<CoastlineIndex> CoastlineReader::build_index(Real domain_xmin, Real domain_ymin, Real domain_xmax, Real domain_ymax) const {
  auto index            = std::make_shared<CoastlineIndex>();
  index->impl_->points  = impl_->points;   // shared, not copied
  index->impl_->offsets = impl_->offsets;

  const PointStore &points   = *impl_->points;
  const OffsetStore &offsets = *impl_->offsets;

    // Create domain bounding box for filtering
  Box2D domain_box(Point2D(domain_xmin, domain_ymin), Point2D(domain_xmax, domain_ymax));

    // Record which segments fall in the domain. The tree over them is not built
    // here - nothing in either app queries it, so it is left to the first
    // intersects() call.
  {
    double ms = 0.0;
    {
      ScopedTimer timer(ms);
      auto &ids = index->impl_->segment_first_point;
      for (size_t k = 0; k + 1 < offsets.size(); ++k) {
        const std::uint32_t begin = offsets[k];
        const std::uint32_t end   = offsets[k + 1];

                // Whole-polyline reject first: load() already applied a spatial filter,
                // so most polylines are entirely inside and this skips their segments.
        Box2D poly_box;
        bg::assign_inverse(poly_box);
        for (std::uint32_t i = begin; i < end; ++i)
          bg::expand(poly_box, points[i]);
        if (!bg::intersects(poly_box, domain_box))
          continue;

        for (std::uint32_t i = begin; i + 1 < end; ++i) {
          if (bg::intersects(Segment2D(points[i], points[i + 1]), domain_box)) {
            ids.push_back(i);
          }
        }
      }
    }
    LOG_INFO("Coastline segment filter: " << index->impl_->segment_first_point.size() << " of " << impl_->num_segments() << " segments in domain (" << ms << " ms)");
  }

    // Stitch the loaded polylines into ordered chains
  std::vector<Chain> chains;
  {
    double ms = 0.0;
    {
      ScopedTimer timer(ms);
      chains = build_chains(points, offsets);
    }
    LOG_INFO("Coastline chain stitching: " << chains.size() << " chains from " << (offsets.size() - 1) << " polylines (" << ms << " ms)");
  }

    // Circumradius at every interior chain vertex inside the domain. Chain lengths
    // span orders of magnitude, hence dynamic scheduling.
  std::vector<CircumradiusValue> samples;
  {
    double ms = 0.0;
    {
      ScopedTimer timer(ms);
      const int num_chains = static_cast<int>(chains.size());
#ifdef _OPENMP
      std::vector<std::vector<CircumradiusValue>> per_thread(omp_get_max_threads());
#pragma omp parallel
      {
        auto &local = per_thread[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 64) nowait
        for (int c = 0; c < num_chains; ++c) {
          const ChainVertices chain(chains[c], points, offsets);
          for (size_t i = 1; i + 1 < chain.size(); ++i) {
            const double radius = compute_circumradius(chain[i - 1], chain[i], chain[i + 1]);
            if (!std::isinf(radius) && bg::within(chain[i], domain_box)) {
              local.emplace_back(chain[i], radius);
            }
          }
        }
      }
      size_t total = 0;
      for (const auto &local : per_thread)
        total += local.size();
      samples.reserve(total);
      for (auto &local : per_thread) {
        samples.insert(samples.end(), local.begin(), local.end());
        std::vector<CircumradiusValue>().swap(local);
      }
#else
      for (int c = 0; c < num_chains; ++c) {
        const ChainVertices chain(chains[c], points, offsets);
        for (size_t i = 1; i + 1 < chain.size(); ++i) {
          const double radius = compute_circumradius(chain[i - 1], chain[i], chain[i + 1]);
          if (!std::isinf(radius) && bg::within(chain[i], domain_box)) {
            samples.emplace_back(chain[i], radius);
          }
        }
      }
#endif
    }
    LOG_INFO("Coastline circumradius: " << samples.size() << " samples (" << ms << " ms)");
  }
  std::vector<Chain>().swap(chains);

    // Bulk-load. The packing constructor runs an STR-style build in one pass;
    // repeated insert() would run the R* insertion path, forced reinsertion and
    // all, once per sample.
  {
    double ms = 0.0;
    {
      ScopedTimer timer(ms);
      index->impl_->circumradius_rtree      = std::make_shared<CircumradiusRTree>(samples.begin(), samples.end());
      index->impl_->num_circumradius_points = samples.size();
    }
        // The tree owns a copy now, so release the staging vector before the caller
        // goes on to allocate a mesh.
    std::vector<CircumradiusValue>().swap(samples);
    LOG_INFO("Coastline circumradius tree packed (" << ms << " ms)");
  }

  return index;
}

// =============================================================================
// CoastlineIndex implementation
// =============================================================================

const SegmentRTree &CoastlineIndex::Impl::segment_tree() const {
    // Built on demand: the refinement pre-pass asks only for num_segments(), so
    // on a continental domain this never runs.
  std::call_once(rtree_once, [this] {
    std::vector<Segment2D> values;
    values.reserve(segment_first_point.size());
    for (std::uint32_t i : segment_first_point) {
      values.emplace_back((*points)[i], (*points)[i + 1]);
    }
    double ms = 0.0;
    {
      ScopedTimer timer(ms);
      rtree = std::make_shared<SegmentRTree>(values.begin(), values.end());
    }
    LOG_INFO("Coastline segment tree packed: " << values.size() << " segments (" << ms << " ms)");
  });
  return *rtree;
}

size_t CoastlineIndex::num_segments() const { return impl_->segment_first_point.size(); }

size_t CoastlineIndex::num_circumradius_points() const { return impl_->num_circumradius_points; }

bool CoastlineIndex::has_circumradius_below(Real xmin, Real ymin, Real xmax, Real ymax, Real threshold) const {
  if (!impl_->circumradius_rtree) {
    return false;
  }

    // Stops at the first sample under the ceiling rather than gathering every
    // sample in the box: at coarse levels a single box can cover the whole coast.
  Box2D box(Point2D(xmin, ymin), Point2D(xmax, ymax));
  auto below = [threshold](const CircumradiusValue &value) { return value.second < threshold; };
  return impl_->circumradius_rtree->qbegin(bgi::intersects(box) && bgi::satisfies(below)) != impl_->circumradius_rtree->qend();
}

bool CoastlineIndex::intersects(Real xmin, Real ymin, Real xmax, Real ymax) const {
  if (impl_->segment_first_point.empty())
    return false;
  Box2D box(Point2D(xmin, ymin), Point2D(xmax, ymax));
    // Stops at the first hit rather than gathering every segment in the box.
  const SegmentRTree &tree = impl_->segment_tree();
  return tree.qbegin(bgi::intersects(box)) != tree.qend();
}

// =============================================================================
// CoastlineRefinement implementation
// =============================================================================

CoastlineRefinement::CoastlineRefinement(std::shared_ptr<CoastlineIndex> index, int max_level) : index_(std::move(index)), max_level_(max_level) {}

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
