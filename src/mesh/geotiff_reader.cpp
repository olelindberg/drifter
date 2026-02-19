#include "mesh/geotiff_reader.hpp"
#include "dg/basis_hexahedron.hpp"
#include "mesh/octree_adapter.hpp"
#include <algorithm>
#include <cmath>

#include <cpl_conv.h>
#include <gdal_priv.h>

namespace drifter {

// =============================================================================
// GeoTiffReader implementation
// =============================================================================

GeoTiffReader::GeoTiffReader() {
    GDALAllRegister();
    gdal_initialized_ = true;
}

GeoTiffReader::~GeoTiffReader() {
    if (gdal_initialized_) {
        GDALDestroyDriverManager();
    }
}

bool GeoTiffReader::is_available() { return true; }

BathymetryData GeoTiffReader::load(const std::string &filename) {
    BathymetryData data;
    error_message_.clear();

    // Open the dataset
    GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpen(filename.c_str(), GA_ReadOnly));

    if (!dataset) {
        error_message_ = "Failed to open GeoTIFF file: " + filename;
        return data;
    }

    // Get dimensions
    data.sizex = dataset->GetRasterXSize();
    data.sizey = dataset->GetRasterYSize();

    // Get geotransform
    if (dataset->GetGeoTransform(data.geotransform.data()) != CE_None) {
        // Use default identity transform if not available
        data.geotransform = {0, 1, 0, 0, 0, -1};
    }

    // Get projection
    const char* proj = dataset->GetProjectionRef();
    if (proj) {
        data.projection = proj;
    }

    // Get the first raster band
    GDALRasterBand* band = dataset->GetRasterBand(1);
    if (!band) {
        error_message_ = "Failed to get raster band from: " + filename;
        GDALClose(dataset);
        return data;
    }

    // Get NoData value
    int has_nodata = 0;
    double nodata = band->GetNoDataValue(&has_nodata);
    if (has_nodata) {
        data.nodata_value = static_cast<float>(nodata);
    }

    // Check band metadata to detect if this is a depth file
    // Danish Depth Model uses "Band_Name=dybde" (Danish for depth)
    const char* band_name = band->GetMetadataItem("Band_Name");
    const char* band_direction = band->GetMetadataItem("Band_Direction");
    if ((band_name && (std::string(band_name) == "dybde" || std::string(band_name) == "depth")) ||
        (band_direction && std::string(band_direction) == "Height")) {
        data.is_depth_positive = true;
    }

    // Also auto-detect based on data statistics: if min > 0, likely depth
    // format
    double min_val, max_val, mean, stddev;
    if (band->GetStatistics(FALSE, TRUE, &min_val, &max_val, &mean, &stddev) == CE_None) {
        // If all values are non-negative, assume it's a depth dataset
        if (min_val >= 0.0 && max_val > 0.0) {
            data.is_depth_positive = true;
        }
    }

    // Allocate and read elevation data
    data.elevation.resize(data.sizex * data.sizey);

    CPLErr err = band->RasterIO(GF_Read, 0, 0, // Start at origin
                                data.sizex, data.sizey, // Read entire raster
                                data.elevation.data(), // Output buffer
                                data.sizex, data.sizey, // Buffer size
                                GDT_Float32, // Data type
                                0, 0); // Default strides

    if (err != CE_None) {
        error_message_ = "Failed to read raster data from: " + filename;
        data.elevation.clear();
        GDALClose(dataset);
        return data;
    }

    // Close dataset
    GDALClose(dataset);

    // Compute bounding box
    data.pixel_to_world(0, 0, data.xmin, data.ymax);
    data.pixel_to_world(data.sizex, data.sizey, data.xmax, data.ymin);

    // Handle inverted axes
    if (data.xmin > data.xmax)
        std::swap(data.xmin, data.xmax);
    if (data.ymin > data.ymax)
        std::swap(data.ymin, data.ymax);

    return data;
}

BathymetryBounds GeoTiffReader::load_bounds_only(const std::string &filename) {
    BathymetryBounds bounds;
    error_message_.clear();

    // Open the dataset
    GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpen(filename.c_str(), GA_ReadOnly));

    if (!dataset) {
        error_message_ = "Failed to open GeoTIFF file: " + filename;
        return bounds;
    }

    // Get dimensions
    int sizex = dataset->GetRasterXSize();
    int sizey = dataset->GetRasterYSize();

    // Get geotransform
    std::array<double, 6> geotransform = {0, 1, 0, 0, 0, -1};
    dataset->GetGeoTransform(geotransform.data());

    // Get the first raster band for metadata
    GDALRasterBand* band = dataset->GetRasterBand(1);
    if (band) {
        // Get NoData value
        int has_nodata = 0;
        double nodata = band->GetNoDataValue(&has_nodata);
        if (has_nodata) {
            bounds.nodata_value = static_cast<float>(nodata);
        }

        // Check band metadata to detect depth format
        const char* band_name = band->GetMetadataItem("Band_Name");
        const char* band_direction = band->GetMetadataItem("Band_Direction");
        if ((band_name &&
             (std::string(band_name) == "dybde" || std::string(band_name) == "depth")) ||
            (band_direction && std::string(band_direction) == "Height")) {
            bounds.is_depth_positive = true;
        }

        // Auto-detect from statistics if available
        double min_val, max_val, mean, stddev;
        if (band->GetStatistics(FALSE, FALSE, &min_val, &max_val, &mean, &stddev) == CE_None) {
            if (min_val >= 0.0 && max_val > 0.0) {
                bounds.is_depth_positive = true;
            }
        }
    }

    // Compute bounding box from geotransform (same logic as load())
    // Top-left corner
    bounds.xmin = geotransform[0];
    bounds.ymax = geotransform[3];

    // Bottom-right corner
    bounds.xmax = geotransform[0] + sizex * geotransform[1] + sizey * geotransform[2];
    bounds.ymin = geotransform[3] + sizex * geotransform[4] + sizey * geotransform[5];

    // Handle inverted axes
    if (bounds.xmin > bounds.xmax)
        std::swap(bounds.xmin, bounds.xmax);
    if (bounds.ymin > bounds.ymax)
        std::swap(bounds.ymin, bounds.ymax);

    // Close dataset - we don't need the raster data
    GDALClose(dataset);

    return bounds;
}

// =============================================================================
// BathymetrySurface implementation
// =============================================================================

BathymetrySurface::BathymetrySurface(std::shared_ptr<BathymetryData> data)
    : data_(std::move(data)) {}

Real BathymetrySurface::depth(Real x, Real y) const {
    return static_cast<Real>(data_->get_depth(x, y));
}

void BathymetrySurface::gradient(Real x, Real y, Real &dh_dx, Real &dh_dy) const {
    // Compute gradient using central differences
    double dx = std::abs(data_->geotransform[1]) * 0.5;
    double dy = std::abs(data_->geotransform[5]) * 0.5;

    float h_xp = data_->get_depth(x + dx, y);
    float h_xm = data_->get_depth(x - dx, y);
    float h_yp = data_->get_depth(x, y + dy);
    float h_ym = data_->get_depth(x, y - dy);

    dh_dx = (h_xp - h_xm) / (2.0 * dx);
    dh_dy = (h_yp - h_ym) / (2.0 * dy);
}

void BathymetrySurface::curvature(Real x, Real y, Real &d2h_dx2, Real &d2h_dxdy,
                                  Real &d2h_dy2) const {
    // Compute second derivatives using central differences
    double dx = std::abs(data_->geotransform[1]) * 0.5;
    double dy = std::abs(data_->geotransform[5]) * 0.5;

    // d²h/dx² = (h(x+dx) - 2*h(x) + h(x-dx)) / dx²
    float h_c = data_->get_depth(x, y);
    float h_xp = data_->get_depth(x + dx, y);
    float h_xm = data_->get_depth(x - dx, y);
    d2h_dx2 = (h_xp - 2.0 * h_c + h_xm) / (dx * dx);

    // d²h/dy² = (h(y+dy) - 2*h(y) + h(y-dy)) / dy²
    float h_yp = data_->get_depth(x, y + dy);
    float h_ym = data_->get_depth(x, y - dy);
    d2h_dy2 = (h_yp - 2.0 * h_c + h_ym) / (dy * dy);

    // d²h/dxdy = (h(x+dx,y+dy) - h(x+dx,y-dy) - h(x-dx,y+dy) + h(x-dx,y-dy)) /
    // (4*dx*dy)
    float h_pp = data_->get_depth(x + dx, y + dy);
    float h_pm = data_->get_depth(x + dx, y - dy);
    float h_mp = data_->get_depth(x - dx, y + dy);
    float h_mm = data_->get_depth(x - dx, y - dy);
    d2h_dxdy = (h_pp - h_pm - h_mp + h_mm) / (4.0 * dx * dy);
}

bool BathymetrySurface::is_water(Real x, Real y, Real min_depth) const {
    return data_->get_depth(x, y) >= min_depth;
}

bool BathymetrySurface::is_land(Real x, Real y) const { return data_->is_land(x, y); }

void BathymetrySurface::get_bounds(Real &xmin, Real &xmax, Real &ymin, Real &ymax) const {
    xmin = data_->xmin;
    xmax = data_->xmax;
    ymin = data_->ymin;
    ymax = data_->ymax;
}

// =============================================================================
// BathymetryMeshGenerator implementation
// =============================================================================

BathymetryMeshGenerator::BathymetryMeshGenerator(std::shared_ptr<BathymetryData> bathymetry)
    : bathymetry_(std::move(bathymetry)) {
    // Initialize config with bathymetry bounds
    if (bathymetry_->is_valid()) {
        config_.xmin = bathymetry_->xmin;
        config_.xmax = bathymetry_->xmax;
        config_.ymin = bathymetry_->ymin;
        config_.ymax = bathymetry_->ymax;
    }
}

void BathymetryMeshGenerator::set_config(const Config &config) { config_ = config; }

std::vector<ElementBounds> BathymetryMeshGenerator::generate() {
    std::vector<ElementBounds> elements;

    if (!bathymetry_->is_valid()) {
        return elements;
    }

    // Use config bounds or bathymetry bounds
    Real xmin = (config_.xmin < config_.xmax) ? config_.xmin : bathymetry_->xmin;
    Real xmax = (config_.xmin < config_.xmax) ? config_.xmax : bathymetry_->xmax;
    Real ymin = (config_.ymin < config_.ymax) ? config_.ymin : bathymetry_->ymin;
    Real ymax = (config_.ymin < config_.ymax) ? config_.ymax : bathymetry_->ymax;

    // Compute base cell sizes
    Real dx = (xmax - xmin) / config_.base_nx;
    Real dy = (ymax - ymin) / config_.base_ny;
    Real dz = (config_.zmax - config_.zmin) / config_.base_nz;

    // Create initial uniform grid, filtering out land cells
    for (int iz = 0; iz < config_.base_nz; ++iz) {
        for (int iy = 0; iy < config_.base_ny; ++iy) {
            for (int ix = 0; ix < config_.base_nx; ++ix) {
                ElementBounds bounds;
                bounds.xmin = xmin + ix * dx;
                bounds.xmax = xmin + (ix + 1) * dx;
                bounds.ymin = ymin + iy * dy;
                bounds.ymax = ymin + (iy + 1) * dy;
                bounds.zmin = config_.zmin + iz * dz;
                bounds.zmax = config_.zmin + (iz + 1) * dz;

                // Check if this cell contains water
                if (config_.mask_land) {
                    Real cx = 0.5 * (bounds.xmin + bounds.xmax);
                    Real cy = 0.5 * (bounds.ymin + bounds.ymax);
                    Real depth = bathymetry_->get_depth(cx, cy);

                    // Skip cells that are entirely on land
                    if (depth < config_.min_depth) {
                        // Also check corners
                        bool any_water = false;
                        for (int j = 0; j <= 1 && !any_water; ++j) {
                            for (int i = 0; i <= 1 && !any_water; ++i) {
                                Real x = (i == 0) ? bounds.xmin : bounds.xmax;
                                Real y = (j == 0) ? bounds.ymin : bounds.ymax;
                                if (bathymetry_->get_depth(x, y) >= config_.min_depth) {
                                    any_water = true;
                                }
                            }
                        }
                        if (!any_water)
                            continue;
                    }
                }

                elements.push_back(bounds);
            }
        }
    }

    return elements;
}

std::vector<VecX>
BathymetryMeshGenerator::compute_element_bathymetry(const std::vector<ElementBounds> &elements,
                                                    int order) const {

    std::vector<VecX> bathymetry(elements.size());

    // Get LGL nodes for this order
    HexahedronBasis basis(order);
    const VecX &nodes_1d = basis.lgl_basis_1d().nodes;
    int n1d = order + 1;
    int ndof = n1d * n1d * n1d;

    for (size_t e = 0; e < elements.size(); ++e) {
        const ElementBounds &bounds = elements[e];
        bathymetry[e].resize(ndof);

        Vec3 center = bounds.center();
        Vec3 size = bounds.size();

        for (int k = 0; k < n1d; ++k) {
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx = i + n1d * (j + n1d * k);

                    // Physical coordinates from reference coordinates
                    Real x = center(0) + 0.5 * nodes_1d(i) * size(0);
                    Real y = center(1) + 0.5 * nodes_1d(j) * size(1);

                    // Get bathymetry depth at this location
                    bathymetry[e](idx) = bathymetry_->get_depth(x, y);
                }
            }
        }
    }

    return bathymetry;
}

void BathymetryMeshGenerator::compute_element_gradients(const std::vector<ElementBounds> &elements,
                                                        int order, std::vector<VecX> &dh_dx,
                                                        std::vector<VecX> &dh_dy) const {

    dh_dx.resize(elements.size());
    dh_dy.resize(elements.size());

    HexahedronBasis basis(order);
    const VecX &nodes_1d = basis.lgl_basis_1d().nodes;
    int n1d = order + 1;
    int ndof = n1d * n1d * n1d;

    BathymetrySurface surface(bathymetry_);

    for (size_t e = 0; e < elements.size(); ++e) {
        const ElementBounds &bounds = elements[e];
        dh_dx[e].resize(ndof);
        dh_dy[e].resize(ndof);

        Vec3 center = bounds.center();
        Vec3 size = bounds.size();

        for (int k = 0; k < n1d; ++k) {
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx = i + n1d * (j + n1d * k);

                    Real x = center(0) + 0.5 * nodes_1d(i) * size(0);
                    Real y = center(1) + 0.5 * nodes_1d(j) * size(1);

                    Real grad_x, grad_y;
                    surface.gradient(x, y, grad_x, grad_y);

                    dh_dx[e](idx) = grad_x;
                    dh_dy[e](idx) = grad_y;
                }
            }
        }
    }
}

std::function<bool(const ElementBounds &)>
BathymetryMeshGenerator::create_refinement_function() const {
    return [this](const ElementBounds &bounds) -> bool { return this->should_refine(bounds); };
}

bool BathymetryMeshGenerator::should_refine(const ElementBounds &bounds) const {
    // Don't refine below minimum element size
    Real min_size = std::min({bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin});
    if (min_size < config_.min_element_size) {
        return false;
    }

    // Refine near coastline
    if (is_near_coastline(bounds)) {
        return true;
    }

    // Refine where bathymetry gradient is steep
    if (has_steep_gradient(bounds)) {
        return true;
    }

    return false;
}

bool BathymetryMeshGenerator::is_near_coastline(const ElementBounds &bounds) const {
    // Sample multiple points in the cell
    int n_samples = 3;
    Real dx = (bounds.xmax - bounds.xmin) / (n_samples - 1);
    Real dy = (bounds.ymax - bounds.ymin) / (n_samples - 1);

    bool found_water = false;
    bool found_land = false;

    for (int j = 0; j < n_samples && !(found_water && found_land); ++j) {
        for (int i = 0; i < n_samples && !(found_water && found_land); ++i) {
            Real x = bounds.xmin + i * dx;
            Real y = bounds.ymin + j * dy;
            Real depth = bathymetry_->get_depth(x, y);

            if (depth >= config_.min_depth) {
                found_water = true;
            } else {
                found_land = true;
            }
        }
    }

    // If cell contains both land and water, it's near coastline
    return found_water && found_land;
}

bool BathymetryMeshGenerator::has_steep_gradient(const ElementBounds &bounds) const {
    // Compute gradient at cell center
    Real cx = 0.5 * (bounds.xmin + bounds.xmax);
    Real cy = 0.5 * (bounds.ymin + bounds.ymax);

    BathymetrySurface surface(bathymetry_);
    Real dh_dx, dh_dy;
    surface.gradient(cx, cy, dh_dx, dh_dy);

    Real gradient_mag = std::sqrt(dh_dx * dh_dx + dh_dy * dh_dy);
    return gradient_mag > config_.bathymetry_gradient_threshold;
}

} // namespace drifter
