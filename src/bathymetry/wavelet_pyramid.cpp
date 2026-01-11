#include "bathymetry/wavelet_pyramid.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace drifter {

// =============================================================================
// BiorthogonalFilters - Using lifting scheme coefficients
// =============================================================================

BiorthogonalFilters BiorthogonalFilters::bior22() {
    BiorthogonalFilters f;
    // Not used in lifting implementation, but kept for interface compatibility
    f.analysis_low.resize(1);
    f.analysis_low << 1.0;
    f.analysis_high = f.analysis_low;
    f.synthesis_low = f.analysis_low;
    f.synthesis_high = f.analysis_low;
    return f;
}

BiorthogonalFilters BiorthogonalFilters::bior44() {
    BiorthogonalFilters f;
    // Not used in lifting implementation, but kept for interface compatibility
    f.analysis_low.resize(1);
    f.analysis_low << 1.0;
    f.analysis_high = f.analysis_low;
    f.synthesis_low = f.analysis_low;
    f.synthesis_high = f.analysis_low;
    return f;
}

// =============================================================================
// WaveletLevel
// =============================================================================

Real WaveletLevel::detail_energy() const {
    return LH.squaredNorm() + HL.squaredNorm() + HH.squaredNorm();
}

Real WaveletLevel::max_detail() const {
    Real max_lh = LH.cwiseAbs().maxCoeff();
    Real max_hl = HL.cwiseAbs().maxCoeff();
    Real max_hh = HH.cwiseAbs().maxCoeff();
    return std::max({max_lh, max_hl, max_hh});
}

// =============================================================================
// CDF 9/7 Lifting Coefficients (used in JPEG 2000)
// =============================================================================

// Lifting scheme coefficients for CDF 9/7 wavelet
static constexpr double ALPHA = -1.586134342;   // Predict 1
static constexpr double BETA  = -0.052980118;   // Update 1
static constexpr double GAMMA =  0.882911075;   // Predict 2
static constexpr double DELTA =  0.443506852;   // Update 2
static constexpr double K     =  1.149604398;   // Scale factor
static constexpr double K_INV =  1.0 / K;       // Inverse scale

// CDF 5/3 lifting coefficients (simpler, for bior22)
static constexpr double ALPHA_53 = -0.5;        // Predict
static constexpr double BETA_53  =  0.25;       // Update

// =============================================================================
// WaveletPyramid - Construction
// =============================================================================

WaveletPyramid::WaveletPyramid(const std::vector<float>& data, int sizex, int sizey,
                               int num_levels, WaveletType type)
    : sizex_(sizex), sizey_(sizey), num_levels_(num_levels), type_(type) {

    geotransform_ = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};

    filters_ = (type == WaveletType::Bior22) ?
               BiorthogonalFilters::bior22() : BiorthogonalFilters::bior44();

    MatX input(sizey, sizex);
    for (int y = 0; y < sizey; ++y) {
        for (int x = 0; x < sizex; ++x) {
            input(y, x) = static_cast<Real>(data[y * sizex + x]);
        }
    }

    build_pyramid(input);
}

WaveletPyramid::WaveletPyramid(const std::vector<float>& data, int sizex, int sizey,
                               int num_levels, const std::array<double, 6>& geotransform,
                               WaveletType type)
    : sizex_(sizex), sizey_(sizey), num_levels_(num_levels),
      type_(type), geotransform_(geotransform) {

    filters_ = (type == WaveletType::Bior22) ?
               BiorthogonalFilters::bior22() : BiorthogonalFilters::bior44();

    MatX input(sizey, sizex);
    for (int y = 0; y < sizey; ++y) {
        for (int x = 0; x < sizex; ++x) {
            input(y, x) = static_cast<Real>(data[y * sizex + x]);
        }
    }

    build_pyramid(input);
}

void WaveletPyramid::build_pyramid(const MatX& input) {
    levels_.resize(num_levels_);

    MatX current = input;
    for (int level = 0; level < num_levels_; ++level) {
        if (current.rows() < 4 || current.cols() < 4) {
            num_levels_ = level;
            levels_.resize(num_levels_);
            break;
        }

        levels_[level] = decompose_level(current);
        current = levels_[level].LL;
    }

    coarsest_approx_ = current;
}

// =============================================================================
// 1D Lifting Forward Transform (in-place on vector)
// =============================================================================

void forward_lifting_97(std::vector<Real>& x) {
    int n = x.size();
    if (n < 2) return;

    // Split into even/odd
    std::vector<Real> even((n + 1) / 2);
    std::vector<Real> odd(n / 2);
    for (int i = 0; i < (n + 1) / 2; ++i) even[i] = x[2 * i];
    for (int i = 0; i < n / 2; ++i) odd[i] = x[2 * i + 1];

    int ne = even.size();
    int no = odd.size();

    // Predict 1: d[i] += alpha * (s[i] + s[i+1])
    for (int i = 0; i < no; ++i) {
        int ip1 = std::min(i + 1, ne - 1);
        odd[i] += ALPHA * (even[i] + even[ip1]);
    }

    // Update 1: s[i] += beta * (d[i-1] + d[i])
    for (int i = 0; i < ne; ++i) {
        int im1 = std::max(i - 1, 0);
        int idx = std::min(i, no - 1);
        odd[im1] = (i == 0) ? odd[0] : odd[im1];  // boundary
        even[i] += BETA * (odd[std::max(0, i-1)] + odd[std::min(i, no-1)]);
    }

    // Predict 2: d[i] += gamma * (s[i] + s[i+1])
    for (int i = 0; i < no; ++i) {
        int ip1 = std::min(i + 1, ne - 1);
        odd[i] += GAMMA * (even[i] + even[ip1]);
    }

    // Update 2: s[i] += delta * (d[i-1] + d[i])
    for (int i = 0; i < ne; ++i) {
        even[i] += DELTA * (odd[std::max(0, i-1)] + odd[std::min(i, no-1)]);
    }

    // Scale
    for (int i = 0; i < ne; ++i) even[i] *= K_INV;
    for (int i = 0; i < no; ++i) odd[i] *= K;

    // Merge: L then H
    for (int i = 0; i < ne; ++i) x[i] = even[i];
    for (int i = 0; i < no; ++i) x[ne + i] = odd[i];
}

void forward_lifting_53(std::vector<Real>& x) {
    int n = x.size();
    if (n < 2) return;

    std::vector<Real> even((n + 1) / 2);
    std::vector<Real> odd(n / 2);
    for (int i = 0; i < (n + 1) / 2; ++i) even[i] = x[2 * i];
    for (int i = 0; i < n / 2; ++i) odd[i] = x[2 * i + 1];

    int ne = even.size();
    int no = odd.size();

    // Predict: d[i] += alpha * (s[i] + s[i+1])
    for (int i = 0; i < no; ++i) {
        int ip1 = std::min(i + 1, ne - 1);
        odd[i] += ALPHA_53 * (even[i] + even[ip1]);
    }

    // Update: s[i] += beta * (d[i-1] + d[i])
    for (int i = 0; i < ne; ++i) {
        even[i] += BETA_53 * (odd[std::max(0, i-1)] + odd[std::min(i, no-1)]);
    }

    for (int i = 0; i < ne; ++i) x[i] = even[i];
    for (int i = 0; i < no; ++i) x[ne + i] = odd[i];
}

// =============================================================================
// 1D Lifting Inverse Transform (in-place on vector)
// =============================================================================

void inverse_lifting_97(std::vector<Real>& x) {
    int n = x.size();
    if (n < 2) return;

    int ne = (n + 1) / 2;
    int no = n / 2;

    // Split: L then H stored contiguously
    std::vector<Real> even(ne);
    std::vector<Real> odd(no);
    for (int i = 0; i < ne; ++i) even[i] = x[i];
    for (int i = 0; i < no; ++i) odd[i] = x[ne + i];

    // Undo scale
    for (int i = 0; i < ne; ++i) even[i] *= K;
    for (int i = 0; i < no; ++i) odd[i] *= K_INV;

    // Undo Update 2: s[i] -= delta * (d[i-1] + d[i])
    for (int i = 0; i < ne; ++i) {
        even[i] -= DELTA * (odd[std::max(0, i-1)] + odd[std::min(i, no-1)]);
    }

    // Undo Predict 2: d[i] -= gamma * (s[i] + s[i+1])
    for (int i = 0; i < no; ++i) {
        int ip1 = std::min(i + 1, ne - 1);
        odd[i] -= GAMMA * (even[i] + even[ip1]);
    }

    // Undo Update 1: s[i] -= beta * (d[i-1] + d[i])
    for (int i = 0; i < ne; ++i) {
        even[i] -= BETA * (odd[std::max(0, i-1)] + odd[std::min(i, no-1)]);
    }

    // Undo Predict 1: d[i] -= alpha * (s[i] + s[i+1])
    for (int i = 0; i < no; ++i) {
        int ip1 = std::min(i + 1, ne - 1);
        odd[i] -= ALPHA * (even[i] + even[ip1]);
    }

    // Interleave back
    for (int i = 0; i < ne; ++i) x[2 * i] = even[i];
    for (int i = 0; i < no; ++i) x[2 * i + 1] = odd[i];
}

void inverse_lifting_53(std::vector<Real>& x) {
    int n = x.size();
    if (n < 2) return;

    int ne = (n + 1) / 2;
    int no = n / 2;

    std::vector<Real> even(ne);
    std::vector<Real> odd(no);
    for (int i = 0; i < ne; ++i) even[i] = x[i];
    for (int i = 0; i < no; ++i) odd[i] = x[ne + i];

    // Undo Update: s[i] -= beta * (d[i-1] + d[i])
    for (int i = 0; i < ne; ++i) {
        even[i] -= BETA_53 * (odd[std::max(0, i-1)] + odd[std::min(i, no-1)]);
    }

    // Undo Predict: d[i] -= alpha * (s[i] + s[i+1])
    for (int i = 0; i < no; ++i) {
        int ip1 = std::min(i + 1, ne - 1);
        odd[i] -= ALPHA_53 * (even[i] + even[ip1]);
    }

    for (int i = 0; i < ne; ++i) x[2 * i] = even[i];
    for (int i = 0; i < no; ++i) x[2 * i + 1] = odd[i];
}

// =============================================================================
// WaveletPyramid - 2D Decomposition using Lifting
// =============================================================================

WaveletLevel WaveletPyramid::decompose_level(const MatX& input) const {
    WaveletLevel level;

    int rows = input.rows();
    int cols = input.cols();

    // Transform rows first
    MatX row_transformed(rows, cols);
    for (int y = 0; y < rows; ++y) {
        std::vector<Real> row(cols);
        for (int x = 0; x < cols; ++x) row[x] = input(y, x);

        if (type_ == WaveletType::Bior22) {
            forward_lifting_53(row);
        } else {
            forward_lifting_97(row);
        }

        for (int x = 0; x < cols; ++x) row_transformed(y, x) = row[x];
    }

    // Transform columns
    MatX full_transformed(rows, cols);
    for (int x = 0; x < cols; ++x) {
        std::vector<Real> col(rows);
        for (int y = 0; y < rows; ++y) col[y] = row_transformed(y, x);

        if (type_ == WaveletType::Bior22) {
            forward_lifting_53(col);
        } else {
            forward_lifting_97(col);
        }

        for (int y = 0; y < rows; ++y) full_transformed(y, x) = col[y];
    }

    // Extract subbands
    // After transform: rows [0, ne_rows) are low-pass, [ne_rows, rows) are high-pass
    // After row transform: cols [0, ne_cols) are low-pass, [ne_cols, cols) are high-pass
    int ne_rows = (rows + 1) / 2;
    int no_rows = rows / 2;
    int ne_cols = (cols + 1) / 2;
    int no_cols = cols / 2;

    level.LL = full_transformed.block(0, 0, ne_rows, ne_cols);
    level.LH = full_transformed.block(0, ne_cols, ne_rows, no_cols);
    level.HL = full_transformed.block(ne_rows, 0, no_rows, ne_cols);
    level.HH = full_transformed.block(ne_rows, ne_cols, no_rows, no_cols);

    return level;
}

// =============================================================================
// WaveletPyramid - 2D Reconstruction using Lifting
// =============================================================================

MatX WaveletPyramid::reconstruct(int level) const {
    if (level < 0) level = 0;
    if (level >= num_levels_) level = num_levels_;

    MatX result = coarsest_approx_;

    for (int l = num_levels_ - 1; l >= level; --l) {
        result = reconstruct_level(levels_[l], result);
    }

    return result;
}

MatX WaveletPyramid::reconstruct_level(const WaveletLevel& level, const MatX& approx) const {
    // Reconstruct the full matrix from subbands
    // approx should equal level.LL (or be computed from deeper levels)

    int ne_rows = approx.rows();
    int no_rows = level.HL.rows();
    int ne_cols = approx.cols();
    int no_cols = level.LH.cols();

    int rows = ne_rows + no_rows;
    int cols = ne_cols + no_cols;

    // Reassemble the full transformed matrix
    MatX full_transformed(rows, cols);
    full_transformed.block(0, 0, ne_rows, ne_cols) = approx;
    full_transformed.block(0, ne_cols, ne_rows, no_cols) = level.LH;
    full_transformed.block(ne_rows, 0, no_rows, ne_cols) = level.HL;
    full_transformed.block(ne_rows, ne_cols, no_rows, no_cols) = level.HH;

    // Inverse transform columns
    MatX col_reconstructed(rows, cols);
    for (int x = 0; x < cols; ++x) {
        std::vector<Real> col(rows);
        for (int y = 0; y < rows; ++y) col[y] = full_transformed(y, x);

        if (type_ == WaveletType::Bior22) {
            inverse_lifting_53(col);
        } else {
            inverse_lifting_97(col);
        }

        for (int y = 0; y < rows; ++y) col_reconstructed(y, x) = col[y];
    }

    // Inverse transform rows
    MatX result(rows, cols);
    for (int y = 0; y < rows; ++y) {
        std::vector<Real> row(cols);
        for (int x = 0; x < cols; ++x) row[x] = col_reconstructed(y, x);

        if (type_ == WaveletType::Bior22) {
            inverse_lifting_53(row);
        } else {
            inverse_lifting_97(row);
        }

        for (int x = 0; x < cols; ++x) result(y, x) = row[x];
    }

    return result;
}

MatX WaveletPyramid::reconstruct_region(const ElementBounds& bounds, int level) const {
    MatX full = reconstruct(level);

    Real px_min, py_min, px_max, py_max;
    world_to_pixel(bounds.xmin, bounds.ymax, level, px_min, py_min);
    world_to_pixel(bounds.xmax, bounds.ymin, level, px_max, py_max);

    int ix_min = std::max(0, static_cast<int>(std::floor(px_min)));
    int iy_min = std::max(0, static_cast<int>(std::floor(py_min)));
    int ix_max = std::min(static_cast<int>(full.cols()) - 1, static_cast<int>(std::ceil(px_max)));
    int iy_max = std::min(static_cast<int>(full.rows()) - 1, static_cast<int>(std::ceil(py_max)));

    if (ix_max < ix_min || iy_max < iy_min) {
        return MatX::Zero(1, 1);
    }

    return full.block(iy_min, ix_min, iy_max - iy_min + 1, ix_max - ix_min + 1);
}

// =============================================================================
// WaveletPyramid - Query Methods
// =============================================================================

const WaveletLevel& WaveletPyramid::detail_coeffs(int level) const {
    if (level < 0 || level >= num_levels_) {
        throw std::out_of_range("Invalid wavelet level");
    }
    return levels_[level];
}

int WaveletPyramid::recommended_level(const ElementBounds& bounds) const {
    Real elem_size = std::min(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin);
    Real pixel_size = std::abs(geotransform_[1]);
    Real target_pixels = 6.0;
    Real pixels_at_level0 = elem_size / pixel_size;

    int level = 0;
    while (pixels_at_level0 / (1 << level) > target_pixels && level < num_levels_) {
        ++level;
    }

    return std::min(level, num_levels_ - 1);
}

Real WaveletPyramid::max_detail_in_region(const ElementBounds& bounds, int level) const {
    if (level < 0 || level >= num_levels_) {
        return 0.0;
    }

    const auto& det = levels_[level];

    int scale = 1 << (level + 1);
    Real px_min, py_min, px_max, py_max;
    world_to_pixel(bounds.xmin, bounds.ymax, 0, px_min, py_min);
    world_to_pixel(bounds.xmax, bounds.ymin, 0, px_max, py_max);

    int ix_min = std::max(0, static_cast<int>(std::floor(px_min / scale)));
    int iy_min = std::max(0, static_cast<int>(std::floor(py_min / scale)));
    int ix_max = std::min(static_cast<int>(det.LH.cols()) - 1,
                         static_cast<int>(std::ceil(px_max / scale)));
    int iy_max = std::min(static_cast<int>(det.LH.rows()) - 1,
                         static_cast<int>(std::ceil(py_max / scale)));

    if (ix_max < ix_min || iy_max < iy_min) {
        return 0.0;
    }

    Real max_val = 0.0;
    for (int y = iy_min; y <= iy_max; ++y) {
        for (int x = ix_min; x <= ix_max; ++x) {
            max_val = std::max(max_val, std::abs(det.LH(y, x)));
            max_val = std::max(max_val, std::abs(det.HL(y, x)));
            max_val = std::max(max_val, std::abs(det.HH(y, x)));
        }
    }

    return max_val;
}

// =============================================================================
// WaveletPyramid - Coordinate Conversion
// =============================================================================

void WaveletPyramid::world_to_pixel(Real wx, Real wy, int level,
                                    Real& px, Real& py) const {
    Real px0 = (wx - geotransform_[0]) / geotransform_[1];
    Real py0 = (wy - geotransform_[3]) / geotransform_[5];

    int scale = 1 << level;
    px = px0 / scale;
    py = py0 / scale;
}

void WaveletPyramid::pixel_to_world(Real px, Real py, int level,
                                    Real& wx, Real& wy) const {
    int scale = 1 << level;
    Real px0 = px * scale;
    Real py0 = py * scale;

    wx = geotransform_[0] + px0 * geotransform_[1] + py0 * geotransform_[2];
    wy = geotransform_[3] + px0 * geotransform_[4] + py0 * geotransform_[5];
}

// =============================================================================
// Unused convolution methods (kept for interface but lifting is used instead)
// =============================================================================

VecX WaveletPyramid::convolve_symmetric(const VecX& signal, const VecX& filter) const {
    // Not used in lifting implementation
    return signal;
}

VecX WaveletPyramid::downsample(const VecX& signal) const {
    int n = (signal.size() + 1) / 2;
    VecX result(n);
    for (int i = 0; i < n; ++i) {
        result(i) = signal(2 * i);
    }
    return result;
}

VecX WaveletPyramid::upsample(const VecX& signal) const {
    int n = signal.size() * 2;
    VecX result = VecX::Zero(n);
    for (int i = 0; i < signal.size(); ++i) {
        result(2 * i) = signal(i);
    }
    return result;
}

MatX WaveletPyramid::filter_rows(const MatX& input, const VecX& filter) const {
    return input;  // Not used
}

MatX WaveletPyramid::filter_cols(const MatX& input, const VecX& filter) const {
    return input;  // Not used
}

MatX WaveletPyramid::downsample_rows(const MatX& input) const {
    return input;  // Not used
}

MatX WaveletPyramid::downsample_cols(const MatX& input) const {
    return input;  // Not used
}

MatX WaveletPyramid::upsample_rows(const MatX& input) const {
    return input;  // Not used
}

MatX WaveletPyramid::upsample_cols(const MatX& input) const {
    return input;  // Not used
}

// =============================================================================
// WaveletPyramid - Serialization
// =============================================================================

void WaveletPyramid::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    out.write(reinterpret_cast<const char*>(&sizex_), sizeof(sizex_));
    out.write(reinterpret_cast<const char*>(&sizey_), sizeof(sizey_));
    out.write(reinterpret_cast<const char*>(&num_levels_), sizeof(num_levels_));
    out.write(reinterpret_cast<const char*>(&type_), sizeof(type_));
    out.write(reinterpret_cast<const char*>(geotransform_.data()),
              sizeof(double) * 6);

    int rows = coarsest_approx_.rows();
    int cols = coarsest_approx_.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(coarsest_approx_.data()),
              sizeof(Real) * rows * cols);

    for (int l = 0; l < num_levels_; ++l) {
        const auto& level = levels_[l];

        auto write_matrix = [&](const MatX& m) {
            int r = m.rows(), c = m.cols();
            out.write(reinterpret_cast<const char*>(&r), sizeof(r));
            out.write(reinterpret_cast<const char*>(&c), sizeof(c));
            out.write(reinterpret_cast<const char*>(m.data()), sizeof(Real) * r * c);
        };

        write_matrix(level.LL);
        write_matrix(level.LH);
        write_matrix(level.HL);
        write_matrix(level.HH);
    }
}

WaveletPyramid WaveletPyramid::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    WaveletPyramid pyramid;

    in.read(reinterpret_cast<char*>(&pyramid.sizex_), sizeof(pyramid.sizex_));
    in.read(reinterpret_cast<char*>(&pyramid.sizey_), sizeof(pyramid.sizey_));
    in.read(reinterpret_cast<char*>(&pyramid.num_levels_), sizeof(pyramid.num_levels_));
    in.read(reinterpret_cast<char*>(&pyramid.type_), sizeof(pyramid.type_));
    in.read(reinterpret_cast<char*>(pyramid.geotransform_.data()), sizeof(double) * 6);

    pyramid.filters_ = (pyramid.type_ == WaveletType::Bior22) ?
                       BiorthogonalFilters::bior22() : BiorthogonalFilters::bior44();

    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    pyramid.coarsest_approx_.resize(rows, cols);
    in.read(reinterpret_cast<char*>(pyramid.coarsest_approx_.data()),
            sizeof(Real) * rows * cols);

    pyramid.levels_.resize(pyramid.num_levels_);
    for (int l = 0; l < pyramid.num_levels_; ++l) {
        auto& level = pyramid.levels_[l];

        auto read_matrix = [&](MatX& m) {
            int r, c;
            in.read(reinterpret_cast<char*>(&r), sizeof(r));
            in.read(reinterpret_cast<char*>(&c), sizeof(c));
            m.resize(r, c);
            in.read(reinterpret_cast<char*>(m.data()), sizeof(Real) * r * c);
        };

        read_matrix(level.LL);
        read_matrix(level.LH);
        read_matrix(level.HL);
        read_matrix(level.HH);
    }

    return pyramid;
}

}  // namespace drifter
