#pragma once

// WaveletPyramid - Biorthogonal wavelet decomposition for multi-resolution bathymetry
//
// Implements Cohen-Daubechies-Feauveau (CDF) biorthogonal wavelets (2.2 and 4.4)
// for multi-resolution analysis of bathymetry raster data. The pyramid structure
// enables:
// - Adaptive resolution selection based on element size
// - Detail coefficient analysis for AMR guidance
// - Efficient reconstruction at arbitrary levels
//
// Usage:
//   WaveletPyramid pyramid(bathy.elevation, bathy.sizex, bathy.sizey, 5);
//   MatX coarse = pyramid.reconstruct(3);  // 3 levels coarser than original
//   const auto& detail = pyramid.detail_coeffs(0);  // Finest level detail

#include "core/types.hpp"
#include "mesh/octree_adapter.hpp"
#include <array>
#include <string>
#include <vector>

namespace drifter {

/// @brief Biorthogonal filter bank for wavelet transform
struct BiorthogonalFilters {
    VecX analysis_low;     // Low-pass decomposition filter (h)
    VecX analysis_high;    // High-pass decomposition filter (g)
    VecX synthesis_low;    // Low-pass reconstruction filter (h~)
    VecX synthesis_high;   // High-pass reconstruction filter (g~)

    /// @brief Create CDF 2.2 (bior2.2) filter bank
    static BiorthogonalFilters bior22();

    /// @brief Create CDF 4.4 (bior4.4) filter bank - 9/7 wavelets
    static BiorthogonalFilters bior44();
};

/// @brief Wavelet coefficients at a single decomposition level
struct WaveletLevel {
    MatX LL;  // Approximation coefficients (input to next level)
    MatX LH;  // Horizontal detail (vertical edges)
    MatX HL;  // Vertical detail (horizontal edges)
    MatX HH;  // Diagonal detail

    /// @brief Total energy in detail coefficients
    Real detail_energy() const;

    /// @brief Maximum absolute detail coefficient
    Real max_detail() const;
};

/// @brief Wavelet type enumeration
enum class WaveletType {
    Bior22,  // CDF 2.2 - simpler, 5/3 taps
    Bior44   // CDF 4.4 - smoother, 9/7 taps (recommended)
};

/// @brief Multi-resolution wavelet pyramid for bathymetry data
///
/// Stores a complete wavelet decomposition of the input raster data.
/// Supports reconstruction at any level and region-based queries.
class WaveletPyramid {
public:
    /// @brief Construct wavelet pyramid from raw bathymetry data
    /// @param data Row-major raster data (size = sizex * sizey)
    /// @param sizex Number of columns (x-dimension)
    /// @param sizey Number of rows (y-dimension)
    /// @param num_levels Number of decomposition levels
    /// @param type Wavelet type (Bior44 recommended)
    WaveletPyramid(const std::vector<float>& data, int sizex, int sizey,
                   int num_levels, WaveletType type = WaveletType::Bior44);

    /// @brief Construct from geotransform for coordinate support
    WaveletPyramid(const std::vector<float>& data, int sizex, int sizey,
                   int num_levels, const std::array<double, 6>& geotransform,
                   WaveletType type = WaveletType::Bior44);

    /// @brief Default constructor for deserialization
    WaveletPyramid() = default;

    /// @brief Reconstruct bathymetry at specified level
    /// @param level 0 = full resolution, higher = coarser
    /// @return Reconstructed data at that resolution
    MatX reconstruct(int level) const;

    /// @brief Reconstruct bathymetry within a bounding box
    /// @param bounds Element bounds in world coordinates
    /// @param level Resolution level
    /// @return Reconstructed data for the region
    MatX reconstruct_region(const ElementBounds& bounds, int level) const;

    /// @brief Get detail coefficients at a level
    /// @param level Decomposition level (0 = finest detail)
    const WaveletLevel& detail_coeffs(int level) const;

    /// @brief Get the coarsest approximation (deepest LL)
    const MatX& coarsest_approximation() const { return coarsest_approx_; }

    /// @brief Get recommended reconstruction level for an element
    /// @param bounds Element bounds
    /// @return Suggested level based on element size vs raster resolution
    int recommended_level(const ElementBounds& bounds) const;

    /// @brief Get maximum detail magnitude in a region
    /// @param bounds Region bounds in world coordinates
    /// @param level Detail level to query
    Real max_detail_in_region(const ElementBounds& bounds, int level) const;

    /// @brief Serialize pyramid to file
    void save(const std::string& filename) const;

    /// @brief Load pyramid from file
    static WaveletPyramid load(const std::string& filename);

    // Accessors
    int sizex() const { return sizex_; }
    int sizey() const { return sizey_; }
    int num_levels() const { return num_levels_; }
    WaveletType wavelet_type() const { return type_; }
    const std::array<double, 6>& geotransform() const { return geotransform_; }

    /// @brief Convert world coordinates to pixel coordinates at given level
    void world_to_pixel(Real wx, Real wy, int level, Real& px, Real& py) const;

    /// @brief Convert pixel coordinates to world coordinates at given level
    void pixel_to_world(Real px, Real py, int level, Real& wx, Real& wy) const;

private:
    int sizex_ = 0;
    int sizey_ = 0;
    int num_levels_ = 0;
    WaveletType type_ = WaveletType::Bior44;
    std::array<double, 6> geotransform_ = {0, 1, 0, 0, 0, -1};  // Identity

    BiorthogonalFilters filters_;
    std::vector<WaveletLevel> levels_;
    MatX coarsest_approx_;

    // Build the pyramid from input data
    void build_pyramid(const MatX& input);

    // Single-level decomposition
    WaveletLevel decompose_level(const MatX& input) const;

    // Single-level reconstruction
    MatX reconstruct_level(const WaveletLevel& level, const MatX& approx) const;

    // 1D convolution with symmetric boundary extension
    VecX convolve_symmetric(const VecX& signal, const VecX& filter) const;

    // Downsample by factor 2 (keep even indices)
    VecX downsample(const VecX& signal) const;

    // Upsample by factor 2 (insert zeros)
    VecX upsample(const VecX& signal) const;

    // Row-wise filtering
    MatX filter_rows(const MatX& input, const VecX& filter) const;

    // Column-wise filtering
    MatX filter_cols(const MatX& input, const VecX& filter) const;

    // Row-wise downsampling
    MatX downsample_rows(const MatX& input) const;

    // Column-wise downsampling
    MatX downsample_cols(const MatX& input) const;

    // Row-wise upsampling
    MatX upsample_rows(const MatX& input) const;

    // Column-wise upsampling
    MatX upsample_cols(const MatX& input) const;
};

}  // namespace drifter
