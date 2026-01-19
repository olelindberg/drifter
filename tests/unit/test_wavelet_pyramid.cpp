#include <gtest/gtest.h>
#include "bathymetry/wavelet_pyramid.hpp"
#include "../test_utils.hpp"
#include <cmath>

using namespace drifter;
using namespace drifter::testing;

class WaveletPyramidTest : public DrifterTestBase {
protected:
    // Create constant raster
    std::vector<float> create_constant_raster(int size, float value) {
        return std::vector<float>(size * size, value);
    }

    // Create linear raster: f(x,y) = x + y
    std::vector<float> create_linear_raster(int size) {
        std::vector<float> data(size * size);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                data[y * size + x] = static_cast<float>(x + y);
            }
        }
        return data;
    }

    // Create quadratic raster: f(x,y) = x^2 + y^2
    std::vector<float> create_quadratic_raster(int size) {
        std::vector<float> data(size * size);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                data[y * size + x] = static_cast<float>(x * x + y * y);
            }
        }
        return data;
    }

    // Create step function (discontinuity at x = size/2)
    std::vector<float> create_step_raster(int size) {
        std::vector<float> data(size * size);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                data[y * size + x] = (x < size / 2) ? 20.0f : 80.0f;
            }
        }
        return data;
    }

    // Create smooth sinusoidal raster
    std::vector<float> create_smooth_raster(int size) {
        std::vector<float> data(size * size);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                Real fx = static_cast<Real>(x) / size;
                Real fy = static_cast<Real>(y) / size;
                data[y * size + x] = static_cast<float>(
                    100.0 + 10.0 * std::sin(2.0 * M_PI * fx) * std::cos(2.0 * M_PI * fy));
            }
        }
        return data;
    }
};

// Test constant field reconstruction
TEST_F(WaveletPyramidTest, ConstantReconstructionExact) {
    int size = 64;
    float value = 42.0f;
    auto data = create_constant_raster(size, value);

    WaveletPyramid pyramid(data, size, size, 3, WaveletType::Bior44);

    // Full resolution reconstruction should match input
    MatX reconstructed = pyramid.reconstruct(0);

    ASSERT_EQ(reconstructed.rows(), size);
    ASSERT_EQ(reconstructed.cols(), size);

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            EXPECT_NEAR(reconstructed(y, x), value, 1e-10)
                << "Mismatch at (" << x << ", " << y << ")";
        }
    }
}

// Test linear field reconstruction (wavelets should preserve linear)
TEST_F(WaveletPyramidTest, LinearReconstructionExact) {
    int size = 64;
    auto data = create_linear_raster(size);

    WaveletPyramid pyramid(data, size, size, 3, WaveletType::Bior44);

    MatX reconstructed = pyramid.reconstruct(0);

    ASSERT_EQ(reconstructed.rows(), size);
    ASSERT_EQ(reconstructed.cols(), size);

    Real max_error = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            Real expected = static_cast<Real>(x + y);
            Real error = std::abs(reconstructed(y, x) - expected);
            max_error = std::max(max_error, error);
        }
    }

    // Biorthogonal wavelets preserve polynomials up to a certain degree
    EXPECT_LT(max_error, 1e-8) << "Linear reconstruction error too large";
}

// Test detail coefficients are very small for constant field
// Note: lifting scheme can produce small numerical artifacts at boundaries
TEST_F(WaveletPyramidTest, ConstantDetailZero) {
    int size = 64;
    auto data = create_constant_raster(size, 50.0f);

    WaveletPyramid pyramid(data, size, size, 3, WaveletType::Bior44);

    for (int level = 0; level < pyramid.num_levels(); ++level) {
        const auto& detail = pyramid.detail_coeffs(level);
        Real detail_norm = detail.LH.norm() + detail.HL.norm() + detail.HH.norm();
        // Allow small numerical noise from boundary handling
        EXPECT_LT(detail_norm, 1e-3)
            << "Detail at level " << level << " should be negligible for constant field";
    }
}

// Test detail coefficients capture discontinuity
TEST_F(WaveletPyramidTest, DiscontinuityDetection) {
    int size = 64;
    auto data = create_step_raster(size);

    WaveletPyramid pyramid(data, size, size, 3, WaveletType::Bior44);

    // Finest level detail should be non-zero at discontinuity
    const auto& detail = pyramid.detail_coeffs(0);
    Real detail_max = detail.max_detail();

    EXPECT_GT(detail_max, 1.0) << "Detail should capture step discontinuity";
}

// Test multi-level reconstruction dimensions
TEST_F(WaveletPyramidTest, MultiLevelDimensions) {
    int size = 128;
    auto data = create_smooth_raster(size);

    WaveletPyramid pyramid(data, size, size, 4, WaveletType::Bior44);

    // Level 0 = full resolution
    MatX level0 = pyramid.reconstruct(0);
    EXPECT_EQ(level0.rows(), 128);
    EXPECT_EQ(level0.cols(), 128);

    // Level 1 = half resolution
    MatX level1 = pyramid.reconstruct(1);
    EXPECT_EQ(level1.rows(), 64);
    EXPECT_EQ(level1.cols(), 64);

    // Level 2 = quarter resolution
    MatX level2 = pyramid.reconstruct(2);
    EXPECT_EQ(level2.rows(), 32);
    EXPECT_EQ(level2.cols(), 32);
}

// Test Bior22 vs Bior44 both work (perfect reconstruction)
TEST_F(WaveletPyramidTest, BothWaveletTypesWork) {
    int size = 64;
    auto data = create_quadratic_raster(size);

    WaveletPyramid pyramid22(data, size, size, 3, WaveletType::Bior22);
    WaveletPyramid pyramid44(data, size, size, 3, WaveletType::Bior44);

    MatX recon22 = pyramid22.reconstruct(0);
    MatX recon44 = pyramid44.reconstruct(0);

    // Both should achieve perfect reconstruction
    Real max_error22 = 0.0, max_error44 = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            Real expected = static_cast<Real>(x * x + y * y);
            max_error22 = std::max(max_error22, std::abs(recon22(y, x) - expected));
            max_error44 = std::max(max_error44, std::abs(recon44(y, x) - expected));
        }
    }

    // Both wavelet types should reconstruct with small error
    EXPECT_LT(max_error22, 1e-8) << "Bior22 should achieve perfect reconstruction";
    EXPECT_LT(max_error44, 1e-8) << "Bior44 should achieve perfect reconstruction";
}

// Test coordinate conversion
TEST_F(WaveletPyramidTest, CoordinateConversion) {
    int size = 64;
    auto data = create_constant_raster(size, 1.0f);

    // Use geotransform: origin at (100, 200), 10m pixel size
    std::array<double, 6> gt = {100.0, 10.0, 0.0, 200.0, 0.0, -10.0};

    WaveletPyramid pyramid(data, size, size, 3, gt, WaveletType::Bior44);

    // Test world to pixel at level 0
    Real px, py;
    pyramid.world_to_pixel(150.0, 150.0, 0, px, py);

    EXPECT_NEAR(px, 5.0, 0.01);  // (150 - 100) / 10 = 5
    EXPECT_NEAR(py, 5.0, 0.01);  // (150 - 200) / (-10) = 5

    // Test at level 1 (half resolution)
    pyramid.world_to_pixel(150.0, 150.0, 1, px, py);
    EXPECT_NEAR(px, 2.5, 0.01);  // 5 / 2 = 2.5
    EXPECT_NEAR(py, 2.5, 0.01);
}

// Test recommended level selection
TEST_F(WaveletPyramidTest, RecommendedLevel) {
    int size = 256;
    auto data = create_smooth_raster(size);

    std::array<double, 6> gt = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};  // 1m pixels
    WaveletPyramid pyramid(data, size, size, 5, gt, WaveletType::Bior44);

    // Small element should use fine resolution
    ElementBounds small_elem{0, 10, 0, 10, 0, 0};
    int small_level = pyramid.recommended_level(small_elem);
    EXPECT_LE(small_level, 2);

    // Large element should use coarse resolution
    ElementBounds large_elem{0, 100, 0, 100, 0, 0};
    int large_level = pyramid.recommended_level(large_elem);
    EXPECT_GE(large_level, 2);
}

// Test serialization round-trip
TEST_F(WaveletPyramidTest, SerializationRoundTrip) {
    int size = 32;
    auto data = create_smooth_raster(size);

    std::array<double, 6> gt = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
    WaveletPyramid original(data, size, size, 3, gt, WaveletType::Bior44);

    // Save and load
    std::string filename = "/tmp/test_wavelet_pyramid.bin";
    original.save(filename);
    WaveletPyramid loaded = WaveletPyramid::load(filename);

    // Check metadata
    EXPECT_EQ(loaded.sizex(), original.sizex());
    EXPECT_EQ(loaded.sizey(), original.sizey());
    EXPECT_EQ(loaded.num_levels(), original.num_levels());

    // Check reconstruction matches
    MatX orig_recon = original.reconstruct(0);
    MatX load_recon = loaded.reconstruct(0);

    Real max_diff = (orig_recon - load_recon).cwiseAbs().maxCoeff();
    EXPECT_LT(max_diff, 1e-12);

    // Cleanup
    std::remove(filename.c_str());
}

// Test detail energy
TEST_F(WaveletPyramidTest, DetailEnergy) {
    int size = 64;

    // Smooth data should have less detail energy
    auto smooth_data = create_smooth_raster(size);
    WaveletPyramid smooth_pyramid(smooth_data, size, size, 3, WaveletType::Bior44);

    // Discontinuous data should have more detail energy
    auto step_data = create_step_raster(size);
    WaveletPyramid step_pyramid(step_data, size, size, 3, WaveletType::Bior44);

    Real smooth_energy = smooth_pyramid.detail_coeffs(0).detail_energy();
    Real step_energy = step_pyramid.detail_coeffs(0).detail_energy();

    EXPECT_GT(step_energy, smooth_energy)
        << "Step function should have more detail energy than smooth function";
}
