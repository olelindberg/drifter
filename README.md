# DRIFTER

A 3D Discontinuous Galerkin (DG) adaptive multi-resolution coastal ocean circulation model written in C++20. DRIFTER implements the primitive equations for ocean modeling with terrain-following sigma coordinates, featuring:

- High-order DG discretization with tensor-product Lagrange basis functions
- Anisotropic adaptive mesh refinement (AMR) via directional octree
- Barotropic/baroclinic mode splitting for efficient time integration
- GeoTIFF bathymetry loading and coastline-adaptive refinement
- VTK and Zarr output formats

## Building

```bash
# Configure (from project root)
cmake -B build

# Build
LD_LIBRARY_PATH=/home/ole/.local/lib cmake --build build --parallel
```

## Running Tests

```bash
# Run all tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build --output-on-failure

# Run only unit tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -L unit --output-on-failure

# Run only integration tests
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -L integration --output-on-failure

# Run a single test by name
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -R "TestName" --output-on-failure

# List available test names
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -N

# List test names with labels
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -N --verbose

# Run tests with verbose output
LD_LIBRARY_PATH=/home/ole/.local/lib ctest --test-dir build -V

# Run test executable directly (shows all logging output)
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_unit_tests
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_integration_tests

# Run specific test(s) with gtest filter
LD_LIBRARY_PATH=/home/ole/.local/lib ./build/tests/drifter_unit_tests --gtest_filter="TestName*"
```

The `LD_LIBRARY_PATH` is needed for GDAL and other libraries installed in `/home/ole/.local/lib`.
