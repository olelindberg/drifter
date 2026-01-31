# Test configuration for DRIFTER
# Uses GoogleTest framework

include(GoogleTest)

# Common test utilities
set(TEST_UTILS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.hpp
)

# Unit tests
set(UNIT_TESTS
    unit/test_basis_hexahedron.cpp
    unit/test_quadrature_3d.cpp
    unit/test_sigma_coordinate.cpp
    unit/test_mortar.cpp
    unit/test_numerical_flux.cpp
    unit/test_equation_of_state.cpp
    unit/test_mode_splitting.cpp
    unit/test_time_stepper.cpp
    unit/test_io.cpp
    unit/test_amr.cpp
    unit/test_seabed_surface.cpp
    unit/test_wavelet_pyramid.cpp
    unit/test_weno_sampler.cpp
    unit/test_adaptive_bathymetry.cpp
    unit/test_quintic_basis_2d.cpp
    unit/test_quadtree_adapter.cpp
    unit/test_cg_dof_manager.cpp
    unit/test_biharmonic_assembler.cpp
    unit/test_bezier_basis_2d.cpp
    unit/test_thin_plate_hessian.cpp
    unit/test_bspline_basis.cpp
    unit/test_hierarchical_solver.cpp
)

# Integration tests
set(INTEGRATION_TESTS
    integration/test_dg_operators.cpp
    integration/test_conservation.cpp
    integration/test_mesh.cpp
    integration/test_initial_conditions.cpp
    integration/test_vtk_output.cpp
    integration/test_simulation_full.cpp
    integration/test_bathymetry.cpp
    integration/test_cg_bathymetry_smoother.cpp
    integration/test_bezier_bathymetry_smoother.cpp
    integration/test_thb_surface.cpp
    integration/test_hierarchical_solver.cpp
    integration/test_adaptive_bezier_smoother.cpp
    integration/test_cg_bezier_bathymetry_smoother.cpp
    integration/test_adaptive_cg_bezier_smoother.cpp
)

# Create unit test executable
add_executable(drifter_unit_tests
    ${UNIT_TESTS}
)

target_link_libraries(drifter_unit_tests PRIVATE
    drifter_lib
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(drifter_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create integration test executable
add_executable(drifter_integration_tests
    ${INTEGRATION_TESTS}
)

target_link_libraries(drifter_integration_tests PRIVATE
    drifter_lib
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(drifter_integration_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Register tests with CTest
gtest_discover_tests(drifter_unit_tests
    PROPERTIES LABELS "unit"
)

gtest_discover_tests(drifter_integration_tests
    PROPERTIES LABELS "integration"
)

