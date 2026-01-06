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
)

# Integration tests
set(INTEGRATION_TESTS
    integration/test_dg_operators.cpp
    integration/test_conservation.cpp
    integration/test_time_stepping.cpp
    integration/test_mesh.cpp
    integration/test_initial_conditions.cpp
    integration/test_vtk_output.cpp
    integration/test_simulation_full.cpp
    integration/test_bathymetry.cpp
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

# MPI tests (if MPI enabled)
if(DRIFTER_USE_MPI)
    add_executable(drifter_mpi_tests
        mpi/test_halo_exchange.cpp
        mpi/test_domain_decomposition.cpp
    )

    target_link_libraries(drifter_mpi_tests PRIVATE
        drifter_lib
        GTest::gtest
        GTest::gtest_main
    )

    # MPI tests need to be run with mpirun
    add_test(
        NAME mpi_tests
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:drifter_mpi_tests>
    )
    set_tests_properties(mpi_tests PROPERTIES LABELS "mpi")
endif()
