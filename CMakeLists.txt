cmake_minimum_required(VERSION 3.20)
project(drifter
    VERSION 0.1.0
    DESCRIPTION "3D Discontinuous Galerkin Adaptive Multi-Resolution Coastal Ocean Circulation Model"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Set RPATH to find libraries in custom locations
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH "/home/ole/.local/lib")

# Build options
option(DRIFTER_USE_MPI "Enable MPI parallelization" ON)
option(DRIFTER_USE_OPENMP "Enable OpenMP threading" ON)
option(DRIFTER_USE_CUDA "Enable CUDA GPU acceleration" OFF)
option(DRIFTER_USE_ZARR "Enable Zarr v3 output via zarrs_ffi" ON)
option(DRIFTER_USE_VTK "Enable VTK output" ON)
option(DRIFTER_BUILD_TESTS "Build unit tests" ON)
option(DRIFTER_BUILD_DOCS "Build documentation" OFF)
option(DRIFTER_USE_METIS "Enable METIS ordering for sparse solvers" ON)
option(DRIFTER_USE_CHOLMOD "Enable CHOLMOD supernodal Cholesky (SuiteSparse)" OFF)
option(DRIFTER_USE_UMFPACK "Enable UMFPACK LU (SuiteSparse)" OFF)
option(DRIFTER_USE_MKL "Enable Intel MKL PARDISO (parallel direct solver)" OFF)

# Find required packages
find_package(Eigen3 3.4 REQUIRED)
find_package(Boost REQUIRED)  # For Boost.Geometry coastline refinement

if(DRIFTER_USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DDRIFTER_USE_MPI)
endif()

if(DRIFTER_USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(DRIFTER_USE_METIS)
    # Try CMake config first, then manual detection for Ubuntu's libmetis-dev
    find_package(METIS QUIET CONFIG)
    if(NOT METIS_FOUND)
        find_library(METIS_LIBRARY NAMES metis)
        find_path(METIS_INCLUDE_DIR NAMES metis.h PATH_SUFFIXES metis)
        if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
            set(METIS_FOUND TRUE)
            include_directories(${METIS_INCLUDE_DIR})
        endif()
    endif()
    if(METIS_FOUND)
        add_definitions(-DDRIFTER_USE_METIS)
        message(STATUS "METIS found: ${METIS_LIBRARY}")
    else()
        message(WARNING "METIS not found, using default COLAMD ordering")
    endif()
endif()

# Optional backends for the Hermite smoother's direct solve. Each option decides
# only whether a backend is *available*; which available one runs is
# "smoother": {"solver": ...} in the JSON config, so they can be compared inside
# a single run (tests/benchmarks/test_hermite_solver_comparison.cpp).
#
# All three are off by default: the default SimplicialLDLT needs no optional
# dependency. When an option is on, a missing library is a hard error rather than
# a quiet downgrade to another backend.

# CHOLMOD supernodal Cholesky. Measured at 18.7 s against SimplicialLDLT's 1.9 s
# on the condensed matrix of a real adaptive run - but under Eigen's default AMD
# ordering, which gives a supernodal factorisation nothing to work with. The
# CholmodSupernodalNesdis solver kind asks for nested dissection instead; Debian's
# libcholmod is linked against libmetis and can supply it.
if(DRIFTER_USE_CHOLMOD)
    find_library(CHOLMOD_LIBRARY NAMES cholmod)
    find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h PATH_SUFFIXES suitesparse)
    if(CHOLMOD_LIBRARY AND CHOLMOD_INCLUDE_DIR)
        add_definitions(-DDRIFTER_USE_CHOLMOD)
        include_directories(${CHOLMOD_INCLUDE_DIR})
        message(STATUS "CHOLMOD found: ${CHOLMOD_LIBRARY}")
    else()
        message(FATAL_ERROR
            "CHOLMOD not found (install libsuitesparse-dev), "
            "or configure with -DDRIFTER_USE_CHOLMOD=OFF to use Eigen's SimplicialLDLT")
    endif()
endif()

# UMFPACK. An unsymmetric LU on a matrix that is SPD, so it does roughly twice
# the arithmetic of a Cholesky and is not expected to win on speed; it is here as
# a reference point and for the case where the system is driven indefinite.
# Eigen's UmfPackSupport calls only umfpack_di_*, and libumfpack already pulls
# amd/suitesparseconfig/blas itself, so the one library is enough.
if(DRIFTER_USE_UMFPACK)
    find_library(UMFPACK_LIBRARY NAMES umfpack)
    find_path(UMFPACK_INCLUDE_DIR NAMES umfpack.h PATH_SUFFIXES suitesparse)
    if(UMFPACK_LIBRARY AND UMFPACK_INCLUDE_DIR)
        add_definitions(-DDRIFTER_USE_UMFPACK)
        include_directories(${UMFPACK_INCLUDE_DIR})
        message(STATUS "UMFPACK found: ${UMFPACK_LIBRARY}")
    else()
        message(FATAL_ERROR
            "UMFPACK not found (install libsuitesparse-dev), "
            "or configure with -DDRIFTER_USE_UMFPACK=OFF")
    endif()
endif()

# Intel MKL PARDISO: the only backend here that uses more than one core for the
# factorisation, which is the whole reason to have it.
#
# The three layers are found by hand rather than through pkg-config because
# Debian ships only the -iomp and -seq variants of the .pc files, and we want
# neither: OpenBLAS here is the OpenMP build, linked against libgomp, and a
# second OpenMP runtime (MKL's libiomp5, or the libomp that libmkl-dev pulls in)
# in the same process oversubscribes the cores at best. libmkl_gnu_thread.so
# declares no OpenMP runtime of its own - it resolves those symbols from the
# executable - so linking it together with libgomp leaves exactly one.
if(DRIFTER_USE_MKL)
    find_library(MKL_INTERFACE_LIBRARY NAMES mkl_intel_lp64)
    find_library(MKL_THREADING_LIBRARY NAMES mkl_gnu_thread)
    find_library(MKL_CORE_LIBRARY NAMES mkl_core)
    find_path(MKL_INCLUDE_DIR NAMES mkl_pardiso.h PATH_SUFFIXES mkl)
    if(MKL_INTERFACE_LIBRARY AND MKL_THREADING_LIBRARY AND MKL_CORE_LIBRARY AND MKL_INCLUDE_DIR)
        add_definitions(-DDRIFTER_USE_MKL)
        include_directories(${MKL_INCLUDE_DIR})
        # "-fopenmp" here is a *link* flag: it makes the driver link libgomp,
        # whose symbols mkl_gnu_thread leaves to the executable to resolve.
        # DRIFTER_USE_OPENMP already brings libgomp in when it is ON, which is
        # its default; this keeps the MKL block correct when it is not, without
        # OpenMP::OpenMP_CXX and the compile option it would carry with it.
        # MKL resolves across its three layers, so they must be a link group.
        set(MKL_LIBRARIES
            -Wl,--start-group
            ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY}
            -Wl,--end-group
            -fopenmp ${CMAKE_DL_LIBS})
        message(STATUS "MKL found: ${MKL_CORE_LIBRARY} (GNU threading layer)")
    else()
        message(FATAL_ERROR
            "MKL not found (install libmkl-dev), "
            "or configure with -DDRIFTER_USE_MKL=OFF")
    endif()
endif()

# Required: GDAL for geospatial data (bathymetry, coastlines)
find_package(GDAL REQUIRED)
message(STATUS "GDAL found: ${GDAL_VERSION}")

# Optional: VTK for visualization output
if(DRIFTER_USE_VTK)
    find_package(VTK QUIET COMPONENTS CommonCore CommonDataModel IOXML)
    if(VTK_FOUND)
        add_definitions(-DDRIFTER_HAS_VTK)
        message(STATUS "VTK found: ${VTK_VERSION}")
    else()
        message(WARNING "VTK not found, VTK output disabled")
    endif()
endif()

# Optional: zarrs_ffi for Zarr v3 output (Rust library with C bindings)
if(DRIFTER_USE_ZARR)
    # Look for zarrs_ffi library and header
    find_library(ZARRS_FFI_LIB
        NAMES zarrs_ffi
        HINTS ${ZARRS_FFI_DIR}/lib ${ZARRS_FFI_DIR}
        PATH_SUFFIXES release
    )
    find_path(ZARRS_FFI_INCLUDE
        NAMES zarrs.h
        HINTS ${ZARRS_FFI_DIR}/include ${ZARRS_FFI_DIR}
    )
    if(ZARRS_FFI_LIB AND ZARRS_FFI_INCLUDE)
        add_definitions(-DDRIFTER_HAS_ZARR)
        message(STATUS "zarrs_ffi found: ${ZARRS_FFI_LIB}")
    else()
        message(WARNING "zarrs_ffi not found (set ZARRS_FFI_DIR), Zarr output disabled")
    endif()
endif()

# Optional legacy packages
find_package(netCDF CONFIG QUIET)
find_package(HDF5 QUIET COMPONENTS CXX)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# GDAL include directories (required)
include_directories(${GDAL_INCLUDE_DIRS})
if(VTK_FOUND)
    include_directories(${VTK_INCLUDE_DIRS})
endif()
if(ZARRS_FFI_INCLUDE)
    include_directories(${ZARRS_FFI_INCLUDE})
endif()

# Source files by module
set(CORE_SOURCES
    src/core/coastline_config.cpp
    src/core/config.cpp
    src/core/config_reader.cpp
    src/core/drifter.cpp
    src/core/logger.cpp
    src/core/timer.cpp
    src/core/lowrider.cpp
    src/core/lowrider_config_reader.cpp
)

set(MESH_SOURCES
    src/mesh/element.cpp
    src/mesh/mesh.cpp
    src/mesh/mesh_io.cpp
    src/mesh/geometry.cpp
    src/mesh/octree_adapter.cpp
    src/mesh/sigma_coordinate.cpp
    src/mesh/vertical_tfi.cpp
    src/mesh/geotiff_reader.cpp
    src/mesh/coastline_refinement.cpp
    src/mesh/seabed_surface.cpp
    src/mesh/multi_source_bathymetry.cpp
    src/mesh/area_of_interest.cpp
)

set(BATHYMETRY_SOURCES
    src/bathymetry/quintic_basis_2d.cpp
    src/bathymetry/lagrange_basis_2d.cpp
    src/bathymetry/quadtree_adapter.cpp
    src/bathymetry/cg_dof_manager.cpp
    src/bathymetry/biharmonic_assembler.cpp
    src/bathymetry/bezier_basis_2d.cpp
    src/bathymetry/thin_plate_hessian.cpp
    src/bathymetry/bezier_data_fitting.cpp
    src/bathymetry/constraint_condenser.cpp
    src/bathymetry/element_data_mask.cpp
    # CG Bezier DOF manager (shared by cubic smoother)
    src/bathymetry/cg_bezier_dof_manager.cpp
    # Base classes for CG Bezier bathymetry smoothers
    src/bathymetry/cg_surface_dof_manager_base.cpp
    src/bathymetry/cg_smoother_base.cpp
    src/bathymetry/adaptive_cg_smoother_base.cpp
    # CG Bezier bathymetry smoother (cubic with C¹ continuity)
    src/bathymetry/cubic_bezier_basis_2d.cpp
    src/bathymetry/cubic_thin_plate_hessian.cpp
    src/bathymetry/cg_cubic_bezier_dof_manager.cpp
    src/bathymetry/cg_cubic_bezier_bathymetry_smoother.cpp
    src/bathymetry/bezier_multigrid_preconditioner.cpp
    src/bathymetry/adaptive_cg_cubic_bezier_smoother.cpp
    # CG Bezier bathymetry smoother (linear with C⁰ continuity)
    src/bathymetry/linear_bezier_basis_2d.cpp
    src/bathymetry/dirichlet_hessian.cpp
    src/bathymetry/cg_linear_bezier_dof_manager.cpp
    src/bathymetry/cg_linear_bezier_bathymetry_smoother.cpp
    src/bathymetry/adaptive_cg_linear_bezier_smoother.cpp
    # CG Hermite bathymetry smoother (C⁰ and C¹, structural continuity, SPD system)
    src/bathymetry/hermite_basis_1d.cpp
    src/bathymetry/hermite_basis_2d.cpp
    src/bathymetry/hermite_energy_matrices.cpp
    src/bathymetry/hermite_hessian.cpp
    src/bathymetry/cg_hermite_dof_manager.cpp
    src/bathymetry/cg_hermite_bathymetry_smoother.cpp
    src/bathymetry/adaptive_cg_hermite_smoother.cpp
    # Standalone iterative methods for smoothing/solving
    src/bathymetry/jacobi_method.cpp
    src/bathymetry/schwarz_method.cpp
    src/bathymetry/schwarz_block_builder.cpp
    src/bathymetry/iterative_solver.cpp
    src/bathymetry/iterative_method_factory.cpp
    # Schur complement preconditioners for constrained CG
    src/bathymetry/diagonal_approx_cg_schur_preconditioner.cpp
    src/bathymetry/block_diag_approx_cg_schur_preconditioner.cpp
    src/bathymetry/flexible_cg.cpp
    # Static condensation for hp-multiresolution ordering
    src/bathymetry/static_condensation.cpp
    # Lowrider linear mesh components
    src/bathymetry/linear_bezier_surface.cpp
    src/bathymetry/linear_mesh_generator.cpp
    src/bathymetry/element_error_estimator.cpp
    src/bathymetry/gauss_quadrature_error_estimator.cpp
)

set(DG_SOURCES
    src/dg/basis.cpp
    src/dg/basis_hexahedron.cpp
    src/dg/bernstein_basis.cpp
    src/dg/nonconforming_projection.cpp
    src/dg/quadrature.cpp
    src/dg/quadrature_3d.cpp
    src/dg/mortar.cpp
    src/dg/operators.cpp
    src/dg/operators_3d.cpp
    src/dg/limiter.cpp
)

set(AMR_SOURCES
    src/amr/refinement.cpp
    src/amr/coarsening.cpp
    src/amr/error_estimator.cpp
    src/amr/load_balancer.cpp
)

set(PHYSICS_SOURCES
    src/physics/shallow_water.cpp
    src/physics/baroclinic.cpp
    src/physics/coriolis.cpp
    src/physics/friction.cpp
    src/physics/turbulence.cpp
    src/physics/boundary_conditions.cpp
    src/physics/vertical_velocity.cpp
    src/physics/primitive_equations.cpp
    src/physics/mode_splitting.cpp
)

set(FLUX_SOURCES
    src/flux/numerical_flux.cpp
)

set(SOLVER_SOURCES
    src/solver/time_stepper.cpp
    src/solver/runge_kutta.cpp
    src/solver/simulation_driver.cpp
    src/solver/flux.cpp
    src/solver/riemann.cpp
)

set(IO_SOURCES
    src/io/bathymetry_vtk_writer.cpp
    src/io/matrix_market_writer.cpp
    src/io/netcdf_io.cpp
    src/io/vtk_io.cpp
    src/io/vtk_writer.cpp
    src/io/water_vtk_writer.cpp
    src/io/zarr_writer.cpp
    src/io/restart.cpp
    src/io/quadtree_vtk_writer.cpp
    src/io/raster_vtk_writer.cpp
)

set(UTILS_SOURCES
    src/utils/interpolation.cpp
    src/utils/coordinates.cpp
)

set(PARALLEL_SOURCES
    src/parallel/domain_decomposition.cpp
    src/parallel/halo_exchange.cpp
)

# Main library
add_library(drifter_lib STATIC
    ${CORE_SOURCES}
    ${MESH_SOURCES}
    ${BATHYMETRY_SOURCES}
    ${DG_SOURCES}
    ${AMR_SOURCES}
    ${PHYSICS_SOURCES}
    ${FLUX_SOURCES}
    ${SOLVER_SOURCES}
    ${IO_SOURCES}
    ${UTILS_SOURCES}
    ${PARALLEL_SOURCES}
)

target_link_libraries(drifter_lib PUBLIC
    Eigen3::Eigen
    $<$<BOOL:${DRIFTER_USE_MPI}>:MPI::MPI_CXX>
    $<$<BOOL:${DRIFTER_USE_OPENMP}>:OpenMP::OpenMP_CXX>
    ${GDAL_LIBRARIES}
    $<$<BOOL:${VTK_FOUND}>:${VTK_LIBRARIES}>
    $<$<BOOL:${ZARRS_FFI_LIB}>:${ZARRS_FFI_LIB}>
    $<$<BOOL:${METIS_FOUND}>:${METIS_LIBRARY}>
    $<$<BOOL:${DRIFTER_USE_CHOLMOD}>:${CHOLMOD_LIBRARY}>
    $<$<BOOL:${DRIFTER_USE_UMFPACK}>:${UMFPACK_LIBRARY}>
    $<$<BOOL:${DRIFTER_USE_MKL}>:${MKL_LIBRARIES}>
)

# Shared library (for lowrider and other projects linking to drifter)
option(BUILD_SHARED_LIBS "Build shared library (libdrifter.so)" OFF)

if(BUILD_SHARED_LIBS)
    add_library(drifter_shared SHARED
        ${CORE_SOURCES}
        ${MESH_SOURCES}
        ${BATHYMETRY_SOURCES}
        ${DG_SOURCES}
        ${AMR_SOURCES}
        ${PHYSICS_SOURCES}
        ${FLUX_SOURCES}
        ${SOLVER_SOURCES}
        ${IO_SOURCES}
        ${UTILS_SOURCES}
        ${PARALLEL_SOURCES}
    )
    set_target_properties(drifter_shared PROPERTIES OUTPUT_NAME "drifter")
    target_link_libraries(drifter_shared PUBLIC
        Eigen3::Eigen
        $<$<BOOL:${DRIFTER_USE_MPI}>:MPI::MPI_CXX>
        $<$<BOOL:${DRIFTER_USE_OPENMP}>:OpenMP::OpenMP_CXX>
        ${GDAL_LIBRARIES}
        $<$<BOOL:${VTK_FOUND}>:${VTK_LIBRARIES}>
        $<$<BOOL:${ZARRS_FFI_LIB}>:${ZARRS_FFI_LIB}>
        $<$<BOOL:${METIS_FOUND}>:${METIS_LIBRARY}>
        $<$<BOOL:${DRIFTER_USE_CHOLMOD}>:${CHOLMOD_LIBRARY}>
        $<$<BOOL:${DRIFTER_USE_UMFPACK}>:${UMFPACK_LIBRARY}>
        $<$<BOOL:${DRIFTER_USE_MKL}>:${MKL_LIBRARIES}>
    )
endif()

# Applications
add_subdirectory(apps/highrider)
add_subdirectory(apps/lowrider)

# Tests
if(DRIFTER_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_subdirectory(tests)
    endif()
endif()

# Install rules
install(TARGETS drifter_lib highrider lowrider
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
if(BUILD_SHARED_LIBS)
    install(TARGETS drifter_shared
        LIBRARY DESTINATION lib
    )
endif()
install(DIRECTORY include/ DESTINATION include)
