#include <iostream>
#include <string>

#ifdef DRIFTER_USE_MPI
#include <mpi.h>
#endif

// #include "core/config.hpp"
// #include "core/logger.hpp"
// #include "mesh/mesh.hpp"
// #include "solver/time_stepper.hpp"

int main(int argc, char* argv[]) {
#ifdef DRIFTER_USE_MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    int rank = 0, size = 1;
#endif

    if (rank == 0) {
        std::cout << "====================================\n";
        std::cout << "  DRIFTER - Coastal Ocean Model\n";
        std::cout << "  3D DG-FEM with Adaptive MR\n";
        std::cout << "====================================\n";
        std::cout << "Running on " << size << " process(es)\n\n";
    }

    // TODO: Parse command line arguments
    // TODO: Load configuration
    // TODO: Initialize mesh
    // TODO: Initialize DG operators
    // TODO: Set initial conditions
    // TODO: Time integration loop
    // TODO: Output results

    if (rank == 0) {
        std::cout << "Simulation complete.\n";
    }

#ifdef DRIFTER_USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
