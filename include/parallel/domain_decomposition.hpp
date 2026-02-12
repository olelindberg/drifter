#pragma once

// Domain Decomposition for Parallel DG Ocean Model
//
// Provides domain decomposition strategies for distributed memory parallelism.
// Key features:
// - Morton curve (space-filling) partitioning from SeaMesh
// - METIS graph partitioning for load balance
// - Support for AMR with dynamic repartitioning
// - Communication map construction
//
// Usage:
//   DomainDecomposition decomp(mesh, comm);
//   decomp.partition_morton();
//   decomp.build_communication_maps();

#include "core/types.hpp"
#include "mesh/morton.hpp"
#include "mesh/octree_adapter.hpp"
#include <map>
#include <memory>
#include <set>
#include <vector>

#ifdef DRIFTER_USE_MPI
#include <mpi.h>
#endif

namespace drifter {

/// @brief Partitioning strategy
enum class PartitionStrategy {
    Morton,   ///< Morton curve (space-filling, preserves locality)
    Hilbert,  ///< Hilbert curve (better locality than Morton)
    Metis,    ///< METIS graph partitioning
    ParMetis, ///< Parallel METIS (for large meshes)
    PTScotch, ///< PT-Scotch partitioner
    Uniform   ///< Uniform block distribution (simple but poor locality)
};

/// @brief Ghost layer configuration
struct GhostConfig {
    int num_layers = 1;           ///< Number of ghost element layers
    bool include_diagonal = true; ///< Include face-diagonal neighbors
    bool include_corners = false; ///< Include corner neighbors
};

/// @brief Element ownership info
struct ElementOwnership {
    Index global_id; ///< Global element ID
    Index local_id;  ///< Local element ID (on owning rank)
    int owner_rank;  ///< Rank that owns this element
    bool is_ghost;   ///< True if this is a ghost element
};

/// @brief Communication neighbor info
struct NeighborComm {
    int rank;                         ///< Neighbor rank
    std::vector<Index> send_elements; ///< Local IDs to send
    std::vector<Index> recv_elements; ///< Ghost IDs to receive into
    size_t send_size = 0;             ///< Total send buffer size (DOFs)
    size_t recv_size = 0;             ///< Total receive buffer size (DOFs)
};

/// @brief Domain decomposition manager
class DomainDecomposition {
public:
#ifdef DRIFTER_USE_MPI
    /// @brief Construct with MPI communicator
    DomainDecomposition(MPI_Comm comm = MPI_COMM_WORLD);
#else
    DomainDecomposition();
#endif

    ~DomainDecomposition() = default;

    /// @brief Set global mesh (before partitioning)
    void set_global_mesh(const OctreeAdapter &mesh);

    /// @brief Partition the mesh
    /// @param strategy Partitioning strategy
    /// @return Local element count
    Index partition(PartitionStrategy strategy = PartitionStrategy::Morton);

    /// @brief Partition using Morton curve (fast, good locality)
    Index partition_morton();

    /// @brief Partition using METIS (best load balance)
    Index partition_metis();

    /// @brief Build communication maps for halo exchange
    void build_communication_maps(const GhostConfig &config = GhostConfig());

    /// @brief Get local elements (owned by this rank)
    const std::vector<Index> &local_elements() const { return local_elements_; }

    /// @brief Get ghost elements (owned by other ranks)
    const std::vector<Index> &ghost_elements() const { return ghost_elements_; }

    /// @brief Get all elements (local + ghost)
    std::vector<Index> all_elements() const;

    /// @brief Get neighbor communication info
    const std::vector<NeighborComm> &neighbors() const { return neighbors_; }

    /// @brief Get element owner rank
    int owner(Index global_elem_id) const;

    /// @brief Get local ID from global ID
    Index global_to_local(Index global_id) const;

    /// @brief Get global ID from local ID
    Index local_to_global(Index local_id) const;

    /// @brief Check if element is owned locally
    bool is_local(Index global_elem_id) const;

    /// @brief Get number of local elements (not including ghosts)
    Index num_local_elements() const { return local_elements_.size(); }

    /// @brief Get number of ghost elements
    Index num_ghost_elements() const { return ghost_elements_.size(); }

    /// @brief Get partition array (element → rank)
    const std::vector<int> &partition() const { return partition_; }

    /// @brief Rebalance after AMR
    void rebalance(const OctreeAdapter &new_mesh);

    /// @brief Get MPI rank
    int rank() const { return rank_; }

    /// @brief Get MPI size
    int size() const { return size_; }

#ifdef DRIFTER_USE_MPI
    /// @brief Get MPI communicator
    MPI_Comm comm() const { return comm_; }
#endif

private:
#ifdef DRIFTER_USE_MPI
    MPI_Comm comm_;
#endif
    int rank_ = 0;
    int size_ = 1;

    const OctreeAdapter *mesh_ = nullptr;

    std::vector<int> partition_;             // Global element → rank
    std::vector<Index> local_elements_;      // Global IDs owned by this rank
    std::vector<Index> ghost_elements_;      // Global IDs needed as ghosts
    std::map<Index, Index> global_to_local_; // Global ID → local ID
    std::map<Index, Index> local_to_global_; // Local ID → global ID

    std::vector<NeighborComm> neighbors_; // Communication with each neighbor

    void compute_morton_partition();
    void compute_metis_partition();
    void identify_ghost_elements(const GhostConfig &config);
    void build_neighbor_lists();
};

/// @brief Load balancer for dynamic repartitioning
class LoadBalancer {
public:
#ifdef DRIFTER_USE_MPI
    LoadBalancer(MPI_Comm comm = MPI_COMM_WORLD);
#else
    LoadBalancer();
#endif

    /// @brief Compute imbalance ratio (max/avg elements per rank)
    Real compute_imbalance(const DomainDecomposition &decomp) const;

    /// @brief Check if rebalancing is needed
    bool needs_rebalance(
        const DomainDecomposition &decomp, Real threshold = 1.2) const;

    /// @brief Compute new partition with better balance
    std::vector<int> compute_balanced_partition(
        const OctreeAdapter &mesh, const DomainDecomposition &current_decomp);

    /// @brief Compute migration plan
    struct MigrationPlan {
        std::map<Index, int> element_destinations; // Element → new rank
        std::map<int, std::vector<Index>> send_to; // Rank → elements to send
        std::map<int, std::vector<Index>>
            recv_from; // Rank → elements to receive
    };

    MigrationPlan compute_migration(
        const std::vector<int> &old_partition,
        const std::vector<int> &new_partition);

private:
#ifdef DRIFTER_USE_MPI
    MPI_Comm comm_;
#endif
    int rank_ = 0;
    int size_ = 1;
};

/// @brief Diffusive load balancing (local element exchange)
class DiffusiveBalancer {
public:
#ifdef DRIFTER_USE_MPI
    DiffusiveBalancer(MPI_Comm comm = MPI_COMM_WORLD);
#else
    DiffusiveBalancer();
#endif

    /// @brief Perform one diffusive balancing step
    /// @return Number of elements migrated
    Index balance_step(DomainDecomposition &decomp, OctreeAdapter &mesh);

    /// @brief Iterate until balanced
    void balance(
        DomainDecomposition &decomp, OctreeAdapter &mesh, Real tolerance = 0.1,
        int max_iterations = 10);

private:
#ifdef DRIFTER_USE_MPI
    MPI_Comm comm_;
#endif
    int rank_ = 0;
    int size_ = 1;

    std::vector<Index> select_boundary_elements(
        const DomainDecomposition &decomp, int neighbor_rank, Index count);
};

} // namespace drifter
