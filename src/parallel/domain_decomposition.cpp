#include "parallel/domain_decomposition.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

namespace drifter {

// =============================================================================
// DomainDecomposition implementation
// =============================================================================

#ifdef DRIFTER_USE_MPI
DomainDecomposition::DomainDecomposition(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}
#else
DomainDecomposition::DomainDecomposition()
    : rank_(0)
    , size_(1)
{
}
#endif

void DomainDecomposition::set_global_mesh(const OctreeAdapter& mesh) {
    mesh_ = &mesh;
}

Index DomainDecomposition::partition(PartitionStrategy strategy) {
    if (!mesh_) {
        throw std::runtime_error("Mesh not set before partitioning");
    }

    switch (strategy) {
        case PartitionStrategy::Morton:
            return partition_morton();
        case PartitionStrategy::Metis:
            return partition_metis();
        case PartitionStrategy::Uniform:
            // Simple uniform distribution
            {
                Index num_elements = mesh_->num_elements();
                partition_.resize(num_elements);
                Index per_rank = num_elements / size_;
                Index remainder = num_elements % size_;

                Index idx = 0;
                for (int r = 0; r < size_; ++r) {
                    Index count = per_rank + (r < remainder ? 1 : 0);
                    for (Index i = 0; i < count; ++i) {
                        partition_[idx++] = r;
                    }
                }

                // Build local element list
                local_elements_.clear();
                for (Index i = 0; i < num_elements; ++i) {
                    if (partition_[i] == rank_) {
                        local_elements_.push_back(i);
                    }
                }
                return local_elements_.size();
            }
        default:
            return partition_morton();
    }
}

Index DomainDecomposition::partition_morton() {
    compute_morton_partition();

    // Build local element list
    local_elements_.clear();
    Index num_elements = mesh_->num_elements();
    for (Index i = 0; i < num_elements; ++i) {
        if (partition_[i] == rank_) {
            local_elements_.push_back(i);
        }
    }

    // Build index maps
    global_to_local_.clear();
    local_to_global_.clear();
    for (Index local = 0; local < static_cast<Index>(local_elements_.size()); ++local) {
        Index global = local_elements_[local];
        global_to_local_[global] = local;
        local_to_global_[local] = global;
    }

    return local_elements_.size();
}

void DomainDecomposition::compute_morton_partition() {
    const auto& elements = mesh_->elements();
    Index num_elements = static_cast<Index>(elements.size());
    partition_.resize(num_elements);

    // Create (morton_code, original_index) pairs and sort
    std::vector<std::pair<uint64_t, Index>> morton_indexed;
    morton_indexed.reserve(num_elements);

    for (Index i = 0; i < num_elements; ++i) {
        morton_indexed.emplace_back(elements[i]->morton, i);
    }

    // Sort by Morton code
    std::sort(morton_indexed.begin(), morton_indexed.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Distribute contiguous chunks to ranks
    Index per_rank = num_elements / size_;
    Index remainder = num_elements % size_;

    Index current_idx = 0;
    for (int r = 0; r < size_; ++r) {
        Index count = per_rank + (r < remainder ? 1 : 0);
        for (Index i = 0; i < count; ++i) {
            Index orig_idx = morton_indexed[current_idx].second;
            partition_[orig_idx] = r;
            ++current_idx;
        }
    }
}

Index DomainDecomposition::partition_metis() {
#ifdef DRIFTER_HAS_METIS
    // METIS partitioning would go here
    // For now, fall back to Morton
    return partition_morton();
#else
    // Fall back to Morton partitioning
    return partition_morton();
#endif
}

void DomainDecomposition::compute_metis_partition() {
    // METIS graph partitioning requires building the dual graph
    // (elements connected if they share a face)
    // This is a placeholder for METIS integration
    compute_morton_partition();
}

void DomainDecomposition::build_communication_maps(const GhostConfig& config) {
    identify_ghost_elements(config);
    build_neighbor_lists();
}

void DomainDecomposition::identify_ghost_elements(const GhostConfig& /*config*/) {
    ghost_elements_.clear();

    if (!mesh_) return;

    std::set<Index> ghost_set;

    // For each local element, find neighbors owned by other ranks
    for (Index local_elem : local_elements_) {
        // Use OctreeAdapter's get_face_neighbors to find neighbor elements
        auto face_neighbors = mesh_->get_face_neighbors(local_elem);

        for (int face = 0; face < 6; ++face) {
            const auto& neighbor_info = face_neighbors[face];
            for (Index neighbor_global : neighbor_info.neighbor_elements) {
                if (neighbor_global < static_cast<Index>(partition_.size()) &&
                    partition_[neighbor_global] != rank_) {
                    ghost_set.insert(neighbor_global);
                }
            }
        }
    }

    ghost_elements_.assign(ghost_set.begin(), ghost_set.end());

    // Add ghost elements to index maps
    Index ghost_local_start = static_cast<Index>(local_elements_.size());
    for (Index i = 0; i < static_cast<Index>(ghost_elements_.size()); ++i) {
        Index global = ghost_elements_[i];
        Index local = ghost_local_start + i;
        global_to_local_[global] = local;
        local_to_global_[local] = global;
    }
}

void DomainDecomposition::build_neighbor_lists() {
    neighbors_.clear();

    if (!mesh_) return;

    // Group ghost elements by owner rank
    std::map<int, std::vector<Index>> ghosts_by_rank;
    for (Index ghost_global : ghost_elements_) {
        int owner = partition_[ghost_global];
        ghosts_by_rank[owner].push_back(ghost_global);
    }

    // Group local elements needed by each neighbor
    std::map<int, std::set<Index>> sends_by_rank;
    for (Index local_elem : local_elements_) {
        auto face_neighbors = mesh_->get_face_neighbors(local_elem);

        for (int face = 0; face < 6; ++face) {
            const auto& neighbor_info = face_neighbors[face];
            for (Index neighbor_global : neighbor_info.neighbor_elements) {
                if (neighbor_global < static_cast<Index>(partition_.size())) {
                    int neighbor_rank = partition_[neighbor_global];
                    if (neighbor_rank != rank_) {
                        sends_by_rank[neighbor_rank].insert(local_elem);
                    }
                }
            }
        }
    }

    // Build neighbor communication structures
    for (const auto& [neighbor_rank, recv_globals] : ghosts_by_rank) {
        NeighborComm nc;
        nc.rank = neighbor_rank;

        // Receive (ghost) elements
        for (Index global : recv_globals) {
            nc.recv_elements.push_back(global_to_local_[global]);
        }

        // Send elements (local IDs)
        for (Index global : sends_by_rank[neighbor_rank]) {
            nc.send_elements.push_back(global_to_local_[global]);
        }

        neighbors_.push_back(nc);
    }

#ifdef DRIFTER_USE_MPI
    // Exchange send/receive counts to ensure consistency
    for (auto& nc : neighbors_) {
        (void)nc;  // Suppress unused warning
        // The other rank's recv_elements should match our send_elements
        // and vice versa. This is handled by the symmetric identification
        // above, but could be verified with MPI communication.
    }
#endif
}

std::vector<Index> DomainDecomposition::all_elements() const {
    std::vector<Index> all;
    all.reserve(local_elements_.size() + ghost_elements_.size());
    all.insert(all.end(), local_elements_.begin(), local_elements_.end());
    all.insert(all.end(), ghost_elements_.begin(), ghost_elements_.end());
    return all;
}

int DomainDecomposition::owner(Index global_elem_id) const {
    if (global_elem_id >= static_cast<Index>(partition_.size())) {
        return -1;
    }
    return partition_[global_elem_id];
}

Index DomainDecomposition::global_to_local(Index global_id) const {
    auto it = global_to_local_.find(global_id);
    if (it == global_to_local_.end()) {
        throw std::runtime_error("Global element not found locally");
    }
    return it->second;
}

Index DomainDecomposition::local_to_global(Index local_id) const {
    auto it = local_to_global_.find(local_id);
    if (it == local_to_global_.end()) {
        throw std::runtime_error("Invalid local element ID");
    }
    return it->second;
}

bool DomainDecomposition::is_local(Index global_elem_id) const {
    if (global_elem_id >= static_cast<Index>(partition_.size())) {
        return false;
    }
    return partition_[global_elem_id] == rank_;
}

void DomainDecomposition::rebalance(const OctreeAdapter& new_mesh) {
    mesh_ = &new_mesh;
    partition(PartitionStrategy::Morton);
    build_communication_maps();
}

// =============================================================================
// LoadBalancer implementation
// =============================================================================

#ifdef DRIFTER_USE_MPI
LoadBalancer::LoadBalancer(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}
#else
LoadBalancer::LoadBalancer()
    : rank_(0)
    , size_(1)
{
}
#endif

Real LoadBalancer::compute_imbalance(const DomainDecomposition& decomp) const {
    Index local_count = decomp.num_local_elements();

#ifdef DRIFTER_USE_MPI
    Index max_count, total_count;
    MPI_Allreduce(&local_count, &max_count, 1, MPI_LONG_LONG, MPI_MAX, comm_);
    MPI_Allreduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, comm_);

    Real avg = static_cast<Real>(total_count) / size_;
    if (avg < 1e-10) return 1.0;

    return static_cast<Real>(max_count) / avg;
#else
    (void)local_count;  // Suppress unused warning
    return 1.0;  // Perfect balance with single process
#endif
}

bool LoadBalancer::needs_rebalance(const DomainDecomposition& decomp,
                                    Real threshold) const {
    return compute_imbalance(decomp) > threshold;
}

std::vector<int> LoadBalancer::compute_balanced_partition(
    const OctreeAdapter& mesh,
    const DomainDecomposition& /*current_decomp*/)
{
    // Use Morton partitioning for balanced result
    const auto& elements = mesh.elements();
    Index num_elements = static_cast<Index>(elements.size());
    std::vector<int> new_partition(num_elements);

    // Sort by Morton and distribute evenly
    std::vector<std::pair<uint64_t, Index>> morton_indexed;
    morton_indexed.reserve(num_elements);

    for (Index i = 0; i < num_elements; ++i) {
        morton_indexed.emplace_back(elements[i]->morton, i);
    }

    std::sort(morton_indexed.begin(), morton_indexed.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    Index per_rank = num_elements / size_;
    Index remainder = num_elements % size_;

    Index current_idx = 0;
    for (int r = 0; r < size_; ++r) {
        Index count = per_rank + (r < remainder ? 1 : 0);
        for (Index i = 0; i < count; ++i) {
            Index orig_idx = morton_indexed[current_idx].second;
            new_partition[orig_idx] = r;
            ++current_idx;
        }
    }

    return new_partition;
}

LoadBalancer::MigrationPlan LoadBalancer::compute_migration(
    const std::vector<int>& old_partition,
    const std::vector<int>& new_partition)
{
    MigrationPlan plan;

    if (old_partition.size() != new_partition.size()) {
        throw std::runtime_error("Partition size mismatch");
    }

    for (Index i = 0; i < static_cast<Index>(old_partition.size()); ++i) {
        int old_rank = old_partition[i];
        int new_rank = new_partition[i];

        if (old_rank != new_rank) {
            plan.element_destinations[i] = new_rank;

            if (old_rank == rank_) {
                plan.send_to[new_rank].push_back(i);
            }
            if (new_rank == rank_) {
                plan.recv_from[old_rank].push_back(i);
            }
        }
    }

    return plan;
}

// =============================================================================
// DiffusiveBalancer implementation
// =============================================================================

#ifdef DRIFTER_USE_MPI
DiffusiveBalancer::DiffusiveBalancer(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}
#else
DiffusiveBalancer::DiffusiveBalancer()
    : rank_(0)
    , size_(1)
{
}
#endif

Index DiffusiveBalancer::balance_step(DomainDecomposition& decomp,
                                       OctreeAdapter& /*mesh*/) {
    Index migrated = 0;

#ifdef DRIFTER_USE_MPI
    // Gather element counts from all ranks
    Index local_count = decomp.num_local_elements();
    std::vector<Index> all_counts(size_);
    MPI_Allgather(&local_count, 1, MPI_LONG_LONG,
                  all_counts.data(), 1, MPI_LONG_LONG, comm_);

    // Compute average
    Index total = std::accumulate(all_counts.begin(), all_counts.end(), Index(0));
    Index avg = total / size_;

    // If we have more than average, offer elements to neighbors with less
    if (local_count > avg + 1) {
        // Find neighbors that need elements
        for (const auto& neighbor : decomp.neighbors()) {
            int nr = neighbor.rank;
            if (all_counts[nr] < avg - 1) {
                // This neighbor needs elements
                Index to_send = std::min(local_count - avg,
                                          avg - all_counts[nr]) / 2;
                if (to_send > 0) {
                    // Select boundary elements to migrate
                    auto elements = select_boundary_elements(decomp, nr, to_send);
                    migrated += static_cast<Index>(elements.size());

                    // Actual migration would involve:
                    // 1. Update partition array
                    // 2. Send element data
                    // 3. Rebuild decomposition
                }
            }
        }
    }
#else
    (void)decomp;  // Suppress unused warning
#endif

    return migrated;
}

void DiffusiveBalancer::balance(DomainDecomposition& decomp,
                                 OctreeAdapter& mesh,
                                 Real tolerance,
                                 int max_iterations) {
    LoadBalancer lb;
#ifdef DRIFTER_USE_MPI
    lb = LoadBalancer(comm_);
#endif

    for (int iter = 0; iter < max_iterations; ++iter) {
        Real imbalance = lb.compute_imbalance(decomp);
        if (imbalance <= 1.0 + tolerance) {
            break;
        }

        Index migrated = balance_step(decomp, mesh);
        if (migrated == 0) {
            break;  // No progress possible
        }

        // Rebuild communication maps after migration
        decomp.build_communication_maps();
    }
}

std::vector<Index> DiffusiveBalancer::select_boundary_elements(
    const DomainDecomposition& decomp,
    int neighbor_rank,
    Index count)
{
    std::vector<Index> selected;

    // Find elements that border the target neighbor
    for (const auto& nc : decomp.neighbors()) {
        if (nc.rank == neighbor_rank) {
            // These elements are adjacent to the neighbor
            for (Index local : nc.send_elements) {
                if (selected.size() >= static_cast<size_t>(count)) {
                    break;
                }
                selected.push_back(decomp.local_to_global(local));
            }
            break;
        }
    }

    return selected;
}

}  // namespace drifter
