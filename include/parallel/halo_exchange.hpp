#pragma once

// Halo Exchange for Parallel DG Ocean Model
//
// Provides ghost data synchronization for distributed memory parallelism.
// Key features:
// - Non-blocking MPI communication
// - Overlapping computation with communication
// - Support for multiple field types (velocity, tracers)
// - Packing/unpacking of DG element data
//
// Usage:
//   HaloExchange halo(decomp);
//   halo.start_exchange(solution);
//   // ... compute interior elements ...
//   halo.finish_exchange(solution);

#include "core/types.hpp"
#include "parallel/domain_decomposition.hpp"
#include <memory>
#include <vector>

#ifdef DRIFTER_USE_MPI
#include <mpi.h>
#endif

namespace drifter {

// Forward declarations
class HexahedronBasis;

/// @brief Field type for halo exchange
enum class FieldType {
    Scalar,         ///< Single scalar per DOF
    Vector2D,       ///< 2D vector (u, v)
    Vector3D,       ///< 3D vector (u, v, w)
    MultiComponent  ///< Arbitrary number of components
};

/// @brief Buffer for send/receive data
struct HaloBuffer {
    std::vector<Real> data;
    int partner_rank = -1;
    size_t num_elements = 0;
    size_t dofs_per_element = 0;

#ifdef DRIFTER_USE_MPI
    MPI_Request request = MPI_REQUEST_NULL;
#endif
};

/// @brief Halo exchange manager
class HaloExchange {
public:
    /// @brief Construct with domain decomposition
    explicit HaloExchange(const DomainDecomposition& decomp);

    ~HaloExchange();

    // Non-copyable
    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    /// @brief Set DOFs per element (must match solution vector layout)
    void set_dofs_per_element(size_t dofs);

    /// @brief Set element basis (for automatic DOF count)
    void set_basis(const HexahedronBasis& basis);

    /// @brief Start non-blocking halo exchange
    /// @param solution Element data vector (local + ghost elements)
    void start_exchange(std::vector<VecX>& solution);

    /// @brief Start exchange for raw buffer
    void start_exchange(Real* data, size_t dofs_per_elem, size_t num_elements);

    /// @brief Finish halo exchange (wait for completion)
    void finish_exchange();

    /// @brief Blocking halo exchange
    void exchange(std::vector<VecX>& solution);

    /// @brief Exchange multiple fields simultaneously
    void exchange_multi(std::vector<std::vector<VecX>*>& fields);

    /// @brief Check if exchange is in progress
    bool is_active() const { return exchange_active_; }

    /// @brief Get communication statistics
    struct Statistics {
        size_t bytes_sent = 0;
        size_t bytes_received = 0;
        double time_packing = 0.0;
        double time_waiting = 0.0;
        double time_unpacking = 0.0;
    };
    Statistics get_statistics() const { return stats_; }

    /// @brief Reset statistics
    void reset_statistics() { stats_ = Statistics(); }

private:
    const DomainDecomposition& decomp_;

    size_t dofs_per_element_ = 0;
    const HexahedronBasis* basis_ = nullptr;

    std::vector<HaloBuffer> send_buffers_;
    std::vector<HaloBuffer> recv_buffers_;

    bool exchange_active_ = false;
    Statistics stats_;

    // Current exchange data pointers
    std::vector<VecX>* current_solution_ = nullptr;

    void allocate_buffers();
    void pack_send_buffers(const std::vector<VecX>& solution);
    void unpack_recv_buffers(std::vector<VecX>& solution);

#ifdef DRIFTER_USE_MPI
    void post_receives();
    void post_sends();
    void wait_all();

    std::vector<MPI_Request> send_requests_;
    std::vector<MPI_Request> recv_requests_;
#endif
};

/// @brief Multi-field halo exchange (optimized for multiple variables)
class MultiFieldHaloExchange {
public:
    explicit MultiFieldHaloExchange(const DomainDecomposition& decomp);

    /// @brief Add a field to exchange
    void add_field(const std::string& name, size_t dofs_per_elem);

    /// @brief Start exchange for all registered fields
    void start_exchange(std::map<std::string, std::vector<VecX>*>& fields);

    /// @brief Finish exchange
    void finish_exchange();

    /// @brief Blocking exchange
    void exchange(std::map<std::string, std::vector<VecX>*>& fields);

private:
    const DomainDecomposition& decomp_;

    struct FieldInfo {
        std::string name;
        size_t dofs_per_elem;
        size_t offset_in_buffer;
    };

    std::vector<FieldInfo> fields_;
    size_t total_dofs_per_element_ = 0;

    std::vector<HaloBuffer> send_buffers_;
    std::vector<HaloBuffer> recv_buffers_;

    std::map<std::string, std::vector<VecX>*>* current_fields_ = nullptr;
    bool exchange_active_ = false;

    void pack_all_fields();
    void unpack_all_fields();

#ifdef DRIFTER_USE_MPI
    std::vector<MPI_Request> requests_;
#endif
};

/// @brief Asynchronous halo exchange with multiple overlap regions
class AsyncHaloExchange {
public:
    AsyncHaloExchange(const DomainDecomposition& decomp);

    /// @brief Classify elements for overlapped computation
    struct ElementClassification {
        std::vector<Index> interior;     ///< Elements with no ghost neighbors
        std::vector<Index> boundary;     ///< Elements needing ghost data
    };

    /// @brief Get element classification
    ElementClassification classify_elements() const;

    /// @brief Start exchange (non-blocking)
    void start(std::vector<VecX>& solution);

    /// @brief Check if exchange is complete
    bool test();

    /// @brief Wait for exchange to complete
    void wait();

    /// @brief Get interior elements (can compute while exchanging)
    const std::vector<Index>& interior_elements() const {
        return classification_.interior;
    }

    /// @brief Get boundary elements (need ghost data)
    const std::vector<Index>& boundary_elements() const {
        return classification_.boundary;
    }

private:
    const DomainDecomposition& decomp_;
    HaloExchange exchanger_;
    ElementClassification classification_;

    void build_classification();
};

/// @brief Face-based halo exchange (for flux computations)
class FaceHaloExchange {
public:
    FaceHaloExchange(const DomainDecomposition& decomp);

    /// @brief Exchange face data only (smaller messages)
    void exchange_faces(std::vector<std::vector<VecX>>& face_data);

    /// @brief Get remote face data for inter-rank faces
    const std::vector<VecX>& get_remote_face_data(Index local_elem,
                                                    int face_id) const;

private:
    const DomainDecomposition& decomp_;

    // Inter-rank face information
    struct InterRankFace {
        Index local_elem;
        int local_face_id;
        int remote_rank;
        Index remote_elem;
        int remote_face_id;
    };

    std::vector<InterRankFace> inter_rank_faces_;
    std::map<std::pair<Index, int>, VecX> remote_face_data_;

    void identify_inter_rank_faces();
};

}  // namespace drifter
