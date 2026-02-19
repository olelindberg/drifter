#include "parallel/halo_exchange.hpp"
#include "dg/basis_hexahedron.hpp"
#include <chrono>
#include <stdexcept>

namespace drifter {

// =============================================================================
// HaloExchange implementation
// =============================================================================

HaloExchange::HaloExchange(const DomainDecomposition &decomp) : decomp_(decomp) {}

HaloExchange::~HaloExchange() {
    if (exchange_active_) {
        finish_exchange();
    }
}

void HaloExchange::set_dofs_per_element(size_t dofs) {
    dofs_per_element_ = dofs;
    allocate_buffers();
}

void HaloExchange::set_basis(const HexahedronBasis &basis) {
    basis_ = &basis;
    dofs_per_element_ = basis.num_dofs_velocity();
    allocate_buffers();
}

void HaloExchange::allocate_buffers() {
    const auto &neighbors = decomp_.neighbors();

    send_buffers_.resize(neighbors.size());
    recv_buffers_.resize(neighbors.size());

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto &nc = neighbors[i];

        send_buffers_[i].partner_rank = nc.rank;
        send_buffers_[i].num_elements = nc.send_elements.size();
        send_buffers_[i].dofs_per_element = dofs_per_element_;
        send_buffers_[i].data.resize(nc.send_elements.size() * dofs_per_element_);

        recv_buffers_[i].partner_rank = nc.rank;
        recv_buffers_[i].num_elements = nc.recv_elements.size();
        recv_buffers_[i].dofs_per_element = dofs_per_element_;
        recv_buffers_[i].data.resize(nc.recv_elements.size() * dofs_per_element_);
    }

#ifdef DRIFTER_USE_MPI
    send_requests_.resize(neighbors.size(), MPI_REQUEST_NULL);
    recv_requests_.resize(neighbors.size(), MPI_REQUEST_NULL);
#endif
}

void HaloExchange::start_exchange(std::vector<VecX> &solution) {
    if (exchange_active_) {
        throw std::runtime_error("Exchange already in progress");
    }

    current_solution_ = &solution;

    auto start = std::chrono::high_resolution_clock::now();
    pack_send_buffers(solution);
    auto end = std::chrono::high_resolution_clock::now();
    stats_.time_packing += std::chrono::duration<double>(end - start).count();

#ifdef DRIFTER_USE_MPI
    post_receives();
    post_sends();
#endif

    exchange_active_ = true;
}

void HaloExchange::start_exchange(Real* data, size_t dofs_per_elem, size_t num_elements) {
    // Not implemented - use vector version
    throw std::runtime_error("Raw pointer exchange not implemented");
}

void HaloExchange::finish_exchange() {
    if (!exchange_active_)
        return;

    auto start = std::chrono::high_resolution_clock::now();

#ifdef DRIFTER_USE_MPI
    wait_all();
#endif

    auto mid = std::chrono::high_resolution_clock::now();
    stats_.time_waiting += std::chrono::duration<double>(mid - start).count();

    if (current_solution_) {
        unpack_recv_buffers(*current_solution_);
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats_.time_unpacking += std::chrono::duration<double>(end - mid).count();

    exchange_active_ = false;
    current_solution_ = nullptr;
}

void HaloExchange::exchange(std::vector<VecX> &solution) {
    start_exchange(solution);
    finish_exchange();
}

void HaloExchange::exchange_multi(std::vector<std::vector<VecX>*> &fields) {
    // Exchange each field sequentially
    // A more optimized version would pack all fields into a single message
    for (auto* field : fields) {
        exchange(*field);
    }
}

void HaloExchange::pack_send_buffers(const std::vector<VecX> &solution) {
    const auto &neighbors = decomp_.neighbors();

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto &nc = neighbors[i];
        Real* buffer = send_buffers_[i].data.data();

        size_t offset = 0;
        for (Index local_elem : nc.send_elements) {
            const VecX &elem_data = solution[local_elem];
            for (Index j = 0; j < elem_data.size(); ++j) {
                buffer[offset++] = elem_data(j);
            }
        }

        stats_.bytes_sent += offset * sizeof(Real);
    }
}

void HaloExchange::unpack_recv_buffers(std::vector<VecX> &solution) {
    const auto &neighbors = decomp_.neighbors();

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto &nc = neighbors[i];
        const Real* buffer = recv_buffers_[i].data.data();

        size_t offset = 0;
        for (Index local_ghost : nc.recv_elements) {
            VecX &elem_data = solution[local_ghost];
            if (elem_data.size() == 0) {
                elem_data.resize(dofs_per_element_);
            }
            for (Index j = 0; j < elem_data.size(); ++j) {
                elem_data(j) = buffer[offset++];
            }
        }

        stats_.bytes_received += offset * sizeof(Real);
    }
}

#ifdef DRIFTER_USE_MPI
void HaloExchange::post_receives() {
    for (size_t i = 0; i < recv_buffers_.size(); ++i) {
        auto &buf = recv_buffers_[i];
        if (buf.data.empty())
            continue;

        MPI_Irecv(buf.data.data(), buf.data.size(), MPI_DOUBLE, buf.partner_rank,
                  0, // tag
                  decomp_.comm(), &recv_requests_[i]);
    }
}

void HaloExchange::post_sends() {
    for (size_t i = 0; i < send_buffers_.size(); ++i) {
        auto &buf = send_buffers_[i];
        if (buf.data.empty())
            continue;

        MPI_Isend(buf.data.data(), buf.data.size(), MPI_DOUBLE, buf.partner_rank,
                  0, // tag
                  decomp_.comm(), &send_requests_[i]);
    }
}

void HaloExchange::wait_all() {
    if (!recv_requests_.empty()) {
        MPI_Waitall(recv_requests_.size(), recv_requests_.data(), MPI_STATUSES_IGNORE);
    }
    if (!send_requests_.empty()) {
        MPI_Waitall(send_requests_.size(), send_requests_.data(), MPI_STATUSES_IGNORE);
    }
}
#endif

// =============================================================================
// MultiFieldHaloExchange implementation
// =============================================================================

MultiFieldHaloExchange::MultiFieldHaloExchange(const DomainDecomposition &decomp)
    : decomp_(decomp) {}

void MultiFieldHaloExchange::add_field(const std::string &name, size_t dofs_per_elem) {
    FieldInfo info;
    info.name = name;
    info.dofs_per_elem = dofs_per_elem;
    info.offset_in_buffer = total_dofs_per_element_;
    fields_.push_back(info);
    total_dofs_per_element_ += dofs_per_elem;

    // Reallocate buffers
    const auto &neighbors = decomp_.neighbors();
    send_buffers_.resize(neighbors.size());
    recv_buffers_.resize(neighbors.size());

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto &nc = neighbors[i];
        send_buffers_[i].data.resize(nc.send_elements.size() * total_dofs_per_element_);
        recv_buffers_[i].data.resize(nc.recv_elements.size() * total_dofs_per_element_);
        send_buffers_[i].partner_rank = nc.rank;
        recv_buffers_[i].partner_rank = nc.rank;
    }
}

void MultiFieldHaloExchange::start_exchange(std::map<std::string, std::vector<VecX>*> &fields) {
    if (exchange_active_) {
        throw std::runtime_error("Exchange already in progress");
    }

    current_fields_ = &fields;
    pack_all_fields();

#ifdef DRIFTER_USE_MPI
    requests_.clear();
    requests_.reserve(send_buffers_.size() + recv_buffers_.size());

    // Post receives
    for (size_t i = 0; i < recv_buffers_.size(); ++i) {
        auto &buf = recv_buffers_[i];
        if (buf.data.empty())
            continue;

        MPI_Request req;
        MPI_Irecv(buf.data.data(), buf.data.size(), MPI_DOUBLE, buf.partner_rank, 0, decomp_.comm(),
                  &req);
        requests_.push_back(req);
    }

    // Post sends
    for (size_t i = 0; i < send_buffers_.size(); ++i) {
        auto &buf = send_buffers_[i];
        if (buf.data.empty())
            continue;

        MPI_Request req;
        MPI_Isend(buf.data.data(), buf.data.size(), MPI_DOUBLE, buf.partner_rank, 0, decomp_.comm(),
                  &req);
        requests_.push_back(req);
    }
#endif

    exchange_active_ = true;
}

void MultiFieldHaloExchange::finish_exchange() {
    if (!exchange_active_)
        return;

#ifdef DRIFTER_USE_MPI
    MPI_Waitall(requests_.size(), requests_.data(), MPI_STATUSES_IGNORE);
#endif

    if (current_fields_) {
        unpack_all_fields();
    }

    exchange_active_ = false;
    current_fields_ = nullptr;
}

void MultiFieldHaloExchange::exchange(std::map<std::string, std::vector<VecX>*> &fields) {
    start_exchange(fields);
    finish_exchange();
}

void MultiFieldHaloExchange::pack_all_fields() {
    const auto &neighbors = decomp_.neighbors();

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto &nc = neighbors[i];
        Real* buffer = send_buffers_[i].data.data();

        size_t elem_offset = 0;
        for (Index local_elem : nc.send_elements) {
            size_t field_offset = 0;
            for (const auto &field_info : fields_) {
                auto it = current_fields_->find(field_info.name);
                if (it == current_fields_->end())
                    continue;

                const VecX &elem_data = (*it->second)[local_elem];
                for (Index j = 0; j < static_cast<Index>(field_info.dofs_per_elem); ++j) {
                    buffer[elem_offset + field_offset + j] = elem_data(j);
                }
                field_offset += field_info.dofs_per_elem;
            }
            elem_offset += total_dofs_per_element_;
        }
    }
}

void MultiFieldHaloExchange::unpack_all_fields() {
    const auto &neighbors = decomp_.neighbors();

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto &nc = neighbors[i];
        const Real* buffer = recv_buffers_[i].data.data();

        size_t elem_offset = 0;
        for (Index local_ghost : nc.recv_elements) {
            size_t field_offset = 0;
            for (const auto &field_info : fields_) {
                auto it = current_fields_->find(field_info.name);
                if (it == current_fields_->end())
                    continue;

                VecX &elem_data = (*it->second)[local_ghost];
                if (elem_data.size() == 0) {
                    elem_data.resize(field_info.dofs_per_elem);
                }
                for (Index j = 0; j < static_cast<Index>(field_info.dofs_per_elem); ++j) {
                    elem_data(j) = buffer[elem_offset + field_offset + j];
                }
                field_offset += field_info.dofs_per_elem;
            }
            elem_offset += total_dofs_per_element_;
        }
    }
}

// =============================================================================
// AsyncHaloExchange implementation
// =============================================================================

AsyncHaloExchange::AsyncHaloExchange(const DomainDecomposition &decomp)
    : decomp_(decomp), exchanger_(decomp) {
    build_classification();
}

void AsyncHaloExchange::build_classification() {
    classification_.interior.clear();
    classification_.boundary.clear();

    // Collect all ghost element global IDs
    std::set<Index> ghost_globals(decomp_.ghost_elements().begin(), decomp_.ghost_elements().end());

    // For each local element, check if any neighbor is a ghost
    for (Index local_idx = 0; local_idx < decomp_.num_local_elements(); ++local_idx) {
        Index global_idx = decomp_.local_to_global(local_idx);
        bool has_ghost_neighbor = false;

        // Check face neighbors
        for (const auto &nc : decomp_.neighbors()) {
            for (Index send_local : nc.send_elements) {
                if (send_local == local_idx) {
                    has_ghost_neighbor = true;
                    break;
                }
            }
            if (has_ghost_neighbor)
                break;
        }

        if (has_ghost_neighbor) {
            classification_.boundary.push_back(local_idx);
        } else {
            classification_.interior.push_back(local_idx);
        }
    }
}

AsyncHaloExchange::ElementClassification AsyncHaloExchange::classify_elements() const {
    return classification_;
}

void AsyncHaloExchange::start(std::vector<VecX> &solution) { exchanger_.start_exchange(solution); }

bool AsyncHaloExchange::test() {
#ifdef DRIFTER_USE_MPI
    // Test if all communications are complete
    // This would require access to the internal requests
    // For now, return false (not complete)
    return false;
#else
    return true;
#endif
}

void AsyncHaloExchange::wait() { exchanger_.finish_exchange(); }

// =============================================================================
// FaceHaloExchange implementation
// =============================================================================

FaceHaloExchange::FaceHaloExchange(const DomainDecomposition &decomp) : decomp_(decomp) {
    identify_inter_rank_faces();
}

void FaceHaloExchange::identify_inter_rank_faces() {
    inter_rank_faces_.clear();

    // This would iterate over all local elements and their faces,
    // identifying those that connect to elements on other ranks
    // Implementation depends on mesh face connectivity structure
}

void FaceHaloExchange::exchange_faces(std::vector<std::vector<VecX>> &face_data) {
    // Exchange only face data, not full element data
    // This is more efficient for flux computation

#ifdef DRIFTER_USE_MPI
    // Pack face data
    // Send/receive
    // Unpack into remote_face_data_
#endif
}

const std::vector<VecX> &FaceHaloExchange::get_remote_face_data(Index local_elem,
                                                                int face_id) const {
    static std::vector<VecX> empty;
    auto it = remote_face_data_.find({local_elem, face_id});
    if (it != remote_face_data_.end()) {
        // Return as single-element vector
        static std::vector<VecX> result;
        result = {it->second};
        return result;
    }
    return empty;
}

} // namespace drifter
