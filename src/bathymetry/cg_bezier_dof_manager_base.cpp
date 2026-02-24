#include "bathymetry/cg_bezier_dof_manager_base.hpp"
#include <algorithm>
#include <stdexcept>

namespace drifter {

CGBezierDofManagerBase::CGBezierDofManagerBase(const QuadtreeAdapter &mesh) : mesh_(mesh) {}

Index CGBezierDofManagerBase::global_dof(Index elem, int local_dof) const {
    if (elem < 0 || elem >= static_cast<Index>(elem_to_global_.size())) {
        throw std::out_of_range("CGBezierDofManagerBase: element index out of range");
    }
    if (local_dof < 0 || local_dof >= num_element_dofs()) {
        throw std::out_of_range("CGBezierDofManagerBase: local DOF index out of range");
    }
    return elem_to_global_[elem][local_dof];
}

const std::vector<Index> &CGBezierDofManagerBase::element_dofs(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(elem_to_global_.size())) {
        throw std::out_of_range("CGBezierDofManagerBase: element index out of range");
    }
    return elem_to_global_[elem];
}

bool CGBezierDofManagerBase::is_boundary_dof(Index dof) const {
    return boundary_dof_set_.count(dof) > 0;
}

bool CGBezierDofManagerBase::is_constrained(Index dof) const {
    return constrained_dofs_.count(dof) > 0;
}

Index CGBezierDofManagerBase::global_to_free(Index global_dof) const {
    if (global_dof < 0 || global_dof >= num_global_dofs_) {
        return -1;
    }
    return global_to_free_[global_dof];
}

Index CGBezierDofManagerBase::free_to_global(Index free_dof) const {
    if (free_dof < 0 || free_dof >= num_free_dofs_) {
        return -1;
    }
    return free_to_global_[free_dof];
}

SpMat CGBezierDofManagerBase::build_constraint_matrix() const {
    Index nrows = num_constraints();
    Index ncols = num_global_dofs_;

    if (nrows == 0) {
        return SpMat(0, ncols);
    }

    std::vector<Eigen::Triplet<Real>> triplets;
    get_constraint_triplets(triplets);

    SpMat A(nrows, ncols);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

Index CGBezierDofManagerBase::find_dof_at_position(const Vec2 &pos) const {
    auto key = quantize_position(pos);
    auto it = position_to_dof_.find(key);
    if (it != position_to_dof_.end()) {
        return it->second;
    }
    return -1;
}

Index CGBezierDofManagerBase::register_dof_at_position(const Vec2 &pos) {
    auto key = quantize_position(pos);
    Index dof = num_global_dofs_++;
    position_to_dof_[key] = dof;
    return dof;
}

void CGBezierDofManagerBase::identify_boundary_dofs_impl(
    const std::function<std::vector<int>(int)> &get_edge_dofs) {
    boundary_dofs_.clear();
    boundary_dof_set_.clear();

    mesh_.for_each_boundary_edge([this, &get_edge_dofs](Index elem, int edge) {
        std::vector<int> edge_dof_list = get_edge_dofs(edge);
        for (int local_dof : edge_dof_list) {
            Index global = elem_to_global_[elem][local_dof];
            if (boundary_dof_set_.insert(global).second) {
                boundary_dofs_.push_back(global);
            }
        }
    });

    std::sort(boundary_dofs_.begin(), boundary_dofs_.end());
}

void CGBezierDofManagerBase::build_dof_mappings() {
    global_to_free_.resize(num_global_dofs_, -1);
    free_to_global_.clear();
    free_to_global_.reserve(num_global_dofs_);

    num_free_dofs_ = 0;
    for (Index g = 0; g < num_global_dofs_; ++g) {
        if (!is_constrained(g)) {
            global_to_free_[g] = num_free_dofs_++;
            free_to_global_.push_back(g);
        }
    }
}

void CGBezierDofManagerBase::initialize_elem_to_global(Index num_elements, int ndof) {
    elem_to_global_.resize(num_elements);
    for (Index e = 0; e < num_elements; ++e) {
        elem_to_global_[e].resize(ndof, -1);
    }
}

} // namespace drifter
