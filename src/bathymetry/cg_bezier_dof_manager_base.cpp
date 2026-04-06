#include "bathymetry/cg_bezier_dof_manager_base.hpp"
#include "mesh/hilbert.hpp"
#include "mesh/morton.hpp"
#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

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
    register_dof_position(dof, pos);
    return dof;
}

void CGBezierDofManagerBase::register_dof_position(Index dof, const Vec2 &pos) {
    if (dof >= static_cast<Index>(dof_positions_.size())) {
        dof_positions_.resize(dof + 1);
    }
    dof_positions_[dof] = pos;
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

std::vector<Index> CGBezierDofManagerBase::reorder_dofs_by_morton() {
    Index n = num_global_dofs_;
    if (n == 0) {
        return {};
    }

    // Compute Morton code for each DOF from its physical position.
    // Map positions to integer grid coordinates relative to the mesh domain.
    const auto &domain = mesh_.domain_bounds();
    Real dx = domain.xmax - domain.xmin;
    Real dy = domain.ymax - domain.ymin;

    // Use a fine integer grid for Morton encoding (21-bit max per axis)
    constexpr uint32_t GRID_RESOLUTION = (1 << 20); // ~1M cells per axis

    std::vector<uint64_t> morton_codes(n);
    for (Index i = 0; i < n; ++i) {
        const Vec2 &pos = dof_positions_[i];
        Real fx = (dx > 0) ? (pos(0) - domain.xmin) / dx : 0.0;
        Real fy = (dy > 0) ? (pos(1) - domain.ymin) / dy : 0.0;
        fx = std::clamp(fx, 0.0, 1.0);
        fy = std::clamp(fy, 0.0, 1.0);
        uint32_t ix = static_cast<uint32_t>(fx * GRID_RESOLUTION);
        uint32_t iy = static_cast<uint32_t>(fy * GRID_RESOLUTION);
        morton_codes[i] = Morton3D::encode(ix, iy, 0);
    }

    // Sort DOF indices by Morton code (stable sort preserves original order for ties)
    std::vector<Index> sorted_dofs(n);
    std::iota(sorted_dofs.begin(), sorted_dofs.end(), 0);
    std::stable_sort(sorted_dofs.begin(), sorted_dofs.end(),
                     [&](Index a, Index b) { return morton_codes[a] < morton_codes[b]; });

    // Build permutation: perm[old_index] = new_index
    std::vector<Index> perm(n);
    for (Index new_idx = 0; new_idx < n; ++new_idx) {
        perm[sorted_dofs[new_idx]] = new_idx;
    }

    // Apply permutation to elem_to_global_
    for (auto &dofs : elem_to_global_) {
        for (auto &d : dofs) {
            d = perm[d];
        }
    }

    // Apply permutation to position_to_dof_ map values
    for (auto &[key, dof] : position_to_dof_) {
        dof = perm[dof];
    }

    // Apply permutation to boundary_dofs_
    for (auto &d : boundary_dofs_) {
        d = perm[d];
    }
    std::sort(boundary_dofs_.begin(), boundary_dofs_.end());

    // Rebuild boundary_dof_set_
    boundary_dof_set_.clear();
    boundary_dof_set_.insert(boundary_dofs_.begin(), boundary_dofs_.end());

    // Apply permutation to constrained_dofs_
    std::set<Index> new_constrained;
    for (Index d : constrained_dofs_) {
        new_constrained.insert(perm[d]);
    }
    constrained_dofs_ = std::move(new_constrained);

    // Permute dof_positions_
    std::vector<Vec2> new_positions(n);
    for (Index i = 0; i < n; ++i) {
        new_positions[perm[i]] = dof_positions_[i];
    }
    dof_positions_ = std::move(new_positions);

    return perm;
}

std::vector<Index> CGBezierDofManagerBase::reorder_dofs_hierarchical(
    const std::function<DOFType(int)> &classify_dof) {

    Index n = num_global_dofs_;
    if (n == 0) {
        return {};
    }

    const int ndof_per_elem = num_element_dofs();
    const Index num_elements = mesh_.num_elements();

    // Build reverse map: global DOF -> list of (element, local_dof)
    std::vector<std::vector<std::pair<Index, int>>> dof_to_elements(n);
    for (Index e = 0; e < num_elements; ++e) {
        const auto &global_dofs = elem_to_global_[e];
        for (int local = 0; local < ndof_per_elem; ++local) {
            Index g = global_dofs[local];
            if (g >= 0 && g < n) {
                dof_to_elements[g].emplace_back(e, local);
            }
        }
    }

    // Compute metadata for each DOF
    std::vector<DOFMetadata> metadata(n);

    // Domain bounds for Hilbert encoding
    const auto &domain = mesh_.domain_bounds();
    Real dx = domain.xmax - domain.xmin;
    Real dy = domain.ymax - domain.ymin;
    constexpr int HILBERT_ORDER = 10; // 2^10 = 1024 grid per axis

    Index skeleton_count = 0;

    for (Index dof = 0; dof < n; ++dof) {
        DOFMetadata &meta = metadata[dof];

        // Determine DOF type and minimum level from adjacent elements
        int min_level = INT_MAX;
        DOFType type = DOFType::Interior;

        for (const auto &[elem, local_dof] : dof_to_elements[dof]) {
            // Get element level (use max of x,y levels for isotropic comparison)
            QuadLevel elem_level = mesh_.element_level(elem);
            int level = std::max(elem_level.x, elem_level.y);
            min_level = std::min(min_level, level);

            // Get DOF type (take the "most shared" type: Vertex < Edge < Interior)
            DOFType local_type = classify_dof(local_dof);
            if (local_type < type) {
                type = local_type;
            }
        }

        meta.type = type;
        meta.level = (min_level == INT_MAX) ? 0 : min_level;
        meta.is_constrained = (constrained_dofs_.count(dof) > 0);

        // Count skeleton DOFs
        if (type != DOFType::Interior) {
            ++skeleton_count;
        }

        // Compute Hilbert key from position
        const Vec2 &pos = dof_positions_[dof];
        Real fx = (dx > 0) ? (pos(0) - domain.xmin) / dx : 0.0;
        Real fy = (dy > 0) ? (pos(1) - domain.ymin) / dy : 0.0;
        fx = std::clamp(fx, 0.0, 1.0);
        fy = std::clamp(fy, 0.0, 1.0);
        uint32_t ix = static_cast<uint32_t>(fx * ((1 << HILBERT_ORDER) - 1));
        uint32_t iy = static_cast<uint32_t>(fy * ((1 << HILBERT_ORDER) - 1));
        meta.hilbert_key = Hilbert2D::encode(ix, iy, HILBERT_ORDER);
    }

    // Sort DOF indices by (level, is_constrained, type, hilbert_key)
    std::vector<Index> sorted_dofs(n);
    std::iota(sorted_dofs.begin(), sorted_dofs.end(), 0);

    std::stable_sort(sorted_dofs.begin(), sorted_dofs.end(),
                     [&metadata](Index a, Index b) {
                         const auto &ma = metadata[a];
                         const auto &mb = metadata[b];

                         // Primary: level (coarse to fine)
                         if (ma.level != mb.level)
                             return ma.level < mb.level;

                         // Secondary: free DOFs before constrained
                         if (ma.is_constrained != mb.is_constrained)
                             return !ma.is_constrained;

                         // Tertiary: DOF type (Vertex < Edge < Interior)
                         if (ma.type != mb.type)
                             return ma.type < mb.type;

                         // Quaternary: Hilbert curve ordering
                         return ma.hilbert_key < mb.hilbert_key;
                     });

    // Build permutation: perm[old_index] = new_index
    std::vector<Index> perm(n);
    for (Index new_idx = 0; new_idx < n; ++new_idx) {
        perm[sorted_dofs[new_idx]] = new_idx;
    }

    // Build inverse permutation
    std::vector<Index> inv_perm(n);
    for (Index old_idx = 0; old_idx < n; ++old_idx) {
        inv_perm[perm[old_idx]] = old_idx;
    }

    // Apply permutation to elem_to_global_
    for (auto &dofs : elem_to_global_) {
        for (auto &d : dofs) {
            if (d >= 0) {
                d = perm[d];
            }
        }
    }

    // Apply permutation to position_to_dof_ map values
    for (auto &[key, dof] : position_to_dof_) {
        dof = perm[dof];
    }

    // Apply permutation to boundary_dofs_
    for (auto &d : boundary_dofs_) {
        d = perm[d];
    }
    std::sort(boundary_dofs_.begin(), boundary_dofs_.end());

    // Rebuild boundary_dof_set_
    boundary_dof_set_.clear();
    boundary_dof_set_.insert(boundary_dofs_.begin(), boundary_dofs_.end());

    // Apply permutation to constrained_dofs_
    std::set<Index> new_constrained;
    for (Index d : constrained_dofs_) {
        new_constrained.insert(perm[d]);
    }
    constrained_dofs_ = std::move(new_constrained);

    // Permute dof_positions_
    std::vector<Vec2> new_positions(n);
    for (Index i = 0; i < n; ++i) {
        new_positions[perm[i]] = dof_positions_[i];
    }
    dof_positions_ = std::move(new_positions);

    // Permute metadata to NEW ordering
    std::vector<DOFMetadata> new_metadata(n);
    for (Index old_idx = 0; old_idx < n; ++old_idx) {
        new_metadata[perm[old_idx]] = metadata[old_idx];
    }

    // Build level blocks
    std::map<int, std::pair<Index, Index>> level_blocks;
    int current_level = -1;
    Index block_start = 0;
    for (Index i = 0; i < n; ++i) {
        int level = new_metadata[i].level;
        if (level != current_level) {
            if (current_level >= 0) {
                level_blocks[current_level] = {block_start, i};
            }
            current_level = level;
            block_start = i;
        }
    }
    if (current_level >= 0) {
        level_blocks[current_level] = {block_start, n};
    }

    // Store hierarchical ordering result
    hierarchical_ordering_.permutation = std::move(perm);
    hierarchical_ordering_.inverse = std::move(inv_perm);
    hierarchical_ordering_.metadata = std::move(new_metadata);
    hierarchical_ordering_.level_blocks = std::move(level_blocks);
    hierarchical_ordering_.num_skeleton_dofs = skeleton_count;

    return hierarchical_ordering_.permutation;
}

} // namespace drifter
