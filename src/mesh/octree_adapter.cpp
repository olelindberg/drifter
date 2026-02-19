// Octree adapter implementation

#include "mesh/octree_adapter.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace drifter {

// =============================================================================
// OctreeAdapter implementation
// =============================================================================

OctreeAdapter::OctreeAdapter(Real xmin, Real xmax, Real ymin, Real ymax, Real zmin, Real zmax) {
    domain_.xmin = xmin;
    domain_.xmax = xmax;
    domain_.ymin = ymin;
    domain_.ymax = ymax;
    domain_.zmin = zmin;
    domain_.zmax = zmax;
}

void OctreeAdapter::build_uniform(int nx, int ny, int nz) {
    // Calculate initial refinement levels
    int level_x = 0, level_y = 0, level_z = 0;
    int cx = 1, cy = 1, cz = 1;

    while (cx < nx) {
        cx *= 2;
        ++level_x;
    }
    while (cy < ny) {
        cy *= 2;
        ++level_y;
    }
    while (cz < nz) {
        cz *= 2;
        ++level_z;
    }

    // Create root node
    root_ = std::make_unique<OctreeNode>();
    root_->morton = 0;
    root_->level = DirectionalLevel(0, 0, 0);
    root_->bounds = domain_;
    root_->mask = RefineMask::XYZ; // Will refine in all directions

    // Build tree to desired level
    std::queue<OctreeNode*> work_queue;
    work_queue.push(root_.get());

    while (!work_queue.empty()) {
        OctreeNode* node = work_queue.front();
        work_queue.pop();

        // Check if we need to refine further
        bool need_x = node->level.level_x < level_x;
        bool need_y = node->level.level_y < level_y;
        bool need_z = node->level.level_z < level_z;

        if (!need_x && !need_y && !need_z) {
            continue; // This is a leaf
        }

        // Set refinement mask
        node->mask = RefineMask::NONE;
        if (need_x)
            node->mask = node->mask | RefineMask::X;
        if (need_y)
            node->mask = node->mask | RefineMask::Y;
        if (need_z)
            node->mask = node->mask | RefineMask::Z;

        // Create children
        create_children(node);

        // Add children to work queue
        for (auto &child : node->children) {
            work_queue.push(child.get());
        }
    }

    rebuild_leaf_list();
}

void OctreeAdapter::build_adaptive(const std::function<bool(const ElementBounds &)> &refine_func,
                                   int max_level_x, int max_level_y, int max_level_z) {

    // Create root node
    root_ = std::make_unique<OctreeNode>();
    root_->morton = 0;
    root_->level = DirectionalLevel(0, 0, 0);
    root_->bounds = domain_;

    // Adaptive refinement
    std::queue<OctreeNode*> work_queue;
    work_queue.push(root_.get());

    while (!work_queue.empty()) {
        OctreeNode* node = work_queue.front();
        work_queue.pop();

        // Check if we should refine
        if (!refine_func(node->bounds)) {
            continue; // Don't refine this node
        }

        // Check level limits
        bool can_x = node->level.level_x < max_level_x;
        bool can_y = node->level.level_y < max_level_y;
        bool can_z = node->level.level_z < max_level_z;

        if (!can_x && !can_y && !can_z) {
            continue; // At max level
        }

        // Set refinement mask (refine all directions that can be refined)
        node->mask = RefineMask::NONE;
        if (can_x)
            node->mask = node->mask | RefineMask::X;
        if (can_y)
            node->mask = node->mask | RefineMask::Y;
        if (can_z)
            node->mask = node->mask | RefineMask::Z;

        // Create children
        create_children(node);

        // Add children to work queue
        for (auto &child : node->children) {
            work_queue.push(child.get());
        }
    }

    rebuild_leaf_list();
}

void OctreeAdapter::balance() {
    // Iterate until no more refinement is needed
    bool changed = true;

    while (changed) {
        changed = false;

        // Collect current leaves
        std::vector<OctreeNode*> current_leaves = leaves_;

        for (OctreeNode* node : current_leaves) {
            // Check all neighbors
            for (int face = 0; face < 6; ++face) {
                OctreeNode* neighbor = find_neighbor_same_or_coarser(node, face);
                if (!neighbor)
                    continue; // Boundary

                // Check 2:1 balance constraint per axis
                DirectionalLevel diff;
                diff.level_x = node->level.level_x - neighbor->level.level_x;
                diff.level_y = node->level.level_y - neighbor->level.level_y;
                diff.level_z = node->level.level_z - neighbor->level.level_z;

                // If neighbor is more than 1 level coarser, refine it
                RefineMask needed = RefineMask::NONE;
                if (diff.level_x > 1)
                    needed = needed | RefineMask::X;
                if (diff.level_y > 1)
                    needed = needed | RefineMask::Y;
                if (diff.level_z > 1)
                    needed = needed | RefineMask::Z;

                if (needed != RefineMask::NONE && neighbor->is_leaf()) {
                    neighbor->mask = needed;
                    create_children(neighbor);
                    changed = true;
                }
            }
        }

        if (changed) {
            rebuild_leaf_list();
        }
    }
}

void OctreeAdapter::refine(const std::vector<Index> &elements,
                           const std::vector<RefineMask> &masks) {
    for (size_t i = 0; i < elements.size(); ++i) {
        Index elem = elements[i];
        RefineMask mask = masks[i];

        if (elem < 0 || elem >= static_cast<Index>(leaves_.size()))
            continue;

        OctreeNode* node = leaves_[elem];
        if (!node->is_leaf())
            continue;

        node->mask = mask;
        create_children(node);
    }

    rebuild_leaf_list();
    balance();
}

void OctreeAdapter::coarsen(const std::vector<Index> &parent_elements) {
    // Mark parents for coarsening
    for (Index elem : parent_elements) {
        if (elem < 0 || elem >= static_cast<Index>(leaves_.size()))
            continue;

        OctreeNode* node = leaves_[elem];
        if (!node->parent)
            continue;

        // Check if all siblings are leaves
        OctreeNode* parent = node->parent;
        bool all_leaves = true;
        for (auto &child : parent->children) {
            if (!child->is_leaf()) {
                all_leaves = false;
                break;
            }
        }

        if (all_leaves) {
            // Coarsen: remove children
            parent->children.clear();
        }
    }

    rebuild_leaf_list();
}

const ElementBounds &OctreeAdapter::element_bounds(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        throw std::out_of_range("Invalid element index");
    }
    return leaves_[elem]->bounds;
}

DirectionalLevel OctreeAdapter::element_level(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        throw std::out_of_range("Invalid element index");
    }
    return leaves_[elem]->level;
}

uint64_t OctreeAdapter::element_morton(Index elem) const {
    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        throw std::out_of_range("Invalid element index");
    }
    return leaves_[elem]->morton;
}

Vec3 OctreeAdapter::element_center(Index elem) const { return element_bounds(elem).center(); }

Vec3 OctreeAdapter::element_size(Index elem) const { return element_bounds(elem).size(); }

NeighborInfo OctreeAdapter::get_neighbor(Index elem, int face_id) const {
    NeighborInfo info;

    if (elem < 0 || elem >= static_cast<Index>(leaves_.size())) {
        info.type = FaceConnectionType::Boundary;
        return info;
    }

    OctreeNode* node = leaves_[elem];

    // Check if at domain boundary
    const ElementBounds &b = node->bounds;
    bool at_boundary = false;

    switch (face_id) {
    case 0:
        at_boundary = (b.xmin <= domain_.xmin + 1e-12);
        break;
    case 1:
        at_boundary = (b.xmax >= domain_.xmax - 1e-12);
        break;
    case 2:
        at_boundary = (b.ymin <= domain_.ymin + 1e-12);
        break;
    case 3:
        at_boundary = (b.ymax >= domain_.ymax - 1e-12);
        break;
    case 4:
        at_boundary = (b.zmin <= domain_.zmin + 1e-12);
        break;
    case 5:
        at_boundary = (b.zmax >= domain_.zmax - 1e-12);
        break;
    }

    if (at_boundary) {
        info.type = FaceConnectionType::Boundary;
        return info;
    }

    // Find neighbor(s)
    OctreeNode* coarse_neighbor = find_neighbor_same_or_coarser(node, face_id);

    if (!coarse_neighbor) {
        info.type = FaceConnectionType::Boundary;
        return info;
    }

    if (coarse_neighbor->is_leaf()) {
        // Single neighbor - check if same level or coarser
        DirectionalLevel node_level = node->level;
        DirectionalLevel neigh_level = coarse_neighbor->level;

        // Determine connection type based on level differences in tangent
        // directions
        auto [t1, t2] = get_face_tangent_axes(face_id);

        int diff_t1 = 0, diff_t2 = 0;
        if (t1 == 0)
            diff_t1 = node_level.level_x - neigh_level.level_x;
        else if (t1 == 1)
            diff_t1 = node_level.level_y - neigh_level.level_y;
        else
            diff_t1 = node_level.level_z - neigh_level.level_z;

        if (t2 == 0)
            diff_t2 = node_level.level_x - neigh_level.level_x;
        else if (t2 == 1)
            diff_t2 = node_level.level_y - neigh_level.level_y;
        else
            diff_t2 = node_level.level_z - neigh_level.level_z;

        // We are finer, neighbor is coarser
        if (diff_t1 <= 0 && diff_t2 <= 0) {
            info.type = FaceConnectionType::SameLevel;
        } else {
            // This shouldn't happen if we're a leaf looking at a coarser
            // neighbor
            info.type = FaceConnectionType::SameLevel;
        }

        info.neighbor_elements.push_back(coarse_neighbor->leaf_index);
        info.neighbor_faces.push_back((face_id % 2 == 0) ? face_id + 1 : face_id - 1);
        info.subface_indices.push_back(0);

    } else {
        // Neighbor is refined - we need to find all fine neighbors
        std::vector<OctreeNode*> fine_neighbors = find_fine_neighbors(node, face_id);

        // Determine connection type
        info.type = get_connection_type(node->level, fine_neighbors[0]->level, face_id);

        for (size_t i = 0; i < fine_neighbors.size(); ++i) {
            info.neighbor_elements.push_back(fine_neighbors[i]->leaf_index);
            info.neighbor_faces.push_back((face_id % 2 == 0) ? face_id + 1 : face_id - 1);
            info.subface_indices.push_back(static_cast<int>(i));
        }
    }

    return info;
}

std::array<NeighborInfo, 6> OctreeAdapter::get_face_neighbors(Index elem) const {
    std::array<NeighborInfo, 6> neighbors;
    for (int f = 0; f < 6; ++f) {
        neighbors[f] = get_neighbor(elem, f);
    }
    return neighbors;
}

std::vector<std::vector<FaceConnection>> OctreeAdapter::build_face_connections() const {
    std::vector<std::vector<FaceConnection>> connections(num_elements());

    for (Index e = 0; e < num_elements(); ++e) {
        connections[e].resize(6);

        for (int f = 0; f < 6; ++f) {
            NeighborInfo info = get_neighbor(e, f);

            FaceConnection &conn = connections[e][f];
            conn.type = info.type;
            conn.coarse_elem = e;
            conn.coarse_face_id = f;
            conn.fine_elems = info.neighbor_elements;
            conn.fine_face_ids = info.neighbor_faces;
            conn.subface_indices = info.subface_indices;
        }
    }

    return connections;
}

Index OctreeAdapter::find_element(const Vec3 &p) const {
    if (!domain_.contains(p)) {
        return -1;
    }

    // Search from root
    OctreeNode* node = root_.get();

    while (!node->is_leaf()) {
        bool found = false;
        for (auto &child : node->children) {
            if (child->bounds.contains(p)) {
                node = child.get();
                found = true;
                break;
            }
        }
        if (!found)
            break;
    }

    return node->is_leaf() ? node->leaf_index : -1;
}

std::vector<Index> OctreeAdapter::morton_order() const {
    std::vector<Index> order(num_elements());
    for (Index i = 0; i < num_elements(); ++i) {
        order[i] = i;
    }

    // Sort by Morton code
    std::sort(order.begin(), order.end(),
              [this](Index a, Index b) { return leaves_[a]->morton < leaves_[b]->morton; });

    return order;
}

std::vector<std::pair<Index, Index>> OctreeAdapter::morton_partition(int num_partitions) const {
    std::vector<Index> order = morton_order();
    std::vector<std::pair<Index, Index>> partitions(num_partitions);

    Index elements_per_partition = num_elements() / num_partitions;
    Index remainder = num_elements() % num_partitions;

    Index start = 0;
    for (int p = 0; p < num_partitions; ++p) {
        Index count = elements_per_partition + (p < remainder ? 1 : 0);
        partitions[p] = {start, start + count};
        start += count;
    }

    return partitions;
}

void OctreeAdapter::rebuild_leaf_list() {
    leaves_.clear();
    morton_lookup_.clear();

    if (root_) {
        collect_leaves(root_.get(), leaves_);
        build_lookup(root_.get());
    }

    // Assign leaf indices
    for (Index i = 0; i < static_cast<Index>(leaves_.size()); ++i) {
        leaves_[i]->leaf_index = i;
    }
}

void OctreeAdapter::collect_leaves(OctreeNode* node, std::vector<OctreeNode*> &leaves) {
    if (node->is_leaf()) {
        leaves.push_back(node);
    } else {
        for (auto &child : node->children) {
            collect_leaves(child.get(), leaves);
        }
    }
}

void OctreeAdapter::build_lookup(OctreeNode* node) {
    morton_lookup_[node->morton] = node;
    for (auto &child : node->children) {
        build_lookup(child.get());
    }
}

void OctreeAdapter::create_children(OctreeNode* node) {
    if (!node->children.empty())
        return;

    bool ref_x = refines_x(node->mask);
    bool ref_y = refines_y(node->mask);
    bool ref_z = refines_z(node->mask);

    int nx = ref_x ? 2 : 1;
    int ny = ref_y ? 2 : 1;
    int nz = ref_z ? 2 : 1;

    const ElementBounds &b = node->bounds;
    Real dx = (b.xmax - b.xmin) / nx;
    Real dy = (b.ymax - b.ymin) / ny;
    Real dz = (b.zmax - b.zmin) / nz;

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                auto child = std::make_unique<OctreeNode>();

                // Compute child Morton code
                child->morton = MortonUtil::refine(node->morton, ix, iy, iz, ref_x, ref_y, ref_z);

                // Compute child level
                child->level.level_x = node->level.level_x + (ref_x ? 1 : 0);
                child->level.level_y = node->level.level_y + (ref_y ? 1 : 0);
                child->level.level_z = node->level.level_z + (ref_z ? 1 : 0);

                // Compute child bounds
                child->bounds.xmin = b.xmin + ix * dx;
                child->bounds.xmax = b.xmin + (ix + 1) * dx;
                child->bounds.ymin = b.ymin + iy * dy;
                child->bounds.ymax = b.ymin + (iy + 1) * dy;
                child->bounds.zmin = b.zmin + iz * dz;
                child->bounds.zmax = b.zmin + (iz + 1) * dz;

                child->parent = node;

                node->children.push_back(std::move(child));
            }
        }
    }
}

OctreeNode* OctreeAdapter::find_neighbor_same_or_coarser(OctreeNode* node, int face_id) const {

    // Use Morton code arithmetic to find neighbor
    // This is a simplified version - full implementation would use
    // SeaMesh's neighbor finding algorithm

    // For now, brute force search
    const ElementBounds &b = node->bounds;
    Vec3 neighbor_center;

    switch (face_id) {
    case 0:
        neighbor_center = Vec3(b.xmin - 0.5 * (b.xmax - b.xmin), 0.5 * (b.ymin + b.ymax),
                               0.5 * (b.zmin + b.zmax));
        break;
    case 1:
        neighbor_center = Vec3(b.xmax + 0.5 * (b.xmax - b.xmin), 0.5 * (b.ymin + b.ymax),
                               0.5 * (b.zmin + b.zmax));
        break;
    case 2:
        neighbor_center = Vec3(0.5 * (b.xmin + b.xmax), b.ymin - 0.5 * (b.ymax - b.ymin),
                               0.5 * (b.zmin + b.zmax));
        break;
    case 3:
        neighbor_center = Vec3(0.5 * (b.xmin + b.xmax), b.ymax + 0.5 * (b.ymax - b.ymin),
                               0.5 * (b.zmin + b.zmax));
        break;
    case 4:
        neighbor_center = Vec3(0.5 * (b.xmin + b.xmax), 0.5 * (b.ymin + b.ymax),
                               b.zmin - 0.5 * (b.zmax - b.zmin));
        break;
    case 5:
        neighbor_center = Vec3(0.5 * (b.xmin + b.xmax), 0.5 * (b.ymin + b.ymax),
                               b.zmax + 0.5 * (b.zmax - b.zmin));
        break;
    }

    // Search for element containing this point
    if (!domain_.contains(neighbor_center)) {
        return nullptr;
    }

    OctreeNode* current = root_.get();
    while (!current->is_leaf()) {
        bool found = false;
        for (auto &child : current->children) {
            if (child->bounds.contains(neighbor_center)) {
                current = child.get();
                found = true;
                break;
            }
        }
        if (!found)
            break;
    }

    return current;
}

std::vector<OctreeNode*> OctreeAdapter::find_fine_neighbors(OctreeNode* node, int face_id) const {

    std::vector<OctreeNode*> neighbors;

    // Find the coarse neighbor first
    OctreeNode* coarse = find_neighbor_same_or_coarser(node, face_id);
    if (!coarse)
        return neighbors;

    // Collect all leaves that share the face
    std::queue<OctreeNode*> work;
    work.push(coarse);

    const ElementBounds &b = node->bounds;
    Real tol = 1e-12;

    while (!work.empty()) {
        OctreeNode* n = work.front();
        work.pop();

        if (n->is_leaf()) {
            // Check if this leaf shares the face
            bool shares_face = false;
            const ElementBounds &nb = n->bounds;

            switch (face_id) {
            case 0:
                shares_face = std::abs(nb.xmax - b.xmin) < tol;
                break;
            case 1:
                shares_face = std::abs(nb.xmin - b.xmax) < tol;
                break;
            case 2:
                shares_face = std::abs(nb.ymax - b.ymin) < tol;
                break;
            case 3:
                shares_face = std::abs(nb.ymin - b.ymax) < tol;
                break;
            case 4:
                shares_face = std::abs(nb.zmax - b.zmin) < tol;
                break;
            case 5:
                shares_face = std::abs(nb.zmin - b.zmax) < tol;
                break;
            }

            if (shares_face) {
                neighbors.push_back(n);
            }
        } else {
            for (auto &child : n->children) {
                work.push(child.get());
            }
        }
    }

    return neighbors;
}

// =============================================================================
// Refinement criteria implementation
// =============================================================================

namespace refinement_criteria {

bool GradientCriterion::should_refine(const VecX &solution, const ElementBounds &bounds,
                                      const DirectionalLevel &level) const {

    // Estimate gradient magnitude from solution variation
    Real max_val = solution.maxCoeff();
    Real min_val = solution.minCoeff();
    Real variation = max_val - min_val;

    Vec3 size = bounds.size();
    Real min_size = size.minCoeff();

    Real gradient_estimate = variation / min_size;

    return gradient_estimate > threshold;
}

RefineMask GradientCriterion::refinement_mask(const VecX &solution,
                                              const ElementBounds &bounds) const {

    // For simplicity, refine in all directions if gradient is high
    // A more sophisticated version would compute directional gradients
    return RefineMask::XYZ;
}

bool BathymetryCriterion::should_refine(const ElementBounds &bounds,
                                        const DirectionalLevel &level) const {

    // Sample bathymetry at corners and center
    std::vector<Real> h_values;
    h_values.push_back(bathymetry(bounds.xmin, bounds.ymin));
    h_values.push_back(bathymetry(bounds.xmax, bounds.ymin));
    h_values.push_back(bathymetry(bounds.xmin, bounds.ymax));
    h_values.push_back(bathymetry(bounds.xmax, bounds.ymax));
    h_values.push_back(
        bathymetry(0.5 * (bounds.xmin + bounds.xmax), 0.5 * (bounds.ymin + bounds.ymax)));

    Real h_max = *std::max_element(h_values.begin(), h_values.end());
    Real h_min = *std::min_element(h_values.begin(), h_values.end());

    return (h_max - h_min) > threshold;
}

bool CoastlineCriterion::should_refine(const ElementBounds &bounds,
                                       const DirectionalLevel &level) const {

    if (level.max_level() >= max_level)
        return false;

    // Check if element contains coastline (transition from wet to dry)
    int n_samples = 4;
    int wet_count = 0;
    int dry_count = 0;

    for (int i = 0; i <= n_samples; ++i) {
        for (int j = 0; j <= n_samples; ++j) {
            Real x = bounds.xmin + i * (bounds.xmax - bounds.xmin) / n_samples;
            Real y = bounds.ymin + j * (bounds.ymax - bounds.ymin) / n_samples;

            if (is_land(x, y)) {
                ++dry_count;
            } else {
                ++wet_count;
            }
        }
    }

    // Refine if mixed wet/dry
    return wet_count > 0 && dry_count > 0;
}

bool CombinedCriterion::should_refine(const ElementBounds &bounds,
                                      const DirectionalLevel &level) const {

    for (const auto &criterion : criteria) {
        if (criterion(bounds, level)) {
            return true;
        }
    }
    return false;
}

} // namespace refinement_criteria

} // namespace drifter
