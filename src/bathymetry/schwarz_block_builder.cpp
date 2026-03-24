#include "bathymetry/schwarz_block_builder.hpp"

namespace drifter {

SchwarzBlockData SchwarzBlockBuilder::build(const SpMat &Q,
                                            const CGBezierDofManagerBase &dof_manager,
                                            bool compute_coloring) {
    SchwarzBlockData data;
    build_element_blocks(data, Q, dof_manager);

    if (compute_coloring) {
        build_element_coloring(data);
    }

    return data;
}

void SchwarzBlockBuilder::build_element_blocks(SchwarzBlockData &data, const SpMat &Q,
                                               const CGBezierDofManagerBase &dof_manager) {
    const auto &all_elem_dofs = dof_manager.all_element_dofs();
    size_t num_elements = all_elem_dofs.size();

    data.element_free_dofs.resize(num_elements);
    data.element_block_lu.resize(num_elements);

    for (size_t e = 0; e < num_elements; ++e) {
        const auto &global_dofs = all_elem_dofs[e];

        // Collect free DOF indices (skip constrained DOFs)
        std::vector<Index> free_dofs;
        free_dofs.reserve(global_dofs.size());
        for (Index g : global_dofs) {
            Index f = dof_manager.global_to_free(g);
            if (f >= 0) {
                free_dofs.push_back(f);
            }
        }

        data.element_free_dofs[e] = free_dofs;

        // Extract element block from Q
        int block_size = static_cast<int>(free_dofs.size());
        if (block_size == 0) {
            // All DOFs constrained, store empty factorization
            data.element_block_lu[e] = Eigen::PartialPivLU<MatX>();
            continue;
        }

        MatX Q_block(block_size, block_size);
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                Q_block(i, j) = Q.coeff(free_dofs[i], free_dofs[j]);
            }
        }

        // LU factorize for local solves
        data.element_block_lu[e] = Q_block.partialPivLu();
    }
}

void SchwarzBlockBuilder::build_element_coloring(SchwarzBlockData &data) {
    // Clear existing coloring
    data.elements_by_color.clear();
    data.num_colors = 0;

    size_t num_elements = data.element_free_dofs.size();
    if (num_elements == 0)
        return;

    // Build DOF -> elements adjacency map
    std::map<Index, std::vector<Index>> dof_to_elements;
    for (size_t e = 0; e < num_elements; ++e) {
        for (Index dof : data.element_free_dofs[e]) {
            dof_to_elements[dof].push_back(static_cast<Index>(e));
        }
    }

    // Greedy graph coloring based on DOF adjacency
    std::vector<int> element_color(num_elements, -1);
    int max_color = -1;

    for (size_t e = 0; e < num_elements; ++e) {
        // Find colors used by adjacent elements (elements sharing DOFs)
        std::set<int> neighbor_colors;
        for (Index dof : data.element_free_dofs[e]) {
            for (Index neighbor : dof_to_elements[dof]) {
                if (neighbor != static_cast<Index>(e) && element_color[neighbor] >= 0) {
                    neighbor_colors.insert(element_color[neighbor]);
                }
            }
        }

        // Find lowest available color
        int color = 0;
        while (neighbor_colors.count(color)) {
            color++;
        }

        element_color[e] = color;
        max_color = std::max(max_color, color);
    }

    // Build color groups
    data.num_colors = max_color + 1;
    data.elements_by_color.resize(data.num_colors);

    for (size_t e = 0; e < num_elements; ++e) {
        data.elements_by_color[element_color[e]].push_back(static_cast<Index>(e));
    }
}

} // namespace drifter
