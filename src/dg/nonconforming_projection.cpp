#include "dg/nonconforming_projection.hpp"
#include "dg/basis_hexahedron.hpp"
#include <cmath>

namespace drifter {

void project_coarse_to_fine_2d(const OctreeAdapter &mesh, std::vector<VecX> &element_data,
                               int order, SeabedInterpolation method) {

    const int n1d = order + 1;
    const int n2d = n1d * n1d;

    // Create interpolator for 2D evaluation
    SeabedInterpolator interp(order, method);
    const VecX &lgl_nodes = interp.lgl_nodes();

    const auto &elements = mesh.elements();
    size_t num_elements = elements.size();

    // For each element, check horizontal faces (0-3) for coarser neighbors
    for (size_t e = 0; e < num_elements; ++e) {
        const auto &my_bounds = elements[e]->bounds;
        Real my_area = (my_bounds.xmax - my_bounds.xmin) * (my_bounds.ymax - my_bounds.ymin);

        // Check faces 0-3 (horizontal: -x, +x, -y, +y)
        for (int face_id = 0; face_id < 4; ++face_id) {
            NeighborInfo info = mesh.get_neighbor(static_cast<Index>(e), face_id);
            if (info.is_boundary() || info.neighbor_elements.empty())
                continue;

            Index neigh_e = info.neighbor_elements[0];
            const auto &neigh_bounds = elements[neigh_e]->bounds;
            Real neigh_area =
                (neigh_bounds.xmax - neigh_bounds.xmin) * (neigh_bounds.ymax - neigh_bounds.ymin);

            // Check if neighbor is coarser (larger area)
            if (neigh_area <= my_area * 1.5)
                continue;

            // Neighbor is coarser - project its polynomial onto our face nodes
            const VecX &coarse_data = element_data[neigh_e];

            // For each DOF on the shared face
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    // Check if this DOF is on the interface face
                    bool on_face = false;
                    int face_i = i, face_j = j;

                    if (face_id == 0 && i == 0)
                        on_face = true; // -x face
                    else if (face_id == 1 && i == n1d - 1)
                        on_face = true; // +x face
                    else if (face_id == 2 && j == 0)
                        on_face = true; // -y face
                    else if (face_id == 3 && j == n1d - 1)
                        on_face = true; // +y face

                    if (!on_face)
                        continue;

                    // Get physical coordinates of this DOF
                    Real xi_ref = lgl_nodes(i);
                    Real eta_ref = lgl_nodes(j);

                    Real phys_x =
                        my_bounds.xmin + 0.5 * (xi_ref + 1.0) * (my_bounds.xmax - my_bounds.xmin);
                    Real phys_y =
                        my_bounds.ymin + 0.5 * (eta_ref + 1.0) * (my_bounds.ymax - my_bounds.ymin);

                    // Transform to coarse element's reference coordinates
                    Real coarse_xi = 2.0 * (phys_x - neigh_bounds.xmin) /
                                         (neigh_bounds.xmax - neigh_bounds.xmin) -
                                     1.0;
                    Real coarse_eta = 2.0 * (phys_y - neigh_bounds.ymin) /
                                          (neigh_bounds.ymax - neigh_bounds.ymin) -
                                      1.0;

                    // Clamp to [-1, 1] to avoid numerical issues at boundaries
                    coarse_xi = std::max(-1.0, std::min(1.0, coarse_xi));
                    coarse_eta = std::max(-1.0, std::min(1.0, coarse_eta));

                    // Interpolate coarse element's polynomial at this point
                    Real projected_value =
                        interp.evaluate_scalar_2d(coarse_data, coarse_xi, coarse_eta);

                    // Overwrite the fine element's DOF value
                    int idx = i + n1d * j;
                    element_data[e](idx) = projected_value;
                }
            }
        }
    }
}

void project_coarse_to_fine_3d(const OctreeAdapter &mesh, std::vector<VecX> &element_data,
                               int order, SeabedInterpolation method) {

    const int n1d = order + 1;
    const int n3d = n1d * n1d * n1d;

    // Create interpolator
    SeabedInterpolator interp(order, method);
    const VecX &lgl_nodes = interp.lgl_nodes();

    const auto &elements = mesh.elements();
    size_t num_elements = elements.size();

    // For each element, check horizontal faces (0-3) for coarser neighbors
    for (size_t e = 0; e < num_elements; ++e) {
        const auto &my_bounds = elements[e]->bounds;
        Real my_area = (my_bounds.xmax - my_bounds.xmin) * (my_bounds.ymax - my_bounds.ymin);

        // Check faces 0-3 (horizontal: -x, +x, -y, +y)
        for (int face_id = 0; face_id < 4; ++face_id) {
            NeighborInfo info = mesh.get_neighbor(static_cast<Index>(e), face_id);
            if (info.is_boundary() || info.neighbor_elements.empty())
                continue;

            Index neigh_e = info.neighbor_elements[0];
            const auto &neigh_bounds = elements[neigh_e]->bounds;
            Real neigh_area =
                (neigh_bounds.xmax - neigh_bounds.xmin) * (neigh_bounds.ymax - neigh_bounds.ymin);

            // Check if neighbor is coarser
            if (neigh_area <= my_area * 1.5)
                continue;

            // Neighbor is coarser - project its polynomial onto our face nodes
            const VecX &coarse_data = element_data[neigh_e];

            // Extract coarse data on bottom face (k=0) for 2D interpolation
            // Since bathymetry is constant in z, we only need the k=0 layer
            VecX coarse_2d(n1d * n1d);
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx_3d = i + n1d * (j + n1d * 0); // k=0
                    coarse_2d(i + n1d * j) = coarse_data(idx_3d);
                }
            }

            // For each DOF on the shared face (all k values)
            for (int k = 0; k < n1d; ++k) {
                for (int j = 0; j < n1d; ++j) {
                    for (int i = 0; i < n1d; ++i) {
                        // Check if this DOF is on the interface face
                        bool on_face = false;

                        if (face_id == 0 && i == 0)
                            on_face = true;
                        else if (face_id == 1 && i == n1d - 1)
                            on_face = true;
                        else if (face_id == 2 && j == 0)
                            on_face = true;
                        else if (face_id == 3 && j == n1d - 1)
                            on_face = true;

                        if (!on_face)
                            continue;

                        // Get physical coordinates
                        Real xi_ref = lgl_nodes(i);
                        Real eta_ref = lgl_nodes(j);

                        Real phys_x = my_bounds.xmin +
                                      0.5 * (xi_ref + 1.0) * (my_bounds.xmax - my_bounds.xmin);
                        Real phys_y = my_bounds.ymin +
                                      0.5 * (eta_ref + 1.0) * (my_bounds.ymax - my_bounds.ymin);

                        // Transform to coarse reference coordinates
                        Real coarse_xi = 2.0 * (phys_x - neigh_bounds.xmin) /
                                             (neigh_bounds.xmax - neigh_bounds.xmin) -
                                         1.0;
                        Real coarse_eta = 2.0 * (phys_y - neigh_bounds.ymin) /
                                              (neigh_bounds.ymax - neigh_bounds.ymin) -
                                          1.0;

                        coarse_xi = std::max(-1.0, std::min(1.0, coarse_xi));
                        coarse_eta = std::max(-1.0, std::min(1.0, coarse_eta));

                        // Interpolate coarse polynomial
                        Real projected_value =
                            interp.evaluate_scalar_2d(coarse_2d, coarse_xi, coarse_eta);

                        // Overwrite fine element's DOF
                        int idx = i + n1d * (j + n1d * k);
                        element_data[e](idx) = projected_value;
                    }
                }
            }
        }
    }
}

} // namespace drifter
