#include "mesh/seabed_surface.hpp"
#include "bathymetry/adaptive_bathymetry.hpp"
#include "dg/bernstein_basis.hpp"
#include "dg/nonconforming_projection.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>

namespace drifter {

SeabedSurface::SeabedSurface(const OctreeAdapter& mesh, int order,
                             SeabedInterpolation method)
    : mesh_(&mesh), order_(order), method_(method), mesh_zmin_(0.0) {
    identify_bottom_elements();
    allocate_storage();
}

void SeabedSurface::identify_bottom_elements() {
    bottom_elements_.clear();
    mesh_to_seabed_.clear();

    // Find minimum z in the mesh
    mesh_zmin_ = std::numeric_limits<Real>::max();
    const auto& elements = mesh_->elements();
    for (const auto* node : elements) {
        mesh_zmin_ = std::min(mesh_zmin_, node->bounds.zmin);
    }

    // Identify elements whose bottom face is at mesh_zmin (seabed)
    const Real tol = 1e-10 * std::abs(mesh_zmin_);
    for (size_t e = 0; e < elements.size(); ++e) {
        if (std::abs(elements[e]->bounds.zmin - mesh_zmin_) < tol) {
            size_t seabed_idx = bottom_elements_.size();
            bottom_elements_.push_back(static_cast<Index>(e));
            mesh_to_seabed_[static_cast<Index>(e)] = seabed_idx;
        }
    }
}

void SeabedSurface::allocate_storage() {
    size_t n_elem = bottom_elements_.size();
    int n1d = order_ + 1;
    int n2d = n1d * n1d;

    depth_coeffs_.resize(n_elem);
    coordinates_.resize(n_elem);

    for (size_t i = 0; i < n_elem; ++i) {
        depth_coeffs_[i] = VecX::Zero(n2d);
        coordinates_[i] = VecX::Zero(3 * n2d);
    }
}

const SeabedInterpolator& SeabedSurface::get_interpolator() const {
    if (!interpolator_ || interpolator_->order() != order_) {
        interpolator_ = std::make_unique<SeabedInterpolator>(order_, method_);
    }
    return *interpolator_;
}

void SeabedSurface::set_from_bathymetry(const BathymetryData& bathy) {
    const int n1d = order_ + 1;
    const int n2d = n1d * n1d;

    // Get 1D LGL nodes for sampling (from interpolator)
    const SeabedInterpolator& interp = get_interpolator();
    const VecX& lgl_1d = interp.lgl_nodes();

    const auto& elements = mesh_->elements();

    // Sample bathymetry at each bottom element's DOF positions
    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        VecX& depths = depth_coeffs_[s];
        VecX& coords = coordinates_[s];

        // Sample on bottom face using 1D LGL nodes
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int idx_2d = i + n1d * j;

                // Reference coords on bottom face (1D LGL nodes)
                Real xi = lgl_1d(i);
                Real eta = lgl_1d(j);

                // Map to physical coords
                Real x = bounds.xmin + 0.5 * (xi + 1.0) * dx;
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * dy;

                // Get bathymetry depth
                Real h = bathy.get_depth(x, y);
                depths(idx_2d) = h;

                // Store coordinates (z = -h at seabed in sigma coords)
                coords(3 * idx_2d + 0) = x;
                coords(3 * idx_2d + 1) = y;
                coords(3 * idx_2d + 2) = -h;  // z at seabed
            }
        }
    }

    // Apply non-conforming projection for interface continuity
    apply_nonconforming_projection();

    // Update coordinates to match projected coefficients
    update_coordinates_from_coefficients();
}

Real SeabedSurface::sample_smoothed(const BathymetryData& bathy, Real x, Real y,
                                    Real filter_radius) const {
    // Compute pixel size from geotransform
    Real pixel_width = std::abs(bathy.geotransform[1]);
    Real pixel_height = std::abs(bathy.geotransform[5]);
    Real pixel_size = std::min(pixel_width, pixel_height);

    // Convert world radius to pixel radius - NO CAP for large elements
    int kernel_radius = std::max(1, static_cast<int>(filter_radius / pixel_size));

    // Get pixel coordinates of center
    double px_d, py_d;
    bathy.world_to_pixel(x, y, px_d, py_d);
    int px = static_cast<int>(std::round(px_d));
    int py = static_cast<int>(std::round(py_d));

    // Box filter: simple average over kernel
    Real sum = 0.0;
    int count = 0;

    for (int di = -kernel_radius; di <= kernel_radius; ++di) {
        for (int dj = -kernel_radius; dj <= kernel_radius; ++dj) {
            int pi = px + di;
            int pj = py + dj;

            // Check bounds
            if (pi >= 0 && pi < bathy.sizex && pj >= 0 && pj < bathy.sizey) {
                float val = bathy.elevation[pj * bathy.sizex + pi];

                // Skip NoData values
                if (std::abs(val - bathy.nodata_value) > 1e-6f && val < 1e30f) {
                    sum += val;
                    ++count;
                }
            }
        }
    }

    if (count == 0) {
        // Fall back to unsmoothed sampling if no valid pixels
        return bathy.get_depth(x, y);
    }

    // Convert elevation to depth
    Real avg_elevation = sum / count;
    if (bathy.is_depth_positive) {
        return avg_elevation > 0.0 ? avg_elevation : 0.0;
    } else {
        // Negative elevation = water depth
        return avg_elevation < 0.0 ? -avg_elevation : 0.0;
    }
}

void SeabedSurface::set_from_bathymetry_smoothed(const BathymetryData& bathy,
                                                 Real smoothing_factor) {
    const int n1d = order_ + 1;

    // Get 1D LGL nodes for sampling (from interpolator)
    const SeabedInterpolator& interp = get_interpolator();
    const VecX& lgl_1d = interp.lgl_nodes();

    const auto& elements = mesh_->elements();

    // Sample bathymetry at each bottom element's DOF positions with smoothing
    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real filter_radius = smoothing_factor * std::min(dx, dy);

        VecX& depths = depth_coeffs_[s];
        VecX& coords = coordinates_[s];

        // Sample on bottom face using 1D LGL nodes
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int idx_2d = i + n1d * j;

                // Reference coords on bottom face (1D LGL nodes)
                Real xi = lgl_1d(i);
                Real eta = lgl_1d(j);

                // Map to physical coords
                Real x = bounds.xmin + 0.5 * (xi + 1.0) * dx;
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * dy;

                // Get smoothed bathymetry depth
                Real h = sample_smoothed(bathy, x, y, filter_radius);
                depths(idx_2d) = h;

                // Store coordinates (z = -h at seabed in sigma coords)
                coords(3 * idx_2d + 0) = x;
                coords(3 * idx_2d + 1) = y;
                coords(3 * idx_2d + 2) = -h;
            }
        }
    }

    // Apply non-conforming projection for interface continuity
    apply_nonconforming_projection();

    // Update coordinates to match projected coefficients
    update_coordinates_from_coefficients();
}

void SeabedSurface::set_from_adaptive_bathymetry(const AdaptiveBathymetry& adaptive) {
    const int n1d = order_ + 1;

    const auto& elements = mesh_->elements();

    // Project bathymetry onto each bottom element using WENO5 + L2 projection
    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        // Use adaptive projection (WENO5 sampling + L2 projection to Bernstein)
        depth_coeffs_[s] = adaptive.project_element(bounds, order_);
    }

    // Apply Bernstein-aware non-conforming projection for interface continuity
    apply_bernstein_nonconforming_projection();

    // Update coordinates AFTER projection to ensure consistency
    update_coordinates_from_coefficients();
}

void SeabedSurface::update_coordinates_from_coefficients() {
    const int n1d = order_ + 1;
    const auto& elements = mesh_->elements();
    const SeabedInterpolator& interp = get_interpolator();
    const VecX& lgl_1d = interp.lgl_nodes();

    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        VecX& coords = coordinates_[s];

        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int idx_2d = i + n1d * j;

                Real xi = lgl_1d(i);
                Real eta = lgl_1d(j);

                Real x = bounds.xmin + 0.5 * (xi + 1.0) * dx;
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * dy;

                // Evaluate depth from (projected) Bernstein coefficients
                Real h = interp.evaluate_scalar_2d(depth_coeffs_[s], xi, eta);

                coords(3 * idx_2d + 0) = x;
                coords(3 * idx_2d + 1) = y;
                coords(3 * idx_2d + 2) = -h;  // z at seabed
            }
        }
    }
}

void SeabedSurface::set_element_coefficients(size_t seabed_elem_idx, const VecX& coeffs) {
    if (seabed_elem_idx < depth_coeffs_.size()) {
        depth_coeffs_[seabed_elem_idx] = coeffs;
    }
}

void SeabedSurface::apply_nonconforming_projection() {
    // The 2D projection function works with mesh elements, but we only have
    // bottom-layer elements. We need to project between bottom elements that
    // have 2:1 size ratios in the horizontal direction.

    const int n1d = order_ + 1;
    const auto& elements = mesh_->elements();

    // Get interpolator for evaluation
    const SeabedInterpolator& interp = get_interpolator();
    const VecX& lgl_nodes = interp.lgl_nodes();

    // For each fine element, check if any horizontal neighbor is coarser
    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index my_mesh_idx = bottom_elements_[s];
        const auto& my_bounds = elements[my_mesh_idx]->bounds;
        Real my_area = (my_bounds.xmax - my_bounds.xmin) *
                       (my_bounds.ymax - my_bounds.ymin);

        // Check horizontal faces (0: -x, 1: +x, 2: -y, 3: +y)
        for (int face_id = 0; face_id < 4; ++face_id) {
            NeighborInfo info = mesh_->get_neighbor(my_mesh_idx, face_id);
            if (info.is_boundary() || info.neighbor_elements.empty()) continue;

            // Find if the neighbor is a bottom element
            Index neigh_mesh_idx = info.neighbor_elements[0];
            auto neigh_it = mesh_to_seabed_.find(neigh_mesh_idx);
            if (neigh_it == mesh_to_seabed_.end()) continue;  // Neighbor not a bottom element

            size_t neigh_s = neigh_it->second;
            const auto& neigh_bounds = elements[neigh_mesh_idx]->bounds;
            Real neigh_area = (neigh_bounds.xmax - neigh_bounds.xmin) *
                              (neigh_bounds.ymax - neigh_bounds.ymin);

            // Check if neighbor is coarser (larger area = coarser element)
            if (neigh_area <= my_area * 1.5) continue;

            // Neighbor is coarser - project its polynomial onto our face nodes
            const VecX& coarse_data = depth_coeffs_[neigh_s];

            // For each DOF on the shared face
            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    // Check if this DOF is on the interface face
                    bool on_face = false;
                    if (face_id == 0 && i == 0) on_face = true;       // -x face
                    else if (face_id == 1 && i == n1d-1) on_face = true;   // +x face
                    else if (face_id == 2 && j == 0) on_face = true;       // -y face
                    else if (face_id == 3 && j == n1d-1) on_face = true;   // +y face

                    if (!on_face) continue;

                    // Get physical coordinates of this DOF
                    Real xi_ref = lgl_nodes(i);
                    Real eta_ref = lgl_nodes(j);

                    Real phys_x = my_bounds.xmin + 0.5 * (xi_ref + 1.0) *
                                  (my_bounds.xmax - my_bounds.xmin);
                    Real phys_y = my_bounds.ymin + 0.5 * (eta_ref + 1.0) *
                                  (my_bounds.ymax - my_bounds.ymin);

                    // Transform to coarse element's reference coordinates
                    Real coarse_xi = 2.0 * (phys_x - neigh_bounds.xmin) /
                                     (neigh_bounds.xmax - neigh_bounds.xmin) - 1.0;
                    Real coarse_eta = 2.0 * (phys_y - neigh_bounds.ymin) /
                                      (neigh_bounds.ymax - neigh_bounds.ymin) - 1.0;

                    // Clamp to [-1, 1]
                    coarse_xi = std::max(-1.0, std::min(1.0, coarse_xi));
                    coarse_eta = std::max(-1.0, std::min(1.0, coarse_eta));

                    // Interpolate coarse element's polynomial at this point
                    Real projected_value = interp.evaluate_scalar_2d(coarse_data, coarse_xi, coarse_eta);

                    // Overwrite the fine element's DOF value
                    int idx = i + n1d * j;
                    depth_coeffs_[s](idx) = projected_value;
                }
            }
        }
    }
}

void SeabedSurface::apply_bernstein_nonconforming_projection() {
    // For Bernstein basis, edge continuity requires matching edge coefficients.
    //
    // Key property: For a 2D tensor-product Bernstein polynomial
    //   f(ξ, η) = Σᵢⱼ cᵢⱼ · Bᵢ(ξ) · Bⱼ(η)
    //
    // On edge ξ = -1: f(-1, η) = Σⱼ c₀ⱼ · Bⱼ(η)  (only i=0 coefficients matter)
    // On edge ξ = +1: f(+1, η) = Σⱼ cₙⱼ · Bⱼ(η)  (only i=n coefficients matter)
    // On edge η = -1: f(ξ, -1) = Σᵢ cᵢ₀ · Bᵢ(ξ)  (only j=0 coefficients matter)
    // On edge η = +1: f(ξ, +1) = Σᵢ cᵢₙ · Bᵢ(ξ)  (only j=n coefficients matter)
    //
    // For ALL interfaces (both conforming and non-conforming):
    // - Conforming: copy edge coefficients from element with smaller index
    // - Non-conforming: use de Casteljau subdivision to match coarse element's edge

    const int n1d = order_ + 1;
    const auto& elements = mesh_->elements();

    // Get interpolator for evaluation
    const SeabedInterpolator& interp = get_interpolator();
    BernsteinBasis1D basis(order_);

    // Lambda: de Casteljau subdivision - extract Bernstein coeffs for [t0, t1] from [0, 1]
    auto subdivide_bernstein = [&](const VecX& coeffs, Real t0, Real t1) -> VecX {
        // First split at t1 to get [0, t1] portion (left part)
        // Then split at t0/t1 to get [t0, t1] portion
        int n = order_;

        // de Casteljau to split at t1: gives us coefficients for [0, t1]
        std::vector<Real> work(coeffs.data(), coeffs.data() + n1d);
        for (int r = 1; r <= n; ++r) {
            for (int i = 0; i <= n - r; ++i) {
                work[i] = (1.0 - t1) * work[i] + t1 * work[i + 1];
            }
        }
        // Now work[0..n] contains the left part [0, t1], but we need to
        // do this properly to get ALL new coefficients

        // Proper subdivision: after splitting at t, left coeffs are the
        // diagonal of the de Casteljau triangle
        std::vector<std::vector<Real>> triangle(n1d);
        for (int i = 0; i < n1d; ++i) {
            triangle[i].resize(n1d - i);
            triangle[i][0] = coeffs(i);
        }

        // Build de Casteljau triangle for parameter t1
        for (int r = 1; r <= n; ++r) {
            for (int i = 0; i <= n - r; ++i) {
                triangle[i][r] = (1.0 - t1) * triangle[i][r-1] + t1 * triangle[i+1][r-1];
            }
        }

        // Left segment [0, t1] coefficients: diagonal c[i][i]
        VecX left_coeffs(n1d);
        for (int i = 0; i <= n; ++i) {
            left_coeffs(i) = triangle[0][i];
        }

        // If t0 > 0, we need to further subdivide [0, t1] at t0/t1
        if (t0 > 1e-10) {
            Real s = t0 / t1;  // Parameter in [0, 1] for the [0, t1] curve

            // Build de Casteljau triangle for parameter s on left_coeffs
            for (int i = 0; i < n1d; ++i) {
                triangle[i][0] = left_coeffs(i);
            }
            for (int r = 1; r <= n; ++r) {
                for (int i = 0; i <= n - r; ++i) {
                    triangle[i][r] = (1.0 - s) * triangle[i][r-1] + s * triangle[i+1][r-1];
                }
            }

            // Right segment [s, 1] of the [0, t1] curve = [t0, t1] of original
            // Right coefficients: anti-diagonal from top-right to bottom-left
            // triangle[0][n] = evaluated point, triangle[n][0] = last original coeff
            VecX result(n1d);
            for (int i = 0; i <= n; ++i) {
                result(i) = triangle[i][n-i];
            }
            return result;
        }

        return left_coeffs;
    };

    // For each element, check all horizontal neighbors and ensure edge continuity
    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index my_mesh_idx = bottom_elements_[s];
        const auto& my_bounds = elements[my_mesh_idx]->bounds;
        Real my_dx = my_bounds.xmax - my_bounds.xmin;
        Real my_dy = my_bounds.ymax - my_bounds.ymin;

        // Check horizontal faces (0: -x, 1: +x, 2: -y, 3: +y)
        for (int face_id = 0; face_id < 4; ++face_id) {
            NeighborInfo info = mesh_->get_neighbor(my_mesh_idx, face_id);
            if (info.is_boundary() || info.neighbor_elements.empty()) continue;

            Index neigh_mesh_idx = info.neighbor_elements[0];
            auto neigh_it = mesh_to_seabed_.find(neigh_mesh_idx);
            if (neigh_it == mesh_to_seabed_.end()) continue;

            size_t neigh_s = neigh_it->second;
            const auto& neigh_bounds = elements[neigh_mesh_idx]->bounds;
            Real neigh_dx = neigh_bounds.xmax - neigh_bounds.xmin;
            Real neigh_dy = neigh_bounds.ymax - neigh_bounds.ymin;

            // Determine relationship: conforming (same size) or non-conforming (2:1)
            bool is_coarser = false;
            bool is_conforming = false;
            if (face_id == 0 || face_id == 1) {
                // x-face: compare y-size
                is_coarser = neigh_dy > my_dy * 1.5;
                is_conforming = std::abs(neigh_dy - my_dy) < my_dy * 0.1;
            } else {
                // y-face: compare x-size
                is_coarser = neigh_dx > my_dx * 1.5;
                is_conforming = std::abs(neigh_dx - my_dx) < my_dx * 0.1;
            }

            // For conforming interfaces: average edge coefficients from both elements
            // This preserves information from both L2 projections rather than discarding one
            if (is_conforming) {
                // Only process once per edge pair (when s < neigh_s)
                if (s < neigh_s) {
                    VecX& my_coeffs = depth_coeffs_[s];
                    VecX& neigh_coeffs = depth_coeffs_[neigh_s];

                    // Average edge coefficients and assign to both elements
                    if (face_id == 0) {
                        // My -x edge (i=0) matches neighbor's +x edge (i=n)
                        for (int j = 0; j < n1d; ++j) {
                            Real avg = 0.5 * (my_coeffs(0 + n1d * j) + neigh_coeffs(order_ + n1d * j));
                            my_coeffs(0 + n1d * j) = avg;
                            neigh_coeffs(order_ + n1d * j) = avg;
                        }
                    } else if (face_id == 1) {
                        // My +x edge (i=n) matches neighbor's -x edge (i=0)
                        for (int j = 0; j < n1d; ++j) {
                            Real avg = 0.5 * (my_coeffs(order_ + n1d * j) + neigh_coeffs(0 + n1d * j));
                            my_coeffs(order_ + n1d * j) = avg;
                            neigh_coeffs(0 + n1d * j) = avg;
                        }
                    } else if (face_id == 2) {
                        // My -y edge (j=0) matches neighbor's +y edge (j=n)
                        for (int i = 0; i < n1d; ++i) {
                            Real avg = 0.5 * (my_coeffs(i + n1d * 0) + neigh_coeffs(i + n1d * order_));
                            my_coeffs(i + n1d * 0) = avg;
                            neigh_coeffs(i + n1d * order_) = avg;
                        }
                    } else {  // face_id == 3
                        // My +y edge (j=n) matches neighbor's -y edge (j=0)
                        for (int i = 0; i < n1d; ++i) {
                            Real avg = 0.5 * (my_coeffs(i + n1d * order_) + neigh_coeffs(i + n1d * 0));
                            my_coeffs(i + n1d * order_) = avg;
                            neigh_coeffs(i + n1d * 0) = avg;
                        }
                    }
                }
                continue;  // Skip to next face
            }

            // For non-conforming: only process if neighbor is coarser
            if (!is_coarser) continue;

            const VecX& coarse_coeffs = depth_coeffs_[neigh_s];

            // Extract the coarse element's edge polynomial (1D Bernstein coeffs)
            // and compute the fine element's edge coefficients via subdivision
            VecX coarse_edge(n1d);
            Real t0, t1;  // Parameter range on coarse edge that fine edge covers

            if (face_id == 0) {
                // Fine's -x face matches part of coarse's +x face
                // Coarse edge: ξ = +1, η varies -> coeffs c[n][j]
                for (int j = 0; j < n1d; ++j) {
                    coarse_edge(j) = coarse_coeffs(order_ + n1d * j);
                }
                // Fine element's y-range in coarse's η parameter space [0, 1]
                t0 = (my_bounds.ymin - neigh_bounds.ymin) / neigh_dy;
                t1 = (my_bounds.ymax - neigh_bounds.ymin) / neigh_dy;

            } else if (face_id == 1) {
                // Fine's +x face matches part of coarse's -x face
                // Coarse edge: ξ = -1, η varies -> coeffs c[0][j]
                for (int j = 0; j < n1d; ++j) {
                    coarse_edge(j) = coarse_coeffs(0 + n1d * j);
                }
                t0 = (my_bounds.ymin - neigh_bounds.ymin) / neigh_dy;
                t1 = (my_bounds.ymax - neigh_bounds.ymin) / neigh_dy;

            } else if (face_id == 2) {
                // Fine's -y face matches part of coarse's +y face
                // Coarse edge: η = +1, ξ varies -> coeffs c[i][n]
                for (int i = 0; i < n1d; ++i) {
                    coarse_edge(i) = coarse_coeffs(i + n1d * order_);
                }
                t0 = (my_bounds.xmin - neigh_bounds.xmin) / neigh_dx;
                t1 = (my_bounds.xmax - neigh_bounds.xmin) / neigh_dx;

            } else {  // face_id == 3
                // Fine's +y face matches part of coarse's -y face
                // Coarse edge: η = -1, ξ varies -> coeffs c[i][0]
                for (int i = 0; i < n1d; ++i) {
                    coarse_edge(i) = coarse_coeffs(i + n1d * 0);
                }
                t0 = (my_bounds.xmin - neigh_bounds.xmin) / neigh_dx;
                t1 = (my_bounds.xmax - neigh_bounds.xmin) / neigh_dx;
            }

            // Clamp parameter range
            t0 = std::max(0.0, std::min(1.0, t0));
            t1 = std::max(0.0, std::min(1.0, t1));

            // Subdivide to get Bernstein coefficients for [t0, t1] portion
            VecX fine_edge = subdivide_bernstein(coarse_edge, t0, t1);

            // Write subdivided coefficients to fine element's edge DOFs
            VecX& fine_coeffs = depth_coeffs_[s];
            if (face_id == 0) {
                // -x face: i = 0
                for (int j = 0; j < n1d; ++j) {
                    fine_coeffs(0 + n1d * j) = fine_edge(j);
                }
            } else if (face_id == 1) {
                // +x face: i = n
                for (int j = 0; j < n1d; ++j) {
                    fine_coeffs(order_ + n1d * j) = fine_edge(j);
                }
            } else if (face_id == 2) {
                // -y face: j = 0
                for (int i = 0; i < n1d; ++i) {
                    fine_coeffs(i + n1d * 0) = fine_edge(i);
                }
            } else {  // face_id == 3
                // +y face: j = n
                for (int i = 0; i < n1d; ++i) {
                    fine_coeffs(i + n1d * order_) = fine_edge(i);
                }
            }
        }
    }

    // =========================================================================
    // Vertex consistency pass
    // =========================================================================
    // All elements sharing a vertex must have the same corner coefficient.
    // For non-conforming meshes, a fine element's corner may lie on a coarse
    // element's edge midpoint - in that case, the fine corner must match
    // the coarse polynomial evaluated at that point.

    // Corner indices in 2D coefficient array c[i + n1d * j]:
    // (0,0) -> index 0,           corner at (xmin, ymin)
    // (n,0) -> index order_,      corner at (xmax, ymin)
    // (0,n) -> index n1d*order_,  corner at (xmin, ymax)
    // (n,n) -> index order_ + n1d*order_, corner at (xmax, ymax)

    struct VertexKey {
        Real x, y;
        bool operator<(const VertexKey& other) const {
            if (std::abs(x - other.x) > 1e-6) return x < other.x;
            return y < other.y;
        }
    };

    struct VertexInfo {
        size_t elem_idx;
        int corner_idx;
        bool is_edge_midpoint;  // True if this is a coarse element's edge midpoint
    };

    std::map<VertexKey, std::vector<VertexInfo>> vertex_to_elements;

    // First pass: collect all element corners
    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        std::array<std::pair<VertexKey, int>, 4> corners = {{
            {{bounds.xmin, bounds.ymin}, 0},
            {{bounds.xmax, bounds.ymin}, order_},
            {{bounds.xmin, bounds.ymax}, n1d * order_},
            {{bounds.xmax, bounds.ymax}, order_ + n1d * order_}
        }};

        for (const auto& [vkey, corner_idx] : corners) {
            vertex_to_elements[vkey].push_back({s, corner_idx, false});
        }
    }

    // Second pass: for each vertex, check if it lies on a coarse element's edge
    // If so, evaluate the coarse polynomial at that point
    for (auto& [vkey, elem_list] : vertex_to_elements) {
        if (elem_list.size() <= 1) continue;

        // Check each element to see if this vertex lies on its edge (but not corner)
        for (size_t s = 0; s < bottom_elements_.size(); ++s) {
            Index mesh_idx = bottom_elements_[s];
            const auto& bounds = elements[mesh_idx]->bounds;

            const Real tol = 1e-6;
            bool on_xmin = std::abs(vkey.x - bounds.xmin) < tol;
            bool on_xmax = std::abs(vkey.x - bounds.xmax) < tol;
            bool on_ymin = std::abs(vkey.y - bounds.ymin) < tol;
            bool on_ymax = std::abs(vkey.y - bounds.ymax) < tol;
            bool in_x = vkey.x > bounds.xmin + tol && vkey.x < bounds.xmax - tol;
            bool in_y = vkey.y > bounds.ymin + tol && vkey.y < bounds.ymax - tol;

            // Check if vertex is on an edge but not at a corner
            bool on_edge_not_corner = false;
            if ((on_xmin || on_xmax) && in_y) on_edge_not_corner = true;
            if ((on_ymin || on_ymax) && in_x) on_edge_not_corner = true;

            if (on_edge_not_corner) {
                // This element is coarser and the vertex lies on its edge
                // Add it to the list with a special marker
                bool already_in_list = false;
                for (const auto& info : elem_list) {
                    if (info.elem_idx == s) {
                        already_in_list = true;
                        break;
                    }
                }
                if (!already_in_list) {
                    // Use -1 as corner_idx to indicate "evaluate at point"
                    elem_list.push_back({s, -1, true});
                }
            }
        }
    }

    // Third pass: enforce consistency at each vertex
    for (auto& [vkey, elem_list] : vertex_to_elements) {
        if (elem_list.size() <= 1) continue;

        // Compute weighted average, giving coarse edge midpoints higher weight
        // since they've already been processed for edge continuity
        Real sum = 0.0;
        Real weight_sum = 0.0;

        for (const auto& info : elem_list) {
            Real val;
            Real weight;

            if (info.is_edge_midpoint) {
                // Evaluate coarse polynomial at this point
                Index mesh_idx = bottom_elements_[info.elem_idx];
                const auto& bounds = elements[mesh_idx]->bounds;

                Real xi = 2.0 * (vkey.x - bounds.xmin) / (bounds.xmax - bounds.xmin) - 1.0;
                Real eta = 2.0 * (vkey.y - bounds.ymin) / (bounds.ymax - bounds.ymin) - 1.0;

                val = interp.evaluate_scalar_2d(depth_coeffs_[info.elem_idx], xi, eta);
                weight = 2.0;  // Higher weight for coarse element
            } else {
                val = depth_coeffs_[info.elem_idx](info.corner_idx);
                weight = 1.0;
            }

            sum += weight * val;
            weight_sum += weight;
        }

        Real avg = sum / weight_sum;

        // Assign to all fine element corners (not to coarse elements since
        // their edge is already set and changing corners would break edge continuity)
        for (const auto& info : elem_list) {
            if (!info.is_edge_midpoint) {
                depth_coeffs_[info.elem_idx](info.corner_idx) = avg;
            }
        }
    }
}

Real SeabedSurface::depth(Real x, Real y) const {
    Index seabed_idx = find_seabed_element(x, y);
    if (seabed_idx < 0) {
        return 0.0;  // Point not found
    }

    Real xi, eta;
    world_to_reference(static_cast<size_t>(seabed_idx), x, y, xi, eta);

    const SeabedInterpolator& interp = get_interpolator();
    return interp.evaluate_scalar_2d(depth_coeffs_[seabed_idx], xi, eta);
}

bool SeabedSurface::gradient(Real x, Real y, Real& dh_dx, Real& dh_dy) const {
    Index seabed_idx = find_seabed_element(x, y);
    if (seabed_idx < 0) {
        dh_dx = dh_dy = 0.0;
        return false;
    }

    // Use finite differences
    const Real eps = 1e-6;
    Real h_xp = depth(x + eps, y);
    Real h_xm = depth(x - eps, y);
    Real h_yp = depth(x, y + eps);
    Real h_ym = depth(x, y - eps);

    dh_dx = (h_xp - h_xm) / (2.0 * eps);
    dh_dy = (h_yp - h_ym) / (2.0 * eps);
    return true;
}

void SeabedSurface::write_vtk(const std::string& filename, int resolution) const {
    const auto& elements = mesh_->elements();

    // Generate VTK output directly - write bottom face quads
    std::ofstream vtk_file(filename + ".vtu");

    // Collect points and cells
    std::vector<Vec3> all_points;
    std::vector<Real> all_depths;
    std::vector<std::array<size_t, 4>> all_quads;

    const SeabedInterpolator& interp = get_interpolator();

    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        size_t base_pt = all_points.size();

        // Generate (resolution+1) x (resolution+1) grid of points
        for (int j = 0; j <= resolution; ++j) {
            for (int i = 0; i <= resolution; ++i) {
                Real xi = -1.0 + 2.0 * i / resolution;
                Real eta = -1.0 + 2.0 * j / resolution;

                Real x = bounds.xmin + 0.5 * (xi + 1.0) * dx;
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * dy;

                Real h = interp.evaluate_scalar_2d(depth_coeffs_[s], xi, eta);

                all_points.push_back(Vec3(x, y, -h));
                all_depths.push_back(h);
            }
        }

        // Generate quads
        for (int j = 0; j < resolution; ++j) {
            for (int i = 0; i < resolution; ++i) {
                size_t p00 = base_pt + i + (resolution + 1) * j;
                size_t p10 = base_pt + (i + 1) + (resolution + 1) * j;
                size_t p01 = base_pt + i + (resolution + 1) * (j + 1);
                size_t p11 = base_pt + (i + 1) + (resolution + 1) * (j + 1);

                all_quads.push_back({p00, p10, p11, p01});
            }
        }
    }

    // Write VTU file
    vtk_file << "<?xml version=\"1.0\"?>\n";
    vtk_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtk_file << "  <UnstructuredGrid>\n";
    vtk_file << "    <Piece NumberOfPoints=\"" << all_points.size()
             << "\" NumberOfCells=\"" << all_quads.size() << "\">\n";

    // Points
    vtk_file << "      <Points>\n";
    vtk_file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& p : all_points) {
        vtk_file << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Points>\n";

    // Cells
    vtk_file << "      <Cells>\n";
    vtk_file << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& q : all_quads) {
        vtk_file << "          " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 0; i < all_quads.size(); ++i) {
        vtk_file << "          " << (i + 1) * 4 << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t i = 0; i < all_quads.size(); ++i) {
        vtk_file << "          9\n";  // VTK_QUAD
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Cells>\n";

    // Point data
    vtk_file << "      <PointData Scalars=\"depth\">\n";
    vtk_file << "        <DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";
    for (Real d : all_depths) {
        vtk_file << "          " << d << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </PointData>\n";

    vtk_file << "    </Piece>\n";
    vtk_file << "  </UnstructuredGrid>\n";
    vtk_file << "</VTKFile>\n";
}

// =========================================================================
// AMR Dynamic Updates
// =========================================================================

void SeabedSurface::on_refine(Index parent_mesh_idx, const std::vector<Index>& child_mesh_indices) {
    auto it = mesh_to_seabed_.find(parent_mesh_idx);
    if (it == mesh_to_seabed_.end()) {
        return;  // Parent wasn't a bottom element
    }

    size_t parent_seabed_idx = it->second;
    const VecX& parent_coeffs = depth_coeffs_[parent_seabed_idx];

    const auto& elements = mesh_->elements();
    const int n1d = order_ + 1;

    const SeabedInterpolator& interp = get_interpolator();
    const VecX& lgl_1d = interp.lgl_nodes();

    // For each child that is also a bottom element
    for (Index child_mesh_idx : child_mesh_indices) {
        // Check if child is in bottom layer
        if (std::abs(elements[child_mesh_idx]->bounds.zmin - mesh_zmin_) > 1e-10) {
            continue;
        }

        // Add to bottom elements if new
        if (mesh_to_seabed_.find(child_mesh_idx) == mesh_to_seabed_.end()) {
            size_t new_seabed_idx = bottom_elements_.size();
            bottom_elements_.push_back(child_mesh_idx);
            mesh_to_seabed_[child_mesh_idx] = new_seabed_idx;
            depth_coeffs_.push_back(VecX::Zero(n1d * n1d));
            coordinates_.push_back(VecX::Zero(3 * n1d * n1d));
        }

        size_t child_seabed_idx = mesh_to_seabed_[child_mesh_idx];
        const auto& child_bounds = elements[child_mesh_idx]->bounds;
        const auto& parent_bounds = elements[parent_mesh_idx]->bounds;

        VecX& child_coeffs = depth_coeffs_[child_seabed_idx];
        VecX& child_coords = coordinates_[child_seabed_idx];

        Real child_dx = child_bounds.xmax - child_bounds.xmin;
        Real child_dy = child_bounds.ymax - child_bounds.ymin;
        Real parent_dx = parent_bounds.xmax - parent_bounds.xmin;
        Real parent_dy = parent_bounds.ymax - parent_bounds.ymin;

        // Interpolate parent's polynomial at child's DOF locations
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int idx_2d = i + n1d * j;

                // Child's DOF in child's reference space (1D LGL nodes)
                Real xi_child = lgl_1d(i);
                Real eta_child = lgl_1d(j);

                // Map to physical
                Real x = child_bounds.xmin + 0.5 * (xi_child + 1.0) * child_dx;
                Real y = child_bounds.ymin + 0.5 * (eta_child + 1.0) * child_dy;

                // Map to parent's reference space
                Real xi_parent = 2.0 * (x - parent_bounds.xmin) / parent_dx - 1.0;
                Real eta_parent = 2.0 * (y - parent_bounds.ymin) / parent_dy - 1.0;

                // Clamp to valid range
                xi_parent = std::max(-1.0, std::min(1.0, xi_parent));
                eta_parent = std::max(-1.0, std::min(1.0, eta_parent));

                // Evaluate parent's polynomial
                Real h = interp.evaluate_scalar_2d(parent_coeffs, xi_parent, eta_parent);
                child_coeffs(idx_2d) = h;

                child_coords(3 * idx_2d + 0) = x;
                child_coords(3 * idx_2d + 1) = y;
                child_coords(3 * idx_2d + 2) = -h;
            }
        }
    }

    // Remove parent from bottom elements
    // (Note: in practice, parent becomes non-leaf, so we might handle this differently)
    // For now, we keep it but it won't be used since it's no longer a leaf
}

void SeabedSurface::on_coarsen(const std::vector<Index>& child_mesh_indices, Index new_parent_mesh_idx) {
    // Average children's coefficients to create parent
    const int n1d = order_ + 1;
    const int n2d = n1d * n1d;

    VecX parent_coeffs = VecX::Zero(n2d);
    int count = 0;

    for (Index child_mesh_idx : child_mesh_indices) {
        auto it = mesh_to_seabed_.find(child_mesh_idx);
        if (it != mesh_to_seabed_.end()) {
            parent_coeffs += depth_coeffs_[it->second];
            ++count;
        }
    }

    if (count > 0) {
        parent_coeffs /= static_cast<Real>(count);
    }

    // Add parent to bottom elements if at bottom layer
    const auto& elements = mesh_->elements();
    if (std::abs(elements[new_parent_mesh_idx]->bounds.zmin - mesh_zmin_) < 1e-10) {
        if (mesh_to_seabed_.find(new_parent_mesh_idx) == mesh_to_seabed_.end()) {
            size_t new_seabed_idx = bottom_elements_.size();
            bottom_elements_.push_back(new_parent_mesh_idx);
            mesh_to_seabed_[new_parent_mesh_idx] = new_seabed_idx;
            depth_coeffs_.push_back(parent_coeffs);
            coordinates_.push_back(VecX::Zero(3 * n2d));

            // Compute coordinates
            const auto& bounds = elements[new_parent_mesh_idx]->bounds;
            Real dx = bounds.xmax - bounds.xmin;
            Real dy = bounds.ymax - bounds.ymin;

            const SeabedInterpolator& interp = get_interpolator();
            const VecX& lgl_1d = interp.lgl_nodes();
            VecX& coords = coordinates_.back();

            for (int j = 0; j < n1d; ++j) {
                for (int i = 0; i < n1d; ++i) {
                    int idx_2d = i + n1d * j;
                    Real xi = lgl_1d(i);
                    Real eta = lgl_1d(j);
                    Real x = bounds.xmin + 0.5 * (xi + 1.0) * dx;
                    Real y = bounds.ymin + 0.5 * (eta + 1.0) * dy;
                    Real h = parent_coeffs(idx_2d);

                    coords(3 * idx_2d + 0) = x;
                    coords(3 * idx_2d + 1) = y;
                    coords(3 * idx_2d + 2) = -h;
                }
            }
        } else {
            size_t seabed_idx = mesh_to_seabed_[new_parent_mesh_idx];
            depth_coeffs_[seabed_idx] = parent_coeffs;
        }
    }

    // Note: Children removal is complex as it changes indices
    // For simplicity, call rebuild_from_mesh() after coarsening
}

void SeabedSurface::rebuild_from_mesh() {
    identify_bottom_elements();
    allocate_storage();
    interpolator_.reset();
}

// =========================================================================
// Accessors
// =========================================================================

Index SeabedSurface::seabed_element_index(Index mesh_idx) const {
    auto it = mesh_to_seabed_.find(mesh_idx);
    return (it != mesh_to_seabed_.end()) ? static_cast<Index>(it->second) : -1;
}

bool SeabedSurface::is_bottom_element(Index mesh_idx) const {
    return mesh_to_seabed_.find(mesh_idx) != mesh_to_seabed_.end();
}

Index SeabedSurface::find_seabed_element(Real x, Real y) const {
    // Linear search through bottom elements
    // TODO: Could use spatial indexing (R-tree) for large meshes
    const auto& elements = mesh_->elements();

    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        if (x >= bounds.xmin && x <= bounds.xmax &&
            y >= bounds.ymin && y <= bounds.ymax) {
            return static_cast<Index>(s);
        }
    }

    return -1;
}

void SeabedSurface::world_to_reference(size_t seabed_idx, Real x, Real y,
                                        Real& xi, Real& eta) const {
    Index mesh_idx = bottom_elements_[seabed_idx];
    const auto& bounds = mesh_->elements()[mesh_idx]->bounds;

    xi = 2.0 * (x - bounds.xmin) / (bounds.xmax - bounds.xmin) - 1.0;
    eta = 2.0 * (y - bounds.ymin) / (bounds.ymax - bounds.ymin) - 1.0;

    // Clamp to valid range
    xi = std::max(-1.0, std::min(1.0, xi));
    eta = std::max(-1.0, std::min(1.0, eta));
}

}  // namespace drifter
