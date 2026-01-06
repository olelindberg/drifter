#include "mesh/seabed_surface.hpp"
#include "dg/nonconforming_projection.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

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
}

void SeabedSurface::set_element_coefficients(size_t seabed_elem_idx, const VecX& coeffs) {
    if (seabed_elem_idx < depth_coeffs_.size()) {
        depth_coeffs_[seabed_elem_idx] = coeffs;
    }
}

void SeabedSurface::apply_nonconforming_projection() {
    // Use the existing projection function
    // But we need to adapt since it expects mesh element indexing
    // We'll create a temporary vector indexed by mesh elements

    const auto& elements = mesh_->elements();
    size_t num_mesh_elements = elements.size();

    // Create mesh-indexed depth data (zeros for non-bottom elements)
    std::vector<VecX> mesh_indexed_depths(num_mesh_elements);
    int n2d = (order_ + 1) * (order_ + 1);

    for (size_t e = 0; e < num_mesh_elements; ++e) {
        auto it = mesh_to_seabed_.find(static_cast<Index>(e));
        if (it != mesh_to_seabed_.end()) {
            mesh_indexed_depths[e] = depth_coeffs_[it->second];
        } else {
            mesh_indexed_depths[e] = VecX::Zero(n2d);
        }
    }

    // Apply projection (modifies in-place)
    project_coarse_to_fine_2d(*mesh_, mesh_indexed_depths, order_, method_);

    // Copy back to our storage
    for (size_t s = 0; s < bottom_elements_.size(); ++s) {
        Index mesh_idx = bottom_elements_[s];
        depth_coeffs_[s] = mesh_indexed_depths[mesh_idx];
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
