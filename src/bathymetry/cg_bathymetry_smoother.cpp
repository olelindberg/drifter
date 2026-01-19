#include "bathymetry/cg_bathymetry_smoother.hpp"
#include "mesh/octree_adapter.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace drifter {

CGBathymetrySmoother::CGBathymetrySmoother(const OctreeAdapter& octree,
                                           Real alpha, Real beta,
                                           int order, Real penalty)
    : alpha_(alpha), beta_(beta), order_(order), penalty_(penalty) {

    // Create 2D mesh from octree bottom face
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();

    init_components();
}

CGBathymetrySmoother::CGBathymetrySmoother(const QuadtreeAdapter& mesh,
                                           Real alpha, Real beta,
                                           int order, Real penalty)
    : quadtree_(&mesh), alpha_(alpha), beta_(beta), order_(order), penalty_(penalty) {

    init_components();
}

void CGBathymetrySmoother::init_components() {
    if (quadtree_->num_elements() == 0) {
        throw std::runtime_error("CGBathymetrySmoother: empty mesh");
    }

    // Create basis with specified order
    basis_ = std::make_unique<LagrangeBasis2D>(order_);

    // Create DOF manager
    dof_manager_ = std::make_unique<CGDofManager>(*quadtree_, *basis_);

    // Create assembler
    assembler_ = std::make_unique<BiharmonicAssembler>(
        *quadtree_, *basis_, *dof_manager_, alpha_, beta_, penalty_);
}

void CGBathymetrySmoother::set_bathymetry_data(
    std::function<Real(Real, Real)> bathy_func) {
    bathy_ = std::make_unique<FunctionBathymetry>(std::move(bathy_func));
    solved_ = false;
}

void CGBathymetrySmoother::set_bathymetry_data(
    std::unique_ptr<BathymetrySource> bathy) {
    bathy_ = std::move(bathy);
    solved_ = false;
}

void CGBathymetrySmoother::solve() {
    if (!bathy_) {
        throw std::runtime_error("CGBathymetrySmoother::solve: no bathymetry data set");
    }

    // Use natural (Neumann) boundary conditions: ∂u/∂n = 0, ∂²u/∂n² = 0
    // This allows the boundary to evolve naturally based on the smoothing
    // rather than being pinned to specific values.
    // The weak form of the biharmonic equation naturally imposes these conditions.
    // The IPDG penalty enforces C¹ continuity at element interfaces.

    // Assemble reduced system (with constraint elimination)
    SpMat K_red;
    VecX f_red;
    assembler_->assemble_reduced_system(*bathy_, K_red, f_red);

    // Solve using direct solver
    Eigen::SparseLU<SpMat> solver;
    solver.compute(K_red);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGBathymetrySmoother::solve: matrix factorization failed");
    }

    solution_free_ = solver.solve(f_red);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGBathymetrySmoother::solve: solve failed");
    }

    // Expand to global DOFs
    solution_global_ = dof_manager_->expand_solution(solution_free_);

    solved_ = true;
}

Real CGBathymetrySmoother::solution_at_dof(Index dof) const {
    if (!solved_) {
        throw std::runtime_error("CGBathymetrySmoother: not solved");
    }
    if (dof < 0 || dof >= solution_global_.size()) {
        throw std::out_of_range("CGBathymetrySmoother: DOF index out of range");
    }
    return solution_global_(dof);
}

Index CGBathymetrySmoother::find_element(Real x, Real y) const {
    return quadtree_->find_element(Vec2(x, y));
}

Real CGBathymetrySmoother::evaluate(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("CGBathymetrySmoother::evaluate: not solved");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        // Point outside domain - return NaN or extrapolate
        return std::numeric_limits<Real>::quiet_NaN();
    }

    return evaluate_in_element(elem, x, y);
}

Real CGBathymetrySmoother::evaluate_in_element(Index elem, Real x, Real y) const {
    const auto& bounds = quadtree_->element_bounds(elem);

    // Map to reference coordinates [-1, 1]
    Real xi = 2.0 * (x - bounds.xmin) / (bounds.xmax - bounds.xmin) - 1.0;
    Real eta = 2.0 * (y - bounds.ymin) / (bounds.ymax - bounds.ymin) - 1.0;

    // Evaluate basis functions
    VecX phi = basis_->evaluate(xi, eta);

    // Get element DOFs
    const auto& elem_dofs = dof_manager_->element_dofs(elem);
    int ndof = basis_->num_dofs();

    // Interpolate solution
    Real value = 0.0;
    for (int i = 0; i < ndof; ++i) {
        value += solution_global_(elem_dofs[i]) * phi(i);
    }

    return value;
}

Vec2 CGBathymetrySmoother::evaluate_gradient(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("CGBathymetrySmoother::evaluate_gradient: not solved");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        return Vec2(std::numeric_limits<Real>::quiet_NaN(),
                    std::numeric_limits<Real>::quiet_NaN());
    }

    return evaluate_gradient_in_element(elem, x, y);
}

Vec2 CGBathymetrySmoother::evaluate_gradient_in_element(Index elem, Real x, Real y) const {
    const auto& bounds = quadtree_->element_bounds(elem);

    // Map to reference coordinates
    Real hx = 0.5 * (bounds.xmax - bounds.xmin);
    Real hy = 0.5 * (bounds.ymax - bounds.ymin);
    Real xi = (x - bounds.xmin) / hx - 1.0;
    Real eta = (y - bounds.ymin) / hy - 1.0;

    // Evaluate gradient of basis functions (in reference coords)
    MatX grad_ref = basis_->evaluate_gradient(xi, eta);  // ndof x 2

    // Get element DOFs
    const auto& elem_dofs = dof_manager_->element_dofs(elem);
    int ndof = basis_->num_dofs();

    // Interpolate gradient
    Vec2 grad_ref_sum(0.0, 0.0);
    for (int i = 0; i < ndof; ++i) {
        grad_ref_sum(0) += solution_global_(elem_dofs[i]) * grad_ref(i, 0);
        grad_ref_sum(1) += solution_global_(elem_dofs[i]) * grad_ref(i, 1);
    }

    // Transform to physical coordinates
    // dx/dxi = hx, dy/deta = hy
    // du/dx = (du/dxi) / (dx/dxi) = du/dxi / hx
    return Vec2(grad_ref_sum(0) / hx, grad_ref_sum(1) / hy);
}

void CGBathymetrySmoother::transfer_to_seabed(SeabedSurface& seabed) const {
    if (!solved_) {
        throw std::runtime_error("CGBathymetrySmoother::transfer_to_seabed: not solved");
    }

    // Get the DG mesh from seabed
    const auto& dg_mesh = seabed.mesh();
    int dg_order = seabed.order();

    // For each seabed element, sample the CG solution at DG DOF positions
    for (Index s = 0; s < seabed.num_elements(); ++s) {
        Index elem_3d = seabed.mesh_element_index(s);
        const auto& bounds_3d = dg_mesh.element_bounds(elem_3d);

        // Get the DG LGL nodes
        int n1d = dg_order + 1;
        int n2d = n1d * n1d;

        VecX coeffs(n2d);

        // Sample CG solution at each DG node position on the bottom face
        // DG nodes are at LGL positions in reference coordinates
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                // Get physical position of DG node
                // Assuming LGL nodes span [-1, 1] uniformly (for now, simplified)
                Real xi = -1.0 + 2.0 * i / dg_order;
                Real eta = -1.0 + 2.0 * j / dg_order;

                Real x = bounds_3d.xmin + 0.5 * (xi + 1.0) * (bounds_3d.xmax - bounds_3d.xmin);
                Real y = bounds_3d.ymin + 0.5 * (eta + 1.0) * (bounds_3d.ymax - bounds_3d.ymin);

                // Evaluate CG solution
                Real depth = evaluate(x, y);

                // Handle NaN (point outside CG mesh)
                if (std::isnan(depth)) {
                    // Fall back to original bathymetry data
                    if (bathy_) {
                        depth = bathy_->evaluate(x, y);
                    } else {
                        depth = 0.0;
                    }
                }

                int dof = i + j * n1d;
                coeffs(dof) = depth;
            }
        }

        seabed.set_element_coefficients(s, coeffs);
    }
}

void CGBathymetrySmoother::apply_corner_dirichlet() {
    // Apply Dirichlet constraints at the 4 domain corners
    // This pins corner values to the input bathymetry data
    const auto& domain = quadtree_->domain_bounds();
    std::vector<std::pair<Real, Real>> corner_positions = {
        {domain.xmin, domain.ymin},
        {domain.xmax, domain.ymin},
        {domain.xmin, domain.ymax},
        {domain.xmax, domain.ymax}
    };

    std::vector<Real> corner_values;
    for (const auto& [x, y] : corner_positions) {
        corner_values.push_back(bathy_->evaluate(x, y));
    }

    dof_manager_->apply_corner_dirichlet(corner_values);
}

void CGBathymetrySmoother::apply_boundary_dirichlet() {
    // Apply Dirichlet constraints on all boundary DOFs to prevent blowup
    // This pins the boundary values to the input bathymetry data

    const auto& boundary_dofs = dof_manager_->boundary_dofs();
    int n1d = basis_->num_nodes_1d();

    // Get positions for each boundary DOF and evaluate bathymetry
    for (Index dof : boundary_dofs) {
        // Find the position of this DOF
        // We need to iterate through elements to find where this DOF is located
        bool found = false;
        for (Index e = 0; e < quadtree_->num_elements() && !found; ++e) {
            const auto& elem_dofs = dof_manager_->element_dofs(e);
            const auto& bounds = quadtree_->element_bounds(e);

            for (int local = 0; local < static_cast<int>(elem_dofs.size()); ++local) {
                if (elem_dofs[local] == dof) {
                    // Found the element containing this DOF
                    // Convert local DOF index to (i, j) and then to physical position
                    int i = local % n1d;
                    int j = local / n1d;

                    // Get reference coordinates from basis nodes
                    const VecX& nodes = basis_->nodes_1d();
                    Real xi = nodes(i);
                    Real eta = nodes(j);

                    // Map to physical coordinates
                    Real x = bounds.xmin + 0.5 * (xi + 1.0) * (bounds.xmax - bounds.xmin);
                    Real y = bounds.ymin + 0.5 * (eta + 1.0) * (bounds.ymax - bounds.ymin);

                    // Get bathymetry value at this position
                    Real value = bathy_->evaluate(x, y);

                    // Apply as Dirichlet constraint
                    dof_manager_->apply_single_dirichlet(dof, value);

                    found = true;
                    break;
                }
            }
        }
    }
}

void CGBathymetrySmoother::write_vtk(const std::string& filename, int resolution) const {
    if (!solved_) {
        throw std::runtime_error("CGBathymetrySmoother::write_vtk: not solved");
    }

    // Generate high-resolution surface mesh by sampling CG solution
    // Each quadtree element is subdivided into resolution x resolution quads

    Index num_elements = quadtree_->num_elements();
    Index points_per_elem = (resolution + 1) * (resolution + 1);
    Index quads_per_elem = resolution * resolution;

    Index total_points = num_elements * points_per_elem;
    Index total_quads = num_elements * quads_per_elem;

    // Collect points and depths
    std::vector<Vec3> points;
    std::vector<Real> depths;
    std::vector<Real> raw_depths;
    std::vector<Index> element_ids;

    points.reserve(total_points);
    depths.reserve(total_points);
    raw_depths.reserve(total_points);

    for (Index e = 0; e < num_elements; ++e) {
        const auto& bounds = quadtree_->element_bounds(e);
        Real hx = bounds.xmax - bounds.xmin;
        Real hy = bounds.ymax - bounds.ymin;

        for (int j = 0; j <= resolution; ++j) {
            for (int i = 0; i <= resolution; ++i) {
                // Reference coordinates
                Real xi = -1.0 + 2.0 * i / resolution;
                Real eta = -1.0 + 2.0 * j / resolution;

                // Physical coordinates
                Real x = bounds.xmin + 0.5 * (xi + 1.0) * hx;
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * hy;

                // Evaluate CG solution using basis
                Real depth = evaluate_in_element(e, x, y);

                // Raw bathymetry for comparison
                Real raw = bathy_ ? bathy_->evaluate(x, y) : 0.0;

                points.emplace_back(x, y, -depth);  // z = -depth for visualization
                depths.push_back(depth);
                raw_depths.push_back(raw);
            }
        }
    }

    // Write VTU file
    std::string vtk_path = filename + ".vtu";
    std::ofstream file(vtk_path);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << total_points
         << "\" NumberOfCells=\"" << total_quads << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    file << std::setprecision(12);
    for (const auto& p : points) {
        file << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Cells (quads)
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index e = 0; e < num_elements; ++e) {
        Index base = e * points_per_elem;
        int n1d = resolution + 1;

        for (int j = 0; j < resolution; ++j) {
            for (int i = 0; i < resolution; ++i) {
                Index p0 = base + i + j * n1d;
                Index p1 = base + (i + 1) + j * n1d;
                Index p2 = base + (i + 1) + (j + 1) * n1d;
                Index p3 = base + i + (j + 1) * n1d;
                file << "          " << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (Index i = 1; i <= total_quads; ++i) {
        file << "          " << (i * 4) << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index i = 0; i < total_quads; ++i) {
        file << "          9\n";  // VTK_QUAD = 9
    }
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    // Point data
    file << "      <PointData Scalars=\"depth\">\n";

    // Smoothed depth
    file << "        <DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";
    for (Real d : depths) {
        file << "          " << d << "\n";
    }
    file << "        </DataArray>\n";

    // Raw bathymetry
    file << "        <DataArray type=\"Float64\" Name=\"raw_bathy\" format=\"ascii\">\n";
    for (Real d : raw_depths) {
        file << "          " << d << "\n";
    }
    file << "        </DataArray>\n";

    // Difference
    file << "        <DataArray type=\"Float64\" Name=\"difference\" format=\"ascii\">\n";
    for (size_t i = 0; i < depths.size(); ++i) {
        file << "          " << (depths[i] - raw_depths[i]) << "\n";
    }
    file << "        </DataArray>\n";

    file << "      </PointData>\n";

    // Cell data (element ID)
    file << "      <CellData>\n";
    file << "        <DataArray type=\"Int64\" Name=\"element_id\" format=\"ascii\">\n";
    for (Index e = 0; e < num_elements; ++e) {
        for (Index q = 0; q < quads_per_elem; ++q) {
            file << "          " << e << "\n";
        }
    }
    file << "        </DataArray>\n";
    file << "      </CellData>\n";

    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

}  // namespace drifter
