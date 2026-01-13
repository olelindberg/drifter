#include "bathymetry/cg_bathymetry_smoother.hpp"
#include "mesh/octree_adapter.hpp"
#include <stdexcept>

namespace drifter {

CGBathymetrySmoother::CGBathymetrySmoother(const OctreeAdapter& octree,
                                           Real alpha, Real beta)
    : alpha_(alpha), beta_(beta) {

    // Create 2D mesh from octree bottom face
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();

    init_components();
}

CGBathymetrySmoother::CGBathymetrySmoother(const QuadtreeAdapter& mesh,
                                           Real alpha, Real beta)
    : quadtree_(&mesh), alpha_(alpha), beta_(beta) {

    init_components();
}

void CGBathymetrySmoother::init_components() {
    if (quadtree_->num_elements() == 0) {
        throw std::runtime_error("CGBathymetrySmoother: empty mesh");
    }

    // Create DOF manager
    dof_manager_ = std::make_unique<CGDofManager>(*quadtree_, basis_);

    // Create assembler
    assembler_ = std::make_unique<BiharmonicAssembler>(
        *quadtree_, basis_, *dof_manager_, alpha_, beta_);
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
    VecX phi = basis_.evaluate(xi, eta);

    // Get element DOFs
    const auto& elem_dofs = dof_manager_->element_dofs(elem);

    // Interpolate solution
    Real value = 0.0;
    for (int i = 0; i < QuinticBasis2D::NDOF; ++i) {
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
    MatX grad_ref = basis_.evaluate_gradient(xi, eta);  // 36 x 2

    // Get element DOFs
    const auto& elem_dofs = dof_manager_->element_dofs(elem);

    // Interpolate gradient
    Vec2 grad_ref_sum(0.0, 0.0);
    for (int i = 0; i < QuinticBasis2D::NDOF; ++i) {
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

}  // namespace drifter
