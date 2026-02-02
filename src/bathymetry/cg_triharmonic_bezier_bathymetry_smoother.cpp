#include "bathymetry/cg_triharmonic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"  // For BathymetrySource, BathymetryPoint
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace drifter {

// =============================================================================
// Gauss-Legendre quadrature points and weights
// =============================================================================

namespace {

// Gauss-Legendre points and weights on [0, 1]
void gauss_legendre_01(int n, std::vector<Real>& pts, std::vector<Real>& wts) {
    pts.resize(n);
    wts.resize(n);

    if (n == 1) {
        pts[0] = 0.5;
        wts[0] = 1.0;
    } else if (n == 2) {
        pts[0] = 0.5 - 0.5 / std::sqrt(3.0);
        pts[1] = 0.5 + 0.5 / std::sqrt(3.0);
        wts[0] = wts[1] = 0.5;
    } else if (n == 3) {
        pts[0] = 0.5 - 0.5 * std::sqrt(0.6);
        pts[1] = 0.5;
        pts[2] = 0.5 + 0.5 * std::sqrt(0.6);
        wts[0] = wts[2] = 5.0 / 18.0;
        wts[1] = 8.0 / 18.0;
    } else if (n == 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        pts[0] = 0.5 * (1.0 - b);
        pts[1] = 0.5 * (1.0 - a);
        pts[2] = 0.5 * (1.0 + a);
        pts[3] = 0.5 * (1.0 + b);
        Real wa = (18.0 + std::sqrt(30.0)) / 72.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 72.0;
        wts[0] = wts[3] = 0.5 * wb;
        wts[1] = wts[2] = 0.5 * wa;
    } else if (n == 5) {
        pts[0] = 0.5 * (1.0 - std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0);
        pts[1] = 0.5 * (1.0 - std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0);
        pts[2] = 0.5;
        pts[3] = 0.5 * (1.0 + std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0);
        pts[4] = 0.5 * (1.0 + std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0);
        wts[0] = wts[4] = 0.5 * (322.0 - 13.0 * std::sqrt(70.0)) / 900.0;
        wts[1] = wts[3] = 0.5 * (322.0 + 13.0 * std::sqrt(70.0)) / 900.0;
        wts[2] = 0.5 * 128.0 / 225.0;
    } else {
        // Default to 6-point quadrature
        pts = {0.03376524289842, 0.16939530676687, 0.38069040695840,
               0.61930959304160, 0.83060469323313, 0.96623475710158};
        wts = {0.08566224618959, 0.18038078652407, 0.23395696728635,
               0.23395696728635, 0.18038078652407, 0.08566224618959};
        for (auto& w : wts) w *= 0.5;
    }
}

}  // anonymous namespace

// =============================================================================
// Construction
// =============================================================================

CGTriharmonicBezierBathymetrySmoother::CGTriharmonicBezierBathymetrySmoother(
    const QuadtreeAdapter& mesh, const CGTriharmonicBezierSmootherConfig& config)
    : quadtree_(&mesh), config_(config) {
    init_components();
}

CGTriharmonicBezierBathymetrySmoother::CGTriharmonicBezierBathymetrySmoother(
    const OctreeAdapter& octree, const CGTriharmonicBezierSmootherConfig& config)
    : config_(config) {
    // Create QuadtreeAdapter from octree bottom face
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();
    init_components();
}

void CGTriharmonicBezierBathymetrySmoother::init_components() {
    basis_ = std::make_unique<BezierBasis2D>();
    triharmonic_hessian_ = std::make_unique<TriharmonicHessian>(
        config_.ngauss_energy, config_.gradient_weight);
    dof_manager_ = std::make_unique<CGBezierDofManager>(*quadtree_);

    // Build edge derivative constraints if enabled (no vertex constraints)
    if (config_.enable_edge_constraints) {
        dof_manager_->build_edge_derivative_constraints(config_.edge_ngauss);
    }

    // Initialize solution vector
    solution_.setZero(dof_manager_->num_global_dofs());
}

// =============================================================================
// Data input
// =============================================================================

void CGTriharmonicBezierBathymetrySmoother::set_bathymetry_data(
    const BathymetrySource& source) {
    set_bathymetry_data([&source](Real x, Real y) { return source.evaluate(x, y); });
}

void CGTriharmonicBezierBathymetrySmoother::set_bathymetry_data(
    std::function<Real(Real, Real)> bathy_func) {
    assemble_triharmonic_hessian();
    assemble_data_fitting(bathy_func);
    data_set_ = true;
}

void CGTriharmonicBezierBathymetrySmoother::set_scattered_points(
    const std::vector<Vec3>& points) {
    std::vector<BathymetryPoint> bathy_points;
    bathy_points.reserve(points.size());
    for (const auto& p : points) {
        bathy_points.emplace_back(p(0), p(1), p(2), 1.0);
    }
    set_scattered_points(bathy_points);
}

void CGTriharmonicBezierBathymetrySmoother::set_scattered_points(
    const std::vector<BathymetryPoint>& points) {
    // Build a function that returns nearest point value
    std::vector<BathymetryPoint> pts = points;

    auto bathy_func = [pts](Real x, Real y) -> Real {
        Real min_dist = std::numeric_limits<Real>::max();
        Real value = 0.0;
        for (const auto& p : pts) {
            Real dx = x - p.x;
            Real dy = y - p.y;
            Real dist = dx * dx + dy * dy;
            if (dist < min_dist) {
                min_dist = dist;
                value = p.z;
            }
        }
        return value;
    };

    set_bathymetry_data(bathy_func);
}

// =============================================================================
// Assembly
// =============================================================================

void CGTriharmonicBezierBathymetrySmoother::assemble_triharmonic_hessian() {
    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_elements = quadtree_->num_elements();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * 36 * 36);

    for (Index elem = 0; elem < num_elements; ++elem) {
        // Get element size for scaling
        Vec2 size = quadtree_->element_size(elem);
        Real dx = size(0);
        Real dy = size(1);

        // Compute scaled local triharmonic Hessian
        MatX H_local = triharmonic_hessian_->scaled_hessian(dx, dy);

        // Get global DOF indices for this element
        const auto& global_dofs = dof_manager_->element_dofs(elem);

        // Assemble into global matrix with CG connectivity
        for (int i = 0; i < BezierBasis2D::NDOF; ++i) {
            Index I = global_dofs[i];
            for (int j = 0; j < BezierBasis2D::NDOF; ++j) {
                Index J = global_dofs[j];
                if (std::abs(H_local(i, j)) > 1e-16) {
                    triplets.emplace_back(I, J, H_local(i, j));
                }
            }
        }
    }

    H_global_.resize(num_dofs, num_dofs);
    H_global_.setFromTriplets(triplets.begin(), triplets.end());
}

void CGTriharmonicBezierBathymetrySmoother::assemble_data_fitting(
    std::function<Real(Real, Real)> bathy_func) {

    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_elements = quadtree_->num_elements();
    int ngauss = config_.ngauss_data;

    // Get Gauss quadrature points and weights
    std::vector<Real> gauss_pts, gauss_wts;
    gauss_legendre_01(ngauss, gauss_pts, gauss_wts);

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * ngauss * ngauss * 36 * 36);

    BtWd_global_.setZero(num_dofs);
    dTWd_global_ = 0.0;

    for (Index elem = 0; elem < num_elements; ++elem) {
        const auto& bounds = quadtree_->element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real jacobian = dx * dy;

        const auto& global_dofs = dof_manager_->element_dofs(elem);

        // Integrate over element using Gauss quadrature
        for (int qi = 0; qi < ngauss; ++qi) {
            Real u = gauss_pts[qi];
            for (int qj = 0; qj < ngauss; ++qj) {
                Real v = gauss_pts[qj];
                Real weight = gauss_wts[qi] * gauss_wts[qj] * jacobian;

                // Physical coordinates
                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;

                // Bathymetry value at this point
                Real d = bathy_func(x, y);

                // Accumulate d^T W d for residual computation
                dTWd_global_ += weight * d * d;

                // Evaluate basis functions at (u, v)
                VecX B = basis_->evaluate(u, v);

                // Assemble B^T W B (outer product with weight)
                for (int i = 0; i < BezierBasis2D::NDOF; ++i) {
                    Index I = global_dofs[i];
                    for (int j = 0; j < BezierBasis2D::NDOF; ++j) {
                        Index J = global_dofs[j];
                        triplets.emplace_back(I, J, weight * B(i) * B(j));
                    }
                    // Assemble B^T W d
                    BtWd_global_(I) += weight * B(i) * d;
                }
            }
        }
    }

    BtWB_global_.resize(num_dofs, num_dofs);
    BtWB_global_.setFromTriplets(triplets.begin(), triplets.end());
}

// =============================================================================
// Solve
// =============================================================================

void CGTriharmonicBezierBathymetrySmoother::solve() {
    if (!data_set_) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: bathymetry data not set");
    }

    // Use constrained solve if we have any constraints (hanging nodes or edge derivatives)
    Index total_constraints = dof_manager_->num_constraints() +
                              dof_manager_->num_edge_derivative_constraints();

    if (total_constraints == 0) {
        solve_unconstrained();
    } else {
        solve_with_constraints();
    }

    solved_ = true;
}

void CGTriharmonicBezierBathymetrySmoother::solve_unconstrained() {
    Index num_dofs = dof_manager_->num_global_dofs();

    // Build Q matrix using ShipMesh formulation:
    // Q = alpha * H + lambda * (B^T W B + epsilon * I)
    // where alpha normalizes H to have similar magnitude to B^T W B

    Real norm_BtWB = BtWB_global_.norm();
    Real norm_H = H_global_.norm();
    Real alpha = 0.0;
    if (norm_H > 1e-14) {
        alpha = norm_BtWB / norm_H;
    }

    // Build Q = alpha * H + lambda * (B^T W B + epsilon * I)
    SpMat Q = alpha * H_global_ + config_.lambda * BtWB_global_;

    // Add ridge regularization
    for (Index i = 0; i < num_dofs; ++i) {
        Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
    }

    // RHS: c = -lambda * B^T W d
    VecX c = -config_.lambda * BtWd_global_;

    // Solve Q * x = -c
    Eigen::SparseLU<SpMat> solver;
    solver.compute(Q);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: SparseLU decomposition failed");
    }

    solution_ = solver.solve(-c);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: SparseLU solve failed");
    }
}

void CGTriharmonicBezierBathymetrySmoother::solve_with_constraints() {
    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_hanging_constraints = dof_manager_->num_constraints();
    Index num_edge_constraints = dof_manager_->num_edge_derivative_constraints();
    Index num_constraints = num_hanging_constraints + num_edge_constraints;

    // Build Q matrix (same as unconstrained)
    Real norm_BtWB = BtWB_global_.norm();
    Real norm_H = H_global_.norm();
    Real alpha = 0.0;
    if (norm_H > 1e-14) {
        alpha = norm_BtWB / norm_H;
    }

    SpMat Q = alpha * H_global_ + config_.lambda * BtWB_global_;
    for (Index i = 0; i < num_dofs; ++i) {
        Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
    }

    VecX c = -config_.lambda * BtWd_global_;

    // Build combined constraint matrix A
    std::vector<Eigen::Triplet<Real>> A_triplets;
    A_triplets.reserve(num_constraints * 50);

    Index row = 0;

    // Add hanging node constraints
    for (const auto& hc : dof_manager_->constraints()) {
        A_triplets.emplace_back(row, hc.slave_dof, 1.0);
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            A_triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
        }
        ++row;
    }

    // Add edge derivative constraints (at Gauss points along shared edges)
    for (const auto& ec : dof_manager_->edge_derivative_constraints()) {
        const auto& global_dofs1 = dof_manager_->element_dofs(ec.elem1);
        const auto& global_dofs2 = dof_manager_->element_dofs(ec.elem2);

        // Element 1 contribution (positive)
        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            Real coeff = ec.coeffs1(k) / ec.scale1;
            if (std::abs(coeff) > 1e-14) {
                A_triplets.emplace_back(row, global_dofs1[k], coeff);
            }
        }

        // Element 2 contribution (negative)
        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            Real coeff = ec.coeffs2(k) / ec.scale2;
            if (std::abs(coeff) > 1e-14) {
                A_triplets.emplace_back(row, global_dofs2[k], -coeff);
            }
        }

        ++row;
    }

    SpMat A(num_constraints, num_dofs);
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());

    // Build KKT system:
    // [Q   A^T] [x]   [-c]
    // [A    0 ] [mu] = [ 0]

    Index kkt_size = num_dofs + num_constraints;
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(Q.nonZeros() + 2 * A.nonZeros());

    // Add Q block
    for (int k = 0; k < Q.outerSize(); ++k) {
        for (SpMat::InnerIterator it(Q, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // Add A block (lower-left)
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A, k); it; ++it) {
            triplets.emplace_back(num_dofs + it.row(), it.col(), it.value());
        }
    }

    // Add A^T block (upper-right)
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A, k); it; ++it) {
            triplets.emplace_back(it.col(), num_dofs + it.row(), it.value());
        }
    }

    SpMat KKT(kkt_size, kkt_size);
    KKT.setFromTriplets(triplets.begin(), triplets.end());

    // Add small regularization to the (2,2) block to handle redundant constraints
    Real constraint_reg = 1e-10;
    for (Index i = num_dofs; i < kkt_size; ++i) {
        KKT.coeffRef(i, i) -= constraint_reg;
    }

    // RHS
    VecX rhs(kkt_size);
    rhs.head(num_dofs) = -c;
    rhs.tail(num_constraints).setZero();

    // Solve KKT system
    Eigen::SparseLU<SpMat> solver;
    solver.compute(KKT);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: KKT SparseLU decomposition failed");
    }

    VecX sol = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: KKT SparseLU solve failed");
    }

    // Extract primal solution
    solution_ = sol.head(num_dofs);

    // Project onto constraint manifold to enforce exact constraint satisfaction
    if (num_constraints > 0) {
        VecX Ax = A * solution_;

        SpMat AAt = A * A.transpose();
        for (Index i = 0; i < num_constraints; ++i) {
            AAt.coeffRef(i, i) += 1e-14;
        }

        Eigen::SparseLU<SpMat> projector;
        projector.compute(AAt);

        if (projector.info() == Eigen::Success) {
            VecX lambda = projector.solve(Ax);
            if (projector.info() == Eigen::Success) {
                VecX correction = A.transpose() * lambda;
                solution_ -= correction;
            }
        }
    }
}

// =============================================================================
// Evaluation
// =============================================================================

Index CGTriharmonicBezierBathymetrySmoother::find_element(Real x, Real y) const {
    for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
        const auto& bounds = quadtree_->element_bounds(elem);
        if (x >= bounds.xmin && x <= bounds.xmax &&
            y >= bounds.ymin && y <= bounds.ymax) {
            return elem;
        }
    }
    return -1;
}

Real CGTriharmonicBezierBathymetrySmoother::evaluate_in_element(
    Index elem, Real x, Real y) const {
    const auto& bounds = quadtree_->element_bounds(elem);

    // Map to parameter space [0, 1]^2
    Real u = (x - bounds.xmin) / (bounds.xmax - bounds.xmin);
    Real v = (y - bounds.ymin) / (bounds.ymax - bounds.ymin);

    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    VecX coeffs = element_coefficients(elem);
    return basis_->evaluate_scalar(coeffs, u, v);
}

Real CGTriharmonicBezierBathymetrySmoother::evaluate(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: must call solve() before evaluate()");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        // Extrapolate from nearest element
        Real min_dist = std::numeric_limits<Real>::max();
        Index closest = 0;
        for (Index e = 0; e < quadtree_->num_elements(); ++e) {
            const auto& b = quadtree_->element_bounds(e);
            Real cx = 0.5 * (b.xmin + b.xmax);
            Real cy = 0.5 * (b.ymin + b.ymax);
            Real dist = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            if (dist < min_dist) {
                min_dist = dist;
                closest = e;
            }
        }
        return evaluate_in_element(closest, x, y);
    }

    return evaluate_in_element(elem, x, y);
}

Vec2 CGTriharmonicBezierBathymetrySmoother::evaluate_gradient_in_element(
    Index elem, Real x, Real y) const {
    const auto& bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;

    Real u = (x - bounds.xmin) / dx;
    Real v = (y - bounds.ymin) / dy;
    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    VecX coeffs = element_coefficients(elem);

    VecX du = basis_->evaluate_du(u, v);
    VecX dv = basis_->evaluate_dv(u, v);

    Real dz_du = coeffs.dot(du);
    Real dz_dv = coeffs.dot(dv);

    return Vec2(dz_du / dx, dz_dv / dy);
}

Vec2 CGTriharmonicBezierBathymetrySmoother::evaluate_gradient(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: must call solve() before "
            "evaluate_gradient()");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        Real min_dist = std::numeric_limits<Real>::max();
        Index closest = 0;
        for (Index e = 0; e < quadtree_->num_elements(); ++e) {
            const auto& b = quadtree_->element_bounds(e);
            Real cx = 0.5 * (b.xmin + b.xmax);
            Real cy = 0.5 * (b.ymin + b.ymax);
            Real dist = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            if (dist < min_dist) {
                min_dist = dist;
                closest = e;
            }
        }
        return evaluate_gradient_in_element(closest, x, y);
    }

    return evaluate_gradient_in_element(elem, x, y);
}

VecX CGTriharmonicBezierBathymetrySmoother::element_coefficients(Index elem) const {
    const auto& global_dofs = dof_manager_->element_dofs(elem);
    VecX coeffs(BezierBasis2D::NDOF);
    for (int i = 0; i < BezierBasis2D::NDOF; ++i) {
        coeffs(i) = solution_(global_dofs[i]);
    }
    return coeffs;
}

// =============================================================================
// Transfer and output
// =============================================================================

void CGTriharmonicBezierBathymetrySmoother::transfer_to_seabed(
    SeabedSurface& seabed) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: must call solve() before "
            "transfer_to_seabed()");
    }

    for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
        VecX coeffs = element_coefficients(elem);
        seabed.set_element_coefficients(elem, coeffs);
    }
}

void CGTriharmonicBezierBathymetrySmoother::write_vtk(
    const std::string& filename, int resolution) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: must call solve() before write_vtk()");
    }

    io::write_cg_bezier_surface_vtk(
        filename,
        *quadtree_,
        [this](Real x, Real y) { return evaluate(x, y); },
        resolution > 0 ? resolution : 11,
        "elevation");
}

void CGTriharmonicBezierBathymetrySmoother::write_control_points_vtk(
    const std::string& filename) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGTriharmonicBezierBathymetrySmoother: must call solve() before "
            "write_control_points_vtk()");
    }

    io::write_bezier_control_points_vtk(
        filename,
        *quadtree_,
        [this](Index e) { return element_coefficients(e); },
        [](int dof) {
            int i = dof % 6;
            int j = dof / 6;
            return Vec2(static_cast<Real>(i) / 5, static_cast<Real>(j) / 5);
        },
        6);
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGTriharmonicBezierBathymetrySmoother::data_residual() const {
    if (!solved_) return 0.0;
    return solution_.dot(BtWB_global_ * solution_) -
           2.0 * solution_.dot(BtWd_global_) + dTWd_global_;
}

Real CGTriharmonicBezierBathymetrySmoother::regularization_energy() const {
    if (!solved_) return 0.0;
    return solution_.dot(H_global_ * solution_);
}

Real CGTriharmonicBezierBathymetrySmoother::objective_value() const {
    if (!solved_) return 0.0;
    Real norm_BtWB = BtWB_global_.norm();
    Real norm_H = H_global_.norm();
    Real alpha = (norm_H > 1e-14) ? norm_BtWB / norm_H : 1.0;

    return alpha * regularization_energy() + config_.lambda * data_residual();
}

Real CGTriharmonicBezierBathymetrySmoother::constraint_violation() const {
    Index num_hanging = dof_manager_->num_constraints();
    Index num_edge = dof_manager_->num_edge_derivative_constraints();
    Index num_constraints = num_hanging + num_edge;

    if (!solved_ || num_constraints == 0) return 0.0;

    // Build full constraint matrix
    Index num_dofs = dof_manager_->num_global_dofs();
    std::vector<Eigen::Triplet<Real>> A_triplets;
    Index row = 0;

    // Hanging node constraints
    for (const auto& hc : dof_manager_->constraints()) {
        A_triplets.emplace_back(row, hc.slave_dof, 1.0);
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            A_triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
        }
        ++row;
    }

    // Edge derivative constraints
    for (const auto& ec : dof_manager_->edge_derivative_constraints()) {
        const auto& global_dofs1 = dof_manager_->element_dofs(ec.elem1);
        const auto& global_dofs2 = dof_manager_->element_dofs(ec.elem2);

        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            Real coeff = ec.coeffs1(k) / ec.scale1;
            if (std::abs(coeff) > 1e-14) {
                A_triplets.emplace_back(row, global_dofs1[k], coeff);
            }
        }
        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            Real coeff = ec.coeffs2(k) / ec.scale2;
            if (std::abs(coeff) > 1e-14) {
                A_triplets.emplace_back(row, global_dofs2[k], -coeff);
            }
        }
        ++row;
    }

    SpMat A(num_constraints, num_dofs);
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());

    VecX violation = A * solution_;
    return violation.norm();
}

}  // namespace drifter
