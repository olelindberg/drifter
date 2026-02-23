#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "core/scoped_timer.hpp"
#include "dg/basis_hexahedron.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drifter {

namespace {

void gauss_legendre_01_cubic(int n, std::vector<Real> &pts, std::vector<Real> &wts) {
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
    } else if (n >= 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        pts.resize(4);
        wts.resize(4);
        pts[0] = 0.5 * (1.0 - b);
        pts[1] = 0.5 * (1.0 - a);
        pts[2] = 0.5 * (1.0 + a);
        pts[3] = 0.5 * (1.0 + b);
        Real wa = (18.0 + std::sqrt(30.0)) / 72.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 72.0;
        wts[0] = wts[3] = 0.5 * wb;
        wts[1] = wts[2] = 0.5 * wa;
    }
}

} // anonymous namespace

// =============================================================================
// Construction
// =============================================================================

CGCubicBezierBathymetrySmoother::CGCubicBezierBathymetrySmoother(
    const QuadtreeAdapter &mesh, const CGCubicBezierSmootherConfig &config)
    : quadtree_(&mesh), config_(config) {
    init_components();
}

CGCubicBezierBathymetrySmoother::CGCubicBezierBathymetrySmoother(
    const OctreeAdapter &octree, const CGCubicBezierSmootherConfig &config)
    : config_(config) {
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();
    init_components();
}

void CGCubicBezierBathymetrySmoother::init_components() {
    basis_ = std::make_unique<CubicBezierBasis2D>();
    thin_plate_hessian_ =
        std::make_unique<CubicThinPlateHessian>(config_.ngauss_energy, config_.gradient_weight);
    dof_manager_ = std::make_unique<CGCubicBezierDofManager>(*quadtree_);
    dof_manager_->build_edge_derivative_constraints(config_.edge_ngauss);

    solution_.setZero(dof_manager_->num_global_dofs());
}

// =============================================================================
// Data input
// =============================================================================

void CGCubicBezierBathymetrySmoother::set_bathymetry_data(const BathymetrySource &source) {
    set_bathymetry_data([&source](Real x, Real y) { return source.evaluate(x, y); });
}

void CGCubicBezierBathymetrySmoother::set_bathymetry_data(
    std::function<Real(Real, Real)> bathy_func) {
    assemble_thin_plate_hessian();
    assemble_data_fitting(bathy_func);
    data_set_ = true;
}

void CGCubicBezierBathymetrySmoother::set_scattered_points(const std::vector<Vec3> &points) {
    std::vector<BathymetryPoint> bathy_points;
    bathy_points.reserve(points.size());
    for (const auto &p : points) {
        bathy_points.emplace_back(p(0), p(1), p(2), 1.0);
    }
    set_scattered_points(bathy_points);
}

void CGCubicBezierBathymetrySmoother::set_scattered_points(
    const std::vector<BathymetryPoint> &points) {
    std::vector<BathymetryPoint> pts = points;

    auto bathy_func = [pts](Real x, Real y) -> Real {
        Real min_dist = std::numeric_limits<Real>::max();
        Real value = 0.0;
        for (const auto &p : pts) {
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

void CGCubicBezierBathymetrySmoother::assemble_thin_plate_hessian() {
    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_elements = quadtree_->num_elements();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * 16 * 16);

    for (Index elem = 0; elem < num_elements; ++elem) {
        Vec2 size = quadtree_->element_size(elem);
        Real dx = size(0);
        Real dy = size(1);

        MatX H_local = thin_plate_hessian_->scaled_hessian(dx, dy);
        const auto &global_dofs = dof_manager_->element_dofs(elem);

        for (int i = 0; i < CubicBezierBasis2D::NDOF; ++i) {
            Index I = global_dofs[i];
            for (int j = 0; j < CubicBezierBasis2D::NDOF; ++j) {
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

void CGCubicBezierBathymetrySmoother::assemble_data_fitting(
    std::function<Real(Real, Real)> bathy_func) {

    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_elements = quadtree_->num_elements();
    int ngauss = config_.ngauss_data;

    std::vector<Real> gauss_pts, gauss_wts;
    gauss_legendre_01_cubic(ngauss, gauss_pts, gauss_wts);

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(num_elements * ngauss * ngauss * 16 * 16);

    BtWd_global_.setZero(num_dofs);
    dTWd_global_ = 0.0;

    for (Index elem = 0; elem < num_elements; ++elem) {
        const auto &bounds = quadtree_->element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real jacobian = dx * dy;

        const auto &global_dofs = dof_manager_->element_dofs(elem);

        for (int qi = 0; qi < static_cast<int>(gauss_pts.size()); ++qi) {
            Real u = gauss_pts[qi];
            for (int qj = 0; qj < static_cast<int>(gauss_pts.size()); ++qj) {
                Real v = gauss_pts[qj];
                Real weight = gauss_wts[qi] * gauss_wts[qj] * jacobian;

                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;
                Real d = bathy_func(x, y);

                dTWd_global_ += weight * d * d;

                VecX B = basis_->evaluate(u, v);

                for (int i = 0; i < CubicBezierBasis2D::NDOF; ++i) {
                    Index I = global_dofs[i];
                    for (int j = 0; j < CubicBezierBasis2D::NDOF; ++j) {
                        Index J = global_dofs[j];
                        triplets.emplace_back(I, J, weight * B(i) * B(j));
                    }
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

void CGCubicBezierBathymetrySmoother::solve() {
    if (!data_set_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: bathymetry data not set");
    }

    Index total_constraints =
        dof_manager_->num_constraints() + dof_manager_->num_edge_derivative_constraints();

    if (total_constraints == 0) {
        solve_unconstrained();
    } else {
        solve_with_constraints();
    }

    solved_ = true;
}

void CGCubicBezierBathymetrySmoother::solve_unconstrained() {
    Index num_dofs = dof_manager_->num_global_dofs();

    SpMat Q;
    VecX c;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->matrix_build_ms : nullptr);

        Real norm_BtWB = BtWB_global_.norm();
        Real norm_H = H_global_.norm();
        Real alpha = 0.0;
        if (norm_H > 1e-14) {
            alpha = norm_BtWB / norm_H;
        }

        Q = alpha * H_global_ + config_.lambda * BtWB_global_;

        for (Index i = 0; i < num_dofs; ++i) {
            Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
        }

        c = -config_.lambda * BtWd_global_;
    }

    Eigen::SparseLU<SpMat> solver;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_compute_ms : nullptr);
        solver.compute(Q);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: SparseLU decomposition failed");
    }

    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_solve_ms : nullptr);
        solution_ = solver.solve(-c);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: SparseLU solve failed");
    }
}

void CGCubicBezierBathymetrySmoother::solve_with_constraints() {
    Index num_dofs = dof_manager_->num_global_dofs();
    Index num_hanging_constraints = dof_manager_->num_constraints();
    Index num_edge_constraints = dof_manager_->num_edge_derivative_constraints();
    Index num_constraints = num_hanging_constraints + num_edge_constraints;

    SpMat Q;
    VecX c;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->matrix_build_ms : nullptr);

        Real norm_BtWB = BtWB_global_.norm();
        Real norm_H = H_global_.norm();
        Real alpha = 0.0;
        if (norm_H > 1e-14) {
            alpha = norm_BtWB / norm_H;
        }

        Q = alpha * H_global_ + config_.lambda * BtWB_global_;
        for (Index i = 0; i < num_dofs; ++i) {
            Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
        }

        c = -config_.lambda * BtWd_global_;
    }

    SpMat A;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_build_ms : nullptr);

        std::vector<Eigen::Triplet<Real>> A_triplets;
        A_triplets.reserve(num_constraints * 20);

        Index row = 0;

        // Hanging node constraints
        for (const auto &hc : dof_manager_->constraints()) {
            A_triplets.emplace_back(row, hc.slave_dof, 1.0);
            for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
                A_triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
            }
            ++row;
        }

        // C¹ edge derivative constraints
        for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
            const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
            const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

            for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
                Real coeff = ec.coeffs1(k) / ec.scale1;
                if (std::abs(coeff) > 1e-14) {
                    A_triplets.emplace_back(row, global_dofs1[k], coeff);
                }
            }

            for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
                Real coeff = ec.coeffs2(k) / ec.scale2;
                if (std::abs(coeff) > 1e-14) {
                    A_triplets.emplace_back(row, global_dofs2[k], -coeff);
                }
            }

            ++row;
        }

        A.resize(num_constraints, num_dofs);
        A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    }

    SpMat KKT;
    VecX rhs;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->kkt_assembly_ms : nullptr);

        // Build KKT system
        Index kkt_size = num_dofs + num_constraints;
        std::vector<Eigen::Triplet<Real>> triplets;
        triplets.reserve(Q.nonZeros() + 2 * A.nonZeros());

        // Copy Q block
        for (int k = 0; k < Q.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Q, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }

        // Single pass over A: add both A (lower-left) and A^T (upper-right) blocks
        for (int k = 0; k < A.outerSize(); ++k) {
            for (SpMat::InnerIterator it(A, k); it; ++it) {
                triplets.emplace_back(num_dofs + it.row(), it.col(), it.value());
                triplets.emplace_back(it.col(), num_dofs + it.row(), it.value());
            }
        }

        KKT.resize(kkt_size, kkt_size);
        KKT.setFromTriplets(triplets.begin(), triplets.end());

        Real constraint_reg = 1e-10;
        for (Index i = num_dofs; i < kkt_size; ++i) {
            KKT.coeffRef(i, i) -= constraint_reg;
        }

        rhs.resize(kkt_size);
        rhs.head(num_dofs) = -c;
        rhs.tail(num_constraints).setZero();
    }

    Eigen::SparseLU<SpMat> solver;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_compute_ms : nullptr);
        solver.compute(KKT);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT "
                                 "SparseLU decomposition failed");
    }

    VecX sol;
    {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->sparse_lu_solve_ms : nullptr);
        sol = solver.solve(rhs);
    }
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: KKT SparseLU solve failed");
    }

    solution_ = sol.head(num_dofs);

    // Project onto constraint manifold
    if (num_constraints > 0) {
        OptionalScopedTimer t(solve_profile_ ? &solve_profile_->constraint_projection_ms : nullptr);

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

Index CGCubicBezierBathymetrySmoother::find_element(Real x, Real y) const {
    for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
        const auto &bounds = quadtree_->element_bounds(elem);
        if (x >= bounds.xmin && x <= bounds.xmax && y >= bounds.ymin && y <= bounds.ymax) {
            return elem;
        }
    }
    return -1;
}

Real CGCubicBezierBathymetrySmoother::evaluate_in_element(Index elem, Real x, Real y) const {
    const auto &bounds = quadtree_->element_bounds(elem);

    Real u = (x - bounds.xmin) / (bounds.xmax - bounds.xmin);
    Real v = (y - bounds.ymin) / (bounds.ymax - bounds.ymin);

    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    VecX coeffs = element_coefficients(elem);
    return basis_->evaluate_scalar(coeffs, u, v);
}

Real CGCubicBezierBathymetrySmoother::evaluate(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: must call "
                                 "solve() before evaluate()");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        Real min_dist = std::numeric_limits<Real>::max();
        Index closest = 0;
        for (Index e = 0; e < quadtree_->num_elements(); ++e) {
            const auto &b = quadtree_->element_bounds(e);
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

Vec2 CGCubicBezierBathymetrySmoother::evaluate_gradient_in_element(Index elem, Real x,
                                                                   Real y) const {
    const auto &bounds = quadtree_->element_bounds(elem);
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

Vec2 CGCubicBezierBathymetrySmoother::evaluate_gradient(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: must call "
                                 "solve() before evaluate_gradient()");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        Real min_dist = std::numeric_limits<Real>::max();
        Index closest = 0;
        for (Index e = 0; e < quadtree_->num_elements(); ++e) {
            const auto &b = quadtree_->element_bounds(e);
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

VecX CGCubicBezierBathymetrySmoother::element_coefficients(Index elem) const {
    const auto &global_dofs = dof_manager_->element_dofs(elem);
    VecX coeffs(CubicBezierBasis2D::NDOF);
    for (int i = 0; i < CubicBezierBasis2D::NDOF; ++i) {
        coeffs(i) = solution_(global_dofs[i]);
    }
    return coeffs;
}

// =============================================================================
// Transfer and output
// =============================================================================

void CGCubicBezierBathymetrySmoother::transfer_to_seabed(SeabedSurface &seabed) const {
    if (!solved_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: must call "
                                 "solve() before transfer_to_seabed()");
    }

    for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
        VecX coeffs = element_coefficients(elem);
        seabed.set_element_coefficients(elem, coeffs);
    }
}

void CGCubicBezierBathymetrySmoother::write_vtk(const std::string &filename, int resolution) const {
    if (!solved_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: must call "
                                 "solve() before write_vtk()");
    }

    // Use CG-aware VTK writer that deduplicates shared vertices at element
    // boundaries, producing a properly connected mesh without visual gaps
    io::write_cg_bezier_surface_vtk(
        filename, *quadtree_, [this](Real x, Real y) { return evaluate(x, y); },
        resolution > 0 ? resolution : 9, "elevation");
}

void CGCubicBezierBathymetrySmoother::write_control_points_vtk(const std::string &filename) const {
    if (!solved_) {
        throw std::runtime_error("CGCubicBezierBathymetrySmoother: must call "
                                 "solve() before write_control_points_vtk()");
    }

    io::write_bezier_control_points_vtk(
        filename, *quadtree_, [this](Index e) { return element_coefficients(e); },
        [](int dof) {
            int i = dof % 4;
            int j = dof / 4;
            return Vec2(static_cast<Real>(i) / 3, static_cast<Real>(j) / 3);
        },
        4);
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGCubicBezierBathymetrySmoother::data_residual() const {
    if (!solved_)
        return 0.0;
    return solution_.dot(BtWB_global_ * solution_) - 2.0 * solution_.dot(BtWd_global_) +
           dTWd_global_;
}

Real CGCubicBezierBathymetrySmoother::regularization_energy() const {
    if (!solved_)
        return 0.0;
    return solution_.dot(H_global_ * solution_);
}

Real CGCubicBezierBathymetrySmoother::objective_value() const {
    if (!solved_)
        return 0.0;
    Real norm_BtWB = BtWB_global_.norm();
    Real norm_H = H_global_.norm();
    Real alpha = (norm_H > 1e-14) ? norm_BtWB / norm_H : 1.0;

    return alpha * regularization_energy() + config_.lambda * data_residual();
}

Real CGCubicBezierBathymetrySmoother::constraint_violation() const {
    Index num_hanging = dof_manager_->num_constraints();
    Index num_edge = dof_manager_->num_edge_derivative_constraints();
    Index num_constraints = num_hanging + num_edge;

    if (!solved_ || num_constraints == 0)
        return 0.0;

    Index num_dofs = dof_manager_->num_global_dofs();
    std::vector<Eigen::Triplet<Real>> A_triplets;
    Index row = 0;

    for (const auto &hc : dof_manager_->constraints()) {
        A_triplets.emplace_back(row, hc.slave_dof, 1.0);
        for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
            A_triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
        }
        ++row;
    }

    for (const auto &ec : dof_manager_->edge_derivative_constraints()) {
        const auto &global_dofs1 = dof_manager_->element_dofs(ec.elem1);
        const auto &global_dofs2 = dof_manager_->element_dofs(ec.elem2);

        for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
            Real coeff = ec.coeffs1(k) / ec.scale1;
            if (std::abs(coeff) > 1e-14) {
                A_triplets.emplace_back(row, global_dofs1[k], coeff);
            }
        }
        for (int k = 0; k < CubicBezierBasis2D::NDOF; ++k) {
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

} // namespace drifter
