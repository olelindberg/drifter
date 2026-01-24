#include "bathymetry/bezier_bathymetry_smoother.hpp"
#include "dg/basis_hexahedron.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>

namespace drifter {

BezierBathymetrySmoother::BezierBathymetrySmoother(const QuadtreeAdapter& mesh,
                                                     const BezierSmootherConfig& config)
    : quadtree_(&mesh)
    , config_(config)
{
    init_components();
}

BezierBathymetrySmoother::BezierBathymetrySmoother(const OctreeAdapter& octree,
                                                     const BezierSmootherConfig& config)
    : config_(config)
{
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();
    init_components();
}

void BezierBathymetrySmoother::init_components() {
    basis_ = std::make_unique<BezierBasis2D>();
    hessian_ = std::make_unique<ThinPlateHessian>(config_.ngauss_energy, config_.gradient_weight);
    constraint_builder_ = std::make_unique<BezierC2ConstraintBuilder>(*quadtree_);
    data_assembler_ = std::make_unique<BezierDataFittingAssembler>(*quadtree_);
}

void BezierBathymetrySmoother::set_bathymetry_data(const BathymetrySource& source) {
    data_assembler_->set_from_bathymetry_source(source, config_.ngauss_data);
    cache_valid_ = false;
    solved_ = false;
}

void BezierBathymetrySmoother::set_bathymetry_data(std::function<Real(Real, Real)> bathy_func) {
    data_assembler_->set_from_function(std::move(bathy_func), config_.ngauss_data);
    cache_valid_ = false;
    solved_ = false;
}

void BezierBathymetrySmoother::set_scattered_points(const std::vector<Vec3>& points) {
    data_assembler_->set_scattered_points(points);
    cache_valid_ = false;
    solved_ = false;
}

void BezierBathymetrySmoother::set_scattered_points(const std::vector<BathymetryPoint>& points) {
    data_assembler_->set_scattered_points(points);
    cache_valid_ = false;
    solved_ = false;
}

void BezierBathymetrySmoother::solve() {
    if (!data_assembler_->has_data()) {
        throw std::runtime_error("BezierBathymetrySmoother: no data set");
    }

    if (config_.lower_bound.has_value() && config_.upper_bound.has_value()) {
        solve_with_bounds();
    } else {
        solve_kkt();
    }

    solved_ = true;
}

SpMat BezierBathymetrySmoother::build_global_hessian_sparse() const {
    Index ndofs = data_assembler_->total_dofs();
    Index num_elements = quadtree_->num_elements();

    // Determine number of threads
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Per-thread triplet vectors (eliminates critical section)
    std::vector<std::vector<Eigen::Triplet<Real>>> thread_triplets(num_threads);

    // Pre-allocate per-thread storage
    for (int t = 0; t < num_threads; ++t) {
        thread_triplets[t].reserve(num_elements * BezierBasis2D::NDOF * BezierBasis2D::NDOF / num_threads);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for nowait schedule(static)
        for (Index e = 0; e < num_elements; ++e) {
            Vec2 size = quadtree_->element_size(e);
            Real dx = size(0);
            Real dy = size(1);

            MatX H_elem = hessian_->scaled_hessian(dx, dy);

            Index base = e * BezierBasis2D::NDOF;

            // Add element Hessian to thread-local triplets (no locking needed)
            for (Index i = 0; i < BezierBasis2D::NDOF; ++i) {
                for (Index j = 0; j < BezierBasis2D::NDOF; ++j) {
                    if (std::abs(H_elem(i, j)) > 1e-16) {  // Skip near-zero entries
                        thread_triplets[tid].emplace_back(base + i, base + j, H_elem(i, j));
                    }
                }
            }
        }
    }

    // Merge thread-local triplets sequentially (no critical section needed)
    std::vector<Eigen::Triplet<Real>> triplets;
    size_t total_size = 0;
    for (const auto& thread_trips : thread_triplets) {
        total_size += thread_trips.size();
    }
    triplets.reserve(total_size);

    for (const auto& thread_trips : thread_triplets) {
        triplets.insert(triplets.end(), thread_trips.begin(), thread_trips.end());
    }

    // Assemble and return sparse matrix
    SpMat H_sparse(ndofs, ndofs);
    H_sparse.setFromTriplets(triplets.begin(), triplets.end());
    return H_sparse;
}

MatX BezierBathymetrySmoother::build_global_hessian() const {
    // Legacy function - converts to dense for backward compatibility
    return MatX(build_global_hessian_sparse());
}

void BezierBathymetrySmoother::solve_kkt() {
    // ShipMesh-style formulation:
    //   minimize: x^T * H * x  (thin plate energy / smoothness)
    //   subject to:
    //     A_c2 * x = 0                        (C² continuity at interior vertices)
    //     A_nat * x = 0                       (natural BCs: z_nn = 0 at boundary)
    //     (A^T W A + λI) * x = A^T W b        (least squares optimality)
    //
    // This prioritizes smoothness while ensuring the data is fit well.
    // The least squares condition is treated as a constraint, not the objective.
    //
    // We solve via a combined KKT system. Since the least squares constraint
    // is dense (n x n), we use a penalty approach for it instead:
    //
    // Augmented objective:
    //   minimize: x^T * H * x + μ * ||(A^T W A + λI) x - A^T W b||²
    //
    // This expands to a standard QP with:
    //   Q = H + μ * (A^T W A + λI)^T (A^T W A + λI)
    //   c = -μ * (A^T W A + λI)^T * A^T W b
    //
    // For large μ, this approaches the constrained solution.
    //
    // Alternative simpler approach: Since (A^T W A + λI) x = A^T W b is the
    // first-order optimality condition for weighted least squares, we can
    // solve the combined system directly using null-space projection.

    // Timing instrumentation (enabled via DRIFTER_PROFILE environment variable)
    bool enable_profiling = std::getenv("DRIFTER_PROFILE") != nullptr;
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_last = t_start;

    // Build normal equations components
    MatX AtWA;
    VecX AtWb;
    data_assembler_->assemble_normal_equations(AtWA, AtWb);

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] Data fitting assembly: " << elapsed << " ms\n";
        t_last = t_now;
    }

    // Ridge regularization (Tikhonov) - matches ShipMesh's lambda = 0.0001
    const Real ridge_lambda = 1e-4;
    Index n = AtWA.rows();

    // Add ridge regularization to AtWA
    MatX AtWA_reg = AtWA;
    for (Index i = 0; i < n; ++i) {
        AtWA_reg(i, i) += ridge_lambda;
    }

    // Build smoothness Hessian (thin plate energy)
    SpMat H_sparse = build_global_hessian_sparse();

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] Hessian assembly: " << elapsed << " ms\n";
        t_last = t_now;
    }

    // Convert AtWA to sparse (it's actually sparse - most elements are zero)
    SpMat AtWA_sparse = AtWA_reg.sparseView();

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] Convert AtWA to sparse: " << elapsed << " ms\n";
        t_last = t_now;
    }

    // Normalize matrices for scale-invariant lambda
    // Physical scaling makes H_global ~1e-8 for km-scale elements, while AtWA ~1.
    // This ensures lambda controls the actual balance regardless of physical scale.
    Real H_norm = H_sparse.norm();
    Real A_norm = AtWA_sparse.norm();
    Real scale_factor = (H_norm > 1e-14) ? (A_norm / H_norm) : 1.0;

    // Combined objective with normalized smoothness:
    // Q = scale * H + lambda * AtWA
    // Compute directly as sparse (MUCH faster than dense operations)
    const Real ls_penalty = config_.lambda;
    SpMat Q_sp = scale_factor * H_sparse + ls_penalty * AtWA_sparse;
    VecX c = -ls_penalty * AtWb;

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] Q matrix construction: " << elapsed << " ms\n";
        t_last = t_now;
    }

    // Apply Dirichlet BCs via row/column elimination in Q and c
    // (Only when natural BCs are disabled)
    // For each Dirichlet DOF i: modify RHS for coupling, then set row to identity
    std::set<Index> dirichlet_dofs;
    if (config_.enable_boundary_dirichlet && !config_.enable_natural_bc) {
        if (!data_assembler_->has_bathymetry_function()) {
            throw std::runtime_error(
                "BezierBathymetrySmoother: Dirichlet BCs require bathymetry function");
        }

        const auto& dir_constraints = constraint_builder_->dirichlet_constraints();

        // First pass: collect all Dirichlet DOFs and their values
        std::vector<std::pair<Index, Real>> dir_values;
        for (const auto& info : dir_constraints) {
            Index dof = info.global_dof;
            Real depth = data_assembler_->evaluate_bathymetry(
                info.position(0), info.position(1));
            dir_values.emplace_back(dof, depth);
            dirichlet_dofs.insert(dof);
        }

        // Second pass: modify RHS to account for known Dirichlet values
        // Need to iterate through sparse matrix Q to find coupling terms
        for (const auto& [dof, depth] : dir_values) {
            // For each row j, find Q(j, dof) and update RHS
            for (int k = 0; k < Q_sp.outerSize(); ++k) {
                for (SpMat::InnerIterator it(Q_sp, k); it; ++it) {
                    if (it.col() == dof && dirichlet_dofs.count(it.row()) == 0) {
                        c(it.row()) += it.value() * depth;
                    }
                }
            }
        }

        // Third pass: apply row/column elimination on sparse matrix
        // Build new triplets with Dirichlet rows/cols zeroed and diagonal set to 1
        std::vector<Eigen::Triplet<Real>> new_triplets;
        new_triplets.reserve(Q_sp.nonZeros());

        for (int k = 0; k < Q_sp.outerSize(); ++k) {
            for (SpMat::InnerIterator it(Q_sp, k); it; ++it) {
                if (dirichlet_dofs.count(it.row()) == 0 && dirichlet_dofs.count(it.col()) == 0) {
                    // Keep non-Dirichlet entries
                    new_triplets.emplace_back(it.row(), it.col(), it.value());
                }
            }
        }

        // Add identity for Dirichlet DOFs
        for (const auto& [dof, depth] : dir_values) {
            new_triplets.emplace_back(dof, dof, 1.0);
            c(dof) = -depth;
        }

        // Rebuild Q_sp
        Q_sp.setFromTriplets(new_triplets.begin(), new_triplets.end());

        if (enable_profiling) {
            auto t_now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
            std::cout << "[Profile] Dirichlet BC application: " << elapsed << " ms\n";
            t_last = t_now;
        }
    }

    // Build constraint matrix
    // If natural BCs are enabled, use C² + natural BC constraints
    // Otherwise, just C² constraints
    SpMat A_constraints;
    if (config_.enable_natural_bc) {
        A_constraints = constraint_builder_->build_c2_and_natural_bc_matrix();
    } else {
        A_constraints = constraint_builder_->build_constraint_matrix();
    }
    Index m = A_constraints.rows();

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] Constraint matrix build: " << elapsed << " ms\n";
        t_last = t_now;
    }

    if (m == 0) {
        // No constraints - solve directly
        MatX Q_dense(Q_sp);  // Convert sparse to dense for LDLT
        Eigen::LDLT<MatX> solver(Q_dense);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("BezierBathymetrySmoother: Q factorization failed");
        }
        solution_ = solver.solve(-c);
        return;
    }

    // Handle Dirichlet DOFs in constraints (only when Dirichlet is enabled)
    // Original: A * x = 0
    // With Dirichlet: A_free * x_free + A_dir * x_dir = 0
    // Rearranged: A_free * x_free = -A_dir * x_dir = b_constraints
    VecX b_constraints = VecX::Zero(m);
    if (!dirichlet_dofs.empty()) {
        // Build vector of Dirichlet values
        VecX x_dir = VecX::Zero(n);
        for (const auto& info : constraint_builder_->dirichlet_constraints()) {
            x_dir(info.global_dof) = data_assembler_->evaluate_bathymetry(
                info.position(0), info.position(1));
        }

        // Compute RHS contribution: b = -A * x_dir
        b_constraints = -(A_constraints * x_dir);

        // Zero out Dirichlet columns in A_constraints
        MatX A_dense = MatX(A_constraints);
        for (Index dof : dirichlet_dofs) {
            A_dense.col(dof).setZero();
        }
        A_constraints = A_dense.sparseView();

        if (enable_profiling) {
            auto t_now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
            std::cout << "[Profile] Dirichlet in constraints: " << elapsed << " ms\n";
            t_last = t_now;
        }
    }

    // Build KKT system:
    // [Q    A^T] [x]   [-c ]
    // [A     0 ] [y] = [b  ]

    // Q_sp is already sparse from earlier construction

    SpMat KKT(n + m, n + m);
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(Q_sp.nonZeros() + 2 * A_constraints.nonZeros());

    // Q block (upper left)
    for (int k = 0; k < Q_sp.outerSize(); ++k) {
        for (SpMat::InnerIterator it(Q_sp, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // A block (lower left)
    for (int k = 0; k < A_constraints.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_constraints, k); it; ++it) {
            triplets.emplace_back(n + it.row(), it.col(), it.value());
        }
    }

    // A^T block (upper right)
    for (int k = 0; k < A_constraints.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A_constraints, k); it; ++it) {
            triplets.emplace_back(it.col(), n + it.row(), it.value());
        }
    }

    // Add small regularization to (2,2) block for numerical stability
    const Real kkt_regularization = 1e-14;
    for (Index i = 0; i < m; ++i) {
        triplets.emplace_back(n + i, n + i, -kkt_regularization);
    }

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] KKT triplet assembly: " << elapsed << " ms\n";
        t_last = t_now;
    }

    KKT.setFromTriplets(triplets.begin(), triplets.end());

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] KKT setFromTriplets: " << elapsed << " ms\n";
        t_last = t_now;
    }

    // Build RHS
    VecX rhs(n + m);
    rhs.head(n) = -c;
    rhs.tail(m) = b_constraints;

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] KKT RHS build: " << elapsed << " ms\n";
        t_last = t_now;
    }

    // Solve
    Eigen::SparseLU<SpMat> solver;
    solver.compute(KKT);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("BezierBathymetrySmoother: KKT factorization failed");
    }

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] SparseLU factorization: " << elapsed << " ms\n";
        t_last = t_now;
    }

    VecX sol = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("BezierBathymetrySmoother: KKT solve failed");
    }

    if (enable_profiling) {
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
        std::cout << "[Profile] SparseLU solve: " << elapsed << " ms\n";
        t_last = t_now;
    }

    solution_ = sol.head(n);

    // Project solution onto constraint manifold for exact satisfaction
    // This corrects any numerical drift from ill-conditioning when lambda is large
    VecX constraint_residual = A_constraints * solution_ - b_constraints;
    Real constraint_violation_before = constraint_residual.norm();

    if (enable_profiling) {
        std::cout << "[Profile] Constraint violation before projection: " << constraint_violation_before << "\n";
    }

    // Only project if constraint violation is significant (> 1e-10)
    // KKT solver typically achieves ~1e-13 accuracy, so projection is usually unnecessary
    if (constraint_violation_before > 1e-10) {
        // Solve (A A^T) lambda = residual, then x = x - A^T * lambda
        // Use sparse ConjugateGradient instead of dense LDLT for better scaling
        // (A·A^T is symmetric positive definite)
        SpMat AAt = A_constraints * A_constraints.transpose();

        Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
        cg.setTolerance(1e-12);  // Tight tolerance for constraint satisfaction
        cg.setMaxIterations(std::max(static_cast<int>(m), 100));
        cg.compute(AAt);

        if (enable_profiling) {
            std::cout << "[Profile] CG compute status: " << (cg.info() == Eigen::Success ? "Success" : "Failed") << "\n";
        }

        if (cg.info() == Eigen::Success) {
            VecX lambda_corr = cg.solve(constraint_residual);

            if (enable_profiling) {
                std::cout << "[Profile] CG solve status: " << (cg.info() == Eigen::Success ? "Success" : "Failed") << "\n";
                std::cout << "[Profile] CG iterations: " << cg.iterations() << "\n";
                std::cout << "[Profile] CG error: " << cg.error() << "\n";
            }

            if (cg.info() == Eigen::Success) {
                solution_ -= A_constraints.transpose() * lambda_corr;

                if (enable_profiling) {
                    VecX final_residual = A_constraints * solution_ - b_constraints;
                    std::cout << "[Profile] Constraint violation after projection: " << final_residual.norm() << "\n";
                }
            }
        }

        if (enable_profiling) {
            auto t_now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
            std::cout << "[Profile] Constraint projection (CG): " << elapsed << " ms\n";
            t_last = t_now;
        }
    } else if (enable_profiling) {
        std::cout << "[Profile] Skipping constraint projection (violation < 1e-14)\n";
    }

    if (enable_profiling) {
        auto t_total = std::chrono::high_resolution_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_total - t_start).count();
        std::cout << "[Profile] TOTAL solve_kkt time: " << total_elapsed << " ms\n";
        std::cout << "[Profile] System size: n=" << n << ", m=" << m << " (total=" << (n+m) << ")\n";
    }
}

void BezierBathymetrySmoother::solve_with_bounds() {
    // First solve without bounds
    solve_kkt();

    // Then apply bounds via active-set iteration
    apply_bound_constraints();
}

void BezierBathymetrySmoother::apply_bound_constraints() {
    Real lower = config_.lower_bound.value();
    Real upper = config_.upper_bound.value();

    for (int iter = 0; iter < config_.max_bound_iterations; ++iter) {
        bool any_violated = false;

        // Check and clamp violated bounds
        for (Index i = 0; i < solution_.size(); ++i) {
            if (solution_(i) < lower - config_.bound_tolerance) {
                solution_(i) = lower;
                any_violated = true;
            } else if (solution_(i) > upper + config_.bound_tolerance) {
                solution_(i) = upper;
                any_violated = true;
            }
        }

        if (!any_violated) break;

        // Simple projection approach: just clamp for now
        // A more sophisticated active-set method would re-solve with
        // fixed DOFs, but that requires more complex bookkeeping
    }
}

Real BezierBathymetrySmoother::evaluate(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("BezierBathymetrySmoother: solve() not called");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        throw std::runtime_error("BezierBathymetrySmoother: point outside mesh");
    }

    return evaluate_in_element(elem, x, y);
}

Vec2 BezierBathymetrySmoother::evaluate_gradient(Real x, Real y) const {
    if (!solved_) {
        throw std::runtime_error("BezierBathymetrySmoother: solve() not called");
    }

    Index elem = find_element(x, y);
    if (elem < 0) {
        throw std::runtime_error("BezierBathymetrySmoother: point outside mesh");
    }

    return evaluate_gradient_in_element(elem, x, y);
}

Index BezierBathymetrySmoother::find_element(Real x, Real y) const {
    return quadtree_->find_element(Vec2(x, y));
}

Real BezierBathymetrySmoother::evaluate_in_element(Index elem, Real x, Real y) const {
    const QuadBounds& bounds = quadtree_->element_bounds(elem);
    Real u = (x - bounds.xmin) / (bounds.xmax - bounds.xmin);
    Real v = (y - bounds.ymin) / (bounds.ymax - bounds.ymin);

    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    VecX coeffs = element_coefficients(elem);
    return basis_->evaluate_scalar(coeffs, u, v);
}

Vec2 BezierBathymetrySmoother::evaluate_gradient_in_element(Index elem, Real x, Real y) const {
    const QuadBounds& bounds = quadtree_->element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;

    Real u = (x - bounds.xmin) / dx;
    Real v = (y - bounds.ymin) / dy;

    u = std::clamp(u, 0.0, 1.0);
    v = std::clamp(v, 0.0, 1.0);

    VecX coeffs = element_coefficients(elem);

    // Evaluate parameter derivatives
    VecX du = basis_->evaluate_du(u, v);
    VecX dv = basis_->evaluate_dv(u, v);

    Real dz_du = coeffs.dot(du);
    Real dz_dv = coeffs.dot(dv);

    // Chain rule: dz/dx = (dz/du) / dx
    return Vec2(dz_du / dx, dz_dv / dy);
}

VecX BezierBathymetrySmoother::element_coefficients(Index elem) const {
    Index base = elem * BezierBasis2D::NDOF;
    return solution_.segment(base, BezierBasis2D::NDOF);
}

void BezierBathymetrySmoother::transfer_to_seabed(SeabedSurface& seabed) const {
    if (!solved_) {
        throw std::runtime_error("BezierBathymetrySmoother: solve() not called");
    }

    // For each seabed element, find corresponding quadtree element
    // and copy the Bezier/Bernstein coefficients
    for (size_t s = 0; s < seabed.num_elements(); ++s) {
        // Get mesh element index and its bounds to find center
        Index mesh_elem = seabed.mesh_element_index(s);
        const ElementBounds& bounds = seabed.mesh().element_bounds(mesh_elem);
        Real cx = 0.5 * (bounds.xmin + bounds.xmax);
        Real cy = 0.5 * (bounds.ymin + bounds.ymax);

        // Find quadtree element at this location
        Index quad_elem = find_element(cx, cy);

        if (quad_elem >= 0) {
            VecX coeffs = element_coefficients(quad_elem);

            // SeabedSurface expects Bernstein depth coefficients
            // Our solution is z-values (which are depths for bathymetry)
            seabed.set_element_coefficients(s, coeffs);
        }
    }
}

void BezierBathymetrySmoother::write_control_points_vtk(const std::string& filename) const {
    if (!solved_) {
        throw std::runtime_error("BezierBathymetrySmoother: solve() not called");
    }

    std::ofstream file(filename + ".vtu");
    if (!file) {
        throw std::runtime_error("BezierBathymetrySmoother: cannot open " + filename + ".vtu");
    }

    // Control points: 6×6 = 36 per element
    constexpr int N1D = BezierBasis2D::N1D;  // 6
    constexpr int NDOF = BezierBasis2D::NDOF;  // 36

    Index num_elems = quadtree_->num_elements();
    Index total_pts = num_elems * NDOF;

    // Cells: connect control points as quads (5×5 = 25 per element)
    int cells_per_elem = (N1D - 1) * (N1D - 1);
    Index total_cells = num_elems * cells_per_elem;

    // VTK header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << total_pts << "\" NumberOfCells=\"" << total_cells << "\">\n";

    // Points: control point positions with z from solution
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
        const QuadBounds& bounds = quadtree_->element_bounds(e);
        VecX coeffs = element_coefficients(e);

        for (int dof = 0; dof < NDOF; ++dof) {
            Vec2 uv = basis_->control_point_position(dof);
            Real x = bounds.xmin + uv(0) * (bounds.xmax - bounds.xmin);
            Real y = bounds.ymin + uv(1) * (bounds.ymax - bounds.ymin);
            Real z = coeffs(dof);

            file << std::setprecision(12) << x << " " << y << " " << z << "\n";
        }
    }

    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells: quads connecting control points
    file << "<Cells>\n";
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";

    Index pt_offset = 0;
    for (Index e = 0; e < num_elems; ++e) {
        for (int j = 0; j < N1D - 1; ++j) {
            for (int i = 0; i < N1D - 1; ++i) {
                // DOF indexing: dof = i + N1D * j
                Index p0 = pt_offset + i + N1D * j;
                Index p1 = pt_offset + (i + 1) + N1D * j;
                Index p2 = pt_offset + (i + 1) + N1D * (j + 1);
                Index p3 = pt_offset + i + N1D * (j + 1);

                file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
        pt_offset += NDOF;
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";

    for (Index c = 1; c <= total_cells; ++c) {
        file << 4 * c << "\n";
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";

    for (Index c = 0; c < total_cells; ++c) {
        file << "9\n";  // VTK_QUAD
    }

    file << "</DataArray>\n";
    file << "</Cells>\n";

    // Point data: control point z-value (depth)
    file << "<PointData Scalars=\"depth\">\n";
    file << "<DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
        VecX coeffs = element_coefficients(e);
        for (int dof = 0; dof < NDOF; ++dof) {
            file << std::setprecision(12) << coeffs(dof) << "\n";
        }
    }

    file << "</DataArray>\n";
    file << "</PointData>\n";

    // Cell data: element index
    file << "<CellData Scalars=\"element\">\n";
    file << "<DataArray type=\"Int64\" Name=\"element\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
        for (int c = 0; c < cells_per_elem; ++c) {
            file << e << "\n";
        }
    }

    file << "</DataArray>\n";
    file << "</CellData>\n";

    // Footer
    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";
}

void BezierBathymetrySmoother::write_vtk(const std::string& filename, [[maybe_unused]] int resolution) const {
    if (!solved_) {
        throw std::runtime_error("BezierBathymetrySmoother: solve() not called");
    }

    std::ofstream file(filename + ".vtu");
    if (!file) {
        throw std::runtime_error("BezierBathymetrySmoother: cannot open " + filename + ".vtu");
    }

    // Use 11x11 LGL points per element (degree 10 polynomial grid)
    constexpr int n_lgl = 11;
    VecX lgl_nodes, lgl_weights;
    compute_gauss_lobatto_nodes(n_lgl, lgl_nodes, lgl_weights);

    // Map LGL nodes from [-1, 1] to [0, 1] for Bezier parameter space
    VecX param_nodes = (lgl_nodes.array() + 1.0) * 0.5;

    // Count points and cells
    Index num_elems = quadtree_->num_elements();
    int pts_per_elem = n_lgl * n_lgl;
    int cells_per_elem = (n_lgl - 1) * (n_lgl - 1);

    Index total_pts = num_elems * pts_per_elem;
    Index total_cells = num_elems * cells_per_elem;

    // VTK header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << total_pts << "\" NumberOfCells=\"" << total_cells << "\">\n";

    // Points
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
        const QuadBounds& bounds = quadtree_->element_bounds(e);
        VecX coeffs = element_coefficients(e);

        for (int j = 0; j < n_lgl; ++j) {
            for (int i = 0; i < n_lgl; ++i) {
                Real u = param_nodes(i);
                Real v = param_nodes(j);

                Real x = bounds.xmin + u * (bounds.xmax - bounds.xmin);
                Real y = bounds.ymin + v * (bounds.ymax - bounds.ymin);
                Real z = basis_->evaluate_scalar(coeffs, u, v);

                file << std::setprecision(12) << x << " " << y << " " << z << "\n";
            }
        }
    }

    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells (quads)
    file << "<Cells>\n";
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";

    Index pt_offset = 0;
    for (Index e = 0; e < num_elems; ++e) {
        for (int j = 0; j < n_lgl - 1; ++j) {
            for (int i = 0; i < n_lgl - 1; ++i) {
                Index p0 = pt_offset + i + n_lgl * j;
                Index p1 = p0 + 1;
                Index p2 = p0 + n_lgl + 1;
                Index p3 = p0 + n_lgl;

                file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
        pt_offset += pts_per_elem;
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";

    for (Index c = 1; c <= total_cells; ++c) {
        file << 4 * c << "\n";
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";

    for (Index c = 0; c < total_cells; ++c) {
        file << "9\n";  // VTK_QUAD
    }

    file << "</DataArray>\n";
    file << "</Cells>\n";

    // Point data: depth
    file << "<PointData Scalars=\"depth\">\n";
    file << "<DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
        VecX coeffs = element_coefficients(e);

        for (int j = 0; j < n_lgl; ++j) {
            for (int i = 0; i < n_lgl; ++i) {
                Real u = param_nodes(i);
                Real v = param_nodes(j);
                Real z = basis_->evaluate_scalar(coeffs, u, v);
                file << std::setprecision(12) << z << "\n";
            }
        }
    }

    file << "</DataArray>\n";
    file << "</PointData>\n";

    // Cell data: element index
    file << "<CellData Scalars=\"element\">\n";
    file << "<DataArray type=\"Int64\" Name=\"element\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elems; ++e) {
        for (int c = 0; c < cells_per_elem; ++c) {
            file << e << "\n";
        }
    }

    file << "</DataArray>\n";
    file << "</CellData>\n";

    // Footer
    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";
}

Real BezierBathymetrySmoother::data_residual() const {
    if (!solved_) return 0.0;

    // Rebuild evaluation if needed
    if (!cache_valid_) {
        data_assembler_->assemble(B_cached_, b_cached_, w_cached_);
        cache_valid_ = true;
    }

    VecX residual = B_cached_ * solution_ - b_cached_;
    Real sum = 0.0;
    for (Index i = 0; i < residual.size(); ++i) {
        sum += w_cached_(i) * residual(i) * residual(i);
    }
    return sum;
}

Real BezierBathymetrySmoother::regularization_energy() const {
    if (!solved_) return 0.0;

    // Sum element energies
    Real total = 0.0;
    for (Index e = 0; e < quadtree_->num_elements(); ++e) {
        VecX coeffs = element_coefficients(e);
        Vec2 size = quadtree_->element_size(e);
        MatX H_scaled = hessian_->scaled_hessian(size(0), size(1));
        total += coeffs.transpose() * H_scaled * coeffs;
    }
    return total;
}

Real BezierBathymetrySmoother::objective_value() const {
    return data_residual() + config_.lambda * regularization_energy();
}

Real BezierBathymetrySmoother::constraint_violation() const {
    if (!solved_) return 0.0;

    // C² constraint violation
    SpMat A_c2 = constraint_builder_->build_constraint_matrix();
    VecX c2_violation = A_c2 * solution_;
    Real total_sq = c2_violation.squaredNorm();

    // Dirichlet constraint violation: solution(dof) should equal depth
    if (config_.enable_boundary_dirichlet && data_assembler_->has_bathymetry_function()) {
        const auto& dir_constraints = constraint_builder_->dirichlet_constraints();

        for (const auto& info : dir_constraints) {
            Real expected = data_assembler_->evaluate_bathymetry(
                info.position(0), info.position(1));
            Real violation = solution_(info.global_dof) - expected;
            total_sq += violation * violation;
        }
    }

    return std::sqrt(total_sq);
}

}  // namespace drifter
