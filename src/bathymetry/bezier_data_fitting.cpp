#include "bathymetry/bezier_data_fitting.hpp"
#include <cmath>
#include <omp.h>
#include <stdexcept>

namespace drifter {

BezierDataFittingAssembler::BezierDataFittingAssembler(
    const QuadtreeAdapter &mesh)
    : mesh_(mesh), basis_(std::make_unique<BezierBasis2D>()) {}

void BezierDataFittingAssembler::set_scattered_points(
    const std::vector<BathymetryPoint> &points) {
    points_ = points;
    assign_points_to_elements();
}

void BezierDataFittingAssembler::set_scattered_points(
    const std::vector<Vec3> &points) {
    points_.clear();
    points_.reserve(points.size());
    for (const auto &p : points) {
        points_.emplace_back(p(0), p(1), p(2), 1.0);
    }
    assign_points_to_elements();
}

void BezierDataFittingAssembler::set_from_bathymetry_source(
    const BathymetrySource &source, int ngauss) {
    // Note: Cannot safely store reference to source (lifetime issue).
    // The bathy_func_ will be set if set_from_function is called instead.
    // For Dirichlet BCs with BathymetrySource, caller should wrap in a
    // FunctionBathymetry or use set_from_function with a capturing lambda.

    compute_gauss_quadrature(ngauss);

    points_.clear();
    point_elements_.clear();

    // Sample at Gauss points in each element
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        const QuadBounds &bounds = mesh_.element_bounds(e);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real area = dx * dy;

        // Area-based weight per point (as in ShipMesh)
        Real base_weight = area / static_cast<Real>(ngauss * ngauss);

        for (int j = 0; j < ngauss; ++j) {
            for (int i = 0; i < ngauss; ++i) {
                Real u = gauss_nodes_(i);
                Real v = gauss_nodes_(j);

                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;
                Real z = source.evaluate(x, y);

                // Weight includes quadrature weight and area
                Real w = base_weight * gauss_weights_(i) * gauss_weights_(j);

                points_.emplace_back(x, y, z, w);
                point_elements_.push_back(e);
            }
        }
    }
}

void BezierDataFittingAssembler::set_from_function(
    std::function<Real(Real, Real)> bathy_func, int ngauss) {
    // Store the function for later evaluation (needed for Dirichlet BCs)
    bathy_func_ = bathy_func;

    compute_gauss_quadrature(ngauss);

    points_.clear();
    point_elements_.clear();

    // Sample at Gauss points in each element
    for (Index e = 0; e < mesh_.num_elements(); ++e) {
        const QuadBounds &bounds = mesh_.element_bounds(e);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        Real area = dx * dy;

        Real base_weight = area / static_cast<Real>(ngauss * ngauss);

        for (int j = 0; j < ngauss; ++j) {
            for (int i = 0; i < ngauss; ++i) {
                Real u = gauss_nodes_(i);
                Real v = gauss_nodes_(j);

                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;
                Real z = bathy_func(x, y);

                Real w = base_weight * gauss_weights_(i) * gauss_weights_(j);

                points_.emplace_back(x, y, z, w);
                point_elements_.push_back(e);
            }
        }
    }
}

void BezierDataFittingAssembler::assign_points_to_elements() {
    point_elements_.clear();
    point_elements_.reserve(points_.size());

    for (const auto &pt : points_) {
        Index elem = find_element(pt.x, pt.y);
        point_elements_.push_back(elem);
    }
}

Index BezierDataFittingAssembler::find_element(Real x, Real y) const {
    return mesh_.find_element(Vec2(x, y));
}

Vec2 BezierDataFittingAssembler::physical_to_param(
    Index elem, Real x, Real y) const {
    const QuadBounds &bounds = mesh_.element_bounds(elem);
    Real u = (x - bounds.xmin) / (bounds.xmax - bounds.xmin);
    Real v = (y - bounds.ymin) / (bounds.ymax - bounds.ymin);
    return Vec2(std::clamp(u, 0.0, 1.0), std::clamp(v, 0.0, 1.0));
}

void BezierDataFittingAssembler::compute_gauss_quadrature(int ngauss) {
    gauss_nodes_.resize(ngauss);
    gauss_weights_.resize(ngauss);

    // Gauss-Legendre on [0, 1]
    if (ngauss == 1) {
        gauss_nodes_ << 0.5;
        gauss_weights_ << 1.0;
    } else if (ngauss == 2) {
        Real a = 0.5 / std::sqrt(3.0);
        gauss_nodes_ << 0.5 - a, 0.5 + a;
        gauss_weights_ << 0.5, 0.5;
    } else if (ngauss == 3) {
        Real a = std::sqrt(3.0 / 5.0) / 2.0;
        gauss_nodes_ << 0.5 - a, 0.5, 0.5 + a;
        gauss_weights_ << 5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0;
    } else if (ngauss == 4) {
        Real a = std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0)) / 2.0;
        Real b = std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0)) / 2.0;
        Real wa = (18.0 + std::sqrt(30.0)) / 72.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 72.0;
        gauss_nodes_ << 0.5 - b, 0.5 - a, 0.5 + a, 0.5 + b;
        gauss_weights_ << wb, wa, wa, wb;
    } else if (ngauss == 5) {
        Real a = std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 6.0;
        Real b = std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 6.0;
        Real wa = (322.0 + 13.0 * std::sqrt(70.0)) / 1800.0;
        Real wb = (322.0 - 13.0 * std::sqrt(70.0)) / 1800.0;
        Real wc = 128.0 / 450.0;
        gauss_nodes_ << 0.5 - b, 0.5 - a, 0.5, 0.5 + a, 0.5 + b;
        gauss_weights_ << wb, wa, wc, wa, wb;
    } else if (ngauss == 6) {
        // 6-point Gauss-Legendre on [0, 1]
        Real x1 = 0.6612093864662645;
        Real x2 = 0.2386191860831969;
        Real x3 = 0.9324695142031521;
        Real w1 = 0.3607615730481386 / 2.0;
        Real w2 = 0.4679139345726910 / 2.0;
        Real w3 = 0.1713244923791704 / 2.0;

        gauss_nodes_ << (1.0 - x3) / 2.0, (1.0 - x1) / 2.0, (1.0 - x2) / 2.0,
            (1.0 + x2) / 2.0, (1.0 + x1) / 2.0, (1.0 + x3) / 2.0;
        gauss_weights_ << w3, w1, w2, w2, w1, w3;
    } else {
        throw std::invalid_argument(
            "BezierDataFittingAssembler: ngauss must be 1-6");
    }
}

void BezierDataFittingAssembler::assemble(SpMat &B, VecX &b, VecX &w) const {
    if (points_.empty()) {
        throw std::runtime_error(
            "BezierDataFittingAssembler: no data points set");
    }

    Index npoints = num_points();
    Index ndofs = total_dofs();

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(npoints * BezierBasis2D::NDOF);

    b.resize(npoints);
    w.resize(npoints);

    for (Index p = 0; p < npoints; ++p) {
        const auto &pt = points_[p];
        Index elem = point_elements_[p];

        if (elem < 0) {
            // Point outside mesh - skip with zero contribution
            b(p) = 0.0;
            w(p) = 0.0;
            continue;
        }

        // Map to parameter space
        Vec2 uv = physical_to_param(elem, pt.x, pt.y);

        // Evaluate basis at this point
        VecX phi = basis_->evaluate(uv(0), uv(1));

        // Add entries to evaluation matrix
        for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
            if (std::abs(phi(k)) > 1e-14) {
                Index col = global_dof(elem, k);
                triplets.emplace_back(p, col, phi(k));
            }
        }

        b(p) = pt.z;
        w(p) = pt.weight;
    }

    B.resize(npoints, ndofs);
    B.setFromTriplets(triplets.begin(), triplets.end());
}

void BezierDataFittingAssembler::assemble_normal_equations(
    MatX &AtWA, VecX &AtWb) const {
    if (points_.empty()) {
        throw std::runtime_error(
            "BezierDataFittingAssembler: no data points set");
    }

    Index ndofs = total_dofs();
    Index num_elements = mesh_.num_elements();

    AtWA.setZero(ndofs, ndofs);
    AtWb.setZero(ndofs);

    // Determine number of threads
    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Per-thread storage (eliminates critical section)
    std::vector<std::vector<Eigen::Triplet<Real>>> thread_triplets(num_threads);
    std::vector<std::vector<Real>> thread_rhs(
        num_threads, std::vector<Real>(ndofs, 0.0));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for nowait schedule(dynamic)
        for (Index e = 0; e < num_elements; ++e) {
            // Find all points in this element
            std::vector<Index> elem_points;
            for (Index p = 0; p < num_points(); ++p) {
                if (point_elements_[p] == e) {
                    elem_points.push_back(p);
                }
            }

            if (elem_points.empty())
                continue;

            // Build local matrices
            MatX local_AtWA =
                MatX::Zero(BezierBasis2D::NDOF, BezierBasis2D::NDOF);
            VecX local_AtWb = VecX::Zero(BezierBasis2D::NDOF);

            for (Index p : elem_points) {
                const auto &pt = points_[p];
                Vec2 uv = physical_to_param(e, pt.x, pt.y);
                VecX phi = basis_->evaluate(uv(0), uv(1));

                // AtWA += w * phi * phi^T
                local_AtWA.noalias() += pt.weight * phi * phi.transpose();

                // AtWb += w * phi * z
                local_AtWb.noalias() += pt.weight * pt.z * phi;
            }

            // Add to thread-local triplets (no locking)
            Index base = global_dof(e, 0);
            for (Index i = 0; i < BezierBasis2D::NDOF; ++i) {
                for (Index j = 0; j < BezierBasis2D::NDOF; ++j) {
                    if (std::abs(local_AtWA(i, j)) > 1e-16) {
                        thread_triplets[tid].emplace_back(
                            base + i, base + j, local_AtWA(i, j));
                    }
                }
                thread_rhs[tid][base + i] += local_AtWb(i);
            }
        }
    }

    // Merge thread-local triplets sequentially
    std::vector<Eigen::Triplet<Real>> triplets;
    for (const auto &thread_trips : thread_triplets) {
        triplets.insert(
            triplets.end(), thread_trips.begin(), thread_trips.end());
    }

    // Merge thread-local RHS vectors
    std::vector<Real> rhs_contributions(ndofs, 0.0);
    for (const auto &thread_rhs_vec : thread_rhs) {
        for (Index i = 0; i < ndofs; ++i) {
            rhs_contributions[i] += thread_rhs_vec[i];
        }
    }

    // Assemble sparse matrix from triplets
    SpMat AtWA_sparse(ndofs, ndofs);
    AtWA_sparse.setFromTriplets(triplets.begin(), triplets.end());
    AtWA = MatX(AtWA_sparse);

    // Set RHS vector
    AtWb = Eigen::Map<VecX>(rhs_contributions.data(), ndofs);
}

void BezierDataFittingAssembler::assemble_normal_equations_sparse(
    SpMat &AtWA, VecX &AtWb) const {
    if (points_.empty()) {
        throw std::runtime_error(
            "BezierDataFittingAssembler: no data points set");
    }

    Index ndofs = total_dofs();
    Index num_elements = mesh_.num_elements();

    // Determine number of threads
    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Per-thread storage for sparse assembly
    std::vector<std::vector<Eigen::Triplet<Real>>> thread_triplets(num_threads);
    std::vector<VecX> thread_rhs(num_threads, VecX::Zero(ndofs));

    // Pre-reserve triplet capacity based on element count
    // Each element contributes at most 36×36 = 1,296 triplets
    Index elements_per_thread = (num_elements + num_threads - 1) / num_threads;
    Index triplets_per_thread =
        elements_per_thread * BezierBasis2D::NDOF * BezierBasis2D::NDOF;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_triplets[tid].reserve(triplets_per_thread);

#pragma omp for nowait schedule(dynamic)
        for (Index e = 0; e < num_elements; ++e) {
            // Find all points in this element
            std::vector<Index> elem_points;
            for (Index p = 0; p < num_points(); ++p) {
                if (point_elements_[p] == e) {
                    elem_points.push_back(p);
                }
            }

            if (elem_points.empty())
                continue;

            // Build local matrices (36×36 block)
            MatX local_AtWA =
                MatX::Zero(BezierBasis2D::NDOF, BezierBasis2D::NDOF);
            VecX local_AtWb = VecX::Zero(BezierBasis2D::NDOF);

            for (Index p : elem_points) {
                const auto &pt = points_[p];
                Vec2 uv = physical_to_param(e, pt.x, pt.y);
                VecX phi = basis_->evaluate(uv(0), uv(1));

                // AtWA += w * phi * phi^T
                local_AtWA.noalias() += pt.weight * phi * phi.transpose();

                // AtWb += w * phi * z
                local_AtWb.noalias() += pt.weight * pt.z * phi;
            }

            // Add to thread-local triplets (36×36 block)
            Index base = global_dof(e, 0);
            for (Index i = 0; i < BezierBasis2D::NDOF; ++i) {
                for (Index j = 0; j < BezierBasis2D::NDOF; ++j) {
                    Real value = local_AtWA(i, j);
                    if (std::abs(value) > 1e-16) {
                        thread_triplets[tid].emplace_back(
                            base + i, base + j, value);
                    }
                }
                // Add to thread-local RHS
                thread_rhs[tid](base + i) += local_AtWb(i);
            }
        }
    }

    // Calculate total triplets for pre-reserve (Phase 4 optimization)
    size_t total_triplets = 0;
    for (const auto &t : thread_triplets) {
        total_triplets += t.size();
    }

    // Merge thread-local triplets with pre-reserve
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(total_triplets);
    for (const auto &thread_trips : thread_triplets) {
        triplets.insert(
            triplets.end(), thread_trips.begin(), thread_trips.end());
    }

    // Build sparse matrix directly (no dense intermediate)
    AtWA.resize(ndofs, ndofs);
    AtWA.setFromTriplets(triplets.begin(), triplets.end());

    // Merge thread-local RHS vectors
    AtWb.setZero(ndofs);
    for (const auto &rhs : thread_rhs) {
        AtWb += rhs;
    }
}

} // namespace drifter
