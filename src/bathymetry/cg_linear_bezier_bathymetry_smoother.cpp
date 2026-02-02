#include "bathymetry/cg_linear_bezier_bathymetry_smoother.hpp"
#include "bathymetry/bezier_data_fitting.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace drifter {

namespace {

void gauss_legendre_01_linear(int n, std::vector<Real> &pts,
                              std::vector<Real> &wts) {
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

CGLinearBezierBathymetrySmoother::CGLinearBezierBathymetrySmoother(
    const QuadtreeAdapter &mesh, const CGLinearBezierSmootherConfig &config)
    : quadtree_(&mesh), config_(config) {
  init_components();
}

CGLinearBezierBathymetrySmoother::CGLinearBezierBathymetrySmoother(
    const OctreeAdapter &octree, const CGLinearBezierSmootherConfig &config)
    : config_(config) {
  quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
  quadtree_ = quadtree_owned_.get();
  init_components();
}

void CGLinearBezierBathymetrySmoother::init_components() {
  basis_ = std::make_unique<LinearBezierBasis2D>();
  dirichlet_hessian_ =
      std::make_unique<DirichletHessian>(config_.ngauss_energy);
  dof_manager_ = std::make_unique<CGLinearBezierDofManager>(*quadtree_);

  solution_.setZero(dof_manager_->num_global_dofs());
}

// =============================================================================
// Data input
// =============================================================================

void CGLinearBezierBathymetrySmoother::set_bathymetry_data(
    const BathymetrySource &source) {
  set_bathymetry_data(
      [&source](Real x, Real y) { return source.evaluate(x, y); });
}

void CGLinearBezierBathymetrySmoother::set_bathymetry_data(
    std::function<Real(Real, Real)> bathy_func) {
  assemble_dirichlet_hessian();
  assemble_data_fitting(bathy_func);
  data_set_ = true;
}

void CGLinearBezierBathymetrySmoother::set_scattered_points(
    const std::vector<Vec3> &points) {
  std::vector<BathymetryPoint> bathy_points;
  bathy_points.reserve(points.size());
  for (const auto &p : points) {
    bathy_points.emplace_back(p(0), p(1), p(2), 1.0);
  }
  set_scattered_points(bathy_points);
}

void CGLinearBezierBathymetrySmoother::set_scattered_points(
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

void CGLinearBezierBathymetrySmoother::assemble_dirichlet_hessian() {
  Index num_dofs = dof_manager_->num_global_dofs();
  Index num_elements = quadtree_->num_elements();

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_elements * 4 * 4);

  for (Index elem = 0; elem < num_elements; ++elem) {
    Vec2 size = quadtree_->element_size(elem);
    Real dx = size(0);
    Real dy = size(1);

    MatX H_local = dirichlet_hessian_->scaled_hessian(dx, dy);
    const auto &global_dofs = dof_manager_->element_dofs(elem);

    for (int i = 0; i < LinearBezierBasis2D::NDOF; ++i) {
      Index I = global_dofs[i];
      for (int j = 0; j < LinearBezierBasis2D::NDOF; ++j) {
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

void CGLinearBezierBathymetrySmoother::assemble_data_fitting(
    std::function<Real(Real, Real)> bathy_func) {

  Index num_dofs = dof_manager_->num_global_dofs();
  Index num_elements = quadtree_->num_elements();
  int ngauss = config_.ngauss_data;

  std::vector<Real> gauss_pts, gauss_wts;
  gauss_legendre_01_linear(ngauss, gauss_pts, gauss_wts);

  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(num_elements * ngauss * ngauss * 4 * 4);

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

        for (int i = 0; i < LinearBezierBasis2D::NDOF; ++i) {
          Index I = global_dofs[i];
          for (int j = 0; j < LinearBezierBasis2D::NDOF; ++j) {
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

  // Compute scale normalization factor once (used in solve and objective_value)
  Real norm_BtWB = BtWB_global_.norm();
  Real norm_H = H_global_.norm();
  alpha_ = (norm_H > 1e-14) ? norm_BtWB / norm_H : 0.0;
}

// =============================================================================
// Solve
// =============================================================================

void CGLinearBezierBathymetrySmoother::solve() {
  if (!data_set_) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: bathymetry data not set");
  }

  Index num_constraints = dof_manager_->num_constraints();

  if (num_constraints == 0) {
    solve_unconstrained();
  } else {
    solve_with_constraints();
  }

  solved_ = true;
}

void CGLinearBezierBathymetrySmoother::solve_unconstrained() {
  Index num_dofs = dof_manager_->num_global_dofs();

  // Q = alpha * H_dirichlet + lambda * BtWB (alpha computed during assembly)
  SpMat Q = alpha_ * H_global_ + config_.lambda * BtWB_global_;

  // Ridge regularization
  for (Index i = 0; i < num_dofs; ++i) {
    Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
  }

  VecX c = -config_.lambda * BtWd_global_;

  Eigen::SparseLU<SpMat> solver;
  solver.compute(Q);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: SparseLU decomposition failed");
  }

  solution_ = solver.solve(-c);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: SparseLU solve failed");
  }
}

void CGLinearBezierBathymetrySmoother::solve_with_constraints() {
  Index num_dofs = dof_manager_->num_global_dofs();
  Index num_constraints = dof_manager_->num_constraints();

  // Q = alpha * H + lambda * BtWB (alpha computed during assembly)
  SpMat Q = alpha_ * H_global_ + config_.lambda * BtWB_global_;
  for (Index i = 0; i < num_dofs; ++i) {
    Q.coeffRef(i, i) += config_.lambda * config_.ridge_epsilon;
  }

  VecX c = -config_.lambda * BtWd_global_;

  // Build constraint matrix from hanging node constraints
  std::vector<Eigen::Triplet<Real>> A_triplets;
  A_triplets.reserve(num_constraints * 3);

  Index row = 0;
  for (const auto &hc : dof_manager_->constraints()) {
    A_triplets.emplace_back(row, hc.slave_dof, 1.0);
    for (size_t i = 0; i < hc.master_dofs.size(); ++i) {
      A_triplets.emplace_back(row, hc.master_dofs[i], -hc.weights[i]);
    }
    ++row;
  }

  SpMat A(num_constraints, num_dofs);
  A.setFromTriplets(A_triplets.begin(), A_triplets.end());

  // Build KKT system: [Q  A^T; A  0]
  Index kkt_size = num_dofs + num_constraints;
  std::vector<Eigen::Triplet<Real>> triplets;
  triplets.reserve(Q.nonZeros() + 2 * A.nonZeros());

  // Add Q block (top-left)
  for (int k = 0; k < Q.outerSize(); ++k) {
    for (SpMat::InnerIterator it(Q, k); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  // Add A and A^T blocks in single loop
  for (int k = 0; k < A.outerSize(); ++k) {
    for (SpMat::InnerIterator it(A, k); it; ++it) {
      triplets.emplace_back(num_dofs + it.row(), it.col(), it.value());  // A block
      triplets.emplace_back(it.col(), num_dofs + it.row(), it.value());  // A^T block
    }
  }

  SpMat KKT(kkt_size, kkt_size);
  KKT.setFromTriplets(triplets.begin(), triplets.end());

  // Small regularization on constraint block for numerical stability
  Real constraint_reg = 1e-10;
  for (Index i = num_dofs; i < kkt_size; ++i) {
    KKT.coeffRef(i, i) -= constraint_reg;
  }

  VecX rhs(kkt_size);
  rhs.head(num_dofs) = -c;
  rhs.tail(num_constraints).setZero();

  Eigen::SparseLU<SpMat> solver;
  solver.compute(KKT);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: KKT SparseLU decomposition "
        "failed");
  }

  VecX sol = solver.solve(rhs);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: KKT SparseLU solve failed");
  }

  solution_ = sol.head(num_dofs);

  // Project onto constraint manifold for numerical precision
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

Index CGLinearBezierBathymetrySmoother::find_element(Real x, Real y) const {
  return quadtree_->find_element(Vec2(x, y));
}

Index CGLinearBezierBathymetrySmoother::find_element_with_fallback(Real x,
                                                                    Real y) const {
  Index elem = find_element(x, y);
  if (elem >= 0) {
    return elem;
  }

  // Point outside domain - find closest element by center distance
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
  return closest;
}

Real CGLinearBezierBathymetrySmoother::evaluate_in_element(Index elem, Real x,
                                                           Real y) const {
  const auto &bounds = quadtree_->element_bounds(elem);

  Real u = (x - bounds.xmin) / (bounds.xmax - bounds.xmin);
  Real v = (y - bounds.ymin) / (bounds.ymax - bounds.ymin);

  u = std::clamp(u, 0.0, 1.0);
  v = std::clamp(v, 0.0, 1.0);

  VecX coeffs = element_coefficients(elem);
  return basis_->evaluate_scalar(coeffs, u, v);
}

Real CGLinearBezierBathymetrySmoother::evaluate(Real x, Real y) const {
  if (!solved_) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: must call solve() before "
        "evaluate()");
  }

  Index elem = find_element_with_fallback(x, y);
  return evaluate_in_element(elem, x, y);
}

Vec2 CGLinearBezierBathymetrySmoother::evaluate_gradient_in_element(
    Index elem, Real x, Real y) const {
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

Vec2 CGLinearBezierBathymetrySmoother::evaluate_gradient(Real x, Real y) const {
  if (!solved_) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: must call solve() before "
        "evaluate_gradient()");
  }

  Index elem = find_element_with_fallback(x, y);
  return evaluate_gradient_in_element(elem, x, y);
}

VecX CGLinearBezierBathymetrySmoother::element_coefficients(Index elem) const {
  const auto &global_dofs = dof_manager_->element_dofs(elem);
  VecX coeffs(LinearBezierBasis2D::NDOF);
  for (int i = 0; i < LinearBezierBasis2D::NDOF; ++i) {
    coeffs(i) = solution_(global_dofs[i]);
  }
  return coeffs;
}

// =============================================================================
// Transfer and output
// =============================================================================

void CGLinearBezierBathymetrySmoother::transfer_to_seabed(
    SeabedSurface &seabed) const {
  if (!solved_) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: must call solve() before "
        "transfer_to_seabed()");
  }

  for (Index elem = 0; elem < quadtree_->num_elements(); ++elem) {
    VecX coeffs = element_coefficients(elem);
    seabed.set_element_coefficients(elem, coeffs);
  }
}

void CGLinearBezierBathymetrySmoother::write_vtk(const std::string &filename,
                                                 int resolution) const {
  if (!solved_) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: must call solve() before "
        "write_vtk()");
  }

  // Use CG-aware VTK writer that deduplicates shared vertices at element
  // boundaries, producing a properly connected mesh without visual gaps
  io::write_cg_bezier_surface_vtk(
      filename, *quadtree_,
      [this](Real x, Real y) { return evaluate(x, y); },
      resolution > 0 ? resolution : 5, "elevation");
}

void CGLinearBezierBathymetrySmoother::write_control_points_vtk(
    const std::string &filename) const {
  if (!solved_) {
    throw std::runtime_error(
        "CGLinearBezierBathymetrySmoother: must call solve() before "
        "write_control_points_vtk()");
  }

  io::write_bezier_control_points_vtk(
      filename, *quadtree_, [this](Index e) { return element_coefficients(e); },
      [](int dof) {
        // Linear: control points at corners
        int i = dof % 2;
        int j = dof / 2;
        return Vec2(static_cast<Real>(i), static_cast<Real>(j));
      },
      2); // 2x2 control points
}

// =============================================================================
// Diagnostics
// =============================================================================

Real CGLinearBezierBathymetrySmoother::data_residual() const {
  if (!solved_)
    return 0.0;
  return solution_.dot(BtWB_global_ * solution_) -
         2.0 * solution_.dot(BtWd_global_) + dTWd_global_;
}

Real CGLinearBezierBathymetrySmoother::regularization_energy() const {
  if (!solved_)
    return 0.0;
  return solution_.dot(H_global_ * solution_);
}

Real CGLinearBezierBathymetrySmoother::objective_value() const {
  if (!solved_)
    return 0.0;
  return alpha_ * regularization_energy() + config_.lambda * data_residual();
}

Real CGLinearBezierBathymetrySmoother::constraint_violation() const {
  Index num_constraints = dof_manager_->num_constraints();

  if (!solved_ || num_constraints == 0)
    return 0.0;

  SpMat A = dof_manager_->build_constraint_matrix();
  VecX violation = A * solution_;
  return violation.norm();
}

} // namespace drifter
