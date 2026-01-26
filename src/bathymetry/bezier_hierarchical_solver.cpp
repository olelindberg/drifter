#include "bathymetry/bezier_hierarchical_solver.hpp"
#include "bathymetry/bezier_bathymetry_smoother.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace drifter {

using Eigen::Triplet;

BezierHierarchicalSolver::BezierHierarchicalSolver(
    const QuadtreeAdapter &mesh,
    const BezierSmootherConfig &config,
    const HierarchicalConfig &hlc_config)
    : mesh_(mesh), config_(config), hlc_config_(hlc_config) {
  // Initialize owned components
  basis_ = std::make_unique<BezierBasis2D>();
  hessian_ = std::make_unique<ThinPlateHessian>(config_.ngauss_energy,
                                                 config_.gradient_weight);

  // Build level hierarchy
  build_level_hierarchy();
}

// ============================================================================
// Hierarchy Building
// ============================================================================

void BezierHierarchicalSolver::build_level_hierarchy() {
  // Group elements by their maximum refinement level
  std::map<int, std::vector<Index>> elements_by_level;
  max_level_ = 0;

  for (Index e = 0; e < mesh_.num_elements(); ++e) {
    QuadLevel level = mesh_.element_level(e);
    int max_ref_level = level.max_level();
    elements_by_level[max_ref_level].push_back(e);
    max_level_ = std::max(max_level_, max_ref_level);
  }

  // Create SubdomainInfo for each level
  subdomains_.clear();
  for (auto &[level, elements] : elements_by_level) {
    SubdomainInfo info;
    info.level = level;
    info.elements = std::move(elements);
    info.boundary_dofs = find_boundary_dofs(info);

    // Compute interior DOFs (all DOFs not on boundary)
    std::set<Index> boundary_set(info.boundary_dofs.begin(),
                                  info.boundary_dofs.end());
    for (Index elem : info.elements) {
      for (int local = 0; local < BezierBasis2D::NDOF; ++local) {
        Index dof = global_dof(elem, local);
        if (boundary_set.find(dof) == boundary_set.end()) {
          info.interior_dofs.push_back(dof);
        }
      }
    }

    subdomains_[level] = std::move(info);
  }

  if (hlc_config_.verbose) {
    std::cout << "[HLC] Built hierarchy with " << (max_level_ + 1)
              << " levels:\n";
    for (const auto &[level, info] : subdomains_) {
      std::cout << "  Level " << level << ": " << info.elements.size()
                << " elements, " << info.boundary_dofs.size() << " boundary DOFs, "
                << info.interior_dofs.size() << " interior DOFs\n";
    }
  }
}

std::vector<Index> BezierHierarchicalSolver::find_boundary_dofs(
    const SubdomainInfo &subdomain) const {
  // Boundary DOFs are on edges where:
  // 1. EdgeNeighborInfo.type == FineToCoarse (refinement boundary)
  // 2. EdgeNeighborInfo.type == Boundary (domain boundary)

  std::set<Index> boundary_set;

  // Build set of elements in this subdomain for quick lookup
  std::set<Index> subdomain_elements(subdomain.elements.begin(),
                                      subdomain.elements.end());

  for (Index elem : subdomain.elements) {
    for (int edge = 0; edge < 4; ++edge) {
      EdgeNeighborInfo info = mesh_.get_neighbor(elem, edge);

      bool is_boundary_edge = false;

      if (info.type == EdgeNeighborInfo::Type::Boundary) {
        // Domain boundary
        is_boundary_edge = true;
      } else if (info.type == EdgeNeighborInfo::Type::FineToCoarse) {
        // Refinement boundary (neighbor is coarser)
        is_boundary_edge = true;
      } else if (info.type == EdgeNeighborInfo::Type::Conforming) {
        // Check if neighbor is in a different (coarser) level
        if (!info.neighbor_elements.empty()) {
          Index neighbor = info.neighbor_elements[0];
          if (subdomain_elements.find(neighbor) == subdomain_elements.end()) {
            // Neighbor is not in this subdomain
            is_boundary_edge = true;
          }
        }
      }

      if (is_boundary_edge) {
        // Add all DOFs on this edge
        std::vector<int> edge_dof_list = basis_->edge_dofs(edge);
        for (int local_dof : edge_dof_list) {
          boundary_set.insert(global_dof(elem, local_dof));
        }
      }
    }
  }

  // Convert to sorted vector
  return std::vector<Index>(boundary_set.begin(), boundary_set.end());
}

// ============================================================================
// Main Solve
// ============================================================================

void BezierHierarchicalSolver::solve() {
  if (!data_assembler_) {
    throw std::runtime_error(
        "BezierHierarchicalSolver: data assembler not set");
  }

  // Check for mixed-level interfaces (elements at different levels sharing edges)
  // The hierarchical solver assumes a nested hierarchy where fine elements are
  // children of coarse elements. If elements at different levels share edges
  // (non-nested AMR), the hierarchical approach won't enforce C² continuity
  // at those interfaces.
  bool has_mixed_level_interfaces = false;
  for (Index e = 0; e < mesh_.num_elements(); ++e) {
    QuadLevel level_e = mesh_.element_level(e);
    int level_e_max = level_e.max_level();

    for (int edge = 0; edge < 4; ++edge) {
      EdgeNeighborInfo info = mesh_.get_neighbor(e, edge);

      // Check conforming interfaces with different levels
      if (info.type == EdgeNeighborInfo::Type::Conforming &&
          !info.neighbor_elements.empty()) {
        Index neighbor = info.neighbor_elements[0];
        QuadLevel level_n = mesh_.element_level(neighbor);
        int level_n_max = level_n.max_level();

        if (level_e_max != level_n_max) {
          has_mixed_level_interfaces = true;
          break;
        }
      }

      // Also check non-conforming interfaces (FineToCoarse/CoarseToFine)
      // These connect elements at different levels
      if (info.type == EdgeNeighborInfo::Type::FineToCoarse ||
          info.type == EdgeNeighborInfo::Type::CoarseToFine) {
        has_mixed_level_interfaces = true;
        break;
      }
    }
    if (has_mixed_level_interfaces) break;
  }

  if (has_mixed_level_interfaces && hlc_config_.verbose) {
    std::cout << "[HLC] WARNING: Mesh has mixed-level interfaces (elements at "
              << "different levels sharing edges). C² continuity may not be "
              << "enforced at these interfaces. Consider using direct solver.\n";
  }

  // Initialize solution vector
  Index total_dofs = mesh_.num_elements() * BezierBasis2D::NDOF;
  solution_ = VecX::Zero(total_dofs);
  laplacian_ = VecX::Zero(total_dofs);
  residual_history_.clear();

  if (hlc_config_.verbose) {
    std::cout << "[HLC] Starting hierarchical solve with " << total_dofs
              << " DOFs\n";
  }

  // Step 1: Solve global coarse problem (level 0 or minimum level)
  int min_level = subdomains_.begin()->first;
  if (hlc_config_.verbose) {
    std::cout << "[HLC] Solving coarse global problem (level " << min_level
              << ")\n";
  }
  solve_coarse_global();
  compute_laplacian();

  // Step 2: Process each level from coarsest+1 to finest
  for (int level = min_level + 1; level <= max_level_; ++level) {
    if (subdomains_.find(level) == subdomains_.end()) {
      continue; // Skip if no elements at this level
    }

    // Initialize fine level DOFs by interpolating from coarse level
    // This ensures b_fine starts with coarse values before corrections are added
    if (hlc_config_.verbose) {
      std::cout << "[HLC] Interpolating level " << level << " from coarse\n";
    }
    interpolate_from_coarse_level(level);

    if (hlc_config_.verbose) {
      std::cout << "[HLC] Solving level " << level << " corrections\n";
    }
    solve_level_corrections(level);
    compute_laplacian();
  }

  if (hlc_config_.verbose) {
    std::cout << "[HLC] Hierarchical solve complete\n";
  }
}

// ============================================================================
// Coarse Global Solve
// ============================================================================

void BezierHierarchicalSolver::solve_coarse_global() {
  // For the coarsest level, solve a standard Bezier bathymetry problem
  // on all elements at that level

  int min_level = subdomains_.begin()->first;
  const SubdomainInfo &subdomain = subdomains_.at(min_level);

  if (subdomain.elements.empty()) {
    return;
  }

  // Build mapping from global DOFs to subdomain DOFs
  std::map<Index, int> global_to_local;
  Index n_local = subdomain.elements.size() * BezierBasis2D::NDOF;
  for (size_t i = 0; i < subdomain.elements.size(); ++i) {
    Index elem = subdomain.elements[i];
    for (int local = 0; local < BezierBasis2D::NDOF; ++local) {
      Index gdof = global_dof(elem, local);
      global_to_local[gdof] = static_cast<int>(i * BezierBasis2D::NDOF + local);
    }
  }

  // Build local Hessian
  SpMat H_local = build_local_hessian(subdomain);

  // Build local data fitting (fitting to actual data, not residual)
  SpMat AtWA_local;
  VecX AtWb_local;

  // For coarse solve, we fit to the actual data
  std::vector<Triplet<Real>> atwA_triplets;
  atwA_triplets.reserve(n_local * 36);

  VecX atWb = VecX::Zero(n_local);

  // Sample data at Gauss points within coarse elements
  for (size_t elem_idx = 0; elem_idx < subdomain.elements.size(); ++elem_idx) {
    Index elem = subdomain.elements[elem_idx];
    QuadBounds bounds = mesh_.element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real area = dx * dy;

    // Gauss quadrature points
    const int ngauss = config_.ngauss_data;
    std::vector<Real> gauss_pts, gauss_wts;
    // Simple Gauss-Legendre points for [0,1]
    if (ngauss == 2) {
      gauss_pts = {0.211324865405187, 0.788675134594813};
      gauss_wts = {0.5, 0.5};
    } else if (ngauss == 3) {
      gauss_pts = {0.112701665379258, 0.5, 0.887298334620742};
      gauss_wts = {0.277777777777778, 0.444444444444444, 0.277777777777778};
    } else {
      // Default: uniform points
      for (int i = 0; i < ngauss; ++i) {
        gauss_pts.push_back((i + 0.5) / ngauss);
        gauss_wts.push_back(1.0 / ngauss);
      }
    }

    // Element-local dense matrices
    MatX B_elem(ngauss * ngauss, BezierBasis2D::NDOF);
    VecX d_elem(ngauss * ngauss);
    VecX w_elem(ngauss * ngauss);

    int pt_idx = 0;
    for (int j = 0; j < ngauss; ++j) {
      for (int i = 0; i < ngauss; ++i) {
        Real u = gauss_pts[i];
        Real v = gauss_pts[j];
        Real x = bounds.xmin + u * dx;
        Real y = bounds.ymin + v * dy;

        // Evaluate basis
        VecX phi = basis_->evaluate(u, v);
        B_elem.row(pt_idx) = phi.transpose();

        // Evaluate data
        d_elem(pt_idx) = data_assembler_->evaluate_bathymetry(x, y);

        // Weight
        w_elem(pt_idx) = area * gauss_wts[i] * gauss_wts[j];

        pt_idx++;
      }
    }

    // Build B^T W B and B^T W d for this element
    MatX BtWB = B_elem.transpose() * w_elem.asDiagonal() * B_elem;
    VecX BtWd = B_elem.transpose() * (w_elem.asDiagonal() * d_elem);

    // Add to global sparse matrix (local indexing)
    Index base = elem_idx * BezierBasis2D::NDOF;
    for (int i = 0; i < BezierBasis2D::NDOF; ++i) {
      for (int j = 0; j < BezierBasis2D::NDOF; ++j) {
        if (std::abs(BtWB(i, j)) > 1e-16) {
          atwA_triplets.emplace_back(base + i, base + j, BtWB(i, j));
        }
      }
      atWb(base + i) += BtWd(i);
    }
  }

  AtWA_local.resize(n_local, n_local);
  AtWA_local.setFromTriplets(atwA_triplets.begin(), atwA_triplets.end());
  AtWb_local = atWb;

  // Ridge regularization
  const Real ridge_lambda = 1e-4;
  for (Index i = 0; i < n_local; ++i) {
    AtWA_local.coeffRef(i, i) += ridge_lambda;
  }

  // Compute scale factor and build Q
  Real alpha = compute_scale_factor(H_local, AtWA_local);
  SpMat Q = alpha * H_local + config_.lambda * AtWA_local;
  VecX c = -config_.lambda * AtWb_local;

  // Build local C² constraints
  SpMat A_local = build_local_constraints(subdomain);
  Index m = A_local.rows();

  // Solve KKT system
  VecX x_local;
  if (m > 0) {
    x_local = solve_local_kkt(Q, c, A_local, VecX::Zero(m));
  } else {
    // No constraints - direct solve
    Eigen::SparseLU<SpMat> solver;
    solver.compute(Q);
    x_local = solver.solve(-c);
  }

  // Copy local solution to global solution
  for (const auto &[gdof, ldof] : global_to_local) {
    solution_(gdof) = x_local(ldof);
  }
}

// ============================================================================
// Level Corrections
// ============================================================================

void BezierHierarchicalSolver::solve_level_corrections(int level) {
  const SubdomainInfo &subdomain = subdomains_.at(level);

  if (subdomain.elements.empty()) {
    return;
  }

  // Solve for local correction delta_b
  VecX delta_b = solve_local_problem(subdomain);

  // Update solution: b_L = b_{L-1} + delta_b
  // delta_b is in subdomain-local ordering, need to map back to global
  std::map<Index, int> global_to_local;
  for (size_t i = 0; i < subdomain.elements.size(); ++i) {
    Index elem = subdomain.elements[i];
    for (int local = 0; local < BezierBasis2D::NDOF; ++local) {
      Index gdof = global_dof(elem, local);
      global_to_local[gdof] = static_cast<int>(i * BezierBasis2D::NDOF + local);
    }
  }

  for (const auto &[gdof, ldof] : global_to_local) {
    solution_(gdof) += delta_b(ldof);
  }
}

VecX BezierHierarchicalSolver::solve_local_problem(
    const SubdomainInfo &subdomain) {
  Index n_local = subdomain.elements.size() * BezierBasis2D::NDOF;

  // Build mapping from global to local DOFs
  std::map<Index, int> global_to_local;
  for (size_t i = 0; i < subdomain.elements.size(); ++i) {
    Index elem = subdomain.elements[i];
    for (int local = 0; local < BezierBasis2D::NDOF; ++local) {
      Index gdof = global_dof(elem, local);
      global_to_local[gdof] = static_cast<int>(i * BezierBasis2D::NDOF + local);
    }
  }

  // 1. Build local thin plate Hessian
  SpMat H_local = build_local_hessian(subdomain);

  // 2. Build local data fitting with residual (d - b_{L-1})
  SpMat AtWA_local;
  VecX AtWb_local;
  build_local_data_fitting_residual(subdomain, AtWA_local, AtWb_local);

  // Ridge regularization
  const Real ridge_lambda = 1e-4;
  for (Index i = 0; i < n_local; ++i) {
    AtWA_local.coeffRef(i, i) += ridge_lambda;
  }

  // 3. Combine: Q = alpha*H + lambda*AtWA, c = -lambda*AtWb
  Real alpha = compute_scale_factor(H_local, AtWA_local);
  SpMat Q = alpha * H_local + config_.lambda * AtWA_local;
  VecX c = -config_.lambda * AtWb_local;

  // 4. Build local C² constraints (within subdomain only)
  SpMat A_local = build_local_constraints(subdomain);
  VecX b_constraints = VecX::Zero(A_local.rows());

  // 5. Apply boundary conditions: delta_b = 0 at boundary DOFs
  // Convert boundary DOFs to local indices
  std::vector<Index> local_boundary_dofs;
  for (Index gdof : subdomain.boundary_dofs) {
    auto it = global_to_local.find(gdof);
    if (it != global_to_local.end()) {
      local_boundary_dofs.push_back(it->second);
    }
  }
  std::sort(local_boundary_dofs.begin(), local_boundary_dofs.end());

  apply_correction_boundary_conditions(Q, c, A_local, b_constraints,
                                        local_boundary_dofs);

  // 6. Solve local KKT system
  Index m = A_local.rows();
  if (m > 0) {
    return solve_local_kkt(Q, c, A_local, b_constraints);
  } else {
    // No constraints - direct solve
    Eigen::SparseLU<SpMat> solver;
    solver.compute(Q);
    return solver.solve(-c);
  }
}

// ============================================================================
// Local Problem Building
// ============================================================================

SpMat BezierHierarchicalSolver::build_local_hessian(
    const SubdomainInfo &subdomain) const {
  Index n_local = subdomain.elements.size() * BezierBasis2D::NDOF;
  std::vector<Triplet<Real>> triplets;
  triplets.reserve(subdomain.elements.size() * BezierBasis2D::NDOF *
                   BezierBasis2D::NDOF);

  for (size_t elem_idx = 0; elem_idx < subdomain.elements.size(); ++elem_idx) {
    Index elem = subdomain.elements[elem_idx];
    Vec2 size = mesh_.element_size(elem);
    Real dx = size(0);
    Real dy = size(1);

    MatX H_elem = hessian_->scaled_hessian(dx, dy);

    Index base = elem_idx * BezierBasis2D::NDOF;
    for (int i = 0; i < BezierBasis2D::NDOF; ++i) {
      for (int j = 0; j < BezierBasis2D::NDOF; ++j) {
        if (std::abs(H_elem(i, j)) > 1e-16) {
          triplets.emplace_back(base + i, base + j, H_elem(i, j));
        }
      }
    }
  }

  SpMat H(n_local, n_local);
  H.setFromTriplets(triplets.begin(), triplets.end());
  return H;
}

void BezierHierarchicalSolver::build_local_data_fitting_residual(
    const SubdomainInfo &subdomain, SpMat &AtWA, VecX &AtWb) const {
  Index n_local = subdomain.elements.size() * BezierBasis2D::NDOF;
  std::vector<Triplet<Real>> triplets;
  triplets.reserve(n_local * 36);

  VecX atWb = VecX::Zero(n_local);

  // Build mapping for evaluating current solution
  std::map<Index, int> global_to_local;
  for (size_t i = 0; i < subdomain.elements.size(); ++i) {
    Index elem = subdomain.elements[i];
    for (int local = 0; local < BezierBasis2D::NDOF; ++local) {
      Index gdof = global_dof(elem, local);
      global_to_local[gdof] = static_cast<int>(i * BezierBasis2D::NDOF + local);
    }
  }

  // Sample data at Gauss points within subdomain elements
  for (size_t elem_idx = 0; elem_idx < subdomain.elements.size(); ++elem_idx) {
    Index elem = subdomain.elements[elem_idx];
    QuadBounds bounds = mesh_.element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;
    Real area = dx * dy;

    // Gauss quadrature points (simplified)
    const int ngauss = config_.ngauss_data;
    std::vector<Real> gauss_pts, gauss_wts;
    if (ngauss == 2) {
      gauss_pts = {0.211324865405187, 0.788675134594813};
      gauss_wts = {0.5, 0.5};
    } else if (ngauss == 3) {
      gauss_pts = {0.112701665379258, 0.5, 0.887298334620742};
      gauss_wts = {0.277777777777778, 0.444444444444444, 0.277777777777778};
    } else {
      for (int i = 0; i < ngauss; ++i) {
        gauss_pts.push_back((i + 0.5) / ngauss);
        gauss_wts.push_back(1.0 / ngauss);
      }
    }

    // Element-local matrices
    MatX B_elem(ngauss * ngauss, BezierBasis2D::NDOF);
    VecX residual_elem(ngauss * ngauss);
    VecX w_elem(ngauss * ngauss);

    // Get current element coefficients for evaluating b_{L-1}
    VecX coeffs = solution_.segment(elem * BezierBasis2D::NDOF,
                                     BezierBasis2D::NDOF);

    int pt_idx = 0;
    for (int j = 0; j < ngauss; ++j) {
      for (int i = 0; i < ngauss; ++i) {
        Real u = gauss_pts[i];
        Real v = gauss_pts[j];
        Real x = bounds.xmin + u * dx;
        Real y = bounds.ymin + v * dy;

        // Evaluate basis
        VecX phi = basis_->evaluate(u, v);
        B_elem.row(pt_idx) = phi.transpose();

        // Evaluate data
        Real data_value = data_assembler_->evaluate_bathymetry(x, y);

        // Evaluate current solution b_{L-1}
        Real current_value = coeffs.dot(phi);

        // Residual = d - b_{L-1}
        residual_elem(pt_idx) = data_value - current_value;

        // Weight
        w_elem(pt_idx) = area * gauss_wts[i] * gauss_wts[j];

        pt_idx++;
      }
    }

    // Build B^T W B and B^T W residual for this element
    MatX BtWB = B_elem.transpose() * w_elem.asDiagonal() * B_elem;
    VecX BtWr = B_elem.transpose() * (w_elem.asDiagonal() * residual_elem);

    // Add to sparse matrix
    Index base = elem_idx * BezierBasis2D::NDOF;
    for (int i = 0; i < BezierBasis2D::NDOF; ++i) {
      for (int j = 0; j < BezierBasis2D::NDOF; ++j) {
        if (std::abs(BtWB(i, j)) > 1e-16) {
          triplets.emplace_back(base + i, base + j, BtWB(i, j));
        }
      }
      atWb(base + i) += BtWr(i);
    }
  }

  AtWA.resize(n_local, n_local);
  AtWA.setFromTriplets(triplets.begin(), triplets.end());
  AtWb = atWb;
}

SpMat BezierHierarchicalSolver::build_local_constraints(
    const SubdomainInfo &subdomain) const {
  // Build C² constraints only for vertices shared within this subdomain
  // Boundary DOFs are handled separately via Dirichlet BCs

  Index n_local = subdomain.elements.size() * BezierBasis2D::NDOF;

  // Build element set and mapping
  std::set<Index> subdomain_elements(subdomain.elements.begin(),
                                      subdomain.elements.end());
  std::map<Index, int> elem_to_local;
  for (size_t i = 0; i < subdomain.elements.size(); ++i) {
    elem_to_local[subdomain.elements[i]] = static_cast<int>(i);
  }

  // Use the mesh's constraint builder to find shared vertices
  BezierC2ConstraintBuilder local_builder(mesh_);

  // Filter to only include constraints between elements in this subdomain
  std::vector<Triplet<Real>> triplets;
  Index constraint_idx = 0;

  // Get constraint info from the mesh's builder
  const auto &vertex_constraints = local_builder.vertex_constraints();

  for (const auto &vc : vertex_constraints) {
    // Check if both elements are in this subdomain
    if (subdomain_elements.find(vc.elem1) == subdomain_elements.end() ||
        subdomain_elements.find(vc.elem2) == subdomain_elements.end()) {
      continue;
    }

    // Both elements are in subdomain - add constraints
    int local_elem1 = elem_to_local[vc.elem1];
    int local_elem2 = elem_to_local[vc.elem2];

    // Get element sizes for scaling
    Vec2 size1 = mesh_.element_size(vc.elem1);
    Vec2 size2 = mesh_.element_size(vc.elem2);
    Real dx1 = size1(0), dy1 = size1(1);
    Real dx2 = size2(0), dy2 = size2(1);

    // Get corner parameters
    Vec2 uv1 = basis_->corner_param(vc.corner1);
    Vec2 uv2 = basis_->corner_param(vc.corner2);

    // C² constraints: match 9 derivatives at shared vertex
    const std::vector<std::pair<int, int>> deriv_orders = {
        {0, 0}, // z
        {1, 0}, // z_u
        {0, 1}, // z_v
        {2, 0}, // z_uu
        {1, 1}, // z_uv
        {0, 2}, // z_vv
        {2, 1}, // z_uuv
        {1, 2}, // z_uvv
        {2, 2}  // z_uuvv
    };

    for (const auto &[nu, nv] : deriv_orders) {
      Real scale1 = std::pow(dx1, nu) * std::pow(dy1, nv);
      Real scale2 = std::pow(dx2, nu) * std::pow(dy2, nv);

      VecX phi1 = basis_->evaluate_derivative(uv1(0), uv1(1), nu, nv);
      VecX phi2 = basis_->evaluate_derivative(uv2(0), uv2(1), nu, nv);

      Index base1 = local_elem1 * BezierBasis2D::NDOF;
      Index base2 = local_elem2 * BezierBasis2D::NDOF;

      for (int k = 0; k < BezierBasis2D::NDOF; ++k) {
        if (std::abs(phi1(k)) > 1e-16) {
          triplets.emplace_back(constraint_idx, base1 + k, phi1(k) / scale1);
        }
        if (std::abs(phi2(k)) > 1e-16) {
          triplets.emplace_back(constraint_idx, base2 + k, -phi2(k) / scale2);
        }
      }
      constraint_idx++;
    }
  }

  SpMat A(constraint_idx, n_local);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

void BezierHierarchicalSolver::apply_correction_boundary_conditions(
    SpMat &Q, VecX &c, SpMat &A_constraints, VecX &b_constraints,
    const std::vector<Index> &boundary_dofs) const {
  // Apply simple Dirichlet boundary conditions: delta_b = 0 at boundary DOFs.
  // Each boundary DOF gets ONE constraint: x[dof] = 0
  // This ensures the correction is zero at the boundary, maintaining C²
  // continuity with the coarser level solution.

  if (boundary_dofs.empty()) {
    return;
  }

  Index n = Q.rows();
  Index num_new_constraints = static_cast<Index>(boundary_dofs.size());
  Index old_num_constraints = A_constraints.rows();
  Index total_constraints = old_num_constraints + num_new_constraints;

  std::vector<Triplet<Real>> a_triplets;
  a_triplets.reserve(A_constraints.nonZeros() + num_new_constraints);

  // Copy existing constraints
  for (int k = 0; k < A_constraints.outerSize(); ++k) {
    for (SpMat::InnerIterator it(A_constraints, k); it; ++it) {
      a_triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  // Add one simple constraint per boundary DOF: x[dof] = 0
  Index constraint_row = old_num_constraints;
  for (Index local_bdof : boundary_dofs) {
    a_triplets.emplace_back(constraint_row, local_bdof, 1.0);
    constraint_row++;
  }

  // Rebuild constraint matrix with new rows
  A_constraints.resize(total_constraints, n);
  A_constraints.setFromTriplets(a_triplets.begin(), a_triplets.end());

  // Extend b_constraints with zeros for the new constraints (delta_b = 0)
  VecX new_b = VecX::Zero(total_constraints);
  new_b.head(old_num_constraints) = b_constraints;
  b_constraints = new_b;
}

VecX BezierHierarchicalSolver::solve_local_kkt(const SpMat &Q, const VecX &c,
                                                const SpMat &A,
                                                const VecX &b) const {
  Index n = Q.rows();
  Index m = A.rows();

  if (m == 0) {
    Eigen::SparseLU<SpMat> solver;
    solver.compute(Q);
    return solver.solve(-c);
  }

  // Build KKT system
  SpMat KKT(n + m, n + m);
  std::vector<Triplet<Real>> triplets;
  triplets.reserve(Q.nonZeros() + 2 * A.nonZeros() + m);

  // Q block
  for (int k = 0; k < Q.outerSize(); ++k) {
    for (SpMat::InnerIterator it(Q, k); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  // A block
  for (int k = 0; k < A.outerSize(); ++k) {
    for (SpMat::InnerIterator it(A, k); it; ++it) {
      triplets.emplace_back(n + it.row(), it.col(), it.value());
    }
  }

  // A^T block
  for (int k = 0; k < A.outerSize(); ++k) {
    for (SpMat::InnerIterator it(A, k); it; ++it) {
      triplets.emplace_back(it.col(), n + it.row(), it.value());
    }
  }

  // Small regularization in (2,2) block
  const Real reg = 1e-14;
  for (Index i = 0; i < m; ++i) {
    triplets.emplace_back(n + i, n + i, -reg);
  }

  KKT.setFromTriplets(triplets.begin(), triplets.end());

  // Build RHS
  VecX rhs(n + m);
  rhs.head(n) = -c;
  rhs.tail(m) = b;

  // Solve
  Eigen::SparseLU<SpMat> solver;
  solver.compute(KKT);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("BezierHierarchicalSolver: KKT factorization failed");
  }

  VecX sol = solver.solve(rhs);
  return sol.head(n);
}

// ============================================================================
// Laplacian Computation
// ============================================================================

void BezierHierarchicalSolver::compute_laplacian() {
  // Compute sigma = z_xx + z_yy at each DOF location
  laplacian_.resize(solution_.size());

  for (Index elem = 0; elem < mesh_.num_elements(); ++elem) {
    VecX coeffs = solution_.segment(elem * BezierBasis2D::NDOF,
                                     BezierBasis2D::NDOF);
    QuadBounds bounds = mesh_.element_bounds(elem);
    Real dx = bounds.xmax - bounds.xmin;
    Real dy = bounds.ymax - bounds.ymin;

    for (int local = 0; local < BezierBasis2D::NDOF; ++local) {
      Vec2 uv = basis_->control_point_position(local);

      VecX phi_uu = basis_->evaluate_d2u(uv(0), uv(1));
      VecX phi_vv = basis_->evaluate_d2v(uv(0), uv(1));

      Real z_uu = coeffs.dot(phi_uu) / (dx * dx);
      Real z_vv = coeffs.dot(phi_vv) / (dy * dy);

      Index gdof = global_dof(elem, local);
      laplacian_(gdof) = z_uu + z_vv;
    }
  }
}

Real BezierHierarchicalSolver::evaluate_laplacian_in_element(Index elem, Real u,
                                                              Real v) const {
  VecX coeffs = solution_.segment(elem * BezierBasis2D::NDOF,
                                   BezierBasis2D::NDOF);
  QuadBounds bounds = mesh_.element_bounds(elem);
  Real dx = bounds.xmax - bounds.xmin;
  Real dy = bounds.ymax - bounds.ymin;

  VecX phi_uu = basis_->evaluate_d2u(u, v);
  VecX phi_vv = basis_->evaluate_d2v(u, v);

  Real z_uu = coeffs.dot(phi_uu) / (dx * dx);
  Real z_vv = coeffs.dot(phi_vv) / (dy * dy);

  return z_uu + z_vv;
}

Real BezierHierarchicalSolver::evaluate_laplacian(Real x, Real y) const {
  Index elem = find_element(x, y);
  if (elem < 0) {
    return 0.0;
  }

  QuadBounds bounds = mesh_.element_bounds(elem);
  Real dx = bounds.xmax - bounds.xmin;
  Real dy = bounds.ymax - bounds.ymin;

  Real u = (x - bounds.xmin) / dx;
  Real v = (y - bounds.ymin) / dy;

  return evaluate_laplacian_in_element(elem, u, v);
}

// ============================================================================
// Utility Methods
// ============================================================================

Index BezierHierarchicalSolver::find_element(Real x, Real y) const {
  for (Index e = 0; e < mesh_.num_elements(); ++e) {
    QuadBounds bounds = mesh_.element_bounds(e);
    if (x >= bounds.xmin - 1e-10 && x <= bounds.xmax + 1e-10 &&
        y >= bounds.ymin - 1e-10 && y <= bounds.ymax + 1e-10) {
      return e;
    }
  }
  return -1;
}

Index BezierHierarchicalSolver::find_coarse_element(Real x, Real y,
                                                     int max_level) const {
  // Search through subdomains at or below max_level
  for (const auto &[level, subdomain] : subdomains_) {
    if (level > max_level) {
      continue;
    }
    for (Index elem : subdomain.elements) {
      QuadBounds bounds = mesh_.element_bounds(elem);
      if (x >= bounds.xmin - 1e-10 && x <= bounds.xmax + 1e-10 &&
          y >= bounds.ymin - 1e-10 && y <= bounds.ymax + 1e-10) {
        return elem;
      }
    }
  }
  return -1;
}

void BezierHierarchicalSolver::interpolate_from_coarse_level(int fine_level) {
  if (subdomains_.find(fine_level) == subdomains_.end()) {
    return;
  }

  const SubdomainInfo &fine_subdomain = subdomains_.at(fine_level);

  int interpolated_from_coarse = 0;
  int interpolated_from_data = 0;

  for (Index fine_elem : fine_subdomain.elements) {
    QuadBounds fine_bounds = mesh_.element_bounds(fine_elem);
    Real fine_dx = fine_bounds.xmax - fine_bounds.xmin;
    Real fine_dy = fine_bounds.ymax - fine_bounds.ymin;

    // For each DOF in fine element
    for (int local = 0; local < BezierBasis2D::NDOF; ++local) {
      Vec2 uv = basis_->control_point_position(local);

      // Physical position of this DOF
      Real x = fine_bounds.xmin + uv(0) * fine_dx;
      Real y = fine_bounds.ymin + uv(1) * fine_dy;

      // Find coarse element containing this DOF position
      Index coarse_elem = find_coarse_element(x, y, fine_level - 1);
      Index fine_dof = global_dof(fine_elem, local);

      if (coarse_elem >= 0) {
        // Interpolate from coarse element
        QuadBounds coarse_bounds = mesh_.element_bounds(coarse_elem);
        Real coarse_dx = coarse_bounds.xmax - coarse_bounds.xmin;
        Real coarse_dy = coarse_bounds.ymax - coarse_bounds.ymin;

        VecX coarse_coeffs =
            solution_.segment(coarse_elem * BezierBasis2D::NDOF, BezierBasis2D::NDOF);

        // Parameter position in coarse element
        Real u_coarse = (x - coarse_bounds.xmin) / coarse_dx;
        Real v_coarse = (y - coarse_bounds.ymin) / coarse_dy;

        // Clamp to [0,1] to handle numerical precision issues
        u_coarse = std::max(0.0, std::min(1.0, u_coarse));
        v_coarse = std::max(0.0, std::min(1.0, v_coarse));

        // Evaluate coarse surface at this position
        VecX phi = basis_->evaluate(u_coarse, v_coarse);
        Real z_coarse = coarse_coeffs.dot(phi);

        solution_(fine_dof) = z_coarse;
        interpolated_from_coarse++;
      } else {
        // No coarse element at this position - initialize from bathymetry data
        // This provides a good starting point for the correction solve
        if (data_assembler_) {
          Real z_data = data_assembler_->evaluate_bathymetry(x, y);
          solution_(fine_dof) = z_data;
          interpolated_from_data++;
        }
      }
    }
  }

  if (hlc_config_.verbose) {
    std::cout << "[HLC]   Interpolated " << interpolated_from_coarse
              << " DOFs from coarse, " << interpolated_from_data
              << " from data\n";
  }
}

int BezierHierarchicalSolver::global_to_subdomain_dof(
    Index global_dof, const SubdomainInfo &subdomain) const {
  Index elem = global_dof / BezierBasis2D::NDOF;
  int local = static_cast<int>(global_dof % BezierBasis2D::NDOF);

  for (size_t i = 0; i < subdomain.elements.size(); ++i) {
    if (subdomain.elements[i] == elem) {
      return static_cast<int>(i * BezierBasis2D::NDOF + local);
    }
  }
  return -1;
}

Index BezierHierarchicalSolver::subdomain_to_global_dof(
    int subdomain_dof, const SubdomainInfo &subdomain) const {
  int elem_idx = subdomain_dof / BezierBasis2D::NDOF;
  int local = subdomain_dof % BezierBasis2D::NDOF;

  if (elem_idx < 0 ||
      static_cast<size_t>(elem_idx) >= subdomain.elements.size()) {
    return -1;
  }

  Index elem = subdomain.elements[elem_idx];
  return global_dof(elem, local);
}

Real BezierHierarchicalSolver::compute_scale_factor(const SpMat &H,
                                                     const SpMat &AtWA) const {
  Real H_norm = H.norm();
  Real A_norm = AtWA.norm();
  return (H_norm > 1e-14) ? (A_norm / H_norm) : 1.0;
}

} // namespace drifter
