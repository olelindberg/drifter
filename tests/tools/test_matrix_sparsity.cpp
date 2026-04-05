#include "bathymetry/block_diag_approx_cg_schur_preconditioner.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/diagonal_approx_cg_schur_preconditioner.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "io/matrix_market_writer.hpp"
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

using namespace drifter;

class MatrixSparsityTest : public ::testing::TestWithParam<int> {
protected:
  std::string output_dir_ = "/tmp/drifter_matrix_sparsity";
};

TEST_P(MatrixSparsityTest, FlatSeabed) {
  int n = GetParam();
  std::string subdir = output_dir_ + "/" + std::to_string(n) + "x" +
                        std::to_string(n);
  std::filesystem::create_directories(subdir);

  // Build NxN uniform quadtree mesh on [0,100]x[0,100]
  QuadtreeAdapter mesh;
  mesh.build_uniform(0.0, 100.0, 0.0, 100.0, n, n);

  // Configure smoother with edge constraints
  CGCubicBezierSmootherConfig config;
  config.lambda = 1.0;
  config.edge_ngauss = 4;
  config.ridge_epsilon = 1e-4;

  // Create smoother and set flat bathymetry
  CGCubicBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data([](Real, Real) { return 100.0; });

  // Write DOF-to-element map
  // For each DOF: element_id if interior (belongs to one element), -1 if shared
  {
    Index num_dofs = smoother.num_global_dofs();
    Index num_elems = mesh.num_elements();
    std::vector<int> dof_count(num_dofs, 0);
    std::vector<Index> dof_elem(num_dofs, -1);
    for (Index e = 0; e < num_elems; ++e) {
      const auto &dofs = smoother.dof_manager().element_dofs(e);
      for (Index d : dofs) {
        dof_count[d]++;
        dof_elem[d] = e;
      }
    }
    // Shared DOFs get element_id = -1
    for (Index d = 0; d < num_dofs; ++d) {
      if (dof_count[d] > 1) {
        dof_elem[d] = -1;
      }
    }
    std::ofstream f(subdir + "/dof_map.txt");
    f << num_dofs << " " << num_elems << "\n";
    for (Index d = 0; d < num_dofs; ++d) {
      f << d << " " << dof_elem[d] << "\n";
    }
  }

  // Write H, BtWB, Q (global DOF matrices)
  const auto &H = smoother.H_global();
  const auto &BtWB = smoother.BtWB_global();
  SpMat Q = smoother.Q_global();

  write_matrix_market(H, subdir + "/H.mtx");
  write_matrix_market(BtWB, subdir + "/BtWB.mtx");
  write_matrix_market(Q, subdir + "/Q.mtx");

  // Build condensed system and write A_edge
  auto sys = smoother.condensed_system();
  write_matrix_market(sys.A_edge, subdir + "/A_edge.mtx");
  write_matrix_market(sys.Q_reduced, subdir + "/Q_reduced.mtx");

  // Compute A * A^T (constraint Gram matrix)
  SpMat AAT;
  if (sys.A_edge.rows() > 0) {
    AAT = (sys.A_edge * SpMat(sys.A_edge.transpose())).pruned(1e-14);
    write_matrix_market(AAT, subdir + "/AAT.mtx");
  }

  // Compute Schur complement S = A * Q^{-1} * A^T
  SpMat S;
  if (sys.A_edge.rows() > 0) {
    Eigen::SparseLU<SpMat> lu;
    lu.compute(sys.Q_reduced);
    ASSERT_EQ(lu.info(), Eigen::Success) << "SparseLU factorization failed";

    // Solve Q * X = A^T  =>  X = Q^{-1} * A^T
    SpMat At = SpMat(sys.A_edge.transpose());
    MatX X(At.rows(), At.cols());
    for (Index j = 0; j < At.cols(); ++j) {
      VecX col = VecX(At.col(j));
      X.col(j) = lu.solve(col);
    }

    // S = A * X = A * Q^{-1} * A^T
    MatX S_dense = MatX(sys.A_edge) * X;
    S = S_dense.sparseView(1e-14);
    write_matrix_market(S, subdir + "/S.mtx");
  }

  // Build blockdiag(Q) and blockdiag(Q)^{-1} using same DOF ownership as preconditioner
  SpMat Q_blkdiag, Q_blkdiag_inv;
  if (sys.A_edge.rows() > 0) {
    const auto &dm = smoother.dof_manager();
    Index n_global = dm.num_global_dofs();
    Index n_free = dm.num_free_dofs();
    Index n_elem = mesh.num_elements();

    // DOF ownership: first element to reference a DOF owns it
    std::vector<Index> dof_owner(n_global, -1);
    for (Index e = 0; e < n_elem; ++e) {
      for (Index g : dm.element_dofs(e)) {
        if (dof_owner[g] < 0)
          dof_owner[g] = e;
      }
    }

    std::vector<Eigen::Triplet<Real>> trips_blk, trips_inv;
    for (Index e = 0; e < n_elem; ++e) {
      // Collect owned free DOFs
      std::vector<Index> owned;
      for (Index g : dm.element_dofs(e)) {
        if (dof_owner[g] == e) {
          Index f = dm.global_to_free(g);
          if (f >= 0)
            owned.push_back(f);
        }
      }
      if (owned.empty())
        continue;

      int bs = static_cast<int>(owned.size());
      MatX block(bs, bs);
      for (int i = 0; i < bs; ++i)
        for (int j = 0; j < bs; ++j)
          block(i, j) = sys.Q_reduced.coeff(owned[i], owned[j]);

      MatX block_inv = block.partialPivLu().solve(MatX::Identity(bs, bs));

      for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < bs; ++j) {
          trips_blk.emplace_back(owned[i], owned[j], block(i, j));
          if (std::abs(block_inv(i, j)) > 1e-14)
            trips_inv.emplace_back(owned[i], owned[j], block_inv(i, j));
        }
      }
    }

    Q_blkdiag.resize(n_free, n_free);
    Q_blkdiag.setFromTriplets(trips_blk.begin(), trips_blk.end());
    write_matrix_market(Q_blkdiag, subdir + "/Q_blkdiag.mtx");

    Q_blkdiag_inv.resize(n_free, n_free);
    Q_blkdiag_inv.setFromTriplets(trips_inv.begin(), trips_inv.end());
    write_matrix_market(Q_blkdiag_inv, subdir + "/Q_blkdiag_inv.mtx");
  }

  // Build Schur preconditioners and write M_S
  SpMat MS_diag, MS_block;
  if (sys.A_edge.rows() > 0) {
    DiagonalApproxCGSchurPreconditioner diag_precond(sys.Q_reduced,
                                                     sys.A_edge);
    MS_diag = diag_precond.assembled_matrix();
    write_matrix_market(MS_diag, subdir + "/MS_diag.mtx");

    BlockDiagApproxCGSchurPreconditioner block_precond(
        sys.Q_reduced, sys.A_edge, smoother.dof_manager());
    MS_block = block_precond.assembled_matrix();
    write_matrix_market(MS_block, subdir + "/MS_block.mtx");
  }

  // Print summary
  std::cout << "\nMatrix sparsity summary (" << n << "x" << n
            << " flat seabed):\n"
            << "  H:         " << H.rows() << " x " << H.cols()
            << ", nnz = " << H.nonZeros() << "\n"
            << "  BtWB:      " << BtWB.rows() << " x " << BtWB.cols()
            << ", nnz = " << BtWB.nonZeros() << "\n"
            << "  Q:         " << Q.rows() << " x " << Q.cols()
            << ", nnz = " << Q.nonZeros() << "\n"
            << "  Q_reduced: " << sys.Q_reduced.rows() << " x "
            << sys.Q_reduced.cols()
            << ", nnz = " << sys.Q_reduced.nonZeros() << "\n"
            << "  A_edge:    " << sys.A_edge.rows() << " x "
            << sys.A_edge.cols() << ", nnz = " << sys.A_edge.nonZeros()
            << "\n";
  if (sys.A_edge.rows() > 0) {
    std::cout << "  AAT:       " << AAT.rows() << " x " << AAT.cols()
              << ", nnz = " << AAT.nonZeros() << "\n"
              << "  S:         " << S.rows() << " x " << S.cols()
              << ", nnz = " << S.nonZeros() << "\n"
              << "  Q_blkdiag:     " << Q_blkdiag.rows() << " x "
              << Q_blkdiag.cols()
              << ", nnz = " << Q_blkdiag.nonZeros() << "\n"
              << "  Q_blkdiag_inv: " << Q_blkdiag_inv.rows() << " x "
              << Q_blkdiag_inv.cols()
              << ", nnz = " << Q_blkdiag_inv.nonZeros() << "\n"
              << "  MS_diag:   " << MS_diag.rows() << " x " << MS_diag.cols()
              << ", nnz = " << MS_diag.nonZeros() << "\n"
              << "  MS_block:  " << MS_block.rows() << " x " << MS_block.cols()
              << ", nnz = " << MS_block.nonZeros() << "\n";
  }
  std::cout << "\nFiles written to: " << subdir << "/\n";

  // Sanity checks
  EXPECT_EQ(H.rows(), H.cols());
  EXPECT_EQ(BtWB.rows(), BtWB.cols());
  EXPECT_GT(H.nonZeros(), 0);
  EXPECT_GT(BtWB.nonZeros(), 0);
  EXPECT_EQ(sys.A_edge.cols(), sys.Q_reduced.cols());

  // 1x1 mesh has no internal edges, so no edge constraints
  if (n > 1) {
    EXPECT_GT(sys.A_edge.rows(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(MeshSizes, MatrixSparsityTest,
                         ::testing::Values(1, 2, 4, 8),
                         [](const auto &info) {
                           return std::to_string(info.param) + "x" +
                                  std::to_string(info.param);
                         });

TEST_P(MatrixSparsityTest, CenterGraded) {
  int num_levels = GetParam();
  if (num_levels < 2) {
    GTEST_SKIP() << "Center-graded mesh requires at least 2 levels";
  }

  std::string subdir =
      output_dir_ + "/center_graded_L" + std::to_string(num_levels);
  std::filesystem::create_directories(subdir);

  // Build center-graded quadtree mesh with 2:1 balance (uses fixed 1000x1000m domain)
  QuadtreeAdapter mesh;
  mesh.build_center_graded(num_levels);

  // Configure smoother with edge constraints
  CGCubicBezierSmootherConfig config;
  config.lambda = 1.0;
  config.edge_ngauss = 4;
  config.ridge_epsilon = 1e-4;

  // Create smoother and set flat bathymetry
  CGCubicBezierBathymetrySmoother smoother(mesh, config);
  smoother.set_bathymetry_data([](Real, Real) { return 100.0; });

  // Write DOF-to-element map
  {
    Index num_dofs = smoother.num_global_dofs();
    Index num_elems = mesh.num_elements();
    std::vector<int> dof_count(num_dofs, 0);
    std::vector<Index> dof_elem(num_dofs, -1);
    for (Index e = 0; e < num_elems; ++e) {
      const auto &dofs = smoother.dof_manager().element_dofs(e);
      for (Index d : dofs) {
        dof_count[d]++;
        dof_elem[d] = e;
      }
    }
    for (Index d = 0; d < num_dofs; ++d) {
      if (dof_count[d] > 1) {
        dof_elem[d] = -1;
      }
    }
    std::ofstream f(subdir + "/dof_map.txt");
    f << num_dofs << " " << num_elems << "\n";
    for (Index d = 0; d < num_dofs; ++d) {
      f << d << " " << dof_elem[d] << "\n";
    }
  }

  // Write H, BtWB, Q (global DOF matrices)
  const auto &H = smoother.H_global();
  const auto &BtWB = smoother.BtWB_global();
  SpMat Q = smoother.Q_global();

  write_matrix_market(H, subdir + "/H.mtx");
  write_matrix_market(BtWB, subdir + "/BtWB.mtx");
  write_matrix_market(Q, subdir + "/Q.mtx");

  // Build condensed system and write A_edge
  auto sys = smoother.condensed_system();
  write_matrix_market(sys.A_edge, subdir + "/A_edge.mtx");
  write_matrix_market(sys.Q_reduced, subdir + "/Q_reduced.mtx");

  // Compute A * A^T (constraint Gram matrix)
  SpMat AAT;
  if (sys.A_edge.rows() > 0) {
    AAT = (sys.A_edge * SpMat(sys.A_edge.transpose())).pruned(1e-14);
    write_matrix_market(AAT, subdir + "/AAT.mtx");
  }

  // Compute Schur complement S = A * Q^{-1} * A^T
  SpMat S;
  if (sys.A_edge.rows() > 0) {
    Eigen::SparseLU<SpMat> lu;
    lu.compute(sys.Q_reduced);
    ASSERT_EQ(lu.info(), Eigen::Success) << "SparseLU factorization failed";

    SpMat At = SpMat(sys.A_edge.transpose());
    MatX X(At.rows(), At.cols());
    for (Index j = 0; j < At.cols(); ++j) {
      VecX col = VecX(At.col(j));
      X.col(j) = lu.solve(col);
    }

    MatX S_dense = MatX(sys.A_edge) * X;
    S = S_dense.sparseView(1e-14);
    write_matrix_market(S, subdir + "/S.mtx");
  }

  // Build blockdiag(Q) and blockdiag(Q)^{-1}
  SpMat Q_blkdiag, Q_blkdiag_inv;
  if (sys.A_edge.rows() > 0) {
    const auto &dm = smoother.dof_manager();
    Index n_global = dm.num_global_dofs();
    Index n_free = dm.num_free_dofs();
    Index n_elem = mesh.num_elements();

    std::vector<Index> dof_owner(n_global, -1);
    for (Index e = 0; e < n_elem; ++e) {
      for (Index g : dm.element_dofs(e)) {
        if (dof_owner[g] < 0)
          dof_owner[g] = e;
      }
    }

    std::vector<Eigen::Triplet<Real>> trips_blk, trips_inv;
    for (Index e = 0; e < n_elem; ++e) {
      std::vector<Index> owned;
      for (Index g : dm.element_dofs(e)) {
        if (dof_owner[g] == e) {
          Index f = dm.global_to_free(g);
          if (f >= 0)
            owned.push_back(f);
        }
      }
      if (owned.empty())
        continue;

      int bs = static_cast<int>(owned.size());
      MatX block(bs, bs);
      for (int i = 0; i < bs; ++i)
        for (int j = 0; j < bs; ++j)
          block(i, j) = sys.Q_reduced.coeff(owned[i], owned[j]);

      MatX block_inv = block.partialPivLu().solve(MatX::Identity(bs, bs));

      for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < bs; ++j) {
          trips_blk.emplace_back(owned[i], owned[j], block(i, j));
          if (std::abs(block_inv(i, j)) > 1e-14)
            trips_inv.emplace_back(owned[i], owned[j], block_inv(i, j));
        }
      }
    }

    Q_blkdiag.resize(n_free, n_free);
    Q_blkdiag.setFromTriplets(trips_blk.begin(), trips_blk.end());
    write_matrix_market(Q_blkdiag, subdir + "/Q_blkdiag.mtx");

    Q_blkdiag_inv.resize(n_free, n_free);
    Q_blkdiag_inv.setFromTriplets(trips_inv.begin(), trips_inv.end());
    write_matrix_market(Q_blkdiag_inv, subdir + "/Q_blkdiag_inv.mtx");
  }

  // Build Schur preconditioners
  SpMat MS_diag, MS_block;
  if (sys.A_edge.rows() > 0) {
    DiagonalApproxCGSchurPreconditioner diag_precond(sys.Q_reduced,
                                                     sys.A_edge);
    MS_diag = diag_precond.assembled_matrix();
    write_matrix_market(MS_diag, subdir + "/MS_diag.mtx");

    BlockDiagApproxCGSchurPreconditioner block_precond(
        sys.Q_reduced, sys.A_edge, smoother.dof_manager());
    MS_block = block_precond.assembled_matrix();
    write_matrix_market(MS_block, subdir + "/MS_block.mtx");
  }

  // Print summary
  std::cout << "\nMatrix sparsity summary (center-graded L" << num_levels
            << ", " << mesh.num_elements() << " elements):\n"
            << "  H:         " << H.rows() << " x " << H.cols()
            << ", nnz = " << H.nonZeros() << "\n"
            << "  BtWB:      " << BtWB.rows() << " x " << BtWB.cols()
            << ", nnz = " << BtWB.nonZeros() << "\n"
            << "  Q:         " << Q.rows() << " x " << Q.cols()
            << ", nnz = " << Q.nonZeros() << "\n"
            << "  Q_reduced: " << sys.Q_reduced.rows() << " x "
            << sys.Q_reduced.cols()
            << ", nnz = " << sys.Q_reduced.nonZeros() << "\n"
            << "  A_edge:    " << sys.A_edge.rows() << " x "
            << sys.A_edge.cols() << ", nnz = " << sys.A_edge.nonZeros()
            << "\n";
  if (sys.A_edge.rows() > 0) {
    std::cout << "  AAT:       " << AAT.rows() << " x " << AAT.cols()
              << ", nnz = " << AAT.nonZeros() << "\n"
              << "  S:         " << S.rows() << " x " << S.cols()
              << ", nnz = " << S.nonZeros() << "\n"
              << "  Q_blkdiag:     " << Q_blkdiag.rows() << " x "
              << Q_blkdiag.cols()
              << ", nnz = " << Q_blkdiag.nonZeros() << "\n"
              << "  Q_blkdiag_inv: " << Q_blkdiag_inv.rows() << " x "
              << Q_blkdiag_inv.cols()
              << ", nnz = " << Q_blkdiag_inv.nonZeros() << "\n"
              << "  MS_diag:   " << MS_diag.rows() << " x " << MS_diag.cols()
              << ", nnz = " << MS_diag.nonZeros() << "\n"
              << "  MS_block:  " << MS_block.rows() << " x " << MS_block.cols()
              << ", nnz = " << MS_block.nonZeros() << "\n";
  }
  std::cout << "\nFiles written to: " << subdir << "/\n";

  // Sanity checks
  EXPECT_EQ(H.rows(), H.cols());
  EXPECT_EQ(BtWB.rows(), BtWB.cols());
  EXPECT_GT(H.nonZeros(), 0);
  EXPECT_GT(BtWB.nonZeros(), 0);
  EXPECT_EQ(sys.A_edge.cols(), sys.Q_reduced.cols());
  EXPECT_GT(sys.A_edge.rows(), 0) << "Center-graded mesh should have edges";
}

INSTANTIATE_TEST_SUITE_P(CenterGradedLevels, MatrixSparsityTest,
                         ::testing::Values(3, 4, 5),
                         [](const auto &info) {
                           return "L" + std::to_string(info.param);
                         });
