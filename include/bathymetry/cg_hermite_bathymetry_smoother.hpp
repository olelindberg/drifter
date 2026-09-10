#pragma once

/// @file cg_hermite_bathymetry_smoother.hpp
/// @brief CG Hermite bathymetry smoother (C0 and C1, SPD system, direct solve)
///
/// Fits a tensor-product Hermite surface to bathymetry data. The degrees of
/// freedom are elevation and physical derivatives at element corners rather than
/// Bernstein control values, which makes C^r continuity **structural**: it holds
/// pointwise along every conforming edge by construction, for arbitrary and
/// unequal element sizes on the two sides.
///
/// Compared with CGCubicBezierBathymetrySmoother, which imposes C1 by
/// collocation and therefore solves an indefinite KKT saddle point:
///
/// |                    | Cubic Bezier            | C1 Hermite       |
/// |--------------------|-------------------------|------------------|
/// | continuity         | approximate C1 (~2.4e-8)| exact C1         |
/// | system             | indefinite KKT          | **SPD**          |
/// | solver             | SparseLU / Schur CG     | direct, SPD      |
/// | constraint rows    | 8N(N-1)                 | 0                |
/// | DOFs on N x N mesh | (3N+1)^2                | 4(N+1)^2         |
///
/// What is given up is Bernstein's convex-hull and variation-diminishing
/// property: a Hermite interpolant can overshoot in steep regions. That is the
/// reason the Bezier smoothers remain in the codebase alongside this one.
///
/// See docs/hermite_bathymetry_system.md and docs/hermite_smoothness_operator.md.

#include "bathymetry/cg_hermite_dof_manager.hpp"
#include "bathymetry/cg_smoother_base.hpp"
#include "bathymetry/hermite_basis_2d.hpp"
#include "bathymetry/hermite_hessian.hpp"
#include "core/types.hpp"
#include <memory>
#include <string>

namespace drifter {

class OctreeAdapter;

/// @brief Per-iteration timings, defined by the adaptive driver that owns them
///
/// Follows the same split as the Bezier smoothers: the profile struct lives with
/// the adaptive driver, and the inner smoother only fills in its own fields.
struct HermiteIterationProfile;

/// @brief Which direct factorisation solves the condensed system
///
/// Q_red is SPD, so a Cholesky-type factorisation is the natural choice and the
/// Cholesky kinds differ only in ordering and in whether they are supernodal.
/// UmfPackLU is an unsymmetric LU: it does roughly twice the arithmetic of a
/// Cholesky and orders for stability rather than symmetric fill, so it is here
/// as a measured reference point and as the kind that still works if the system
/// is ever driven indefinite - not as a speed candidate.
///
/// Availability is a build decision: a kind whose backend was not compiled in
/// throws from make_factorization() naming the CMake option that enables it.
/// There is no fallback to another kind. Which *available* kind runs is a
/// config decision, so the backends can be compared within a single run; see
/// tests/benchmarks/test_hermite_solver_comparison.cpp.
enum class HermiteSolverKind {
    SimplicialLDLT,          ///< Eigen simplicial LDL^T, AMD ordering (default)
    SimplicialLDLTMetis,     ///< As above with nested dissection; needs DRIFTER_USE_METIS
    SimplicialLLT,           ///< Eigen simplicial LL^T, AMD ordering
    PardisoLDLT,             ///< MKL PARDISO LDL^T, parallel; needs DRIFTER_USE_MKL
    PardisoLLT,              ///< MKL PARDISO LL^T, parallel; needs DRIFTER_USE_MKL
    UmfPackLU,               ///< SuiteSparse UMFPACK LU; needs DRIFTER_USE_UMFPACK
    CholmodSimplicialLDLT,   ///< CHOLMOD simplicial LDL^T; needs DRIFTER_USE_CHOLMOD
    CholmodSupernodalLLT,    ///< CHOLMOD supernodal LL^T, AMD; needs DRIFTER_USE_CHOLMOD
    CholmodSupernodalNesdis  ///< CHOLMOD supernodal LL^T, nested dissection
};

/// @brief Configuration for the CG Hermite smoother
struct CGHermiteSmootherConfig {
    /// Continuity order: 0 (bilinear, C0) or 1 (bicubic Bogner-Fox-Schmit, C1)
    int continuity_order = 1;

    /// Data fitting weight. lambda -> 0 is pure smoothness, lambda -> infinity
    /// approaches a least-squares fit.
    Real lambda = 0.01;

    /// Gauss points per direction for the data fitting term. The integrand has
    /// degree 2p per direction, so r = 0 needs 2 and r = 1 needs 4.
    int ngauss_data = 4;

    /// Ridge regularization, covering the null space of the smoothness energy
    Real ridge_epsilon = 1e-4;

    /// Zero normal gradient (symmetry) on the domain boundary. Exact and free:
    /// two DOF eliminations per boundary node, no constraint rows. Not
    /// expressible at r = 0, where it is silently inactive.
    bool enable_zero_gradient_bc = false;

    /// Symmetric equilibration of the mixed-unit Hermite DOFs before the solve.
    ///
    /// This is a change of variables inside solve() only - it improves the
    /// conditioning of the factorised matrix but does not alter the problem, so
    /// enabling or disabling it changes the answer only by solver accuracy. (The
    /// ridge is always applied in the equilibrated space, independently of this
    /// flag, because a uniform ridge across DOFs of different physical units is
    /// dimensionally inconsistent either way.)
    /// See docs/hermite_bathymetry_system.md S11.
    bool use_equilibration = true;

    /// Direct factorisation used by solve(). The default is the only kind that
    /// needs no optional dependency.
    HermiteSolverKind solver = HermiteSolverKind::SimplicialLDLT;

    bool verbose = false;

    /// Boundary relaxation zone (reduces data fitting weight near boundaries)
    BoundaryRelaxationConfig boundary_relaxation;
};

/// @brief CG Hermite bathymetry smoother
class CGHermiteBathymetrySmoother : public CGSmootherBase {
public:
    /// @brief Construct smoother for a quadtree mesh
    explicit CGHermiteBathymetrySmoother(const QuadtreeAdapter &mesh,
                                         const CGHermiteSmootherConfig &config = {});

    /// @brief Construct smoother from an octree (uses the bottom face)
    explicit CGHermiteBathymetrySmoother(const OctreeAdapter &octree,
                                         const CGHermiteSmootherConfig &config = {});

    // =========================================================================
    // Configuration
    // =========================================================================

    void set_smoothing_weight(Real lambda) { config_.lambda = lambda; }
    const CGHermiteSmootherConfig &config() const { return config_; }
    void set_profile(HermiteIterationProfile *profile) { profile_ = profile; }

    // =========================================================================
    // Solve
    // =========================================================================

    /// @brief Solve the smoothing problem
    ///
    /// Condenses the master/slave substitutions to Q_red = T' Q T, which is SPD,
    /// and factorises it directly with the backend named by config.solver. There
    /// is no KKT path, no Schur complement and no iterative fallback.
    ///
    /// @throws std::runtime_error if data is unset or the factorisation fails
    /// @throws std::invalid_argument if config.solver was not compiled in
    void solve();

    // =========================================================================
    // Output
    // =========================================================================

    /// @brief Polynomial degree of the fitted surface: 1 for r=0, 3 for r=1
    int surface_degree() const { return 2 * config_.continuity_order + 1; }

    /// @brief Write the fitted surface as per-element VTK_LAGRANGE_QUAD cells
    ///
    /// @param order Degree of the emitted cells; <= 0 uses surface_degree().
    ///              A higher degree resamples the same polynomial on more nodes
    ///              so ParaView tessellates it more finely; values below
    ///              surface_degree() are raised to it, since a lower degree
    ///              could not represent the surface.
    void write_vtk(const std::string &filename, int order = 0) const;

    /// @brief Write the Bernstein control net of the fitted surface
    ///
    /// Hermite DOFs are not control points, so the surface is converted through
    /// c_e = M_e q_e first. Useful for inspecting whether the fit overshoots the
    /// convex hull, which Hermite does not guarantee.
    void write_control_points_vtk(const std::string &filename) const;

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// @brief Continuity constraint violation
    ///
    /// Identically zero: C^r is structural in this basis, and the hanging-node
    /// substitutions are satisfied exactly by back-substitution. Provided for
    /// interface parity with the Bezier smoothers, whose collocated C1 is only
    /// approximate. See docs/hermite_bathymetry_system.md S3.
    Real constraint_violation() const { return 0.0; }

    const CGHermiteDofManager &dof_manager() const {
        ensure_dof_system();
        return *dof_manager_;
    }
    const Basis2DBase &get_basis() const { return *basis_; }

    /// @brief The Hermite basis, with the change of basis and midpoint matrices
    const HermiteBasis2D &hermite_basis() const { return *basis_; }

    /// @brief Assembled system matrix Q (before condensation)
    SpMat Q_global() const { return assemble_Q(); }

    /// @brief Assembled smoothness operator H
    const SpMat &H_global() const { return H_global_; }

    /// @brief Assembled data fitting operator B'WB
    const SpMat &BtWB_global() const { return BtWB_global_; }

    /// @brief The condensed system actually factorised, Q_red = T' Q T
    ///
    /// Exposed so tests can verify it is symmetric positive definite.
    SpMat condensed_matrix() const;

    /// @brief c_e = M_e q_e, for consumers that need Bernstein control values
    ///
    /// Public, mirroring element_coefficients() on the base.
    VecX element_bernstein_coefficients(Index elem) const override;

protected:
    // =========================================================================
    // CGSmootherBase virtual method implementations
    // =========================================================================

    void set_bathymetry_data_impl(std::function<Real(Real, Real)> bathy_func) override;
    Index dof_manager_num_global_dofs() const override {
        ensure_dof_system();
        return dof_manager_->num_global_dofs();
    }
    Index dof_manager_num_free_dofs() const override {
        ensure_dof_system();
        return dof_manager_->num_free_dofs();
    }
    Index dof_manager_num_constraints() const override {
        ensure_dof_system();
        return dof_manager_->num_constraints();
    }
    const std::vector<Index> &element_global_dofs(Index elem) const override {
        ensure_dof_system();
        return dof_manager_->element_dofs(elem);
    }
    const Basis2DBase &basis() const override { return *basis_; }
    int ngauss_data() const override { return config_.ngauss_data; }
    Real lambda() const override { return config_.lambda; }
    Real ridge_epsilon() const override { return config_.ridge_epsilon; }

    /// @brief Lambda_e = diag(h_x^a h_y^b) for physical-derivative DOFs
    VecX element_dof_scaling(Real dx, Real dy) const override;

    /// @brief lambda*epsilon*S^-2, so the ridge is dimensionally consistent
    VecX ridge_diagonal() const override;

private:
    /// @brief Build the basis and the smoothness operator (construction time)
    void init_basis();

    /// @brief Build the element mask, the DOF numbering and its constraints
    ///
    /// Depends on the data masks, so it runs once they are known rather than in
    /// the constructor.
    void build_dof_system();

    /// @brief Build the DOF numbering if it has not been built yet
    void ensure_dof_system() const;

    /// @brief Expand a global DOF into free DOFs and weights (the T operator)
    std::vector<std::pair<Index, Real>> expand_dof(Index global) const;

    /// @brief Build Q_red = T' Q T and b_red = T' b
    void build_condensed_system(SpMat &Q_reduced, VecX &b_reduced) const;

    CGHermiteSmootherConfig config_;

    std::unique_ptr<HermiteBasis2D> basis_;
    std::unique_ptr<HermiteHessian> hessian_;
    std::unique_ptr<CGHermiteDofManager> dof_manager_;

    /// slave global DOF -> index into dof_manager_->constraints()
    std::unordered_map<Index, size_t> slave_to_constraint_;

    HermiteIterationProfile *profile_ = nullptr;
};

} // namespace drifter
