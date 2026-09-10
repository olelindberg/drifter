#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
#include "bathymetry/adaptive_cg_hermite_smoother.hpp" // HermiteIterationProfile
#include "bathymetry/constraint_condenser.hpp"
#include "core/enum_strings.hpp"
#include "core/logger.hpp"
#include "core/scoped_timer.hpp"
#include "io/bathymetry_vtk_writer.hpp"
#include "mesh/octree_adapter.hpp"
#include <Eigen/SparseCholesky>
#ifdef DRIFTER_USE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#ifdef DRIFTER_USE_METIS
#include <Eigen/MetisSupport>
#endif
#ifdef DRIFTER_USE_MKL
#include <Eigen/PardisoSupport>
#endif
#ifdef DRIFTER_USE_UMFPACK
#include <Eigen/UmfPackSupport>
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace drifter {

namespace {

/// The factorised system is stored column-major: every backend here reads a
/// sparse matrix as CSC, and SpMat is row-major, so the conversion is made once
/// and explicitly rather than left to an interpretation of the storage.
using HermiteMatrix = Eigen::SparseMatrix<Real>;

/// Q_red is symmetric positive definite, so a Cholesky-type factorisation
/// replaces the SparseLU / Schur-complement machinery the Bezier path needs.
///
/// Which backend factorises it is a runtime choice (CGHermiteSmootherConfig::
/// solver), because comparing them is the only way to answer which is fastest on
/// a given matrix, and doing that through the build system meant one reconfigure
/// and rebuild per candidate. Availability stays a build decision: a kind whose
/// library was not compiled in throws below rather than quietly running another.
///
/// Time to factorise, uniform C1 mesh, 66564 free DOFs, 4 cores (i5-8350U); all
/// nine agree on the fitted surface to ~1e-12. Reproduce with
/// tests/benchmarks/test_hermite_solver_comparison.cpp:
///
///     PardisoLDLT                540 ms
///     CholmodSupernodalLLT       556 ms
///     PardisoLLT                 636 ms
///     CholmodSupernodalNesdis    800 ms
///     UmfPackLU                 1445 ms
///     SimplicialLDLTMetis       3613 ms
///     SimplicialLDLT            3743 ms
///     SimplicialLLT             3855 ms
///     CholmodSimplicialLDLT     4165 ms
///
/// The split is not parallel-vs-serial, it is whether the factorisation reaches
/// dense BLAS3 kernels at all. The four fastest are supernodal or multifrontal
/// and hand blocks to a threaded OpenBLAS; the four slowest are simplicial,
/// which is scalar code walking a sparse structure one column at a time. The
/// ordering barely matters within a family - METIS buys SimplicialLDLT 3%, and
/// asking CHOLMOD for nested dissection actually costs it 44% against its own
/// AMD default, which is the opposite of what the supernodal literature would
/// suggest and worth remembering before reaching for NESDIS again.
///
/// Note the ranking inverts below a few thousand DOFs, where setup dominates and
/// the simplicial solvers win; the benchmark covers n = 16 and n = 64 as well so
/// that crossover stays visible.
///
/// This replaces an earlier table that put CholmodSupernodalLLT at 18694 ms,
/// an order of magnitude off what it measures at here, and blamed Debian's
/// libcholmod for being built without a graph partitioner. That diagnosis was
/// wrong on the facts - libcholmod.so.3 is linked against libmetis and exports
/// cholmod_metis - and the timing has not reproduced.
///
/// SimplicialLDLT remains the default because it is the only one of the nine
/// that needs no optional dependency, not because it is fast.
class HermiteFactorization {
public:
    virtual ~HermiteFactorization() = default;

    /// @return false if the factorisation failed
    virtual bool compute(const HermiteMatrix &A) = 0;

    /// @return false if the triangular solves failed
    virtual bool solve(const VecX &b, VecX &x) = 0;
};

/// Wraps any Eigen sparse solver, which share an interface but no base class.
/// The configure hook runs after construction and before compute(), which is the
/// only window CHOLMOD offers for setting its ordering.
template <typename Solver> class EigenFactorization final : public HermiteFactorization {
public:
    using Configure = void (*)(Solver &);

    explicit EigenFactorization(Configure configure = nullptr) {
        if (configure) {
            configure(solver_);
        }
    }

    bool compute(const HermiteMatrix &A) override {
        solver_.compute(A);
        return solver_.info() == Eigen::Success;
    }

    bool solve(const VecX &b, VecX &x) override {
        x = solver_.solve(b);
        return solver_.info() == Eigen::Success;
    }

private:
    Solver solver_;
};

/// Names the CMake option that would have made `kind` available, so a config
/// naming an uncompiled backend fails with the fix in the message.
[[noreturn]] void throw_unavailable(HermiteSolverKind kind, const char *option) {
    throw std::invalid_argument("CGHermiteBathymetrySmoother: solver '" + to_string(kind) +
                                "' was not compiled in; configure with -D" + std::string(option) +
                                "=ON");
}

std::unique_ptr<HermiteFactorization> make_factorization(HermiteSolverKind kind) {
    switch (kind) {
    case HermiteSolverKind::SimplicialLDLT:
        return std::make_unique<EigenFactorization<Eigen::SimplicialLDLT<HermiteMatrix>>>();

    case HermiteSolverKind::SimplicialLLT:
        return std::make_unique<EigenFactorization<Eigen::SimplicialLLT<HermiteMatrix>>>();

    case HermiteSolverKind::SimplicialLDLTMetis:
#ifdef DRIFTER_USE_METIS
        return std::make_unique<EigenFactorization<
            Eigen::SimplicialLDLT<HermiteMatrix, Eigen::Lower, Eigen::MetisOrdering<int>>>>();
#else
        throw_unavailable(kind, "DRIFTER_USE_METIS");
#endif

    case HermiteSolverKind::PardisoLDLT:
#ifdef DRIFTER_USE_MKL
        return std::make_unique<
            EigenFactorization<Eigen::PardisoLDLT<HermiteMatrix, Eigen::Lower>>>();
#else
        throw_unavailable(kind, "DRIFTER_USE_MKL");
#endif

    case HermiteSolverKind::PardisoLLT:
#ifdef DRIFTER_USE_MKL
        return std::make_unique<
            EigenFactorization<Eigen::PardisoLLT<HermiteMatrix, Eigen::Lower>>>();
#else
        throw_unavailable(kind, "DRIFTER_USE_MKL");
#endif

    case HermiteSolverKind::UmfPackLU:
#ifdef DRIFTER_USE_UMFPACK
        return std::make_unique<EigenFactorization<Eigen::UmfPackLU<HermiteMatrix>>>();
#else
        throw_unavailable(kind, "DRIFTER_USE_UMFPACK");
#endif

    case HermiteSolverKind::CholmodSimplicialLDLT:
#ifdef DRIFTER_USE_CHOLMOD
        return std::make_unique<
            EigenFactorization<Eigen::CholmodSimplicialLDLT<HermiteMatrix, Eigen::Lower>>>();
#else
        throw_unavailable(kind, "DRIFTER_USE_CHOLMOD");
#endif

    case HermiteSolverKind::CholmodSupernodalLLT:
#ifdef DRIFTER_USE_CHOLMOD
        return std::make_unique<
            EigenFactorization<Eigen::CholmodSupernodalLLT<HermiteMatrix, Eigen::Lower>>>();
#else
        throw_unavailable(kind, "DRIFTER_USE_CHOLMOD");
#endif

    case HermiteSolverKind::CholmodSupernodalNesdis:
#ifdef DRIFTER_USE_CHOLMOD
        // CHOLMOD's default is to try AMD and settle for it. Supernodal
        // factorisation only pays off when the ordering leaves dense blocks to
        // work on, which is what nested dissection produces and AMD does not.
        using Nesdis = Eigen::CholmodSupernodalLLT<HermiteMatrix, Eigen::Lower>;
        return std::make_unique<EigenFactorization<Nesdis>>([](Nesdis &s) {
            s.cholmod().nmethods = 1;
            s.cholmod().method[0].ordering = CHOLMOD_NESDIS;
        });
#else
        throw_unavailable(kind, "DRIFTER_USE_CHOLMOD");
#endif
    }
    throw std::invalid_argument("CGHermiteBathymetrySmoother: unknown HermiteSolverKind");
}

} // namespace

// =============================================================================
// Construction
// =============================================================================

CGHermiteBathymetrySmoother::CGHermiteBathymetrySmoother(const QuadtreeAdapter &mesh,
                                                         const CGHermiteSmootherConfig &config)
    : config_(config) {
    quadtree_ = &mesh;
    init_basis();
}

CGHermiteBathymetrySmoother::CGHermiteBathymetrySmoother(const OctreeAdapter &octree,
                                                         const CGHermiteSmootherConfig &config)
    : config_(config) {
    quadtree_owned_ = std::make_unique<QuadtreeAdapter>(octree);
    quadtree_ = quadtree_owned_.get();
    init_basis();
}

void CGHermiteBathymetrySmoother::init_basis() {
    const int r = config_.continuity_order;
    if (r < 0 || r > 1) {
        throw std::invalid_argument(
            "CGHermiteBathymetrySmoother: continuity_order must be 0 (C0) or 1 (C1), got " +
            std::to_string(r));
    }

    basis_ = std::make_unique<HermiteBasis2D>(r);
    hessian_ = std::make_unique<HermiteHessian>(r);
}

void CGHermiteBathymetrySmoother::build_dof_system() {
    // Classify first: the pins are decided per element, and the DOF manager needs
    // that classification while it is numbering.
    build_element_mask();
    dof_manager_ = std::make_unique<CGHermiteDofManager>(
        *quadtree_, config_.continuity_order, config_.enable_zero_gradient_bc,
        element_mask_.get());

    slave_to_constraint_.clear();
    const auto &constraints = dof_manager_->constraints();
    for (size_t ci = 0; ci < constraints.size(); ++ci) {
        slave_to_constraint_[constraints[ci].slave_dof] = ci;
    }

    solution_.setZero(dof_manager_->num_global_dofs());
}

void CGHermiteBathymetrySmoother::ensure_dof_system() const {
    if (dof_manager_) {
        return;
    }
    // The numbering depends on the data masks, which arrive after construction,
    // so it is built when they are final - set_bathymetry_data_impl() - and here
    // for a caller that queries the smoother without ever setting data. Building
    // it eagerly in the constructor instead meant building it twice on every
    // refinement, once for a mask that was not yet attached.
    //
    // The const_cast is the usual lazy-initialisation one: the members it fills
    // are logically part of construction, and no caller ever holds a genuinely
    // const CGHermiteBathymetrySmoother.
    const_cast<CGHermiteBathymetrySmoother *>(this)->build_dof_system();
}

// =============================================================================
// Basis-dependent hooks
// =============================================================================

VecX CGHermiteBathymetrySmoother::element_dof_scaling(Real dx, Real dy) const {
    return basis_->dof_scaling(dx, dy);
}

VecX CGHermiteBathymetrySmoother::element_bernstein_coefficients(Index elem) const {
    const auto &bounds = quadtree_->element_bounds(elem);
    const Real dx = bounds.xmax - bounds.xmin;
    const Real dy = bounds.ymax - bounds.ymin;

    // element_coefficients() returns Lambda_e * q_e; M_e expects the raw physical
    // derivative DOFs q_e, so undo the scaling before the change of basis.
    const VecX scaled = element_coefficients(elem);
    const VecX q = scaled.cwiseQuotient(basis_->dof_scaling(dx, dy));

    return basis_->bernstein_change_of_basis(dx, dy) * q;
}

VecX CGHermiteBathymetrySmoother::ridge_diagonal() const {
    // lambda*epsilon applied in the equilibrated space is lambda*epsilon*S^-2 in
    // original coordinates. A uniform ridge would penalise z and z_xy identically
    // despite their entries differing by orders of magnitude, distorting exactly
    // the null-space modes the ridge exists to control.
    ensure_dof_system();
    const VecX S = dof_manager_->equilibration_scaling();
    return lambda() * ridge_epsilon() * S.cwiseProduct(S).cwiseInverse();
}

// =============================================================================
// Data input
// =============================================================================

void CGHermiteBathymetrySmoother::set_bathymetry_data_impl(
    std::function<Real(Real, Real)> bathy_func) {
    relaxation_config_ = config_.boundary_relaxation;

    // The non-water region is only known once the data source is attached, and it
    // changes the constraint set, so the DOF numbering is built here rather than in
    // the constructor. Cheap next to the assembly that follows, and re-running it
    // on every re-fit is what keeps the classification correct after refinement.
    build_dof_system();
    if (element_mask_) {
        LOG_DEBUG("Hermite elements: " << element_mask_->num_water() << " water, "
                                       << element_mask_->num_beach() << " beach (pinned to 0), "
                                       << element_mask_->num_inland() << " inland (excluded); "
                                       << dof_manager_->num_pinned_dofs() << " DOFs pinned");
    }

    {
        OptionalScopedTimer t(profile_ ? &profile_->hessian_assembly_ms : nullptr);
        assemble_hessian_global(*hessian_);
    }
    {
        OptionalScopedTimer t(profile_ ? &profile_->data_fitting_ms : nullptr);
        assemble_data_fitting_global(bathy_func);
    }
}

// =============================================================================
// Constraint condensation
// =============================================================================

std::vector<std::pair<Index, Real>> CGHermiteBathymetrySmoother::expand_dof(Index g) const {
    const Index f = dof_manager_->global_to_free(g);
    if (f >= 0) {
        return {{f, 1.0}};
    }

    auto it = slave_to_constraint_.find(g);
    if (it == slave_to_constraint_.end()) {
        return {};
    }

    // Constraints are transitively closed, so every master here is free
    const auto &c = dof_manager_->constraints()[it->second];
    std::vector<std::pair<Index, Real>> result;
    result.reserve(c.master_dofs.size());
    for (size_t i = 0; i < c.master_dofs.size(); ++i) {
        const Index mf = dof_manager_->global_to_free(c.master_dofs[i]);
        if (mf >= 0) {
            result.emplace_back(mf, c.weights[i]);
        }
    }
    return result;
}

void CGHermiteBathymetrySmoother::build_condensed_system(SpMat &Q_reduced,
                                                         VecX &b_reduced) const {
    ensure_dof_system();

    SpMat Q;
    VecX b;
    {
        OptionalScopedTimer t(profile_ ? &profile_->matrix_build_ms : nullptr);
        Q = assemble_Q();
        b = assemble_b();
    }
    {
        OptionalScopedTimer t(profile_ ? &profile_->constraint_condense_ms : nullptr);
        condense_matrix_and_rhs(Q, b, [this](Index g) { return expand_dof(g); },
                                dof_manager_->num_free_dofs(), Q_reduced, b_reduced);
    }
}

SpMat CGHermiteBathymetrySmoother::condensed_matrix() const {
    SpMat Q_reduced;
    VecX b_reduced;
    build_condensed_system(Q_reduced, b_reduced);
    return Q_reduced;
}

// =============================================================================
// Solve
// =============================================================================

void CGHermiteBathymetrySmoother::solve() {
    if (!data_set_) {
        throw std::runtime_error("CGHermiteBathymetrySmoother: bathymetry data not set");
    }

    const Index num_dofs = dof_manager_->num_global_dofs();
    const Index num_free = dof_manager_->num_free_dofs();

    // Times its own two phases: assembling Q and b, then condensing them
    SpMat Q_reduced;
    VecX b_reduced;
    build_condensed_system(Q_reduced, b_reduced);

    // Symmetric equilibration: solve (S Q S)(S^-1 x) = S b. A congruence, so
    // symmetry and positive definiteness are preserved.
    VecX S_free = VecX::Ones(num_free);
    if (config_.use_equilibration) {
        const VecX S = dof_manager_->equilibration_scaling();
        for (Index f = 0; f < num_free; ++f) {
            S_free(f) = S(dof_manager_->free_to_global(f));
        }
        Q_reduced = S_free.asDiagonal() * Q_reduced * S_free.asDiagonal();
        b_reduced = b_reduced.cwiseProduct(S_free);
    }

    // Named, not a temporary passed straight into compute(): UmfPackLU keeps raw
    // pointers into the matrix it was given whenever the storage order already
    // matches, rather than copying it, and dereferences them again in solve().
    // A temporary here dies at the end of the compute() statement and the
    // triangular solves then read freed memory. The Cholesky backends copy
    // during factorisation and so never noticed.
    const HermiteMatrix A(Q_reduced);

    auto solver = make_factorization(config_.solver);
    bool factorised = false;
    {
        OptionalScopedTimer t(profile_ ? &profile_->factorize_ms : nullptr);
        factorised = solver->compute(A);
    }
    if (!factorised) {
        // An LU tolerates indefiniteness, so for it a failure means the matrix is
        // genuinely singular and the ridge advice would send the reader nowhere.
        const bool is_cholesky = config_.solver != HermiteSolverKind::UmfPackLU;
        throw std::runtime_error(
            "CGHermiteBathymetrySmoother: " + to_string(config_.solver) +
            " factorisation of the condensed system failed. " +
            (is_cholesky
                 ? "The system should be symmetric positive definite; check that ridge_epsilon > 0" +
                       std::string(config_.use_equilibration
                                       ? "."
                                       : ", or enable use_equilibration on a deeply refined mesh.")
                 : std::string("The system is singular.")));
    }

    VecX y;
    {
        OptionalScopedTimer t(profile_ ? &profile_->substitute_ms : nullptr);
        if (!solver->solve(b_reduced, y)) {
            throw std::runtime_error("CGHermiteBathymetrySmoother: " + to_string(config_.solver) +
                                     " solve failed");
        }
    }

    // Undo the change of variables: x = S y
    const VecX x_free = config_.use_equilibration ? y.cwiseProduct(S_free) : y;

    solution_.setZero(num_dofs);
    for (Index f = 0; f < num_free; ++f) {
        solution_(dof_manager_->free_to_global(f)) = x_free(f);
    }
    back_substitute_slaves(solution_, dof_manager_->constraints());

    solved_ = true;

    if (config_.verbose) {
        LOG_INFO("CGHermiteBathymetrySmoother: r="
                 << config_.continuity_order << " dofs=" << num_dofs << " free=" << num_free
                 << " constraints=" << dof_manager_->num_constraints()
                 << " solver=" << to_string(config_.solver)
                 << " data_residual=" << data_residual());
    }
}

// =============================================================================
// Output
// =============================================================================

void CGHermiteBathymetrySmoother::write_vtk(const std::string &filename, int order) const {
    if (!solved_) {
        throw std::runtime_error("CGHermiteBathymetrySmoother: must solve() before write_vtk()");
    }

    // Inland elements carry no data and are not solved for, so they are left out of
    // the file entirely - a hole in the surface rather than a misleading flat patch.
    // Water and Beach are emitted, tagged so the rim is identifiable in ParaView.
    std::vector<std::pair<std::string, std::vector<Real>>> cell_data;
    std::function<bool(Index)> include_element;
    if (element_mask_) {
        std::vector<Real> element_class(static_cast<size_t>(quadtree_->num_elements()));
        for (Index e = 0; e < quadtree_->num_elements(); ++e) {
            element_class[static_cast<size_t>(e)] =
                static_cast<Real>(static_cast<int>((*element_mask_)[e]));
        }
        cell_data.emplace_back("element_class", std::move(element_class));

        const auto mask = element_mask_;
        include_element = [mask](Index elem) { return !mask->is_excluded(elem); };
    }

    io::write_high_order_surface_vtk(
        filename, *quadtree_, element_major_evaluator(*this),
        order > 0 ? std::max(order, surface_degree()) : surface_degree(), "elevation", cell_data,
        include_element);
}

void CGHermiteBathymetrySmoother::write_control_points_vtk(const std::string &filename) const {
    if (!solved_) {
        throw std::runtime_error(
            "CGHermiteBathymetrySmoother: must solve() before write_control_points_vtk()");
    }

    const int n1d = basis_->num_nodes_1d();
    io::write_bezier_control_points_vtk(
        filename, *quadtree_,
        [this](Index elem) { return element_bernstein_coefficients(elem); },
        [n1d](int dof) {
            return Vec2(static_cast<Real>(dof % n1d) / (n1d - 1),
                        static_cast<Real>(dof / n1d) / (n1d - 1));
        },
        n1d);
}

} // namespace drifter
