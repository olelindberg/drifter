/// @file test_hermite_solver_comparison.cpp
/// @brief Times every compiled-in Hermite direct solver on the same system
///
/// Q_red is SPD, so the candidates are mostly orderings and supernodal-or-not
/// variations of a Cholesky. The one structural difference is threading:
/// everything Eigen and SuiteSparse offer here is single-threaded, and PARDISO
/// is not, which is why the thread count is reported alongside the timings.
///
/// Which backends exist is a build decision, so run this configured with the lot:
///
///     cmake -B build -DDRIFTER_USE_UMFPACK=ON -DDRIFTER_USE_CHOLMOD=ON \
///                    -DDRIFTER_USE_MKL=ON
///     MKL_NUM_THREADS=4 ./build/tests/drifter_benchmarks \
///         --gtest_filter="HermiteSolverComparison*"
///
/// The numbers quoted at the top of src/bathymetry/cg_hermite_bathymetry_smoother.cpp
/// come from here.

#include "bathymetry/adaptive_cg_hermite_smoother.hpp"
#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "core/enum_strings.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#ifdef DRIFTER_USE_MKL
#include <mkl_service.h>
#endif

using namespace drifter;

namespace {

/// One backend's result on one mesh
struct SolverResult {
    HermiteSolverKind kind;
    double factorize_ms = 0.0;
    double substitute_ms = 0.0;
    Real data_residual = 0.0;
    Index num_free_dofs = 0;
    /// Largest pointwise difference from the SimplicialLDLT reference solution
    Real max_diff_from_reference = 0.0;
};

/// Every kind whose backend was compiled into this binary. Deliberately built
/// from the #ifdefs rather than from the enum, so a backend that is not linked
/// is skipped here instead of throwing.
std::vector<HermiteSolverKind> available_kinds() {
    std::vector<HermiteSolverKind> kinds{HermiteSolverKind::SimplicialLDLT,
                                         HermiteSolverKind::SimplicialLLT};
#ifdef DRIFTER_USE_METIS
    kinds.push_back(HermiteSolverKind::SimplicialLDLTMetis);
#endif
#ifdef DRIFTER_USE_MKL
    kinds.push_back(HermiteSolverKind::PardisoLDLT);
    kinds.push_back(HermiteSolverKind::PardisoLLT);
#endif
#ifdef DRIFTER_USE_UMFPACK
    kinds.push_back(HermiteSolverKind::UmfPackLU);
#endif
#ifdef DRIFTER_USE_CHOLMOD
    kinds.push_back(HermiteSolverKind::CholmodSimplicialLDLT);
    kinds.push_back(HermiteSolverKind::CholmodSupernodalLLT);
    kinds.push_back(HermiteSolverKind::CholmodSupernodalNesdis);
#endif
    return kinds;
}

/// Reports the threads MKL will actually use, so that a PARDISO run pinned to
/// one core is visible as a misconfiguration rather than read as "PARDISO lost".
int mkl_threads() {
#ifdef DRIFTER_USE_MKL
    return mkl_get_max_threads();
#else
    return 0;
#endif
}

} // namespace

class HermiteSolverComparisonTest : public ::testing::Test {
protected:
    static constexpr Real L = 1000.0;

    /// Enough structure that the fit is not trivially smooth, so the
    /// factorisation sees a realistic operator rather than a near-null one
    static Real bathymetry(Real x, Real y) {
        const Real kx = 3.0 * M_PI / L;
        const Real ky = 2.0 * M_PI / L;
        return -50.0 + 20.0 * std::sin(kx * x) * std::cos(ky * y) +
               5.0 * std::sin(4.0 * kx * x) * std::sin(3.0 * ky * y);
    }

    static CGHermiteSmootherConfig config_for(HermiteSolverKind kind) {
        CGHermiteSmootherConfig config;
        config.continuity_order = 1;
        config.lambda = 100.0;
        config.ngauss_data = 4;
        config.use_equilibration = true;
        config.solver = kind;
        return config;
    }

    /// Solves on a uniform n x n mesh and records the two solver phases
    static SolverResult run_one(HermiteSolverKind kind, int n,
                                const VecX *reference_surface,
                                VecX *out_surface) {
        QuadtreeAdapter mesh;
        mesh.build_uniform(0.0, L, 0.0, L, n, n);

        CGHermiteBathymetrySmoother smoother(mesh, config_for(kind));
        smoother.set_bathymetry_data(bathymetry);

        HermiteIterationProfile profile;
        smoother.set_profile(&profile);
        smoother.solve();

        SolverResult result;
        result.kind = kind;
        result.factorize_ms = profile.factorize_ms;
        result.substitute_ms = profile.substitute_ms;
        result.data_residual = smoother.data_residual();
        result.num_free_dofs = smoother.num_free_dofs();

        // Sample the fitted surface rather than the DOF vector: the backends
        // agree on the surface, which is the quantity that matters, and the DOF
        // ordering is the same only because the mesh is.
        constexpr int SAMPLES = 21;
        VecX surface(SAMPLES * SAMPLES);
        for (int i = 0; i < SAMPLES; ++i) {
            for (int j = 0; j < SAMPLES; ++j) {
                const Real x = L * i / (SAMPLES - 1);
                const Real y = L * j / (SAMPLES - 1);
                surface(i * SAMPLES + j) = smoother.evaluate(x, y);
            }
        }
        if (reference_surface) {
            result.max_diff_from_reference =
                (surface - *reference_surface).cwiseAbs().maxCoeff();
        }
        if (out_surface) {
            *out_surface = surface;
        }
        return result;
    }

    static void print_table(int n, const std::vector<SolverResult> &results) {
        std::cout << "\n=== Hermite direct solvers, C1, uniform " << n << " x " << n
                  << " (" << results.front().num_free_dofs << " free DOFs) ===\n"
                  << std::left << std::setw(26) << "solver" << std::right << std::setw(12)
                  << "factor_ms" << std::setw(12) << "subst_ms" << std::setw(14) << "residual"
                  << std::setw(14) << "vs_ref" << "\n";
        for (const auto &r : results) {
            std::cout << std::left << std::setw(26) << to_string(r.kind) << std::right
                      << std::fixed << std::setprecision(1) << std::setw(12) << r.factorize_ms
                      << std::setw(12) << r.substitute_ms << std::scientific
                      << std::setprecision(3) << std::setw(14) << r.data_residual << std::setw(14)
                      << r.max_diff_from_reference << "\n";
        }
        std::cout << std::defaultfloat;
    }
};

// =============================================================================
// The comparison itself
// =============================================================================

// Every compiled-in backend factorises the same system and must reach the same
// surface. The timings are the output; the agreement is the assertion, since a
// fast solver that returns a different answer is not a faster solver.
TEST_F(HermiteSolverComparisonTest, AllBackendsAgreeAndAreTimed) {
    const auto kinds = available_kinds();
    std::cout << "\nCompiled-in backends: " << kinds.size()
              << "; MKL threads: " << mkl_threads() << " (0 = MKL not compiled in)\n";

    // n = 128 puts ~67k free DOFs on the mesh, within reach of the ~88k of the
    // Kattegat run this is meant to inform. The small sizes are kept because the
    // ranking inverts below a few thousand DOFs, and knowing where it inverts is
    // part of the answer.
    for (int n : {16, 64, 128}) {
        // SimplicialLDLT is the reference: it is the default and needs no
        // optional dependency, so it is the one always present to compare to.
        VecX reference;
        std::vector<SolverResult> results;
        results.push_back(run_one(HermiteSolverKind::SimplicialLDLT, n, nullptr, &reference));

        for (const auto kind : kinds) {
            if (kind == HermiteSolverKind::SimplicialLDLT) {
                continue;
            }
            results.push_back(run_one(kind, n, &reference, nullptr));
        }

        print_table(n, results);

        const Real scale = reference.cwiseAbs().maxCoeff();
        for (const auto &r : results) {
            EXPECT_LT(r.max_diff_from_reference, 1e-7 * scale)
                << to_string(r.kind) << " disagrees with SimplicialLDLT at n=" << n;
            EXPECT_GT(r.factorize_ms, 0.0) << to_string(r.kind) << " reported no factorisation time";
        }
    }
}

// The point of PARDISO is the cores. If MKL is compiled in but pinned to one
// thread, its timings say nothing about what it would do in production, so make
// that visible rather than letting it be read as a loss.
#ifdef DRIFTER_USE_MKL
TEST_F(HermiteSolverComparisonTest, PardisoHasMoreThanOneThread) {
    const int threads = mkl_threads();
    std::cout << "MKL_NUM_THREADS in effect: " << threads << "\n";
    EXPECT_GT(threads, 1) << "PARDISO is running single-threaded, which removes its only "
                             "structural advantage over SimplicialLDLT. Set MKL_NUM_THREADS.";
}
#endif
