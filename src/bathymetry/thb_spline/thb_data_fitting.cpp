#include "bathymetry/thb_spline/thb_data_fitting.hpp"

#include <Eigen/SparseCore>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace drifter {

namespace {

/// Compute Gauss-Legendre nodes and weights on [0, 1]
void compute_gauss_quadrature(int ngauss, VecX& nodes, VecX& weights) {
    nodes.resize(ngauss);
    weights.resize(ngauss);

    if (ngauss == 1) {
        nodes << 0.5;
        weights << 1.0;
    } else if (ngauss == 2) {
        Real a = 0.5 / std::sqrt(3.0);
        nodes << 0.5 - a, 0.5 + a;
        weights << 0.5, 0.5;
    } else if (ngauss == 3) {
        Real a = 0.5 * std::sqrt(0.6);
        nodes << 0.5 - a, 0.5, 0.5 + a;
        weights << 5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0;
    } else if (ngauss == 4) {
        Real a = 0.5 * std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real b = 0.5 * std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0));
        Real wa = (18.0 + std::sqrt(30.0)) / 72.0;
        Real wb = (18.0 - std::sqrt(30.0)) / 72.0;
        nodes << 0.5 - b, 0.5 - a, 0.5 + a, 0.5 + b;
        weights << wb, wa, wa, wb;
    } else if (ngauss == 5) {
        Real a = std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 6.0;
        Real b = std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 6.0;
        Real wa = (322.0 + 13.0 * std::sqrt(70.0)) / 1800.0;
        Real wb = (322.0 - 13.0 * std::sqrt(70.0)) / 1800.0;
        Real wc = 128.0 / 450.0;
        nodes << 0.5 - b, 0.5 - a, 0.5, 0.5 + a, 0.5 + b;
        weights << wb, wa, wc, wa, wb;
    } else if (ngauss == 6) {
        Real x1 = 0.2386191860831969;
        Real x2 = 0.6612093864662645;
        Real x3 = 0.9324695142031521;
        Real w1 = 0.4679139345726910;
        Real w2 = 0.3607615730481386;
        Real w3 = 0.1713244923791704;
        nodes << (1.0 - x3) / 2.0, (1.0 - x2) / 2.0, (1.0 - x1) / 2.0,
            (1.0 + x1) / 2.0, (1.0 + x2) / 2.0, (1.0 + x3) / 2.0;
        weights << w3 / 2.0, w2 / 2.0, w1 / 2.0, w1 / 2.0, w2 / 2.0, w3 / 2.0;
    } else {
        throw std::invalid_argument("THBDataFitting: ngauss must be 1-6");
    }
}

}  // namespace

THBDataFitting::THBDataFitting(const QuadtreeAdapter& quadtree,
                               const THBHierarchy& hierarchy,
                               const THBRefinementMask& mask,
                               const THBTruncation& truncation)
    : quadtree_(quadtree),
      hierarchy_(hierarchy),
      mask_(mask),
      truncation_(truncation) {}

void THBDataFitting::set_from_bathymetry_source(const BathymetrySource& source,
                                                int ngauss) {
    // Get Gauss-Legendre nodes and weights on [0, 1]
    VecX gauss_nodes, gauss_weights;
    compute_gauss_quadrature(ngauss, gauss_nodes, gauss_weights);

    // Clear existing data
    data_x_.clear();
    data_y_.clear();
    data_z_.clear();
    data_w_.clear();

    // Reserve space (estimate)
    const Index num_elements = quadtree_.num_elements();
    const Index points_per_elem = ngauss * ngauss;
    data_x_.reserve(num_elements * points_per_elem);
    data_y_.reserve(num_elements * points_per_elem);
    data_z_.reserve(num_elements * points_per_elem);
    data_w_.reserve(num_elements * points_per_elem);

    // Sample at Gauss points in each element
    for (Index e = 0; e < num_elements; ++e) {
        const QuadBounds& bounds = quadtree_.element_bounds(e);
        const Real dx = bounds.xmax - bounds.xmin;
        const Real dy = bounds.ymax - bounds.ymin;
        const Real area = dx * dy;

        for (int j = 0; j < ngauss; ++j) {
            for (int i = 0; i < ngauss; ++i) {
                // Map Gauss point from [0,1] to element
                Real u = gauss_nodes(i);  // In [0, 1]
                Real v = gauss_nodes(j);  // In [0, 1]
                Real x = bounds.xmin + u * dx;
                Real y = bounds.ymin + v * dy;

                // Evaluate bathymetry
                Real z = source.evaluate(x, y);

                // Weight includes Jacobian and Gauss weight (already on [0,1])
                Real w = gauss_weights(i) * gauss_weights(j) * area;

                data_x_.push_back(x);
                data_y_.push_back(y);
                data_z_.push_back(z);
                data_w_.push_back(w);
            }
        }
    }
}

void THBDataFitting::set_from_function(std::function<Real(Real, Real)> bathy_func,
                                       int ngauss) {
    // Wrap function as BathymetrySource
    class FunctionSource : public BathymetrySource {
      public:
        std::function<Real(Real, Real)> func;
        Real evaluate(Real x, Real y) const override { return func(x, y); }
    };

    FunctionSource source;
    source.func = bathy_func;
    set_from_bathymetry_source(source, ngauss);
}

VecX THBDataFitting::evaluate_active_basis(Real x, Real y) const {
    // Find the finest refinement level at this point
    int point_level = mask_.level_at_point(x, y);

    // Use truncation to evaluate active basis functions at appropriate levels
    const auto& active_funcs = mask_.active_functions();
    VecX result = VecX::Zero(active_funcs.size());

    for (Index idx = 0; idx < static_cast<Index>(active_funcs.size()); ++idx) {
        const auto& [level, i, j] = active_funcs[idx];

        // Only evaluate functions at the SAME level as the point
        // This ensures partition of unity: at each point, only one level's
        // functions contribute, and their sum equals 1.0
        if (level != point_level) {
            continue;
        }

        // Convert physical to parameter coordinates at this level
        auto [u, v] = hierarchy_.physical_to_parameter(level, x, y);

        // Evaluate basis function (no truncation needed with single-level evaluation)
        const auto& basis = hierarchy_.basis(level);
        result(idx) = basis.evaluate(i, j, u, v);
    }

    return result;
}

void THBDataFitting::assemble_normal_equations(SpMat& AtWA, VecX& AtWb) const {
    const Index num_active = num_active_dofs();
    const Index num_pts = num_points();

    // Initialize outputs
    AtWb.setZero(num_active);

    // Use triplets for sparse matrix assembly
    std::vector<Eigen::Triplet<Real>> triplets;

    // Estimate non-zeros (sparse due to local support)
    // Each point contributes to at most 16 x max_level active functions
    const int max_nonzeros_per_point = 16 * (hierarchy_.max_level() + 1);
    triplets.reserve(num_pts * max_nonzeros_per_point * max_nonzeros_per_point / 4);

#ifdef _OPENMP
    // Parallel assembly with thread-local storage
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<Eigen::Triplet<Real>>> thread_triplets(num_threads);
    std::vector<VecX> thread_AtWb(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        thread_AtWb[t].setZero(num_active);
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_triplets = thread_triplets[tid];
        auto& local_AtWb = thread_AtWb[tid];

#pragma omp for schedule(static)
        for (Index p = 0; p < num_pts; ++p) {
            Real x = data_x_[p];
            Real y = data_y_[p];
            Real z = data_z_[p];
            Real w = data_w_[p];

            // Evaluate active basis functions at this point
            VecX phi = evaluate_active_basis(x, y);

            // Find non-zero entries
            for (Index i = 0; i < num_active; ++i) {
                if (std::abs(phi(i)) < 1e-14)
                    continue;

                // Add to RHS
                local_AtWb(i) += w * phi(i) * z;

                // Add to normal matrix
                for (Index j = 0; j < num_active; ++j) {
                    if (std::abs(phi(j)) < 1e-14)
                        continue;

                    local_triplets.push_back({static_cast<int>(i), static_cast<int>(j),
                                              w * phi(i) * phi(j)});
                }
            }
        }
    }

    // Merge thread-local results
    for (int t = 0; t < num_threads; ++t) {
        triplets.insert(triplets.end(), thread_triplets[t].begin(),
                        thread_triplets[t].end());
        AtWb += thread_AtWb[t];
    }

#else
    // Serial assembly
    for (Index p = 0; p < num_pts; ++p) {
        Real x = data_x_[p];
        Real y = data_y_[p];
        Real z = data_z_[p];
        Real w = data_w_[p];

        // Evaluate active basis functions at this point
        VecX phi = evaluate_active_basis(x, y);

        // Find non-zero entries
        for (Index i = 0; i < num_active; ++i) {
            if (std::abs(phi(i)) < 1e-14)
                continue;

            // Add to RHS
            AtWb(i) += w * phi(i) * z;

            // Add to normal matrix
            for (Index j = 0; j < num_active; ++j) {
                if (std::abs(phi(j)) < 1e-14)
                    continue;

                triplets.push_back(
                    {static_cast<int>(i), static_cast<int>(j), w * phi(i) * phi(j)});
            }
        }
    }
#endif

    // Build sparse matrix
    AtWA.resize(num_active, num_active);
    AtWA.setFromTriplets(triplets.begin(), triplets.end());
}

}  // namespace drifter
