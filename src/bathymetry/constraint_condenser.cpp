#include "bathymetry/constraint_condenser.hpp"

namespace drifter {

std::pair<SpMat, VecX> assemble_kkt(const SpMat &Q, const SpMat &A, const VecX &b,
                                    Real constraint_reg) {
    Index num_primal = Q.rows();
    Index num_constraints = A.rows();
    Index kkt_size = num_primal + num_constraints;

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(Q.nonZeros() + 2 * A.nonZeros());

    // Q block (upper-left)
    for (int k = 0; k < Q.outerSize(); ++k) {
        for (SpMat::InnerIterator it(Q, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }

    // A and A^T blocks
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A, k); it; ++it) {
            triplets.emplace_back(num_primal + it.row(), it.col(), it.value());
            triplets.emplace_back(it.col(), num_primal + it.row(), it.value());
        }
    }

    SpMat KKT(kkt_size, kkt_size);
    KKT.setFromTriplets(triplets.begin(), triplets.end());

    // -εI on constraint block
    for (Index i = num_primal; i < kkt_size; ++i) {
        KKT.coeffRef(i, i) -= constraint_reg;
    }

    VecX rhs(kkt_size);
    rhs.head(num_primal) = b;
    rhs.tail(num_constraints).setZero();

    return {std::move(KKT), std::move(rhs)};
}

void condense_matrix_and_rhs(
    const SpMat &Q, const VecX &c,
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof, Index num_free,
    SpMat &Q_reduced, VecX &c_reduced) {

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(Q.nonZeros());
    c_reduced = VecX::Zero(num_free);

    // Condense Q matrix
    // For each entry Q(I, J), expand both I and J to free DOF indices
    // and add contributions weighted by the expansion weights
    for (int k = 0; k < Q.outerSize(); ++k) {
        for (SpMat::InnerIterator it(Q, k); it; ++it) {
            Index I = it.row();
            Index J = it.col();
            Real val = it.value();

            auto I_expanded = expand_dof(I);
            auto J_expanded = expand_dof(J);

            for (const auto &[If, Iw] : I_expanded) {
                for (const auto &[Jf, Jw] : J_expanded) {
                    triplets.emplace_back(If, Jf, val * Iw * Jw);
                }
            }
        }
    }

    // Condense RHS vector
    // For each entry c(g), expand g to free DOF indices and add weighted contribution
    Index num_dofs = c.size();
    for (Index g = 0; g < num_dofs; ++g) {
        auto g_expanded = expand_dof(g);
        for (const auto &[gf, gw] : g_expanded) {
            c_reduced(gf) += c(g) * gw;
        }
    }

    Q_reduced.resize(num_free, num_free);
    Q_reduced.setFromTriplets(triplets.begin(), triplets.end());
}

} // namespace drifter
