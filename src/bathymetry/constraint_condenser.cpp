#include "bathymetry/constraint_condenser.hpp"

namespace drifter {

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
