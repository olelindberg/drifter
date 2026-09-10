#include "bathymetry/constraint_condenser.hpp"

namespace drifter {

namespace {

/// @brief The T operator as a flat CSR table
///
/// Global DOF g expands to the (free index, weight) pairs in
/// entries[offsets[g] .. offsets[g+1]). Building this once turns condensation
/// into a pass with no allocation: expand_dof() is a std::function returning a
/// fresh std::vector, so calling it twice per nonzero - as the loops below used
/// to - costs two heap allocations per nonzero, tens of millions of them on a
/// refined mesh, for a result that depends only on the DOF.
struct ExpansionTable {
    std::vector<Index> offsets;
    std::vector<std::pair<Index, Real>> entries;

    Index begin(Index g) const { return offsets[static_cast<size_t>(g)]; }
    Index end(Index g) const { return offsets[static_cast<size_t>(g) + 1]; }
    Index size(Index g) const { return end(g) - begin(g); }
    const std::pair<Index, Real> &at(Index i) const {
        return entries[static_cast<size_t>(i)];
    }
};

ExpansionTable build_expansion_table(
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof,
    Index num_dofs) {
    ExpansionTable table;
    table.offsets.resize(static_cast<size_t>(num_dofs) + 1);
    table.entries.reserve(static_cast<size_t>(num_dofs));

    for (Index g = 0; g < num_dofs; ++g) {
        table.offsets[static_cast<size_t>(g)] = static_cast<Index>(table.entries.size());
        for (const auto &e : expand_dof(g)) {
            table.entries.push_back(e);
        }
    }
    table.offsets[static_cast<size_t>(num_dofs)] = static_cast<Index>(table.entries.size());
    return table;
}

/// @brief Number of triplets the condensation of M will emit
///
/// Counted rather than guessed so the triplet vector is allocated once: it holds
/// one entry per (expanded row, expanded column) pair, which is at least
/// nonZeros() and more wherever a slave DOF is involved.
size_t count_condensed_triplets(const SpMat &M, const ExpansionTable &table) {
    size_t count = 0;
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMat::InnerIterator it(M, k); it; ++it) {
            count += static_cast<size_t>(table.size(it.row())) *
                     static_cast<size_t>(table.size(it.col()));
        }
    }
    return count;
}

/// @brief Emit the triplets of T' M T
void condense_into_triplets(const SpMat &M, const ExpansionTable &table,
                            std::vector<Eigen::Triplet<Real>> &triplets) {
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMat::InnerIterator it(M, k); it; ++it) {
            const Index I = it.row();
            const Index J = it.col();
            const Real val = it.value();

            for (Index ii = table.begin(I); ii < table.end(I); ++ii) {
                const auto &[If, Iw] = table.at(ii);
                const Real row_val = val * Iw;
                for (Index jj = table.begin(J); jj < table.end(J); ++jj) {
                    const auto &[Jf, Jw] = table.at(jj);
                    triplets.emplace_back(If, Jf, row_val * Jw);
                }
            }
        }
    }
}

} // namespace

std::pair<SpMat, VecX> assemble_kkt(const SpMat &Q, const SpMat &A, const VecX &b,
                                    const VecX &b_constraint,
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
    if (b_constraint.size() == num_constraints) {
        rhs.tail(num_constraints) = b_constraint;
    } else {
        rhs.tail(num_constraints).setZero();
    }

    return {std::move(KKT), std::move(rhs)};
}

std::pair<SpMat, VecX> assemble_kkt(const SpMat &Q, const SpMat &A, const VecX &b,
                                    Real constraint_reg) {
    return assemble_kkt(Q, A, b, VecX(), constraint_reg);
}

void condense_matrix_and_rhs(
    const SpMat &Q, const VecX &c,
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof, Index num_free,
    SpMat &Q_reduced, VecX &c_reduced) {

    const ExpansionTable table = build_expansion_table(expand_dof, Q.rows());

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(count_condensed_triplets(Q, table));
    c_reduced = VecX::Zero(num_free);

    // Condense Q matrix
    // For each entry Q(I, J), expand both I and J to free DOF indices
    // and add contributions weighted by the expansion weights
    condense_into_triplets(Q, table, triplets);

    // Condense RHS vector
    // For each entry c(g), expand g to free DOF indices and add weighted contribution
    Index num_dofs = c.size();
    for (Index g = 0; g < num_dofs; ++g) {
        for (Index i = table.begin(g); i < table.end(g); ++i) {
            const auto &[gf, gw] = table.at(i);
            c_reduced(gf) += c(g) * gw;
        }
    }

    Q_reduced.resize(num_free, num_free);
    Q_reduced.setFromTriplets(triplets.begin(), triplets.end());
}

SpMat condense_matrix(
    const SpMat &M,
    const std::function<std::vector<std::pair<Index, Real>>(Index)> &expand_dof,
    Index num_free) {

    const ExpansionTable table = build_expansion_table(expand_dof, M.rows());

    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(count_condensed_triplets(M, table));

    // Condense matrix: for each entry M(I, J), expand both I and J to free DOF indices
    // and add contributions weighted by the expansion weights
    condense_into_triplets(M, table, triplets);

    SpMat M_reduced(num_free, num_free);
    M_reduced.setFromTriplets(triplets.begin(), triplets.end());
    return M_reduced;
}

} // namespace drifter
