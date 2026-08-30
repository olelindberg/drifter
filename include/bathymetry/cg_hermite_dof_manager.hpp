#pragma once

/// @file cg_hermite_dof_manager.hpp
/// @brief Global DOF numbering for the CG Hermite bathymetry smoother
///
/// A Hermite node carries (r+1)^2 degrees of freedom - the value and its
/// derivatives up to order r in each direction - so unlike a Bezier basis a
/// quantized position alone does not identify a DOF. This manager keeps a map
/// from position to *node* and derives
///
///     global_dof = node_index * (r+1)^2 + derivative_index
///
/// with derivative_index = a + (r+1)*b for the multi-index (a, b).
///
/// Identifying all (r+1)^2 DOFs of a shared node between neighbouring elements
/// makes C^r continuity structural along every conforming edge - there are no
/// edge derivative constraints and no KKT system.
/// See docs/hermite_bathymetry_system.md S3.
///
/// Two kinds of constraint remain, and both are pure master/slave
/// **substitutions** rather than side conditions, so they condense to a smaller
/// SPD system rather than producing a saddle point:
///   - hanging nodes at 2:1 T-junctions (S8)
///   - zero normal gradient boundary conditions, as eliminations (S9)
///   - Dirichlet pins holding the surface at depth 0 over non-water elements

#include "bathymetry/cg_surface_dof_manager_base.hpp"
#include "bathymetry/element_data_mask.hpp"
#include "bathymetry/hermite_basis_2d.hpp"
#include "core/types.hpp"
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace drifter {

/// @brief Master/slave substitution for a Hermite DOF
///
/// x_slave = sum_i weights[i] * x_(master_dofs[i]). An empty master list means
/// the DOF is pinned to zero, which is how strong boundary conditions are
/// expressed. Layout matches back_substitute_slaves() in constraint_condenser.hpp.
struct HermiteConstraint {
    Index slave_dof = -1;
    std::vector<Index> master_dofs;
    std::vector<Real> weights;
};

/// @brief DOF manager for the Hermite bathymetry smoother
class CGHermiteDofManager : public CGSurfaceDofManagerBase {
public:
    /// @brief Build the DOF map for a quadtree mesh
    /// @param mesh 2D quadtree
    /// @param r Continuity order (0 or 1; 2 is supported by the element library
    ///          but not yet exercised by a smoother)
    /// @param enable_zero_gradient_bc Pin normal-derivative DOFs on the domain
    ///        boundary, giving an exact zero normal gradient (symmetry) condition
    /// @param mask Per-element water / beach / inland classification. Non-water
    ///        elements are held at depth 0. Null (the default) pins nothing. The
    ///        mask must outlive the constructor call only; it is not retained.
    CGHermiteDofManager(const QuadtreeAdapter &mesh, int r,
                        bool enable_zero_gradient_bc = false,
                        const ElementDataMask *mask = nullptr);

    // =========================================================================
    // CGSurfaceDofManagerBase interface
    // =========================================================================

    int num_element_dofs() const override { return basis_.num_dofs(); }

    // =========================================================================
    // Hermite-specific queries
    // =========================================================================

    /// @brief Continuity order
    int r() const { return r_; }

    /// @brief Number of DOFs carried by each node, (r+1)^2
    int dofs_per_node() const { return dofs_per_node_; }

    /// @brief Number of distinct nodes
    Index num_nodes() const { return num_nodes_; }

    /// @brief All substitutions (hanging nodes and pinned boundary DOFs)
    const std::vector<HermiteConstraint> &constraints() const { return constraints_; }

    /// @brief Derivative multi-index (a, b) of a global DOF
    std::pair<int, int> deriv_order(Index global_dof) const;

    /// @brief Total derivative order a + b of a global DOF
    ///
    /// Governs the DOF's physical units, and hence its row scaling in Q.
    int total_deriv_order(Index global_dof) const;

    /// @brief Nodal length scale of a global DOF (smallest adjacent element size)
    Real length_scale(Index global_dof) const;

    /// @brief Symmetric equilibration scaling S = diag(l_I^|alpha_I|)
    ///
    /// Hermite DOFs are dimensionally inhomogeneous - z has units of length, z_x
    /// is dimensionless, z_xy has units of 1/length - so rows of Q scale like
    /// h^-|alpha| and the condition number picks up a factor (h_max/h_min)^2r of
    /// pure scaling on an adaptive mesh. Solving (S Q S)(S^-1 x) = S b is a
    /// symmetric congruence that restores the DOF types to comparable magnitude
    /// while preserving symmetry and definiteness.
    /// See docs/hermite_bathymetry_system.md S11.
    ///
    /// @return Vector of length num_global_dofs()
    VecX equilibration_scaling() const;

    /// @brief The element basis
    const HermiteBasis2D &basis() const { return basis_; }

    /// @brief Number of DOFs held at zero by the non-water Dirichlet condition
    Index num_pinned_dofs() const { return num_pinned_dofs_; }

protected:
    std::pair<int64_t, int64_t> quantize_position(const Vec2 &pos) const override;
    size_t num_constraints_impl() const override { return constraints_.size(); }
    void get_constraint_triplets(std::vector<Eigen::Triplet<Real>> &triplets) const override;

private:
    /// @brief Assign node indices by position and fill elem_to_global_
    void assign_node_dofs();

    /// @brief Build 2:1 T-junction substitutions from the midpoint matrix G_r
    void build_hanging_node_constraints();

    /// @brief Pin normal-derivative DOFs on the domain boundary
    void build_zero_gradient_constraints();

    /// @brief Pin every DOF of every non-water element to zero
    ///
    /// A homogeneous Dirichlet condition, expressed as a slave with no masters, so
    /// it costs nothing beyond the existing condensation: Q_red = T^T Q T stays SPD.
    ///
    /// **All** (r+1)^2 DOFs of each corner are pinned, not just the value. Pinning
    /// the value alone leaves the corner's derivative DOFs free, and since those are
    /// shared with the water element next door, the fitted surface arrives at the
    /// rim still sloping downward and the bicubic carries that slope on into the
    /// pinned element: a 100 m gap fitted that way reaches -321 m at its centre,
    /// because a Hermite basis has no convex-hull property to bound it. Pinning the
    /// full node makes the pinned element identically zero, so the value the solver
    /// holds, the value written to VTK and the value handed to SeabedSurface all
    /// agree.
    ///
    /// Because the corners are shared, this is simultaneously the water region's
    /// boundary condition: z = 0 with zero normal gradient at the shoreline and at
    /// the rim of every gap. A coastal water element therefore runs from its
    /// offshore depth to 0 as a smoothstep - monotone, with no overshoot.
    ///
    /// Every DOF of a pinned element being constrained is also what lets
    /// CGSmootherBase skip those elements in assembly: no free DOF is left without
    /// an assembled element to support it.
    ///
    /// This is nodal, so it only bites where the mesh has a node.
    void build_dirichlet_pin_constraints(const ElementDataMask &mask);

    /// @brief Resolve masters that are themselves slaves, to a fixpoint
    ///
    /// 2:1 balance permits an element at level L to neighbour elements at L-1 and
    /// L+1 simultaneously, so a coarse-element corner can itself be the midpoint
    /// of an even coarser edge. Constraints must be closed transitively before
    /// condensation. See docs/hermite_bathymetry_system.md S8.
    void close_constraints_transitively();

    /// @brief Compute per-DOF length scales for equilibration
    void compute_length_scales();

    /// @brief Node index at a position, or -1
    Index find_node(const Vec2 &pos) const;

    /// @brief Register (or find) the node at a position
    Index find_or_register_node(const Vec2 &pos);

    /// @brief Global DOF for a node and derivative multi-index
    Index node_dof(Index node, int a, int b) const {
        return node * dofs_per_node_ + a + (r_ + 1) * b;
    }

    int r_;
    int dofs_per_node_;
    HermiteBasis2D basis_;
    bool enable_zero_gradient_bc_;

    Index num_nodes_ = 0;
    Index num_pinned_dofs_ = 0;

    /// Position -> node index (nodes carry dofs_per_node_ DOFs each)
    std::map<std::pair<int64_t, int64_t>, Index> position_to_node_;

    std::vector<HermiteConstraint> constraints_;

    /// Per-global-DOF metadata, permuted alongside the Morton reordering.
    /// Stored explicitly because reordering destroys any arithmetic relation
    /// between a DOF index and its node/derivative multi-index.
    std::vector<std::pair<int, int>> dof_deriv_order_;
    std::vector<Real> dof_length_scale_;

    /// Domain origin and quantization scale (mesh-relative, as the linear manager)
    Real xmin_domain_ = 0.0;
    Real ymin_domain_ = 0.0;
    Real inv_quantization_tol_ = 1.0;
};

} // namespace drifter
