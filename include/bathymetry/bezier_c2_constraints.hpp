#pragma once

/// @file bezier_c2_constraints.hpp
/// @brief C² continuity constraint builder for Bezier bathymetry patches
///
/// Builds equality constraint matrix for enforcing C² (curvature) continuity
/// across element interfaces. At shared vertices, this requires matching:
///   - Value: z
///   - First derivatives: dz/du, dz/dv
///   - Second derivatives: d²z/du², d²z/dv², d²z/dudv
///   - Third derivatives: d³z/du²dv, d³z/dudv²
///   - Fourth derivative: d⁴z/du²dv²
///
/// For degree-5 Bezier, this gives 9 constraints per shared vertex.

#include "core/types.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "bathymetry/bezier_basis_2d.hpp"
#include <memory>
#include <vector>

namespace drifter {

/// @brief Constraint type for a shared vertex
struct VertexConstraintInfo {
    Index elem1;          ///< First element sharing the vertex
    Index elem2;          ///< Second element sharing the vertex
    int corner1;          ///< Corner index (0-3) in elem1
    int corner2;          ///< Corner index (0-3) in elem2

    /// Physical position of shared vertex
    Vec2 position;
};

/// @brief Constraint type for a hanging node on a non-conforming edge
struct HangingNodeConstraintInfo {
    Index coarse_elem;    ///< Coarse element
    Index fine_elem;      ///< Fine element
    int coarse_edge;      ///< Edge ID on coarse element
    int fine_edge;        ///< Edge ID on fine element

    /// Parameter position of hanging node within coarse element's edge
    Real t_coarse;

    /// Control point indices on fine element edge that are constrained
    std::vector<int> fine_dofs;
};

/// @brief Constraint type for fine-to-fine continuity at a T-junction vertex
///
/// When two fine elements share a T-junction (hanging node) vertex on a coarse
/// edge, they must be constrained to each other (not just to the coarse element)
/// to ensure C² continuity between them.
struct TJunctionVertexInfo {
    Index fine_elem1;     ///< First fine element at the T-junction
    Index fine_elem2;     ///< Second fine element at the T-junction
    int corner1;          ///< Corner index (0-3) in fine_elem1
    int corner2;          ///< Corner index (0-3) in fine_elem2
    Real dx1, dy1;        ///< Size of fine_elem1
    Real dx2, dy2;        ///< Size of fine_elem2
};

/// @brief Constraint type for a Dirichlet boundary condition
struct DirichletConstraintInfo {
    Index elem;           ///< Element index
    int edge;             ///< Edge ID (0-3) on domain boundary
    int local_dof;        ///< Local DOF index within element (0-35)
    Index global_dof;     ///< Global DOF index

    /// Physical position of the control point
    Vec2 position;
};

/// @brief Constraint type for a conforming shared edge between elements
/// For C² continuity along the entire edge (not just vertices), interior
/// edge DOFs must match between the two elements.
struct EdgeConstraintInfo {
    Index elem1;          ///< First element sharing the edge
    Index elem2;          ///< Second element sharing the edge
    int edge1;            ///< Edge ID (0-3) on elem1
    int edge2;            ///< Edge ID (0-3) on elem2
};

/// @brief Constraint info for natural boundary condition (z_nn = 0)
///
/// Natural boundary conditions enforce zero normal curvature at boundary DOFs.
/// For axis-aligned edges:
///   - Left/right edges (n = (±1, 0)): z_xx = 0
///   - Bottom/top edges (n = (0, ±1)): z_yy = 0
///
/// At domain corners where two edges meet, both constraints are applied:
///   z_xx = 0 AND z_yy = 0
struct NaturalBCConstraintInfo {
    Index elem;           ///< Element index
    int local_dof;        ///< Local DOF index within element (0-35)
    Index global_dof;     ///< Global DOF index
    Vec2 position;        ///< Physical position of the control point
    Vec2 normal;          ///< Outward boundary normal direction
    int edge;             ///< Boundary edge ID (0=left, 1=right, 2=bottom, 3=top)
};

/// @brief Builds C² continuity constraint matrix for Bezier patches
///
/// Creates a sparse constraint matrix A_eq such that A_eq * x = 0 enforces
/// C² continuity at all shared vertices and edges.
class BezierC2ConstraintBuilder {
public:
    /// @brief Construct constraint builder for a mesh
    /// @param mesh 2D quadtree mesh
    explicit BezierC2ConstraintBuilder(const QuadtreeAdapter& mesh);

    /// @brief Build the full constraint matrix
    /// @return Sparse matrix (num_constraints x total_dofs) where total_dofs = 36 * num_elements
    SpMat build_constraint_matrix() const;

    /// @brief Get the number of constraints
    Index num_constraints() const;

    /// @brief Get number of DOFs per element (always 36 for quintic Bezier)
    int dofs_per_element() const { return BezierBasis2D::NDOF; }

    /// @brief Get total number of DOFs
    Index total_dofs() const {
        return dofs_per_element() * mesh_.num_elements();
    }

    /// @brief Get global DOF index for an element-local DOF
    /// @param elem Element index
    /// @param local_dof Local DOF index (0-35)
    Index global_dof(Index elem, int local_dof) const {
        return elem * dofs_per_element() + local_dof;
    }

    /// @brief Get all vertex constraints found
    const std::vector<VertexConstraintInfo>& vertex_constraints() const {
        return vertex_constraints_;
    }

    /// @brief Get all hanging node constraints found
    const std::vector<HangingNodeConstraintInfo>& hanging_node_constraints() const {
        return hanging_node_constraints_;
    }

    /// @brief Get all conforming edge constraints found
    const std::vector<EdgeConstraintInfo>& edge_constraints() const {
        return edge_constraints_;
    }

    /// @brief Get all T-junction vertex constraints (fine-to-fine)
    const std::vector<TJunctionVertexInfo>& tjunction_vertex_constraints() const {
        return t_junction_vertex_constraints_;
    }

    // =========================================================================
    // Dirichlet boundary constraint support
    // =========================================================================

    /// @brief Find all DOFs on domain boundary edges
    ///
    /// Returns a list of DirichletConstraintInfo for each DOF on the
    /// domain boundary. Corner DOFs shared by multiple boundary edges
    /// are included only once.
    void find_boundary_dofs() const;

    /// @brief Get all Dirichlet constraints found
    const std::vector<DirichletConstraintInfo>& dirichlet_constraints() const {
        if (!dirichlet_built_) {
            find_boundary_dofs();
        }
        return dirichlet_constraints_;
    }

    /// @brief Get number of Dirichlet boundary constraints
    Index num_dirichlet_constraints() const {
        if (!dirichlet_built_) {
            find_boundary_dofs();
        }
        return static_cast<Index>(dirichlet_constraints_.size());
    }

    /// @brief Build constraint matrix for Dirichlet BCs only
    /// @return Sparse matrix (num_dirichlet_constraints x total_dofs)
    /// Each row has a single 1.0 at the constrained DOF column
    SpMat build_dirichlet_constraint_matrix() const;

    /// @brief Build combined constraint matrix (C² continuity + Dirichlet)
    /// @return Sparse matrix with C² constraints first, then Dirichlet constraints
    SpMat build_combined_constraint_matrix() const;

    /// @brief Get total number of constraints including Dirichlet
    Index num_combined_constraints() const {
        return num_constraints() + num_dirichlet_constraints();
    }

    // =========================================================================
    // Natural boundary condition support (z_nn = 0 at boundary DOFs)
    // =========================================================================

    /// @brief Find all boundary DOFs that need natural BC constraints
    ///
    /// For each DOF on the domain boundary, creates a natural moment constraint
    /// enforcing z_nn = 0 where n is the outward normal to the boundary edge.
    /// At domain corners where two edges meet, two constraints are created
    /// (one for each edge normal, giving z_xx = 0 and z_yy = 0).
    void find_natural_bc_constraints() const;

    /// @brief Get all natural BC constraints found
    const std::vector<NaturalBCConstraintInfo>& natural_bc_constraints() const {
        if (!natural_bc_built_) {
            find_natural_bc_constraints();
        }
        return natural_bc_constraints_;
    }

    /// @brief Get number of natural BC constraints
    Index num_natural_bc_constraints() const {
        if (!natural_bc_built_) {
            find_natural_bc_constraints();
        }
        return static_cast<Index>(natural_bc_constraints_.size());
    }

    /// @brief Build constraint matrix for natural BCs only
    /// @return Sparse matrix (num_natural_bc_constraints x total_dofs)
    /// Each row encodes z_nn = 0 at a boundary DOF
    SpMat build_natural_bc_matrix() const;

    /// @brief Build combined constraint matrix (C² continuity + natural BC)
    /// @return Sparse matrix with C² constraints first, then natural BC constraints
    SpMat build_c2_and_natural_bc_matrix() const;

    /// @brief Get total number of C² + natural BC constraints
    Index num_c2_and_natural_bc_constraints() const {
        return num_constraints() + num_natural_bc_constraints();
    }

private:
    const QuadtreeAdapter& mesh_;
    std::unique_ptr<BezierBasis2D> basis_;

    /// Cached constraint info (C² continuity)
    mutable std::vector<VertexConstraintInfo> vertex_constraints_;
    mutable std::vector<HangingNodeConstraintInfo> hanging_node_constraints_;
    mutable std::vector<TJunctionVertexInfo> t_junction_vertex_constraints_;
    mutable std::vector<EdgeConstraintInfo> edge_constraints_;
    mutable bool constraints_built_ = false;

    /// Cached Dirichlet boundary constraints
    mutable std::vector<DirichletConstraintInfo> dirichlet_constraints_;
    mutable bool dirichlet_built_ = false;

    /// Cached natural BC constraints
    mutable std::vector<NaturalBCConstraintInfo> natural_bc_constraints_;
    mutable bool natural_bc_built_ = false;

    /// Find all shared vertices and hanging nodes
    void find_all_constraints() const;

    /// Add derivative matching constraints at a conforming vertex
    /// @param info Vertex constraint information
    /// @param triplets Output triplets for sparse matrix
    /// @param constraint_idx Current constraint row index (updated)
    void add_vertex_constraints(
        const VertexConstraintInfo& info,
        std::vector<Eigen::Triplet<Real>>& triplets,
        Index& constraint_idx) const;

    /// Add constraints for hanging nodes on non-conforming edges
    /// @param info Hanging node constraint information
    /// @param triplets Output triplets for sparse matrix
    /// @param constraint_idx Current constraint row index (updated)
    void add_hanging_node_constraints(
        const HangingNodeConstraintInfo& info,
        std::vector<Eigen::Triplet<Real>>& triplets,
        Index& constraint_idx) const;

    /// Add constraints for interior DOFs along conforming shared edges
    /// @param info Edge constraint information
    /// @param triplets Output triplets for sparse matrix
    /// @param constraint_idx Current constraint row index (updated)
    void add_conforming_edge_constraints(
        const EdgeConstraintInfo& info,
        std::vector<Eigen::Triplet<Real>>& triplets,
        Index& constraint_idx) const;

    /// Add C² constraints between two fine elements at a T-junction vertex
    /// @param info T-junction vertex constraint information
    /// @param triplets Output triplets for sparse matrix
    /// @param constraint_idx Current constraint row index (updated)
    void add_tjunction_vertex_constraints(
        const TJunctionVertexInfo& info,
        std::vector<Eigen::Triplet<Real>>& triplets,
        Index& constraint_idx) const;

    /// Get (u, v) parameter for a corner in element
    /// @param elem Element index
    /// @param corner_id Corner ID (0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1))
    /// @return Parameter coordinates in [0,1]²
    Vec2 get_corner_param(int corner_id) const;

    /// Get physical position of a corner
    Vec2 get_corner_position(Index elem, int corner_id) const;

    /// Map derivative order (nu, nv) to constraint index within vertex
    /// Returns index 0-8 for the 9 C² constraints
    int derivative_to_constraint_index(int nu, int nv) const;

    /// Check if two positions are the same (within tolerance)
    bool positions_match(const Vec2& p1, const Vec2& p2) const;

    /// Get the two edges meeting at a corner
    /// @param corner_id Corner ID (0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1))
    /// @return Pair of edge IDs meeting at this corner
    std::pair<int, int> corner_to_edges(int corner_id) const;

    /// Check if a vertex is on the domain boundary
    /// A vertex is on the boundary if ANY of the edges meeting at any element's
    /// corner at that position is a boundary edge
    bool is_boundary_vertex(const std::vector<std::pair<Index, int>>& elem_corners) const;

    /// Check if two elements share an interior edge at a given vertex
    /// This is used to determine if C² constraints should be applied between them.
    /// Returns true only if the elements share an edge (not just a vertex) and
    /// that edge is interior (not on the domain boundary).
    bool share_interior_edge_at_vertex(Index elem1, int corner1, Index elem2, int corner2) const;

    /// Get outward normal for a boundary edge
    /// @param edge Edge ID (0=left, 1=right, 2=bottom, 3=top)
    /// @return Unit outward normal vector
    Vec2 edge_outward_normal(int edge) const;

    /// Get physical position of a DOF within an element
    Vec2 get_dof_position(Index elem, int local_dof) const;

    /// Get list of local DOF indices on a given edge
    /// @param edge Edge ID (0=left, 1=right, 2=bottom, 3=top)
    /// @return Vector of local DOF indices (6 DOFs for quintic Bezier)
    std::vector<int> edge_dofs(int edge) const;
};

}  // namespace drifter
