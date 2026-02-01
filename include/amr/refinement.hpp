#pragma once

// Adaptive Mesh Refinement (AMR) for DG Ocean Model
//
// Provides h-refinement capabilities integrated with SeaMesh octree.
// Key features:
// - Directional (anisotropic) refinement following octree structure
// - 2:1 balancing constraint maintained by SeaMesh
// - L2 projection for solution transfer during refinement/coarsening
// - Non-conforming interface handling via mortar elements
//
// Refinement workflow:
// 1. Error estimation (gradient-based, feature-based, or physics-based)
// 2. Mark elements for refinement/coarsening
// 3. Apply refinement maintaining 2:1 balance
// 4. Project solution to new mesh
// 5. Rebuild connectivity and mortar interfaces

#include "core/types.hpp"
#include "dg/basis_hexahedron.hpp"
#include "dg/mortar.hpp"
#include "dg/face_connection.hpp"
#include "mesh/octree_adapter.hpp"
#include "mesh/refine_mask.hpp"
#include <functional>
#include <memory>
#include <vector>
#include <set>
#include <map>

namespace drifter {

// Forward declarations
class Element;
class Mesh;

/// @brief Refinement action for an element
enum class RefinementAction {
    None,      ///< No change
    Refine,    ///< Refine this element
    Coarsen    ///< Coarsen (merge with siblings)
};

/// @brief Legacy refinement flag (for compatibility)
enum class RefinementFlag : std::uint8_t {
    None,
    Refine,     // h-refine (split element)
    Coarsen,    // h-coarsen (merge with siblings)
    PRefine,    // Increase polynomial order
    PCoarsen    // Decrease polynomial order
};

/// @brief Error indicator result for an element
struct ErrorIndicator {
    Real value = 0.0;            ///< Error indicator value
    RefineMask suggested_mask = RefineMask::NONE;  ///< Suggested refinement directions
    bool refine = false;         ///< Should refine
    bool coarsen = false;        ///< Can coarsen
};

/// @brief Parameters for refinement criteria
struct RefinementParams {
    Real refine_threshold = 0.1;   ///< Refine if error > threshold
    Real coarsen_threshold = 0.01; ///< Coarsen if error < threshold
    int max_level = 8;             ///< Maximum refinement level per axis
    int min_level = 0;             ///< Minimum refinement level per axis
    bool allow_directional = true; ///< Allow directional (anisotropic) refinement
    Real smoothness_factor = 0.5;  ///< Factor for gradient smoothness in error
    bool enforce_balance = true;   ///< Enforce 2:1 balance constraint
};

/// @brief Legacy refinement criteria (for compatibility)
struct RefinementCriteria {
    Real refine_threshold = 0.7;
    Real coarsen_threshold = 0.1;
    int max_level = 10;
    int min_level = 0;
    int max_polynomial_order = 6;
    bool allow_p_refinement = true;
    bool allow_h_refinement = true;
    bool enforce_balance = true;
};

/// @brief Base class for error estimators
class ErrorEstimator {
public:
    virtual ~ErrorEstimator() = default;

    /// @brief Estimate error for all elements (octree version)
    virtual void estimate(const std::vector<VecX>& solution,
                           const OctreeAdapter& mesh,
                           std::vector<ErrorIndicator>& indicators) const = 0;

    /// @brief Estimate error for a single element (octree version)
    virtual ErrorIndicator estimate_element(const VecX& solution,
                                              const OctreeNode& node) const = 0;

    /// @brief Legacy interface: estimate for Element
    virtual Real estimate(const Element& elem) const { return 0.0; }

    // Factory methods
    static std::unique_ptr<ErrorEstimator> gradient_based();
    static std::unique_ptr<ErrorEstimator> residual_based();
    static std::unique_ptr<ErrorEstimator> spectral_decay();
    static std::unique_ptr<ErrorEstimator> feature_based(
        std::function<Real(const Vec3&)> feature_detector);
};

/// @brief Gradient-based error estimator
class GradientErrorEstimator : public ErrorEstimator {
public:
    /// @brief Construct gradient estimator
    /// @param basis Hexahedron basis for gradient computation
    /// @param params Refinement parameters
    GradientErrorEstimator(const HexahedronBasis& basis,
                            const RefinementParams& params = RefinementParams());

    void estimate(const std::vector<VecX>& solution,
                   const OctreeAdapter& mesh,
                   std::vector<ErrorIndicator>& indicators) const override;

    ErrorIndicator estimate_element(const VecX& solution,
                                      const OctreeNode& node) const override;

private:
    const HexahedronBasis& basis_;
    RefinementParams params_;

    /// @brief Compute directional gradients and suggest refinement mask
    RefineMask compute_directional_mask(const VecX& solution) const;
};

/// @brief Jump-based error estimator (face residual)
class JumpErrorEstimator : public ErrorEstimator {
public:
    JumpErrorEstimator(const HexahedronBasis& basis,
                        const RefinementParams& params = RefinementParams());

    void estimate(const std::vector<VecX>& solution,
                   const OctreeAdapter& mesh,
                   std::vector<ErrorIndicator>& indicators) const override;

    ErrorIndicator estimate_element(const VecX& solution,
                                      const OctreeNode& node) const override;

    /// @brief Set neighbor solution for jump computation
    void set_neighbor_data(const std::vector<std::vector<VecX>>& face_neighbors);

private:
    const HexahedronBasis& basis_;
    RefinementParams params_;
    std::vector<std::vector<VecX>> face_neighbors_;
};

/// @brief Feature-based refinement (bathymetry, coastline)
class FeatureBasedEstimator : public ErrorEstimator {
public:
    /// @brief Add bathymetry gradient criterion
    void add_bathymetry_criterion(std::function<Real(Real, Real)> bathymetry,
                                    Real threshold);

    /// @brief Add coastline proximity criterion
    void add_coastline_criterion(std::function<bool(Real, Real)> is_land,
                                   Real max_distance);

    void estimate(const std::vector<VecX>& solution,
                   const OctreeAdapter& mesh,
                   std::vector<ErrorIndicator>& indicators) const override;

    ErrorIndicator estimate_element(const VecX& solution,
                                      const OctreeNode& node) const override;

private:
    std::vector<std::pair<std::function<Real(Real, Real)>, Real>> bathymetry_criteria_;
    std::vector<std::pair<std::function<bool(Real, Real)>, Real>> coastline_criteria_;
};

/// @brief Solution projection during refinement/coarsening
class SolutionProjection {
public:
    /// @brief Construct projector
    /// @param basis Hexahedron basis
    SolutionProjection(const HexahedronBasis& basis);

    /// @brief Project solution to children during refinement
    void project_to_children(const VecX& U_parent, RefineMask mask,
                              std::vector<VecX>& U_children) const;

    /// @brief Project solution from children during coarsening
    void project_from_children(const std::vector<VecX>& U_children, RefineMask mask,
                                 VecX& U_parent) const;

    /// @brief L2 projection operator from parent to child
    MatX child_projection_matrix(int child_index, RefineMask mask) const;

    /// @brief L2 projection operator from children to parent
    MatX parent_projection_matrix(RefineMask mask) const;

private:
    const HexahedronBasis& basis_;
    std::map<RefineMask, std::vector<MatX>> child_projections_;
    std::map<RefineMask, MatX> parent_projections_;

    void build_projection_matrices();

    /// @brief Interpolate 3D solution at reference coordinates
    Real interpolate_3d(const VecX& U, Real xi, Real eta, Real zeta) const;
};

/// @brief Main AMR controller (octree-based)
class AdaptiveMeshRefinement {
public:
    /// @brief Construct AMR controller
    AdaptiveMeshRefinement(int order, const RefinementParams& params = RefinementParams());

    /// @brief Set error estimator
    void set_error_estimator(std::unique_ptr<ErrorEstimator> estimator);

    /// @brief Set refinement parameters
    void set_params(const RefinementParams& params) { params_ = params; }

    /// @brief Perform one adaptation cycle
    int adapt(OctreeAdapter& mesh,
              std::vector<VecX>& solution,
              std::vector<std::vector<FaceConnection>>& face_connections);

    /// @brief Mark elements for refinement/coarsening
    void mark_elements(const OctreeAdapter& mesh,
                        const std::vector<VecX>& solution,
                        std::vector<RefinementAction>& actions,
                        std::vector<RefineMask>& masks);

    /// @brief Apply refinement to mesh
    void refine_mesh(OctreeAdapter& mesh,
                      const std::vector<Index>& elements_to_refine,
                      const std::vector<RefineMask>& masks);

    /// @brief Apply coarsening to mesh
    void coarsen_mesh(OctreeAdapter& mesh,
                       const std::vector<Index>& elements_to_coarsen);

    /// @brief Transfer solution after mesh changes
    void transfer_solution(const std::vector<VecX>& old_solution,
                            const std::vector<std::pair<Index, std::vector<Index>>>& refinement_map,
                            std::vector<VecX>& new_solution);

    /// @brief Rebuild face connections after adaptation
    void rebuild_face_connections(const OctreeAdapter& mesh,
                                    std::vector<std::vector<FaceConnection>>& face_connections);

    /// @brief Update mortar interfaces
    void update_mortar_interfaces(const std::vector<std::vector<FaceConnection>>& face_connections);

    /// @brief Get mortar interface manager
    MortarInterfaceManager& mortar_manager() { return *mortar_manager_; }

private:
    int order_;
    RefinementParams params_;

    HexahedronBasis basis_;
    std::unique_ptr<ErrorEstimator> error_estimator_;
    std::unique_ptr<SolutionProjection> solution_projection_;
    std::unique_ptr<MortarInterfaceManager> mortar_manager_;

    bool can_refine(const OctreeNode& node, RefineMask mask) const;
    bool can_coarsen(const OctreeNode& node) const;
    void enforce_balance(OctreeAdapter& mesh);
};

/// @brief Legacy AMR controller (for compatibility)
class AMRController {
public:
    AMRController(std::shared_ptr<Mesh> mesh,
                  std::unique_ptr<ErrorEstimator> estimator);

    void set_criteria(const RefinementCriteria& criteria);
    void mark_elements();
    RefinementFlag get_flag(Index elem_id) const;
    void adapt();
    void project_solution();
    void enforce_balance();

    Index num_marked_refine() const;
    Index num_marked_coarsen() const;
    Real global_error_estimate() const;

private:
    std::shared_ptr<Mesh> mesh_;
    std::unique_ptr<ErrorEstimator> estimator_;
    RefinementCriteria criteria_;
    std::vector<RefinementFlag> flags_;
    std::vector<Real> error_indicators_;

    void refine_element(Index elem_id);
    void coarsen_element(Index elem_id);
    void interpolate_to_children(const Element& parent, std::vector<Element>& children);
    void project_from_children(const std::vector<Element>& children, Element& parent);
};

/// @brief Load balancing for parallel AMR
class LoadBalancer {
public:
    enum class Strategy {
        SpaceFillingCurve,
        GraphPartition,
        DiffusiveBalance
    };

    void set_strategy(Strategy strategy);

    std::vector<int> compute_partition(const Mesh& mesh, int num_procs);
    std::vector<int> compute_partition(const OctreeAdapter& mesh, int num_procs);

    void migrate(Mesh& mesh, const std::vector<int>& new_partition);

private:
    Strategy strategy_ = Strategy::SpaceFillingCurve;

    std::vector<Index> hilbert_order(const Mesh& mesh);
    std::vector<Index> morton_order(const OctreeAdapter& mesh);
};

/// @brief AMR flux computation with mortar elements
class AMRFluxComputation {
public:
    AMRFluxComputation(const HexahedronBasis& basis,
                        MortarInterfaceManager& mortar_manager);

    void compute_nonconforming_flux(
        const FaceConnection& conn,
        const VecX& U_coarse,
        const std::vector<VecX>& U_fine,
        const NumericalFluxFunc& numerical_flux,
        VecX& rhs_coarse,
        std::vector<VecX>& rhs_fine) const;

    void compute_conforming_flux(
        int face_id_left, int face_id_right,
        const VecX& U_left, const VecX& U_right,
        const Vec3& normal,
        const NumericalFluxFunc& numerical_flux,
        VecX& rhs_left, VecX& rhs_right) const;

private:
    const HexahedronBasis& basis_;
    MortarInterfaceManager& mortar_manager_;
};

}  // namespace drifter
