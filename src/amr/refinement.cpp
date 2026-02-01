// AMR Refinement implementation
#include "amr/refinement.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace drifter {

// ----------------------------------------------------------------------------
// ErrorEstimator factory methods
// ----------------------------------------------------------------------------

std::unique_ptr<ErrorEstimator> ErrorEstimator::gradient_based() {
    HexahedronBasis basis(2);  // Default order
    return std::make_unique<GradientErrorEstimator>(basis);
}

std::unique_ptr<ErrorEstimator> ErrorEstimator::residual_based() {
    HexahedronBasis basis(2);
    return std::make_unique<JumpErrorEstimator>(basis);
}

std::unique_ptr<ErrorEstimator> ErrorEstimator::spectral_decay() {
    // Return gradient-based estimator as fallback (wavelet indicator removed)
    HexahedronBasis basis(2);
    return std::make_unique<GradientErrorEstimator>(basis);
}

std::unique_ptr<ErrorEstimator> ErrorEstimator::feature_based(
    std::function<Real(const Vec3&)> /*feature_detector*/) {
    return std::make_unique<FeatureBasedEstimator>();
}

// ----------------------------------------------------------------------------
// GradientErrorEstimator
// ----------------------------------------------------------------------------

GradientErrorEstimator::GradientErrorEstimator(const HexahedronBasis& basis,
                                                 const RefinementParams& params)
    : basis_(basis), params_(params) {}

void GradientErrorEstimator::estimate(const std::vector<VecX>& solution,
                                        const OctreeAdapter& mesh,
                                        std::vector<ErrorIndicator>& indicators) const {
    indicators.resize(mesh.num_elements());

    const auto& elements = mesh.elements();
    for (Index e = 0; e < mesh.num_elements(); ++e) {
        if (static_cast<size_t>(e) < solution.size()) {
            indicators[e] = estimate_element(solution[e], *elements[e]);
        }
    }

    // Normalize indicators by max value
    Real max_error = 0.0;
    for (const auto& ind : indicators) {
        max_error = std::max(max_error, ind.value);
    }

    if (max_error > 1e-14) {
        for (auto& ind : indicators) {
            ind.value /= max_error;
            ind.refine = ind.value > params_.refine_threshold;
            ind.coarsen = ind.value < params_.coarsen_threshold;
        }
    }
}

ErrorIndicator GradientErrorEstimator::estimate_element(const VecX& solution,
                                                           const OctreeNode& node) const {
    ErrorIndicator result;

    if (solution.size() == 0) return result;

    // Compute gradient-based error indicator
    // Using variation in solution as proxy for gradient
    Real max_val = solution.maxCoeff();
    Real min_val = solution.minCoeff();
    Real variation = max_val - min_val;

    // Scale by element size
    Vec3 size = node.bounds.size();
    Real h = std::cbrt(size(0) * size(1) * size(2));  // Characteristic length

    result.value = variation * h;

    // Compute directional gradients for refinement mask
    result.suggested_mask = compute_directional_mask(solution);

    return result;
}

RefineMask GradientErrorEstimator::compute_directional_mask(const VecX& solution) const {
    if (solution.size() == 0) return RefineMask::NONE;

    int n1d = basis_.order() + 1;
    if (solution.size() != n1d * n1d * n1d) return RefineMask::XYZ;

    RefineMask mask = RefineMask::NONE;

    // Compute variation in each direction
    Real var_x = 0.0, var_y = 0.0, var_z = 0.0;

    for (int k = 0; k < n1d; ++k) {
        for (int j = 0; j < n1d; ++j) {
            Real min_x = solution(k * n1d * n1d + j * n1d);
            Real max_x = min_x;
            for (int i = 1; i < n1d; ++i) {
                Real val = solution(k * n1d * n1d + j * n1d + i);
                min_x = std::min(min_x, val);
                max_x = std::max(max_x, val);
            }
            var_x = std::max(var_x, max_x - min_x);
        }
    }

    for (int k = 0; k < n1d; ++k) {
        for (int i = 0; i < n1d; ++i) {
            Real min_y = solution(k * n1d * n1d + i);
            Real max_y = min_y;
            for (int j = 1; j < n1d; ++j) {
                Real val = solution(k * n1d * n1d + j * n1d + i);
                min_y = std::min(min_y, val);
                max_y = std::max(max_y, val);
            }
            var_y = std::max(var_y, max_y - min_y);
        }
    }

    for (int j = 0; j < n1d; ++j) {
        for (int i = 0; i < n1d; ++i) {
            Real min_z = solution(j * n1d + i);
            Real max_z = min_z;
            for (int k = 1; k < n1d; ++k) {
                Real val = solution(k * n1d * n1d + j * n1d + i);
                min_z = std::min(min_z, val);
                max_z = std::max(max_z, val);
            }
            var_z = std::max(var_z, max_z - min_z);
        }
    }

    Real max_var = std::max({var_x, var_y, var_z});
    if (max_var < 1e-14) return RefineMask::NONE;

    // Refine in directions with significant variation
    Real threshold = params_.smoothness_factor * max_var;
    if (var_x > threshold) mask = mask | RefineMask::X;
    if (var_y > threshold) mask = mask | RefineMask::Y;
    if (var_z > threshold) mask = mask | RefineMask::Z;

    return mask;
}

// ----------------------------------------------------------------------------
// JumpErrorEstimator
// ----------------------------------------------------------------------------

JumpErrorEstimator::JumpErrorEstimator(const HexahedronBasis& basis,
                                         const RefinementParams& params)
    : basis_(basis), params_(params) {}

void JumpErrorEstimator::estimate(const std::vector<VecX>& /*solution*/,
                                    const OctreeAdapter& mesh,
                                    std::vector<ErrorIndicator>& indicators) const {
    indicators.resize(mesh.num_elements());
}

ErrorIndicator JumpErrorEstimator::estimate_element(const VecX& /*solution*/,
                                                       const OctreeNode& /*node*/) const {
    return ErrorIndicator();
}

void JumpErrorEstimator::set_neighbor_data(const std::vector<std::vector<VecX>>& face_neighbors) {
    face_neighbors_ = face_neighbors;
}

// ----------------------------------------------------------------------------
// FeatureBasedEstimator
// ----------------------------------------------------------------------------

void FeatureBasedEstimator::add_bathymetry_criterion(
    std::function<Real(Real, Real)> bathymetry, Real threshold) {
    bathymetry_criteria_.emplace_back(std::move(bathymetry), threshold);
}

void FeatureBasedEstimator::add_coastline_criterion(
    std::function<bool(Real, Real)> is_land, Real max_distance) {
    coastline_criteria_.emplace_back(std::move(is_land), max_distance);
}

void FeatureBasedEstimator::estimate(const std::vector<VecX>& /*solution*/,
                                       const OctreeAdapter& mesh,
                                       std::vector<ErrorIndicator>& indicators) const {
    indicators.resize(mesh.num_elements());
}

ErrorIndicator FeatureBasedEstimator::estimate_element(const VecX& /*solution*/,
                                                          const OctreeNode& /*node*/) const {
    return ErrorIndicator();
}

// ----------------------------------------------------------------------------
// SolutionProjection
// ----------------------------------------------------------------------------

SolutionProjection::SolutionProjection(const HexahedronBasis& basis)
    : basis_(basis) {
    build_projection_matrices();
}

void SolutionProjection::project_to_children(const VecX& U_parent, RefineMask mask,
                                               std::vector<VecX>& U_children) const {
    int n_children = num_children(mask);
    U_children.resize(n_children);

    // Get parent nodes in reference coordinates
    const VecX& parent_nodes = basis_.lgl_basis_1d().nodes;
    int n1d = basis_.order() + 1;
    int ndof = n1d * n1d * n1d;

    // Each child occupies a sub-region of the parent
    bool ref_x = refines_x(mask);
    bool ref_y = refines_y(mask);
    bool ref_z = refines_z(mask);

    int nx = ref_x ? 2 : 1;
    int ny = ref_y ? 2 : 1;
    int nz = ref_z ? 2 : 1;

    int child_idx = 0;
    for (int cz = 0; cz < nz; ++cz) {
        for (int cy = 0; cy < ny; ++cy) {
            for (int cx = 0; cx < nx; ++cx) {
                U_children[child_idx].resize(ndof);

                // Map child reference coords to parent reference coords
                // Child [−1,1]^3 → parent sub-region
                Real xi_min = ref_x ? (-1.0 + cx) : -1.0;
                Real xi_max = ref_x ? (cx) : 1.0;
                Real eta_min = ref_y ? (-1.0 + cy) : -1.0;
                Real eta_max = ref_y ? (cy) : 1.0;
                Real zeta_min = ref_z ? (-1.0 + cz) : -1.0;
                Real zeta_max = ref_z ? (cz) : 1.0;

                // Evaluate parent polynomial at child nodes
                for (int k = 0; k < n1d; ++k) {
                    Real zeta_child = parent_nodes(k);
                    Real zeta_parent = zeta_min + 0.5 * (zeta_child + 1.0) * (zeta_max - zeta_min);

                    for (int j = 0; j < n1d; ++j) {
                        Real eta_child = parent_nodes(j);
                        Real eta_parent = eta_min + 0.5 * (eta_child + 1.0) * (eta_max - eta_min);

                        for (int i = 0; i < n1d; ++i) {
                            Real xi_child = parent_nodes(i);
                            Real xi_parent = xi_min + 0.5 * (xi_child + 1.0) * (xi_max - xi_min);

                            // Interpolate parent solution to this point
                            int child_dof = k * n1d * n1d + j * n1d + i;
                            U_children[child_idx](child_dof) = interpolate_3d(
                                U_parent, xi_parent, eta_parent, zeta_parent);
                        }
                    }
                }

                ++child_idx;
            }
        }
    }
}

void SolutionProjection::project_from_children(const std::vector<VecX>& U_children,
                                                  RefineMask mask,
                                                  VecX& U_parent) const {
    int n1d = basis_.order() + 1;
    int ndof = n1d * n1d * n1d;
    U_parent.resize(ndof);

    bool ref_x = refines_x(mask);
    bool ref_y = refines_y(mask);
    bool ref_z = refines_z(mask);

    int nx = ref_x ? 2 : 1;
    int ny = ref_y ? 2 : 1;
    int nz = ref_z ? 2 : 1;

    // L2 projection: use quadrature to integrate children over parent domain
    const VecX& nodes = basis_.lgl_basis_1d().nodes;
    const VecX& weights = basis_.lgl_basis_1d().weights;

    // Initialize with zeros
    U_parent.setZero();
    VecX mass_diag = VecX::Zero(ndof);

    // For each parent node, find which child(ren) contribute
    for (int pk = 0; pk < n1d; ++pk) {
        Real zeta_p = nodes(pk);
        int cz = ref_z ? (zeta_p >= 0 ? 1 : 0) : 0;

        for (int pj = 0; pj < n1d; ++pj) {
            Real eta_p = nodes(pj);
            int cy = ref_y ? (eta_p >= 0 ? 1 : 0) : 0;

            for (int pi = 0; pi < n1d; ++pi) {
                Real xi_p = nodes(pi);
                int cx = ref_x ? (xi_p >= 0 ? 1 : 0) : 0;

                int child_idx = cz * ny * nx + cy * nx + cx;
                int parent_dof = pk * n1d * n1d + pj * n1d + pi;

                if (child_idx < static_cast<int>(U_children.size())) {
                    // Map parent coords to child reference coords
                    Real xi_min = ref_x ? (-1.0 + cx) : -1.0;
                    Real xi_max = ref_x ? (cx) : 1.0;
                    Real eta_min = ref_y ? (-1.0 + cy) : -1.0;
                    Real eta_max = ref_y ? (cy) : 1.0;
                    Real zeta_min = ref_z ? (-1.0 + cz) : -1.0;
                    Real zeta_max = ref_z ? (cz) : 1.0;

                    Real xi_c = 2.0 * (xi_p - xi_min) / (xi_max - xi_min) - 1.0;
                    Real eta_c = 2.0 * (eta_p - eta_min) / (eta_max - eta_min) - 1.0;
                    Real zeta_c = 2.0 * (zeta_p - zeta_min) / (zeta_max - zeta_min) - 1.0;

                    // Clamp to valid range
                    xi_c = std::clamp(xi_c, -1.0, 1.0);
                    eta_c = std::clamp(eta_c, -1.0, 1.0);
                    zeta_c = std::clamp(zeta_c, -1.0, 1.0);

                    U_parent(parent_dof) = interpolate_3d(U_children[child_idx],
                                                           xi_c, eta_c, zeta_c);
                }

                // Mass lumping weight
                mass_diag(parent_dof) = weights(pi) * weights(pj) * weights(pk);
            }
        }
    }
}

Real SolutionProjection::interpolate_3d(const VecX& U, Real xi, Real eta, Real zeta) const {
    int n1d = basis_.order() + 1;
    const VecX& nodes = basis_.lgl_basis_1d().nodes;

    // Compute Lagrange interpolation basis at (xi, eta, zeta)
    VecX L_xi(n1d), L_eta(n1d), L_zeta(n1d);

    for (int i = 0; i < n1d; ++i) {
        L_xi(i) = 1.0;
        L_eta(i) = 1.0;
        L_zeta(i) = 1.0;

        for (int j = 0; j < n1d; ++j) {
            if (j != i) {
                L_xi(i) *= (xi - nodes(j)) / (nodes(i) - nodes(j));
                L_eta(i) *= (eta - nodes(j)) / (nodes(i) - nodes(j));
                L_zeta(i) *= (zeta - nodes(j)) / (nodes(i) - nodes(j));
            }
        }
    }

    // Evaluate interpolation
    Real result = 0.0;
    for (int k = 0; k < n1d; ++k) {
        for (int j = 0; j < n1d; ++j) {
            for (int i = 0; i < n1d; ++i) {
                int dof = k * n1d * n1d + j * n1d + i;
                result += U(dof) * L_xi(i) * L_eta(j) * L_zeta(k);
            }
        }
    }

    return result;
}

MatX SolutionProjection::child_projection_matrix(int child_index, RefineMask mask) const {
    auto it = child_projections_.find(mask);
    if (it != child_projections_.end() &&
        child_index < static_cast<int>(it->second.size())) {
        return it->second[child_index];
    }

    // Build on-the-fly
    int n1d = basis_.order() + 1;
    int ndof = n1d * n1d * n1d;
    return MatX::Identity(ndof, ndof);
}

MatX SolutionProjection::parent_projection_matrix(RefineMask mask) const {
    auto it = parent_projections_.find(mask);
    if (it != parent_projections_.end()) {
        return it->second;
    }

    int n1d = basis_.order() + 1;
    int ndof = n1d * n1d * n1d;
    return MatX::Identity(ndof, ndof);
}

void SolutionProjection::build_projection_matrices() {
    // Pre-build matrices for common refinement patterns
    // For now, use on-the-fly evaluation in project_to_children/project_from_children
}

// ----------------------------------------------------------------------------
// AdaptiveMeshRefinement
// ----------------------------------------------------------------------------

AdaptiveMeshRefinement::AdaptiveMeshRefinement(int order, const RefinementParams& params)
    : order_(order)
    , params_(params)
    , basis_(order)
    , solution_projection_(std::make_unique<SolutionProjection>(basis_))
    , mortar_manager_(std::make_unique<MortarInterfaceManager>(order)) {}

void AdaptiveMeshRefinement::set_error_estimator(std::unique_ptr<ErrorEstimator> estimator) {
    error_estimator_ = std::move(estimator);
}

int AdaptiveMeshRefinement::adapt(OctreeAdapter& mesh,
                                    std::vector<VecX>& solution,
                                    std::vector<std::vector<FaceConnection>>& face_connections) {
    // Step 1: Mark elements for refinement/coarsening
    std::vector<RefinementAction> actions;
    std::vector<RefineMask> masks;
    mark_elements(mesh, solution, actions, masks);

    // Count changes
    std::vector<Index> to_refine, to_coarsen;
    std::vector<RefineMask> refine_masks;

    for (Index e = 0; e < mesh.num_elements(); ++e) {
        if (actions[e] == RefinementAction::Refine) {
            to_refine.push_back(e);
            refine_masks.push_back(masks[e]);
        } else if (actions[e] == RefinementAction::Coarsen) {
            to_coarsen.push_back(e);
        }
    }

    if (to_refine.empty() && to_coarsen.empty()) {
        return 0;  // No changes
    }

    // Step 2: Store old solution for transfer
    std::vector<VecX> old_solution = solution;
    Index old_num_elements = mesh.num_elements();

    // Step 3: Apply refinement
    if (!to_refine.empty()) {
        // Build refinement map for solution transfer
        std::vector<std::pair<Index, std::vector<Index>>> refinement_map;

        for (size_t i = 0; i < to_refine.size(); ++i) {
            Index parent = to_refine[i];
            int n_children = num_children(refine_masks[i]);

            std::vector<Index> children;
            // Children will be assigned indices after refinement
            // For now, store relative indices
            for (int c = 0; c < n_children; ++c) {
                children.push_back(-1);  // Placeholder
            }
            refinement_map.push_back({parent, children});
        }

        mesh.refine(to_refine, refine_masks);

        // Step 4: Transfer solution
        transfer_solution(old_solution, refinement_map, solution);
    }

    // Step 5: Apply coarsening
    if (!to_coarsen.empty()) {
        mesh.coarsen(to_coarsen);
    }

    // Step 6: Rebuild connectivity
    rebuild_face_connections(mesh, face_connections);

    // Step 7: Update mortar interfaces
    update_mortar_interfaces(face_connections);

    return static_cast<int>(to_refine.size() + to_coarsen.size());
}

void AdaptiveMeshRefinement::mark_elements(const OctreeAdapter& mesh,
                                             const std::vector<VecX>& solution,
                                             std::vector<RefinementAction>& actions,
                                             std::vector<RefineMask>& masks) {
    Index num_elem = mesh.num_elements();
    actions.resize(num_elem, RefinementAction::None);
    masks.resize(num_elem, RefineMask::NONE);

    if (!error_estimator_) return;

    // Compute error indicators
    std::vector<ErrorIndicator> indicators;
    error_estimator_->estimate(solution, mesh, indicators);

    const auto& elements = mesh.elements();

    for (Index e = 0; e < num_elem; ++e) {
        const auto& ind = indicators[e];
        const auto& node = *elements[e];

        if (ind.refine && can_refine(node, ind.suggested_mask)) {
            actions[e] = RefinementAction::Refine;
            masks[e] = ind.suggested_mask != RefineMask::NONE ?
                       ind.suggested_mask : RefineMask::XYZ;
        } else if (ind.coarsen && can_coarsen(node)) {
            actions[e] = RefinementAction::Coarsen;
        }
    }
}

void AdaptiveMeshRefinement::refine_mesh(OctreeAdapter& mesh,
                                           const std::vector<Index>& elements_to_refine,
                                           const std::vector<RefineMask>& masks) {
    mesh.refine(elements_to_refine, masks);
}

void AdaptiveMeshRefinement::coarsen_mesh(OctreeAdapter& mesh,
                                            const std::vector<Index>& elements_to_coarsen) {
    mesh.coarsen(elements_to_coarsen);
}

void AdaptiveMeshRefinement::transfer_solution(
    const std::vector<VecX>& old_solution,
    const std::vector<std::pair<Index, std::vector<Index>>>& refinement_map,
    std::vector<VecX>& new_solution) {

    // For elements not in refinement map, copy directly
    // For refined elements, project to children

    // Start with copy
    new_solution.reserve(old_solution.size() * 2);

    std::set<Index> refined_parents;
    for (const auto& [parent, children] : refinement_map) {
        refined_parents.insert(parent);
    }

    // Copy non-refined elements
    for (Index e = 0; e < static_cast<Index>(old_solution.size()); ++e) {
        if (refined_parents.find(e) == refined_parents.end()) {
            new_solution.push_back(old_solution[e]);
        }
    }

    // Project refined elements to children
    for (const auto& [parent, children] : refinement_map) {
        if (parent < static_cast<Index>(old_solution.size())) {
            std::vector<VecX> child_solutions;
            solution_projection_->project_to_children(
                old_solution[parent], RefineMask::XYZ, child_solutions);

            for (const auto& child_sol : child_solutions) {
                new_solution.push_back(child_sol);
            }
        }
    }
}

void AdaptiveMeshRefinement::rebuild_face_connections(
    const OctreeAdapter& mesh,
    std::vector<std::vector<FaceConnection>>& face_connections) {
    face_connections = mesh.build_face_connections();
}

void AdaptiveMeshRefinement::update_mortar_interfaces(
    const std::vector<std::vector<FaceConnection>>& face_connections) {
    if (!mortar_manager_) return;

    // Rebuild mortar manager
    mortar_manager_ = std::make_unique<MortarInterfaceManager>(order_);

    for (size_t e = 0; e < face_connections.size(); ++e) {
        for (int f = 0; f < 6; ++f) {
            const auto& conn = face_connections[e][f];
            if (!conn.fine_elems.empty() &&
                conn.type != FaceConnectionType::SameLevel &&
                conn.type != FaceConnectionType::Boundary) {
                // Register non-conforming interface
                mortar_manager_->register_interface(conn);
            }
        }
    }

    // Build mortar operators
    mortar_manager_->build_operators();
}

bool AdaptiveMeshRefinement::can_refine(const OctreeNode& node, RefineMask /*mask*/) const {
    return node.level.level_x < params_.max_level &&
           node.level.level_y < params_.max_level &&
           node.level.level_z < params_.max_level;
}

bool AdaptiveMeshRefinement::can_coarsen(const OctreeNode& node) const {
    return node.level.level_x > params_.min_level ||
           node.level.level_y > params_.min_level ||
           node.level.level_z > params_.min_level;
}

void AdaptiveMeshRefinement::enforce_balance(OctreeAdapter& /*mesh*/) {
    // Stub: uses SeaMesh balance_octree
}

// ----------------------------------------------------------------------------
// AMRController (legacy)
// ----------------------------------------------------------------------------

AMRController::AMRController(std::shared_ptr<Mesh> mesh,
                               std::unique_ptr<ErrorEstimator> estimator)
    : mesh_(std::move(mesh)), estimator_(std::move(estimator)) {}

void AMRController::set_criteria(const RefinementCriteria& criteria) {
    criteria_ = criteria;
}

void AMRController::mark_elements() {
    // Stub
}

RefinementFlag AMRController::get_flag(Index elem_id) const {
    if (static_cast<size_t>(elem_id) < flags_.size()) {
        return flags_[elem_id];
    }
    return RefinementFlag::None;
}

void AMRController::adapt() {
    // Stub
}

void AMRController::project_solution() {
    // Stub
}

void AMRController::enforce_balance() {
    // Stub
}

Index AMRController::num_marked_refine() const {
    return 0;  // Stub
}

Index AMRController::num_marked_coarsen() const {
    return 0;  // Stub
}

Real AMRController::global_error_estimate() const {
    return 0.0;  // Stub
}

void AMRController::refine_element(Index /*elem_id*/) {
    // Stub
}

void AMRController::coarsen_element(Index /*elem_id*/) {
    // Stub
}

void AMRController::interpolate_to_children(const Element& /*parent*/,
                                              std::vector<Element>& /*children*/) {
    // Stub
}

void AMRController::project_from_children(const std::vector<Element>& /*children*/,
                                            Element& /*parent*/) {
    // Stub
}

// ----------------------------------------------------------------------------
// LoadBalancer
// ----------------------------------------------------------------------------

void LoadBalancer::set_strategy(Strategy strategy) {
    strategy_ = strategy;
}

std::vector<int> LoadBalancer::compute_partition(const Mesh& /*mesh*/, int num_procs) {
    return std::vector<int>(100, 0);  // Stub
}

std::vector<int> LoadBalancer::compute_partition(const OctreeAdapter& mesh, int num_procs) {
    std::vector<int> partition(mesh.num_elements());
    for (size_t i = 0; i < partition.size(); ++i) {
        partition[i] = static_cast<int>(i % num_procs);
    }
    return partition;
}

void LoadBalancer::migrate(Mesh& /*mesh*/, const std::vector<int>& /*new_partition*/) {
    // Stub
}

std::vector<Index> LoadBalancer::hilbert_order(const Mesh& /*mesh*/) {
    return {};  // Stub
}

std::vector<Index> LoadBalancer::morton_order(const OctreeAdapter& /*mesh*/) {
    return {};  // Stub
}

// ----------------------------------------------------------------------------
// AMRFluxComputation
// ----------------------------------------------------------------------------

AMRFluxComputation::AMRFluxComputation(const HexahedronBasis& basis,
                                         MortarInterfaceManager& mortar_manager)
    : basis_(basis), mortar_manager_(mortar_manager) {}

void AMRFluxComputation::compute_nonconforming_flux(
    const FaceConnection& /*conn*/,
    const VecX& /*U_coarse*/,
    const std::vector<VecX>& /*U_fine*/,
    const NumericalFluxFunc& /*numerical_flux*/,
    VecX& /*rhs_coarse*/,
    std::vector<VecX>& /*rhs_fine*/) const {
    // Stub
}

void AMRFluxComputation::compute_conforming_flux(
    int /*face_id_left*/, int /*face_id_right*/,
    const VecX& /*U_left*/, const VecX& /*U_right*/,
    const Vec3& /*normal*/,
    const NumericalFluxFunc& /*numerical_flux*/,
    VecX& /*rhs_left*/, VecX& /*rhs_right*/) const {
    // Stub
}

}  // namespace drifter
