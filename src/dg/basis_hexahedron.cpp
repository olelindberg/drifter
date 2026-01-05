// 3D Hexahedron basis implementation
// Adapted from wobbler: LegendrePolynomial.cpp

#include "dg/basis_hexahedron.hpp"
#include <cmath>
#include <stdexcept>

namespace drifter {

// =============================================================================
// Gauss-Legendre nodes and weights computation
// =============================================================================

void compute_gauss_legendre_nodes(int n, VecX& nodes, VecX& weights) {
    nodes.resize(n);
    weights.resize(n);

    if (n == 1) {
        nodes(0) = 0.0;
        weights(0) = 2.0;
        return;
    }

    if (n == 2) {
        nodes(0) = -std::sqrt(1.0 / 3.0);
        nodes(1) = -nodes(0);
        weights(0) = 1.0;
        weights(1) = 1.0;
        return;
    }

    // General case: Newton iteration for roots of P_n(x)
    for (int i = 0; i <= (n - 1) / 2; ++i) {
        // Initial guess
        Real xi = -std::cos((2.0 * i + 1.0) / (2.0 * n) * M_PI);

        // Newton iteration
        Real L, dL;
        for (int iter = 0; iter < 100; ++iter) {
            Real xi_old = xi;
            legendre_poly_and_derivative(n, xi, L, dL);
            xi = xi - L / dL;
            if (std::abs(xi - xi_old) < 1e-15) break;
        }

        nodes(i) = xi;
        nodes(n - 1 - i) = -xi;

        // Weights
        legendre_poly_and_derivative(n, xi, L, dL);
        Real w = 2.0 / ((1.0 - xi * xi) * dL * dL);
        weights(i) = w;
        weights(n - 1 - i) = w;
    }

    // Center node for odd n
    if (n % 2 == 1) {
        Real L, dL;
        legendre_poly_and_derivative(n, 0.0, L, dL);
        nodes(n / 2) = 0.0;
        weights(n / 2) = 2.0 / (dL * dL);
    }
}

// =============================================================================
// Gauss-Lobatto nodes and weights computation
// =============================================================================

void compute_gauss_lobatto_nodes(int n, VecX& nodes, VecX& weights) {
    nodes.resize(n);
    weights.resize(n);

    if (n < 2) {
        throw std::invalid_argument("LGL requires at least 2 nodes");
    }

    if (n == 2) {
        nodes(0) = -1.0;
        nodes(1) = 1.0;
        weights(0) = 1.0;
        weights(1) = 1.0;
        return;
    }

    // Endpoints are fixed at -1 and +1
    nodes(0) = -1.0;
    nodes(n - 1) = 1.0;
    weights(0) = 2.0 / (n * (n - 1));
    weights(n - 1) = weights(0);

    // Interior nodes are roots of P'_{n-1}(x)
    // which equals (1 - x^2) * P_{n-1}(x) derivative condition
    int N = n - 1;  // Polynomial degree for Lobatto

    for (int i = 1; i <= (n - 1) / 2; ++i) {
        // Initial guess
        Real xi = -std::cos((0.25 + i) * M_PI / N - 3.0 / (8.0 * N * M_PI * (0.25 + i)));

        // Newton iteration to find roots of q = P_{N+1} - P_{N-1}
        for (int iter = 0; iter < 100; ++iter) {
            Real xi_old = xi;

            // Compute P_{N-1}, P_N, P_{N+1} and their derivatives
            Real LNm1, dLNm1, LN, dLN;
            if (N >= 2) {
                Real Lnm2 = 1.0, Lnm1 = xi;
                Real dLnm2 = 0.0, dLnm1 = 1.0;

                for (int k = 2; k <= N; ++k) {
                    Real Lk = (2.0 * k - 1.0) / k * xi * Lnm1 - (k - 1.0) / k * Lnm2;
                    Real dLk = dLnm2 + (2.0 * k - 1.0) * Lnm1;

                    if (k == N - 1) {
                        LNm1 = Lk;
                        dLNm1 = dLk;
                    }
                    if (k == N) {
                        LN = Lk;
                        dLN = dLk;
                    }

                    Lnm2 = Lnm1;
                    Lnm1 = Lk;
                    dLnm2 = dLnm1;
                    dLnm1 = dLk;
                }

                // P_{N+1}
                int k = N + 1;
                Real LNp1 = (2.0 * k - 1.0) / k * xi * LN - (k - 1.0) / k * LNm1;
                Real dLNp1 = dLNm1 + (2.0 * k - 1.0) * LN;

                // q = P_{N+1} - P_{N-1}, dq
                Real q = LNp1 - LNm1;
                Real dq = dLNp1 - dLNm1;

                xi = xi - q / dq;
            }

            if (std::abs(xi - xi_old) < 1e-15) break;
        }

        nodes(i) = xi;
        nodes(n - 1 - i) = -xi;

        // Compute weights using P_N(xi)
        Real LN, dLN;
        legendre_poly_and_derivative(N, xi, LN, dLN);
        Real w = 2.0 / (N * (N + 1) * LN * LN);
        weights(i) = w;
        weights(n - 1 - i) = w;
    }

    // Center node for even n
    if (n % 2 == 1) {
        nodes(n / 2) = 0.0;
        Real LN, dLN;
        legendre_poly_and_derivative(N, 0.0, LN, dLN);
        weights(n / 2) = 2.0 / (N * (N + 1) * LN * LN);
    }
}

// =============================================================================
// Barycentric weights
// =============================================================================

VecX compute_barycentric_weights(const VecX& nodes) {
    int n = nodes.size();
    VecX w(n);

    for (int j = 0; j < n; ++j) {
        w(j) = 1.0;
        for (int k = 0; k < n; ++k) {
            if (k != j) {
                w(j) *= (nodes(j) - nodes(k));
            }
        }
        w(j) = 1.0 / w(j);
    }

    return w;
}

// =============================================================================
// 1D derivative matrix
// =============================================================================

MatX compute_derivative_matrix_1d(const VecX& nodes) {
    int n = nodes.size();
    VecX bw = compute_barycentric_weights(nodes);
    MatX D(n, n);

    for (int i = 0; i < n; ++i) {
        Real diag_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                D(i, j) = (bw(j) / bw(i)) / (nodes(i) - nodes(j));
                diag_sum += D(i, j);
            }
        }
        D(i, i) = -diag_sum;
    }

    return D;
}

// =============================================================================
// 1D interpolation matrix
// =============================================================================

MatX compute_interpolation_matrix_1d(const VecX& from_nodes, const VecX& to_nodes) {
    int n_from = from_nodes.size();
    int n_to = to_nodes.size();
    VecX bw = compute_barycentric_weights(from_nodes);
    MatX I(n_to, n_from);

    for (int i = 0; i < n_to; ++i) {
        Real xi = to_nodes(i);

        // Check if we're at a from_node
        bool at_node = false;
        int node_idx = -1;
        for (int j = 0; j < n_from; ++j) {
            if (std::abs(xi - from_nodes(j)) < 1e-15) {
                at_node = true;
                node_idx = j;
                break;
            }
        }

        if (at_node) {
            I.row(i).setZero();
            I(i, node_idx) = 1.0;
        } else {
            // Barycentric interpolation
            VecX temp(n_from);
            Real sum = 0.0;
            for (int j = 0; j < n_from; ++j) {
                temp(j) = bw(j) / (xi - from_nodes(j));
                sum += temp(j);
            }
            for (int j = 0; j < n_from; ++j) {
                I(i, j) = temp(j) / sum;
            }
        }
    }

    return I;
}

// =============================================================================
// LagrangeBasis1D factory methods
// =============================================================================

LagrangeBasis1D LagrangeBasis1D::create_lgl(int order) {
    LagrangeBasis1D basis;
    basis.order = order;
    compute_gauss_lobatto_nodes(order + 1, basis.nodes, basis.weights);
    basis.bary_weights = compute_barycentric_weights(basis.nodes);
    basis.D = compute_derivative_matrix_1d(basis.nodes);
    return basis;
}

LagrangeBasis1D LagrangeBasis1D::create_gl(int order) {
    LagrangeBasis1D basis;
    basis.order = order;
    compute_gauss_legendre_nodes(order + 1, basis.nodes, basis.weights);
    basis.bary_weights = compute_barycentric_weights(basis.nodes);
    basis.D = compute_derivative_matrix_1d(basis.nodes);
    return basis;
}

// =============================================================================
// HexahedronBasis implementation
// =============================================================================

HexahedronBasis::HexahedronBasis(int order, bool lgl_for_velocity, bool gl_for_tracers)
    : order_(order) {

    int np = order + 1;
    num_dofs_lgl_ = np * np * np;
    num_dofs_gl_ = np * np * np;

    // Build all operators
    build_1d_operators();
    build_3d_nodes();
    build_3d_differentiation_matrices();
    build_mass_matrices();
    build_grid_interpolation();
    build_face_interpolation();
}

void HexahedronBasis::build_1d_operators() {
    lgl_1d_ = LagrangeBasis1D::create_lgl(order_);
    gl_1d_ = LagrangeBasis1D::create_gl(order_);
}

void HexahedronBasis::build_3d_nodes() {
    int np = order_ + 1;

    lgl_nodes_3d_.resize(num_dofs_lgl_);
    gl_nodes_3d_.resize(num_dofs_gl_);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = dof_index(i, j, k, order_);
                lgl_nodes_3d_[idx] = Vec3(lgl_1d_.nodes(i),
                                          lgl_1d_.nodes(j),
                                          lgl_1d_.nodes(k));
                gl_nodes_3d_[idx] = Vec3(gl_1d_.nodes(i),
                                         gl_1d_.nodes(j),
                                         gl_1d_.nodes(k));
            }
        }
    }
}

namespace {

/// Build 3D tensor-product differentiation matrices from 1D derivative matrix
/// D_xi = D ⊗ I ⊗ I, D_eta = I ⊗ D ⊗ I, D_zeta = I ⊗ I ⊗ D
void build_3d_diff_matrices_from_1d(const MatX& D_1d, int order,
                                     MatX& D_xi, MatX& D_eta, MatX& D_zeta) {
    int np = order + 1;
    int ndof = np * np * np;

    D_xi.resize(ndof, ndof);
    D_eta.resize(ndof, ndof);
    D_zeta.resize(ndof, ndof);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int row = HexahedronBasis::dof_index(i, j, k, order);

                for (int kp = 0; kp < np; ++kp) {
                    for (int jp = 0; jp < np; ++jp) {
                        for (int ip = 0; ip < np; ++ip) {
                            int col = HexahedronBasis::dof_index(ip, jp, kp, order);

                            // D_xi: differentiate in i, identity in j and k
                            D_xi(row, col) = D_1d(i, ip) *
                                (j == jp ? 1.0 : 0.0) *
                                (k == kp ? 1.0 : 0.0);

                            // D_eta: differentiate in j, identity in i and k
                            D_eta(row, col) = (i == ip ? 1.0 : 0.0) *
                                D_1d(j, jp) *
                                (k == kp ? 1.0 : 0.0);

                            // D_zeta: differentiate in k, identity in i and j
                            D_zeta(row, col) = (i == ip ? 1.0 : 0.0) *
                                (j == jp ? 1.0 : 0.0) *
                                D_1d(k, kp);
                        }
                    }
                }
            }
        }
    }
}

/// Build 3D tensor-product interpolation matrix from 1D interpolation matrix
/// I_3d = I_1d ⊗ I_1d ⊗ I_1d
void build_3d_interp_matrix_from_1d(const MatX& I_1d, int order, MatX& I_3d) {
    int np = order + 1;
    int ndof = np * np * np;

    I_3d = MatX::Zero(ndof, ndof);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int row = HexahedronBasis::dof_index(i, j, k, order);

                for (int kp = 0; kp < np; ++kp) {
                    for (int jp = 0; jp < np; ++jp) {
                        for (int ip = 0; ip < np; ++ip) {
                            int col = HexahedronBasis::dof_index(ip, jp, kp, order);

                            I_3d(row, col) = I_1d(i, ip) * I_1d(j, jp) * I_1d(k, kp);
                        }
                    }
                }
            }
        }
    }
}

}  // anonymous namespace

void HexahedronBasis::build_3d_differentiation_matrices() {
    build_3d_diff_matrices_from_1d(lgl_1d_.D, order_, D_xi_lgl_, D_eta_lgl_, D_zeta_lgl_);
    build_3d_diff_matrices_from_1d(gl_1d_.D, order_, D_xi_gl_, D_eta_gl_, D_zeta_gl_);
}

void HexahedronBasis::build_mass_matrices() {
    int np = order_ + 1;
    int ndof = num_dofs_lgl_;

    // Mass matrices are diagonal for tensor-product basis with exact integration
    // M_ijk,ijk = w_i * w_j * w_k

    mass_lgl_ = MatX::Zero(ndof, ndof);
    mass_inv_lgl_ = MatX::Zero(ndof, ndof);
    mass_gl_ = MatX::Zero(ndof, ndof);
    mass_inv_gl_ = MatX::Zero(ndof, ndof);

    for (int k = 0; k < np; ++k) {
        for (int j = 0; j < np; ++j) {
            for (int i = 0; i < np; ++i) {
                int idx = dof_index(i, j, k, order_);

                Real w_lgl = lgl_1d_.weights(i) * lgl_1d_.weights(j) * lgl_1d_.weights(k);
                mass_lgl_(idx, idx) = w_lgl;
                mass_inv_lgl_(idx, idx) = 1.0 / w_lgl;

                Real w_gl = gl_1d_.weights(i) * gl_1d_.weights(j) * gl_1d_.weights(k);
                mass_gl_(idx, idx) = w_gl;
                mass_inv_gl_(idx, idx) = 1.0 / w_gl;
            }
        }
    }
}

void HexahedronBasis::build_grid_interpolation() {
    // Build 3D interpolation matrices between LGL and GL grids
    MatX I_lgl_to_gl_1d = compute_interpolation_matrix_1d(lgl_1d_.nodes, gl_1d_.nodes);
    MatX I_gl_to_lgl_1d = compute_interpolation_matrix_1d(gl_1d_.nodes, lgl_1d_.nodes);

    build_3d_interp_matrix_from_1d(I_lgl_to_gl_1d, order_, lgl_to_gl_);
    build_3d_interp_matrix_from_1d(I_gl_to_lgl_1d, order_, gl_to_lgl_);
}

void HexahedronBasis::build_face_interpolation() {
    int np = order_ + 1;
    int nface = np * np;  // DOFs per face

    // Face quadrature uses the same nodes as volume in tangent directions
    // Face normals:
    // Face 0: xi = -1, tangent (eta, zeta)
    // Face 1: xi = +1, tangent (eta, zeta)
    // Face 2: eta = -1, tangent (xi, zeta)
    // Face 3: eta = +1, tangent (xi, zeta)
    // Face 4: zeta = -1, tangent (xi, eta)
    // Face 5: zeta = +1, tangent (xi, eta)

    for (int face = 0; face < 6; ++face) {
        face_quad_nodes_[face].resize(nface);
        face_quad_weights_[face].resize(nface);
        interp_to_face_lgl_[face].resize(nface, num_dofs_lgl_);
        interp_to_face_gl_[face].resize(nface, num_dofs_gl_);
        interp_to_face_lgl_[face].setZero();
        interp_to_face_gl_[face].setZero();

        auto [t1_axis, t2_axis] = get_face_tangent_axes(face);
        int normal_axis = get_face_normal_axis(face);
        Real normal_val = is_positive_face(face) ? 1.0 : -1.0;

        // Build face quadrature nodes and interpolation matrices
        for (int jt = 0; jt < np; ++jt) {
            for (int it = 0; it < np; ++it) {
                int face_idx = it + np * jt;

                // Face quadrature node in tangent coordinates
                face_quad_nodes_[face][face_idx] = Vec2(lgl_1d_.nodes(it),
                                                         lgl_1d_.nodes(jt));
                face_quad_weights_[face](face_idx) = lgl_1d_.weights(it) *
                                                      lgl_1d_.weights(jt);

                // Build interpolation matrix row
                // Evaluate volume basis at face quadrature point
                Vec3 xi_face;
                xi_face(normal_axis) = normal_val;
                xi_face(t1_axis) = lgl_1d_.nodes(it);
                xi_face(t2_axis) = lgl_1d_.nodes(jt);

                // For LGL: this is just extraction (nodes are collocated at faces)
                // For GL: we need to interpolate (GL nodes are interior)
                for (int k = 0; k < np; ++k) {
                    for (int j = 0; j < np; ++j) {
                        for (int i = 0; i < np; ++i) {
                            int vol_idx = dof_index(i, j, k, order_);

                            // LGL: basis function value at face point
                            Real phi_lgl = 1.0;
                            int idx[3] = {i, j, k};
                            Real nodes_face[3];
                            nodes_face[0] = lgl_1d_.nodes(idx[0]);
                            nodes_face[1] = lgl_1d_.nodes(idx[1]);
                            nodes_face[2] = lgl_1d_.nodes(idx[2]);

                            // For LGL, nodes at face boundary match
                            // Check if this volume node projects to this face point
                            if (normal_axis == 0) {
                                // xi face: check if i corresponds to boundary
                                bool at_face = (normal_val < 0) ? (i == 0) : (i == np - 1);
                                if (at_face && j == it && k == jt) {
                                    interp_to_face_lgl_[face](face_idx, vol_idx) = 1.0;
                                }
                            } else if (normal_axis == 1) {
                                bool at_face = (normal_val < 0) ? (j == 0) : (j == np - 1);
                                if (at_face && i == it && k == jt) {
                                    interp_to_face_lgl_[face](face_idx, vol_idx) = 1.0;
                                }
                            } else {
                                bool at_face = (normal_val < 0) ? (k == 0) : (k == np - 1);
                                if (at_face && i == it && j == jt) {
                                    interp_to_face_lgl_[face](face_idx, vol_idx) = 1.0;
                                }
                            }

                            // GL: need full interpolation since GL nodes are interior
                            VecX phi_gl_1d_xi = gl_1d_.evaluate(xi_face(0));
                            VecX phi_gl_1d_eta = gl_1d_.evaluate(xi_face(1));
                            VecX phi_gl_1d_zeta = gl_1d_.evaluate(xi_face(2));

                            interp_to_face_gl_[face](face_idx, vol_idx) =
                                phi_gl_1d_xi(i) * phi_gl_1d_eta(j) * phi_gl_1d_zeta(k);
                        }
                    }
                }
            }
        }
    }
}

MatX HexahedronBasis::interp_to_subface(int face_id, int subface_idx,
                                         FaceConnectionType conn_type,
                                         bool use_lgl) const {
    int np = order_ + 1;
    int nface = np * np;

    // Determine sub-face bounds in [-1, 1] x [-1, 1] reference face
    Real t1_min = -1.0, t1_max = 1.0;
    Real t2_min = -1.0, t2_max = 1.0;

    switch (conn_type) {
        case FaceConnectionType::Fine2x1:
            // Split in first tangent direction
            if (subface_idx == 0) {
                t1_max = 0.0;
            } else {
                t1_min = 0.0;
            }
            break;

        case FaceConnectionType::Fine1x2:
            // Split in second tangent direction
            if (subface_idx == 0) {
                t2_max = 0.0;
            } else {
                t2_min = 0.0;
            }
            break;

        case FaceConnectionType::Fine2x2:
            // 2x2 split
            if (subface_idx == 0) { t1_max = 0.0; t2_max = 0.0; }
            else if (subface_idx == 1) { t1_min = 0.0; t2_max = 0.0; }
            else if (subface_idx == 2) { t1_max = 0.0; t2_min = 0.0; }
            else { t1_min = 0.0; t2_min = 0.0; }
            break;

        case FaceConnectionType::Fine3_2plus1:
            // L-shaped: 2 in dir1, 1 in dir2 for first two, then 1 full in dir2
            if (subface_idx == 0) { t1_max = 0.0; t2_max = 0.0; }
            else if (subface_idx == 1) { t1_min = 0.0; t2_max = 0.0; }
            else { t2_min = 0.0; }  // Full width in t1
            break;

        case FaceConnectionType::Fine3_1plus2:
            // Transpose of above
            if (subface_idx == 0) { t1_max = 0.0; t2_max = 0.0; }
            else if (subface_idx == 1) { t1_max = 0.0; t2_min = 0.0; }
            else { t1_min = 0.0; }  // Full width in t2
            break;

        case FaceConnectionType::SameLevel:
        case FaceConnectionType::Boundary:
        default:
            // Full face
            break;
    }

    // Map from sub-face reference [-1,1]^2 to coarse face sub-region
    auto map_to_subface = [&](Real s, Real t) {
        Real t1 = 0.5 * ((t1_max - t1_min) * s + (t1_max + t1_min));
        Real t2 = 0.5 * ((t2_max - t2_min) * t + (t2_max + t2_min));
        return std::make_pair(t1, t2);
    };

    // Build interpolation matrix from volume to sub-face quadrature
    MatX interp(nface, use_lgl ? num_dofs_lgl_ : num_dofs_gl_);
    interp.setZero();

    auto [t1_axis, t2_axis] = get_face_tangent_axes(face_id);
    int normal_axis = get_face_normal_axis(face_id);
    Real normal_val = is_positive_face(face_id) ? 1.0 : -1.0;

    const LagrangeBasis1D& basis_1d = use_lgl ? lgl_1d_ : gl_1d_;

    for (int jt = 0; jt < np; ++jt) {
        for (int it = 0; it < np; ++it) {
            int face_idx = it + np * jt;

            // Sub-face quadrature point in sub-face reference coords
            Real s = basis_1d.nodes(it);
            Real t = basis_1d.nodes(jt);

            // Map to coarse face coordinates
            auto [t1, t2] = map_to_subface(s, t);

            // Build volume evaluation point
            Vec3 xi_vol;
            xi_vol(normal_axis) = normal_val;
            xi_vol(t1_axis) = t1;
            xi_vol(t2_axis) = t2;

            // Evaluate volume basis at this point
            for (int k = 0; k < np; ++k) {
                for (int j = 0; j < np; ++j) {
                    for (int i = 0; i < np; ++i) {
                        int vol_idx = dof_index(i, j, k, order_);

                        VecX phi_1d_xi = basis_1d.evaluate(xi_vol(0));
                        VecX phi_1d_eta = basis_1d.evaluate(xi_vol(1));
                        VecX phi_1d_zeta = basis_1d.evaluate(xi_vol(2));

                        interp(face_idx, vol_idx) =
                            phi_1d_xi(i) * phi_1d_eta(j) * phi_1d_zeta(k);
                    }
                }
            }
        }
    }

    return interp;
}

}  // namespace drifter
