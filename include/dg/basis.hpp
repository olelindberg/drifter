#pragma once

#include "core/types.hpp"
#include <memory>
#include <vector>

namespace drifter {

// Nodal basis functions on reference element
class Basis {
public:
    virtual ~Basis() = default;

    // Polynomial order
    virtual int order() const = 0;

    // Number of DOFs/modes
    virtual int num_modes() const = 0;

    // Evaluate basis functions at point in reference coordinates
    virtual VecX evaluate(const Vec3& xi) const = 0;

    // Evaluate gradients of basis functions
    virtual MatX evaluate_gradient(const Vec3& xi) const = 0;

    // Get mass matrix on reference element
    virtual const MatX& mass_matrix() const = 0;

    // Get stiffness matrices (derivatives)
    virtual const MatX& stiffness_x() const = 0;
    virtual const MatX& stiffness_y() const = 0;
    virtual const MatX& stiffness_z() const = 0;

    // Get lift matrix for surface integrals
    virtual const MatX& lift_matrix(int face_id) const = 0;

    // Nodal points in reference element
    virtual const std::vector<Vec3>& nodes() const = 0;

    // Vandermonde matrix (for modal <-> nodal conversion)
    virtual const MatX& vandermonde() const = 0;
    virtual const MatX& vandermonde_inverse() const = 0;

    // Factory methods
    static std::unique_ptr<Basis> create_triangle(int order);
    static std::unique_ptr<Basis> create_tetrahedron(int order);
    static std::unique_ptr<Basis> create_prism(int order);
};

// Legendre polynomials and derivatives
namespace legendre {
    Real evaluate(int n, Real x);
    Real derivative(int n, Real x);
    VecX gauss_nodes(int n);
    VecX gauss_weights(int n);
    VecX lobatto_nodes(int n);
    VecX lobatto_weights(int n);
}

// Dubiner basis for triangles (orthogonal on reference triangle)
class DubinerBasis : public Basis {
public:
    explicit DubinerBasis(int order);

    int order() const override { return order_; }
    int num_modes() const override { return num_modes_; }

    VecX evaluate(const Vec3& xi) const override;
    MatX evaluate_gradient(const Vec3& xi) const override;

    const MatX& mass_matrix() const override { return mass_; }
    const MatX& stiffness_x() const override { return stiff_x_; }
    const MatX& stiffness_y() const override { return stiff_y_; }
    const MatX& stiffness_z() const override { return stiff_z_; }
    const MatX& lift_matrix(int face_id) const override;

    const std::vector<Vec3>& nodes() const override { return nodes_; }
    const MatX& vandermonde() const override { return vander_; }
    const MatX& vandermonde_inverse() const override { return vander_inv_; }

private:
    int order_;
    int num_modes_;
    std::vector<Vec3> nodes_;
    MatX mass_, stiff_x_, stiff_y_, stiff_z_;
    MatX vander_, vander_inv_;
    std::vector<MatX> lift_;

    void build_operators();
};

} // namespace drifter
