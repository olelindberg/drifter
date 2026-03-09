#include "bathymetry/iterative_method_factory.hpp"
#include "bathymetry/jacobi_method.hpp"
#include "bathymetry/schwarz_method.hpp"
#include <stdexcept>

namespace drifter {

std::unique_ptr<IIterativeMethod> IterativeMethodFactory::create(
    SmootherType type, const SpMat &Q,
    const std::vector<std::vector<Index>> &element_free_dofs,
    const std::vector<Eigen::PartialPivLU<MatX>> &element_block_lu,
    const std::vector<std::vector<Index>> &elements_by_color, Real omega) {

    switch (type) {
    case SmootherType::Jacobi:
        return std::make_unique<JacobiMethod>(Q, omega);

    case SmootherType::MultiplicativeSchwarz:
        return std::make_unique<MultiplicativeSchwarzMethod>(
            Q, element_free_dofs, element_block_lu);

    case SmootherType::AdditiveSchwarz:
        return std::make_unique<AdditiveSchwarzMethod>(Q, element_free_dofs,
                                                       element_block_lu, omega);

    case SmootherType::ColoredMultiplicativeSchwarz:
        return std::make_unique<ColoredSchwarzMethod>(
            Q, element_free_dofs, element_block_lu, elements_by_color);

    default:
        throw std::invalid_argument(
            "IterativeMethodFactory: unknown SmootherType");
    }
}

} // namespace drifter
