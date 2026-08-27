#pragma once

/// @file matrix_market_writer.hpp
/// @brief Write Eigen sparse matrices in Matrix Market coordinate format

#include "core/types.hpp"
#include <string>

namespace drifter {

/// @brief Write a sparse matrix to a Matrix Market (.mtx) file
/// @param mat Sparse matrix to write
/// @param filename Output file path
void write_matrix_market(const SpMat &mat, const std::string &filename);

} // namespace drifter
