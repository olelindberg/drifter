#include "io/matrix_market_writer.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace drifter {

void write_matrix_market(const SpMat &mat, const std::string &filename) {
  std::ofstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  f << "%%MatrixMarket matrix coordinate real general\n";
  f << mat.rows() << " " << mat.cols() << " " << mat.nonZeros() << "\n";
  f << std::setprecision(16);

  for (int k = 0; k < mat.outerSize(); ++k) {
    for (SpMat::InnerIterator it(mat, k); it; ++it) {
      f << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value()
        << "\n";
    }
  }
}

} // namespace drifter
