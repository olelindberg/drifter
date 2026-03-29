#pragma once

/// @file enum_strings.hpp
/// @brief Bidirectional enum-string conversion for configuration parsing

#include "bathymetry/adaptive_smoother_types.hpp"
#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/smoother_types.hpp"
#include <stdexcept>
#include <string>

namespace drifter {

// =============================================================================
// ErrorMetricType
// =============================================================================

inline std::string to_string(ErrorMetricType e) {
  switch (e) {
  case ErrorMetricType::NormalizedError:
    return "NormalizedError";
  case ErrorMetricType::MeanDifference:
    return "MeanDifference";
  case ErrorMetricType::VolumeChange:
    return "VolumeChange";
  }
  throw std::invalid_argument("Unknown ErrorMetricType");
}

inline ErrorMetricType error_metric_type_from_string(const std::string &s) {
  if (s == "NormalizedError")
    return ErrorMetricType::NormalizedError;
  if (s == "MeanDifference")
    return ErrorMetricType::MeanDifference;
  if (s == "VolumeChange")
    return ErrorMetricType::VolumeChange;
  throw std::invalid_argument("Unknown ErrorMetricType: '" + s + "'. Valid values: NormalizedError, MeanDifference, VolumeChange");
}

// =============================================================================
// SmootherType
// =============================================================================

inline std::string to_string(SmootherType e) {
  switch (e) {
  case SmootherType::Jacobi:
    return "Jacobi";
  case SmootherType::MultiplicativeSchwarz:
    return "MultiplicativeSchwarz";
  case SmootherType::AdditiveSchwarz:
    return "AdditiveSchwarz";
  case SmootherType::ColoredMultiplicativeSchwarz:
    return "ColoredMultiplicativeSchwarz";
  }
  throw std::invalid_argument("Unknown SmootherType");
}

inline SmootherType smoother_type_from_string(const std::string &s) {
  if (s == "Jacobi")
    return SmootherType::Jacobi;
  if (s == "MultiplicativeSchwarz")
    return SmootherType::MultiplicativeSchwarz;
  if (s == "AdditiveSchwarz")
    return SmootherType::AdditiveSchwarz;
  if (s == "ColoredMultiplicativeSchwarz")
    return SmootherType::ColoredMultiplicativeSchwarz;
  throw std::invalid_argument("Unknown SmootherType: '" + s +
                              "'. Valid values: Jacobi, MultiplicativeSchwarz, AdditiveSchwarz, "
                              "ColoredMultiplicativeSchwarz");
}

// =============================================================================
// SchurPreconditionerType
// =============================================================================

inline std::string to_string(SchurPreconditionerType e) {
  switch (e) {
  case SchurPreconditionerType::None:
    return "None";
  case SchurPreconditionerType::DiagonalApproxCG:
    return "DiagonalApproxCG";
  case SchurPreconditionerType::BlockDiagApproxCG:
    return "BlockDiagApproxCG";
  }
  throw std::invalid_argument("Unknown SchurPreconditionerType");
}

inline SchurPreconditionerType schur_preconditioner_type_from_string(const std::string &s) {
  if (s == "None")
    return SchurPreconditionerType::None;
  if (s == "DiagonalApproxCG")
    return SchurPreconditionerType::DiagonalApproxCG;
  if (s == "BlockDiagApproxCG")
    return SchurPreconditionerType::BlockDiagApproxCG;
  throw std::invalid_argument("Unknown SchurPreconditionerType: '" + s +
                              "'. Valid values: None, DiagonalApproxCG, BlockDiagApproxCG");
}

// =============================================================================
// TransferOperatorStrategy
// =============================================================================

inline std::string to_string(TransferOperatorStrategy e) {
  switch (e) {
  case TransferOperatorStrategy::L2Projection:
    return "L2Projection";
  case TransferOperatorStrategy::BezierSubdivision:
    return "BezierSubdivision";
  }
  throw std::invalid_argument("Unknown TransferOperatorStrategy");
}

inline TransferOperatorStrategy transfer_operator_strategy_from_string(const std::string &s) {
  if (s == "L2Projection")
    return TransferOperatorStrategy::L2Projection;
  if (s == "BezierSubdivision")
    return TransferOperatorStrategy::BezierSubdivision;
  throw std::invalid_argument("Unknown TransferOperatorStrategy: '" + s + "'. Valid values: L2Projection, BezierSubdivision");
}

// =============================================================================
// CoarseGridStrategy
// =============================================================================

inline std::string to_string(CoarseGridStrategy e) {
  switch (e) {
  case CoarseGridStrategy::Galerkin:
    return "Galerkin";
  case CoarseGridStrategy::CachedRediscretization:
    return "CachedRediscretization";
  }
  throw std::invalid_argument("Unknown CoarseGridStrategy");
}

inline CoarseGridStrategy coarse_grid_strategy_from_string(const std::string &s) {
  if (s == "Galerkin")
    return CoarseGridStrategy::Galerkin;
  if (s == "CachedRediscretization")
    return CoarseGridStrategy::CachedRediscretization;
  throw std::invalid_argument("Unknown CoarseGridStrategy: '" + s + "'. Valid values: Galerkin, CachedRediscretization");
}

} // namespace drifter
