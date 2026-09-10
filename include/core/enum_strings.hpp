#pragma once

/// @file enum_strings.hpp
/// @brief Bidirectional enum-string conversion for configuration parsing

#include "bathymetry/adaptive_smoother_types.hpp"
#include "bathymetry/bezier_multigrid_preconditioner.hpp"
#include "bathymetry/cg_cubic_bezier_bathymetry_smoother.hpp"
#include "bathymetry/cg_hermite_bathymetry_smoother.hpp"
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
  case ErrorMetricType::PixelRMSE:
    return "PixelRMSE";
  case ErrorMetricType::PixelMaxError:
    return "PixelMaxError";
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
  if (s == "PixelRMSE")
    return ErrorMetricType::PixelRMSE;
  if (s == "PixelMaxError")
    return ErrorMetricType::PixelMaxError;
  throw std::invalid_argument("Unknown ErrorMetricType: '" + s + "'. Valid values: NormalizedError, MeanDifference, VolumeChange, PixelRMSE, PixelMaxError");
}

// =============================================================================
// BathySmootherKind
// =============================================================================

/// @brief Which bathymetry smoother family the application should use
///
/// CubicBezier is the historical default: Bernstein control values with C1
/// imposed by collocation, solved as an indefinite KKT system. The Hermite kinds
/// use corner value/derivative DOFs, which makes continuity structural and the
/// system SPD. See docs/hermite_bathymetry_system.md.
enum class BathySmootherKind {
  CubicBezier, ///< Cubic Bezier, approximate C1 via KKT (default)
  HermiteC0,   ///< Bilinear Hermite, exact C0, SPD
  HermiteC1    ///< Bicubic Bogner-Fox-Schmit Hermite, exact C1, SPD
};

inline std::string to_string(BathySmootherKind e) {
  switch (e) {
  case BathySmootherKind::CubicBezier:
    return "CubicBezier";
  case BathySmootherKind::HermiteC0:
    return "HermiteC0";
  case BathySmootherKind::HermiteC1:
    return "HermiteC1";
  }
  throw std::invalid_argument("Unknown BathySmootherKind");
}

inline BathySmootherKind bathy_smoother_kind_from_string(const std::string &s) {
  if (s == "CubicBezier")
    return BathySmootherKind::CubicBezier;
  if (s == "HermiteC0")
    return BathySmootherKind::HermiteC0;
  if (s == "HermiteC1")
    return BathySmootherKind::HermiteC1;
  throw std::invalid_argument("Unknown BathySmootherKind: '" + s + "'. Valid values: CubicBezier, HermiteC0, HermiteC1");
}

// =============================================================================
// HermiteSolverKind
// =============================================================================

/// @brief Which direct factorisation the Hermite smoother uses
///
/// The enum itself is declared with the config it belongs to, in
/// bathymetry/cg_hermite_bathymetry_smoother.hpp. Availability is a build
/// decision, so a name that parses here can still be rejected at solve() time
/// if its backend was not compiled in.
inline std::string to_string(HermiteSolverKind e) {
  switch (e) {
  case HermiteSolverKind::SimplicialLDLT:
    return "SimplicialLDLT";
  case HermiteSolverKind::SimplicialLDLTMetis:
    return "SimplicialLDLTMetis";
  case HermiteSolverKind::SimplicialLLT:
    return "SimplicialLLT";
  case HermiteSolverKind::PardisoLDLT:
    return "PardisoLDLT";
  case HermiteSolverKind::PardisoLLT:
    return "PardisoLLT";
  case HermiteSolverKind::UmfPackLU:
    return "UmfPackLU";
  case HermiteSolverKind::CholmodSimplicialLDLT:
    return "CholmodSimplicialLDLT";
  case HermiteSolverKind::CholmodSupernodalLLT:
    return "CholmodSupernodalLLT";
  case HermiteSolverKind::CholmodSupernodalNesdis:
    return "CholmodSupernodalNesdis";
  }
  throw std::invalid_argument("Unknown HermiteSolverKind");
}

inline HermiteSolverKind hermite_solver_kind_from_string(const std::string &s) {
  if (s == "SimplicialLDLT")
    return HermiteSolverKind::SimplicialLDLT;
  if (s == "SimplicialLDLTMetis")
    return HermiteSolverKind::SimplicialLDLTMetis;
  if (s == "SimplicialLLT")
    return HermiteSolverKind::SimplicialLLT;
  if (s == "PardisoLDLT")
    return HermiteSolverKind::PardisoLDLT;
  if (s == "PardisoLLT")
    return HermiteSolverKind::PardisoLLT;
  if (s == "UmfPackLU")
    return HermiteSolverKind::UmfPackLU;
  if (s == "CholmodSimplicialLDLT")
    return HermiteSolverKind::CholmodSimplicialLDLT;
  if (s == "CholmodSupernodalLLT")
    return HermiteSolverKind::CholmodSupernodalLLT;
  if (s == "CholmodSupernodalNesdis")
    return HermiteSolverKind::CholmodSupernodalNesdis;
  throw std::invalid_argument("Unknown HermiteSolverKind: '" + s + "'. Valid values: SimplicialLDLT, SimplicialLDLTMetis, SimplicialLLT, PardisoLDLT, PardisoLLT, UmfPackLU, CholmodSimplicialLDLT, CholmodSupernodalLLT, CholmodSupernodalNesdis");
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
