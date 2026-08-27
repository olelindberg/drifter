/// @file config_reader.cpp
/// @brief JSON configuration file reader implementation

#include "core/config_reader.hpp"
#include "core/enum_strings.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace pt = boost::property_tree;

namespace drifter {

namespace {

/// @brief Parse MultigridConfig from property tree
MultigridConfig parse_multigrid_config(const pt::ptree &tree) {
  MultigridConfig config;

  config.pre_smoothing    = tree.get<int>("pre_smoothing", config.pre_smoothing);
  config.post_smoothing   = tree.get<int>("post_smoothing", config.post_smoothing);
  config.verbose          = tree.get<bool>("verbose", config.verbose);
  config.max_vcycles      = tree.get<int>("max_vcycles", config.max_vcycles);
  config.vcycle_tolerance = tree.get<Real>("vcycle_tolerance", config.vcycle_tolerance);

  if (auto str = tree.get_optional<std::string>("smoother_type")) {
    config.smoother_type = smoother_type_from_string(*str);
  }
  if (auto str = tree.get_optional<std::string>("transfer_strategy")) {
    config.transfer_strategy = transfer_operator_strategy_from_string(*str);
  }
  if (auto str = tree.get_optional<std::string>("coarse_grid_strategy")) {
    config.coarse_grid_strategy = coarse_grid_strategy_from_string(*str);
  }

  return config;
}

/// @brief Parse CGCubicBezierSmootherConfig from property tree
CGCubicBezierSmootherConfig parse_smoother_config(const pt::ptree &tree) {
  CGCubicBezierSmootherConfig config;

  config.lambda                  = tree.get<Real>("lambda", config.lambda);
  config.ngauss_data             = tree.get<int>("ngauss_data", config.ngauss_data);
  config.ngauss_energy           = tree.get<int>("ngauss_energy", config.ngauss_energy);
  config.ridge_epsilon           = tree.get<Real>("ridge_epsilon", config.ridge_epsilon);
  config.max_bound_iterations    = tree.get<int>("max_bound_iterations", config.max_bound_iterations);
  config.bound_tolerance         = tree.get<Real>("bound_tolerance", config.bound_tolerance);
  config.edge_ngauss             = tree.get<int>("edge_ngauss", config.edge_ngauss);
  config.enable_natural_bc       = tree.get<bool>("enable_natural_bc", config.enable_natural_bc);
  config.enable_zero_gradient_bc = tree.get<bool>("enable_zero_gradient_bc", config.enable_zero_gradient_bc);
  config.use_condensation        = tree.get<bool>("use_condensation", config.use_condensation);
  config.use_iterative_solver    = tree.get<bool>("use_iterative_solver", config.use_iterative_solver);
  config.tolerance               = tree.get<Real>("tolerance", config.tolerance);
  config.max_iterations          = tree.get<int>("max_iterations", config.max_iterations);
  config.inner_tolerance         = tree.get<Real>("inner_tolerance", config.inner_tolerance);
  config.inner_max_iterations    = tree.get<int>("inner_max_iterations", config.inner_max_iterations);
  config.icc_shift               = tree.get<Real>("icc_shift", config.icc_shift);
  config.use_multigrid           = tree.get<bool>("use_multigrid", config.use_multigrid);
  config.use_hierarchical_ordering = tree.get<bool>("use_hierarchical_ordering", config.use_hierarchical_ordering);
  config.use_static_condensation = tree.get<bool>("use_static_condensation", config.use_static_condensation);
  config.verbose                 = tree.get<bool>("verbose", config.verbose);

  if (auto str = tree.get_optional<std::string>("schur_preconditioner")) {
    config.schur_preconditioner = schur_preconditioner_type_from_string(*str);
  }

  // Optional bounds
  if (auto val = tree.get_optional<Real>("lower_bound")) {
    config.lower_bound = *val;
  }
  if (auto val = tree.get_optional<Real>("upper_bound")) {
    config.upper_bound = *val;
  }

  // Nested multigrid config
  if (auto mg_tree = tree.get_child_optional("multigrid")) {
    config.multigrid_config = parse_multigrid_config(*mg_tree);
  }

  return config;
}

/// @brief Parse AdaptiveCGCubicBezierConfig from property tree
AdaptiveCGCubicBezierConfig parse_adaptive_config(const pt::ptree &tree, const pt::ptree &smoother_tree) {
  AdaptiveCGCubicBezierConfig config;

  config.error_threshold      = tree.get<Real>("error_threshold", config.error_threshold);
  config.max_iterations       = tree.get<int>("max_iterations", config.max_iterations);
  config.max_elements         = tree.get<int>("max_elements", config.max_elements);
  config.max_refinement_level = tree.get<int>("max_refinement_level", config.max_refinement_level);
  config.dorfler_theta        = tree.get<Real>("dorfler_theta", config.dorfler_theta);
  config.symmetry_tolerance   = tree.get<Real>("symmetry_tolerance", config.symmetry_tolerance);
  config.ngauss_error         = tree.get<int>("ngauss_error", config.ngauss_error);
  config.verbose              = tree.get<bool>("verbose", config.verbose);
  config.error_output_dir     = tree.get<std::string>("error_output_dir", config.error_output_dir);
  config.vtk_output_prefix    = tree.get<std::string>("vtk_output_prefix", config.vtk_output_prefix);

  if (auto str = tree.get_optional<std::string>("error_metric_type")) {
    config.error_metric_type = error_metric_type_from_string(*str);
  }

  // Parse smoother config from separate section
  config.smoother_config = parse_smoother_config(smoother_tree);

  return config;
}

/// @brief Parse CGHermiteSmootherConfig from property tree
CGHermiteSmootherConfig parse_hermite_smoother_config(const pt::ptree &tree,
                                                     BathySmootherKind kind) {
  CGHermiteSmootherConfig config;

  // The continuity order comes from the smoother kind, not a separate key
  config.continuity_order = (kind == BathySmootherKind::HermiteC0) ? 0 : 1;

  config.lambda                  = tree.get<Real>("lambda", config.lambda);
  config.ridge_epsilon           = tree.get<Real>("ridge_epsilon", config.ridge_epsilon);
  config.enable_zero_gradient_bc = tree.get<bool>("enable_zero_gradient_bc", config.enable_zero_gradient_bc);
  config.use_equilibration       = tree.get<bool>("use_equilibration", config.use_equilibration);
  config.verbose                 = tree.get<bool>("verbose", config.verbose);

  // The data-fitting integrand has degree 2p per direction, so C0 needs 2 points
  // and C1 needs 4. Default to the requirement rather than the struct default.
  const int default_ngauss = (config.continuity_order == 0) ? 2 : 4;
  config.ngauss_data = tree.get<int>("ngauss_data", default_ngauss);

  return config;
}

/// @brief Parse AdaptiveCGHermiteConfig from property tree
///
/// Shares the "adaptive" JSON section with the cubic Bezier path; only the
/// nested smoother config differs.
AdaptiveCGHermiteConfig parse_hermite_adaptive_config(const pt::ptree &tree,
                                                     const pt::ptree &smoother_tree,
                                                     BathySmootherKind kind) {
  AdaptiveCGHermiteConfig config;

  config.error_threshold      = tree.get<Real>("error_threshold", config.error_threshold);
  config.max_iterations       = tree.get<int>("max_iterations", config.max_iterations);
  config.max_elements         = tree.get<int>("max_elements", config.max_elements);
  config.max_refinement_level = tree.get<int>("max_refinement_level", config.max_refinement_level);
  config.dorfler_theta        = tree.get<Real>("dorfler_theta", config.dorfler_theta);
  config.symmetry_tolerance   = tree.get<Real>("symmetry_tolerance", config.symmetry_tolerance);
  config.ngauss_error         = tree.get<int>("ngauss_error", config.ngauss_error);
  config.verbose              = tree.get<bool>("verbose", config.verbose);
  config.error_output_dir     = tree.get<std::string>("error_output_dir", config.error_output_dir);
  config.vtk_output_prefix    = tree.get<std::string>("vtk_output_prefix", config.vtk_output_prefix);

  if (auto str = tree.get_optional<std::string>("error_metric_type")) {
    config.error_metric_type = error_metric_type_from_string(*str);
  }

  config.smoother_config = parse_hermite_smoother_config(smoother_tree, kind);

  return config;
}

/// @brief Serialize MultigridConfig to property tree
pt::ptree serialize_multigrid_config(const MultigridConfig &config) {
  pt::ptree tree;

  tree.put("pre_smoothing", config.pre_smoothing);
  tree.put("post_smoothing", config.post_smoothing);
  tree.put("smoother_type", to_string(config.smoother_type));
  tree.put("verbose", config.verbose);
  tree.put("coarse_grid_strategy", to_string(config.coarse_grid_strategy));
  tree.put("transfer_strategy", to_string(config.transfer_strategy));
  tree.put("max_vcycles", config.max_vcycles);
  tree.put("vcycle_tolerance", config.vcycle_tolerance);

  return tree;
}

/// @brief Serialize CGCubicBezierSmootherConfig to property tree
pt::ptree serialize_smoother_config(const CGCubicBezierSmootherConfig &config) {
  pt::ptree tree;

  tree.put("lambda", config.lambda);
  tree.put("ngauss_data", config.ngauss_data);
  tree.put("ngauss_energy", config.ngauss_energy);
  tree.put("ridge_epsilon", config.ridge_epsilon);
  tree.put("max_bound_iterations", config.max_bound_iterations);
  tree.put("bound_tolerance", config.bound_tolerance);
  tree.put("edge_ngauss", config.edge_ngauss);
  tree.put("enable_natural_bc", config.enable_natural_bc);
  tree.put("enable_zero_gradient_bc", config.enable_zero_gradient_bc);
  tree.put("use_condensation", config.use_condensation);
  tree.put("use_iterative_solver", config.use_iterative_solver);
  tree.put("tolerance", config.tolerance);
  tree.put("max_iterations", config.max_iterations);
  tree.put("inner_tolerance", config.inner_tolerance);
  tree.put("inner_max_iterations", config.inner_max_iterations);
  tree.put("icc_shift", config.icc_shift);
  tree.put("use_multigrid", config.use_multigrid);
  tree.put("use_hierarchical_ordering", config.use_hierarchical_ordering);
  tree.put("use_static_condensation", config.use_static_condensation);
  tree.put("schur_preconditioner", to_string(config.schur_preconditioner));
  tree.put("verbose", config.verbose);

  if (config.lower_bound) {
    tree.put("lower_bound", *config.lower_bound);
  }
  if (config.upper_bound) {
    tree.put("upper_bound", *config.upper_bound);
  }

  tree.add_child("multigrid", serialize_multigrid_config(config.multigrid_config));

  return tree;
}

/// @brief Serialize CGHermiteSmootherConfig to property tree
pt::ptree serialize_hermite_smoother_config(const CGHermiteSmootherConfig &config,
                                            BathySmootherKind kind) {
  pt::ptree tree;

  // continuity_order is implied by the kind, so it is not written separately
  tree.put("type", to_string(kind));
  tree.put("lambda", config.lambda);
  tree.put("ngauss_data", config.ngauss_data);
  tree.put("ridge_epsilon", config.ridge_epsilon);
  tree.put("enable_zero_gradient_bc", config.enable_zero_gradient_bc);
  tree.put("use_equilibration", config.use_equilibration);
  tree.put("verbose", config.verbose);

  return tree;
}

/// @brief Serialize AdaptiveCGCubicBezierConfig to property tree
pt::ptree serialize_adaptive_config(const AdaptiveCGCubicBezierConfig &config) {
  pt::ptree tree;

  tree.put("error_threshold", config.error_threshold);
  tree.put("max_iterations", config.max_iterations);
  tree.put("max_elements", config.max_elements);
  tree.put("max_refinement_level", config.max_refinement_level);
  tree.put("error_metric_type", to_string(config.error_metric_type));
  tree.put("dorfler_theta", config.dorfler_theta);
  tree.put("symmetry_tolerance", config.symmetry_tolerance);
  tree.put("ngauss_error", config.ngauss_error);
  tree.put("verbose", config.verbose);
  tree.put("error_output_dir", config.error_output_dir);
  tree.put("vtk_output_prefix", config.vtk_output_prefix);

  return tree;
}

} // anonymous namespace

DrifterConfig ConfigReader::load(const std::string &filepath) {
  pt::ptree root;

  try {
    pt::read_json(filepath, root);
  } catch (const pt::json_parser_error &e) {
    throw std::runtime_error("Failed to parse config file '" + filepath + "': " + e.what());
  }

  DrifterConfig config;

  // =========================================================================
  // Data paths (required section)
  // =========================================================================
  auto data_tree = root.get_child_optional("data");
  if (!data_tree) {
    throw std::runtime_error("Missing required section 'data' in config file");
  }

  config.data_dir = data_tree->get<std::string>("data_dir", "");
  if (config.data_dir.empty()) {
    throw std::runtime_error("Missing required field 'data.data_dir'");
  }

  config.primary_file = data_tree->get<std::string>("primary_file", "");
  if (config.primary_file.empty()) {
    throw std::runtime_error("Missing required field 'data.primary_file'");
  }

  // Parse tile_files array
  if (auto tiles = data_tree->get_child_optional("tile_files")) {
    for (const auto &item : *tiles) {
      config.tile_files.push_back(item.second.get_value<std::string>());
    }
  }

  // =========================================================================
  // Domain definition
  // =========================================================================
  if (auto domain_tree = root.get_child_optional("domain")) {
    config.center_x    = domain_tree->get<Real>("center_x", config.center_x);
    config.center_y    = domain_tree->get<Real>("center_y", config.center_y);
    config.domain_size = domain_tree->get<Real>("domain_size", config.domain_size);
  }

  // =========================================================================
  // Initial grid
  // =========================================================================
  if (auto grid_tree = root.get_child_optional("initial_grid")) {
    config.nx = grid_tree->get<int>("nx", config.nx);
    config.ny = grid_tree->get<int>("ny", config.ny);
  }

  // =========================================================================
  // Adaptive and smoother configuration
  // =========================================================================
  pt::ptree adaptive_tree;
  pt::ptree smoother_tree;

  if (auto tree = root.get_child_optional("adaptive")) {
    adaptive_tree = *tree;
  }
  if (auto tree = root.get_child_optional("smoother")) {
    smoother_tree = *tree;
  }

  // Which smoother family to run; defaults to the historical cubic Bezier path
  if (auto str = smoother_tree.get_optional<std::string>("type")) {
    config.smoother_kind = bathy_smoother_kind_from_string(*str);
  }

  // Both configs are parsed regardless of the selected kind: they are cheap
  // value structs, and populating both keeps save() lossless.
  config.adaptive = parse_adaptive_config(adaptive_tree, smoother_tree);
  config.hermite_adaptive =
      parse_hermite_adaptive_config(adaptive_tree, smoother_tree, config.smoother_kind);

  // =========================================================================
  // Auto-compute min_tree_level from initial grid
  // =========================================================================
  // Multigrid is specific to the cubic Bezier path; the Hermite path solves
  // directly and has no multigrid config to seed.
  int initial_level = static_cast<int>(std::log2(std::max(config.nx, config.ny)));
  config.adaptive.smoother_config.multigrid_config.min_tree_level = initial_level;

  // =========================================================================
  // Output configuration
  // =========================================================================
  if (auto output_tree = root.get_child_optional("output")) {
    config.output_file     = output_tree->get<std::string>("output_file", config.output_file);
    config.vtk_subdivision = output_tree->get<int>("vtk_subdivision", config.vtk_subdivision);
  }

  return config;
}

void ConfigReader::save(const DrifterConfig &config, const std::string &filepath) {
  pt::ptree root;

  // Data section
  pt::ptree data_tree;
  data_tree.put("data_dir", config.data_dir);
  data_tree.put("primary_file", config.primary_file);
  pt::ptree tiles_array;
  for (const auto &tile : config.tile_files) {
    pt::ptree tile_node;
    tile_node.put("", tile);
    tiles_array.push_back(std::make_pair("", tile_node));
  }
  data_tree.add_child("tile_files", tiles_array);
  root.add_child("data", data_tree);

  // Domain section
  pt::ptree domain_tree;
  domain_tree.put("center_x", config.center_x);
  domain_tree.put("center_y", config.center_y);
  domain_tree.put("domain_size", config.domain_size);
  root.add_child("domain", domain_tree);

  // Initial grid section
  pt::ptree grid_tree;
  grid_tree.put("nx", config.nx);
  grid_tree.put("ny", config.ny);
  root.add_child("initial_grid", grid_tree);

  // Adaptive section
  root.add_child("adaptive", serialize_adaptive_config(config.adaptive));

  // Smoother section
  // Write the smoother block for whichever family is selected, so a saved config
  // round-trips back to the same run.
  if (config.smoother_kind == BathySmootherKind::CubicBezier) {
    pt::ptree smoother_tree = serialize_smoother_config(config.adaptive.smoother_config);
    smoother_tree.put("type", to_string(config.smoother_kind));
    root.add_child("smoother", smoother_tree);
  } else {
    root.add_child("smoother",
                   serialize_hermite_smoother_config(config.hermite_adaptive.smoother_config,
                                                     config.smoother_kind));
  }

  // Output section
  pt::ptree output_tree;
  output_tree.put("output_file", config.output_file);
  output_tree.put("vtk_subdivision", config.vtk_subdivision);
  root.add_child("output", output_tree);

  // Write to file
  try {
    pt::write_json(filepath, root);
  } catch (const pt::json_parser_error &e) {
    throw std::runtime_error("Failed to write config file '" + filepath + "': " + e.what());
  }
}

namespace {

/// @brief Print the adaptive-refinement section
///
/// The Bezier and Hermite adaptive configs are unrelated types with the same
/// field names, so one template serves both.
template <typename AdaptiveConfig>
void print_adaptive_section(int w, const AdaptiveConfig &a) {
  std::cout << "\nAdaptive Refinement:\n";
  std::cout << "  " << std::left << std::setw(w) << "error_threshold" << ": " << a.error_threshold << "\n";
  std::cout << "  " << std::left << std::setw(w) << "error_metric_type" << ": " << to_string(a.error_metric_type) << "\n";
  std::cout << "  " << std::left << std::setw(w) << "max_iterations" << ": " << a.max_iterations << "\n";
  std::cout << "  " << std::left << std::setw(w) << "max_elements" << ": " << a.max_elements << "\n";
  std::cout << "  " << std::left << std::setw(w) << "max_refinement_level" << ": " << a.max_refinement_level << "\n";
  std::cout << "  " << std::left << std::setw(w) << "dorfler_theta" << ": " << a.dorfler_theta << "\n";
  std::cout << "  " << std::left << std::setw(w) << "ngauss_error" << ": " << a.ngauss_error << "\n";
  std::cout << "  " << std::left << std::setw(w) << "verbose" << ": " << (a.verbose ? "true" : "false") << "\n";
}

/// @brief Print the Hermite smoother section
///
/// The Hermite path solves the condensed SPD system directly, so it has neither
/// an iterative-solver nor a multigrid section to report.
void print_hermite_smoother_section(int w, const CGHermiteSmootherConfig &sc) {
  std::cout << "\nSmoother (Hermite):\n";
  std::cout << "  " << std::left << std::setw(w) << "continuity_order" << ": " << sc.continuity_order << "\n";
  std::cout << "  " << std::left << std::setw(w) << "lambda" << ": " << sc.lambda << "\n";
  std::cout << "  " << std::left << std::setw(w) << "ngauss_data" << ": " << sc.ngauss_data << "\n";
  std::cout << "  " << std::left << std::setw(w) << "ridge_epsilon" << ": " << sc.ridge_epsilon << "\n";
  std::cout << "  " << std::left << std::setw(w) << "enable_zero_gradient_bc" << ": " << (sc.enable_zero_gradient_bc ? "true" : "false") << "\n";
  std::cout << "  " << std::left << std::setw(w) << "use_equilibration" << ": " << (sc.use_equilibration ? "true" : "false") << "\n";
  std::cout << "  " << std::left << std::setw(w) << "solver" << ": SimplicialLDLT (direct, SPD)\n";
}

} // namespace

void print_config(const DrifterConfig &config) {
  const int w = 26;

  std::cout << "\n=== Configuration ===\n";
  std::cout << "\nSmoother family:\n";
  std::cout << "  " << std::left << std::setw(w) << "type" << ": " << to_string(config.smoother_kind) << "\n";

  std::cout << "\nData:\n";
  std::cout << "  " << std::left << std::setw(w) << "data_dir" << ": " << config.data_dir << "\n";
  std::cout << "  " << std::left << std::setw(w) << "primary_file" << ": " << config.primary_file << "\n";
  std::cout << "  " << std::left << std::setw(w) << "tile_files" << ": " << config.tile_files.size() << " files\n";

  std::cout << "\nDomain:\n";
  std::cout << "  " << std::left << std::setw(w) << "center_x" << ": " << config.center_x << "\n";
  std::cout << "  " << std::left << std::setw(w) << "center_y" << ": " << config.center_y << "\n";
  std::cout << "  " << std::left << std::setw(w) << "domain_size" << ": " << config.domain_size << "\n";
  std::cout << "  " << std::left << std::setw(w) << "nx" << ": " << config.nx << "\n";
  std::cout << "  " << std::left << std::setw(w) << "ny" << ": " << config.ny << "\n";

  // Only the family that will actually run is reported. Both configs are parsed
  // so that save() is lossless, but printing the unused one is misleading.
  if (config.smoother_kind != BathySmootherKind::CubicBezier) {
    print_adaptive_section(w, config.hermite_adaptive);
    print_hermite_smoother_section(w, config.hermite_adaptive.smoother_config);

    std::cout << "\nOutput:\n";
    std::cout << "  " << std::left << std::setw(w) << "output_file" << ": " << config.output_file << "\n";
    std::cout << "  " << std::left << std::setw(w) << "vtk_subdivision" << ": " << config.vtk_subdivision << "\n";
    std::cout << "\n";
    return;
  }

  print_adaptive_section(w, config.adaptive);

  const auto &sc = config.adaptive.smoother_config;
  const auto &mg = sc.multigrid_config;

  std::cout << "\nSmoother:\n";
  std::cout << "  " << std::left << std::setw(w) << "lambda" << ": " << sc.lambda << "\n";
  std::cout << "  " << std::left << std::setw(w) << "ngauss_data" << ": " << sc.ngauss_data << "\n";
  std::cout << "  " << std::left << std::setw(w) << "ngauss_energy" << ": " << sc.ngauss_energy << "\n";
  std::cout << "  " << std::left << std::setw(w) << "ridge_epsilon" << ": " << sc.ridge_epsilon << "\n";
  std::cout << "  " << std::left << std::setw(w) << "edge_ngauss" << ": " << sc.edge_ngauss << "\n";
  std::cout << "  " << std::left << std::setw(w) << "enable_natural_bc" << ": " << (sc.enable_natural_bc ? "true" : "false") << "\n";
  std::cout << "  " << std::left << std::setw(w) << "enable_zero_gradient_bc" << ": " << (sc.enable_zero_gradient_bc ? "true" : "false") << "\n";
  std::cout << "  " << std::left << std::setw(w) << "use_condensation" << ": " << (sc.use_condensation ? "true" : "false") << "\n";

  std::cout << "\nIterative Solver:\n";
  std::cout << "  " << std::left << std::setw(w) << "use_iterative_solver" << ": " << (sc.use_iterative_solver ? "true" : "false") << "\n";
  std::cout << "  " << std::left << std::setw(w) << "tolerance" << ": " << sc.tolerance << "\n";
  std::cout << "  " << std::left << std::setw(w) << "max_iterations" << ": " << sc.max_iterations << "\n";
  std::cout << "  " << std::left << std::setw(w) << "inner_tolerance" << ": " << sc.inner_tolerance << "\n";
  std::cout << "  " << std::left << std::setw(w) << "inner_max_iterations" << ": " << sc.inner_max_iterations << "\n";
  std::cout << "  " << std::left << std::setw(w) << "icc_shift" << ": " << sc.icc_shift << "\n";
  std::cout << "  " << std::left << std::setw(w) << "schur_preconditioner" << ": " << to_string(sc.schur_preconditioner) << "\n";

  std::cout << "\nMultigrid:\n";
  std::cout << "  " << std::left << std::setw(w) << "use_multigrid" << ": " << (sc.use_multigrid ? "true" : "false") << "\n";
  std::cout << "  " << std::left << std::setw(w) << "min_tree_level" << ": " << mg.min_tree_level << "\n";
  std::cout << "  " << std::left << std::setw(w) << "pre_smoothing" << ": " << mg.pre_smoothing << "\n";
  std::cout << "  " << std::left << std::setw(w) << "post_smoothing" << ": " << mg.post_smoothing << "\n";
  std::cout << "  " << std::left << std::setw(w) << "smoother_type" << ": " << to_string(mg.smoother_type) << "\n";
  std::cout << "  " << std::left << std::setw(w) << "transfer_strategy" << ": " << to_string(mg.transfer_strategy) << "\n";
  std::cout << "  " << std::left << std::setw(w) << "coarse_grid_strategy" << ": " << to_string(mg.coarse_grid_strategy) << "\n";
  std::cout << "  " << std::left << std::setw(w) << "max_vcycles" << ": " << mg.max_vcycles << "\n";
  std::cout << "  " << std::left << std::setw(w) << "vcycle_tolerance" << ": " << mg.vcycle_tolerance << "\n";

  std::cout << "\nOutput:\n";
  std::cout << "  " << std::left << std::setw(w) << "output_file" << ": " << config.output_file << "\n";
  std::cout << "  " << std::left << std::setw(w) << "vtk_subdivision" << ": " << config.vtk_subdivision << "\n";
  std::cout << "\n";
}

} // namespace drifter
