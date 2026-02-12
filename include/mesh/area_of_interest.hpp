#pragma once

#include "core/types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace drifter {

// Forward declarations
class OctreeAdapter;
struct ElementBounds;

/// @brief Bit flags for AOI membership (supports up to 8 named regions)
/// Uses uint8_t for minimal memory: 1 byte per element
using AOIFlags = std::uint8_t;

/// @brief No region membership
constexpr AOIFlags AOI_NONE = 0;

/// @brief Area of Interest manager
///
/// Tracks which elements belong to named polygon regions. Uses 1 byte per
/// element with bit flags for up to 8 simultaneous regions.
///
/// @par Zero-overhead design:
/// - AOI membership is precomputed once per mesh configuration
/// - Hot loops (compute_rhs) never touch AOI data
/// - Output/diagnostic loops can filter using O(1) bit test
///
/// @par Usage:
/// @code
/// AreaOfInterest aoi;
/// aoi.define_region("harbor",
///     "POLYGON((400000 6200000, 420000 6200000, 420000 6220000, 400000
///     6220000, 400000 6200000))");
/// aoi.rebuild(mesh);  // Precompute membership
///
/// // In output code:
/// for (Index e = 0; e < mesh.num_elements(); ++e) {
///     if (aoi.is_in_region(e, "harbor")) {
///         // Write element e to output
///     }
/// }
/// @endcode
class AreaOfInterest {
public:
  AreaOfInterest();
  ~AreaOfInterest();

  // Move-only (PIMPL for polygon storage)
  AreaOfInterest(const AreaOfInterest &) = delete;
  AreaOfInterest &operator=(const AreaOfInterest &) = delete;
  AreaOfInterest(AreaOfInterest &&) noexcept;
  AreaOfInterest &operator=(AreaOfInterest &&) noexcept;

  // =========================================================================
  // Region definition (call before rebuild)
  // =========================================================================

  /// @brief Define a named AOI region with WKT polygon
  /// @param name Unique region name
  /// @param wkt WKT polygon string (e.g., "POLYGON((x1 y1, x2 y2, ...))")
  /// @return Bit index (0-7) for this region
  /// @throws std::runtime_error if 8 regions already defined or WKT is invalid
  int define_region(const std::string &name, const std::string &wkt);

  /// @brief Define a named AOI region with point vector
  /// @param name Unique region name
  /// @param polygon_points Polygon vertices as (x, y) pairs (closed
  /// automatically)
  /// @return Bit index (0-7) for this region
  /// @throws std::runtime_error if 8 regions already defined
  int define_region(const std::string &name,
                    const std::vector<std::pair<Real, Real>> &polygon_points);

  /// @brief Remove a named region
  /// @param name Region name to remove
  void remove_region(const std::string &name);

  /// @brief Clear all regions
  void clear_regions();

  // =========================================================================
  // Membership computation (call after mesh changes)
  // =========================================================================

  /// @brief Rebuild membership flags for all elements
  /// @param mesh The mesh to compute membership for
  /// @note Call after mesh construction or AMR
  void rebuild(const OctreeAdapter &mesh);

  /// @brief Handle element creation during refinement
  /// @param mesh The mesh (for computing child bounds)
  /// @param children Child element indices
  /// @note Recomputes membership from geometry for each child
  void on_refine(const OctreeAdapter &mesh, const std::vector<Index> &children);

  /// @brief Handle element deletion during coarsening
  /// @param children Child element indices being merged
  /// @param parent_elem New parent element index
  /// @note Parent inherits union of children's memberships
  void on_coarsen(const std::vector<Index> &children, Index parent_elem);

  /// @brief Resize internal storage for new mesh size
  /// @param num_elements New number of elements
  void resize(Index num_elements);

  // =========================================================================
  // Membership queries (O(1), thread-safe)
  // =========================================================================

  /// @brief Check if element is in a named region
  /// @param elem Element index
  /// @param region_name Region name
  /// @return true if element is in the region
  bool is_in_region(Index elem, const std::string &region_name) const;

  /// @brief Check if element is in a region by bit index
  /// @param elem Element index
  /// @param bit_index Region bit index (0-7)
  /// @return true if element is in the region
  bool is_in_region(Index elem, int bit_index) const;

  /// @brief Check if element is in any AOI region
  /// @param elem Element index
  /// @return true if element is in any region
  bool is_in_any_region(Index elem) const;

  /// @brief Get all region flags for an element
  /// @param elem Element index
  /// @return Bit flags for all regions
  AOIFlags get_flags(Index elem) const;

  /// @brief Get region bit index by name
  /// @param name Region name
  /// @return Bit index, or -1 if not found
  int get_region_index(const std::string &name) const;

  /// @brief Get list of region names for an element
  /// @param elem Element index
  /// @return Vector of region names the element belongs to
  std::vector<std::string> get_region_names(Index elem) const;

  // =========================================================================
  // Bulk queries for output/diagnostics
  // =========================================================================

  /// @brief Get elements in a named region
  /// @param region_name Region name
  /// @return Vector of element indices in the region
  std::vector<Index> elements_in_region(const std::string &region_name) const;

  /// @brief Get number of elements in a named region
  /// @param region_name Region name
  /// @return Number of elements
  Index count_in_region(const std::string &region_name) const;

  /// @brief Get membership as Real vector for VTK cell data
  /// @param region_name Region name
  /// @return Vector of 1.0 (in region) or 0.0 (not in region) per element
  std::vector<Real> as_cell_data(const std::string &region_name) const;

  /// @brief Get combined region ID for VTK cell data
  /// @return Vector of AOIFlags cast to Real per element
  std::vector<Real> combined_region_ids() const;

  // =========================================================================
  // Iterators for filtered loops
  // =========================================================================

  /// @brief Iterate over elements in a region
  /// @tparam Func Callable with signature void(Index elem)
  /// @param region_name Region name
  /// @param f Function to call for each element in region
  template <typename Func>
  void for_each_in_region(const std::string &region_name, Func &&f) const {
    int bit = get_region_index(region_name);
    if (bit < 0)
      return;
    AOIFlags mask = static_cast<AOIFlags>(1 << bit);
    for (Index e = 0; e < static_cast<Index>(flags_.size()); ++e) {
      if (flags_[e] & mask) {
        f(e);
      }
    }
  }

  // =========================================================================
  // Configuration
  // =========================================================================

  /// @brief Get number of defined regions
  /// @return Number of regions (0-8)
  int num_regions() const;

  /// @brief Get all region names
  /// @return Vector of region names
  std::vector<std::string> region_names() const;

  /// @brief Check if a region is defined
  /// @param name Region name
  /// @return true if region exists
  bool has_region(const std::string &name) const;

private:
  /// @brief Per-element membership flags (1 byte each)
  std::vector<AOIFlags> flags_;

  /// @brief Region name to bit index mapping
  std::unordered_map<std::string, int> region_bits_;

  /// @brief Next available bit index
  int next_bit_ = 0;

  /// @brief PIMPL for boost::geometry polygon operations
  struct Impl;
  std::unique_ptr<Impl> impl_;

  /// @brief Compute membership for a single element
  AOIFlags compute_element_flags(const ElementBounds &bounds) const;
};

} // namespace drifter
