#pragma once

/// @file element_data_mask.hpp
/// @brief Per-element water / beach / inland classification for the CG smoothers
///
/// The point-wise masks on CGSmootherBase answer "is there a measurement at this
/// exact coordinate". That is the right granularity for dropping a single sample
/// from the least-squares sum, but not for deciding what to do with a *region*
/// carrying no water data. This class answers the element-level question instead:
///
///   - **Water**  - at least one sample in the element is a real depth reading.
///                  Assembled and solved as usual.
///   - **Beach**  - no water data anywhere in the element, but it shares a node
///                  with a Water element. Its nodes are pinned to depth 0, which
///                  is what gives the water region its Dirichlet boundary.
///   - **Inland** - no water data, and no Water element shares any of its nodes.
///                  Dropped from assembly entirely; every DOF it owns ends up
///                  pinned, so it leaves the solved system.
///
/// "Not water" covers both NoData and land: the interior of a landmass is as
/// pointless to solve for as the interior of a survey hole.
///
/// Adjacency is by **shared node**, i.e. edge *and* corner neighbours. That is
/// what makes dropping Inland elements safe: every node of an Inland element is
/// touched only by non-water elements, so it is interior to the pinned region and
/// all of its DOFs are pinned. Skipping its assembly can never strand a free DOF
/// with no operator support. Edge-only adjacency would break that - an element
/// touching water diagonally would be classified Inland while still sharing a
/// node whose derivative DOFs the water element leaves free.

#include "bathymetry/quadtree_adapter.hpp"
#include "core/types.hpp"
#include <cstdint>
#include <functional>
#include <vector>

namespace drifter {

/// @brief What an element is, for the purposes of the smoothing solve
enum class ElementDataClass : uint8_t {
    Water = 0,  ///< Carries at least one real depth reading
    Beach = 1,  ///< No water data, but shares a node with a Water element
    Inland = 2  ///< No water data, and no Water element shares any of its nodes
};

/// @brief Element classification over a quadtree, from the point-wise data masks
///
/// Cheap to build and cheap to throw away: it is rebuilt from scratch on every
/// re-fit, which is what keeps it correct across adaptive refinement.
class ElementDataMask {
public:
    /// @brief Classify every element of a mesh
    ///
    /// @param mesh The quadtree being fitted
    /// @param has_data Where a measurement exists; empty means "everywhere"
    /// @param is_land Where the surface is known to be at depth 0; empty means
    ///        "nowhere". Both empty classifies the whole mesh as Water, which is
    ///        the analytic-function path and must stay unchanged.
    /// @param nsamples Interior sample grid per direction, in addition to the
    ///        four corners and the centre. Larger is more conservative about
    ///        calling an element non-water.
    ElementDataMask(const QuadtreeAdapter &mesh,
                    const std::function<bool(Real, Real)> &has_data,
                    const std::function<bool(Real, Real)> &is_land, int nsamples = 3);

    /// @brief Classification of an element
    ElementDataClass operator[](Index elem) const {
        return classes_[static_cast<size_t>(elem)];
    }

    /// @brief Number of classified elements
    Index size() const { return static_cast<Index>(classes_.size()); }

    /// @brief Is this element solved for against data?
    bool is_water(Index elem) const { return (*this)[elem] == ElementDataClass::Water; }

    /// @brief Is this element held at depth 0? (Beach or Inland)
    bool is_pinned(Index elem) const { return !is_water(elem); }

    /// @brief Is this element dropped from assembly? (Inland only)
    bool is_excluded(Index elem) const { return (*this)[elem] == ElementDataClass::Inland; }

    Index num_water() const { return num_water_; }
    Index num_beach() const { return num_beach_; }
    Index num_inland() const { return num_inland_; }

    /// @brief Whether any element is non-water
    ///
    /// False for an analytic source, which is the signal that nothing changes.
    bool has_pinned_elements() const { return num_beach_ > 0 || num_inland_ > 0; }

private:
    /// @brief Whether any sample in an element is a real depth reading
    static bool sample_is_water(const QuadBounds &bounds,
                                const std::function<bool(Real, Real)> &has_data,
                                const std::function<bool(Real, Real)> &is_land, int nsamples);

    std::vector<ElementDataClass> classes_;
    Index num_water_ = 0;
    Index num_beach_ = 0;
    Index num_inland_ = 0;
};

} // namespace drifter
