#include "bathymetry/element_data_mask.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace drifter {

namespace {

/// Position quantization for node sharing, matching CGHermiteDofManager: a
/// mesh-relative origin and a tolerance scaled to the smallest element, so that
/// large projected coordinates (UTM and similar) do not lose resolution.
class PositionQuantizer {
public:
    explicit PositionQuantizer(const QuadtreeAdapter &mesh) {
        Real xmin = std::numeric_limits<Real>::max();
        Real ymin = std::numeric_limits<Real>::max();
        Real min_element_size = std::numeric_limits<Real>::max();
        for (Index e = 0; e < mesh.num_elements(); ++e) {
            const auto &b = mesh.element_bounds(e);
            xmin = std::min(xmin, b.xmin);
            ymin = std::min(ymin, b.ymin);
            min_element_size =
                std::min(min_element_size, std::min(b.xmax - b.xmin, b.ymax - b.ymin));
        }
        if (mesh.num_elements() > 0) {
            xmin_ = xmin;
            ymin_ = ymin;
            inv_tol_ = 1.0 / (min_element_size * 1e-8);
        }
    }

    std::pair<int64_t, int64_t> operator()(Real x, Real y) const {
        return {static_cast<int64_t>(std::round((x - xmin_) * inv_tol_)),
                static_cast<int64_t>(std::round((y - ymin_) * inv_tol_))};
    }

private:
    Real xmin_ = 0.0;
    Real ymin_ = 0.0;
    Real inv_tol_ = 1.0;
};

} // namespace

bool ElementDataMask::sample_is_water(const QuadBounds &bounds,
                                      const std::function<bool(Real, Real)> &has_data,
                                      const std::function<bool(Real, Real)> &is_land,
                                      int nsamples) {
    const Real dx = bounds.xmax - bounds.xmin;
    const Real dy = bounds.ymax - bounds.ymin;

    auto is_water_at = [&](Real x, Real y) {
        if (has_data && !has_data(x, y)) {
            return false;
        }
        return !(is_land && is_land(x, y));
    };

    // The corners and the centre first: they are the cheapest way to find water in
    // an element that has any, and the raster lookups behind these predicates are
    // the expensive part of classification.
    //
    // The corners are nudged inward by a fraction of the element. A corner is shared
    // with up to four elements, so probing it exactly would let a coastline passing
    // through that one point decide the class of all four - the classification would
    // turn on a measure-zero set. Inset, it is a question about this element's own
    // interior.
    constexpr Real CORNER_INSET = 1e-3;
    const Real ix = CORNER_INSET * dx;
    const Real iy = CORNER_INSET * dy;
    const std::array<Vec2, 5> probes = {
        Vec2(bounds.xmin + ix, bounds.ymin + iy), Vec2(bounds.xmax - ix, bounds.ymin + iy),
        Vec2(bounds.xmin + ix, bounds.ymax - iy), Vec2(bounds.xmax - ix, bounds.ymax - iy),
        Vec2(bounds.xmin + 0.5 * dx, bounds.ymin + 0.5 * dy)};
    for (const Vec2 &p : probes) {
        if (is_water_at(p(0), p(1))) {
            return true;
        }
    }

    // Then an interior grid, which catches a channel narrow enough to slip between
    // the corners and the centre.
    for (int j = 0; j < nsamples; ++j) {
        const Real v = (static_cast<Real>(j) + 0.5) / static_cast<Real>(nsamples);
        for (int i = 0; i < nsamples; ++i) {
            const Real u = (static_cast<Real>(i) + 0.5) / static_cast<Real>(nsamples);
            if (is_water_at(bounds.xmin + u * dx, bounds.ymin + v * dy)) {
                return true;
            }
        }
    }

    return false;
}

ElementDataMask::ElementDataMask(const QuadtreeAdapter &mesh,
                                 const std::function<bool(Real, Real)> &has_data,
                                 const std::function<bool(Real, Real)> &is_land, int nsamples) {
    const Index num_elements = mesh.num_elements();
    classes_.assign(static_cast<size_t>(num_elements), ElementDataClass::Water);

    // No masks at all: an analytic source has neither gaps nor land, so the whole
    // mesh is water and every downstream path behaves exactly as before.
    if (!has_data && !is_land) {
        num_water_ = num_elements;
        return;
    }

    // Pass 1: water or not, per element
    for (Index e = 0; e < num_elements; ++e) {
        if (!sample_is_water(mesh.element_bounds(e), has_data, is_land, nsamples)) {
            classes_[static_cast<size_t>(e)] = ElementDataClass::Inland;
        }
    }

    // Pass 2: promote a non-water element to Beach if any element sharing one of
    // its corners is Water. Corner sharing rather than edge adjacency - see the
    // header for why that distinction is what makes dropping Inland safe.
    const PositionQuantizer quantize(mesh);
    std::map<std::pair<int64_t, int64_t>, std::vector<Index>> node_to_elements;
    for (Index e = 0; e < num_elements; ++e) {
        const auto &b = mesh.element_bounds(e);
        const std::array<Vec2, 4> corners = {Vec2(b.xmin, b.ymin), Vec2(b.xmax, b.ymin),
                                             Vec2(b.xmin, b.ymax), Vec2(b.xmax, b.ymax)};
        for (const Vec2 &c : corners) {
            node_to_elements[quantize(c(0), c(1))].push_back(e);
        }
    }

    // A T-junction puts a fine element's corner at the midpoint of a coarse edge,
    // where the coarse element has no corner of its own, so corner-position
    // matching alone would miss that pair. Walk the edge neighbours as well.
    auto mark_beach = [&](Index elem) {
        if (classes_[static_cast<size_t>(elem)] == ElementDataClass::Inland) {
            classes_[static_cast<size_t>(elem)] = ElementDataClass::Beach;
        }
    };

    for (const auto &[node, elements] : node_to_elements) {
        const bool touches_water =
            std::any_of(elements.begin(), elements.end(), [this](Index e) {
                return classes_[static_cast<size_t>(e)] == ElementDataClass::Water;
            });
        if (!touches_water) {
            continue;
        }
        for (Index e : elements) {
            mark_beach(e);
        }
    }

    for (Index e = 0; e < num_elements; ++e) {
        if (classes_[static_cast<size_t>(e)] == ElementDataClass::Water) {
            continue;
        }
        for (int edge = 0; edge < 4; ++edge) {
            const EdgeNeighborInfo info = mesh.get_neighbor(e, edge);
            for (Index n : info.neighbor_elements) {
                if (classes_[static_cast<size_t>(n)] == ElementDataClass::Water) {
                    mark_beach(e);
                }
            }
        }
    }

    for (const auto c : classes_) {
        switch (c) {
        case ElementDataClass::Water: ++num_water_; break;
        case ElementDataClass::Beach: ++num_beach_; break;
        case ElementDataClass::Inland: ++num_inland_; break;
        }
    }
}

} // namespace drifter
