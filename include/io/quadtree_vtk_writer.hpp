#pragma once

/// @file quadtree_vtk_writer.hpp
/// @brief VTK output for linear quadrilateral meshes

#include "bathymetry/element_error_estimator.hpp"
#include "bathymetry/linear_bezier_surface.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include <functional>
#include <string>
#include <vector>

namespace drifter {

/// @brief VTK writer for linear quadrilateral elements
///
/// Outputs VTK unstructured grid (.vtu) files with:
/// - Linear quadrilateral cells (VTK type 9)
/// - Point data: z-coordinate (surface height)
/// - Cell data: refinement level, element index, area
class QuadtreeVTKWriter {
public:
    /// @brief Write mesh and surface to VTK file
    /// @param filename Output filename (without extension)
    /// @param mesh Quadtree mesh
    /// @param surface Linear Bezier surface
    void write(const std::string& filename,
               const QuadtreeAdapter& mesh,
               const LinearBezierSurface& surface);

    /// @brief Write only water elements (depth > 0) to VTK file
    /// @param filename Output filename (without extension)
    /// @param mesh Quadtree mesh
    /// @param surface Linear Bezier surface
    /// @param depth_func Function returning depth at (x, y); elements with depth > 0 are written
    void write_water_only(const std::string& filename,
                          const QuadtreeAdapter& mesh,
                          const LinearBezierSurface& surface,
                          std::function<Real(Real, Real)> depth_func);

    /// @brief Write mesh only (no surface data)
    /// @param filename Output filename (without extension)
    /// @param mesh Quadtree mesh
    void write_mesh_only(const std::string& filename,
                         const QuadtreeAdapter& mesh);

    /// @brief Write mesh with error and depth data for visualization
    /// @param filename Output filename (without extension)
    /// @param mesh Quadtree mesh
    /// @param surface Linear Bezier surface
    /// @param errors Per-element error estimates
    /// @param depth_func Function returning depth at (x, y)
    /// @param source_id_func Function returning source index at (x, y): 0=primary, 1..N=tiles, -1=none
    void write_with_errors(const std::string& filename,
                           const QuadtreeAdapter& mesh,
                           const LinearBezierSurface& surface,
                           const std::vector<ElementError>& errors,
                           std::function<Real(Real, Real)> depth_func = nullptr,
                           std::function<int(Real, Real)> source_id_func = nullptr);
};

} // namespace drifter
