#pragma once

// Bathymetry VTK Writer
//
// Provides VTK output for 2D bathymetry surfaces. Separates IO code
// from algorithmic code in the bathymetry smoothers.
//
// Supports:
// - Bezier surfaces (quintic, cubic, or any degree)
// - Lagrange surfaces (CG smoother)
// - Control point visualization
// - Seabed surface from 3D mesh
//
// Usage:
//   io::write_bezier_surface_vtk("output", mesh,
//       [&](Index e) { return smoother.element_coefficients(e); },
//       [&](const VecX& c, Real u, Real v) { return basis.evaluate_scalar(c, u,
//       v); }, 11);  // resolution

#include "core/types.hpp"
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace drifter {

// Forward declarations
class QuadtreeAdapter;
class OctreeAdapter;
struct QuadBounds;

namespace io {

/// @brief Write a Bezier/polynomial surface to VTK format
///
/// Samples the surface at LGL nodes within each element and outputs
/// as VTU with quad cells. Suitable for Bezier smoothers.
///
/// @param filename Output filename (without extension, .vtu will be added)
/// @param mesh The quadtree mesh
/// @param get_coefficients Function returning coefficients for element e
/// @param evaluate Function evaluating surface at (u,v) given coefficients
/// @param resolution Number of LGL sample points per direction (default 11)
/// @param scalar_name Name for the elevation/depth scalar field
void write_bezier_surface_vtk(const std::string &filename, const QuadtreeAdapter &mesh,
                              const std::function<VecX(Index)> &get_coefficients,
                              const std::function<Real(const VecX &, Real, Real)> &evaluate,
                              int resolution = 11, const std::string &scalar_name = "elevation");

/// @brief Write Bezier control points to VTK format
///
/// Outputs the control point mesh as VTU with quad cells connecting
/// adjacent control points. Useful for visualizing the Bezier control net.
///
/// @param filename Output filename (full path with extension, e.g.,
/// "output.vtu")
/// @param mesh The quadtree mesh
/// @param get_coefficients Function returning coefficients for element e
/// @param control_point_position Function returning (u,v) for DOF index
/// @param n1d Number of control points per direction (e.g., 6 for quintic, 4
/// for cubic)
void write_bezier_control_points_vtk(const std::string &filename, const QuadtreeAdapter &mesh,
                                     const std::function<VecX(Index)> &get_coefficients,
                                     const std::function<Vec2(int)> &control_point_position,
                                     int n1d);

/// @brief Write Lagrange surface to VTK format
///
/// Samples the surface at uniform grid within each element. Suitable for
/// CG Lagrange bathymetry smoother.
///
/// @param filename Output filename (without extension, .vtu will be added)
/// @param mesh The quadtree mesh
/// @param evaluate_in_element Function evaluating depth at (x,y) in element e
/// @param evaluate_raw Optional function for raw bathymetry comparison
/// @param resolution Number of sample points per direction (default 10)
void write_lagrange_surface_vtk(const std::string &filename, const QuadtreeAdapter &mesh,
                                const std::function<Real(Index, Real, Real)> &evaluate_in_element,
                                const std::function<Real(Real, Real)> &evaluate_raw = nullptr,
                                int resolution = 10);

/// @brief Write seabed surface from 3D mesh to VTK format
///
/// Outputs the seabed (bottom face) of a 3D ocean mesh as a 2D VTU surface.
/// Uses the provided interpolator for high-order evaluation.
///
/// @param filename Output filename (without extension, .vtu will be added)
/// @param mesh The 3D octree mesh
/// @param depth_coeffs Per-element depth coefficients
/// @param bottom_elements Indices of bottom-layer elements
/// @param evaluate_2d Function evaluating depth at (xi, eta) in reference
/// coords
/// @param resolution Number of sample points per direction (default 10)
void write_seabed_surface_vtk(const std::string &filename, const OctreeAdapter &mesh,
                              const std::vector<VecX> &depth_coeffs,
                              const std::vector<Index> &bottom_elements,
                              const std::function<Real(const VecX &, Real, Real)> &evaluate_2d,
                              int resolution = 10);

/// @brief Write CG Bezier/polynomial surface to VTK format with shared vertices
///
/// Unlike write_bezier_surface_vtk() which creates duplicate vertices at
/// element boundaries (suitable for DG), this function deduplicates vertices at
/// shared boundaries for CG (Continuous Galerkin) elements. This produces a
/// properly connected mesh without visual gaps in ParaView.
///
/// @param filename Output filename (without extension, .vtu will be added)
/// @param mesh The quadtree mesh
/// @param evaluate_at Function evaluating surface z-value at physical (x, y)
/// @param resolution Number of sample points per direction (default 11)
/// @param scalar_name Name for the elevation/depth scalar field
void write_cg_bezier_surface_vtk(
    const std::string &filename, const QuadtreeAdapter &mesh,
    const std::function<Real(Real, Real)> &evaluate_at, int resolution = 11,
    const std::string &scalar_name = "elevation",
    const std::vector<std::pair<std::string, std::vector<Real>>> &element_cell_data = {});

/// @brief Write CG Bezier surface with explicit quantization parameters
///
/// This overload allows passing quantization parameters from a DOF manager
/// to ensure consistent vertex deduplication between DOF sharing and VTK
/// output.
///
/// @param filename Output filename (without extension, .vtu will be added)
/// @param mesh The quadtree mesh
/// @param evaluate_at Function evaluating surface z-value at physical (x, y)
/// @param xmin_domain Domain origin x for coordinate normalization
/// @param ymin_domain Domain origin y for coordinate normalization
/// @param inv_quantization_tol Inverse tolerance for position quantization
/// @param resolution Number of sample points per direction (default 11)
/// @param scalar_name Name for the elevation/depth scalar field
void write_cg_bezier_surface_vtk(
    const std::string &filename, const QuadtreeAdapter &mesh,
    const std::function<Real(Real, Real)> &evaluate_at, Real xmin_domain, Real ymin_domain,
    Real inv_quantization_tol, int resolution = 11, const std::string &scalar_name = "elevation",
    const std::vector<std::pair<std::string, std::vector<Real>>> &element_cell_data = {});

/// @brief Write a polygon boundary to VTK PolyData format
///
/// Outputs a closed polygon as a VTP file. Useful for visualizing
/// region boundaries in ParaView (e.g., domain exclusion zones).
///
/// @param filename Output filename (without extension, .vtp will be added)
/// @param polygon_points Polygon vertices as (x, y) pairs in world coordinates
/// @param z_value Z coordinate for all points (default 0 for 2D overlay)
/// @param polygon_name Name attribute for the polygon (optional)
void write_polygon_vtk(const std::string &filename,
                       const std::vector<std::pair<Real, Real>> &polygon_points, Real z_value = 0.0,
                       const std::string &polygon_name = "polygon");

} // namespace io
} // namespace drifter
