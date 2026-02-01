#include "io/bathymetry_vtk_writer.hpp"
#include "bathymetry/quadtree_adapter.hpp"
#include "mesh/octree_adapter.hpp"
#include "dg/basis_hexahedron.hpp"  // For compute_gauss_lobatto_nodes
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace drifter {
namespace io {

void write_bezier_surface_vtk(
    const std::string& filename,
    const QuadtreeAdapter& mesh,
    const std::function<VecX(Index)>& get_coefficients,
    const std::function<Real(const VecX&, Real, Real)>& evaluate,
    int resolution,
    const std::string& scalar_name) {

    std::ofstream file(filename + ".vtu");
    if (!file) {
        throw std::runtime_error("write_bezier_surface_vtk: cannot open " +
                                 filename + ".vtu");
    }

    // Use LGL nodes for high-order accurate interpolation
    int n_lgl = resolution > 0 ? resolution : 11;
    VecX lgl_nodes, lgl_weights;
    compute_gauss_lobatto_nodes(n_lgl, lgl_nodes, lgl_weights);

    // Map LGL nodes from [-1, 1] to [0, 1] for Bezier parameter space
    VecX param_nodes = (lgl_nodes.array() + 1.0) * 0.5;

    Index num_elements = mesh.num_elements();
    int pts_per_elem = n_lgl * n_lgl;
    int cells_per_elem = (n_lgl - 1) * (n_lgl - 1);
    Index total_points = num_elements * pts_per_elem;
    Index total_cells = num_elements * cells_per_elem;

    // VTU XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << total_points << "\" NumberOfCells=\"" << total_cells << "\">\n";

    // Points - evaluate at LGL nodes
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (Index elem = 0; elem < num_elements; ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;
        VecX coeffs = get_coefficients(elem);

        for (int j = 0; j < n_lgl; ++j) {
            Real v = param_nodes(j);
            Real y = bounds.ymin + v * dy;
            for (int i = 0; i < n_lgl; ++i) {
                Real u = param_nodes(i);
                Real x = bounds.xmin + u * dx;
                Real z = evaluate(coeffs, u, v);
                file << std::setprecision(12) << x << " " << y << " " << z << "\n";
            }
        }
    }
    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells (quads connecting LGL points)
    file << "<Cells>\n";
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index elem = 0; elem < num_elements; ++elem) {
        Index base = elem * pts_per_elem;
        for (int j = 0; j < n_lgl - 1; ++j) {
            for (int i = 0; i < n_lgl - 1; ++i) {
                Index p0 = base + j * n_lgl + i;
                Index p1 = p0 + 1;
                Index p2 = p0 + n_lgl + 1;
                Index p3 = p0 + n_lgl;
                file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
    }
    file << "</DataArray>\n";

    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (Index i = 1; i <= total_cells; ++i) {
        file << (i * 4) << "\n";
    }
    file << "</DataArray>\n";

    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index i = 0; i < total_cells; ++i) {
        file << "9\n";  // VTK_QUAD
    }
    file << "</DataArray>\n";
    file << "</Cells>\n";

    // Point data: elevation
    file << "<PointData Scalars=\"" << scalar_name << "\">\n";
    file << "<DataArray type=\"Float64\" Name=\"" << scalar_name << "\" format=\"ascii\">\n";
    for (Index elem = 0; elem < num_elements; ++elem) {
        VecX coeffs = get_coefficients(elem);

        for (int j = 0; j < n_lgl; ++j) {
            Real v = param_nodes(j);
            for (int i = 0; i < n_lgl; ++i) {
                Real u = param_nodes(i);
                Real z = evaluate(coeffs, u, v);
                file << std::setprecision(12) << z << "\n";
            }
        }
    }
    file << "</DataArray>\n";
    file << "</PointData>\n";

    // Cell data: element ID
    file << "<CellData Scalars=\"element_id\">\n";
    file << "<DataArray type=\"Int64\" Name=\"element_id\" format=\"ascii\">\n";
    for (Index elem = 0; elem < num_elements; ++elem) {
        for (int c = 0; c < cells_per_elem; ++c) {
            file << elem << "\n";
        }
    }
    file << "</DataArray>\n";
    file << "</CellData>\n";

    // VTU footer
    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

void write_bezier_control_points_vtk(
    const std::string& filename,
    const QuadtreeAdapter& mesh,
    const std::function<VecX(Index)>& get_coefficients,
    const std::function<Vec2(int)>& control_point_position,
    int n1d) {

    // Use filename as-is (caller provides full path with extension)
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("write_bezier_control_points_vtk: cannot open " +
                                 filename);
    }

    int ndof = n1d * n1d;
    Index num_elements = mesh.num_elements();
    Index total_points = num_elements * ndof;

    // Cells: connect control points as quads ((n1d-1) × (n1d-1) per element)
    int cells_per_elem = (n1d - 1) * (n1d - 1);
    Index total_cells = num_elements * cells_per_elem;

    // VTK header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << total_points << "\" NumberOfCells=\"" << total_cells << "\">\n";

    // Points: control point positions with z from solution
    file << "<Points>\n";
    file << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elements; ++e) {
        const auto& bounds = mesh.element_bounds(e);
        VecX coeffs = get_coefficients(e);

        for (int dof = 0; dof < ndof; ++dof) {
            Vec2 uv = control_point_position(dof);
            Real x = bounds.xmin + uv(0) * (bounds.xmax - bounds.xmin);
            Real y = bounds.ymin + uv(1) * (bounds.ymax - bounds.ymin);
            Real z = coeffs(dof);

            file << std::setprecision(12) << x << " " << y << " " << z << "\n";
        }
    }

    file << "</DataArray>\n";
    file << "</Points>\n";

    // Cells: quads connecting control points
    file << "<Cells>\n";
    file << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";

    Index pt_offset = 0;
    for (Index e = 0; e < num_elements; ++e) {
        for (int j = 0; j < n1d - 1; ++j) {
            for (int i = 0; i < n1d - 1; ++i) {
                // DOF indexing: dof = i + n1d * j
                Index p0 = pt_offset + i + n1d * j;
                Index p1 = pt_offset + (i + 1) + n1d * j;
                Index p2 = pt_offset + (i + 1) + n1d * (j + 1);
                Index p3 = pt_offset + i + n1d * (j + 1);

                file << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
        pt_offset += ndof;
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";

    for (Index c = 1; c <= total_cells; ++c) {
        file << 4 * c << "\n";
    }

    file << "</DataArray>\n";
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";

    for (Index c = 0; c < total_cells; ++c) {
        file << "9\n";  // VTK_QUAD
    }

    file << "</DataArray>\n";
    file << "</Cells>\n";

    // Point data: control point z-value (depth)
    file << "<PointData Scalars=\"depth\">\n";
    file << "<DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elements; ++e) {
        VecX coeffs = get_coefficients(e);
        for (int dof = 0; dof < ndof; ++dof) {
            file << std::setprecision(12) << coeffs(dof) << "\n";
        }
    }

    file << "</DataArray>\n";
    file << "</PointData>\n";

    // Cell data: element index
    file << "<CellData Scalars=\"element\">\n";
    file << "<DataArray type=\"Int64\" Name=\"element\" format=\"ascii\">\n";

    for (Index e = 0; e < num_elements; ++e) {
        for (int c = 0; c < cells_per_elem; ++c) {
            file << e << "\n";
        }
    }

    file << "</DataArray>\n";
    file << "</CellData>\n";

    // Footer
    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";
}

void write_lagrange_surface_vtk(
    const std::string& filename,
    const QuadtreeAdapter& mesh,
    const std::function<Real(Index, Real, Real)>& evaluate_in_element,
    const std::function<Real(Real, Real)>& evaluate_raw,
    int resolution) {

    Index num_elements = mesh.num_elements();
    Index points_per_elem = (resolution + 1) * (resolution + 1);
    Index quads_per_elem = resolution * resolution;

    Index total_points = num_elements * points_per_elem;
    Index total_quads = num_elements * quads_per_elem;

    // Collect points and depths
    std::vector<Vec3> points;
    std::vector<Real> depths;
    std::vector<Real> raw_depths;

    points.reserve(total_points);
    depths.reserve(total_points);
    if (evaluate_raw) {
        raw_depths.reserve(total_points);
    }

    for (Index e = 0; e < num_elements; ++e) {
        const auto& bounds = mesh.element_bounds(e);
        Real hx = bounds.xmax - bounds.xmin;
        Real hy = bounds.ymax - bounds.ymin;

        for (int j = 0; j <= resolution; ++j) {
            for (int i = 0; i <= resolution; ++i) {
                // Physical coordinates
                Real x = bounds.xmin + hx * i / resolution;
                Real y = bounds.ymin + hy * j / resolution;

                // Evaluate CG solution
                Real depth = evaluate_in_element(e, x, y);

                points.emplace_back(x, y, -depth);  // z = -depth for visualization
                depths.push_back(depth);

                if (evaluate_raw) {
                    raw_depths.push_back(evaluate_raw(x, y));
                }
            }
        }
    }

    // Write VTU file
    std::ofstream file(filename + ".vtu");
    if (!file) {
        throw std::runtime_error("write_lagrange_surface_vtk: cannot open " +
                                 filename + ".vtu");
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << total_points
         << "\" NumberOfCells=\"" << total_quads << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    file << std::setprecision(12);
    for (const auto& p : points) {
        file << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Cells (quads)
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index e = 0; e < num_elements; ++e) {
        Index base = e * points_per_elem;
        int n1d = resolution + 1;

        for (int j = 0; j < resolution; ++j) {
            for (int i = 0; i < resolution; ++i) {
                Index p0 = base + i + j * n1d;
                Index p1 = base + (i + 1) + j * n1d;
                Index p2 = base + (i + 1) + (j + 1) * n1d;
                Index p3 = base + i + (j + 1) * n1d;
                file << "          " << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
            }
        }
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (Index i = 1; i <= total_quads; ++i) {
        file << "          " << (i * 4) << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index i = 0; i < total_quads; ++i) {
        file << "          9\n";  // VTK_QUAD = 9
    }
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    // Point data
    file << "      <PointData Scalars=\"depth\">\n";

    // Smoothed depth
    file << "        <DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";
    for (Real d : depths) {
        file << "          " << d << "\n";
    }
    file << "        </DataArray>\n";

    // Raw bathymetry (if available)
    if (evaluate_raw) {
        file << "        <DataArray type=\"Float64\" Name=\"raw_bathy\" format=\"ascii\">\n";
        for (Real d : raw_depths) {
            file << "          " << d << "\n";
        }
        file << "        </DataArray>\n";

        // Difference
        file << "        <DataArray type=\"Float64\" Name=\"difference\" format=\"ascii\">\n";
        for (size_t i = 0; i < depths.size(); ++i) {
            file << "          " << (depths[i] - raw_depths[i]) << "\n";
        }
        file << "        </DataArray>\n";
    }

    file << "      </PointData>\n";

    // Cell data (element ID)
    file << "      <CellData>\n";
    file << "        <DataArray type=\"Int64\" Name=\"element_id\" format=\"ascii\">\n";
    for (Index e = 0; e < num_elements; ++e) {
        for (Index q = 0; q < quads_per_elem; ++q) {
            file << "          " << e << "\n";
        }
    }
    file << "        </DataArray>\n";
    file << "      </CellData>\n";

    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

void write_seabed_surface_vtk(
    const std::string& filename,
    const OctreeAdapter& mesh,
    const std::vector<VecX>& depth_coeffs,
    const std::vector<Index>& bottom_elements,
    const std::function<Real(const VecX&, Real, Real)>& evaluate_2d,
    int resolution) {

    const auto& elements = mesh.elements();

    // Collect points and cells
    std::vector<Vec3> all_points;
    std::vector<Real> all_depths;
    std::vector<std::array<size_t, 4>> all_quads;

    for (size_t s = 0; s < bottom_elements.size(); ++s) {
        Index mesh_idx = bottom_elements[s];
        const auto& bounds = elements[mesh_idx]->bounds;

        Real dx = bounds.xmax - bounds.xmin;
        Real dy = bounds.ymax - bounds.ymin;

        size_t base_pt = all_points.size();

        // Generate (resolution+1) x (resolution+1) grid of points
        for (int j = 0; j <= resolution; ++j) {
            for (int i = 0; i <= resolution; ++i) {
                Real xi = -1.0 + 2.0 * i / resolution;
                Real eta = -1.0 + 2.0 * j / resolution;

                Real x = bounds.xmin + 0.5 * (xi + 1.0) * dx;
                Real y = bounds.ymin + 0.5 * (eta + 1.0) * dy;

                Real h = evaluate_2d(depth_coeffs[s], xi, eta);

                all_points.push_back(Vec3(x, y, -h));
                all_depths.push_back(h);
            }
        }

        // Generate quads
        for (int j = 0; j < resolution; ++j) {
            for (int i = 0; i < resolution; ++i) {
                size_t p00 = base_pt + i + (resolution + 1) * j;
                size_t p10 = base_pt + (i + 1) + (resolution + 1) * j;
                size_t p01 = base_pt + i + (resolution + 1) * (j + 1);
                size_t p11 = base_pt + (i + 1) + (resolution + 1) * (j + 1);

                all_quads.push_back({p00, p10, p11, p01});
            }
        }
    }

    // Write VTU file
    std::ofstream vtk_file(filename + ".vtu");
    if (!vtk_file) {
        throw std::runtime_error("write_seabed_surface_vtk: cannot open " +
                                 filename + ".vtu");
    }

    vtk_file << "<?xml version=\"1.0\"?>\n";
    vtk_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtk_file << "  <UnstructuredGrid>\n";
    vtk_file << "    <Piece NumberOfPoints=\"" << all_points.size()
             << "\" NumberOfCells=\"" << all_quads.size() << "\">\n";

    // Points
    vtk_file << "      <Points>\n";
    vtk_file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& p : all_points) {
        vtk_file << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Points>\n";

    // Cells
    vtk_file << "      <Cells>\n";
    vtk_file << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& q : all_quads) {
        vtk_file << "          " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 0; i < all_quads.size(); ++i) {
        vtk_file << "          " << (i + 1) * 4 << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t i = 0; i < all_quads.size(); ++i) {
        vtk_file << "          9\n";  // VTK_QUAD
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Cells>\n";

    // Point data
    vtk_file << "      <PointData Scalars=\"depth\">\n";
    vtk_file << "        <DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";
    for (Real d : all_depths) {
        vtk_file << "          " << d << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </PointData>\n";

    vtk_file << "    </Piece>\n";
    vtk_file << "  </UnstructuredGrid>\n";
    vtk_file << "</VTKFile>\n";
}

}  // namespace io
}  // namespace drifter
