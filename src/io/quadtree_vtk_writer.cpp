#include "io/quadtree_vtk_writer.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace drifter {

void QuadtreeVTKWriter::write(const std::string& filename,
                              const QuadtreeAdapter& mesh,
                              const LinearBezierSurface& surface) {
    std::string full_path = filename + ".vtu";
    std::ofstream f(full_path);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open " << full_path << " for writing" << std::endl;
        return;
    }

    Index ncells = mesh.num_elements();
    Index npoints = ncells * 4;  // 4 corners per quad (not shared in VTK)

    // VTK XML header
    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <UnstructuredGrid>\n";
    f << "    <Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << ncells << "\">\n";

    // Points
    f << "      <Points>\n";
    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        // Get z values at corners from surface
        Eigen::Vector4d z = surface.element_coefficients(elem);

        // Corner order: (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax) - VTK quad order
        f << std::setprecision(12);
        f << "          " << bounds.xmin << " " << bounds.ymin << " " << z(0) << "\n";  // 0,0
        f << "          " << bounds.xmax << " " << bounds.ymin << " " << z(1) << "\n";  // 1,0
        f << "          " << bounds.xmax << " " << bounds.ymax << " " << z(3) << "\n";  // 1,1
        f << "          " << bounds.xmin << " " << bounds.ymax << " " << z(2) << "\n";  // 0,1
    }
    f << "        </DataArray>\n";
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n";
    f << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        Index base = elem * 4;
        f << "          " << base << " " << base+1 << " " << base+2 << " " << base+3 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          " << (elem + 1) * 4 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          9\n";  // VTK_QUAD = 9
    }
    f << "        </DataArray>\n";
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n";

    // Refinement level
    f << "        <DataArray type=\"Int32\" Name=\"level\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        auto level = mesh.element_level(elem);
        f << "          " << level.max_level() << "\n";
    }
    f << "        </DataArray>\n";

    // Element index
    f << "        <DataArray type=\"Int64\" Name=\"element_id\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          " << elem << "\n";
    }
    f << "        </DataArray>\n";

    // Element area
    f << "        <DataArray type=\"Float64\" Name=\"area\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        auto size = mesh.element_size(elem);
        f << "          " << (size(0) * size(1)) << "\n";
    }
    f << "        </DataArray>\n";

    f << "      </CellData>\n";

    // Close tags
    f << "    </Piece>\n";
    f << "  </UnstructuredGrid>\n";
    f << "</VTKFile>\n";

    f.close();
}

void QuadtreeVTKWriter::write_water_only(const std::string& filename,
                                          const QuadtreeAdapter& mesh,
                                          const LinearBezierSurface& surface,
                                          std::function<Real(Real, Real)> depth_func) {
    if (!depth_func) {
        throw std::runtime_error(
            "QuadtreeVTKWriter::write_water_only: depth_func is null. "
            "Provide a valid depth function.");
    }

    // First pass: identify water elements and store depths
    std::vector<Index> water_elements;
    std::vector<Real> element_depths;

    for (Index elem = 0; elem < mesh.num_elements(); ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        Real cx = 0.5 * (bounds.xmin + bounds.xmax);
        Real cy = 0.5 * (bounds.ymin + bounds.ymax);
        Real depth = depth_func(cx, cy);

        if (depth > 0.0) {
            water_elements.push_back(elem);
            element_depths.push_back(depth);
        }
    }

    if (water_elements.empty()) {
        throw std::runtime_error(
            "QuadtreeVTKWriter::write_water_only: no water elements found. "
            "All " + std::to_string(mesh.num_elements()) +
            " elements have depth <= 0 at their centers.");
    }

    std::string full_path = filename + ".vtu";
    std::ofstream f(full_path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "QuadtreeVTKWriter::write_water_only: could not open '" + full_path + "' for writing.");
    }

    Index ncells = static_cast<Index>(water_elements.size());
    Index npoints = ncells * 4;

    // VTK XML header
    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <UnstructuredGrid>\n";
    f << "    <Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << ncells << "\">\n";

    // Points
    f << "      <Points>\n";
    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (size_t i = 0; i < water_elements.size(); ++i) {
        Index elem = water_elements[i];
        const auto& bounds = mesh.element_bounds(elem);
        Eigen::Vector4d z = surface.element_coefficients(elem);

        f << std::setprecision(12);
        f << "          " << bounds.xmin << " " << bounds.ymin << " " << z(0) << "\n";
        f << "          " << bounds.xmax << " " << bounds.ymin << " " << z(1) << "\n";
        f << "          " << bounds.xmax << " " << bounds.ymax << " " << z(3) << "\n";
        f << "          " << bounds.xmin << " " << bounds.ymax << " " << z(2) << "\n";
    }
    f << "        </DataArray>\n";
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n";
    f << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index i = 0; i < ncells; ++i) {
        Index base = i * 4;
        f << "          " << base << " " << base+1 << " " << base+2 << " " << base+3 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (Index i = 0; i < ncells; ++i) {
        f << "          " << (i + 1) * 4 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index i = 0; i < ncells; ++i) {
        f << "          9\n";
    }
    f << "        </DataArray>\n";
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n";

    // Depth field
    f << "        <DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";
    for (size_t i = 0; i < element_depths.size(); ++i) {
        f << "          " << std::setprecision(12) << element_depths[i] << "\n";
    }
    f << "        </DataArray>\n";

    // Refinement level
    f << "        <DataArray type=\"Int32\" Name=\"level\" format=\"ascii\">\n";
    for (size_t i = 0; i < water_elements.size(); ++i) {
        auto level = mesh.element_level(water_elements[i]);
        f << "          " << level.max_level() << "\n";
    }
    f << "        </DataArray>\n";

    // Element index (original mesh index)
    f << "        <DataArray type=\"Int64\" Name=\"element_id\" format=\"ascii\">\n";
    for (size_t i = 0; i < water_elements.size(); ++i) {
        f << "          " << water_elements[i] << "\n";
    }
    f << "        </DataArray>\n";

    // Element area
    f << "        <DataArray type=\"Float64\" Name=\"area\" format=\"ascii\">\n";
    for (size_t i = 0; i < water_elements.size(); ++i) {
        auto size = mesh.element_size(water_elements[i]);
        f << "          " << (size(0) * size(1)) << "\n";
    }
    f << "        </DataArray>\n";

    f << "      </CellData>\n";

    // Close tags
    f << "    </Piece>\n";
    f << "  </UnstructuredGrid>\n";
    f << "</VTKFile>\n";

    f.close();
}

void QuadtreeVTKWriter::write_mesh_only(const std::string& filename,
                                         const QuadtreeAdapter& mesh) {
    std::string full_path = filename + ".vtu";
    std::ofstream f(full_path);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open " << full_path << " for writing" << std::endl;
        return;
    }

    Index ncells = mesh.num_elements();
    Index npoints = ncells * 4;

    // VTK XML header
    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <UnstructuredGrid>\n";
    f << "    <Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << ncells << "\">\n";

    // Points (z=0 for mesh only)
    f << "      <Points>\n";
    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        f << std::setprecision(12);
        f << "          " << bounds.xmin << " " << bounds.ymin << " 0\n";
        f << "          " << bounds.xmax << " " << bounds.ymin << " 0\n";
        f << "          " << bounds.xmax << " " << bounds.ymax << " 0\n";
        f << "          " << bounds.xmin << " " << bounds.ymax << " 0\n";
    }
    f << "        </DataArray>\n";
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n";
    f << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        Index base = elem * 4;
        f << "          " << base << " " << base+1 << " " << base+2 << " " << base+3 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          " << (elem + 1) * 4 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          9\n";
    }
    f << "        </DataArray>\n";
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n";
    f << "        <DataArray type=\"Int32\" Name=\"level\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        auto level = mesh.element_level(elem);
        f << "          " << level.max_level() << "\n";
    }
    f << "        </DataArray>\n";
    f << "      </CellData>\n";

    f << "    </Piece>\n";
    f << "  </UnstructuredGrid>\n";
    f << "</VTKFile>\n";

    f.close();
}

void QuadtreeVTKWriter::write_with_errors(const std::string& filename,
                                           const QuadtreeAdapter& mesh,
                                           const LinearBezierSurface& surface,
                                           const std::vector<ElementError>& errors,
                                           std::function<Real(Real, Real)> depth_func,
                                           std::function<int(Real, Real)> source_id_func) {
    std::string full_path = filename + ".vtu";
    std::ofstream f(full_path);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open " << full_path << " for writing" << std::endl;
        return;
    }

    Index ncells = mesh.num_elements();
    Index npoints = ncells * 4;

    // Build error lookup map (element index -> error value)
    std::vector<Real> error_values(ncells, 0.0);
    for (const auto& err : errors) {
        if (err.element >= 0 && err.element < ncells) {
            error_values[err.element] = err.error;
        }
    }

    // VTK XML header
    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <UnstructuredGrid>\n";
    f << "    <Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << ncells << "\">\n";

    // Points
    f << "      <Points>\n";
    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        const auto& bounds = mesh.element_bounds(elem);
        Eigen::Vector4d z = surface.element_coefficients(elem);

        f << std::setprecision(12);
        f << "          " << bounds.xmin << " " << bounds.ymin << " " << z(0) << "\n";
        f << "          " << bounds.xmax << " " << bounds.ymin << " " << z(1) << "\n";
        f << "          " << bounds.xmax << " " << bounds.ymax << " " << z(3) << "\n";
        f << "          " << bounds.xmin << " " << bounds.ymax << " " << z(2) << "\n";
    }
    f << "        </DataArray>\n";
    f << "      </Points>\n";

    // Cells
    f << "      <Cells>\n";
    f << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        Index base = elem * 4;
        f << "          " << base << " " << base+1 << " " << base+2 << " " << base+3 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          " << (elem + 1) * 4 << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          9\n";
    }
    f << "        </DataArray>\n";
    f << "      </Cells>\n";

    // Cell data
    f << "      <CellData>\n";

    // Error field (primary field of interest)
    f << "        <DataArray type=\"Float64\" Name=\"error\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          " << std::setprecision(6) << error_values[elem] << "\n";
    }
    f << "        </DataArray>\n";

    // Depth field (if depth function provided)
    if (depth_func) {
        f << "        <DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";
        for (Index elem = 0; elem < ncells; ++elem) {
            const auto& bounds = mesh.element_bounds(elem);
            Real cx = 0.5 * (bounds.xmin + bounds.xmax);
            Real cy = 0.5 * (bounds.ymin + bounds.ymax);
            Real depth = depth_func(cx, cy);
            f << "          " << std::setprecision(6) << depth << "\n";
        }
        f << "        </DataArray>\n";

        // Surface elevation at center (for comparison)
        f << "        <DataArray type=\"Float64\" Name=\"surface_z\" format=\"ascii\">\n";
        for (Index elem = 0; elem < ncells; ++elem) {
            const auto& bounds = mesh.element_bounds(elem);
            Real cx = 0.5 * (bounds.xmin + bounds.xmax);
            Real cy = 0.5 * (bounds.ymin + bounds.ymax);
            Real z = surface.evaluate(cx, cy);
            f << "          " << std::setprecision(6) << z << "\n";
        }
        f << "        </DataArray>\n";
    }

    // Refinement level
    f << "        <DataArray type=\"Int32\" Name=\"level\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        auto level = mesh.element_level(elem);
        f << "          " << level.max_level() << "\n";
    }
    f << "        </DataArray>\n";

    // Element index
    f << "        <DataArray type=\"Int64\" Name=\"element_id\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        f << "          " << elem << "\n";
    }
    f << "        </DataArray>\n";

    // Element area
    f << "        <DataArray type=\"Float64\" Name=\"area\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        auto size = mesh.element_size(elem);
        f << "          " << (size(0) * size(1)) << "\n";
    }
    f << "        </DataArray>\n";

    // Element size (width and height)
    f << "        <DataArray type=\"Float64\" Name=\"element_size_x\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        auto size = mesh.element_size(elem);
        f << "          " << size(0) << "\n";
    }
    f << "        </DataArray>\n";

    f << "        <DataArray type=\"Float64\" Name=\"element_size_y\" format=\"ascii\">\n";
    for (Index elem = 0; elem < ncells; ++elem) {
        auto size = mesh.element_size(elem);
        f << "          " << size(1) << "\n";
    }
    f << "        </DataArray>\n";

    // Source ID field (which TIF file provides data for this element)
    if (source_id_func) {
        f << "        <DataArray type=\"Int32\" Name=\"source_id\" format=\"ascii\">\n";
        for (Index elem = 0; elem < ncells; ++elem) {
            const auto& bounds = mesh.element_bounds(elem);
            Real cx = 0.5 * (bounds.xmin + bounds.xmax);
            Real cy = 0.5 * (bounds.ymin + bounds.ymax);
            int source_id = source_id_func(cx, cy);
            f << "          " << source_id << "\n";
        }
        f << "        </DataArray>\n";
    }

    f << "      </CellData>\n";

    // Close tags
    f << "    </Piece>\n";
    f << "  </UnstructuredGrid>\n";
    f << "</VTKFile>\n";

    f.close();
    std::cout << "Wrote error field to: " << full_path << std::endl;
}

} // namespace drifter
