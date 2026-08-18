#include "io/quadtree_vtk_writer.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

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

} // namespace drifter
