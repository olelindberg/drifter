#pragma once

/// @file vtk_binary_utils.hpp
/// @brief Shared utilities for binary VTK output encoding

#include "core/types.hpp"
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

namespace drifter {
namespace vtk {

// =============================================================================
// Base64 encoding
// =============================================================================

namespace detail {

inline constexpr char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                       "abcdefghijklmnopqrstuvwxyz"
                                       "0123456789+/";

} // namespace detail

/// @brief Encode binary data as base64 string
/// @param data Pointer to binary data
/// @param len Length of data in bytes
/// @return Base64 encoded string
inline std::string base64_encode(const unsigned char* data, size_t len) {
    std::string result;
    result.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        unsigned int n = data[i] << 16;
        if (i + 1 < len)
            n |= data[i + 1] << 8;
        if (i + 2 < len)
            n |= data[i + 2];

        result += detail::base64_chars[(n >> 18) & 0x3F];
        result += detail::base64_chars[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? detail::base64_chars[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? detail::base64_chars[n & 0x3F] : '=';
    }

    return result;
}

// =============================================================================
// Binary data array writers
// =============================================================================

/// @brief Write Float64 data array in VTK binary format
/// @param out Output stream
/// @param name Data array name
/// @param num_components Number of components per value (1 for scalar, 3 for vector)
/// @param data Data values
inline void write_binary_float64(std::ostream& out, const std::string& name,
                                 int num_components, const std::vector<Real>& data) {
    out << "<DataArray type=\"Float64\" Name=\"" << name << "\" NumberOfComponents=\""
        << num_components << "\" format=\"binary\">";

    // VTK binary format: prepend size as 64-bit unsigned integer
    uint64_t size = data.size() * sizeof(Real);
    std::vector<unsigned char> buffer(sizeof(uint64_t) + size);
    std::memcpy(buffer.data(), &size, sizeof(uint64_t));
    std::memcpy(buffer.data() + sizeof(uint64_t), data.data(), size);

    out << base64_encode(buffer.data(), buffer.size());
    out << "</DataArray>\n";
}

/// @brief Write Int64 data array in VTK binary format
/// @param out Output stream
/// @param name Data array name
/// @param data Data values
inline void write_binary_int64(std::ostream& out, const std::string& name,
                               const std::vector<int64_t>& data) {
    out << "<DataArray type=\"Int64\" Name=\"" << name << "\" format=\"binary\">";

    uint64_t size = data.size() * sizeof(int64_t);
    std::vector<unsigned char> buffer(sizeof(uint64_t) + size);
    std::memcpy(buffer.data(), &size, sizeof(uint64_t));
    std::memcpy(buffer.data() + sizeof(uint64_t), data.data(), size);

    out << base64_encode(buffer.data(), buffer.size());
    out << "</DataArray>\n";
}

/// @brief Write Int32 data array in VTK binary format
/// @param out Output stream
/// @param name Data array name
/// @param data Data values
inline void write_binary_int32(std::ostream& out, const std::string& name,
                               const std::vector<int32_t>& data) {
    out << "<DataArray type=\"Int32\" Name=\"" << name << "\" format=\"binary\">";

    uint64_t size = data.size() * sizeof(int32_t);
    std::vector<unsigned char> buffer(sizeof(uint64_t) + size);
    std::memcpy(buffer.data(), &size, sizeof(uint64_t));
    std::memcpy(buffer.data() + sizeof(uint64_t), data.data(), size);

    out << base64_encode(buffer.data(), buffer.size());
    out << "</DataArray>\n";
}

/// @brief Write UInt8 data array in VTK binary format
/// @param out Output stream
/// @param name Data array name
/// @param data Data values
inline void write_binary_uint8(std::ostream& out, const std::string& name,
                               const std::vector<uint8_t>& data) {
    out << "<DataArray type=\"UInt8\" Name=\"" << name << "\" format=\"binary\">";

    uint64_t size = data.size() * sizeof(uint8_t);
    std::vector<unsigned char> buffer(sizeof(uint64_t) + size);
    std::memcpy(buffer.data(), &size, sizeof(uint64_t));
    if (!data.empty()) {
        std::memcpy(buffer.data() + sizeof(uint64_t), data.data(), size);
    }

    out << base64_encode(buffer.data(), buffer.size());
    out << "</DataArray>\n";
}

/// @brief Write Float64 points in VTK binary format (3 components)
/// @param out Output stream
/// @param data Point coordinates (x,y,z interleaved)
inline void write_binary_points(std::ostream& out, const std::vector<Real>& data) {
    out << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"binary\">";

    uint64_t size = data.size() * sizeof(Real);
    std::vector<unsigned char> buffer(sizeof(uint64_t) + size);
    std::memcpy(buffer.data(), &size, sizeof(uint64_t));
    std::memcpy(buffer.data() + sizeof(uint64_t), data.data(), size);

    out << base64_encode(buffer.data(), buffer.size());
    out << "</DataArray>\n";
}

} // namespace vtk
} // namespace drifter
