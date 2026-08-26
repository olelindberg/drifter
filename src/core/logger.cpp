/// @file logger.cpp
/// @brief Logger implementation

#include "core/logger.hpp"

#include <iomanip>

namespace drifter {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

std::string_view Logger::basename(const char* path) {
    std::string_view sv(path);
    auto pos = sv.find_last_of("/\\");
    return (pos == std::string_view::npos) ? sv : sv.substr(pos + 1);
}

const char* Logger::level_string(LogLevel level) {
    switch (level) {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::WARNING:
        return "WARNING";
    case LogLevel::ERROR:
        return "ERROR";
    }
    return "UNKNOWN";
}

void Logger::log(LogLevel level, const char* file, int line, const char* func, const std::string& message) {
    if (level < min_level_)
        return;

    // Build context string: "file.cpp:123 funcname"
    std::ostringstream ctx;
    ctx << basename(file) << ":" << line << " " << func;

    std::lock_guard<std::mutex> lock(mutex_);

    auto& out = (level >= LogLevel::WARNING) ? std::cerr : std::cout;
    out << "[" << std::left << std::setw(context_width_) << ctx.str() << "] "
        << "[" << std::left << std::setw(level_width_) << level_string(level) << "] " << message << "\n";
}

} // namespace drifter
