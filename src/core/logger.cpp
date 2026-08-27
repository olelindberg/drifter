/// @file logger.cpp
/// @brief Logger implementation

#include "core/logger.hpp"

#include <iomanip>

namespace drifter {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
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

void Logger::log(LogLevel level, const char* func, const std::string& message) {
    if (level < min_level_)
        return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto& out = (level >= LogLevel::WARNING) ? std::cerr : std::cout;
    out << "[" << std::left << std::setw(context_width_) << func << "] "
        << "[" << std::left << std::setw(level_width_) << level_string(level) << "] " << message << "\n";
}

} // namespace drifter
