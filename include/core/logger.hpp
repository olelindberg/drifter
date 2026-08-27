#pragma once

/// @file logger.hpp
/// @brief Logging utilities with file/line/function context

#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace drifter {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
  public:
    static Logger& instance();

    void set_level(LogLevel level) { min_level_ = level; }
    LogLevel level() const { return min_level_; }

    void log(LogLevel level, const char* file, int line, const char* func, const std::string& message);

  private:
    Logger() = default;
    LogLevel min_level_ = LogLevel::INFO; // DEBUG disabled by default
    std::mutex mutex_;

    static std::string_view basename(const char* path);
    static const char* level_string(LogLevel level);

    static constexpr int context_width_ = 40; // [file:line func] column width
    static constexpr int level_width_   = 7;  // [LEVEL] column width
};

} // namespace drifter

// Macros - short-circuit to avoid building message when level disabled
#define LOG_DEBUG(msg)                                                                                                 \
    do {                                                                                                               \
        if (drifter::Logger::instance().level() <= drifter::LogLevel::DEBUG) {                                         \
            std::ostringstream _ss;                                                                                    \
            _ss << msg;                                                                                                \
            drifter::Logger::instance().log(drifter::LogLevel::DEBUG, __FILE__, __LINE__, __func__, _ss.str());        \
        }                                                                                                              \
    } while (0)

#define LOG_INFO(msg)                                                                                                  \
    do {                                                                                                               \
        if (drifter::Logger::instance().level() <= drifter::LogLevel::INFO) {                                          \
            std::ostringstream _ss;                                                                                    \
            _ss << msg;                                                                                                \
            drifter::Logger::instance().log(drifter::LogLevel::INFO, __FILE__, __LINE__, __func__, _ss.str());         \
        }                                                                                                              \
    } while (0)

#define LOG_WARNING(msg)                                                                                               \
    do {                                                                                                               \
        if (drifter::Logger::instance().level() <= drifter::LogLevel::WARNING) {                                       \
            std::ostringstream _ss;                                                                                    \
            _ss << msg;                                                                                                \
            drifter::Logger::instance().log(drifter::LogLevel::WARNING, __FILE__, __LINE__, __func__, _ss.str());      \
        }                                                                                                              \
    } while (0)

#define LOG_ERROR(msg)                                                                                                 \
    do {                                                                                                               \
        std::ostringstream _ss;                                                                                        \
        _ss << msg;                                                                                                    \
        drifter::Logger::instance().log(drifter::LogLevel::ERROR, __FILE__, __LINE__, __func__, _ss.str());            \
    } while (0)
