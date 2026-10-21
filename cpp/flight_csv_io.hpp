#pragma once

// Shared CLI-only CSV reader and offline frame diagnostics.
#include "flight_kalman.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace flight_csv {
namespace fs = std::filesystem;
namespace fk = flight_kalman;
constexpr double pi = 3.14159265358979323846;
constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct InputOptions {
    fs::path input, output;
    double dt = 0.05, time_scale = 1;
    bool dt_explicit = false, scale_explicit = false;
    std::string time_column, dt_column, sigma_acc_column, position_sigma_column;
    std::string direction = "auto", yaw_unit = "auto";
};

struct Row {
    std::array<double, 15> v{}; // same order as Python read_flight()
    double dt = 0, sigma_acc = 1;
    std::optional<double> position_sigma;
    std::size_t line = 0;
    fk::Quaternion quaternion() const { return {v[5], v[6], v[7], v[8]}; }
};

inline std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

inline long double number(std::string text, const std::string& label) {
    text = trim(text);
    std::replace(text.begin(), text.end(), ',', '.');
    std::istringstream in(text);
    in.imbue(std::locale::classic());
    long double value = 0;
    if (!(in >> value) || !std::isfinite(value))
        throw std::runtime_error("Invalid finite number: " + label);
    in >> std::ws;
    if (!in.eof()) throw std::runtime_error("Invalid number: " + label);
    return value;
}

inline double realNumber(const std::string& text, const std::string& label) {
    const double value = static_cast<double>(number(text, label));
    if (!std::isfinite(value)) throw std::runtime_error("Number outside double range: " + label);
    return value;
}

// Semicolon CSV with quoted fields, escaped quotes, CRLF and multiline fields.
inline bool record(std::istream& in, std::vector<std::string>& fields, std::size_t& line) {
    fields.clear();
    std::string field;
    bool quoted = false, closed = false, consumed = false;
    char c;
    while (in.get(c)) {
        consumed = true;
        if (c == '\n') ++line;
        if (quoted) {
            if (c == '"') {
                if (in.peek() == '"') { in.get(c); field += '"'; }
                else { quoted = false; closed = true; }
            } else field += c;
        } else if (c == ';') {
            fields.push_back(field); field.clear(); closed = false;
        } else if (c == '\n' || c == '\r') {
            if (c == '\r' && in.peek() == '\n') { in.get(c); ++line; }
            else if (c == '\r') ++line;
            fields.push_back(field);
            return true;
        } else if (c == '"' && field.empty() && !closed) quoted = true;
        else if (closed && c != ' ' && c != '\t')
            throw std::runtime_error("Characters after quoted CSV field at line " + std::to_string(line));
        else field += c;
    }
    if (in.bad()) throw std::runtime_error("CSV read failed");
    if (quoted) throw std::runtime_error("Unclosed CSV quote at line " + std::to_string(line));
    if (consumed) fields.push_back(field);
    return consumed;
}

inline std::size_t column(const std::vector<std::string>& header,
                   std::initializer_list<std::string> names, bool optional = false) {
    for (const auto& name : names) {
        auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) continue;
        if (std::count(header.begin(), header.end(), name) != 1)
            throw std::runtime_error("Duplicate column: " + name);
        return static_cast<std::size_t>(it - header.begin());
    }
    if (!optional) throw std::runtime_error("Missing column: " + *names.begin());
    return missing;
}

inline std::vector<Row> read(const InputOptions& o, double default_acceleration_sigma = 1) {
    std::ifstream in(o.input, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input: " + o.input.u8string());
    // Strip BOM before parsing, including when the first header cell is quoted.
    char bom[3]{};
    in.read(bom, 3);
    if (std::string(bom, 3) != "\xEF\xBB\xBF") { in.clear(); in.seekg(0); }
    std::vector<std::string> header, fields;
    std::size_t line = 1;
    if (!record(in, header, line)) throw std::runtime_error("Empty CSV");
    for (auto& s : header) s = trim(s);
    const std::array<std::size_t, 15> indices{
        column(header,{"vis_x"}), column(header,{"vis_y"}), column(header,{"mheight"}),
        column(header,{"acc_x","acc.x"}), column(header,{"acc_y","acc.y"}),
        column(header,{"quat.x","quat_x"}), column(header,{"quat.y","quat_y"}),
        column(header,{"quat.z","quat_z"}), column(header,{"quat.w","quat_w"}), column(header,{"yaw"}),
        column(header,{"rawacc.x","rawacc_x"},true), column(header,{"rawacc.y","rawacc_y"},true),
        column(header,{"rawacc.z","rawacc_z"},true), column(header,{"acc.z","acc_z"},true),
        column(header,{"G"},true)};
    const auto time_index = o.time_column.empty() ? missing : column(header,{o.time_column});
    const auto dt_index = o.dt_column.empty() ? missing : column(header,{o.dt_column});
    const auto acc_index = o.sigma_acc_column.empty() ? missing : column(header,{o.sigma_acc_column});
    const auto pos_index = o.position_sigma_column.empty() ? missing : column(header,{o.position_sigma_column});
    std::vector<Row> rows;
    long double previous_time = 0;
    while (true) {
        const auto row_line = line;
        if (!record(in, fields, line)) break;
        if (std::all_of(fields.begin(), fields.end(), [](const auto& s) { return trim(s).empty(); })) continue;
        auto get = [&](std::size_t i) -> const std::string& {
            if (i >= fields.size()) throw std::runtime_error("Missing value at CSV line " + std::to_string(row_line));
            return fields[i];
        };
        Row row;
        row.line = row_line;
        try {
            for (std::size_t j = 0; j < indices.size(); ++j) {
                row.v[j] = nan;
                if (indices[j] == missing) continue;
                try { row.v[j] = realNumber(get(indices[j]), header[indices[j]]); }
                catch (const std::runtime_error&) { if (j < 10) throw; }
            }
            row.dt = rows.empty() ? 0 : o.dt;
            if (time_index != missing) {
                const auto t = number(get(time_index), o.time_column);
                if (!rows.empty()) row.dt = static_cast<double>((t-previous_time)*o.time_scale);
                previous_time = t;
            } else if (dt_index != missing) {
                const auto& cell = get(dt_index);
                const double value = rows.empty() && trim(cell).empty() ? 0 : realNumber(cell, o.dt_column);
                if (value < 0) throw std::runtime_error("dt column must be nonnegative on row 0");
                row.dt = rows.empty() ? 0 : value*o.time_scale;
            }
            if (!rows.empty() && (!std::isfinite(row.dt) || row.dt <= 0))
                throw std::runtime_error("Elapsed time must be finite and >0 (timestamps must increase)");
            row.sigma_acc = acc_index == missing ? default_acceleration_sigma :
                                                   realNumber(get(acc_index), o.sigma_acc_column);
            if (pos_index != missing) row.position_sigma = realNumber(get(pos_index), o.position_sigma_column);
            rows.push_back(row);
        } catch (const std::exception& e) {
            throw std::runtime_error("CSV line " + std::to_string(row_line) + ": " + e.what());
        }
    }
    if (rows.size() < 2) throw std::runtime_error("At least two data rows are required");
    return rows;
}

inline double wrap(double degrees) {
    double v = std::fmod(degrees + 180, 360);
    if (v < 0) v += 360;
    return v - 180;
}

struct Attitude {
    fk::QuaternionDirection direction;
    bool yaw_radians;
    double yaw_rms, raw_rms;
};

inline Attitude checkAttitude(const std::vector<Row>& rows, const InputOptions& o) {
    std::array<std::array<double, 2>, 2> yaw_sum{};
    std::array<double, 2> raw_sum{}, yaw_score{}, raw_score{};
    std::array<int, 2> unit{};
    std::size_t raw_count = 0;
    for (const auto& row : rows) {
        const bool raw_valid = std::isfinite(row.v[10]) && std::isfinite(row.v[11]) && std::isfinite(row.v[12]);
        if (raw_valid) ++raw_count;
        for (int d = 0; d < 2; ++d) {
            const auto r = fk::rotationMatrix(row.quaternion(), d == 0 ? fk::QuaternionDirection::BodyToNed : fk::QuaternionDirection::NedToBody);
            const double yaw = std::atan2(r[1][0], r[0][0])*180/pi;
            for (int u = 0; u < 2; ++u) {
                const double error = wrap(yaw - row.v[9]*(u == 0 ? 1 : 180/pi));
                yaw_sum[d][u] += error*error;
            }
            if (raw_valid) for (int j = 0; j < 2; ++j) {
                const double error = r[j][0]*row.v[10]+r[j][1]*row.v[11]+r[j][2]*row.v[12]-row.v[3+j];
                raw_sum[d] += error*error;
            }
        }
    }
    for (int d = 0; d < 2; ++d) {
        unit[d] = o.yaw_unit == "auto" ? (yaw_sum[d][1] < yaw_sum[d][0] ? 1 : 0) : (o.yaw_unit == "rad" ? 1 : 0);
        yaw_score[d] = std::sqrt(yaw_sum[d][unit[d]] / static_cast<double>(rows.size()));
        raw_score[d] = raw_count >= 2 ? std::sqrt(raw_sum[d] / (2*static_cast<double>(raw_count))) : nan;
    }
    int direction = 0;
    if (o.direction == "auto") {
        const int best_raw = raw_score[1] < raw_score[0] ? 1 : 0;
        const double lo = raw_score[best_raw], hi = raw_score[1-best_raw];
        if (raw_count >= 2 && lo < 0.25 && hi > std::max(2*lo, lo+0.001)) direction = best_raw;
        else {
            direction = yaw_score[1] < yaw_score[0] ? 1 : 0;
            if (yaw_score[1-direction]-yaw_score[direction] < std::max(0.05, 0.2*yaw_score[direction])) {
                direction = 0;
                std::cerr << "Frame check: quaternion direction is weakly identified; assuming body-to-ned.\n";
            }
        }
    } else direction = o.direction == "ned-to-body" ? 1 : 0;
    if (yaw_score[direction] > 2) std::cerr << "Frame check: quaternion/yaw RMS mismatch exceeds 2 degrees.\n";
    if (raw_score[direction] > 0.25) std::cerr << "Frame check: rotated raw acceleration differs from NED acceleration.\n";
    return {direction == 0 ? fk::QuaternionDirection::BodyToNed : fk::QuaternionDirection::NedToBody,
            unit[direction] == 1, yaw_score[direction], raw_score[direction]};
}

} // namespace flight_csv
