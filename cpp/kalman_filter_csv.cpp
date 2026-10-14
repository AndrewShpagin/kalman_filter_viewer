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

namespace fs = std::filesystem;
namespace fk = flight_kalman;
namespace {
constexpr double pi = 3.14159265358979323846;
constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct Options {
    fs::path input, output;
    fk::Parameters parameters;
    fk::Vec2 initial_velocity{};
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

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

long double number(std::string text, const std::string& label) {
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

double realNumber(const std::string& text, const std::string& label) {
    const double value = static_cast<double>(number(text, label));
    if (!std::isfinite(value)) throw std::runtime_error("Number outside double range: " + label);
    return value;
}

void help() {
    std::cout << R"(Usage: kalman_filter_csv input.csv [options]
  --export PATH                 Output (default: result_cpp.csv next to input)
  --dt SECONDS                  Fixed interval (default: 0.05, matches Python)
  --time-column NAME            Derive dt from strictly increasing timestamps
  --dt-column NAME              Read elapsed time from this column
  --time-scale FACTOR           Convert time/dt column to seconds (ms: 0.001)
  --sigma-acc VALUE             Per-sample acceleration sigma, m/s^2 (1)
  --min-pos-sigma VALUE         Position sigma floor, m (0.001)
  --k-height VALUE              Height error fraction (0.1 = 10%)
  --initial-vel-sigma VALUE     Initial speed sigma, m/s (1)
  --coord-scale VALUE           Vision position multiplier (1)
  --vx0 VALUE --vy0 VALUE       Initial NED velocity, m/s (0,0)
  --sigma-acc-column NAME       Per-row acceleration sigma override
  --position-sigma-column NAME  Per-row position sigma override (metres, >0)
  --quat-direction MODE        auto | body-to-ned | ned-to-body
  --yaw-unit UNIT              auto | deg | rad
  --help

CSV uses ';'. Required columns match the Python viewer, including yaw for
frame diagnostics. The header-only class itself needs no yaw measurement.
Row 0 initializes state: output time_s=0, dt_s=0; acceleration is not integrated.
Choose only one time mode. No guessing of timestamp units or skipped bad rows.
Example: kalman_filter_csv flight.csv --time-column timestamp --time-scale 0.001
)";
}

Options parse(const std::vector<std::string>& args) {
    Options o;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg.rfind("--", 0) != 0) {
            if (!o.input.empty()) throw std::runtime_error("Only one input CSV is allowed");
            o.input = fs::u8path(arg);
            continue;
        }
        if (++i == args.size()) throw std::runtime_error("Missing value for " + arg);
        const auto& value = args[i];
        if (arg == "--export" || arg == "--output") o.output = fs::u8path(value);
        else if (arg == "--time-column") o.time_column = value;
        else if (arg == "--dt-column") o.dt_column = value;
        else if (arg == "--sigma-acc-column") o.sigma_acc_column = value;
        else if (arg == "--position-sigma-column") o.position_sigma_column = value;
        else if (arg == "--quat-direction") o.direction = value;
        else if (arg == "--yaw-unit") o.yaw_unit = value;
        else if (arg == "--dt") { o.dt = realNumber(value, arg); o.dt_explicit = true; }
        else if (arg == "--time-scale") { o.time_scale = realNumber(value, arg); o.scale_explicit = true; }
        else if (arg == "--sigma-acc") o.parameters.noise.acceleration_sigma = realNumber(value, arg);
        else if (arg == "--min-pos-sigma") o.parameters.noise.min_position_sigma = realNumber(value, arg);
        else if (arg == "--k-height") o.parameters.noise.height_error_fraction = realNumber(value, arg);
        else if (arg == "--initial-vel-sigma") o.parameters.noise.initial_velocity_sigma = realNumber(value, arg);
        else if (arg == "--coord-scale") o.parameters.coordinate_scale = realNumber(value, arg);
        else if (arg == "--vx0") o.initial_velocity[0] = realNumber(value, arg);
        else if (arg == "--vy0") o.initial_velocity[1] = realNumber(value, arg);
        else throw std::runtime_error("Unknown option: " + arg);
    }
    if (o.input.empty()) throw std::runtime_error("An input CSV is required; see --help");
    if (o.output.empty()) o.output = o.input.parent_path() / "result_cpp.csv";
    if (o.dt <= 0 || o.time_scale <= 0) throw std::runtime_error("dt and time-scale must be >0");
    const int column_modes = (!o.time_column.empty()) + (!o.dt_column.empty());
    if (column_modes > 1 || (column_modes && o.dt_explicit))
        throw std::runtime_error("Choose only one of --dt, --time-column, --dt-column");
    if (!column_modes && o.scale_explicit)
        throw std::runtime_error("--time-scale requires --time-column or --dt-column");
    if (o.direction != "auto" && o.direction != "body-to-ned" && o.direction != "ned-to-body")
        throw std::runtime_error("Invalid --quat-direction");
    if (o.yaw_unit != "auto" && o.yaw_unit != "deg" && o.yaw_unit != "rad")
        throw std::runtime_error("Invalid --yaw-unit");
    fk::FlightKalmanFilter validate(o.parameters);
    if (fs::weakly_canonical(o.input) == fs::weakly_canonical(o.output) ||
        (fs::exists(o.output) && fs::equivalent(o.input, o.output)))
        throw std::runtime_error("Input and output must be different files");
    return o;
}

// Semicolon CSV with quoted fields, escaped quotes, CRLF and multiline fields.
bool record(std::istream& in, std::vector<std::string>& fields, std::size_t& line) {
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

std::size_t column(const std::vector<std::string>& header,
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

std::vector<Row> read(const Options& o) {
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
            row.sigma_acc = acc_index == missing ? o.parameters.noise.acceleration_sigma :
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

double wrap(double degrees) {
    double v = std::fmod(degrees + 180, 360);
    if (v < 0) v += 360;
    return v - 180;
}

struct Attitude {
    fk::QuaternionDirection direction;
    bool yaw_radians;
    double yaw_rms, raw_rms;
};

Attitude checkAttitude(const std::vector<Row>& rows, const Options& o) {
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

void write(const Options& o, const std::vector<Row>& rows,
           const std::vector<fk::Estimate>& states, const Attitude& attitude) {
    auto temporary = o.output;
    temporary += "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 "." + std::to_string(std::random_device{}()) + ".tmp";
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ec; fs::remove(path, ec); }
    } cleanup{temporary};
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(temporary, std::ios::binary);
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "time_s;mheight;sigma_position;coord_scale";
    for (const char* prefix : {"vis", "vis_scaled", "estimated_position", "estimated_speed", "acc_raw",
                              "bias_ned_used", "acc_used", "bias_ned_estimated", "integral_acc_raw",
                              "bias_body_estimated", "bias_body_used", "bias_fit_rhs"})
        out << ';' << prefix << "_x;" << prefix << "_y";
    for (const char* axis : {"x", "y", "z", "w"}) out << ";quat_" << axis << "_normalized_input";
    out << ";yaw_csv_deg;yaw_quat_deg;yaw_error_deg;bias_fit_normal_xx;bias_fit_normal_xy;bias_fit_normal_yy";
    for (const char* axis : {"x", "y"})
        for (const char* label : {"Pxx", "Pxv", "Pvv", "K_position", "K_velocity"}) out << ';' << axis << '_' << label;
    out << ";dt_s;sigma_acc\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& s = states[i];
        out << s.time_s << ';' << s.height << ';' << s.sigma_position << ';' << s.coordinate_scale;
        for (const auto& pair : {s.raw_position, s.measured_position, s.position, s.velocity, s.raw_acceleration,
                                s.bias_ned_used, s.corrected_acceleration, s.bias_ned, s.integral_raw,
                                s.bias_body, s.bias_body_used, s.bias_rhs}) out << ';' << pair[0] << ';' << pair[1];
        out << ';' << s.quaternion.x << ';' << s.quaternion.y << ';' << s.quaternion.z << ';' << s.quaternion.w;
        const auto r = fk::rotationMatrix(rows[i].quaternion(), attitude.direction);
        const double yaw = std::atan2(r[1][0], r[0][0])*180/pi;
        const double input_yaw = rows[i].v[9]*(attitude.yaw_radians ? 180/pi : 1);
        out << ';' << input_yaw << ';' << yaw << ';' << wrap(yaw-input_yaw);
        for (double value : s.bias_normal) out << ';' << value;
        for (int j = 0; j < 2; ++j) {
            const auto& p = s.covariance[j];
            out << ';' << p.xx << ';' << p.xv << ';' << p.vv << ';' << s.gain[j].position << ';' << s.gain[j].velocity;
        }
        out << ';' << s.dt_s << ';' << rows[i].sigma_acc << '\n';
    }
    out.close();
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), o.output.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot replace output CSV (Windows error " + std::to_string(GetLastError()) + "); close it in other applications");
#else
    fs::rename(temporary, o.output);
#endif
}

int run(const std::vector<std::string>& args) {
    try {
        if (std::find(args.begin()+1, args.end(), "--help") != args.end()) { help(); return 0; }
        auto o = parse(args);
        const auto rows = read(o);
        const auto attitude = checkAttitude(rows, o);
        o.parameters.quaternion_direction = attitude.direction;
        fk::FlightKalmanFilter filter(o.parameters);
        filter.reset(o.initial_velocity);
        std::vector<fk::Estimate> states;
        states.reserve(rows.size());
        for (const auto& row : rows) {
            try {
                auto noise = filter.parameters().noise;
                noise.acceleration_sigma = row.sigma_acc;
                filter.setNoiseParameters(noise);
                states.push_back(filter.update(fk::Measurement{
                    {row.v[3], row.v[4]}, row.quaternion(), {row.v[0], row.v[1]}, row.v[2], row.dt, row.position_sigma}));
            } catch (const std::exception& e) {
                throw std::runtime_error("CSV line " + std::to_string(row.line) + ": " + e.what());
            }
        }
        write(o, rows, states, attitude);
        std::cout << std::setprecision(10) << "Quaternion: "
                  << (attitude.direction == fk::QuaternionDirection::BodyToNed ? "body-to-ned" : "ned-to-body")
                  << "; yaw: " << (attitude.yaw_radians ? "rad" : "deg")
                  << "; yaw RMS: " << attitude.yaw_rms << " deg\n";
        if (std::isfinite(attitude.raw_rms)) std::cout << "Raw acceleration RMS: " << attitude.raw_rms << " m/s^2\n";
        std::cout << rows.size() << " samples; " << states.back().time_s << " s; time mode: "
                  << (!o.time_column.empty() ? "timestamp differences" : !o.dt_column.empty() ? "dt column" : "fixed dt")
                  << "\nFinal BODY bias: " << states.back().bias_body[0] << ", " << states.back().bias_body[1]
                  << "\nExported: " << o.output.u8string() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        if (n <= 0) { std::cerr << "Cannot decode command line\n"; return 1; }
        std::string s(static_cast<std::size_t>(n), '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, s.data(), n, nullptr, nullptr);
        s.pop_back();
        args.push_back(std::move(s));
    }
    return run(args);
}
#else
int main(int argc, char** argv) { return run(std::vector<std::string>(argv, argv+argc)); }
#endif
