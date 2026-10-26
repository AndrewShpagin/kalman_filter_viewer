#include "flight_velocity.hpp"
#include "flight_csv_io.hpp"

namespace fs = std::filesystem;
namespace fv = flight_velocity;
namespace {
using namespace flight_csv;

struct Options : InputOptions { fv::Parameters parameters; };

void help() {
    std::cout << R"(Usage: velocity_filter_csv input.csv [options]
  --export PATH                 Default: result_velocity_cpp.csv beside input
  --dt SECONDS                  Fixed interval (default 0.05, matches Python)
  --time-column NAME            Derive dt from strictly increasing timestamps
  --dt-column NAME              Read elapsed time from a column
  --time-scale FACTOR           Time/dt column conversion to seconds (ms: 0.001)
  --fade-in SECONDS             Oldest fade (6)
  --flat SECONDS                Middle plateau (6)
  --fade-out SECONDS            Newest fade (6)
  --vision-delay SECONDS        Vision age relative to IMU (0); exact time shift
  --lead SECONDS                Forward position horizon (1)
  --coord-scale FACTOR          Vision position multiplier (1)
  --bias-warmup SECONDS         Minimum bias information time (6; 0 disables)
  --bias-memory SECONDS         Exponential bias memory (60; 0 = whole flight)
  --bias-x0 VALUE --bias-y0 VALUE  Initial BODY bias, m/s^2 (0,0)
  --vx0 VALUE --vy0 VALUE       Initial NED velocity, m/s (0,0)
  --no-learn-bias               Hold initial body bias fixed
  --quat-direction MODE        auto | body-to-ned | ned-to-body
  --yaw-unit UNIT              auto | deg | rad
  --help

CSV separator: ';'. Same columns and quaternion diagnostics as Python viewer.
Acceleration is gravity-compensated NED, BEFORE this estimator's bias removal.
Row 0 initializes: time_s=dt_s=0; no integration. Height is exported only.
The class requires neither height nor yaw. The runner uses yaw for diagnostics.
Variable-dt averages use w*dt. A partially overlapping oldest interval is
clipped, with linear visual-position interpolation. Fades use interval midpoints.
)";
}

Options parse(const std::vector<std::string>& args) {
    Options o;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg == "--no-learn-bias") { o.parameters.learn_bias = false; continue; }
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
        else if (arg == "--quat-direction") o.direction = value;
        else if (arg == "--yaw-unit") o.yaw_unit = value;
        else if (arg == "--dt") { o.dt = realNumber(value, arg); o.dt_explicit = true; }
        else if (arg == "--time-scale") { o.time_scale = realNumber(value, arg); o.scale_explicit = true; }
        else if (arg == "--fade-in") o.parameters.fade_in = realNumber(value, arg);
        else if (arg == "--flat") o.parameters.flat = realNumber(value, arg);
        else if (arg == "--fade-out") o.parameters.fade_out = realNumber(value, arg);
        else if (arg == "--vision-delay") o.parameters.vision_delay = realNumber(value, arg);
        else if (arg == "--lead") o.parameters.lead = realNumber(value, arg);
        else if (arg == "--coord-scale") o.parameters.coordinate_scale = realNumber(value, arg);
        else if (arg == "--bias-warmup") o.parameters.bias_warmup = realNumber(value, arg);
        else if (arg == "--bias-memory") o.parameters.bias_memory = realNumber(value, arg);
        else if (arg == "--bias-x0") o.parameters.initial_bias_body[0] = realNumber(value, arg);
        else if (arg == "--bias-y0") o.parameters.initial_bias_body[1] = realNumber(value, arg);
        else if (arg == "--vx0") o.parameters.initial_velocity[0] = realNumber(value, arg);
        else if (arg == "--vy0") o.parameters.initial_velocity[1] = realNumber(value, arg);
        else throw std::runtime_error("Unknown option: " + arg);
    }
    if (o.input.empty()) throw std::runtime_error("An input CSV is required; see --help");
    if (o.output.empty()) o.output = o.input.parent_path() / "result_velocity_cpp.csv";
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
    fv::FlightVelocityEstimator validate(o.parameters);
    if (fs::weakly_canonical(o.input) == fs::weakly_canonical(o.output) ||
        (fs::exists(o.output) && fs::equivalent(o.input, o.output)))
        throw std::runtime_error("Input and output must be different files");
    return o;
}

void write(const Options& o, const std::vector<Row>& rows,
           const std::vector<fv::Estimate>& states, const Attitude& attitude) {
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
    out << "time_s;mheight;dt;fade_in;flat;fade_out;lead;coord_scale;window_span_s;window_complete;weight_sum;learn_bias;bias_x0;bias_y0;bias_warmup;bias_memory;vx0;vy0";
    for (const char* prefix : {"vis", "vis_scaled", "estimated_speed", "forward_position", "visual_interval_speed",
                              "mean_visual_speed", "mean_inertial_speed", "window_integral_acc", "velocity_offset",
                              "acc_raw", "acc_used", "integral_acc_raw", "bias_body_used", "bias_body_estimated",
                              "bias_ned_used", "bias_ned_estimated", "bias_fit_rhs"})
        out << ';' << prefix << "_x;" << prefix << "_y";
    out << ";niacc.x;niacc.y";
    for (const char* axis : {"x", "y", "z", "w"}) out << ";quat_" << axis << "_normalized_input";
    out << ";yaw_csv_deg;yaw_quat_deg;yaw_error_deg;bias_fit_normal_xx;bias_fit_normal_xy;bias_fit_normal_yy";
    out << ";dt_s;weighted_duration_s;bias_response_xx;bias_response_xy;bias_response_yx;bias_response_yy";
    out << ";vision_delay;vision_time_s;alignment_ready;bias_fit_dt_s;delayed_speed_x;delayed_speed_y;prediction_delta_v_x;prediction_delta_v_y\n";
    const auto& p = o.parameters;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& s = states[i];
        const double dt = o.time_column.empty() && o.dt_column.empty() ? o.dt : s.dt_s;
        out << s.time_s << ';' << rows[i].v[2] << ';' << dt << ';' << p.fade_in << ';' << p.flat << ';' << p.fade_out
            << ';' << p.lead << ';' << p.coordinate_scale << ';' << s.window_span_s << ';' << s.window_complete
            << ';' << s.weight_sum << ';' << p.learn_bias << ';' << p.initial_bias_body[0] << ';' << p.initial_bias_body[1]
            << ';' << p.bias_warmup << ';' << p.bias_memory << ';' << p.initial_velocity[0] << ';' << p.initial_velocity[1];
        fv::Vec2 visual_speed = p.initial_velocity;
        if (i > 0) for (int j = 0; j < 2; ++j) {
            // Diagnostic only: the estimator itself accumulates displacements
            // without forming these potentially noisy individual derivatives.
            visual_speed[j] = (s.measured_position[j]-states[i-1].measured_position[j])/s.dt_s;
            if (!std::isfinite(visual_speed[j])) throw std::runtime_error("Visual speed diagnostic overflow");
        }
        for (const auto& v : {s.raw_position, s.measured_position, s.velocity, s.forward_position, visual_speed,
                             s.mean_visual_velocity, s.mean_integrated_velocity, s.window_integral, s.normalization_offset,
                             s.raw_acceleration, s.corrected_acceleration, s.integral_raw, s.bias_body_used, s.bias_body,
                             s.bias_ned_used, s.bias_ned, s.bias_rhs, s.velocity}) out << ';' << v[0] << ';' << v[1];
        out << ';' << s.quaternion.x << ';' << s.quaternion.y << ';' << s.quaternion.z << ';' << s.quaternion.w;
        const auto r = fk::rotationMatrix(rows[i].quaternion(), attitude.direction);
        const double yaw = std::atan2(r[1][0], r[0][0])*180/pi;
        const double input_yaw = rows[i].v[9]*(attitude.yaw_radians ? 180/pi : 1);
        out << ';' << input_yaw << ';' << yaw << ';' << wrap(yaw-input_yaw);
        for (double value : s.bias_normal) out << ';' << value;
        out << ';' << s.dt_s << ';' << s.weighted_duration_s;
        for (const auto& row : s.bias_response) for (double x : row) out << ';' << x;
        out << ';' << p.vision_delay << ';' << s.vision_time_s << ';' << s.alignment_ready << ';' << s.bias_fit_dt_s;
        for (const auto& v : {s.delayed_velocity, s.prediction_delta_velocity}) for (double x : v) out << ';' << x;
        out << '\n';
    }
    out.close();
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), o.output.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot replace output CSV (Windows error " + std::to_string(GetLastError()) + ")");
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
        o.parameters.quaternion_direction = attitude.direction == fk::QuaternionDirection::BodyToNed
            ? fv::QuaternionDirection::BodyToNed : fv::QuaternionDirection::NedToBody;
        fv::FlightVelocityEstimator estimator(o.parameters);
        std::vector<fv::Estimate> states;
        states.reserve(rows.size());
        for (const auto& row : rows) {
            try {
                states.push_back(estimator.update({row.v[3], row.v[4]},
                    {row.v[5], row.v[6], row.v[7], row.v[8]}, {row.v[0], row.v[1]}, row.dt));
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
