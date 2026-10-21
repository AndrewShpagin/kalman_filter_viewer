#include "flight_csv_io.hpp"

namespace fs = std::filesystem;
namespace fk = flight_kalman;
namespace {
using namespace flight_csv;
struct Options : flight_csv::InputOptions {
    fk::Parameters parameters;
    fk::Vec2 initial_velocity{};
};

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
        const auto rows = read(o, o.parameters.noise.acceleration_sigma);
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
