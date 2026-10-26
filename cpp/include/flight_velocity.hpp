#pragma once

// Standalone C++17 header; no Eigen, Python or Kalman dependencies.
// Window-normalized horizontal NED velocity, matching velocity_flight_viewer.py.
// Acceleration is gravity-compensated NED acceleration BEFORE our bias removal.
// Quaternion: x,y,z,w. Body-Z bias is assumed zero; full attitude is used.
// Each update supplies position at the interval END and acceleration/attitude
// held over that interval; vision is timestamped current_time - vision_delay.
// The first update initializes; its dt is not integrated.
//
// Детальний розбір із виведенням формул: ../VELOCITY_ALGORITHM_UK.md.
// Це віконний оцінювач швидкості + окремий МНК для біаса, не фільтр Кальмана.
// Послідовність: старий BODY-біас -> швидкість -> новий BODY-біас -> forward.
// Ні 20 Гц, ні 60 Гц тут не зашито: кожен наступний update має власний dt.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <stdexcept>

namespace flight_velocity {

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;
using Rotation = std::array<std::array<double, 3>, 3>;
struct Quaternion { double x = 0, y = 0, z = 0, w = 1; };
enum class QuaternionDirection { BodyToNed, NedToBody };

inline Quaternion normalized(Quaternion q) {
    // Масштаб кватерніона не повинен змінювати обертання. Вкладені hypot
    // стійкіші до переповнення, ніж sqrt(x*x+y*y+z*z+w*w).
    const double n = std::hypot(std::hypot(q.x, q.y), std::hypot(q.z, q.w));
    if (!std::isfinite(n) || n < 1e-8)
        throw std::invalid_argument("Quaternion has a zero/non-finite norm");
    return {q.x/n, q.y/n, q.z/n, q.w/n};
}

inline Rotation rotationMatrix(Quaternion input,
                               QuaternionDirection direction = QuaternionDirection::BodyToNed) {
    const auto q = normalized(input);
    const auto x = q.x, y = q.y, z = q.z, w = q.w;
    Rotation r{{
        {{1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)}},
        {{2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)}},
        {{2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)}}
    }};
    // Завжди повертаємо BODY -> NED. Для зворотної вхідної конвенції
    // обернена ортогональна матриця дорівнює транспонованій.
    if (direction == QuaternionDirection::NedToBody)
        for (int i = 0; i < 3; ++i)
            for (int j = i+1; j < 3; ++j) std::swap(r[i][j], r[j][i]);
    return r;
}

struct Parameters {
    double fade_in = 6, flat = 6, fade_out = 6; // seconds, oldest -> newest
    double lead = 1;                          // forward_position = scaled vision + lead*v
    double vision_delay = 0;                  // vision capture age relative to IMU, seconds
    double coordinate_scale = 1;
    double bias_warmup = 6;                   // initial information floor, seconds (not a delay)
    double bias_memory = 60;                  // exponential memory, seconds; 0 = whole flight
    Vec2 initial_velocity{};                 // NED, m/s
    Vec2 initial_bias_body{};                // BODY X/Y, m/s^2
    bool learn_bias = true;
    QuaternionDirection quaternion_direction = QuaternionDirection::BodyToNed;
    double windowSeconds() const noexcept { return fade_in + flat + fade_out; }
};

struct Measurement {
    // Це прискорення переносу в NED, а не specific force сирого IMU:
    // гравітацію вже прибрано, наш BODY-біас ще НЕ віднято.
    // Одне значення представляє весь інтервал (previous_time, current_time].
    // Vision належить current_time - vision_delay; IMU — current_time.
    // Клас не збирає високочастотні IMU-підвимірювання.
    Vec2 acceleration_ned{};
    Quaternion quaternion{};
    Vec2 visual_position{};
    double dt = 0;                           // seconds, >0 after initialization
};

struct Estimate {
    // time_s накопичує фактичні dt після першого виклику. window_span_s —
    // наявна тривалість історії, яка на старті менша за налаштоване вікно.
    double time_s = 0, dt_s = 0, window_span_s = 0;
    double vision_time_s = 0, bias_fit_dt_s = 0;
    bool alignment_ready = false;            // IMU history reaches the vision epoch
    Vec2 delayed_velocity{}, prediction_delta_velocity{};
    // Для усереднення використовуємо weighted_duration_s = sum(w*h),
    // не weight_sum = sum(w): кадр за 50 мс має більшу вагу, ніж за 16.7 мс.
    double weight_sum = 0, weighted_duration_s = 0;
    double bias_prior_information_s = 0;     // decaying startup floor
    bool window_complete = false;
    // raw_position: вхід; measured_position: вхід*coordinate_scale;
    // forward_position: measured_position + lead*velocity. Усі позиції в м.
    Vec2 raw_position{}, measured_position{}, forward_position{}, velocity{};
    // Два середніх у м/с: візуальні прирости / час та інтеграл прискорення.
    Vec2 mean_visual_velocity{}, mean_integrated_velocity{};
    // delayed_velocity = window_integral + normalization_offset (when ready).
    // velocity = delayed_velocity + prediction_delta_velocity.
    // window_integral починається з НУЛЯ на лівій межі поточного вікна.
    Vec2 window_integral{}, normalization_offset{};
    // corrected_acceleration: поточне прискорення мінус bias_ned_used.
    // integral_raw: інтеграл СИРОГО прискорення від початку всього логу;
    // це лише діагностика, він не використовується для швидкості/біаса.
    Vec2 raw_acceleration{}, corrected_acceleration{}, integral_raw{};
    // *_used: біас попереднього кроку, застосований зараз; bias_body/bias_ned:
    // щойно оцінений біас, який вплине на швидкість наступного update.
    Vec2 bias_body{}, bias_body_used{}, bias_ned{}, bias_ned_used{}, bias_rhs{};
    std::array<double, 3> bias_normal{};       // xx,xy,yy; information, NOT covariance
    Mat2 bias_response{};                    // C: DELAYED velocity response to BODY bias
    // C має одиниці секунд: v_delayed(b + db) = v_delayed(b) - C*db.
    // Це чутливість оцінки, не матриця коваріації/обертання.
    Quaternion quaternion{};                // normalized INPUT convention
};

class FlightVelocityEstimator {
public:
    explicit FlightVelocityEstimator(Parameters parameters = {}) { reset(parameters); }

    const Parameters& parameters() const noexcept { return parameters_; }
    const Estimate& state() const noexcept { return state_; }
    bool initialized() const noexcept { return initialized_; }
    std::size_t historySize() const noexcept { return history_.size(); }

    // Applies on the NEXT update, preserving bias, time and retained history.
    // Growing the window uses available intervals until enough new data arrives.
    // Repeated calls with unchanged values are a no-op (safe to call each frame).
    void setFades(double fade_in, double flat, double fade_out) {
        auto p = parameters_;
        p.fade_in = fade_in; p.flat = flat; p.fade_out = fade_out;
        validate(p);
        if (p.fade_in == parameters_.fade_in && p.flat == parameters_.flat &&
            p.fade_out == parameters_.fade_out) return;
        parameters_ = p;
        window_changed_ = true;
    }

    // forward_time is Parameters::lead, in seconds. Immediately updates the
    // published forward position; velocity, bias and window history are intact.
    void setForwardTime(double forward_time) {
        auto p = parameters_;
        p.lead = forward_time;
        validate(p);
        Vec2 forward = state_.forward_position;
        if (initialized_) for (int j = 0; j < 2; ++j)
            forward[j] = state_.measured_position[j]+forward_time*state_.velocity[j];
        finite(forward);
        parameters_ = p;
        state_.forward_position = forward;
    }

    // Full configuration/reset starts a new flight. Use the setters above to
    // change window/forward time without discarding state. Replay is required
    // to compare a whole flight under different parameters.
    void reset(const Parameters& parameters) {
        validate(parameters);
        parameters_ = parameters;
        reset();
    }
    void reset() noexcept {
        // Починаємо новий відлік. За замовчуванням початкові v і b нульові.
        // Якщо запуск відбувається в русі, initial_velocity треба задати.
        history_.clear();
        time_ = 0;
        initialized_ = false;
        window_changed_ = false;
        state_ = Estimate{};
        state_.velocity = parameters_.initial_velocity;
        state_.delayed_velocity = parameters_.initial_velocity;
        state_.vision_time_s = -parameters_.vision_delay;
        state_.alignment_ready = parameters_.vision_delay == 0;
        state_.bias_body = state_.bias_body_used = parameters_.initial_bias_body;
        state_.bias_prior_information_s = parameters_.bias_warmup;
    }

    Estimate update(Vec2 raw_acceleration_ned, Quaternion quaternion,
                    Vec2 visual_position, double dt) {
        return update(Measurement{raw_acceleration_ned, quaternion, visual_position, dt});
    }

    // Invalid input / numeric overflow throws without changing state or history.
    // O(samples in window + vision_delay) time and memory; deque may allocate. Not thread-safe.
    Estimate update(const Measurement& input) {
        // 1. Перевіряємо вхід до зміни історії. Нульовий dt дозволений лише
        // при ініціалізації; наступні 1/60...1/20 с приймаються без округлення.
        finite(input.acceleration_ned); finite(input.visual_position);
        if (!std::isfinite(input.dt) || input.dt < 0 || (initialized_ && input.dt == 0))
            throw std::invalid_argument("dt must be finite and >0 after initialization");
        Estimate next = state_;
        // next — робоча копія: невдала операція не псує опублікований state_.
        next.quaternion = normalized(input.quaternion);
        const auto r = rotationMatrix(next.quaternion, parameters_.quaternion_direction);
        // M відображає [bx,by,0]_BODY у горизонтальні компоненти NED.
        // Беремо блок повної 3D-матриці: roll/pitch враховані. M не обов'язково
        // ортогональна 2D-матриця; при сильному нахилі напрям може бути слабко
        // спостережуваним. BODY-Z біас модель не оцінює.
        const Mat2 m{{{{r[0][0], r[0][1]}}, {{r[1][0], r[1][1]}}}};
        next.raw_position = input.visual_position;
        next.raw_acceleration = input.acceleration_ned;
        // 2. Беремо біас, отриманий на ПОПЕРЕДНЬОМУ update. Поточної оцінки
        // біаса ще немає: жодної ітерації швидкість <-> біас всередині кроку.
        next.bias_body_used = state_.bias_body;
        next.bias_ned_used = multiply(m, next.bias_body_used);
        for (int j = 0; j < 2; ++j) {
            next.measured_position[j] = input.visual_position[j]*parameters_.coordinate_scale;
            next.corrected_acceleration[j] = input.acceleration_ned[j]-next.bias_ned_used[j];
        }
        if (!initialized_) {
            // Перший кадр задає початкову позицію. Для нього немає попередньої
            // позиції/інтервалу: input.dt не інтегруємо, біас не навчаємо.
            next.mean_visual_velocity = next.normalization_offset = next.velocity;
            finish(next, m);
            state_ = next;
            initialized_ = true;
            window_changed_ = false;
            return state_;
        }

        Estimate previous = state_;
        if (window_changed_ && !history_.empty()) {
            // Evaluate the previous timestamp under the NEW window, with the
            // SAME bias that was used at that timestamp. The public state is
            // retained until this update succeeds. A configuration-induced
            // speed/C jump must not become acceleration or bias information.
            const auto epoch = time_ - parameters_.vision_delay;
            const auto previous_left = windowLeft(epoch);
            windowStatus(previous, previous_left, epoch);
            if (epoch > previous_left) average(previous, previous_left, epoch);
            finite(previous.delayed_velocity);
            for (const auto& row : previous.bias_response) finite(row);
        }
        const long double time = time_ + input.dt;
        // long double зменшує втрату точності при довгому накопиченні часу;
        // на платформах, де він дорівнює double, додаткової точності немає.
        if (!std::isfinite(time) || time <= time_)
            throw std::invalid_argument("Elapsed time is outside representable range");
        next.time_s = static_cast<double>(time);
        next.dt_s = input.dt;
        const auto epoch = time - parameters_.vision_delay;
        next.vision_time_s = static_cast<double>(epoch);
        next.alignment_ready = epoch >= 0;
        next.bias_fit_dt_s = 0;
        Vec2 displacement{};
        // 3. Зберігаємо приріст позиції, а не dp/dt: це дозволяє далі
        // врахувати тривалість без обчислення шумної миттєвої швидкості.
        for (int j = 0; j < 2; ++j) {
            displacement[j] = next.measured_position[j]-state_.measured_position[j];
            next.integral_raw[j] += input.acceleration_ned[j]*input.dt;
        }
        finite(displacement);
        // У пам'яті лишаються СИРІ прискорення і власна M кожного інтервалу.
        // Тому всю історію можна перерахувати з новою оцінкою BODY-біаса.
        history_.push_back({time, input.dt, input.acceleration_ned, displacement, m});
        const auto left = windowLeft(epoch);
        windowStatus(next, left, epoch);
        try {
            // 4. Вікно -> швидкість зі старим біасом. 5. МНК -> новий біас.
            // 6. Випередження використовує вже отриману швидкість, не
            // перераховуючи її повторно з щойно знайденим біасом.
            if (epoch > left) {
                average(next, left, epoch);
                if (parameters_.learn_bias && previous.alignment_ready) {
                    // Обидва кінці швидкості та інтеграл IMU — на одній
                    // фізичній шкалі часу. Край може розрізати IMU-інтервал.
                    const auto delta = integrate(time_ - parameters_.vision_delay, epoch);
                    learn(next, delta, previous);
                    next.bias_fit_dt_s = input.dt;
                }
            } else {
                // До першої спільної епохи не вигадуємо IMU до початку логу
                // і не навчаємо біас. initial_velocity задано для time=0.
                next.delayed_velocity = parameters_.initial_velocity;
                next.mean_visual_velocity = next.normalization_offset = next.delayed_velocity;
                next.mean_integrated_velocity = next.window_integral = {};
                next.bias_response = {};
                next.weight_sum = next.weighted_duration_s = 0;
            }
            const auto prediction = integrate(std::max(0.0L, epoch), time);
            const auto predicted_bias = multiply(prediction.rotation, next.bias_body_used);
            for (int j = 0; j < 2; ++j) {
                next.prediction_delta_velocity[j] = prediction.raw[j]-predicted_bias[j];
                next.velocity[j] = next.delayed_velocity[j]+next.prediction_delta_velocity[j];
            }
            finish(next, m);
        } catch (...) {
            // Відкочуємо єдину зміну історії; state_ досі не змінений.
            history_.pop_back();
            throw;
        }
        while (!history_.empty() && history_.front().end <= left) history_.pop_front();
        // Пам'ять обмежена window + vision_delay (у секундах, не весь політ).
        // Найстаріший інтервал, що перетинає left, зберігаємо повністю;
        // потрібну частину обрізає average(). Решта вже не знадобиться.
        time_ = time;
        state_ = next;
        window_changed_ = false;
        return state_;
    }

private:
    struct Interval {
        long double end;                 // правий край у накопичених секундах
        double dt;                       // повна тривалість, не обрізана вікном
        Vec2 acceleration, displacement; // сире NED a; масштабований приріст p
        Mat2 projection;                 // M саме цього інтервалу
    };
    struct Integral { Vec2 raw{}; Mat2 rotation{}; };
    Parameters parameters_{};
    Estimate state_{};
    std::deque<Interval> history_;
    long double time_ = 0;
    bool initialized_ = false;
    bool window_changed_ = false;

    long double windowLeft(long double time) const {
        // Не використовуємо майбутнє й не домальовуємо втрачену історію при
        // збільшенні вікна: left обмежена початком реально наявних інтервалів.
        long double left = std::max(0.0L, time-parameters_.windowSeconds());
        if (!history_.empty())
            left = std::max(left, history_.front().end-history_.front().dt);
        return left;
    }
    void windowStatus(Estimate& s, long double left, long double time) const {
        const long double span = std::max(0.0L, time-left);
        s.window_span_s = static_cast<double>(span);
        s.window_complete = span + 1e-12L*parameters_.windowSeconds() >= parameters_.windowSeconds();
    }

    static Vec2 multiply(const Mat2& a, const Vec2& b) {
        return {a[0][0]*b[0]+a[0][1]*b[1], a[1][0]*b[0]+a[1][1]*b[1]};
    }
    static void finite(double value) {
        if (!std::isfinite(value)) throw std::invalid_argument("Non-finite input or numerical overflow");
    }
    static void finite(const Vec2& v) { for (double x : v) finite(x); }
    static void validate(const Parameters& p) {
        for (double x : {p.fade_in, p.flat, p.fade_out, p.lead, p.vision_delay, p.bias_warmup, p.bias_memory}) {
            finite(x);
            if (x < 0) throw std::invalid_argument("Durations must be nonnegative");
        }
        finite(p.windowSeconds()); finite(p.windowSeconds()+p.vision_delay); finite(p.coordinate_scale);
        if (p.windowSeconds() <= 0 || p.coordinate_scale <= 0)
            throw std::invalid_argument("Window and coordinate scale must be positive");
        finite(p.initial_velocity); finite(p.initial_bias_body);
        if (p.quaternion_direction != QuaternionDirection::BodyToNed &&
            p.quaternion_direction != QuaternionDirection::NedToBody)
            throw std::invalid_argument("Invalid quaternion direction");
    }
    double weight(double age, double span) const {
        // age = 0 біля НОВОГО краю, age = span біля СТАРОГО.
        // fade_in зростає від старого краю; fade_out спадає до нового.
        // На старті краї можуть перекриватися: виходить трикутник, а не
        // повна трапеція. Нижче все одно нормуємо на фактичну суму w*h.
        double w = 1;
        if (parameters_.fade_in > 0) w = std::min(w, (span-age)/parameters_.fade_in);
        if (parameters_.fade_out > 0) w = std::min(w, age/parameters_.fade_out);
        return std::max(0.0, w);
    }

    Integral integrate(long double left, long double right) const {
        Integral result;
        for (const auto& interval : history_) {
            if (interval.end <= left) continue;
            const auto begin = interval.end - interval.dt;
            if (begin >= right) break;
            const double h = static_cast<double>(std::min(right, interval.end)-std::max(left, begin));
            if (h <= 0) continue;
            for (int j = 0; j < 2; ++j) {
                result.raw[j] += interval.acceleration[j]*h;
                for (int k = 0; k < 2; ++k)
                    result.rotation[j][k] += interval.projection[j][k]*h;
            }
        }
        return result;
    }

    void average(Estimate& s, long double left, long double epoch) const {
        // Два курсори: IMU закінчується в interval.end, vision — на delay
        // раніше. Розбиваємо вікно на спільні відрізки. Це O(history size),
        // без копій історії та без округлення затримки до кількості кадрів.
        Vec2 integral{}, visual_sum{}, inertial_sum{};
        Mat2 rotation_integral{}, rotation_sum{};
        s.weight_sum = s.weighted_duration_s = 0;
        std::size_t ai = 0, vi = 0;
        auto cursor = left;
        while (cursor < epoch) {
            while (ai < history_.size() && history_[ai].end <= cursor) ++ai;
            while (vi < history_.size() && history_[vi].end-parameters_.vision_delay <= cursor) ++vi;
            if (ai == history_.size() || vi == history_.size())
                throw std::invalid_argument("Missing aligned history");
            const auto& imu = history_[ai];
            const auto& vision = history_[vi];
            const auto end = std::min({epoch, imu.end, vision.end-parameters_.vision_delay});
            const double h = static_cast<double>(end-cursor);
            const double age = static_cast<double>(epoch-end)+h/2;
            const double w = weight(age, s.window_span_s), mass = w*h;
            // Один BODY-біас на все вікно; власна історична M кожного IMU.
            // Fade — наближення середньою точкою, без додаткової модуляції a.
            const auto bias_ned = multiply(imu.projection, s.bias_body_used);
            for (int j = 0; j < 2; ++j) {
                const double delta = (imu.acceleration[j]-bias_ned[j])*h;
                inertial_sum[j] += mass*(integral[j]+delta/2);
                integral[j] += delta; // acceleration is NEVER faded
                // Частина візуального приросту: лінійна інтерполяція позиції.
                // Однакові w*h для обох середніх; dp/dt не формуємо.
                visual_sum[j] += w*(vision.displacement[j]*(h/vision.dt));
                for (int k = 0; k < 2; ++k) {
                    const double dr = imu.projection[j][k]*h;
                    rotation_sum[j][k] += mass*(rotation_integral[j][k]+dr/2);
                    rotation_integral[j][k] += dr;
                }
            }
            s.weight_sum += w;
            s.weighted_duration_s += mass;
            cursor = end;
        }
        if (!(s.weighted_duration_s > 0))
            throw std::invalid_argument("Window has no positive weighted duration");
        for (int j = 0; j < 2; ++j) {
            s.window_integral[j] = integral[j];
            s.mean_visual_velocity[j] = visual_sum[j]/s.weighted_duration_s;
            s.mean_integrated_velocity[j] = inertial_sum[j]/s.weighted_duration_s;
            s.normalization_offset[j] = s.mean_visual_velocity[j]-s.mean_integrated_velocity[j];
            // Зсуваємо весь інтеграл на константу. Беремо швидкість на
            // правому краї (vision epoch), не значення посередині вікна.
            s.delayed_velocity[j] = integral[j]+s.normalization_offset[j];
            for (int k = 0; k < 2; ++k)
                s.bias_response[j][k] = rotation_integral[j][k]-rotation_sum[j][k]/s.weighted_duration_s;
        }
    }

    void learn(Estimate& s, const Integral& delta, const Estimate& previous) const {
        // Модель: a_raw = a_true + M*b_true, де b_true приблизно сталий
        // у BODY протягом вікна. Для ідеальних узгоджених даних:
        // v_est = v_true + C*(b_true - b_used).
        // Визначимо u = v_est + C*b_used: оцінку швидкості, яку те саме
        // вікно дало б БЕЗ віднімання біаса. Тоді
        // r = delta_raw - (u_now-u_prev) = (delta_M - C_now+C_prev)*b_true.
        // delta_raw та delta_M інтегруються між двома VISION-епохами,
        // а не між поточними IMU-кадрами; v_est тут — delayed_velocity.
        // H = (delta_M - C_now + C_prev)/dt; за delay=0 delta_M=M*dt.
        // Формули прибирають прямий вплив зміни нашої попередньої оцінки b.
        // Оцінена швидкість не стає незалежним вимірюванням: в ній ті самі
        // a і vision. Помилки/затримки vision можуть перейти у fitted bias.
        Mat2 h{};
        const auto response_now = multiply(s.bias_response, s.bias_body_used);
        const auto response_prev = multiply(previous.bias_response, previous.bias_body_used);
        Vec2 residual{};
        for (int j = 0; j < 2; ++j) {
            // r у м/с, H безрозмірна. Різницю швидкостей беремо в NED,
            // не у BODY: обертання BODY саме змінює компоненти швидкості.
            residual[j] = delta.raw[j]-(s.delayed_velocity[j]-previous.delayed_velocity[j])
                          -response_now[j]+response_prev[j];
            for (int k = 0; k < 2; ++k)
                h[j][k] = (delta.rotation[j][k]-s.bias_response[j][k]+previous.bias_response[j][k])/s.dt_s;
        }
        double information_dt = s.dt_s, residual_weight = 1;
        // МНК мінімізує sum(weight_time * |H*b-r/dt|^2).
        // Без забування weight_time=dt: N += dt*H^T*H, rhs += H^T*r.
        // У коді не формуємо r/dt; ділення на dt у H лишається необхідним.
        if (parameters_.bias_memory > 0) {
            const double ratio = s.dt_s/parameters_.bias_memory;
            const double decay = std::exp(-ratio);
            // Точна маса експоненційного забування на інтервалі за сталих
            // H і r/dt: tau*(1-exp(-dt/tau)). expm1 точніша для малого dt.
            information_dt = ratio == 0 ? s.dt_s : -parameters_.bias_memory*std::expm1(-ratio);
            residual_weight = information_dt/s.dt_s;
            // Забуваємо і дані, і стартовий поріг. Постійний стартовий поріг
            // при кінцевій пам'яті міг би назавжди пригнічувати слабкий напрям.
            for (double& x : s.bias_normal) x *= decay;
            for (double& x : s.bias_rhs) x *= decay;
            s.bias_prior_information_s *= decay;
        }
        for (int j = 0; j < 2; ++j) {
            // Додаємо по одному рівнянню з NED X та NED Y. Недіагональний
            // N_xy зв'язує BODY bx/by через повороти. N — інформація МНК
            // у секундах, rhs — м/с; це НЕ дисперсія і не P Кальмана.
            s.bias_normal[0] += information_dt*h[j][0]*h[j][0];
            s.bias_normal[1] += information_dt*h[j][0]*h[j][1];
            s.bias_normal[2] += information_dt*h[j][1]*h[j][1];
            for (int k = 0; k < 2; ++k) s.bias_rhs[k] += residual_weight*h[j][k]*residual[j];
        }
        for (double x : s.bias_normal) finite(x);
        finite(s.bias_rhs);
        const auto a = s.bias_normal[0], b = s.bias_normal[1], d = s.bias_normal[2];
        // Симетрична N = [[a,b],[b,d]]. Тут локальна b — N_xy, НЕ біас.
        // high/low — власні значення (скільки інформації у двох напрямах).
        // clamp до нуля прибирає можливу від'ємність від округлення.
        const double center = a/2+d/2, radius = std::hypot(a/2-d/2, b);
        const double high = std::max(0.0, center+radius), low = std::max(0.0, center-radius);
        const double angle = 0.5*std::atan2(2*b, a-d), c = std::cos(angle), sn = std::sin(angle);
        // Стовпці U = [(c,sn),(-sn,c)] — власні вектори high/low.
        // project(v)=U^T*v: перехід до незалежних напрямів інформації.
        auto project = [&](Vec2 v) -> Vec2 { return {c*v[0]+sn*v[1], -sn*v[0]+c*v[1]}; };
        Vec2 components{};
        if (parameters_.bias_warmup > 0) {
            const auto b0 = parameters_.initial_bias_body;
            const auto prior = project(b0);
            const auto error = project({s.bias_rhs[0]-a*b0[0]-b*b0[1], s.bias_rhs[1]-b*b0[0]-d*b0[1]});
            // У власному напрямі: b = b0 + (rhs-lambda*b0)/max(lambda,floor).
            // Коли lambda мала, не ділимо на майже нуль: відхилення від b0
            // набирає довіру поступово. Це не жорстка межа біаса/його зміни
            // і не очікування bias_warmup секунд перед початком навчання.
            const double floor = std::max(s.bias_prior_information_s, std::max(high*1e-8, 1e-12));
            components = {prior[0]+error[0]/std::max(high, floor),
                          prior[1]+error[1]/std::max(low, floor)};
        } else {
            // Без startup floor розв'язуємо тільки спостережувані напрями.
            // У напрямі з майже нульовою інформацією зберігаємо старий біас.
            const double threshold = std::max(high*1e-8, 1e-12);
            const auto rhs = project(s.bias_rhs);
            components = project(s.bias_body);
            if (high > threshold) components[0] = rhs[0]/high;
            if (low > threshold) components[1] = rhs[1]/low;
        }
        s.bias_body = {c*components[0]-sn*components[1], sn*components[0]+c*components[1]};
        // Зворотне U*components повертає оцінку в BODY X/Y. Вона буде
        // bias_body_used наступного кроку; швидкість s.velocity вже обчислена.
    }

    void finish(Estimate& s, const Mat2& m) const {
        // Новий біас у NED — для діагностики. corrected_acceleration лишається
        // узгодженим зі СТАРИМ bias_ned_used, застосованим до цього кроку.
        s.bias_ned = multiply(m, s.bias_body);
        for (int j = 0; j < 2; ++j)
            // lead (forward_time) — горизонт випередження, НЕ dt кадру.
            // Це лінійна екстраполяція з оціненою швидкістю. Вона не
            // переносить саму measured_position до now: тут вона ще має свій
            // vision_time_s. Компенсація vision_delay стосується velocity.
            // Це задана користувачем формула vis + lead*v, без a*lead^2/2.
            s.forward_position[j] = s.measured_position[j]+parameters_.lead*s.velocity[j];
        for (double x : {s.time_s, s.dt_s, s.vision_time_s, s.bias_fit_dt_s, s.window_span_s, s.weight_sum, s.weighted_duration_s, s.bias_prior_information_s}) finite(x);
        for (const auto& v : {s.raw_position, s.measured_position, s.forward_position, s.velocity, s.delayed_velocity, s.prediction_delta_velocity,
                             s.mean_visual_velocity, s.mean_integrated_velocity, s.window_integral,
                             s.normalization_offset, s.raw_acceleration, s.corrected_acceleration,
                             s.integral_raw, s.bias_body, s.bias_body_used, s.bias_ned, s.bias_ned_used, s.bias_rhs}) finite(v);
        for (const auto& row : s.bias_response) finite(row);
        for (double x : s.bias_normal) finite(x);
    }
};

} // namespace flight_velocity
