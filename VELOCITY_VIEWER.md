# Experimental inertial velocity viewer

This Python experiment estimates horizontal velocity from NED acceleration, anchors its mean to visual motion over a past window, and learns accelerometer bias in body axes from the resulting velocity. It does not run a Kalman filter or subtract a fitted linear velocity trend.

## Run

Use the repository's existing dependencies and keep `velocity_flight_viewer.py`, `kalman_flight_viewer.py` and `viewer_widgets.py` together. The new script imports the shared CSV/attitude helpers and desktop controls.

```bash
python -m pip install -r requirements.txt
python velocity_flight_viewer.py
python velocity_flight_viewer.py flight.csv
```

The empty window supports **Open CSV** and **drag and drop**, with the same Tk/Qt requirements as the Kalman viewer. Loading a file starts it paused at time zero. A file supplied on the command line plays at real time unless `--paused` is given. Space toggles playback, arrows seek one second, Home/End seek to the limits, `+`/`-` adjust playback rate, and the time slider seeks and pauses.

X and Y are separate rows, each showing:

1. Scaled visual position and `scaled_visual_position + lead * estimated_speed`.
2. Estimated speed and the window's weighted mean visual speed.
3. Learned body bias and its current NED projection.
4. Input NED acceleration and acceleration corrected by the bias used at that step.

The default visible plot width is 10 s. This is independent of the 18 s estimation window.

## Controls

| Setting | Default | Meaning |
| --- | --- | --- |
| Old-edge fade | 6 s | Weight grows linearly from zero at the oldest edge |
| Flat part | 6 s | Full weight |
| Recent-edge fade | 6 s | Weight falls linearly to zero at the newest edge |
| Forward time | 1 s | Lead horizon in `vis + lead * speed`, independent of sample step |
| Coordinate multiplier | 1 | Applied to original `vis_x/y` before deriving visual speed |
| Sample step | 0.05 s | Fixed interval; CSV timestamps do not set the time base |
| Plot window | 10 s | Visible time span only |
| Initial body bias X/Y | 0 m/s² | Starting estimates in body axes |
| Bias warmup | 6 s | Minimum information time before fully trusting a learned body-bias direction; 0 disables the startup ramp |
| Learn body bias | Enabled | Disable to hold the entered body biases fixed |

**Apply + export** recomputes the whole flight from row zero using the edited settings, saves the CSV, and pauses at the current time. Editing fields does not change the active result until Apply, Open CSV, or Open in PlotJuggler. Reapplying a coordinate scale always starts from the original positions. Invalid parameters or a failed export retain the previously computed result.

Typing, cursor movement and changing field focus repaint only the controls on TkAgg/QtAgg. They do not redraw the eight plots, recalculate estimates or write the CSV, even when replay was already paused. Other backends without partial drawing fall back to a full redraw.

Initial NED speed is zero by default. If the log starts while moving, set `--vx0` and `--vy0` in m/s. These are the assumed starting speed for cumulative bias fitting, not a velocity measurement from a later frame.

## Input and output

Input is the same semicolon CSV used by the Kalman viewer: `vis_x`, `vis_y`, `mheight`, `acc.x/y` or `acc_x/y`, `quat.x/y/z/w` (underscore aliases accepted), and `yaw`. Positions are in metres and accelerations in m/s², in the same NED axes. Pass acceleration **before subtracting this experiment's own estimated bias**. It must already be translational acceleration in NED; raw body specific force including gravity is not this input.

Quaternion normalization, direction checks, yaw-unit checks, and optional raw-IMU diagnostics reuse the existing implementation. `--quat-direction body-to-ned` / `ned-to-body` and `--yaw-unit deg` / `rad` override automatic conventions. Auto-detection inspects the CSV offline. Once frame conventions are fixed, all estimates use only the current and previous samples.

Opening or applying automatically writes **`result_velocity.csv` next to the input**. An explicit `--export` path stays fixed when opening another flight. Source files are protected from output-path collisions. The output is replaced atomically, without publishing a partial CSV.

Output columns (suffix `_x/y` means two separate columns). Positions, velocities,
accelerations and NED biases use X=North, Y=East; `bias_body_*` uses rotating body
axes. Body Z bias is assumed zero. All means refer to the current past window.

| Columns | Meaning |
| --- | --- |
| `time_s`, `dt` | Row-based elapsed time and sample step, seconds; input timestamps are not used |
| `mheight` | Input height above ground, metres; logged but not used to weight this velocity estimator |
| `vis_x/y`, `vis_scaled_x/y` | Original and scaled positions |
| `estimated_speed_x/y`, `niacc.x/y` | Estimated velocity; the two naming schemes contain identical values |
| `forward_position_x/y` | Scaled position plus `lead * estimated_speed` |
| `visual_interval_speed_x/y` | Visual position difference divided by sample step; row zero contains the assumed initial speed |
| `mean_visual_speed_x/y` | Weighted visual interval-velocity mean |
| `mean_inertial_speed_x/y` | Weighted mean of the locally integrated inertial velocity, before offset correction |
| `window_integral_acc_x/y` | Inertial integral at the current endpoint, starting at zero at the window's left sample boundary |
| `velocity_offset_x/y` | Mean visual speed minus mean inertial speed |
| `bias_body_used_x/y` | Previous body estimate used for this velocity update |
| `bias_body_estimated_x/y` | Body bias after the update, available to the next sample |
| `bias_ned_used_x/y`, `bias_ned_estimated_x/y` | Corresponding projections at the current attitude |
| `acc_raw_x/y`, `acc_used_x/y` | Raw and bias-corrected NED accelerations |
| `window_complete`, `window_span_s` | Whether the full requested window exists, and available span |

Position columns are in metres. All speed, offset and acceleration-integral
columns are in m/s, including **`window_integral_acc_x/y`** despite its name.
Acceleration and bias columns are in m/s². `window_span_s` is in seconds and
`window_complete` is 0 or 1.

For either axis, the exported identities are:

```text
velocity_offset = mean_visual_speed - mean_inertial_speed
estimated_speed = window_integral_acc + velocity_offset
forward_position = vis_scaled + lead * estimated_speed
```

`mean_inertial_speed` is the weighted mean over the window, not the velocity
at the window centre. The integral starts from zero at its left boundary, so
its absolute value depends on that boundary; `velocity_offset` restores the
velocity reference. `bias_*_used` is the previous estimate used on this step,
while `bias_*_estimated` is the newly updated estimate for the following step.

Additional parameters and diagnostics:

| Columns | Meaning / units |
| --- | --- |
| `fade_in`, `flat`, `fade_out` | Old-edge fade, full-weight part, recent-edge fade durations, seconds |
| `lead` | Forward-position horizon, seconds; not the sample step |
| `coord_scale` | Visual-coordinate multiplier, dimensionless |
| `weight_sum` | Sum of interval fade weights **before** normalization; dimensionless, not a confidence score |
| `learn_bias` | 1 learns body bias; 0 holds the entered initial bias fixed |
| `bias_x0`, `bias_y0` | Initial body bias, m/s² |
| `bias_warmup` | Minimum information time for the body-bias solve, seconds |
| `vx0`, `vy0` | Assumed initial NED speed, m/s |
| `integral_acc_raw_x/y` | Integral of raw NED acceleration since row zero, without bias subtraction, m/s |
| `quat_x_normalized_input`, `quat_y_normalized_input`, `quat_z_normalized_input`, `quat_w_normalized_input` | Unit-length input quaternion, dimensionless; retains the input direction convention |
| `yaw_csv_deg` | Input yaw converted to degrees |
| `yaw_quat_deg` | Yaw from the selected body-to-NED quaternion rotation, degrees |
| `yaw_error_deg` | `yaw_quat_deg - yaw_csv_deg`, wrapped to [-180, 180) degrees |
| `bias_fit_normal_xx`, `bias_fit_normal_xy`, `bias_fit_normal_yy` | Symmetric 2×2 cumulative body-bias fit matrix, seconds |
| `bias_fit_rhs_x/y` | Right-hand side of the raw data fit, m/s; startup trust damping is applied when solving it |

Settings are repeated on every row for reproducibility. There are no Kalman
covariances or gains in this export. Row zero initializes the result; integration
and bias learning begin at row one. The first visual-interval speed is the
assumed initial speed rather than a measured difference.

```bash
# Compute without opening a desktop window.
python velocity_flight_viewer.py flight.csv --no-gui

# Change the estimation window and lead, and choose an output path.
python velocity_flight_viewer.py flight.csv --fade-in 3 --flat 4 --fade-out 3 --lead 0.8 --export speed.csv

# Compare with a known, fixed BODY bias.
python velocity_flight_viewer.py flight.csv --fixed-bias --bias-x0 0.3 --bias-y0 -0.2

# Save a 20-second view ending at 330 seconds, without a desktop.
python velocity_flight_viewer.py flight.csv --no-gui --window 20 --start 330 --snapshot speed.png
```

**Open in PlotJuggler** applies edited settings, exports, and opens the result using the same executable discovery as the Kalman viewer. Use `time_s` as the time column and `;` as the delimiter if prompted.

## Exact update order

Let `R_j` map body to NED, and `M_j = R_j[:2, :2]`. Only body X/Y bias is estimated; **body Z bias is fixed to zero**. Roll, pitch and yaw all participate in the projection.

At sample `i`, take the previously available body bias `b_(i-1)`. Hold this vector fixed in body axes while reintegrating the current past window. Rotate it with **each historical sample's own quaternion**:

```text
a_corrected_j = a_raw_j - M_j * b_(i-1)
I_start = 0
I_j = I_(j-1) + a_corrected_j * dt
```

The implementation uses prefix integrals of acceleration and rotation matrices to evaluate this efficiently. It does not retain older, differently corrected accelerations in the current window.

Visual speed from a position difference is an average over a sample interval. Compare it with inertial speed at the same midpoint, assuming constant acceleration within that sample:

```text
visual_speed_j = (p_j - p_(j-1)) / dt
inertial_midpoint_speed_j = (I_j + I_(j-1)) / 2
```

At interval midpoint age `u` seconds before now, the full-window weight is:

```text
T = fade_in + flat + fade_out
w(u) = max(0, min(1, (T-u)/fade_in, u/fade_out)),  0 <= u <= T
```

Omit a division term when its fade duration is zero. Both means use precisely the same interval weights and divide by their actual sum:

```text
offset_i = weighted_mean(visual_speed) - weighted_mean(inertial_midpoint_speed)
estimated_speed_i = I_i + offset_i
```

The fade weights apply to the **means**, not to the physical `a * dt` increments. Only the integration constant is adjusted. This preserves acceleration dynamics; it is different from the earlier experiment that removed a linear trend from integrated acceleration. The lead is used only to produce `forward_position`; it does not feed velocity normalization or bias learning.

After obtaining this sample's velocity, update the next body bias. The speed
comes from a faded past window, so its error response to a bias is not simply
the current attitude. Let `B_j` be the integral of `M_j` from the current
window's left boundary. Using the same interval-midpoint fade weights as speed:

```text
C_i = B_end - weighted_mean((B_(j-1) + B_j) / 2)
```

For exact visual data and a constant true body bias, this gives
`estimated_speed_i = true_speed_i + C_i*(true_bias - bias_used_i)`.
Consequently, the residual used to learn bias must account both for the window
response and for the different bias estimates used at the two adjacent steps:

```text
delta_v = estimated_speed_i - estimated_speed_(i-1)       # both in NED
H_i = M_i - (C_i - C_(i-1)) / dt
r_i = a_raw_i*dt - delta_v - C_i*bias_used_i + C_(i-1)*bias_used_(i-1)
normal += dt * transpose(H_i) * H_i
rhs    += transpose(H_i) * r_i
b_i     = solve_with_startup_trust(normal, rhs, initial_body_bias, bias_warmup)
```

Here `H_i` is a bias observation matrix, not an orthogonal rotation matrix.
Initialize `C_0=0`. The correction keeps changes caused by updating the bias
from being interpreted as physical acceleration; it also accounts for the
window delay during body rotation. Simply damping the previous residual fit
could hide the startup spike but fail to converge during sustained rotation.

Accumulate over the whole flight. The normal matrix and right-hand side in the
CSV contain these cumulative observations, without any artificial time added.

At startup, the first velocity update may correct an incorrect assumed initial
speed or follow visual noise. Dividing that correction by only 0.05 s used to
produce a large false accelerometer bias, which was fed back into the following
velocity update. A fading visual mean does not protect a one-interval window:
its sole weight cancels when the mean is normalized.

Use a minimum information time `T = bias_warmup` (default 6 s). If
`normal = U * diag(lambda) * U.T`, solve for the departure from the configured
initial body bias `b0`:

```text
b_i = b0 + U * ((U.T * (rhs - normal*b0)) / max(lambda, T))
```

The maximum and division are componentwise. This implements a linear trust ramp
`min(lambda/T, 1)` in each body direction without first dividing by a tiny
eigenvalue. Weakly observed directions remain close to the initial bias. The
information time is not necessarily the wall-clock time: the window and body
rotation affect how much information each interval contributes.

Row zero is exactly the configured initial bias, zero by default. After a
direction accumulates at least `T` seconds of information, its solution is the
ordinary cumulative fit; there is no permanent attenuation of the result.
The estimate need not grow monotonically: it responds to the actual residuals.
This startup setting is independent of the visual fade and display windows.
`--bias-warmup 0` disables only the trust ramp, retaining the corrected window
response and the observable-eigenvalue check; it can again amplify startup noise.
With `--fixed-bias`, the configured initial biases are retained throughout and
warmup has no effect. The Kalman viewer keeps its previous default behavior.

The new `b_i` is used at sample `i+1`. There is no same-sample iteration or use of future velocity. Differencing NED velocities before transporting the residual avoids the rotating-frame derivative term introduced by subtracting velocities expressed in different body frames.

Row zero initializes state without integration or bias learning. During startup use the available past intervals and the same fade slopes, normalize their actual weights, and mark `window_complete=0`. Overlapping fades produce a shorter triangular window. Startup estimates are shown, rather than padding the past or waiting 18 seconds for the first output.

This is an experiment with coupled velocity and bias estimates, not an independent measurement of accelerometer bias. Visual-motion errors and attitude errors can enter the learned bias. Correct arithmetic and consistent synthetic motion do not establish closed-loop flight stability or accuracy on a real log without ground truth.

## Requirement for the final variable-time implementation

The current CSV experiment uses one fixed `dt` (default 0.05 s). It does **not**
yet read variable sample timestamps. The eventual streaming implementation must
use actual elapsed time both in integration and in the two means, rather than
giving a densely sampled part of the window more influence just because it has
more samples.

For interval `i`, let `dt_i = t_i - t_(i-1) > 0`, and evaluate the same fade
`w_i` at that interval's time midpoint for both signals. With acceleration held
constant within each interval, the midpoint approximation is:

```text
I_i = I_(i-1) + (a_raw_i - M_i * previous_body_bias) * dt_i
I_mid_i = (I_(i-1) + I_i) / 2
D = sum(w_i * dt_i)

mean_visual_speed = sum(w_i * (p_i - p_(i-1))) / D
mean_inertial_speed = sum(w_i * dt_i * I_mid_i) / D

estimated_speed = I_end + mean_visual_speed - mean_inertial_speed
```

This is a time-weighted mean, not a confidence weight. The visual numerator is
computed directly from position increments: `dt_i * (delta_p / dt_i)` cancels
algebraically, so the mean needs no division by each potentially tiny `dt_i`.
Both means divide only by the same total weighted duration `D`. With fixed `dt`,
this reduces to the current algorithm. Acceleration integration still has no
fade factor.

Window and fade lengths must be measured in seconds. If the left window boundary
cuts an interval, integrate and weight only its overlapping duration, with the
corresponding interpolated position increment. Large intervals crossing fade
corners need subdivision or integration of the fade rather than treating the
whole interval as a single midpoint sample. Reject non-increasing timestamps;
do not replace zero duration with an arbitrary epsilon. At startup with `D=0`,
retain the initialized velocity until a usable interval exists.

Cumulative bias learning must also use actual `dt_i` in `H_i`, the acceleration
increment and `normal += dt_i*H_i.T*H_i`. Compute `C_i` with the same time-weighted
mean used for speed, then accumulate `rhs += H_i.T*r_i` without dividing the
residual by `dt_i`. The new bias remains available only to the next sample. If
IMU and visual observations have different timestamps, align their integration
intervals before applying these same-window formulas.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests include known motion with changing acceleration and full attitude, body-bias recovery during rotation from zero and incorrect initial biases, constant NED velocity during rotation, causal prefix consistency, bias recovery during accelerated motion, suppression of a first-frame visual-speed correction in bias learning, warmup with unequal elapsed-time steps and weakly observed body directions, short-log startup, asymmetric/rectangular windows, CSV export, GUI recalculation, drag/drop dispatch, input protection, and PlotJuggler launch wiring. Existing Kalman-viewer tests run alongside them.
