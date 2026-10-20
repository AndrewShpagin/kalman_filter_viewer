# Experimental inertial velocity viewer

This Python experiment estimates horizontal velocity from NED acceleration, anchors its mean to visual motion over a past window, and learns accelerometer bias in body axes from the resulting velocity. It does not run a Kalman filter or subtract a fitted linear velocity trend.

## Run

Use the repository's existing dependencies and keep `velocity_flight_viewer.py` and `kalman_flight_viewer.py` together. The new script imports the CSV/attitude helpers and desktop controls from the latter.

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
| Learn body bias | Enabled | Disable to hold the entered body biases fixed |

**Apply + export** recomputes the whole flight from row zero using the edited settings, saves the CSV, and pauses at the current time. Editing fields does not change the active result until Apply, Open CSV, or Open in PlotJuggler. Reapplying a coordinate scale always starts from the original positions. Invalid parameters or a failed export retain the previously computed result.

Initial NED speed is zero by default. If the log starts while moving, set `--vx0` and `--vy0` in m/s. These are the assumed starting speed for cumulative bias fitting, not a velocity measurement from a later frame.

## Input and output

Input is the same semicolon CSV used by the Kalman viewer: `vis_x`, `vis_y`, `mheight`, `acc.x/y` or `acc_x/y`, `quat.x/y/z/w` (underscore aliases accepted), and `yaw`. Positions are in metres and accelerations in m/s², in the same NED axes. Pass acceleration **before subtracting this experiment's own estimated bias**. It must already be translational acceleration in NED; raw body specific force including gravity is not this input.

Quaternion normalization, direction checks, yaw-unit checks, and optional raw-IMU diagnostics reuse the existing implementation. `--quat-direction body-to-ned` / `ned-to-body` and `--yaw-unit deg` / `rad` override automatic conventions. Auto-detection inspects the CSV offline. Once frame conventions are fixed, all estimates use only the current and previous samples.

Opening or applying automatically writes **`result_velocity.csv` next to the input**. An explicit `--export` path stays fixed when opening another flight. Source files are protected from output-path collisions. The output is replaced atomically, without publishing a partial CSV.

Useful output columns:

| Columns | Meaning |
| --- | --- |
| `time_s`, `dt` | Row-based elapsed time and sample step |
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

The export also records raw acceleration integrals, normalized input quaternions, yaw checks, body-bias normal equations, and active numerical parameters. It contains no Kalman covariances or gains.

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

After obtaining this sample's velocity, update the next body bias:

```text
delta_v = estimated_speed_i - estimated_speed_(i-1)       # both in NED
normal += dt * transpose(M_i) * M_i
rhs    += transpose(M_i) * (a_raw_i * dt - delta_v)
b_i     = solve(normal, rhs)
```

Accumulate over the whole flight. Preserve the previous estimate in unobservable body directions, using the existing eigenvalue check. For a fixed identity attitude this reduces to `(integral(a_raw) - (v_i-v_0)) / elapsed_time`.

The new `b_i` is used at sample `i+1`. There is no same-sample iteration or use of future velocity. Differencing NED velocities before transporting the residual avoids the rotating-frame derivative term introduced by subtracting velocities expressed in different body frames.

Row zero initializes state without integration or bias learning. During startup use the available past intervals and the same fade slopes, normalize their actual weights, and mark `window_complete=0`. Overlapping fades produce a shorter triangular window. Startup estimates are shown, rather than padding the past or waiting 18 seconds for the first output.

This is an experiment with coupled velocity and bias estimates, not an independent measurement of accelerometer bias. Visual-motion errors and attitude errors can enter the learned bias. Correct arithmetic and consistent synthetic motion do not establish closed-loop flight stability or accuracy on a real log without ground truth.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests include known motion with changing acceleration and full attitude, body-bias recovery during rotation, constant NED velocity during rotation, causal prefix consistency, the fixed-attitude integral identity, short-log startup, asymmetric/rectangular windows, CSV export, GUI recalculation, drag/drop dispatch, input protection, and PlotJuggler launch wiring. Existing Kalman-viewer tests run alongside them.
