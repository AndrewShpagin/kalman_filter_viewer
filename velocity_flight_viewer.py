#!/usr/bin/env python3
"""Experimental window-normalized inertial velocity with a body-frame bias.

Run:    python velocity_flight_viewer.py [flight.csv]
Export: python velocity_flight_viewer.py flight.csv --no-gui

Uses the same CSV reader, quaternion checks, desktop replay, drag/drop and
PlotJuggler integration as kalman_flight_viewer.py (also keep viewer_widgets.py).
No Kalman filter is used here. Positions are vis_x/y in NED, acceleration is
acc.x/y or acc_x/y in NED, before subtraction of this estimator's own bias.
Time is row index * --dt (default 0.05 s), independent of CSV timestamps.

For each sample i:
  1. Use the PREVIOUS body bias b; M_j = R_body_to_ned_j[:2, :2].
  2. Integrate a_j - M_j @ b over the trailing window, with b fixed in body
     axes throughout that window. Historical quaternions rotate it at each j.
  3. Compute visual interval velocities (p_j - p_(j-1)) / dt. Match them to
     inertial velocities at the same interval midpoints. With constant a_j
     over a sample, the latter are (I_(j-1) + I_j) / 2.
  4. Apply the 6/6/6 s trapezoidal weights to these two averages:
        v_i = I_i - weighted_mean(I_mid) + weighted_mean(v_visual).
     Weights do NOT attenuate a*dt. Only an additive velocity offset is fitted;
     no linear detrending and no extra position smoother are applied.
  5. AFTER computing v_i, accumulate the next body-bias estimate. Account for
     the known velocity response C_i to a constant BODY bias over its window:
        H_i = M_i - (C_i - C_(i-1)) / dt
        residual = a_raw_i*dt - (v_i-v_(i-1))
                   - C_i*b_used_i + C_(i-1)*b_used_(i-1)
        normal += dt * H_i.T @ H_i
        rhs    += H_i.T @ residual
        b_next  = a gradually trusted solve of normal * b = rhs.
     Floor normal-matrix eigenvalues at --bias-warmup seconds (default 6).
     Blend from the configured initial body bias until each direction has
     enough accumulated information; 0 disables the startup trust ramp.
     Thus new bias first affects the NEXT velocity update. There is no
     same-sample fixed-point iteration and no future velocity in bias fitting.
  6. Plot/export p_forward_i = p_visual_i + lead_seconds * v_i (default 1 s).

Bias accumulation covers the whole flight, not the moving plot/window.
Body Z bias is assumed zero. The minimum information time prevents a first
visual-speed correction from being divided by almost zero accumulated time
and immediately fed back as a huge acceleration bias. The response correction
prevents changing bias from being mistaken for real acceleration, and keeps
learning aligned with the window's averaging even while the drone rotates.
NED velocity increments must be
differenced BEFORE rotating; body velocity differences add a rotation term.

Startup: row 0 uses --vx0/--vy0 and does not integrate. Before a full window
exists, normalize the available past intervals with faded edges; mark them
window_complete=0. No future samples or zero padding are used. The initial
velocity is an assumption; supply it if the log starts while already moving.
All historical accelerations in the current window are re-corrected with the
latest available bias. Prefix integrals make this equivalent to reintegration.
Frame auto-detection is an offline check of the CSV; once conventions are fixed,
the estimator is causal. Use explicit frame/unit flags for streaming comparisons.

The new viewer saves result_velocity.csv next to the input. Apply recalculates
from the beginning. It can start empty, open/drop another CSV, export without
a GUI, save a snapshot, or open the result in PlotJuggler.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

from kalman_flight_viewer import (
    Attitude, BodyBiasEstimator, FlightViewer, check_attitude, read_flight,
    wrap_degrees,
)


@dataclass
class VelocityConfig:
    dt: float = 0.05
    fade_in: float = 6.0
    flat: float = 6.0
    fade_out: float = 6.0
    lead: float = 1.0
    coord_scale: float = 1.0
    view_window: float = 10.0
    bias_x0: float = 0.0
    bias_y0: float = 0.0
    bias_warmup: float = 6.0
    vx0: float = 0.0
    vy0: float = 0.0
    learn_bias: bool = True
    quaternion_direction: str = 'auto'
    yaw_unit: str = 'auto'

    @property
    def fit_window(self):
        return self.fade_in + self.flat + self.fade_out

    def validate(self):
        for name in ('dt', 'coord_scale', 'view_window'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}: must be finite and > 0.')
        for name in ('fade_in', 'flat', 'fade_out', 'lead', 'bias_warmup'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}: must be finite and >= 0.')
        for name in ('bias_x0', 'bias_y0', 'vx0', 'vy0'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name}: must be finite.')
        if not math.isfinite(self.fit_window) or self.fit_window < self.dt:
            raise ValueError('Fade in + flat + fade out must be at least one time step.')
        if self.quaternion_direction not in ('auto', 'body-to-ned', 'ned-to-body'):
            raise ValueError('Invalid quaternion direction.')
        if self.yaw_unit not in ('auto', 'deg', 'rad'):
            raise ValueError('Invalid yaw unit.')


def interval_weights(age: np.ndarray, span: float, cfg: VelocityConfig) -> np.ndarray:
    """Trapezoid from oldest -> newest: fade-in, flat, fade-out.

    During startup span is shortened, retaining the edge slopes; if the two
    fades overlap, the weights form a triangle. Normalize by actual sum later.
    """
    weights = np.ones_like(age)
    if cfg.fade_in > 0:
        weights = np.minimum(weights, (span - age) / cfg.fade_in)
    if cfg.fade_out > 0:
        weights = np.minimum(weights, age / cfg.fade_out)
    return np.where((age >= 0) & (age <= span), np.maximum(weights, 0), 0)


def estimate_velocity(data: np.ndarray, cfg: VelocityConfig,
                      attitude: Attitude | None = None) -> dict[str, np.ndarray]:
    cfg.validate()
    if data.ndim != 2 or len(data) < 2 or data.shape[1] < 10:
        raise ValueError('At least two rows of position, acceleration and attitude are required.')
    if not np.isfinite(data[:, :10]).all():
        raise ValueError('Required input values must be finite.')
    if attitude is None:
        attitude = check_attitude(data, cfg)
    n = len(data)
    if attitude.rotation.shape != (n, 3, 3) or not np.isfinite(attitude.rotation).all():
        raise ValueError('Attitude rotations do not match the flight.')
    dt = cfg.dt
    t = np.arange(n, dtype=float) * dt
    measured = data[:, :2] * cfg.coord_scale
    raw_acc = data[:, 3:5]
    matrices = attitude.rotation[:, :2, :2]
    # Prefixes are functions only of samples up to their own row.
    raw_integral = np.zeros((n, 2))
    raw_integral[1:] = np.cumsum(raw_acc[1:] * dt, axis=0)
    rotation_integral = np.zeros((n, 2, 2))
    rotation_integral[1:] = np.cumsum(matrices[1:] * dt, axis=0)
    visual_velocity = np.empty((n, 2))
    visual_velocity[0] = cfg.vx0, cfg.vy0
    visual_velocity[1:] = np.diff(measured, axis=0) / dt
    out = {
        'time': t, 'height': data[:, 2], 'raw_position': data[:, :2],
        'measured_position': measured, 'raw_acceleration': raw_acc,
        'visual_velocity': visual_velocity, 'integral_raw': raw_integral,
        'quaternion': attitude.quaternion, 'yaw_csv_deg': attitude.yaw_csv_deg,
        'yaw_quat_deg': attitude.yaw_quat_deg,
        'yaw_error_deg': wrap_degrees(attitude.yaw_quat_deg - attitude.yaw_csv_deg),
        'window_span_s': np.minimum(t, cfg.fit_window),
        'window_complete': (t + dt * 1e-9 >= cfg.fit_window).astype(float),
        'weight_sum': np.zeros(n),
        'bias_normal': np.zeros((n, 3)),
    }
    for key in ('velocity', 'forward_position', 'mean_visual_velocity',
                'mean_integrated_velocity', 'window_integral', 'normalization_offset',
                'corrected_acceleration', 'bias_body', 'bias_body_used',
                'bias', 'bias_used', 'bias_rhs'):
        out[key] = np.zeros((n, 2))
    for name in ('dt', 'fade_in', 'flat', 'fade_out', 'lead', 'coord_scale',
                 'bias_x0', 'bias_y0', 'bias_warmup', 'vx0', 'vy0', 'learn_bias'):
        out[name] = np.full(n, float(getattr(cfg, name)))
    out['velocity'][0] = cfg.vx0, cfg.vy0
    out['mean_visual_velocity'][0] = out['velocity'][0]
    out['normalization_offset'][0] = out['velocity'][0]
    bias_fit = BodyBiasEstimator((cfg.bias_x0, cfg.bias_y0), cfg.bias_warmup)
    out['bias_body'][0] = out['bias_body_used'][0] = bias_fit.bias
    out['bias'][0] = out['bias_used'][0] = matrices[0] @ bias_fit.bias
    out['corrected_acceleration'][0] = raw_acc[0] - out['bias_used'][0]

    # Bound the allocation by flight length even for a very long requested window.
    max_intervals = n - 1 if cfg.fit_window >= t[-1] else int(math.ceil(cfg.fit_window / dt))
    full_ages = (np.arange(max_intervals, 0, -1) - 0.5) * dt
    full_weights = interval_weights(full_ages, cfg.fit_window, cfg)
    response_previous = np.zeros((2, 2))
    bias_used_previous = bias_fit.bias.copy()
    for i in range(1, n):
        b = bias_fit.bias.copy()
        m = matrices[i]
        out['bias_body_used'][i] = b
        out['bias_used'][i] = m @ b
        out['corrected_acceleration'][i] = raw_acc[i] - m @ b
        count = min(i, max_intervals)
        start = i - count
        ages = full_ages[-count:]
        weights = (full_weights if count == max_intervals and t[i] >= cfg.fit_window
                   else interval_weights(ages, min(t[i], cfg.fit_window), cfg))
        total = float(weights.sum())
        if total <= 0:
            raise ValueError('Window has no positively weighted sample intervals.')
        weights = weights / total
        out['weight_sum'][i] = total

        # Reintegrate the entire past window using the currently available BODY
        # bias. Subtract the prefix at start to keep local integrals well scaled.
        a_local = raw_integral[start:i + 1] - raw_integral[start]
        r_local = rotation_integral[start:i + 1] - rotation_integral[start]
        integrated = a_local - r_local @ b
        midpoint_velocity = 0.5 * (integrated[:-1] + integrated[1:])
        mean_inertial = weights @ midpoint_velocity
        mean_visual = weights @ visual_velocity[start + 1:i + 1]
        offset = mean_visual - mean_inertial
        out['window_integral'][i] = integrated[-1]
        out['mean_integrated_velocity'][i] = mean_inertial
        out['mean_visual_velocity'][i] = mean_visual
        out['normalization_offset'][i] = offset
        out['velocity'][i] = integrated[-1] + offset

        # v_i = v_true_i + C_i @ (b_true - b_used_i), for exact visual data
        # and a constant body bias. C_i uses the SAME window and fade as speed.
        rotation_midpoints = 0.5 * (r_local[:-1] + r_local[1:])
        response = r_local[-1] - np.einsum('i,ijk->jk', weights, rotation_midpoints)
        # Current velocity completes this update; the resulting new bias is only
        # consumed by sample i+1. Account for the window response and for using
        # different bias estimates on the two adjacent velocity outputs.
        if cfg.learn_bias:
            delta_v = out['velocity'][i] - out['velocity'][i - 1]
            effective_projection = m - (response - response_previous) / dt
            residual = (raw_acc[i] * dt - delta_v - response @ b
                        + response_previous @ bias_used_previous)
            bias_fit.update_residual(effective_projection, residual, dt)
        response_previous = response
        bias_used_previous = b
        out['bias_body'][i] = bias_fit.bias
        out['bias'][i] = m @ bias_fit.bias
        normal = bias_fit.normal
        out['bias_normal'][i] = normal[0, 0], normal[0, 1], normal[1, 1]
        out['bias_rhs'][i] = bias_fit.rhs

    out['forward_position'] = measured + cfg.lead * out['velocity']
    for key, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Estimate became non-finite: {key}; check input and parameters.')
    return out


def export_velocity(path: Path, result: dict[str, np.ndarray]) -> None:
    """Atomic semicolon export, including enough intermediates to audit the fit."""
    path = Path(path)
    names = ['time_s', 'mheight']
    columns = [result['time'], result['height']]
    for key in ('dt', 'fade_in', 'flat', 'fade_out', 'lead', 'coord_scale',
                'window_span_s', 'window_complete', 'weight_sum', 'learn_bias',
                'bias_x0', 'bias_y0', 'bias_warmup', 'vx0', 'vy0'):
        names.append(key)
        columns.append(result[key])
    for key, prefix in (
        ('raw_position', 'vis'), ('measured_position', 'vis_scaled'),
        ('velocity', 'estimated_speed'), ('forward_position', 'forward_position'),
        ('visual_velocity', 'visual_interval_speed'),
        ('mean_visual_velocity', 'mean_visual_speed'),
        ('mean_integrated_velocity', 'mean_inertial_speed'),
        ('window_integral', 'window_integral_acc'),
        ('normalization_offset', 'velocity_offset'),
        ('raw_acceleration', 'acc_raw'), ('corrected_acceleration', 'acc_used'),
        ('integral_raw', 'integral_acc_raw'),
        ('bias_body_used', 'bias_body_used'), ('bias_body', 'bias_body_estimated'),
        ('bias_used', 'bias_ned_used'), ('bias', 'bias_ned_estimated'),
        ('bias_rhs', 'bias_fit_rhs'),
    ):
        for axis, suffix in enumerate(('x', 'y')):
            names.append(prefix + '_' + suffix)
            columns.append(result[key][:, axis])
    for axis, suffix in enumerate(('x', 'y')):
        names.append('niacc.' + suffix)
        columns.append(result['velocity'][:, axis])
    for j, axis in enumerate(('x', 'y', 'z', 'w')):
        names.append('quat_' + axis + '_normalized_input')
        columns.append(result['quaternion'][:, j])
    for field in ('yaw_csv_deg', 'yaw_quat_deg', 'yaw_error_deg'):
        names.append(field)
        columns.append(result[field])
    for j, suffix in enumerate(('xx', 'xy', 'yy')):
        names.append('bias_fit_normal_' + suffix)
        columns.append(result['bias_normal'][:, j])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, delimiter=';')
            writer.writerow(names)
            writer.writerows(zip(*columns))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class VelocityViewer(FlightViewer):
    viewer_name = 'Inertial velocity viewer'
    replay_name = 'Windowed velocity replay'
    default_export_name = 'result_velocity.csv'
    panel_titles = ('Vision / forward position', 'Velocity', 'Bias', 'Acceleration')

    def __init__(self, *args, **kwargs):
        from matplotlib.widgets import CheckButtons
        super().__init__(*args, **kwargs)
        self.bias_toggle = CheckButtons(self.fig.add_axes([0.815, 0.735, 0.16, 0.035]),
                                       ['Learn body bias'], [self.cfg.learn_bias])
        self.bias_toggle.on_clicked(self.parameters_edited)

    def parameter_specs(self, cfg):
        return (
            ('fade_in', 'Old-edge fade (s)', cfg.fade_in),
            ('flat', 'Flat part (s)', cfg.flat),
            ('fade_out', 'Recent-edge fade (s)', cfg.fade_out),
            ('lead', 'Forward time (s)', cfg.lead),
            ('coord_scale', 'Coordinate multiplier', cfg.coord_scale),
            ('dt', 'Sample step (s)', cfg.dt),
            ('view_window', 'Plot window (s)', cfg.view_window),
            ('bias_x0', 'Initial body bias X', cfg.bias_x0),
            ('bias_y0', 'Initial body bias Y', cfg.bias_y0),
            ('bias_warmup', 'Bias warmup (s)', cfg.bias_warmup),
        )

    def plot_groups(self, axis):
        groups = list(super().plot_groups(axis))
        groups[0] = [('measured_position', 'Vision x multiplier', '#7e8793', '--'),
                     ('forward_position', 'Vision + lead * speed', '#0072B2', '-')]
        groups[1] = [('velocity', 'Inertial + mean correction', '#009E73', '-'),
                     ('mean_visual_velocity', 'Window mean visual speed', '#CC79A7', '--')]
        return groups

    def parameter_text(self):
        c = self.cfg
        return (f'Active: window {c.fade_in:g} + {c.flat:g} + {c.fade_out:g} = {c.fit_window:g} s; '
                f'lead={c.lead:g} s; scale={c.coord_scale:g}; dt={c.dt:g} s; '
                f'view={self.window:g} s; body bias: {"learning" if c.learn_bias else "fixed"}; '
                f'bias warmup={c.bias_warmup:g} s.')

    def status_details(self, index):
        r = self.result
        state = 'full window' if r['window_complete'][index] else 'warmup'
        vx, vy = r['velocity'][index]
        return f'vx={vx:.3f}, vy={vy:.3f} m/s; {state} ({r["window_span_s"][index]:.2f} s)'

    def config_from_fields(self):
        values = {}
        for key, widget in self.fields.items():
            try:
                values[key] = float(widget.text.strip().replace(',', '.'))
            except ValueError:
                raise ValueError(f'{key}: enter a number.') from None
        values['learn_bias'] = (self.bias_toggle.get_status()[0] if hasattr(self, 'bias_toggle')
                                else self.cfg.learn_bias)
        cfg = replace(self.cfg, **values)
        cfg.validate()
        return cfg

    def run_estimate(self, data, cfg, attitude):
        return estimate_velocity(data, cfg, attitude)

    def save_estimates(self, path, result):
        export_velocity(path, result)

    def after_apply(self):
        self.window = self.cfg.view_window


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, nargs='?', help='CSV; optional in GUI mode')
    for name, default, help_text in (
        ('dt', 0.05, 'Fixed sample step in seconds; CSV timestamps are not used'),
        ('fade-in', 6., 'Fade from zero at the oldest edge, seconds'),
        ('flat', 6., 'Full-weight part of the window, seconds'),
        ('fade-out', 6., 'Fade to zero at the current edge, seconds'),
        ('lead', 1., 'Forward position = scaled vision + this many seconds * speed'),
        ('coord-scale', 1., 'Multiplier on original vis_x/y'),
        ('bias-x0', 0., 'Initial body X accelerometer bias, m/s^2'),
        ('bias-y0', 0., 'Initial body Y accelerometer bias, m/s^2'),
        ('bias-warmup', 6., 'Minimum accumulated information time for body bias, seconds; 0 = immediate fit'),
        ('vx0', 0., 'Initial NED X velocity, m/s'),
        ('vy0', 0., 'Initial NED Y velocity, m/s'),
    ):
        parser.add_argument('--' + name, type=float, default=default, help=help_text)
    parser.add_argument('--window', dest='view_window', type=float, default=10.,
                        help='Visible plot duration in seconds; independent of the estimation window')
    parser.add_argument('--fixed-bias', dest='learn_bias', action='store_false',
                        help='Keep --bias-x0/y0 fixed instead of learning from estimated speed')
    parser.add_argument('--quat-direction', dest='quaternion_direction',
                        choices=('auto', 'body-to-ned', 'ned-to-body'), default='auto')
    parser.add_argument('--yaw-unit', choices=('auto', 'deg', 'rad'), default='auto')
    parser.add_argument('--playback-speed', type=float, default=1.)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--export', type=Path,
                        help='Fixed output path; otherwise result_velocity.csv next to the input')
    parser.add_argument('--plotjuggler', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--no-gui', action='store_true')
    args = parser.parse_args(argv)
    cfg = VelocityConfig(**{key: getattr(args, key) for key in VelocityConfig.__dataclass_fields__})
    try:
        cfg.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.no_gui and args.csv is None:
        parser.error('--no-gui requires an input CSV')
    if not math.isfinite(args.start) or args.start < 0:
        parser.error('--start must be finite and >= 0')
    if not math.isfinite(args.playback_speed) or not 0.0625 <= args.playback_speed <= 64:
        parser.error('--playback-speed must be between 0.0625 and 64')
    args.export_follows_input = args.export is None
    if args.export is None:
        args.export = (args.csv.parent if args.csv else Path.cwd()) / VelocityViewer.default_export_name
    paths = [p.resolve() for p in (args.csv, args.export, args.snapshot) if p is not None]
    if len(set(paths)) != len(paths):
        parser.error('Input, export and snapshot paths must be different')
    return args, cfg


def main(argv=None):
    args, cfg = parse_args(argv)
    try:
        data = attitude = result = None
        if args.csv is not None:
            data, _ = read_flight(args.csv)
            attitude = check_attitude(data, cfg)
            report = attitude.report
            print(f'Quaternion: {report["direction"]}; yaw: {report["yaw_unit"]}; '
                  f'yaw RMS: {report["yaw_rms_deg"]:.6f} deg')
            for warning in report['warnings']:
                print('Frame check: ' + warning, file=sys.stderr)
            result = estimate_velocity(data, cfg, attitude)
            export_velocity(args.export, result)
            print(f'{len(data)} samples; {result["time"][-1]:.2f} s; dt={cfg.dt:g} s; '
                  f'window={cfg.fade_in:g}/{cfg.flat:g}/{cfg.fade_out:g} s; lead={cfg.lead:g} s')
            print(f'Final body bias: {result["bias_body"][-1]} m/s^2. Exported: {args.export}')
        if not args.no_gui or args.snapshot is not None:
            import matplotlib
            if args.no_gui:
                matplotlib.use('Agg')
            elif matplotlib.get_backend().lower() in ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template'):
                raise RuntimeError('No GUI backend. Use a desktop with Tk/Qt, or --no-gui.')
            viewer = VelocityViewer(result, cfg, window=cfg.view_window,
                                    playback_speed=args.playback_speed, start=args.start,
                                    paused=args.paused or args.no_gui,
                                    title=str(args.csv) if args.csv else 'Flight', data=data,
                                    attitude=attitude, input_path=args.csv, export_path=args.export,
                                    export_follows_input=args.export_follows_input,
                                    plotjuggler_path=args.plotjuggler)
            if args.snapshot is not None:
                viewer.fig.savefig(args.snapshot, dpi=130)
                print(f'Snapshot: {args.snapshot}')
            if not args.no_gui:
                viewer.show()
            else:
                viewer.plt.close(viewer.fig)
    except (ValueError, OSError, RuntimeError, OverflowError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
