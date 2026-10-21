#!/usr/bin/env python3
"""Replay horizontal drone navigation from a semicolon-separated CSV.

Install:  python -m pip install numpy matplotlib "tkinterdnd2>=0.6"
Run:      python kalman_flight_viewer.py
          Drop a CSV onto the window, or click Open CSV.
          python kalman_flight_viewer.py height.csv
          Automatically writes result.csv NEXT TO the input CSV.
Example:  python kalman_flight_viewer.py height.csv --sigma-acc 1.0 --start 200
Export:   python kalman_flight_viewer.py height.csv --no-gui --export estimates.csv
Snapshot: python kalman_flight_viewer.py height.csv --no-gui --start 200 --snapshot view.png

Columns: vis_x, vis_y, mheight, acc_x/acc_y (or acc.x/acc.y),
         quat.x/y/z/w, yaw. Optional diagnostics: rawacc.x/y/z, acc.z, G.
Units: m, m/s, m/s^2. Input horizontal acceleration is already in NED.
Quaternion order is x,y,z,w. Yaw units and quaternion direction are checked
against the data; quaternion is authoritative for rotation, yaw is a check.
Override auto detection with --quat-direction or --yaw-unit when necessary.

Model:
  * Independent Kalman filters for X and Y, each with state [position, velocity].
  * The coordinate multiplier (default 1) scales vis_x and vis_y BEFORE filter
    initialization and every measurement update. The plot shows this scaled
    measurement. CSV contains both original vis_x/y and vis_scaled_x/y, plus
    coord_scale. Scaling is always applied to the original data, never twice.
    Height-based position sigma remains controlled by its own UI settings.
  * sigma_position = max(K_height * abs(mheight), min_position_sigma).
    Exy is interpreted as the per-axis STANDARD DEVIATION, so R = Exy^2.
  * sigma_acc is the per-sample acceleration standard deviation. Its independent
    error is held constant within a step: Q = sigma_acc^2 * G * G.T,
    G = [dt^2 / 2, dt]. This is not a continuous noise spectral density.
  * Fixed dt = 0.05 s by default, even when CSV timestamps differ.
  * First CSV row initializes x/y; vx/vy start at zero unless overridden.
    Row i's acceleration represents the interval ((i-1)*dt, i*dt].
  * Bias is stored in BODY axes: [bx, by, 0]. This horizontal-only estimator
    assumes body-Z accelerometer bias is zero; it does not estimate it.
    R_i maps body -> NED using the full quaternion (roll, pitch AND yaw).
    M_i = R_i[:2, :2] maps body bx/by into horizontal NED acceleration.
    Step i uses the previous body estimate, rotated with CURRENT attitude:
        bias_ned_used = M_i @ bias_body_previous
        a_used = a_ned_raw - bias_ned_used
    After Kalman prediction and position correction, form a NED increment:
        delta_v = estimated_velocity_i - estimated_velocity_previous
        residual = a_ned_raw * dt - delta_v
    Accumulate a constant-body-bias least-squares estimate OUTSIDE Kalman:
        normal += dt * M_i.T @ M_i
        rhs    += M_i.T @ residual
        bias_body = solve(normal, rhs)
    Unobservable body directions keep their previous estimate (eigenvalue test).
    For yaw-only motion M is orthogonal, hence normal=t*I and this reduces to
        bias_body = sum(M_i.T @ (a_ned_raw_i*dt - delta_v_i)) / t.
    For a fixed identity attitude it reproduces the old (integral(acc)-dv)/t.
    Row 0 initializes state and attitude; it does not integrate or update bias.
  * Bias is a cumulative estimate over the WHOLE elapsed flight. Moving/scrolling
    the 10-second view never resets it. Bias uncertainty is not a Kalman state
    and is not included in P: this remains an external adaptation, not a joint
    optimal Kalman estimator. Filter velocity errors can influence learned bias.
  * Do NOT subtract current body velocity from an integral of body acceleration:
    d(v_body)/dt = R.T @ a_ned - omega_body cross v_body. Transforming each NED
    velocity INCREMENT avoids this rotating-frame derivative term, without gyro
    differentiation. For full tilt, the 2x2 normal equations avoid inventing an
    unavailable vertical velocity. acc.z and height derivatives are not fused.

Frame conventions are diagnosed from the file before filtering. With these
conventions fixed, the filter is evaluated in one forward-only, causal pass
before playback so seeking is immediate; estimates do not use future samples.
Charts replay the results at real time (1x) by default. Slower rendering can
skip displayed frames but never skips a filter update.

Controls: Space = pause/play; Left/Right = seek 1 s; Home/End = beginning/end;
          +/- = slower/faster playback; bottom slider = seek and pause.
The top fields edit acceleration sigma, initial speed sigma, minimum position
sigma, height error in PERCENT (10 means K_height=0.1), and coordinate multiplier.
All uncertainty inputs are STANDARD DEVIATIONS, not variances.
Apply + export recalculates the WHOLE flight from row 0, updates all plots,
and replaces result.csv (or the path passed with --export). The playback
position is preserved and playback is paused. Editing fields alone does not
change estimates or the CSV. Invalid values leave the previous result intact.
Open CSV or dropping one CSV anywhere on the plot window selects another flight
using the current fields, checks its quaternion
and yaw, and recalculates from row 0. Playback starts paused at t=0. By default,
result.csv is saved next to the newly opened file; an explicit --export path
stays fixed. Cancel or a failed load keeps the previous flight available.
The initial CSV argument is optional in GUI mode. Until a file is loaded,
playback/export are inactive, parameters remain editable, and no CSV is written.
Open in PlotJuggler applies the current fields, exports the result, and starts
PlotJuggler with -d and the absolute CSV path. It searches the saved path, PATH,
Windows installation registry entries and common installation directories.
If needed, select the installed executable once; the path is remembered in
the user's settings directory. --plotjuggler PATH overrides automatic lookup.
Drag/drop works with TkAgg (tkinterdnd2 >= 0.6) and QtAgg (PySide/PyQt).
File paths with spaces and non-ASCII characters are supported. Drop one file
at a time; the source is opened, never moved. --no-gui requires an input CSV.
The eight plots are X/Y rows and position/velocity/bias/acceleration columns.
Bias plots show both body components and their current NED projections.
At the end, playback stops and the window remains open for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass
class Config:
    dt: float = 0.05
    k_height: float = 0.1
    sigma_acc: float = 1.0
    min_pos_sigma: float = 0.001
    initial_vel_sigma: float = 1.0
    coord_scale: float = 1.0
    vx0: float = 0.0
    vy0: float = 0.0
    quaternion_direction: str = 'auto'
    yaw_unit: str = 'auto'


@dataclass
class Attitude:
    rotation: np.ndarray       # (n,3,3), body -> NED
    quaternion: np.ndarray     # Normalized input, x/y/z/w order.
    yaw_csv_deg: np.ndarray
    yaw_quat_deg: np.ndarray
    norm: np.ndarray
    report: dict


def quaternion_matrices(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Active right-handed rotation; q=(x,y,z,w)."""
    norm = np.linalg.norm(q, axis=1)
    if not np.isfinite(norm).all() or np.any(norm < 1e-8):
        raise ValueError('Quaternion has a zero/non-finite norm.')
    q = q / norm[:, None]
    x, y, z, w = q.T
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y*y + z*z)
    r[:, 0, 1] = 2 * (x*y - z*w)
    r[:, 0, 2] = 2 * (x*z + y*w)
    r[:, 1, 0] = 2 * (x*y + z*w)
    r[:, 1, 1] = 1 - 2 * (x*x + z*z)
    r[:, 1, 2] = 2 * (y*z - x*w)
    r[:, 2, 0] = 2 * (x*z - y*w)
    r[:, 2, 1] = 2 * (y*z + x*w)
    r[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return r, norm


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def check_attitude(data: np.ndarray, cfg: Config) -> Attitude:
    """Cross-check quaternion direction, yaw units, and optional raw IMU data.

    Required data columns 0:10: vis_x/y, height, acc_x/y, qx/qy/qz/qw, yaw.
    Optional diagnostic columns 10:15: rawacc_x/y/z, acc_z, G (NaN if absent).
    """
    if data.shape[1] < 10:
        raise ValueError('Body bias requires quaternion x/y/z/w and yaw columns.')
    rotation, norm = quaternion_matrices(data[:, 5:9])
    candidates = {'body-to-ned': rotation, 'ned-to-body': rotation.transpose(0, 2, 1)}
    diagnostics = {}
    for name, r in candidates.items():
        yq = np.rad2deg(np.arctan2(r[:, 1, 0], r[:, 0, 0]))
        scores = {unit: float(np.sqrt(np.mean(wrap_degrees(yq - yaw)**2)))
                  for unit, yaw in (('deg', data[:, 9]), ('rad', np.rad2deg(data[:, 9])))}
        unit = min(scores, key=scores.get) if cfg.yaw_unit == 'auto' else cfg.yaw_unit
        diagnostics[name] = {'yaw_unit': unit, 'yaw_rms_deg': scores[unit],
                             'yaw_unit_scores': scores}
        if data.shape[1] >= 13:
            valid = np.isfinite(data[:, 10:13]).all(axis=1)
            if np.count_nonzero(valid) >= 2:
                rotated = np.einsum('nij,nj->ni', r[valid], data[valid, 10:13])
                error = rotated[:, :2] - data[valid, 3:5]
                diagnostics[name]['raw_horizontal_rms'] = float(np.sqrt(np.mean(error**2)))
                diagnostics[name]['raw_horizontal_rms_axes'] = np.sqrt(np.mean(error**2, axis=0)).tolist()
    warnings = []
    if cfg.quaternion_direction == 'auto':
        # Prefer a direct vector reconstruction when it strongly discriminates.
        raw_names = [name for name in candidates if 'raw_horizontal_rms' in diagnostics[name]]
        if len(raw_names) == 2:
            best = min(raw_names, key=lambda name: diagnostics[name]['raw_horizontal_rms'])
            other = next(name for name in candidates if name != best)
            lo, hi = diagnostics[best]['raw_horizontal_rms'], diagnostics[other]['raw_horizontal_rms']
            decisive_raw = lo < 0.25 and hi > max(2.0 * lo, lo + 0.001)
        else:
            decisive_raw = False
        if decisive_raw:
            direction, evidence = best, 'rawacc horizontal reconstruction'
        else:
            direction = min(candidates, key=lambda name: diagnostics[name]['yaw_rms_deg'])
            other = next(name for name in candidates if name != direction)
            best_score = diagnostics[direction]['yaw_rms_deg']
            if diagnostics[other]['yaw_rms_deg'] - best_score < max(0.05, 0.2 * best_score):
                direction = 'body-to-ned'
                warnings.append('Quaternion direction is weakly identified; assuming body-to-ned. '
                                'Use --quat-direction if your convention differs.')
            evidence = 'yaw consistency'
    else:
        direction, evidence = cfg.quaternion_direction, 'explicit command-line setting'
    r = candidates[direction]
    unit = diagnostics[direction]['yaw_unit']
    yaw_csv = data[:, 9].copy() if unit == 'deg' else np.rad2deg(data[:, 9])
    yaw_quat = np.rad2deg(np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    yaw_error = wrap_degrees(yaw_quat - yaw_csv)
    report = {'direction': direction, 'direction_evidence': evidence, 'yaw_unit': unit,
              'yaw_rms_deg': float(np.sqrt(np.mean(yaw_error**2))),
              'yaw_max_deg': float(np.max(np.abs(yaw_error))),
              'quaternion_norm_min': float(norm.min()), 'quaternion_norm_max': float(norm.max()),
              'candidates': diagnostics, 'warnings': warnings}
    if report['yaw_rms_deg'] > 2.0:
        warnings.append('Quaternion/yaw mismatch exceeds 2 degrees RMS; quaternion is used for rotation.')
    raw_rms = diagnostics[direction].get('raw_horizontal_rms')
    if raw_rms is not None:
        report['raw_horizontal_rms'] = raw_rms
        if raw_rms > 0.25:
            warnings.append('Rotated rawacc differs from acc.x/y; check frames, units and timing.')
    if np.max(np.abs(norm - 1)) > 0.02:
        warnings.append('Quaternion norms differ from 1 by more than 2%; they have been normalized.')
    if data.shape[1] >= 15:
        valid = np.isfinite(data[:, 10:15]).all(axis=1)
        if np.count_nonzero(valid) >= 2:
            rz = np.einsum('ni,ni->n', r[valid, 2, :], data[valid, 10:13])
            az, g = data[valid, 13], data[valid, 14]
            report['acc_z_ned_relation_rms'] = float(np.sqrt(np.mean((az - (rz + g))**2)))
            report['acc_z_up_relation_rms'] = float(np.sqrt(np.mean((az + rz + g)**2)))
    return Attitude(r, data[:, 5:9] / norm[:, None], yaw_csv, yaw_quat, norm, report)


class BodyBiasEstimator:
    """Cumulative external least squares for body bx/by, with body bz fixed to 0."""
    def __init__(self):
        self.normal = np.zeros((2, 2))
        self.rhs = np.zeros(2)
        self.bias = np.zeros(2)

    def update(self, m, raw_acc_ned, delta_velocity_ned, dt):
        residual = raw_acc_ned * dt - delta_velocity_ned
        self.normal += dt * (m.T @ m)
        self.rhs += m.T @ residual
        # Preserve the old estimate in unobservable directions instead of
        # creating a huge bias when the vehicle is near a 90-degree tilt.
        eigenvalues, vectors = np.linalg.eigh(self.normal)
        observable = eigenvalues > max(float(eigenvalues[-1]) * 1e-8, 1e-12)
        coordinates = vectors.T @ self.bias
        rhs_coordinates = vectors.T @ self.rhs
        coordinates[observable] = rhs_coordinates[observable] / eigenvalues[observable]
        self.bias = vectors @ coordinates
        return self.bias.copy()


@dataclass
class Kalman1D:
    x: float
    v: float
    pxx: float
    pxv: float
    pvv: float

    def step(self, a: float, z: float, r: float, dt: float,
             sigma_acc: float) -> tuple[float, float]:
        """Predict and correct once; return K_position and K_velocity."""
        # Predict state. Use the OLD velocity in the position prediction.
        self.x += self.v * dt + 0.5 * a * dt * dt
        self.v += a * dt

        # Predict covariance. Every RHS uses the OLD covariance.
        q = sigma_acc * sigma_acc
        pxx = self.pxx + 2.0 * dt * self.pxv + dt**2 * self.pvv + 0.25 * dt**4 * q
        pxv = self.pxv + dt * self.pvv + 0.5 * dt**3 * q
        pvv = self.pvv + dt**2 * q

        # Correct both position and velocity with the SAME position innovation.
        innovation = z - self.x
        s = pxx + r
        kx = pxx / s
        kv = pxv / s
        self.x += kx * innovation
        self.v += kv * innovation

        # Joseph covariance update: (I-KH) P (I-KH)^T + K R K^T.
        # Equivalent to P - K H P in exact arithmetic, with less cancellation.
        h = r / s  # 1 - kx, evaluated without subtracting almost equal numbers.
        self.pxx = h * h * pxx + kx * kx * r
        self.pxv = h * (pxv - kv * pxx) + kx * kv * r
        self.pvv = pvv - 2.0 * kv * pxv + kv * kv * (pxx + r)
        return kx, kv


def read_flight(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Strict reader: do not drop bad rows and silently change the time base."""
    aliases = {
        'vis_x': ('vis_x',), 'vis_y': ('vis_y',), 'mheight': ('mheight',),
        'acc_x': ('acc_x', 'acc.x'), 'acc_y': ('acc_y', 'acc.y'),
        'quat_x': ('quat.x', 'quat_x'), 'quat_y': ('quat.y', 'quat_y'),
        'quat_z': ('quat.z', 'quat_z'), 'quat_w': ('quat.w', 'quat_w'),
        'yaw': ('yaw',),
        'rawacc_x': ('rawacc.x', 'rawacc_x'),
        'rawacc_y': ('rawacc.y', 'rawacc_y'),
        'rawacc_z': ('rawacc.z', 'rawacc_z'),
        'acc_z': ('acc.z', 'acc_z'), 'G': ('G',),
    }
    optional = {'rawacc_x', 'rawacc_y', 'rawacc_z', 'acc_z', 'G'}
    values = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            raise ValueError('CSV is empty.') from None
        selected = {}
        indices = []
        for key, names in aliases.items():
            found = next((name for name in names if name in header), None)
            if found is None:
                if key in optional:
                    selected[key] = ''
                    indices.append(None)
                    continue
                raise ValueError(f'Missing CSV column: {" or ".join(names)}')
            if header.count(found) != 1:
                raise ValueError(f'Duplicate CSV column: {found}')
            selected[key] = found
            indices.append(header.index(found))
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            parsed = []
            for key, column in zip(aliases, indices):
                if column is None:
                    parsed.append(float('nan'))
                    continue
                try:
                    # Also accept a decimal comma in this semicolon format.
                    number = float(row[column].strip().replace(',', '.'))
                except (ValueError, IndexError):
                    if key in optional:
                        parsed.append(float('nan'))
                        continue
                    raise ValueError(f'CSV line {reader.line_num}: invalid {selected[key]}') from None
                if not math.isfinite(number) and key not in optional:
                    raise ValueError(f'CSV line {reader.line_num}: non-finite {selected[key]}')
                parsed.append(number)
            values.append(parsed)
    if len(values) < 2:
        raise ValueError('At least two data rows are required.')
    return np.asarray(values, dtype=float), selected


def estimate(data: np.ndarray, cfg: Config, attitude: Attitude | None = None) -> dict[str, np.ndarray]:
    """Causal NED state filter with a separately accumulated body-frame bias."""
    if attitude is None:
        attitude = check_attitude(data, cfg)
    n = len(data)
    rotation = attitude.rotation
    if len(rotation) != n:
        raise ValueError('Attitude and data row counts do not match.')
    t = np.arange(n, dtype=float) * cfg.dt
    sigma = np.maximum(cfg.k_height * np.abs(data[:, 2]), cfg.min_pos_sigma)
    if not math.isfinite(cfg.coord_scale) or cfg.coord_scale <= 0:
        raise ValueError('Coordinate multiplier must be finite and > 0.')
    measured_position = data[:, :2] * cfg.coord_scale
    v0 = np.array([cfg.vx0, cfg.vy0], dtype=float)
    filters = [Kalman1D(measured_position[0, j], v0[j], sigma[0]**2, 0.0,
                        cfg.initial_vel_sigma**2) for j in range(2)]
    out = {
        'time': t, 'height': data[:, 2], 'sigma_position': sigma,
        'raw_position': data[:, :2], 'measured_position': measured_position,
        'coord_scale': np.full(n, cfg.coord_scale), 'raw_acceleration': data[:, 3:5],
        'position': np.zeros((n, 2)), 'velocity': np.zeros((n, 2)),
        'bias': np.zeros((n, 2)), 'bias_used': np.zeros((n, 2)),
        'bias_body': np.zeros((n, 2)), 'bias_body_used': np.zeros((n, 2)),
        'corrected_acceleration': np.zeros((n, 2)),
        'integral_raw': np.zeros((n, 2)),
        'bias_normal': np.zeros((n, 3)), 'bias_rhs': np.zeros((n, 2)),
        'quaternion': attitude.quaternion,
        'yaw_csv_deg': attitude.yaw_csv_deg,
        'yaw_quat_deg': attitude.yaw_quat_deg,
        'yaw_error_deg': wrap_degrees(attitude.yaw_quat_deg - attitude.yaw_csv_deg),
        'covariance': np.zeros((n, 2, 3)),  # Pxx, Pxv, Pvv, for each axis.
        'gain': np.zeros((n, 2, 2)),         # Kx, Kv, for each axis.
    }
    out['position'][0] = measured_position[0]
    out['velocity'][0] = v0
    out['corrected_acceleration'][0] = data[0, 3:5]
    for axis, kf in enumerate(filters):
        out['covariance'][0, axis] = kf.pxx, kf.pxv, kf.pvv
    integral_raw = np.zeros(2)
    bias_estimator = BodyBiasEstimator()
    for i in range(1, n):
        raw_acc = data[i, 3:5]
        # A body-fixed bias projects differently into NED as attitude changes.
        # This is the horizontal projection of R_i @ [bx, by, 0].
        m = rotation[i, :2, :2]
        out['bias_body_used'][i] = bias_estimator.bias
        out['bias_used'][i] = m @ bias_estimator.bias
        corrected_acc = raw_acc - out['bias_used'][i]
        integral_raw += raw_acc * cfg.dt
        for axis, kf in enumerate(filters):
            out['gain'][i, axis] = kf.step(
                corrected_acc[axis], measured_position[i, axis], sigma[i]**2,
                cfg.dt, cfg.sigma_acc)
            out['position'][i, axis] = kf.x
            out['velocity'][i, axis] = kf.v
            out['covariance'][i, axis] = kf.pxx, kf.pxv, kf.pvv
        # Difference the velocities in NED BEFORE transporting the residual.
        # R_i.T @ v_i - R_(i-1).T @ v_(i-1) would include axis rotation!
        delta_velocity = out['velocity'][i] - out['velocity'][i - 1]
        body_bias = bias_estimator.update(m, raw_acc, delta_velocity, cfg.dt)
        out['bias_body'][i] = body_bias
        out['bias'][i] = m @ body_bias  # Current NED projection of NEW estimate.
        normal = bias_estimator.normal
        out['bias_normal'][i] = normal[0, 0], normal[0, 1], normal[1, 1]
        out['bias_rhs'][i] = bias_estimator.rhs
        out['corrected_acceleration'][i] = corrected_acc
        out['integral_raw'][i] = integral_raw
    for key in ('position', 'velocity', 'bias', 'bias_body', 'covariance', 'gain'):
        if not np.isfinite(out[key]).all():
            raise ValueError(f'Filter became non-finite: {key}; check inputs and noise settings.')
    return out


def export_estimates(path: Path, result: dict[str, np.ndarray]) -> None:
    names = ['time_s', 'mheight', 'sigma_position', 'coord_scale']
    columns = [result['time'], result['height'], result['sigma_position'], result['coord_scale']]
    for field, prefix in (
        ('raw_position', 'vis'), ('measured_position', 'vis_scaled'),
        ('position', 'estimated_position'),
        ('velocity', 'estimated_speed'), ('raw_acceleration', 'acc_raw'),
        ('bias_used', 'bias_ned_used'), ('corrected_acceleration', 'acc_used'),
        ('bias', 'bias_ned_estimated'), ('integral_raw', 'integral_acc_raw'),
        ('bias_body', 'bias_body_estimated'), ('bias_body_used', 'bias_body_used'),
        ('bias_rhs', 'bias_fit_rhs'),
    ):
        for j, axis in enumerate(('x', 'y')):
            names.append(f'{prefix}_{axis}')
            columns.append(result[field][:, j])
    for j, axis in enumerate(('x', 'y', 'z', 'w')):
        names.append(f'quat_{axis}_normalized_input')
        columns.append(result['quaternion'][:, j])
    for field in ('yaw_csv_deg', 'yaw_quat_deg', 'yaw_error_deg'):
        names.append(field)
        columns.append(result[field])
    for j, label in enumerate(('xx', 'xy', 'yy')):
        names.append('bias_fit_normal_' + label)
        columns.append(result['bias_normal'][:, j])
    for j, axis in enumerate(('x', 'y')):
        for k, label in enumerate(('Pxx', 'Pxv', 'Pvv')):
            names.append(f'{axis}_{label}')
            columns.append(result['covariance'][:, j, k])
        for k, label in enumerate(('K_position', 'K_velocity')):
            names.append(f'{axis}_{label}')
            columns.append(result['gain'][:, j, k])
    # Atomic replacement: an error must not leave a half-written result.csv.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            writer = csv.writer(f, delimiter=';')
            writer.writerow(names)
            writer.writerows(zip(*columns))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def viewer_settings_path() -> Path:
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'KalmanFlightViewer' / 'settings.json'


def saved_plotjuggler_path() -> str | None:
    try:
        settings = json.loads(viewer_settings_path().read_text(encoding='utf-8'))
        value = settings.get('plotjuggler_path') if isinstance(settings, dict) else None
        return value if isinstance(value, str) else None
    except (OSError, ValueError):
        return None


def save_plotjuggler_path(path: Path) -> None:
    settings_path = viewer_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({'plotjuggler_path': str(path)}, indent=2) + '\n',
                             encoding='utf-8')


def resolve_plotjuggler_executable(value) -> Path | None:
    if not value:
        return None
    try:
        path = Path(os.path.expandvars(str(value).strip().strip('"'))).expanduser()
        candidates = ([path / 'Contents' / 'MacOS' / name
                       for name in ('plotjuggler', 'PlotJuggler')]
                      if path.suffix.lower() == '.app' else [path])
        for candidate in candidates:
            if candidate.is_file() and (
                    candidate.suffix.lower() == '.exe' if sys.platform == 'win32'
                    else os.access(candidate, os.X_OK)):
                return candidate.resolve()
    except (OSError, ValueError):
        pass
    return None


def windows_plotjuggler_locations():
    """Read only PlotJuggler's App Paths / uninstall entries, including 32-bit views."""
    if sys.platform != 'win32':
        return
    import winreg
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, r'Software\Microsoft\Windows\CurrentVersion\App Paths\plotjuggler.exe',
                                    0, winreg.KEY_READ | view) as key:
                    yield winreg.QueryValueEx(key, '')[0]
            except OSError:
                pass
            try:
                with winreg.OpenKey(hive, r'Software\Microsoft\Windows\CurrentVersion\Uninstall',
                                    0, winreg.KEY_READ | view) as parent:
                    for i in range(winreg.QueryInfoKey(parent)[0]):
                        try:
                            with winreg.OpenKey(parent, winreg.EnumKey(parent, i)) as key:
                                name = str(winreg.QueryValueEx(key, 'DisplayName')[0])
                                if 'plotjuggler' not in name.lower():
                                    continue
                                for field in ('InstallLocation', 'DisplayIcon'):
                                    try:
                                        value = str(winreg.QueryValueEx(key, field)[0])
                                        if field == 'DisplayIcon':
                                            value = value.rsplit(',', 1)[0].strip('"')
                                        if value:
                                            yield value
                                    except OSError:
                                        pass
                        except OSError:
                            continue
            except OSError:
                pass


def common_plotjuggler_locations():
    if sys.platform == 'win32':
        for value in windows_plotjuggler_locations():
            root = Path(os.path.expandvars(str(value).strip('"')))
            yield root
            yield root / 'plotjuggler.exe'
            yield root / 'bin' / 'plotjuggler.exe'
        roots = [Path(os.environ.get(name) or default) for name, default in (
            ('ProgramFiles', r'C:\Program Files'), ('ProgramFiles(x86)', r'C:\Program Files (x86)'),
            ('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local')))]
        roots += [roots[-1] / 'Programs', Path.home(), Path.home() / 'Downloads',
                  Path(os.environ.get('SystemDrive', 'C:') + '/')]
        patterns = ('PlotJuggler*/plotjuggler.exe', 'PlotJuggler*/bin/plotjuggler.exe')
    elif sys.platform == 'darwin':
        roots = [Path('/Applications'), Path.home() / 'Applications']
        patterns = ('PlotJuggler*.app',)
        for root in (Path('/opt/homebrew/bin'), Path('/usr/local/bin')):
            yield root / 'plotjuggler'
    else:
        for root in (Path('/usr/bin'), Path('/usr/local/bin'), Path('/snap/bin'),
                     Path.home() / '.local' / 'bin'):
            yield root / 'plotjuggler'
        roots = [Path('/opt'), Path.home() / 'Applications', Path.home() / 'Downloads']
        patterns = ('PlotJuggler*/bin/plotjuggler', 'PlotJuggler*/plotjuggler',
                    '*[Pp]lot[Jj]uggler*.AppImage', 'ros/*/lib/plotjuggler/plotjuggler')
    # Bounded directory patterns: do not crawl the whole disk or block the UI for minutes.
    for root in roots:
        for pattern in patterns:
            try:
                yield from sorted(root.glob(pattern), reverse=True)
            except OSError:
                continue


def find_plotjuggler(preferred=None) -> Path | None:
    for value in (preferred, os.environ.get('PLOTJUGGLER_PATH'), saved_plotjuggler_path()):
        found = resolve_plotjuggler_executable(value)
        if found is not None:
            return found
    for command in ('plotjuggler', 'PlotJuggler', 'plotjuggler.exe'):
        found = resolve_plotjuggler_executable(shutil.which(command))
        if found is not None:
            return found
    for value in common_plotjuggler_locations():
        found = resolve_plotjuggler_executable(value)
        if found is not None:
            return found
    return None


class FlightViewer:
    viewer_name = 'Kalman flight viewer'
    replay_name = 'Kalman replay'
    default_export_name = 'result.csv'
    panel_titles = ('Position', 'Velocity', 'Bias', 'Acceleration')

    def parameter_specs(self, cfg):
        return (
            ('sigma_acc', 'Accel sigma (m/s^2)', cfg.sigma_acc),
            ('initial_vel_sigma', 'Initial speed sigma (m/s)', cfg.initial_vel_sigma),
            ('min_pos_sigma', 'Min position sigma (m)', cfg.min_pos_sigma),
            ('height_percent', 'Height error (%)', 100.0 * cfg.k_height),
            ('coord_scale', 'Coordinate multiplier', cfg.coord_scale),
        )

    def plot_groups(self, axis):
        return (
            [('measured_position', 'Vision x multiplier', '#7e8793', '--'),
             ('position', 'Kalman', '#0072B2', '-')],
            [('velocity', 'Estimated speed', '#009E73', '-')],
            [('bias_body', 'Body ' + ('X' if axis == 0 else 'Y'), '#D55E00', '-'),
             ('bias', 'NED ' + ('X' if axis == 0 else 'Y'), '#0072B2', '--')],
            [('raw_acceleration', 'Raw', '#8c9199', '--'),
             ('corrected_acceleration', 'Raw - bias used', '#8e44ad', '-')],
        )

    def parameter_text(self):
        return (f'Active: sigma_pos = max({self.cfg.min_pos_sigma:g} m, '
                f'{100 * self.cfg.k_height:g}% of height); '
                f'sigma_acc={self.cfg.sigma_acc:g} m/s^2; '
                f'sigma_v0={self.cfg.initial_vel_sigma:g} m/s. '
                f'coord_scale={self.cfg.coord_scale:g}; '
                f'dt={self.cfg.dt:g} s; window={self.window:g} s.')

    def status_details(self, index):
        return f'sigma_position={self.result["sigma_position"][index]:.2f} m'

    def run_estimate(self, data, cfg, attitude):
        return estimate(data, cfg, attitude)

    def save_estimates(self, path, result):
        export_estimates(path, result)

    def after_apply(self):
        """Hook for replay modes with editable display/time settings."""

    def __init__(self, result, cfg, window=10.0, playback_speed=1.0,
                 start=0.0, paused=False, title='Flight', data=None,
                 export_path=None, attitude=None, input_path=None,
                 export_follows_input=False, plotjuggler_path=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        from matplotlib.widgets import Button, Slider
        from viewer_widgets import ResponsiveTextBox

        self.plt = plt
        self.result, self.cfg, self.window = result, cfg, window
        if result is not None and data is None:
            raise ValueError('Original CSV data including quaternion/yaw is required by the viewer.')
        self.data = data
        self.attitude = attitude if attitude is not None else (
            check_attitude(data, cfg) if data is not None else None)
        self.export_path = Path(export_path) if export_path is not None else Path(self.default_export_name)
        self.input_path = Path(input_path).resolve() if input_path is not None else None
        self.input_paths = {self.input_path} if self.input_path is not None else set()
        self.export_follows_input = export_follows_input
        self.plotjuggler_path = plotjuggler_path
        self.external_processes = []
        self.duration = float(result['time'][-1]) if result is not None else 0.0
        self._drop_pending = False
        self._closed = False
        self.playback_speed = playback_speed
        self.current_time = min(max(start, 0.0), self.duration)
        self.paused = paused or self.current_time >= self.duration
        self.last_clock = time.perf_counter()
        self.last_index = -1
        specs = self.parameter_specs(cfg)
        extra_rows = (len(specs) - 1) // 5
        controls_shift = 0.09 * extra_rows
        self.fig, self.axes = plt.subplots(2, 4, figsize=(17, 9.5 + extra_rows), sharex=True)
        self.fig.subplots_adjust(left=0.065, right=0.985, top=0.745 - controls_shift,
                                 bottom=0.20, wspace=0.33, hspace=0.32)
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(self.viewer_name)
        self.fig.suptitle(Path(title).name + ' - ' + self.replay_name if result is not None
                          else self.viewer_name, y=0.98, fontsize=16)
        self.status = self.fig.text(0.065, 0.935, '', fontsize=11)
        self.parameter_summary = self.fig.text(0.065, 0.812 - controls_shift, '', fontsize=10, color='#555555')
        self.message = self.fig.text(0.065, 0.782 - controls_shift, '', fontsize=10)
        self.fields = {}
        for number, (key, label, value) in enumerate(specs):
            row, col = divmod(number, 5)
            left = 0.065 + 0.148 * col
            self.fig.text(left, 0.89 - row * 0.09, label, fontsize=10)
            widget = ResponsiveTextBox(
                self.fig.add_axes([left, 0.842 - row * 0.09, 0.128, 0.035]),
                '', initial=f'{value:g}')
            self.fields[key] = widget
            widget.on_text_change(self.parameters_edited)
        self.apply_button = Button(self.fig.add_axes([0.815, 0.84, 0.16, 0.045]),
                                   'Apply + export')
        self.apply_button.on_clicked(self.apply_parameters)
        self.open_button = Button(self.fig.add_axes([0.815, 0.89, 0.16, 0.04]), 'Open CSV')
        self.open_button.on_clicked(self.open_csv)
        self.plotjuggler_button = Button(self.fig.add_axes([0.815, 0.788, 0.16, 0.04]),
                                        'Open in PlotJuggler')
        self.plotjuggler_button.on_clicked(self.open_in_plotjuggler)
        if self.attitude is not None:
            frame_report = self.attitude.report
            self.message.set_text(
                f'CSV: {self.export_path.name}. Bias: body XY (body Z=0). '
                f'Quaternion: {frame_report["direction"]}; yaw: {frame_report["yaw_unit"]}; '
                f'RMS difference: {frame_report["yaw_rms_deg"]:.3f} deg.')
        else:
            self.message.set_text('Set parameters above, then open or drop a flight CSV.')
        self.bindings = []
        for j, direction in enumerate(('X / North', 'Y / East')):
            groups = self.plot_groups(j)
            for col, (title_text, unit, traces) in enumerate(zip(
                self.panel_titles,
                ('m', 'm/s', 'm/s^2', 'm/s^2'), groups)):
                ax = self.axes[j, col]
                ax.set_title(f'{direction}: {title_text}' if col != 2
                             else f'Bias: body/NED {"X" if j == 0 else "Y"}', fontsize=11)
                ax.set_ylabel(unit)
                if j == 1:
                    ax.set_xlabel('Time (s)')
                ax.grid(True, alpha=0.22)
                ax.tick_params(labelsize=9)
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_major_locator(MaxNLocator(5))
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(formatter)
                pairs = []
                for field, label, color, style in traces:
                    line, = ax.plot([], [], label=label, color=color,
                                    linestyle=style, linewidth=1.6)
                    # Resolve against self.result on each draw after recalculation.
                    pairs.append((line, field, j))
                ax.legend(loc='upper left', fontsize=8, framealpha=0.75)
                self.bindings.append((ax, pairs, (0.10, 0.10, 0.01, 0.10)[col]))
        self.slider = Slider(self.fig.add_axes([0.22, 0.095, 0.65, 0.035]),
                             'Time (s)', 0.0, max(self.duration, self.cfg.dt),
                             valinit=self.current_time, valfmt='%0.2f')
        self.button = Button(self.fig.add_axes([0.065, 0.085, 0.085, 0.05]), 'Pause')
        self.button.on_clicked(self.toggle_pause)
        self.slider.on_changed(self.seek)
        self.fig.text(0.065, 0.025,
                      'Space: pause/play   Left/Right: +/-1 s   Home/End: seek   '
                      '+/-: playback speed   Slider: seek and pause', fontsize=10)
        self.drop_status = self.fig.text(0.065, 0.057, '', fontsize=10, color='#555555')
        self.empty_hint = self.fig.text(
            0.525, 0.48, 'Drop a CSV file here\nor click Open CSV\n\n'
            'Current parameters will be used for the new flight.',
            ha='center', va='center', fontsize=19, color='#555555')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.timer = self.fig.canvas.new_timer(interval=25)
        self.timer.add_callback(self.tick)
        self.set_loaded_controls()
        self.enable_file_drop()
        self.draw(force=True)

    def set_loaded_controls(self):
        loaded = self.result is not None
        self.empty_hint.set_visible(not loaded)
        for ax in self.axes.flat:
            ax.set_visible(loaded)
        for control in (self.slider, self.button, self.apply_button, self.plotjuggler_button):
            control.set_active(loaded)
            control.ax.set_alpha(1.0 if loaded else 0.35)
            control.label.set_color('black' if loaded else '#aaaaaa')

    def draw(self, force=False):
        result = self.result
        self.parameter_summary.set_text(self.parameter_text())
        if result is None:
            self.status.set_text('No CSV loaded')
            self.button.label.set_text('Play')
            self.fig.canvas.draw_idle()
            return
        index = min(int((self.current_time + 1e-9) / self.cfg.dt), len(result['time']) - 1)
        if not force and index == self.last_index:
            return
        self.last_index = index
        # Keep a full-width window, with the current sample at its right edge.
        right = float(result['time'][index])
        left = right - self.window
        begin = int(np.searchsorted(result['time'], left - 1e-9))
        selection = slice(begin, index + 1)
        times = result['time'][selection]
        self.axes[0, 0].set_xlim(left, right)
        for ax, pairs, min_span in self.bindings:
            segments = []
            for line, field, axis in pairs:
                segment = result[field][selection, axis]
                line.set_data(times, segment)
                segments.append(segment)
            lo = min(float(np.min(v)) for v in segments)
            hi = max(float(np.max(v)) for v in segments)
            span = max(hi - lo, min_span)
            center = 0.5 * (lo + hi)
            # Headroom keeps lines clear of the legend.
            ax.set_ylim(center - 0.65 * span, center + 0.95 * span)
        self.update_playback_status()
        # Updating the display must not trigger the seek callback or extra draws.
        self.slider.eventson = False
        self.slider.drawon = False
        self.slider.set_val(right)
        self.slider.eventson = True
        self.slider.drawon = True
        self.fig.canvas.draw_idle()

    def update_playback_status(self):
        if self.result is None:
            self.status.set_text('No CSV loaded')
        else:
            index = min(int((self.current_time + 1e-9) / self.cfg.dt),
                        len(self.result['time']) - 1)
            mode = 'Paused' if self.paused else 'Playing'
            if index == len(self.result['time']) - 1:
                mode = 'End'
            self.status.set_text(
                f'{mode}   {self.playback_speed:g}x    '
                f't={self.result["time"][index]:.2f} / {self.duration:.2f} s    '
                f'height={self.result["height"][index]:.2f} m    '
                + self.status_details(index))
        self.button.label.set_text('Play' if self.paused else 'Pause')

    def parameters_edited(self, _text):
        # Keep plots and the slider untouched while typing, including when
        # already paused. Only Apply triggers computation and disk writes.
        from viewer_widgets import redraw_controls
        old_bounds = [artist.get_window_extent().frozen()
                      for artist in (self.status, self.message)]
        self.paused = True
        self.message.set_color('#555555')
        self.message.set_text('Parameters edited. Click Apply + export to recalculate and save.'
                              if self.result is not None else
                              'Parameters will be applied when you open or drop a CSV.')
        if hasattr(self, 'button'):
            self.update_playback_status()
            redraw_controls(self.fig, [self.status, self.message, self.button.ax], old_bounds)

    def config_from_fields(self):
        values = {}
        for key, widget in self.fields.items():
            try:
                value = float(widget.text.strip().replace(',', '.'))
            except ValueError:
                raise ValueError(f'{key}: enter a number.') from None
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key}: must be finite and >= 0.')
            if key == 'min_pos_sigma' and value == 0:
                raise ValueError('Minimum position sigma must be > 0.')
            if key == 'coord_scale' and value == 0:
                raise ValueError('Coordinate multiplier must be > 0.')
            values[key] = value
        return replace(self.cfg,
                       sigma_acc=values['sigma_acc'],
                       initial_vel_sigma=values['initial_vel_sigma'],
                       min_pos_sigma=values['min_pos_sigma'],
                       k_height=values['height_percent'] / 100.0,
                       coord_scale=values['coord_scale'])

    def queue_dropped_files(self, paths, schedule):
        """Validate a native file list, then release the drag source before processing."""
        if self._closed or self._drop_pending:
            return False
        try:
            if len(paths) != 1:
                raise ValueError('Drop one CSV file at a time.')
            path = Path(paths[0])
            if path.suffix.lower() != '.csv' or not path.is_file():
                raise ValueError('Drop an existing .csv file (not a folder or a link).')
        except (ValueError, OSError) as error:
            self.message.set_color('#b42318')
            self.message.set_text(str(error))
            self.fig.canvas.draw_idle()
            return False
        self._drop_pending = True
        self.paused = True

        def load_when_idle():
            try:
                if not self._closed:
                    self.load_csv(path)
            finally:
                self._drop_pending = False

        schedule(load_when_idle)
        return True

    def enable_file_drop(self):
        """Attach native drop handlers to the existing Matplotlib desktop window."""
        canvas = self.fig.canvas
        window = getattr(canvas.manager, 'window', None)
        backend = self.plt.get_backend().lower()
        try:
            if 'tk' in backend and hasattr(canvas, 'get_tk_widget'):
                from tkinterdnd2 import TkinterDnD, DND_FILES, COPY, REFUSE_DROP
                if not hasattr(TkinterDnD, 'require'):
                    raise ImportError('Upgrade tkinterdnd2 to version 0.6 or newer.')
                TkinterDnD.require(window)

                def on_drop(event):
                    # TkDND sends a Tcl list, not whitespace-separated filenames.
                    paths = window.tk.splitlist(event.data)
                    accepted = self.queue_dropped_files(paths, window.after_idle)
                    return COPY if accepted else REFUSE_DROP

                # The canvas contains all plots, controls and the empty start screen.
                # Plain tkinter.Tk is not a BaseWidget and has no DnD mixin methods.
                widget = canvas.get_tk_widget()
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<DropEnter>>', lambda event: COPY)
                widget.dnd_bind('<<DropPosition>>', lambda event: COPY)
                widget.dnd_bind('<<Drop>>', on_drop)
            elif 'qt' in backend and window is not None:
                from matplotlib.backends.qt_compat import QtCore
                viewer = self
                event_types = (QtCore.QEvent if hasattr(QtCore.QEvent, 'DragEnter')
                               else QtCore.QEvent.Type)
                copy_action = (QtCore.Qt.CopyAction if hasattr(QtCore.Qt, 'CopyAction')
                               else QtCore.Qt.DropAction.CopyAction)

                class FileDropFilter(QtCore.QObject):
                    def eventFilter(self, watched, event):
                        kind = event.type()
                        if kind not in (event_types.DragEnter, event_types.DragMove, event_types.Drop):
                            return False
                        urls = event.mimeData().urls()
                        if len(urls) != 1 or not urls[0].isLocalFile():
                            event.ignore()
                            return True
                        path = urls[0].toLocalFile()
                        if Path(path).suffix.lower() != '.csv':
                            event.ignore()
                            return True
                        if kind == event_types.Drop and not viewer.queue_dropped_files(
                                [path], lambda callback: QtCore.QTimer.singleShot(0, callback)):
                            event.ignore()
                            return True
                        # Explicitly COPY: accepting a proposed MOVE could remove the source.
                        event.setDropAction(copy_action)
                        event.accept()
                        return True

                self._drop_filter = FileDropFilter(window)
                for widget in (window, canvas):
                    widget.setAcceptDrops(True)
                    widget.installEventFilter(self._drop_filter)
            else:
                self.drop_status.set_text('Drag & drop needs a TkAgg or QtAgg desktop window. Use Open CSV.')
                return
        except Exception as error:
            self.drop_status.set_color('#b42318')
            self.drop_status.set_text(
                'Drag & drop unavailable. Install: python -m pip install -U "tkinterdnd2>=0.6". Open CSV still works.')
            print(f'Drag & drop unavailable: {error}', file=sys.stderr)
            return
        self.drop_status.set_text('Drag & drop: drop one CSV anywhere in this window to open it.')

    def choose_csv_path(self):
        initial_dir = self.input_path.parent if self.input_path else self.export_path.resolve().parent
        return self.choose_file('Open flight CSV', initial_dir, 'CSV files (*.csv);;All files (*)',
                                [('CSV files', '*.csv'), ('All files', '*')])

    def choose_file(self, title, initial_dir, qt_filter, tk_filetypes):
        """Use the active desktop backend's native file dialog; no GUI imports in CLI mode."""
        parent = getattr(self.fig.canvas.manager, 'window', None)
        if 'qt' in self.plt.get_backend().lower():
            from matplotlib.backends.qt_compat import QtWidgets
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent, title, str(initial_dir), qt_filter)
            return filename
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as error:
            raise RuntimeError('File dialog needs Tk (python3-tk on Linux) or a Qt backend.') from error
        root = None
        try:
            if not isinstance(parent, tk.Misc):
                root = tk.Tk()
                root.withdraw()
                parent = root
            return filedialog.askopenfilename(
                parent=parent, title=title, initialdir=str(initial_dir), filetypes=tk_filetypes)
        finally:
            if root is not None:
                root.destroy()

    def choose_plotjuggler_path(self):
        initial_dir = (Path(self.plotjuggler_path).parent if self.plotjuggler_path
                       else Path.home())
        return self.choose_file(
            'Locate installed PlotJuggler executable (not its installer)', initial_dir,
            'Executables (*.exe);;All files (*)' if sys.platform == 'win32'
            else 'All files (*)',
            [('Executable', '*.exe'), ('All files', '*')] if sys.platform == 'win32'
            else [('All files', '*')])

    def open_in_plotjuggler(self, _event=None):
        if self.result is None:
            return False
        was_paused = self.paused
        self.paused = True
        try:
            cfg = self.config_from_fields()
            self.message.set_color('#555555')
            self.message.set_text('Looking for PlotJuggler...')
            self.draw(force=True)
            executable = find_plotjuggler(self.plotjuggler_path)
            if executable is None:
                self.message.set_text('PlotJuggler not found. Select its installed executable.')
                self.fig.canvas.draw()
                selected = self.choose_plotjuggler_path()
                if not selected:
                    self.paused = was_paused
                    self.message.set_text('PlotJuggler selection cancelled.')
                    return False
                executable = resolve_plotjuggler_executable(selected)
                if executable is None:
                    raise ValueError('Selected file is not an executable. Select the installed PlotJuggler application.')
            # Apply edited parameters, or restore the current export if it was moved/changed.
            if cfg != self.cfg:
                if not self.apply_parameters():
                    return False
            else:
                self.save_estimates(self.export_path, self.result)
            command = [str(executable), '-d', str(self.export_path.resolve())]
            options = {'cwd': str(executable.parent), 'shell': False}
            if sys.platform == 'win32':
                options['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options['start_new_session'] = True
            process = subprocess.Popen(command, **options)
            self.external_processes.append(process)
            self.plotjuggler_path = executable
            self.message.set_color('#257343')
            self.message.set_text(f'PlotJuggler started: {self.export_path.name[:65]}')
            print(f'PlotJuggler: {executable}; data: {self.export_path.resolve()}')
            try:
                save_plotjuggler_path(executable)
            except OSError as error:
                self.message.set_text('PlotJuggler started; could not remember its path for the next session.')
                print(f'Could not save PlotJuggler path: {error}', file=sys.stderr)
            return True
        except Exception as error:
            self.message.set_color('#b42318')
            self.message.set_text('PlotJuggler: ' + str(error).replace('\n', ' ')[:135])
            print(f'Open in PlotJuggler failed: {error}', file=sys.stderr)
            return False
        finally:
            self.last_clock = time.perf_counter()
            self.draw(force=True)

    def open_csv(self, _event=None):
        was_paused = self.paused
        self.paused = True
        self.draw(force=True)
        try:
            filename = self.choose_csv_path()
        except Exception as error:
            # Dialog/backend failures must not close the current flight window.
            self.paused = was_paused
            self.message.set_color('#b42318')
            self.message.set_text('Cannot open dialog: ' + str(error).replace('\n', ' ')[:135])
            print(f'File dialog failed: {error}', file=sys.stderr)
            self.draw(force=True)
            return False
        finally:
            # A modal dialog may keep the GUI event loop running for a long time.
            self.last_clock = time.perf_counter()
        if not filename:
            self.paused = was_paused
            self.draw(force=True)
            return False
        return self.load_csv(Path(filename))

    def load_csv(self, path):
        """Commit a new flight only after reading, estimating and exporting succeed."""
        self.paused = True
        try:
            cfg = self.config_from_fields()
            path = Path(path).resolve()
            export_path = path.parent / self.default_export_name if self.export_follows_input else self.export_path
            if export_path.resolve() in self.input_paths | {path}:
                raise ValueError('Export would overwrite an input CSV. Choose a different file/output path.')
            self.message.set_color('#555555')
            self.message.set_text(f'Opening {path.name[:70]} and saving CSV...')
            self.fig.canvas.draw()
            data, _ = read_flight(path)
            attitude = check_attitude(data, cfg)
            result = self.run_estimate(data, cfg, attitude)
            self.save_estimates(export_path, result)
        except (ValueError, OSError, RuntimeError, OverflowError) as error:
            self.message.set_color('#b42318')
            self.message.set_text('Not opened: ' + str(error).replace('\n', ' ')[:145])
            print(f'Open CSV failed: {error}', file=sys.stderr)
            self.last_clock = time.perf_counter()
            self.draw(force=True)
            return False
        self.cfg, self.data, self.attitude, self.result = cfg, data, attitude, result
        self.after_apply()
        self.input_path, self.export_path = path, export_path
        self.input_paths.add(path)
        self.duration = float(result['time'][-1])
        self.current_time = 0.0
        self.last_index = -1
        self.last_clock = time.perf_counter()
        self.slider.valmax = self.duration
        self.slider.valinit = 0.0
        self.slider.ax.set_xlim(0.0, self.duration)
        self.slider.vline.set_xdata([0.0, 0.0])
        self.set_loaded_controls()
        self.fig.suptitle(path.name + ' - ' + self.replay_name, y=0.98, fontsize=16)
        report = attitude.report
        self.message.set_color('#257343')
        self.message.set_text(
            f'Opened {len(data)} rows. Saved {export_path.name[:35]}. '
            f'Quaternion: {report["direction"]}; yaw: {report["yaw_unit"]}; '
            f'RMS: {report["yaw_rms_deg"]:.3f} deg.')
        print(f'Opened: {path}; {len(data)} samples; {self.duration:.2f} s. Exported: {export_path}')
        for warning in report['warnings']:
            print('Frame check: ' + warning, file=sys.stderr)
        self.draw(force=True)
        return True

    def apply_parameters(self, _event=None):
        if self.result is None:
            return False
        self.paused = True
        try:
            cfg = self.config_from_fields()
            self.message.set_color('#555555')
            self.message.set_text('Recalculating the full flight and saving CSV...')
            self.fig.canvas.draw()
            result = self.run_estimate(self.data, cfg, self.attitude)
            # Only commit new settings/curves after the corresponding CSV is saved.
            self.save_estimates(self.export_path, result)
        except (ValueError, OSError, OverflowError) as error:
            self.message.set_color('#b42318')
            detail = str(error).replace('\n', ' ')
            if isinstance(error, PermissionError):
                detail = 'Cannot write CSV. Close it in Excel/another app and retry.'
            print(f'Apply failed: {error}', file=sys.stderr)
            self.message.set_text('Not applied: ' + detail[:145])
            self.draw(force=True)
            return False
        self.cfg, self.result = cfg, result
        self.after_apply()
        self.duration = float(result['time'][-1])
        self.current_time = min(self.current_time, self.duration)
        self.slider.valmax = self.duration
        self.slider.ax.set_xlim(0.0, self.duration)
        self.last_clock = time.perf_counter()
        self.message.set_color('#257343')
        self.message.set_text(f'Applied. Saved {len(result["time"])} rows to {self.export_path.name}.')
        print(f'Updated {self.export_path}: {self.parameter_text()}')
        self.draw(force=True)
        return True

    def tick(self):
        self.external_processes = [p for p in self.external_processes if p.poll() is None]
        now = time.perf_counter()
        elapsed = now - self.last_clock
        self.last_clock = now
        if not self.paused:
            self.current_time = min(self.duration, self.current_time + elapsed * self.playback_speed)
            if self.current_time >= self.duration:
                self.paused = True
            self.draw()

    def seek(self, seconds):
        if self.result is None:
            return
        self.current_time = float(np.clip(seconds, 0, self.duration))
        self.paused = True
        self.last_clock = time.perf_counter()
        self.draw(force=True)

    def toggle_pause(self, _event=None):
        if self.result is None:
            return
        if self.paused and self.current_time >= self.duration:
            self.current_time = 0.0
        self.paused = not self.paused
        self.last_clock = time.perf_counter()
        self.draw(force=True)

    def on_key(self, event):
        if any(widget.capturekeystrokes for widget in self.fields.values()):
            return  # Arrow/space/+/- keys belong to the text editor while typing.
        if event.key == ' ':
            self.toggle_pause()
        elif event.key in ('left', 'right'):
            self.seek(self.current_time + (-1.0 if event.key == 'left' else 1.0))
        elif event.key in ('home', 'end'):
            self.seek(0.0 if event.key == 'home' else self.duration)
        elif event.key in ('+', '=', '-', '_'):
            factor = 0.5 if event.key in ('-', '_') else 2.0
            self.playback_speed = float(np.clip(self.playback_speed * factor, 0.0625, 64))
            self.last_clock = time.perf_counter()
            self.draw(force=True)

    def on_close(self, _event):
        self._closed = True
        self.timer.stop()

    def show(self):
        self.last_clock = time.perf_counter()
        self.timer.start()
        self.plt.show()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, nargs='?',
                        help='Optional flight CSV; otherwise open an empty window with Open CSV / drag & drop')
    parser.add_argument('--dt', type=float, default=0.05, help='Fixed sample step, seconds (default: 0.05)')
    parser.add_argument('--window', type=float, default=10.0, help='Visible time span, seconds (default: 10)')
    parser.add_argument('--k-height', type=float, default=0.1, help='Position stddev / height (default: 0.1)')
    parser.add_argument('--sigma-acc', type=float, default=1.0, help='Per-sample acceleration stddev, m/s^2 (default: 1)')
    parser.add_argument('--min-pos-sigma', type=float, default=0.001, help='Position stddev floor, m (default: 0.001)')
    parser.add_argument('--initial-vel-sigma', type=float, default=1.0, help='Initial speed stddev, m/s (default: 1)')
    parser.add_argument('--coord-scale', type=float, default=1.0,
                        help='Multiply vis_x/y before filtering and plotting (default: 1)')
    parser.add_argument('--vx0', type=float, default=0.0, help='Initial north velocity, m/s (default: 0)')
    parser.add_argument('--vy0', type=float, default=0.0, help='Initial east velocity, m/s (default: 0)')
    parser.add_argument('--quat-direction', dest='quaternion_direction',
                        choices=('auto', 'body-to-ned', 'ned-to-body'), default='auto',
                        help='Input quaternion direction; default: check against yaw/rawacc')
    parser.add_argument('--yaw-unit', choices=('auto', 'deg', 'rad'), default='auto',
                        help='CSV yaw units; diagnostic only, quaternion drives rotation')
    parser.add_argument('--playback-speed', type=float, default=1.0, help='Playback rate multiplier (default: 1)')
    parser.add_argument('--start', type=float, default=0.0, help='Start displaying at this second; estimate from row 0')
    parser.add_argument('--paused', action='store_true', help='Start with playback paused')
    parser.add_argument('--export', type=Path,
                        help='Fixed output CSV (default: result.csv next to current input); overwritten on startup, Open and Apply')
    parser.add_argument('--plotjuggler', type=Path,
                        help='Path to the PlotJuggler executable; otherwise locate automatically or choose in the UI')
    parser.add_argument('--snapshot', type=Path, help='Save the window at --start as an image')
    parser.add_argument('--no-gui', action='store_true', help='Compute/export without opening a desktop window')
    args = parser.parse_args(argv)
    if args.csv is None and args.no_gui:
        parser.error('--no-gui requires an input CSV')
    args.export_follows_input = args.export is None
    if args.export is None:
        args.export = (args.csv.parent if args.csv is not None else Path.cwd()) / 'result.csv'
    for name in ('dt', 'window', 'min_pos_sigma', 'playback_speed', 'coord_scale'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and > 0')
    for name in ('k_height', 'sigma_acc', 'initial_vel_sigma', 'start'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and >= 0')
    for name in ('vx0', 'vy0'):
        if not math.isfinite(getattr(args, name)):
            parser.error(f'--{name} must be finite')
    if args.playback_speed > 64 or args.playback_speed < 0.0625:
        parser.error('--playback-speed must be between 0.0625 and 64')
    # Input is read-only; do not allow an export or snapshot to overwrite it.
    paths = [p for p in (args.csv, args.export, args.snapshot) if p is not None]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error('Input CSV, export and snapshot must have different paths')
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    try:
        data = attitude = result = None
        if args.csv is not None:
            data, mapping = read_flight(args.csv)
            attitude = check_attitude(data, cfg)
            report = attitude.report
            print(f'Quaternion input: {report["direction"]} ({report["direction_evidence"]}); '
                  f'yaw in {report["yaw_unit"]}')
            print(f'Yaw consistency: RMS={report["yaw_rms_deg"]:.6f} deg, '
                  f'max={report["yaw_max_deg"]:.6f} deg')
            print(f'Quaternion norm: {report["quaternion_norm_min"]:.8f} .. '
                  f'{report["quaternion_norm_max"]:.8f} (normalized before use)')
            if 'raw_horizontal_rms' in report:
                print(f'R @ rawacc versus acc.x/y: RMS={report["raw_horizontal_rms"]:.8f} m/s^2')
            if report.get('acc_z_up_relation_rms', float('inf')) < 0.001:
                print('Vertical diagnostic: acc.z ~= -(R @ rawacc).z - G (UP sign). '
                      'acc.z is not used in this horizontal filter.')
            for warning in report['warnings']:
                print('Frame check: ' + warning, file=sys.stderr)
            result = estimate(data, cfg, attitude)
            print(f'{len(data)} samples; {result["time"][-1]:.2f} s; dt={cfg.dt:g} s')
            print(f'Acceleration columns: {mapping["acc_x"]}, {mapping["acc_y"]}')
            print(f'R = max({cfg.k_height:g} * abs(mheight), {cfg.min_pos_sigma:g})^2')
            print(f'Coordinate multiplier: {cfg.coord_scale:g} (vis_scaled = vis * multiplier)')
            print(f'Final BODY bias: x={result["bias_body"][-1, 0]:.6f}, '
                  f'y={result["bias_body"][-1, 1]:.6f} m/s^2; body z fixed to 0')
            print(f'Final NED projection: x={result["bias"][-1, 0]:.6f}, '
                  f'y={result["bias"][-1, 1]:.6f} m/s^2')
            export_estimates(args.export, result)
            print(f'Exported: {args.export}')
        else:
            print('Open CSV or drop a flight CSV into the window to begin.')
        if not args.no_gui or args.snapshot is not None:
            import matplotlib
            if args.no_gui:
                matplotlib.use('Agg')
            elif matplotlib.get_backend().lower() in ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template'):
                raise RuntimeError('No GUI backend. Run on a desktop with Tk/Qt, '
                                   'or use --no-gui --snapshot view.png.')
            viewer = FlightViewer(result, cfg, args.window, args.playback_speed,
                                  args.start, args.paused or args.no_gui,
                                  str(args.csv) if args.csv is not None else 'Flight',
                                  data=data, export_path=args.export, attitude=attitude,
                                  input_path=args.csv, export_follows_input=args.export_follows_input,
                                  plotjuggler_path=args.plotjuggler)
            if args.snapshot is not None:
                viewer.fig.savefig(args.snapshot, dpi=130)
                print(f'Snapshot: {args.snapshot}')
            if not args.no_gui:
                viewer.show()
            else:
                viewer.plt.close(viewer.fig)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
