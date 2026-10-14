#!/usr/bin/env python3
"""Compare every shared export column with the existing NumPy implementation.

Requires only NumPy. Synthetic inputs and outputs live in a temporary directory.
--csv adds a fixed-step comparison using your own flight; --time-column also
checks its actual intervals against the Python components with variable dt.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kalman_flight_viewer as py


def reference(data, cfg, intervals=None, acc_sigmas=None, pos_sigmas=None):
    attitude = py.check_attitude(data, cfg)
    out = py.estimate(data, cfg, attitude)
    if intervals is None:
        return out
    # Reuse the original, independently implemented NumPy/Kalman components.
    # Only the orchestration substitutes a per-row interval and noise values.
    n = len(data)
    sigma = out['sigma_position'] if pos_sigmas is None else pos_sigmas
    acceleration_sigma = np.full(n, cfg.sigma_acc) if acc_sigmas is None else acc_sigmas
    out['time'] = np.cumsum(intervals)
    out['sigma_position'] = sigma
    filters = [py.Kalman1D(out['measured_position'][0, j], out['velocity'][0, j],
                          sigma[0]**2, 0.0, cfg.initial_vel_sigma**2) for j in range(2)]
    for j, kf in enumerate(filters):
        out['covariance'][0, j] = kf.pxx, kf.pxv, kf.pvv
    bias = py.BodyBiasEstimator()
    integral = np.zeros(2)
    for i in range(1, n):
        dt = intervals[i]
        m = attitude.rotation[i, :2, :2]
        raw = data[i, 3:5]
        out['bias_body_used'][i] = bias.bias
        out['bias_used'][i] = m @ bias.bias
        corrected = raw - out['bias_used'][i]
        integral += raw * dt
        for j, kf in enumerate(filters):
            out['gain'][i, j] = kf.step(corrected[j], out['measured_position'][i, j],
                                       sigma[i]**2, dt, acceleration_sigma[i])
            out['position'][i, j], out['velocity'][i, j] = kf.x, kf.v
            out['covariance'][i, j] = kf.pxx, kf.pxv, kf.pvv
        body = bias.update(m, raw, out['velocity'][i]-out['velocity'][i-1], dt)
        out['bias_body'][i], out['bias'][i] = body, m @ body
        out['bias_normal'][i] = bias.normal[0, 0], bias.normal[0, 1], bias.normal[1, 1]
        out['bias_rhs'][i] = bias.rhs
        out['corrected_acceleration'][i] = corrected
        out['integral_raw'][i] = integral
    return out


def compare(exe, source, work, cfg, flags=(), intervals=None, acc_sigmas=None, pos_sigmas=None):
    output = work / 'result_cpp.csv'
    command = [str(exe), str(source), '--export', str(output),
               '--sigma-acc', str(cfg.sigma_acc), '--min-pos-sigma', str(cfg.min_pos_sigma),
               '--k-height', str(cfg.k_height), '--initial-vel-sigma', str(cfg.initial_vel_sigma),
               '--coord-scale', str(cfg.coord_scale), '--vx0', str(cfg.vx0), '--vy0', str(cfg.vy0),
               '--quat-direction', cfg.quaternion_direction, '--yaw-unit', cfg.yaw_unit]
    if intervals is None:
        command += ['--dt', str(cfg.dt)]
    completed = subprocess.run(command + list(flags), capture_output=True, text=True)
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    data, _ = py.read_flight(source)
    expected_path = work / 'reference.csv'
    py.export_estimates(expected_path, reference(data, cfg, intervals, acc_sigmas, pos_sigmas))
    expected = np.genfromtxt(expected_path, delimiter=';', names=True)
    actual = np.genfromtxt(output, delimiter=';', names=True)
    assert actual.dtype.names == expected.dtype.names + ('dt_s', 'sigma_acc')
    for name in expected.dtype.names:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-9, atol=1e-9, err_msg=name)
    dt = np.full(len(data), cfg.dt) if intervals is None else intervals.copy()
    dt[0] = 0
    np.testing.assert_allclose(actual['dt_s'], dt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual['sigma_acc'], cfg.sigma_acc if acc_sigmas is None else acc_sigmas)
    error = max(float(np.max(np.abs(actual[k]-expected[k])))
                for k in expected.dtype.names if k != 'time_s')
    print(f'PASS {source.name}: {len(data)} rows, 48 shared columns; max numeric error {error:.3g}')


def synthetic(path, inverse=False):
    count = 320
    dt = np.resize([.017, .053, .14, .05], count)
    dt[0] = 0
    t = np.cumsum(dt)
    roll, pitch, yaw = .25*np.sin(t), .3*np.cos(t*.6), .4+t*.15
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    q = np.column_stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                         cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy))
    rotation, _ = py.quaternion_matrices(q)
    a = np.column_stack((.3*np.sin(t), .2*np.cos(t)))
    raw = a + np.einsum('nij,j->ni', rotation[:, :2, :2], [.12, -.08])
    raw_body = np.einsum('nji,nj->ni', rotation, np.column_stack((raw, np.full(count, -9.81))))
    p, v = np.array([1., 2.]), np.array([.7, -.4])
    positions = []
    for i in range(count):
        p += v*dt[i]+.5*a[i]*dt[i]**2
        v += a[i]*dt[i]
        positions.append(p.copy()+.03*np.array([np.sin(3*t[i]), np.cos(2*t[i])]))
    if inverse:
        q[:, :3] *= -1
    q *= 1.7  # Non-unit input must be normalized in both implementations.
    acc_sigma = .2 + .7*(np.arange(count) % 11)/10
    pos_sigma = .07 + .04*np.sin(t)**2
    header = ['vis_x', 'vis_y', 'mheight', 'acc_x', 'acc_y', 'quat.x', 'quat.y',
              'quat.z', 'quat.w', 'yaw', 'rawacc.x', 'rawacc.y', 'rawacc.z',
              'timestamp', 'elapsed', 'acc_sigma', 'pos_sigma', 'note']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for i in range(count):
            row = [*positions[i], 2+np.sin(t[i]), *raw[i], *q[i],
                   yaw[i] if inverse else np.rad2deg(yaw[i]), *raw_body[i],
                   1000000+round(t[i]*1000), '' if i == 0 else dt[i], acc_sigma[i], pos_sigma[i]]
            writer.writerow([str(value).replace('.', ',') for value in row] + ['quoted; "value"\nnext line'])
    return dt, acc_sigma, pos_sigma


def rejection_tests(exe, source, work):
    target = work / 'protected.csv'
    target.write_text('previous valid output', encoding='utf-8')
    for flags in (['--dt', '0'], ['--time-column', 'timestamp', '--dt', '.05'],
                  ['--dt-column', 'acc_y'], ['--time-column', 'pos_sigma'],
                  ['--sigma-acc-column', 'note']):
        # acc_y changes sign, so it cannot be used as a positive time interval.
        p = subprocess.run([str(exe), str(source), '--export', str(target), *flags], capture_output=True)
        assert p.returncode != 0, flags
        assert target.read_text(encoding='utf-8') == 'previous valid output'
    original = source.read_bytes()
    p = subprocess.run([str(exe), str(source), '--export', str(source)], capture_output=True)
    assert p.returncode != 0 and source.read_bytes() == original
    print('PASS invalid time/noise/input-output cases preserve existing files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--time-column')
    parser.add_argument('--time-scale', type=float, default=1)
    args = parser.parse_args()
    if args.time_column and not args.csv:
        parser.error('--time-column requires --csv')
    exe = args.exe.resolve()
    with tempfile.TemporaryDirectory(prefix='kalman_cpp_') as directory:
        work = Path(directory)
        source = work / 'політ with spaces.csv'
        intervals, acc, pos = synthetic(source)
        compare(exe, source, work, py.Config())
        cfg = py.Config(coord_scale=1.12, sigma_acc=.4, k_height=.03,
                        min_pos_sigma=.02, initial_vel_sigma=.8, vx0=.7, vy0=-.4)
        compare(exe, source, work, cfg)
        flags = ['--sigma-acc-column', 'acc_sigma', '--position-sigma-column', 'pos_sigma']
        compare(exe, source, work, cfg, flags+['--time-column', 'timestamp', '--time-scale', '.001'], intervals, acc, pos)
        compare(exe, source, work, cfg, flags+['--dt-column', 'elapsed'], intervals, acc, pos)
        # Auto-detect inverse quaternions and yaw in radians, then verify explicit overrides.
        synthetic(source, inverse=True)
        compare(exe, source, work, cfg)
        cfg.quaternion_direction, cfg.yaw_unit = 'ned-to-body', 'rad'
        compare(exe, source, work, cfg)
        rejection_tests(exe, source, work)
        if args.csv:
            compare(exe, args.csv.resolve(), work, py.Config())
            if args.time_column:
                with args.csv.open(encoding='utf-8-sig', newline='') as f:
                    times = np.array([np.longdouble(row[args.time_column].replace(',', '.'))
                                      for row in csv.DictReader(f, delimiter=';')])
                intervals = np.r_[0., np.asarray(np.diff(times)*args.time_scale, dtype=float)]
                compare(exe, args.csv.resolve(), work, py.Config(),
                        ['--time-column', args.time_column, '--time-scale', str(args.time_scale)], intervals)


if __name__ == '__main__':
    main()
