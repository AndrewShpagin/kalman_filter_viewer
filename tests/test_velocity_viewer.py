import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

import kalman_flight_viewer as shared
import velocity_flight_viewer as viewer


def flight(acceleration, dt=0.05, initial_velocity=(0., 0.), body_bias=(0., 0.), rotating=False):
    """Exact motion with acceleration held constant over each sample interval."""
    n = len(acceleration)
    t = np.arange(n) * dt
    data = np.zeros((n, 15))
    data[:, 2] = 100.
    data[:, 8] = 1.
    data[:, 10:] = np.nan
    if rotating:
        yaw = 0.37 * t
        roll = 0.35 * np.sin(t * 0.23)
        pitch = 0.28 * np.cos(t * 0.31)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        data[:, 5:9] = np.column_stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                                       cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy))
        data[:, 9] = np.rad2deg(yaw)
    rotation, _ = shared.quaternion_matrices(data[:, 5:9])
    data[:, 3:5] = acceleration + rotation[:, :2, :2] @ np.asarray(body_bias)
    velocity = np.zeros((n, 2))
    velocity[0] = initial_velocity
    for i in range(1, n):
        velocity[i] = velocity[i - 1] + acceleration[i] * dt
        data[i, :2] = data[i - 1, :2] + (velocity[i - 1] + velocity[i]) * (dt / 2)
    return data, velocity


def config(**kwargs):
    return viewer.VelocityConfig(quaternion_direction='body-to-ned', yaw_unit='deg', **kwargs)


class VelocityModelTests(unittest.TestCase):
    def test_rotating_body_does_not_create_bias_for_constant_ned_velocity(self):
        data, truth = flight(np.zeros((1000, 2)), initial_velocity=(3., -2.), rotating=True)
        out = viewer.estimate_velocity(data, config(vx0=3., vy0=-2.))
        np.testing.assert_allclose(out['velocity'], truth, atol=2e-11)
        np.testing.assert_allclose(out['bias_body'], 0., atol=2e-11)

    def test_preserves_acceleration_with_known_body_bias_and_full_attitude(self):
        t = np.arange(1400) * 0.05
        acceleration = np.column_stack((0.4*np.sin(t/3) + 0.7*(t > 14), 0.6*np.cos(t/5)))
        data, truth = flight(acceleration, initial_velocity=(1., -0.5),
                             body_bias=(0.3, -0.2), rotating=True)
        cfg = config(learn_bias=False, bias_x0=0.3, bias_y0=-0.2, vx0=1., vy0=-0.5)
        out = viewer.estimate_velocity(data, cfg)
        # Includes a real acceleration step: fading the acceleration itself or
        # subtracting a linear speed trend would fail this physics check.
        np.testing.assert_allclose(out['velocity'], truth, atol=2e-10)
        np.testing.assert_allclose(out['corrected_acceleration'], acceleration, atol=1e-12)
        np.testing.assert_allclose(out['forward_position'], data[:, :2] + truth, atol=2e-10)

    def test_learns_stationary_body_bias_while_rotating(self):
        bias = np.array([0.3, -0.2])
        data, _ = flight(np.zeros((4000, 2)), body_bias=bias, rotating=True)
        for initial in ((0., 0.), (-0.5, 0.4)):
            out = viewer.estimate_velocity(data, config(bias_x0=initial[0], bias_y0=initial[1]))
            np.testing.assert_allclose(out['bias_body'][-1], bias, atol=1e-10)
            self.assertLess(np.max(np.abs(out['velocity'][-400:])), 1e-10)

    def test_feedback_is_next_sample_and_learns_bias_during_accelerated_motion(self):
        t = np.arange(900) * 0.05
        acceleration = np.column_stack((np.sin(t/4), np.cos(t/7)))
        data, _ = flight(acceleration, initial_velocity=(1., -2.), body_bias=(0.15, -0.1))
        for warmup in (0., 6.):
            out = viewer.estimate_velocity(data, config(vx0=1., vy0=-2., bias_warmup=warmup))
            np.testing.assert_array_equal(out['bias_body_used'][1:], out['bias_body'][:-1])
            np.testing.assert_allclose(out['bias_body'][-1], [.15, -.1], atol=2e-12)

    def test_first_visual_speed_correction_cannot_create_large_startup_bias(self):
        # A ~4 cm visual change in 50 ms, as in the reported log, must not be
        # interpreted as a 17-19 m/s² accelerometer bias on the first update.
        data, _ = flight(np.zeros((100, 2)))
        data[1:, :2] = [-0.04389, -0.04606]
        data[1, 3:5] = [-0.14639, 0.12965]
        unprotected = viewer.estimate_velocity(data, config(bias_warmup=0.))
        out = viewer.estimate_velocity(data, config())
        np.testing.assert_array_equal(out['bias_body'][0], [0., 0.])
        self.assertGreater(np.min(np.abs(unprotected['bias_body'][1])), 17.)
        self.assertLess(np.max(np.abs(out['bias_body'])), 0.08)
        # The first visual-speed estimate is unchanged; the correction is to
        # bias learning, not clipping or hiding the measured coordinate change.
        np.testing.assert_array_equal(out['velocity'][1], unprotected['velocity'][1])

    def test_bias_trust_ramps_with_elapsed_time_including_unequal_steps(self):
        initial = np.array([0.04, -0.02])
        true_bias = np.array([0.3, -0.2])
        acceleration = np.array([0.8, -0.6])
        fit = shared.BodyBiasEstimator(initial, warmup_seconds=2.)
        elapsed = 0.
        for dt in (0.005, 0.05, 0.3, 0.645, 1., 2.):
            elapsed += dt
            value = fit.update(np.eye(2), acceleration + true_bias, acceleration*dt, dt)
            expected = initial + min(elapsed/2., 1.) * (true_bias-initial)
            np.testing.assert_allclose(value, expected, atol=1e-14)
        # After enough information, the initial prior no longer attenuates bias.
        np.testing.assert_allclose(fit.bias, true_bias, atol=1e-14)

    def test_weakly_observed_body_direction_stays_close_to_initial_bias(self):
        initial = np.array([0.1, -0.2])
        fit = shared.BodyBiasEstimator(initial, warmup_seconds=18.)
        # Pitch near 90 degrees: body X has almost no horizontal projection.
        m = np.diag([1e-7, 1.])
        for _ in range(100):
            fit.update(m, np.array([0.5, 0.2]), np.zeros(2), .05)
        self.assertLess(abs(fit.bias[0] - initial[0]), 1e-6)
        self.assertLess(abs(fit.bias[1] - initial[1]), .12)

    def test_bias_memory_tracks_a_changed_offset_after_long_initial_history(self):
        # Stationary vision, rotating body, changing bias, large zero-mean noise.
        # A whole-flight fit leaves a persistent signed acceleration/speed error.
        count = 8000
        data, _ = flight(np.zeros((count, 2)), rotating=True)
        t = np.arange(count)*.05
        rotation, _ = shared.quaternion_matrices(data[:, 5:9])
        bias = np.where((t < 160)[:, None], [.05, -.05], [.35, .2])
        noise = np.column_stack((.8*np.sin(9*t), .7*np.cos(11*t)))
        data[:, 3:5] = np.einsum('nij,nj->ni', rotation[:, :2, :2], bias+noise)
        fixed = viewer.estimate_velocity(data, config())
        cumulative = viewer.estimate_velocity(data, config(bias_memory=0))
        np.testing.assert_allclose(fixed['bias_body'][-1], bias[-1], atol=.012)
        self.assertGreater(np.linalg.norm(cumulative['bias_body'][-1]-bias[-1]), .12)
        # The fit is still causal, starts at zero, and applies the new bias later.
        np.testing.assert_array_equal(fixed['bias_body'][0], [0., 0.])
        np.testing.assert_array_equal(fixed['bias_body_used'][1:], fixed['bias_body'][:-1])

    def test_bias_memory_uses_seconds_and_exact_interval_weights(self):
        # Identical observations over equal elapsed time must give equal
        # information even when dt varies by orders of magnitude.
        reference = shared.BodyBiasEstimator(warmup_seconds=6., memory_seconds=60.)
        irregular = shared.BodyBiasEstimator(warmup_seconds=6., memory_seconds=60.)
        m = np.array([[.8, -.2], [.1, .9]])
        bias = np.array([.3, -.2])
        for _ in range(1000):
            reference.update_residual(m, (m@bias)*.01, .01)
        for dt in (.000001, .07, 2.4, 7.529999):
            irregular.update_residual(m, (m@bias)*dt, dt)
        np.testing.assert_allclose(irregular.normal, reference.normal, atol=1e-12)
        np.testing.assert_allclose(irregular.rhs, reference.rhs, atol=1e-12)
        np.testing.assert_allclose(irregular.bias, reference.bias, atol=1e-12)
        expected = -60*np.expm1(-10/60)*(m.T@m)
        np.testing.assert_allclose(irregular.normal, expected, atol=1e-12)

    def test_future_rows_cannot_change_past_estimates(self):
        t = np.arange(600) * 0.05
        data, _ = flight(np.column_stack((np.sin(t), np.cos(t))), body_bias=(0.1, 0.2), rotating=True)
        full = viewer.estimate_velocity(data, config())
        short = viewer.estimate_velocity(data[:245], config())
        for key in ('velocity', 'bias_body', 'forward_position', 'mean_visual_velocity'):
            np.testing.assert_allclose(full[key][:245], short[key], rtol=0., atol=1e-12)

    def test_matching_means_and_short_log_warmup(self):
        data, truth = flight(np.zeros((3, 2)), initial_velocity=(2., -1.))
        out = viewer.estimate_velocity(data, config(vx0=2., vy0=-1.))
        np.testing.assert_allclose(out['velocity'], truth, atol=1e-12)
        np.testing.assert_array_equal(out['window_complete'], 0.)
        np.testing.assert_allclose(out['mean_integrated_velocity'] + out['normalization_offset'],
                                   out['mean_visual_velocity'], atol=1e-12)
        data, _ = flight(np.zeros((362, 2)))
        out = viewer.estimate_velocity(data, config())
        self.assertEqual(out['window_complete'][359], 0)
        self.assertEqual(out['window_complete'][360], 1)

    def test_rectangular_and_asymmetric_windows(self):
        acceleration = np.full((240, 2), [0.2, -0.1])
        data, truth = flight(acceleration)
        for fade_in, flat, fade_out in ((0, 2, 0), (1, 2, 0), (0, 1, 2), (1, 0, 1), (0.13, 0.11, 0.17)):
            out = viewer.estimate_velocity(data, config(learn_bias=False, fade_in=fade_in,
                                                        flat=flat, fade_out=fade_out))
            # With a fractional left edge, linear interpolation of position
            # differs from the quadratic analytic trajectory. Its known error
            # is a*(original midpoint - clipped midpoint), weighted once.
            expected = truth.copy()
            times = np.arange(len(data))*.05
            for i in range(1, len(data)):
                left = max(0., times[i]-fade_in-flat-fade_out)
                k = np.arange(1, i+1)
                lo = np.maximum(times[k-1], left)
                valid = times[k] > lo
                k, lo = k[valid], lo[valid]
                h = times[k]-lo
                mid = (times[k]+lo)/2
                cfg = config(fade_in=fade_in, flat=flat, fade_out=fade_out)
                weights = viewer.interval_weights(times[i]-mid, times[i]-left, cfg)*h
                shift = np.average((times[k-1]+times[k])/2-mid, weights=weights)
                expected[i] += acceleration[i]*shift
            np.testing.assert_allclose(out['velocity'], expected, atol=1e-11)

    def test_delayed_variable_time_motion_and_causal_prefix(self):
        dt = np.r_[0., np.resize([1/60, 1/23, 1/37, 1/20, 1/53], 2000)]
        t = np.cumsum(dt)
        delay = .237
        speed = np.column_stack((1.6*np.cos(.8*t), -.6*np.sin(.6*t)))
        data = np.zeros((len(t), 15)); data[:, 2] = 100.; data[:, 10:] = np.nan
        data[:, :2] = np.column_stack((2*np.sin(.8*(t-delay)), np.cos(.6*(t-delay))))
        data[:, 7] = np.sin(.3*t/2); data[:, 8] = np.cos(.3*t/2)
        data[:, 9] = np.rad2deg(.3*t)
        rotation, _ = shared.quaternion_matrices(data[:, 5:9])
        data[1:, 3:5] = np.diff(speed, axis=0)/dt[1:, None]
        data[:, 3:5] += rotation[:, :2, :2]@np.array([.12, -.08])
        cfg = config(vision_delay=delay, vx0=1.6, fade_in=.6, flat=.4, fade_out=.3,
                     bias_memory=4., bias_warmup=2.)
        result = viewer.estimate_velocity(data, cfg, intervals=dt)
        np.testing.assert_allclose(result['velocity'][t>30], speed[t>30], atol=.003)
        np.testing.assert_allclose(result['bias_body'][t>30], np.tile([.12, -.08], (sum(t>30), 1)), atol=.003)
        np.testing.assert_array_equal(result['bias_body'][t<delay], 0.)
        self.assertTrue(np.all(result['bias_fit_dt_s'][t<delay] == 0))
        prefix = viewer.estimate_velocity(data[:300], cfg, intervals=dt[:300])
        for key in result:
            np.testing.assert_allclose(result[key][:300], prefix[key], rtol=0, atol=1e-12, err_msg=key)
        for invalid in (dt[:-1], np.where(t<1, 0, dt), np.full(len(t), np.nan)):
            with self.assertRaises(ValueError):
                viewer.estimate_velocity(data, cfg, intervals=invalid)

    def test_rejects_invalid_parameters(self):
        for changes in ({'dt': 0}, {'vision_delay': -1}, {'vision_delay': float('inf')}, {'fade_in': -1}, {'coord_scale': 0},
                        {'lead': float('nan')}, {'bias_warmup': -1}, {'bias_memory': -1},
                        {'bias_memory': float('inf')},
                        {'bias_warmup': float('inf')}, {'fade_in': 0, 'flat': 0, 'fade_out': 0}):
            with self.assertRaises(ValueError):
                config(**changes).validate()


class VelocityUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'Політ with spaces'
        self.folder.mkdir()
        self.source = self.folder / 'flight.csv'
        self.source.write_text(
            'vis_x;vis_y;mheight;acc.x;acc.y;quat.x;quat.y;quat.z;quat.w;yaw\n'
            + ''.join(f'{i*0.05};{-i*0.1};10;0;0;0;0;0;1;0\n' for i in range(30)),
            encoding='utf-8')
        self.viewer = viewer.VelocityViewer(None, config(vx0=1., vy0=-2.), paused=True,
                                            export_path=self.folder / 'result_velocity.csv',
                                            export_follows_input=True)
        self.addCleanup(self.viewer.plt.close, self.viewer.fig)

    def test_open_export_apply_time_scale_and_mean_columns(self):
        v = self.viewer
        self.assertFalse(v.apply_button.active)
        self.assertTrue(v.load_csv(self.source))
        self.assertTrue(v.apply_button.active)
        self.assertEqual(v.export_path.name, 'result_velocity.csv')
        v.fields['coord_scale'].set_val('2')
        v.fields['dt'].set_val('0.1')
        v.fields['lead'].set_val('0.5')
        v.fields['view_window'].set_val('20')
        v.fields['bias_warmup'].set_val('6')
        v.fields['bias_memory'].set_val('45')
        self.assertTrue(v.apply_parameters())
        self.assertAlmostEqual(v.duration, 2.9)
        self.assertAlmostEqual(v.slider.valmax, 2.9)
        self.assertEqual(v.window, 20)
        np.testing.assert_allclose(v.result['velocity'], [[1., -2.]]*30, atol=1e-11)
        with v.export_path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter=';'))
        self.assertEqual(len(rows), 30)
        self.assertAlmostEqual(float(rows[-1]['forward_position_x']), 2.9 + 0.5)
        self.assertAlmostEqual(float(rows[-1]['estimated_speed_x']), float(rows[-1]['niacc.x']))
        self.assertIn('mean_visual_speed_x', rows[-1])
        self.assertIn('bias_body_used_x', rows[-1])
        self.assertEqual(float(rows[-1]['bias_warmup']), 6.)
        self.assertEqual(float(rows[-1]['bias_memory']), 45.)
        original = v.result
        previous_csv = v.export_path.read_bytes()
        v.fields['dt'].set_val('0')
        self.assertFalse(v.apply_parameters())
        self.assertIs(v.result, original)
        self.assertEqual(v.export_path.read_bytes(), previous_csv)

    def test_drop_then_fixed_bias_and_plotjuggler_export(self):
        v = self.viewer
        pending = []
        self.assertTrue(v.queue_dropped_files([str(self.source)], pending.append))
        self.assertIsNone(v.result)
        pending.pop()()
        self.assertIsNotNone(v.result)
        v.fields['bias_x0'].set_val('-0.1')
        v.bias_toggle.set_active(0)
        self.assertTrue(v.apply_parameters())
        np.testing.assert_allclose(v.result['bias_body'][:, 0], -0.1)
        self.assertFalse(v.cfg.learn_bias)
        binary = self.folder / 'plotjuggler'
        with patch.object(shared, 'find_plotjuggler', return_value=binary), \
                patch.object(shared, 'save_plotjuggler_path'), \
                patch.object(shared.subprocess, 'Popen') as launch:
            self.assertTrue(v.open_in_plotjuggler())
            self.assertEqual(launch.call_args.args[0], [str(binary), '-d', str(v.export_path.resolve())])
        self.assertIn('estimated_speed_x', v.export_path.read_text().splitlines()[0])

    def test_delay_control_and_timestamp_replay(self):
        v = self.viewer
        t = np.r_[0., np.cumsum(np.resize([.02, .05, .017], 60))]
        source = self.folder/'timed.csv'
        source.write_text('timestamp;vis_x;vis_y;mheight;acc.x;acc.y;quat.x;quat.y;quat.z;quat.w;yaw\n'
                          + ''.join(f'{1e9+ti*1000};{ti};{-2*ti};10;0;0;0;0;0;1;0\n' for ti in t),
                          encoding='utf-8')
        v.cfg = replace(v.cfg, time_column='timestamp', time_scale=.001)
        self.assertTrue(v.load_csv(source))
        v.fields['vision_delay'].set_val('.137')
        self.assertTrue(v.apply_parameters())
        np.testing.assert_allclose(v.result['time'], t, atol=1e-9)
        np.testing.assert_allclose(v.result['velocity'], np.tile([1., -2.], (len(t), 1)), atol=1e-7)
        v.current_time = t[27]+.001
        v.draw(force=True)
        self.assertEqual(v.last_index, 27)
        with v.export_path.open() as stream:
            row = list(csv.DictReader(stream, delimiter=';'))[-1]
        self.assertEqual(float(row['vision_delay']), .137)
        self.assertIn('delayed_speed_x', row)
        self.assertIn('prediction_delta_v_x', row)
        bad = self.folder/'bad_time.csv'
        bad.write_text(source.read_text().replace(str(1e9+20), str(1e9)), encoding='utf-8')
        original = v.result
        self.assertFalse(v.load_csv(bad))
        self.assertIs(v.result, original)

    def test_protects_input_and_failed_load_keeps_flight(self):
        v = self.viewer
        self.assertTrue(v.load_csv(self.source))
        previous = v.result
        self.assertFalse(v.load_csv(self.folder / 'missing.csv'))
        self.assertIs(v.result, previous)
        v.export_follows_input = False
        v.export_path = self.source
        before = self.source.read_bytes()
        self.assertFalse(v.load_csv(self.source))
        self.assertEqual(self.source.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
