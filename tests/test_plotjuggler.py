import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')

import kalman_flight_viewer as viewer


class PlotJugglerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'Flight files with spaces'
        self.folder.mkdir()
        self.settings = self.folder / 'settings.json'
        self.settings_patch = patch.object(viewer, 'viewer_settings_path', return_value=self.settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.environment_patch = patch.dict(os.environ, {'PLOTJUGGLER_PATH': ''})
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.executable = self.folder / ('plotjuggler.exe' if sys.platform == 'win32' else 'plotjuggler')
        self.executable.touch()
        self.executable.chmod(0o755)

    def make_viewer(self, loaded=True):
        source = self.folder / 'flight.csv'
        source.write_text(
            'vis_x;vis_y;mheight;acc_x;acc_y;quat.x;quat.y;quat.z;quat.w;yaw\n'
            '1;2;10;0;0;0;0;0;1;0\n'
            '1;2;10;0;0;0;0;0;1;0\n', encoding='utf-8')
        cfg = viewer.Config(quaternion_direction='body-to-ned', yaw_unit='deg')
        data = viewer.read_flight(source)[0] if loaded else None
        result = viewer.estimate(data, cfg) if loaded else None
        v = viewer.FlightViewer(result, cfg, data=data, input_path=source if loaded else None,
                                export_path=self.folder / 'результат ; flight.csv', paused=True)
        self.addCleanup(v.plt.close, v.fig)
        return v

    def test_saved_and_explicit_executable(self):
        viewer.save_plotjuggler_path(self.executable)
        self.assertEqual(viewer.find_plotjuggler(), self.executable.resolve())
        override = self.folder / ('other.exe' if sys.platform == 'win32' else 'other')
        override.touch()
        override.chmod(0o755)
        self.assertEqual(viewer.find_plotjuggler(override), override.resolve())

    def test_path_lookup_after_stale_setting(self):
        viewer.save_plotjuggler_path(self.folder / 'deleted')
        with patch.object(viewer.shutil, 'which', return_value=str(self.executable)):
            self.assertEqual(viewer.find_plotjuggler(), self.executable.resolve())
        self.settings.write_text('not JSON', encoding='utf-8')
        self.assertIsNone(viewer.saved_plotjuggler_path())

    def test_windows_program_files_lookup(self):
        binary = self.folder / 'PlotJuggler' / 'bin' / 'plotjuggler.exe'
        binary.parent.mkdir(parents=True)
        binary.touch()
        with patch.object(sys, 'platform', 'win32'), \
                patch.dict(os.environ, {'ProgramFiles': str(self.folder)}), \
                patch.object(viewer, 'windows_plotjuggler_locations', return_value=iter(())), \
                patch.object(viewer.shutil, 'which', return_value=None):
            self.assertEqual(viewer.find_plotjuggler(), binary.resolve())

    def test_windows_custom_registry_install(self):
        binary = self.folder / 'custom' / 'bin' / 'plotjuggler.exe'
        binary.parent.mkdir(parents=True)
        binary.touch()
        with patch.object(sys, 'platform', 'win32'), \
                patch.object(viewer, 'windows_plotjuggler_locations', return_value=iter([str(binary.parent.parent)])), \
                patch.object(viewer.shutil, 'which', return_value=None):
            self.assertEqual(viewer.find_plotjuggler(), binary.resolve())

    def test_empty_window_cannot_launch(self):
        v = self.make_viewer(False)
        self.assertFalse(v.plotjuggler_button.active)
        with patch.object(viewer.subprocess, 'Popen') as launch:
            self.assertFalse(v.open_in_plotjuggler())
            launch.assert_not_called()

    def test_launch_applies_pending_fields_and_exports(self):
        v = self.make_viewer()
        v.fields['coord_scale'].set_val('1.4')
        with patch.object(viewer, 'find_plotjuggler', return_value=self.executable), \
                patch.object(viewer.subprocess, 'Popen') as launch:
            self.assertTrue(v.open_in_plotjuggler())
            args, options = launch.call_args
        self.assertEqual(args[0], [str(self.executable), '-d', str(v.export_path.resolve())])
        self.assertFalse(options['shell'])
        self.assertEqual(options['cwd'], str(self.executable.parent))
        with v.export_path.open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f, delimiter=';'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]['vis_scaled_x']), 1.4)
        self.assertEqual(float(rows[0]['vis_x']), 1)
        self.assertEqual(viewer.saved_plotjuggler_path(), str(self.executable))

    def test_missing_program_asks_once_and_remembers_selection(self):
        v = self.make_viewer()
        with patch.object(viewer, 'find_plotjuggler', return_value=None), \
                patch.object(v, 'choose_plotjuggler_path', return_value=str(self.executable)) as choose, \
                patch.object(viewer.subprocess, 'Popen'):
            self.assertTrue(v.open_in_plotjuggler())
            choose.assert_called_once()
        self.assertEqual(viewer.saved_plotjuggler_path(), str(self.executable.resolve()))

    def test_cancel_preserves_output_and_playback(self):
        v = self.make_viewer()
        v.paused = False
        v.export_path.write_text('old export', encoding='utf-8')
        with patch.object(viewer, 'find_plotjuggler', return_value=None), \
                patch.object(v, 'choose_plotjuggler_path', return_value=''), \
                patch.object(viewer.subprocess, 'Popen') as launch:
            self.assertFalse(v.open_in_plotjuggler())
            launch.assert_not_called()
        self.assertFalse(v.paused)
        self.assertEqual(v.export_path.read_text(encoding='utf-8'), 'old export')

    def test_failed_export_does_not_launch_or_commit_settings(self):
        v = self.make_viewer()
        previous = v.result
        v.fields['coord_scale'].set_val('2')
        with patch.object(viewer, 'find_plotjuggler', return_value=self.executable), \
                patch.object(viewer, 'export_estimates', side_effect=PermissionError('CSV locked')), \
                patch.object(viewer.subprocess, 'Popen') as launch:
            self.assertFalse(v.open_in_plotjuggler())
            launch.assert_not_called()
        self.assertEqual(v.cfg.coord_scale, 1)
        self.assertIs(v.result, previous)
        self.assertFalse(self.settings.exists())

    def test_failed_launch_is_reported(self):
        v = self.make_viewer()
        with patch.object(viewer, 'find_plotjuggler', return_value=self.executable), \
                patch.object(viewer.subprocess, 'Popen', side_effect=OSError('Cannot execute')):
            self.assertFalse(v.open_in_plotjuggler())
        self.assertIn('Cannot execute', v.message.get_text())
        self.assertFalse(self.settings.exists())

    @unittest.skipIf(sys.platform == 'win32', 'POSIX executable stub for subprocess integration')
    def test_real_child_receives_csv_as_one_argument(self):
        v = self.make_viewer()
        received = self.folder / 'received.json'
        self.executable.write_text(
            '#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n'
            f'Path({str(received)!r}).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")\n',
            encoding='utf-8')
        with patch.object(viewer, 'find_plotjuggler', return_value=self.executable):
            self.assertTrue(v.open_in_plotjuggler())
        self.assertEqual(v.external_processes[-1].wait(timeout=5), 0)
        self.assertEqual(json.loads(received.read_text(encoding='utf-8')),
                         ['-d', str(v.export_path.resolve())])


if __name__ == '__main__':
    unittest.main()
