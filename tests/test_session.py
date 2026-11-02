import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))
from lab_session import check, parse_state, prepare, restore, run_bounded


class FakeDevice:
    def __init__(self):
        self.values = {("global", "stay_on_while_plugged_in"): "0", ("system", "screen_off_timeout"): "60000"}
        self.identity = "first-device"
        self.locked = False

    def get(self, namespace, key): return self.values[(namespace, key)]
    def put(self, namespace, key, value): self.values[(namespace, key)] = value
    def fingerprint(self): return self.identity
    def adb(self, *args): return b""
    def frame(self): return {"content_assessed": False}
    def state(self): return {"wakefulness": "Awake", "locked": self.locked, "admin_max_screen_off_ms": 60000}


class SessionTest(unittest.TestCase):
    def test_enforced_timeout_and_unknown_lock_remain_visible(self):
        power = "mWakefulness=Awake mIsPowered=true mStayOn=false mMaximumScreenOffTimeoutFromDeviceAdmin=60000 (enforced=true)"
        self.assertEqual(parse_state(power, "")['locked'], None)
        state = parse_state(power, "KeyguardServiceDelegate\n showing=true")
        self.assertEqual(state['admin_max_screen_off_ms'], 60000)
        self.assertFalse(state['stay_on_effective'])
        self.assertTrue(state['locked'])

    def test_prepare_preserves_original_backup_and_restore_is_device_bound(self):
        device = FakeDevice()
        with tempfile.TemporaryDirectory() as folder:
            original = dict(device.values)
            prepare(device, folder)
            self.assertEqual(device.values[("system", "screen_off_timeout")], "1800000")
            with self.assertRaises(FileExistsError): prepare(device, folder)
            device.identity = "second-device"
            with self.assertRaisesRegex(ValueError, "another USB"): restore(device, folder)
            device.identity = "first-device"
            restore(device, folder)
            self.assertEqual(device.values, original)
            self.assertFalse((Path(folder) / "lab-display.json").exists())

    def test_locked_background_use_is_optional_but_unknown_state_fails(self):
        device = FakeDevice()
        device.locked = True
        self.assertTrue(check(device)['locked'])
        with self.assertRaisesRegex(ValueError, "unlock"): check(device, require_unlocked=True)
        device.locked = None
        with self.assertRaisesRegex(ValueError, "unknown"): check(device)

    def test_frame_failure_never_uploads_or_starts_a_guest_job(self):
        device = FakeDevice()
        with tempfile.TemporaryDirectory() as folder, patch('lab_session.Path', wraps=Path) as paths:
            # Keep this test's receipt under its own directory.
            paths.side_effect = lambda p: Path(folder) / p if p == "runs" else Path(p)
            with patch.object(device, 'frame', side_effect=ValueError('capture stalled')), \
                    patch('lab_session.subprocess.run') as remote:
                self.assertEqual(run_bounded(device, ['ssh'], ROOT, ['true']), 1)
            remote.assert_not_called()
            receipt = next((Path(folder) / 'runs').glob('*/session.json'))
            self.assertEqual(json.loads(receipt.read_text())['status'], 'fail')

    def test_guest_deadline_survives_host_hangup_and_kills_ignoring_child(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'result'
            command = [sys.executable, '-IS', str(ROOT / 'guest/bounded_job.py'), '--seconds', '0.4', '--output', str(out),
                       '--', sys.executable, '-IS', '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']
            child = subprocess.Popen(command)
            time.sleep(0.1)
            child.send_signal(signal.SIGHUP)
            self.assertEqual(child.wait(timeout=4), 1)
            result = json.loads((out / 'job.json').read_text())
            self.assertEqual(result['stop_reason'], 'deadline')
            self.assertEqual(result['exit_code'], -signal.SIGKILL)

    def test_guest_cancel_and_success_have_distinct_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            for cancel in (False, True):
                out = Path(folder) / str(cancel)
                out.mkdir()
                if cancel: (out / 'cancel').touch()
                command = [sys.executable, '-IS', str(ROOT / 'guest/bounded_job.py'), '--seconds', '1', '--output', str(out),
                           '--', sys.executable, '-IS', '-c', 'import time; time.sleep(0.2)']
                run = subprocess.run(command, timeout=4)
                result = json.loads((out / 'job.json').read_text())
                self.assertEqual(result['status'], 'fail' if cancel else 'pass')
                self.assertEqual(run.returncode, 1 if cancel else 0)
                self.assertEqual(result['stop_reason'], 'cancelled' if cancel else None)


if __name__ == '__main__': unittest.main()
