"""Optional, reversible Android lab setup and bounded experiment supervision."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess
import time
import uuid


def parse_state(power, window):
    def field(name):
        match = re.search(r"\b" + re.escape(name) + r"=([^\s]+)", power)
        return match.group(1) if match else None

    delegate = window.split("KeyguardServiceDelegate", 1)
    locked = re.search(r"\bshowing=(true|false)", delegate[1]) if len(delegate) == 2 else None
    timeout = field("mMaximumScreenOffTimeoutFromDeviceAdmin")
    return {"wakefulness": field("mWakefulness"), "powered": field("mIsPowered") == "true",
            "stay_on_effective": field("mStayOn") == "true",
            "admin_max_screen_off_ms": int(timeout) if timeout and timeout.isdigit() else None,
            "locked": locked.group(1) == "true" if locked else None}


class Device:
    def adb(self, *arguments, timeout=4):
        result = subprocess.run(["adb", "-d", *arguments], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, check=True)
        return result.stdout

    def state(self):
        return parse_state(self.adb("shell", "dumpsys", "power").decode(),
                           self.adb("shell", "dumpsys", "window", "policy").decode())

    def fingerprint(self):
        return hashlib.sha256(self.adb("get-serialno").strip()).hexdigest()

    def frame(self):
        raw = self.adb("exec-out", "screencap", "-p")
        if len(raw) < 45 or raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" or raw[-8:-4] != b"IEND":
            raise ValueError("Android frame capture did not return a complete PNG")
        width, height = struct.unpack(">II", raw[16:24])
        if not 0 < width <= 8192 or not 0 < height <= 8192:
            raise ValueError("unexpected display dimensions")
        # Screen contents never leave this process or enter the public receipt.
        return {"width": width, "height": height, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "content_assessed": False}

    def get(self, namespace, key):
        value = self.adb("shell", "settings", "get", namespace, key).decode().strip()
        if value != "null" and not re.fullmatch(r"[0-9]+", value):
            raise ValueError("unexpected Android setting value")
        return value

    def put(self, namespace, key, value):
        action = "delete" if value == "null" else "put"
        arguments = (value,) if action == "put" else ()
        self.adb("shell", "settings", action, namespace, key, *arguments)


SETTINGS = (("global", "stay_on_while_plugged_in", "3"),
            ("system", "screen_off_timeout", "1800000"))


def prepare(device, state_dir):
    state_dir = Path(state_dir)
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    backup = state_dir / "lab-display.json"
    original = {"device_sha256": device.fingerprint(), "settings": [
        {"namespace": namespace, "key": key, "value": device.get(namespace, key)}
        for namespace, key, _ in SETTINGS]}
    # An existing backup must never be replaced by already-modified settings.
    with backup.open("x") as stream:
        json.dump(original, stream, indent=2)
        stream.write("\n")
    for namespace, key, value in SETTINGS:
        device.put(namespace, key, value)
    device.adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    return {"requested": "AC/USB stay-awake and 30-minute timeout", "effective": device.state(),
            "backup": str(backup), "policies_changed": False, "sync_changed": False}


def restore(device, state_dir):
    backup = Path(state_dir) / "lab-display.json"
    original = json.loads(backup.read_text())
    if original["device_sha256"] != device.fingerprint():
        raise ValueError("display backup belongs to another USB device")
    for setting in original["settings"]:
        device.put(setting["namespace"], setting["key"], setting["value"])
        if device.get(setting["namespace"], setting["key"]) != setting["value"]:
            raise ValueError("display setting restoration was not confirmed; backup retained")
    backup.unlink()
    return {"restored": True, "effective": device.state()}


def check(device, require_unlocked=False):
    state = device.state()
    if state["wakefulness"] is None or state["locked"] is None:
        raise ValueError("Android screen/lock state is unknown")
    if require_unlocked and (state["locked"] or state["wakefulness"] != "Awake"):
        raise ValueError("unlock and wake the Pixel before this display-dependent experiment")
    return {**state, "frame": device.frame(),
            "scope": "ADB and frame-capture responsiveness; not visual correctness or GPU recovery guarantee"}


def recover(device, stop_vm=False, reboot=False):
    if reboot:
        device.adb("reboot")
        return {"reboot_requested": True, "unlock_required_after_boot": True}
    if stop_vm:
        device.adb("shell", "am", "force-stop", "com.android.virtualization.terminal")
    device.adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    device.adb("shell", "input", "keyevent", "KEYCODE_HOME")
    return {"terminal_stopped": stop_vm, "check": check(device)}


def run_bounded(device, ssh, root, command, seconds=20, require_unlocked=False):
    if not command or not 0 < seconds <= 120:
        raise ValueError("provide a guest command and a deadline of 0..120 seconds")
    name = "session-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    output = Path("runs") / name
    output.mkdir(parents=True)
    remote = ".local/share/SpiralPixcel/sessions/" + name
    report = {"status": "fail", "seconds": seconds, "command": command,
              "require_unlocked": require_unlocked, "auto_reboot": False}
    job = None
    uploaded = False
    try:
        report["before"] = check(device, require_unlocked)
        source = (Path(root) / "guest/bounded_job.py").read_bytes()
        report["supervisor_sha256"] = hashlib.sha256(source).hexdigest()
        subprocess.run(ssh + [f"umask 077 && mkdir -p {remote}/result && cat > {remote}/bounded_job.py"],
                       input=source, check=True, timeout=10)
        uploaded = True
        remote_command = (f"cd {remote} && python3 -IS bounded_job.py --seconds {seconds} -- " + shlex.join(command))
        with (output / "transport.log").open("wb") as log:
            job = subprocess.Popen(ssh + [remote_command], stdout=log, stderr=log)
            deadline = time.monotonic() + seconds + 10
            while job.poll() is None:
                if time.monotonic() >= deadline:
                    raise ValueError("guest supervisor exceeded its deadline")
                time.sleep(1)
                if job.poll() is None:
                    check(device, require_unlocked)
            report["ssh_exit_code"] = job.returncode
        report["after"] = check(device, require_unlocked)
        if job.returncode:
            raise ValueError("bounded guest job failed; inspect job.json and stderr.log")
        report["status"] = "pass"
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        report["error"] = str(error)
        if job and job.poll() is None:
            try:
                subprocess.run(ssh + [f": > {remote}/result/cancel"], timeout=5, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                job.wait(timeout=3)
            except (OSError, subprocess.SubprocessError):
                report["cancel_unconfirmed"] = True
                job.terminate()
                try:
                    job.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    job.kill()
                    job.wait()
    finally:
        for filename in (("job.json", "stdout.log", "stderr.log") if uploaded else ()):
            try:
                result = subprocess.run(ssh + [f"cat {remote}/result/{filename}"], capture_output=True,
                                        check=True, timeout=5)
                (output / filename).write_bytes(result.stdout)
            except (OSError, subprocess.SubprocessError):
                report.setdefault("uncollected", []).append(filename)
        if (output / "job.json").is_file():
            try:
                guest = json.loads((output / "job.json").read_text())
                report["guest"] = guest
                if guest.get("status") != "pass":
                    report["status"] = "fail"
            except (OSError, ValueError):
                report["status"] = "fail"
        else:
            report["status"] = "fail"
        (output / "session.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output.resolve()), "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1
