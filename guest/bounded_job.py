"""Run one experiment in a process group with a guest-owned deadline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(command, seconds, output):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    result = {"status": "fail", "command": command, "deadline_seconds": seconds}
    started = time.monotonic()
    stop = []
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, signal.SIG_IGN if sig == signal.SIGHUP else lambda signum, frame: stop.append(signum))
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        reason = None
        try:
            while child.poll() is None:
                if stop:
                    reason = "signal"
                    break
                if (output / "cancel").exists():
                    reason = "cancelled"
                    break
                if time.monotonic() - started >= seconds:
                    reason = "deadline"
                    break
                time.sleep(0.05)
        finally:
            if reason or child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                # Always kill the remaining group, even if its leader exited.
                # Otherwise a descendant ignoring TERM could outlive the lease.
                time.sleep(0.2)
                child.poll()  # Reap an exited leader before signalling its group again.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            result.update(exit_code=child.wait(), stop_reason=reason,
                          wall_seconds=round(time.monotonic() - started, 3))
    if result["exit_code"] == 0 and not reason:
        result["status"] = "pass"
    (output / "job.tmp").write_text(json.dumps(result, indent=2) + "\n")
    (output / "job.tmp").replace(output / "job.json")
    for sig, handler in old_handlers.items():
        signal.signal(sig, handler)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path, default=Path("result"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.seconds <= 120:
        parser.error("provide a command and a deadline of 0..120 seconds")
    sys.exit(0 if run(command, args.seconds, args.output)["status"] == "pass" else 1)
