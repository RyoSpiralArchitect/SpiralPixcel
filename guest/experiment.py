"""Run inside a private per-run directory in Debian, using only the standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

OUT = Path("result")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version(cmd):
    return subprocess.check_output(cmd, text=True).splitlines()[0]


def dispatch(width, height, steps, rate, decay, software, output):
    raw = subprocess.check_output(["./compute", str(OUT / "shader.spv"), str(OUT / "input.bin"),
                                   str(output), *map(str, (width, height, steps, rate, decay)),
                                   "allow-software" if software else "hardware"], text=True)
    fields, extensions = {}, []
    for line in raw.splitlines():
        key, value = line.split("\t", 1)
        if key == "extension":
            extensions.append(value)
        else:
            fields[key] = value
    fields["extensions"] = sorted(extensions)
    for key in ("queue_family", "device_type", "timestamp_valid_bits", "driver_version"):
        fields[key] = int(fields[key])
    fields["gpu_ms"] = float(fields["gpu_ms"]) if "gpu_ms" in fields else None
    fields["submit_wait_ms"] = float(fields["submit_wait_ms"])
    return fields


def doctor(software):
    width, height = 37, 7  # Both dimensions exercise the shader's edge guard.
    values = [(0xA5A50001 + 2654435761*i) & 0xffffffff for i in range(width*height)]
    (OUT / "input.bin").write_bytes(struct.pack(f"<{len(values)}I", *values))
    dest = OUT / "gpu.bin"
    vk = dispatch(width, height, 1, 0, 1, software, dest)
    actual = struct.unpack(f"<{len(values)}I", dest.read_bytes())
    wrong = sum(v != (2*x & 0xffffffff) for x, v in zip(values, actual))
    return {"vulkan": vk, "elements": len(values), "mismatches": wrong, "status": "fail" if wrong else "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["doctor"])
    parser.add_argument("--allow-software", action="store_true", help="explicit software Vulkan for CI")
    args = parser.parse_args()
    if sys.byteorder != "little":
        parser.error("this experiment expects a little-endian host")
    OUT.mkdir(exist_ok=True)
    shader = Path("shaders") / (args.experiment + ".comp")
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-missing-field-initializers",
                    "compute.cpp", "-lvulkan", "-o", "compute"], check=True)
    subprocess.run(["glslc", "--target-env=vulkan1.0", str(shader), "-o", OUT / "shader.spv"], check=True)
    result = doctor(args.allow_software)
    result.update(experiment=args.experiment, guest_arch=platform.machine(), kernel=platform.release(),
                  compiler=version(["c++", "--version"]), glslc=version(["glslc", "--version"]),
                  sha256={str(p): digest(p) for p in ["compute.cpp", "experiment.py", shader, OUT / "shader.spv", OUT / "input.bin", OUT / "gpu.bin"]})
    tmp = OUT / "receipt.tmp"
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(OUT / "result.json")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
