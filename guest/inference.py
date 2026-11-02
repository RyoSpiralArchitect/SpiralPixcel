"""Bounded llama-bench run in the Pixel Debian VM, with retained raw evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

OUT = Path("result")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_benchmark(raw, repetitions, gpu_layers, decode_only=False):
    rows = json.loads(raw)
    expected = {(0, 32)} if decode_only else {(64, 0), (0, 32)}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("expected completed benchmark rows for the requested tests")
    by_test = {}
    for row in rows:
        test = (row.get("n_prompt"), row.get("n_gen"))
        if test not in expected or test in by_test:
            raise ValueError("unexpected or duplicate benchmark test")
        rate = row.get("avg_ts")
        samples = row.get("samples_ts")
        if (not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
                or not isinstance(samples, list) or len(samples) != repetitions
                or any(not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in samples)):
            raise ValueError("invalid or incomplete benchmark samples")
        if row.get("n_gpu_layers") != gpu_layers:
            raise ValueError("GPU layer request changed during benchmarking")
        if gpu_layers and row.get("devices") != "Vulkan0":
            raise ValueError("benchmark did not select Vulkan0")
        by_test[test] = row
    rates = {"decode_tg32_tokens_s": by_test[(0, 32)]["avg_ts"]}
    if not decode_only:
        rates["prefill_pp64_tokens_s"] = by_test[(64, 0)]["avg_ts"]
    return rates


def memory_sample(pid):
    """Return Linux process RSS/HWM, in KiB. Device allocations need separate accounting."""
    fields = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                fields[name] = int(value.split()[0])
    except (FileNotFoundError, ProcessLookupError):
        return {}
    return fields


def run_benchmark(command, environment, timeout):
    start = time.monotonic()
    peaks = {"VmRSS": 0, "VmHWM": 0}
    timed_out = False
    with (OUT / "bench.json").open("wb") as stdout, (OUT / "bench.log").open("wb") as stderr:
        with subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr) as child:
            while child.poll() is None:
                for name, value in memory_sample(child.pid).items():
                    peaks[name] = max(peaks[name], value)
                if time.monotonic() - start > timeout:
                    timed_out = True
                    child.kill()
                    break
                time.sleep(0.02)
            returncode = child.wait()
    return {"exit_code": returncode, "timed_out": timed_out,
            "benchmark_wall_seconds": round(time.monotonic() - start, 3),
            "sampled_peak_rss_kib": peaks["VmRSS"] or None,
            "sampled_peak_hwm_kib": peaks["VmHWM"] or None,
            "memory_method": "/proc/PID/status sampled every 20 ms; process memory, excludes separate GPU accounting"}


def logged_memory(raw):
    """Peak allocation totals reported by ggml's Vulkan logger, rounded by that logger."""
    pattern = re.compile(r"ggml_vulkan memory: ([^\s:]+): .*?Total device: ([0-9.]+) (B|KiB|MiB|GiB), total host: ([0-9.]+) (B|KiB|MiB|GiB)")
    scale = {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
    peaks = {}
    current = {}
    peak_device, peak_host = 0., 0.
    for device, dev_num, dev_unit, host_num, host_unit in pattern.findall(raw):
        allocated = float(dev_num) * scale[dev_unit]
        host = float(host_num) * scale[host_unit]
        entry = peaks.setdefault(device, {"device_mib": 0., "host_mib": 0.})
        entry["device_mib"] = max(entry["device_mib"], allocated)
        entry["host_mib"] = max(entry["host_mib"], host)
        current[device] = (allocated, host)
        peak_device = max(peak_device, sum(x[0] for x in current.values()))
        peak_host = max(peak_host, sum(x[1] for x in current.values()))
    return {"peak_device_allocated_mib": peak_device if peaks else None,
            "peak_host_allocated_mib": peak_host if peaks else None,
            "memory_logger_devices": peaks,
            "device_memory_method": "GGML_VK_MEMORY_LOGGER allocation totals; logger-rounded values, separate from process RSS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path.home() / ".local/share/spiral-pixel/vulkan-core16-v1/llama-b11960")
    parser.add_argument("--model", type=Path, default=Path.home() / ".local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile", choices=("baseline", "host-memory", "mmvq"), default="baseline")
    parser.add_argument("--disable-fp16", action="store_true", help="same-runtime FP32 control")
    parser.add_argument("--workgroup", type=int, choices=(32, 64, 128), help="opt-in patched PowerVR MMV specialization")
    parser.add_argument("--rms-subgroups", action="store_true")
    parser.add_argument("--flash-attn", choices=("off", "on"), default="off")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--repack", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--library-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.runtime.is_absolute() or not args.model.is_absolute() or any(not p.is_absolute() for p in args.library_path):
        parser.error("runtime, model and library paths must be absolute VM paths")
    if not (0 <= args.gpu_layers <= 999 and 1 <= args.repetitions <= 20 and 1 <= args.timeout <= 3600):
        parser.error("gpu layers: 0..999, repetitions: 1..20, timeout: 1..3600")
    OUT.mkdir(exist_ok=True)
    # A failed retry must never leave the previous success receipt visible.
    result = {"experiment": "inference", "status": "fail", "guest_arch": platform.machine(),
              "kernel": platform.release(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "vulkan": {"device": "unverified"}, "runtime": str(args.runtime), "model": str(args.model), "profile": args.profile}
    receipt = OUT / "result.json"
    receipt.unlink(missing_ok=True)
    command = [str(args.runtime / "llama-bench"), "-m", str(args.model), "-p", "0" if args.decode_only else "64", "-n", "32",
               "-b", "16", "-ub", "16", "-t", "4", "-fa", args.flash_attn, "-ngl", str(args.gpu_layers),
               "-dev", "Vulkan0" if args.gpu_layers else "none", "-r", str(args.repetitions),
               "--repack", str(args.repack), "--no-warmup", "-o", "json"]
    if not args.gpu_layers:
        command += ["--no-op-offload", "1"]
    result["command"] = command
    environment = os.environ.copy()
    for key in ("GGML_VK_PREFER_HOST_MEMORY", "GGML_VK_FORCE_MMVQ", "GGML_VK_DISABLE_MMVQ",
                "GGML_VK_PWR_MMV_WG", "GGML_VK_PWR_RMS_SUBGROUPS", "GGML_VK_PERF_LOGGER", "GGML_VK_DISABLE_F16"):
        environment.pop(key, None)
    environment["GGML_VK_MEMORY_LOGGER"] = "1"
    if args.profile == "host-memory":
        environment["GGML_VK_PREFER_HOST_MEMORY"] = "1"
    elif args.profile == "mmvq":
        environment["GGML_VK_FORCE_MMVQ"] = "1"
    if args.disable_fp16:
        environment["GGML_VK_DISABLE_F16"] = "1"
    if args.workgroup:
        environment["GGML_VK_PWR_MMV_WG"] = str(args.workgroup)
    if args.rms_subgroups:
        environment["GGML_VK_PWR_RMS_SUBGROUPS"] = "1"
    paths = [str(args.runtime), *map(str, args.library_path)]
    if environment.get("LD_LIBRARY_PATH"):
        paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(paths)
    result["environment"] = {key: value for key, value in environment.items()
                             if key.startswith("GGML_VK_") or key == "LD_LIBRARY_PATH"}
    try:
        if args.workgroup or args.rms_subgroups or (args.runtime / "spiralpixcel-kernel.json").exists():
            marker = json.loads((args.runtime / "spiralpixcel-kernel.json").read_text())
            if marker.get("libggml-vulkan_sha256") != sha256(args.runtime / "libggml-vulkan.so"):
                raise ValueError("patched runtime identity differs from its build receipt")
            if marker.get("check-mmv_sha256") != sha256(args.runtime / "check-mmv"):
                raise ValueError("kernel checker identity differs from its build receipt")
            with (OUT / "kernel-check.jsonl").open("wb") as out, (OUT / "kernel-check.log").open("wb") as err:
                subprocess.run([str(args.runtime / "check-mmv"), str(args.runtime)], env=environment,
                               stdout=out, stderr=err, check=True, timeout=180)
            checks = [json.loads(line) for line in (OUT / "kernel-check.jsonl").read_text().splitlines()]
            gate = checks[-1]
            operations = {}
            for row in checks[:-1]:
                operations[row.get("op")] = operations.get(row.get("op"), 0) + 1
            if (gate.get("status") != "pass" or gate.get("cases") != 56 or gate.get("failed") != 0
                    or operations != {"MUL_MAT": 40, "RMS_NORM": 8, "RMS_NORM_MUL": 8}):
                raise ValueError("CPU vs Vulkan numerical gate did not pass")
            result["kernel_check"] = gate
            result["kernel_build"] = marker
        result["model_bytes"] = args.model.stat().st_size
        result["model_sha256"] = sha256(args.model)
        files = [args.runtime / "llama-bench", *sorted(args.runtime.glob("*.so*"))]
        cache, hashes = {}, {}
        for path in files:
            resolved = path.resolve(strict=True)
            if resolved not in cache:
                cache[resolved] = sha256(resolved)
            hashes[path.name] = cache[resolved]
        result["runtime_sha256"] = hashes
        result.update(run_benchmark(command, environment, args.timeout))
        result.update(logged_memory((OUT / "bench.log").read_text(errors="replace")))
        if result["timed_out"] or result["exit_code"]:
            raise ValueError("benchmark did not complete; see bench.log")
        raw = (OUT / "bench.json").read_text()
        result.update(parse_benchmark(raw, args.repetitions, args.gpu_layers, args.decode_only))
        rows = json.loads(raw)
        result["vulkan"] = {"device": rows[0].get("gpu_info", "unreported") if args.gpu_layers else "CPU",
                            "selected": rows[0].get("devices")}
        result["status"] = "pass"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = OUT / "inference.tmp"
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(receipt)
    print(json.dumps({key: result[key] for key in ("experiment", "status", "vulkan")}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
