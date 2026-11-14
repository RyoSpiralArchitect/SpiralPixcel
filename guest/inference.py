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


def validate_core_libraries(runtime, marker, required=False):
    """Verify every core SONAME, versioned file and alias against the build receipt."""
    if not isinstance(marker, dict):
        raise ValueError("runtime build receipt must be an object")
    if "core_library_sha256" not in marker:
        if required:
            raise ValueError("split runtime requires a matched core library manifest")
        return {}
    manifest = marker["core_library_sha256"]
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("core library manifest must be a nonempty object")
    name_pattern = re.compile(r"libggml(?:-base)?\.so(?:\.[0-9]+)*")
    for name, digest in manifest.items():
        if not isinstance(name, str) or not name_pattern.fullmatch(name):
            raise ValueError("invalid core library manifest filename")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid core library digest for {name}")
    runtime = Path(runtime)
    actual_names = {p.name for p in runtime.iterdir() if name_pattern.fullmatch(p.name)}
    # These are the pinned runtime's dependency SONAMEs, which must resolve locally.
    required_aliases = {"libggml.so.0", "libggml-base.so.0"}
    if not required_aliases <= actual_names or set(manifest) != actual_names:
        raise ValueError("core library manifest does not cover the runtime's complete core library family")
    verified = {}
    for name, expected in manifest.items():
        try:
            # Hash aliases too: resolving a symlink to different bytes must fail.
            observed = sha256((runtime / name).resolve(strict=True))
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"cannot read core library {name}") from exc
        if observed != expected:
            raise ValueError(f"{name} differs from the matched core library build receipt")
        verified[name] = observed
    return verified


def validate_runtime_files(runtime, marker):
    """Bind the split experiment to every bundled library alias and used executable."""
    runtime = Path(runtime)
    executables = {"llama-bench", "llama-completion", "check-mmv", "check-large", "check-model", "check-readback"}
    libraries = {p.name for p in runtime.glob("*.so*")}
    if "libllama.so.0" not in libraries:
        raise ValueError("split runtime requires its local libllama.so.0 dependency alias")
    specifications = (
        ("runtime_library_sha256", libraries, r"lib[A-Za-z0-9_.+-]+\.so(?:\.[0-9]+)*"),
        ("runtime_executable_sha256", executables, r"[A-Za-z0-9_-]+"),
    )
    verified = {}
    for key, expected_names, pattern in specifications:
        manifest = marker.get(key)
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError(f"split runtime requires a complete {key} manifest")
        for name, digest in manifest.items():
            if not isinstance(name, str) or not re.fullmatch(pattern, name):
                raise ValueError(f"invalid filename in {key}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid digest in {key}")
        if set(manifest) != expected_names:
            raise ValueError(f"{key} does not cover the complete split runtime")
        for name, expected in manifest.items():
            try:
                actual = sha256((runtime / name).resolve(strict=True))
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"cannot read split runtime file {name}") from exc
            if actual != expected:
                raise ValueError(f"{name} differs from {key}")
        verified[key] = dict(manifest)
    return verified


# check-mmv layouts: the original 56 cases, 68 with Q8_0 mat-vec cases added for the unsigned
# PowerVR kernel, 76 with single-token attention-score cases for the PowerVR decode path, and 102
# with 4864-row mat-vecs for the unsigned kernel's tall-matrix variant. Each switch must be checked
# on a layout that covers it.
KERNEL_GATES = {56: {"MUL_MAT": 40, "RMS_NORM": 8, "RMS_NORM_MUL": 8},
                68: {"MUL_MAT": 52, "RMS_NORM": 8, "RMS_NORM_MUL": 8},
                76: {"MUL_MAT": 52, "ATTN_SCORES": 8, "RMS_NORM": 8, "RMS_NORM_MUL": 8},
                102: {"MUL_MAT": 78, "ATTN_SCORES": 8, "RMS_NORM": 8, "RMS_NORM_MUL": 8}}


# The u8 patch logs the first use of its kernel per weight type. The extended gate layout
# alone does not show that the kernel ran: an unavailable pipeline falls back silently.
U8_SELECTED = re.compile(r"ggml_vulkan: mul_mat_vecq_u8 selected for (\S+)")
# Runtimes built before the Q6_K kernel do not list their types in the receipt.
U8_GATE_TYPES = {"q4_K", "q5_0", "q8_0"}
# Runtimes whose receipt lists shape variants of the kernel log the first use of each.
U8_VARIANT_SELECTED = re.compile(r"ggml_vulkan: mul_mat_vecq_u8 variant (\S+) selected for (\S+)")
# The decode patch logs the first use of its attention-score kernel in the same way.
ATTN_SELECTED = "ggml_vulkan: mul_mat_vec_p021_pvr selected"


def u8_selected_types(log_path):
    return sorted(set(U8_SELECTED.findall(Path(log_path).read_text(errors="replace"))))


def u8_selected_variants(log_path):
    found = U8_VARIANT_SELECTED.findall(Path(log_path).read_text(errors="replace"))
    return sorted({f"{type_name}/{variant}" for variant, type_name in found})


def attention_kernel_selected(log_path):
    return ATTN_SELECTED in Path(log_path).read_text(errors="replace").splitlines()


def check_kernel_gate(raw, require_q8_0=False, require_attention=False, require_tall=False):
    checks = [json.loads(line) for line in raw.splitlines()]
    if not checks or any(not isinstance(row, dict) for row in checks):
        raise ValueError("CPU vs Vulkan numerical gate has an incomplete receipt")
    gate, rows = checks[-1], checks[:-1]
    expected = KERNEL_GATES.get(len(rows))
    if (expected is None or (require_q8_0 and len(rows) < 68) or (require_attention and "ATTN_SCORES" not in expected)
            or (require_tall and expected["MUL_MAT"] < 78)):
        raise ValueError("CPU vs Vulkan numerical gate has an incomplete receipt")
    operations = {}
    for row in rows:
        operations[row.get("op")] = operations.get(row.get("op"), 0) + 1
    if (gate.get("status") != "pass" or gate.get("cases") != len(rows) or gate.get("failed") != 0
            or any(row.get("status") != "pass" for row in rows) or operations != expected):
        raise ValueError("CPU vs Vulkan numerical gate did not pass")
    return gate


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


def run_model_gate(runtime, model, gpu_layers, environment, output, *, model_sha256, runtime_manifest):
    """Compare a fixed prompt's complete CPU/GPU logits before collecting timings."""
    from model_check import compare_vectors, read_vector

    runtime, model, output = Path(runtime), Path(model), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = json.dumps(runtime_manifest, sort_keys=True, separators=(",", ":")).encode()
    report = {"status": "fail", "technical_validation_only": True, "nmse_limit": 5e-3,
              "model_sha256": model_sha256, "runtime_manifest": runtime_manifest,
              "runtime_manifest_sha256": hashlib.sha256(identity).hexdigest(),
              "checker_sha256": sha256(runtime / "check-model"), "executions": {},
              "environment": {key: value for key, value in environment.items()
                              if key.startswith("GGML_VK_") or key == "LD_LIBRARY_PATH"},
              "parameters": {"context_tokens": 512, "batch_tokens": 16, "ubatch_tokens": 16,
                             "threads": 4, "flash_attn": "off", "output_positions": 1,
                             "weight_repack": "runtime default"}}
    paths = [output / f"model-{arm}.{suffix}" for arm in ("reference", "candidate")
             for suffix in ("bin", "json", "log")]
    for path in paths:
        path.unlink(missing_ok=True)
    try:
        for arm, layers in (("reference", 0), ("candidate", gpu_layers)):
            command = [str(runtime / "check-model"), str(runtime), str(model),
                       str(output / f"model-{arm}.bin"), str(layers)]
            env = environment.copy()
            split = "0" if arm == "reference" else environment.get("GGML_VK_SPLIT_OUTPUT", "0")
            env["GGML_VK_SPLIT_OUTPUT"] = split
            env.pop("GGML_SCHED_DEBUG", None)
            execution = {"command": command, "gpu_layers": layers,
                         "environment_overrides": {"GGML_VK_SPLIT_OUTPUT": split}}
            report["executions"][arm] = execution
            with (output / f"model-{arm}.json").open("wb") as stdout, \
                    (output / f"model-{arm}.log").open("wb") as stderr:
                try:
                    child = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=180)
                except subprocess.TimeoutExpired:
                    execution["timed_out"] = True
                    raise
            execution["exit_code"] = child.returncode
            if child.returncode:
                raise ValueError(f"full-model {arm} check exited with {child.returncode}")
        reference, _ = read_vector(output / "model-reference.bin")
        candidate, _ = read_vector(output / "model-candidate.bin")
        reference_meta = json.loads((output / "model-reference.json").read_text())
        candidate_meta = json.loads((output / "model-candidate.json").read_text())
        if not isinstance(candidate_meta, dict) or candidate_meta.get("gpu_layers") != gpu_layers:
            raise ValueError("full-model candidate changed the requested GPU layer count")
        report.update(compare_vectors(reference, candidate, reference_meta, candidate_meta, nmse_limit=5e-3))
        if report["status"] != "pass":
            raise ValueError("full-model logits exceed the NMSE bound or change the top1 token")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
    report["sha256"] = {path.name: sha256(path) for path in paths if path.is_file()}
    temporary = output / "model-check.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "model-check.json")
    if report["status"] != "pass":
        raise ValueError("full-model CPU/GPU logit gate failed; see model-check.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path.home() / ".local/share/spiral-pixel/vulkan-core16-v1/llama-b11960")
    parser.add_argument("--model", type=Path, default=Path.home() / ".local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile", choices=("baseline", "host-memory", "mmvq"), default="baseline")
    parser.add_argument("--split-output", action="store_true", help="opt-in bounded Qwen2 output-head row views")
    parser.add_argument("--schedule-report", action="store_true", help="record CPU/GPU graph placement before timing")
    parser.add_argument("--sync-transfers", action="store_true", help="use Vulkan's synchronous backend mode; automatic with --split-output")
    parser.add_argument("--completion-wait", choices=("queue", "serial"), default="queue",
                        help="queue completion wait (patched runtime), or slow per-operation diagnostic waits")
    parser.add_argument("--disable-fp16", action="store_true", help="same-runtime FP32 control")
    parser.add_argument("--workgroup", type=int, choices=(32, 64, 128), help="opt-in patched PowerVR MMV specialization")
    parser.add_argument("--rms-subgroups", action="store_true")
    parser.add_argument("--u8-mmvq", action="store_true",
                        help="patched PowerVR Q5_0/Q8_0/Q4_K/Q6_K single-column mat-vec on unsigned 4x8 dots")
    parser.add_argument("--powervr-decode", action="store_true",
                        help="patched PowerVR decode path: attention-score kernel, compute-only barriers, "
                             "device-local buffers")
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
    if args.split_output and not args.gpu_layers:
        parser.error("--split-output requires at least one GPU layer")
    if args.split_output and args.flash_attn != "off":
        parser.error("--split-output requires --flash-attn off to match its fixed-prompt gate")
    if args.sync_transfers and (not args.gpu_layers or args.flash_attn != "off"):
        parser.error("--sync-transfers requires GPU layers and --flash-attn off for its fixed-prompt gate")
    if args.u8_mmvq and not (args.split_output or args.sync_transfers):
        parser.error("--u8-mmvq requires --split-output or --sync-transfers, so its full-model logit gate runs")
    if args.powervr_decode and not args.split_output:
        parser.error("--powervr-decode requires --split-output, the runtime it is built into, so its full-model logit gate runs")
    if args.powervr_decode and args.profile == "host-memory":
        # The decode path places device buffers in device-local memory, which would override the profile.
        parser.error("--powervr-decode cannot be combined with --profile host-memory")
    if (args.split_output or args.sync_transfers) and args.repack != 1:
        parser.error("checked transfer profiles require --repack 1 to match the pinned model checker's default")
    OUT.mkdir(exist_ok=True)
    # A failed retry must never leave the previous success receipt visible.
    result = {"experiment": "inference", "status": "fail", "guest_arch": platform.machine(),
              "kernel": platform.release(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "vulkan": {"device": "unverified"}, "runtime": str(args.runtime), "model": str(args.model), "profile": args.profile}
    receipt = OUT / "result.json"
    receipt.unlink(missing_ok=True)
    # Comparisons across runs need to show they shared one VM boot.
    try:
        result["vm"] = {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        "uptime_seconds": float(Path("/proc/uptime").read_text().split()[0])}
    except (OSError, ValueError, IndexError):
        result["vm"] = None
    command = [str(args.runtime / "llama-bench"), "-m", str(args.model), "-p", "0" if args.decode_only else "64", "-n", "32",
               "-b", "16", "-ub", "16", "-t", "4", "-fa", args.flash_attn, "-ngl", str(args.gpu_layers),
               "-dev", "Vulkan0" if args.gpu_layers else "none", "-r", str(args.repetitions),
               "--repack", str(args.repack), "--no-warmup", "-o", "json"]
    if not args.gpu_layers:
        command += ["--no-op-offload", "1"]
    result["command"] = command
    environment = os.environ.copy()
    for key in ("GGML_VK_PREFER_HOST_MEMORY", "GGML_VK_FORCE_MMVQ", "GGML_VK_DISABLE_MMVQ",
                "GGML_VK_PWR_MMV_WG", "GGML_VK_PWR_RMS_SUBGROUPS", "GGML_VK_PWR_U8_MMVQ", "GGML_VK_PERF_LOGGER", "GGML_VK_DISABLE_F16", "GGML_VK_SPLIT_OUTPUT", "GGML_VK_DISABLE_ASYNC", "GGML_VK_SERIALIZE_SUBMISSIONS", "GGML_VK_WAIT_IDLE",
                "GGML_VK_PWR_ATTN", "GGML_VK_PWR_BARRIER", "GGML_VK_PREFER_DEVICE_LOCAL"):
        environment.pop(key, None)
    environment.pop("GGML_SCHED_DEBUG", None)
    environment["GGML_VK_MEMORY_LOGGER"] = "1"
    if args.profile == "host-memory":
        environment["GGML_VK_PREFER_HOST_MEMORY"] = "1"
    elif args.profile == "mmvq":
        environment["GGML_VK_FORCE_MMVQ"] = "1"
    if args.split_output:
        environment["GGML_VK_SPLIT_OUTPUT"] = "1"
    if args.sync_transfers or args.split_output:
        environment["GGML_VK_DISABLE_ASYNC"] = "1"
        # The graph-completion wait preserves batching; per-operation waits are a diagnostic control.
        environment["GGML_VK_WAIT_IDLE" if args.completion_wait == "queue" else "GGML_VK_SERIALIZE_SUBMISSIONS"] = "1"
    if args.disable_fp16:
        environment["GGML_VK_DISABLE_F16"] = "1"
    if args.workgroup:
        environment["GGML_VK_PWR_MMV_WG"] = str(args.workgroup)
    if args.rms_subgroups:
        environment["GGML_VK_PWR_RMS_SUBGROUPS"] = "1"
    if args.u8_mmvq:
        environment["GGML_VK_PWR_U8_MMVQ"] = "1"
    if args.powervr_decode:
        for key in ("GGML_VK_PWR_ATTN", "GGML_VK_PWR_BARRIER", "GGML_VK_PREFER_DEVICE_LOCAL"):
            environment[key] = "1"
    paths = [str(args.runtime), *map(str, args.library_path)]
    if environment.get("LD_LIBRARY_PATH"):
        paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(paths)
    result["environment"] = {key: value for key, value in environment.items()
                             if key.startswith("GGML_VK_") or key == "LD_LIBRARY_PATH"}
    variants = None
    try:
        if (args.split_output or args.sync_transfers or args.workgroup or args.rms_subgroups or args.u8_mmvq
                or args.powervr_decode or (args.runtime / "spiralpixcel-kernel.json").exists()):
            marker = json.loads((args.runtime / "spiralpixcel-kernel.json").read_text())
            core = validate_core_libraries(args.runtime, marker, required=args.split_output or args.sync_transfers)
            result["core_library_manifest_verified"] = bool(core)
            result["kernel_build"] = marker
            if args.split_output or args.sync_transfers:
                runtime_manifest = validate_runtime_files(args.runtime, marker)
                result["runtime_manifest_verified"] = True
                if args.completion_wait == "queue" and marker.get("queue_idle_completion") is not True:
                    raise ValueError("runtime lacks queue completion waits; rebuild it or choose --completion-wait serial")
            if args.u8_mmvq and marker.get("powervr_u8_mmvq") is not True:
                raise ValueError("runtime lacks the PowerVR unsigned mat-vec patch; rebuild with --powervr-u8-mmvq")
            if args.powervr_decode and marker.get("powervr_decode") is not True:
                raise ValueError("runtime lacks the PowerVR decode patch; rebuild with --powervr-decode")
            if marker.get("libggml-vulkan_sha256") != sha256(args.runtime / "libggml-vulkan.so"):
                raise ValueError("patched runtime identity differs from its build receipt")
            if marker.get("check-mmv_sha256") != sha256(args.runtime / "check-mmv"):
                raise ValueError("kernel checker identity differs from its build receipt")
            if args.split_output or args.sync_transfers:
                from readback_check import parse_readback
                with (OUT / "readback-check.jsonl").open("wb") as out, (OUT / "readback-check.log").open("wb") as err:
                    subprocess.run([str(args.runtime / "check-readback"), str(args.runtime)], env=environment,
                                   stdout=out, stderr=err, check=True, timeout=30)
                result["readback_check"] = parse_readback((OUT / "readback-check.jsonl").read_text())
            with (OUT / "kernel-check.jsonl").open("wb") as out, (OUT / "kernel-check.log").open("wb") as err:
                subprocess.run([str(args.runtime / "check-mmv"), str(args.runtime)], env=environment,
                               stdout=out, stderr=err, check=True, timeout=180)
            variants = marker.get("powervr_u8_mmvq_variants") if args.u8_mmvq else None
            result["kernel_check"] = check_kernel_gate((OUT / "kernel-check.jsonl").read_text(), args.u8_mmvq,
                                                       args.powervr_decode, bool(variants))
            if args.u8_mmvq:
                selected = u8_selected_types(OUT / "kernel-check.log")
                if not set(marker.get("powervr_u8_mmvq_types", U8_GATE_TYPES)) <= set(selected):
                    raise ValueError("the unsigned mat-vec kernel did not run for every gated type; see kernel-check.log")
                result["u8_mmvq_selected"] = {"kernel_check": selected}
            if variants:
                selected = u8_selected_variants(OUT / "kernel-check.log")
                if not set(variants) <= set(selected):
                    raise ValueError("a shape variant of the unsigned mat-vec kernel did not run; see kernel-check.log")
                result["u8_mmvq_variants_selected"] = {"kernel_check": selected}
            if args.powervr_decode:
                if not attention_kernel_selected(OUT / "kernel-check.log"):
                    raise ValueError("the attention-score kernel did not run in the kernel gate; see kernel-check.log")
                result["attention_kernel_selected"] = {"kernel_check": True}
            if args.split_output:
                if not marker.get("output_row_split"):
                    raise ValueError("runtime does not advertise output row splitting")
                for name, key in (("libllama.so.0.8.0", "libllama_sha256"), ("check-large", "check-large_sha256"),
                                  ("check-model", "check-model_sha256")):
                    if marker.get(key) != sha256(args.runtime / name):
                        raise ValueError(f"{name} differs from the split runtime build receipt")
                with (OUT / "large-check.jsonl").open("wb") as out, (OUT / "large-check.log").open("wb") as err:
                    subprocess.run([str(args.runtime / "check-large"), str(args.runtime)], env=environment,
                                   stdout=out, stderr=err, check=True, timeout=180)
                large = [json.loads(line) for line in (OUT / "large-check.jsonl").read_text().splitlines()]
                if (len(large) != 3 or any(not isinstance(row, dict) for row in large)
                        or large[-1].get("status") != "pass" or large[-1].get("cases") != 2
                        or large[-1].get("failed") != 0 or {r.get("n") for r in large[:-1]} != {1, 16}
                        or any(r.get("status") != "pass" or r.get("weight_bytes", 0) <= 128*1024*1024
                               or r.get("reference") != "dequantized_q8_0_basis_values"
                               or not r.get("separate_weight_buffer") or r.get("readback_probes", 0) < 20
                               or r.get("mismatched_weight_bytes") != 0 or r.get("mismatched_input_bytes") != 0
                               or r.get("mismatched_elements") != 0 for r in large[:-1])):
                    raise ValueError("large-output CPU vs Vulkan numerical gate did not pass")
                result["large_kernel_check"] = large[-1]
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
        if args.split_output or args.sync_transfers:
            result["model_check"] = run_model_gate(
                args.runtime, args.model, args.gpu_layers, environment, OUT,
                model_sha256=result["model_sha256"], runtime_manifest=runtime_manifest)
            if args.u8_mmvq:
                result["u8_mmvq_selected"]["model_candidate"] = u8_selected_types(OUT / "model-candidate.log")
                if not result["u8_mmvq_selected"]["model_candidate"]:
                    raise ValueError("the unsigned mat-vec kernel did not run in the full-model gate")
            if variants:
                # Which variants a model reaches depends on its shapes; the kernel gate covers them all.
                result["u8_mmvq_variants_selected"]["model_candidate"] = u8_selected_variants(OUT / "model-candidate.log")
            if args.powervr_decode:
                # The checker's last ubatch is a single token, which takes the attention-score kernel.
                if not attention_kernel_selected(OUT / "model-candidate.log"):
                    raise ValueError("the attention-score kernel did not run in the full-model gate")
                result["attention_kernel_selected"]["model_candidate"] = True
        if args.schedule_report or args.split_output or args.sync_transfers:
            from schedule_report import collect
            placement = collect(args.runtime, args.model, args.gpu_layers, args.repack, args.flash_attn, environment, OUT)
            if args.split_output:
                chunks = [node for node in placement["output_nodes"] if node["name"].startswith("output_mm.chunk.")]
                if len(chunks) < 2 or any(not node["backend"].startswith("Vulkan") for node in chunks):
                    raise ValueError("no bounded GPU output-head chunks found in the schedule report")
            result["schedule"] = {"nodes_per_backend": placement["nodes_per_backend"],
                                  "cpu_nodes": placement["cpu_nodes"], "output_nodes": placement["output_nodes"],
                                  "rejected_weights": placement["rejected_weights"]}
        result.update(run_benchmark(command, environment, args.timeout))
        result.update(logged_memory((OUT / "bench.log").read_text(errors="replace")))
        if result["timed_out"] or result["exit_code"]:
            raise ValueError("benchmark did not complete; see bench.log")
        if args.u8_mmvq:
            # llama-bench drops library log messages unless -v, so this is usually empty; the
            # full-model gate above shows the selection for the same runtime, model and switches.
            result["u8_mmvq_selected"]["benchmark"] = u8_selected_types(OUT / "bench.log")
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
