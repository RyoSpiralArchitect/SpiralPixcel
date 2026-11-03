"""Keep incomplete and CPU-fallback benchmark output from becoming GPU success."""
from array import array
import json
import hashlib
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import unittest
import tempfile
import subprocess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guest"))
import inference
from inference import parse_benchmark, logged_memory, validate_core_libraries, validate_runtime_files
from model_check import PROMPT_SHA256


def rows():
    return [{"n_prompt": prompt, "n_gen": gen, "avg_ts": 8.0, "samples_ts": [8.0, 8.1, 7.9],
             "n_gpu_layers": 99, "devices": "Vulkan0"} for prompt, gen in [(64, 0), (0, 32)]]


class InferenceReceipt(unittest.TestCase):
    def test_rejects_interrupted_or_fallback_runs(self):
        self.assertEqual(parse_benchmark(json.dumps(rows()), 3, 99)["decode_tg32_tokens_s"], 8.0)
        for field, value in [("samples_ts", [8.0]), ("devices", "none"), ("avg_ts", float("nan"))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                broken = rows()
                broken[1][field] = value
                parse_benchmark(json.dumps(broken), 3, 99)
        with self.assertRaises(ValueError):
            parse_benchmark(json.dumps(rows()[:1]), 3, 99)

    def test_decode_only_and_memory_units(self):
        self.assertEqual(parse_benchmark(json.dumps(rows()[1:]), 3, 99, True), {"decode_tg32_tokens_s": 8.0})
        raw = """ggml_vulkan memory: Vulkan0: +1.00 GiB device at 0xaa. Total device: 1.00 GiB, total host: 512.00 KiB
        ggml_vulkan memory: ggml_vk_host_malloc(1048608)\n        ggml_vulkan memory: Vulkan0: -1.00 GiB device at 0xaa. Total device: 0 B, total host: 0 B"""
        parsed = logged_memory(raw)
        self.assertEqual(parsed["peak_device_allocated_mib"], 1024.)
        self.assertEqual(parsed["peak_host_allocated_mib"], .5)
        self.assertIsNone(logged_memory("")["peak_device_allocated_mib"])


class KernelGate(unittest.TestCase):
    def records(self, matmuls, status="pass"):
        rows = [{"op": op, "status": status} for op, count in
                (("MUL_MAT", matmuls), ("RMS_NORM", 8), ("RMS_NORM_MUL", 8)) for _ in range(count)]
        rows.append({"status": "pass", "cases": matmuls + 16, "failed": 0})
        return "\n".join(map(json.dumps, rows))

    def test_accepts_both_layouts_and_requires_q8_0_on_request(self):
        self.assertEqual(inference.check_kernel_gate(self.records(40))["cases"], 56)
        self.assertEqual(inference.check_kernel_gate(self.records(52), True)["cases"], 68)
        for raw, require in ((self.records(40), True), (self.records(44), False), (self.records(52, "fail"), False)):
            with self.subTest(raw=raw[:40], require=require), self.assertRaises(ValueError):
                inference.check_kernel_gate(raw, require)


class MatchedCoreManifest(unittest.TestCase):
    def make_runtime(self, directory):
        runtime = Path(directory)
        for family in ("libggml", "libggml-base"):
            filename = family + ".so.0.28.0"
            (runtime / filename).write_bytes(("matched " + family).encode())
            (runtime / (family + ".so.0")).symlink_to(filename)
            (runtime / (family + ".so")).symlink_to(family + ".so.0")
        return runtime

    def marker(self, runtime):
        return {"core_library_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in runtime.iterdir()}}

    def test_split_requires_manifest_and_all_dependency_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.make_runtime(folder)
            self.assertEqual(validate_core_libraries(runtime, {}, required=False), {})
            with self.assertRaisesRegex(ValueError, "requires a matched"):
                validate_core_libraries(runtime, {}, required=True)
            marker = self.marker(runtime)
            verified = validate_core_libraries(runtime, marker, required=True)
            self.assertEqual(verified, marker["core_library_sha256"])
            del marker["core_library_sha256"]["libggml.so.0"]
            with self.assertRaisesRegex(ValueError, "complete core library family"):
                validate_core_libraries(runtime, marker, required=True)

    def test_missing_split_manifest_stops_before_running_checkers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "spiralpixcel-kernel.json").write_text(json.dumps({"output_row_split": True}))
            output = root / "result"
            command = ["inference.py", "--runtime", str(runtime), "--model", str(root / "model.gguf"), "--split-output"]
            with patch.object(inference, "OUT", output), patch.object(sys, "argv", command), \
                    patch("inference.subprocess.run") as run, redirect_stdout(io.StringIO()):
                self.assertEqual(inference.main(), 1)
            run.assert_not_called()
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["status"], "fail")
            self.assertIn("requires a matched core", receipt["error"])

    def test_mutated_core_bytes_and_retargeted_alias_fail(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as other:
            runtime = self.make_runtime(folder)
            marker = self.marker(runtime)
            target = runtime / "libggml.so.0.28.0"
            original = target.read_bytes()
            target.write_bytes(b"mutated core")
            with self.assertRaisesRegex(ValueError, "differs from"):
                validate_core_libraries(runtime, marker, required=True)
            target.write_bytes(original)
            external = Path(other) / target.name
            external.write_bytes(b"other build")
            alias = runtime / "libggml.so.0"
            alias.unlink()
            alias.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "differs from"):
                validate_core_libraries(runtime, marker, required=True)

    def test_manifest_paths_and_unknown_hashes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = self.make_runtime(folder)
            for filename in ("../libggml.so.0", "/tmp/libggml.so.0", "libggml.so/../secret"):
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "filename"):
                    marker = self.marker(runtime)
                    marker["core_library_sha256"][filename] = "0" * 64
                    validate_core_libraries(runtime, marker, required=True)
            marker = self.marker(runtime)
            marker["core_library_sha256"]["libggml.so"] = "not a digest"
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_core_libraries(runtime, marker, required=True)


class SplitModelAcceptance(unittest.TestCase):
    def make_fixture(self, root):
        runtime = root / "runtime"
        runtime.mkdir()
        for family in ("libggml", "libggml-base"):
            filename = family + ".so.0.28.0"
            (runtime / filename).write_bytes(family.encode())
            (runtime / (family + ".so.0")).symlink_to(filename)
            (runtime / (family + ".so")).symlink_to(family + ".so.0")
        core = {p.name: inference.sha256(p) for p in runtime.iterdir()}
        for name in ("libggml-vulkan.so", "libllama.so.0.8.0", "check-mmv", "check-large", "check-model", "check-readback", "llama-bench", "llama-completion"):
            (runtime / name).write_bytes(name.encode())
        marker = {"output_row_split": True, "core_library_sha256": core, "queue_idle_completion": True}
        for name, key in (("libggml-vulkan.so", "libggml-vulkan_sha256"), ("check-mmv", "check-mmv_sha256"),
                          ("libllama.so.0.8.0", "libllama_sha256"), ("check-large", "check-large_sha256"),
                          ("check-model", "check-model_sha256")):
            marker[key] = inference.sha256(runtime / name)
        (runtime / "libllama.so.0").symlink_to("libllama.so.0.8.0")
        (runtime / "libllama.so").symlink_to("libllama.so.0")
        marker["runtime_library_sha256"] = {p.name: inference.sha256(p) for p in runtime.glob("*.so*")}
        marker["runtime_executable_sha256"] = {name: inference.sha256(runtime / name) for name in
                                               ("llama-bench", "llama-completion", "check-mmv", "check-large", "check-model", "check-readback")}
        (runtime / "spiralpixcel-kernel.json").write_text(json.dumps(marker))
        model = root / "model.gguf"
        model.write_bytes(b"model fixture")
        return runtime, model

    def fake_checkers(self, mode):
        def run(command, *, env, stdout, stderr, timeout, check=False):
            name = Path(command[0]).name
            stderr.write(("mock " + name + "\n").encode())
            exit_code = 0
            if name == "check-readback":
                self.assertEqual(env["GGML_VK_DISABLE_ASYNC"], "1")
                self.assertTrue((env.get("GGML_VK_WAIT_IDLE") == "1" and "GGML_VK_SERIALIZE_SUBMISSIONS" not in env)
                                or (env.get("GGML_VK_SERIALIZE_SUBMISSIONS") == "1" and "GGML_VK_WAIT_IDLE" not in env))
                records = [{"route": route, "bytes": size, "trial": trial, "offset": 64,
                            "mismatched_bytes": 0, "status": "pass"}
                           for route in ("sync_normal", "async_api_normal", "async_api_vulkan_host")
                           for size in (64, 4096, 1048576) for trial in range(3)]
                records.append({"status": "pass", "cases": 27, "failed": 0,
                                "transfer_only": True, "max_payload_bytes": 1048576})
                if mode == "bad-readback":
                    records[0].update(mismatched_bytes=64, status="fail")
            elif name == "check-mmv":
                extended = mode in ("u8", "u8-legacy-gate", "u8-fallback")
                if mode in ("u8", "u8-legacy-gate"):
                    for type_name in ("q4_K", "q5_0", "q8_0"):
                        stderr.write(f"ggml_vulkan: mul_mat_vecq_u8 selected for {type_name}\n".encode())
                self.assertEqual(env.get("GGML_VK_PWR_U8_MMVQ"), "1" if extended else None)
                matmuls = 52 if mode in ("u8", "u8-fallback") else 40
                records = [{"op": op, "status": "pass"} for op, count in
                           (("MUL_MAT", matmuls), ("RMS_NORM", 8), ("RMS_NORM_MUL", 8)) for _ in range(count)]
                records.append({"status": "pass", "cases": matmuls + 16, "failed": 0})
            elif name == "check-large":
                case = {"status": "pass", "weight_bytes": 129*1024*1024,
                        "reference": "dequantized_q8_0_basis_values", "separate_weight_buffer": True,
                        "readback_probes": 20, "mismatched_weight_bytes": 0,
                        "mismatched_input_bytes": 0, "mismatched_elements": 0}
                records = [{**case, "n": 1}, {**case, "n": 16}]
                if mode == "extra-large-case":
                    records.append({**case, "n": 16})
                records.append({"status": "pass", "cases": 2, "failed": 0})
            elif name == "check-model":
                layers = int(command[4])
                self.assertEqual(env.get("GGML_VK_PWR_U8_MMVQ"), "1" if mode == "u8" else None)
                if mode == "u8" and layers:
                    stderr.write(b"ggml_vulkan: mul_mat_vecq_u8 selected for q5_0\n")
                self.assertEqual(env.get("GGML_VK_SPLIT_OUTPUT", "0"), "0" if layers == 0 or mode == "matching-baseline" else "1")
                values = [1., 2., 3.]
                if layers and mode == "zero":
                    values = [0., 0., 0.]
                if layers and mode == "changed-top1":
                    values = [1., 3., 2.]
                if layers and mode == "failed-process":
                    exit_code = 7
                binary = array("f", values)
                if sys.byteorder != "little":
                    binary.byteswap()
                Path(command[3]).write_bytes(binary.tobytes())
                records = [{"status": "pass", "format": "float32-le", "prompt_sha256": PROMPT_SHA256,
                            "prompt_tokens": 33, "logit_count": len(values), "gpu_layers": layers,
                            "nonzero_count": sum(value != 0 for value in values), "nonfinite_count": 0,
                            "finite": True, "top1_token": max(range(len(values)), key=values.__getitem__)}]
            else:
                raise AssertionError("unexpected checker " + name)
            stdout.write(("\n".join(map(json.dumps, records)) + "\n").encode())
            return subprocess.CompletedProcess(command, exit_code)
        return run

    def test_bad_full_model_never_reaches_timing(self):
        for mode in ("zero", "changed-top1", "failed-process"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                runtime, model = self.make_fixture(root)
                output = root / "result"
                command = ["inference.py", "--runtime", str(runtime), "--model", str(model), "--split-output"]
                with patch.object(inference, "OUT", output), patch.object(sys, "argv", command), \
                        patch("inference.subprocess.run", side_effect=self.fake_checkers(mode)), \
                        patch("inference.run_benchmark") as benchmark, redirect_stdout(io.StringIO()):
                    self.assertEqual(inference.main(), 1)
                benchmark.assert_not_called()
                receipt = json.loads((output / "result.json").read_text())
                self.assertEqual(receipt["status"], "fail")
                gate = json.loads((output / "model-check.json").read_text())
                self.assertEqual(gate["status"], "fail")
                self.assertEqual(gate["executions"]["reference"]["gpu_layers"], 0)
                self.assertEqual(gate["executions"]["candidate"]["gpu_layers"], 99)
                self.assertEqual(len(gate["sha256"]), 6)
                self.assertEqual(gate["model_sha256"], inference.sha256(model))
                self.assertEqual(gate["checker_sha256"], inference.sha256(runtime / "check-model"))
                self.assertEqual(gate["parameters"]["flash_attn"], "off")
                self.assertIn("libllama.so.0", gate["runtime_manifest"]["runtime_library_sha256"])
                self.assertIn("error", gate)
                if mode == "zero":
                    self.assertIn("all-zero", gate["error"])

    def test_failed_small_transfer_never_starts_large_or_model_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, model = self.make_fixture(Path(folder))
            output = Path(folder) / "result"
            command = ["inference.py", "--runtime", str(runtime), "--model", str(model), "--split-output"]
            with patch.object(inference, "OUT", output), patch.object(sys, "argv", command), \
                    patch("inference.subprocess.run", side_effect=self.fake_checkers("bad-readback")) as checks, \
                    patch("inference.run_benchmark") as benchmark, redirect_stdout(io.StringIO()):
                self.assertEqual(inference.main(), 1)
            benchmark.assert_not_called()
            self.assertEqual([Path(call.args[0][0]).name for call in checks.call_args_list], ["check-readback"])
            self.assertFalse((output / "large-check.jsonl").exists())

    def test_matching_full_model_can_reach_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime, model = self.make_fixture(root)
            output = root / "result"
            command = ["inference.py", "--runtime", str(runtime), "--model", str(model), "--split-output"]
            placement = {"nodes_per_backend": {"Vulkan0": 2}, "cpu_nodes": [], "rejected_weights": [],
                         "output_nodes": [{"name": f"output_mm.chunk.{i}", "backend": "Vulkan0"} for i in range(2)]}
            def timed_run(command, environment, timeout):
                self.assertEqual(environment["GGML_VK_SPLIT_OUTPUT"], "1")
                (output / "bench.json").write_text(json.dumps(rows()))
                (output / "bench.log").write_text("")
                return {"exit_code": 0, "timed_out": False, "sampled_peak_rss_kib": 1}

            with patch.object(inference, "OUT", output), patch.object(sys, "argv", command), \
                    patch("inference.subprocess.run", side_effect=self.fake_checkers("matching")), \
                    patch("schedule_report.collect", return_value=placement), \
                    patch("inference.run_benchmark", side_effect=timed_run) as benchmark, redirect_stdout(io.StringIO()):
                self.assertEqual(inference.main(), 0)
            benchmark.assert_called_once()
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["model_check"]["status"], "pass")
            self.assertTrue(receipt["model_check"]["top1_equal"])
            self.assertEqual(receipt["model_check"]["nmse"], 0)

    def test_unsigned_matvec_reaches_timing_on_the_extended_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime, model = self.make_fixture(root)
            marker_path = runtime / "spiralpixcel-kernel.json"
            marker_path.write_text(json.dumps({**json.loads(marker_path.read_text()), "powervr_u8_mmvq": True}))
            output = root / "result"
            command = ["inference.py", "--runtime", str(runtime), "--model", str(model), "--split-output", "--u8-mmvq"]
            placement = {"nodes_per_backend": {"Vulkan0": 2}, "cpu_nodes": [], "rejected_weights": [],
                         "output_nodes": [{"name": f"output_mm.chunk.{i}", "backend": "Vulkan0"} for i in range(2)]}
            def timed_run(command, environment, timeout):
                self.assertEqual(environment["GGML_VK_PWR_U8_MMVQ"], "1")
                (output / "bench.json").write_text(json.dumps(rows()))
                (output / "bench.log").write_text("")
                return {"exit_code": 0, "timed_out": False, "sampled_peak_rss_kib": 1}

            with patch.object(inference, "OUT", output), patch.object(sys, "argv", command), \
                    patch("inference.subprocess.run", side_effect=self.fake_checkers("u8")), \
                    patch("schedule_report.collect", return_value=placement), \
                    patch("inference.run_benchmark", side_effect=timed_run) as benchmark, redirect_stdout(io.StringIO()):
                self.assertEqual(inference.main(), 0)
            benchmark.assert_called_once()
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["kernel_check"]["cases"], 68)
            self.assertEqual(receipt["u8_mmvq_selected"], {"kernel_check": ["q4_K", "q5_0", "q8_0"],
                                                           "model_candidate": ["q5_0"], "benchmark": []})
            self.assertEqual(receipt["model_check"]["environment"]["GGML_VK_PWR_U8_MMVQ"], "1")

    def test_unsigned_matvec_requires_its_patch_and_q8_0_cases(self):
        for flagged, mode, message in ((False, "u8", "unsigned mat-vec patch"),
                                       (True, "u8-legacy-gate", "incomplete receipt"),
                                       (True, "u8-fallback", "did not run for every gated type")):
            with self.subTest(flagged=flagged), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                runtime, model = self.make_fixture(root)
                if flagged:
                    marker_path = runtime / "spiralpixcel-kernel.json"
                    marker_path.write_text(json.dumps({**json.loads(marker_path.read_text()), "powervr_u8_mmvq": True}))
                command = ["inference.py", "--runtime", str(runtime), "--model", str(model), "--split-output", "--u8-mmvq"]
                with patch.object(inference, "OUT", root / "result"), patch.object(sys, "argv", command), \
                        patch("inference.subprocess.run", side_effect=self.fake_checkers(mode)), \
                        patch("inference.run_benchmark") as benchmark, redirect_stdout(io.StringIO()):
                    self.assertEqual(inference.main(), 1)
                benchmark.assert_not_called()
                self.assertIn(message, json.loads((root / "result/result.json").read_text())["error"])

    def test_large_gate_requires_exactly_two_case_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime, model = self.make_fixture(root)
            output = root / "result"
            command = ["inference.py", "--runtime", str(runtime), "--model", str(model), "--split-output"]
            with patch.object(inference, "OUT", output), patch.object(sys, "argv", command), \
                    patch("inference.subprocess.run", side_effect=self.fake_checkers("extra-large-case")), \
                    patch("inference.run_benchmark") as benchmark, redirect_stdout(io.StringIO()):
                self.assertEqual(inference.main(), 1)
            benchmark.assert_not_called()
            self.assertIn("large-output", json.loads((output / "result.json").read_text())["error"])
            self.assertFalse((output / "model-reference.bin").exists())

    def test_sync_control_does_not_silently_enable_row_splitting(self):
        for completion in ("queue", "serial"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                runtime, model = self.make_fixture(root)
                output = root / "result"
                command = ["inference.py", "--runtime", str(runtime), "--model", str(model), "--sync-transfers", "--completion-wait", completion]
                placement = {"nodes_per_backend": {"CPU": 1}, "cpu_nodes": [], "rejected_weights": [], "output_nodes": []}
                def timing(command, env, timeout):
                    self.assertNotIn("GGML_VK_SPLIT_OUTPUT", env)
                    self.assertEqual(env.get("GGML_VK_WAIT_IDLE" if completion == "queue" else "GGML_VK_SERIALIZE_SUBMISSIONS"), "1")
                    (output / "bench.json").write_text(json.dumps(rows()))
                    (output / "bench.log").write_text("")
                    return {"exit_code": 0, "timed_out": False}
                with patch.object(inference, "OUT", output), patch.object(sys, "argv", command), \
                        patch("inference.subprocess.run", side_effect=self.fake_checkers("matching-baseline")), \
                        patch("schedule_report.collect", return_value=placement), \
                        patch("inference.run_benchmark", side_effect=timing), redirect_stdout(io.StringIO()):
                    self.assertEqual(inference.main(), 0)
                gate = json.loads((output / "model-check.json").read_text())
                self.assertEqual(gate["executions"]["candidate"]["environment_overrides"]["GGML_VK_SPLIT_OUTPUT"], "0")
                self.assertFalse((output / "large-check.jsonl").exists())

    def test_full_runtime_manifest_rejects_changed_llama_alias_or_missing_executable(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, _ = self.make_fixture(Path(folder))
            marker = json.loads((runtime / "spiralpixcel-kernel.json").read_text())
            validate_runtime_files(runtime, marker)
            old = runtime / "libllama.so.0.7.0"
            old.write_bytes(b"old llama build")
            # Updating the inventory cannot conceal retargeting the loaded SONAME.
            marker["runtime_library_sha256"][old.name] = inference.sha256(old)
            (runtime / "libllama.so.0").unlink()
            (runtime / "libllama.so.0").symlink_to(old.name)
            with self.assertRaisesRegex(ValueError, "differs from"):
                validate_runtime_files(runtime, marker)
            (runtime / "libllama.so.0").unlink()
            (runtime / "libllama.so.0").symlink_to("libllama.so.0.8.0")
            del marker["runtime_executable_sha256"]["check-model"]
            with self.assertRaisesRegex(ValueError, "complete split runtime"):
                validate_runtime_files(runtime, marker)

    def test_split_output_rejects_unvalidated_flash_attention(self):
        with patch.object(sys, "argv", ["inference.py", "--split-output", "--flash-attn", "on"]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            inference.main()
        self.assertEqual(error.exception.code, 2)

    def test_split_output_rejects_cpu_only_request(self):
        with patch.object(sys, "argv", ["inference.py", "--split-output", "--gpu-layers", "0"]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            inference.main()
        self.assertEqual(error.exception.code, 2)

    def test_unsigned_matvec_requires_the_logit_gate(self):
        with patch.object(sys, "argv", ["inference.py", "--u8-mmvq"]), redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            inference.main()
        self.assertEqual(error.exception.code, 2)

    def test_checked_profiles_reject_repack_parameters_not_used_by_model_checker(self):
        for option in ("--sync-transfers", "--split-output"):
            with self.subTest(option=option), patch.object(sys, "argv", ["inference.py", option, "--repack", "0"]), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                inference.main()
            self.assertEqual(error.exception.code, 2)

    def test_queue_wait_requires_runtime_support_before_gpu_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime, model = self.make_fixture(root)
            marker_path = runtime / "spiralpixcel-kernel.json"
            marker = json.loads(marker_path.read_text())
            del marker["queue_idle_completion"]
            marker_path.write_text(json.dumps(marker))
            with patch.object(inference, "OUT", root / "result"), \
                    patch.object(sys, "argv", ["inference.py", "--runtime", str(runtime), "--model", str(model), "--split-output"]), \
                    patch("inference.subprocess.run") as run, redirect_stdout(io.StringIO()):
                self.assertEqual(inference.main(), 1)
            run.assert_not_called()
            self.assertIn("queue completion", json.loads((root / "result/result.json").read_text())["error"])


if __name__ == "__main__":
    unittest.main()
