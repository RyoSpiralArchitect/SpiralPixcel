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
        for name in ("libggml-vulkan.so", "libllama.so.0.8.0", "check-mmv", "check-large", "check-model", "llama-bench", "llama-completion"):
            (runtime / name).write_bytes(name.encode())
        marker = {"output_row_split": True, "core_library_sha256": core}
        for name, key in (("libggml-vulkan.so", "libggml-vulkan_sha256"), ("check-mmv", "check-mmv_sha256"),
                          ("libllama.so.0.8.0", "libllama_sha256"), ("check-large", "check-large_sha256"),
                          ("check-model", "check-model_sha256")):
            marker[key] = inference.sha256(runtime / name)
        (runtime / "libllama.so.0").symlink_to("libllama.so.0.8.0")
        (runtime / "libllama.so").symlink_to("libllama.so.0")
        marker["runtime_library_sha256"] = {p.name: inference.sha256(p) for p in runtime.glob("*.so*")}
        marker["runtime_executable_sha256"] = {name: inference.sha256(runtime / name) for name in
                                               ("llama-bench", "llama-completion", "check-mmv", "check-large", "check-model")}
        (runtime / "spiralpixcel-kernel.json").write_text(json.dumps(marker))
        model = root / "model.gguf"
        model.write_bytes(b"model fixture")
        return runtime, model

    def fake_checkers(self, mode):
        def run(command, *, env, stdout, stderr, timeout, check=False):
            name = Path(command[0]).name
            stderr.write(("mock " + name + "\n").encode())
            exit_code = 0
            if name == "check-mmv":
                records = [{"op": op, "status": "pass"} for op, count in
                           (("MUL_MAT", 40), ("RMS_NORM", 8), ("RMS_NORM_MUL", 8)) for _ in range(count)]
                records.append({"status": "pass", "cases": 56, "failed": 0})
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
                self.assertEqual(env["GGML_VK_SPLIT_OUTPUT"], "0" if layers == 0 else "1")
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


if __name__ == "__main__":
    unittest.main()
