"""Keep scheduler placement receipts tied to the final dump and a completed probe."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guest"))
from schedule_report import collect, parse_schedule


REJECTION = "done_getting_tensors: tensor 'output.weight' (q8_0) (and 0 others) cannot be used with preferred buffer type Vulkan0, using CPU_REPACK instead"
CPU_OUTPUT = """## SPLIT #0: CPU # 0 inputs
node #  0 (  GET_ROWS):                 embd (   3K) [  CPU         ] use=2,c=1:    token_embd.weight (  89M) [  CPU         ]           inp_tokens (   0K) [ CPU ]
## SPLIT #1: Vulkan0 # 1 inputs
node #943 (       MUL):          result_norm (   3K) [Vulka         ] use=1,c=1:                 norm (   3K) [Vulka ]   output_norm.weight (   3K) [Vulka ]
## SPLIT #2: CPU # 1 inputs
node #944 (   MUL_MAT):        result_output ( 593K) [  CPU         ] use=0,c=1:        output.weight ( 137M) [  CPU         ]    CPU#result_norm#0 (   3K) [ NULL ]
"""
GPU_OUTPUT = """## SPLIT #0: CPU # 0 inputs
node #  0 (  GET_ROWS):                 embd (   3K) [  CPU         ] use=2,c=1:    token_embd.weight (  89M) [  CPU         ]           inp_tokens (   0K) [ CPU ]
## SPLIT #1: Vulkan0 # 1 inputs
node #944 (   MUL_MAT):   result_output_chunk ( 300K) [Vulka         ] use=1,c=1:        output.weight (  69M) [Vulka         ]          result_norm (   3K) [Vulka ]
"""


def probe_row():
    return {"n_prompt": 0, "n_gen": 1, "n_gpu_layers": 99,
            "devices": "Vulkan0", "samples_ts": [8.0]}


class ScheduleReceipt(unittest.TestCase):
    def test_later_graph_replaces_prior_cpu_output(self):
        report = parse_schedule(CPU_OUTPUT + GPU_OUTPUT)
        self.assertEqual(report["logged_graphs"], 2)
        self.assertEqual(report["nodes_per_backend"], {"CPU": 1, "Vulkan0": 1})
        self.assertEqual([node["name"] for node in report["cpu_nodes"]], ["embd"])
        self.assertEqual(report["output_nodes"][0]["backend"], "Vulkan0")
        self.assertEqual(report["output_nodes"][0]["buffer_label"], "Vulka")

    def test_empty_final_dump_cannot_reuse_previous_placement(self):
        for previous in (CPU_OUTPUT, GPU_OUTPUT):
            with self.subTest(previous=previous), self.assertRaisesRegex(ValueError, "empty graph"):
                parse_schedule(previous + "## SPLIT #0: Vulkan0 # 0 inputs\n")

    def test_output_weight_rejection_is_retained(self):
        report = parse_schedule(REJECTION + "\n" + CPU_OUTPUT)
        self.assertEqual(report["cpu_nodes"][-1]["name"], "result_output")
        self.assertEqual(report["rejected_weights"], [{
            "tensor": "output.weight", "type": "q8_0", "preferred_buffer": "Vulkan0",
            "selected_buffer": "CPU_REPACK", "log_line": REJECTION,
        }])

    def test_failed_or_malformed_probe_cannot_complete(self):
        cases = [(1, [probe_row()]), (0, []), (0, [1]),
                 (0, [{**probe_row(), "devices": "none"}]),
                 (0, [{**probe_row(), "samples_ts": []}])]
        for exit_code, rows in cases:
            with self.subTest(exit_code=exit_code, rows=rows), tempfile.TemporaryDirectory() as folder:
                def run(command, *, env, stdout, stderr, timeout):
                    stdout.write(json.dumps(rows).encode())
                    stderr.write(CPU_OUTPUT.encode())
                    return subprocess.CompletedProcess(command, exit_code)

                output = Path(folder)
                with patch("schedule_report.subprocess.run", side_effect=run), self.assertRaises(ValueError):
                    collect(Path("/runtime"), Path("/model.gguf"), 99, 1, "off", {}, output)
                report = json.loads((output / "schedule.json").read_text())
                self.assertFalse(report["completed"])
                self.assertIn("error", report)
                self.assertEqual(set(report["sha256"]), {"scheduler.log", "scheduler-bench.json"})

    def test_diagnostic_flags_are_local_and_cleared_for_the_probe(self):
        environment = {"GGML_VK_PERF_LOGGER": "1", "GGML_VK_SPLIT_OUTPUT": "1"}
        def run(command, *, env, stdout, stderr, timeout):
            self.assertEqual(env["GGML_SCHED_DEBUG"], "2")
            self.assertNotIn("GGML_VK_PERF_LOGGER", env)
            self.assertEqual(env["GGML_VK_SPLIT_OUTPUT"], "1")
            stdout.write(json.dumps([probe_row()]).encode())
            stderr.write(GPU_OUTPUT.encode())
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as folder, patch("schedule_report.subprocess.run", side_effect=run):
            report = collect(Path("/runtime"), Path("/model.gguf"), 99, 1, "off", environment, Path(folder))
        self.assertTrue(report["completed"])
        self.assertEqual(environment, {"GGML_VK_PERF_LOGGER": "1", "GGML_VK_SPLIT_OUTPUT": "1"})


if __name__ == "__main__":
    unittest.main()
