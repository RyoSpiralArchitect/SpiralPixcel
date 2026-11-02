"""Keep incomplete and CPU-fallback benchmark output from becoming GPU success."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guest"))
from inference import parse_benchmark, logged_memory


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


if __name__ == "__main__":
    unittest.main()
