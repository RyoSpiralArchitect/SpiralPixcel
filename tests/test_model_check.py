"""Full-logit validation must reject broken readback even if a kernel run exits zero."""
from array import array
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guest"))
from model_check import PROMPT_SHA256, compare_vectors, read_vector


def receipt(values, layers):
    return {"status": "pass", "format": "float32-le", "prompt_sha256": PROMPT_SHA256,
            "prompt_tokens": 33, "logit_count": len(values), "gpu_layers": layers,
            "nonzero_count": sum(value != 0 for value in values), "nonfinite_count": 0,
            "finite": True, "top1_token": max(range(len(values)), key=values.__getitem__)}


class ModelCheck(unittest.TestCase):
    def test_matching_vectors_pass_and_changed_top1_fails(self):
        reference = [1., 2., 3.]
        candidate = [1.001, 1.999, 3.001]
        result = compare_vectors(reference, candidate, receipt(reference, 0), receipt(candidate, 99))
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["top1_equal"])
        candidate = [1., 3., 2.]
        result = compare_vectors(reference, candidate, receipt(reference, 0), receipt(candidate, 99))
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["top1_equal"])

    def test_zero_nan_and_count_mismatch_are_rejected(self):
        reference = [1., 2., 3.]
        reference_meta = receipt(reference, 0)
        candidate_meta = receipt(reference, 99)
        for candidate in ([0., 0., 0.], [1., math.nan, 3.], [1., 2.]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare_vectors(reference, candidate, reference_meta, candidate_meta)
        with self.assertRaises(ValueError):
            compare_vectors(reference, reference, reference_meta, {**candidate_meta, "logit_count": 4})

    def test_binary_vector_must_be_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "logits.bin"
            values = array("f", [1., -2., 3.])
            if sys.byteorder != "little":
                values.byteswap()
            path.write_bytes(values.tobytes())
            actual, fingerprint = read_vector(path)
            self.assertEqual(list(actual), [1., -2., 3.])
            self.assertEqual(len(fingerprint), 64)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                read_vector(path)


if __name__ == "__main__":
    unittest.main()
