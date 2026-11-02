import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guest"))
from readback_check import parse_readback, ROUTES, SIZES


def rows():
    result = [{"route": route, "bytes": size, "trial": trial, "offset": 64,
               "mismatched_bytes": 0, "status": "pass"}
              for route in sorted(ROUTES) for size in sorted(SIZES) for trial in range(3)]
    result.append({"status": "pass", "cases": 27, "failed": 0, "transfer_only": True,
                   "max_payload_bytes": 1048576})
    return result


def encode(items): return "\n".join(map(json.dumps, items))


class ReadbackTest(unittest.TestCase):
    def test_complete_transfer_coverage(self):
        self.assertEqual(parse_readback(encode(rows()))["cases"], 27)

    def test_stale_or_zero_bytes_cannot_hide_behind_passing_summary(self):
        for route in ROUTES:
            data = rows()
            row = next(row for row in data[:-1] if row["route"] == route)
            row.update(mismatched_bytes=64, status="fail")
            with self.subTest(route=route), self.assertRaisesRegex(ValueError, "readback detected"):
                parse_readback(encode(data))

    def test_missing_or_duplicate_or_wrong_dimension_cases_fail(self):
        missing, repeated, wrong = rows(), rows(), rows()
        del missing[0]
        repeated[1] = repeated[0]
        wrong[0]["bytes"] = 134217728
        for data in [missing, repeated, wrong, [rows()[-1]], []]:
            with self.subTest(data=data[:1]), self.assertRaises(ValueError): parse_readback(encode(data))

    def test_failed_control_receipt_is_rejected(self):
        path = Path(__file__).resolve().parents[1] / "examples/inference/readback-study/async-control.jsonl"
        self.assertTrue(path.is_file(), "the failed Pixel control must remain in the evidence set")
        with self.assertRaises(ValueError): parse_readback(path.read_text())


if __name__ == "__main__": unittest.main()
