"""Regression checks for lifting, holding still and exact tick boundaries."""
import json
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guest"))
from touch import decode, pack_ticks, schedule


class TouchSchedule(unittest.TestCase):
    def test_hold_gap_and_separate_tap(self):
        ticks = schedule([[[0, 4, 0], [150, 4, 0]], [[300, 127, 127]]])
        self.assertEqual(ticks, [{4}, {4}, {4}, {4}, set(), set(), {16383}])
        words = pack_ticks(ticks)
        self.assertEqual(struct.unpack_from("<I", words)[0], 1 << 4)
        self.assertEqual(struct.unpack_from("<I", words, len(words)-4)[0], 1 << 31)

    def test_reject_backwards_time_and_invalid_coordinates(self):
        for strokes in ([[[0, 0, 0], [1, 1, 0]], [[0, 2, 2]]], [[[0, -1, 0]]], [[[0, 0, True]]]):
            with self.subTest(strokes=strokes), self.assertRaises(ValueError):
                decode(json.dumps({"format": "SpiralPixcel.touch.v1", "strokes": strokes}).encode())


if __name__ == "__main__":
    unittest.main()
