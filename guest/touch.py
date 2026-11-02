"""SpiralPixcel touch/v1. Time-ordered contact strokes on a 128-square grid."""
from array import array
import json

SIZE = 128
TICK_MS = 50
MAX_BYTES = 128 * 1024


def decode(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("trace exceeds 128 KiB")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("format") != "SpiralPixcel.touch.v1":
        raise ValueError("unsupported touch trace")
    strokes = data.get("strokes")
    if not isinstance(strokes, list) or not strokes:
        raise ValueError("empty touch trace")
    count, previous = 0, 0
    for stroke in strokes:
        if not isinstance(stroke, list) or not stroke:
            raise ValueError("empty stroke")
        for point in stroke:
            if not isinstance(point, list) or len(point) != 3 or any(type(n) is not int for n in point):
                raise ValueError("expected [integer milliseconds, x, y]")
            t, x, y = point
            if not previous <= t <= 10000 or not 0 <= x < SIZE or not 0 <= y < SIZE:
                raise ValueError("trace time or coordinates out of range")
            count += 1
            previous = t
    if count > 4096 or strokes[0][0][0] != 0:
        raise ValueError("expected at most 4096 samples starting at 0 ms")
    return strokes


def schedule(strokes):
    """Rasterize each segment spatially and temporally, never bridging two contacts."""
    ticks = [set() for _ in range(strokes[-1][-1][0] // TICK_MS + 1)]
    for stroke in strokes:
        t, x, y = stroke[0]
        ticks[t // TICK_MS].add(y * SIZE + x)
        for nt, nx, ny in stroke[1:]:
            dt, dx, dy = nt-t, nx-x, ny-y
            n = max(abs(dx), abs(dy), (dt + TICK_MS - 1) // TICK_MS, 1)
            for k in range(1, n+1):
                tick = (t*n + dt*k) // (n*TICK_MS)
                col = (2*(x*n + dx*k) + n) // (2*n)
                row = (2*(y*n + dy*k) + n) // (2*n)
                ticks[tick].add(row * SIZE + col)
            t, x, y = nt, nx, ny
    return ticks


def pack_ticks(ticks):
    stride = SIZE * SIZE // 32
    words = array("I", [0]) * (len(ticks) * stride)
    for tick, cells in enumerate(ticks):
        for cell in cells:
            words[tick*stride + cell//32] |= 1 << (cell % 32)
    return words.tobytes()
