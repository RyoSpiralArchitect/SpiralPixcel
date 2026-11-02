"""CPU reference, checkpoint comparison and a fixed-scale PNG for a 128x128 impulse."""
from array import array
import math
import struct
import zlib

SIZE = 128
ABS_TOL = 2e-7


def cpu_step(cells, rate, decay):
    # Clamping the neighbor coordinate implements the same no-flux boundary as the shader.
    out = array("f", [0]) * len(cells)
    for y in range(SIZE):
        row = y * SIZE
        north, south = max(y-1, 0) * SIZE, min(y+1, SIZE-1) * SIZE
        for x in range(SIZE):
            i = row + x
            neighbors = cells[row + max(0, x-1)] + cells[row + min(SIZE-1, x+1)] + cells[north+x] + cells[south+x]
            out[i] = decay * (cells[i] + rate * (neighbors - 4*cells[i]))
    return out


def write_png(dest, values):
    # Same scale for every run; log intensity reveals the low-amplitude outer field.
    palette = [(9, 12, 25), (29, 55, 101), (25, 163, 161), (245, 198, 107), (255, 249, 222)]
    scale = 3
    raw = bytearray()
    for y in range(SIZE):
        row = bytearray([0])
        for v in values[y*SIZE:(y+1)*SIZE]:
            t = max(0., min(1., math.log1p(65535*max(0., v)) / math.log(65536))) * (len(palette)-1)
            i = min(int(t), len(palette)-2)
            rgb = bytes(round(a + (b-a)*(t-i)) for a, b in zip(palette[i], palette[i+1]))
            row.extend(rgb*scale)
        raw.extend(row*scale)

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload))
    header = struct.pack(">2I5B", SIZE*scale, SIZE*scale, 8, 2, 0, 0, 0)
    dest.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def run(out, dispatch, args):
    cells = array("f", [0]) * (SIZE*SIZE)
    cells[(SIZE//2)*SIZE + SIZE//2] = 1.
    (out / "input.bin").write_bytes(cells.tobytes())
    checkpoints = sorted({1, max(1, args.steps//2), args.steps})
    # Push constants are binary32. Round the parameters once for the reference too.
    rate, decay = array("f", [args.rate, args.decay])
    results = []
    for step in range(1, args.steps+1):
        cells = cpu_step(cells, rate, decay)
        if step not in checkpoints:
            continue
        suffix = "" if step == args.steps else f"-{step}"
        path = out / f"gpu{suffix}.bin"
        vk = dispatch(SIZE, SIZE, step, rate, decay, args.allow_software, path)
        actual = array("f")
        actual.frombytes(path.read_bytes())
        if len(actual) != len(cells) or not all(map(math.isfinite, actual)):
            raise ValueError(f"invalid GPU field at step {step}")
        error = max(abs(g-c) for g, c in zip(actual, cells))
        (out / f"cpu{suffix}.bin").write_bytes(cells.tobytes())
        results.append({"step": step, "cells": len(cells), "max_abs_error": error, "pass": error <= ABS_TOL,
                        "gpu_ms": vk["gpu_ms"], "submit_wait_ms": vk["submit_wait_ms"],
                        "gpu_sum": math.fsum(actual), "gpu_peak": max(actual)})
    write_png(out / "field.png", actual)
    return {"vulkan": vk, "grid": [SIZE, SIZE], "initial": "unit impulse at (64,64)", "boundary": "clamp",
            "steps": args.steps, "rate": args.rate, "decay": args.decay, "absolute_tolerance": ABS_TOL,
            "checkpoints": results, "status": "pass" if all(r["pass"] for r in results) else "fail"}
