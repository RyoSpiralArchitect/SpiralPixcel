"""Require every bounded transfer case, including repeated reads into Vulkan host memory."""
import json


ROUTES = {"sync_normal", "async_api_normal", "async_api_vulkan_host"}
SIZES = {64, 4096, 1048576}


def parse_readback(raw):
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 28 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("readback gate requires 27 completed cases and one summary")
    summary = rows[-1]
    if (summary.get("status") != "pass" or summary.get("cases") != 27 or summary.get("failed") != 0
            or summary.get("transfer_only") is not True or summary.get("max_payload_bytes") != 1048576):
        raise ValueError("readback gate failed")
    expected = {(route, size, trial) for route in ROUTES for size in SIZES for trial in range(3)}
    observed = set()
    for row in rows[:-1]:
        if any(type(row.get(key)) is not int for key in ("bytes", "trial", "offset", "mismatched_bytes")):
            raise ValueError("readback case has invalid numeric fields")
        key = row.get("route"), row["bytes"], row["trial"]
        if key not in expected or key in observed:
            raise ValueError("unexpected or repeated readback case")
        observed.add(key)
        if row.get("status") != "pass" or row["mismatched_bytes"] != 0 or row["offset"] != 64:
            raise ValueError("stale or zero readback detected")
    if observed != expected:
        raise ValueError("readback case coverage is incomplete")
    return summary
