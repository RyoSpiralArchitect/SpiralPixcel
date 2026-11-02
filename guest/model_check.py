"""Compare complete fixed-prompt CPU and GPU logit vectors before benchmarking."""
from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

PROMPT_SHA256 = "3ae35bc99e935fea913bbae40ac007931405b8242c50f63bd06b32aac923a762"
MAX_LOGIT_BYTES = 64 * 1024 * 1024


def read_vector(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_LOGIT_BYTES + 1)
    if not raw or len(raw) % 4 or len(raw) > MAX_LOGIT_BYTES:
        raise ValueError("expected a nonempty float32 vector of at most 64 MiB")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values, hashlib.sha256(raw).hexdigest()


def validate_vector(values, metadata):
    if not isinstance(metadata, dict) or metadata.get("status") != "pass":
        raise ValueError("model execution did not pass its technical checks")
    if metadata.get("format") != "float32-le" or metadata.get("prompt_sha256") != PROMPT_SHA256:
        raise ValueError("model receipt uses a different prompt or vector format")
    if metadata.get("logit_count") != len(values):
        raise ValueError("logit vector length differs from its receipt")
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("logit vector is empty or nonfinite")
    nonzero = sum(value != 0 for value in values)
    if not nonzero:
        raise ValueError("all-zero logits are invalid")
    if (metadata.get("nonzero_count") != nonzero or metadata.get("nonfinite_count") != 0
            or metadata.get("finite") is not True):
        raise ValueError("logit values disagree with the model receipt")
    top1 = max(range(len(values)), key=values.__getitem__)
    if metadata.get("top1_token") != top1:
        raise ValueError("top1 token differs from the logit vector")
    if type(metadata.get("prompt_tokens")) is not int or metadata["prompt_tokens"] <= 0:
        raise ValueError("invalid prompt token count")
    return top1


def compare_vectors(reference, candidate, reference_metadata, candidate_metadata, nmse_limit=5e-3):
    if not math.isfinite(nmse_limit) or nmse_limit <= 0:
        raise ValueError("NMSE limit must be finite and positive")
    reference_top1 = validate_vector(reference, reference_metadata)
    candidate_top1 = validate_vector(candidate, candidate_metadata)
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate logit counts differ")
    if reference_metadata["prompt_tokens"] != candidate_metadata["prompt_tokens"]:
        raise ValueError("reference and candidate prompt token counts differ")
    if reference_metadata.get("gpu_layers") != 0:
        raise ValueError("the reference must use the CPU route")
    layers = candidate_metadata.get("gpu_layers")
    if type(layers) is not int or not 1 <= layers <= 999:
        raise ValueError("the candidate must request GPU layers")
    energy = math.fsum(float(value) * value for value in reference)
    squared_error = math.fsum((float(actual)-expected)**2 for expected, actual in zip(reference, candidate))
    if energy <= 0:
        raise ValueError("reference logit energy must be positive")
    nmse = squared_error / energy
    top1_equal = reference_top1 == candidate_top1
    return {
        "status": "pass" if nmse <= nmse_limit and top1_equal else "fail",
        "technical_validation_only": True,
        "prompt_sha256": PROMPT_SHA256,
        "prompt_tokens": reference_metadata["prompt_tokens"],
        "logit_count": len(reference),
        "nmse": nmse,
        "nmse_limit": nmse_limit,
        "max_abs_error": max(abs(float(actual)-expected) for expected, actual in zip(reference, candidate)),
        "top1_equal": top1_equal,
        "reference_top1_token": reference_top1,
        "candidate_top1_token": candidate_top1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--nmse-limit", type=float, default=5e-3)
    parser.add_argument("--output", type=Path, default=Path("result/model-check.json"))
    args = parser.parse_args()
    result = {"status": "fail", "technical_validation_only": True}
    try:
        reference, reference_sha = read_vector(args.reference)
        candidate, candidate_sha = read_vector(args.candidate)
        result = compare_vectors(reference, candidate, json.loads(args.reference_json.read_text()),
                                 json.loads(args.candidate_json.read_text()), args.nmse_limit)
        result["reference_logits_sha256"] = reference_sha
        result["candidate_logits_sha256"] = candidate_sha
    except (OSError, ValueError) as error:
        result["error"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
