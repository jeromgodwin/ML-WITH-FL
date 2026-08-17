"""Endpoint pipeline validation: real PE -> features -> preprocessing -> model.

Proves the trained bundle accepts real-PE inputs end-to-end:
    PE bytes -> extract_feature_vector (lief 1.0, EMBER v2 formulas)
             -> bundle.preprocess (scaler) -> model -> probabilities

A mismatch (wrong shape, non-finite, or parse failure on a valid PE) fails
the script loudly. Run:

    python scripts/validate_endpoint.py [--bundle data/ember_2018_2/models/<v>/bundle]
                                        [--files path1 path2 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fedshield.logging_setup import get_logger  # noqa: E402
from src.endpoint.ember_features import extract_feature_vector, extract_raw_features  # noqa: E402
from src.federated.model_bundle import BUNDLE_MANIFEST, InferenceBundle  # noqa: E402

logger = get_logger(__name__)

DEFAULT_SYSTEM_PES = [
    r"C:\Windows\System32\kernel32.dll",
    r"C:\Windows\System32\user32.dll",
    r"C:\Windows\System32\ntdll.dll",
    r"C:\Windows\System32\cmd.exe",
    r"C:\Windows\System32\notepad.exe",
]


def validate_bundle(bundle_dir: Path, files: list[Path]) -> dict:
    bundle = InferenceBundle.load(bundle_dir)
    expected_dim = bundle.config.input_dim

    rows = []
    failures = 0
    for path in files:
        path = Path(path)
        if not path.exists():
            rows.append({"file": str(path), "status": "missing"})
            failures += 1
            continue
        bytez = path.read_bytes()
        raw = extract_raw_features(bytez)
        if raw is None:
            rows.append({"file": path.name, "status": "parse_failed",
                         "size": len(bytez)})
            failures += 1
            continue
        vec = extract_feature_vector(bytez)
        if vec is None or vec.shape != (expected_dim,):
            rows.append({"file": path.name, "status": "shape_mismatch",
                         "shape": None if vec is None else list(vec.shape)})
            failures += 1
            continue
        if not np.isfinite(vec).all():
            rows.append({"file": path.name, "status": "non_finite"})
            failures += 1
            continue
        prob = float(bundle.predict_proba(vec.reshape(1, -1))[0])
        verdict = "malicious" if prob >= 0.5 else "benign"
        rows.append({
            "file": path.name,
            "status": "ok",
            "size": len(bytez),
            "sha256": raw.get("sha256"),
            "machine": raw["header"]["coff"]["machine"],
            "p_malicious": round(prob, 4),
            "verdict": verdict,
        })

    summary = {
        "bundle": str(bundle_dir),
        "version": bundle.manifest.get("version"),
        "n_features": expected_dim,
        "files_checked": len(files),
        "files_ok": sum(1 for r in rows if r["status"] == "ok"),
        "files_failed": failures,
        "results": rows,
    }
    ok = failures == 0 and len(rows) > 0
    summary["passed"] = ok
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate endpoint pipeline")
    parser.add_argument("--bundle", default=None,
                        help="bundle dir; default: data/ember_2018_2/models/mlp-central-v1/bundle")
    parser.add_argument("--files", nargs="*", default=None,
                        help="PE files to validate (default: system binaries)")
    parser.add_argument("--report", default=None, help="write summary JSON here")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle) if args.bundle else \
        Path("data") / "ember_2018_2" / "models" / "mlp-central-v1" / "bundle"
    if not (bundle_dir / BUNDLE_MANIFEST).exists():
        print(f"bundle not found: {bundle_dir} (run scripts/train_centralized.py first)")
        sys.exit(1)

    files = [Path(f) for f in args.files] if args.files else \
        [Path(p) for p in DEFAULT_SYSTEM_PES]

    summary = validate_bundle(bundle_dir, files)
    print(json.dumps(summary, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not summary["passed"]:
        print("ENDPOINT VALIDATION FAILED")
        sys.exit(1)
    print(f"ENDPOINT VALIDATION PASSED ({summary['files_ok']}/{summary['files_checked']} files)")


if __name__ == "__main__":
    main()