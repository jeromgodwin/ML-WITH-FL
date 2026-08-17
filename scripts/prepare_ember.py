"""Prepare the EMBER 2018_2 dataset for FedShield.

Pipeline (idempotent; each stage skips if its outputs exist):
  1. vectorize raw JSONL features -> float32 .npy matrices + metadata
  2. seeded stratified train/val index split (official test untouched)
  3. fit StandardScaler on train only -> artifact + manifest
  4. quality report per split -> JSON

Usage:
    python scripts/prepare_ember.py [--data-dir data] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fedshield.logging_setup import get_logger  # noqa: E402
from src.federated.data.dataset import summarize_split  # noqa: E402
from src.federated.data.preprocess import fit_scaler_from_split  # noqa: E402
from src.federated.data.split import check_split_reproducible, make_train_val_indices  # noqa: E402
from src.federated.data.vectorize import vectorize_all  # noqa: E402
from src.federated.data.dataset import load_split  # noqa: E402

logger = get_logger(__name__)

TRAIN_JSONL = "train_features_{i}.jsonl"
TEST_JSONL = "test_features.jsonl"


def rebuild_train_meta(meta_path: Path, raw_dir: Path, n_x: int) -> None:
    """Write meta_train.jsonl from the pristine raw feature files.

    One metadata line per raw row (order == X_train row order); the 'label'
    field is coerced to int like vectorize_jsonl does.
    """
    with open(meta_path, "w", encoding="utf-8") as mf:
        written = 0
        for i in range(6):
            src = raw_dir / TRAIN_JSONL.format(i=i)
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    label = record.get("label")
                    row_meta = {k: record.get(k) for k in ("sha256", "md5", "appeared", "avclass", "label")}
                    row_meta["label"] = int(label) if label in (0, 1) else None
                    mf.write(json.dumps(row_meta) + "\n")
                    written += 1
    if written != n_x:
        raise RuntimeError(f"rebuilt meta has {written} rows, expected {n_x}")
    logger.info("rebuilt %s (%d rows)", meta_path, written)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EMBER 2018_2 for FedShield")
    parser.add_argument("--data-dir", default="data", help="root data directory (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="reproducibility seed (default: 42)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="val fraction of train (default: 0.2)")
    parser.add_argument("--skip-scaler", action="store_true", help="skip scaler fitting")
    parser.add_argument("--force-vectorize", action="store_true", help="re-vectorize from JSONL")
    args = parser.parse_args()

    root = Path(args.data_dir)
    raw_dir = root / "ember_2018_2" / "ember2018"
    vectorized_dir = root / "ember_2018_2" / "vectorized"
    artifacts_dir = root / "ember_2018_2" / "artifacts"
    indices_path = artifacts_dir / "split_indices.npz"

    if not raw_dir.is_dir():
        logger.error("raw dataset directory not found: %s (run scripts/fetch_ember.py first)", raw_dir)
        sys.exit(1)

    # 1. Vectorize -----------------------------------------------------------
    train_files = {}
    for i in range(6):
        p = raw_dir / TRAIN_JSONL.format(i=i)
        if p.exists():
            train_files[f"train_{i}"] = p
    test_file = raw_dir / TEST_JSONL
    if not train_files:
        logger.error("no train_features_*.jsonl found in %s", raw_dir)
        sys.exit(1)

    splits_to_vectorize = {"test": test_file} if test_file.exists() else {}
    for name, path in train_files.items():
        splits_to_vectorize[name] = path

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    combined_ok = all((vectorized_dir / f"{n}{s}").exists()
                      for n, s in (("X_", "train.npy"), ("y_", "train.npy"), ("meta_", "train.jsonl")))
    vectorized_ok = (vectorized_dir / "X_test.npy").exists() and combined_ok
    if args.force_vectorize or not vectorized_ok:
        logger.info("Stage 1: vectorizing %d files", len(splits_to_vectorize))
        stats = vectorize_all(splits_to_vectorize, vectorized_dir)
        (artifacts_dir / "vectorize_stats.json").write_text(
            json.dumps({k: v.to_dict() for k, v in stats.items()}, indent=2), encoding="utf-8")
    else:
        logger.info("Stage 1: vectorized outputs already present; skipping")

    # Combine per-file train chunks into a single train split.
    train_parts = [vectorized_dir / f"X_train_{i}.npy" for i in range(6)]
    combined_x = vectorized_dir / "X_train.npy"
    if all(p.exists() for p in train_parts):
        if not combined_x.exists():
            logger.info("Merging per-file train chunks into single train split ...")
            shapes = [np.load(p, mmap_mode="r").shape[0] for p in train_parts]
            total = sum(shapes)
            out = np.lib.format.open_memmap(str(combined_x), mode="w+", dtype=np.float32,
                                            shape=(total, 2381))
            pos = 0
            for part in train_parts:
                block = np.load(part, mmap_mode="r")
                out[pos:pos + block.shape[0]] = block
                pos += block.shape[0]
                del block  # release the memmap handle (Windows locks open files)
            out.flush()
            del out
            import gc  # noqa: PLC0415
            gc.collect()
            with open(vectorized_dir / "meta_train.jsonl", "w", encoding="utf-8") as mf:
                for part in [vectorized_dir / f"meta_train_{i}.jsonl" for i in range(6)]:
                    mf.write(part.read_text(encoding="utf-8"))
            y_concat = np.concatenate([np.load(vectorized_dir / f"y_train_{i}.npy") for i in range(6)])
            np.save(vectorized_dir / "y_train.npy", y_concat.astype(np.int8))
            del y_concat
            gc.collect()
        else:
            logger.info("Combined train split already exists; skipping merge")
        # Remove per-file chunks now that the combined split exists.
        for i in range(6):
            for suffix in (f"X_train_{i}.npy", f"y_train_{i}.npy", f"meta_train_{i}.jsonl"):
                (vectorized_dir / suffix).unlink(missing_ok=True)

    # Self-heal: meta_train.jsonl must have exactly one line per X_train row.
    # Earlier buggy runs appended duplicate records; rebuild from the pristine
    # raw feature files (row order == X_train row order by construction).
    n_x = int(np.load(combined_x, mmap_mode="r").shape[0])
    meta_path = vectorized_dir / "meta_train.jsonl"
    n_meta = sum(1 for _ in meta_path.open(encoding="utf-8")) if meta_path.exists() else 0
    if n_meta != n_x:
        logger.warning("meta_train.jsonl has %d rows, X_train has %d; rebuilding from raw JSONL",
                       n_meta, n_x)
        rebuild_train_meta(meta_path, raw_dir, n_x)
    else:
        logger.info("meta_train.jsonl aligned (rows=%d); OK", n_meta)

    # 2. Train/val split ------------------------------------------------------
    if not indices_path.exists() or not check_split_reproducible(indices_path, args.seed, args.val_fraction):
        logger.info("Stage 2: creating train/val split (seed=%d, val=%.0f%%)",
                    args.seed, args.val_fraction * 100)
        _, y_train = load_split(vectorized_dir, "train")
        make_train_val_indices(y_train, indices_path, seed=args.seed,
                               val_fraction=args.val_fraction)
    else:
        logger.info("Stage 2: split indices already present; skipping")

    # 3. Scaler ---------------------------------------------------------------
    if not args.skip_scaler:
        manifest = fit_scaler_from_split(vectorized_dir, indices_path, artifacts_dir, seed=args.seed)
        logger.info("Stage 3: scaler fit on %d labeled train rows", manifest.n_train_rows)
    else:
        logger.info("Stage 3: skipped (--skip-scaler)")

    # 4. Report ---------------------------------------------------------------
    stats_file = artifacts_dir / "vectorize_stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else {}
    report = {"dataset": "ember_2018_2", "feature_version": 2, "n_features": 2381,
              "seed": args.seed, "val_fraction": args.val_fraction}
    report["splits"] = {}
    for split in ["train", "test"]:
        report["splits"][split] = summarize_split(vectorized_dir, split,
                                                  stats.get(split, {})).to_dict()
    if indices_path.exists():
        indices = np.load(indices_path)
        report["val_rows"] = int(len(indices["val_idx"]))
    report_path = artifacts_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Stage 4: report written to %s", report_path)
    for split, s in report["splits"].items():
        logger.info("  %s: rows=%d benign=%d malicious=%d unlabeled=%d (nan=%d inf=%d)",
                    split, s["n_rows"], s["n_benign"], s["n_malicious"], s["n_unlabeled"],
                    s["n_nan"], s["n_inf"])


if __name__ == "__main__":
    main()