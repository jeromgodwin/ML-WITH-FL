"""Phase 16: Communication efficiency — actual federated communication.

Measures real serialized bytes (no estimates):
- model parameter count / serialized size from src/federated/fl/server.py:357
- upload / download bytes from src/federated/fl/client.py:50 and strategy.py:95
- clients participating, rounds, bytes/round, total bytes

Compares FedAvg / FedProx / Personalized under:
- different client counts (via partition building if needed)
- different Non-IID severity (iid/mild/moderate/severe)
- different round counts (5/10/20/30)
- different client fractions (sampling)

Analyzes tradeoffs: communication vs F1, vs convergence, vs clients, vs training time.
Optional (client sampling, reduced frequency, quantization) only after baseline — not added here to avoid feature bloat.

Usage:
    python scripts/run_phase16_experiment.py --rounds 5 --strategies iid,mild --algorithms fedavg,fedprox
    python scripts/run_phase16_experiment.py --full-matrix
    python scripts/run_phase16_experiment.py --aggregate data/experiments_phase16
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fedshield.config import ExperimentConfig
from fedshield.logging_setup import get_logger
from src.federated.experiments.engine import run_unified_experiment, DEFAULT_VECTORIZED, DEFAULT_SCALER
from src.federated.communication.analysis import analyze_communication_efficiency, generate_tradeoff_plots, to_csv_rows
from src.federated.experiments.storage import DEFAULT_EXPERIMENTS_ROOT

logger = get_logger(__name__)

DEFAULT_OUTPUT_ROOT = Path("data/experiments_phase16")
PHASE16_SEED = 42


def ensure_partition(strategy: str, clients: int, seed: int = PHASE16_SEED) -> Path:
    """Ensure partition exists; build it if missing (actual non-IID simulation)."""
    part_dir = Path("data") / f"{strategy}-c{clients}-s{seed}"
    if part_dir.exists():
        return part_dir
    logger.info("partition %s not found — building (%d clients, seed %d)", part_dir, clients, seed)
    # Build using partition logic directly (no subprocess)
    from src.federated.data.partition import ClientPartitionConfig, build_client_partition, save_partition
    from src.federated.data.split import load_split_indices

    vectorized_dir = Path("data/ember_2018_2/vectorized")
    indices_path = Path("data/ember_2018_2/artifacts/split_indices.npz")
    if not (vectorized_dir / "X_train.npy").exists():
        raise FileNotFoundError(f"vectorized train missing: {vectorized_dir} (run prepare_ember)")
    n_train_rows = int(np.load(vectorized_dir / "X_train.npy", mmap_mode="r").shape[0])
    y_train = np.load(vectorized_dir / "y_train.npy", mmap_mode="r")
    indices = load_split_indices(indices_path)
    labeled_train = indices["train_idx"][(y_train[indices["train_idx"]] == 0) | (y_train[indices["train_idx"]] == 1)]

    # Severity map matches build_clients.py FIXED_SEVERITY
    severity_map = {"iid": "moderate", "mild": "mild", "moderate": "moderate", "severe": "severe",
                    "label_skew": 0.1, "quantity_skew": 0.3}
    severity = severity_map.get(strategy, "moderate")
    cfg = ClientPartitionConfig(strategy=strategy, severity=severity, clients=clients, seed=seed,
                                val_fraction=0.1, min_samples_per_client=5000, test_strategy="global_test")
    partition = build_client_partition(y_train, cfg, pool_idx=labeled_train)
    save_partition(part_dir, partition, y_train, None)
    logger.info("built partition %s", part_dir)
    return part_dir


def run_one_cell(
    algorithm: str,
    strategy: str,
    n_rounds: int,
    n_clients: int,
    client_fraction: float,
    proximal_mu: float,
    output_parent: Path,
    seed: int = PHASE16_SEED,
) -> dict:
    """Run one Phase-16 cell and return its communication summary via unified engine."""
    ensure_partition(strategy, n_clients, seed=seed)
    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    cfg.fl.algorithm = algorithm
    cfg.fl.num_clients = n_clients
    cfg.fl.num_rounds = n_rounds
    cfg.fl.client_fraction = client_fraction
    cfg.fl.proximal_mu = proximal_mu
    cfg.partition.strategy = strategy
    cfg.partition.clients = n_clients
    cfg.train.local_epochs = 5
    # Keep model fixed for fair communication comparison (same param count)
    cfg.seed = seed
    cfg.partition.seed = seed
    cfg.train.seed = seed

    # Run via unified engine (stores under data/experiments with unique ID)
    res = run_unified_experiment(
        cfg,
        raw_config=cfg.to_dict(),
        root=output_parent,
        vectorized=DEFAULT_VECTORIZED,
        scaler_path=DEFAULT_SCALER,
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16 communication efficiency")
    parser.add_argument("--strategies", default="iid,mild,moderate,severe", help="comma-separated partition strategies")
    parser.add_argument("--algorithms", default="fedavg,fedprox,personalized", help="comma-separated algorithms")
    parser.add_argument("--rounds-list", default="5,10,20", help="comma-separated round counts")
    parser.add_argument("--clients-list", default="10", help="comma-separated client counts")
    parser.add_argument("--fractions", default="1.0", help="comma-separated client fractions")
    parser.add_argument("--proximal-mu", type=float, default=0.1, help="FedProx mu")
    parser.add_argument("--seed", type=int, default=PHASE16_SEED)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--full-matrix", action="store_true", help="run full matrix (algorithms x strategies x rounds)")
    parser.add_argument("--aggregate", default=None, help="aggregate existing runs in OUTPUT_ROOT (no new training)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate is not None:
        agg_root = Path(args.aggregate)
        return do_aggregate(agg_root)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    rounds_list = [int(x) for x in args.rounds_list.split(",") if x.strip()]
    clients_list = [int(x) for x in args.clients_list.split(",") if x.strip()]
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]

    # Build matrix: algorithm x strategy x rounds x clients x fraction
    # For --full-matrix, expand to all combos; otherwise default is minimal (current lists)
    matrix = []
    for algo in algorithms:
        for strat in strategies:
            for rnd in rounds_list:
                for cli in clients_list:
                    for frac in fractions:
                        mu = args.proximal_mu if algo == "fedprox" else 0.0
                        matrix.append((algo, strat, rnd, cli, frac, mu))

    if args.dry_run:
        for algo, strat, rnd, cli, frac, mu in matrix:
            print(f"{algo:13s} {strat:10s} rounds={rnd:2d} clients={cli:2d} frac={frac:.1f} mu={mu}")
        print(f"dry-run: {len(matrix)} cells")
        return

    # Run each cell sequentially (each is a fresh process-like via unified engine's fresh server)
    results: list[dict] = []
    t_all0 = time.perf_counter()
    for idx, (algo, strat, rnd, cli, frac, mu) in enumerate(matrix, 1):
        logger.info("Phase16 cell %d/%d: %s %s rounds=%d clients=%d frac=%.2f", idx, len(matrix), algo, strat, rnd, cli, frac)
        try:
            res = run_one_cell(algo, strat, rnd, cli, frac, mu, output_parent=output_root, seed=args.seed)
            results.append(res)
        except Exception as e:
            logger.error("cell %s/%s rounds=%d clients=%d failed: %s", algo, strat, rnd, cli, e, exc_info=True)
            # Record failure as None (no invention)
            continue

    # Aggregate from the just-completed runs
    exp_dirs = [Path(r["storage_dir"]) for r in results if "storage_dir" in r]
    if not exp_dirs:
        logger.warning("no successful cells to aggregate")
        return

    summaries = []
    for d in exp_dirs:
        # summaries are in data/experiments_phase16/<id>/metrics/summary.json or summary.json
        candidates = [d / "metrics" / "summary.json", d / "summary.json"]
        for c in candidates:
            if c.exists():
                try:
                    summaries.append((d.name, json.loads(c.read_text(encoding="utf-8"))))
                    break
                except Exception:
                    continue

    analysis = analyze_communication_efficiency(summaries)

    # Write comparison artifacts under output_root/_phase16_<timestamp>
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    agg_dir = output_root / f"_aggregate_{ts}"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # CSV (no invented values)
    rows = to_csv_rows([__import__("src.federated.communication.analysis", fromlist=["from_summary"]).from_summary(s, eid) for eid, s in summaries])
    # Write via analysis helper or manually
    import csv as _csv
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        # Put core columns first
        core = ["experiment_id", "algorithm", "strategy", "n_clients", "n_rounds", "total_bytes", "bytes_per_round", "final_f1", "training_time_s"]
        ordered = [c for c in core if c in fieldnames] + [c for c in fieldnames if c not in core]
        with open(agg_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ordered})
    (agg_dir / "comparison.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, default=str), encoding="utf-8")

    # Markdown table
    md_lines = ["| experiment_id | algorithm | strategy | n_clients | n_rounds | total_bytes | bytes/round | F1 | training_time_s |",
                "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md_lines.append(f"| {r.get('experiment_id','')} | {r.get('algorithm','')} | {r.get('strategy','')} | {r.get('n_clients','')} | {r.get('n_rounds','')} | {r.get('total_bytes','')} | {r.get('bytes_per_round','')} | {r.get('final_f1','')} | {r.get('training_time_s','')} |")
    (agg_dir / "comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Tradeoff CSVs
    for key, data in analysis.get("tradeoffs", {}).items():
        if not data:
            continue
        p = agg_dir / f"tradeoff_{key}.csv"
        if data and isinstance(data[0], dict):
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=sorted(data[0].keys()))
                w.writeheader()
                w.writerows(data)

    # Plots (best-effort)
    try:
        generate_tradeoff_plots(analysis.get("tradeoffs", {}), agg_dir / "plots")
    except Exception as e:
        logger.warning("plot generation failed: %s", e)

    # Comparison JSON already written; also print summary
    print(json.dumps(analysis, indent=2, sort_keys=True, default=str))
    print(f"Phase 16 done: {len(results)} cells -> {agg_dir}")
    print(f"total wall time: {time.perf_counter() - t_all0:.1f}s")


def do_aggregate(agg_root: Path) -> None:
    """Aggregate existing runs under agg_root without new training."""
    agg_root = Path(agg_root)
    exp_dirs = [d for d in agg_root.iterdir() if d.is_dir() and (d / "config_resolved.json").exists()]
    summaries = []
    for d in exp_dirs:
        for c in [d / "metrics" / "summary.json", d / "summary.json"]:
            if c.exists():
                try:
                    summaries.append((d.name, json.loads(c.read_text(encoding="utf-8"))))
                    break
                except Exception:
                    continue
    if not summaries:
        print(f"no experiments found under {agg_root}")
        return
    from src.federated.communication.analysis import analyze_communication_efficiency

    analysis = analyze_communication_efficiency(summaries)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = agg_root / f"_aggregate_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(analysis, indent=2, sort_keys=True, default=str))
    print(f"aggregated {len(summaries)} experiments -> {out}")


if __name__ == "__main__":
    main()
