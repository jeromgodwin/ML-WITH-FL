"""Unified experiment runner for Phase 15.

- YAML or JSON config
- Every run gets a unique experiment ID, never overwrites
- Stores: config, env, per-round, per-client, final, training time,
  resource, communication, drift, security, privacy, model metadata, logs, plots
- Supports full matrix: centralized, FedAvg/FedProx/Personalized x IID severities,
  plus controlled experiments for resource/drift/poisoning/privacy
- Reproducibility: seed, dataset version, partition seed, model, preprocessing,
  algorithm, software versions
- Aggregation: CSV, JSON, comparison tables, plots (no invented values)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedshield.config import ExperimentConfig
from fedshield.logging_setup import get_logger
from src.federated.experiments.engine import run_unified_experiment, DEFAULT_VECTORIZED, DEFAULT_SCALER
from src.federated.experiments.matrix import full_matrix, controlled_entries, matrix_from_config
from src.federated.experiments.aggregation import collect_summaries, write_csv, write_json, write_comparison_table, generate_plots
from src.federated.experiments.storage import DEFAULT_EXPERIMENTS_ROOT

logger = get_logger(__name__)


def _load_config(path: str | Path) -> tuple[ExperimentConfig, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    if p.suffix.lower() == ".json":
        cfg = ExperimentConfig.from_json(p)
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        cfg = ExperimentConfig.from_yaml(p)
        import yaml

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return cfg, raw


def main() -> None:
    parser = argparse.ArgumentParser(description="FedShield unified experiment runner (Phase 15)")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML or JSON config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="dotted override, e.g. fl.num_rounds=5 (repeatable)")
    parser.add_argument("--experiment-id", default=None, help="force experiment ID (must not exist)")
    parser.add_argument("--root", default=str(DEFAULT_EXPERIMENTS_ROOT), help="experiments root directory")
    parser.add_argument("--vectorized", default=str(DEFAULT_VECTORIZED))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    # Matrix modes
    parser.add_argument("--matrix", action="store_true", help="run full matrix (centralized + 12 FL cells)")
    parser.add_argument("--matrix-controlled", action="store_true", help="run controlled experiments (resource/drift/poisoning/privacy)")
    parser.add_argument("--matrix-config", default=None, help="YAML/JSON file describing a custom matrix (see matrix.py)")
    parser.add_argument("--aggregate", default=None, help="aggregate existing experiments in ROOT: write CSV/JSON/tables/plots to this output dir")
    parser.add_argument("--dry-run", action="store_true", help="print what would run without executing")
    args = parser.parse_args()

    # Aggregation mode (no training)
    if args.aggregate is not None:
        root = Path(args.root)
        out = Path(args.aggregate)
        exp_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "metrics" / "summary.json").exists() or (d / "summary.json").exists()]
        # More robust: collect all subdirs with config_resolved.json
        exp_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "config_resolved.json").exists()]
        rows = collect_summaries(exp_dirs)
        write_csv(rows, out / "comparison.csv")
        write_json(rows, out / "comparison.json")
        write_comparison_table(rows, out / "comparison.md")
        generate_plots(rows, out / "plots")
        print(f"aggregated {len(rows)} experiments -> {out}")
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
        return

    # Determine matrix or single run
    entries = None
    if args.matrix or args.matrix_controlled or args.matrix_config:
        if args.matrix_config:
            p = Path(args.matrix_config)
            import yaml

            if p.suffix.lower() == ".json":
                raw = json.loads(p.read_text(encoding="utf-8"))
            else:
                raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            entries = matrix_from_config(raw)
        else:
            entries = []
            if args.matrix:
                entries.extend(full_matrix())
            if args.matrix_controlled:
                entries.extend(controlled_entries())
            if not entries:
                entries = full_matrix()
        if args.dry_run:
            for e in entries:
                print(f"{e['label']}: algorithm={e['meta'].get('algorithm')} strategy={e['meta'].get('strategy')}")
            print(f"dry-run: {len(entries)} entries")
            return
        # Run each entry as a unified experiment
        results = []
        for e in entries:
            cfg: ExperimentConfig = e["config"]
            # Apply CLI overrides to each
            for ov in args.overrides:
                if "=" not in ov:
                    logger.warning("ignoring malformed --set %r (expected k=v)", ov)
                    continue
                k, v = ov.split("=", 1)
                # try to parse v as yaml literal
                try:
                    import yaml

                    v_parsed = yaml.safe_load(v)
                except Exception:
                    v_parsed = v
                from fedshield.config import set_dotted

                set_dotted(cfg, k, v_parsed)
            # Also apply per-entry overrides? already in cfg
            res = run_unified_experiment(
                cfg,
                raw_config=cfg.to_dict(),
                root=Path(args.root),
                experiment_id=None,
                vectorized=Path(args.vectorized),
                scaler_path=Path(args.scaler),
            )
            results.append(res)
        # Aggregate matrix results
        exp_dirs = [Path(r["storage_dir"]) for r in results]
        rows = collect_summaries(exp_dirs)
        # Write aggregate next to matrix runs (under root/_matrix_<timestamp>)
        from datetime import datetime, timezone

        agg_dir = Path(args.root) / f"_matrix_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        agg_dir.mkdir(parents=True, exist_ok=True)
        write_csv(rows, agg_dir / "comparison.csv")
        write_json(rows, agg_dir / "comparison.json")
        write_comparison_table(rows, agg_dir / "comparison.md")
        generate_plots(rows, agg_dir / "plots")
        (agg_dir / "matrix_runs.json").write_text(json.dumps([r["experiment_id"] for r in results], indent=2), encoding="utf-8")
        print(f"matrix done: {len(results)} experiments -> {agg_dir}")
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
        return

    # Single experiment mode
    cfg, raw = _load_config(args.config)
    for ov in args.overrides:
        if "=" not in ov:
            logger.warning("ignoring malformed --set %r (expected k=v)", ov)
            continue
        k, v = ov.split("=", 1)
        try:
            import yaml

            v_parsed = yaml.safe_load(v)
        except Exception:
            v_parsed = v
        from fedshield.config import set_dotted

        set_dotted(cfg, k, v_parsed)
    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        return
    res = run_unified_experiment(
        cfg,
        raw_config=raw,
        root=Path(args.root),
        experiment_id=args.experiment_id,
        vectorized=Path(args.vectorized),
        scaler_path=Path(args.scaler),
    )
    print(json.dumps(res["summary"], indent=2, sort_keys=True, default=str))
    print(f"experiment {res['experiment_id']} -> {res['storage_dir']}")


if __name__ == "__main__":
    main()
