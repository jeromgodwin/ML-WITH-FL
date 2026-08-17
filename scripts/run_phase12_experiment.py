"""Phase 12 experiment: normal FL vs resource-aware FL.

Both runs use the IDENTICAL partition (iid, 10 clients), model, optimizer,
seeds and rounds — only the training gate differs:

    A) normal           no resource gating (baseline)
    B) resource-aware   an external CPU-burning workload (N subprocesses,
                        square wave: 10s busy / 10s idle) creates REAL CPU
                        contention; the policy (max_cpu_percent on the real
                        system CPU, check_interval) defers FL training
                        during busy windows, so FL yields the machine to
                        the competing workload.

Reported per run:
    wall training time, REAL process CPU%/RAM (sampled every 0.5s with
    psutil during training), communication bytes, client F1 and global F1
    per round (convergence), and — for B — pause count and deferred/wait
    time.

Battery: the battery state is merely REPORTED (availability + level) when
psutil exposes it. No battery-savings claim is made: with no controlled
battery measurements the experiment records the battery state and nothing
more.

Usage:
    python scripts/run_phase12_experiment.py [--rounds 5]
    python scripts/run_phase12_experiment.py --skip-normal   # re-run B only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from fedshield.config import ExperimentConfig, ResourceConfig  # noqa: E402
from fedshield.logging_setup import get_logger  # noqa: E402
from src.endpoint.resource import (  # noqa: E402
    ResourceMonitor, ResourcePolicy, TrainingController,
)
from src.federated.data.dataset import load_split  # noqa: E402
from src.federated.fl.server import run_fl_experiment  # noqa: E402

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_VECTORIZED = Path("data/ember_2018_2/vectorized")
DEFAULT_SCALER = Path("data/ember_2018_2/artifacts/scaler.joblib")
OUTPUT_ROOT = Path("data/fl/phase12")


class ProcessSampler:
    """Samples this process's real CPU% and RSS every ``interval`` seconds."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = False
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import psutil
        proc = psutil.Process(os.getpid())
        # prime cpu_percent (first call returns 0.0)
        proc.cpu_percent(interval=None)
        while not self._stop:
            cpu = proc.cpu_percent(interval=None)
            ram_mb = proc.memory_info().rss / 1048576.0
            self.samples.append({"cpu_percent": cpu, "ram_mb": ram_mb})
            time.sleep(self.interval)

    def stop(self) -> dict:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if not self.samples:
            return {"n_samples": 0}
        cpus = [s["cpu_percent"] for s in self.samples]
        rss = [s["ram_mb"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "mean_cpu_percent": round(float(np.mean(cpus)), 2),
            "max_cpu_percent": round(float(np.max(cpus)), 2),
            "mean_ram_mb": round(float(np.mean(rss)), 1),
            "max_ram_mb": round(float(np.max(rss)), 1),
        }


class CpuBurner:
    """External workload: N subprocesses spinning one CPU core each.

    A separate process keeps the FL process's own CPU/RAM measurement
    clean — the burner's load shows up as REAL system CPU utilization that
    the resource policy observes (and defers training for).
    """

    def __init__(self, n_procs: int = 3):
        self.n_procs = n_procs
        self._procs: list[subprocess.Popen] = []

    def start(self) -> None:
        for _ in range(self.n_procs):
            self._procs.append(subprocess.Popen(
                [sys.executable, "-c", "while True: pass"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    def stop(self) -> None:
        for p in self._procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
        for p in self._procs:
            try:
                p.wait(timeout=3)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
        self._procs = []


def run_burner_wave(burner: CpuBurner, busy_sec: float, idle_sec: float,
                    stop: threading.Event) -> None:
    """Toggle the burner on/off on a square wave (busy window first)."""
    t0 = time.time()
    running = False
    while not stop.is_set():
        on = (time.time() - t0) % (busy_sec + idle_sec) < busy_sec
        if on and not running:
            burner.start()
            running = True
        elif not on and running:
            burner.stop()
            running = False
        time.sleep(0.1)
    burner.stop()


def battery_note() -> dict:
    """Report battery state when available; NEVER a savings claim."""
    try:
        import psutil
        b = psutil.sensors_battery()
        if b is None:
            return {"battery_available": False}
        return {
            "battery_available": True,
            "percent": float(b.percent),
            "ac_powered": b.power_plugged,
        }
    except Exception as exc:  # noqa: BLE001
        return {"battery_available": False, "error": str(exc)}


def load_run_inputs(cfg: ExperimentConfig):
    scaler = joblib.load(DEFAULT_SCALER)
    scale_inv = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
    scale_inv = (1.0 / scale_inv).astype(np.float32)
    X_train, y_train = load_split(DEFAULT_VECTORIZED, "train")
    X_test, y_test = load_split(DEFAULT_VECTORIZED, "test")
    partition_dir = Path("data") / f"iid-c{cfg.fl.num_clients}-s{cfg.seed}"
    return partition_dir, X_train, y_train, X_test, y_test, scale_inv


def summarize_run(results: dict, sampler: dict, label: str) -> dict:
    rounds = results["rounds"]
    return {
        "label": label,
        "wall_training_time_s": results["training_time_s"],
        "total_bytes_exchanged": results["communication"]["totals"][
            "total_bytes_exchanged"],
        "final_global_test_f1": (results["final_global_test_metrics"] or {}).get("f1"),
        "client_f1_per_round": [r.get("avg_client_f1") for r in rounds],
        "global_f1_per_round": [
            (r.get("global_eval") or {}).get("f1") for r in rounds],
        "round_time_s_mean": results["round_time_s_mean"],
        "process": sampler,
        "resource": results.get("resource"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 12 normal vs resource-aware FL")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check-interval", type=float, default=1.0)
    parser.add_argument("--max-cpu", type=float, default=70.0,
                        help="policy: defer training above this real system CPU%%")
    parser.add_argument("--burners", type=int, default=3,
                        help="CPU-burning subprocesses for the contention windows")
    parser.add_argument("--skip-normal", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.fl.num_rounds = args.rounds
    cfg.fl.algorithm = "fedavg"
    partition_dir, X_train, y_train, X_test, y_test, scale_inv = load_run_inputs(cfg)

    battery = battery_note()
    logger.info("battery state: %s", battery)

    records = []
    if not args.skip_normal:
        sampler = ProcessSampler(interval=0.5)
        sampler.start()
        res = run_fl_experiment(
            cfg, partition_dir, X_train, y_train, X_test, y_test,
            scale_inv=scale_inv, output_dir=OUTPUT_ROOT / "normal", seed=cfg.seed)
        process_stats = sampler.stop()
        records.append(summarize_run(res, process_stats, "normal"))
        logger.info("=== normal done: %s ===", res["final_global_test_metrics"])

    # B) resource-aware: REAL CPU contention from an external burner workload.
    resource_cfg = ResourceConfig(
        enabled=True,
        max_cpu_percent=args.max_cpu,
        check_interval_sec=args.check_interval,
    )
    controller = TrainingController(
        ResourcePolicy(resource_cfg), ResourceMonitor(),
        check_interval_sec=args.check_interval)
    burner = CpuBurner(n_procs=args.burners)
    stop_wave = threading.Event()
    wave = threading.Thread(target=run_burner_wave,
                            args=(burner, 10.0, 10.0, stop_wave), daemon=True)
    wave.start()
    try:
        sampler = ProcessSampler(interval=0.5)
        sampler.start()
        res = run_fl_experiment(
            cfg, partition_dir, X_train, y_train, X_test, y_test,
            scale_inv=scale_inv, output_dir=OUTPUT_ROOT / "resource-aware",
            seed=cfg.seed, controller=controller)
        process_stats = sampler.stop()
    finally:
        stop_wave.set()
        wave.join(timeout=5.0)
        burner.stop()
    records.append(summarize_run(res, process_stats, "resource-aware"))
    logger.info("=== resource-aware done: %s ===",
                res["final_global_test_metrics"])

    comparison = {
        "experiment": {
            "phase": 12,
            "split": "iid",
            "algorithm": "fedavg",
            "rounds": cfg.fl.num_rounds,
            "clients": cfg.fl.num_clients,
            "seed": cfg.seed,
            "contention_profile": (
                f"external CPU burner ({args.burners} subprocesses), "
                "square wave: 10s busy / 10s idle"),
            "policy": {
                "max_cpu_percent": args.max_cpu,
                "check_interval_sec": args.check_interval,
            },
        },
        "battery": battery,
        "runs": records,
    }
    (OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

