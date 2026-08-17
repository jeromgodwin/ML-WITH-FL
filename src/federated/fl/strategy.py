"""Flower FedAvg / FedProx server strategies with per-round tracking.

Subclasses flwr.server.strategy.FedAvg — real Flower aggregation — and adds
honest measurement/tracking:

- round time (duration of the fit-aggregation step)
- per-round communication totals (summed from client-measured upload/download
  payload bytes)
- per-client eval metrics + aggregated avg/worst client F1 + F1 variance
- server-side global evaluation on the official test set (evaluate_fn)

FedProx (Phase 10): server aggregation is identical to FedAvg (weighted
averaging); the proximal regularizer is a CLIENT-side objective change. The
server only passes ``proximal_mu`` to clients via the fit config.
``FedProxTracked`` is thus a thin subclass of ``FedAvgTracked``.

The round history is exposed via ``rounds`` / ``summary()`` after the run.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import FitRes, Metrics, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from fedshield.logging_setup import get_logger

logger = get_logger(__name__)


def _weighted_avg(metric: str, results: List[Tuple[int, Metrics]]) -> Optional[float]:
    """Weighted (by n_samples) average of a metric across client results."""
    total, weighted = 0, 0.0
    for n_samples, metrics in results:
        if metric in metrics and metrics[metric] is not None:
            weighted += float(metrics[metric]) * n_samples
            total += n_samples
    return (weighted / total) if total else None


class FedAvgTracked(FedAvg):
    def __init__(
        self,
        num_clients: int,
        fraction_fit: float,
        num_rounds: int,
        local_epochs: int,
        learning_rate: float,
        batch_size: int,
        evaluate_every: int,
        initial_parameters: Parameters,
        evaluate_fn: Optional[Callable] = None,
        fraction_evaluate: float = 1.0,
        proximal_mu: float = 0.0,
    ):
        min_fit = max(1, int(round(fraction_fit * num_clients)))
        min_eval = max(1, int(round(fraction_evaluate * num_clients)))
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit,
            min_evaluate_clients=min_eval,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda r: {
                "server_round": r,
                "local_epochs": local_epochs,
                "lr": learning_rate,
                "batch_size": batch_size,
                "proximal_mu": proximal_mu,
            },
            on_evaluate_config_fn=lambda r: {"server_round": r},
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=self._aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=self._aggregate_eval_metrics,
            accept_failures=True,
        )
        self.num_rounds = num_rounds
        self.evaluate_every = evaluate_every
        self.proximal_mu = proximal_mu
        self.rounds: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # aggregation hooks (tracking only; math is FedAvg's)
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        t0 = time.perf_counter()
        aggregated, agg_metrics = super().aggregate_fit(server_round, results, failures)
        round_time_ms = (time.perf_counter() - t0) * 1000.0

        upload = sum(int(fitres.metrics.get("upload_bytes", 0)) for _, fitres in results)
        download = sum(int(fitres.metrics.get("download_bytes", 0)) for _, fitres in results)
        record = {
            "round": server_round,
            "round_time_ms": round(round_time_ms, 3),
            "n_clients_fit": len(results),
            "n_failures": len(failures),
            "download_bytes": download,
            "upload_bytes": upload,
            "bytes_this_round": upload + download,
            "fit_metrics": agg_metrics,
        }
        self.rounds.append(record)
        logger.info("round %d: %d clients fit in %.1f ms, %d bytes exchanged",
                    server_round, len(results), round_time_ms, upload + download)
        return aggregated, agg_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, aggregated = super().aggregate_evaluate(server_round, results, failures)
        if not results:
            return loss, aggregated
        for rec in self.rounds:
            if rec["round"] == server_round:
                client_f1 = [float(m.metrics.get("f1", 0.0)) for _, m in results if m.metrics.get("f1") is not None]
                if client_f1:
                    mean_f1 = sum(client_f1) / len(client_f1)
                    rec["avg_client_f1"] = mean_f1
                    rec["worst_client_f1"] = min(client_f1)
                    rec["client_f1_variance"] = round(
                        sum((f - mean_f1) ** 2 for f in client_f1) / len(client_f1), 8)
                rec["per_client_metrics"] = [
                    {"cid": str(cid.cid), **m.metrics} for cid, m in results]
                break
        return loss, aggregated

    def evaluate(self, server_round, parameters):
        """Server-side global evaluation on the official test set."""
        if server_round % self.evaluate_every != 0:
            return None
        t0 = time.perf_counter()
        result = super().evaluate(server_round, parameters)
        eval_ms = (time.perf_counter() - t0) * 1000.0
        if result is None:
            return None
        loss, metrics = result
        for rec in self.rounds:
            if rec["round"] == server_round:
                rec["global_eval"] = dict(metrics or {})
                rec["global_eval"]["eval_time_ms"] = round(eval_ms, 3)
                break
        return result

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        totals = {
            "rounds": len(self.rounds),
            "total_round_time_ms": round(sum(r["round_time_ms"] for r in self.rounds), 3),
            "total_download_bytes": sum(r["download_bytes"] for r in self.rounds),
            "total_upload_bytes": sum(r["upload_bytes"] for r in self.rounds),
            "total_bytes_exchanged": sum(r["bytes_this_round"] for r in self.rounds),
        }
        return {"totals": totals, "rounds": self.rounds}

    # ------------------------------------------------------------------
    def _aggregate_fit_metrics(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        out: Metrics = {}
        for name in ("train_loss", "train_accuracy"):
            v = _weighted_avg(name, metrics)
            if v is not None:
                out[name] = v
        out["total_fit_samples"] = int(sum(n for n, _ in metrics))
        out["fit_time_ms_total"] = round(
            sum(float(m.get("fit_time_ms", 0.0)) for _, m in metrics), 3)
        return out

    def _aggregate_eval_metrics(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        out: Metrics = {}
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            v = _weighted_avg(name, metrics)
            if v is not None:
                out[name] = v
        out["n_eval_samples"] = int(sum(n for n, _ in metrics))
        return out


class FedProxTracked(FedAvgTracked):
    """FedProx strategy: FedAvg aggregation + proximal_mu passed to clients.

    FedProx (Li et al. 2020) differs from FedAvg ONLY on the client side:
    clients add ``(mu/2) * ||w - w_global||^2`` to their local objective with a
    frozen ``w_global``. The server aggregation is identical to FedAvg, so this
    strategy only forwards ``proximal_mu`` in the fit config. ``mu=0`` is
    exactly FedAvg.
    """

    def __init__(self, proximal_mu: float = 0.0, **kwargs):
        super().__init__(proximal_mu=proximal_mu, **kwargs)


class PersonalizedTracked(FedAvgTracked):
    """FedRep-style personalized FL strategy (Phase 11).

    The server aggregates ONLY the shared body parameters with plain weighted
    FedAvg; personal heads never reach the server, so no server-side change to
    the aggregation is needed. The strategy forwards the client-side head
    adaptation hyperparameters (``head_epochs``, ``head_lr``) through the fit
    config; the per-round client evaluation reflects each client's FULL
    personalized model (global body + its own head).
    """

    def __init__(self, head_epochs: int = 2, head_learning_rate: float = 1e-2,
                 **kwargs):
        super().__init__(**kwargs)
        self.head_epochs = head_epochs
        self.head_learning_rate = head_learning_rate
        base_cfg_fn = self.on_fit_config_fn

        def on_fit_config_fn(server_round: int) -> Dict[str, Any]:
            cfg = dict(base_cfg_fn(server_round))
            cfg.update({"head_epochs": head_epochs,
                        "head_lr": head_learning_rate})
            return cfg

        self.on_fit_config_fn = on_fit_config_fn