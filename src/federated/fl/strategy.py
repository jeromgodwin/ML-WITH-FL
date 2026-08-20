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


class DefendedTracked(FedAvgTracked):
    """FedAvg with configurable server-side poisoning defenses (Phase 14).

    Defends the round aggregation against simulated malicious updates:

    - clipping: each client's parameter UPDATE (returned - received) is
      clipped to a maximum L2 norm before aggregation.
    - anomaly detection: per-client updates are scored (magnitude, deviation
      from peer updates, distance from the reference update); HIGHLY_ANOMALOUS
      clients are excluded from the aggregation.
    - model validation: the candidate aggregated model is validated on a
      controlled held-out set; if it degrades unexpectedly vs the currently
      trusted model, it is REJECTED and the previous trusted model is
      retained.
    - robust aggregation: coordinate-wise median or trimmed mean over client
      updates (experimental).

    Every mode still records the per-client update norms and anomaly
    classifications, so a comparison run reports detection/FP rates for the
    baseline too. The strategy NEVER claims complete poisoning detection.
    """

    def __init__(
        self,
        defense_mode: str = "none",
        clip_norm: Optional[float] = None,
        anomaly_suspect_mult: float = 3.0,
        anomaly_detect_mult: float = 6.0,
        exclude_highly_anomalous: bool = True,
        robust_trim_frac: float = 0.2,
        validation_gate: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if defense_mode not in ("none", "clipping", "anomaly", "validation",
                                "robust_median", "robust_trimmed"):
            raise ValueError(f"unsupported defense mode: {defense_mode!r}")
        self.defense_mode = defense_mode
        self.clip_norm = clip_norm
        self.anomaly_suspect_mult = anomaly_suspect_mult
        self.anomaly_detect_mult = anomaly_detect_mult
        self.exclude_highly_anomalous = exclude_highly_anomalous
        self.robust_trim_frac = robust_trim_frac
        self.validation_gate = validation_gate
        from flwr.common import parameters_to_ndarrays
        self._global = parameters_to_ndarrays(self.initial_parameters)
        self._reference_norms: List[float] = []
        self._anomaly: Optional[Any] = None
        self._clipper: Optional[Any] = None
        from src.federated.defense.anomaly import UpdateAnomalyDetector
        from src.federated.defense.clipping import UpdateClipper
        self._anomaly = UpdateAnomalyDetector(
            suspect_mult=anomaly_suspect_mult, detect_mult=anomaly_detect_mult)
        self._clipper = UpdateClipper(max_norm=clip_norm)
        self.defense_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        t0 = time.perf_counter()
        from flwr.common import parameters_to_ndarrays

        # Per-client parameter arrays + update vectors vs the last global.
        # The partition index (0..n-1) reported by each client is used as the
        # stable id; Flower's generated cid (random hex per connection) is only
        # a fallback.
        client_params = [
            (str(fitres.metrics.get("partition_cid", cid.cid)),
             parameters_to_ndarrays(fitres.parameters),
             int(fitres.num_examples), dict(fitres.metrics or {}))
            for cid, fitres in results
        ]
        layer_sizes = [int(g.size) for g in self._global]
        layer_bounds = np.cumsum([0] + layer_sizes)
        update_vectors = []
        for cid_str, params, n, _m in client_params:
            upd = [(np.asarray(p, dtype=np.float32) - g)
                   for p, g in zip(params, self._global)]
            update_vectors.append((cid_str, np.concatenate(
                [u.reshape(-1) for u in upd])))

        # Clipping records norm before/after for every update.
        clipped_vectors = []
        for cid_str, vec in update_vectors:
            c = self._clipper.clip(vec)
            clipped_vectors.append((cid_str, c))

        # Anomaly detection (always measured, applied when configured).
        reference = (float(np.median(self._reference_norms))
                     if self._reference_norms else None)
        anomaly_records = self._anomaly.score_and_classify(
            clipped_vectors, reference_norm=reference)

        # Select aggregation participants.
        from src.federated.defense.anomaly import AnomalyClass
        exclude = set()
        if self.defense_mode == "anomaly" and self.exclude_highly_anomalous:
            exclude = {r.cid for r in anomaly_records
                       if r.classification == AnomalyClass.HIGHLY_ANOMALOUS}
            if exclude:
                logger.warning("round %d: excluding %d HIGHLY_ANOMALOUS client(s): %s",
                               server_round, len(exclude), sorted(exclude))

        # Aggregate updates (weighted mean / robust), then add to global.
        chosen = [(cid_str, v, n) for (cid_str, v), (_, _n, n, _m)
                  in zip(clipped_vectors, client_params)
                  if cid_str not in exclude]
        if not chosen:
            logger.warning("round %d: all clients excluded; keeping previous model",
                           server_round)
            aggregated = list(self._global)
            agg_update = np.zeros_like(update_vectors[0][1])
        else:
            if self.defense_mode == "robust_median":
                from src.federated.defense.robust import coordinate_wise_median
                agg_update = coordinate_wise_median([v for _, v, _ in chosen])
            elif self.defense_mode == "robust_trimmed":
                from src.federated.defense.robust import trimmed_mean
                agg_update = trimmed_mean([v for _, v, _ in chosen],
                                          trim_frac=self.robust_trim_frac)
            else:
                total = sum(n for _, _, n in chosen)
                weights = np.array([n / total for _, _, n in chosen], dtype=float)
                stack = np.stack([v for _, v, _ in chosen], axis=0)
                agg_update = (stack.T @ weights).astype(np.float32)
            agg_layers = [agg_update[lb:le].reshape(self._global[i].shape)
                          for i, (lb, le) in enumerate(
                              zip(layer_bounds[:-1], layer_bounds[1:]))]
            aggregated = [np.asarray(g, dtype=np.float32) + a
                          for g, a in zip(self._global, agg_layers)]

        # Model validation gate: accept / flag / reject the candidate.
        validation_record = None
        if self.validation_gate is not None:
            validation_record = self.validation_gate.validate(
                server_round, aggregated)
            if validation_record.decision.value == "REJECT":
                aggregated = [np.asarray(p, dtype=np.float32).copy()
                              for p in self.validation_gate.trusted_parameters()]
                agg_update = np.zeros_like(agg_update)
        self._global = [np.asarray(p, dtype=np.float32) for p in aggregated]
        self._reference_norms.append(float(np.linalg.norm(agg_update)))

        round_time_ms = (time.perf_counter() - t0) * 1000.0
        upload = sum(int(fitres.metrics.get("upload_bytes", 0))
                     for _, fitres in results)
        download = sum(int(fitres.metrics.get("download_bytes", 0))
                       for _, fitres in results)
        record = {
            "round": server_round,
            "round_time_ms": round(round_time_ms, 3),
            "n_clients_fit": len(results),
            "n_failures": len(failures),
            "n_clients_aggregated": len(chosen),
            "n_excluded_anomalous": len(exclude),
            "excluded_cids": sorted(exclude),
            "download_bytes": download,
            "upload_bytes": upload,
            "bytes_this_round": upload + download,
            "fit_metrics": self._aggregate_fit_metrics(
                [(n, m) for _, _, n, m in client_params]),
            "update_norms": {cid: round(float(np.linalg.norm(v)), 6)
                             for cid, v in update_vectors},
            "anomaly": [r.to_dict() for r in anomaly_records],
            "clipping": [r.to_dict() for r in self._clipper.records[-len(results):]],
            "global_update_norm": round(float(np.linalg.norm(agg_update)), 6),
            "validation": None if validation_record is None
            else validation_record.to_dict(),
        }
        self.rounds.append(record)
        logger.info("round %d: %d clients fit, %d aggregated in %.1f ms (%s, %d bytes)",
                    server_round, len(results), len(chosen), round_time_ms,
                    self.defense_mode, upload + download)

        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(aggregated), record["fit_metrics"]

    # ------------------------------------------------------------------
    def summarize_defense(self) -> Dict[str, Any]:
        """Defense statistics across all rounds."""
        anomaly = self._anomaly.summary() if self._anomaly else {"n_updates": 0}
        clipping = self._clipper.summary() if self._clipper else {"n_updates": 0}
        validation = (self.validation_gate.summary()
                      if self.validation_gate is not None else None)
        n_flagged = 0
        for r in self.rounds:
            n_flagged += sum(1 for a in r.get("anomaly", [])
                             if a["classification"] != "NORMAL")
        total = sum(r.get("n_clients_fit", 0) for r in self.rounds)
        self.defense_summary = {
            "defense_mode": self.defense_mode,
            "total_client_updates": total,
            "anomaly": anomaly,
            "clipping": clipping,
            "validation": validation,
            "updates_flagged_non_normal": n_flagged,
            "updates_flagged_rate": round(n_flagged / total, 6) if total else 0.0,
            "n_rounds": len(self.rounds),
        }
        return self.defense_summary