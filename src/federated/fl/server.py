"""Federated experiment runner: local gRPC Flower server + in-process clients.

Runs a REAL Flower round loop over local gRPC (no Ray required): a genuine
flwr.server with a tracked strategy and genuine NumPyClient workers, each with
its own Phase-4 client data. Records per-round global metrics, per-client
metrics, round times, and actual serialized communication bytes.

Phase 10: dispatches on ``cfg.fl.algorithm`` — ``fedavg`` (FedAvgTracked) or
``fedprox`` (FedProxTracked with ``cfg.fl.proximal_mu``). Both consume the
SAME saved partition, model, seeds, optimizer and hyperparameters so results
are directly comparable (only the algorithm differs).

Phase 11: ``personalized`` (PersonalizedTracked + PersonalizedClient) —
FedPer-style personalization. Clients keep a personal head; only the shared
body crosses the wire. The server-side global evaluation then evaluates the
body with a probe head trained on a balanced sample of the training set, so a
single global test number still exists for comparison with FedAvg/FedProx.

Raw client data never leaves the workers — only parameters and metrics flow
over the wire.
"""

from __future__ import annotations

import copy
import json
import math
import socket
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from flwr.client import start_client
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, Server
from flwr.server.client_manager import SimpleClientManager
from flwr.compat.server.app import init_defaults, run_fl, start_grpc_server

from fedshield.config import ExperimentConfig
from fedshield.logging_setup import get_logger
from src.federated.evaluation.metrics import predict_proba_chunked
from src.federated.fl.client import build_client_fn, build_personalized_client_fn
from src.federated.fl.dataset import PartitionClientData
from src.federated.fl.strategy import (
    FedAvgTracked, FedProxTracked, PersonalizedTracked, DefendedTracked,
)
from src.federated.models.mlp import (
    MLPConfig, build_mlp, build_personalized_mlp,
)
from src.utils.reproducibility import set_all_seeds

logger = get_logger(__name__)

DEFAULT_MAX_MESSAGE_LENGTH = 1_073_741_824  # 1 GiB


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _server_validation_set(
    data: PartitionClientData,
    X_train: np.ndarray,
    y_train: np.ndarray,
    frac: float,
    seed: int,
    max_rows: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Controlled server-side validation set: labeled rows never seen by clients.

    The partition strictly consumes its pool, so the validation set is drawn
    from labeled ``X_train`` rows OUTSIDE the pool. Deterministic (seeded), so
    every round and every defense configuration validate on the SAME rows.
    """
    pool = set(np.asarray(data.pool, dtype=np.int64).tolist())
    held_out = np.asarray([i for i in range(len(y_train)) if i not in pool],
                          dtype=np.int64)
    if len(held_out) == 0:
        raise ValueError(
            "no held-out rows available for the server validation set "
            "(partition consumes the entire training matrix)")
    n_want = min(int(frac * len(pool)), len(held_out), max_rows)
    # Stratify by label so the controlled validation set always contains both
    # classes (a single-class set breaks binary F1 and is not a meaningful
    # candidate-vs-trusted comparison).
    pos = held_out[np.asarray(y_train[held_out]) == 1]
    neg = held_out[np.asarray(y_train[held_out]) == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            "held-out rows do not contain both classes; cannot build a "
            "stratified server validation set")
    rng = np.random.default_rng(seed)
    half = n_want // 2
    k_pos = min(half, len(pos))
    k_neg = min(half, len(neg))
    sel = np.concatenate([
        rng.choice(pos, k_pos, replace=False),
        rng.choice(neg, k_neg, replace=False),
    ])
    return (np.asarray(X_train[sel], dtype=np.float32),
            np.asarray(y_train[sel], dtype=np.int64))


def _run_grpc_server(address: str, config: ServerConfig, strategy, max_message_length: int):
    """Run a Flower server over local gRPC (no signal handlers: thread-safe).

    Mirrors flwr.compat.server.app.start_server but skips signal registration,
    which is only allowed on the main thread.
    """
    server, initialized_config = init_defaults(
        server=None, config=config, strategy=strategy, client_manager=None)
    grpc_server = start_grpc_server(
        client_manager=server.client_manager(),
        server_address=address,  # raw "host:port"; start_grpc_server parses it
        max_message_length=max_message_length,
    )
    history = run_fl(server=server, config=initialized_config)
    grpc_server.stop(grace=1)
    return history


def _initial_parameters(model_cfg: MLPConfig, seed: int, personalized: bool = False):
    set_all_seeds(seed)
    model = build_personalized_mlp(model_cfg) if personalized else build_mlp(model_cfg)
    params = (list(model.body.parameters()) if personalized
              else list(model.parameters()))
    return ndarrays_to_parameters([p.detach().cpu().numpy() for p in params])


def _probe_auc(logits: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney) of the probe head on its own training sample.

    Used as the restart criterion: heads stuck in a degenerate basin rank no
    better than chance (AUC ~0.5-0.6), good heads reach ~0.8+, while in-sample
    accuracy is useless as a gate because the 129-parameter head is
    threshold-calibrated (bias-dominated), not rank-limited.
    """
    order = np.argsort(logits, kind="mergesort")  # ascending: lowest logit -> rank 1
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order)) + 1
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        return 0.5
    return float((ranks[pos].sum() - m * (m + 1) / 2) / (m * n))


def _train_probe_head(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    probe_samples: int,
    probe_epochs: int,
    learning_rate: float,
    batch_size: int,
    max_restarts: int = 3,
) -> Tuple[int, float, int]:
    """Train the 129-parameter probe head on a balanced sample (Phase 11).

    The body is frozen; only the head is optimized, on an equal-sized
    positive/negative sample drawn deterministically from ``seed`` so every
    round evaluates the same probe protocol. ``batch_size`` is clipped to the
    sample size so small evaluations still take real gradient steps.

    A single from-scratch head can get stuck in a degenerate basin (a flipped
    or constant predictor — observed as probe AUC < 0.5 while client F1 keeps
    rising), so the head is trained up to ``max_restarts + 1`` times from
    consecutive deterministic seeds and the BEST (highest rank AUC on the
    probe's own balanced sample) is kept. All restarts are seeded, so the
    protocol stays reproducible.

    Returns ``(n_samples, best_in_sample_auc, restarts_used)``.
    """
    for p in model.body.parameters():
        p.requires_grad = False
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_train == 1)
    neg = np.flatnonzero(y_train == 0)
    k = min(probe_samples // 2, len(pos), len(neg))
    idx = np.concatenate([
        rng.choice(pos, k, replace=False),
        rng.choice(neg, k, replace=False),
    ])
    batch_size = min(batch_size, len(idx))
    best_auc, best_restarts = -1.0, max_restarts
    for attempt in range(max_restarts + 1):
        _init_probe_head_deterministic(model, seed + attempt)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
        for _ in range(probe_epochs):
            perm = rng.permutation(len(idx))
            for start in range(0, len(idx), batch_size):
                sel = idx[perm[start:start + batch_size]]
                xb = torch.from_numpy(X_train[sel])
                yb = torch.from_numpy(y_train[sel]).float()
                optimizer.zero_grad()
                logits = model(xb).ravel()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            all_logits = np.empty(len(idx), dtype=np.float32)
            for start in range(0, len(idx), batch_size):
                sel = idx[start:start + batch_size]
                logits = model(torch.from_numpy(X_train[sel])).ravel()
                all_logits[start:start + batch_size] = logits.numpy()
        auc = _probe_auc(all_logits, y_train[idx])
        if auc > best_auc:
            best_auc = auc
            best_restarts = attempt
        if auc >= 0.65:
            break
    return int(len(idx)), best_auc, best_restarts


def _init_probe_head_deterministic(model, probe_seed: int) -> None:
    """Re-init the probe head with a LOCAL generator (no global RNG use).

    The probe head must start from the same init every round so round-to-round
    global evaluations are comparable, but the server thread must not touch the
    global torch RNG (client threads draw dropout masks from it concurrently).
    Mirrors nn.Linear's default init (kaiming_uniform, bias uniform).
    """
    g = torch.Generator().manual_seed(probe_seed)
    with torch.no_grad():
        for module in model.head.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5),
                                           generator=g)
            fan_in = module.weight.size(1)
            bound = 1.0 / math.sqrt(fan_in)
            # uniform_ WITHOUT a generator would draw from the GLOBAL RNG,
            # which races with client-construction threads; use a local one.
            torch.nn.init.uniform_(module.bias, -bound, bound, generator=g)


def _global_evaluate_fn(
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_inv: Optional[np.ndarray],
    model_cfg: MLPConfig,
    chunk: int = 20000,
    personalized: bool = False,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    probe_train_samples: int = 100_000,
    probe_epochs: int = 10,
    probe_seed: int = 42,
    probe_learning_rate: float = 1e-2,
    probe_batch_size: int = 512,
):
    def evaluate_fn(server_round, parameters, config):
        from src.federated.evaluation.metrics import compute_metrics

        model = build_personalized_mlp(model_cfg) if personalized else build_mlp(model_cfg)
        with torch.no_grad():
            for p, new_p in zip(
                (model.body.parameters() if personalized else model.parameters()),
                parameters):
                p.copy_(torch.from_numpy(new_p))
        model.eval()
        probe_n = None
        if personalized:
            probe_n, probe_train_auc, probe_restarts = _train_probe_head(
                model, X_train, y_train, probe_seed,
                probe_train_samples, probe_epochs,
                learning_rate=probe_learning_rate, batch_size=probe_batch_size)
        probs = predict_proba_chunked(model, X_test, chunk=chunk,
                                      scale_inv=scale_inv)
        y = np.asarray(y_test)
        loss = float(np.mean(-(y * np.log(np.clip(probs, 1e-7, 1 - 1e-7))
                               + (1 - y) * np.log(np.clip(1 - probs, 1e-7, 1 - 1e-7)))))
        m = compute_metrics(y, probs)
        metrics = {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
        metrics["n_test"] = int(len(y))
        if personalized:
            metrics["probe_head_trained"] = True
            metrics["probe_train_samples"] = probe_n
            metrics["probe_train_auc"] = probe_train_auc
            metrics["probe_restarts"] = probe_restarts
        logger.info("round %d global test: acc=%.4f f1=%.4f auc=%s%s",
                    server_round, m["accuracy"], m["f1"],
                    f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "None",
                    f" (probe head, n={probe_n}, in-sample auc={probe_train_auc:.3f},"
                    f" restarts={probe_restarts})" if personalized else "")
        return loss, metrics

    return evaluate_fn


def run_fl_experiment(
    cfg: ExperimentConfig,
    partition_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_inv: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    grpc_max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    controller: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one FL experiment on a saved Phase-4 partition; return results.

    The algorithm comes from ``cfg.fl.algorithm`` (fedavg | fedprox |
    personalized). All algorithms share the identical partition, model,
    optimizer, learning rate, batch size, epochs, rounds and seed — only the
    strategy differs, so the results are directly comparable. For
    ``personalized`` only the shared body crosses the wire and the global
    evaluation uses a probe head trained on a balanced sample.

    Phase 14: when ``cfg.defense.mode`` is not "none" or ``cfg.attack.enabled``
    is set, the run uses the DefendedTracked strategy (clipping / anomaly
    detection / model validation / robust aggregation) and clients simulate
    the configured abnormal updates (synthetic only — no real malware).

    controller: optional Phase-12 TrainingController; when None and
    ``cfg.endpoint.resource.enabled`` is set, one is built from the config.
    The controller gates every client's local epochs — real-time detection
    never consults it.
    """
    algorithm = cfg.fl.algorithm
    if algorithm not in ("fedavg", "fedprox", "personalized"):
        raise ValueError(
            f"unsupported FL algorithm: {algorithm!r} "
            f"(supported: fedavg, fedprox, personalized)")
    personalized = algorithm == "personalized"
    proximal_mu = cfg.fl.proximal_mu
    seed = seed if seed is not None else cfg.seed
    set_all_seeds(seed)
    t_run0 = time.perf_counter()

    data = PartitionClientData(partition_dir, X_train, y_train, scale_inv=scale_inv)
    if data.n_clients != cfg.fl.num_clients:
        raise ValueError(
            f"partition has {data.n_clients} clients but config says {cfg.fl.num_clients}; "
            f"the FL system consumes the saved partition configuration")

    model_cfg = MLPConfig(
        input_dim=int(X_train.shape[1]),
        hidden_layers=tuple(cfg.model.hidden_layers),
        dropout=cfg.model.dropout,
    )

    initial_parameters = _initial_parameters(model_cfg, seed, personalized=personalized)
    from flwr.common import parameters_to_ndarrays

    model_params_bytes = int(sum(
        np.asarray(a).nbytes for a in parameters_to_ndarrays(initial_parameters)))
    model_param_count = int(sum(
        np.asarray(a).size for a in parameters_to_ndarrays(initial_parameters)))
    # For personalized, also compute full model size for reference (actual communication is body only)
    if personalized:
        from src.federated.models.mlp import build_mlp

        _tmp_full = build_mlp(model_cfg)
        _full_params = [p.detach().cpu().numpy() for p in _tmp_full.parameters()]
        full_model_bytes = int(sum(p.nbytes for p in _full_params))
        full_model_param_count = int(sum(p.size for p in _full_params))
    else:
        full_model_bytes = model_params_bytes
        full_model_param_count = model_param_count

    evaluate_fn = _global_evaluate_fn(
        X_test, y_test, scale_inv, model_cfg,
        personalized=personalized,
        X_train=X_train if personalized else None,
        y_train=y_train if personalized else None,
        probe_train_samples=cfg.fl.personalized_probe_samples,
        probe_epochs=cfg.fl.personalized_probe_epochs,
        probe_seed=seed,
        probe_learning_rate=cfg.fl.personalized_probe_learning_rate,
        probe_batch_size=cfg.train.batch_size,
    )
    strategy_kwargs = dict(
        num_clients=data.n_clients,
        fraction_fit=cfg.fl.client_fraction,
        num_rounds=cfg.fl.num_rounds,
        local_epochs=cfg.train.local_epochs,
        learning_rate=cfg.train.learning_rate,
        batch_size=cfg.train.batch_size,
        evaluate_every=cfg.fl.evaluate_every,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        fraction_evaluate=cfg.fl.client_fraction,
    )
    if algorithm == "fedprox":
        strategy = FedProxTracked(proximal_mu=proximal_mu, **strategy_kwargs)
    elif algorithm == "personalized":
        strategy = PersonalizedTracked(
            head_epochs=cfg.fl.personalized_head_epochs,
            head_learning_rate=cfg.fl.personalized_head_learning_rate,
            **strategy_kwargs)
    else:
        strategy = FedAvgTracked(**strategy_kwargs)

    # Phase 14: simulated attack + server-side defenses.
    from src.federated.defense.attack import AttackSpec
    from src.federated.defense.validation import ValidationGate
    from src.federated.privacy.dp import PrivacySpec, privacy_report
    attack_spec = AttackSpec.from_config(cfg.attack)
    privacy_spec = PrivacySpec.from_config(cfg.privacy)
    validation_gate = None
    defense_active = cfg.defense.mode != "none" or attack_spec.is_malicious
    privacy_active = bool(privacy_spec.enabled)
    if defense_active:
        if cfg.defense.mode == "validation":
            X_val, y_val = _server_validation_set(
                data, X_train, y_train, cfg.defense.validation_frac, seed)
            validation_gate = ValidationGate(
                X_val, y_val, scale_inv, model_cfg,
                tolerance=cfg.defense.validation_tolerance)
            logger.info("validation gate: %d held-out rows (never in any client)",
                        len(X_val))
        strategy = DefendedTracked(
            defense_mode=cfg.defense.mode,
            clip_norm=cfg.defense.clip_norm,
            anomaly_suspect_mult=cfg.defense.anomaly_suspect_mult,
            anomaly_detect_mult=cfg.defense.anomaly_detect_mult,
            exclude_highly_anomalous=cfg.defense.exclude_highly_anomalous,
            robust_trim_frac=cfg.defense.robust_trim_frac,
            validation_gate=validation_gate,
            **strategy_kwargs)
    logger.info("%s experiment: %s strategy, %d rounds, %d clients, proximal_mu=%s%s%s%s",
                algorithm, data.strategy(), cfg.fl.num_rounds, data.n_clients,
                proximal_mu, " (personalized head)" if personalized else "",
                f" | defense={cfg.defense.mode} attack={attack_spec.attack_type}"
                if defense_active else "",
                f" | privacy={privacy_spec.strength_label()} sigma={privacy_spec.sigma:.3f}"
                if privacy_active else "")

    # Phase 19: configurable server address (never hardcode localhost in config)
    # For in-process simulation, bind to cfg.server.host if it is localhost/LAN,
    # otherwise use 127.0.0.1 for local sim; internet deployment uses cfg.server.host via separate server process
    host = getattr(cfg, "server", None).host if hasattr(cfg, "server") and getattr(cfg.server, "host", None) else "127.0.0.1"
    # For local simulation, always bind to 127.0.0.1 if host is not 0.0.0.0/LAN — keep Internet host for URL construction only
    bind_host = "127.0.0.1" if host in ("127.0.0.1", "localhost") else "0.0.0.0"
    port = _free_port()
    address = f"{bind_host}:{port}"
    external_address = f"{host}:{port}" if host not in ("0.0.0.0",) else address

    # Phase 12: resource-aware training gate (one controller, shared by all
    # client threads; built from config unless an explicit one is injected).
    if controller is None and cfg.endpoint.resource.enabled:
        from src.endpoint.resource import create_controller_from_config
        controller = create_controller_from_config(cfg.endpoint.resource)
        logger.info("resource-aware FL enabled: %s",
                    cfg.endpoint.resource)
    if controller is not None:
        controller.request_start()

    server_config = ServerConfig(num_rounds=cfg.fl.num_rounds)
    server_thread = threading.Thread(
        target=_run_grpc_server,
        args=(address, server_config, strategy, grpc_max_message_length),
        daemon=True,
    )
    server_thread.start()

    # give the gRPC server a moment to bind before workers connect
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection((bind_host if bind_host != "0.0.0.0" else "127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)

    client_threads = []
    for cid in range(data.n_clients):
        # bind the worker to its partition index: in-process start_client
        # workers all receive the same node id, so the cid string is unreliable
        builder = build_personalized_client_fn if personalized else build_client_fn
        client_fn = builder(data, model_cfg, seed=seed, cid=cid,
                            controller=controller,
                            attack_spec=attack_spec if defense_active else None,
                            privacy_spec=privacy_spec if privacy_active else None)
        t = threading.Thread(
            target=start_client,
            kwargs=dict(server_address=address, client_fn=client_fn,
                        grpc_max_message_length=grpc_max_message_length,
                        max_retries=5, max_wait_time=1.0),
            daemon=True,
        )
        t.start()
        client_threads.append(t)

    server_thread.join()
    # Phase 14: join the client threads after the server stops so their
    # local data is released before the next experiment (sequential cells
    # otherwise accumulate several GB per run and OOM on this machine).
    for t in client_threads:
        t.join(timeout=20)
    if controller is not None:
        controller.finish()
    training_time_s = time.perf_counter() - t_run0

    summary = strategy.summary()
    final_metrics = None
    for rec in reversed(strategy.rounds):
        if rec.get("global_eval"):
            final_metrics = rec["global_eval"]
            break

    results: Dict[str, Any] = {
        "experiment": {
            "algorithm": algorithm,
            "proximal_mu": proximal_mu,
            "strategy": data.strategy(),
            "partition_dir": str(partition_dir),
            "partition_config": data.config.to_dict(),
            "fl_config": {
                "algorithm": algorithm,
                "proximal_mu": proximal_mu,
                "num_clients": data.n_clients,
                "num_rounds": cfg.fl.num_rounds,
                "client_fraction": cfg.fl.client_fraction,
                "local_epochs": cfg.train.local_epochs,
                "learning_rate": cfg.train.learning_rate,
                "batch_size": cfg.train.batch_size,
                "seed": seed,
            },
            "model": {
                "input_dim": int(X_train.shape[1]),
                "hidden_layers": list(cfg.model.hidden_layers),
                "dropout": cfg.model.dropout,
            },
        },
        "communication": {
            "model_parameter_count": model_param_count,
            "model_parameter_bytes": model_params_bytes,
            "full_model_parameter_count": full_model_param_count,
            "full_model_bytes": full_model_bytes,
            "per_round": [{
                "round": r["round"],
                "download_bytes": r["download_bytes"],
                "upload_bytes": r["upload_bytes"],
                "bytes_this_round": r["bytes_this_round"],
                "n_clients": r["n_clients_fit"],
            } for r in summary["rounds"]],
            "totals": summary["totals"],
        },
        "rounds": [{
            "round": r["round"],
            "round_time_ms": r["round_time_ms"],
            "n_clients_fit": r["n_clients_fit"],
            "avg_client_f1": r.get("avg_client_f1"),
            "worst_client_f1": r.get("worst_client_f1"),
            "client_f1_variance": r.get("client_f1_variance"),
            "global_eval": r.get("global_eval"),
            "defense": {
                "n_clients_aggregated": r.get("n_clients_aggregated"),
                "n_excluded_anomalous": r.get("n_excluded_anomalous", 0),
                "excluded_cids": r.get("excluded_cids", []),
                "update_norms": r.get("update_norms"),
                "anomaly": r.get("anomaly"),
                "clipping": r.get("clipping"),
                "validation": r.get("validation"),
                "global_update_norm": r.get("global_update_norm"),
            },
        } for r in summary["rounds"]],
        "per_client_metrics": [
            {"round": r["round"], "clients": r.get("per_client_metrics")}
            for r in summary["rounds"] if r.get("per_client_metrics")],
        "final_global_test_metrics": final_metrics,
        "training_time_s": round(training_time_s, 3),
        "round_time_s_mean": round(
            summary["totals"]["total_round_time_ms"] / max(len(summary["rounds"]), 1) / 1000.0, 3),
    }

    if controller is not None:
        results["resource"] = {
            "enabled": True,
            "policy": asdict(cfg.endpoint.resource),
            "controller": controller.status(),
        }

    if defense_active:
        from src.federated.defense.attack import attack_report
        from src.federated.defense.attack import is_malicious_cid
        malicious_cids = [c for c in range(data.n_clients)
                          if is_malicious_cid(c, attack_spec)]
        results["attack"] = attack_report(attack_spec, malicious_cids)
        results["defense"] = strategy.summarize_defense()

    if privacy_active:
        results["privacy"] = privacy_report(privacy_spec, cfg.fl.num_rounds, sampling_rate=cfg.fl.client_fraction)
        results["privacy"]["config"] = {
            "max_grad_norm": privacy_spec.max_grad_norm,
            "noise_multiplier": privacy_spec.noise_multiplier,
            "sigma": privacy_spec.sigma,
            "delta": privacy_spec.delta,
            "enabled": True,
        }
    else:
        results["privacy"] = {"enabled": False, "mode": "none"}

    if personalized:
        head_model = build_personalized_mlp(model_cfg)
        results["experiment"]["personalized"] = {
            "shared_params": "body (all hidden layers)",
            "personalized_params": "head (final Linear)",
            "head_parameter_count": int(sum(
                p.numel() for p in head_model.head.parameters())),
            "body_parameter_count": int(sum(
                p.numel() for p in head_model.body.parameters())),
            "probe_train_samples": cfg.fl.personalized_probe_samples,
            "probe_epochs": cfg.fl.personalized_probe_epochs,
            "probe_learning_rate": cfg.fl.personalized_probe_learning_rate,
            "head_epochs": cfg.fl.personalized_head_epochs,
            "head_learning_rate": cfg.fl.personalized_head_learning_rate,
        }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(results["experiment"], indent=2), encoding="utf-8")
        (output_dir / "rounds.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results["rounds"]) + "\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
        logger.info("experiment results written to %s", output_dir)

    return results


def run_fedavg_experiment(
    cfg: ExperimentConfig,
    partition_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_inv: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    grpc_max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
) -> Dict[str, Any]:
    """Backwards-compatible FedAvg-only entry point (Phase 9 behaviour).

    Forces ``cfg.fl.algorithm = "fedavg"`` and delegates to run_fl_experiment.
    """
    cfg = copy.deepcopy(cfg)
    cfg.fl.algorithm = "fedavg"
    return run_fl_experiment(
        cfg, partition_dir, X_train, y_train, X_test, y_test,
        scale_inv=scale_inv, output_dir=output_dir, seed=seed,
        grpc_max_message_length=grpc_max_message_length)