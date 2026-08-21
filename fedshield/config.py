"""Experiment configuration: typed dataclasses loaded from YAML files.

Configuration drives every experiment (algorithm, partitioning, model, training).
All values have defaults in configs/default.yaml so experiments are reproducible
by recording the fully-resolved configuration.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    """Small configurable MLP for static PE tabular features."""

    input_dim: int = 2381
    hidden_layers: tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    activation: str = "relu"  # relu | gelu | tanh
    version: str = "mlp-v1"


@dataclass
class TrainConfig:
    """Local training hyperparameters."""

    batch_size: int = 512
    local_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # adam | adamw | sgd
    seed: int = 42
    # Centralized baseline training:
    epochs: int = 20
    early_stopping_patience: int = 3  # 0 disables early stopping


@dataclass
class FlConfig:
    """Federated protocol configuration."""

    num_clients: int = 10
    num_rounds: int = 30
    client_fraction: float = 1.0
    algorithm: str = "fedavg"  # fedavg | fedprox | personalized
    proximal_mu: float = 0.0
    evaluate_every: int = 1
    # Phase 11 personalized FL: the server-side probe head used for the global
    # test evaluation is trained on a balanced sample of this size for this
    # many epochs (body frozen) at this learning rate. The head operates on
    # LayerNorm-standardized body features.
    personalized_probe_samples: int = 100_000
    personalized_probe_epochs: int = 10
    personalized_probe_learning_rate: float = 1e-2
    # FedRep-style client training: each round the personal head first adapts
    # to the current global body for this many epochs (body frozen) at this
    # learning rate, then the body trains for local_epochs with the head
    # frozen.
    personalized_head_epochs: int = 5
    personalized_head_learning_rate: float = 1e-2


@dataclass
class PartitionConfig:
    """Data partitioning (non-IID simulation) configuration.

    strategy: iid | mild | moderate | severe | quantity_skew | label_skew |
              family_skew | combined_severe
    severity: "mild" | "moderate" | "severe" (label-Dirichlet alpha
              1.0 / 0.5 / 0.1) or a raw Dirichlet alpha float; used by
              non-IID strategies (mild/moderate/severe/label_skew).
    test_strategy: "global_test" keeps the official test split out of every
              client (used for all clients); "per_client" additionally
              reports the local holdout (client val) as a test signal.
    """

    strategy: str = "iid"
    severity: Any = None
    clients: int = 10
    seed: int = 42
    val_fraction: float = 0.1
    min_samples_per_client: int = 5000
    test_strategy: str = "global_test"
    # limit applied to the pooled dataset before partitioning (None = all)
    max_samples: Optional[int] = None


@dataclass
class DataConfig:
    """Dataset paths and versions."""

    data_dir: str = "data"
    ember_version: str = "2018_2"
    # subsample of the raw EMBER train split to use as the pool (None = all)
    train_cap: Optional[int] = None
    test_cap: Optional[int] = None


@dataclass
class MonitorConfig:
    """File system monitoring configuration."""

    watched_directories: tuple[str, ...] = (
        "~/Downloads",
    )
    recursive: bool = True
    # Debounce time (seconds) to wait for file writes to complete
    stability_wait: float = 2.0
    # Scan cycle interval (seconds)
    poll_interval: float = 1.0
    # Max file size to analyze (bytes), 0 = no limit
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    # Extensions considered for PE analysis (case-insensitive); the monitor
    # also sniffs the MZ header, so extensions are never the only signal
    pe_extensions: tuple[str, ...] = (".exe", ".dll", ".sys", ".scr", ".com")


@dataclass
class InferenceConfig:
    """Local inference service configuration."""

    model_path: str = "models/local_model.pt"
    # Path to feature schema (JSON) matching the model's expected features
    feature_schema_path: str = "models/feature_schema.json"
    # Device for inference: cpu | cuda | mps
    device: str = "cpu"
    # Batch size for inference (usually 1 for real-time)
    batch_size: int = 1
    # Confidence threshold for "uncertain" verdict
    uncertainty_threshold: float = 0.5


@dataclass
class RiskConfig:
    """Risk policy configuration: maps malware probability to action."""

    # Probability thresholds: [allow_max, warn_max, quarantine_min]
    # e.g., [0.3, 0.7] => p<0.3 ALLOW, 0.3<=p<0.7 WARN, p>=0.7 QUARANTINE
    thresholds: tuple[float, float] = (0.3, 0.7)
    # Action labels corresponding to thresholds
    actions: tuple[str, str, str] = ("ALLOW", "WARN", "QUARANTINE")


@dataclass
class QuarantineConfig:
    """Quarantine system configuration."""

    quarantine_dir: str = "quarantine"
    # Keep original filename in quarantine (with hash prefix to avoid collisions)
    preserve_filename: bool = True
    # Record metadata JSON alongside quarantined file
    record_metadata: bool = True


@dataclass
class NotificationConfig:
    """Notification service configuration."""

    enabled: bool = True
    # Channels: console | file | webhook | toast (Windows)
    channels: tuple[str, ...] = ("console", "file")
    # File path for file-based notifications
    notification_log: str = "logs/notifications.log"
    # Minimum risk level to notify: allow | warn | quarantine
    min_level: str = "warn"


@dataclass
class HistoryConfig:
    """Detection history configuration."""

    enabled: bool = True
    history_db: str = "history/detections.db"
    # Retention: days to keep history (0 = forever)
    retention_days: int = 0


@dataclass
class ResourceConfig:
    """Resource-aware FL training policy (Phase 12).

    Every constraint is optional: None/False disables it and the code then
    never applies it (missing metrics are skipped the same way, so a policy
    works identically on machines that cannot measure a given signal). The
    values themselves are configured in configs/default.yaml — there are no
    hardcoded universal thresholds.

    enabled: master switch; when False training is always permitted.
    max_cpu_percent: pause training above this CPU utilization.
    min_battery_percent: pause training below this battery level.
    require_ac_power: pause training while running on battery.
    idle_only: pause training while the user is active (idle time <
        idle_min_seconds); requires an activity metric, else skipped.
    idle_min_seconds: user-inactivity threshold for idle_only.
    max_training_duration_sec: cancel training once a single fit has run
        this long (0/None = unlimited).
    min_free_memory_mb: pause training below this free RAM.
    check_interval_sec: how often the controller re-samples resources.
    """

    enabled: bool = False
    max_cpu_percent: Optional[float] = None
    min_battery_percent: Optional[float] = None
    require_ac_power: bool = False
    idle_only: bool = False
    idle_min_seconds: Optional[float] = None
    max_training_duration_sec: Optional[float] = None
    min_free_memory_mb: Optional[float] = None
    check_interval_sec: float = 5.0


@dataclass
class DriftConfig:
    """Concept drift detection and adaptive retraining (Phase 13).

    All thresholds and windows are configurable in configs/default.yaml —
    no hardcoded universal defaults.

    enabled: master switch for adaptive retraining workflow.
    timestamp_feature: name of the feature column used for temporal splits.
    reference_frac: fraction of data used as the reference (older) distribution
        for drift detection; the remainder forms the streaming "current" data.
    psi_feature_subset: optional list of feature indices to monitor; None = all.
    psi_bins: number of bins for Population Stability Index (PSI) calculation.
    psi_suspect_threshold: PSI >= this → DRIFT_SUSPECTED.
    psi_detected_threshold: PSI >= this → DRIFT_DETECTED.
    cooldown_hours: minimum hours between retraining events.
    min_new_samples: minimum new samples required to trigger retraining.
    max_retraining_rounds: FL rounds for adaptive retraining runs.
    max_frequency_per_day: maximum retraining events per 24h.
    validation_frac: fraction of current data reserved for candidate validation.
    """

    enabled: bool = False
    timestamp_feature: str = "header_timestamp"
    reference_frac: float = 0.5
    psi_feature_subset: Optional[list[int]] = None
    psi_bins: int = 10
    psi_suspect_threshold: float = 0.1
    psi_detected_threshold: float = 0.2
    cooldown_hours: float = 24.0
    min_new_samples: int = 10000
    max_retraining_rounds: int = 5
    max_frequency_per_day: int = 1
    validation_frac: float = 0.2


@dataclass
class EndpointConfig:
    """Endpoint Protection Engine configuration."""

    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    quarantine: QuarantineConfig = field(default_factory=QuarantineConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
    log_dir: str = "logs"
    # Write logs to a file in log_dir
    file_output: bool = True
    # Optional structured (JSON) file output in addition to human-readable
    structured_output: bool = False


@dataclass
class ServerNetworkConfig:
    """Secure client/server networking (Phase 19). Never hardcode localhost.

    host: configurable (localhost | LAN IP | hostname | internet endpoint)
    port: configurable
    secure: when True, TLS/HTTPS is used; when False, plain channel (local sim)
    tls_cert / tls_key / ca_cert: paths to PEM files (optional, for TLS)
    """

    host: str = "127.0.0.1"
    port: int = 8080
    secure: bool = True
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None
    ca_cert: Optional[str] = None


@dataclass
class ClientIdentityConfig:
    """Client authentication identity (Phase 19)."""

    client_id: str = "client-001"
    token: Optional[str] = None  # bearer token / HMAC credential
    role: str = "client"  # client | admin


@dataclass
class PrivacyConfig:
    """Differential-privacy / privacy accounting (Phase 15).

    enabled: master switch; when False no DP mechanism is applied.
    noise_multiplier: Gaussian noise scale (sigma) for DP-SGD style.
    max_grad_norm: per-sample gradient clipping norm.
    delta: target delta for (epsilon, delta)-DP accounting.
    secure_rng: use cryptographically secure RNG when True (slower).
    accounting_mode: epsilon accounting method identifier.
    """

    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    secure_rng: bool = False
    accounting_mode: str = "rdp"


@dataclass
class AttackConfig:
    """Controlled malicious-client simulation (Phase 14).

    Produces intentionally abnormal model updates ONLY — no real malware,
    no operational attack tooling. Purpose is measurement: attack impact,
    detection capability, mitigation capability.

    enabled: master switch; when False all clients are honest.
    attack_type: none | label_flip | scaled_update | replacement.
        label_flip: malicious clients train on flipped labels (data-level).
        scaled_update: malicious clients scale their honest update by
            update_scale (abnormal-magnitude update).
        replacement: malicious clients return a large random parameter
            vector unrelated to local training (out-of-distribution update).
    n_malicious: number of malicious clients (first n of the partition).
    update_scale: magnitude multiplier for scaled_update.
    flip_frac: fraction of the malicious client's labels flipped.
    """

    enabled: bool = False
    attack_type: str = "none"
    n_malicious: int = 2
    update_scale: float = 20.0
    flip_frac: float = 1.0
    seed: int = 42


@dataclass
class DefenseConfig:
    """Server-side poisoning defenses (Phase 14).

    mode: none | clipping | anomaly | validation | robust_median |
        robust_trimmed. All metrics are recorded for every mode, so a
        comparison run can report detection/FP rates even for the baseline.

    clip_norm: maximum L2 norm of a client's parameter update (before
        aggregation). Updates above the threshold are scaled down to it.
        None = clipping disabled (even in clipping mode).
    anomaly_suspect_mult: per-client anomaly score >= this multiple of the
        robust scale (MAD of peer scores) → SUSPICIOUS.
    anomaly_detect_mult: score >= this multiple → HIGHLY_ANOMALOUS.
    exclude_highly_anomalous: drop HIGHLY_ANOMALOUS clients from the
        aggregation when anomaly detection is active.
    robust_trim_frac: fraction trimmed per side in robust_trimmed mode.
    validation_frac: fraction of the training rows NOT used by any client
        held out by the server as the controlled validation set.
    validation_tolerance: candidate F1 below (trusted F1 - tolerance) is
        REJECTED (previous trusted model retained).
    """

    mode: str = "none"
    clip_norm: Optional[float] = None
    anomaly_suspect_mult: float = 3.0
    anomaly_detect_mult: float = 6.0
    exclude_highly_anomalous: bool = True
    robust_trim_frac: float = 0.2
    validation_frac: float = 0.05
    validation_tolerance: float = 0.01


@dataclass
class ExperimentConfig:
    """Root configuration for one experiment run.

    Top-level fields map to the Phase 15 unified-experiment schema:
    dataset/data, seed, algorithm/fl.algorithm, clients/fl.num_clients,
    client fraction/fl.client_fraction, partition strategy/severity,
    model, learning rate/train.learning_rate, batch size/train.batch_size,
    local epochs/train.local_epochs, FL rounds/fl.num_rounds,
    FedProx mu/fl.proximal_mu, personalization/fl.personalized_*,
    resource policy/endpoint.resource, drift/endpoint.drift,
    privacy/privacy, security/attack+defense.
    """

    name: str = "default"
    seed: int = 42
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fl: FlConfig = field(default_factory=FlConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    server: ServerNetworkConfig = field(default_factory=ServerNetworkConfig)
    client_identity: ClientIdentityConfig = field(default_factory=ClientIdentityConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate a YAML config file into an ExperimentConfig."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load a JSON config file into an ExperimentConfig."""
        import json as _json

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = _json.load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        """Load YAML or JSON (by extension) into an ExperimentConfig."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        return cls.from_yaml(path)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        """Build an ExperimentConfig from a nested dict, keeping defaults for absent keys."""
        raw = _normalize_aliases(raw)
        return cls(**{k: v for k, v in _build(root_fields(cls), raw).items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        """Fully-resolved configuration as a JSON-serializable dict (for DB/experiment log)."""
        return _flatten(self)

    def with_overrides(self, **overrides: Any) -> "ExperimentConfig":
        """Return a deep copy with dotted-key overrides, e.g. fl.num_clients=5."""
        new = copy.deepcopy(self)
        for key, value in overrides.items():
            set_dotted(new, key, value)
        return new


def _resolve_type(typ: Any) -> Any:
    """Resolve string annotations (PEP 563) to actual classes via module globals."""
    if isinstance(typ, str):
        return globals().get(typ, typ)
    return typ


def _sub_spec(cls: type) -> dict[str, Any]:
    """Field-name -> resolved-type map for a dataclass (skips non-config keys)."""
    return {f.name: _resolve_type(f.type) for f in fields(cls) if f.name != "name"}


def root_fields(cls) -> dict[str, Any]:
    return _sub_spec(cls)


def _normalize_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    """Map Phase-15 flat aliases to the nested config structure.

    Supports both flat keys (dataset, algorithm, clients, ...) and nested
    dicts. Flat keys take precedence and are merged into the nested dicts.
    Never mutates the input.
    """
    import copy as _copy

    d = _copy.deepcopy(raw)
    # dataset alias -> data
    if "dataset" in d and "data" not in d:
        v = d.pop("dataset")
        if isinstance(v, str):
            d["data"] = {"ember_version": v}
        elif isinstance(v, dict):
            d["data"] = v
    elif "dataset" in d and isinstance(d.get("data"), dict) and isinstance(d["dataset"], dict):
        # merge
        merged = {**d.pop("dataset")}
        d["data"] = {**merged, **d["data"]}

    # algorithm -> fl.algorithm
    if "algorithm" in d:
        d.setdefault("fl", {})["algorithm"] = d.pop("algorithm")
    # clients -> fl.num_clients + partition.clients
    if "clients" in d:
        v = d.pop("clients")
        d.setdefault("fl", {})["num_clients"] = v
        d.setdefault("partition", {})["clients"] = v
    if "client_fraction" in d:
        d.setdefault("fl", {})["client_fraction"] = d.pop("client_fraction")
    if "partition_strategy" in d:
        d.setdefault("partition", {})["strategy"] = d.pop("partition_strategy")
    if "non_iid_severity" in d:
        d.setdefault("partition", {})["severity"] = d.pop("non_iid_severity")
    if "severity" in d and "partition" not in d:
        d.setdefault("partition", {})["severity"] = d.pop("severity")
    # model alias (already nested, pass through)
    if "learning_rate" in d:
        d.setdefault("train", {})["learning_rate"] = d.pop("learning_rate")
    if "batch_size" in d:
        d.setdefault("train", {})["batch_size"] = d.pop("batch_size")
    if "local_epochs" in d:
        d.setdefault("train", {})["local_epochs"] = d.pop("local_epochs")
    for k in ("fl_rounds", "num_rounds", "rounds"):
        if k in d:
            d.setdefault("fl", {})["num_rounds"] = d.pop(k)
            break
    for k in ("fedprox_mu", "proximal_mu", "mu"):
        if k in d:
            d.setdefault("fl", {})["proximal_mu"] = d.pop(k)
            break
    # personalization settings
    if "personalization" in d:
        v = d.pop("personalization")
        if isinstance(v, dict):
            for pk, pv in v.items():
                d.setdefault("fl", {})[pk] = pv
    if "personalization_settings" in d:
        v = d.pop("personalization_settings")
        if isinstance(v, dict):
            for pk, pv in v.items():
                d.setdefault("fl", {})[pk] = pv
    # resource/drift/privacy aliases
    for flat, nested in (
        ("resource_policy", ("endpoint", "resource")),
        ("resource", ("endpoint", "resource")),
        ("drift_settings", ("endpoint", "drift")),
        ("drift", ("endpoint", "drift")),
        ("privacy_settings", ("privacy",)),
        ("privacy_config", ("privacy",)),
    ):
        if flat in d:
            v = d.pop(flat)
            if isinstance(v, dict):
                cur = d
                for part in nested[:-1]:
                    cur = cur.setdefault(part, {})
                leaf = nested[-1]
                cur.setdefault(leaf, {}).update(v)
            else:
                # scalar not expected, ignore
                pass
    # security_settings -> attack + defense + optional privacy
    if "security_settings" in d or "security" in d:
        v = d.pop("security_settings", d.pop("security", None))
        if isinstance(v, dict):
            if "attack" in v:
                d.setdefault("attack", {}).update(v["attack"] if isinstance(v["attack"], dict) else {})
            if "defense" in v:
                d.setdefault("defense", {}).update(v["defense"] if isinstance(v["defense"], dict) else {})
            if "privacy" in v:
                d.setdefault("privacy", {}).update(v["privacy"] if isinstance(v["privacy"], dict) else {})
            # flat security keys that look like attack/defense keys
            for k in ("attack_type", "n_malicious", "update_scale", "flip_frac"):
                if k in v:
                    d.setdefault("attack", {})[k] = v[k]
            for k in ("mode", "clip_norm", "anomaly_suspect_mult", "anomaly_detect_mult"):
                if k in v:
                    d.setdefault("defense", {})[k] = v[k]
    return d


def _build(spec: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Recursively fill dataclass specs from a raw dict."""
    result: dict[str, Any] = {}
    for key, typ in spec.items():
        if key not in raw:
            continue
        value = raw[key]
        resolved = _resolve_type(typ)
        if isinstance(value, dict) and hasattr(resolved, "__dataclass_fields__"):
            sub = _build(_sub_spec(resolved), value)
            result[key] = resolved(**sub)
        else:
            result[key] = value
    return result


def _flatten(obj: Any) -> dict[str, Any]:
    """Serialize a dataclass tree to a plain dict (lists/tuples remain)."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _flatten(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def set_dotted(obj: Any, dotted_key: str, value: Any) -> None:
    """Set a value at a dotted path, e.g. fl.num_rounds=10, on a dataclass tree."""
    parts = dotted_key.split(".")
    current = obj
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)
