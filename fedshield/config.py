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
class EndpointConfig:
    """Endpoint Protection Engine configuration."""

    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    quarantine: QuarantineConfig = field(default_factory=QuarantineConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)


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
class ExperimentConfig:
    """Root configuration for one experiment run."""

    name: str = "default"
    seed: int = 42
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fl: FlConfig = field(default_factory=FlConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate a YAML config file into an ExperimentConfig."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        """Build an ExperimentConfig from a nested dict, keeping defaults for absent keys."""
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
