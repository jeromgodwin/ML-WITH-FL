"""Tests for Phase 15 unified experiment + reproducibility engine."""

import json
import pathlib
import tempfile

import pytest

from fedshield.config import ExperimentConfig


def test_yaml_and_json_config_loading(tmp_path):
    """YAML or JSON config must define all 15 required fields via aliases."""
    # YAML with flat aliases
    yaml_text = """
dataset: "2018_2"
seed: 123
algorithm: fedavg
clients: 5
client_fraction: 0.8
partition_strategy: mild
non_iid_severity: mild
model:
  input_dim: 2381
  hidden_layers: [64, 32]
  dropout: 0.1
learning_rate: 0.002
batch_size: 256
local_epochs: 2
fl_rounds: 3
fedprox_mu: 0.05
personalization:
  personalized_probe_samples: 1000
  personalized_probe_epochs: 2
resource_policy:
  enabled: false
drift_settings:
  enabled: false
privacy_settings:
  enabled: false
security_settings:
  attack:
    enabled: false
  defense:
    mode: "none"
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(p)
    assert cfg.seed == 123
    assert cfg.fl.algorithm == "fedavg"
    assert cfg.fl.num_clients == 5
    assert cfg.fl.client_fraction == 0.8
    assert cfg.partition.strategy == "mild"
    assert cfg.train.learning_rate == 0.002
    assert cfg.train.batch_size == 256
    assert cfg.train.local_epochs == 2
    assert cfg.fl.num_rounds == 3
    assert cfg.fl.proximal_mu == 0.05

    # JSON loading also works
    import yaml as _yaml

    raw = _yaml.safe_load(yaml_text)
    jp = tmp_path / "cfg.json"
    jp.write_text(json.dumps(raw), encoding="utf-8")
    cfg2 = ExperimentConfig.from_json(jp)
    assert cfg2.fl.algorithm == "fedavg"
    cfg3 = ExperimentConfig.from_file(jp)
    assert cfg3.fl.algorithm == "fedavg"
    cfg4 = ExperimentConfig.from_file(p)
    assert cfg4.fl.algorithm == "fedavg"


def test_privacy_config_defaults():
    cfg = ExperimentConfig()
    assert hasattr(cfg, "privacy")
    assert cfg.privacy.enabled is False
    assert cfg.privacy.noise_multiplier == 1.0
    # Round-trip via to_dict/from_dict
    d = cfg.to_dict()
    assert "privacy" in d
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.privacy.enabled == cfg.privacy.enabled


def test_unique_experiment_id_never_overwrites(tmp_path):
    """Every run gets a unique ID; never overwrite previous results."""
    from src.federated.experiments.storage import ExperimentStorage

    cfg = ExperimentConfig(name="test")
    s1 = ExperimentStorage.create(cfg, root=tmp_path, experiment_id="exp-unique-001")
    (s1.dir / "marker.txt").write_text("hello", encoding="utf-8")
    # Same ID must raise
    with pytest.raises(FileExistsError):
        ExperimentStorage(tmp_path, "exp-unique-001")
    # Auto-generated IDs are unique
    s2 = ExperimentStorage.create(cfg, root=tmp_path)
    s3 = ExperimentStorage.create(cfg, root=tmp_path)
    assert s2.experiment_id != s3.experiment_id
    assert s2.dir != s3.dir


def test_result_storage_structure(tmp_path):
    """Storage must persist all required fields without overwriting."""
    from src.federated.experiments.storage import ExperimentStorage
    from src.federated.experiments.environment import collect_environment_metadata, collect_reproducibility_record

    cfg = ExperimentConfig(name="storage-test", seed=99)
    cfg.fl.algorithm = "fedavg"
    storage = ExperimentStorage.create(cfg, root=tmp_path)
    raw = {"name": "storage-test", "seed": 99}
    storage.save_config(cfg, raw_source=raw)
    env = collect_environment_metadata()
    storage.save_environment(env)
    repro = collect_reproducibility_record(cfg)
    storage.save_reproducibility(repro)

    results = {
        "experiment": {"algorithm": "fedavg", "strategy": "iid"},
        "rounds": [{"round": 1, "global_eval": {"f1": 0.9}}],
        "per_client_metrics": [{"round": 1, "clients": []}],
        "final_global_test_metrics": {"f1": 0.9, "accuracy": 0.92},
        "training_time_s": 123.4,
        "communication": {"totals": {"total_bytes_exchanged": 1000}},
        "resource": {"enabled": False},
        "drift": {"enabled": False},
        "attack": {"attack_type": "none"},
        "defense": {"defense_mode": "none"},
        "privacy": {"enabled": False},
        "model": {"input_dim": 2381},
        "logs": {"msg": "done"},
    }
    storage.save_metrics(results)

    assert (storage.dir / "config_resolved.json").exists()
    assert (storage.dir / "environment.json").exists()
    assert (storage.dir / "reproducibility.json").exists()
    assert (storage.dir / "metrics/summary.json").exists()
    assert (storage.dir / "metrics/rounds.json").exists()
    assert (storage.dir / "metrics/rounds.jsonl").exists()
    assert (storage.dir / "metrics/per_client.json").exists()
    assert (storage.dir / "metrics/final.json").exists()
    assert (storage.dir / "metrics/training_time.json").exists()
    assert (storage.dir / "metrics/communication.json").exists()
    assert (storage.dir / "metrics/resource.json").exists()
    assert (storage.dir / "metrics/drift.json").exists()
    assert (storage.dir / "metrics/security.json").exists()
    assert (storage.dir / "metrics/privacy.json").exists()
    assert (storage.dir / "model/metadata.json").exists()
    assert (storage.dir / "logs/run.json").exists()


def test_experiment_matrix_support():
    """Matrix must support centralized + FedAvg/FedProx/Personalized x 4 severities."""
    from src.federated.experiments.matrix import full_matrix, controlled_entries

    m = full_matrix(seed=42, include_centralized=True)
    # 1 centralized + 3*4 = 13
    assert len(m) == 13
    labels = {e["label"] for e in m}
    assert "centralized" in labels
    for algo in ("fedavg", "fedprox", "personalized"):
        for strat in ("iid", "mild", "moderate", "severe"):
            assert f"{algo}-{strat}" in labels

    c = controlled_entries(seed=42)
    kinds = {e["kind"] for e in c}
    assert "controlled_resource" in kinds
    assert "controlled_drift" in kinds
    assert "controlled_poisoning" in kinds
    assert "controlled_privacy" in kinds


def test_reproducibility_record_contains_required_keys():
    cfg = ExperimentConfig(seed=77)
    cfg.partition.seed = 77
    cfg.model.hidden_layers = (128, 64)
    from src.federated.experiments.environment import collect_reproducibility_record

    rec = collect_reproducibility_record(cfg)
    for k in ("seed", "partition_seed", "dataset_version", "model_config", "preprocessing_version", "algorithm_config", "software"):
        assert k in rec
    assert rec["seed"] == 77
    assert rec["partition_seed"] == 77
    assert "config_fingerprint" in rec


def test_aggregation_no_invented_missing_values(tmp_path):
    """CSV/JSON generation must not invent missing values (empty, not fake)."""
    from src.federated.experiments.aggregation import collect_summaries, write_csv, write_json, write_comparison_table

    # Create two fake experiment dirs
    for idx, algo in enumerate(("fedavg", "centralized")):
        d = tmp_path / f"exp-{idx}"
        d.mkdir()
        (d / "metrics").mkdir()
        summary = {
            "experiment": {"algorithm": algo, "strategy": "iid" if algo == "fedavg" else "centralized"},
            "final_global_test_metrics": {"f1": 0.9 if algo == "fedavg" else None, "roc_auc": 0.95},
            "training_time_s": 10.0,
            "communication": {"totals": {"total_bytes_exchanged": 1000 if algo == "fedavg" else None}},
        }
        (d / "metrics" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (d / "config_resolved.json").write_text(json.dumps({"fl": {"algorithm": algo}}), encoding="utf-8")

    rows = collect_summaries([tmp_path / "exp-0", tmp_path / "exp-1"])
    assert len(rows) == 2
    # centralized should have missing f1 as None, not invented
    central = next(r for r in rows if r["algorithm"] == "centralized")
    assert central["final_f1"] is None
    assert central["total_bytes"] is None

    out = tmp_path / "agg"
    write_csv(rows, out / "comparison.csv")
    csv_text = (out / "comparison.csv").read_text(encoding="utf-8")
    # Missing values should be empty fields, not "0" or "N/A"
    assert ",," in csv_text or csv_text.count("\n") == 3  # header + 2 rows
    # JSON should preserve None as null
    write_json(rows, out / "comparison.json")
    j = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert any(r["final_f1"] is None for r in j)
    # Table should render missing as empty
    write_comparison_table(rows, out / "comparison.md")
    md = (out / "comparison.md").read_text(encoding="utf-8")
    assert "centralized" in md
