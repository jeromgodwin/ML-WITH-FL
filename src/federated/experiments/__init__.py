"""Unified experiment engine (Phase 15). Exports for external use."""

from src.federated.experiments.engine import run_unified_experiment
from src.federated.experiments.storage import ExperimentStorage, DEFAULT_EXPERIMENTS_ROOT
from src.federated.experiments.matrix import full_matrix, controlled_entries, matrix_from_config
from src.federated.experiments.aggregation import collect_summaries, write_csv, write_json, write_comparison_table, generate_plots
from src.federated.experiments.environment import collect_environment_metadata, collect_reproducibility_record

__all__ = [
    "run_unified_experiment",
    "ExperimentStorage",
    "DEFAULT_EXPERIMENTS_ROOT",
    "full_matrix",
    "controlled_entries",
    "matrix_from_config",
    "collect_summaries",
    "write_csv",
    "write_json",
    "write_comparison_table",
    "generate_plots",
    "collect_environment_metadata",
    "collect_reproducibility_record",
]
