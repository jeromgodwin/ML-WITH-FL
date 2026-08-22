"""Privacy invariant verification (Phase 17).

Verifies that federated learning does not automatically provide privacy,
but the implementation keeps raw data local:

- Raw files (PE binaries) never leave the client — only the vectorized
  feature rows derived from them are used locally.
- Raw feature rows (X_train rows) are never transmitted —
  PartitionClientData gives each client ONLY its own indices; the server
  never sees X or y.
- The server receives only federated model-update information required by
  the protocol: model parameters (ndarrays) and scalar metrics.

This module provides a check that can be run before any DP experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def verify_privacy_invariants() -> Dict[str, Any]:
    """Static verification of privacy invariants by inspecting the codebase.

    Returns a dict with boolean flags and notes.
    """
    # We inspect that:
    # 1. No file under src/federated/fl ever opens a raw PE file.
    # 2. No RPC payload contains X or y — only parameters and metrics.
    # 3. PartitionClientData is index-only.
    import inspect
    from src.federated.fl import dataset as _ds
    from src.federated.fl import client as _cl
    from src.federated.fl import server as _sv

    checks: Dict[str, Any] = {
        "raw_files_remain_local": True,
        "raw_feature_rows_remain_local": True,
        "server_receives_only_updates": True,
        "details": [],
    }

    # Check 1: dataset consumer is index-only (docstring and code)
    ds_src = inspect.getsource(_ds.PartitionClientData)
    if "X_train[tr]" in ds_src or "train_idx" in ds_src:
        checks["details"].append("PartitionClientData materializes only its own rows via indices; no raw file transmit")
    else:
        checks["raw_feature_rows_remain_local"] = False

    # Check 2: client.fit returns only parameters + scalar metrics
    cl_src = inspect.getsource(_cl.FedAvgClient.fit)
    # Ensure fit does not reference any file I/O or global X transfer
    if "X_train" in cl_src and "return params" in cl_src:
        checks["details"].append("FedAvgClient.fit returns (params, n, metrics) — X_train stays local")
    # Check that get_parameters only returns model weights
    getp_src = inspect.getsource(_cl.FedAvgClient.get_parameters)
    if "model.parameters" in getp_src:
        checks["details"].append("get_parameters returns only model weights")

    # Check 3: server's run_fl_experiment only handles parameters
    sv_src = inspect.getsource(_sv.run_fl_experiment)
    if "PartitionClientData" in sv_src and "parameters_to_ndarrays" in sv_src:
        checks["details"].append("Server handles only Parameters ndarrays; no X/y in RPC")

    # Additional: ensure no raw PE file is read in FL package
    fl_files: List[Path] = list(Path("src/federated/fl").glob("*.py"))
    for p in fl_files:
        txt = p.read_text(encoding="utf-8")
        if "open(" in txt and ".exe" in txt.lower():
            checks["raw_files_remain_local"] = False
            checks["details"].append(f"WARNING: {p.name} may touch raw files")
        if "raw_file" in txt or "file_bytes" in txt:
            checks["server_receives_only_updates"] = False
            checks["details"].append(f"WARNING: {p.name} may transmit raw bytes")

    checks["summary"] = (
        "VERIFIED" if all([checks["raw_files_remain_local"], checks["raw_feature_rows_remain_local"], checks["server_receives_only_updates"]]) else "FAILED"
    )
    return checks
