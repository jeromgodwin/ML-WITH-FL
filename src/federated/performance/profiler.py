"""Endpoint performance profiler — measures actual bottlenecks (Enhancement 9)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import psutil


def profile_endpoint(detector, sample_file: Path) -> Dict[str, Any]:
    """Profile endpoint: startup, memory, CPU, file-event latency, feature extraction, inference, quarantine, DB."""
    # Persistent model loading is already done in AutoDetector (loaded once)
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024 / 1024
    t0 = time.perf_counter()
    result = detector.scan(sample_file)
    total_ms = (time.perf_counter() - t0) * 1000
    mem_after = proc.memory_info().rss / 1024 / 1024
    return {
        "total_scan_ms": total_ms,
        "feature_extraction_ms": result.extraction_ms if hasattr(result, "extraction_ms") else None,
        "inference_ms": result.inference_ms if hasattr(result, "inference_ms") else None,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": mem_after - mem_before,
        "cpu_percent": proc.cpu_percent(interval=0.1),
        "idle_cpu": psutil.cpu_percent(interval=0.1),
    }
