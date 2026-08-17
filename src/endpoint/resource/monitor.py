"""Resource metrics for resource-aware FL training (Phase 12).

The monitor is a SEPARATE module from training: real-time protection never
consults it, and training only samples it between epochs. It reports:
CPU utilization, RAM (used/total/free/percent), battery level + AC state,
and user idle time (Windows: GetLastInputInfo).

Graceful degradation contract
-----------------------------
No metric is assumed to exist. Every snapshot field carries an
``*_available`` flag; when a metric cannot be measured (no psutil, no
battery on the machine, non-Windows idle detection, provider failure) the
value is None and the flag is False. Policy evaluation skips constraints
whose metric is unavailable, so a policy works identically on machines that
cannot measure a given signal.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, Optional

from fedshield.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # psutil is optional; metrics degrade gracefully
    psutil = None  # type: ignore[assignment]


def _idle_seconds_windows() -> Optional[float]:
    """Seconds since the last user input on Windows (None elsewhere).

    Uses GetLastInputInfo: the system tick at the last keyboard/mouse event.
    Returns None when the API is unavailable (non-Windows, ctypes failure),
    which callers treat as an unsupported metric.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        tick = int(info.dwTime)
        now = int(ctypes.windll.kernel32.GetTickCount())
        # dwTime wraps at 2^32 ms; unwrap the difference (signed 32-bit)
        diff = (now - tick) & 0xFFFFFFFF
        if diff > 0x7FFFFFFF:
            diff -= 0x100000000
        return max(0.0, diff / 1000.0)
    except Exception as exc:  # noqa: BLE001 - unsupported metric must not crash
        logger.debug("idle detection unavailable: %s", exc)
        return None


def _default_providers() -> Dict[str, Callable[[], Any]]:
    """psutil-backed providers; every callable returns None on failure."""

    def cpu() -> Optional[float]:
        if psutil is None:
            return None
        return float(psutil.cpu_percent(interval=0.2))

    def ram() -> Optional[tuple]:
        if psutil is None:
            return None
        m = psutil.virtual_memory()
        return (float(m.used), float(m.total), float(m.free))

    def battery() -> Optional[tuple]:
        if psutil is None:
            return None
        b = psutil.sensors_battery()
        if b is None:
            return None
        plugged = b.power_plugged if b.power_plugged is not None else None
        return (float(b.percent), plugged)

    return {
        "cpu": cpu,
        "ram": ram,
        "battery": battery,
        "idle": _idle_seconds_windows,
    }


class ResourceMonitor:
    """Samples host resource metrics with graceful unsupported handling.

    ``providers`` maps metric name -> callable returning the raw value
    (or None on failure); tests inject deterministic providers. With no
    providers the default psutil/OS implementations are used.
    """

    def __init__(self, providers: Optional[Dict[str, Callable[[], Any]]] = None):
        self._providers = providers or _default_providers()

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """JSON-serializable resource snapshot with availability flags."""
        out: Dict[str, Any] = {
            "timestamp": round(time.time(), 3),
            "cpu_percent": None,
            "cpu_available": False,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "ram_free_mb": None,
            "ram_percent": None,
            "ram_available": False,
            "battery_percent": None,
            "ac_powered": None,
            "battery_available": False,
            "idle_seconds": None,
            "user_active": None,
            "activity_available": False,
        }
        self._sample_cpu(out)
        self._sample_ram(out)
        self._sample_battery(out)
        self._sample_activity(out)
        return out

    # ------------------------------------------------------------------
    def _sample(self, key: str) -> Any:
        provider = self._providers.get(key)
        if provider is None:
            return None
        try:
            return provider()
        except Exception as exc:  # noqa: BLE001 - providers must not crash sampling
            logger.debug("resource metric %r unavailable: %s", key, exc)
            return None

    def _sample_cpu(self, out: Dict[str, Any]) -> None:
        value = self._sample("cpu")
        if value is None:
            return
        out["cpu_percent"] = round(float(value), 1)
        out["cpu_available"] = True

    def _sample_ram(self, out: Dict[str, Any]) -> None:
        value = self._sample("ram")
        if value is None:
            return
        used, total, free = value
        if total <= 0:
            return
        out["ram_used_mb"] = round(used / 1048576.0, 1)
        out["ram_total_mb"] = round(total / 1048576.0, 1)
        out["ram_free_mb"] = round(free / 1048576.0, 1)
        out["ram_percent"] = round(100.0 * used / total, 1)
        out["ram_available"] = True

    def _sample_battery(self, out: Dict[str, Any]) -> None:
        value = self._sample("battery")
        if value is None:
            return
        percent, plugged = value
        out["battery_percent"] = round(float(percent), 1)
        out["ac_powered"] = plugged  # None stays None: state unknown
        out["battery_available"] = True

    def _sample_activity(self, out: Dict[str, Any]) -> None:
        value = self._sample("idle")
        if value is None:
            return
        out["idle_seconds"] = round(float(value), 1)
        out["activity_available"] = True