"""Structured audit logging — endpoint, federated, security (Enhancement 19)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


class AuditLogger:
    def __init__(self, log_path: Path | str = "logs/audit.jsonl"):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, category: str, event: str, **fields: Any) -> None:
        entry = {"timestamp": time.time(), "category": category, "event": event, **fields}
        # Avoid sensitive raw content
        entry.pop("raw_file", None)
        entry.pop("file_bytes", None)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def search(self, category: str = None, limit: int = 100) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-1000:]:
            try:
                e = json.loads(line)
                if category and e.get("category") != category:
                    continue
                out.append(e)
            except Exception:
                continue
        return out[-limit:]
