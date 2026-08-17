"""Persistent scan history (SQLite).

Stores every detection record. Queries:
- latest detections
- blocked/quarantined files (action == QUARANTINE)
- safe files (action == ALLOW)
- search/filter by verdict, action, timestamp range, model version

A new connection is opened per call, so concurrent callers (monitor thread +
dashboard) are safe without shared state.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fedshield.config import HistoryConfig
from fedshield.logging_setup import get_logger
from src.interfaces import DetectionRecord

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    detection_id          TEXT PRIMARY KEY,
    timestamp             TEXT NOT NULL,
    filename              TEXT NOT NULL,
    filepath              TEXT NOT NULL,
    sha256                TEXT NOT NULL,
    file_type             TEXT NOT NULL,
    model_version         TEXT NOT NULL,
    malware_probability   REAL NOT NULL,
    benign_probability    REAL,
    risk_score            INTEGER NOT NULL,
    risk_level            TEXT NOT NULL,
    verdict               TEXT NOT NULL,
    action                TEXT NOT NULL,
    model_algorithm       TEXT NOT NULL,
    analysis_duration_ms  REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_action     ON detections(action);
CREATE INDEX IF NOT EXISTS idx_detections_verdict    ON detections(verdict);
CREATE INDEX IF NOT EXISTS idx_detections_model      ON detections(model_version);
"""


class HistoryStore:
    def __init__(self, config: HistoryConfig | None = None, db_path: Optional[Path] = None):
        self.config = config or HistoryConfig()
        self.db_path = Path(db_path) if db_path else Path(self.config.history_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        self._prune()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    def add(self, record: DetectionRecord) -> None:
        d = record.to_dict()
        cols = list(d.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO detections ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [d[c] for c in cols])

    def latest(self, limit: int = 50) -> List[Dict]:
        return self.search(limit=limit)

    def by_action(self, action: str, limit: int = 100) -> List[Dict]:
        return self.search(action=action, limit=limit)

    def by_verdict(self, verdict: str, limit: int = 100) -> List[Dict]:
        return self.search(verdict=verdict, limit=limit)

    def quarantined(self, limit: int = 100) -> List[Dict]:
        """Blocked/quarantined files."""
        return self.by_action("QUARANTINE", limit=limit)

    def safe(self, limit: int = 100) -> List[Dict]:
        return self.by_action("ALLOW", limit=limit)

    def search(
        self,
        verdict: Optional[str] = None,
        action: Optional[str] = None,
        model_version: Optional[str] = None,
        since: Optional[str] = None,       # ISO timestamp (inclusive)
        until: Optional[str] = None,       # ISO timestamp (inclusive)
        min_risk_score: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict]:
        where, params = [], []
        if verdict:
            where.append("verdict = ?")
            params.append(verdict)
        if action:
            where.append("action = ?")
            params.append(action)
        if model_version:
            where.append("model_version = ?")
            params.append(model_version)
        if since:
            where.append("timestamp >= ?")
            params.append(since)
        if until:
            where.append("timestamp <= ?")
            params.append(until)
        if min_risk_score is not None:
            where.append("risk_score >= ?")
            params.append(min_risk_score)
        sql = "SELECT * FROM detections"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    # ------------------------------------------------------------------
    def _prune(self) -> None:
        if self.config.retention_days <= 0:
            return
        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=self.config.retention_days)).isoformat()
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
            if cur.rowcount:
                logger.info("pruned %d history rows older than %s", cur.rowcount, cutoff)