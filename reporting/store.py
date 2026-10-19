"""SQLite store: inspection_runs, defect_events, reports.

A fresh connection is opened per operation and guarded by a lock, so it is safe
to call from the reporting worker thread and from the GUI thread.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT, video_name TEXT, module TEXT DEFAULT 'track',
    structure_id TEXT,
    is_live BOOLEAN DEFAULT 0,
    source_type TEXT DEFAULT 'file',
    model_type TEXT, model_path TEXT, model_hash TEXT,
    fps REAL, total_frames INTEGER,
    frames_processed INTEGER DEFAULT 0, dropped_frames INTEGER DEFAULT 0,
    conf_threshold REAL, operator TEXT,
    started_at TEXT, ended_at TEXT, run_date TEXT
);
CREATE TABLE IF NOT EXISTS defect_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT, run_id INTEGER, module TEXT DEFAULT 'track',
    cls TEXT, severity TEXT, routing TEXT, note TEXT,
    first_frame INTEGER, last_frame INTEGER, best_frame INTEGER,
    hits INTEGER, best_conf REAL, mean_conf REAL, bbox TEXT,
    lat REAL, lon REAL, location_name TEXT, location_source TEXT,
    first_ts REAL, last_ts REAL,
    detected_at TEXT, event_date TEXT,
    evidence_dir TEXT, status TEXT DEFAULT 'Unverified', report_id TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY, kind TEXT, event_id TEXT, report_date TEXT,
    version INTEGER, path TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_date ON defect_events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_cls ON defect_events(cls);
"""

_RUN_COLS = {"video_path", "video_name", "module", "structure_id", "is_live", "source_type",
             "model_type", "model_path", "model_hash", "fps", "total_frames",
             "frames_processed", "dropped_frames", "conf_threshold", "operator",
             "started_at", "ended_at", "run_date"}
_EVENT_COLS = {"run_id", "module", "cls", "severity", "routing", "note", "first_frame", "last_frame",
               "best_frame", "hits", "best_conf", "mean_conf", "bbox", "lat", "lon",
               "location_name", "location_source", "first_ts", "last_ts", "detected_at",
               "event_date", "evidence_dir", "status", "report_id"}


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)
            self._migrate(c)

    def _migrate(self, c) -> None:
        """Run startup migrations for backward compatibility with existing databases."""
        cols_events = {row[1] for row in c.execute("PRAGMA table_info(defect_events)").fetchall()}
        if "module" not in cols_events:
            c.execute("ALTER TABLE defect_events ADD COLUMN module TEXT DEFAULT 'track'")
            c.execute("UPDATE defect_events SET module='track' WHERE module IS NULL")

        cols_runs = {row[1] for row in c.execute("PRAGMA table_info(inspection_runs)").fetchall()}
        if "structure_id" not in cols_runs:
            c.execute("ALTER TABLE inspection_runs ADD COLUMN structure_id TEXT")
        if "module" not in cols_runs:
            c.execute("ALTER TABLE inspection_runs ADD COLUMN module TEXT DEFAULT 'track'")
        if "is_live" not in cols_runs:
            c.execute("ALTER TABLE inspection_runs ADD COLUMN is_live BOOLEAN DEFAULT 0")
        if "source_type" not in cols_runs:
            c.execute("ALTER TABLE inspection_runs ADD COLUMN source_type TEXT DEFAULT 'file'")

        c.execute("CREATE INDEX IF NOT EXISTS idx_events_mod_cls ON defect_events(module, cls);")

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=15)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _insert(conn, table, allowed, values: dict) -> int:
        keys = [k for k in values if k in allowed]
        sql = f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})"
        return conn.execute(sql, [values[k] for k in keys]).lastrowid

    @staticmethod
    def _update(conn, table, allowed, key_col, key, values: dict) -> None:
        keys = [k for k in values if k in allowed]
        if not keys:
            return
        sql = f"UPDATE {table} SET {','.join(k + '=?' for k in keys)} WHERE {key_col}=?"
        conn.execute(sql, [values[k] for k in keys] + [key])

    # -- runs ---------------------------------------------------------------
    def create_run(self, **values) -> int:
        with self._conn() as c:
            return self._insert(c, "inspection_runs", _RUN_COLS, values)

    def set_run_fields(self, run_id: int, **values) -> None:
        with self._conn() as c:
            self._update(c, "inspection_runs", _RUN_COLS, "id", run_id, values)

    def get_run(self, run_id: int) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM inspection_runs WHERE id=?", (run_id,)).fetchone()
            return dict(r) if r else {}

    def runs_for_date(self, date: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM inspection_runs WHERE run_date=? ORDER BY id", (date,)).fetchall()
            return [dict(r) for r in rows]

    # -- events -------------------------------------------------------------
    def add_event(self, values: dict) -> str:
        vals = dict(values)
        if "module" not in vals:
            vals["module"] = "track"
        with self._conn() as c:
            row_id = self._insert(c, "defect_events", _EVENT_COLS, vals)
            event_id = f"EVT-{vals['event_date'].replace('-', '')}-{row_id:04d}"
            c.execute("UPDATE defect_events SET event_id=? WHERE id=?", (event_id, row_id))
            return event_id

    def set_event_fields(self, event_id: str, **values) -> None:
        with self._conn() as c:
            self._update(c, "defect_events", _EVENT_COLS, "event_id", event_id, values)

    def get_event(self, event_id: str) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM defect_events WHERE event_id=?", (event_id,)).fetchone()
            return dict(r) if r else {}

    def events_for_date(self, date: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM defect_events WHERE event_date=? ORDER BY detected_at, id", (date,)
            ).fetchall()
            return [dict(r) for r in rows]

    def previous_events(self, cls: str, exclude_event_id: str, limit: int = 5, module: str = "track") -> list[dict]:
        with self._conn() as c:
            if module:
                rows = c.execute(
                    "SELECT * FROM defect_events WHERE cls=? AND module=? AND event_id!=? ORDER BY id DESC LIMIT ?",
                    (cls, module, exclude_event_id, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM defect_events WHERE cls=? AND event_id!=? ORDER BY id DESC LIMIT ?",
                    (cls, exclude_event_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    def assign_digest(self, date: str, report_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE defect_events SET report_id=? WHERE event_date=? AND routing='digest'",
                (report_id, date),
            )

    # -- reports ------------------------------------------------------------
    def next_incident_id(self, date: str) -> str:
        with self._conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM reports WHERE kind='incident' AND report_date=?", (date,)
            ).fetchone()[0]
        return f"INC-{date.replace('-', '')}-{n + 1:04d}"

    def report_version(self, report_id: str) -> int:
        with self._conn() as c:
            r = c.execute("SELECT version FROM reports WHERE report_id=?", (report_id,)).fetchone()
            return int(r[0]) if r else 0

    def record_report(self, report_id, kind, event_id, report_date, version, path) -> None:
        import datetime as dt
        with self._conn() as c:
            c.execute(
                "INSERT INTO reports (report_id, kind, event_id, report_date, version, path, created_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(report_id) DO UPDATE SET "
                "version=excluded.version, path=excluded.path, created_at=excluded.created_at",
                (report_id, kind, event_id, report_date, version, path,
                 dt.datetime.now().isoformat(timespec="seconds")),
            )
