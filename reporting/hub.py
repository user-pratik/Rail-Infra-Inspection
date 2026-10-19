"""
ReportingHub: the one object gui_app.py needs.

    hub = ReportingHub(on_report=callback)           # once, at app start
    tok = hub.start_session(video, model_path, "rcnn", fps, total_frames, conf, module="track")
    hub.ingest(tok, frame_idx, raw_frame_or_None, [(class_name, score, (x1,y1,x2,y2)), ...])
    hub.end_session(tok)                             # end of video or Stop (safe to call twice)
    hub.generate_digest()                            # daily digest button / CLI

The inference thread only does a non-blocking queue put. Aggregation, evidence
saving, SQLite and PDF generation all happen on one background worker thread.
If the worker falls behind, frames are dropped (counted and disclosed in the
digest) instead of stalling playback.
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2

from .aggregator import Aggregator, Detection
from .classifier import classify
from .config import load_config
from .evidence import save_event_evidence
from .store import Store
from .telemetry import DummyTelemetry

log = logging.getLogger("reporting")


@dataclass
class _Session:
    token: int
    run_id: int
    video_name: str
    model_path: str
    fps: float
    telemetry: DummyTelemetry
    agg: Aggregator
    module: str = "track"
    structure_id: str | None = None
    frames_seen: int = 0
    incidents: int = 0
    queued: int = 0
    inventory: int = 0
    hashed: bool = False


class ReportingHub:
    def __init__(self, on_report=None, config_path=None, base_dir=None):
        self.cfg = load_config(config_path)
        self.base = Path(base_dir) if base_dir else self.cfg.output_dir
        for sub in ("incidents", "daily", "evidence"):
            (self.base / sub).mkdir(parents=True, exist_ok=True)
        self.store = Store(self.base / "reports.db")
        self.on_report = on_report          # called from the worker thread with a dict

        self._q: queue.Queue = queue.Queue(maxsize=int(self.cfg.agg["queue_size"]))
        self._lock = threading.Lock()
        self._token = 0
        self._active_token = 0
        self._dropped = 0
        self._stats = {"incidents": 0, "queued": 0, "inventory": 0, "dropped": 0,
                       "last_report": "", "last_report_id": ""}
        today = self.store.events_for_date(dt.date.today().isoformat())
        self._stats["incidents"] = sum(1 for e in today if e["routing"] == "incident")
        self._stats["queued"] = sum(1 for e in today if e["routing"] == "digest" and not e.get("report_id"))
        self._cur: _Session | None = None
        self._hash_cache: dict[str, str] = {}
        self._thread = threading.Thread(target=self._run, name="reporting-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ public API
    def start_session(self, video_path, model_path, model_type, fps, total_frames,
                      conf_threshold=None, module="track", structure_id=None) -> int:
        with self._lock:
            self._token += 1
            token = self._active_token = self._token
            self._dropped = 0
        self._put(("start", token, dict(
            video_path=str(video_path), model_path=str(model_path), model_type=str(model_type),
            fps=float(fps) if fps and fps > 0 else 15.0, total_frames=int(total_frames or 0),
            conf_threshold=conf_threshold, module=module, structure_id=structure_id)))
        return token

    def ingest(self, token, frame_idx, frame, detections) -> None:
        """Called on the inference thread every frame. Never blocks."""
        if token != self._active_token:
            return
        dets_list = list(detections) if detections else []
        if frame is not None and dets_list:
            max_dim = int(self.cfg.evidence_cfg.get("max_dim", 1280))
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = float(max_dim) / float(max(h, w))
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                scaled_dets = []
                for item in dets_list:
                    c_name, c_conf, bbox = item[0], item[1], item[2]
                    s_bbox = (
                        int(round(bbox[0] * scale)),
                        int(round(bbox[1] * scale)),
                        int(round(bbox[2] * scale)),
                        int(round(bbox[3] * scale)),
                    )
                    scaled_dets.append((c_name, c_conf, s_bbox))
                dets_list = scaled_dets
        try:
            self._q.put_nowait(("frame", token, int(frame_idx), frame, dets_list))
        except queue.Full:
            with self._lock:
                self._dropped += 1
                self._stats["dropped"] += 1

    def end_session(self, token) -> None:
        """Safe to call more than once for the same token."""
        with self._lock:
            if token != self._active_token or token == 0:
                return
            self._active_token = 0
            dropped = self._dropped
        self._put(("end", token, dropped))

    def generate_digest(self, date: str | None = None, auto_open: bool = False) -> None:
        self._put(("digest", date or dt.date.today().isoformat(), auto_open))

    def compile_report(self, auto_open: bool = True) -> None:
        """Flush active detections mid-flow and open/create the latest report."""
        self._put(("compile_report", auto_open))

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def wait_idle(self) -> None:
        self._q.join()

    def shutdown(self) -> None:
        """Flush any open session, finish queued work, stop the worker."""
        self.end_session(self._active_token)
        self._put(("stop",))
        self._thread.join(timeout=30)

    def open_path(self, path) -> None:
        path = str(path)
        if not path or not os.path.exists(path):
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # noqa: S606 (Windows only)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            log.exception("open_path failed for %s", path)

    def open_reports_folder(self) -> None:
        self.open_path(self.base)

    # ------------------------------------------------------------------ internals
    def _put(self, msg) -> None:
        self._q.put(msg, timeout=10)

    def _emit(self, info: dict) -> None:
        if self.on_report:
            try:
                self.on_report(info)
            except Exception:  # never let a UI callback kill the worker
                log.exception("on_report callback failed")

    def _run(self) -> None:
        while True:
            msg = self._q.get()
            try:
                kind = msg[0]
                if kind == "stop":
                    if self._cur:
                        self._on_end(self._cur.token, 0)
                    return
                if kind == "start":
                    self._on_start(msg[1], msg[2])
                elif kind == "frame":
                    self._on_frame(*msg[1:])
                elif kind == "end":
                    self._on_end(msg[1], msg[2])
                elif kind == "digest":
                    date = msg[1]
                    auto_open = msg[2] if len(msg) > 2 else False
                    self._on_digest(date, auto_open=auto_open)
                elif kind == "compile_report":
                    auto_open = msg[1] if len(msg) > 1 else True
                    self._on_compile_report(auto_open=auto_open)
            except Exception as exc:
                log.exception("reporting worker error")
                self._emit({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                self._q.task_done()

    # -- session lifecycle ---------------------------------------------------
    def _on_start(self, token: int, p: dict) -> None:
        if self._cur:
            self._on_end(self._cur.token, 0)
        now = dt.datetime.now()
        operator = self.cfg.reports_cfg.get("operator") or getpass.getuser()
        module = p.get("module", "track")
        structure_id = p.get("structure_id")
        run_id = self.store.create_run(
            video_path=p["video_path"], video_name=os.path.basename(p["video_path"]), module=module,
            structure_id=structure_id,
            model_type=p["model_type"], model_path=p["model_path"], fps=p["fps"],
            total_frames=p["total_frames"], conf_threshold=p["conf_threshold"], operator=operator,
            started_at=now.isoformat(timespec="seconds"), run_date=now.date().isoformat())
        agg_cfg = self.cfg.aggregation_cfg(module)
        self._cur = _Session(token=token, run_id=run_id, video_name=os.path.basename(p["video_path"]),
                             model_path=p["model_path"], fps=p["fps"],
                             telemetry=DummyTelemetry(fps=p["fps"]), agg=Aggregator(agg_cfg),
                             module=module, structure_id=structure_id)

    def _on_frame(self, token, frame_idx, frame, raw_dets) -> None:
        s = self._cur
        if s is None or s.token != token:
            return
        s.frames_seen += 1
        dets = []
        for name, conf, bbox in raw_dets:
            cls = self.cfg.canonical(s.module, name)
            if cls:
                dets.append(Detection(cls, float(conf), tuple(int(v) for v in bbox)))
        for track in s.agg.update(frame_idx, dets, frame):
            self._finalize(s, track)

    def _on_end(self, token, dropped) -> None:
        s = self._cur
        if s is None or s.token != token:
            return
        for track in s.agg.flush():
            self._finalize(s, track)
        self.store.set_run_fields(s.run_id, frames_processed=s.frames_seen, dropped_frames=dropped,
                                  ended_at=dt.datetime.now().isoformat(timespec="seconds"))
        self._emit({"kind": "run_complete", "incidents": s.incidents, "queued": s.queued,
                    "inventory": s.inventory, "module": s.module})
        self._cur = None

    # -- one closed, confirmed event -----------------------------------------
    def _model_hash(self, path: str) -> str:
        if path in self._hash_cache:
            return self._hash_cache[path]
        digest = ""
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()[:16]
        except OSError:
            pass
        self._hash_cache[path] = digest
        return digest

    def _finalize(self, s: _Session, t) -> None:
        cfg = self.cfg
        area_frac = t.best_area_frac
        dec = classify(cfg, s.module, t.cls, t.best_conf, area_frac=area_frac)
        geo = s.telemetry.get(t.best_frame)
        ev = {
            "run_id": s.run_id, "module": s.module, "cls": t.cls, "severity": dec.severity,
            "routing": dec.routing, "note": dec.note, "first_frame": t.first_frame,
            "last_frame": t.last_frame, "best_frame": t.best_frame, "hits": t.hits,
            "best_conf": round(t.best_conf, 4), "mean_conf": round(t.conf_sum / t.hits, 4),
            "bbox": json.dumps(list(t.best_bbox)), "lat": geo.latitude, "lon": geo.longitude,
            "location_name": geo.location_name, "location_source": "simulated",
            "first_ts": round(t.first_frame / s.fps, 3), "last_ts": round(t.last_frame / s.fps, 3),
            "detected_at": t.created_wall.isoformat(timespec="seconds"),
            "event_date": t.created_wall.date().isoformat(),
        }
        event_id = self.store.add_event(ev)
        rel = f"evidence/{event_id}"
        save_event_evidence(self.base / rel, t, cfg, dec.severity)
        self.store.set_event_fields(event_id, evidence_dir=rel)

        if not s.hashed:
            self.store.set_run_fields(s.run_id, model_hash=self._model_hash(s.model_path))
            s.hashed = True

        if dec.routing == "incident":
            self._make_incident(s, event_id, t.cls, ev["event_date"])
        elif dec.routing == "digest":
            s.queued += 1
            with self._lock:
                self._stats["queued"] += 1
            self._emit({"kind": "queued", "event_id": event_id, "cls": t.cls, "conf": t.best_conf,
                        "needs_review": dec.severity == "critical", "module": s.module})
        else:
            s.inventory += 1
            with self._lock:
                self._stats["inventory"] += 1

    def _render_module(self):
        try:
            from . import render
            return render
        except ImportError as exc:
            self._emit({"kind": "error",
                        "message": f"PDF library missing ({exc}). Run: pip install reportlab"})
            return None

    def _make_incident(self, s: _Session, event_id: str, cls: str, date: str) -> None:
        render = self._render_module()
        if render is None:
            return
        rid = self.store.next_incident_id(date)
        rel = f"incidents/{rid}.pdf"
        history = self.store.previous_events(cls, event_id, 5, module=s.module)
        render.render_incident(
            self.base / rel, base=self.base, cfg=self.cfg, report_id=rid,
            event=self.store.get_event(event_id), run=self.store.get_run(s.run_id),
            history=history, generated_at=dt.datetime.now())
        self.store.record_report(rid, "incident", event_id, date, 1, rel)
        self.store.set_event_fields(event_id, report_id=rid)
        s.incidents += 1
        full = str(self.base / rel)
        with self._lock:
            self._stats.update(incidents=self._stats["incidents"] + 1, last_report=full, last_report_id=rid)
        ev = self.store.get_event(event_id)
        self._emit({"kind": "incident", "report_id": rid, "path": full, "cls": cls, "conf": ev["best_conf"],
                    "module": s.module})

    def _flush_midflow(self) -> None:
        s = self._cur
        if s is None:
            return
        if hasattr(s.agg, "flush_confirmed"):
            for track in s.agg.flush_confirmed():
                self._finalize(s, track)
        self.store.set_run_fields(s.run_id, frames_processed=s.frames_seen)

    # -- daily digest ----------------------------------------------------------
    def _on_digest(self, date: str, auto_open: bool = False) -> None:
        render = self._render_module()
        if render is None:
            return
        self._flush_midflow()
        events = self.store.events_for_date(date)
        runs = self.store.runs_for_date(date)
        if not events and not runs:
            self._emit({"kind": "digest_empty", "date": date})
            return
        prev_date = (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()
        rid = f"DIG-{date.replace('-', '')}"
        version = self.store.report_version(rid) + 1
        rel = f"daily/{rid}.pdf"
        target_path = self.base / rel
        try:
            render.render_digest(
                target_path, base=self.base, cfg=self.cfg, report_id=rid, date=date, events=events,
                prev_events=self.store.events_for_date(prev_date), runs=runs,
                generated_at=dt.datetime.now(), version=version)
        except PermissionError:
            rel = f"daily/{rid}_v{version}_{int(dt.datetime.now().timestamp())}.pdf"
            target_path = self.base / rel
            render.render_digest(
                target_path, base=self.base, cfg=self.cfg, report_id=rid, date=date, events=events,
                prev_events=self.store.events_for_date(prev_date), runs=runs,
                generated_at=dt.datetime.now(), version=version)

        self.store.record_report(rid, "digest", None, date, version, rel)
        self.store.assign_digest(date, rid)
        full = str(target_path)
        with self._lock:
            self._stats.update(queued=0, last_report=full, last_report_id=rid)
        self._emit({"kind": "digest", "report_id": rid, "path": full, "date": date,
                    "count": sum(1 for e in events if e["routing"] == "digest")})
        if auto_open:
            self.open_path(full)

    def _on_compile_report(self, auto_open: bool = True) -> None:
        self._flush_midflow()
        last = self._stats.get("last_report")
        if last and os.path.exists(last):
            if auto_open:
                self.open_path(last)
            self._emit({"kind": "report_opened", "path": last,
                        "report_id": self._stats.get("last_report_id", "")})
        else:
            self._on_digest(dt.date.today().isoformat(), auto_open=auto_open)
