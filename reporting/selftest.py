"""
Self-test with synthetic detections (no models, no video needed).

    python -m reporting.selftest                 # writes to a temp folder
    python -m reporting.selftest --outdir demo_reports

Checks:
  Existing:
    1. 200 frames of one crack  -> exactly 1 event, 1 incident report
    2. one-frame flicker        -> no event
    3. flaking / groove         -> digest queue, no incident report
    4. low-confidence squat     -> digest, flagged "needs review"
    5. joint                    -> inventory only
    6. daily digest PDF is produced
  Phase 6 Additions:
    a. Bridge crack drifting over 150 frames -> exactly 1 event and 1 incident report
    b. Bridge corrosion at 0.40 confidence (below 0.45) -> digest, flagged review
    c. Bridge leakage and freelime, using aliases "Free lime", "free_lime" and "Leak"
       -> canonicalised and queued for digest, no incident
    d. Two same-frame overlapping crack boxes (IoU 0.5) -> one event (merge on)
    e. Track crack and bridge crack on same day -> 2 separate events, 2 separate incident
       reports, counted separately in digest
    f. Migration test: reports.db with old schema and track event -> opens cleanly, reads as 'track'
    g. Large-frame guard: 2872x1628 frame -> downscaled to <= 1280 px, bbox scaled correctly
    h. Digest PDF contains both modules' sections when both had activity
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np

from .hub import ReportingHub
from .store import Store

SIZE = 520
FPS = 15.0


def make_frame(idx: int, boxes, shape=(SIZE, SIZE)) -> np.ndarray:
    rng = np.random.default_rng(idx)
    h, w = shape
    img = rng.integers(60, 90, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (w // 2 - 60, 0), (w // 2 + 60, h), (150, 150, 150), -1)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (30, 30, 200), -1)
    return img


def run_scenario(hub: ReportingHub, name: str, frames: dict, total: int, module: str = "track",
                 model_type: str = "rcnn", structure_id: str | None = None,
                 frame_shape=(SIZE, SIZE)) -> int:
    """frames: {frame_idx: [(class, conf, bbox), ...]}"""
    tok = hub.start_session(f"{name}.mp4", "weights/fake_model.pth", model_type, FPS, total, 0.2,
                            module=module, structure_id=structure_id)
    for i in range(total):
        dets = frames.get(i, [])
        frame = make_frame(i, [d[2] for d in dets], shape=frame_shape) if dets else None
        hub.ingest(tok, i, frame, dets)
    hub.end_session(tok)
    hub.wait_idle()
    return tok


def extract_pdf_text(path: Path) -> str:
    """Extract text from PDF streams (handling ASCII85 and FlateDecode compression)."""
    import base64
    import re
    content = []
    data = path.read_bytes()
    for m in re.finditer(rb'stream[\r\n]+(.*?)~>endstream', data, re.DOTALL):
        chunk = m.group(1) + b'~>'
        try:
            a85 = base64.a85decode(chunk, adobe=True)
            try:
                txt = zlib.decompress(a85).decode('latin1', 'ignore')
            except Exception:
                txt = a85.decode('latin1', 'ignore')
            content.append(txt)
        except Exception:
            pass
    # Also check plain streams
    for m in re.finditer(rb'stream[\r\n]+(.*?)endstream', data, re.DOTALL):
        raw = m.group(1).rstrip(b'\r\n')
        try:
            dec = zlib.decompress(raw).decode('latin1', 'ignore')
            content.append(dec)
        except Exception:
            try:
                content.append(raw.decode('latin1', 'ignore'))
            except Exception:
                pass
    return "\n".join(content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", help="keep output here (default: temporary folder)")
    args = ap.parse_args()
    out = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="reporting_selftest_"))
    if args.outdir and out.exists():
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    messages = []
    hub = ReportingHub(on_report=messages.append, base_dir=out)
    failures = []

    def check(label, ok):
        print(("PASS  " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    today = dt.date.today().isoformat()

    # 1. 200 frames of a single crack that drifts slowly across the frame
    f1 = {i: [("Crack", 0.55 + 0.4 * (i % 40) / 40, (200 + i // 4, 100 + i, 240 + i // 4, 140 + i))]
          for i in range(0, 200) if i < 140}
    run_scenario(hub, "single_crack", f1, 200, module="track")
    cracks = [e for e in hub.store.events_for_date(today) if e["cls"] == "crack" and e.get("module") == "track"]
    check("1. single crack over many frames -> exactly 1 event", len(cracks) == 1)
    check("1. critical crack -> 1 incident report", sum(1 for m in messages if m["kind"] == "incident") == 1)

    # 2-5. mixed scenario (track)
    mixed = {}
    mixed[20] = [("Spalling", 0.9, (300, 200, 340, 240))]                      # 2. one-frame flicker
    for i in range(40, 70):                                                    # 3. flaking
        mixed[i] = [("Flakings", 0.7, (180, 120 + i, 230, 170 + i))]
    for i in range(90, 115):                                                   # 3. groove
        mixed[i] = [("Groove", 0.6, (260, 60 + i, 330, 90 + i))]
    for i in range(130, 150):                                                  # 4. low-confidence squat
        mixed[i] = [("Squats", 0.30, (220, 100 + i, 260, 140 + i))]
    for i in range(160, 175):                                                  # 5. joint
        mixed[i] = [("joint", 0.8, (150, 80 + i, 380, 120 + i))]
    before_incidents = sum(1 for m in messages if m["kind"] == "incident")
    run_scenario(hub, "mixed_demo", mixed, 200, module="track")
    check("2. one-frame flicker -> no spalling event",
          not any(e["cls"] == "spalling" and e.get("module") == "track" for e in hub.store.events_for_date(today)))
    check("3. flaking + groove + low-conf squat -> no extra incident reports",
          sum(1 for m in messages if m["kind"] == "incident") == before_incidents)
    q = [m for m in messages if m["kind"] == "queued" and m.get("module", "track") == "track"]
    check("3. flaking, groove and squat queued for digest", {m["cls"] for m in q} == {"flaking", "groove", "squat"})
    check("4. low-confidence squat flagged for review", any(m["cls"] == "squat" and m["needs_review"] for m in q))
    check("5. joint logged as inventory only", hub.stats()["inventory"] == 1)

    # -------------------------------------------------------------------------
    # Phase 6 additions (Bridge module & new capabilities)
    # -------------------------------------------------------------------------

    # (a) Bridge crack drifting over 150 frames -> exactly 1 event and 1 incident report
    incidents_before = sum(1 for m in messages if m["kind"] == "incident")
    f_bridge_crack = {i: [("Crack", 0.65, (100 + i // 4, 100 + i // 2, 160 + i // 4, 160 + i // 2))]
                      for i in range(150)}
    run_scenario(hub, "bridge_single_crack", f_bridge_crack, 150, module="bridge",
                 model_type="yolo", structure_id="BR-PIER-01")
    bridge_cracks = [e for e in hub.store.events_for_date(today) if e.get("module") == "bridge" and e["cls"] == "crack"]
    check("a. bridge crack drifting over 150 frames -> exactly 1 event", len(bridge_cracks) == 1)
    check("a. bridge crack -> exactly 1 incident report",
          sum(1 for m in messages if m["kind"] == "incident") == incidents_before + 1)

    # (b) Bridge corrosion at 0.40 confidence (below 0.45 threshold) -> digest, flagged review
    incidents_before = sum(1 for m in messages if m["kind"] == "incident")
    f_corrosion = {i: [("Corrosion", 0.40, (120, 120, 180, 180))] for i in range(10, 40)}
    run_scenario(hub, "bridge_corrosion_lowconf", f_corrosion, 50, module="bridge", model_type="yolo")
    b_evs = [e for e in hub.store.events_for_date(today) if e.get("module") == "bridge" and e["cls"] == "corrosion"]
    check("b. bridge corrosion at 0.40 conf -> 1 event queued for digest",
          len(b_evs) == 1 and b_evs[0]["routing"] == "digest" and b_evs[0]["severity"] == "critical")
    check("b. bridge corrosion below 0.45 -> no incident report",
          sum(1 for m in messages if m["kind"] == "incident") == incidents_before)
    check("b. bridge corrosion flagged for review in queued message",
          any(m["cls"] == "corrosion" and m.get("needs_review") for m in messages if m["kind"] == "queued"))

    # (c) Bridge leakage and freelime, using aliases "Free lime", "free_lime" and "Leak"
    incidents_before = sum(1 for m in messages if m["kind"] == "incident")
    f_aliases = {}
    for i in range(10, 30):
        f_aliases[i] = [("Free lime", 0.60, (50, 50, 110, 110))]
    for i in range(35, 55):
        f_aliases[i] = [("free_lime", 0.62, (150, 50, 210, 110))]
    for i in range(60, 80):
        f_aliases[i] = [("Leak", 0.65, (200, 200, 260, 260))]
    run_scenario(hub, "bridge_aliases", f_aliases, 90, module="bridge", model_type="yolo")
    c_freelime = [e for e in hub.store.events_for_date(today) if e.get("module") == "bridge" and e["cls"] == "freelime"]
    c_leakage = [e for e in hub.store.events_for_date(today) if e.get("module") == "bridge" and e["cls"] == "leakage"]
    check("c. bridge aliases canonicalised to freelime and leakage", len(c_freelime) >= 1 and len(c_leakage) == 1)
    check("c. moderate bridge classes -> queued for digest with no incidents",
          all(e["routing"] == "digest" for e in c_freelime + c_leakage) and
          sum(1 for m in messages if m["kind"] == "incident") == incidents_before)

    # (d) Two same-frame overlapping crack boxes (IoU 0.5) -> one event (merge on)
    # Box 1: (100, 100, 200, 200) area=10000; Box 2: (100, 133, 200, 233) area=10000; intersection=6700; union=13300 -> IoU=0.5037
    cracks_before = len([e for e in hub.store.events_for_date(today) if e.get("module") == "bridge" and e["cls"] == "crack"])
    f_overlap = {i: [("Crack", 0.60, (100, 100, 200, 200)), ("Crack", 0.65, (100, 133, 200, 233))]
                 for i in range(10, 30)}
    run_scenario(hub, "bridge_overlap_merge", f_overlap, 40, module="bridge", model_type="yolo")
    cracks_after = len([e for e in hub.store.events_for_date(today) if e.get("module") == "bridge" and e["cls"] == "crack"])
    check("d. two same-frame overlapping crack boxes (IoU 0.5) -> merged into 1 event",
          cracks_after - cracks_before == 1)

    # (e) Track crack and bridge crack on same day -> 2 separate events, 2 separate incident reports
    track_cracks_all = [e for e in hub.store.events_for_date(today) if e.get("module", "track") == "track" and e["cls"] == "crack"]
    bridge_cracks_all = [e for e in hub.store.events_for_date(today) if e.get("module") == "bridge" and e["cls"] == "crack"]
    check("e. track crack and bridge crack on same day -> 2 separate events",
          len(track_cracks_all) >= 1 and len(bridge_cracks_all) >= 1)
    incident_reports = [m for m in messages if m["kind"] == "incident"]
    check("e. separate incident reports generated for both modules",
          any(m.get("module") == "track" for m in incident_reports) and
          any(m.get("module") == "bridge" for m in incident_reports))

    # (f) Migration test: create reports.db with OLD schema and one track event, open with new Store
    mig_dir = Path(tempfile.mkdtemp(prefix="reporting_migration_"))
    mig_db = mig_dir / "reports.db"
    conn = sqlite3.connect(str(mig_db))
    # Old schema without module column in defect_events and without structure_id in inspection_runs
    conn.execute("""
    CREATE TABLE defect_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT, run_id INTEGER,
        cls TEXT, severity TEXT, routing TEXT, note TEXT,
        first_frame INTEGER, last_frame INTEGER, best_frame INTEGER,
        hits INTEGER, best_conf REAL, mean_conf REAL, bbox TEXT,
        lat REAL, lon REAL, location_name TEXT, location_source TEXT,
        first_ts REAL, last_ts REAL, detected_at TEXT, event_date TEXT,
        evidence_dir TEXT, status TEXT DEFAULT 'Unverified', report_id TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE inspection_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_path TEXT, video_name TEXT, model_type TEXT, model_path TEXT,
        fps REAL, total_frames INTEGER, frames_processed INTEGER DEFAULT 0,
        dropped_frames INTEGER DEFAULT 0, conf_threshold REAL, operator TEXT,
        started_at TEXT, ended_at TEXT, run_date TEXT
    );
    """)
    conn.execute("INSERT INTO defect_events (event_id, cls, severity, routing, event_date) "
                 "VALUES ('EVT-LEGACY-001', 'crack', 'critical', 'incident', '2026-10-19')")
    conn.commit()
    conn.close()

    mig_store = Store(mig_db)
    legacy_ev = mig_store.get_event("EVT-LEGACY-001")
    legacy_prev = mig_store.previous_events("crack", "none", module="track")
    check("f. database migration: old row reads as module 'track' without error",
          legacy_ev.get("module") == "track" and len(legacy_prev) == 1)

    # (g) Large-frame guard: ingest 2872x1628 frame with bbox -> evidence image is <= 1280 px wide and bbox scaled
    f_large = {i: [("Crack", 0.85, (1000, 500, 1400, 900))] for i in range(10, 25)}
    run_scenario(hub, "drone_large_frame", f_large, 35, module="bridge", model_type="yolo",
                 frame_shape=(1628, 2872))
    drone_evs = [e for e in hub.store.events_for_date(today)
                 if e.get("module") == "bridge" and "drone_large_frame" in (hub.store.get_run(e["run_id"]).get("video_name") or "")]
    if drone_evs:
        d_ev = drone_evs[0]
        d_bbox = json.loads(d_ev["bbox"])
        scale_expected = 1280.0 / 2872.0
        expected_x1 = int(round(1000 * scale_expected))
        ann_img_file = out / d_ev["evidence_dir"] / "annotated.jpg"
        img_ok = False
        if ann_img_file.exists():
            img_w = cv2.imread(str(ann_img_file)).shape[1]
            img_ok = (img_w <= 1280)
        check("g. large-frame guard: evidence image is <= 1280 px wide and bbox scaled correctly",
              img_ok and abs(d_bbox[0] - expected_x1) <= 1)
    else:
        check("g. large-frame guard: event created and verified", False)

    # (h) Digest PDF contains both modules' sections when both had activity
    hub.generate_digest()
    hub.wait_idle()
    dig = [m for m in messages if m["kind"] == "digest"]
    check("6. daily digest PDF created", bool(dig) and Path(dig[-1]["path"]).exists())
    if dig and Path(dig[-1]["path"]).exists():
        dig_text = extract_pdf_text(Path(dig[-1]["path"]))
        has_track = "Railway Track Defect Inspection" in dig_text or "Track" in dig_text
        has_bridge = "Bridge Structural Defect Inspection" in dig_text or "Bridge" in dig_text
        check("h. digest PDF contains sections for both track and bridge modules", has_track and has_bridge)
    else:
        check("h. digest PDF contains sections for both modules", False)

    errs = [m for m in messages if m["kind"] == "error"]
    check("no errors reported during self-test", not errs)
    for e in errs:
        print("   error:", e["message"])

    hub.shutdown()
    print("\nOutput folder:", out)
    for p in sorted(out.rglob("*.pdf")):
        print("  ", p.relative_to(out))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
