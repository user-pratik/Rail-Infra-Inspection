"""
PDF rendering (ReportLab). Two report bodies share one base layout:

  * render_incident()  - Critical Defect Incident Report
  * render_digest()    - Daily Inspection Digest

Shared frame: header band (project, report type, report ID), footer (AI-generated
notice, system/model, generated time, page X of Y).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#12263A")
GREY = colors.HexColor("#5F6B7A")
LIGHT = colors.HexColor("#EEF1F5")
LINE = colors.HexColor("#C9D1DB")
SEV_COLOR = {"critical": colors.HexColor("#C62828"), "moderate": colors.HexColor("#EF6C00"),
             "info": colors.HexColor("#B8860B")}
SEV_LABEL = {"critical": "CRITICAL", "moderate": "MODERATE", "info": "INFORMATIONAL"}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

LIMITATIONS = [
    "Detections come from a machine-learning model and may include false positives or miss defects.",
    "Confidence is the model's score, not a calibrated probability.",
    "Sizes are in image pixels. No real-world scale is available unless the camera is calibrated.",
    "Location is simulated whenever the location source says so. Do not use it to dispatch crews.",
    "This report must be verified by a qualified person before any maintenance or operational decision.",
]


# --------------------------------------------------------------------------- helpers
def _styles() -> dict:
    return {
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=NAVY),
        "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=13, textColor=GREY),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=15,
                             textColor=NAVY, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=11.5),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=10, textColor=GREY),
        "tiny": ParagraphStyle("tiny", fontName="Helvetica", fontSize=7.5, leading=9.5),
        "hdr": ParagraphStyle("hdr", fontName="Helvetica-Bold", fontSize=8.5, leading=10.5,
                              textColor=colors.white),
        "badge": ParagraphStyle("badge", fontName="Helvetica-Bold", fontSize=10, leading=12,
                                textColor=colors.white),
        "cap": ParagraphStyle("cap", fontName="Helvetica", fontSize=8, leading=10, textColor=GREY,
                              alignment=1),
    }


def _txt(value) -> str:
    return escape(str(value)).replace("\n", "<br/>")


def _fmt_dt(iso: str | None) -> str:
    if not iso:
        return "n/a"
    try:
        return dt.datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M:%S")
    except ValueError:
        return str(iso)


def _fmt_ts(seconds) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def _fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m {seconds % 60:02d}s" if seconds >= 3600 \
        else f"{seconds // 60}m {seconds % 60:02d}s"


def _location_label(ev: dict) -> str:
    base = f"{ev.get('lat', 0):.6f}, {ev.get('lon', 0):.6f}"
    return base + ("  (SIMULATED)" if ev.get("location_source") == "simulated" else "")


def _heading(text: str, st) -> Paragraph:
    return Paragraph(_txt(text), st["h2"])


def _kv(rows, st, label_w=44 * mm, badge: dict | None = None, row_heights=None) -> Table:
    data = []
    for label, value in rows:
        v = Paragraph(_txt(value), st["cell"]) if isinstance(value, str) else value
        data.append([Paragraph(f"<b>{_txt(label)}</b>", st["cell"]), v])
    t = Table(data, colWidths=[label_w, CONTENT_W - label_w], rowHeights=row_heights)
    style = [("GRID", (0, 0), (-1, -1), 0.5, LINE), ("BACKGROUND", (0, 0), (0, -1), LIGHT),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for idx, col in (badge or {}).items():
        style.append(("BACKGROUND", (1, idx), (1, idx), col))
    t.setStyle(TableStyle(style))
    return t


def _grid(header, rows, widths, st, cell_style="cell") -> Table:
    data = [[Paragraph(_txt(h), st["hdr"]) for h in header]]
    for r in rows:
        data.append([c if not isinstance(c, str) else Paragraph(_txt(c), st[cell_style]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _image(path: Path, max_w: float, max_h: float):
    if not path.exists():
        return None
    try:
        iw, ih = ImageReader(str(path)).getSize()
        scale = min(max_w / iw, max_h / ih)
        return Image(str(path), width=iw * scale, height=ih * scale)
    except Exception:
        return None


def _canvas_factory(meta: dict):
    class _NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved)
            for state in self._saved:
                self.__dict__.update(state)
                self._decorate(total)
                super().showPage()
            super().save()

        def _decorate(self, total):
            self.saveState()
            self.setFillColor(NAVY)
            self.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, stroke=0, fill=1)
            self.setFillColor(colors.white)
            self.setFont("Helvetica-Bold", 10)
            self.drawString(MARGIN, PAGE_H - 9 * mm, meta["project"])
            self.setFont("Helvetica", 9)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - 9 * mm, f'{meta["type"]}   |   {meta["id"]}')
            self.setStrokeColor(LINE)
            self.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
            self.setFillColor(GREY)
            self.setFont("Helvetica", 7.5)
            self.drawString(MARGIN, 11 * mm, "AI-generated report - requires human verification before action.")
            self.drawString(MARGIN, 7 * mm, f'System v{meta["version"]}   |   {meta["model"]}   |   '
                                            f'Generated {meta["generated"]}   |   Report version {meta["rev"]}')
            self.drawRightString(PAGE_W - MARGIN, 11 * mm, f"Page {self._pageNumber} of {total}")
            self.restoreState()

    return _NumberedCanvas


def _keep_headings(story):
    """Glue every section heading to the flowable after it so headings are never stranded."""
    out, i = [], 0
    while i < len(story):
        item = story[i]
        if isinstance(item, Paragraph) and item.style.name == "h2" and i + 1 < len(story):
            out.append(KeepTogether([item, story[i + 1]]))
            i += 2
        else:
            out.append(item)
            i += 1
    return out


def _build(path: Path, story, meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    story = _keep_headings(story)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=22 * mm, bottomMargin=20 * mm,
                            title=f'{meta["type"]} {meta["id"]}', author=meta["project"])
    doc.build(story, canvasmaker=_canvas_factory(meta))


def _signoff(st):
    return _kv([
        ("Reviewer name", ""), ("Role / designation", ""),
        ("Decision", "[   ] Confirmed defect        [   ] False positive        [   ] Needs re-inspection"),
        ("Notes", ""), ("Signature", ""), ("Date", ""),
    ], st, row_heights=[8 * mm, 8 * mm, 8 * mm, 20 * mm, 11 * mm, 8 * mm])


def _appendix(cfg, run: dict, st, extra_rows=None, module: str = "track"):
    model = cfg.get_model_info(module, run.get("model_type", ""))
    agg = cfg.aggregation_cfg(module)
    rows = [
        ("Model", model.get("name", run.get("model_type", "n/a"))),
        ("Model benchmark", model.get("metrics", "n/a")),
        ("Weights file", Path(run.get("model_path") or "n/a").name),
        ("Weights SHA-256 (first 16)", run.get("model_hash") or "n/a"),
        ("GUI confidence threshold", f'{run["conf_threshold"]:.2f}' if run.get("conf_threshold") else "n/a"),
        ("Event confirmation rule", f'{agg["confirm_hits"]} detections within {agg["confirm_window"]} frames; '
                                    f'event closes after {agg["max_gap_frames"]} frames unseen'),
    ] + (extra_rows or [])
    items = [_heading("Appendix: Model, Limitations and Disclaimer", st), _kv(rows, st), Spacer(1, 5),
             Paragraph("<b>Limitations</b>", st["body"])]
    items += [Paragraph("&bull; " + _txt(x), st["body"]) for x in LIMITATIONS]
    return items


def _meta(cfg, rtype, rid, run_model, generated_at, rev):
    r = cfg.reports_cfg
    return {"project": r["project_name"], "type": rtype, "id": rid, "version": r["system_version"],
            "model": run_model, "generated": generated_at.strftime("%d %b %Y %H:%M"), "rev": rev}


# --------------------------------------------------------------------------- incident report
def render_incident(path, *, base: Path, cfg, report_id: str, event: dict, run: dict,
                    history: list[dict], generated_at: dt.datetime, version: int = 1) -> None:
    st = _styles()
    base = Path(base)
    path = Path(path)
    module = event.get("module") or run.get("module") or "track"
    mod_cfg = cfg.module_cfg(module)
    rule = cfg.rule(module, event["cls"])
    sev = event["severity"]
    cls_label = event["cls"].capitalize()
    bbox = json.loads(event["bbox"])
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    model_info = cfg.get_model_info(module, run.get("model_type", ""))
    model_name = model_info.get("name", run.get("model_type", "model"))
    ev_dir = base / (event.get("evidence_dir") or "")
    sim = event.get("location_source") == "simulated"

    story = [Paragraph("Critical Defect Incident Report", st["h1"]),
             Paragraph(f"{_txt(cls_label)} detected on {_txt(run.get('video_name', 'n/a'))} "
                       f"at {_txt(_fmt_ts(event['first_ts']))}", st["sub"]), Spacer(1, 6)]

    # 1 Summary
    badge = Paragraph(SEV_LABEL[sev], st["badge"])
    story += [_heading("1. Summary", st), _kv([
        ("Report ID", report_id), ("Severity", badge), ("Defect type", cls_label),
        ("Detection confidence", f"{event['best_conf']:.0%} (best), {event['mean_conf']:.0%} (mean)"),
        ("Location", f"{event['location_name']}\n{_location_label(event)}"),
        ("Recommended action", rule.get("recommended_action", "n/a")),
        ("Status", f"{event['status']}  (AI-detected, pending verification)"),
    ], st, badge={1: SEV_COLOR[sev]})]

    # 2 Inspection context
    inspection_label = mod_cfg.get("inspection_label", "Defect inspection")
    story += [_heading("2. Inspection Context", st), _kv([
        ("Source video", run.get("video_name", "n/a")),
        ("Inspection date / time", _fmt_dt(run.get("started_at"))),
        ("Inspection module", inspection_label),
        ("Model", f"{model_name} ({Path(run.get('model_path') or 'n/a').name})"),
        ("Weights SHA-256 (first 16)", run.get("model_hash") or "n/a"),
        ("GUI confidence threshold", f'{run["conf_threshold"]:.2f}' if run.get("conf_threshold") else "n/a"),
        ("Operator", run.get("operator") or "n/a"),
        ("Event detected at", _fmt_dt(event["detected_at"])),
    ], st)]

    # 3 Defect details
    # For bridge events, calculate bbox area percentage of frame
    area_note = ""
    if module == "bridge":
        ann_img_path = ev_dir / "annotated.jpg"
        if ann_img_path.exists():
            try:
                iw, ih = ImageReader(str(ann_img_path)).getSize()
                if iw > 0 and ih > 0:
                    area_pct = (w * h) / float(iw * ih) * 100.0
                    area_note = f" (covers {area_pct:.1f}% of the frame)"
            except Exception:
                pass

    story += [_heading("3. Defect Details", st),
              Paragraph(_txt(rule.get("description", "")), st["body"]), Spacer(1, 4), _kv([
                  ("Class", cls_label),
                  ("Bounding box (px)", f"x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}"),
                  ("Size in image", f"{w} x {h} px{area_note} (real-world size unavailable: camera not calibrated)"),
                  ("Frame range", f"{event['first_frame']} to {event['last_frame']} "
                                  f"(best frame {event['best_frame']})"),
                  ("Video time range", f"{_fmt_ts(event['first_ts'])} to {_fmt_ts(event['last_ts'])}"),
                  ("Confirmed detections", f"{event['hits']} frames"),
              ], st)]

    # 4 Evidence
    ann = _image(ev_dir / "annotated.jpg", 92 * mm, 85 * mm)
    crop = _image(ev_dir / "crop.jpg", 70 * mm, 85 * mm)
    ev_items = [_heading("4. Evidence", st)]
    if ann or crop:
        t = Table([[ann or "", crop or ""],
                   [Paragraph("Annotated frame (best detection)", st["cap"]),
                    Paragraph("Defect crop (enlarged)", st["cap"])]],
                  colWidths=[CONTENT_W * 0.55, CONTENT_W * 0.45])
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, 0), "MIDDLE")]))
        ev_items.append(t)
        strip = []
        for name, cap in (("strip_first.jpg", f"First seen (frame {event['first_frame']})"),
                          ("strip_best.jpg", f"Best (frame {event['best_frame']})"),
                          ("strip_last.jpg", f"Last seen (frame {event['last_frame']})")):
            im = _image(ev_dir / name, 52 * mm, 52 * mm)
            if im:
                strip.append([im, Paragraph(cap, st["cap"])])
        if strip:
            s = Table([[c[0] for c in strip], [c[1] for c in strip]], colWidths=[CONTENT_W / len(strip)] * len(strip))
            s.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            ev_items += [Spacer(1, 6), s]
    else:
        ev_items.append(Paragraph("No evidence images were saved for this event.", st["body"]))
    story += ev_items

    # 5 Location
    loc_rows = [
        ("Place", event["location_name"]),
        ("GPS (lat, lon)", _location_label(event)),
    ]
    if module == "bridge":
        loc_rows.append(("Structure / bridge ID", run.get("structure_id") or "Not available"))
        loc_rows.append(("Structural element (pier / girder / abutment)", "Not available"))
    else:
        loc_rows.append(("Chainage / km marker", "Not available"))
        loc_rows.append(("Section / line", "Not available"))
    loc_rows.append(("Location source", "SIMULATED: dummy telemetry, not a real position" if sim else "Telemetry"))
    story += [_heading("5. Location", st), _kv(loc_rows, st)]

    # 6 Risk and action
    story += [_heading("6. Risk and Recommended Action", st), _kv([
        ("Risk", rule.get("risk", "n/a")),
        ("Recommended action", rule.get("recommended_action", "n/a")),
        ("Required response time", rule.get("response_time", "n/a")),
    ], st)]
    if event.get("note"):
        story += [Spacer(1, 4), Paragraph(f"<b>Note:</b> {_txt(event['note'])}", st["body"])]

    # 7 History
    story.append(_heading("7. History", st))
    if history:
        story.append(_grid(["Event", "Detected", "Confidence", "Status", "Report"],
                           [[h["event_id"], _fmt_dt(h["detected_at"]), f"{h['best_conf']:.0%}",
                             h["status"], h.get("report_id") or "-"] for h in history],
                           [34 * mm, 42 * mm, 24 * mm, 28 * mm, 46 * mm], st))
    else:
        story.append(Paragraph("No previous events of this class are recorded.", st["body"]))
    story.append(Paragraph("Matching is by class only because the location is simulated; "
                           "match by position once real GPS or chainage is available.", st["small"]))

    # 8 Sign-off
    story.append(KeepTogether([_heading("8. Verification and Sign-off", st), _signoff(st)]))

    # 9 Appendix
    story += _appendix(cfg, run, st, module=module)

    _build(path, story, _meta(cfg, "Critical Defect Incident Report", report_id, model_name,
                              generated_at, version))


# --------------------------------------------------------------------------- daily digest
def render_digest(path, *, base: Path, cfg, report_id: str, date: str, events: list[dict],
                  prev_events: list[dict], runs: list[dict], generated_at: dt.datetime,
                  version: int = 1) -> None:
    st = _styles()
    base = Path(base)
    path = Path(path)
    incidents = [e for e in events if e["routing"] == "incident"]
    digest_items = [e for e in events if e["routing"] == "digest"]
    inventory = [e for e in events if e["routing"] == "inventory"]
    review = [e for e in digest_items if e["severity"] == "critical"]

    # Active modules on this date (from events and runs)
    all_modules = cfg.module_names()
    active_modules = [m for m in all_modules if any(e.get("module", "track") == m for e in events)
                      or any(r.get("module", "track") == m for r in runs)]
    if not active_modules:
        active_modules = ["track"]

    footage = sum((r.get("frames_processed") or 0) / (r.get("fps") or 15.0) for r in runs)
    dropped = sum(r.get("dropped_frames") or 0 for r in runs)
    model_types = sorted({r.get("model_type") for r in runs if r.get("model_type")})
    model_names_list = []
    for r in runs:
        m_mod = r.get("module", "track")
        m_type = r.get("model_type", "")
        name = cfg.get_model_info(m_mod, m_type).get("name", m_type)
        if name and name not in model_names_list:
            model_names_list.append(name)
    model_name = ", ".join(model_names_list) or "n/a"
    pretty = dt.date.fromisoformat(date).strftime("%A, %d %B %Y")

    story = [Paragraph("Daily Inspection Digest", st["h1"]), Paragraph(_txt(pretty), st["sub"]), Spacer(1, 6)]

    # 1 Summary (with totals per module)
    summary_rows = [
        ("Report ID", report_id), ("Date", date),
        ("Inspection runs", str(len(runs))),
        ("Footage analysed", _fmt_dur(footage) if runs else "n/a"),
        ("Confirmed events", f"{len(events)} total: {len(incidents)} critical incidents, "
                             f"{len(digest_items)} in this digest, {len(inventory)} inventory"),
    ]
    # Add per-module totals row
    mod_totals = []
    for m in active_modules:
        m_disp = cfg.module_cfg(m).get("display_name", m.capitalize())
        m_evs = [e for e in events if e.get("module", "track") == m]
        m_inc = sum(1 for e in m_evs if e["routing"] == "incident")
        m_dig = sum(1 for e in m_evs if e["routing"] == "digest")
        m_inv = sum(1 for e in m_evs if e["routing"] == "inventory")
        mod_totals.append(f"<b>{m_disp}:</b> {len(m_evs)} event(s) ({m_inc} incident, {m_dig} digest, {m_inv} inventory)")
    summary_rows.append(("Module breakdown", Paragraph("<br/>".join(mod_totals), st["cell"])))
    summary_rows.append(("Status", "AI-detected, pending verification"))
    story += [_heading("1. Summary", st), _kv(summary_rows, st)]

    # Class-count tables: one block per module that had runs or events that day, headed by display_name
    for m in active_modules:
        m_disp = cfg.module_cfg(m).get("display_name", m.capitalize())
        story += [Spacer(1, 6), Paragraph(f"<b>{_txt(m_disp)} - Defect Class Summary</b>", st["body"])]
        m_order = cfg.class_order(m)
        counts = [[c.capitalize(), str(sum(1 for e in events if e.get("module", "track") == m and e["cls"] == c)),
                   (cfg.rule(m, c).get("severity", "n/a")).capitalize()] for c in m_order]
        story += [_grid(["Class", "Events", "Severity"], counts, [60 * mm, 30 * mm, 40 * mm], st)]

    # 2 Escalations / review (rows show a Module column)
    story.append(_heading("2. Escalations and Items Needing Review", st))
    if incidents or review:
        esc_rows = []
        for e in incidents + review:
            e_mod = e.get("module", "track").capitalize()
            action_desc = "Incident report issued" if e["routing"] == "incident" else "Low confidence: review"
            esc_rows.append([e_mod, e.get("report_id") or e["event_id"], e["cls"].capitalize(),
                             f"{e['best_conf']:.0%}", action_desc, _fmt_ts(e["first_ts"])])
        # Widths sum to 174 mm (CONTENT_W)
        story.append(_grid(["Module", "Reference", "Class", "Confidence", "Action", "Video time"], esc_rows,
                           [22 * mm, 36 * mm, 26 * mm, 22 * mm, 44 * mm, 24 * mm], st))
    else:
        story.append(Paragraph("No critical incidents or items needing review.", st["body"]))

    # 3 Findings table (with Module column)
    story.append(_heading("3. Findings Compiled in This Digest", st))
    if digest_items:
        find_rows = []
        for e in digest_items:
            thumb = _image(base / (e.get("evidence_dir") or "") / "crop.jpg", 20 * mm, 16 * mm)
            e_mod = e.get("module", "track").capitalize()
            loc_str = f"{e['lat']:.5f},\n{e['lon']:.5f}" + (" (sim.)" if e.get("location_source") == "simulated" else "")
            find_rows.append([thumb or "-", e_mod, e["event_id"], _fmt_ts(e["first_ts"]), e["cls"].capitalize(),
                              SEV_LABEL[e["severity"]].capitalize(), f"{e['best_conf']:.0%}", loc_str])
        # Widths sum to 174 mm (CONTENT_W)
        story.append(_grid(["Image", "Module", "Event", "Video time", "Class", "Severity", "Conf.", "Location"], find_rows,
                           [20 * mm, 20 * mm, 28 * mm, 18 * mm, 20 * mm, 20 * mm, 14 * mm, 34 * mm], st, "tiny"))
    else:
        story.append(Paragraph("No non-critical findings today.", st["body"]))

    # 4 Trend vs previous day: one block per module that had runs or events that day, headed by display_name
    story.append(_heading("4. Trend vs Previous Day", st))
    for m in active_modules:
        m_disp = cfg.module_cfg(m).get("display_name", m.capitalize())
        story += [Spacer(1, 4), Paragraph(f"<b>{_txt(m_disp)} - Trends</b>", st["body"])]
        trend = []
        for c in cfg.class_order(m):
            today_cnt = sum(1 for e in events if e.get("module", "track") == m and e["cls"] == c)
            prev_cnt = sum(1 for e in prev_events if e.get("module", "track") == m and e["cls"] == c)
            trend.append([c.capitalize(), str(today_cnt), str(prev_cnt), f"{today_cnt - prev_cnt:+d}"])
        story.append(_grid(["Class", "Today", "Previous day", "Change"], trend,
                           [60 * mm, 30 * mm, 40 * mm, 30 * mm], st))
    if not prev_events:
        story.append(Paragraph("No events were recorded on the previous day.", st["small"]))

    # 5 Coverage (with Module column)
    story.append(_heading("5. Inspection Coverage", st))
    if runs:
        cov_rows = []
        for r in runs:
            r_mod = r.get("module", "track")
            m_info = cfg.get_model_info(r_mod, r.get("model_type", ""))
            cov_rows.append([f"#{r['id']}", r_mod.capitalize(), r.get("video_name") or "n/a",
                             m_info.get("name", r.get("model_type", "n/a")),
                             str(r.get("frames_processed") or 0), _fmt_dt(r.get("started_at"))])
        # Widths sum to 174 mm (CONTENT_W)
        story.append(_grid(["Run", "Module", "Source", "Model", "Frames", "Started"], cov_rows,
                           [12 * mm, 22 * mm, 44 * mm, 38 * mm, 16 * mm, 42 * mm], st))
    else:
        story.append(Paragraph("No inspection runs were recorded for this date.", st["body"]))
    if dropped:
        story.append(Paragraph(f"Note: {dropped} frame(s) were skipped by the reporting queue under load.", st["small"]))
    if inventory:
        story.append(Paragraph(f"Inventory only: {len(inventory)} rail joint/feature event(s) logged, not reported as defects.",
                               st["small"]))

    # 6 Sign-off + appendix
    story.append(KeepTogether([_heading("6. Verification and Sign-off", st), _signoff(st)]))
    primary_run = runs[0] if runs else {}
    primary_module = primary_run.get("module", "track") if runs else "track"
    extra_appendix = []
    if len(active_modules) > 1:
        extra_appendix.append(("Active modules", ", ".join(m.capitalize() for m in active_modules)))
    if len(model_types) > 1:
        extra_appendix.append(("Models used", model_name))
    story += _appendix(cfg, primary_run, st, extra_rows=extra_appendix or None, module=primary_module)

    _build(path, story, _meta(cfg, "Daily Inspection Digest", report_id, model_name, generated_at, version))
