"""
Small Tkinter panel for gui_app.py.

    self.report_panel = ReportingPanel(parent, self.reporter, self.root,
                                       log_fn=self._report_log, bg=..., fg=...)
    self.report_panel.pack(fill="x", pady=(8, 0))

The hub calls back from its worker thread; the callback only appends to a
thread-safe queue. The panel drains that queue on the Tk main thread, so no
Tk call ever happens off the main thread.
"""

from __future__ import annotations

import queue
import tkinter as tk


class ReportingPanel:
    def __init__(self, parent, hub, root, log_fn=None, bg="#1e2330", fg="#e8eaed",
                 accent="#2d7ff9", font=("Segoe UI", 9)):
        self.hub, self.root, self.log_fn = hub, root, log_fn
        self._events: queue.Queue = queue.Queue()
        hub.on_report = self._events.put  # thread-safe hand-off

        self.frame = tk.Frame(parent, bg=bg, highlightbackground="#3a4255", highlightthickness=1)
        tk.Label(self.frame, text="Automated Reporting", bg=bg, fg=fg,
                 font=(font[0], font[1] + 1, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.status = tk.StringVar()
        tk.Label(self.frame, textvariable=self.status, bg=bg, fg=fg, font=font,
                 anchor="w", justify="left", wraplength=300).pack(fill="x", padx=8)
        self.last = tk.StringVar(value="No reports generated yet.")
        tk.Label(self.frame, textvariable=self.last, bg=bg, fg="#9aa4b5", font=font,
                 anchor="w", justify="left", wraplength=300).pack(fill="x", padx=8, pady=(0, 4))

        row = tk.Frame(self.frame, bg=bg)
        row.pack(fill="x", padx=8, pady=(0, 8))
        for text, cmd in (("Daily Digest", lambda: self.hub.generate_digest(auto_open=True)),
                          ("Report", lambda: self.hub.compile_report(auto_open=True)),
                          ("Folder", self.hub.open_reports_folder)):
            tk.Button(row, text=text, command=cmd, bg=accent, fg="white", relief="flat",
                      font=font, padx=8, pady=3, cursor="hand2").pack(side="left", padx=(0, 6))

        self._refresh_status()
        self.root.after(250, self._poll)

    def pack(self, **kw):
        self.frame.pack(**kw)

    def grid(self, **kw):
        self.frame.grid(**kw)

    # ------------------------------------------------------------------
    def _refresh_status(self):
        s = self.hub.stats()
        text = f"Incident reports: {s['incidents']}   |   Digest queue: {s['queued']}"
        if s["dropped"]:
            text += f"   |   Skipped frames: {s['dropped']}"
        self.status.set(text)

    def _log(self, msg: str):
        self.last.set(msg)
        if self.log_fn:
            self.log_fn(msg)

    def _poll(self):
        try:
            while True:
                self._handle(self._events.get_nowait())
        except queue.Empty:
            pass
        self._refresh_status()
        self.root.after(250, self._poll)

    def _handle(self, info: dict):
        kind = info.get("kind")
        if kind == "incident":
            self._log(f"INCIDENT {info['report_id']}: {info['cls']} ({info['conf']:.0%})")
        elif kind == "queued":
            tag = " [review]" if info.get("needs_review") else ""
            self._log(f"Queued for digest: {info['cls']} ({info['conf']:.0%}){tag}")
        elif kind == "digest":
            self._log(f"Digest ready: {info['report_id']} ({info['count']} findings)")
        elif kind == "digest_empty":
            self._log(f"No inspections recorded for {info['date']}.")
        elif kind == "run_complete":
            self._log(f"Run complete: {info['incidents']} incident(s), {info['queued']} queued for digest.")
        elif kind == "report_opened":
            rid = info.get("report_id") or "report"
            self._log(f"Opened report: {rid}")
        elif kind == "error":
            self._log(f"Reporting error: {info['message']}")
