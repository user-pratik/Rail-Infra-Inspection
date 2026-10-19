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

import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox


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
                          ("Folder", self.hub.open_reports_folder),
                          ("📧 Send Email", self._prompt_send_email)):
            tk.Button(row, text=text, command=cmd, bg=accent, fg="white", relief="flat",
                      font=font, padx=8, pady=3, cursor="hand2").pack(side="left", padx=(0, 6))

        self._cached_password = ""
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

    def _prompt_send_email(self):
        try:
            from reporting.emailer import prepare_email_package, send_email
            pkg = prepare_email_package(self.hub.store, self.hub)
        except Exception as e:
            messagebox.showerror("Email Preparation Error", f"Failed to prepare email package:\n{e}")
            return

        cfg = getattr(self.hub, "cfg", None)
        email_cfg = getattr(cfg, "email_cfg", {}) if cfg else {}
        smtp_host = email_cfg.get("smtp_host", "localhost")
        smtp_port = int(email_cfg.get("smtp_port", 587))
        use_tls = bool(email_cfg.get("use_tls", True))
        username = email_cfg.get("username", "")
        password_env = email_cfg.get("password_env", "SMTP_PASSWORD")
        from_addr = email_cfg.get("from_addr", "Railway Inspection Suite <inspections@railway-infra.org>")

        env_pw = os.environ.get(password_env, "")
        cached_pw = getattr(self, "_cached_password", "")
        active_pw = env_pw or cached_pw

        dlg = tk.Toplevel(self.root)
        dlg.title("📧 Send Inspection Reports")
        dlg.geometry("540x440")
        dlg.configure(bg="#1a1b26")
        dlg.transient(self.root)
        dlg.grab_set()

        try:
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            dw, dh = 540, 440
            dx = max(0, rx + (rw - dw) // 2)
            dy = max(0, ry + (rh - dh) // 2)
            dlg.geometry(f"{dw}x{dh}+{dx}+{dy}")
        except Exception:
            pass

        tk.Label(
            dlg, text="📧 Dispatch Daily Digest & Critical Reports",
            font=("Segoe UI", 12, "bold"), bg="#1a1b26", fg="#7aa2f7"
        ).pack(anchor="w", padx=16, pady=(14, 8))

        info_f = tk.Frame(dlg, bg="#24283b", padx=12, pady=10)
        info_f.pack(fill="x", padx=16, pady=4)

        crit_color = "#f7768e" if pkg.critical_count > 0 else "#9ece6a"
        tk.Label(
            info_f, text=f"• Inspection Date: {pkg.date_str}   |   Critical Incidents: {pkg.critical_count}",
            font=("Segoe UI", 10, "bold"), bg="#24283b", fg=crit_color
        ).pack(anchor="w")

        att_names = ", ".join(p.name for p in pkg.attachments)
        tk.Label(
            info_f, text=f"• Attachments ({len(pkg.attachments)} files):\n  {att_names}",
            font=("Segoe UI", 8), bg="#24283b", fg="#a9b1d6", justify="left", wraplength=480
        ).pack(anchor="w", pady=(4, 0))

        tk.Label(dlg, text="Recipients (comma separated):", font=("Segoe UI", 9, "bold"), bg="#1a1b26", fg="#c0caf5").pack(anchor="w", padx=16, pady=(10, 2))
        recip_var = tk.StringVar(value=", ".join(pkg.recipients))
        recip_entry = tk.Entry(dlg, textvariable=recip_var, font=("Segoe UI", 9), bg="#24283b", fg="white", insertbackground="white")
        recip_entry.pack(fill="x", padx=16)

        tk.Label(
            dlg, text=f"SMTP: {smtp_host}:{smtp_port}  |  TLS: {use_tls}  |  From: {from_addr}",
            font=("Segoe UI", 8), bg="#1a1b26", fg="#787c99"
        ).pack(anchor="w", padx=16, pady=(6, 4))

        pw_var = tk.StringVar(value=active_pw)
        rem_var = tk.BooleanVar(value=True)

        if not active_pw:
            pw_frame = tk.Frame(dlg, bg="#1a1b26")
            pw_frame.pack(fill="x", padx=16, pady=4)
            tk.Label(pw_frame, text=f"SMTP Password (or set ${password_env}):", font=("Segoe UI", 9), bg="#1a1b26", fg="#e0af68").pack(anchor="w")
            pw_entry = tk.Entry(pw_frame, textvariable=pw_var, show="*", font=("Segoe UI", 9), bg="#24283b", fg="white", insertbackground="white")
            pw_entry.pack(fill="x", pady=2)
            tk.Checkbutton(
                pw_frame, text="Remember password for this session", variable=rem_var,
                bg="#1a1b26", fg="#a9b1d6", selectcolor="#24283b", activebackground="#1a1b26"
            ).pack(anchor="w")
        else:
            source_desc = f"environment variable ${password_env}" if env_pw else "cached session memory"
            tk.Label(
                dlg, text=f"🔒 Password loaded from {source_desc}",
                font=("Segoe UI", 8, "italic"), bg="#1a1b26", fg="#9ece6a"
            ).pack(anchor="w", padx=16, pady=2)

        btn_box = tk.Frame(dlg, bg="#1a1b26")
        btn_box.pack(fill="x", padx=16, pady=(16, 12), side="bottom")

        def _do_send():
            final_recips = [r.strip() for r in recip_var.get().split(",") if r.strip()]
            if not final_recips:
                messagebox.showerror("Error", "Please provide at least one recipient email address.")
                return
            final_pw = pw_var.get()
            if rem_var.get() and final_pw:
                self._cached_password = final_pw
            pkg.recipients = final_recips
            dlg.destroy()

            self._log(f"Connecting to {smtp_host}:{smtp_port} to email digest...")
            def _thread_send():
                ok, msg = send_email(
                    pkg, smtp_host=smtp_host, smtp_port=smtp_port,
                    use_tls=use_tls, username=username, password=final_pw,
                    from_addr=from_addr
                )
                if ok:
                    self.root.after(0, lambda: self._log(f"✅ Email delivered to {len(final_recips)} recipient(s)"))
                    self.root.after(0, lambda: messagebox.showinfo("Email Sent", msg))
                else:
                    self.root.after(0, lambda: self._log(f"❌ Email error: {msg}"))
                    self.root.after(0, lambda: messagebox.showerror("Email Error", f"Failed to send email:\n{msg}"))
            threading.Thread(target=_thread_send, daemon=True).start()

        tk.Button(
            btn_box, text="Cancel", command=dlg.destroy,
            bg="#414868", fg="white", relief="flat", font=("Segoe UI", 9), padx=14, pady=4, cursor="hand2"
        ).pack(side="right", padx=(6, 0))

        tk.Button(
            btn_box, text="🚀 Send Email Now", command=_do_send,
            bg="#7aa2f7", fg="#1a1b26", relief="flat", font=("Segoe UI", 9, "bold"), padx=14, pady=4, cursor="hand2"
        ).pack(side="right")
