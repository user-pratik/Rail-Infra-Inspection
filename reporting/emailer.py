"""
Email delivery module for Railway Infrastructure Inspection Suite.
Builds and sends daily digests, critical incident PDFs, and incident CSV records
via standard smtplib + email.message.EmailMessage.
"""

from __future__ import annotations

import csv
import datetime as dt
import email.utils
import html
import logging
import mimetypes
import os
import smtplib
import zipfile
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("reporting.emailer")


def export_critical_incidents_csv(
    events: List[Dict[str, Any]],
    runs_by_id: Dict[int, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Exports critical incident events to CSV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Timestamp",
        "Structure/Run",
        "Defect Type",
        "Confidence",
        "Location/Chainage",
        "Severity",
        "Status",
        "Evidence Filename",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            run = runs_by_id.get(ev.get("run_id", 0), {})
            structure_or_run = (
                run.get("structure_id")
                or run.get("video_name")
                or f"Run #{ev.get('run_id', 'n/a')}"
            )
            lat = ev.get("lat") or 0.0
            lon = ev.get("lon") or 0.0
            loc_name = ev.get("location_name") or "Track Corridor"
            loc_str = f"{lat:.5f}, {lon:.5f} ({loc_name})"
            ev_dir = ev.get("evidence_dir") or ""
            evidence_file = os.path.basename(ev_dir) if ev_dir else "annotated.jpg"

            writer.writerow({
                "Timestamp": ev.get("detected_at") or dt.datetime.now().isoformat(),
                "Structure/Run": structure_or_run,
                "Defect Type": (ev.get("cls") or "").capitalize(),
                "Confidence": f"{float(ev.get('best_conf', 0.0)):.1%}",
                "Location/Chainage": loc_str,
                "Severity": (ev.get("severity") or "critical").upper(),
                "Status": ev.get("status") or "Unverified",
                "Evidence Filename": evidence_file,
            })
    return output_path


def build_email_html(
    date_str: str,
    critical_events: List[Dict[str, Any]],
    runs_by_id: Dict[int, Dict[str, Any]],
    total_runs: int,
    digest_pdf_name: str,
) -> str:
    """Builds clean HTML email body."""
    crit_count = len(critical_events)
    rows_html = ""
    for ev in critical_events:
        run = runs_by_id.get(ev.get("run_id", 0), {})
        source_name = run.get("video_name") or f"Run #{ev.get('run_id')}"
        if run.get("is_live"):
            source_name = f"[LIVE] {source_name}"
        cls_name = (ev.get("cls") or "").capitalize()
        conf_str = f"{float(ev.get('best_conf', 0)):.0%}"
        report_id = ev.get("report_id") or ev.get("event_id") or "N/A"
        time_str = ev.get("detected_at") or ""
        rows_html += f"""
        <tr style="border-bottom: 1px solid #e2e8f0;">
            <td style="padding: 10px; font-weight: bold; color: #b91c1c;">{html.escape(cls_name)}</td>
            <td style="padding: 10px;">{html.escape(report_id)}</td>
            <td style="padding: 10px;">{html.escape(source_name)}</td>
            <td style="padding: 10px;">{conf_str}</td>
            <td style="padding: 10px; font-size: 12px; color: #4a5568;">{html.escape(time_str)}</td>
        </tr>
        """

    if not rows_html:
        rows_html = """
        <tr>
            <td colspan="5" style="padding: 15px; text-align: center; color: #2b6cb0;">
                No critical incidents recorded for this date.
            </td>
        </tr>
        """

    alert_banner = (
        f"""<div style="background-color: #fee2e2; border-left: 5px solid #dc2626; padding: 14px; margin-bottom: 20px;">
            <h3 style="margin: 0; color: #991b1b; font-size: 16px;">
                ⚠️ ACTION REQUIRED: {crit_count} Critical Structural Defect(s) Detected
            </h3>
            <p style="margin: 6px 0 0 0; color: #7f1d1d; font-size: 13px;">
                Automated vision models flagged critical defects (cracks, spalling, or obstacles). Human inspector verification is required.
            </p>
        </div>"""
        if crit_count > 0
        else """<div style="background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 14px; margin-bottom: 20px;">
            <h3 style="margin: 0; color: #065f46; font-size: 16px;">
                ✅ Normal Operations: No Critical Incidents
            </h3>
            <p style="margin: 6px 0 0 0; color: #047857; font-size: 13px;">
                All inspection runs completed nominal or non-critical observations.
            </p>
        </div>"""
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f7fafc; margin: 0; padding: 24px; color: #1a202c;">
        <div style="max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; padding: 28px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="border-bottom: 2px solid #2b6cb0; padding-bottom: 12px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 20px; color: #1a365d;">🚄 Railway Infrastructure & Defect Inspection Suite</h1>
                <p style="margin: 4px 0 0 0; font-size: 13px; color: #718096;">Automated Daily Maintenance & Incident Dispatch Report</p>
            </div>

            {alert_banner}

            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px;">
                <tr>
                    <td style="padding: 6px 0; color: #718096; width: 140px;">Inspection Date:</td>
                    <td style="padding: 6px 0; font-weight: bold;">{html.escape(date_str)}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; color: #718096;">Inspection Runs:</td>
                    <td style="padding: 6px 0; font-weight: bold;">{total_runs} session(s) analysed</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; color: #718096;">Critical Incidents:</td>
                    <td style="padding: 6px 0; font-weight: bold; color: {'#dc2626' if crit_count > 0 else '#16a34a'};">{crit_count} defect(s)</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; color: #718096;">Digest Attachment:</td>
                    <td style="padding: 6px 0;">{html.escape(digest_pdf_name)}</td>
                </tr>
            </table>

            <h3 style="margin: 20px 0 10px 0; font-size: 15px; color: #2d3748; border-bottom: 1px solid #edf2f7; padding-bottom: 6px;">
                Critical Defects Summary ({crit_count})
            </h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px;">
                <thead>
                    <tr style="background-color: #f8fafc; border-bottom: 2px solid #cbd5e1; text-align: left;">
                        <th style="padding: 10px;">Defect</th>
                        <th style="padding: 10px;">Report ID</th>
                        <th style="padding: 10px;">Source</th>
                        <th style="padding: 10px;">Confidence</th>
                        <th style="padding: 10px;">Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <div style="background-color: #f1f5f9; border-radius: 6px; padding: 14px; font-size: 13px; color: #334155;">
                <b style="color: #0f172a;">Next Steps for Track & Bridge Engineering Team:</b>
                <ol style="margin: 8px 0 0 0; padding-left: 20px; line-height: 1.6;">
                    <li>Open attached critical incident PDFs to inspect evidence crops and coordinate locations.</li>
                    <li>Refer to <code>critical_incident_records_{date_str.replace('-', '')}.csv</code> for automated import into Maintenance Asset Database.</li>
                    <li>Dispatch localized track/bridge patrol to complete sign-off verification.</li>
                </ol>
            </div>

            <div style="margin-top: 24px; padding-top: 14px; border-top: 1px solid #e2e8f0; font-size: 11px; color: #94a3b8; text-align: center;">
                Generated automatically by Railway Infrastructure Inspection Suite AI Pipeline.<br>
                Confidence thresholds: Track R-CNN &ge; 35%, Bridge YOLO &ge; 45%.
            </div>
        </div>
    </body>
    </html>
    """


class EmailPackage:
    """Prepared package containing all attachments and metadata ready to send."""
    def __init__(
        self,
        date_str: str,
        subject: str,
        html_body: str,
        attachments: List[Path],
        recipients: List[str],
        critical_count: int,
    ):
        self.date_str = date_str
        self.subject = subject
        self.html_body = html_body
        self.attachments = attachments
        self.recipients = recipients
        self.critical_count = critical_count


def prepare_email_package(
    store,
    hub,
    date: Optional[str] = None,
    recipients: Optional[List[str]] = None,
) -> EmailPackage:
    """
    Gathers today's daily digest, critical incident PDFs, and exports critical incident CSV.
    """
    if date is None:
        date = dt.date.today().isoformat()

    cfg = hub.cfg
    base_dir = hub.base
    email_cfg = getattr(cfg, "email_cfg", {})

    target_recipients = recipients or email_cfg.get("default_recipients") or ["railway-inspections@example.com"]
    if isinstance(target_recipients, str):
        target_recipients = [r.strip() for r in target_recipients.split(",") if r.strip()]

    # 1. Fetch runs and events
    runs = store.runs_for_date(date)
    runs_by_id = {r["id"]: r for r in runs}
    events = store.events_for_date(date)

    critical_events = [e for e in events if e.get("severity") == "critical" or e.get("routing") == "incident"]

    # 2. Ensure daily digest exists
    digest_pdf_name = f"DIG-{date.replace('-', '')}.pdf"
    digest_pdf_path = base_dir / "daily" / digest_pdf_name
    if not digest_pdf_path.exists():
        # Compile digest now
        log.info(f"Digest PDF not found for {date}, compiling automatically...")
        hub.generate_digest(date=date, auto_open=False)
        hub.wait_idle()

    attachments: List[Path] = []
    if digest_pdf_path.exists():
        attachments.append(digest_pdf_path)

    # 3. Gather critical incident PDFs
    incidents_dir = base_dir / "incidents"
    incident_pdfs: List[Path] = []
    for ev in critical_events:
        rep_id = ev.get("report_id")
        if rep_id:
            pdf_file = incidents_dir / f"{rep_id}.pdf"
            if pdf_file.exists() and pdf_file not in incident_pdfs:
                incident_pdfs.append(pdf_file)

    # Cap directly attached incident PDFs to keep total email size under ~5 MB
    # and avoid socket write timeouts / SMTP server disconnects over residential links.
    max_direct = int(email_cfg.get("max_incident_attachments", 6))
    if len(incident_pdfs) <= max_direct:
        attachments.extend(incident_pdfs)
    else:
        # Attach the top priority incidents directly
        attachments.extend(incident_pdfs[:max_direct])

    # 4. Generate CSV export (always included)
    csv_name = f"critical_incident_records_{date.replace('-', '')}.csv"
    csv_path = base_dir / "daily" / csv_name
    export_critical_incidents_csv(critical_events, runs_by_id, csv_path)
    if csv_path.exists():
        attachments.append(csv_path)

    # 5. Build Subject & Body
    crit_count = len(critical_events)
    status_tag = "URGENT / ACTION REQUIRED" if crit_count > 0 else "DAILY REPORT"
    subject = f"[{status_tag}] Daily Rail & Structure Inspection Digest - {date} ({crit_count} Critical Incidents)"

    html_body = build_email_html(
        date_str=date,
        critical_events=critical_events,
        runs_by_id=runs_by_id,
        total_runs=len(runs),
        digest_pdf_name=digest_pdf_name if digest_pdf_path.exists() else "Digest compiled inline",
    )

    return EmailPackage(
        date_str=date,
        subject=subject,
        html_body=html_body,
        attachments=attachments,
        recipients=target_recipients,
        critical_count=crit_count,
    )


def send_email(
    package: EmailPackage,
    smtp_host: str,
    smtp_port: int = 587,
    use_tls: bool = True,
    username: str = "",
    password: str = "",
    from_addr: str = "inspections@railway-infra.org",
) -> Tuple[bool, str]:
    """
    Connects via smtplib and transmits EmailPackage.
    Supports TLS, SSL, or plain SMTP.
    Returns (True, msg) or (False, error_details).
    """
    try:
        msg = EmailMessage()
        msg["Subject"] = package.subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(package.recipients)
        msg["Date"] = email.utils.formatdate(localtime=True)

        # Plain text fallback
        plain_text = (
            f"Railway Infrastructure Inspection Digest - {package.date_str}\n"
            f"Critical Incidents: {package.critical_count}\n\n"
            f"Please refer to the attached PDF report(s) and CSV record for full defect details."
        )
        msg.set_content(plain_text)
        msg.add_alternative(package.html_body, subtype="html")

        # Attach files
        for att_path in package.attachments:
            if not att_path.exists():
                continue
            ctype, encoding = mimetypes.guess_type(str(att_path))
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)

            with open(att_path, "rb") as f:
                file_bytes = f.read()

            msg.add_attachment(
                file_bytes,
                maintype=maintype,
                subtype=subtype,
                filename=att_path.name,
            )

        log.info(f"Connecting to SMTP server {smtp_host}:{smtp_port}...")
        socket_timeout = 120  # 2 minutes for reliable upload over residential Wi-Fi
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=socket_timeout)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=socket_timeout)

        with server:
            server.ehlo()
            if use_tls and smtp_port != 465:
                server.starttls()
                server.ehlo()
            if username and password:
                clean_pw = password.strip()
                try:
                    server.login(username, clean_pw)
                except smtplib.SMTPAuthenticationError:
                    server.login(username, clean_pw.replace(" ", ""))
            server.send_message(msg)

        log.info(f"Email sent successfully to {package.recipients}")
        return True, f"Sent email to {', '.join(package.recipients)} with {len(package.attachments)} attachments."

    except (smtplib.SMTPServerDisconnected, TimeoutError) as exc:
        err_msg = (
            f"SMTP Connection Disconnected or Timed Out ({exc}). "
            "This usually indicates network upload latency or attachment size limits."
        )
        log.error(f"Failed to send email: {err_msg}")
        return False, err_msg
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        log.error(f"Failed to send email: {err_msg}")
        return False, err_msg

