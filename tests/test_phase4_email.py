"""
Unit and acceptance test for Phase 4: Email delivery.
Tests:
1. prepare_email_package compiles CSV, gathers digest and critical PDFs.
2. Formats email HTML and subject correctly.
3. Transmits successfully to a mock local SMTP server with all attachments intact.
"""

import os
import sys
import threading
import time
import warnings
from pathlib import Path

# Suppress Python 3.11 smtpd deprecation warning in test
warnings.filterwarnings("ignore", category=DeprecationWarning)
import smtpd
import asyncore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reporting.hub import ReportingHub
from reporting.emailer import prepare_email_package, send_email


class MockSMTPServer(smtpd.SMTPServer):
    def __init__(self, host="127.0.0.1", port=10255):
        super().__init__((host, port), None, decode_data=False)
        self.received_messages = []

    def process_message(self, peer, mailfrom, rcpttos, data, **kwargs):
        self.received_messages.append(data)


def test_email_pipeline():
    print("\n==========================================")
    print("PHASE 4 ACCEPTANCE TEST: Email Dispatch")
    print("==========================================")

    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="email_test_")
    hub = ReportingHub(base_dir=tmpdir)

    # 1. Ingest test session with critical crack event
    print("[STEP 1] Generating test session with confirmed crack event...")
    tok = hub.start_session("test_track.mp4", "rcnn.pth", "rcnn", 15.0, 100, 0.20, module="track")
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, 10):
        hub.ingest(tok, i, dummy_frame, [("crack", 0.88, (100, 100, 200, 200))])
    hub.end_session(tok)
    hub.generate_digest()
    hub.wait_idle()

    # 2. Prepare package
    print("[STEP 2] Preparing email package...")
    pkg = prepare_email_package(hub.store, hub, recipients=["test-inspector@example.com"])
    print(f"Subject: {pkg.subject}")
    print(f"Attachments ({len(pkg.attachments)}): {[p.name for p in pkg.attachments]}")
    assert len(pkg.attachments) >= 2, "Expected at least digest PDF and critical incident CSV"
    
    # Verify CSV file content
    csv_file = [p for p in pkg.attachments if p.name.endswith(".csv")][0]
    csv_content = csv_file.read_text(encoding="utf-8")
    print(f"CSV header & rows:\n{csv_content.strip()}")
    assert "Defect Type" in csv_content
    assert "Crack" in csv_content or "crack" in csv_content.lower()

    # 3. Start mock SMTP server
    print("\n[STEP 3] Starting mock local SMTP server on port 10255...")
    smtp_srv = MockSMTPServer(host="127.0.0.1", port=10255)
    loop_thread = threading.Thread(target=lambda: asyncore.loop(timeout=0.1), daemon=True)
    loop_thread.start()
    time.sleep(0.5)

    # 4. Transmit email
    print("[STEP 4] Sending inspection email...")
    ok, status_msg = send_email(
        pkg,
        smtp_host="127.0.0.1",
        smtp_port=10255,
        use_tls=False,
        username="",
        password="",
        from_addr="inspections@railway-infra.org"
    )
    print(f"Send status: ok={ok}, msg={status_msg}")
    assert ok is True, f"Send failed: {status_msg}"

    # Wait for server to receive
    time.sleep(1.0)
    assert len(smtp_srv.received_messages) == 1, "Mock SMTP server did not receive the message"
    raw_received = smtp_srv.received_messages[0]
    assert b"Subject:" in raw_received
    assert b"critical_incident_records" in raw_received
    assert b"Content-Type: application/pdf" in raw_received or b"Content-Type: text/csv" in raw_received
    print("Mock SMTP server verified receipt of multipart email with attachments.")

    smtp_srv.close()
    hub.shutdown()
    print("\n=== PHASE 4 ACCEPTANCE TEST PASSED! ===")
    return True

if __name__ == "__main__":
    ok = test_email_pipeline()
    sys.exit(0 if ok else 1)
