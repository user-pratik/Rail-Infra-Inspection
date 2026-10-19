"""
Generate the daily digest without opening the GUI.

    python -m reporting.daily                    # today
    python -m reporting.daily --date 2026-10-18

Schedule it with Windows Task Scheduler (e.g. every night at 23:59):
    Program:   D:\\RAIL-INFRA-INSPECTION\\venv\\Scripts\\python.exe
    Arguments: -m reporting.daily
    Start in:  D:\\RAIL-INFRA-INSPECTION
"""

from __future__ import annotations

import argparse

from .hub import ReportingHub


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the daily inspection digest.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    result = {}
    hub = ReportingHub(on_report=lambda info: result.update(info))
    hub.generate_digest(args.date)
    hub.wait_idle()
    hub.shutdown()

    kind = result.get("kind")
    if kind == "digest":
        print(f"Digest {result['report_id']} written: {result['path']}")
    elif kind == "digest_empty":
        print(f"Nothing recorded for {result['date']}; no digest created.")
    else:
        print(f"Digest not created: {result.get('message', 'unknown error')}")


if __name__ == "__main__":
    main()
