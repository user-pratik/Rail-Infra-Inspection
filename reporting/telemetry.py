"""
Dummy telemetry provider for the defect-reporting module.

Simulates a GPS track that starts at VIT Chennai and drifts a small, fixed
amount for every frame, so each frame/image gets a slightly different
latitude/longitude. Replace with a real provider (CSV sidecar, GPS log,
chainage lookup) later; the interface stays the same.

Usage:
    provider = DummyTelemetry(fps=15)
    loc = provider.get(frame_index=142)
    print(loc.latitude, loc.longitude, loc.location_name)

Generate a CSV sidecar:
    python -m reporting.telemetry --frames 527 --fps 30 --out telemetry.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

# VIT Chennai campus (Kelambakkam-Vandalur Road), approximate centre
VIT_CHENNAI_LAT = 12.8406
VIT_CHENNAI_LON = 80.1534
VIT_CHENNAI_NAME = "VIT Chennai, Kelambakkam-Vandalur Road, Chennai, Tamil Nadu"

# Drift applied per frame, in degrees.
# 0.000005 deg is roughly 0.55 m, so ~8 m per second at 15 FPS.
# Tune these to make the movement bigger or smaller.
LAT_STEP_PER_FRAME = 0.000005
LON_STEP_PER_FRAME = 0.000004


@dataclass(frozen=True)
class GeoPoint:
    frame_index: int
    timestamp_s: float
    latitude: float
    longitude: float
    location_name: str

    @property
    def coords_str(self) -> str:
        return f"{self.latitude:.6f}, {self.longitude:.6f}"


class DummyTelemetry:
    """Returns a deterministic, slowly drifting position for any frame index."""

    def __init__(
        self,
        fps: float = 15.0,
        start_lat: float = VIT_CHENNAI_LAT,
        start_lon: float = VIT_CHENNAI_LON,
        lat_step: float = LAT_STEP_PER_FRAME,
        lon_step: float = LON_STEP_PER_FRAME,
        location_name: str = VIT_CHENNAI_NAME,
    ):
        self.fps = fps if fps > 0 else 15.0
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.lat_step = lat_step
        self.lon_step = lon_step
        self.location_name = location_name

    def get(self, frame_index: int) -> GeoPoint:
        return GeoPoint(
            frame_index=frame_index,
            timestamp_s=frame_index / self.fps,
            latitude=round(self.start_lat + self.lat_step * frame_index, 6),
            longitude=round(self.start_lon + self.lon_step * frame_index, 6),
            location_name=self.location_name,
        )


def write_csv(path: str, frames: int, fps: float) -> None:
    provider = DummyTelemetry(fps=fps)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_s", "latitude", "longitude", "location_name"])
        for i in range(frames):
            p = provider.get(i)
            writer.writerow([p.frame_index, f"{p.timestamp_s:.3f}", p.latitude, p.longitude, p.location_name])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a dummy GPS telemetry CSV (VIT Chennai).")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default="telemetry.csv")
    args = ap.parse_args()
    write_csv(args.out, args.frames, args.fps)
    print(f"Wrote {args.frames} rows to {args.out}")
